# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any

import torch
import torch.nn.functional as F

from ..distillation.distillation_trainer import DistillationTrainer
from .server_distillation_config import ServerDistillationConfig


# Self-contained copy of the base trainer's `_jsd_divergence` (together with the `_compute_prompt_length` /
# `_reduce_divergence_loss` methods below). The base deletes these while migrating to the chunked loss, but this
# quarantined server path still depends on them; it is re-pointed at the stable base wholesale later (issue #6449).
def _jsd_divergence(student_log_probs, teacher_log_probs, beta, support_mask=None):
    """Compute JSD (or forward/reverse KL) from log-probability tensors.

    When *support_mask* is not None, uses manual computation with masked positions zeroed. When None, uses
    ``F.kl_div``.
    """
    if support_mask is not None:
        safe_student = torch.where(support_mask, student_log_probs, torch.zeros_like(student_log_probs))
        safe_teacher = torch.where(support_mask, teacher_log_probs, torch.zeros_like(teacher_log_probs))
        student_probs = torch.where(support_mask, student_log_probs.exp(), torch.zeros_like(student_log_probs))
        teacher_probs = torch.where(support_mask, teacher_log_probs.exp(), torch.zeros_like(teacher_log_probs))

        if beta == 0:
            return torch.nan_to_num(teacher_probs * (safe_teacher - safe_student), nan=0.0)
        elif beta == 1:
            return torch.nan_to_num(student_probs * (safe_student - safe_teacher), nan=0.0)
        else:
            beta_t = torch.tensor(beta, dtype=student_log_probs.dtype, device=student_log_probs.device)
            tiny = torch.finfo(student_probs.dtype).tiny
            mixture_probs = (1 - beta_t) * student_probs + beta_t * teacher_probs
            safe_mixture = torch.where(
                support_mask,
                torch.log(mixture_probs.clamp_min(tiny)),
                torch.zeros_like(student_log_probs),
            )
            kl_teacher = torch.nan_to_num(teacher_probs * (safe_teacher - safe_mixture), nan=0.0)
            kl_student = torch.nan_to_num(student_probs * (safe_student - safe_mixture), nan=0.0)
            return beta_t * kl_teacher + (1 - beta_t) * kl_student
    else:
        if beta == 0:
            return F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True)
        elif beta == 1:
            return F.kl_div(teacher_log_probs, student_log_probs, reduction="none", log_target=True)
        else:
            beta_t = torch.tensor(beta, dtype=student_log_probs.dtype, device=student_log_probs.device)
            mixture_log_probs = torch.logsumexp(
                torch.stack([student_log_probs + torch.log1p(-beta_t), teacher_log_probs + torch.log(beta_t)]),
                dim=0,
            )
            kl_teacher = F.kl_div(mixture_log_probs, teacher_log_probs, reduction="none", log_target=True)
            kl_student = F.kl_div(mixture_log_probs, student_log_probs, reduction="none", log_target=True)
            return beta_t * kl_teacher + (1 - beta_t) * kl_student


def _add_tail_bucket(log_probs, valid_mask):
    """Append a (K+1)-th tail element: log(1 - sum(exp(top_k_logps))).

    This creates a proper probability distribution over K+1 elements, preventing trivial zero loss when top_k is small
    (especially top_k=1).
    """
    log_sum = torch.logsumexp(log_probs, dim=-1, keepdim=True)
    log_sum = torch.clamp(log_sum, max=-1e-7)  # ensure sum < 1
    tail = torch.log(-torch.expm1(log_sum))  # log(1 - exp(log_sum))
    tail_mask = torch.ones_like(valid_mask[..., :1], dtype=torch.bool)
    return torch.cat([log_probs, tail], dim=-1), torch.cat([valid_mask, tail_mask], dim=-1)


def build_teacher_request_inputs(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_attention_mask: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
) -> tuple[list[list[int]], list[int], list[int]]:
    """Trim padded batch tensors into per-sample sequences for teacher-server requests."""

    if input_ids.shape != attention_mask.shape:
        raise ValueError(
            f"input_ids and attention_mask must have the same shape, got {input_ids.shape} and {attention_mask.shape}."
        )

    input_ids_cpu = input_ids.detach().cpu()
    attention_mask_cpu = attention_mask.detach().cpu().bool()

    if prompt_attention_mask is not None:
        prompt_lengths = prompt_attention_mask.detach().cpu().sum(dim=1).to(torch.long)
    else:
        if labels is None:
            raise ValueError("labels are required when prompt_attention_mask is not provided.")
        if labels.shape != input_ids.shape:
            raise ValueError(f"labels must match input_ids shape, got {labels.shape} and {input_ids.shape}.")
        full_lengths = attention_mask_cpu.sum(dim=1).to(torch.long)
        completion_lengths = (labels.detach().cpu() != -100).sum(dim=1).to(torch.long)
        prompt_lengths = full_lengths - completion_lengths

    trimmed_input_ids: list[list[int]] = []
    prompt_lengths_list: list[int] = []
    completion_lengths_list: list[int] = []

    for row, mask, prompt_length in zip(input_ids_cpu, attention_mask_cpu, prompt_lengths, strict=True):
        trimmed_row = row[mask]
        prompt_len = int(prompt_length.item())
        if prompt_len < 0 or prompt_len > trimmed_row.numel():
            raise ValueError(
                f"Invalid prompt length {prompt_len} for trimmed sequence of length {trimmed_row.numel()}."
            )
        trimmed_input_ids.append(trimmed_row.tolist())
        prompt_lengths_list.append(prompt_len)
        completion_lengths_list.append(int(trimmed_row.numel()) - prompt_len)

    return trimmed_input_ids, prompt_lengths_list, completion_lengths_list


class ServerDistillationTrainer(DistillationTrainer):
    """Distillation from a teacher hosted on an external vLLM server.

    Instead of running a local teacher forward pass, per-token teacher logprobs are fetched from a vLLM server via
    [`~generation.vllm_client.VLLMClient`]. The server only returns the teacher's top-k logprobs, so the divergence is
    restricted to a sparse support (top-1 for `beta > 0`, top-k for the pure forward path `beta = 0`). Everything else
    — the student forward, generation, buffering, metrics — is inherited from
    [`experimental.distillation.DistillationTrainer`].
    """

    _tag_names = ["trl", "server-distillation"]
    _name = "Server Distillation"

    def __init__(
        self,
        model=None,
        args: ServerDistillationConfig | None = None,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        processing_class=None,
        compute_metrics=None,
        callbacks=None,
        optimizers=(None, None),
        preprocess_logits_for_metrics=None,
        peft_config=None,
    ):
        if args is None:
            args = ServerDistillationConfig(output_dir="tmp_server_distillation")

        # No local teacher: the base sets `self.teacher_model = None`, then we attach the server client below.
        super().__init__(
            model=model,
            teacher_model=None,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            peft_config=peft_config,
        )

        self.reverse_kl_top_1_mode = args.reverse_kl_top_1_mode
        self.loss_top_k = args.loss_top_k
        self.loss_add_tail = args.loss_add_tail

        from ...generation.vllm_client import VLLMClient

        self.teacher_client = VLLMClient(base_url=args.teacher_model_server_url, connection_timeout=60.0)

    def _get_reverse_kl_top_1_tokens(
        self, student_scores: torch.Tensor, completion_tokens: torch.Tensor
    ) -> torch.Tensor:
        """Return the reverse-KL top-1 token IDs for the mixed top-1 loss path.

        Args:
            student_scores: Any (B, T, V) tensor whose argmax selects the student's top token
                (logits or log-probs — both are order-preserving).
            completion_tokens: (B, T) actual token IDs in the completion.
        """
        if self.reverse_kl_top_1_mode == "argmax":
            return student_scores.argmax(dim=-1)
        return completion_tokens

    def _compute_sparse_top_1_divergence_loss(
        self,
        student_log_probs: torch.Tensor,
        teacher_top1_token_ids: torch.Tensor,
        teacher_top1_logprobs: torch.Tensor,
        reverse_token_ids: torch.Tensor,
        reverse_teacher_logprobs: torch.Tensor,
        labels: torch.Tensor,
        num_items_in_batch=None,
    ) -> torch.Tensor:
        """Compute exact generalized JSD/KL on top-1 support for the mixed beta>0 path."""
        neg_inf = torch.full((), float("-inf"), dtype=student_log_probs.dtype, device=student_log_probs.device)

        if self.beta == 1:
            support = reverse_token_ids.unsqueeze(-1)
            support_mask = torch.ones_like(support, dtype=torch.bool)
            teacher_support_logprobs = reverse_teacher_logprobs.unsqueeze(-1)
        else:
            teacher_support = teacher_top1_token_ids.unsqueeze(-1)
            reverse_support = reverse_token_ids.unsqueeze(-1)
            support = torch.cat([teacher_support, reverse_support], dim=-1)
            support_mask = torch.ones_like(support, dtype=torch.bool)
            support_mask[..., 1] = support[..., 1] != support[..., 0]
            teacher_support_logprobs = torch.stack([teacher_top1_logprobs, reverse_teacher_logprobs], dim=-1)
            support = torch.where(support_mask, support, torch.zeros_like(support))

        student_support_logprobs = student_log_probs.gather(-1, support)
        student_support_logprobs = torch.where(support_mask, student_support_logprobs, neg_inf)
        teacher_support_logprobs = torch.where(support_mask, teacher_support_logprobs, neg_inf)

        if self.loss_add_tail:
            base_support_mask = support_mask
            student_sparse_log_probs, support_mask = _add_tail_bucket(student_support_logprobs, base_support_mask)
            teacher_sparse_log_probs, _ = _add_tail_bucket(teacher_support_logprobs, base_support_mask)
        else:
            student_sparse_log_probs = student_support_logprobs - torch.logsumexp(
                student_support_logprobs, dim=-1, keepdim=True
            )
            teacher_sparse_log_probs = teacher_support_logprobs - torch.logsumexp(
                teacher_support_logprobs, dim=-1, keepdim=True
            )

        jsd = _jsd_divergence(student_sparse_log_probs, teacher_sparse_log_probs, self.beta, support_mask)
        return self._reduce_divergence_loss(
            jsd, completion_mask=labels != -100, reduction="batchmean", num_items_in_batch=num_items_in_batch
        )

    def _compute_prompt_length(self, inputs: dict[str, torch.Tensor | Any]) -> int:
        """Compute the earliest prompt boundary that still includes every completion token in the batch."""
        if inputs.get("labels") is not None:
            attention_mask = inputs["attention_mask"]
            labels = inputs["labels"]
            full_lengths = attention_mask.sum(dim=1)
            completion_lengths = (labels != -100).sum(dim=1)
            return int((full_lengths - completion_lengths).min().item())
        return inputs["prompts"].shape[1]

    @staticmethod
    def _reduce_divergence_loss(jsd, completion_mask=None, reduction="batchmean", num_items_in_batch=None):
        """Reduce a per-token divergence tensor over the valid completion tokens.

        When `num_items_in_batch` is provided (as under gradient accumulation), the divergence is reduced as `sum /
        num_items_in_batch`, matching the gradient-accumulation-correct behavior of HF's default cross-entropy.
        Otherwise it falls back to the local `reduction` (default `batchmean`). See issue #4719.
        """
        mask = None
        if completion_mask is not None:
            mask = completion_mask.bool()
            jsd = jsd[mask]

        if num_items_in_batch is not None:
            # Normalize by the global number of valid tokens for gradient-accumulation-correct loss.
            jsd_sum = jsd.sum()
            if isinstance(num_items_in_batch, torch.Tensor):
                num_items_in_batch = num_items_in_batch.to(jsd_sum.device)
            return jsd_sum / num_items_in_batch
        if reduction == "batchmean":
            # clamp_min(1) avoids 0/0 -> nan when a sample has no unmasked positions
            # (e.g. completion fully truncated). jsd[mask] is empty -> jsd.sum() == 0,
            # so 0/1 == 0 with a valid grad path.
            denom = mask.sum().clamp_min(1) if completion_mask is not None else max(jsd.size(0), 1)
            return jsd.sum() / denom
        elif reduction == "sum":
            return jsd.sum()
        elif reduction == "mean":
            return jsd.mean()
        else:
            return jsd

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Reconstruct the old `input_ids` / `attention_mask` / `labels` layout this server path is built on from GRPO's
        # keys, so it rides on the base's (evolving) generation. The teacher-server loss is migrated onto the stable
        # base wholesale later (issue #6449, item 58). The first four lines mirror `DistillationTrainer.compute_loss`.
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        # Server-only: the teacher-server loss masks on `labels`/`prompt_attention_mask`, so rebuild them too.
        completion_labels = torch.where(completion_mask.bool(), completion_ids, torch.full_like(completion_ids, -100))
        labels = torch.cat([torch.full_like(prompt_ids, -100), completion_labels], dim=1)
        inputs = {
            **inputs,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "prompt_attention_mask": prompt_mask,
        }

        # Student forward pass
        student_outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        prompt_length = self._compute_prompt_length(inputs)
        labels = inputs["labels"][:, prompt_length:]
        completion_tokens = inputs["input_ids"][:, prompt_length:]

        # Server path: token-level divergence using teacher logprobs.
        # The server returns:
        #   actual_logprobs  – (B, T)    teacher log p(x_actual)  (for reverse KL)
        #   topk_logprobs    – (B, T, K) teacher top-k sorted logprobs (for forward KL)
        #   topk_token_ids   – (B, T, K) corresponding token IDs
        teacher_result = self._get_teacher_token_logprobs_from_server(inputs, prompt_length)

        student_logits = student_outputs.logits[:, prompt_length - 1 : -1, :]
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)

        comp_len = teacher_result["actual_logprobs"].shape[1]
        completion_tokens = completion_tokens[:, :comp_len]
        trimmed_labels = labels[:, :comp_len]

        if self.beta > 0:
            loss = self._compute_server_sparse_top_1_divergence_loss(
                teacher_result=teacher_result,
                student_log_probs=student_log_probs[:, :comp_len, :],
                completion_tokens=completion_tokens,
                labels=trimmed_labels,
            )
        else:
            loss = self._compute_server_forward_kl_loss(
                teacher_result=teacher_result,
                student_log_probs=student_log_probs[:, :comp_len, :],
                labels=trimmed_labels,
            )

        # The base trainer disables Trainer's built-in grad-accum loss scaling (via `compute_loss_func`) because it
        # normalizes by the global completion-token count. The server path normalizes locally with `batchmean` and does
        # not consume `num_items_in_batch`, so it must re-apply that scaling itself.
        if self.model.training:
            loss = loss / self.current_gradient_accumulation_steps

        return (loss, student_outputs) if return_outputs else loss

    def _get_teacher_token_logprobs_from_server(
        self,
        inputs: dict[str, torch.Tensor | Any],
        aligned_prompt_length: int,
    ) -> dict[str, torch.Tensor]:
        """Fetch per-token teacher logprobs from an external vLLM server.

        Returns a dict with:
            ``actual_logprobs`` - (batch, completion_length) teacher log-prob for the actual
                                   token at each position (for reverse KL).
            ``topk_logprobs`` - (batch, completion_length, K) teacher top-k sorted logprobs
                                   (for forward KL).
            ``topk_token_ids`` - (batch, completion_length, K) corresponding token IDs.
        """
        import numpy as np

        input_ids = inputs["input_ids"]
        batch_size = input_ids.shape[0]
        sequences, prompt_lengths, completion_lengths = build_teacher_request_inputs(
            input_ids,
            inputs["attention_mask"],
            prompt_attention_mask=inputs.get("prompt_attention_mask"),
            labels=inputs.get("labels"),
        )

        # The pure forward server path can use the requested teacher top-k support.
        # When beta > 0, config validation restricts the server-backed path to top-1.
        requested_top_k = self.loss_top_k
        result = self.teacher_client.get_sequence_logprobs(
            sequences=sequences,
            prompt_lengths=prompt_lengths,
            top_logprobs=requested_top_k,
            temperature=self.temperature,
        )
        K = requested_top_k

        device = input_ids.device
        labels = inputs.get("labels")
        if labels is None:
            raise ValueError("labels are required to align teacher-server logprobs with the student loss tensors.")

        # The student loss slices tensors in padded-sequence coordinates starting at `aligned_prompt_length`.
        # Place each teacher completion into that same coordinate system by locating the first non-masked completion
        # token in `labels`. This works for both left-padded off-policy batches and on-policy batches where
        # completions are right-padded after a fixed-width prompt block.
        completion_offsets = []
        label_mask = labels != -100
        for sample_mask, comp_len in zip(label_mask, completion_lengths, strict=True):
            if comp_len == 0:
                completion_offsets.append(0)
                continue
            completion_start = int(torch.nonzero(sample_mask, as_tuple=False)[0].item())
            completion_offsets.append(completion_start - aligned_prompt_length)

        # Size the output tensors to tightly fit the teacher logprobs. Using the full padded
        # sequence length would include padding positions with -inf teacher logprobs, producing
        # +inf in the forward pass and NaN gradients in the backward pass (0 * inf = NaN).
        # Shorter samples in variable-length batches still need the -inf sentinel at the tail;
        # downstream loss consumers (_compute_server_sparse_top_1_divergence_loss,
        # _compute_server_forward_kl_loss) neutralise those positions before the divergence
        # math runs.
        completion_length = max(
            (offset + len(lps) for offset, lps in zip(completion_offsets, result["logprobs"], strict=True)),
            default=0,
        )

        # actual_logprobs: (B, T) - teacher logprob for the actual token
        def _actual_to_tensor(key):
            arr = np.full((batch_size, completion_length), float("-inf"), dtype=np.float32)
            for i, (offset, seq_lps) in enumerate(zip(completion_offsets, result[key], strict=True)):
                if seq_lps:
                    vals = np.array(seq_lps, dtype=np.float32)  # (comp_len_i, 1)
                    arr[i, offset : offset + vals.shape[0]] = vals[:, 0]
            return torch.from_numpy(arr).to(device)

        # topk: (B, T, K)
        def _topk_to_tensor(key, k, np_dtype, fill):
            arr = np.full((batch_size, completion_length, k), fill, dtype=np_dtype)
            for i, (offset, seq_vals) in enumerate(zip(completion_offsets, result[key], strict=True)):
                if seq_vals:
                    vals = np.array(seq_vals, dtype=np_dtype)  # (comp_len_i, k)
                    arr[i, offset : offset + vals.shape[0], :] = vals
            return torch.from_numpy(arr).to(device)

        return {
            "actual_logprobs": _actual_to_tensor("actual_logprobs"),
            "topk_logprobs": _topk_to_tensor("logprobs", K, np.float32, float("-inf")),
            "topk_token_ids": _topk_to_tensor("logprob_token_ids", K, np.int64, 0),
        }

    def _compute_server_sparse_top_1_divergence_loss(
        self,
        teacher_result: dict[str, torch.Tensor],
        student_log_probs: torch.Tensor,
        completion_tokens: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute exact sparse top-1 generalized JSD/KL from server-provided teacher logprobs.

        Args:
            teacher_result: dict with ``actual_logprobs`` (B, T), ``topk_logprobs`` (B, T, K),
                ``topk_token_ids`` (B, T, K).
            student_log_probs: (B, T, V) student log-softmax over vocabulary.
            completion_tokens: (B, T) actual token IDs in the completion.
            labels: (B, T) with -100 for positions to ignore.
        """
        topk_teacher_lps = teacher_result["topk_logprobs"]  # (B, T, 1)
        topk_token_ids = teacher_result["topk_token_ids"]  # (B, T, 1)
        actual_teacher_lps = teacher_result["actual_logprobs"]  # (B, T)
        required = labels != -100

        missing_actual = required & ~torch.isfinite(actual_teacher_lps)
        if missing_actual.any():
            missing_count = int(missing_actual.sum().item())
            total_required = int(required.sum().item())
            raise ValueError(
                "Teacher server is missing actual-token logprobs for required reverse-KL positions: "
                f"{missing_count}/{total_required}."
            )
        if self.beta < 1:
            teacher_top1_logprobs = topk_teacher_lps.squeeze(-1)
            missing_top1 = required & ~torch.isfinite(teacher_top1_logprobs)
            if missing_top1.any():
                missing_count = int(missing_top1.sum().item())
                total_required = int(required.sum().item())
                raise ValueError(
                    "Teacher server is missing top-1 logprobs for required forward-KL positions: "
                    f"{missing_count}/{total_required}."
                )

        # Replace -inf teacher logprobs at intra-batch padding (labels == -100) with 0 so
        # reverse-KL's student_probs·(log_s - log_t) does not leak +inf into the backward pass.
        pad_mask_2d = ~required
        pad_mask_3d = pad_mask_2d.unsqueeze(-1)
        topk_teacher_lps = torch.where(pad_mask_3d, 0.0, topk_teacher_lps)
        actual_teacher_lps = torch.where(pad_mask_2d, 0.0, actual_teacher_lps)

        # Server path only supports "sampled" mode — config validation enforces this, but we guard
        # explicitly so future relaxations of the config check don't silently change behaviour.
        reverse_token_ids = self._get_reverse_kl_top_1_tokens(student_log_probs, completion_tokens)
        # The server path normalizes locally (batchmean), not by num_items_in_batch: teacher logprobs may not cover
        # every student completion token (the loss is summed over the trimmed teacher window), so the global token
        # count would be the wrong denominator. Gradient-accumulation normalization for the server path is left as a
        # follow-up.
        return self._compute_sparse_top_1_divergence_loss(
            student_log_probs=student_log_probs,
            teacher_top1_token_ids=topk_token_ids.squeeze(-1),
            teacher_top1_logprobs=topk_teacher_lps.squeeze(-1),
            reverse_token_ids=reverse_token_ids,
            reverse_teacher_logprobs=actual_teacher_lps,
            labels=labels,
        )

    def _compute_server_forward_kl_loss(
        self,
        teacher_result: dict[str, torch.Tensor],
        student_log_probs: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute sparse forward KL from server-provided teacher top-k logprobs (beta==0 path).

        Args:
            teacher_result: dict with ``topk_logprobs`` (B, T, K) and ``topk_token_ids`` (B, T, K).
            student_log_probs: (B, T, V) student log-softmax over vocabulary.
            labels: (B, T) with -100 for positions to ignore.
        """
        teacher_topk_logprobs = teacher_result["topk_logprobs"]
        teacher_topk_token_ids = teacher_result["topk_token_ids"]
        valid = teacher_topk_logprobs > float("-inf")
        neg_inf = torch.full((), float("-inf"), dtype=student_log_probs.dtype, device=student_log_probs.device)
        student_topk_logprobs = student_log_probs.gather(dim=-1, index=teacher_topk_token_ids)
        student_topk_logprobs = torch.where(valid, student_topk_logprobs, neg_inf)
        teacher_topk_logprobs = torch.where(valid, teacher_topk_logprobs, neg_inf)

        if self.loss_add_tail:
            base_support_mask = valid
            student_sparse_log_probs, support_mask = _add_tail_bucket(student_topk_logprobs, base_support_mask)
            teacher_sparse_log_probs, _ = _add_tail_bucket(teacher_topk_logprobs, base_support_mask)
        else:
            support_mask = valid
            student_sparse_log_probs = student_topk_logprobs - torch.logsumexp(
                student_topk_logprobs, dim=-1, keepdim=True
            )
            teacher_sparse_log_probs = teacher_topk_logprobs - torch.logsumexp(
                teacher_topk_logprobs, dim=-1, keepdim=True
            )

        jsd = _jsd_divergence(
            student_sparse_log_probs,
            teacher_sparse_log_probs,
            beta=0.0,
            support_mask=support_mask,
        )
        # See `_compute_server_sparse_top_1_divergence_loss`: the server path normalizes locally, not by
        # num_items_in_batch, because the teacher window may not cover every student completion token.
        return self._reduce_divergence_loss(jsd, completion_mask=labels != -100, reduction="batchmean")
