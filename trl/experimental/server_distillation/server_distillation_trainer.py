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

from ..distillation.distillation_trainer import DistillationTrainer, _add_tail_bucket, _jsd_divergence
from .server_distillation_config import ServerDistillationConfig


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
    — the student forward, generation, buffering, metrics — is inherited from [`experimental.distillation.DistillationTrainer`].
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

        from ...generation.vllm_client import VLLMClient

        self.teacher_client = VLLMClient(base_url=args.teacher_model_server_url, connection_timeout=60.0)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
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
        return self._reduce_divergence_loss(jsd, labels=labels, reduction="batchmean")
