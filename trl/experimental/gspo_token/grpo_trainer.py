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

import math

import torch

from ...trainer.grpo_trainer import GRPOTrainer as _GRPOTrainer
from ...trainer.utils import nanmax, nanmin


class GRPOTrainer(_GRPOTrainer):
    def _compute_loss(self, model, inputs):
        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        mask = completion_mask if "tool_mask" not in inputs else completion_mask * inputs["tool_mask"]

        # Compute the per_token_logps and the entropy at each position in the completion
        per_token_logps, entropies, _ = self._get_per_token_logps_and_entropies(
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            compute_entropy=True,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            num_images=inputs.get("num_images"),
            pixel_attention_mask=inputs.get("pixel_attention_mask"),
            image_sizes=inputs.get("image_sizes"),
            token_type_ids=inputs.get("token_type_ids"),
        )

        if self.top_entropy_quantile < 1.0:
            entropy_mask = self.get_high_entropy_mask(entropies, mask, 1 - self.top_entropy_quantile)
        else:
            entropy_mask = None

        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        # Compute the loss
        advantages = inputs["advantages"]
        # In the base GRPO implementation, advantages are expected to have shape (B,). To support subclasses that
        # provide advantages with shape (B, T) (e.g., MiniLLM), we *conditionally* unsqueeze the tensor.
        if advantages.dim() == 1:
            advantages = advantages.unsqueeze(1)
        # When gradient_accumulation_steps % generate_every == 0 (on-policy),
        # old_per_token_logps == per_token_logps on the first iteration. In this case we can skip
        # its computation (see _generate_and_score_completions) and instead use per_token_logps.detach().
        # The exception is the liger kernel path with vLLM importance sampling correction,
        # where old_per_token_logps is always pre-computed.
        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps

        # Compute inline IS ratio for non-liger vLLM path. This must happen before the loss type switch
        # because vespo needs it in get_gamma_weights. For the liger path, IS ratio is pre-computed in
        # _generate_and_score_completions and passed via inputs["importance_sampling_ratio"]. Store
        # old_per_token_logps and the ratio back into inputs so that subsequent iterations (num_iterations > 1)
        # reuse the same generation-time values.
        vllm_importance_sampling_ratio = None
        if self.use_vllm and self.vllm_importance_sampling_correction:
            sampling_per_token_logps = inputs.get("sampling_per_token_logps")
            if sampling_per_token_logps is not None and inputs.get("importance_sampling_ratio") is None:
                inputs["old_per_token_logps"] = old_per_token_logps
                per_token_logps_diff = (old_per_token_logps - sampling_per_token_logps) * mask
                # Tokens whose sampling logprob was NaN (unavailable from vLLM) get a zero difference, so their
                # importance ratio is exactly 1 (no correction) rather than propagating NaN through the loss.
                per_token_logps_diff = torch.nan_to_num(per_token_logps_diff, nan=0.0)

                sequence_level_is = self.vllm_importance_sampling_mode in ["sequence_mask", "sequence_truncate"]
                if sequence_level_is:
                    per_sequence_logps_diff = per_token_logps_diff.sum(dim=-1, keepdim=True)
                    logps_diff = per_sequence_logps_diff
                else:
                    logps_diff = per_token_logps_diff

                vllm_importance_sampling_ratio = torch.exp(logps_diff)

                # vllm_importance_sampling_ratio.shape:
                #   token_* modes:     (B, T)  (per-token ratio)
                #   sequence_* modes:  (B, 1)  (per-sequence ratio)

                if self.vllm_importance_sampling_mode in ["sequence_truncate", "token_truncate"]:
                    vllm_importance_sampling_ratio = torch.clamp(
                        vllm_importance_sampling_ratio,
                        min=self.vllm_importance_sampling_clip_min,
                        max=self.vllm_importance_sampling_clip_max,
                    )
                elif self.vllm_importance_sampling_mode in ["sequence_mask", "token_mask"]:
                    min_val = (
                        self.vllm_importance_sampling_clip_min
                        if self.vllm_importance_sampling_clip_min is not None
                        else -math.inf
                    )
                    max_val = (
                        self.vllm_importance_sampling_clip_max
                        if self.vllm_importance_sampling_clip_max is not None
                        else math.inf
                    )

                    invalid_mis_mask = (vllm_importance_sampling_ratio < min_val) | (
                        vllm_importance_sampling_ratio > max_val
                    )
                    vllm_importance_sampling_ratio = vllm_importance_sampling_ratio.masked_fill(
                        invalid_mis_mask, value=0.0
                    )
                else:
                    raise ValueError(
                        f"Unknown vLLM importance sampling level: {self.vllm_importance_sampling_mode}. Possible values are 'token_truncate', 'token_mask', 'sequence_truncate', and 'sequence_mask'."
                    )

                inputs["importance_sampling_ratio"] = vllm_importance_sampling_ratio

        log_ratio = per_token_logps - old_per_token_logps
        if self.importance_sampling_level == "token":
            log_importance_weights = log_ratio
        elif self.importance_sampling_level == "sequence":
            log_importance_weights = (log_ratio * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)
            log_importance_weights = log_importance_weights.unsqueeze(-1)
        elif self.importance_sampling_level == "sequence_token":
            # GSPO-token: sg[si(θ)] * πθ(yi,t)/sg[πθ(yi,t)]
            seq_level_log_weight = (log_ratio * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)
            seq_level_log_weight = seq_level_log_weight.detach().unsqueeze(-1)  # Stop gradient
            log_importance_weights = per_token_logps - per_token_logps.detach() + seq_level_log_weight
        else:
            raise ValueError(
                f"Unknown importance sampling level: {self.importance_sampling_level}. Possible values are 'token' "
                "and 'sequence'."
            )
        # From here, log_importance_weights (and all subsequent tensors, coef_1, coef_2, etc.) shape depends on
        # importance_sampling_level: "token" level: (B, T); "sequence" level: (B, 1)

        coef_1 = torch.exp(log_importance_weights)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)

        # Two-sided clipping
        if self.args.delta is not None:
            coef_1 = torch.clamp(coef_1, max=self.args.delta)

        per_token_loss1 = coef_1 * advantages
        per_token_loss2 = coef_2 * advantages
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if entropy_mask is not None:
            per_token_loss = per_token_loss * entropy_mask

        if self.use_vllm and self.vllm_importance_sampling_correction:
            per_token_loss = per_token_loss * inputs["importance_sampling_ratio"]

        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        mode = "train" if self.model.training else "eval"
        if self.loss_type == "grpo":
            loss = ((per_token_loss * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()
            normalizer = self.current_gradient_accumulation_steps if mode == "train" else 1.0  # no accum in eval
            loss = loss / normalizer
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * mask).sum() / mask.sum().clamp(min=1.0)
            normalizer = self.current_gradient_accumulation_steps if mode == "train" else 1.0  # no accum in eval
            loss = loss / normalizer
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
            normalizer = self.current_gradient_accumulation_steps if mode == "train" else 1.0  # no accum in eval
            loss = loss / normalizer
        elif self.loss_type == "dapo":
            # `num_items_in_batch` spans the generation batch, so rescale it to one accumulation window
            normalizer = inputs["num_items_in_batch"].clamp(min=1.0) / self.accelerator.num_processes
            if mode == "train":  # in eval, the batch is neither split across steps nor accumulated
                normalizer = normalizer * self.current_gradient_accumulation_steps / self.args.steps_per_generation
            loss = (per_token_loss * mask).sum() / normalizer
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Log the metrics
        def masked_seq_mean(x):
            if x.shape[1] == 1:  # when importance_sampling_level == "sequence": already one value per sequence
                return x.squeeze(1)
            return (x * mask).sum(-1) / mask.sum(-1)

        def global_masked_mean(x):
            if x.shape[1] == 1:  # when importance_sampling_level == "sequence": one value per sequence
                local_sum, local_count = x.sum(), torch.tensor(float(x.shape[0]), device=x.device)
            else:
                local_sum, local_count = (x * mask).sum(), mask.sum().float()
            totals = self.accelerator.reduce(torch.stack([local_sum, local_count]), reduction="sum")
            return (totals[0] / totals[1].clamp(min=1.0)).item()

        if self.beta != 0.0:
            self._metrics[mode]["kl"].append(global_masked_mean(per_token_kl))

        self._metrics[mode]["entropy"].append(global_masked_mean(entropies))

        # Compute the clipped probability ratios
        is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages < 0)
        is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages > 0)
        is_region_clipped = is_low_clipped | is_high_clipped
        self._metrics[mode]["clip_ratio/low_mean"].append(global_masked_mean(is_low_clipped.float()))
        self._metrics[mode]["clip_ratio/high_mean"].append(global_masked_mean(is_high_clipped.float()))
        self._metrics[mode]["clip_ratio/region_mean"].append(global_masked_mean(is_region_clipped.float()))
        gathered_low_clip = self.accelerator.gather(masked_seq_mean(is_low_clipped.float()))
        self._metrics[mode]["clip_ratio/low_min"].append(nanmin(gathered_low_clip).item())
        gathered_high_clip = self.accelerator.gather(masked_seq_mean(is_high_clipped.float()))
        self._metrics[mode]["clip_ratio/high_max"].append(nanmax(gathered_high_clip).item())
        # The vLLM importance sampling diagnostics are logged once per generation batch: on the liger path by
        # _generate_and_score_completions, and on the non-liger path here, once the last micro-batch of the generation
        # batch has been through the loss, over the generation batch reassembled from the buffered micro-batches
        # (each of which stored its old_per_token_logps and ratio in inputs above).
        if vllm_importance_sampling_ratio is not None:
            batches = self._buffered_inputs if mode == "train" else [inputs]
            if all("importance_sampling_ratio" in batch for batch in batches):
                old_per_token_logps = torch.cat([batch["old_per_token_logps"] for batch in batches])
                sampling_per_token_logps = torch.cat([batch["sampling_per_token_logps"] for batch in batches])
                vllm_importance_sampling_ratio = torch.cat([batch["importance_sampling_ratio"] for batch in batches])
                completion_mask = torch.cat([batch["completion_mask"] for batch in batches])
                tool_mask = torch.cat([batch["tool_mask"] for batch in batches]) if "tool_mask" in inputs else None
                device = self.accelerator.device
                delta = torch.abs(old_per_token_logps - sampling_per_token_logps)
                mask = completion_mask.bool() if tool_mask is None else (completion_mask * tool_mask).bool()
                # Tokens vLLM could not score carry NaN, so exclude them rather than let them turn the reported
                # divergence into NaN. Counting them as zero instead would understate the divergence.
                delta = delta[mask & ~torch.isnan(delta)]
                mean_delta = torch.mean(delta) if delta.numel() > 0 else torch.tensor(0.0, device=device)
                max_delta = torch.max(delta) if delta.numel() > 0 else torch.tensor(0.0, device=device)
                self._metrics[mode]["sampling/sampling_logp_difference/mean"].append(
                    self.accelerator.gather(mean_delta).mean().item()
                )
                self._metrics[mode]["sampling/sampling_logp_difference/max"].append(
                    self.accelerator.gather(max_delta).max().item()
                )
                if sequence_level_is:
                    flat_is_ratio = vllm_importance_sampling_ratio.flatten()
                else:
                    flat_is_ratio = vllm_importance_sampling_ratio[mask]

                min_importance_sampling_ratio = (
                    torch.min(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
                )
                mean_importance_sampling_ratio = (
                    torch.mean(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
                )
                max_importance_sampling_ratio = (
                    torch.max(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
                )
                self._metrics[mode]["sampling/importance_sampling_ratio/min"].append(
                    nanmin(self.accelerator.gather(min_importance_sampling_ratio)).item()
                )
                self._metrics[mode]["sampling/importance_sampling_ratio/mean"].append(
                    self.accelerator.gather(mean_importance_sampling_ratio).nanmean().item()
                )
                self._metrics[mode]["sampling/importance_sampling_ratio/max"].append(
                    nanmax(self.accelerator.gather(max_importance_sampling_ratio)).item()
                )

        return loss
