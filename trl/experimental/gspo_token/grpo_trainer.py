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
        per_token_logps, entropies, aux_loss = self._get_per_token_logps_and_entropies(
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            compute_entropy=True,
            compute_aux_loss=self.aux_loss_enabled,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            num_images=inputs.get("num_images"),
            pixel_attention_mask=inputs.get("pixel_attention_mask"),
            spatial_shapes=inputs.get("spatial_shapes"),
            num_tiles=inputs.get("num_tiles"),
            image_sizes=inputs.get("image_sizes"),
            token_type_ids=inputs.get("token_type_ids"),
            mm_token_type_ids=inputs.get("mm_token_type_ids"),
            image_position_ids=inputs.get("image_position_ids"),
        )

        if self.top_entropy_quantile < 1.0:
            entropy_mask = self.get_high_entropy_mask(entropies, mask, 1 - self.top_entropy_quantile)
        else:
            entropy_mask = None

        # Compute the loss
        advantages = inputs["advantages"]
        # In the base GRPO implementation, advantages are expected to have shape (B,). To support subclasses that
        # provide advantages with shape (B, T) (e.g., MiniLLM), we *conditionally* unsqueeze the tensor.
        if advantages.dim() == 1:
            advantages = advantages.unsqueeze(1)
        # When num_iterations == 1 and steps_per_generation <= gradient_accumulation_steps,
        # old_per_token_logps == per_token_logps. In this case we can skip its computation
        # (see _generate_and_score_completions) and instead use per_token_logps.detach().
        # The exception is when using vLLM, where we always compute old_per_token_logps
        # for importance sampling
        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps

        if self.off_policy_mask_threshold is not None:
            # OPSM should use inference-time logprobs to detect both sources of off-policyness:
            # 1. Drift from gradient updates (always present)
            # 2. Drift from training-inference mismatch (when using vLLM)
            # When using vLLM, prioritize sampling_per_token_logps, otherwise use old_per_token_logps
            sampling_per_token_logps = inputs.get("sampling_per_token_logps", old_per_token_logps)

            off_policy_mask = self.get_off_policy_mask(
                advantages=advantages,
                per_token_logps=per_token_logps,
                sampling_per_token_logps=sampling_per_token_logps,
                mask=mask,
                off_policy_threshold=self.off_policy_mask_threshold,
            )

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
                f"Unknown importance sampling level: {self.importance_sampling_level}. Possible values are "
                "'token', 'sequence' and 'sequence_token'."
            )

        coef_1 = torch.exp(log_importance_weights)

        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )
            # Importance sampling correction for the KL divergence
            if self.args.use_bias_correction_kl:
                per_token_kl = per_token_kl * coef_1

        # From here, log_importance_weights (and all subsequent tensors, coef_1, coef_2, etc.) shape depends on
        # importance_sampling_level: "token" level: (B, T); "sequence" level: (B, 1)
        if self.loss_type == "cispo":
            clamped_ratios = torch.clamp(coef_1, max=self.epsilon_high).detach()
            per_token_loss = -clamped_ratios * advantages * per_token_logps
        elif self.loss_type in ["grpo", "bnpo", "dr_grpo", "dapo", "luspo"]:
            coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
            # Two-sided clipping
            if self.args.delta is not None:
                coef_1 = torch.clamp(coef_1, max=self.args.delta)

            per_token_loss1 = coef_1 * advantages
            per_token_loss2 = coef_2 * advantages
            per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        elif self.loss_type == "sapo":
            temperatures = torch.where(advantages > 0, self.args.sapo_temperature_pos, self.args.sapo_temperature_neg)
            soft_coef_1 = torch.sigmoid(temperatures * (coef_1 - 1)) * 4 / temperatures
            per_token_loss = -soft_coef_1 * advantages
        elif self.loss_type == "vespo":
            phi_seq = self.get_gamma_weights(
                advantages=advantages,
                log_ratio_per_token=log_ratio,
                mask=mask,
                importance_sampling_ratio=inputs.get("importance_sampling_ratio"),
                k_pos=self.args.vespo_k_pos,
                lambda_pos=self.args.vespo_lambda_pos,
                k_neg=self.args.vespo_k_neg,
                lambda_neg=self.args.vespo_lambda_neg,
            )
            per_token_loss = -phi_seq * advantages * per_token_logps
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        if self.off_policy_mask_threshold is not None:
            per_token_loss = per_token_loss * off_policy_mask

        if entropy_mask is not None:
            per_token_loss = per_token_loss * entropy_mask

        if self.use_vllm and self.vllm_importance_sampling_correction and self.loss_type != "vespo":
            per_token_loss = per_token_loss * inputs["importance_sampling_ratio"]

        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        mode = "train" if self.model.training else "eval"
        if self.loss_type in ["grpo", "sapo"]:
            loss = ((per_token_loss * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()
            normalizer = self.current_gradient_accumulation_steps if mode == "train" else 1.0  # no accum in eval
            policy_loss = loss.detach()
            loss = loss / normalizer
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * mask).sum() / mask.sum().clamp(min=1.0)
            normalizer = self.current_gradient_accumulation_steps if mode == "train" else 1.0  # no accum in eval
            policy_loss = loss.detach()
            loss = loss / normalizer
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
            normalizer = self.current_gradient_accumulation_steps if mode == "train" else 1.0  # no accum in eval
            policy_loss = loss.detach()
            loss = loss / normalizer
        elif self.loss_type in ["cispo", "dapo", "vespo"]:
            # `num_items_in_batch` spans the generation batch, so rescale it to one accumulation window
            normalizer = inputs["num_items_in_batch"].clamp(min=1.0) / self.accelerator.num_processes
            if mode == "train":  # in eval, the batch is neither split across steps nor accumulated
                normalizer = normalizer * self.current_gradient_accumulation_steps / self.args.steps_per_generation
            loss = (per_token_loss * mask).sum() / normalizer
            policy_loss = loss.detach()
        elif self.loss_type == "luspo":
            # `per_token_loss` is (B, 1) only in the recommended sequence-level setup; importance_sampling_level=
            # "token" (the config default), the KL term, token-level vLLM IS ratios, and the entropy mask all
            # broadcast it to (B, T), so mask before aggregating.
            loss = (per_token_loss * mask).sum(-1).mean()
            normalizer = self.current_gradient_accumulation_steps if mode == "train" else 1.0
            policy_loss = loss.detach()
            loss = loss / normalizer
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Entropy bonus: add entropy regularization to encourage exploration. _entropy_bonus_enabled is set
        # whenever a non-zero static coef is set OR adaptive mode is enabled (adaptive stays enabled even when
        # entropy_coef has been decremented to entropy_coef_min so it can recover once entropy drops again).
        if self._entropy_bonus_enabled:
            # When top_entropy_quantile < 1.0, entropy_mask restricts policy gradients to high-entropy
            # tokens. Use the same effective mask for the entropy bonus so it acts on the same tokens.
            effective_mask = mask if entropy_mask is None else mask * entropy_mask
            # Entropy bonus = mean per-token entropy H (the documented objective L = L_policy - coef * H), so
            # H does not depend on how each loss type normalizes its policy term. The bonus is a mean over the
            # tokens it acts on (effective_mask), scaled only for gradient accumulation, never by a loss-type-
            # specific policy normalizer. (The adaptive controller below tracks a window-global token-weighted mean,
            # which can differ from this per-micro-batch mean when token counts vary across micro-batches.)
            accumulation_factor = self.current_gradient_accumulation_steps if mode == "train" else 1.0
            entropy_loss = (
                (entropies * effective_mask).sum() / effective_mask.sum().clamp(min=1.0) / accumulation_factor
            )

            # Apply the coefficient and gating from the end of the previous optimizer step, so that every
            # micro-batch in the current accumulation window applies the same entropy bonus. The adaptive
            # update below only takes effect on the next step.
            if self.use_adaptive_entropy:
                apply_coef = self.entropy_coef if self._last_world_entropy <= self.args.entropy_target else 0.0
            else:
                apply_coef = self.entropy_coef

            loss = loss - apply_coef * entropy_loss

            self._metrics[mode]["policy_loss"].append(self.accelerator.gather(policy_loss).nanmean().item())

            # Adaptive update. Gated on train mode so evaluation cannot mutate the entropy controller state.
            if self.use_adaptive_entropy and mode == "train":
                # Accumulate the entropy sum and active-token count of every micro-batch into a running window
                # buffer, so the controller measures the exact window-global entropy rather than just the last
                # micro-batch (which would be a 1 / gradient_accumulation_steps subsample).
                stats = torch.stack([(entropies * effective_mask).sum(), effective_mask.sum()]).detach()
                if self._entropy_window_stats is None:
                    self._entropy_window_stats = stats
                else:
                    self._entropy_window_stats = self._entropy_window_stats + stats
                # At the optimizer-step boundary, reduce the window totals across ranks (sum and token count
                # jointly, for a true global mean unbiased when ranks have different completion lengths),
                # update the coefficient for the next step, then reset the window buffer.
                if self.accelerator.sync_gradients:
                    window_stats = self.accelerator.reduce(self._entropy_window_stats, reduction="sum")
                    world_entropy = (window_stats[0] / window_stats[1].clamp(min=1.0)).item()
                    if world_entropy <= self.args.entropy_target:
                        self.entropy_coef = min(
                            self.entropy_coef + self.args.entropy_coef_delta, self.args.entropy_coef_max
                        )
                    else:
                        self.entropy_coef = max(
                            self.entropy_coef - self.args.entropy_coef_delta, self.args.entropy_coef_min
                        )
                    self._last_world_entropy = world_entropy
                    self._entropy_window_stats = None

            # Log entropy_coef on train optimizer-step boundaries (constant for static control; updated just
            # above for adaptive control). sync_gradients is always True in eval (no accumulation context).
            if mode == "train" and self.accelerator.sync_gradients:
                self._metrics[mode]["entropy_coef"].append(self.entropy_coef)

        # The policy loss above is scaled for gradient accumulation (HF auto-scaling is off here), so scale aux too
        if self.aux_loss_enabled:
            normalizer = self.current_gradient_accumulation_steps if mode == "train" else 1.0
            loss = loss + self.router_aux_loss_coef * aux_loss / normalizer
            self._metrics[mode]["aux_loss"].append(self.accelerator.gather_for_metrics(aux_loss).mean().item())

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

        if self.loss_type in ["grpo", "bnpo", "dr_grpo", "dapo", "luspo"]:
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
        elif self.loss_type == "cispo":
            is_cispo_clipped = (coef_1 > self.epsilon_high) & (advantages > 0)
            self._metrics[mode]["cispo_clip_ratio"].append(global_masked_mean(is_cispo_clipped.float()))
        elif self.loss_type == "vespo":
            self._metrics[mode]["vespo/phi_seq_mean"].append(global_masked_mean(phi_seq))

        return loss
