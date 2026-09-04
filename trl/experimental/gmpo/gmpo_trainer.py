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

from ...trainer.grpo_trainer import GRPOTrainer
from ...trainer.utils import get_config_model_id, nanmax, nanmin
from .gmpo_config import GMPOConfig


class GMPOTrainer(GRPOTrainer):
    """
    Trainer for Geometric-Mean Policy Optimization (GMPO).

    GMPO (https://huggingface.co/papers/2507.20673) is a GRPO variant that maximizes the *geometric* mean of the
    token-level importance ratios instead of the arithmetic mean. Because the geometric mean is far less sensitive to
    outlier ratios, the policy update is more stable and a much wider clipping range can be used.

    The only change w.r.t. [`GRPOTrainer`] is `_compute_loss`. Everything else (generation, reward computation, weight
    syncing, metric logging) is inherited unchanged
    """

    _tag_names = ["trl", "gmpo"]

    def __init__(self, model, reward_funcs, args=None, **kwargs):
        if args is None:
            model_name = model if isinstance(model, str) else get_config_model_id(model.config)
            args = GMPOConfig(f"{model_name.split('/')[-1]}-GMPO")
        if args.use_liger_kernel:
            raise NotImplementedError("Liger kernel is not supported by GMPOTrainer.")

        super().__init__(model, reward_funcs, args=args, **kwargs)

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

        advantages = inputs["advantages"]
        # When num_iterations == 1 and steps_per_generation <= gradient_accumulation_steps,
        # old_per_token_logps == per_token_logps, so we skip its computation and use per_token_logps.detach() instead.
        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps

        # GMPO Objective
        # Per-token log importance ratio
        log_ratio = per_token_logps - old_per_token_logps

        # Token-level clipping, performed in *log space* for numerical stability. The clip range in
        # ratio space is (exp(-epsilon_low), exp(epsilon_high)); the paper recommends exp(±0.4), markedly wider than
        # GRPO/DAPO, to encourage exploration.
        clamped_log_ratio = torch.clamp(log_ratio, min=-self.epsilon_low, max=self.epsilon_high)

        # sign-aware, one-sided clipping = PPO's trust-region "min" trick written in log-spaces:
        advantages_col = advantages.unsqueeze(1)
        clipped_log_ratio = torch.where(
            advantages_col > 0,
            torch.minimum(log_ratio, clamped_log_ratio),
            torch.maximum(log_ratio, clamped_log_ratio),
        )

        # Optionally drop low-entropy tokens from the geometric mean
        seq_mask = mask * entropy_mask if entropy_mask is not None else mask

        # Geometric mean of the clipped token ratios = exp(mean of clipped log-ratios over valid tokens). The 1/|o_i|
        # exponent is the geometric-mean normalization; the paper's ablation shows it is essential.
        log_importance_weights = (clipped_log_ratio * seq_mask).sum(-1) / seq_mask.sum(-1).clamp(min=1.0)  # (B,)
        coef = torch.exp(log_importance_weights)  # (B,) sequence-level (geometric-mean) importance weight

        per_sequence_loss = -coef * advantages  # (B,)

        # KL regularization toward the reference model (optional; sequence-averaged to match GMPO's sequence-level
        # objective). Disabled by default (beta == 0)
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )
            seq_kl = (per_token_kl * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)  # (B,)
            per_sequence_loss = per_sequence_loss + self.beta * seq_kl

        # GMPO aggregates with a plain mean over sequences, per token-norm
        # already lives inside the geometric mean.
        mode = "train" if self.model.training else "eval"
        loss = per_sequence_loss.mean()
        policy_loss = loss.detach()
        normalizer = self.current_gradient_accumulation_steps if mode == "train" else 1.0  # no accum in eval
        loss = loss / normalizer

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

        # Fraction of the tokens pushed into the clipped region, in log-space.
        is_low_clipped = (log_ratio < -self.epsilon_low) & (advantages_col < 0)
        is_high_clipped = (log_ratio > self.epsilon_high) & (advantages_col > 0)
        is_region_clipped = is_low_clipped | is_high_clipped
        self._metrics[mode]["clip_ratio/low_mean"].append(global_masked_mean(is_low_clipped.float()))
        self._metrics[mode]["clip_ratio/high_mean"].append(global_masked_mean(is_high_clipped.float()))
        self._metrics[mode]["clip_ratio/region_mean"].append(global_masked_mean(is_region_clipped.float()))
        gathered_low_clip = self.accelerator.gather(masked_seq_mean(is_low_clipped.float()))
        self._metrics[mode]["clip_ratio/low_min"].append(nanmin(gathered_low_clip).item())
        gathered_high_clip = self.accelerator.gather(masked_seq_mean(is_high_clipped.float()))
        self._metrics[mode]["clip_ratio/high_max"].append(nanmax(gathered_high_clip).item())

        return loss
