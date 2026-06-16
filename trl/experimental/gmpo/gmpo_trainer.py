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

    _tag_names = ['trl', 'gmpo']

    def __init__(self, model, reward_funcs, args=None, **kwargs):
        if args is None:
            model_name = model if isinstance(model, str) else get_config_model_id(model.config)
            args = GMPOConfig(f"{model_name.split('/')[-1]}-GMPO")

        super().__init__(model, reward_funcs, args=args, **kwargs)
    
    def _compute_loss(self, model, inputs):
        # compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs['prompt_ids'], inputs['prompt_mask']
        completion_ids, completion_mask = inputs['completion_ids'], inputs['completion_mask']
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1) # we only need to complete the logits for the completion tokens
        mask = completion_mask if "tool_mask" not in inputs else completion_mask * inputs["tool_mask"]

        # Compute the per_token_logps and the entropy at each position in the completion
        per_token_logps, entropies = self._get_per_token_logps_and_entropies(
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
            mm_token_type_ids=inputs.get("mm_token_type_ids"),
            image_position_ids=inputs.get("image_position_ids"),
        )

        if self.top_entropy_quantile < 1.0:
            entropy_mask = self.get_high_entropy_mask(entropies, mask, 1 - self.top_entropy_quantile)
        else:
            entropy_mask = None

        advantages = inputs['advantages']
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

        per_sequence_loss = -coef * advantages # (B,)

        # KL regularization toward the reference model (optional; sequence-averaged to match GMPO's sequence-level
        # objective). Disabled by default (beta == 0)

        if self.beta != 0.0:
            ref_per_token_logps = inputs['ref_per_token_logps']
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )
            seq_kl = (per_token_kl * mask).sum(-1) / mask.sum(-1).clamp(min=1.0) # (B,)
            per_sequence_loss = per_sequence_loss + self.beta * seq_kl

        # GMPO aggregates with a plain mean over sequences, per token-norm
        # already lives inside the geometric mean.
        mode = "train" if self.model.training else "eval"
        loss = per_sequence_loss.mean()
        normalizer = self.current_gradient_accumulation_steps if mode == "train" else 1.0  # no accum in eval
        loss = loss / normalizer

        # logging
        completion_token_count = mask.sum().clamp(min=1.0)

        if self.beta != 0.0:
            mean_kl = (per_token_kl * mask).sum() / completion_token_count
            self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).nanmean().item())

        mean_entropy = (entropies * mask).sum() / completion_token_count
        self._metrics[mode]["entropy"].append(self.accelerator.gather(mean_entropy).nanmean().item())

        # Fraction of the tokens pushed into the clipped region, in log-space.
        is_high_clipped = (log_ratio > self.epsilon_high) & (advantages_col > 0)
        is_low_clipped = (log_ratio < -self.epsilon_low) & (advantages_col < 0)
        is_region_clipped = is_high_clipped | is_low_clipped

        low_clip = (is_low_clipped.float() * mask).sum() / completion_token_count
        high_clip = (is_high_clipped.float() * mask).sum() / completion_token_count
        clip_ratio = (is_region_clipped.float() * mask).sum() / completion_token_count

        gathered_low_clip = self.accelerator.gather(low_clip)
        self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/low_min"].append(nanmin(gathered_low_clip).item())
        gathered_high_clip = self.accelerator.gather(high_clip)
        self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/high_max"].append(nanmax(gathered_high_clip).item())
        gathered_clip_ratio = self.accelerator.gather(clip_ratio)
        self._metrics[mode]["clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean().item())
        
        return loss