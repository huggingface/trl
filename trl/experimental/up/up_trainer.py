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

import textwrap

import torch
from accelerate.logging import get_logger

from ...trainer.grpo_trainer import GRPOTrainer
from ...trainer.utils import get_config_model_id, nanmin
from .up_config import UPConfig


logger = get_logger(__name__)


class UPTrainer(GRPOTrainer):
    """
    Trainer for Unbounded Positive Asymmetric Optimization (UP).

    UP (https://huggingface.co/papers/2607.06987) is a GRPO variant that routes tokens asymmetrically on the sign of
    the advantage. For positive advantages the importance sampling ratio is replaced by the self-anchored ratio `πθ /
    sg(πθ)`, whose forward value is exactly 1 and whose gradient is the unclipped REINFORCE gradient `Â ∇ log πθ`,
    independent of the old policy, so correct but low-confidence tokens are never truncated by the trust region.
    Non-positive advantages keep the standard clipped surrogate as a safeguard.

    Aggregation follows DAPO's global active-token normalization, matching the paper's UP-DAPO instantiation.

    The only change w.r.t. [`GRPOTrainer`] is `_compute_loss`. Everything else (generation, reward computation, weight
    syncing, metric logging) is inherited unchanged.
    """

    _tag_names = ["trl", "up"]
    _name = "UP"
    _paper = {
        "title": "UP: Unbounded Positive Asymmetric Optimization for Breaking the Exploration-Stability Dilemma",
        "id": "2607.06987",
        # docstyle-ignore
        "citation": textwrap.dedent("""\
            @article{fan2026up,
                title        = {{UP: Unbounded Positive Asymmetric Optimization for Breaking the Exploration-Stability Dilemma}},
                author       = {Chongyu Fan and Pengfei Liu and Jingjia Huang and Sijia Liu and Yi Lin},
                year         = 2026,
                journal      = {arXiv preprint arXiv:2607.06987}
            }"""),
    }

    def __init__(self, model, reward_funcs, args=None, **kwargs):
        if args is None:
            model_name = model if isinstance(model, str) else get_config_model_id(model.config)
            args = UPConfig(f"{model_name.split('/')[-1]}-UP")

        if args.use_liger_kernel:
            # `GRPOTrainer.compute_loss` routes to the fused Liger loss before `_compute_loss` is reached, so the UP
            # objective would be silently replaced by the GRPO one.
            raise NotImplementedError(
                "`use_liger_kernel=True` is not supported by `UPTrainer`: the fused Liger loss bypasses the UP "
                "objective. Set `use_liger_kernel=False`."
            )

        super().__init__(model, reward_funcs, args=args, **kwargs)

        if self.importance_sampling_level == "sequence":
            # `GRPOTrainer.__init__` emits a generic warning here whose remedy (`loss_type='grpo'`) does nothing in
            # this trainer, since UP ignores `loss_type` entirely. State the applicable one.
            logger.warning(
                "With `importance_sampling_level='sequence'`, the positive branch follows the sequence-level UP "
                "objective (UP-GSPO), but token losses are still aggregated with the global active-token "
                "normalization, which weights each sequence by its completion length instead of averaging "
                "per-sequence losses as in the paper's UP-GSPO setup. To reproduce the paper's token-level UP-DAPO "
                "setup, set `importance_sampling_level='token'` (the default)."
            )

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
        if advantages.dim() == 1:
            advantages = advantages.unsqueeze(1)
        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps

        if self.off_policy_mask_threshold is not None:
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
        else:
            raise ValueError(
                f"Unknown importance sampling level: {self.importance_sampling_level}. Possible values are 'token' "
                "and 'sequence'."
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

        # UP objective (Eqs. 12-14). Tokens with positive advantage bypass clipping entirely: the importance
        # sampling ratio is replaced by the self-anchored ratio πθ / sg(πθ), whose value is 1 and whose gradient is
        # the unclipped REINFORCE gradient Â ∇ log πθ, independent of the old policy. Tokens with non-positive
        # advantage keep the standard clipped surrogate as a safeguard against aggressive penalization.
        if self.importance_sampling_level == "token":
            anchored_log_ratio = per_token_logps - per_token_logps.detach()
        else:
            # "sequence": length-normalized sequence-level ratio (UP-GSPO, Eq. 16). Under the global active-token
            # normalizer this is numerically identical to the token-level branch; only the non-positive branch
            # actually changes.
            anchored_log_ratio = ((per_token_logps - per_token_logps.detach()) * mask).sum(-1) / mask.sum(-1).clamp(
                min=1.0
            )
            anchored_log_ratio = anchored_log_ratio.unsqueeze(-1)
        up_per_token_loss = -anchored_log_ratio.exp() * advantages
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        # Two-sided clipping
        if self.args.delta is not None:
            coef_1 = torch.clamp(coef_1, max=self.args.delta)
        clipped_per_token_loss = -torch.min(coef_1 * advantages, coef_2 * advantages)
        per_token_loss = torch.where(advantages > 0, up_per_token_loss, clipped_per_token_loss)

        if self.off_policy_mask_threshold is not None:
            per_token_loss = per_token_loss * off_policy_mask

        if entropy_mask is not None:
            per_token_loss = per_token_loss * entropy_mask

        if self.use_vllm and self.vllm_importance_sampling_correction:
            per_token_loss = per_token_loss * inputs["importance_sampling_ratio"]

        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        # UP aggregates like DAPO: normalize by the number of active tokens in the global accumulated batch.
        # `num_items_in_batch` spans the generation batch, so rescale it to one accumulation window.
        mode = "train" if self.model.training else "eval"
        normalizer = inputs["num_items_in_batch"].clamp(min=1.0) / self.accelerator.num_processes
        if mode == "train":  # in eval, the batch is neither split across steps nor accumulated
            normalizer = normalizer * self.current_gradient_accumulation_steps / self.args.steps_per_generation
        loss = (per_token_loss * mask).sum() / normalizer
        policy_loss = loss.detach()

        # Entropy bonus: add entropy regularization to encourage exploration.
        if self._entropy_bonus_enabled:
            effective_mask = mask if entropy_mask is None else mask * entropy_mask
            # `normalizer` is a global token count, so summing the entropies (instead of averaging them again) makes
            # the term accumulate over the optimizer step to the global mean per-token entropy, as it does for the
            # other global-token-normalized loss types.
            entropy_loss = (entropies * effective_mask).sum() / normalizer

            if self.use_adaptive_entropy:
                apply_coef = self.entropy_coef if self._last_world_entropy <= self.args.entropy_target else 0.0
            else:
                apply_coef = self.entropy_coef

            loss = loss - apply_coef * entropy_loss

            self._metrics[mode]["policy_loss"].append(self.accelerator.gather(policy_loss).nanmean().item())

            if self.use_adaptive_entropy and mode == "train":
                stats = torch.stack([(entropies * effective_mask).sum(), effective_mask.sum()]).detach()
                if self._entropy_window_stats is None:
                    self._entropy_window_stats = stats
                else:
                    self._entropy_window_stats = self._entropy_window_stats + stats
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

            if mode == "train" and self.accelerator.sync_gradients:
                self._metrics[mode]["entropy_coef"].append(self.entropy_coef)

        # The policy loss above is scaled for gradient accumulation (HF auto-scaling is off here), so scale aux too
        if self.aux_loss_enabled:
            aux_normalizer = self.current_gradient_accumulation_steps if mode == "train" else 1.0
            loss = loss + self.router_aux_loss_coef * aux_loss / aux_normalizer
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

        # Only the negative-advantage branch of UP is clipped (the positive branch is unbounded by design), so only
        # the low-side clip ratio is meaningful.
        is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages < 0)
        self._metrics[mode]["clip_ratio/low_mean"].append(global_masked_mean(is_low_clipped.float()))
        gathered_low_clip = self.accelerator.gather(masked_seq_mean(is_low_clipped.float()))
        self._metrics[mode]["clip_ratio/low_min"].append(nanmin(gathered_low_clip).item())

        return loss
