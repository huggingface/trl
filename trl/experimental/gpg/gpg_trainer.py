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

from ...trainer.grpo_trainer import GRPOTrainer
from ...trainer.utils import get_config_model_id
from .gpg_config import GPGConfig


class GPGTrainer(GRPOTrainer):
    """
    Trainer for Group Policy Gradient (GPG).

    GPG (https://huggingface.co/papers/2504.02546) is a minimalist GRPO variant that drops the critic, the reference
    model and the KL constraint, and optimizes the policy-gradient objective directly instead of a surrogate. Those
    three properties are already expressible with [`GRPOConfig`], so [`GPGConfig`] simply defaults `beta` to `0.0` and
    `scale_rewards` to `"none"`.

    That leaves the gradient-bias correction as the only behavioral change, and it is the one thing this trainer adds.
    A group whose completions all receive the same reward has a zero advantage, so it contributes nothing to the
    gradient, yet it still counts toward the loss denominator. The update is therefore scaled down by the fraction of
    such degenerate groups, and GPG multiplies it back up. That is paper eq. 7, whose multiplier is `alpha = B / (B -
    M)`, with `M` counting the samples that belong to groups whose responses are all right or all wrong. This trainer
    stores the reciprocal, `(B - M) / B`, and divides by it, which is the same operation written the other way around
    and matches how the authors' own implementation stores it.

    A group is informative when its scorable rewards produce distinct advantages. Every scorable completion in an
    informative group counts toward the fraction, while unscorable completions do not because their advantages are
    forced to zero. The correction also zeroes the shared floating-point residual left by group-mean subtraction in
    degenerate groups. `frac_reward_zero_std`, which [`GRPOTrainer`] logs, is close but not equal, because it is
    derived from the reward standard deviation rather than from the advantages: a group no reward function could score
    has a NaN standard deviation, which is not close to zero, so the metric counts it valid although its advantages are
    all equal, and the metric turns into a batch-wide statistic that collapses to 0 or 1 when `scale_rewards="batch"`.

    The paper's eq. 5 normalizes by the total completion-token count, which is what `loss_type="bnpo"` and `"dapo"` do
    and what the authors' implementation uses. Against that denominator a completion-count multiplier is exact only
    when every completion has the same length. Against `"grpo"` and `"sapo"`, which average one token-mean per
    completion, it is exact at any lengths, so [`GPGConfig`] defaults `loss_type` to `"grpo"`. That is a deliberate
    departure from eq. 5 in favor of making eq. 7 exact; set `loss_type="bnpo"` to reproduce the published objective
    instead. Only `"luspo"` is rejected, because it sums each completion's token losses rather than averaging them and
    so cannot be canceled at any length.

    `__init__` also rejects the settings that would misapply the multiplier rather than merely blunt it: the three that
    silence a completion's tokens while leaving its advantage intact (`mask_truncated_completions`,
    `top_entropy_quantile`, `off_policy_mask_threshold`), a non-zero `beta`,
    `multi_objective_aggregation="normalize_then_sum"`, `use_liger_kernel` (which bypasses `_compute_loss` entirely),
    and the entropy or MoE router auxiliary terms.

    One published component is deliberately left out. The paper also thresholds the valid-sample proportion at
    `beta_th` and accumulates valid samples into the next resampled batch whenever the proportion falls below it, which
    curbs the variance a large correction factor introduces. This trainer applies the correction alone, so a batch
    where nearly every group is degenerate still takes one large, high-variance step.

    The generation period must divide `gradient_accumulation_steps`, so every generated batch is consumed before an
    optimizer update. The GRPO surrogate is then *gradient-identical* to the plain policy gradient GPG writes down: the
    importance ratio is exactly one at the point of evaluation, and clipping around one is inert, so both reduce to
    `advantages * dlogp/dtheta`.

    Everything else (generation, reward computation, weight syncing, metric logging) is inherited unchanged.
    """

    _tag_names = ["trl", "gpg"]
    _name = "GPG"
    _paper = {
        "title": "GPG: A Simple and Strong Reinforcement Learning Baseline for Model Reasoning",
        "id": "2504.02546",
        # docstyle-ignore
        "citation": textwrap.dedent("""\
            @inproceedings{chu2026gpg,
                title        = {{GPG: A Simple and Strong Reinforcement Learning Baseline for Model Reasoning}},
                author       = {Xiangxiang Chu and Hailang Huang and Xiao Zhang and Fei Wei and Yong Wang},
                year         = 2026,
                booktitle    = {International Conference on Learning Representations}
            }"""),
    }

    def __init__(self, model, reward_funcs, args=None, **kwargs):
        if args is None:
            model_name = model if isinstance(model, str) else get_config_model_id(model.config)
            args = GPGConfig(f"{model_name.split('/')[-1]}-GPG")

        if not isinstance(args, GPGConfig):
            raise ValueError(
                f"GPGTrainer requires a `GPGConfig`, but got `{type(args).__name__}`. GPG's defaults and its "
                f"`bias_correction` switch live on `GPGConfig`, so a plain `GRPOConfig` would silently train "
                f"uncorrected GRPO. Pass `GPGConfig(...)`, or omit `args` to build one."
            )

        generate_every = args.steps_per_generation * args.num_iterations
        if args.gradient_accumulation_steps % generate_every != 0:
            raise ValueError(
                "GPG requires `steps_per_generation * num_iterations` to divide `gradient_accumulation_steps` so "
                "every generated batch is consumed before an optimizer update. Otherwise later slices use stale "
                "log-probabilities and activate the clipped GRPO surrogate instead of plain policy gradient."
            )

        # Validate the settings that are decided by `args` alone before the parent builds anything: a rejected
        # configuration then costs no model load, and `use_liger_kernel` reports GPG's reason rather than the
        # parent's "Liger is required" ImportError, which fires first and reads as a missing dependency.
        if args.bias_correction:
            # Only `luspo` is rejected outright. Every other reduction still produces a usable objective; they
            # differ in how exactly the group factor cancels the denominator, which `GPGConfig` handles by
            # defaulting to `grpo` rather than by forbidding the alternatives. `luspo` is different: it averages
            # per-row sums instead of per-row means, so it does not reduce to any of them. On two completions of
            # length 3 with a unit per-token loss it returns 3.0 where both eq. 5 and `grpo` return 1.0, and that
            # gap grows with completion length rather than canceling.
            if args.loss_type == "luspo":
                raise ValueError(
                    "GPG's bias correction rescales a loss that is assumed to average over completions, but "
                    "`loss_type='luspo'` sums each completion's token losses instead of averaging them, so the "
                    "loss scales with completion length and the factor cannot cancel it at any length. Use 'grpo' "
                    "(the GPGConfig default) or 'sapo' for an exact correction, 'bnpo' or 'dapo' to reproduce the "
                    "paper's own objective, or set `bias_correction=False`."
                )
            # Four inherited settings silence a completion's token losses while leaving its advantage untouched
            # (grpo_trainer.py:3229-3233). The reduction still divides by the completion mask, so a silenced
            # completion keeps its slot in the denominator while contributing no gradient. That is the dilution the
            # correction exists to undo, and it cannot see it, because it reads only the advantages. The vLLM one is
            # the trap: `vllm_importance_sampling_correction` defaults to True and its mode defaults to
            # "sequence_mask", so simply passing `use_vllm=True` masks every sequence whose ratio leaves the cap.
            silencing = [
                name
                for name, enabled in [
                    ("mask_truncated_completions=True", args.mask_truncated_completions),
                    ("top_entropy_quantile<1.0", args.top_entropy_quantile < 1.0),
                    ("off_policy_mask_threshold", args.off_policy_mask_threshold is not None),
                    (
                        f"vllm_importance_sampling_mode={args.vllm_importance_sampling_mode!r}",
                        args.use_vllm
                        and args.vllm_importance_sampling_correction
                        and args.vllm_importance_sampling_mode in ("sequence_mask", "token_mask"),
                    ),
                ]
                if enabled
            ]
            if silencing:
                raise ValueError(
                    f"GPG's bias correction reads the advantages to decide which groups carry a gradient, but "
                    f"{', '.join(silencing)} zeroes a completion's token losses while leaving its advantage intact. "
                    f"That completion then contributes no gradient yet still counts in the loss denominator, which "
                    f"is the dilution the correction exists to undo, so it would under-correct by an amount it "
                    f"cannot measure. Disable it, or set `bias_correction=False`."
                )
            if args.beta != 0.0:
                raise ValueError(
                    "GPG's bias correction divides the loss returned by `_compute_loss`, which already includes the "
                    "KL term. KL does not vanish for a group whose rewards are identical, so dividing by the "
                    "gradient-carrying fraction would raise its effective coefficient to `beta / fraction`. Use "
                    "`beta=0.0` (the GPGConfig default, and what GPG prescribes) or set `bias_correction=False`."
                )
            if args.multi_objective_aggregation == "normalize_then_sum":
                raise ValueError(
                    "GPG's bias correction identifies a degenerate group by its advantages being exactly zero, but "
                    "`multi_objective_aggregation='normalize_then_sum'` normalizes per group and then subtracts a "
                    "batch mean that is only zero up to floating-point error. A degenerate group is then left with "
                    "a residual advantage rather than a clean zero, and the correction counts it as informative. "
                    "Over 2000 randomized batches the residual was non-zero in 1745 of them, peaking at 2.5e-07. "
                    "Use 'sum_then_normalize', which subtracts the group mean and leaves an exact zero, or set "
                    "`bias_correction=False`."
                )
            if args.use_liger_kernel:
                raise ValueError(
                    "GPG's bias correction is applied in `_compute_loss`, which `use_liger_kernel=True` bypasses in "
                    "favor of `compute_liger_loss`, so the correction would silently not be applied. Disable "
                    "`use_liger_kernel` or set `bias_correction=False`."
                )

        super().__init__(model, reward_funcs, args=args, **kwargs)

        self.bias_correction = args.bias_correction
        # Both flags depend on what the parent built: `_entropy_bonus_enabled` folds in `use_adaptive_entropy`, and
        # `aux_loss_enabled` needs the loaded model to be a MoE. Neither is knowable from `args` alone, so these two
        # rejections cannot join the block above. They are separate so the error names the term that actually fired.
        if self.bias_correction and self._entropy_bonus_enabled:
            raise ValueError(
                "GPG's bias correction rescales the loss returned by `_compute_loss`, which already includes the "
                "entropy bonus, so the bonus would be rescaled with it and its effective coefficient would move "
                "with the reward spread. Set `entropy_coef=0.0` and `use_adaptive_entropy=False`, or set "
                "`bias_correction=False`."
            )
        if self.bias_correction and self.aux_loss_enabled:
            raise ValueError(
                "GPG's bias correction rescales the loss returned by `_compute_loss`, which already includes the "
                "MoE router auxiliary loss, so that term would be rescaled with it and its effective coefficient "
                "would move with the reward spread. Set `router_aux_loss_coef=0.0` (the GPGConfig default), or set "
                "`bias_correction=False`."
            )

        # Fraction of completion slots that still carry a gradient, refreshed once per generation batch. Keyed by mode
        # like `_metrics`: the parent buffers one train generation across `steps_per_generation` optimizer steps and
        # only regenerates periodically, while eval generates per batch, so an eval landing inside that window would
        # otherwise leave the remaining buffered train steps rescaling with the eval fraction. Starts at 1.0 so the
        # correction is a no-op if the loss is ever computed before a generation batch has been scored.
        self._valid_sample_fraction = {"train": 1.0, "eval": 1.0}
        self._scorable_mask = None

    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        rewards_per_func = super()._calculate_rewards(inputs, prompts, completions, completion_ids_list)

        self._scorable_mask = ~torch.isnan(rewards_per_func).all(dim=1)

        return rewards_per_func

    def _generate_and_score_completions(self, inputs):
        inputs = super()._generate_and_score_completions(inputs)

        # A group is informative when its scorable advantages have some spread. Use the scorability mask rather than
        # excluding exactly-zero advantages: a zero can be a valid group-mean sample, while an unscorable completion
        # is forced to zero and must not make a degenerate group's shared floating-point residual look informative.
        # Every scorable completion in an informative group carries the group's learning signal. Unscorable
        # completions do not, so exclude their slots from the correction factor as well.
        #
        # Read it here rather than at loss time: one generation batch feeds several gradient-accumulation
        # micro-batches, each of which calls `_compute_loss`.
        # Gather before averaging: the parent slices `advantages` to the local process before returning it, so a local
        # mean would give each rank its own factor. Gradients are mean-reduced across ranks, so that would weight every
        # rank equally however many informative groups it happens to hold, and make the result depend on the world size.
        # The slice is contiguous and rank-indexed and `gather` concatenates in rank order, so the gathered tensor is
        # the parent's pre-slice tensor, which the parent itself already reshapes by `num_generations`. A rank holding
        # a partial group therefore cannot misalign the reshape.
        mode = "train" if self.model.training else "eval"
        advantages = self.accelerator.gather(inputs["advantages"])
        num_generations = self.num_generations if mode == "train" else self.num_generations_eval
        groups = advantages.view(-1, num_generations)
        scorable_groups = self._scorable_mask.view(-1, num_generations)
        highest = groups.masked_fill(~scorable_groups, -torch.inf).max(dim=1).values
        lowest = groups.masked_fill(~scorable_groups, torch.inf).min(dim=1).values
        informative_groups = (highest != lowest) & scorable_groups.any(dim=1)
        valid_samples = informative_groups.repeat_interleave(num_generations) & self._scorable_mask
        self._valid_sample_fraction[mode] = valid_samples.float().mean().item()

        process_slice = slice(
            self.accelerator.process_index * len(inputs["advantages"]),
            (self.accelerator.process_index + 1) * len(inputs["advantages"]),
        )
        inputs["advantages"] = inputs["advantages"] * valid_samples[process_slice]

        return inputs

    def _compute_loss(self, model, inputs):
        loss = super()._compute_loss(model, inputs)

        mode = "train" if self.model.training else "eval"
        fraction = self._valid_sample_fraction[mode]

        # Skip the correction when no completion carries a gradient: the loss is then flat and there is nothing to
        # rescale, while the factor itself would be a division by zero.
        if self.bias_correction and fraction > 0.0:
            loss = loss / fraction

        return loss
