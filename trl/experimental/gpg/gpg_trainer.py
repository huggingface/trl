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
    such degenerate completions. GPG divides the loss by the fraction that are *not* degenerate, restoring the gradient
    magnitude the informative completions should have produced.

    The factor is the fraction of completions whose advantage is non-zero. That is the quantity the denominator
    actually counts, and the parent zeroes an advantage in the two cases the correction must discount: the group-mean
    subtraction zeroes every member of a group whose rewards are identical, and `nan_to_num` zeroes a completion that
    no reward function could score. `frac_reward_zero_std`, which [`GRPOTrainer`] logs, sees only the first of those
    and collapses to 0 or 1 under a batch-wide standard deviation, so it is not used here.

    Because the factor counts completions, it cancels the denominator only when the denominator does too. `__init__`
    rejects the inherited settings that break that identity rather than rescaling by a factor that would be silently
    wrong: a token-normalized `loss_type`, a non-zero `beta`, `multi_objective_aggregation="normalize_then_sum"`,
    `use_liger_kernel` (which bypasses `_compute_loss` entirely), and the entropy or MoE router auxiliary terms.

    Note that with the default `num_iterations=1` the GRPO surrogate is *gradient-identical* to the plain policy
    gradient GPG writes down: the importance ratio is exactly one at the point of evaluation, and clipping around one
    is inert, so both reduce to `advantages * dlogp/dtheta`. Raising `num_iterations` above 1 makes the objective
    genuinely off-policy and departs from the method as published.

    Everything else (generation, reward computation, weight syncing, metric logging) is inherited unchanged.
    """

    _tag_names = ["trl", "gpg"]

    def __init__(self, model, reward_funcs, args=None, **kwargs):
        if args is None:
            model_name = model if isinstance(model, str) else get_config_model_id(model.config)
            args = GPGConfig(f"{model_name.split('/')[-1]}-GPG")

        # Validate the settings that are decided by `args` alone before the parent builds anything: a rejected
        # configuration then costs no model load, and `use_liger_kernel` reports GPG's reason rather than the
        # parent's "Liger is required" ImportError, which fires first and reads as a missing dependency.
        if args.bias_correction:
            # The correction rescales the loss by a count of completions, which only cancels the parent's denominator
            # when that denominator is also a completion count. Reject the settings that break that identity
            # rather than applying a factor that is silently wrong, since the loss carries no signal that it was.
            # `grpo`/`sapo` average one token-mean per completion, `dr_grpo` divides by rows times a constant
            # `max_completion_length`, and `luspo` averages per-row sums: every one of those denominators is
            # proportional to a completion count, which the factor cancels exactly. `bnpo`, `dapo`, `cispo` and
            # `vespo` divide by a completion-token total instead, which it cancels only if all groups are equally long.
            if args.loss_type not in ["grpo", "sapo", "dr_grpo", "luspo"]:
                raise ValueError(
                    f"GPG's bias correction counts completions, so it requires a per-completion loss normalizer, but "
                    f"`loss_type='{args.loss_type}'` normalizes by completion tokens. Use one of 'grpo' (the "
                    f"GPGConfig default), 'sapo', 'dr_grpo' or 'luspo', or set `bias_correction=False`."
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
                    "GPG's bias correction assumes group-relative advantages, but "
                    "`multi_objective_aggregation='normalize_then_sum'` centers on the batch mean rather than each "
                    "group's, so a group whose rewards are identical keeps a non-zero advantage and the correction "
                    "cannot identify it. Use 'sum_then_normalize' or set `bias_correction=False`."
                )
            if args.use_liger_kernel:
                raise ValueError(
                    "GPG's bias correction is applied in `_compute_loss`, which `use_liger_kernel=True` bypasses in "
                    "favour of `compute_liger_loss`, so the correction would silently not be applied. Disable "
                    "`use_liger_kernel` or set `bias_correction=False`."
                )

        super().__init__(model, reward_funcs, args=args, **kwargs)

        self.bias_correction = args.bias_correction
        # `aux_loss_enabled` also depends on the loaded model being a MoE, so it is only known after the parent runs.
        if self.bias_correction and (self._entropy_bonus_enabled or self.aux_loss_enabled):
            raise ValueError(
                "GPG's bias correction rescales the loss returned by `_compute_loss`, which already includes the "
                "entropy bonus and the MoE router auxiliary loss, so those regularizers would be rescaled with it "
                "and their effective coefficients would move with the reward spread. Disable them or set "
                "`bias_correction=False`."
            )

        # Fraction of completions that still carry a gradient, refreshed once per generation batch. Keyed by mode
        # like `_metrics`: the parent buffers one train generation across `steps_per_generation` optimizer steps and
        # only regenerates periodically, while eval generates per batch, so an eval landing inside that window would
        # otherwise leave the remaining buffered train steps rescaling with the eval fraction. Starts at 1.0 so the
        # correction is a no-op if the loss is ever computed before a generation batch has been scored.
        self._valid_group_fraction = {"train": 1.0, "eval": 1.0}

    def _generate_and_score_completions(self, inputs):
        inputs = super()._generate_and_score_completions(inputs)

        # Count the rows that survive into the gradient rather than reading `frac_reward_zero_std`. A completion
        # contributes exactly when its advantage is non-zero, and the parent zeroes the advantage in both cases the
        # correction must skip: every member of a group whose rewards are identical is zeroed by the group-mean
        # subtraction, and an unscorable completion is zeroed by `nan_to_num`. The logged metric sees only the first
        # of those, and under a batch-wide standard deviation it collapses to 0 or 1 for the whole batch.
        #
        # Read it here rather than at loss time: the metric lists are cleared on every log, and one generation batch
        # feeds several gradient-accumulation micro-batches, each of which calls `_compute_loss`.
        # Gather before averaging: the parent slices `advantages` to the local process before returning it, so a local
        # mean would give each rank its own factor. Gradients are mean-reduced across ranks, so that would weight every
        # rank equally however many informative completions it happens to hold, and make the result depend on the world
        # size. `frac_reward_zero_std` did not have that problem, because the parent computes it before the slice.
        mode = "train" if self.model.training else "eval"
        advantages = self.accelerator.gather(inputs["advantages"])
        self._valid_group_fraction[mode] = (advantages != 0).float().mean().item()

        return inputs

    def _compute_loss(self, model, inputs):
        loss = super()._compute_loss(model, inputs)

        mode = "train" if self.model.training else "eval"
        fraction = self._valid_group_fraction[mode]

        # Skip the correction when no completion carries a gradient: the loss is then flat and there is nothing to
        # rescale, while the factor itself would be a division by zero.
        if self.bias_correction and fraction > 0.0:
            loss = loss / fraction

        return loss
