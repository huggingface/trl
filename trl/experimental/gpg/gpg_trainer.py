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
    gradient, yet its tokens still count toward the loss denominator. The update is therefore scaled down by the
    fraction of such degenerate groups. GPG divides the loss by the fraction of groups that are *not* degenerate,
    restoring the gradient magnitude the non-degenerate groups should have produced.

    [`GRPOTrainer`] already computes that fraction: it logs `frac_reward_zero_std`, the fraction of completions whose
    group reward standard deviation is zero. The correction factor is its complement, so `_compute_loss` only has to
    rescale.

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

        super().__init__(model, reward_funcs, args=args, **kwargs)

        self.bias_correction = args.bias_correction
        # Fraction of groups with a non-degenerate reward spread, refreshed once per generation batch. Starts at 1.0
        # so the correction is a no-op if the loss is ever computed before a generation batch has been scored.
        self._valid_group_fraction = 1.0

    def _generate_and_score_completions(self, inputs):
        inputs = super()._generate_and_score_completions(inputs)

        # Read `frac_reward_zero_std` immediately after the parent appends it, rather than at loss time: the metric
        # lists are cleared on every log, and a single generation batch feeds several gradient-accumulation
        # micro-batches, each of which calls `_compute_loss`.
        mode = "train" if self.model.training else "eval"
        self._valid_group_fraction = 1.0 - self._metrics[mode]["frac_reward_zero_std"][-1]

        return inputs

    def _compute_loss(self, model, inputs):
        loss = super()._compute_loss(model, inputs)

        # Skip the correction when every group is degenerate: the advantages are then all zero, so the loss carries no
        # gradient and there is nothing to rescale, while the factor itself would be a division by zero.
        if self.bias_correction and self._valid_group_fraction > 0.0:
            loss = loss / self._valid_group_fraction

        return loss
