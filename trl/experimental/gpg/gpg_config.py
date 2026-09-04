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

from dataclasses import dataclass, field

from ...trainer.grpo_config import GRPOConfig


@dataclass
class GPGConfig(GRPOConfig):
    # docstyle-ignore
    r"""
    Configuration class for the [`GPGTrainer`].

    [`GPGConfig`] inherits every parameter from [`GRPOConfig`] and changes only the defaults that define GPG. The
    [GPG paper](https://huggingface.co/papers/2504.02546) drops the critic, the reference model and the KL constraint,
    and optimizes the plain policy-gradient objective rather than a surrogate. GRPO already defaults `beta` to
    `0.0`, so that one is restated here to make the method self-describing; the rest are genuine changes.

    Parameters:
        beta (`float`, *optional*, defaults to `0.0`):
            KL coefficient. GPG removes the KL constraint entirely, so this defaults to `0.0`, which also avoids
            loading a reference model.
        scale_rewards (`str` or `bool`, *optional*, defaults to `"none"`):
            Scaling strategy for the rewards, taking the same values as in [`GRPOConfig`]. GPG uses the mean-centered
            advantage with no standard-deviation scaling, so this defaults to `"none"` where [`GRPOConfig`] defaults
            to `"group"`.
        loss_type (`str`, *optional*, defaults to `"grpo"`):
            Loss normalizer. The correction counts equal-sized groups, so it cancels the denominator only when the
            loss is normalized per completion, which is what `"grpo"` and `"sapo"` do. The paper's eq. 5 instead
            normalizes by a completion-token total, which is `"bnpo"` and GRPO's own default `"dapo"`; against that
            denominator the correction is exact only when every completion has the same length. This defaults to
            `"grpo"` to keep the correction exact, and [`GPGTrainer`] rejects only `"luspo"`.
        router_aux_loss_coef (`float`, *optional*, defaults to `0.0`):
            Coefficient of the load-balancing auxiliary loss, which only has an effect on a Mixture-of-Experts model.
            The auxiliary loss is added to the loss before the correction divides it, which would move its
            effective coefficient with the reward spread, so [`GPGTrainer`] rejects a non-zero value once the
            loaded model turns out to be a Mixture-of-Experts. This defaults to `0.0` where [`GRPOConfig`] defaults to
            `0.001`, so such a model trains under GPG without further configuration.
        bias_correction (`bool`, *optional*, defaults to `True`):
            Whether to rescale the loss by the fraction of completion slots that carry an informative group's signal
            (paper eq. 7). Degenerate groups and unscorable completions contribute nothing to the gradient, yet they
            still count toward the loss denominator and shrink the update. GPG divides the loss by the remaining
            fraction to compensate. Set to `False` to recover the uncorrected GRPO gradient magnitude.
    """

    beta: float = field(
        default=0.0,
        metadata={
            "help": "KL coefficient. GPG removes the KL constraint, so this defaults to 0.0, which also avoids "
            "loading a reference model."
        },
    )
    scale_rewards: str = field(
        default="none",
        metadata={
            "help": "Scaling strategy for the rewards, taking the same values as in GRPOConfig. GPG uses the "
            "mean-centered advantage with no standard-deviation scaling, so this defaults to 'none'."
        },
    )
    loss_type: str = field(
        default="grpo",
        metadata={
            "help": "Loss normalizer. The correction counts equal-sized groups, so it is exact only under a "
            "per-completion normalizer such as 'grpo' or 'sapo'. The paper's eq. 5 normalizes by a "
            "completion-token total instead, which is 'bnpo' and GRPO's own default 'dapo'; against those the "
            "correction is exact only at equal completion lengths. Defaults to 'grpo' to keep it exact."
        },
    )
    router_aux_loss_coef: float = field(
        default=0.0,
        metadata={
            "help": "Coefficient of the load-balancing auxiliary loss, which only has an effect on a "
            "Mixture-of-Experts model. It is added to the loss before the correction divides it, so GPGTrainer "
            "rejects a non-zero value; this defaults to 0.0 where GRPOConfig defaults to 0.001."
        },
    )
    bias_correction: bool = field(
        default=True,
        metadata={
            "help": "Whether to rescale the loss by the fraction of completion slots that carry an informative "
            "group's signal. Degenerate groups and unscorable completions contribute nothing to the gradient yet "
            "still count toward the loss denominator. GPG divides the loss by the remaining fraction to compensate."
        },
    )
