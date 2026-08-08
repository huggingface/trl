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
    and optimizes the plain policy-gradient objective rather than a surrogate, so the GRPO defaults for `beta` and
    `scale_rewards` are already the ones GPG prescribes and are restated here to make the method self-describing.

    Parameters:
        beta (`float`, *optional*, defaults to `0.0`):
            KL coefficient. GPG removes the KL constraint entirely, so this defaults to `0.0`, which also avoids
            loading a reference model.
        scale_rewards (`str`, *optional*, defaults to `"none"`):
            Whether to scale the advantage by the standard deviation of the rewards. GPG uses the mean-centered
            advantage without standard-deviation scaling, so this defaults to `"none"`.
        bias_correction (`bool`, *optional*, defaults to `True`):
            Whether to rescale the loss by the fraction of groups whose rewards are not all identical. Groups in which
            every completion receives the same reward have a zero advantage and contribute nothing to the gradient,
            yet they still count toward the loss denominator, which shrinks the update by the fraction of such groups.
            GPG divides the loss by that fraction to compensate. Set to `False` to recover the uncorrected GRPO
            gradient magnitude.
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
            "help": "Whether to scale the advantage by the standard deviation of the rewards. GPG uses the "
            "mean-centered advantage without standard-deviation scaling, so this defaults to 'none'."
        },
    )
    bias_correction: bool = field(
        default=True,
        metadata={
            "help": "Whether to rescale the loss by the fraction of groups whose rewards are not all identical. "
            "Groups where every completion gets the same reward have a zero advantage and contribute nothing to the "
            "gradient, yet still count toward the loss denominator. GPG divides the loss by that fraction to "
            "compensate."
        },
    )
