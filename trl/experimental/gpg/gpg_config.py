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
        loss_type (`str`, *optional*, defaults to `"grpo"`):
            Loss normalizer. The correction counts completions, so it is exact only when the loss is normalized per
            completion. GRPO's own default, `"dapo"`, divides by a completion-token count instead, and a
            completion-count factor cancels a token-count denominator only when every group emits the same number of
            tokens.
        bias_correction (`bool`, *optional*, defaults to `True`):
            Whether to rescale the loss by the fraction of completions whose advantage is non-zero. A completion in a
            group whose rewards are all identical, or one that no reward function could score, has a zero advantage
            and contributes nothing to the gradient, yet it still counts toward the loss denominator and so shrinks
            the update. GPG divides the loss by that fraction to compensate. Set to `False` to recover the uncorrected
            GRPO gradient magnitude.
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
    loss_type: str = field(
        default="grpo",
        metadata={
            "help": "Loss normalizer. The correction counts completions, so it is exact only under a "
            "per-completion normalizer. GRPO's own default, 'dapo', divides by a completion-token count, which a "
            "completion-count factor cancels only when every group emits the same number of tokens."
        },
    )
    bias_correction: bool = field(
        default=True,
        metadata={
            "help": "Whether to rescale the loss by the fraction of completions whose advantage is non-zero. A "
            "completion in a group with identical rewards, or one no reward function could score, contributes "
            "nothing to the gradient yet still counts toward the loss denominator. GPG divides the loss by that "
            "fraction to compensate."
        },
    )
