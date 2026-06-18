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
class GMPOConfig(GRPOConfig):
    # docstyle-ignore
    r"""
    Configuration class for the [`GMPOTrainer`].

    [`GMPOConfig`] inherits every parameter from [`GRPOConfig`]; it only changes the meaning and default of the
    clipping range. In GMPO, clipping is applied to the per-token *log*-importance ratios (i.e. in log space) before
    the geometric mean is taken, so `epsilon` and `epsilon_high` are expressed in log space: the effective ratio
    clipping range is `(exp(-epsilon), exp(epsilon_high))`. The [GMPO paper](https://huggingface.co/papers/2507.20673)
    recommends a markedly wider range than GRPO/DAPO, `(exp(-0.4), exp(0.4))`, to encourage exploration.

    Parameters:
        epsilon (`float`, *optional*, defaults to `0.4`):
            Lower-bound clipping value, expressed in log space. The lower bound of the per-token importance ratio is
            `exp(-epsilon)`.
        epsilon_high (`float`, *optional*):
            Upper-bound clipping value, expressed in log space. If `None`, it defaults to the value of `epsilon`. The
            upper bound of the per-token importance ratio is `exp(epsilon_high)`.
    """

    epsilon: float = field(
        default=0.4,
        metadata={
            "help": "Lower-bound clipping value, expressed in log space. The lower bound of the per-token importance "
            "ratio is exp(-epsilon). GMPO recommends 0.4."
        },
    )

    loss_type: str = field(
        default="grpo",
        metadata={
            "help": "Ignored in GMPO. GMPO always uses per-sequence-mean / grad-accum normalization, regardless of "
            "the value of this parameter."
        },
    )
