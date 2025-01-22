# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import os
from dataclasses import dataclass, field

from ..trainer.utils import OnPolicyConfig


@dataclass
class RLOOConfig(OnPolicyConfig):
    r"""
    Configuration class for the [`RLOOTrainer`].

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        exp_name (`str`, *optional*, defaults to `os.path.basename(__file__)[: -len(".py")]`):
            Name of this experiment.
        reward_model_path (`str`, *optional*, defaults to `"EleutherAI/pythia-160m"`):
            Path to the reward model.
        num_ppo_epochs (`int`, *optional*, defaults to `4`):
            Number of epochs to train.
        whiten_rewards (`bool`, *optional*, defaults to `False`):
            Whether to whiten the rewards.
        kl_coef (`float`, *optional*, defaults to `0.05`):
            KL coefficient.
        cliprange (`float`, *optional*, defaults to `0.2`):
            Clip range.
        rloo_k (`int`, *optional*, defaults to `2`):
            REINFORCE Leave-One-Out (RLOO) number of online samples per prompt.
        normalize_reward (`bool`, *optional*, defaults to `False`):
            Whether to normalize rewards.
        reward_clip_range (`float`, *optional*, defaults to `10.0`):
            Clip range for rewards.
        normalize_advantage (`bool`, *optional*, defaults to `False`):
            Whether to normalize advantages.
        token_level_kl (`bool`, *optional*, defaults to `True`):
            Whether to use token-level KL penalty or sequence-level KL penalty.
        ds3_gather_for_generation (`bool`, *optional*, defaults to `True`):
            This setting applies to DeepSpeed ZeRO-3. If enabled, the policy model weights are gathered for generation,
            improving generation speed. However, disabling this option allows training models that exceed the VRAM
            capacity of a single GPU, albeit at the cost of slower generation.
    """

    exp_name: str = field(
        default=os.path.basename(__file__)[:-3],
        metadata={"help": "Name of this experiment."},
    )
    reward_model_path: str = field(
        default="EleutherAI/pythia-160m",
        metadata={"help": "Path to the reward model."},
    )
    num_ppo_epochs: int = field(
        default=4,
        metadata={"help": "Number of epochs to train."},
    )
    whiten_rewards: bool = field(
        default=False,
        metadata={"help": "Whether to whiten the rewards."},
    )
    kl_coef: float = field(
        default=0.05,
        metadata={"help": "KL coefficient."},
    )
    cliprange: float = field(
        default=0.2,
        metadata={"help": "Clip range."},
    )
    rloo_k: int = field(
        default=2,
        metadata={"help": "REINFORCE Leave-One-Out (RLOO) number of online samples per prompt."},
    )
    normalize_reward: bool = field(
        default=False,
        metadata={"help": "Whether to normalize rewards"},
    )
    reward_clip_range: float = field(
        default=10.0,
        metadata={"help": "Clip range for rewards"},
    )
    normalize_advantage: bool = field(
        default=False,
        metadata={"help": "Whether to normalize advantages"},
    )
    token_level_kl: bool = field(
        default=False,
        metadata={"help": "Whether to use token-level KL penalty or sequence-level KL penalty"},
    )
    ds3_gather_for_generation: bool = field(
        default=True,
        metadata={
            "help": "This setting applies to DeepSpeed ZeRO-3. If enabled, the policy model weights are gathered for "
            "generation, improving generation speed. However, disabling this option allows training models that "
            "exceed the VRAM capacity of a single GPU, albeit at the cost of slower generation."
        },
    )
