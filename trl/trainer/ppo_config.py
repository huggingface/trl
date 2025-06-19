# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
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
from typing import Literal, Optional

from ..trainer.utils import OnPolicyConfig


@dataclass
class PPOConfig(OnPolicyConfig):
    r"""
    Configuration class for the [`PPOTrainer`].

    This class includes only the parameters that are specific to PPO training. For a full list of training arguments,
    please refer to the [`~transformers.TrainingArguments`] and [`OnPolicyConfig`] documentation. Note that default
    values in this class may differ from those in [`~transformers.TrainingArguments`].

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        exp_name (`str`, *optional*, defaults to `os.path.basename(__file__)[:-3]`):
            Name of this experiment.
        reward_model_path (`str`, *optional*, defaults to `"EleutherAI/pythia-160m"`):
            Path to the reward model.
        model_adapter_name (`str` or `None`, *optional*, defaults to `None`):
            Name of the train target PEFT adapter, when using LoRA with multiple adapters.
        ref_adapter_name (`str` or `None`, *optional*, defaults to `None`):
            Name of the reference PEFT adapter, when using LoRA with multiple adapters.
        num_ppo_epochs (`int`, *optional*, defaults to `4`):
            Number of epochs to train.
        whiten_rewards (`bool`, *optional*, defaults to `False`):
            Whether to whiten the rewards.
        kl_coef (`float`, *optional*, defaults to `0.05`):
            KL coefficient.
        kl_estimator (`Literal["k1", "k3"]`, *optional*, defaults to `"k1"`):
            Which estimator for KL-Divergence to use from [Approximating KL
            Divergence](http://joschu.net/blog/kl-approx.html). Defaults to "k1", a straightforward, unbiased
            estimator. Can be set to "k3", an unbiased estimator with lower variance which "appears to be a strictly
            better estimator". Cannot be set to "k2", as it is used for logging purposes.
        cliprange (`float`, *optional*, defaults to `0.2`):
            Clip range.
        vf_coef (`float`, *optional*, defaults to `0.1`):
            Value function coefficient.
        cliprange_value (`float`, *optional*, defaults to `0.2`):
            Clip range for the value function.
        gamma (`float`, *optional*, defaults to `1.0`):
            Discount factor.
        lam (`float`, *optional*, defaults to `0.95`):
            Lambda value for GAE.
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
    model_adapter_name: Optional[str] = field(
        default=None,
        metadata={"help": "Name of the train target PEFT adapter, when using LoRA with multiple adapters."},
    )
    ref_adapter_name: Optional[str] = field(
        default=None,
        metadata={"help": "Name of the reference PEFT adapter, when using LoRA with multiple adapters."},
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
    kl_estimator: Literal["k1", "k3"] = field(
        default="k1",
        metadata={
            "help": "Which estimator for KL-Divergence to use from Approximating KL Divergence "
            "(http://joschu.net/blog/kl-approx.html). Defaults to 'k1', a straightforward, unbiased estimator. Can be "
            "set to 'k3', an unbiased estimator with lower variance which 'appears to be a strictly better "
            "estimator'. Cannot be set to 'k2', as it is used for logging purposes."
        },
    )
    cliprange: float = field(
        default=0.2,
        metadata={"help": "Clip range."},
    )
    vf_coef: float = field(
        default=0.1,
        metadata={"help": "Value function coefficient."},
    )
    cliprange_value: float = field(
        default=0.2,
        metadata={"help": "Clip range for the value function."},
    )
    gamma: float = field(
        default=1.0,
        metadata={"help": "Discount factor."},
    )
    lam: float = field(
        default=0.95,
        metadata={"help": "Lambda value for GAE."},
    )
    ds3_gather_for_generation: bool = field(
        default=True,
        metadata={
            "help": "This setting applies to DeepSpeed ZeRO-3. If enabled, the policy model weights are gathered for "
            "generation, improving generation speed. However, disabling this option allows training models that "
            "exceed the VRAM capacity of a single GPU, albeit at the cost of slower generation."
        },
    )
