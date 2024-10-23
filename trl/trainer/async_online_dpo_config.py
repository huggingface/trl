# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
from typing import List, Literal
import os

from ..trainer.utils import OnPolicyConfig


@dataclass
class AsyncOnlineDPOConfig(OnPolicyConfig):
    r"""
    Configuration class for the [`AsyncOnlineDPOTrainer`].

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        exp_name (`str`, *optional*, defaults to `os.path.basename(__file__)[:-3]`):
            Name of this experiment.
        num_ppo_epochs (`int`, *optional*, defaults to `1`):
            Number of updates to train on the same minibatch
        learning_rate (`float`, *optional*, defaults to `5e-7`):
            Initial learning rate for [`AdamW`] optimizer. The default value replaces that of
            [`~transformers.TrainingArguments`].
        reward_model_path (`Optional[str]`, *optional*, defaults to `None`):
            Path to the reward model.
        beta (`float` or `list[float]`, *optional*, defaults to `0.1`):
            Parameter controlling the deviation from the reference model. Higher β means less deviation from the
            reference model. For the IPO loss (`loss_type="ipo"`), β is the regularization parameter denoted by τ in
            the [paper](https://huggingface.co/papers/2310.12036). If a list of floats is provided then the β is
            selected for each new epoch and the last β is used for the rest of the epochs.
        loss_type (`str`, *optional*, defaults to `"sigmoid"`):
            Type of loss to use. Possible values are:

                - `"sigmoid"`: sigmoid loss from the original [DPO](https://huggingface.co/papers/2305.18290) paper.
                - `"ipo"`: IPO loss from the [IPO](https://huggingface.co/papers/2310.12036) paper.

        vllm_device (`str`, *optional*, defaults to `None`):
            device to put the vllm generation on, defaults to accelerate.num_processes + 1"
        vllm_gpu_memory_utilization (`float`, defaults to 0.9)
            the percentage of the GPU's memory for vllm to reserve, reduce if exection graph takes too much space

    """

    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    num_ppo_epochs: int = 1
    learning_rate: float = 5e-7
    reward_model_path: str = None
    beta: List[float] = field(default_factory=lambda: [0.1])
    loss_type: Literal["sigmoid", "ipo"] = "sigmoid"

    vllm_device: str | None = None
    vllm_gpu_memory_utilization: float = 0.9
