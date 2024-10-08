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

from dataclasses import dataclass
from typing import Literal, Optional

from transformers import TrainingArguments


@dataclass
class CGPOConfig(TrainingArguments):
    r"""
    Configuration class for the [`CGPOTrainer`].

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        k (`int`, *optional*, defaults to `4`):
            Number of responses generated by the policy for each prompt at each iteration.
        rlhf_optimizer (`Literal["crraft", "codpo", "crpg"]`, *optional*, defaults to `crraft`):
            The RLHF optimizer to update the policy.
        kl_threshold (`float`, *optional*, defaults to `None`):
            Pre-definied maximum KL-divergence threshold.
        beta (`float`, *optional*, defaults to `0.1`):
            Only used when rlhf_optimizer is set to `codpo`.
            Parameter controlling the deviation from the reference model. Higher beta means less deviation from the reference model.
        lamb (`float`, *optional*, defaults to `5.0`):
            Only used when rlhf_optimizer is set to `codpo`.
            Parameter controlling the importance of the regularization term added to the DPO loss.
        local_generation_batch_size (`int`, *optional*, defaults to `None`):
             The size of the local mini-batch used during the generation phase.
        max_new_tokens (`int`, *optional*, defaults to `64`):
            Maximum number of tokens to generate per completion.
        max_length (`int`, *optional*, defaults to `None`):
            Maximum number of tokens to generate per completion.
        temperature (`float`, *optional*, defaults to `0.9`):
            Temperature for sampling. The higher the temperature, the more random the completions.
        missing_eos_penalty (`Optional[float]`, *optional*, defaults to `None`):
            Penalty applied to the score when the model fails to generate an EOS token. This is useful to encourage
            to generate completions shorter than the maximum length (`max_new_tokens`). The penalty must be a positive
            value.
        disable_dropout (`bool`, *optional*, defaults to `True`):
            Whether to disable dropout in the model.
    """

    k: int = 4
    rlhf_optimizer: Literal["crraft", "codpo", "crpg"] = "crraft"
    kl_threshold: float = None
    beta: float = 0.1
    lamb: float = 5.0
    local_generation_batch_size: int = None
    max_new_tokens: int = 64
    max_length: int = None
    temperature: float = 0.9
    missing_eos_penalty: Optional[float] = None
    disable_dropout: bool = True

    def __post_init__(self):
        super().__post_init__()

        if self.rlhf_optimizer not in {"crraft", "codpo", "crpg"}:
            raise ValueError(
                f"Invalid value for rlhf_optimizer: {self.rlhf_optimizer}. Must be one of 'crraft', 'codpo', or 'crpg'."
            )

        if self.kl_threshold is None:
            raise ValueError("Training without setting the KL divergence threshold is not supported.")
