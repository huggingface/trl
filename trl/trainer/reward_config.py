# Copyright 2024 The HuggingFace Team. All rights reserved.
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
from typing import Optional

from transformers import TrainingArguments


@dataclass
class RewardConfig(TrainingArguments):
    r"""
    Configuration class for the [`RewardTrainer`].

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        max_length (`Optional[int]`, *optional*, defaults to `None`):
            Maximum length of the sequences (prompt + completion) in the batch. This argument is required if you want
            to use the default data collator.
        dataset_num_proc (`int`, *optional*, defaults to `None`):
            Number of processes to use for processing the dataset.
        center_rewards_coefficient (`float`, *optional*, defaults to `None`):
            Coefficient to incentivize the reward model to output mean-zero rewards (proposed by
            https://huggingface.co/papers/2312.09244, Eq. 2). Recommended value: `0.01`.
    """

    max_length: Optional[int] = None
    dataset_num_proc: Optional[int] = None
    center_rewards_coefficient: Optional[float] = None
