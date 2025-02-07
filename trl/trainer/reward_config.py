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

from dataclasses import dataclass, field
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
        max_length (`int` or `None`, *optional*, defaults to `1024`):
            Maximum length of the sequences (prompt + completion) in the batch, filters out entries that exceed the
            limit. This argument is required if you want to use the default data collator.
        disable_dropout (`bool`, *optional*, defaults to `True`):
            Whether to disable dropout in the model.
        dataset_num_proc (`int`, *optional*, defaults to `None`):
            Number of processes to use for processing the dataset.
        center_rewards_coefficient (`float`, *optional*, defaults to `None`):
            Coefficient to incentivize the reward model to output mean-zero rewards (proposed by
            https://huggingface.co/papers/2312.09244, Eq. 2). Recommended value: `0.01`.
        remove_unused_columns (`bool`, *optional*, defaults to `False`):
            Whether to remove the columns that are not used by the model's forward pass. Can be `True` only if
            the dataset is pretokenized.
    """

    max_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "Maximum length of the sequences (prompt + completion) in the batch, filters out entries that "
            "exceed the limit. This argument is required if you want to use the default data collator."
        },
    )
    disable_dropout: bool = field(
        default=True,
        metadata={"help": "Whether to disable dropout in the model and reference model."},
    )
    dataset_num_proc: Optional[int] = field(
        default=None,
        metadata={"help": "Number of processes to use for processing the dataset."},
    )
    center_rewards_coefficient: Optional[float] = field(
        default=None,
        metadata={
            "help": "Coefficient to incentivize the reward model to output mean-zero rewards (proposed by "
            "https://huggingface.co/papers/2312.09244, Eq. 2). Recommended value: `0.01`."
        },
    )
    remove_unused_columns: bool = field(
        default=False,
        metadata={
            "help": "Whether to remove the columns that are not used by the model's forward pass. Can be `True` only "
            "if the dataset is pretokenized."
        },
    )
