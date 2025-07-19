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

from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments


@dataclass
class RewardConfig(TrainingArguments):
    r"""
    Configuration class for the [`RewardTrainer`].

    This class includes only the parameters that are specific to Reward training. For a full list of training
    arguments, please refer to the [`~transformers.TrainingArguments`] documentation. Note that default values in this
    class may differ from those in [`~transformers.TrainingArguments`].

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
            Whether to remove the columns that are not used by the model's forward pass. Can be `True` only if the
            dataset is pretokenized.
    """

    # Parameters whose default values are overridden from TrainingArguments
    logging_steps: float = field(
        default=10,
        metadata={
            "help": "Log every X updates steps. Should be an integer or a float in range `[0,1)`. If smaller than 1, "
            "will be interpreted as ratio of total training steps."
        },
    )
    bf16: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether to use bf16 (mixed) precision instead of 32-bit. Requires Ampere or higher NVIDIA "
            "architecture or Intel XPU or using CPU (use_cpu) or Ascend NPU. If not set, it defaults to `True` if "
            "`fp16` is not set."
        },
    )
    # Note: In transformers>=4.54.0, `average_tokens_across_devices` defaults to True. Overriding this setting is only
    # needed for earlier versions. Once we require transformers>=4.54.0, this line can be safely removed.
    # See https://github.com/huggingface/transformers/pull/39395
    average_tokens_across_devices: bool = field(
        default=True,
        metadata={
            "help": "Whether or not to average tokens across devices. If enabled, will use all_reduce to synchronize "
            "num_tokens_in_batch for precise loss calculation. Reference: https://github.com/huggingface/transformers/issues/34242 "
        },
    )

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

    def __post_init__(self):
        self.bf16 = not (self.fp16) if self.bf16 is None else self.bf16

        super().__post_init__()
