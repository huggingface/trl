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
class PRMConfig(TrainingArguments):
    r"""
    Configuration class for the [`PRMTrainer`].

    This class includes only the parameters that are specific to PRM training. For a full list of training arguments,
    please refer to the [`~transformers.TrainingArguments`] documentation. Note that default values in this class may
    differ from those in [`~transformers.TrainingArguments`].

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        max_length (`int` or `None`, *optional*, defaults to `1024`):
            Maximum length of the sequences (prompt + completion) used for truncation.
        max_prompt_length (`int` or `None`, *optional*, defaults to `512`):
            Maximum length of the prompt used for truncation.
        max_completion_length (`int` or `None`, *optional*, defaults to `None`):
            Maximum length of the completion used for truncation. The completion is the concatenation of the steps.
        disable_dropout (`bool`, *optional*, defaults to `True`):
            Whether to disable dropout in the model.
        step_separator (`str`, *optional*, defaults to `"\n"`):
            Separator used to separate each step of the reasoning process.
        train_on_last_step_only (`bool`, *optional*, defaults to `False`):
            Whether to train only on the last step.
        dataset_num_proc (`int`, *optional*, defaults to `None`):
            Number of processes to use for processing the dataset.
    """

    # Parameters whose default values are overridden from TrainingArguments
    learning_rate: float = field(
        default=1e-5,
        metadata={"help": "The initial learning rate for AdamW."},
    )
    logging_steps: float = field(
        default=10,
        metadata={
            "help": (
                "Log every X updates steps. Should be an integer or a float in range `[0,1)`. "
                "If smaller than 1, will be interpreted as ratio of total training steps."
            )
        },
    )
    average_tokens_across_devices: bool = field(
        default=True,
        metadata={
            "help": "Whether or not to average tokens across devices. If enabled, will use all_reduce to synchronize "
            "num_tokens_in_batch for precise loss calculation. Reference: https://github.com/huggingface/transformers/issues/34242 "
        },
    )

    max_length: Optional[int] = field(
        default=1024,
        metadata={"help": "Maximum length of the sequences (prompt + completion) used for truncation."},
    )
    max_prompt_length: Optional[int] = field(
        default=512,
        metadata={"help": "Maximum length of the prompt used for truncation."},
    )
    max_completion_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "Maximum length of the completion used for truncation. The completion is the concatenation of the "
            "steps."
        },
    )
    disable_dropout: bool = field(
        default=True,
        metadata={"help": "Whether to disable dropout in the model and reference model."},
    )
    step_separator: str = field(
        default="\n",
        metadata={"help": "Separator used to separate each step of the reasoning process."},
    )
    train_on_last_step_only: bool = field(
        default=False,
        metadata={"help": "Whether to train only on the last step."},
    )
    dataset_num_proc: Optional[int] = field(
        default=None,
        metadata={"help": "Number of processes to use for processing the dataset."},
    )
