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
class GRPOConfig(TrainingArguments):
    r"""
    Configuration class for the [`GRPOTrainer`].

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        learning_rate (`float`, *optional*, defaults to `5e-7`):
            Initial learning rate for [`AdamW`] optimizer. The default value replaces that of
            [`~transformers.TrainingArguments`].
        max_length (`int` or `None`, *optional*, defaults to `None`):
            Maximum length of the sequences (prompt + completion) in the batch. This argument is required if you want
            to use the default data collator.
        max_prompt_length (`int` or `None`, *optional*, defaults to `None`):
            Maximum length of the prompt. This argument is required if you want to use the default data collator.
        max_completion_length (`int` or `None`, *optional*, defaults to `None`):
            Maximum length of the completion. This argument is required if you want to use the default data collator
            and your model is an encoder-decoder.
        dataset_num_proc: (`int` or `None`, *optional*, defaults to `None`):
            Number of processes to use for processing the dataset.
    """

    learning_rate: float = field(
        default=1e-6,
        metadata={
            "help": "Initial learning rate for `AdamW` optimizer. The default value replaces that of "
            "`transformers.TrainingArguments`."
        },
    )
    max_length: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum length of the sequences (prompt + completion) in the batch."},
    )
    max_prompt_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "Maximum length of the prompt. This argument is required if you want to use the default data "
            "collator and your model is an encoder-decoder."
        },
    )
    max_completion_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "Maximum length of the completion. This argument is required if you want to use the default data "
            "collator and your model is an encoder-decoder."
        },
    )
    dataset_num_proc: Optional[int] = field(
        default=None,
        metadata={"help": "Number of processes to use for processing the dataset."},
    )
