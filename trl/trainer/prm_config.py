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
class PRMConfig(TrainingArguments):
    r"""
    Configuration class for the [`PRMTrainer`].

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        learning_rate (`float`, *optional*, defaults to `1e-5`):
            Initial learning rate for [`AdamW`] optimizer. The default value replaces that of
            [`~transformers.TrainingArguments`].
        max_length (`Optional[int]`, *optional*, defaults to `None`):
            Maximum length of the sequences (prompt + completion) used for truncation.
        max_completion_length (`Optional[int]`, *optional*, defaults to `None`):
            Maximum length of the completion used for truncation. The completion is the concatenation of the steps.
        step_separator (`str`, *optional*, defaults to `"\n"`):
            Separator used to separate each step of the reasoning process.
        train_on_last_step_only (`bool`, *optional*, defaults to `False`):
            Whether to train only on the last step.
        dataset_num_proc (`int`, *optional*, defaults to `None`):
            Number of processes to use for processing the dataset.
    """

    learning_rate: float = 1e-5
    max_length: Optional[int] = None
    max_completion_length: Optional[int] = None
    step_separator: str = "\n"
    train_on_last_step_only: bool = False
    dataset_num_proc: Optional[int] = None
