# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
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

from ...trainer.base_config import BaseConfig


@dataclass
class PRMConfig(BaseConfig):
    # docstyle-ignore
    r"""
    Configuration class for the [`experimental.prm.PRMTrainer`].

    This class includes only the parameters that are specific to PRM training. For a full list of training arguments,
    please refer to the [`~transformers.TrainingArguments`] documentation. Note that default values in this class may
    differ from those in [`~transformers.TrainingArguments`].

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        max_length (`int` or `None`, *optional*, defaults to `1024`):
            Maximum length of the sequences (prompt + completion) used for truncation.
        max_completion_length (`int`, *optional*):
            Maximum length of the completion used for truncation. The completion is the concatenation of the steps.
        disable_dropout (`bool`, *optional*, defaults to `True`):
            Whether to disable dropout in the model.
        step_separator (`str`, *optional*, defaults to `"\n"`):
            Separator used to separate each step of the reasoning process.
        train_on_last_step_only (`bool`, *optional*, defaults to `False`):
            Whether to train only on the last step.
        dataset_num_proc (`int`, *optional*):
            Number of processes to use for processing the dataset.

    > [!NOTE]
    > These parameters have default values different from [`~transformers.TrainingArguments`]:
    > - `logging_steps`: Defaults to `10` instead of `500`.
    > - `gradient_checkpointing`: Defaults to `True` instead of `False`.
    > - `bf16`: Defaults to `True` if `fp16` is not set, instead of `False`.
    > - `learning_rate`: Defaults to `1e-5` instead of `5e-5`.
    """

    # Parameters whose default values are overridden from TrainingArguments
    learning_rate: float = field(
        default=1e-5,
        metadata={"help": "The initial learning rate for AdamW."},
    )

    max_length: int | None = field(
        default=1024,
        metadata={"help": "Maximum length of the sequences (prompt + completion) used for truncation."},
    )
    max_completion_length: int | None = field(
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
    dataset_num_proc: int | None = field(
        default=None,
        metadata={"help": "Number of processes to use for processing the dataset."},
    )
