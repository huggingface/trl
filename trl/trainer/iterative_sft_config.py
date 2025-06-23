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
from typing import Any, Optional

from transformers import TrainingArguments


@dataclass
class IterativeSFTConfig(TrainingArguments):
    r"""
    Configuration class for the [`IterativeSFTTrainer`].

    This class includes only the parameters that are specific to Iterative SFT training. For a full list of training
    arguments, please refer to the [`~transformers.TrainingArguments`] documentation. Note that default values in this
    class may differ from those in [`~transformers.TrainingArguments`].

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        > Parameters that control the model

        model_init_kwargs (`dict[str, Any]` or `None`, *optional*, defaults to `None`):
            Keyword arguments for [`~transformers.AutoModelForCausalLM.from_pretrained`], used when the `model`
            argument of the [`IterativeSFTTrainer`] is provided as a string.

        > Parameters that control the data preprocessing

        max_length (`int` or `None`, *optional*, defaults to `None`):
            Maximum length of the tokenized sequence. Sequences longer than `max_length` are truncated.
        truncation_mode (`str`, *optional*, defaults to `"keep_end"`):
            The truncation mode to use, either `"keep_end"` or `"keep_start"`.
        optimize_device_cache (`bool`, *optional*, defaults to `False`):
            Whether to optimize accelerator cache for slightly more memory-efficient training.
    """

    _VALID_DICT_FIELDS = TrainingArguments._VALID_DICT_FIELDS + ["model_init_kwargs"]

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

    # Parameters that control the model
    model_init_kwargs: Optional[dict[str, Any]] = field(
        default=None,
        metadata={
            "help": "Keyword arguments for `AutoModelForCausalLM.from_pretrained`, used when the `model` argument of "
            "the `IterativeSFTTrainer` is provided as a string."
        },
    )

    # Parameters that control the data preprocessing
    max_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "Maximum length of the tokenized sequence. Sequences longer than `max_length` are truncated."
        },
    )
    truncation_mode: str = field(
        default="keep_end",
        metadata={"help": "The truncation mode to use, either 'keep_end' or 'keep_start'."},
    )
    optimize_device_cache: bool = field(
        default=False,
        metadata={"help": "Whether to optimize accelerator cache for slightly more memory-efficient training."},
    )

    def __post_init__(self):
        self.bf16 = not (self.fp16) if self.bf16 is None else self.bf16

        super().__post_init__()

        if self.truncation_mode not in ["keep_end", "keep_start"]:
            raise ValueError(f"truncation_mode must be either 'keep_end' or 'keep_start', got {self.truncation_mode}")
