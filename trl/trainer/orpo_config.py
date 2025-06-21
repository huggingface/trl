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
class ORPOConfig(TrainingArguments):
    r"""
    Configuration class for the [`ORPOTrainer`].

    This class includes only the parameters that are specific to ORPO training. For a full list of training arguments,
    please refer to the [`~transformers.TrainingArguments`] documentation. Note that default values in this class may
    differ from those in [`~transformers.TrainingArguments`].

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        max_length (`int` or `None`, *optional*, defaults to `1024`):
            Maximum length of the sequences (prompt + completion) in the batch. This argument is required if you want
            to use the default data collator.
        max_prompt_length (`int` or `None`, *optional*, defaults to `512`):
            Maximum length of the prompt. This argument is required if you want to use the default data collator.
        max_completion_length (`int` or `None`, *optional*, defaults to `None`):
            Maximum length of the completion. This argument is required if you want to use the default data collator
            and your model is an encoder-decoder.
        beta (`float`, *optional*, defaults to `0.1`):
            Parameter controlling the relative ratio loss weight in the ORPO loss. In the
            [paper](https://huggingface.co/papers/2403.07691), it is denoted by λ. In the
            [code](https://github.com/xfactlab/orpo), it is denoted by `alpha`.
        disable_dropout (`bool`, *optional*, defaults to `True`):
            Whether to disable dropout in the model.
        label_pad_token_id (`int`, *optional*, defaults to `-100`):
            Label pad token id. This argument is required if you want to use the default data collator.
        padding_value (`int` or `None`, *optional*, defaults to `None`):
            Padding value to use. If `None`, the padding value of the tokenizer is used.
        truncation_mode (`str`, *optional*, defaults to `"keep_end"`):
            Truncation mode to use when the prompt is too long. Possible values are `"keep_end"` or `"keep_start"`.
            This argument is required if you want to use the default data collator.
        generate_during_eval (`bool`, *optional*, defaults to `False`):
            If `True`, generates and logs completions from the model to W&B or Comet during evaluation.
        is_encoder_decoder (`bool` or `None`, *optional*, defaults to `None`):
            When using the `model_init` argument (callable) to instantiate the model instead of the `model` argument,
            you need to specify if the model returned by the callable is an encoder-decoder model.
        model_init_kwargs (`dict[str, Any]` or `None`, *optional*, defaults to `None`):
            Keyword arguments to pass to `AutoModelForCausalLM.from_pretrained` when instantiating the model from a
            string.
        dataset_num_proc (`int` or `None`, *optional*, defaults to `None`):
            Number of processes to use for processing the dataset.
    """

    _VALID_DICT_FIELDS = TrainingArguments._VALID_DICT_FIELDS + ["model_init_kwargs"]

    # Parameters whose default values are overridden from TrainingArguments
    learning_rate: float = field(
        default=1e-6,
        metadata={"help": "The initial learning rate for AdamW."},
    )
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

    max_length: Optional[int] = field(
        default=1024,
        metadata={"help": "Maximum length of the sequences (prompt + completion) in the batch."},
    )
    max_prompt_length: Optional[int] = field(
        default=512,
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
    beta: float = field(
        default=0.1,
        metadata={
            "help": "Parameter controlling the relative ratio loss weight in the ORPO loss. In the paper, it is "
            "denoted by λ."
        },
    )
    disable_dropout: bool = field(
        default=True,
        metadata={"help": "Whether to disable dropout in the model."},
    )
    label_pad_token_id: int = field(
        default=-100,
        metadata={
            "help": "Label pad token id. This argument is required if you want to use the default data collator."
        },
    )
    padding_value: Optional[int] = field(
        default=None,
        metadata={"help": "Padding value to use. If `None`, the padding value of the tokenizer is used."},
    )
    truncation_mode: str = field(
        default="keep_end",
        metadata={
            "help": "Truncation mode to use when the prompt is too long.",
            "choices": ["keep_end", "keep_start"],
        },
    )
    generate_during_eval: bool = field(
        default=False,
        metadata={"help": "If `True`, generates and logs completions from the model to W&B during evaluation."},
    )
    is_encoder_decoder: Optional[bool] = field(
        default=None,
        metadata={
            "help": "When using the `model_init` argument (callable) to instantiate the model instead of the `model` "
            "argument, you need to specify if the model returned by the callable is an encoder-decoder model."
        },
    )
    model_init_kwargs: Optional[dict[str, Any]] = field(
        default=None,
        metadata={
            "help": "Keyword arguments to pass to `AutoModelForCausalLM.from_pretrained` when instantiating the model "
            "from a string."
        },
    )
    dataset_num_proc: Optional[int] = field(
        default=None,
        metadata={"help": "Number of processes to use for processing the dataset."},
    )

    def __post_init__(self):
        self.bf16 = not (self.fp16) if self.bf16 is None else self.bf16

        super().__post_init__()
