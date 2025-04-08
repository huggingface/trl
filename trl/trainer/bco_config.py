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
class BCOConfig(TrainingArguments):
    r"""
    Configuration class for the [`BCOTrainer`].

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
            Parameter controlling the deviation from the reference model. Higher β means less deviation from the
            reference model.
        label_pad_token_id (`int`,  *optional*, defaults to `-100`):
            Label pad token id. This argument is required if you want to use the default data collator.
        padding_value (`int` or `None`, *optional*, defaults to `None`):
            Padding value to use. If `None`, the padding value of the tokenizer is used.
        truncation_mode (`str`, *optional*, defaults to `"keep_end"`):
            Truncation mode to use when the prompt is too long. Possible values are `"keep_end"` or `"keep_start"`.
            This argument is required if you want to use the default data collator.
        disable_dropout (`bool`, *optional*, defaults to `True`):
            Whether to disable dropout in the model and reference model.
        generate_during_eval (`bool`, *optional*, defaults to `False`):
            If `True`, generates and logs completions from both the model and the reference model to W&B or Comet during
            evaluation.
        is_encoder_decoder (`bool` or `None`, *optional*, defaults to `None`):
            When using the `model_init` argument (callable) to instantiate the model instead of the `model` argument,
            you need to specify if the model returned by the callable is an encoder-decoder model.
        precompute_ref_log_probs (`bool`, *optional*, defaults to `False`):
            Whether to precompute reference model log probabilities for training and evaluation datasets. This is
            useful when training without the reference model to reduce the total GPU memory needed.
        model_init_kwargs (`dict[str, Any]` or `None`, *optional*, defaults to `None`):
            Keyword arguments to pass to `AutoModelForCausalLM.from_pretrained` when instantiating the model from a
            string.
        ref_model_init_kwargs (`dict[str, Any]` or `None`, *optional*, defaults to `None`):
            Keyword arguments to pass to `AutoModelForCausalLM.from_pretrained` when instantiating the reference model
            from a string.
        dataset_num_proc (`int` or `None`, *optional*, defaults to `None`):
            Number of processes to use for processing the dataset.
        prompt_sample_size (`int`, *optional*, defaults to `1024`):
            Number of prompts that are fed to density ratio classifier.
        min_density_ratio (`float`, *optional*, defaults to `0.5`):
            Minimum value of the density ratio. The estimated density ratio is clamped to this value.
        max_density_ratio (`float`, *optional*, defaults to `10.0`):
            Maximum value of the density ratio. The estimated density ratio is clamped to this value.
    """

    max_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "Maximum length of the sequences (prompt + completion) in the batch. "
            "This argument is required if you want to use the default data collator."
        },
    )
    max_prompt_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "Maximum length of the prompt. "
            "This argument is required if you want to use the default data collator."
        },
    )
    max_completion_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "Maximum length of the completion. This argument is required if you want to use the "
            "default data collator and your model is an encoder-decoder."
        },
    )
    beta: float = field(
        default=0.1,
        metadata={
            "help": "Parameter controlling the deviation from the reference model. "
            "Higher β means less deviation from the reference model."
        },
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
            "help": "Truncation mode to use when the prompt is too long. Possible values are "
            "`keep_end` or `keep_start`. This argument is required if you want to use the "
            "default data collator."
        },
    )
    disable_dropout: bool = field(
        default=True,
        metadata={"help": "Whether to disable dropout in the model and reference model."},
    )
    generate_during_eval: bool = field(
        default=False,
        metadata={
            "help": "If `True`, generates and logs completions from both the model and the reference model "
            "to W&B during evaluation."
        },
    )
    is_encoder_decoder: Optional[bool] = field(
        default=None,
        metadata={
            "help": "When using the `model_init` argument (callable) to instantiate the model instead of the "
            "`model` argument, you need to specify if the model returned by the callable is an "
            "encoder-decoder model."
        },
    )
    precompute_ref_log_probs: bool = field(
        default=False,
        metadata={
            "help": "Whether to precompute reference model log probabilities for training and evaluation datasets. "
            "This is useful when training without the reference model to reduce the total GPU memory "
            "needed."
        },
    )
    model_init_kwargs: Optional[dict[str, Any]] = field(
        default=None,
        metadata={
            "help": "Keyword arguments to pass to `AutoModelForCausalLM.from_pretrained` when instantiating the "
            "model from a string."
        },
    )
    ref_model_init_kwargs: Optional[dict[str, Any]] = field(
        default=None,
        metadata={
            "help": "Keyword arguments to pass to `AutoModelForCausalLM.from_pretrained` when instantiating the "
            "reference model from a string."
        },
    )
    dataset_num_proc: Optional[int] = field(
        default=None,
        metadata={"help": "Number of processes to use for processing the dataset."},
    )
    prompt_sample_size: int = field(
        default=1024,
        metadata={"help": "Number of prompts that are fed to density ratio classifier."},
    )
    min_density_ratio: float = field(
        default=0.5,
        metadata={"help": "Minimum value of the density ratio. The estimated density ratio is clamped to this value."},
    )
    max_density_ratio: float = field(
        default=10.0,
        metadata={"help": "Maximum value of the density ratio. The estimated density ratio is clamped to this value."},
    )
