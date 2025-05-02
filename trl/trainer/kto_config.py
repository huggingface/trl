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
class KTOConfig(TrainingArguments):
    r"""
    Configuration class for the [`KTOTrainer`].

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        learning_rate (`float`, *optional*, defaults to `1e-6`):
            Initial learning rate for [`AdamW`] optimizer. The default value replaces that of
            [`~transformers.TrainingArguments`].
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
        loss_type (`str`, *optional*, defaults to `"kto"`):
            Type of loss to use. Possible values are:

                - `"kto"`: KTO loss from the [KTO](https://huggingface.co/papers/2402.01306) paper.
                - `"apo_zero_unpaired"`: Unpaired variant of APO-zero loss from the [APO](https://huggingface.co/papers/2408.06266) paper.

        desirable_weight (`float`, *optional*, defaults to `1.0`):
            Desirable losses are weighed by this factor to counter unequal number of desirable and undesirable paris.
        undesirable_weight (`float`, *optional*, defaults to `1.0`):
            Undesirable losses are weighed by this factor to counter unequal number of desirable and undesirable pairs.
        label_pad_token_id (`int`, *optional*, defaults to `-100`):
            Label pad token id. This argument is required if you want to use the default data collator.
        padding_value (`int` or `None`, *optional*, defaults to `None`):
            Padding value to use. If `None`, the padding value of the tokenizer is used.
        truncation_mode (`str`, *optional*, defaults to `"keep_end"`):
            Truncation mode to use when the prompt is too long. Possible values are `"keep_end"` or `"keep_start"`.
            This argument is required if you want to use the default data collator.
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
        dataset_num_proc: (`int` or `None`, *optional*, defaults to `None`):
            Number of processes to use for processing the dataset.
        disable_dropout (`bool`, *optional*, defaults to `True`):
            Whether to disable dropout in the model and reference model.
        use_liger_loss (`bool`, *optional*, defaults to `False`):
            Whether to use Liger loss. It requires liger-kernel to be installed.
        base_model_attribute_name (`str`, *optional*, defaults to `"model"`):
            Name of the attribute in the model that contains the base model. This is used to get the base model from
            the model when the model does not have a `get_decoder` method in the case when `use_liger_loss` is `True`.
    """

    learning_rate: float = field(
        default=1e-6,
        metadata={
            "help": "Initial learning rate for `AdamW` optimizer. The default value replaces that of "
            "`transformers.TrainingArguments`."
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
            "help": "Parameter controlling the deviation from the reference model. Higher β means less deviation from "
            "the reference model."
        },
    )
    loss_type: str = field(
        default="kto",
        metadata={
            "help": "Type of loss to use.",
            "choices": ["kto", "apo_zero_unpaired"],
        },
    )
    desirable_weight: float = field(
        default=1.0,
        metadata={
            "help": "Desirable losses are weighed by this factor to counter unequal number of desirable and "
            "undesirable pairs.",
        },
    )
    undesirable_weight: float = field(
        default=1.0,
        metadata={
            "help": "Undesirable losses are weighed by this factor to counter unequal number of desirable and "
            "undesirable pairs.",
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
            "help": "Truncation mode to use when the prompt is too long.",
            "choices": ["keep_end", "keep_start"],
        },
    )
    generate_during_eval: bool = field(
        default=False,
        metadata={
            "help": "If `True`, generates and logs completions from both the model and the reference model to W&B "
            "during evaluation."
        },
    )
    is_encoder_decoder: Optional[bool] = field(
        default=None,
        metadata={
            "help": "When using the `model_init` argument (callable) to instantiate the model instead of the `model` "
            "argument, you need to specify if the model returned by the callable is an encoder-decoder model."
        },
    )
    disable_dropout: bool = field(
        default=True,
        metadata={"help": "Whether to disable dropout in the model."},
    )
    precompute_ref_log_probs: bool = field(
        default=False,
        metadata={
            "help": "Whether to precompute reference model log probabilities for training and evaluation datasets. "
            "This is useful when training without the reference model to reduce the total GPU memory needed."
        },
    )
    model_init_kwargs: Optional[dict[str, Any]] = field(
        default=None,
        metadata={
            "help": "Keyword arguments to pass to `AutoModelForCausalLM.from_pretrained` when instantiating the model "
            "from a string."
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
    use_liger_loss: bool = field(
        default=False,
        metadata={"help": "Whether to use Liger loss. It requires liger-kernel to be installed."},
    )
    base_model_attribute_name: str = field(
        default="model",
        metadata={
            "help": "Name of the attribute in the model that contains the base model. This is used to get the base "
            "model from the model when the model does not have a `get_decoder` method in the case when "
            "`use_liger_loss` is `True`."
        },
    )
