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
from typing import Any, Optional

from transformers import TrainingArguments


@dataclass
class CPOConfig(TrainingArguments):
    r"""
    Configuration class for the [`CPOTrainer`].

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
            reference model. For the IPO loss (`loss_type="ipo"`), β is the regularization parameter denoted by τ in
            the [paper](https://huggingface.co/papers/2310.12036).
        label_smoothing (`float`, *optional*, defaults to `0.0`):
            Label smoothing factor. This argument is required if you want to use the default data collator.
        loss_type (`str`, *optional*, defaults to `"sigmoid"`):
            Type of loss to use. Possible values are:

                - `"sigmoid"`: sigmoid loss from the original [DPO](https://huggingface.co/papers/2305.18290) paper.
                - `"hinge"`: hinge loss on the normalized likelihood from the [SLiC](https://huggingface.co/papers/2305.10425) paper.
                - `"ipo"`: IPO loss from the [IPO](https://huggingface.co/papers/2310.12036) paper.
                - `"simpo"`: SimPO loss from the [SimPO](https://huggingface.co/papers/2405.14734) paper.

        disable_dropout (`bool`, *optional*, defaults to `True`):
            Whether to disable dropout in the model.
        cpo_alpha (`float`, *optional*, defaults to `1.0`):
            Weight of the BC regularizer in CPO training.
        simpo_gamma (`float`, *optional*, defaults to `0.5`):
            Target reward margin for the SimPO loss, used only when the `loss_type="simpo"`.
        label_pad_token_id (`int`, *optional*, defaults to `-100`):
            Label pad token id. This argument is required if you want to use the default data collator.
        padding_value (`int` or `None`, *optional*, defaults to `None`):
            Padding value to use. If `None`, the padding value of the tokenizer is used.
        truncation_mode (`str`,*optional*,  defaults to `"keep_end"`):
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
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "Label smoothing factor."},
    )
    loss_type: str = field(
        default="sigmoid",
        metadata={
            "help": "Type of loss to use.",
            "choices": ["sigmoid", "hinge", "ipo", "simpo"],
        },
    )
    disable_dropout: bool = field(
        default=True,
        metadata={"help": "Whether to disable dropout in the model."},
    )
    cpo_alpha: float = field(
        default=1.0,
        metadata={"help": "Weight of the BC regularizer in CPO training."},
    )
    simpo_gamma: float = field(
        default=0.5,
        metadata={"help": "Target reward margin for the SimPO loss, used only when the `loss_type='simpo'`."},
    )
    label_pad_token_id: int = field(
        default=-100,
        metadata={"help": "Label pad token id."},
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
        metadata={"help": "Whether the model is an encoder-decoder model."},
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
