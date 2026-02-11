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
from typing import Any

from transformers import TrainingArguments


@dataclass
class BCOConfig(TrainingArguments):
    r"""
    Configuration class for the [`experimental.bco.BCOTrainer`].

    This class includes only the parameters that are specific to BCO training. For a full list of training arguments,
    please refer to the [`~transformers.TrainingArguments`] documentation. Note that default values in this class may
    differ from those in [`~transformers.TrainingArguments`].

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        max_length (`int` or `None`, *optional*, defaults to `1024`):
            Maximum length of the sequences (prompt + completion) in the batch. This argument is required if you want
            to use the default data collator.
        max_completion_length (`int`, *optional*):
            Maximum length of the completion. This argument is required if you want to use the default data collator
            and your model is an encoder-decoder.
        beta (`float`, *optional*, defaults to `0.1`):
            Parameter controlling the deviation from the reference model. Higher β means less deviation from the
            reference model.
        truncation_mode (`str`, *optional*, defaults to `"keep_end"`):
            Truncation mode to use when the prompt is too long. Possible values are `"keep_end"` or `"keep_start"`.
            This argument is required if you want to use the default data collator.
        disable_dropout (`bool`, *optional*, defaults to `True`):
            Whether to disable dropout in the model and reference model.
        generate_during_eval (`bool`, *optional*, defaults to `False`):
            If `True`, generates and logs completions from both the model and the reference model to W&B or Comet
            during evaluation.
        is_encoder_decoder (`bool`, *optional*):
            When using the `model_init` argument (callable) to instantiate the model instead of the `model` argument,
            you need to specify if the model returned by the callable is an encoder-decoder model.
        precompute_ref_log_probs (`bool`, *optional*, defaults to `False`):
            Whether to precompute reference model log probabilities for training and evaluation datasets. This is
            useful when training without the reference model to reduce the total GPU memory needed.
        model_init_kwargs (`dict[str, Any]`, *optional*):
            Keyword arguments to pass to `AutoModelForCausalLM.from_pretrained` when instantiating the model and
            reference model from strings.
        dataset_num_proc (`int`, *optional*):
            Number of processes to use for processing the dataset.
        prompt_sample_size (`int`, *optional*, defaults to `1024`):
            Number of prompts that are fed to density ratio classifier.
        min_density_ratio (`float`, *optional*, defaults to `0.5`):
            Minimum value of the density ratio. The estimated density ratio is clamped to this value.
        max_density_ratio (`float`, *optional*, defaults to `10.0`):
            Maximum value of the density ratio. The estimated density ratio is clamped to this value.
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
    gradient_checkpointing: bool = field(
        default=True,
        metadata={
            "help": "If True, use gradient checkpointing to save memory at the expense of slower backward pass."
        },
    )
    bf16: bool | None = field(
        default=None,
        metadata={
            "help": "Whether to use bf16 (mixed) precision instead of 32-bit. Requires Ampere or higher NVIDIA "
            "architecture or Intel XPU or using CPU (use_cpu) or Ascend NPU. If not set, it defaults to `True` if "
            "`fp16` is not set."
        },
    )
    # Transformers 4.57.0 introduced a bug that caused the dtype of `lr_scheduler_kwargs` to be unparsable. This issue
    # was fixed in https://github.com/huggingface/transformers/pull/41322 and released in 4.57.5. We add a temporary
    # workaround here, which can be removed once we drop support for versions older than 4.57.5.
    lr_scheduler_kwargs: dict | str | None = field(
        default=None,
        metadata={
            "help": "Additional parameters for the lr_scheduler, such as {'num_cycles': 1} for cosine with hard "
            "restarts."
        },
    )

    max_length: int | None = field(
        default=1024,
        metadata={
            "help": "Maximum length of the sequences (prompt + completion) in the batch. "
            "This argument is required if you want to use the default data collator."
        },
    )
    max_completion_length: int | None = field(
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
    is_encoder_decoder: bool | None = field(
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
    model_init_kwargs: dict[str, Any] | None = field(
        default=None,
        metadata={
            "help": "Keyword arguments to pass to `AutoModelForCausalLM.from_pretrained` when instantiating the "
            "model from a string."
        },
    )
    dataset_num_proc: int | None = field(
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

    def __post_init__(self):
        self.bf16 = not (self.fp16) if self.bf16 is None else self.bf16

        super().__post_init__()
