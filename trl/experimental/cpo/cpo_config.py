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

from ...trainer.base_config import BaseConfig


@dataclass
class CPOConfig(BaseConfig):
    # docstyle-ignore
    r"""
    Configuration class for the [`experimental.cpo.CPOTrainer`].

    This class includes only the parameters that are specific to CPO training. For a full list of training arguments,
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
            reference model. For the IPO loss (`loss_type="ipo"`), β is the regularization parameter denoted by τ in
            the [paper](https://huggingface.co/papers/2310.12036).
        label_smoothing (`float`, *optional*, defaults to `0.0`):
            Label smoothing factor. This argument is required if you want to use the default data collator.
        loss_type (`str`, *optional*, defaults to `"sigmoid"`):
            Type of loss to use. Possible values are:

                - `"sigmoid"`: sigmoid loss from the original [DPO](https://huggingface.co/papers/2305.18290) paper.
                - `"hinge"`: hinge loss on the normalized likelihood from the
                  [SLiC](https://huggingface.co/papers/2305.10425) paper.
                - `"ipo"`: IPO loss from the [IPO](https://huggingface.co/papers/2310.12036) paper.
                - `"simpo"`: SimPO loss from the [SimPO](https://huggingface.co/papers/2405.14734) paper.
                - `"alphapo"`: AlphaPO loss from the [AlphaPO](https://huggingface.co/papers/2501.03884) paper. This
                  automatically sets `loss_type="simpo"` and `cpo_alpha=0.0`.

        disable_dropout (`bool`, *optional*, defaults to `True`):
            Whether to disable dropout in the model.
        cpo_alpha (`float`, *optional*, defaults to `1.0`):
            Weight of the BC regularizer in CPO training.
        simpo_gamma (`float`, *optional*, defaults to `0.5`):
            Target reward margin for the SimPO loss, used only when the `loss_type="simpo"`.
        alpha (`float`, *optional*, defaults to `0.0`):
            Alpha parameter that controls reward function shape across all loss types. When alpha=0 (default), uses
            standard log probability rewards. When `alpha != 0`, applies AlphaPO transformation: `r = (1 - p^(-alpha))
            / alpha` from the [AlphaPO paper](https://huggingface.co/papers/2501.03884). This parameter works with all
            loss types.
        truncation_mode (`str`,*optional*,  defaults to `"keep_end"`):
            Truncation mode to use when the prompt is too long. Possible values are `"keep_end"` or `"keep_start"`.
            This argument is required if you want to use the default data collator.
        generate_during_eval (`bool`, *optional*, defaults to `False`):
            If `True`, generates and logs completions from the model to W&B or Comet during evaluation.
        is_encoder_decoder (`bool`, *optional*):
            When using the `model_init` argument (callable) to instantiate the model instead of the `model` argument,
            you need to specify if the model returned by the callable is an encoder-decoder model.
        model_init_kwargs (`dict[str, Any]`, *optional*):
            Keyword arguments to pass to `AutoModelForCausalLM.from_pretrained` when instantiating the model from a
            string.
        dataset_num_proc (`int`, *optional*):
            Number of processes to use for processing the dataset.

    > [!NOTE]
    > These parameters have default values different from [`~transformers.TrainingArguments`]:
    > - `logging_steps`: Defaults to `10` instead of `500`.
    > - `gradient_checkpointing`: Defaults to `True` instead of `False`.
    > - `bf16`: Defaults to `True` if `fp16` is not set, instead of `False`.
    > - `learning_rate`: Defaults to `1e-6` instead of `5e-5`.
    """

    _VALID_DICT_FIELDS = TrainingArguments._VALID_DICT_FIELDS + ["model_init_kwargs"]

    # Parameters whose default values are overridden from TrainingArguments
    learning_rate: float = field(
        default=1e-6,
        metadata={"help": "The initial learning rate for AdamW."},
    )

    max_length: int | None = field(
        default=1024,
        metadata={"help": "Maximum length of the sequences (prompt + completion) in the batch."},
    )
    max_completion_length: int | None = field(
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
            "choices": ["sigmoid", "hinge", "ipo", "simpo", "alphapo"],
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
    alpha: float = field(
        default=0.0,
        metadata={
            "help": "Alpha parameter that controls reward function shape across all loss types. When alpha=0 "
            "(default), uses standard log probability rewards. When `alpha != 0`, applies AlphaPO transformation: "
            "`r = (1 - p^(-alpha)) / alpha` from the AlphaPO paper. This parameter works with all loss types."
        },
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
    is_encoder_decoder: bool | None = field(
        default=None,
        metadata={"help": "Whether the model is an encoder-decoder model."},
    )
    model_init_kwargs: dict[str, Any] | None = field(
        default=None,
        metadata={
            "help": "Keyword arguments to pass to `AutoModelForCausalLM.from_pretrained` when instantiating the model "
            "from a string."
        },
    )
    dataset_num_proc: int | None = field(
        default=None,
        metadata={"help": "Number of processes to use for processing the dataset."},
    )

    def __post_init__(self):
        # Syntactic sugar for AlphaPO: set loss_type to "simpo" and cpo_alpha to 0.0
        if self.loss_type == "alphapo":
            self.loss_type = "simpo"
            self.cpo_alpha = 0.0

        super().__post_init__()
