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

from ...trainer.base_config import _BaseConfig


@dataclass
class TPOConfig(_BaseConfig):
    # docstyle-ignore
    r"""
    Configuration class for the [`experimental.tpo.TPOTrainer`].

    This class includes only the parameters that are specific to TPO training. For a full list of training arguments,
    please refer to the [`~transformers.TrainingArguments`] documentation. Note that default values in this class may
    differ from those in [`~transformers.TrainingArguments`].

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        > Parameters that control the model

        model_init_kwargs (`dict[str, Any]`, *optional*):
            Keyword arguments for [`~transformers.AutoModelForCausalLM.from_pretrained`], used when the `model`
            argument of the [`experimental.tpo.TPOTrainer`] is provided as a string.
        disable_dropout (`bool`, *optional*, defaults to `True`):
            Whether to disable dropout in the model.

        > Parameters that control the data preprocessing

        dataset_num_proc (`int`, *optional*):
            Number of processes to use for processing the dataset.
        max_length (`int` or `None`, *optional*, defaults to `1024`):
            Maximum length of the tokenized sequence. Sequences longer than `max_length` are truncated from the left or
            right depending on the `truncation_mode`. If `None`, no truncation is applied.
        truncation_mode (`str`, *optional*, defaults to `"keep_start"`):
            Truncation mode to use when the sequence exceeds `max_length`. Possible values are `"keep_start"` and
            `"keep_end"`.
        pad_to_multiple_of (`int`, *optional*):
            If set, the sequences will be padded to a multiple of this value.

        > Parameters that control the training

        loss_type (`str`, *optional*, defaults to `"sigmoid"`):
            Type of loss to use. Possible values are:

                - `"sigmoid"`: sigmoid loss from the original [TPO](https://huggingface.co/papers/2405.16681) paper.
                - `"hinge"`: hinge loss on the normalized likelihood from the
                  [SLiC](https://huggingface.co/papers/2305.10425) paper.
                - `"ipo"`: IPO loss from the [IPO](https://huggingface.co/papers/2310.12036) paper.
                - `"tpo-l"`: length-normalized TPO variant from the
                  [TPO](https://huggingface.co/papers/2405.16681) paper, which adds a target reward margin
                  `tpo_l_gamma` to the Bradley-Terry objective.

        beta (`float`, *optional*, defaults to `0.01`):
            Parameter controlling the temperature of the TPO loss. For the IPO loss (`loss_type="ipo"`), β is the
            regularization parameter denoted by τ in the [paper](https://huggingface.co/papers/2310.12036).
        label_smoothing (`float`, *optional*, defaults to `0.0`):
            Label smoothing factor.
        tpo_alpha (`float`, *optional*, defaults to `1.0`):
            Weight of the supervised negative log-likelihood term computed on the gold (`reference`) response in TPO
            training. Setting `tpo_alpha=0.0` disables the NLL term and skips the corresponding forward pass.
        tpo_l_gamma (`float`, *optional*, defaults to `0.5`):
            Target reward margin γ for the TPO-L loss, used only when `loss_type="tpo-l"`.

    > [!NOTE]
    > These parameters have default values different from [`~transformers.TrainingArguments`]:
    > - `logging_steps`: Defaults to `10` instead of `500`.
    > - `gradient_checkpointing`: Defaults to `True` instead of `False`.
    > - `bf16`: Defaults to `True` if `fp16` is not set, instead of `False`.
    > - `learning_rate`: Defaults to `5e-7` instead of `5e-5`.
    """

    _VALID_DICT_FIELDS = _BaseConfig._VALID_DICT_FIELDS + ["model_init_kwargs"]

    # Parameters whose default values are overridden from TrainingArguments
    learning_rate: float = field(
        default=5e-7,
        metadata={"help": "The initial learning rate for AdamW."},
    )

    # Parameters that control the model
    model_init_kwargs: dict[str, Any] | str | None = field(
        default=None,
        metadata={
            "help": "Keyword arguments for `AutoModelForCausalLM.from_pretrained`, used when the `model` argument of "
            "the `TPOTrainer` is provided as a string."
        },
    )
    disable_dropout: bool = field(
        default=True,
        metadata={"help": "Whether to disable dropout in the model."},
    )

    # Parameters that control the data preprocessing
    dataset_num_proc: int | None = field(
        default=None,
        metadata={"help": "Number of processes to use for processing the dataset."},
    )
    max_length: int | None = field(
        default=1024,
        metadata={
            "help": "Maximum length of the tokenized sequence. Sequences longer than `max_length` are truncated from "
            "the left or right depending on the `truncation_mode`. If `None`, no truncation is applied."
        },
    )
    truncation_mode: str = field(
        default="keep_start",
        metadata={
            "help": "Truncation mode to use when the sequence exceeds `max_length`.",
            "choices": ["keep_end", "keep_start"],
        },
    )
    pad_to_multiple_of: int | None = field(
        default=None,
        metadata={"help": "If set, the sequences will be padded to a multiple of this value."},
    )

    # Parameters that control the training
    loss_type: str = field(
        default="sigmoid",
        metadata={
            "help": "Type of loss to use.",
            "choices": ["sigmoid", "hinge", "ipo", "tpo-l"],
        },
    )
    beta: float = field(
        default=0.01,
        metadata={
            "help": "Parameter controlling the temperature of the TPO loss. For the IPO loss (`loss_type='ipo'`), this "
            "value is the regularization parameter denoted by τ in the IPO paper."
        },
    )
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "Label smoothing factor."},
    )
    tpo_alpha: float = field(
        default=1.0,
        metadata={
            "help": "Weight of the supervised NLL term computed on the gold (`reference`) response in TPO training. "
            "Setting `tpo_alpha=0.0` disables the NLL term and skips the corresponding forward pass."
        },
    )
    tpo_l_gamma: float = field(
        default=0.5,
        metadata={"help": "Target reward margin γ for the TPO-L loss, used only when `loss_type='tpo-l'`."},
    )
