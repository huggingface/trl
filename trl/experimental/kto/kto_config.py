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
class KTOConfig(_BaseConfig):
    # docstyle-ignore
    r"""
    Configuration class for the [`experimental.kto.KTOTrainer`].

    This class includes only the parameters that are specific to KTO training. For a full list of training arguments,
    please refer to the [`~transformers.TrainingArguments`] documentation. Note that default values in this class may
    differ from those in [`~transformers.TrainingArguments`].

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        > Parameters that control the model

        model_init_kwargs (`dict[str, Any]`, *optional*):
            Keyword arguments for [`~transformers.AutoModelForCausalLM.from_pretrained`], used when the `model`
            argument of the [`KTOTrainer`] is provided as a string.
        disable_dropout (`bool`, *optional*, defaults to `True`):
            Whether to disable dropout in the model and reference model.

        > Parameters that control the data preprocessing

        dataset_num_proc (`int`, *optional*):
            Number of processes to use for processing the dataset.
        max_length (`int` or `None`, *optional*, defaults to `1024`):
            Maximum length of the tokenized sequence. Sequences longer than `max_length` are truncated from the left.
            If `None`, no truncation is applied.
        precompute_ref_log_probs (`bool`, *optional*, defaults to `False`):
            Whether to precompute the reference model log probabilities for the entire training dataset before
            training. This allows to save memory during training, as the reference model does not need to be kept in
            memory.

        > Parameters that control the training

        loss_type (`str`, *optional*, defaults to `"kto"`):
            Type of loss to use. Possible values are:

                - `"kto"`: KTO loss from the [KTO](https://huggingface.co/papers/2402.01306) paper.
                - `"apo_zero_unpaired"`: Unpaired variant of APO-zero loss from the
                  [APO](https://huggingface.co/papers/2408.06266) paper.

        beta (`float`, *optional*, defaults to `0.1`):
            Parameter controlling the deviation from the reference model. Higher β means less deviation from the
            reference model.
        desirable_weight (`float`, *optional*, defaults to `1.0`):
            Desirable losses are weighed by this factor to counter unequal number of desirable and undesirable pairs.
        undesirable_weight (`float`, *optional*, defaults to `1.0`):
            Undesirable losses are weighed by this factor to counter unequal number of desirable and undesirable pairs.
        generate_during_eval (`bool`, *optional*, defaults to `False`):
            If `True`, generates and logs completions from both the model and the reference model to W&B or Comet
            during evaluation.

    > [!NOTE]
    > These parameters have default values different from [`~transformers.TrainingArguments`]:
    > - `logging_steps`: Defaults to `10` instead of `500`.
    > - `gradient_checkpointing`: Defaults to `True` instead of `False`.
    > - `bf16`: Defaults to `True` if `fp16` is not set, instead of `False`.
    > - `learning_rate`: Defaults to `1e-6` instead of `5e-5`.
    """

    _VALID_DICT_FIELDS = _BaseConfig._VALID_DICT_FIELDS + ["model_init_kwargs"]

    # Parameters whose default values are overridden from TrainingArguments
    learning_rate: float = field(
        default=1e-6,
        metadata={"help": "The initial learning rate for AdamW."},
    )

    # Parameters that control the model
    model_init_kwargs: dict[str, Any] | str | None = field(
        default=None,
        metadata={
            "help": "Keyword arguments for `AutoModelForCausalLM.from_pretrained`, used when the `model` argument of "
            "the `KTOTrainer` is provided as a string."
        },
    )
    disable_dropout: bool = field(
        default=True,
        metadata={"help": "Whether to disable dropout in the model and reference model."},
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
            "the left. If `None`, no truncation is applied."
        },
    )
    precompute_ref_log_probs: bool = field(
        default=False,
        metadata={
            "help": "Whether to precompute the reference model log probabilities for the entire training dataset "
            "before training. This allows to save memory during training, as the reference model does not need to be "
            "kept in memory."
        },
    )

    # Parameters that control the training
    loss_type: str = field(
        default="kto",
        metadata={
            "help": "Type of loss to use.",
            "choices": ["kto", "apo_zero_unpaired"],
        },
    )
    beta: float = field(
        default=0.1,
        metadata={
            "help": "Parameter controlling the deviation from the reference model. Higher β means less deviation from "
            "the reference model."
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
    generate_during_eval: bool = field(
        default=False,
        metadata={
            "help": "If `True`, generates and logs completions from both the model and the reference model to W&B "
            "during evaluation."
        },
    )
