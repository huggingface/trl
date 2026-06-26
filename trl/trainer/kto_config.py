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

from .base_config import _BaseConfig


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
        trust_remote_code (`bool`, *optional*, defaults to `False`):
            Whether to allow loading models and tokenizers that ship custom Python code from the Hub. Forwarded to
            [`~transformers.AutoModelForCausalLM.from_pretrained`] and
            [`~transformers.AutoProcessor.from_pretrained`].
        disable_dropout (`bool`, *optional*, defaults to `True`):
            Whether to disable dropout in the model and reference model.

        > Parameters that control the data preprocessing

        dataset_num_proc (`int`, *optional*):
            Number of processes to use for processing the dataset.
        max_length (`int` or `None`, *optional*, defaults to `1024`):
            Maximum length of the tokenized sequence. Sequences longer than `max_length` are truncated from the left.
            If `None`, no truncation is applied.
        pad_to_multiple_of (`int`, *optional*):
            If set, the sequences will be padded to a multiple of this value.
        precompute_ref_log_probs (`bool`, *optional*, defaults to `False`):
            Whether to precompute the reference model log probabilities for the entire training dataset before
            training. This allows to save memory during training, as the reference model does not need to be kept in
            memory.
        precompute_ref_batch_size (`int`, *optional*):
            Batch size to use when precomputing reference model log probabilities. This can be set higher than the
            training batch size to speed up preprocessing. If `None`, defaults to `per_device_train_batch_size` for
            training and `per_device_eval_batch_size` for evaluation.

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
        activation_offloading (`bool`, *optional*, defaults to `False`):
            Whether to offload the activations to the CPU.
        sync_ref_model (`bool`, *optional*, defaults to `False`):
            Whether to synchronize the reference model with the active model every `ref_model_sync_steps` steps, using
            the `ref_model_mixup_alpha` parameter. This synchronization originates from the
            [TR-DPO](https://huggingface.co/papers/2404.09656) paper. `sync_ref_model=True` is not yet compatible with
            PEFT or `precompute_ref_log_probs=True`.
        ref_model_mixup_alpha (`float`, *optional*, defaults to `0.6`):
            α parameter from the TR-DPO paper, which controls the mix between the current policy and the previous
            reference policy during updates. The reference policy is updated according to the equation: `π_ref = α *
            π_θ + (1 - α) * π_ref_prev`. To use this parameter, you must set `sync_ref_model=True`.
        ref_model_sync_steps (`int`, *optional*, defaults to `512`):
            τ parameter from the TR-DPO paper, which determines how frequently the current policy is synchronized with
            the reference policy. To use this parameter, you must set `sync_ref_model=True`.
    > [!NOTE]
    > These parameters have default values different from [`~transformers.TrainingArguments`]:
    > - `logging_steps`: Defaults to `10` instead of `500`.
    > - `gradient_checkpointing`: Defaults to `True` instead of `False`.
    > - `bf16`: Defaults to `True` if `fp16` is not set, instead of `False`.
    > - `learning_rate`: Defaults to `1e-6` instead of `5e-5`.
    > - `train_sampling_strategy`: Defaults to `"sequential"` instead of `"random"`. Loss types
    >   that estimate the KL divergence term (all except `"apo_zero_unpaired"`) require sequential
    >   sampling because the KL completion for each example is precomputed against its neighbors in
    >   a fixed-order batch; any other strategy breaks that pairing.
    """

    _VALID_DICT_FIELDS = _BaseConfig._VALID_DICT_FIELDS + ["model_init_kwargs"]

    # Parameters whose default values are overridden from TrainingArguments
    learning_rate: float = field(
        default=1e-6,
        metadata={"help": "The initial learning rate for AdamW."},
    )
    train_sampling_strategy: str = field(
        default="sequential",
        metadata={
            "help": "Sampler to use for the training dataloader. Loss types that estimate the KL divergence term "
            "(all except `'apo_zero_unpaired'`) require `'sequential'` because the KL completion for each example is "
            "precomputed against its neighbors in a fixed-order batch; any other strategy breaks that pairing. "
            "Possible values are `'random'`, `'sequential'`, and `'group_by_length'`.",
            "choices": ["random", "sequential", "group_by_length"],
        },
    )

    # Parameters that control the model
    model_init_kwargs: dict[str, Any] | str | None = field(
        default=None,
        metadata={
            "help": "Keyword arguments for `AutoModelForCausalLM.from_pretrained`, used when the `model` argument of "
            "the `KTOTrainer` is provided as a string."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": "Whether to allow loading models and tokenizers that ship custom Python code from the Hub. "
            "Forwarded to `AutoModelForCausalLM.from_pretrained` and `AutoProcessor.from_pretrained`."
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
    pad_to_multiple_of: int | None = field(
        default=None,
        metadata={"help": "If set, the sequences will be padded to a multiple of this value."},
    )
    precompute_ref_log_probs: bool = field(
        default=False,
        metadata={
            "help": "Whether to precompute the reference model log probabilities for the entire training dataset "
            "before training. This allows to save memory during training, as the reference model does not need to be "
            "kept in memory."
        },
    )
    precompute_ref_batch_size: int | None = field(
        default=None,
        metadata={
            "help": "Batch size to use when precomputing reference model log probabilities. This can be set higher "
            "than the training batch size to speed up preprocessing. If `None`, defaults to "
            "`per_device_train_batch_size` for training and `per_device_eval_batch_size` for evaluation."
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
    activation_offloading: bool = field(
        default=False,
        metadata={"help": "Whether to offload the activations to the CPU."},
    )
    sync_ref_model: bool = field(
        default=False,
        metadata={
            "help": "Whether to synchronize the reference model with the active model every `ref_model_sync_steps` "
            "steps, using the `ref_model_mixup_alpha` parameter. This synchronization originates from the "
            "[TR-DPO](https://huggingface.co/papers/2404.09656) paper. `sync_ref_model=True` is not yet compatible "
            "with PEFT or `precompute_ref_log_probs=True`."
        },
    )
    ref_model_mixup_alpha: float = field(
        default=0.6,
        metadata={
            "help": "α parameter from the TR-DPO paper, which controls the mix between the current policy and the "
            "previous reference policy during updates. The reference policy is updated according to the equation: "
            "`π_ref = α * π_θ + (1 - α) * π_ref_prev`. To use this parameter, you must set `sync_ref_model=True`."
        },
    )
    ref_model_sync_steps: int = field(
        default=512,
        metadata={
            "help": "τ parameter from the TR-DPO paper, which determines how frequently the current policy is "
            "synchronized with the reference policy. To use this parameter, you must set `sync_ref_model=True`."
        },
    )
