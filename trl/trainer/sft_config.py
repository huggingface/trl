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

import warnings
from dataclasses import dataclass, field
from typing import Any, Optional

from transformers import TrainingArguments


@dataclass
class SFTConfig(TrainingArguments):
    r"""
    Configuration class for the [`SFTTrainer`].

    Only the parameters specific to SFT training are listed here. For details on other parameters, refer to the
    [`~transformers.TrainingArguments`] documentation.

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        > Parameters that control the model

        model_init_kwargs (`dict[str, Any]` or `None`, *optional*, defaults to `None`):
            Keyword arguments for [`~transformers.AutoModelForCausalLM.from_pretrained`], used when the `model`
            argument of the [`SFTTrainer`] is provided as a string.

        > Parameters that control the data preprocessing

        dataset_text_field (`str`, *optional*, defaults to `"text"`):
            Name of the column that contains text data in the dataset.
        dataset_kwargs (`dict[str, Any]` or `None`, *optional*, defaults to `None`):
            Dictionary of optional keyword arguments for the dataset preparation. The only supported key is
            `skip_prepare_dataset`.
        dataset_num_proc (`int` or `None`, *optional*, defaults to `None`):
            Number of processes to use for processing the dataset.
        pad_token (`str` or `None`, *optional*, defaults to `None`):
            Token used for padding. If `None`, it defaults to `processing_class.pad_token`, or if that is also `None`,
            it falls back to `processing_class.eos_token`.
        max_length (`int` or `None`, *optional*, defaults to `1024`):
            Maximum length of the tokenized sequence. Sequences longer than `max_length` are truncated from the right.
            If `None`, no truncation is applied. When packing is enabled, this value sets the sequence length.
        packing (`bool`, *optional*, defaults to `False`):
            Whether to pack multiple sequences into a fixed-length format. Uses `max_length` to define sequence length.
        padding_free (`bool`, *optional*, defaults to `False`):
            Whether to perform forward passes without padding by flattening all sequences in the batch into a single
            continuous sequence. This reduces memory usage by eliminating padding overhead. Currently, this is only
            supported with the `flash_attention_2` attention implementation, which can efficiently handle the flattened
            batch structure.
        eval_packing (`bool` or `None`, *optional*, defaults to `None`):
            Whether to pack the eval dataset. If `None`, uses the same value as `packing`.

        > Parameters that control the training

        learning_rate (`float`, *optional*, defaults to `2e-5`):
            Initial learning rate for [`AdamW`] optimizer. The default value replaces that of
            [`~transformers.TrainingArguments`].
    """

    # Parameters that control the model
    model_init_kwargs: Optional[dict[str, Any]] = field(
        default=None,
        metadata={
            "help": "Keyword arguments for `AutoModelForCausalLM.from_pretrained`, used when the `model` argument of "
            "the `SFTTrainer` is provided as a string."
        },
    )

    # Parameters that control the data preprocessing
    dataset_text_field: str = field(
        default="text",
        metadata={"help": "Name of the column that contains text data in the dataset."},
    )
    dataset_kwargs: Optional[dict[str, Any]] = field(
        default=None,
        metadata={
            "help": "Dictionary of optional keyword arguments for the dataset preparation. The only supported key is "
            "`skip_prepare_dataset`."
        },
    )
    dataset_num_proc: Optional[int] = field(
        default=None,
        metadata={"help": "Number of processes to use for processing the dataset."},
    )
    pad_token: Optional[str] = field(
        default=None,
        metadata={
            "help": "Token used for padding. If `None`, it defaults to `processing_class.pad_token`, or if that "
            "is also `None`, it falls back to `processing_class.eos_token`."
        },
    )
    max_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "Maximum length of the tokenized sequence. Sequences longer than `max_length` are truncated from"
            "the right. If `None`, no truncation is applied. When packing is enabled, this value sets the "
            "sequence length."
        },
    )
    packing: bool = field(
        default=False,
        metadata={
            "help": "Whether to pack multiple sequences into a fixed-length format. Uses `max_length` to define "
            "sequence length."
        },
    )
    padding_free: bool = field(
        default=False,
        metadata={
            "help": "Whether to perform forward passes without padding by flattening all sequences in the batch into "
            "a single continuous sequence. This reduces memory usage by eliminating padding overhead. Currently, "
            "this is only supported with the `flash_attention_2` attention implementation, which can efficiently "
            "handle the flattened batch structure."
        },
    )
    eval_packing: Optional[bool] = field(
        default=None,
        metadata={"help": "Whether to pack the eval dataset. If `None`, uses the same value as `packing`."},
    )

    # Parameters that control the training
    learning_rate: float = field(
        default=2.0e-5,
        metadata={
            "help": "Initial learning rate for `AdamW` optimizer. The default value replaces that of "
            "`TrainingArguments`."
        },
    )

    # Deprecated parameters
    dataset_batch_size: Optional[int] = field(
        default=None,
        metadata={"help": "Deprecated. You can safely remove this parameter from your configuration."},
    )
    num_of_sequences: Optional[int] = field(
        default=None,
        metadata={
            "help": "Deprecated. Use `max_length` instead, which specifies the maximum length of the tokenized "
            "sequence, unlike `num_of_sequences`, which referred to string sequences."
        },
    )
    chars_per_token: Optional[float] = field(
        default=None,
        metadata={"help": "Deprecated. If you want to customize the packing length, use `max_length`."},
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={"help": "Deprecated. Use `max_length` instead."},
    )
    use_liger: Optional[bool] = field(
        default=None,
        metadata={"help": "Deprecated. Use `use_liger_kernel` instead."},
    )

    def __post_init__(self):
        super().__post_init__()

        if self.dataset_batch_size is not None:
            warnings.warn(
                "`dataset_batch_size` is deprecated and will be remove in version 0.18.0. You can safely remove this "
                "parameter from your configuration.",
                DeprecationWarning,
            )

        if self.num_of_sequences is not None:
            warnings.warn(
                "`num_of_sequences` is deprecated and will be remove in version 0.18.0. Use `max_length` instead, "
                "which specifies the maximum length of the tokenized sequence, unlike `num_of_sequences`, which r"
                "eferred to string sequences.",
                DeprecationWarning,
            )

        if self.chars_per_token is not None:
            warnings.warn(
                "`chars_per_token` is deprecated and will be remove in version 0.18.0. If you want to customize the "
                "packing length, use `max_length`.",
                DeprecationWarning,
            )

        if self.max_seq_length is not None:
            warnings.warn(
                "`max_seq_length` is deprecated and will be remove in version 0.20.0. Use `max_length` instead.",
                DeprecationWarning,
            )
            self.max_length = self.max_seq_length

        if self.use_liger is not None:
            warnings.warn(
                "`use_liger` is deprecated and will be remove in version 0.18.0. Use `use_liger_kernel` instead.",
                DeprecationWarning,
            )
            self.use_liger_kernel = self.use_liger
