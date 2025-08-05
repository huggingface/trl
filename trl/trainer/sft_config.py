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
class SFTConfig(TrainingArguments):
    r"""
    Configuration class for the [`SFTTrainer`].

    This class includes only the parameters that are specific to SFT training. For a full list of training arguments,
    please refer to the [`~transformers.TrainingArguments`] documentation. Note that default values in this class may
    differ from those in [`~transformers.TrainingArguments`].

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        > Parameters that control the model

        model_init_kwargs (`dict[str, Any]` or `None`, *optional*, defaults to `None`):
            Keyword arguments for [`~transformers.AutoModelForCausalLM.from_pretrained`], used when the `model`
            argument of the [`SFTTrainer`] is provided as a string.
        chat_template_path (`str` or `None`, *optional*, defaults to `None`):
            If specified, sets the model's chat template. This can either be the path to a tokenizer (local directory
            or Hugging Face Hub model) or a direct path to a Jinja template file. When using a Jinja file, you must
            ensure that any special tokens referenced in the template are added to the tokenizer and that the model's
            embedding layer is resized accordingly.

        > Parameters that control the data preprocessing

        dataset_text_field (`str`, *optional*, defaults to `"text"`):
            Name of the column that contains text data in the dataset.
        dataset_kwargs (`dict[str, Any]` or `None`, *optional*, defaults to `None`):
            Dictionary of optional keyword arguments for the dataset preparation. The only supported key is
            `skip_prepare_dataset`.
        dataset_num_proc (`int` or `None`, *optional*, defaults to `None`):
            Number of processes to use for processing the dataset.
        eos_token (`str` or `None`, *optional*, defaults to `None`):
            Token used to indicate the end of a turn or sequence. If `None`, it defaults to
            `processing_class.eos_token`.
        pad_token (`int` or `None`, *optional*, defaults to `None`):
            Token used for padding. If `None`, it defaults to `processing_class.pad_token`, or if that is also `None`,
            it falls back to `processing_class.eos_token`.
        max_length (`int` or `None`, *optional*, defaults to `1024`):
            Maximum length of the tokenized sequence. Sequences longer than `max_length` are truncated from the right.
            If `None`, no truncation is applied. When packing is enabled, this value sets the sequence length.
        packing (`bool`, *optional*, defaults to `False`):
            Whether to group multiple sequences into fixed-length blocks to improve computational efficiency and reduce
            padding. Uses `max_length` to define sequence length.
        packing_strategy (`str`, *optional*, defaults to `"bfd"`):
            Strategy for packing sequences. Can be either `"bfd"` (best-fit decreasing, default), or `"wrapped"`.
        padding_free (`bool`, *optional*, defaults to `False`):
            Whether to perform forward passes without padding by flattening all sequences in the batch into a single
            continuous sequence. This reduces memory usage by eliminating padding overhead. Currently, this is only
            supported with the `flash_attention_2` attention implementation, which can efficiently handle the flattened
            batch structure. When packing is enabled with strategy `"bfd"`, padding-free is enabled, regardless of the
            value of this parameter.
        pad_to_multiple_of (`int` or `None`, *optional*, defaults to `None`):
            If set, the sequences will be padded to a multiple of this value.
        eval_packing (`bool` or `None`, *optional*, defaults to `None`):
            Whether to pack the eval dataset. If `None`, uses the same value as `packing`.

        > Parameters that control the training

        completion_only_loss (`bool` or `None`, *optional*, defaults to `None`):
            Whether to compute loss only on the completion part of the sequence. If set to `True`, loss is computed
            only on the completion, which is supported only for [prompt-completion](#prompt-completion) datasets. If
            `False`, loss is computed on the entire sequence. If `None` (default), the behavior depends on the dataset:
            loss is computed on the completion for [prompt-completion](#prompt-completion) datasets, and on the full
            sequence for [language modeling](#language-modeling) datasets.
        assistant_only_loss (`bool`, *optional*, defaults to `False`):
            Whether to compute loss only on the assistant part of the sequence. If set to `True`, loss is computed
            only on the assistant responses, which is supported only for [conversational](#conversational) datasets. If `False`,
            loss is computed on the entire sequence.
        activation_offloading (`bool`, *optional*, defaults to `False`):
            Whether to offload the activations to the CPU.
    """

    _VALID_DICT_FIELDS = TrainingArguments._VALID_DICT_FIELDS + ["model_init_kwargs"]

    # Parameters whose default values are overridden from TrainingArguments
    learning_rate: float = field(
        default=2e-5,
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
    # Note: In transformers>=4.54.0, `average_tokens_across_devices` defaults to True. Overriding this setting is only
    # needed for earlier versions. Once we require transformers>=4.54.0, this line can be safely removed.
    # See https://github.com/huggingface/transformers/pull/39395
    average_tokens_across_devices: bool = field(
        default=True,
        metadata={
            "help": "Whether or not to average tokens across devices. If enabled, will use all_reduce to synchronize "
            "num_tokens_in_batch for precise loss calculation. Reference: https://github.com/huggingface/transformers/issues/34242 "
        },
    )

    # Parameters that control the model
    model_init_kwargs: Optional[dict[str, Any]] = field(
        default=None,
        metadata={
            "help": "Keyword arguments for `AutoModelForCausalLM.from_pretrained`, used when the `model` argument of "
            "the `SFTTrainer` is provided as a string."
        },
    )
    chat_template_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "If specified, sets the model's chat template. This can either be the path to a tokenizer (local "
            "directory or Hugging Face Hub model) or a direct path to a Jinja template file. When using a Jinja file, "
            "you must ensure that any special tokens referenced in the template are added to the tokenizer and "
            "that the model's embedding layer is resized accordingly."
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
    eos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": "Token used to indicate the end of a turn or sequence. If `None`, it defaults to `processing_class.eos_token`."
        },
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
            "help": "Whether to group multiple sequences into fixed-length blocks to improve computational efficiency "
            "and reduce padding. Uses `max_length` to define sequence length."
        },
    )
    packing_strategy: str = field(
        default="bfd",
        metadata={
            "help": "Strategy for packing sequences. Can be either `'bfd'` (best-fit decreasing, default), or "
            "`'wrapped'`."
        },
    )
    padding_free: bool = field(
        default=False,
        metadata={
            "help": "Whether to perform forward passes without padding by flattening all sequences in the batch into "
            "a single continuous sequence. This reduces memory usage by eliminating padding overhead. Currently, "
            "this is only supported with the `flash_attention_2` attention implementation, which can efficiently "
            "handle the flattened batch structure. When packing is enabled with strategy `'bfd'`, padding-free is "
            "enabled, regardless of the value of this parameter."
        },
    )
    pad_to_multiple_of: Optional[int] = field(
        default=None,
        metadata={"help": "If set, the sequences will be padded to a multiple of this value."},
    )
    eval_packing: Optional[bool] = field(
        default=None,
        metadata={"help": "Whether to pack the eval dataset. If `None`, uses the same value as `packing`."},
    )

    # Parameters that control the training
    completion_only_loss: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "Whether to compute loss only on the completion part of the sequence. If set to `True`, loss is "
                "computed only on the completion, which is supported only for prompt-completion datasets. If `False`, "
                "loss is computed on the entire sequence. If `None` (default), the behavior depends on the dataset: "
                "loss is computed on the completion for prompt-completion datasets, and on the full sequence for "
                "language modeling datasets."
            )
        },
    )
    assistant_only_loss: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to compute loss only on the assistant part of the sequence. If set to `True`, loss is "
                "computed only on the assistant responses, which is supported only for conversational datasets. If `False`, "
                "loss is computed on the entire sequence."
            )
        },
    )
    activation_offloading: bool = field(
        default=False,
        metadata={"help": "Whether to offload the activations to the CPU."},
    )

    def __post_init__(self):
        self.bf16 = not (self.fp16) if self.bf16 is None else self.bf16
        super().__post_init__()
