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
class DPOConfig(TrainingArguments):
    r"""
    Configuration class for the [`DPOTrainer`].

    This class includes only the parameters that are specific to DPO training. For a full list of training arguments,
    please refer to the [`~transformers.TrainingArguments`] documentation. Note that default values in this class may
    differ from those in [`~transformers.TrainingArguments`].

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        > Parameters that control the model and reference model

        model_init_kwargs (`dict[str, Any]`, *optional*):
            Keyword arguments for [`~transformers.AutoModelForCausalLM.from_pretrained`], used when the `model`
            argument of the [`DPOTrainer`] is provided as a string.

        > Parameters that control the data preprocessing

        dataset_num_proc (`int`, *optional*):
            Number of processes to use for processing the dataset.
        pad_token (`str`, *optional*):
            Token used for padding. If `None`, it defaults to `processing_class.pad_token`, or if that is also `None`,
            it falls back to `processing_class.eos_token`.
        max_prompt_length (`int` or `None`, *optional*, defaults to `512`):
            Maximum length of the prompt part of the sequence. If `None`, no truncation is applied.
        max_completion_length (`int` or `None`, *optional*, defaults to `None`):
            Maximum length of the completion part of the sequence. If `None`, no truncation is applied.
        max_length (`int` or `None`, *optional*, defaults to `1024`):
            Maximum length of the tokenized sequence. Sequences longer than `max_length` are truncated from the right.
            If `None`, no truncation is applied.
        truncation_mode (`str`, *optional*, defaults to `"keep_end"`):
            Truncation mode to use when the sequence exceeds `max_length`. Possible values are `"keep_end"` and
            `"keep_start"`.
        padding_free (`bool`, *optional*, defaults to `False`):
            Whether to perform forward passes without padding by flattening all sequences in the batch into a single
            continuous sequence. This reduces memory usage by eliminating padding overhead. Currently, this is only
            supported with the FlashAttention 2 or 3, which can efficiently handle the flattened batch structure.
        pad_to_multiple_of (`int`, *optional*):
            If set, the sequences will be padded to a multiple of this value.
        precompute_ref_log_probs (`bool`, *optional*, defaults to `True`):
            Whether to precompute the reference model log probabilities for the entire training dataset before
            training. This allows to save memory during training, as the reference model does not need to be kept in
            memory.

        > Parameters that control the training

        loss_type (`str` or `list[str]`, *optional*, defaults to `"sigmoid"`):
            Type of loss to use. Possible values are:

                - `"sigmoid"`: sigmoid loss from the original [DPO](https://huggingface.co/papers/2305.18290) paper.
                - `"hinge"`: hinge loss on the normalized likelihood from the
                [SLiC](https://huggingface.co/papers/2305.10425) paper.
        beta (`float`, *optional*, defaults to `0.1`):
            Parameter controlling the deviation from the reference model. Higher β means less deviation from the
            reference model.
        activation_offloading (`bool`, *optional*, defaults to `False`):
            Whether to offload the activations to the CPU.
    """

    _VALID_DICT_FIELDS = TrainingArguments._VALID_DICT_FIELDS + ["model_init_kwargs"]

    # Parameters whose default values are overridden from TrainingArguments
    learning_rate: float = field(
        default=1e-6,
        metadata={"help": "The initial learning rate for AdamW."},
    )
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
    bf16: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether to use bf16 (mixed) precision instead of 32-bit. Requires Ampere or higher NVIDIA "
            "architecture or Intel XPU or using CPU (use_cpu) or Ascend NPU. If not set, it defaults to `True` if "
            "`fp16` is not set."
        },
    )

    # Parameters that control the model
    model_init_kwargs: Optional[dict[str, Any]] = field(
        default=None,
        metadata={
            "help": "Keyword arguments for `AutoModelForCausalLM.from_pretrained`, used when the `model` argument of "
            "the `DPOTrainer` is provided as a string."
        },
    )

    # Parameters that control the data preprocessing
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
    max_prompt_length: Optional[int] = field(
        default=512,
        metadata={"help": "Maximum length of the prompt part of the sequence. If `None`, no truncation is applied."},
    )
    max_completion_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "Maximum length of the completion part of the sequence. If `None`, no truncation is applied."
        },
    )
    max_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "Maximum length of the tokenized sequence. Sequences longer than `max_length` are truncated from "
            "the right. If `None`, no truncation is applied."
        },
    )
    truncation_mode: str = field(
        default="keep_end",
        metadata={
            "help": "Truncation mode to use when the sequence exceeds `max_length`. Possible values are `'keep_end'` "
            "and `'keep_start'`.",
            "choices": ["keep_end", "keep_start"],
        },
    )
    padding_free: bool = field(
        default=False,
        metadata={
            "help": "Whether to perform forward passes without padding by flattening all sequences in the batch into "
            "a single continuous sequence. This reduces memory usage by eliminating padding overhead. Currently, this "
            "is only supported with the FlashAttention 2 or 3, which can efficiently handle the flattened batch "
            "structure."
        },
    )
    pad_to_multiple_of: Optional[int] = field(
        default=None,
        metadata={"help": "If set, the sequences will be padded to a multiple of this value."},
    )
    precompute_ref_log_probs: bool = field(
        default=True,
        metadata={
            "help": "Whether to precompute the reference model log probabilities for the entire training dataset "
            "before training. This allows to save memory during training, as the reference model does not need to be "
            "kept in memory."
        },
    )

    # Parameters that control the training
    loss_type: list[str] = field(
        default_factory=lambda: ["sigmoid"],
        metadata={
            "help": "Type of loss to use. Possible values are: `'sigmoid'`, `'hinge'`.",
        },
    )
    beta: float = field(
        default=0.1,
        metadata={
            "help": "Parameter controlling the deviation from the reference model. Higher β means less deviation from "
            "the reference model."
        },
    )
    activation_offloading: bool = field(
        default=False,
        metadata={"help": "Whether to offload the activations to the CPU."},
    )

    def __post_init__(self):
        self.bf16 = not (self.fp16) if self.bf16 is None else self.bf16

        # Normalize loss_type to string format for internal use
        if hasattr(self.loss_type, "__len__") and len(self.loss_type) == 1:
            self.loss_type = self.loss_type[0]
        super().__post_init__()
