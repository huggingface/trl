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
from typing import Any

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
        TODO
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
    bf16: bool | None = field(
        default=None,
        metadata={
            "help": "Whether to use bf16 (mixed) precision instead of 32-bit. Requires Ampere or higher NVIDIA "
            "architecture or Intel XPU or using CPU (use_cpu) or Ascend NPU. If not set, it defaults to `True` if "
            "`fp16` is not set."
        },
    )

    # Parameters that control the model
    model_init_kwargs: dict[str, Any] | None = field(
        default=None,
        metadata={
            "help": "Keyword arguments for `AutoModelForCausalLM.from_pretrained`, used when the `model` argument of "
            "the `DPOTrainer` is provided as a string."
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
    pad_token: str | None = field(
        default=None,
        metadata={
            "help": "Token used for padding. If `None`, it defaults to `processing_class.pad_token`, or if that "
            "is also `None`, it falls back to `processing_class.eos_token`."
        },
    )
    max_prompt_length: int | None = field(
        default=None,
        metadata={"help": "Maximum length of the prompt part of the sequence. If `None`, no truncation is applied."},
    )
    max_completion_length: int | None = field(
        default=None,
        metadata={
            "help": "Maximum length of the completion part of the sequence. If `None`, no truncation is applied."
        },
    )
    max_length: int | None = field(
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

    # Parameters that control the training
    loss_type: list[str] = field(
        default_factory=lambda: ["sigmoid"],
        metadata={
            "help": "Type of loss to use. Possible values are: `'sigmoid'`, `'hinge'`, `'robust'`.",
        },
    )
    label_smoothing: float = field(
        default=0.0,
        metadata={
            "help": (
                "Label smoothing parameter used in Robust DPO, interpreted as the probability that a preference "
                "label is flipped. Must lie in [0.0, 0.5). A typical value recommended by the Robust DPO paper is 0.1."
            ),
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
    sync_ref_model: bool = field(
        default=False,
        metadata={
            "help": "Whether to synchronize the reference model with the main model at each training step. This is "
            "useful when training with PEFT methods, where the main model's weights change during training."
        },
    )

    def __post_init__(self):
        self.bf16 = not (self.fp16) if self.bf16 is None else self.bf16

        # Normalize loss_type to string format for internal use
        if hasattr(self.loss_type, "__len__") and len(self.loss_type) == 1:
            self.loss_type = self.loss_type[0]
        super().__post_init__()
