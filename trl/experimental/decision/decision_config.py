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
class DecisionConfig(_BaseConfig):
    """
    Configuration for [`experimental.decision.DecisionTrainer`].

    All standard [`~transformers.TrainingArguments`] are available.

    Args:
        model_init_kwargs (`dict[str, Any]`, *optional*):
            Arguments passed to [`~transformers.AutoModelForDecision.from_pretrained`] for a model identifier.
        mode (`str`, *optional*, defaults to `"head"`):
            Train the projections and decision stack (`"head"`), also train the encoder's adapter (`"lora"`), or train
            the complete model (`"full"`).
        max_state_length (`int`, *optional*, defaults to `9000`):
            Maximum number of tokens in a state/question input.
        max_option_length (`int`, *optional*, defaults to `8192`):
            Maximum number of tokens per option.
        truncation (`str`, *optional*, defaults to `"error"`):
            Reject overflowing inputs (`"error"`) or explicitly truncate them (`"truncate"`).
        option_chunk_size (`int`, *optional*, defaults to `8`):
            Number of options per encoder forward. `None` encodes all options in one forward.
        router_aux_loss_coef (`float`, *optional*):
            MoE load-balancing coefficient. Defaults to the encoder configuration's coefficient, or zero for a dense
            encoder. Set to zero to disable the auxiliary loss.
        remove_unused_columns (`bool`, *optional*, defaults to `False`):
            Retain the raw decision fields required by the collator.
    """

    _VALID_DICT_FIELDS = _BaseConfig._VALID_DICT_FIELDS + ["model_init_kwargs"]

    model_init_kwargs: dict[str, Any] | None = field(default=None, metadata={"help": "Model loading arguments."})
    mode: str = field(default="head", metadata={"help": "Train head, lora, or full model."})
    max_state_length: int = field(default=9000, metadata={"help": "Maximum state/question token length."})
    max_option_length: int = field(default=8192, metadata={"help": "Maximum option token length."})
    truncation: str = field(default="error", metadata={"help": "Overflow policy: error or truncate."})
    option_chunk_size: int | None = field(default=8, metadata={"help": "Options per encoder forward."})
    router_aux_loss_coef: float | None = field(default=None, metadata={"help": "MoE load-balancing coefficient."})
    remove_unused_columns: bool = field(default=False, metadata={"help": "Keep raw decision fields."})

    def __post_init__(self):
        super().__post_init__()
        if self.mode not in {"head", "lora", "full"}:
            raise ValueError("mode must be head, lora, or full")
        if self.truncation not in {"error", "truncate"}:
            raise ValueError("truncation must be error or truncate")
        if self.max_state_length < 1 or self.max_option_length < 1:
            raise ValueError("max_state_length and max_option_length must be positive")
        if self.option_chunk_size is not None and self.option_chunk_size < 1:
            raise ValueError("option_chunk_size must be positive or None")
        if self.remove_unused_columns:
            raise ValueError("The decision collator requires remove_unused_columns=False")
        # Flattened options have a different leading dimension from states. Shard examples before collation.
        self.accelerator_config.dispatch_batches = False
        self.accelerator_config.split_batches = False
        if self.gradient_checkpointing_kwargs is None:
            self.gradient_checkpointing_kwargs = {"use_reentrant": False}
