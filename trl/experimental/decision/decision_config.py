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
        remove_unused_columns (`bool`, *optional*, defaults to `False`):
            Retain the raw decision fields required by the collator.
    """

    _VALID_DICT_FIELDS = _BaseConfig._VALID_DICT_FIELDS + ["model_init_kwargs"]

    model_init_kwargs: dict[str, Any] | None = field(default=None, metadata={"help": "Model loading arguments."})
    remove_unused_columns: bool = field(default=False, metadata={"help": "Keep raw decision fields."})

    def __post_init__(self):
        super().__post_init__()
        if self.gradient_checkpointing_kwargs is None:
            self.gradient_checkpointing_kwargs = {"use_reentrant": False}
