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

import logging

import torch
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)


logger = logging.getLogger(__name__)


class PEFTAdapterEMACallback(TrainerCallback):
    """
    Callback that maintains an EMA copy of PEFT adapter weights for use as a teacher model in self-distillation.

    The callback creates a secondary adapter ("teacher") with zero-initialized weights and maintains shadow weights
    that are updated via exponential moving average: `teacher_weight = (1-α) * teacher_weight + α * student_weight`

    Usage:
        ```python
        trainer.add_callback(
            PEFTAdapterEMACallback(
                model=model,
                teacher_adapter_name="teacher",
                update_rate=0.05,
            )
        )
        ```
    """

    def __init__(
        self,
        model,
        teacher_adapter_name: str = "teacher",
        update_rate: float = 0.05,
        sync_steps: int = 1,
        accelerator=None,
    ):
        self.model = model
        self.teacher_adapter_name = teacher_adapter_name
        self.update_rate = update_rate
        self.sync_steps = sync_steps
        self.accelerator = accelerator
        self.shadow_weights: dict[str, torch.Tensor] | None = None
        self.teacher_adapter_config = None
        self._initialized = False

    def _get_student_state_dict(self):
        """Get student adapter state dict using PEFT keys (without adapter name)."""
        from peft import get_peft_model_state_dict

        if self.accelerator is not None:
            model = self.accelerator.unwrap_model(self.model)
        else:
            model = self.model
        return get_peft_model_state_dict(model)

    def _initialize_teacher_adapter(self):
        """Create teacher adapter with zero weights initialized from student adapter."""
        from peft import get_peft_model_state_dict, set_peft_model_state_dict

        if self._initialized:
            return

        if self.accelerator is not None:
            model = self.accelerator.unwrap_model(self.model)
        else:
            model = self.model

        adapter_name = model.active_adapter
        if adapter_name is None:
            adapter_name = "default"

        self.teacher_adapter_config = model.peft_config.get(adapter_name)

        student_state = get_peft_model_state_dict(model)

        teacher_state = {k: torch.zeros_like(v) for k, v in student_state.items()}

        model.add_adapter(self.teacher_adapter_name, self.teacher_adapter_config)

        model.set_adapter(self.teacher_adapter_name)
        set_peft_model_state_dict(model, teacher_state, adapter_name=self.teacher_adapter_name)

        model.set_adapter(adapter_name)

        self.shadow_weights = {k: v.clone().zero_() for k, v in teacher_state.items()}

        self._initialized = True
        logger.info(f"Initialized PEFT adapter EMA teacher with adapter name: {self.teacher_adapter_name}")

    @torch.no_grad()
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step % self.sync_steps != 0:
            return

        if not self._initialized:
            self._initialize_teacher_adapter()

        if self.shadow_weights is None:
            return

        if self.accelerator is None and "accelerator" in kwargs:
            self.accelerator = kwargs["accelerator"]

        student_state = self._get_student_state_dict()

        for key, student_param in student_state.items():
            if key in self.shadow_weights:
                shadow = self.shadow_weights[key]
                shadow.data = (1 - self.update_rate) * shadow.data + self.update_rate * student_param.data

        from peft import set_peft_model_state_dict

        if self.accelerator is not None:
            unwrapped_model = self.accelerator.unwrap_model(self.model)
        else:
            unwrapped_model = self.model

        original_adapter = unwrapped_model.active_adapter
        unwrapped_model.set_adapter(self.teacher_adapter_name)
        set_peft_model_state_dict(unwrapped_model, self.shadow_weights, adapter_name=self.teacher_adapter_name)
        unwrapped_model.set_adapter(original_adapter)

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if self.accelerator is None and "accelerator" in kwargs:
            self.accelerator = kwargs["accelerator"]
        self._initialize_teacher_adapter()
