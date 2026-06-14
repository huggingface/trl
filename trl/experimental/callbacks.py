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
from accelerate.utils import is_peft_model
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from ..trainer.callbacks import SyncRefModelCallback


logger = logging.getLogger(__name__)


def is_pure_lora_training(model, accelerator=None) -> bool:
    """Return `True` when the active adapter is LoRA and every trainable parameter is a LoRA parameter."""
    if not is_peft_model(model):
        return False

    if accelerator is not None:
        model = accelerator.unwrap_model(model)

    adapter_name = model.active_adapter
    if adapter_name is None:
        adapter_name = "default"
    adapter_config = model.peft_config.get(adapter_name)
    peft_type = adapter_config.peft_type
    if peft_type is None or str(peft_type).split(".")[-1] != "LORA":
        return False

    for name, param in model.named_parameters():
        if param.requires_grad and "lora_" not in name:
            return False
    return True


class SyncTeacherModelCallback(SyncRefModelCallback):
    """Synchronize an EMA teacher model with the student model on each configured sync step."""

    def __init__(self, teacher_model, accelerator=None):
        super().__init__(ref_model=teacher_model, accelerator=accelerator)

    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs["model"]
        if self.ref_model is not None and state.global_step % args.teacher_sync_steps == 0:
            if self.accelerator:
                model = self.accelerator.unwrap_model(model)
            self.sync_target_model(model, self.ref_model, args.teacher_update_rate)


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

    def _unwrapped_model(self):
        return self.accelerator.unwrap_model(self.model) if self.accelerator is not None else self.model

    def _get_student_state_dict(self):
        """Student adapter state with full (unsharded) tensors, keyed without the adapter name."""
        from peft import get_peft_model_state_dict
        from peft.utils.integrations import gather_params_ctx
        from torch.distributed.tensor import DTensor

        model = self._unwrapped_model()
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        # ZeRO-3 shards every parameter, so gather before reading; FSDP2 hands back DTensors, so materialize them.
        with gather_params_ctx(trainable_params, modifier_rank=None):
            state = get_peft_model_state_dict(model)
            return {k: (v.full_tensor() if isinstance(v, DTensor) else v).detach().clone() for k, v in state.items()}

    @torch.no_grad()
    def _write_teacher_weights(self, model):
        """Copy the shadow weights into the teacher adapter, re-sharding when the target is sharded."""
        from peft.utils.integrations import gather_params_ctx
        from torch.distributed.tensor import DTensor, distribute_tensor

        infix = f".{self.teacher_adapter_name}."
        targets = [(name, param) for name, param in model.named_parameters() if infix in name]
        # Under ZeRO-3 the teacher parameters are sharded too: gather them, write the full values (identical on
        # every rank), and let DeepSpeed re-partition on exit.
        with gather_params_ctx([param for _, param in targets], modifier_rank=0):
            for name, param in targets:
                shadow = self.shadow_weights.get(name.replace(infix, "."))
                if shadow is None:
                    continue
                value = shadow.to(device=param.device, dtype=param.dtype)
                if isinstance(param.data, DTensor):
                    value = distribute_tensor(value, param.data.device_mesh, param.data.placements)
                param.data.copy_(value)

    def _initialize_teacher_adapter(self):
        """Create the zero-initialized teacher adapter and its full-tensor shadow weights."""
        if self._initialized:
            return

        model = self._unwrapped_model()
        adapter_name = model.active_adapter
        if adapter_name is None:
            adapter_name = "default"
        self.teacher_adapter_config = model.peft_config.get(adapter_name)

        student_state = self._get_student_state_dict()
        self.shadow_weights = {k: torch.zeros_like(v) for k, v in student_state.items()}

        if self.teacher_adapter_name not in model.peft_config:
            model.add_adapter(self.teacher_adapter_name, self.teacher_adapter_config)
        self._write_teacher_weights(model)

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
            shadow = self.shadow_weights.get(key)
            if shadow is not None:
                shadow.mul_(1 - self.update_rate).add_(student_param, alpha=self.update_rate)

        self._write_teacher_weights(self._unwrapped_model())

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if self.accelerator is None and "accelerator" in kwargs:
            self.accelerator = kwargs["accelerator"]
        self._initialize_teacher_adapter()
