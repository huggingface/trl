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

"""
Delta-compressed weight synchronization utilities.

- ``BF16ChangeDetector``: hooks into the optimizer to detect which bf16 elements
  actually changed after each step.
- ``PatchMetadata``: structured metadata stored in safetensors headers for both
  anchor (full) and delta (sparse) weight files.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import torch


logger = logging.getLogger(__name__)


class BF16ChangeDetector:
    """Detects which bf16 weights actually changed across an optimizer step.

    Hooks into the optimizer via ``register_step_pre_hook`` / ``register_step_post_hook``
    (PyTorch >= 2.1). Snapshots bf16 values before the step, compares after.

    ``_validated_masks[name]`` is a boolean tensor with True for each element that changed.
    """

    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer):
        self._validated_masks: dict[str, torch.Tensor] = {}
        self._pre_step_bf16: dict[str, torch.Tensor] = {}

        # Match model param names to optimizer param objects via data_ptr()
        # (id() doesn't work because Accelerate wraps params as different objects)
        model_params = {p.data_ptr(): name.removeprefix("module.") for name, p in model.named_parameters()}
        self._param_id_to_name: dict[int, str] = {}
        for group in optimizer.param_groups:
            for p in group["params"]:
                name = model_params.get(p.data_ptr())
                if name is not None:
                    self._param_id_to_name[id(p)] = name

        logger.info(
            "BF16ChangeDetector: matched %d/%d optimizer params",
            len(self._param_id_to_name),
            sum(1 for _ in model.named_parameters()),
        )

        self._pre_hook_handle = optimizer.register_step_pre_hook(self._pre_step_hook)
        self._post_hook_handle = optimizer.register_step_post_hook(self._post_step_hook)

    def _pre_step_hook(self, optimizer, args, kwargs) -> None:
        self._pre_step_bf16.clear()
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                name = self._param_id_to_name.get(id(p))
                if name is None:
                    continue
                self._pre_step_bf16[name] = p.detach().to(torch.bfloat16).cpu().clone()

    def _post_step_hook(self, optimizer, args, kwargs) -> None:
        self._validated_masks.clear()
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                name = self._param_id_to_name.get(id(p))
                if name is None or name not in self._pre_step_bf16:
                    continue
                self._validated_masks[name] = p.detach().to(torch.bfloat16).cpu() != self._pre_step_bf16[name]

    def close(self):
        self._pre_hook_handle.remove()
        self._post_hook_handle.remove()


@dataclass
class PatchMetadata:
    format: str = "sparse_weight_patch"
    version: str = "1"
    sparse: bool = False
    model_version: int = 0
    num_changed_params: int = 0
    total_changed_elements: int = 0
    total_elements: int = 0
    sparsity: float = 0.0
    changed_params: str = "[]"

    def to_metadata_dict(self) -> dict[str, str]:
        return {k: str(v) for k, v in asdict(self).items()}

    @classmethod
    def from_metadata_dict(cls, d: dict[str, str]) -> PatchMetadata:
        field_types = {f.name: f.type for f in cls.__dataclass_fields__.values()}
        kwargs = {}
        for k, v in d.items():
            if k not in field_types:
                continue
            ft = field_types[k]
            if ft == "int":
                kwargs[k] = int(v)
            elif ft == "float":
                kwargs[k] = float(v)
            elif ft == "bool":
                kwargs[k] = v == "True"
            else:
                kwargs[k] = v
        return cls(**kwargs)
