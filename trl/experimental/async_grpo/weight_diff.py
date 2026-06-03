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
from enum import Enum

import torch

from .delta_codec import Encoding


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


def _low_byte(p: torch.Tensor, device: torch.device | str | None = None) -> torch.Tensor:
    """Low byte of each element's bf16 bit pattern as ``uint8`` (1 byte/elem).

    bf16 layout (16 bits): ``[sign | exp(8) | mantissa(7)]``. The low byte holds all 7
    mantissa bits plus the exponent LSB, so any sub-ULP / mantissa-level update flips it.
    Snapshotting just this byte costs 1 B/elem — half a full bf16 clone.

    Stays on ``p``'s device by default (so the change mask is computed on-GPU and only the
    sparse payload crosses PCIe). Pass ``device="cpu"`` to keep the snapshot in host memory
    when a full on-device snapshot would not fit (e.g. DeepSpeed-Z2 holds full params/rank).
    """
    bf16 = p.detach().to(torch.bfloat16)
    if device is not None:
        bf16 = bf16.to(device)
    bf16 = bf16.contiguous()
    # TODO: 0xFF should be configurable either at module level or via some number of precision bits.
    return bf16.view(torch.int16).bitwise_and(0xFF).to(torch.uint8)


class LowByteChangeDetector:
    """Detects changed bf16 weights from a 1-byte-per-element snapshot.

    Like [`BF16ChangeDetector`], hooks the optimizer (PyTorch >= 2.1) and diffs pre/post
    step — but snapshots only the **low byte** of each weight's bf16 pattern (1 B/elem)
    instead of the full bf16 value (2 B/elem). A flipped low byte implies the bf16 value
    changed, so the detected mask is a strict subset of the true change set: **no false
    positives**, but rare false negatives (mantissa + exp-LSB unchanged while a high
    exponent/sign bit changed). Those misses cause inference-side drift, bounded by
    periodic anchors — set ``validate_recall=True`` to measure the miss rate.

    ``_validated_masks[name]`` is a boolean tensor, True for each element detected as changed.

    Args:
        model ([`~torch.nn.Module`]):
            Model whose parameters are tracked.
        optimizer ([`~torch.optim.Optimizer`]):
            Optimizer to hook. Must expose native ``register_step_*_hook`` (unwrap Accelerate first).
        validate_recall (`bool`, *optional*, defaults to `False`):
            Also keep a full bf16 snapshot to score low-byte detection against the true diff.
            Doubles the snapshot cost; for diagnostics only.
        snapshot_to_cpu (`bool`, *optional*, defaults to `False`):
            Keep the low-byte snapshot in host memory instead of on the param's device. Masks
            are then produced on CPU. Use when a full on-device snapshot would not fit (e.g.
            DeepSpeed-Z2 holds full params per rank). Default keeps it on-GPU so the change
            mask and sparse extraction stay on the device.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        validate_recall: bool = False,
        snapshot_to_cpu: bool = False,
    ):
        self.validate_recall = validate_recall
        self._snap_device = "cpu" if snapshot_to_cpu else None
        self._validated_masks: dict[str, torch.Tensor] = {}
        self._pre_step_low: dict[str, torch.Tensor] = {}
        self._pre_step_bf16: dict[str, torch.Tensor] = {}  # only populated when validate_recall
        self._accuracy: dict[str, float] = {}

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
            "LowByteChangeDetector: matched %d/%d optimizer params (validate_recall=%s)",
            len(self._param_id_to_name),
            sum(1 for _ in model.named_parameters()),
            validate_recall,
        )

        self._pre_hook_handle = optimizer.register_step_pre_hook(self._pre_step_hook)
        self._post_hook_handle = optimizer.register_step_post_hook(self._post_step_hook)

    def _pre_step_hook(self, optimizer, args, kwargs) -> None:
        self._pre_step_low.clear()
        self._pre_step_bf16.clear()
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                name = self._param_id_to_name.get(id(p))
                if name is None:
                    continue
                self._pre_step_low[name] = _low_byte(p, self._snap_device)
                if self.validate_recall:
                    self._pre_step_bf16[name] = p.detach().to(torch.bfloat16).to(self._snap_device or p.device).clone()

    def _post_step_hook(self, optimizer, args, kwargs) -> None:
        self._validated_masks.clear()
        total_tp, total_true, total_elements = 0, 0, 0
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                name = self._param_id_to_name.get(id(p))
                if name is None or name not in self._pre_step_low:
                    continue
                detected = _low_byte(p, self._snap_device) != self._pre_step_low[name]
                self._validated_masks[name] = detected
                if self.validate_recall:
                    # True bf16 diff, computed here while p still holds the post-step value.
                    post_bf16 = p.detach().to(torch.bfloat16).to(self._snap_device or p.device)
                    true_mask = post_bf16 != self._pre_step_bf16[name]
                    total_tp += (detected & true_mask).sum().item()
                    total_true += true_mask.sum().item()
                    total_elements += true_mask.numel()
        if self.validate_recall:
            # Low-byte changes ⊆ bf16 changes, so precision is 1.0 by construction;
            # recall = fraction of truly-changed elements the low byte detected.
            self._accuracy = {
                "recall": total_tp / max(total_true, 1),
                "true_changed": total_true,
                "detected_changed": total_tp,
                "total_elements": total_elements,
                "sparsity": 1.0 - total_true / max(total_elements, 1),
            }

    def get_prediction_accuracy(self) -> dict[str, float]:
        return dict(self._accuracy)

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
    encoding: Encoding = Encoding.GAP_DELTA  # index encoding (delta files only)

    def to_metadata_dict(self) -> dict[str, str]:
        return {k: (v.value if isinstance(v, Enum) else str(v)) for k, v in asdict(self).items()}

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
            elif ft == "Encoding":
                kwargs[k] = Encoding(v)
            else:
                kwargs[k] = v
        return cls(**kwargs)
