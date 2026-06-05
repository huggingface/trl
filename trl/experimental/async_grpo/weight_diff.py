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

- ``BF16ChangeDetector``: hooks into the optimizer to detect which bf16 elements actually changed after each step.
- ``PatchMetadata``: structured metadata stored in safetensors headers for both anchor (full) and delta (sparse) weight
  files.
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

    Hooks into the optimizer via ``register_step_pre_hook`` / ``register_step_post_hook`` (PyTorch >= 2.1). Snapshots
    bf16 values before the step, compares after.

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
                # pop so each pre-step snapshot is freed right after it's diffed (peak ~1× snapshot, not 2×).
                self._validated_masks[name] = p.detach().to(torch.bfloat16).cpu() != self._pre_step_bf16.pop(name)

    def close(self):
        self._pre_hook_handle.remove()
        self._post_hook_handle.remove()


def _low_byte(p: torch.Tensor, device: torch.device | str | None = None) -> torch.Tensor:
    """Low byte of each element's bf16 bit pattern as ``uint8`` (1 byte/elem).

    bf16 layout (16 bits): ``[sign | exp(8) | mantissa(7)]``. The low byte holds all 7 mantissa bits plus the exponent
    LSB, so any sub-ULP / mantissa-level update flips it. Snapshotting just this byte costs 1 B/elem — half a full bf16
    clone.

    Stays on ``p``'s device by default (so the change mask is computed on-GPU and only the sparse payload crosses
    PCIe). Pass ``device="cpu"`` to keep the snapshot in host memory when a full on-device snapshot would not fit (e.g.
    DeepSpeed-Z2 holds full params/rank).
    """
    bf16 = p.detach().to(torch.bfloat16)
    if device is not None:
        bf16 = bf16.to(device)
    bf16 = bf16.contiguous()
    # TODO: 0xFF should be configurable either at module level or via some number of precision bits.
    return bf16.view(torch.int16).bitwise_and(0xFF).to(torch.uint8)


class LowByteChangeDetector:
    """Detects changed bf16 weights from a 1-byte-per-element snapshot kept in host (CPU) memory.

    Like [`BF16ChangeDetector`], hooks the optimizer (PyTorch >= 2.1) and diffs pre/post step — but snapshots only the
    **low byte** of each weight's bf16 pattern (1 B/elem) instead of the full bf16 value (2 B/elem). A flipped low byte
    implies the bf16 value changed, so the detected mask is a strict subset of the true change set: **no false
    positives**, but rare false negatives.

    The snapshot is kept on CPU (0 GPU memory footprint) using pre-allocated pinned memory to maximize transfer
    bandwidth and avoid runtime allocation overhead. GPU-side diffing is performed in bounded buckets to prevent VRAM
    explosion while maintaining maximum PCIe saturation.

    ``_validated_masks[name]`` is a boolean tensor, True for each element detected as changed.

    Args:
        model ([`~torch.nn.Module`]):
            Model whose parameters are tracked.
        optimizer ([`~torch.optim.Optimizer`]):
            Optimizer to hook. Must expose native ``register_step_*_hook`` (unwrap Accelerate first).
        validate_recall (`bool`, *optional*, defaults to `False`):
            Also keep a full bf16 snapshot to score low-byte detection against the true diff. Doubles the snapshot
            cost; for diagnostics only.
        snapshot_to_cpu (`bool`, *optional*, defaults to `True`):
            Kept for backwards compatibility. Snapshots are always kept in host memory.
        bucket_mb (`int`, *optional*, defaults to `128`):
            Cap peak GPU memory staging to ~3x this size (e.g. ~384 MB for 128 MB) during transfers and diffing.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        validate_recall: bool = False,
        snapshot_to_cpu: bool = True,
        bucket_mb: int = 128,
    ):
        self.validate_recall = validate_recall
        self.bucket_bytes = bucket_mb * 1024 * 1024
        self._validated_masks: dict[str, torch.Tensor] = {}

        model_params = {p.data_ptr(): name.removeprefix("module.") for name, p in model.named_parameters()}
        self._params: list[tuple[str, torch.Tensor]] = []
        for group in optimizer.param_groups:
            for p in group["params"]:
                name = model_params.get(p.data_ptr())
                if name is not None and p.requires_grad:
                    self._params.append((name, p))

        self._buckets: list[
            list[tuple[str, torch.Tensor, int, int]]
        ] = []  # List of buckets: [(name, p, offset_in_bucket, length)]
        if self._params:
            self._device = self._params[0][1].device

            current_bucket = []
            bucket_size = 0
            for name, p in self._params:
                current_bucket.append((name, p, bucket_size, p.numel()))
                bucket_size += p.numel()
                if bucket_size >= self.bucket_bytes:
                    self._buckets.append(current_bucket)
                    current_bucket = []
                    bucket_size = 0
            if current_bucket:
                self._buckets.append(current_bucket)

            # Pre-allocate pinned CPU memory matching the bucket structures
            self._pinned_pre_low: list[torch.Tensor] = []
            self._pinned_post_mask: list[torch.Tensor] = []
            for bucket in self._buckets:
                total_numel = sum(length for _, _, _, length in bucket)
                self._pinned_pre_low.append(torch.empty(total_numel, dtype=torch.uint8, device="cpu", pin_memory=True))
                self._pinned_post_mask.append(
                    torch.empty(total_numel, dtype=torch.bool, device="cpu", pin_memory=True)
                )

        self._pre_step_bf16: dict[str, torch.Tensor] = {}
        self._accuracy: dict[str, float] = {}

        self._pre_hook_handle = optimizer.register_step_pre_hook(self._pre_step_hook)
        self._post_hook_handle = optimizer.register_step_post_hook(self._post_step_hook)

    def _pre_step_hook(self, optimizer, args, kwargs) -> None:
        self._pre_step_bf16.clear()
        if not self._buckets:
            return

        #  Process and copy bucket-by-bucket asynchronously
        for b_idx, bucket in enumerate(self._buckets):
            gpu_buf = torch.cat([_low_byte(p).view(-1) for _, p, _, _ in bucket])
            self._pinned_pre_low[b_idx].copy_(gpu_buf, non_blocking=True)

        if self.validate_recall:
            for name, p in self._params:
                self._pre_step_bf16[name] = p.detach().to(torch.bfloat16).cpu().clone()

    def _post_step_hook(self, optimizer, args, kwargs) -> None:
        self._validated_masks.clear()
        if not self._buckets:
            return

        for b_idx, bucket in enumerate(self._buckets):
            cur_buf = torch.cat([_low_byte(p).view(-1) for _, p, _, _ in bucket])

            # Fetch only this bucket's pre-step snapshot to GPU and diff
            prev_buf = self._pinned_pre_low[b_idx].to(self._device, non_blocking=True)
            diff = cur_buf != prev_buf

            self._pinned_post_mask[b_idx].copy_(diff, non_blocking=True)

        torch.cuda.synchronize()

        # Unpack masks back to model format
        for b_idx, bucket in enumerate(self._buckets):
            mask_buf = self._pinned_post_mask[b_idx]
            for name, p, offset, length in bucket:
                self._validated_masks[name] = mask_buf[offset : offset + length].view(p.shape).clone()

        if self.validate_recall:
            total_tp, total_true, total_elements = 0, 0, 0
            for name, p in self._params:
                if name not in self._validated_masks or name not in self._pre_step_bf16:
                    continue
                detected = self._validated_masks[name]
                post_bf16 = p.detach().to(torch.bfloat16).cpu()
                true_mask = post_bf16 != self._pre_step_bf16.pop(name)
                total_tp += (detected.cpu() & true_mask).sum().item()
                total_true += true_mask.sum().item()
                total_elements += true_mask.numel()

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
