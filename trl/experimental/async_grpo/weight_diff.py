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
Delta-compressed weight synchronization engine.

Implements sparse weight patching for AsyncGRPOTrainer: detects which bf16 weights changed
between consecutive optimizer steps, encodes only the changed elements as sparse safetensors
patches, and provides a checkpoint chain (anchor + deltas) for reconstructing any step.

References:
- PULSE paper: arXiv:2602.03839 (Feb 2026)
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from safetensors import safe_open


logger = logging.getLogger(__name__)


def bf16_absorption_threshold(w: torch.Tensor) -> torch.Tensor:
    """BF16 absorption threshold per element: |delta_w| must exceed this to survive rounding.

    BF16 has 7 mantissa bits. An fp32 update is absorbed when |delta_w| < |w| / 256.
    Reference: PULSE paper Definition A.3, Equation (4).
    """
    return w.abs() * (2.0**-8)


class ULPChangeDetector:
    """Detects which bf16 weights change across an optimizer step.

    Hooks into the Adam optimizer via ``register_step_pre_hook`` / ``register_step_post_hook``
    (PyTorch >= 2.1). Runs two passes per optimizer step:

    Pre-step (ULP prediction): uses existing Adam state (m, v) to predict which weights will
    change after casting back to bf16.

    Post-step (ground truth): compares post-step bf16 cast against pre-step snapshot.
    """

    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer):
        self.model = model
        self.optimizer = optimizer

        self._predicted_masks: dict[str, torch.Tensor] = {}
        self._validated_masks: dict[str, torch.Tensor] = {}
        self._pre_step_bf16: dict[str, torch.Tensor] = {}

        # Build param_id -> name mapping
        self._param_id_to_name: dict[int, str] = {}
        for name, param in model.named_parameters():
            name = name.removeprefix("module.")
            self._param_id_to_name[id(param)] = name

        self._pre_hook_handle = optimizer.register_step_pre_hook(self._pre_step_hook)
        self._post_hook_handle = optimizer.register_step_post_hook(self._post_step_hook)

    def _pre_step_hook(self, optimizer, args, kwargs) -> None:
        self._predicted_masks.clear()
        self._pre_step_bf16.clear()

        for group in optimizer.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group.get("weight_decay", 0.0)

            for p in group["params"]:
                if p.grad is None:
                    continue
                pid = id(p)
                name = self._param_id_to_name.get(pid)
                if name is None:
                    continue

                state = optimizer.state.get(p, {})
                if "exp_avg" not in state or "exp_avg_sq" not in state:
                    self._pre_step_bf16[name] = p.detach().to(torch.bfloat16).cpu().clone()
                    continue

                step_count = state.get("step", torch.tensor(1)).item() if "step" in state else 1
                m = state["exp_avg"]
                v = state["exp_avg_sq"]

                with torch.no_grad():
                    m_hat = m / (1 - beta1**step_count)
                    v_hat = v / (1 - beta2**step_count)
                    predicted_delta = lr * m_hat / (v_hat.sqrt() + eps)
                    if weight_decay > 0:
                        predicted_delta = predicted_delta + lr * weight_decay * p.data
                    threshold = bf16_absorption_threshold(p.data)
                    self._predicted_masks[name] = (predicted_delta.abs() > threshold).cpu()

                self._pre_step_bf16[name] = p.detach().to(torch.bfloat16).cpu().clone()

    def _post_step_hook(self, optimizer, args, kwargs) -> None:
        self._validated_masks.clear()

        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                pid = id(p)
                name = self._param_id_to_name.get(pid)
                if name is None or name not in self._pre_step_bf16:
                    continue

                post_bf16 = p.detach().to(torch.bfloat16).cpu()
                self._validated_masks[name] = post_bf16 != self._pre_step_bf16[name]

    def get_changed_params(self, use_validated: bool = True) -> dict[str, torch.Tensor]:
        masks = self._validated_masks if use_validated else self._predicted_masks
        return {name: mask for name, mask in masks.items() if mask.any()}

    def get_prediction_accuracy(self) -> dict[str, float]:
        total_tp, total_fp, total_fn = 0, 0, 0
        total_changed, total_elements = 0, 0

        for name, validated in self._validated_masks.items():
            predicted = self._predicted_masks.get(name)
            n_validated = validated.sum().item()
            total_changed += n_validated
            total_elements += validated.numel()

            if predicted is None:
                total_fn += n_validated
                continue

            tp = (predicted & validated).sum().item()
            fp = (predicted & ~validated).sum().item()
            fn = (~predicted & validated).sum().item()
            total_tp += tp
            total_fp += fp
            total_fn += fn

        precision = total_tp / max(total_tp + total_fp, 1)
        recall = total_tp / max(total_tp + total_fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        sparsity = 1.0 - total_changed / max(total_elements, 1)

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "sparsity": sparsity,
            "total_changed": total_changed,
            "total_elements": total_elements,
        }

    def close(self):
        self._pre_hook_handle.remove()
        self._post_hook_handle.remove()


@dataclass
class PatchMetadata:
    format: str = "sparse_weight_patch"
    version: str = "1"
    model_version: int = 0
    prev_model_version: int = -1
    anchor_step: int = 0
    base_model_id: str = ""
    num_changed_params: int = 0
    total_changed_elements: int = 0
    total_elements: int = 0
    sparsity: float = 0.0
    checksum_sha256: str = ""
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
            else:
                kwargs[k] = v
        return cls(**kwargs)


def compute_bf16_checksum(bf16_state_dict: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for name in sorted(bf16_state_dict.keys()):
        h.update(bf16_state_dict[name].detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def encode_sparse_patch(
    prev_bf16: dict[str, torch.Tensor],
    curr_bf16: dict[str, torch.Tensor],
    model_version: int,
    prev_model_version: int,
    anchor_step: int,
    base_model_id: str = "",
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Encode a sparse weight patch between two bf16 snapshots.

    Returns ``(tensors_dict, metadata_dict)`` for ``safetensors.save_file()``.
    Values are actual values, not additive deltas, so reconstruction is bit-exact (PULSE Prop A.6).
    """
    tensors: dict[str, torch.Tensor] = {}
    changed_param_names: list[str] = []
    total_changed = 0
    total_elements = 0

    for name in sorted(curr_bf16.keys()):
        total_elements += curr_bf16[name].numel()
        changed_mask = curr_bf16[name] != prev_bf16[name]
        num_changed = changed_mask.sum().item()
        if num_changed == 0:
            continue
        changed_param_names.append(name)
        total_changed += num_changed
        flat_indices = changed_mask.flatten().nonzero(as_tuple=False).squeeze(1).to(torch.int32)
        flat_values = curr_bf16[name].flatten()[flat_indices.long()]
        tensors[f"{name}.indices"] = flat_indices.cpu()
        tensors[f"{name}.values"] = flat_values.cpu()

    checksum = compute_bf16_checksum(curr_bf16)
    sparsity = 1.0 - total_changed / max(total_elements, 1)

    meta = PatchMetadata(
        model_version=model_version,
        prev_model_version=prev_model_version,
        anchor_step=anchor_step,
        base_model_id=base_model_id,
        num_changed_params=len(changed_param_names),
        total_changed_elements=total_changed,
        total_elements=total_elements,
        sparsity=sparsity,
        checksum_sha256=checksum,
        changed_params=json.dumps(changed_param_names),
    )

    return tensors, meta.to_metadata_dict()


def apply_sparse_patch(
    base_bf16: dict[str, torch.Tensor],
    patch_path: str | Path,
    verify_checksum: bool = True,
) -> dict[str, torch.Tensor]:
    """Apply a sparse patch to ``base_bf16`` in-place.

    For each changed parameter: ``flat[indices] = values`` (direct assignment, no FP arithmetic).
    Optionally verifies SHA256 checksum. Raises ``ValueError`` on mismatch.
    """
    with safe_open(str(patch_path), framework="pt", device="cpu") as f:
        meta = f.metadata()
        changed_names = json.loads(meta["changed_params"])
        for name in changed_names:
            indices = f.get_tensor(f"{name}.indices").long()
            values = f.get_tensor(f"{name}.values")
            flat = base_bf16[name].flatten()
            flat[indices] = values
            base_bf16[name] = flat.reshape(base_bf16[name].shape)

    if verify_checksum:
        expected = meta.get("checksum_sha256", "")
        if expected:
            actual = compute_bf16_checksum(base_bf16)
            if actual != expected:
                raise ValueError(
                    f"Checksum mismatch after applying patch {patch_path}: "
                    f"expected {expected[:16]}..., got {actual[:16]}..."
                )

    return base_bf16
