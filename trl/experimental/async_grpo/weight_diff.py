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

- ``AdamWInversionChangeDetector``: recovers which bf16 elements changed across an optimizer step by *inverting* the
  AdamW update from the resident moments — no pre-step snapshot kept.
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


def _adamw_reconstruct_pre_step(param: torch.Tensor, state: dict, group: dict) -> torch.Tensor:
    """Reconstruct a parameter's pre-step value from its post-step value and the resident AdamW moments.

    Decoupled AdamW (the ``torch.optim.AdamW`` update):

        ``theta_t = theta_{t-1} * (1 - lr*wd) - (lr/bc1) * m_t / (sqrt(v_t)/sqrt(bc2) + eps)``

    with bias corrections ``bc1 = 1 - beta1**t`` and ``bc2 = 1 - beta2**t``. The moments ``m_t`` / ``v_t`` resident in
    ``optimizer.state`` after the step are exactly the ones the step used, so inverting it recovers ``theta_{t-1}``:

        ``theta_{t-1} = (theta_t + (lr/bc1) * m_t / (sqrt(v_t)/sqrt(bc2) + eps)) / (1 - lr*wd)``

    Returns an ``fp32`` reconstruction on ``param``'s device. The result is exact up to floating-point error, so a
    bf16-rounded comparison may flip rare elements near a rounding boundary (bounded by periodic anchors).
    """
    beta1, beta2 = group["betas"]
    lr, eps, weight_decay = group["lr"], group["eps"], group["weight_decay"]
    step = state["step"]
    t = step.item() if torch.is_tensor(step) else step
    bias_correction1 = 1 - beta1**t
    bias_correction2 = 1 - beta2**t
    exp_avg = state["exp_avg"].float()
    exp_avg_sq = state["exp_avg_sq"].float()
    denom = exp_avg_sq.sqrt() / (bias_correction2**0.5) + eps
    update = (lr / bias_correction1) * exp_avg / denom
    return (param.detach().float() + update) / (1 - lr * weight_decay)


class AdamWInversionChangeDetector:
    """Detects changed bf16 weights by *inverting* the AdamW step from the resident moments.

    The AdamW update is invertible, so we instead reconstruct each param's pre-step value on the fly from the
    ``exp_avg`` / ``exp_avg_sq`` moments already living in ``optimizer.state`` (see [`_adamw_reconstruct_pre_step`]),
    then diff the **low byte** of the bf16 pattern (a flipped low byte ⊆ a changed bf16 value). Persistent extra
    storage is **zero**, the reconstruction is transient and thrown away.


    ``_validated_masks[name]`` is a boolean tensor, True for each element detected as changed in the last step;
    populated by [`compute_masks`].

    Args:
        model ([`~torch.nn.Module`]):
            Model whose parameters are tracked.
        optimizer ([`~torch.optim.Optimizer`]):
            Optimizer driving the step. Must be a [`torch.optim.AdamW`] (unwrap Accelerate first).
    """

    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer):
        if not isinstance(optimizer, torch.optim.AdamW):
            raise TypeError(
                f"Sparse weight sync reconstructs the pre-step weights from the resident AdamW moments, so it "
                f"requires a `torch.optim.AdamW` optimizer, but got `{type(optimizer).__name__}`. Set "
                f"`weight_sync_mode='full'` to broadcast the full policy over NCCL instead."
            )
        self.optimizer = optimizer
        self._validated_masks: dict[str, torch.Tensor] = {}

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
            "AdamWInversionChangeDetector: matched %d/%d optimizer params",
            len(self._param_id_to_name),
            sum(1 for _ in model.named_parameters()),
        )

    def compute_masks(self) -> dict[str, torch.Tensor]:
        """Reconstruct the pre-step weights from the current AdamW state and diff against the live weights.

        Returns ``_validated_masks`` (``{name: bool tensor}``, True where the bf16 weight changed in the last step).
        The reconstruction is rounded to bf16 before the comparison: an unchanged bf16 weight differs from its fp32
        reconstruction by a sub-ULP residual, so an fp32 comparison would flag ~every element — the bf16 round makes
        the mask track the actual bf16 changes (sparse). Params that have never stepped (no optimizer state) are
        omitted. Runs entirely on each param's device.
        """
        self._validated_masks.clear()
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                name = self._param_id_to_name.get(id(p))
                if name is None:
                    continue
                state = self.optimizer.state.get(p)
                if not state:  # never stepped (e.g. frozen / no grad) -> nothing changed
                    continue
                theta_old = _adamw_reconstruct_pre_step(p, state, group)
                self._validated_masks[name] = p != theta_old.to(torch.bfloat16)
        return self._validated_masks


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
