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

from __future__ import annotations

import json
import logging
import tempfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import torch
from huggingface_hub import batch_bucket_files, download_bucket_files
from safetensors import safe_open
from safetensors.torch import save
from vllm.config.parallel import ParallelConfig
from vllm.config.weight_transfer import WeightTransferConfig
from vllm.distributed.weight_transfer.base import (
    WeightTransferEngine,
    WeightTransferInitInfo,
    WeightTransferUpdateInfo,
)
from vllm.distributed.weight_transfer.factory import WeightTransferEngineFactory

from .weight_diff import PatchMetadata


logger = logging.getLogger(__name__)


@dataclass
class DeltaWeightTransferInitInfo(WeightTransferInitInfo):
    pass


@dataclass
class DeltaWeightTransferUpdateInfo(WeightTransferUpdateInfo):
    """Metadata sent via ``/update_weights`` — just bucket coordinates."""

    repo_id: str = ""  # bucket_id
    filename: str = ""


class DeltaWeightTransferEngine(WeightTransferEngine[DeltaWeightTransferInitInfo, DeltaWeightTransferUpdateInfo]):
    """Weight transfer engine that uses HF Hub (Xet) as the data plane.

    Worker side: downloads safetensors from Hub, feeds to ``load_weights``.
    Trainer side: uploads changed params as safetensors to Hub.
    """

    init_info_cls = DeltaWeightTransferInitInfo
    update_info_cls = DeltaWeightTransferUpdateInfo

    def __init__(self, config: WeightTransferConfig, parallel_config: ParallelConfig) -> None:
        super().__init__(config, parallel_config)
        # TODO: might be able to eliminate completely
        # CPU-side bf16 snapshot — needed because vLLM's load_weights expects full
        # tensors, so we must reconstruct them from sparse (indices, values) patches.
        # Kept on CPU to avoid GPU memory overhead (~2 bytes/param, e.g. ~1.2 GB for 0.6B model).
        self._bf16_snapshot: dict[str, torch.Tensor] | None = None

    def init_transfer_engine(self, init_info: DeltaWeightTransferInitInfo) -> None:
        pass

    def receive_weights(
        self,
        update_info: DeltaWeightTransferUpdateInfo,
        load_weights: Callable[[list[tuple[str, torch.Tensor]]], None],
    ) -> None:
        """Download safetensors from Hub and feed to load_weights.

        Handles two formats based on the ``sparse`` metadata flag:

        - **Full** (first sync): keys are param names → feed directly to load_weights,
          build snapshot for future sparse applies.
        - **Sparse** (subsequent): keys are ``{name}.indices`` + ``{name}.values`` →
          apply to snapshot, feed reconstructed full tensors to load_weights.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = f"{tmpdir}/weights.safetensors"
            download_bucket_files(
                update_info.repo_id,
                files=[(update_info.filename, local_path)],
            )

            with safe_open(local_path, framework="pt", device="cpu") as f:
                meta = PatchMetadata.from_metadata_dict(f.metadata())

                if not meta.sparse:
                    self._bf16_snapshot = {}
                    for name in f.keys():
                        tensor = f.get_tensor(name)
                        self._bf16_snapshot[name] = tensor.to(torch.bfloat16).clone()
                        load_weights([(name, tensor)])
                    logger.info("Applied anchor (step %d, %d params)", meta.model_version, meta.num_changed_params)
                else:
                    changed_names = json.loads(meta.changed_params)
                    for name in changed_names:
                        indices = f.get_tensor(f"{name}.indices").long()
                        values = f.get_tensor(f"{name}.values")
                        snap = self._bf16_snapshot[name].flatten()
                        snap[indices] = values
                        self._bf16_snapshot[name] = snap.reshape(self._bf16_snapshot[name].shape)
                        load_weights([(name, self._bf16_snapshot[name])])
                    logger.info(
                        "Applied delta (step %d, %d params, sparsity=%.4f)",
                        meta.model_version,
                        meta.num_changed_params,
                        meta.sparsity,
                    )

    def shutdown(self) -> None:
        self._bf16_snapshot = None

    @staticmethod
    def trainer_send_weights(
        iterator: Iterator[tuple[str, torch.Tensor]],
        trainer_args: dict[str, Any] | Any,
    ) -> None:
        """Not used directly — the rollout worker manages upload + signaling."""
        raise NotImplementedError("Use AsyncRolloutWorker._send_weights_delta instead")

    @staticmethod
    def upload(
        iterator: Iterator[tuple[str, torch.Tensor, torch.Tensor | None]],
        bucket_id: str,
        filename: str,
        model_version: int = 0,
    ) -> PatchMetadata | None:
        """Encode params as safetensors and upload to HF Hub.

        Each item is ``(name, tensor, mask)``:

        - ``mask is None``: full tensor stored as ``name`` (anchor).
        - ``mask`` provided: sparse encoding — only changed elements stored
          as ``{name}.indices`` (int32) + ``{name}.values`` (bf16).

        Returns :class:`PatchMetadata` or ``None`` if the iterator was empty.
        """
        tensors: dict[str, torch.Tensor] = {}
        changed_names: list[str] = []
        total_changed = 0
        total_elements = 0
        sparse = False

        for name, tensor, mask in iterator:
            bf16 = tensor.to(torch.bfloat16).cpu()
            total_elements += bf16.numel()
            if mask is None:
                tensors[name] = bf16.clone()
                changed_names.append(name)
                total_changed += bf16.numel()
            else:
                sparse = True
                indices = mask.flatten().nonzero(as_tuple=False).squeeze(1).to(torch.int32)
                values = bf16.flatten()[indices.long()]
                tensors[f"{name}.indices"] = indices
                tensors[f"{name}.values"] = values
                changed_names.append(name)
                total_changed += len(indices)

        if not tensors:
            return None

        meta = PatchMetadata(
            sparse=sparse,
            model_version=model_version,
            num_changed_params=len(changed_names),
            total_changed_elements=total_changed,
            total_elements=total_elements,
            sparsity=1.0 - total_changed / max(total_elements, 1),
            changed_params=json.dumps(changed_names),
        )
        buf = save(tensors, metadata=meta.to_metadata_dict())

        batch_bucket_files(bucket_id, add=[(buf, filename)])

        logger.info(
            "[delta_engine] uploaded %s/%s (%.1f MB, %d params, sparse=%s, sparsity=%.4f)",
            bucket_id,
            filename,
            len(buf) / 1e6,
            len(changed_names),
            sparse,
            meta.sparsity,
        )
        return meta


class DeltaWorkerExtension:
    """vLLM worker extension for the delta weight transfer backend.

    This class is intentionally minimal. Its import (via ``--worker-extension-cls``)
    registers the engine and overrides the ``"nccl"`` factory entry.

    ``backend`` must be ``"nccl"`` in the CLI (pydantic ``Literal`` validation).
    This module overrides the ``"nccl"`` factory entry so that the actual engine
    created is ``DeltaWeightTransferEngine``.
    """

    pass


if "delta" not in WeightTransferEngineFactory._registry:
    WeightTransferEngineFactory.register_engine("delta", DeltaWeightTransferEngine)

# Override "nccl" so --weight-transfer-config '{"backend":"nccl"}' creates our engine.
WeightTransferEngineFactory._registry["nccl"] = lambda: DeltaWeightTransferEngine
