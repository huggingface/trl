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

import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import torch
from huggingface_hub import hf_hub_download
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


logger = logging.getLogger(__name__)


@dataclass
class DeltaWeightTransferInitInfo(WeightTransferInitInfo):
    pass


@dataclass
class DeltaWeightTransferUpdateInfo(WeightTransferUpdateInfo):
    """Metadata sent via ``/update_weights`` — just Hub coordinates."""

    repo_id: str = ""
    filename: str = ""
    revision: str = "main"


class DeltaWeightTransferEngine(WeightTransferEngine[DeltaWeightTransferInitInfo, DeltaWeightTransferUpdateInfo]):
    """Weight transfer engine that uses HF Hub (Xet) as the data plane.

    Worker side: downloads safetensors from Hub, feeds to ``load_weights``.
    Trainer side: uploads changed params as safetensors to Hub.
    """

    init_info_cls = DeltaWeightTransferInitInfo
    update_info_cls = DeltaWeightTransferUpdateInfo

    def __init__(self, config: WeightTransferConfig, parallel_config: ParallelConfig) -> None:
        super().__init__(config, parallel_config)
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
        local_path = hf_hub_download(
            repo_id=update_info.repo_id,
            filename=update_info.filename,
            revision=update_info.revision,
            force_download=True,
        )

        with safe_open(local_path, framework="pt", device="cpu") as f:
            is_sparse = f.metadata().get("sparse", "False") == "True"

            if not is_sparse:
                # Full tensors — feed to load_weights and build snapshot
                self._bf16_snapshot = {}
                for name in f.keys():
                    tensor = f.get_tensor(name)
                    self._bf16_snapshot[name] = tensor.to(torch.bfloat16).clone()
                    load_weights([(name, tensor)])
                logger.info("Applied full weights (%d params)", len(self._bf16_snapshot))
            else:
                # Sparse — apply indices/values to snapshot, feed full tensors
                changed = set()
                for key in f.keys():
                    if key.endswith(".indices"):
                        changed.add(key.removesuffix(".indices"))

                for name in changed:
                    indices = f.get_tensor(f"{name}.indices").long()
                    values = f.get_tensor(f"{name}.values")
                    snap = self._bf16_snapshot[name].flatten()
                    snap[indices] = values
                    self._bf16_snapshot[name] = snap.reshape(self._bf16_snapshot[name].shape)
                    load_weights([(name, self._bf16_snapshot[name])])

                logger.info("Applied sparse weights (%d params changed)", len(changed))

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
        repo_id: str,
        filename: str,
        hf_api: Any,
        revision: str = "main",
    ) -> int:
        """Encode params as safetensors and upload to HF Hub.

        Each item is ``(name, tensor, mask)``:

        - ``mask is None``: full tensor stored as ``name`` (anchor).
        - ``mask`` provided: sparse encoding — only changed elements stored
          as ``{name}.indices`` (int32) + ``{name}.values`` (bf16).

        Returns the number of params encoded.
        """
        tensors: dict[str, torch.Tensor] = {}
        num_params = 0
        sparse = False

        for name, tensor, mask in iterator:
            num_params += 1
            bf16 = tensor.to(torch.bfloat16).cpu()
            if mask is None:
                tensors[name] = bf16.clone()
            else:
                sparse = True
                indices = mask.flatten().nonzero(as_tuple=False).squeeze(1).to(torch.int32)
                values = bf16.flatten()[indices.long()]
                tensors[f"{name}.indices"] = indices
                tensors[f"{name}.values"] = values

        if not tensors:
            return 0

        metadata = {"num_params": str(num_params), "sparse": str(sparse)}
        buf = save(tensors, metadata=metadata)

        hf_api.upload_file(
            path_or_fileobj=buf,
            path_in_repo=filename,
            repo_id=repo_id,
            revision=revision,
            commit_message=f"weight update ({num_params} params, {len(buf) / 1e6:.1f} MB, sparse={sparse})",
        )

        logger.info(
            "[delta_engine] uploaded %s/%s (%.1f MB, %d params, sparse=%s)",
            repo_id,
            filename,
            len(buf) / 1e6,
            num_params,
            sparse,
        )
        return num_params


# ---------------------------------------------------------------------------
# Worker extension — its import triggers engine registration
# ---------------------------------------------------------------------------


class DeltaWorkerExtension:
    """vLLM worker extension for the delta weight transfer backend.

    This class is intentionally minimal. Its import (via ``--worker-extension-cls``)
    registers the engine and overrides the ``"nccl"`` factory entry.

    ``backend`` must be ``"nccl"`` in the CLI (pydantic ``Literal`` validation).
    This module overrides the ``"nccl"`` factory entry so that the actual engine
    created is ``DeltaWeightTransferEngine``.
    """

    pass


# ---------------------------------------------------------------------------
# Module-level registration
# ---------------------------------------------------------------------------

if "delta" not in WeightTransferEngineFactory._registry:
    WeightTransferEngineFactory.register_engine("delta", DeltaWeightTransferEngine)

# Override "nccl" so --weight-transfer-config '{"backend":"nccl"}' creates our engine.
WeightTransferEngineFactory._registry["nccl"] = lambda: DeltaWeightTransferEngine
