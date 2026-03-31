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

"""
Delta weight transfer engine for vLLM.

Uses HuggingFace Hub (Xet storage) as the data plane for sparse weight patches.
The trainer uploads patches to HF Hub, then sends a lightweight metadata signal
to vLLM via ``/update_weights``. The vLLM worker downloads and applies patches.

Registration happens at module import time so that vLLM's ``WeightTransferEngineFactory``
can find the ``"delta"`` backend. Use ``--worker-extension-cls`` to trigger the import::

    vllm serve model_name \\
        --worker-extension-cls trl.experimental.async_grpo.delta_engine.DeltaWorkerExtension \\
        --weight-transfer-backend delta
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

import requests
import torch
from huggingface_hub import HfApi, hf_hub_download
from safetensors import safe_open
from safetensors.torch import load_file, save
from vllm.config.parallel import ParallelConfig
from vllm.config.weight_transfer import WeightTransferConfig
from vllm.distributed.weight_transfer.base import (
    WeightTransferEngine,
    WeightTransferInitInfo,
    WeightTransferUpdateInfo,
)
from vllm.distributed.weight_transfer.factory import WeightTransferEngineFactory

from .weight_diff import compute_bf16_checksum, encode_sparse_patch


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class DeltaWeightTransferInitInfo(WeightTransferInitInfo):
    """No initialization needed for file-based Hub transport."""

    pass


@dataclass
class DeltaWeightTransferUpdateInfo(WeightTransferUpdateInfo):
    """Metadata sent via ``/update_weights`` — no weight data, just Hub coordinates."""

    repo_id: str = ""
    filename: str = ""
    revision: str = "main"
    patch_type: str = "anchor"  # "anchor" | "delta"
    expected_checksum: str = ""
    # is_checkpoint_format: True for anchor (layerwise reload), False for delta (param.copy_)


@dataclass
class DeltaTrainerSendWeightsArgs:
    """Trainer-side state passed to ``trainer_send_weights``.

    This is a mutable object — ``prev_bf16_snapshot`` and ``model_version``
    are updated after each call so the next call can compute a diff.
    """

    repo_id: str
    url: str  # vLLM server URL (for the /update_weights signal only)
    hf_api: HfApi = field(default_factory=HfApi)
    anchor_interval: int = 10
    verify_checksum: bool = True
    revision: str = "main"
    base_model_id: str = ""
    # Mutable state — updated after each call:
    prev_bf16_snapshot: dict[str, torch.Tensor] | None = None
    model_version: int = 0
    _last_anchor_step: int = 0


class DeltaWeightTransferEngine(WeightTransferEngine[DeltaWeightTransferInitInfo, DeltaWeightTransferUpdateInfo]):
    """Weight transfer engine that uses HF Hub (Xet) for sparse delta patches.

    Worker side: downloads patches from Hub and applies them via a CPU bf16 snapshot.
    Trainer side: encodes sparse patches and uploads them to Hub.
    """

    init_info_cls = DeltaWeightTransferInitInfo
    update_info_cls = DeltaWeightTransferUpdateInfo

    def __init__(self, config: WeightTransferConfig, parallel_config: ParallelConfig) -> None:
        super().__init__(config, parallel_config)
        self._bf16_snapshot: dict[str, torch.Tensor] | None = None

    def init_transfer_engine(self, init_info: DeltaWeightTransferInitInfo) -> None:
        pass  # No process group setup needed

    def receive_weights(
        self,
        update_info: DeltaWeightTransferUpdateInfo,
        load_weights: Callable[[list[tuple[str, torch.Tensor]]], None],
    ) -> None:
        local_path = hf_hub_download(
            repo_id=update_info.repo_id,
            filename=update_info.filename,
            revision=update_info.revision,
            force_download=True,
        )

        if update_info.patch_type == "anchor":
            self._receive_anchor(local_path, load_weights)
        else:
            self._receive_delta(local_path, update_info.expected_checksum, load_weights)

    def _receive_anchor(
        self,
        local_path: str,
        load_weights: Callable[[list[tuple[str, torch.Tensor]]], None],
    ) -> None:
        """Load a full anchor checkpoint and rebuild the snapshot."""
        state = load_file(local_path, device="cpu")
        self._bf16_snapshot = {}
        for name, tensor in state.items():
            self._bf16_snapshot[name] = tensor.to(torch.bfloat16).clone()
            load_weights([(name, tensor)])
        logger.info("Loaded anchor checkpoint with %d parameters", len(self._bf16_snapshot))

    def _receive_delta(
        self,
        local_path: str,
        expected_checksum: str,
        load_weights: Callable[[list[tuple[str, torch.Tensor]]], None],
    ) -> None:
        """Apply a sparse delta patch to the snapshot, then feed changed params to load_weights."""
        if self._bf16_snapshot is None:
            raise RuntimeError(
                "Cannot apply delta patch without a prior anchor. "
                "Ensure the first weight sync is an anchor (is_checkpoint_format=True)."
            )

        with safe_open(local_path, framework="pt", device="cpu") as f:
            meta = f.metadata()
            changed_names = json.loads(meta.get("changed_params", "[]"))

            for name in changed_names:
                indices = f.get_tensor(f"{name}.indices").long()
                values = f.get_tensor(f"{name}.values")
                # Apply to CPU snapshot (bit-exact, no FP arithmetic)
                snap_flat = self._bf16_snapshot[name].flatten()
                snap_flat[indices] = values
                self._bf16_snapshot[name] = snap_flat.reshape(self._bf16_snapshot[name].shape)
                # Pass reconstructed full tensor to load_weights
                load_weights([(name, self._bf16_snapshot[name].to("cuda"))])

        if expected_checksum:
            actual = compute_bf16_checksum(self._bf16_snapshot)
            if actual != expected_checksum:
                raise ValueError(f"Checksum mismatch: expected {expected_checksum[:16]}..., got {actual[:16]}...")

        logger.info(
            "Applied delta patch: %d params changed, sparsity=%s",
            len(changed_names),
            meta.get("sparsity", "?"),
        )

    def shutdown(self) -> None:
        self._bf16_snapshot = None

    @staticmethod
    def trainer_send_weights(
        iterator: Iterator[tuple[str, torch.Tensor]],
        trainer_args: dict[str, Any] | DeltaTrainerSendWeightsArgs,
    ) -> None:
        """Encode a sparse patch, upload to HF Hub, and signal vLLM.

        Args:
            iterator: (name, tensor) pairs from the model (e.g. FSDP streaming iterator).
            trainer_args: :class:`DeltaTrainerSendWeightsArgs` (mutable, updated in-place).
        """
        if isinstance(trainer_args, dict):
            args = DeltaTrainerSendWeightsArgs(**trainer_args)
        else:
            args = trainer_args

        # 1. Collect bf16 snapshot from the streaming iterator
        curr_bf16: dict[str, torch.Tensor] = {}
        for name, tensor in iterator:
            curr_bf16[name] = tensor.to(torch.bfloat16).cpu().clone()

        args.model_version += 1
        is_anchor = args.prev_bf16_snapshot is None or args.model_version % args.anchor_interval == 0

        # 2. Encode to safetensors bytes (no local disk write)
        if is_anchor:
            checksum = compute_bf16_checksum(curr_bf16)
            metadata = {
                "format": "anchor_checkpoint",
                "version": "1",
                "model_version": str(args.model_version),
                "base_model_id": args.base_model_id,
                "checksum_sha256": checksum,
            }
            buf = save(curr_bf16, metadata=metadata)
            filename = f"anchors/step_{args.model_version:06d}.safetensors"
            args._last_anchor_step = args.model_version
        else:
            tensors, meta_dict = encode_sparse_patch(
                prev_bf16=args.prev_bf16_snapshot,
                curr_bf16=curr_bf16,
                model_version=args.model_version,
                prev_model_version=args.model_version - 1,
                anchor_step=args._last_anchor_step,
                base_model_id=args.base_model_id,
            )
            checksum = meta_dict["checksum_sha256"]
            # safetensors requires at least one tensor
            if not tensors:
                tensors["__empty_delta__"] = torch.zeros(1, dtype=torch.int32)
            buf = save(tensors, metadata=meta_dict)
            filename = f"deltas/step_{args.model_version:06d}.safetensors"

        # 3. Upload to HF Hub (Xet handles chunking/dedup)
        args.hf_api.upload_file(
            path_or_fileobj=buf,
            path_in_repo=filename,
            repo_id=args.repo_id,
            revision=args.revision,
            commit_message=f"step {args.model_version} ({'anchor' if is_anchor else 'delta'})",
        )

        logger.info(
            "[delta_engine] uploaded %s to %s/%s (%.1f MB)",
            "anchor" if is_anchor else "delta",
            args.repo_id,
            filename,
            len(buf) / 1e6,
        )

        # 4. Signal vLLM (metadata only — no weight data)
        update_info = {
            "repo_id": args.repo_id,
            "filename": filename,
            "revision": args.revision,
            "patch_type": "anchor" if is_anchor else "delta",
            "expected_checksum": checksum if args.verify_checksum else "",
            "is_checkpoint_format": True,  # Always True: vLLM fuses params (e.g. gate_up_proj), needs model.load_weights for name mapping
        }
        resp = requests.post(
            f"{args.url}/update_weights",
            json={"update_info": update_info},
            timeout=300,
        )
        resp.raise_for_status()

        # 5. Update mutable state for next call
        args.prev_bf16_snapshot = curr_bf16

    @staticmethod
    def trainer_init(
        repo_id: str,
        url: str,
        anchor_interval: int = 10,
        verify_checksum: bool = True,
        revision: str = "main",
        base_model_id: str = "",
        token: str | None = None,
    ) -> DeltaTrainerSendWeightsArgs:
        """Initialize trainer-side state: create/ensure HF repo and return args.

        Call once at startup. Pass the returned object to ``trainer_send_weights``
        on every weight sync.
        """
        api = HfApi(token=token)
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        logger.info("[delta_engine] trainer_init: repo=%s, anchor_interval=%d", repo_id, anchor_interval)
        return DeltaTrainerSendWeightsArgs(
            repo_id=repo_id,
            url=url,
            hf_api=api,
            anchor_interval=anchor_interval,
            verify_checksum=verify_checksum,
            revision=revision,
            base_model_id=base_model_id,
        )


class DeltaWorkerExtension:
    """vLLM worker extension for the delta weight transfer backend.

    This class is intentionally minimal. Its primary role is to trigger the
    import of this module (via ``--worker-extension-cls``) which registers the
    engine with ``WeightTransferEngineFactory`` at module level below.

    Usage with standard ``vllm serve``::

        VLLM_SERVER_DEV_MODE=1 vllm serve Qwen/Qwen3-0.6B \\
            --worker-extension-cls trl.experimental.async_grpo.delta_engine.DeltaWorkerExtension \\
            --weight-transfer-config '{"backend":"nccl"}' \\
            --max-model-len 8192 --enforce-eager --logprobs-mode processed_logprobs

    Note: ``backend`` must be ``"nccl"`` in the CLI (pydantic ``Literal`` validation).
    This module overrides the ``"nccl"`` factory entry so that the actual engine
    created is ``DeltaWeightTransferEngine``.
    """

    pass


# ---------------------------------------------------------------------------
# Module-level registration — runs when this module is first imported
# ---------------------------------------------------------------------------

if "delta" not in WeightTransferEngineFactory._registry:
    WeightTransferEngineFactory.register_engine("delta", DeltaWeightTransferEngine)

# Override the "nccl" factory entry so that --weight-transfer-config '{"backend":"nccl"}'
# (which passes pydantic Literal["nccl","ipc"] validation) actually creates a
# DeltaWeightTransferEngine.  This is safe: the trainer side never reads the factory,
# and the worker side is explicitly opting in via --worker-extension-cls.
WeightTransferEngineFactory._registry["nccl"] = lambda: DeltaWeightTransferEngine
