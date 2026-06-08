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

import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from typing import TypedDict

import requests
import torch
from accelerate.logging import get_logger
from huggingface_hub import create_bucket

from trl.import_utils import is_vllm_available

from .delta_codec import UpdateKind, extract_sparse_batched
from .delta_engine import HFBucketWeightTransferEngine


if is_vllm_available(min_version="0.17.1"):
    from vllm.distributed.weight_transfer.base import SparseWeightPatch
    from vllm.distributed.weight_transfer.nccl_engine import NCCLTrainerSendWeightsArgs, NCCLWeightTransferEngine
    from vllm.utils.network_utils import get_ip, get_open_port


logger = get_logger(__name__)

# A weight iterator yields ``(name, full_tensor, mask)`` per parameter; ``mask`` is None for a full transfer.
WeightIterFn = Callable[[bool], Iterator[tuple[str, torch.Tensor, torch.Tensor | None]]]


class PendingPatch(TypedDict):
    """A bucket patch uploaded in [`BucketWeightTransfer`] phase 1, awaiting the phase-2 apply."""

    repo_id: str  # bucket id
    filename: str  # path within the bucket
    update_kind: UpdateKind  # DENSE (anchor) or SPARSE_FLAT (delta)


class WeightTransfer(ABC):
    """Base for the trainer-side weight-sync transports. Holds the vLLM HTTP plumbing shared by every transport.

    Args:
        vllm_server_url (`str`):
            Base URL of the vLLM server.
        weight_update_info (`dict`):
            Full-model metadata with keys ``names`` / ``dtype_names`` / ``shapes`` (one entry per parameter, in
            ``model.named_parameters()`` order). Used to build the update_info sent to vLLM.
        server_timeout (`float`, *optional*, defaults to `240.0`):
            Seconds to wait for the vLLM server to become ready.
        init_weight_transfer_timeout (`int`, *optional*, defaults to `1800`):
            Timeout for the one-off ``/init_weight_transfer_engine`` call.
    """

    def __init__(
        self,
        vllm_server_url: str,
        weight_update_info: dict,
        server_timeout: float = 240.0,
        init_weight_transfer_timeout: int = 1800,
    ):
        if not is_vllm_available(min_version="0.17.1"):
            raise ImportError(
                "vLLM >= 0.17.1 is required to use the weight-sync transports. Install it with: "
                "pip install 'vllm>=0.17.1'"
            )
        self.vllm_server_url = vllm_server_url.rstrip("/")
        self.server_timeout = server_timeout
        self.init_weight_transfer_timeout = init_weight_transfer_timeout
        self._names = weight_update_info["names"]
        self._dtype_names = weight_update_info["dtype_names"]
        self._shapes = weight_update_info["shapes"]

    @abstractmethod
    def init(self, accelerator) -> None:
        """Set up the transport (rank 0 only). Called once before the first sync."""

    @abstractmethod
    def sync(self, *, iter_fn: WeightIterFn, sparse: bool, is_anchor: bool, version: int, accelerator) -> None:
        """Push the current policy to vLLM. Runs on every rank; rank 0 drives the transport, the others only walk
        ``iter_fn`` so the FSDP2 collectives line up."""

    def destroy(self) -> None:  # noqa: B027 - intentional no-op default; NCCL overrides, bucket needs none
        """Tear down the transport (rank 0 only). Default: nothing to do."""

    def _wait_for_server_ready_sync(self, timeout_s: float | None = None, poll_interval_s: float = 2.0) -> None:
        timeout_s = timeout_s if timeout_s is not None else self.server_timeout
        logger.info(f"Waiting for vLLM server at {self.vllm_server_url} ...")
        start = time.time()
        while True:
            elapsed = time.time() - start
            try:
                response = requests.get(f"{self.vllm_server_url}/health", timeout=5)
                if response.status_code == 200:
                    logger.info(f"vLLM server ready after {elapsed:.1f}s")
                    return
            except (requests.ConnectionError, requests.Timeout, OSError):
                pass
            if elapsed >= timeout_s:
                raise TimeoutError(
                    f"Timed out after {timeout_s:.0f}s waiting for vLLM server at {self.vllm_server_url}. "
                    "Make sure the vLLM server is running and reachable. If the server needs more time to load "
                    "the model, increase `vllm_server_timeout` in your AsyncGRPOConfig."
                )
            if int(elapsed) % 10 < poll_interval_s:
                logger.info(f"Still waiting for vLLM server... ({elapsed:.0f}s)")
            time.sleep(poll_interval_s)

    def _post_vllm(self, path: str, json_body: dict, retries: int = 1, timeout: int = 300) -> None:
        """POST to a vLLM server endpoint with bounded retry on 429 / connection errors."""
        url = f"{self.vllm_server_url}{path}"
        for attempt in range(retries):
            try:
                resp = requests.post(url, json=json_body, timeout=timeout)
                if resp.status_code < 429:
                    resp.raise_for_status()
                    return
                status = resp.status_code
            except requests.RequestException as e:
                logger.warning(f"[weight_sync] POST {path} failed: {e}")
                status = "connection error"
            if attempt < retries - 1:
                wait = min(2**attempt, 30)
                logger.warning(f"[weight_sync] POST {path} -> {status}, retry in {wait}s ({attempt + 1}/{retries})")
                time.sleep(wait)
        raise RuntimeError(f"[weight_sync] POST {path} failed after {retries} attempt(s)")

    def pause(self) -> None:
        requests.post(f"{self.vllm_server_url}/pause", params={"mode": "keep"})

    def resume(self) -> None:
        requests.post(f"{self.vllm_server_url}/resume")


class NCCLWeightTransfer(WeightTransfer):
    """Broadcast weights to vLLM over a shared NCCL group. Single-phase: pause, broadcast, resume."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_update_group = None

    def init(self, accelerator) -> None:
        if not accelerator.is_main_process:
            return
        self._wait_for_server_ready_sync()
        inference_world_size = self._get_world_size()
        world_size = inference_world_size + 1
        master_address = get_ip()
        master_port = get_open_port()
        init_info = {
            "master_address": master_address,
            "master_port": master_port,
            "rank_offset": 1,
            "world_size": world_size,
        }
        # vLLM's /init joins the group on the worker side; the trainer joins concurrently from this rank.
        t_init = threading.Thread(
            target=requests.post,
            args=(f"{self.vllm_server_url}/init_weight_transfer_engine",),
            kwargs={"json": {"init_info": init_info}, "timeout": self.init_weight_transfer_timeout},
        )
        t_init.start()
        self.model_update_group = NCCLWeightTransferEngine.trainer_init(
            {"master_address": master_address, "master_port": master_port, "world_size": world_size}
        )
        t_init.join()
        logger.info("Initialised weight-transfer NCCL group with vLLM")

    def _get_world_size(self, attempts: int = 30, poll_interval_s: float = 2.0) -> int:
        """Read vLLM's world size, retrying until the engine RPC is ready. ``/health`` can go green before
        ``/get_world_size`` answers, so poll instead of indexing the first response blindly."""
        for _ in range(attempts):
            try:
                world_size = requests.get(f"{self.vllm_server_url}/get_world_size", timeout=5).json().get("world_size")
                if world_size is not None:
                    return int(world_size)
            except (requests.RequestException, ValueError):
                pass
            time.sleep(poll_interval_s)
        raise RuntimeError(f"vLLM /get_world_size did not return a world_size after {attempts} attempts")

    def sync(self, *, iter_fn: WeightIterFn, sparse: bool, is_anchor: bool, version: int, accelerator) -> None:
        is_main = accelerator.is_main_process
        t0 = time.time()
        if is_main:
            self.pause()
        accelerator.wait_for_everyone()  # broadcast must start in lockstep across FSDP ranks
        if is_main:
            kind = "sparse" if sparse else ("anchor" if is_anchor else "full")
            logger.info("Weight sync: NCCL %s broadcast...", kind)
            if sparse:
                self._send_sparse(iter_fn(True))
            else:
                self._send_full(iter_fn(False))
        else:
            for _ in iter_fn(sparse):  # participate in the full_tensor() collectives
                pass
        accelerator.wait_for_everyone()
        if is_main:
            self.resume()
            logger.info("Weight sync: done (NCCL, %.1fs)", time.time() - t0)

    def _send_full(self, iterator) -> None:
        """Dense full-policy broadcast (checkpoint format). The worker enters ``receive_weights`` inside the
        ``/update_weights`` call, so the NCCL send must run concurrently with that POST."""
        update_info = {
            "update_kind": UpdateKind.DENSE,
            "names": self._names,
            "dtype_names": self._dtype_names,
            "shapes": self._shapes,
            "packed": True,
        }
        self._post_vllm("/start_weight_update", {"is_checkpoint_format": True})
        t_update = threading.Thread(
            target=requests.post,
            args=(f"{self.vllm_server_url}/update_weights",),
            kwargs={"json": {"update_info": update_info}, "timeout": 1800},
        )
        t_update.start()
        NCCLWeightTransferEngine.trainer_send_weights(
            iterator=((name, tensor) for name, tensor, _mask in iterator),
            trainer_args=NCCLTrainerSendWeightsArgs(group=self.model_update_group, packed=True),
        )
        t_update.join()
        self._post_vllm("/finish_weight_update", {})

    def _send_sparse(self, iterator) -> None:
        """Sparse delta broadcast (kernel format, applied in place via ``index_copy_`` on vLLM). Changed
        ``(int32 flat-index, bf16 value)`` pairs are extracted on the GPU; the per-param counts go to vLLM as
        ``num_updates_list`` so it can pre-allocate the receive buffers, then each patch is broadcast in that order.

        Extraction runs in **bounded chunks**: the iterator yields one ``full_tensor()``-gathered param at a time, and
        we keep only its sparse payload (~1% of the param) before dropping the dense tensor. So rank 0 never holds the
        whole gathered model at once — essential under FSDP2 at large scale, where the full model is tens of GB.
        """
        # patches: (name, int32 indices, bf16 values, shape) — only the sparse payload is retained across chunks.
        patches: list[tuple] = []
        chunk: list[tuple] = []
        chunk_numel = 0
        for name, tensor, mask in iterator:
            chunk.append((name, tensor, mask))
            chunk_numel += tensor.numel()
            if chunk_numel >= 256_000_000:
                patches.extend(self._extract_chunk(chunk))
                chunk, chunk_numel = [], 0  # drop the chunk's dense gathered tensors
        if chunk:
            patches.extend(self._extract_chunk(chunk))

        names = [name for name, _, _, _ in patches]
        if not names:  # nothing changed this step -> vLLM rejects an empty sparse update; skip
            logger.debug("[weight_sync] sparse NCCL: no changed params, skipping transfer")
            return
        update_info = {
            "update_kind": UpdateKind.SPARSE_FLAT,
            "names": names,
            "dtype_names": ["bfloat16"] * len(names),
            "shapes": [shape for _, _, _, shape in patches],
            "num_updates_list": [int(idx.numel()) for _, idx, _, _ in patches],
            "packed": False,
        }
        self._post_vllm("/start_weight_update", {"is_checkpoint_format": False})
        t_update = threading.Thread(
            target=requests.post,
            args=(f"{self.vllm_server_url}/update_weights",),
            kwargs={"json": {"update_info": update_info}, "timeout": 1800},
        )
        t_update.start()
        NCCLWeightTransferEngine.trainer_send_sparse_weights(
            iterator=(SparseWeightPatch(name=name, indices=idx, values=vals) for name, idx, vals, _ in patches),
            trainer_args=NCCLTrainerSendWeightsArgs(group=self.model_update_group, src=0, packed=False),
        )
        t_update.join()
        self._post_vllm("/finish_weight_update", {})

    @staticmethod
    def _extract_chunk(chunk) -> list[tuple]:
        """Extract ``(name, int32 indices, bf16 values, shape)`` for a chunk of ``(name, full, mask)`` triples.
        Values are cast to bf16 to match the dtype vLLM allocates for the receive buffer (the served policy is bf16)."""
        shape_by_name = {name: list(tensor.shape) for name, tensor, _ in chunk}
        return [
            (name, idx.contiguous(), vals.to(torch.bfloat16).contiguous(), shape_by_name[name])
            for name, idx, vals in extract_sparse_batched(chunk)
        ]

    def destroy(self) -> None:
        if self.model_update_group is None:
            return
        self.model_update_group.group.store = None
        self.model_update_group.group.socket = None
        self.model_update_group = None


class BucketWeightTransfer(WeightTransfer):
    """Route the patch through an HF Storage Bucket. Two-phase: upload while inference runs, then pause → apply →
    resume (vLLM fetches the patch from the bucket inside the apply).

    Args:
        bucket_id (`str`):
            HF Storage Bucket for the patches/anchors (created if missing).
        encoding (`str`, *optional*, defaults to `"gap_delta"`):
            Index encoding for the patches: ``"raw"``, ``"gap_delta"``, or ``"nvcomp_cascaded"``.
    """

    def __init__(self, *args, bucket_id: str, encoding: str = "gap_delta", **kwargs):
        super().__init__(*args, **kwargs)
        self._bucket_id = bucket_id
        self._encoding = encoding
        self._pending: PendingPatch | None = None

    def init(self, accelerator) -> None:
        if not accelerator.is_main_process:
            return
        self._wait_for_server_ready_sync()
        create_bucket(self._bucket_id, exist_ok=True)
        requests.post(
            f"{self.vllm_server_url}/init_weight_transfer_engine",
            json={"init_info": {}},
            timeout=self.init_weight_transfer_timeout,
        )
        logger.info("Initialised bucket weight transfer (bucket %s)", self._bucket_id)

    def sync(self, *, iter_fn: WeightIterFn, sparse: bool, is_anchor: bool, version: int, accelerator) -> None:
        is_main = accelerator.is_main_process
        t0 = time.time()
        # Phase 1: encode + upload the patch while inference keeps running (weights materialized here).
        if is_main:
            logger.info("Weight sync: uploading %s patch to bucket...", "anchor" if is_anchor else "delta")
            self._upload(iter_fn(sparse), is_anchor=is_anchor, version=version)
        else:
            for _ in iter_fn(sparse):  # participate in the full_tensor() collectives
                pass
        accelerator.wait_for_everyone()
        # Phase 2: pause, then signal vLLM to fetch + apply the uploaded patch.
        if is_main:
            self.pause()
            try:
                self._apply()
            except Exception as e:
                logger.warning(f"Weight sync: bucket apply failed ({e}), skipping; vLLM keeps stale weights")
            self.resume()
            logger.info("Weight sync: done (bucket, %.1fs)", time.time() - t0)
        accelerator.wait_for_everyone()

    def _upload(self, iterator, is_anchor: bool, version: int) -> None:
        if is_anchor:
            iterator = ((name, tensor, None) for name, tensor, _mask in iterator)  # strip masks -> full tensors
        subdir = "anchors" if is_anchor else "deltas"
        filename = f"{subdir}/step_{version:06d}.safetensors"
        meta = HFBucketWeightTransferEngine.upload(
            iterator=iterator,
            bucket_id=self._bucket_id,
            filename=filename,
            model_version=version,
            encoding=self._encoding,
        )
        self._pending = (
            None
            if meta is None
            else PendingPatch(
                repo_id=self._bucket_id,
                filename=filename,
                update_kind=UpdateKind.DENSE if is_anchor else UpdateKind.SPARSE_FLAT,
            )
        )

    def _apply(self) -> None:
        """No-op when nothing was uploaded this step; ``_pending`` is cleared up front so a failed apply leaves no
        stale state."""
        pending, self._pending = self._pending, None
        if pending is None:
            return
        # Anchors are HF-checkpoint-format full tensors; deltas are sparse kernel-format.
        self._post_vllm("/start_weight_update", {"is_checkpoint_format": pending["update_kind"] is UpdateKind.DENSE})
        # vLLM fetches the patch inside this call; a full anchor can take minutes, so the timeout must cover the
        # download, otherwise a read-timeout would retry into a re-download.
        self._post_vllm("/update_weights", {"update_info": pending}, retries=5, timeout=1800)
        self._post_vllm("/finish_weight_update", {})


def make_weight_transfer(
    backend: str,
    *,
    vllm_server_url: str,
    weight_update_info: dict,
    server_timeout: float = 240.0,
    bucket_id: str | None = None,
    encoding: str = "gap_delta",
) -> WeightTransfer:
    """Build the [`WeightTransfer`] for ``backend`` (``"nccl"`` or ``"bucket"``)."""
    if backend == "nccl":
        return NCCLWeightTransfer(vllm_server_url, weight_update_info, server_timeout=server_timeout)
    if backend == "bucket":
        return BucketWeightTransfer(
            vllm_server_url, weight_update_info, server_timeout=server_timeout, bucket_id=bucket_id, encoding=encoding
        )
    raise ValueError(f"Unknown weight_sync_backend: {backend!r}")
