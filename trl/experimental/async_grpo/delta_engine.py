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
import os
import tempfile
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from huggingface_hub import batch_bucket_files, download_bucket_files
from safetensors import safe_open
from safetensors.torch import save_file
from vllm.distributed.weight_transfer.base import (
    SparseWeightPatch,
    WeightTransferEngine,
    WeightTransferInitInfo,
    WeightTransferUpdateInfo,
)
from vllm.distributed.weight_transfer.factory import WeightTransferEngineFactory

from .delta_codec import (
    Encoding,
    extract_sparse_batched,
    gap_delta_decode,
    gap_delta_encode,
    nvcomp_decode,
    nvcomp_encode,
)
from .weight_diff import PatchMetadata


try:
    from vllm.logger import init_logger

    logger = init_logger(f"vllm.{__name__}")
except Exception:  # pragma: no cover - vLLM always present where this module is used
    logger = logging.getLogger(__name__)


@contextmanager
def _fetch(update_info):
    """Download a patch from the bucket to a temp file and yield an open safetensors handle."""
    # TODO(@aminediro): writes only 1 safetensors, for very large model, this needs to be multiple files
    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/weights.safetensors"
        download_bucket_files(update_info.repo_id, files=[(update_info.filename, path)])
        with safe_open(path, framework="pt", device="cpu") as f:
            yield f


def _encode_idx(idx: torch.Tensor, encoding: Encoding) -> torch.Tensor:
    """Encode absolute int32 indices for storage. Returns a CPU tensor (I32, U16/U32, or U8 bytes)."""
    encoding = Encoding(encoding)
    if encoding is Encoding.RAW:
        return idx.to(torch.int32).cpu().contiguous()
    if encoding is Encoding.GAP_DELTA:
        return gap_delta_encode(idx).cpu().contiguous()  # native uint16/uint32 (dtype = width)
    return nvcomp_encode(idx)  # Encoding.NVCOMP_CASCADED -> uint8 CPU bytes


def _decode_idx(raw: torch.Tensor, encoding: Encoding) -> torch.Tensor:
    """Inverse of [`_encode_idx`] → 1D int32 absolute indices.

    Self-describing: ``raw.dtype`` carries the gap-delta width, so no element count is needed.
    """
    encoding = Encoding(encoding)
    if encoding is Encoding.RAW:
        return raw.to(torch.int32)
    if encoding is Encoding.GAP_DELTA:
        return gap_delta_decode(raw)
    return nvcomp_decode(raw)


@dataclass
class DeltaWeightTransferInitInfo(WeightTransferInitInfo):
    pass


@dataclass
class DeltaWeightTransferUpdateInfo(WeightTransferUpdateInfo):
    """Per-sync info sent via ``/update_weights`` — just bucket coordinates + kind.

    Names and per-param nnz are read from the downloaded file, so ``num_updates_list`` is not required here (we
    override the base validation that would otherwise demand it for sparse).
    """

    repo_id: str = ""  # bucket_id
    filename: str = ""

    def __post_init__(self) -> None:
        if self.update_kind not in ("dense", "sparse_flat"):
            raise ValueError(f"Unsupported update_kind: {self.update_kind}")


class DeltaWeightTransferEngine(WeightTransferEngine[DeltaWeightTransferInitInfo, DeltaWeightTransferUpdateInfo]):
    """Weight transfer engine using an HF Storage Bucket as the data plane."""

    init_info_cls = DeltaWeightTransferInitInfo
    update_info_cls = DeltaWeightTransferUpdateInfo

    def init_transfer_engine(self, init_info: DeltaWeightTransferInitInfo) -> None:
        pass

    def receive_weights(
        self,
        update_info: DeltaWeightTransferUpdateInfo,
        load_weights: Callable[[list[tuple[str, torch.Tensor]]], None],
    ) -> None:
        """Anchor path: download full safetensors from the bucket and load them directly."""
        t0 = time.time()
        with _fetch(update_info) as f:
            t_dl = time.time()
            for name in f.keys():
                load_weights([(name, f.get_tensor(name))])
            meta = PatchMetadata.from_metadata_dict(f.metadata())
            t_apply = time.time()
        logger.info(
            "Applied anchor (step %d, %d params) | download %.2fs load %.2fs",
            meta.model_version,
            meta.num_changed_params,
            t_dl - t0,
            t_apply - t_dl,
        )

    def receive_sparse_weights(
        self,
        update_info: DeltaWeightTransferUpdateInfo,
        apply_patches: Callable[[list[SparseWeightPatch]], None],
    ) -> None:
        t0 = time.time()
        patches = []
        with _fetch(update_info) as f:
            t_dl = time.time()
            names, idxs, vals = [], [], []
            meta = PatchMetadata.from_metadata_dict(f.metadata())
            for name, idx, values in iter_sparse_patches(f):
                names.append(name)
                idxs.append(idx)
                vals.append(values)
            if names:
                device = torch.accelerator.current_device_index()
                sizes = [i.numel() for i in idxs]
                all_idx = torch.cat(idxs).to(device)
                all_val = torch.cat(vals).to(device)
                off = 0
                for name, n in zip(names, sizes, strict=False):
                    patches.append(
                        SparseWeightPatch(name=name, indices=all_idx[off : off + n], values=all_val[off : off + n])
                    )
                    off += n
        if patches:
            apply_patches(patches)

        t_apply = time.time()
        logger.info(
            "Applied delta (step %d, %d params, sparsity=%.4f) | download %.2fs decode+apply %.2fs",
            meta.model_version,
            meta.num_changed_params,
            meta.sparsity,
            t_dl - t0,
            t_apply - t_dl,
        )

    def shutdown(self) -> None:
        pass

    @staticmethod
    def trainer_send_weights(iterator, trainer_args) -> None:
        raise NotImplementedError("Use AsyncRolloutWorker.upload_weights / apply_weights instead")

    @staticmethod
    def upload(
        iterator: Iterator[tuple[str, torch.Tensor, torch.Tensor | None]],
        bucket_id: str,
        filename: str,
        model_version: int = 0,
        encoding: Encoding = Encoding.GAP_DELTA,
    ) -> PatchMetadata | None:
        """Encode params as a safetensors patch and push to the bucket.

        Returns the :class:`PatchMetadata` (also written to the safetensors header), or ``None`` if the iterator was
        empty.
        """
        tensors, meta = encode_patch(iterator, model_version=model_version, encoding=encoding)
        if tensors is None:
            return None
        # Write to a temp file and upload the path: hf-xet's in-memory `upload_bytes` panics on
        # multi-GB buffers (e.g. a full-model anchor); the file path uses the large-file code path.
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = f"{tmpdir}/patch.safetensors"
            save_file(tensors, local_path, metadata=meta.to_metadata_dict())
            size_mb = os.path.getsize(local_path) / 1e6
            batch_bucket_files(bucket_id, add=[(local_path, filename)])
        logger.info(
            "[delta_engine] uploaded %s/%s (%.1f MB, %d params, sparse=%s, enc=%s, sparsity=%.4f)",
            bucket_id,
            filename,
            size_mb,
            meta.num_changed_params,
            meta.sparse,
            meta.encoding.value,
            meta.sparsity,
        )
        return meta


def encode_patch(
    iterator: Iterator[tuple[str, torch.Tensor, torch.Tensor | None]],
    model_version: int = 0,
    encoding: Encoding = Encoding.GAP_DELTA,
) -> tuple[dict[str, torch.Tensor] | None, PatchMetadata | None]:
    """Build the safetensors tensor dict + metadata for a patch (no I/O).

    Each item is ``(name, tensor, mask)``:

    - ``mask is None``: full tensor stored as ``name`` (anchor).
    - ``mask`` provided: GPU sparse-extract; store encoded indices as ``{name}.idx`` and values as ``{name}.val``.
      ``encoding`` is ``"raw"`` (int32) or ``"gap_delta"`` (uint16 gap bytes, uint32 fallback per param).
    """
    encoding = Encoding(encoding)
    tensors: dict[str, torch.Tensor] = {}
    delta_items: list[tuple[str, torch.Tensor, torch.Tensor]] = []
    n_params = 0
    total_changed = 0
    total_elements = 0

    for name, tensor, mask in iterator:
        n_params += 1
        total_elements += tensor.numel()
        if mask is None:  # anchor: store the full tensor
            tensors[name] = tensor.detach().to(torch.bfloat16).cpu().contiguous().clone()
            total_changed += tensor.numel()
        else:
            delta_items.append((name, tensor, mask))

    for name, idx, vals in extract_sparse_batched(delta_items):
        total_changed += idx.numel()
        tensors[f"{name}.val"] = vals.to(torch.bfloat16).cpu().contiguous()
        tensors[f"{name}.idx"] = _encode_idx(idx, encoding)

    sparse = bool(delta_items)
    if not tensors:
        return None, None

    meta = PatchMetadata(
        sparse=sparse,
        model_version=model_version,
        num_changed_params=n_params,
        total_changed_elements=total_changed,
        total_elements=total_elements,
        sparsity=1.0 - total_changed / max(total_elements, 1),
        encoding=encoding,
    )
    return tensors, meta


def iter_sparse_patches(f) -> Iterator[tuple[str, torch.Tensor, torch.Tensor]]:
    """Yield ``(name, int32 indices, values)`` from an open sparse safetensors handle.

    Flat, self-describing format: param names are recovered from the ``{name}.val`` tensor keys and the index encoding
    is a single global header field (the gap-delta width is carried by the index tensor's own dtype). ``f`` is a
    ``safetensors.safe_open`` handle.
    """
    encoding = PatchMetadata.from_metadata_dict(f.metadata()).encoding
    names = sorted({k[: -len(".val")] for k in f.keys() if k.endswith(".val")})
    for name in names:
        idx = _decode_idx(f.get_tensor(f"{name}.idx"), encoding)
        yield name, idx, f.get_tensor(f"{name}.val")


class DeltaWorkerExtension:
    """vLLM worker-extension hook (pass via ``--worker-extension-cls``).

    Required: ``--worker-extension-cls`` makes the vLLM *worker* process import this module, which runs the
    ``register_engine`` call below so the ``"delta"`` backend exists in the worker (the factory registry is
    per-process)"""

    pass


if "delta" not in WeightTransferEngineFactory._registry:
    WeightTransferEngineFactory.register_engine("delta", DeltaWeightTransferEngine)
