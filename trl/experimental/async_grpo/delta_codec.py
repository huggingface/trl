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
GPU-resident sparse delta extraction + index encoding.

The change mask, ``nonzero``, and value gather all run on the device, so only the sparse payload — and, for the disk
transport, its compressed index form — crosses PCIe. This replaces the dense ``tensor.to(bf16).cpu()`` + CPU
``nonzero``/gather in the v1 upload path, which copied the full dense tensor (~100%) to host just to keep ~1-3% of it.

Index encodings (lossless, shrink only the index half — values are sent raw):

- ``raw`` : int32 absolute flat positions (4 B/elem) — what NCCL / vLLM #40096 expect.
- ``gap_delta`` : ``idx[k] - idx[k-1] - 1`` packed to uint16 (2 B), uint32 fallback per param.
- ``nvcomp`` : nvCOMP "Cascaded" (delta + bit-pack [+ RLE]) over the int32 indices, on GPU.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
import torch


try:
    from nvidia import nvcomp  # pip: nvidia-nvcomp

    _NVCOMP_OK = True
except Exception:  # pragma: no cover - environment dependent
    nvcomp = None
    _NVCOMP_OK = False


class Encoding(str, Enum):
    """Index-encoding scheme for a sparse delta patch (values are always stored raw)."""

    RAW = "raw"  # int32 absolute positions (4 B/elem)
    GAP_DELTA = "gap_delta"  # uint16 gaps, uint32 fallback per param (~2 B/elem)
    NVCOMP_CASCADED = "nvcomp_cascaded"  # nvCOMP Cascaded delta+bitpack on the GPU (~1.3 B/elem)


class UpdateKind(str, Enum):
    """vLLM weight-update format (the ``update_kind`` field in the update_info sent to vLLM)."""

    DENSE = "dense"
    SPARSE_FLAT = "sparse_flat"


def extract_sparse(
    param: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pull changed ``(indices, values)`` for one param, entirely on ``param``'s device.

    Args:
        param (`torch.Tensor`):
            Full dense parameter (kept on the GPU; never copied to host here).
        mask (`torch.Tensor`):
            Boolean change mask, same numel as ``param``, same device.

    Returns:
        `tuple` of:
            - indices (`torch.Tensor`): 1D `int32` flat positions (ascending), on device.
            - values (`torch.Tensor`): 1D values at those positions, param dtype, on device.
    """
    flat = param.detach().reshape(-1)
    idx = mask.reshape(-1).nonzero(as_tuple=True)[0]  # device; one sync (dynamic size)
    vals = flat.index_select(0, idx)
    return idx.to(torch.int32), vals


def extract_sparse_batched(
    items: list[tuple[str, torch.Tensor, torch.Tensor]],
) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
    """Batched [`extract_sparse`] over many params with a single ``nonzero``.

    ``nonzero`` forces a device→host sync (its output size is data-dependent), so the per-param loop costs one sync per
    param. This concatenates all flattened masks, runs **one** ``nonzero`` over the whole set, then splits the global
    positions back per param with ``searchsorted`` on the cumulative sizes — collapsing ~N syncs into 2 (the
    ``nonzero`` and one boundary D2H). Indices are returned local to each param's flat space (ready for that param's
    ``index_copy_``).

    All tensors must be on the same device. (Transient cost: a concatenated full-size bool mask — fine for the model
    sizes here; very large models should shard, see the multi-file TODO.)

    Args:
        items (`list[tuple[str, torch.Tensor, torch.Tensor]]`):
            ``(name, tensor, mask)`` triples; ``mask`` is the per-param boolean change mask.

    Returns:
        `list[tuple[str, torch.Tensor, torch.Tensor]]`: ``(name, int32 local indices, values)`` per input param, in
        input order.
    """
    if not items:
        return []
    device = items[0][1].device
    flats = [tensor.detach().reshape(-1) for _, tensor, _ in items]
    sizes = torch.tensor([f.numel() for f in flats], device=device)
    offsets = torch.cat([sizes.new_zeros(1), torch.cumsum(sizes, 0)])
    global_idx = torch.cat([mask.reshape(-1) for _, _, mask in items]).nonzero(as_tuple=True)[0]
    bounds = torch.searchsorted(global_idx, offsets[1:]).tolist()

    out = []
    prev = 0
    for i, (name, _, _) in enumerate(items):
        g = global_idx[prev : bounds[i]] - offsets[i]  # local positions within this param
        out.append((name, g.to(torch.int32), flats[i].index_select(0, g)))
        prev = bounds[i]
    return out


def gap_delta_encode(idx: torch.Tensor) -> torch.Tensor:
    """Gap-encode sorted positions: ``delta[k] = idx[k] - idx[k-1] - 1`` (idx[-1] := -1).

    Returns the gaps as ``uint16`` if the max gap fits, else ``uint32`` — so one outlier never bumps the whole param to
    4 B. The dtype *is* the width (no separate width needed); the receiver inverts with [`gap_delta_decode`].

    Args:
        idx (`torch.Tensor`):
            1D ascending `int32` positions (as returned by [`extract_sparse`]).

    Returns:
        `torch.Tensor`: 1D gaps, dtype `uint16` or `uint32`, same device.
    """
    if idx.numel() == 0:
        return idx.to(torch.uint16)
    idx64 = idx.long()
    prev = torch.cat([idx64.new_full((1,), -1), idx64[:-1]])
    deltas = idx64 - prev - 1
    return deltas.to(torch.uint16 if int(deltas.max()) <= 0xFFFF else torch.uint32)


def gap_delta_decode(deltas: torch.Tensor) -> torch.Tensor:
    """Invert [`gap_delta_encode`] → 1D ascending `int32` positions on ``deltas``' device."""
    if deltas.numel() == 0:
        return deltas.to(torch.int32)
    return (torch.cumsum(deltas.long() + 1, dim=0) - 1).to(torch.int32)


def nvcomp_available() -> bool:
    return _NVCOMP_OK


def nvcomp_encode(idx: torch.Tensor) -> torch.Tensor:
    """Compress absolute int32 indices with nvCOMP Cascaded → ``uint8`` byte tensor (CPU).

    Cascaded does the delta + bit-pack internally, so callers pass raw int32 positions (no gap-encoding needed). The
    self-describing bitstream carries dtype + length, so [`nvcomp_decode`] needs nothing else. Cascaded is a GPU codec,
    so indices are moved to CUDA.
    """
    if not _NVCOMP_OK:
        raise RuntimeError("nvidia-nvcomp not installed")
    comp = nvcomp.Codec(algorithm="Cascaded").encode(nvcomp.as_array(idx.to("cuda", torch.int32).contiguous()))
    return torch.from_numpy(np.asarray(comp.cpu()).view(np.uint8).copy())


def nvcomp_decode(raw: torch.Tensor) -> torch.Tensor:
    """Inverse of [`nvcomp_encode`]: ``uint8`` bytes → 1D ``int32`` indices (CPU)."""
    if not _NVCOMP_OK:
        raise RuntimeError("nvidia-nvcomp not installed")
    dec = nvcomp.Codec(algorithm="Cascaded").decode(nvcomp.as_array(raw.to("cuda").contiguous()))
    return torch.from_numpy(np.asarray(dec.cpu()).view(np.int32).copy())  # reinterpret bytes → int32
