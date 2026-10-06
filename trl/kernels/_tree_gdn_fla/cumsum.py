# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in this directory.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
import triton.language as tl


@triton.heuristics(
    {
        "HAS_SCALE": lambda args: args["scale"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4, 8]],
    key=["B", "H", "BT", "IS_VARLEN", "REVERSE"],
)
@triton.jit(do_not_specialize=["T"])
def chunk_local_cumsum_scalar_kernel(
    s,
    o,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    BT: tl.constexpr,
    REVERSE: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    if HEAD_FIRST:
        p_s = s + bos * H + i_h * T + o_t
        p_o = o + bos * H + i_h * T + o_t
    else:
        p_s = s + bos * H + i_h + o_t * H
        p_o = o + bos * H + i_h + o_t * H
    # [BT]
    b_s = tl.load(p_s, mask=m_t, other=0.0).to(tl.float32)
    b_o = tl.cumsum(b_s, axis=0)
    if REVERSE:
        b_z = tl.sum(b_s, axis=0)
        b_o = -b_o + b_z[None] + b_s
    if HAS_SCALE:
        b_o *= scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=m_t)


def chunk_local_cumsum_scalar(
    g: torch.Tensor,
    chunk_size: int,
    reverse: bool = False,
    scale: float = None,
    cu_seqlens: torch.Tensor | None = None,
    head_first: bool = False,
    output_dtype: torch.dtype | None = torch.float,
    chunk_indices: torch.LongTensor | None = None,
) -> torch.Tensor:
    if head_first:
        B, H, T = g.shape
    else:
        B, T, H = g.shape
    assert chunk_size == 2 ** (chunk_size.bit_length() - 1), "chunk_size must be a power of 2"
    BT = chunk_size
    assert chunk_indices is not None  # Built once in TreeGDNPlan, never inside a layer.
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    g_org, g = g, torch.empty_like(g, dtype=output_dtype or g.dtype)
    grid = (NT, B * H)
    chunk_local_cumsum_scalar_kernel[grid](
        s=g_org,
        o=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        B=B,
        H=H,
        BT=BT,
        HEAD_FIRST=head_first,
        REVERSE=reverse,
    )
    return g
