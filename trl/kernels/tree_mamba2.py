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
#
# `grouped_bwd_dqkg_kernel` below is adapted from `chunk_bwd_kernel_dqkwg` of flash-linear-attention 0.5.2
# (https://github.com/fla-org/flash-linear-attention): fla/ops/common/chunk_o.py, distributed under the MIT license:
#
# MIT License
#
# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Mamba2's chunked scan (SSD) over a tree-packed row, run by the tree kernels of the gated delta rule.

SSD is gated linear attention with a scalar decay per head (https://huggingface.co/papers/2405.21060): the state `S_t =
exp(dt_t A) S_{t-1} + B_t (dt_t x_t)^T` is read out as `y_t = C_t^T S_t + D x_t`. With queries `C`, keys `B`, values
`dt x` and log-gates `dt A`, that is the gated delta rule without its delta (WY) term, so the same chunk and tree-state
kernels run it: each segment starts from its parent's final state, and in backward sums its children's state gradients.
SSD's own kernels compute the elementwise `dt x` and `D x` terms, and the query and key gradients, summed over the
heads that share a group of `B` and `C`.
"""

import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from .tree_gated_delta_rule import (
    IS_NVIDIA_HOPPER,
    TRITON_ABOVE_3_4_0,
    TRITON_ABOVE_3_7_1,
    backward_states,
    chunk_bwd_dv_local,
    chunk_fwd_o,
    chunk_local_cumsum_scalar,
    exp2,
    forward_states,
    tree_causal_conv1d,
    tree_plan,
)


def tree_mamba2_chunk_scan(
    scan, hidden_states, dt, A, B, C, chunk_size, D=None, dt_bias=None, dt_limit=(0.0, math.inf), **tree
):
    """
    Tree version of transformers' `mamba2_chunk_scan` (`scan`, unused): `(1, tokens, heads, head_dim)` inputs
    `hidden_states`, `(1, tokens, heads)` raw time steps `dt`, `(1, tokens, groups, state)` `B` and `C`.
    """
    dt = F.softplus(dt + dt_bias.to(dt.dtype)).clamp(*dt_limit).float()
    return TreeSSD.apply(C, B, hidden_states, dt, A * dt, D, tree_plan(tree))


def tree_mamba2_split_conv1d_scan_combined(
    fused,
    zxbcdt,
    conv1d_weight,
    conv1d_bias,
    dt_bias,
    A,
    D,
    chunk_size,
    initial_states=None,
    dt_limit=(0.0, math.inf),
    return_final_states=False,
    activation="silu",
    rmsnorm_weight=None,
    rmsnorm_eps=1e-6,
    outproj_weight=None,
    outproj_bias=None,
    headdim=None,
    ngroups=1,
    norm_before_gate=True,
    **tree,
):
    """
    Tree version of transformers' fused `mamba2_split_conv1d_scan_combined`: the whole mixer after `in_proj`, from
    `zxbcdt` `(1, tokens, inner + conv channels + heads)` to the output projection, through the tree convolution, the
    tree scan and a gated RMSNorm over groups of `inner / groups` channels. Like `mamba_ssm`'s, it saves only `zxbcdt`
    and the scan output for backward, and recomputes the rest.
    """
    args = (zxbcdt, conv1d_weight, conv1d_bias, dt_bias, A, D, rmsnorm_weight, outproj_weight, outproj_bias)
    return TreeMamba2Mixer.apply(*args, activation, dt_limit, headdim, ngroups, rmsnorm_eps, tree_plan(tree))


def ssd_forward(q, k, x, dt, g, D, tree):
    layout = dict(cu_seqlens=tree["tree_cu_seqlens_q"], chunk_indices=tree["tree_chunk_indices"])
    g = chunk_local_cumsum_scalar(g, scale=1.4426950408889634, **layout)
    v = scale_values(x, dt)
    h, _ = forward_states(k, None, v, g, tree)
    out = chunk_fwd_o(q, k, v, h, g=g, scale=1.0, **layout)
    add_skip(out, x, D)
    return out, g


def ssd_backward(q, k, x, dt, g, D, do, tree):
    layout = dict(cu_seqlens=tree["tree_cu_seqlens_q"], chunk_indices=tree["tree_chunk_indices"])
    v = scale_values(x, dt)
    h, _ = forward_states(k, None, v, g, tree)
    dv = chunk_bwd_dv_local(q, k, do, g=g, scale=1.0, **layout)
    dh, dv = backward_states(q, k, None, g, do, dv, tree, 1.0)
    dq, dk, dg = grouped_bwd_dqkg(q, k, v, g, h, do, dh, **layout)
    dg = chunk_local_cumsum_scalar(dg, reverse=True, **layout)
    dx, ddt, dD = skip_bwd(do, dv, x, dt, D)
    return dq, dk, dx, ddt, dg, dD


class TreeSSD(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, x, dt, g, D, tree):
        q, k, x, dt, g = [t.contiguous() for t in (q, k, x, dt, g)]
        out, g = ssd_forward(q, k, x, dt, g, D, tree)
        ctx.save_for_backward(q, k, x, dt, g, D)
        ctx.tree = tree
        return out

    @staticmethod
    def backward(ctx, do):
        return *ssd_backward(*ctx.saved_tensors, do.contiguous(), ctx.tree), None


class TreeMamba2Mixer(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, zxbcdt, conv_w, conv_b, dt_bias, A, D, norm_w, out_w, out_b, act, dt_limit, P, G, eps, tree):
        ctx.meta = act, dt_limit, P, G, eps, tree
        q, k, x, dt, g, z = scan_inputs(zxbcdt, conv_w, conv_b, dt_bias, A, act, dt_limit, P, G, tree)
        y, _ = ssd_forward(q, k, x, dt, g, D, tree)
        y = y.flatten(2)
        normed, rstd = gated_rmsnorm(y, z, norm_w, G, eps)
        ctx.save_for_backward(zxbcdt, conv_w, conv_b, dt_bias, A, D, norm_w, out_w, y, rstd)
        ctx.has_out_b = out_b is not None
        return F.linear(normed, out_w, out_b)

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, dout):
        zxbcdt, conv_w, conv_b, dt_bias, A, D, norm_w, out_w, y, rstd = ctx.saved_tensors
        act, dt_limit, P, G, eps, tree = ctx.meta
        layout = dict(cu_seqlens=tree["tree_cu_seqlens_q"], chunk_indices=tree["tree_chunk_indices"])
        dout = dout.contiguous()
        z = zxbcdt[..., : y.shape[-1]]
        normed, _ = gated_rmsnorm(y, z, norm_w, G, eps)
        d_out_w = dout.flatten(0, 1).T @ normed.flatten(0, 1)
        d_out_b = dout.flatten(0, 1).sum(0) if ctx.has_out_b else None
        dy, dz, d_norm_w = gated_rmsnorm_bwd(dout @ out_w, y, z, norm_w, rstd, G)
        with torch.enable_grad():
            leaves = [t.detach().requires_grad_() for t in (zxbcdt, conv_w, conv_b, dt_bias, A)]
            q, k, x, dt, g, _ = scan_inputs(*leaves, act, dt_limit, P, G, tree)
        gc = chunk_local_cumsum_scalar(g.detach(), scale=1.4426950408889634, **layout)
        dq, dk, dx, ddt, dg, dD = ssd_backward(
            q.detach(), k.detach(), x.detach(), dt.detach(), gc, D, dy.view_as(x), tree
        )
        dzxbcdt, d_conv_w, d_conv_b, d_dt_bias, dA = torch.autograd.grad(
            (q, k, x, dt, g), leaves, (dq, dk, dx, ddt, dg)
        )
        dzxbcdt[..., : y.shape[-1]] += dz
        d_out_b = d_out_b.to(out_w.dtype) if d_out_b is not None else None
        grads = dzxbcdt, d_conv_w, d_conv_b, d_dt_bias, dA, dD.to(D.dtype), d_norm_w, d_out_w.to(out_w.dtype), d_out_b
        return *grads, None, None, None, None, None, None


def scan_inputs(zxbcdt, conv_w, conv_b, dt_bias, A, act, dt_limit, P, G, tree):
    inner, heads = A.shape[0] * P, A.shape[0]
    z, xBC, dt = zxbcdt.split([inner, conv_w.shape[0], heads], dim=-1)
    xBC = tree_causal_conv1d(None, xBC.transpose(1, 2), conv_w, conv_b, activation=act, **tree).transpose(1, 2)
    x, B, C = xBC.split([inner, (conv_w.shape[0] - inner) // 2, (conv_w.shape[0] - inner) // 2], dim=-1)
    dt = F.softplus(dt + dt_bias.to(dt.dtype)).clamp(*dt_limit).float()
    q, k, x = (t.contiguous() for t in (C.unflatten(-1, (G, -1)), B.unflatten(-1, (G, -1)), x.unflatten(-1, (-1, P))))
    return q, k, x, dt.contiguous(), (A * dt).contiguous(), z


# --- Elementwise terms: values `dt x` and skip `D x` ---


@triton.jit(do_not_specialize=["T"])
def scale_values_kernel(x, dt, v, T, HV: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BV: tl.constexpr):
    i_t, i_h = tl.program_id(0), tl.program_id(1)
    o_t = i_t * BT + tl.arange(0, BT)
    o_v = tl.arange(0, BV)
    m = (o_t < T)[:, None] & (o_v < V)[None, :]
    p = o_t[:, None].to(tl.int64) * (HV * V) + i_h * V + o_v[None, :]
    b_dt = tl.load(dt + o_t * HV + i_h, mask=o_t < T, other=0.0)
    b_x = tl.load(x + p, mask=m, other=0.0).to(tl.float32)
    tl.store(v + p, (b_x * b_dt[:, None]).to(v.dtype.element_ty), mask=m)


@triton.jit(do_not_specialize=["T"])
def add_skip_kernel(o, x, D, T, HV: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BV: tl.constexpr):
    i_t, i_h = tl.program_id(0), tl.program_id(1)
    o_t = i_t * BT + tl.arange(0, BT)
    o_v = tl.arange(0, BV)
    m = (o_t < T)[:, None] & (o_v < V)[None, :]
    p = o_t[:, None].to(tl.int64) * (HV * V) + i_h * V + o_v[None, :]
    b_x = tl.load(x + p, mask=m, other=0.0).to(tl.float32)
    b_o = tl.load(o + p, mask=m, other=0.0).to(tl.float32)
    tl.store(o + p, (b_o + tl.load(D + i_h).to(tl.float32) * b_x).to(o.dtype.element_ty), mask=m)


@triton.jit(do_not_specialize=["T"])
def skip_bwd_kernel(
    do, dv, x, dt, D, dx, ddt, dD, T, HV: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BV: tl.constexpr
):
    i_t, i_h = tl.program_id(0), tl.program_id(1)
    o_t = i_t * BT + tl.arange(0, BT)
    o_v = tl.arange(0, BV)
    m_t = o_t < T
    m = m_t[:, None] & (o_v < V)[None, :]
    p = o_t[:, None].to(tl.int64) * (HV * V) + i_h * V + o_v[None, :]
    b_x = tl.load(x + p, mask=m, other=0.0).to(tl.float32)
    b_do = tl.load(do + p, mask=m, other=0.0).to(tl.float32)
    b_dv = tl.load(dv + p, mask=m, other=0.0).to(tl.float32)
    b_dt = tl.load(dt + o_t * HV + i_h, mask=m_t, other=0.0)
    b_D = tl.load(D + i_h).to(tl.float32)
    tl.store(dx + p, (b_dv * b_dt[:, None] + b_D * b_do).to(dx.dtype.element_ty), mask=m)
    tl.store(ddt + o_t * HV + i_h, tl.sum(b_dv * b_x, axis=1), mask=m_t)
    tl.store(dD + i_t * HV + i_h, tl.sum(b_do * b_x))


def elementwise_grid(x):
    _, tokens, value_heads, value_dim = x.shape
    # Disable FP fusion, as Triton 3.10 does by default (see #8136)
    meta = dict(
        T=tokens, HV=value_heads, V=value_dim, BT=64, BV=triton.next_power_of_2(value_dim), enable_fp_fusion=False
    )
    return (triton.cdiv(tokens, 64), value_heads), meta


def scale_values(x, dt):
    v = torch.empty_like(x)
    grid, meta = elementwise_grid(x)
    scale_values_kernel[grid](x, dt, v, **meta)
    return v


def add_skip(o, x, D):
    grid, meta = elementwise_grid(x)
    add_skip_kernel[grid](o, x, D, **meta)


def skip_bwd(do, dv, x, dt, D):
    grid, meta = elementwise_grid(x)
    dx = torch.empty_like(x)
    ddt = torch.empty_like(dt)
    dD = dt.new_empty(grid)
    skip_bwd_kernel[grid](do, dv, x, dt, D, dx, ddt, dD, **meta)
    return dx, ddt, dD.sum(0).to(D.dtype)


# --- Query and key gradients, summed over the heads of a group ---


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages) for num_warps in [4, 8] for num_stages in [2, 3]
    ],
    key=["H", "HV", "K", "V", "BK", "BV"],
)
@triton.jit(do_not_specialize=["T"])
def grouped_bwd_dqkg_kernel(
    q,
    k,
    v,
    g,
    h,
    do,
    dh,
    dq,
    dk,
    dg,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    HP: tl.constexpr,
):
    i_k, i_tg, i_hs = tl.program_id(0), tl.program_id(1).to(tl.int64), tl.program_id(2)
    S: tl.constexpr = HV // H // HP
    i_hq, i_s = i_hs // S, i_hs % S
    dg += i_k.to(tl.int64) * T * HV
    dq += i_s.to(tl.int64) * T * H * K
    dk += i_s.to(tl.int64) * T * H * K
    i_n, i_t = tl.load(chunk_indices + i_tg * 2).to(tl.int32), tl.load(chunk_indices + i_tg * 2 + 1).to(tl.int32)
    bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
    T = (eos - bos).to(tl.int32)

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    o_k = i_k * BK + tl.arange(0, BK)
    m_k = o_k < K
    m_qk = m_t[:, None] & m_k[None, :]
    m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
    last = min(i_t * BT + BT, T) - 1

    p_qk = (bos + o_t[:, None]) * (H * K) + i_hq * K + o_k[None, :]
    b_q = tl.load(q + p_qk, mask=m_qk, other=0.0)
    b_k = tl.load(k + p_qk, mask=m_qk, other=0.0)
    b_dq_sum = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk_sum = tl.zeros([BT, BK], dtype=tl.float32)

    for i_g in range(HP):
        i_h = (i_hq * S + i_s) * HP + i_g
        b_dg_last = tl.zeros([1], dtype=tl.float32)
        b_dq = tl.zeros([BT, BK], dtype=tl.float32)
        b_dk = tl.zeros([BT, BK], dtype=tl.float32)
        b_ds = tl.zeros([BT, BT], dtype=tl.float32)
        for i_v in range(tl.cdiv(V, BV)):
            o_v = i_v * BV + tl.arange(0, BV)
            m_v = o_v < V
            m_tv = m_t[:, None] & m_v[None, :]
            p_tv = (bos + o_t[:, None]) * (HV * V) + i_h * V + o_v[None, :]
            p_kv = (i_tg * HV + i_h) * K * V + o_v[:, None] + o_k[None, :] * V
            m_kv = m_v[:, None] & m_k[None, :]
            b_v = tl.load(v + p_tv, mask=m_tv, other=0.0)
            b_do = tl.load(do + p_tv, mask=m_tv, other=0.0)
            b_h = tl.load(h + p_kv, mask=m_kv, other=0.0)
            b_dh = tl.load(dh + p_kv, mask=m_kv, other=0.0)
            b_dg_last += tl.sum(b_h * b_dh)
            b_ds += tl.dot(b_do, tl.trans(b_v))
            b_dq += tl.dot(b_do, b_h.to(b_do.dtype))
            b_dk += tl.dot(b_v, b_dh.to(b_v.dtype))

        b_g = tl.load(g + (bos + o_t) * HV + i_h, mask=m_t, other=0.0)
        b_g_last = tl.load(g + (bos + last) * HV + i_h)
        b_dg_last *= exp2(b_g_last)
        b_dq = b_dq * exp2(b_g)[:, None]
        b_dk = b_dk * tl.where(m_t, exp2(-b_g + b_g_last), 0)[:, None]
        b_dg_last += tl.sum(b_dk * b_k)
        b_ds = tl.where(m_A, b_ds * exp2(b_g[:, None] - b_g[None, :]), 0).to(b_k.dtype)
        b_dq += tl.dot(b_ds, b_k)
        b_dk += tl.dot(tl.trans(b_ds), b_q)
        b_dg = tl.sum(b_dq * b_q, axis=1) - tl.sum(b_dk * b_k, axis=1)
        b_dg = tl.where(o_t < last, b_dg, b_dg + b_dg_last)
        tl.store(dg + (bos + o_t) * HV + i_h, b_dg, mask=m_t)
        b_dq_sum += b_dq
        b_dk_sum += b_dk

    tl.store(dq + p_qk, b_dq_sum.to(dq.dtype.element_ty), mask=m_qk)
    tl.store(dk + p_qk, b_dk_sum.to(dk.dtype.element_ty), mask=m_qk)


def grouped_bwd_dqkg(q, k, v, g, h, do, dh, cu_seqlens, chunk_indices):
    if IS_NVIDIA_HOPPER and TRITON_ABOVE_3_4_0 and not TRITON_ABOVE_3_7_1:
        raise RuntimeError(
            "Triton >= 3.4.0 and < 3.7.1 on Hopper GPUs produces incorrect results for gated chunk_bwd_dqkwg (see "
            "#640). Please upgrade Triton to >= 3.7.1."
        )
    _, tokens, heads, key_dim = q.shape
    value_heads, value_dim = v.shape[2:]
    BK = min(max(triton.next_power_of_2(key_dim), 16), 64)
    BV = min(max(triton.next_power_of_2(value_dim), 16), 64)
    NK = triton.cdiv(key_dim, BK)
    HP = math.gcd(value_heads // heads, 8)
    NS = value_heads // heads // HP
    dq = q.new_empty(NS, *q.shape, dtype=q.dtype if NS == 1 else torch.float32)
    dk = torch.empty_like(dq)
    dg = g.new_empty(NK, *g.shape)
    # Disable FP fusion, as Triton 3.10 does by default (see #8136)
    grouped_bwd_dqkg_kernel[(NK, len(chunk_indices), value_heads // HP)](
        q, k, v, g, h, do, dh, dq, dk, dg, cu_seqlens, chunk_indices,
        T=tokens, H=heads, HV=value_heads, K=key_dim, V=value_dim, BT=64, BK=BK, BV=BV, HP=HP,
        enable_fp_fusion=False,
    )  # fmt: skip
    return dq.sum(0).to(q.dtype), dk.sum(0).to(k.dtype), dg.sum(0)


# --- Gated RMSNorm over groups: `norm(y * silu(z)) * weight` ---


@triton.jit(do_not_specialize=["T"])
def gated_rmsnorm_fwd_kernel(
    y, z, w, o, rstd, eps, T, stride_z, D: tl.constexpr, N: tl.constexpr, BN: tl.constexpr, ROWS: tl.constexpr
):
    i_r, i_g = tl.program_id(0), tl.program_id(1)
    o_n = tl.arange(0, BN)
    m = o_n < N
    b_w = tl.load(w + i_g * N + o_n, mask=m, other=0.0).to(tl.float32)
    for i in range(ROWS):
        r = i_r * ROWS + i
        if r < T:
            p = r.to(tl.int64) * D + i_g * N + o_n
            b_z = tl.load(z + r.to(tl.int64) * stride_z + i_g * N + o_n, mask=m, other=0.0).to(tl.float32)
            b_x = tl.load(y + p, mask=m, other=0.0).to(tl.float32) * b_z * tl.sigmoid(b_z)
            b_rstd = 1.0 / tl.sqrt(tl.sum(b_x * b_x) / N + eps)
            tl.store(o + p, (b_x * b_rstd * b_w).to(o.dtype.element_ty), mask=m)
            tl.store(rstd + r * (D // N) + i_g, b_rstd)


@triton.jit(do_not_specialize=["T"])
def gated_rmsnorm_bwd_kernel(
    do, y, z, w, rstd, dy, dz, dw, T, stride_z, D: tl.constexpr, N: tl.constexpr, BN: tl.constexpr, ROWS: tl.constexpr
):
    i_r, i_g = tl.program_id(0), tl.program_id(1)
    o_n = tl.arange(0, BN)
    m = o_n < N
    b_w = tl.load(w + i_g * N + o_n, mask=m, other=0.0).to(tl.float32)
    b_dw = tl.zeros([BN], dtype=tl.float32)
    for i in range(ROWS):
        r = i_r * ROWS + i
        if r < T:
            p = r.to(tl.int64) * D + i_g * N + o_n
            b_y = tl.load(y + p, mask=m, other=0.0).to(tl.float32)
            b_z = tl.load(z + r.to(tl.int64) * stride_z + i_g * N + o_n, mask=m, other=0.0).to(tl.float32)
            b_do = tl.load(do + p, mask=m, other=0.0).to(tl.float32)
            b_s = tl.sigmoid(b_z)
            b_rstd = tl.load(rstd + r * (D // N) + i_g)
            b_xhat = b_y * b_z * b_s * b_rstd
            b_dxhat = b_do * b_w
            b_dw += b_do * b_xhat
            b_dx = b_rstd * (b_dxhat - b_xhat * tl.sum(b_dxhat * b_xhat) / N)
            tl.store(dy + p, (b_dx * b_z * b_s).to(dy.dtype.element_ty), mask=m)
            tl.store(dz + p, (b_dx * b_y * b_s * (1 + b_z * (1 - b_s))).to(dz.dtype.element_ty), mask=m)
    tl.store(dw + i_r.to(tl.int64) * D + i_g * N + o_n, b_dw, mask=m)


def gated_rmsnorm(y, z, w, groups, eps):
    tokens, width = y.shape[1], y.shape[2]
    out = torch.empty_like(y)
    rstd = y.new_empty(tokens, groups, dtype=torch.float32)
    N = width // groups
    # Disable FP fusion, as Triton 3.10 does by default (see #8136)
    gated_rmsnorm_fwd_kernel[(triton.cdiv(tokens, 16), groups)](
        y,
        z,
        w,
        out,
        rstd,
        eps,
        tokens,
        z.stride(1),
        D=width,
        N=N,
        BN=triton.next_power_of_2(N),
        ROWS=16,
        enable_fp_fusion=False,
    )
    return out, rstd


def gated_rmsnorm_bwd(do, y, z, w, rstd, groups):
    tokens, width = y.shape[1], y.shape[2]
    dy, dz = torch.empty_like(y), torch.empty_like(y)
    N = width // groups
    grid = (triton.cdiv(tokens, 64), groups)
    dw = y.new_empty(grid[0], width, dtype=torch.float32)
    # Disable FP fusion, as Triton 3.10 does by default (see #8136)
    gated_rmsnorm_bwd_kernel[grid](
        do, y, z, w, rstd, dy, dz, dw, tokens, z.stride(1), D=width, N=N, BN=triton.next_power_of_2(N), ROWS=64,
        enable_fp_fusion=False,
    )  # fmt: skip
    return dy, dz, dw.sum(0).to(w.dtype)
