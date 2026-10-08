# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
# Copyright 2026 The HuggingFace Team. All rights reserved.
# Licensed under the MIT license in this directory.

"""Tree-packed Gated DeltaNet: FLA's chunk math with tree-native state propagation.

One flattened forest, 64-token chunks, scalar gates, key dimensions <= 128, and key-first states. The pipeline is
followed by its Triton kernels below.
"""

import torch
import triton
import triton.language as tl
from packaging.version import Version


# --- Device/compiler checks and FP32 exponential ---


IS_NVIDIA_HOPPER = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9
IS_NVIDIA_BLACKWELL = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10
TRITON_ABOVE_3_4_0 = Version(triton.__version__) >= Version("3.4.0")
TRITON_ABOVE_3_7_1 = Version(triton.__version__) >= Version("3.7.1")


def check_shared_mem(arch="ampere", device=None):
    if not torch.cuda.is_available():
        return False
    required = {"ampere": 163840, "ada": 101376, "hopper": 232448}[arch]
    return torch.cuda.get_device_properties(device).shared_memory_per_block_optin >= required


@triton.jit
def exp2(x):
    return tl.math.exp2(x.to(tl.float32))


# --- Autograd pipeline and depth-ordered state launches ---


def forward_states(k, w, u, g, plan):
    _, tokens, heads, key_dim = k.shape
    value_heads, value_dim = u.shape[2:]
    h = k.new_empty(1, len(plan.chunk_indices), value_heads, key_dim, value_dim)
    final = k.new_empty(len(plan.parents), value_heads, key_dim, value_dim, dtype=torch.float32)
    v_new = torch.empty_like(u)
    for segments in plan.levels:
        chunk_gated_delta_rule_fwd_kernel_h_blockdim64[
            lambda meta, segments=segments: (triton.cdiv(value_dim, meta["BV"]), len(segments) * value_heads)
        ](
            k=k,
            v=u,
            w=w,
            v_new=v_new,
            g=g,
            h=h,
            h0=final,
            ht=final,
            cu_seqlens=plan.cu_seqlens,
            chunk_offsets=plan.chunk_offsets,
            segments=segments,
            parents=plan.parents,
            T=tokens,
            H=heads,
            HV=value_heads,
            K=key_dim,
            V=value_dim,
            BT=64,
        )
    return h, v_new


def backward_states(q, k, w, g, do, dv, plan, scale):
    _, tokens, heads, key_dim = k.shape
    value_heads, value_dim = do.shape[2:]
    dh = k.new_empty(1, len(plan.chunk_indices), value_heads, key_dim, value_dim)
    d_initial = k.new_empty(len(plan.parents), value_heads, key_dim, value_dim, dtype=torch.float32)
    du = torch.empty_like(dv)
    for segments in reversed(plan.levels):
        chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64[
            lambda meta, segments=segments: (triton.cdiv(value_dim, meta["BV"]), len(segments) * value_heads)
        ](
            q=q,
            k=k,
            w=w,
            g=g,
            dht=d_initial,
            dh0=d_initial,
            do=do,
            dh=dh,
            dv=dv,
            dv2=du,
            cu_seqlens=plan.cu_seqlens,
            chunk_offsets=plan.chunk_offsets,
            segments=segments,
            child_offsets=plan.child_offsets,
            children=plan.children,
            scale=scale,
            T=tokens,
            H=heads,
            HV=value_heads,
            K=key_dim,
            V=value_dim,
            BT=64,
        )
    return dh, du


class TreeGatedDeltaRule(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, beta, plan, scale, normalize):
        q, k, v, g, beta = [x.contiguous() for x in (q, k, v, g, beta)]
        q_rstd, k_rstd = None, None
        if normalize:
            q, q_rstd = l2norm_fwd(q)
            k, k_rstd = l2norm_fwd(k)
        layout = dict(cu_seqlens=plan.cu_seqlens, chunk_indices=plan.chunk_indices)
        g = chunk_local_cumsum_scalar(g, scale=1.4426950408889634, **layout)
        # Retain upstream's fused 64-token KKT + triangular solve, and low-precision WY representation.
        a = k.new_zeros(1, k.shape[1], v.shape[2], 64)
        chunk_gated_delta_rule_fwd_kkt_solve_kernel[(len(plan.chunk_indices), v.shape[2])](
            k=k,
            g=g,
            beta=beta,
            A=a,
            T=k.shape[1],
            H=k.shape[2],
            HV=v.shape[2],
            K=k.shape[3],
            BT=64,
            BC=16,
            **layout,
        )
        w, u = recompute_w_u_fwd(k, v, beta, a, g, **layout)
        h, v_new = forward_states(k, w, u, g, plan)
        out = chunk_fwd_o(q, k, v_new, h, g=g, scale=scale, **layout)
        ctx.save_for_backward(q, q_rstd, k, k_rstd, v, g, beta, a)
        ctx.plan, ctx.scale, ctx.normalize = plan, scale, normalize
        return out

    @staticmethod
    def backward(ctx, do):
        q, q_rstd, k, k_rstd, v, g, beta, a = ctx.saved_tensors
        plan, scale = ctx.plan, ctx.scale
        do = do.contiguous()
        layout = dict(cu_seqlens=plan.cu_seqlens, chunk_indices=plan.chunk_indices)
        w, u = recompute_w_u_fwd(k, v, beta, a, g, **layout)
        h, v_new = forward_states(k, w, u, g, plan)
        dv = chunk_bwd_dv_local(q, k, do, g=g, scale=scale, **layout)
        dh, du = backward_states(q, k, w, g, do, dv, plan, scale)
        dq, dk, dw, dg = chunk_bwd_dqkwg(q, k, v_new, do, h, dh, w=w, g=g, dv=du, scale=scale, **layout)
        dk2, dv, db, dg2 = prepare_wy_repr_bwd(k, v, beta, a, dw, du, g, **layout)
        dk.add_(dk2)
        dg.add_(dg2)
        dg = chunk_local_cumsum_scalar(dg, reverse=True, **layout)
        if ctx.normalize:
            dq = l2norm_bwd(q, q_rstd, dq)
            dk = l2norm_bwd(k, k_rstd, dk)
        return dq.to(q), dk.to(k), dv.to(v), dg, db.to(beta), None, None, None


# --- Tree state scan: parent reads forward, child-gradient sums backward ---


GATED_DELTA_RULE_FWD_H_NUM_WARPS = [2] if IS_NVIDIA_BLACKWELL else [2, 4]


@triton.autotune(
    configs=[
        triton.Config({"BV": BV}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in GATED_DELTA_RULE_FWD_H_NUM_WARPS
        for num_stages in ([2, 3, 4] if check_shared_mem("ampere") else [2, 1])
        for BV in ([32, 64] if check_shared_mem("ada") else [32])
    ],
    key=["H", "HV", "K", "V", "BT"],
)
@triton.jit(do_not_specialize=["T"])
def chunk_gated_delta_rule_fwd_kernel_h_blockdim64(
    k,
    v,
    w,
    v_new,
    g,
    h,
    h0,
    ht,
    cu_seqlens,
    chunk_offsets,
    segments,
    parents,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = tl.load(segments + i_nh // HV), i_nh % HV
    i_nh = i_n * HV + i_h
    bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    T = eos - bos
    NT = tl.cdiv(T, BT)
    boh = tl.load(chunk_offsets + i_n).to(tl.int32)

    b_h1 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 64:
        b_h2 = tl.zeros([64, BV], dtype=tl.float32)

    # calculate offset
    h += (boh * HV + i_h).to(tl.int64) * K * V
    v += (bos * HV + i_h).to(tl.int64) * V
    k += (bos * H + i_h // (HV // H)).to(tl.int64) * K
    w += (bos * HV + i_h).to(tl.int64) * K
    v_new += (bos * HV + i_h).to(tl.int64) * V

    parent = tl.load(parents + i_n)
    h0 = h0 + (parent * HV + i_h) * K * V
    ht = ht + i_nh * K * V

    # load initial state
    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V
    o_k1 = tl.arange(0, 64)
    m_k1 = o_k1 < K
    o_k2 = 64 + o_k1
    m_k2 = o_k2 < K
    if parent >= 0:
        p_h0_1 = h0 + o_k1[:, None] * V + o_v[None, :]
        m_h0_1 = m_k1[:, None] & m_v[None, :]
        b_h1 += tl.load(p_h0_1, mask=m_h0_1, other=0.0).to(tl.float32)
        if K > 64:
            p_h0_2 = h0 + o_k2[:, None] * V + o_v[None, :]
            m_h0_2 = m_k2[:, None] & m_v[None, :]
            b_h2 += tl.load(p_h0_2, mask=m_h0_2, other=0.0).to(tl.float32)

    # main recurrence
    for i_t in range(NT):
        i_t_int64 = i_t.to(tl.int64)
        o_t = i_t * BT + tl.arange(0, BT)
        m_t = o_t < T
        p_h1 = h + i_t_int64 * HV * K * V + o_k1[:, None] * V + o_v[None, :]
        m_h1 = m_k1[:, None] & m_v[None, :]
        tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), mask=m_h1)
        if K > 64:
            p_h2 = h + i_t_int64 * HV * K * V + o_k2[:, None] * V + o_v[None, :]
            m_h2 = m_k2[:, None] & m_v[None, :]
            tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), mask=m_h2)

        p_w = w + o_t[:, None] * (HV * K) + o_k1[None, :]
        b_w = tl.load(p_w, mask=m_t[:, None] & m_k1[None, :], other=0.0)
        b_v = tl.dot(b_w, b_h1.to(b_w.dtype))
        if K > 64:
            p_w = w + o_t[:, None] * (HV * K) + o_k2[None, :]
            b_w = tl.load(p_w, mask=m_t[:, None] & m_k2[None, :], other=0.0)
            b_v += tl.dot(b_w, b_h2.to(b_w.dtype))
        p_v = v + o_t[:, None] * (HV * V) + o_v[None, :]
        b_v = tl.load(p_v, mask=m_t[:, None] & m_v[None, :], other=0.0) - b_v

        p_v = v_new + o_t[:, None] * (HV * V) + o_v[None, :]
        tl.store(p_v, b_v.to(p_v.dtype.element_ty), mask=m_t[:, None] & m_v[None, :])

        last_idx = min((i_t + 1) * BT, T) - 1
        b_g_last = tl.load(g + (bos * HV + last_idx * HV + i_h).to(tl.int64)).to(tl.float32)
        p_g = g + (bos * HV + i_h).to(tl.int64) + o_t * HV
        b_g = tl.load(p_g, mask=m_t, other=0.0).to(tl.float32)
        b_v = b_v * tl.where(m_t, exp2(b_g_last - b_g), 0)[:, None]
        b_g_last = exp2(b_g_last)
        b_h1 *= b_g_last
        if K > 64:
            b_h2 *= b_g_last

        b_v = b_v.to(k.dtype.element_ty)

        p_k = k + o_k1[:, None] + o_t[None, :] * (H * K)
        b_k = tl.load(p_k, mask=m_k1[:, None] & m_t[None, :], other=0.0)
        b_h1 += tl.dot(b_k, b_v)
        if K > 64:
            p_k = k + o_k2[:, None] + o_t[None, :] * (H * K)
            b_k = tl.load(p_k, mask=m_k2[:, None] & m_t[None, :], other=0.0)
            b_h2 += tl.dot(b_k, b_v)

    p_ht = ht + o_k1[:, None] * V + o_v[None, :]
    m_ht = m_k1[:, None] & m_v[None, :]
    tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), mask=m_ht)
    if K > 64:
        p_ht = ht + o_k2[:, None] * V + o_v[None, :]
        m_ht = m_k2[:, None] & m_v[None, :]
        tl.store(p_ht, b_h2.to(p_ht.dtype.element_ty), mask=m_ht)


@triton.autotune(
    configs=[
        triton.Config({"BV": BV}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4]
        for num_stages in ([2, 3, 4] if check_shared_mem("ampere") else [1])
        for BV in ([32, 64] if check_shared_mem("ada") else [32])
    ],
    key=["H", "HV", "K", "V", "BT", "BV"],
)
@triton.jit(do_not_specialize=["T"])
def chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64(
    q,
    k,
    w,
    g,
    dht,
    dh0,
    do,
    dh,
    dv,
    dv2,
    cu_seqlens,
    chunk_offsets,
    segments,
    child_offsets,
    children,
    scale,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = tl.load(segments + i_nh // HV), i_nh % HV
    i_nh = i_n * HV + i_h
    bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    T = eos - bos
    NT = tl.cdiv(T, BT)
    boh = tl.load(chunk_offsets + i_n).to(tl.int32)

    b_dh1 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 64:
        b_dh2 = tl.zeros([64, BV], dtype=tl.float32)

    # calculate offset
    q += (bos * H + i_h // (HV // H)).to(tl.int64) * K
    k += (bos * H + i_h // (HV // H)).to(tl.int64) * K
    w += (bos * HV + i_h).to(tl.int64) * K
    do += (bos * HV + i_h).to(tl.int64) * V
    dv += (bos * HV + i_h).to(tl.int64) * V
    dv2 += (bos * HV + i_h).to(tl.int64) * V
    dh += (boh * HV + i_h).to(tl.int64) * K * V

    dh0 += i_nh * K * V
    child_grad = dht

    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V
    o_k1 = tl.arange(0, 64)
    m_k1 = o_k1 < K
    o_k2 = 64 + o_k1
    m_k2 = o_k2 < K
    # Tree boundary: sum child input-state gradients before scanning this segment backward.
    first = tl.load(child_offsets + i_n)
    last = tl.load(child_offsets + i_n + 1)
    for edge in range(first, last):
        child = tl.load(children + edge)
        dht = child_grad + (child * HV + i_h) * K * V
        p_dht1 = dht + o_k1[:, None] * V + o_v[None, :]
        m_dht1 = m_k1[:, None] & m_v[None, :]
        b_dh1 += tl.load(p_dht1, mask=m_dht1, other=0.0)
        if K > 64:
            p_dht2 = dht + o_k2[:, None] * V + o_v[None, :]
            m_dht2 = m_k2[:, None] & m_v[None, :]
            b_dh2 += tl.load(p_dht2, mask=m_dht2, other=0.0)

    for i_t in range(NT - 1, -1, -1):
        i_t_int64 = i_t.to(tl.int64)
        o_t = i_t * BT + tl.arange(0, BT)
        m_t = o_t < T
        p_dh1 = dh + i_t_int64 * HV * K * V + o_k1[:, None] * V + o_v[None, :]
        m_dh1 = m_k1[:, None] & m_v[None, :]
        tl.store(p_dh1, b_dh1.to(p_dh1.dtype.element_ty), mask=m_dh1)
        if K > 64:
            p_dh2 = dh + i_t_int64 * HV * K * V + o_k2[:, None] * V + o_v[None, :]
            m_dh2 = m_k2[:, None] & m_v[None, :]
            tl.store(p_dh2, b_dh2.to(p_dh2.dtype.element_ty), mask=m_dh2)

        last_idx = min((i_t + 1) * BT, T) - 1
        bg_last = tl.load(g + (bos + last_idx) * HV + i_h).to(tl.float32)
        p_g = g + bos * HV + i_h + o_t * HV
        b_g = tl.load(p_g, mask=m_t, other=0.0).to(tl.float32)
        bg_last_exp = exp2(bg_last)
        b_g_exp = exp2(b_g)
        p_dv = dv + o_t[:, None] * (HV * V) + o_v[None, :]
        p_dv2 = dv2 + o_t[:, None] * (HV * V) + o_v[None, :]
        p_do = do + o_t[:, None] * (HV * V) + o_v[None, :]

        b_do = tl.load(p_do, mask=m_t[:, None] & m_v[None, :], other=0.0)

        # Update dv
        p_k = k + o_t[:, None] * (H * K) + o_k1[None, :]
        b_k = tl.load(p_k, mask=m_t[:, None] & m_k1[None, :], other=0.0)
        b_dv = tl.dot(b_k, b_dh1.to(b_k.dtype))

        if K > 64:
            p_k = k + o_t[:, None] * (H * K) + o_k2[None, :]
            b_k = tl.load(p_k, mask=m_t[:, None] & m_k2[None, :], other=0.0)
            b_dv += tl.dot(b_k, b_dh2.to(b_k.dtype))

        b_dv *= tl.where(m_t, exp2(bg_last - b_g), 0)[:, None]
        b_dv += tl.load(p_dv, mask=m_t[:, None] & m_v[None, :], other=0.0)

        tl.store(p_dv2, b_dv.to(p_dv.dtype.element_ty), mask=m_t[:, None] & m_v[None, :])
        # Update dh
        p_w = w + o_k1[:, None] + o_t[None, :] * (HV * K)
        p_q = q + o_k1[:, None] + o_t[None, :] * (H * K)
        b_w = tl.load(p_w, mask=m_k1[:, None] & m_t[None, :], other=0.0)
        b_q = tl.load(p_q, mask=m_k1[:, None] & m_t[None, :], other=0.0)
        b_dh1 *= bg_last_exp
        b_q = b_q * b_g_exp[None, :]
        b_dh1 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype))
        if K > 64:
            p_q = q + o_k2[:, None] + o_t[None, :] * (H * K)
            p_w = w + o_k2[:, None] + o_t[None, :] * (HV * K)
            b_q = tl.load(p_q, mask=m_k2[:, None] & m_t[None, :], other=0.0)
            b_w = tl.load(p_w, mask=m_k2[:, None] & m_t[None, :], other=0.0)
            b_dh2 *= bg_last_exp
            b_q = b_q * b_g_exp[None, :]
            b_dh2 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype))

    p_dh0 = dh0 + o_k1[:, None] * V + o_v[None, :]
    m_dh0 = m_k1[:, None] & m_v[None, :]
    tl.store(p_dh0, b_dh1.to(p_dh0.dtype.element_ty), mask=m_dh0)
    if K > 64:
        p_dh1 = dh0 + o_k2[:, None] * V + o_v[None, :]
        m_dh1 = m_k2[:, None] & m_v[None, :]
        tl.store(p_dh1, b_dh2.to(p_dh1.dtype.element_ty), mask=m_dh1)


# --- Fused 64-token Gram matrix and triangular solve ---


SOLVE_TRIL_DOT_PRECISION = tl.constexpr("tf32")


@triton.autotune(
    configs=[triton.Config({"BK": BK}, num_warps=num_warps) for BK in [32, 64] for num_warps in [1, 2, 4]],
    key=["H", "HV", "K", "BC"],
)
@triton.jit(do_not_specialize=["T"])
def chunk_gated_delta_rule_fwd_kkt_solve_kernel(
    k,
    g,
    beta,
    A,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
):
    """
    Fused kernel: compute beta * K @ K^T (lower triangular) + solve_tril (I+A)^{-1} in one pass.

    This kernel fuses chunk_scaled_dot_kkt_fwd and solve_tril into a single kernel, avoiding the HBM round-trip for the
    intermediate A matrix.

    Steps:
    1. Compute all 10 lower-triangular [BC, BC] blocks of beta * K @ K^T in registers
    2. Apply gate and beta scaling
    3. Forward substitution on diagonal blocks
    4. Block merge to get full (I+A)^{-1}
    5. Write result to A (output)
    """
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1)
    i_h = i_bh % HV

    i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
    bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    T = eos - bos

    if i_t * BT >= T:
        return

    i_tc0 = i_t * BT
    i_tc1 = i_t * BT + BC
    i_tc2 = i_t * BT + 2 * BC
    i_tc3 = i_t * BT + 3 * BC

    k += (bos * H + i_h // (HV // H)) * K
    A += (bos * HV + i_h) * BT

    o_i = tl.arange(0, BC)
    m_tc0 = (i_tc0 + o_i) < T
    m_tc1 = (i_tc1 + o_i) < T
    m_tc2 = (i_tc2 + o_i) < T
    m_tc3 = (i_tc3 + o_i) < T

    # load beta for each sub-chunk
    p_b0 = beta + bos * HV + i_h + (i_tc0 + o_i) * HV
    p_b1 = beta + bos * HV + i_h + (i_tc1 + o_i) * HV
    p_b2 = beta + bos * HV + i_h + (i_tc2 + o_i) * HV
    p_b3 = beta + bos * HV + i_h + (i_tc3 + o_i) * HV
    b_b0 = tl.load(p_b0, mask=m_tc0, other=0.0).to(tl.float32)
    b_b1 = tl.load(p_b1, mask=m_tc1, other=0.0).to(tl.float32)
    b_b2 = tl.load(p_b2, mask=m_tc2, other=0.0).to(tl.float32)
    b_b3 = tl.load(p_b3, mask=m_tc3, other=0.0).to(tl.float32)

    # load gate if used
    p_g0 = g + bos * HV + i_h + (i_tc0 + o_i) * HV
    p_g1 = g + bos * HV + i_h + (i_tc1 + o_i) * HV
    p_g2 = g + bos * HV + i_h + (i_tc2 + o_i) * HV
    p_g3 = g + bos * HV + i_h + (i_tc3 + o_i) * HV

    b_g0 = tl.load(p_g0, mask=m_tc0, other=0.0).to(tl.float32)
    b_g1 = tl.load(p_g1, mask=m_tc1, other=0.0).to(tl.float32)
    b_g2 = tl.load(p_g2, mask=m_tc2, other=0.0).to(tl.float32)
    b_g3 = tl.load(p_g3, mask=m_tc3, other=0.0).to(tl.float32)

    ############################################################################
    # Step 1: compute all 10 lower-triangular [BC, BC] blocks of K @ K^T
    ############################################################################

    # 4 diagonal blocks
    b_A00 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A11 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A22 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A33 = tl.zeros([BC, BC], dtype=tl.float32)

    # 6 off-diagonal blocks
    b_A10 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A20 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A21 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A30 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A31 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A32 = tl.zeros([BC, BC], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        p_k0 = k + (i_tc0 + o_i)[:, None] * (H * K) + o_k[None, :]
        b_k0 = tl.load(p_k0, mask=m_tc0[:, None] & (o_k[None, :] < K), other=0.0)
        # diagonal block 0
        b_A00 += tl.dot(b_k0, tl.trans(b_k0))

        if i_tc1 < T:
            p_k1 = k + (i_tc1 + o_i)[:, None] * (H * K) + o_k[None, :]
            b_k1 = tl.load(p_k1, mask=m_tc1[:, None] & (o_k[None, :] < K), other=0.0)
            # diagonal block 1
            b_A11 += tl.dot(b_k1, tl.trans(b_k1))
            # off-diagonal (1,0)
            b_A10 += tl.dot(b_k1, tl.trans(b_k0))

            if i_tc2 < T:
                p_k2 = k + (i_tc2 + o_i)[:, None] * (H * K) + o_k[None, :]
                b_k2 = tl.load(p_k2, mask=m_tc2[:, None] & (o_k[None, :] < K), other=0.0)
                # diagonal block 2
                b_A22 += tl.dot(b_k2, tl.trans(b_k2))
                # off-diagonal (2,0), (2,1)
                b_A20 += tl.dot(b_k2, tl.trans(b_k0))
                b_A21 += tl.dot(b_k2, tl.trans(b_k1))

                if i_tc3 < T:
                    p_k3 = k + (i_tc3 + o_i)[:, None] * (H * K) + o_k[None, :]
                    b_k3 = tl.load(p_k3, mask=m_tc3[:, None] & (o_k[None, :] < K), other=0.0)
                    # diagonal block 3
                    b_A33 += tl.dot(b_k3, tl.trans(b_k3))
                    # off-diagonal (3,0), (3,1), (3,2)
                    b_A30 += tl.dot(b_k3, tl.trans(b_k0))
                    b_A31 += tl.dot(b_k3, tl.trans(b_k1))
                    b_A32 += tl.dot(b_k3, tl.trans(b_k2))

    ############################################################################
    # Step 2: apply gate and beta scaling
    ############################################################################

    # apply gate, beta scaling, and masking
    # m_d: strictly lower triangular mask for diagonal blocks
    # m_tc: boundary mask to prevent NaN from 0 * inf (IEEE 754) when
    #   out-of-bounds g loads as 0 via boundary_check and exp2(0 - g_inbounds) overflows
    m_d = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]

    b_A00 *= tl.where(m_d & m_tc0[:, None] & m_tc0[None, :], exp2(b_g0[:, None] - b_g0[None, :]), 0.0)
    b_A11 *= tl.where(m_d & m_tc1[:, None] & m_tc1[None, :], exp2(b_g1[:, None] - b_g1[None, :]), 0.0)
    b_A22 *= tl.where(m_d & m_tc2[:, None] & m_tc2[None, :], exp2(b_g2[:, None] - b_g2[None, :]), 0.0)
    b_A33 *= tl.where(m_d & m_tc3[:, None] & m_tc3[None, :], exp2(b_g3[:, None] - b_g3[None, :]), 0.0)

    b_A10 *= tl.where(m_tc1[:, None] & m_tc0[None, :], exp2(b_g1[:, None] - b_g0[None, :]), 0.0)
    b_A20 *= tl.where(m_tc2[:, None] & m_tc0[None, :], exp2(b_g2[:, None] - b_g0[None, :]), 0.0)
    b_A21 *= tl.where(m_tc2[:, None] & m_tc1[None, :], exp2(b_g2[:, None] - b_g1[None, :]), 0.0)
    b_A30 *= tl.where(m_tc3[:, None] & m_tc0[None, :], exp2(b_g3[:, None] - b_g0[None, :]), 0.0)
    b_A31 *= tl.where(m_tc3[:, None] & m_tc1[None, :], exp2(b_g3[:, None] - b_g1[None, :]), 0.0)
    b_A32 *= tl.where(m_tc3[:, None] & m_tc2[None, :], exp2(b_g3[:, None] - b_g2[None, :]), 0.0)

    # diagonal blocks: scaled by beta
    b_A00 = b_A00 * b_b0[:, None]
    b_A11 = b_A11 * b_b1[:, None]
    b_A22 = b_A22 * b_b2[:, None]
    b_A33 = b_A33 * b_b3[:, None]

    # off-diagonal blocks: full block, scaled by beta
    b_A10 = b_A10 * b_b1[:, None]
    b_A20 = b_A20 * b_b2[:, None]
    b_A21 = b_A21 * b_b2[:, None]
    b_A30 = b_A30 * b_b3[:, None]
    b_A31 = b_A31 * b_b3[:, None]
    b_A32 = b_A32 * b_b3[:, None]

    ############################################################################
    # Step 3: forward substitution on diagonal blocks -> (I + A_diag)^{-1}
    #
    # Same algorithm as solve_tril, but rows are extracted from in-register
    # [BC, BC] tensor via tl.sum(tl.where(mask, tensor, 0), 0) instead of
    # tl.load from HBM.
    ############################################################################

    b_Ai00 = -b_A00
    b_Ai11 = -b_A11
    b_Ai22 = -b_A22
    b_Ai33 = -b_A33

    for i in range(2, min(BC, T - i_tc0)):
        b_a00 = tl.sum(tl.where((o_i == i)[:, None], -b_A00, 0.0), 0)
        b_a00 = tl.where(o_i < i, b_a00, 0.0)
        b_a00 = b_a00 + tl.sum(b_a00[:, None] * b_Ai00, 0)
        b_Ai00 = tl.where((o_i == i)[:, None], b_a00, b_Ai00)
    for i in range(2, min(BC, T - i_tc1)):
        b_a11 = tl.sum(tl.where((o_i == i)[:, None], -b_A11, 0.0), 0)
        b_a11 = tl.where(o_i < i, b_a11, 0.0)
        b_a11 = b_a11 + tl.sum(b_a11[:, None] * b_Ai11, 0)
        b_Ai11 = tl.where((o_i == i)[:, None], b_a11, b_Ai11)
    for i in range(2, min(BC, T - i_tc2)):
        b_a22 = tl.sum(tl.where((o_i == i)[:, None], -b_A22, 0.0), 0)
        b_a22 = tl.where(o_i < i, b_a22, 0.0)
        b_a22 = b_a22 + tl.sum(b_a22[:, None] * b_Ai22, 0)
        b_Ai22 = tl.where((o_i == i)[:, None], b_a22, b_Ai22)
    for i in range(2, min(BC, T - i_tc3)):
        b_a33 = tl.sum(tl.where((o_i == i)[:, None], -b_A33, 0.0), 0)
        b_a33 = tl.where(o_i < i, b_a33, 0.0)
        b_a33 = b_a33 + tl.sum(b_a33[:, None] * b_Ai33, 0)
        b_Ai33 = tl.where((o_i == i)[:, None], b_a33, b_Ai33)

    b_Ai00 += m_I
    b_Ai11 += m_I
    b_Ai22 += m_I
    b_Ai33 += m_I

    ############################################################################
    # Step 4: block merge -> full (I + A)^{-1}
    ############################################################################

    b_Ai10 = -tl.dot(
        tl.dot(b_Ai11, b_A10, input_precision=SOLVE_TRIL_DOT_PRECISION),
        b_Ai00,
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )
    b_Ai21 = -tl.dot(
        tl.dot(b_Ai22, b_A21, input_precision=SOLVE_TRIL_DOT_PRECISION),
        b_Ai11,
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )
    b_Ai32 = -tl.dot(
        tl.dot(b_Ai33, b_A32, input_precision=SOLVE_TRIL_DOT_PRECISION),
        b_Ai22,
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )

    b_Ai20 = -tl.dot(
        b_Ai22,
        tl.dot(b_A20, b_Ai00, input_precision=SOLVE_TRIL_DOT_PRECISION)
        + tl.dot(b_A21, b_Ai10, input_precision=SOLVE_TRIL_DOT_PRECISION),
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )
    b_Ai31 = -tl.dot(
        b_Ai33,
        tl.dot(b_A31, b_Ai11, input_precision=SOLVE_TRIL_DOT_PRECISION)
        + tl.dot(b_A32, b_Ai21, input_precision=SOLVE_TRIL_DOT_PRECISION),
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )
    b_Ai30 = -tl.dot(
        b_Ai33,
        tl.dot(b_A30, b_Ai00, input_precision=SOLVE_TRIL_DOT_PRECISION)
        + tl.dot(b_A31, b_Ai10, input_precision=SOLVE_TRIL_DOT_PRECISION)
        + tl.dot(b_A32, b_Ai20, input_precision=SOLVE_TRIL_DOT_PRECISION),
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )

    ############################################################################
    # Step 5: store full (I + A)^{-1} to output A
    ############################################################################

    p_A00 = A + (i_tc0 + o_i)[:, None] * (HV * BT) + o_i[None, :]
    p_A10 = A + (i_tc1 + o_i)[:, None] * (HV * BT) + o_i[None, :]
    p_A11 = A + (i_tc1 + o_i)[:, None] * (HV * BT) + (BC + o_i)[None, :]
    p_A20 = A + (i_tc2 + o_i)[:, None] * (HV * BT) + o_i[None, :]
    p_A21 = A + (i_tc2 + o_i)[:, None] * (HV * BT) + (BC + o_i)[None, :]
    p_A22 = A + (i_tc2 + o_i)[:, None] * (HV * BT) + (2 * BC + o_i)[None, :]
    p_A30 = A + (i_tc3 + o_i)[:, None] * (HV * BT) + o_i[None, :]
    p_A31 = A + (i_tc3 + o_i)[:, None] * (HV * BT) + (BC + o_i)[None, :]
    p_A32 = A + (i_tc3 + o_i)[:, None] * (HV * BT) + (2 * BC + o_i)[None, :]
    p_A33 = A + (i_tc3 + o_i)[:, None] * (HV * BT) + (3 * BC + o_i)[None, :]

    m_A0 = m_tc0[:, None] & (o_i[None, :] < BT)
    m_A1 = m_tc1[:, None] & (o_i[None, :] < BT)
    m_A2 = m_tc2[:, None] & (o_i[None, :] < BT)
    m_A3 = m_tc3[:, None] & (o_i[None, :] < BT)
    m_A11 = m_tc1[:, None] & ((BC + o_i)[None, :] < BT)
    m_A21 = m_tc2[:, None] & ((BC + o_i)[None, :] < BT)
    m_A22 = m_tc2[:, None] & ((2 * BC + o_i)[None, :] < BT)
    m_A31 = m_tc3[:, None] & ((BC + o_i)[None, :] < BT)
    m_A32 = m_tc3[:, None] & ((2 * BC + o_i)[None, :] < BT)
    m_A33 = m_tc3[:, None] & ((3 * BC + o_i)[None, :] < BT)

    tl.store(p_A00, b_Ai00.to(A.dtype.element_ty), mask=m_A0)
    tl.store(p_A10, b_Ai10.to(A.dtype.element_ty), mask=m_A1)
    tl.store(p_A11, b_Ai11.to(A.dtype.element_ty), mask=m_A11)
    tl.store(p_A20, b_Ai20.to(A.dtype.element_ty), mask=m_A2)
    tl.store(p_A21, b_Ai21.to(A.dtype.element_ty), mask=m_A21)
    tl.store(p_A22, b_Ai22.to(A.dtype.element_ty), mask=m_A22)
    tl.store(p_A30, b_Ai30.to(A.dtype.element_ty), mask=m_A3)
    tl.store(p_A31, b_Ai31.to(A.dtype.element_ty), mask=m_A31)
    tl.store(p_A32, b_Ai32.to(A.dtype.element_ty), mask=m_A32)
    tl.store(p_A33, b_Ai33.to(A.dtype.element_ty), mask=m_A33)


# --- Chunk-local WY representation and its gradients ---


PREPARE_WY_REPR_BWD_NUM_WARPS = [2] if IS_NVIDIA_BLACKWELL else [2, 4]
PREPARE_WY_REPR_BWD_NUM_STAGES = [4] if IS_NVIDIA_BLACKWELL else [2, 3, 4]


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["H", "HV", "K", "V", "BT", "BK", "BV"],
)
@triton.jit(do_not_specialize=["T"])
def recompute_w_u_fwd_kernel(
    k,
    v,
    beta,
    w,
    u,
    A,
    g,
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
):
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_h = i_bh % HV
    i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
    bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    T = eos - bos
    o_t = i_t * BT + tl.arange(0, BT)
    o_A = tl.arange(0, BT)
    m_t = o_t < T
    m_A = m_t[:, None] & (o_A[None, :] < BT)
    p_b = beta + bos * HV + i_h + o_t * HV
    b_b = tl.load(p_b, mask=m_t, other=0.0)

    p_A = A + (bos * HV + i_h) * BT + o_t[:, None] * (HV * BT) + o_A[None, :]
    b_A = tl.load(p_A, mask=m_A, other=0.0)

    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = m_t[:, None] & (o_v[None, :] < V)
        p_v = v + (bos * HV + i_h) * V + o_t[:, None] * (HV * V) + o_v[None, :]
        p_u = u + (bos * HV + i_h) * V + o_t[:, None] * (HV * V) + o_v[None, :]
        b_v = tl.load(p_v, mask=m_v, other=0.0)
        b_vb = (b_v * b_b[:, None]).to(b_v.dtype)
        b_u = tl.dot(b_A, b_vb, allow_tf32=False)
        tl.store(p_u, b_u.to(p_u.dtype.element_ty), mask=m_v)

    p_g = g + (bos * HV + i_h) + o_t * HV
    b_g = exp2(tl.load(p_g, mask=m_t, other=0.0))

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = m_t[:, None] & (o_k[None, :] < K)
        p_k = k + (bos * H + i_h // (HV // H)) * K + o_t[:, None] * (H * K) + o_k[None, :]
        p_w = w + (bos * HV + i_h) * K + o_t[:, None] * (HV * K) + o_k[None, :]
        b_k = tl.load(p_k, mask=m_k, other=0.0)
        b_kb = b_k * b_b[:, None]
        b_kb *= b_g[:, None]
        b_w = tl.dot(b_A, b_kb.to(b_k.dtype))
        tl.store(p_w, b_w.to(p_w.dtype.element_ty), mask=m_k)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in PREPARE_WY_REPR_BWD_NUM_WARPS
        for num_stages in PREPARE_WY_REPR_BWD_NUM_STAGES
    ],
    key=["H", "HV", "K", "V", "BT", "BK", "BV"],
)
@triton.jit(do_not_specialize=["T"])
def prepare_wy_repr_bwd_kernel(
    k,
    v,
    beta,
    g,
    A,
    dw,
    du,
    dk,
    dv,
    db,
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
):
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_h = i_bh % HV
    i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
    bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    T = eos - bos

    o_t = i_t * BT + tl.arange(0, BT)
    o_A = tl.arange(0, BT)
    m_t = o_t < T
    m_AT = (o_A[:, None] < BT) & m_t[None, :]
    p_b = beta + (bos * HV + i_h) + o_t * HV
    p_db = db + (bos * HV + i_h) + o_t * HV
    p_A = A + (bos * HV + i_h) * BT + o_A[:, None] + o_t[None, :] * (HV * BT)

    b_b = tl.load(p_b, mask=m_t, other=0.0)
    b_db = tl.zeros([BT], dtype=tl.float32)
    b_A = tl.load(p_A, mask=m_AT, other=0.0)
    b_dA = tl.zeros([BT, BT], dtype=tl.float32)

    p_g = g + (bos * HV + i_h) + o_t * HV
    b_g = tl.load(p_g, mask=m_t, other=0.0)
    b_g_exp = exp2(b_g)
    b_dg = tl.zeros([BT], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = m_t[:, None] & (o_k[None, :] < K)
        p_k = k + (bos * H + i_h // (HV // H)) * K + o_t[:, None] * (H * K) + o_k[None, :]
        p_dk = dk + (bos * HV + i_h) * K + o_t[:, None] * (HV * K) + o_k[None, :]
        p_dw = dw + (bos * HV + i_h) * K + o_t[:, None] * (HV * K) + o_k[None, :]
        # [BT, BK]
        b_k = tl.load(p_k, mask=m_k, other=0.0)
        b_kbg = b_k * (b_b * b_g_exp)[:, None]
        b_dw = tl.load(p_dw, mask=m_k, other=0.0)

        b_dA += tl.dot(b_dw, tl.trans(b_kbg).to(b_dw.dtype))
        b_dkbg = tl.dot(b_A, b_dw)
        b_dk = b_dkbg * (b_g_exp * b_b)[:, None]
        b_db += tl.sum(b_dkbg * b_k * b_g_exp[:, None], 1)
        b_dg += tl.sum(b_dkbg * b_kbg, 1)
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), mask=m_k)

    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = m_t[:, None] & (o_v[None, :] < V)
        p_v = v + (bos * HV + i_h) * V + o_t[:, None] * (HV * V) + o_v[None, :]
        p_dv = dv + (bos * HV + i_h) * V + o_t[:, None] * (HV * V) + o_v[None, :]
        p_du = du + (bos * HV + i_h) * V + o_t[:, None] * (HV * V) + o_v[None, :]
        b_v = tl.load(p_v, mask=m_v, other=0.0)
        b_vb = (b_v * b_b[:, None]).to(b_v.dtype)
        b_du = tl.load(p_du, mask=m_v, other=0.0)
        b_dA += tl.dot(b_du, tl.trans(b_vb))
        b_dvb = tl.dot(b_A, b_du)
        b_dv = b_dvb * b_b[:, None]
        b_db += tl.sum(b_dvb * b_v, 1)
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), mask=m_v)

    m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)
    b_dA = tl.where(m_A, b_dA, 0)
    b_dA = tl.dot(b_dA.to(b_A.dtype), b_A)
    b_dA = tl.dot(b_A, b_dA.to(b_A.dtype))

    b_dA *= exp2(b_g[:, None] - b_g[None, :])

    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    b_dA = tl.where(m_A, -b_dA, 0).to(k.dtype.element_ty)

    tl.debug_barrier()
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = m_t[:, None] & (o_k[None, :] < K)
        p_k = k + (bos * H + i_h // (HV // H)) * K + o_t[:, None] * (H * K) + o_k[None, :]
        p_dk = dk + (bos * HV + i_h) * K + o_t[:, None] * (HV * K) + o_k[None, :]
        b_k = tl.load(p_k, mask=m_k, other=0.0)
        b_kt = tl.trans(b_k)
        b_kb = b_k * b_b[:, None]

        b_A += tl.dot(b_k, b_kt)
        b_dkb = tl.dot(b_dA, b_k)
        b_db += tl.sum(b_dkb * b_k, 1)
        b_dk = b_dkb * b_b[:, None] + tl.trans(tl.dot(tl.trans(b_kb).to(b_dA.dtype), b_dA))
        b_dk += tl.load(p_dk, mask=m_k, other=0.0)

        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), mask=m_k)
    tl.store(p_db, b_db.to(p_db.dtype.element_ty), mask=m_t)

    b_A *= b_b[:, None]
    b_AdA = b_dA * b_A
    p_dg = dg + (bos * HV + i_h) + o_t * HV
    b_dg += tl.sum(b_AdA, axis=1) - tl.sum(b_AdA, axis=0)
    tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), mask=m_t)


def recompute_w_u_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    g: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, V, HV = *k.shape, v.shape[-1], v.shape[2]
    BT = A.shape[-1]
    BK = 64
    BV = 64

    NT = len(chunk_indices)

    w = k.new_empty(B, T, HV, K)
    u = torch.empty_like(v)
    recompute_w_u_fwd_kernel[(NT, B * HV)](
        k=k,
        v=v,
        beta=beta,
        w=w,
        u=u,
        A=A,
        g=g,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
    )
    return w, u


def prepare_wy_repr_bwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    dw: torch.Tensor,
    du: torch.Tensor,
    g: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, K, V, HV = *k.shape, v.shape[-1], v.shape[2]
    BT = A.shape[-1]
    NT = len(chunk_indices)
    CONST_TILING = 64 if check_shared_mem() else 32
    BK = min(max(triton.next_power_of_2(K), 16), CONST_TILING)
    BV = min(max(triton.next_power_of_2(V), 16), CONST_TILING)

    dk = k.new_empty(B, T, HV, K)
    dv = torch.empty_like(v)
    dg = torch.empty_like(g)
    db = torch.empty_like(beta)
    prepare_wy_repr_bwd_kernel[(NT, B * HV)](
        k=k,
        v=v,
        beta=beta,
        g=g,
        A=A,
        dw=dw,
        du=du,
        dk=dk,
        dv=dv,
        db=db,
        dg=dg,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
    )
    if H != HV:
        dk = dk.view(B, T, H, HV // H, K).sum(3)
    return dk, dv, db, dg


# --- Chunk outputs and local gradients ---


NUM_WARPS = [2, 4] if IS_NVIDIA_HOPPER else [2, 4, 8]


@triton.autotune(
    configs=[
        triton.Config({"BK": 128, "BV": 128}, num_warps=8, num_stages=3),
        triton.Config({"BK": 64, "BV": 64}, num_warps=4, num_stages=3),
        triton.Config({"BK": 32, "BV": 32}, num_warps=2, num_stages=3),
    ],
    key=["H", "HV", "K", "V", "BT"],
)
@triton.jit(do_not_specialize=["T"])
def chunk_fwd_kernel_o(
    q,
    k,
    v,
    h,
    g,
    o,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    i_h = i_bh % HV

    i_tg = i_t
    i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
    bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    T = eos - bos

    # offset calculation
    q += (bos * H + i_h // (HV // H)) * K
    k += (bos * H + i_h // (HV // H)) * K
    v += (bos * HV + i_h) * V
    o += (bos * HV + i_h) * V
    h += (i_tg * HV + i_h).to(tl.int64) * K * V

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_A = tl.zeros([BT, BT], dtype=tl.float32)

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    o_v = i_v * BV + tl.arange(0, BV)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        p_q = q + o_t[:, None] * (H * K) + o_k[None, :]
        p_k = k + o_k[:, None] + o_t[None, :] * (H * K)
        p_h = h + o_k[:, None] * V + o_v[None, :]
        m_h = m_k[:, None] & (o_v[None, :] < V)
        # [BT, BK]
        b_q = tl.load(p_q, mask=m_t[:, None] & m_k[None, :], other=0.0)
        # [BK, BT]
        b_k = tl.load(p_k, mask=m_k[:, None] & m_t[None, :], other=0.0)
        b_h = tl.load(p_h, mask=m_h, other=0.0)

        # [BT, BK] @ [BK, BV] -> [BT, BV]
        b_o += tl.dot(b_q, b_h)
        # [BT, BK] @ [BK, BT] -> [BT, BT]
        b_A += tl.dot(b_q, b_k)

    g += bos * HV + i_h
    p_g = g + o_t * HV
    b_g = tl.load(p_g, mask=m_t, other=0.0)
    b_o = b_o * exp2(b_g)[:, None]
    b_A = b_A * exp2(b_g[:, None] - b_g[None, :])
    m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A, b_A, 0)

    p_v = v + o_t[:, None] * (HV * V) + o_v[None, :]
    p_o = o + o_t[:, None] * (HV * V) + o_v[None, :]

    b_v = tl.load(p_v, mask=m_t[:, None] & (o_v < V)[None, :], other=0.0)
    # to fix mma -> mma layout conversion
    # already solved by triton v3.2 or higher
    b_o = b_o * scale + tl.dot(b_A.to(b_v.dtype), b_v) * scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=m_t[:, None] & (o_v < V)[None, :])


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in NUM_WARPS
        for num_stages in [2, 3, 4]
    ],
    key=["H", "HV", "K", "V", "BT", "BK", "BV"],
)
@triton.jit(do_not_specialize=["T"])
def chunk_bwd_kernel_dqkwg(
    q,
    k,
    v,
    g,
    h,
    do,
    dh,
    dq,
    dk,
    dw,
    dv,
    dg,
    cu_seqlens,
    chunk_indices,
    scale,
    B: tl.constexpr,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    i_h = i_bh % HV

    all = B * T
    i_tg = i_t
    i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
    bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    T = eos - bos

    # offset calculation
    v += (bos * HV + i_h) * V
    do += (bos * HV + i_h) * V
    h += (i_tg * HV + i_h).to(tl.int64) * K * V
    dh += (i_tg * HV + i_h).to(tl.int64) * K * V
    q += (bos * H + i_h // (HV // H)) * K
    k += (bos * H + i_h // (HV // H)) * K
    dq += (bos * HV + i_h) * K
    dk += (bos * HV + i_h) * K

    # for delta rule only
    dw += (bos * HV + i_h) * K
    dv += (bos * HV + i_h) * V

    dg += i_k * all * HV
    b_dg_last = tl.zeros([1], dtype=tl.float32)
    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_ds = tl.zeros([BT, BT], dtype=tl.float32)
    b_dw = tl.zeros([BT, BK], dtype=tl.float32)

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    o_k = i_k * BK + tl.arange(0, BK)
    m_k = o_k < K
    m_qk = m_t[:, None] & m_k[None, :]
    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = o_v < V
        m_hv = m_t[:, None] & m_v[None, :]
        p_v = v + o_t[:, None] * (HV * V) + o_v[None, :]
        p_do = do + o_t[:, None] * (HV * V) + o_v[None, :]
        p_h = h + o_v[:, None] + o_k[None, :] * V
        p_dh = dh + o_v[:, None] + o_k[None, :] * V
        m_h = m_v[:, None] & m_k[None, :]
        # [BT, BV]
        b_v = tl.load(p_v, mask=m_hv, other=0.0)
        b_do = tl.load(p_do, mask=m_hv, other=0.0)
        # [BV, BK]
        b_h = tl.load(p_h, mask=m_h, other=0.0)
        b_dh = tl.load(p_dh, mask=m_h, other=0.0)
        b_dg_last += tl.sum(b_h * b_dh)
        # [BT, BV] @ [BV, BT] -> [BT, BT]
        b_ds += tl.dot(b_do, tl.trans(b_v))
        # [BT, BV] @ [BV, BK] -> [BT, BK]
        b_dq += tl.dot(b_do, b_h.to(b_do.dtype))
        # [BT, BV] @ [BV, BK] -> [BT, BK]
        b_dk += tl.dot(b_v, b_dh.to(b_v.dtype))
        p_dv = dv + o_t[:, None] * (HV * V) + o_v[None, :]
        b_dv = tl.load(p_dv, mask=m_hv, other=0.0)
        b_dw += tl.dot(b_dv.to(b_v.dtype), b_h.to(b_v.dtype))

    p_dw = dw + o_t[:, None] * (HV * K) + o_k[None, :]
    tl.store(p_dw, -b_dw.to(p_dw.dtype.element_ty), mask=m_qk)

    tl.debug_barrier()
    p_q = q + o_t[:, None] * (H * K) + o_k[None, :]
    p_k = k + o_t[:, None] * (H * K) + o_k[None, :]
    b_q = tl.load(p_q, mask=m_qk, other=0.0)
    b_k = tl.load(p_k, mask=m_qk, other=0.0)

    p_dq = dq + o_t[:, None] * (HV * K) + o_k[None, :]
    p_dk = dk + o_t[:, None] * (HV * K) + o_k[None, :]

    m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
    g += bos * HV + i_h
    dg += bos * HV + i_h
    p_g = g + o_t * HV
    b_g = tl.load(p_g, mask=m_t, other=0.0)
    b_g_last = tl.load(g + (min(i_t * BT + BT, T) - 1) * HV)
    b_dg_last *= exp2(b_g_last)
    b_dq = b_dq * exp2(b_g)[:, None] * scale
    b_dk = b_dk * tl.where(m_t, exp2(-b_g + b_g_last), 0)[:, None]
    b_dg_last += tl.sum(b_dk * b_k)

    b_ds = tl.where(m_A, b_ds * exp2(b_g[:, None] - b_g[None, :]), 0) * scale
    b_ds = b_ds.to(b_k.dtype)
    # [BT, BK]
    b_dq += tl.dot(b_ds, b_k)
    b_dk += tl.dot(tl.trans(b_ds), b_q)

    b_dg = tl.sum(b_dq * b_q, axis=1) - tl.sum(b_dk * b_k, axis=1)

    p_dg = dg + o_t * HV
    # (SY 09/21) revcumsum in a separate kernel due to strange triton compiler issue
    b_dg = tl.where(o_t < min(i_t * BT + BT, T) - 1, b_dg, b_dg + b_dg_last)
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), mask=m_qk)
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), mask=m_qk)
    tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), mask=m_t)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in NUM_WARPS
        for num_stages in [2, 3, 4]
    ],
    key=["H", "HV", "K", "V", "BT", "BK", "BV"],
)
@triton.jit(do_not_specialize=["T"])
def chunk_bwd_kernel_dv_local(
    q,
    k,
    g,
    do,
    dv,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_h = i_bh % HV
    i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
    bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    T = eos - bos

    # offset calculation
    q += (bos * H + i_h // (HV // H)) * K
    k += (bos * H + i_h // (HV // H)) * K
    do += (bos * HV + i_h) * V
    dv += (bos * HV + i_h) * V

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    g += bos * HV + i_h
    p_g = g + o_t * HV
    b_g = tl.load(p_g, mask=m_t, other=0.0)

    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        p_k = k + o_t[:, None] * (H * K) + o_k[None, :]
        p_q = q + o_k[:, None] + o_t[None, :] * (H * K)

        b_k = tl.load(p_k, mask=m_t[:, None] & m_k[None, :], other=0.0)
        b_q = tl.load(p_q, mask=m_k[:, None] & m_t[None, :], other=0.0)
        b_A += tl.dot(b_k, b_q) * scale
    b_A *= exp2(b_g[None, :] - b_g[:, None])
    m_A = (o_t[:, None] <= o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A, b_A, 0).to(do.dtype.element_ty)

    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = o_v < V
        p_do = do + o_t[:, None] * (HV * V) + o_v[None, :]
        p_dv = dv + o_t[:, None] * (HV * V) + o_v[None, :]
        b_do = tl.load(p_do, mask=m_t[:, None] & m_v[None, :], other=0.0)
        b_dv = tl.dot(b_A.to(b_do.dtype), b_do)
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), mask=m_t[:, None] & m_v[None, :])


def chunk_fwd_o(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    g: torch.Tensor,
    scale: float,
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
) -> torch.Tensor:
    B, T, H, K, V, HV = *q.shape, v.shape[-1], v.shape[2]
    BT = 64
    NT = len(chunk_indices)

    o = torch.empty_like(v)

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), NT, B * HV)

    chunk_fwd_kernel_o[grid](
        q=q,
        k=k,
        v=v,
        h=h,
        g=g,
        o=o,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=BT,
    )
    return o


def chunk_bwd_dv_local(
    q: torch.Tensor,
    k: torch.Tensor,
    do: torch.Tensor,
    g: torch.Tensor,
    scale: float,
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
) -> torch.Tensor:
    B, T, H, K, V, HV = *k.shape, do.shape[-1], do.shape[2]
    BT = 64
    # H100 can have larger block size
    if check_shared_mem("hopper", k.device.index):
        CONST_TILING = 128
    elif check_shared_mem("ada", k.device.index):
        CONST_TILING = 64
    else:
        CONST_TILING = 32
    BK = min(max(triton.next_power_of_2(K), 16), CONST_TILING)
    BV = min(max(triton.next_power_of_2(V), 16), CONST_TILING)
    NT = len(chunk_indices)

    dv = torch.empty_like(do)
    grid = (NT, B * HV)
    chunk_bwd_kernel_dv_local[grid](
        q=q,
        k=k,
        g=g,
        do=do,
        dv=dv,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
    )
    return dv


def chunk_bwd_dqkwg(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    h: torch.Tensor,
    dh: torch.Tensor,
    w: torch.Tensor,
    g: torch.Tensor,
    dv: torch.Tensor,
    scale: float,
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if IS_NVIDIA_HOPPER and TRITON_ABOVE_3_4_0 and not TRITON_ABOVE_3_7_1:
        raise RuntimeError(
            "Triton >= 3.4.0 and < 3.7.1 on Hopper GPUs produces incorrect results for "
            "gated chunk_bwd_dqkwg (see #640). Please upgrade Triton to >= 3.7.1 or "
            "use a supported Triton version (this vendored path has no TileLang backend)."
        )

    B, T, H, K, V, HV = *k.shape, v.shape[-1], v.shape[2]
    BT = 64
    NT = len(chunk_indices)

    if check_shared_mem("hopper", k.device.index):
        CONST_TILING = 128
    elif check_shared_mem("ada", k.device.index):
        CONST_TILING = 64
    else:
        CONST_TILING = 32
    BK = min(max(triton.next_power_of_2(K), 16), CONST_TILING)
    BV = min(max(triton.next_power_of_2(V), 16), CONST_TILING)
    NK = triton.cdiv(K, BK)
    dq = q.new_empty(B, T, HV, K)
    dk = k.new_empty(B, T, HV, K)
    dg = torch.empty(NK, *g.shape, dtype=torch.float32, device=g.device)
    dw = torch.empty_like(w)

    grid = (NK, NT, B * HV)
    chunk_bwd_kernel_dqkwg[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        h=h,
        do=do,
        dh=dh,
        dw=dw,
        dq=dq,
        dk=dk,
        dv=dv,
        dg=dg,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        B=B,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
    )

    if H != HV:
        dq = dq.view(B, T, H, HV // H, K).sum(3)
        dk = dk.view(B, T, H, HV // H, K).sum(3)
    dg = dg.sum(0)
    return dq, dk, dw, dg


# --- Chunk-local gate cumulative sums ---


@triton.heuristics(
    {
        "HAS_SCALE": lambda args: args["scale"] is not None,
    }
)
@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4, 8]],
    key=["B", "H", "BT", "REVERSE"],
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
):
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1)
    i_h = i_bh % H
    i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
    bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    T = eos - bos

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
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
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
    reverse: bool = False,
    scale: float | None = None,
) -> torch.Tensor:
    B, T, H = g.shape
    BT = 64
    NT = len(chunk_indices)
    g_org, g = g, torch.empty_like(g, dtype=torch.float32)
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
        REVERSE=reverse,
    )
    return g


# --- Q/K normalization ---


BT_LIST = [32, 64, 128]


@triton.autotune(
    configs=[triton.Config({"BT": BT}, num_warps=num_warps) for num_warps in [1, 2, 4, 8, 16] for BT in BT_LIST],
    key=["D", "NB"],
)
@triton.jit(do_not_specialize=["T"])
def l2norm_fwd_kernel(
    x,
    y,
    rstd,
    eps,
    T,
    D: tl.constexpr,
    BD: tl.constexpr,
    NB: tl.constexpr,
    BT: tl.constexpr,
):
    i_t = tl.program_id(0).to(tl.int64)
    o_t = i_t * BT + tl.arange(0, BT)
    o_d = tl.arange(0, BD)
    m_t = o_t < T
    m_x = m_t[:, None] & (o_d[None, :] < D)
    p_x = x + o_t[:, None] * D + o_d[None, :]
    p_y = y + o_t[:, None] * D + o_d[None, :]
    p_rstd = rstd + o_t

    b_x = tl.load(p_x, mask=m_x, other=0.0).to(tl.float32)
    b_rstd = 1 / tl.sqrt(tl.sum(b_x * b_x, 1) + eps)
    b_y = b_x * b_rstd[:, None]

    tl.store(p_y, b_y.to(p_y.dtype.element_ty), mask=m_x)
    tl.store(p_rstd, b_rstd.to(p_rstd.dtype.element_ty), mask=m_t)


@triton.autotune(
    configs=[triton.Config({"BT": BT}, num_warps=num_warps) for num_warps in [1, 2, 4, 8, 16] for BT in BT_LIST],
    key=["D", "NB"],
)
@triton.jit(do_not_specialize=["T"])
def l2norm_bwd_kernel(
    y,
    rstd,
    dy,
    dx,
    eps,
    T,
    D: tl.constexpr,
    BD: tl.constexpr,
    NB: tl.constexpr,
    BT: tl.constexpr,
):
    i_t = tl.program_id(0).to(tl.int64)
    o_t = i_t * BT + tl.arange(0, BT)
    o_d = tl.arange(0, BD)
    m_t = o_t < T
    m_x = m_t[:, None] & (o_d[None, :] < D)
    p_y = y + o_t[:, None] * D + o_d[None, :]
    p_rstd = rstd + o_t
    p_dy = dy + o_t[:, None] * D + o_d[None, :]
    p_dx = dx + o_t[:, None] * D + o_d[None, :]

    b_y = tl.load(p_y, mask=m_x, other=0.0).to(tl.float32)
    b_rstd = tl.load(p_rstd, mask=m_t, other=0.0).to(tl.float32)
    b_dy = tl.load(p_dy, mask=m_x, other=0.0).to(tl.float32)
    b_dx = b_dy * b_rstd[:, None] - tl.sum(b_dy * b_y, 1)[:, None] * b_y * b_rstd[:, None]
    tl.store(p_dx, b_dx.to(p_dx.dtype.element_ty), mask=m_x)


def l2norm_fwd(
    x: torch.Tensor,
    eps: float = 1e-6,
):
    x_shape_og = x.shape
    x = x.view(-1, x.shape[-1])
    # allocate output
    y = torch.empty_like(x)
    assert y.stride(-1) == 1
    T, D = x.shape[0], x.shape[-1]
    BD = triton.next_power_of_2(D)

    rstd = torch.empty((T,), dtype=torch.float32, device=x.device)
    # NOTE(tylerr): Avoid excessive recompilation and autotuning by tolerating a larger range
    # of T before recompiling the kernel.
    NB = triton.cdiv(T, 2048 * 32)

    def grid(meta):
        return (triton.cdiv(T, meta["BT"]),)

    l2norm_fwd_kernel[grid](
        x=x,
        y=y,
        rstd=rstd,
        eps=eps,
        T=T,
        D=D,
        BD=BD,
        NB=NB,
    )
    return y.view(x_shape_og), rstd.view(x_shape_og[:-1])


def l2norm_bwd(
    y: torch.Tensor,
    rstd: torch.Tensor,
    dy: torch.Tensor,
    eps: float = 1e-6,
):
    y_shape_og = y.shape
    y = y.view(-1, dy.shape[-1])
    dy = dy.view(-1, dy.shape[-1])
    assert dy.shape == y.shape
    # allocate output
    dx = torch.empty_like(y)
    T, D = y.shape[0], y.shape[-1]
    BD = triton.next_power_of_2(D)

    # NOTE(tylerr): Avoid excessive recompilation and autotuning by tolerating a larger range
    # of T before recompiling the kernel.
    NB = triton.cdiv(T, 2048 * 32)

    def grid(meta):
        return (triton.cdiv(T, meta["BT"]),)

    l2norm_bwd_kernel[grid](
        y=y,
        rstd=rstd,
        dy=dy,
        dx=dx,
        eps=eps,
        T=T,
        D=D,
        BD=BD,
        NB=NB,
    )

    return dx.view(y_shape_og)
