# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in this directory.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import triton
import triton.language as tl

from .utils import IS_NVIDIA_BLACKWELL, check_shared_mem, exp2


GATED_DELTA_RULE_FWD_H_NUM_WARPS = [2] if IS_NVIDIA_BLACKWELL else [2, 4]


@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "USE_GK": lambda args: args["gk"] is not None,
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
        "SAVE_NEW_VALUE": lambda args: args["v_new"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BV": BV}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in GATED_DELTA_RULE_FWD_H_NUM_WARPS
        for num_stages in ([2, 3, 4] if check_shared_mem("ampere") else [2, 1])
        for BV in ([32, 64] if check_shared_mem("ada") else [32])
    ],
    key=["H", "HV", "K", "V", "BT", "STATE_V_FIRST"],
)
@triton.jit(do_not_specialize=["T"])
def chunk_gated_delta_rule_fwd_kernel_h_blockdim64(
    k,
    v,
    w,
    v_new,
    g,
    gk,
    h,
    h0,
    ht,
    cu_seqlens,
    chunk_offsets,
    segments,
    parents,
    child_offsets,
    children,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    SAVE_NEW_VALUE: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = tl.load(segments + i_nh // HV), i_nh % HV
    i_nh = i_n * HV + i_h
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    if STATE_V_FIRST:
        b_h1 = tl.zeros([BV, 64], dtype=tl.float32)
        if K > 64:
            b_h2 = tl.zeros([BV, 64], dtype=tl.float32)
        if K > 128:
            b_h3 = tl.zeros([BV, 64], dtype=tl.float32)
        if K > 192:
            b_h4 = tl.zeros([BV, 64], dtype=tl.float32)
    else:
        b_h1 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 64:
            b_h2 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 128:
            b_h3 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 192:
            b_h4 = tl.zeros([64, BV], dtype=tl.float32)

    # calculate offset
    h += (boh * HV + i_h).to(tl.int64) * K * V
    v += (bos * HV + i_h).to(tl.int64) * V
    k += (bos * H + i_h // (HV // H)).to(tl.int64) * K
    w += (bos * HV + i_h).to(tl.int64) * K
    if SAVE_NEW_VALUE:
        v_new += (bos * HV + i_h).to(tl.int64) * V

    parent = tl.load(parents + i_n)
    if USE_INITIAL_STATE:
        h0 = h0 + (parent * HV + i_h) * K * V
    if STORE_FINAL_STATE:
        ht = ht + i_nh * K * V

    # load initial state
    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V
    o_k1 = tl.arange(0, 64)
    m_k1 = o_k1 < K
    o_k2 = 64 + o_k1
    m_k2 = o_k2 < K
    o_k3 = 128 + o_k1
    m_k3 = o_k3 < K
    o_k4 = 192 + o_k1
    m_k4 = o_k4 < K
    if parent >= 0:
        if STATE_V_FIRST:
            p_h0_1 = h0 + o_v[:, None] * K + o_k1[None, :]
            m_h0_1 = m_v[:, None] & m_k1[None, :]
        else:
            p_h0_1 = h0 + o_k1[:, None] * V + o_v[None, :]
            m_h0_1 = m_k1[:, None] & m_v[None, :]
        b_h1 += tl.load(p_h0_1, mask=m_h0_1, other=0.0).to(tl.float32)
        if K > 64:
            if STATE_V_FIRST:
                p_h0_2 = h0 + o_v[:, None] * K + o_k2[None, :]
                m_h0_2 = m_v[:, None] & m_k2[None, :]
            else:
                p_h0_2 = h0 + o_k2[:, None] * V + o_v[None, :]
                m_h0_2 = m_k2[:, None] & m_v[None, :]
            b_h2 += tl.load(p_h0_2, mask=m_h0_2, other=0.0).to(tl.float32)
        if K > 128:
            if STATE_V_FIRST:
                p_h0_3 = h0 + o_v[:, None] * K + o_k3[None, :]
                m_h0_3 = m_v[:, None] & m_k3[None, :]
            else:
                p_h0_3 = h0 + o_k3[:, None] * V + o_v[None, :]
                m_h0_3 = m_k3[:, None] & m_v[None, :]
            b_h3 += tl.load(p_h0_3, mask=m_h0_3, other=0.0).to(tl.float32)
        if K > 192:
            if STATE_V_FIRST:
                p_h0_4 = h0 + o_v[:, None] * K + o_k4[None, :]
                m_h0_4 = m_v[:, None] & m_k4[None, :]
            else:
                p_h0_4 = h0 + o_k4[:, None] * V + o_v[None, :]
                m_h0_4 = m_k4[:, None] & m_v[None, :]
            b_h4 += tl.load(p_h0_4, mask=m_h0_4, other=0.0).to(tl.float32)

    # main recurrence
    for i_t in range(NT):
        i_t_int64 = i_t.to(tl.int64)
        o_t = i_t * BT + tl.arange(0, BT)
        m_t = o_t < T
        if STATE_V_FIRST:
            p_h1 = h + i_t_int64 * HV * K * V + o_v[:, None] * K + o_k1[None, :]
            m_h1 = m_v[:, None] & m_k1[None, :]
        else:
            p_h1 = h + i_t_int64 * HV * K * V + o_k1[:, None] * V + o_v[None, :]
            m_h1 = m_k1[:, None] & m_v[None, :]
        tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), mask=m_h1)
        if K > 64:
            if STATE_V_FIRST:
                p_h2 = h + i_t_int64 * HV * K * V + o_v[:, None] * K + o_k2[None, :]
                m_h2 = m_v[:, None] & m_k2[None, :]
            else:
                p_h2 = h + i_t_int64 * HV * K * V + o_k2[:, None] * V + o_v[None, :]
                m_h2 = m_k2[:, None] & m_v[None, :]
            tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), mask=m_h2)
        if K > 128:
            if STATE_V_FIRST:
                p_h3 = h + i_t_int64 * HV * K * V + o_v[:, None] * K + o_k3[None, :]
                m_h3 = m_v[:, None] & m_k3[None, :]
            else:
                p_h3 = h + i_t_int64 * HV * K * V + o_k3[:, None] * V + o_v[None, :]
                m_h3 = m_k3[:, None] & m_v[None, :]
            tl.store(p_h3, b_h3.to(p_h3.dtype.element_ty), mask=m_h3)
        if K > 192:
            if STATE_V_FIRST:
                p_h4 = h + i_t_int64 * HV * K * V + o_v[:, None] * K + o_k4[None, :]
                m_h4 = m_v[:, None] & m_k4[None, :]
            else:
                p_h4 = h + i_t_int64 * HV * K * V + o_k4[:, None] * V + o_v[None, :]
                m_h4 = m_k4[:, None] & m_v[None, :]
            tl.store(p_h4, b_h4.to(p_h4.dtype.element_ty), mask=m_h4)

        p_w = w + o_t[:, None] * (HV * K) + o_k1[None, :]
        b_w = tl.load(p_w, mask=m_t[:, None] & m_k1[None, :], other=0.0)
        if STATE_V_FIRST:
            b_v = tl.dot(b_w, tl.trans(b_h1).to(b_w.dtype))
        else:
            b_v = tl.dot(b_w, b_h1.to(b_w.dtype))
        if K > 64:
            p_w = w + o_t[:, None] * (HV * K) + o_k2[None, :]
            b_w = tl.load(p_w, mask=m_t[:, None] & m_k2[None, :], other=0.0)
            if STATE_V_FIRST:
                b_v += tl.dot(b_w, tl.trans(b_h2).to(b_w.dtype))
            else:
                b_v += tl.dot(b_w, b_h2.to(b_w.dtype))
        if K > 128:
            p_w = w + o_t[:, None] * (HV * K) + o_k3[None, :]
            b_w = tl.load(p_w, mask=m_t[:, None] & m_k3[None, :], other=0.0)
            if STATE_V_FIRST:
                b_v += tl.dot(b_w, tl.trans(b_h3).to(b_w.dtype))
            else:
                b_v += tl.dot(b_w, b_h3.to(b_w.dtype))
        if K > 192:
            p_w = w + o_t[:, None] * (HV * K) + o_k4[None, :]
            b_w = tl.load(p_w, mask=m_t[:, None] & m_k4[None, :], other=0.0)
            if STATE_V_FIRST:
                b_v += tl.dot(b_w, tl.trans(b_h4).to(b_w.dtype))
            else:
                b_v += tl.dot(b_w, b_h4.to(b_w.dtype))
        p_v = v + o_t[:, None] * (HV * V) + o_v[None, :]
        b_v = tl.load(p_v, mask=m_t[:, None] & m_v[None, :], other=0.0) - b_v

        if SAVE_NEW_VALUE:
            p_v = v_new + o_t[:, None] * (HV * V) + o_v[None, :]
            tl.store(p_v, b_v.to(p_v.dtype.element_ty), mask=m_t[:, None] & m_v[None, :])

        last_idx = min((i_t + 1) * BT, T) - 1
        if USE_G:
            b_g_last = tl.load(g + (bos * HV + last_idx * HV + i_h).to(tl.int64)).to(tl.float32)
            p_g = g + (bos * HV + i_h).to(tl.int64) + o_t * HV
            b_g = tl.load(p_g, mask=m_t, other=0.0).to(tl.float32)
            b_v = b_v * tl.where(m_t, exp2(b_g_last - b_g), 0)[:, None]
            b_g_last = exp2(b_g_last)
            b_h1 *= b_g_last
            if K > 64:
                b_h2 *= b_g_last
            if K > 128:
                b_h3 *= b_g_last
            if K > 192:
                b_h4 *= b_g_last

        if USE_GK:
            o_k1 = tl.arange(0, 64)
            b_gk_last1 = tl.load(gk + (bos + last_idx) * HV * K + i_h * K + o_k1, mask=(o_k1 < K), other=0.0).to(
                tl.float32
            )
            if STATE_V_FIRST:
                b_h1 *= exp2(b_gk_last1)[None, :]
            else:
                b_h1 *= exp2(b_gk_last1)[:, None]
            if K > 64:
                o_k2 = 64 + o_k1
                b_gk_last2 = tl.load(gk + (bos + last_idx) * HV * K + i_h * K + o_k2, mask=(o_k2 < K), other=0.0).to(
                    tl.float32
                )
                if STATE_V_FIRST:
                    b_h2 *= exp2(b_gk_last2)[None, :]
                else:
                    b_h2 *= exp2(b_gk_last2)[:, None]
            if K > 128:
                o_k3 = 128 + o_k1
                b_gk_last3 = tl.load(gk + (bos + last_idx) * HV * K + i_h * K + o_k3, mask=(o_k3 < K), other=0.0).to(
                    tl.float32
                )
                if STATE_V_FIRST:
                    b_h3 *= exp2(b_gk_last3)[None, :]
                else:
                    b_h3 *= exp2(b_gk_last3)[:, None]
            if K > 192:
                o_k4 = 192 + o_k1
                b_gk_last4 = tl.load(gk + (bos + last_idx) * HV * K + i_h * K + o_k4, mask=(o_k4 < K), other=0.0).to(
                    tl.float32
                )
                if STATE_V_FIRST:
                    b_h4 *= exp2(b_gk_last4)[None, :]
                else:
                    b_h4 *= exp2(b_gk_last4)[:, None]
        b_v = b_v.to(k.dtype.element_ty)

        p_k = k + o_k1[:, None] + o_t[None, :] * (H * K)
        b_k = tl.load(p_k, mask=m_k1[:, None] & m_t[None, :], other=0.0)
        if STATE_V_FIRST:
            b_h1 += tl.trans(tl.dot(b_k, b_v))
        else:
            b_h1 += tl.dot(b_k, b_v)
        if K > 64:
            p_k = k + o_k2[:, None] + o_t[None, :] * (H * K)
            b_k = tl.load(p_k, mask=m_k2[:, None] & m_t[None, :], other=0.0)
            if STATE_V_FIRST:
                b_h2 += tl.trans(tl.dot(b_k, b_v))
            else:
                b_h2 += tl.dot(b_k, b_v)
        if K > 128:
            p_k = k + o_k3[:, None] + o_t[None, :] * (H * K)
            b_k = tl.load(p_k, mask=m_k3[:, None] & m_t[None, :], other=0.0)
            if STATE_V_FIRST:
                b_h3 += tl.trans(tl.dot(b_k, b_v))
            else:
                b_h3 += tl.dot(b_k, b_v)
        if K > 192:
            p_k = k + o_k4[:, None] + o_t[None, :] * (H * K)
            b_k = tl.load(p_k, mask=m_k4[:, None] & m_t[None, :], other=0.0)
            if STATE_V_FIRST:
                b_h4 += tl.trans(tl.dot(b_k, b_v))
            else:
                b_h4 += tl.dot(b_k, b_v)

    if STORE_FINAL_STATE:
        if STATE_V_FIRST:
            p_ht = ht + o_v[:, None] * K + o_k1[None, :]
            m_ht = m_v[:, None] & m_k1[None, :]
        else:
            p_ht = ht + o_k1[:, None] * V + o_v[None, :]
            m_ht = m_k1[:, None] & m_v[None, :]
        tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), mask=m_ht)
        if K > 64:
            if STATE_V_FIRST:
                p_ht = ht + o_v[:, None] * K + o_k2[None, :]
                m_ht = m_v[:, None] & m_k2[None, :]
            else:
                p_ht = ht + o_k2[:, None] * V + o_v[None, :]
                m_ht = m_k2[:, None] & m_v[None, :]
            tl.store(p_ht, b_h2.to(p_ht.dtype.element_ty), mask=m_ht)
        if K > 128:
            if STATE_V_FIRST:
                p_ht = ht + o_v[:, None] * K + o_k3[None, :]
                m_ht = m_v[:, None] & m_k3[None, :]
            else:
                p_ht = ht + o_k3[:, None] * V + o_v[None, :]
                m_ht = m_k3[:, None] & m_v[None, :]
            tl.store(p_ht, b_h3.to(p_ht.dtype.element_ty), mask=m_ht)
        if K > 192:
            if STATE_V_FIRST:
                p_ht = ht + o_v[:, None] * K + o_k4[None, :]
                m_ht = m_v[:, None] & m_k4[None, :]
            else:
                p_ht = ht + o_k4[:, None] * V + o_v[None, :]
                m_ht = m_k4[:, None] & m_v[None, :]
            tl.store(p_ht, b_h4.to(p_ht.dtype.element_ty), mask=m_ht)


@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "USE_GK": lambda args: args["gk"] is not None,
        "USE_INITIAL_STATE": lambda args: args["dh0"] is not None,
        "USE_FINAL_STATE_GRADIENT": lambda args: args["dht"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BV": BV}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4]
        for num_stages in ([2, 3, 4] if check_shared_mem("ampere") else [1])
        for BV in ([32, 64] if check_shared_mem("ada") else [32])
    ],
    key=["H", "HV", "K", "V", "BT", "BV", "USE_G", "STATE_V_FIRST"],
)
@triton.jit(do_not_specialize=["T"])
def chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64(
    q,
    k,
    w,
    g,
    gk,
    dht,
    dh0,
    do,
    dh,
    dv,
    dv2,
    cu_seqlens,
    chunk_offsets,
    segments,
    parents,
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
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = tl.load(segments + i_nh // HV), i_nh % HV
    i_nh = i_n * HV + i_h
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    if STATE_V_FIRST:
        b_dh1 = tl.zeros([BV, 64], dtype=tl.float32)
        if K > 64:
            b_dh2 = tl.zeros([BV, 64], dtype=tl.float32)
        if K > 128:
            b_dh3 = tl.zeros([BV, 64], dtype=tl.float32)
        if K > 192:
            b_dh4 = tl.zeros([BV, 64], dtype=tl.float32)
    else:
        b_dh1 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 64:
            b_dh2 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 128:
            b_dh3 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 192:
            b_dh4 = tl.zeros([64, BV], dtype=tl.float32)

    # calculate offset
    q += (bos * H + i_h // (HV // H)).to(tl.int64) * K
    k += (bos * H + i_h // (HV // H)).to(tl.int64) * K
    w += (bos * HV + i_h).to(tl.int64) * K
    do += (bos * HV + i_h).to(tl.int64) * V
    dv += (bos * HV + i_h).to(tl.int64) * V
    dv2 += (bos * HV + i_h).to(tl.int64) * V
    dh += (boh * HV + i_h).to(tl.int64) * K * V
    if USE_GK:
        gk += (bos * HV + i_h).to(tl.int64) * K

    if USE_INITIAL_STATE:
        dh0 += i_nh * K * V
    child_grad = dht

    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V
    o_k1 = tl.arange(0, 64)
    m_k1 = o_k1 < K
    o_k2 = 64 + o_k1
    m_k2 = o_k2 < K
    o_k3 = 128 + o_k1
    m_k3 = o_k3 < K
    o_k4 = 192 + o_k1
    m_k4 = o_k4 < K
    # Tree boundary: sum child input-state gradients before scanning this segment backward.
    first = tl.load(child_offsets + i_n)
    last = tl.load(child_offsets + i_n + 1)
    for edge in range(first, last):
        child = tl.load(children + edge)
        dht = child_grad + (child * HV + i_h) * K * V
        if STATE_V_FIRST:
            p_dht1 = dht + o_v[:, None] * K + o_k1[None, :]
            m_dht1 = m_v[:, None] & m_k1[None, :]
        else:
            p_dht1 = dht + o_k1[:, None] * V + o_v[None, :]
            m_dht1 = m_k1[:, None] & m_v[None, :]
        b_dh1 += tl.load(p_dht1, mask=m_dht1, other=0.0)
        if K > 64:
            if STATE_V_FIRST:
                p_dht2 = dht + o_v[:, None] * K + o_k2[None, :]
                m_dht2 = m_v[:, None] & m_k2[None, :]
            else:
                p_dht2 = dht + o_k2[:, None] * V + o_v[None, :]
                m_dht2 = m_k2[:, None] & m_v[None, :]
            b_dh2 += tl.load(p_dht2, mask=m_dht2, other=0.0)
        if K > 128:
            if STATE_V_FIRST:
                p_dht3 = dht + o_v[:, None] * K + o_k3[None, :]
                m_dht3 = m_v[:, None] & m_k3[None, :]
            else:
                p_dht3 = dht + o_k3[:, None] * V + o_v[None, :]
                m_dht3 = m_k3[:, None] & m_v[None, :]
            b_dh3 += tl.load(p_dht3, mask=m_dht3, other=0.0)
        if K > 192:
            if STATE_V_FIRST:
                p_dht4 = dht + o_v[:, None] * K + o_k4[None, :]
                m_dht4 = m_v[:, None] & m_k4[None, :]
            else:
                p_dht4 = dht + o_k4[:, None] * V + o_v[None, :]
                m_dht4 = m_k4[:, None] & m_v[None, :]
            b_dh4 += tl.load(p_dht4, mask=m_dht4, other=0.0)

    for i_t in range(NT - 1, -1, -1):
        i_t_int64 = i_t.to(tl.int64)
        o_t = i_t * BT + tl.arange(0, BT)
        m_t = o_t < T
        if STATE_V_FIRST:
            p_dh1 = dh + i_t_int64 * HV * K * V + o_v[:, None] * K + o_k1[None, :]
            m_dh1 = m_v[:, None] & m_k1[None, :]
        else:
            p_dh1 = dh + i_t_int64 * HV * K * V + o_k1[:, None] * V + o_v[None, :]
            m_dh1 = m_k1[:, None] & m_v[None, :]
        tl.store(p_dh1, b_dh1.to(p_dh1.dtype.element_ty), mask=m_dh1)
        if K > 64:
            if STATE_V_FIRST:
                p_dh2 = dh + i_t_int64 * HV * K * V + o_v[:, None] * K + o_k2[None, :]
                m_dh2 = m_v[:, None] & m_k2[None, :]
            else:
                p_dh2 = dh + i_t_int64 * HV * K * V + o_k2[:, None] * V + o_v[None, :]
                m_dh2 = m_k2[:, None] & m_v[None, :]
            tl.store(p_dh2, b_dh2.to(p_dh2.dtype.element_ty), mask=m_dh2)
        if K > 128:
            if STATE_V_FIRST:
                p_dh3 = dh + i_t_int64 * HV * K * V + o_v[:, None] * K + o_k3[None, :]
                m_dh3 = m_v[:, None] & m_k3[None, :]
            else:
                p_dh3 = dh + i_t_int64 * HV * K * V + o_k3[:, None] * V + o_v[None, :]
                m_dh3 = m_k3[:, None] & m_v[None, :]
            tl.store(p_dh3, b_dh3.to(p_dh3.dtype.element_ty), mask=m_dh3)
        if K > 192:
            if STATE_V_FIRST:
                p_dh4 = dh + i_t_int64 * HV * K * V + o_v[:, None] * K + o_k4[None, :]
                m_dh4 = m_v[:, None] & m_k4[None, :]
            else:
                p_dh4 = dh + i_t_int64 * HV * K * V + o_k4[:, None] * V + o_v[None, :]
                m_dh4 = m_k4[:, None] & m_v[None, :]
            tl.store(p_dh4, b_dh4.to(p_dh4.dtype.element_ty), mask=m_dh4)

        last_idx = min((i_t + 1) * BT, T) - 1
        if USE_G:
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
        if USE_GK:
            o_k1 = tl.arange(0, 64)
            b_gk_last1 = tl.load(gk + last_idx * HV * K + o_k1, mask=(o_k1 < K), other=0.0).to(tl.float32)
        if STATE_V_FIRST:
            b_dv = tl.dot(b_k, tl.trans(b_dh1).to(b_k.dtype))
        else:
            b_dv = tl.dot(b_k, b_dh1.to(b_k.dtype))

        if K > 64:
            p_k = k + o_t[:, None] * (H * K) + o_k2[None, :]
            b_k = tl.load(p_k, mask=m_t[:, None] & m_k2[None, :], other=0.0)
            if USE_GK:
                b_gk_last2 = tl.load(gk + last_idx * HV * K + o_k2, mask=(o_k2 < K), other=0.0).to(tl.float32)
            if STATE_V_FIRST:
                b_dv += tl.dot(b_k, tl.trans(b_dh2).to(b_k.dtype))
            else:
                b_dv += tl.dot(b_k, b_dh2.to(b_k.dtype))

        if K > 128:
            p_k = k + o_t[:, None] * (H * K) + o_k3[None, :]
            b_k = tl.load(p_k, mask=m_t[:, None] & m_k3[None, :], other=0.0)
            if USE_GK:
                b_gk_last3 = tl.load(gk + last_idx * HV * K + o_k3, mask=(o_k3 < K), other=0.0).to(tl.float32)
            if STATE_V_FIRST:
                b_dv += tl.dot(b_k, tl.trans(b_dh3).to(b_k.dtype))
            else:
                b_dv += tl.dot(b_k, b_dh3.to(b_k.dtype))

        if K > 192:
            p_k = k + o_t[:, None] * (H * K) + o_k4[None, :]
            b_k = tl.load(p_k, mask=m_t[:, None] & m_k4[None, :], other=0.0)
            if USE_GK:
                b_gk_last4 = tl.load(gk + last_idx * HV * K + o_k4, mask=(o_k4 < K), other=0.0).to(tl.float32)
            if STATE_V_FIRST:
                b_dv += tl.dot(b_k, tl.trans(b_dh4).to(b_k.dtype))
            else:
                b_dv += tl.dot(b_k, b_dh4.to(b_k.dtype))

        if USE_G:
            b_dv *= tl.where(m_t, exp2(bg_last - b_g), 0)[:, None]
        b_dv += tl.load(p_dv, mask=m_t[:, None] & m_v[None, :], other=0.0)

        tl.store(p_dv2, b_dv.to(p_dv.dtype.element_ty), mask=m_t[:, None] & m_v[None, :])
        # Update dh
        p_w = w + o_k1[:, None] + o_t[None, :] * (HV * K)
        p_q = q + o_k1[:, None] + o_t[None, :] * (H * K)
        b_w = tl.load(p_w, mask=m_k1[:, None] & m_t[None, :], other=0.0)
        b_q = tl.load(p_q, mask=m_k1[:, None] & m_t[None, :], other=0.0)
        if USE_G:
            b_dh1 *= bg_last_exp
            b_q = b_q * b_g_exp[None, :]
        if USE_GK:
            if STATE_V_FIRST:
                b_dh1 *= exp2(b_gk_last1)[None, :]
            else:
                b_dh1 *= exp2(b_gk_last1[:, None])
        if STATE_V_FIRST:
            b_dh1 += tl.trans(tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype)))
        else:
            b_dh1 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype))
        if K > 64:
            p_q = q + o_k2[:, None] + o_t[None, :] * (H * K)
            p_w = w + o_k2[:, None] + o_t[None, :] * (HV * K)
            b_q = tl.load(p_q, mask=m_k2[:, None] & m_t[None, :], other=0.0)
            b_w = tl.load(p_w, mask=m_k2[:, None] & m_t[None, :], other=0.0)
            if USE_G:
                b_dh2 *= bg_last_exp
                b_q = b_q * b_g_exp[None, :]
            if USE_GK:
                if STATE_V_FIRST:
                    b_dh2 *= exp2(b_gk_last2)[None, :]
                else:
                    b_dh2 *= exp2(b_gk_last2[:, None])
            if STATE_V_FIRST:
                b_dh2 += tl.trans(
                    tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype))
                )
            else:
                b_dh2 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype))
        if K > 128:
            p_q = q + o_k3[:, None] + o_t[None, :] * (H * K)
            p_w = w + o_k3[:, None] + o_t[None, :] * (HV * K)
            b_q = tl.load(p_q, mask=m_k3[:, None] & m_t[None, :], other=0.0)
            b_w = tl.load(p_w, mask=m_k3[:, None] & m_t[None, :], other=0.0)
            if USE_G:
                b_dh3 *= bg_last_exp
                b_q = b_q * b_g_exp[None, :]
            if USE_GK:
                if STATE_V_FIRST:
                    b_dh3 *= exp2(b_gk_last3)[None, :]
                else:
                    b_dh3 *= exp2(b_gk_last3[:, None])
            if STATE_V_FIRST:
                b_dh3 += tl.trans(
                    tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype))
                )
            else:
                b_dh3 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype))
        if K > 192:
            p_q = q + o_k4[:, None] + o_t[None, :] * (H * K)
            p_w = w + o_k4[:, None] + o_t[None, :] * (HV * K)
            b_q = tl.load(p_q, mask=m_k4[:, None] & m_t[None, :], other=0.0)
            b_w = tl.load(p_w, mask=m_k4[:, None] & m_t[None, :], other=0.0)
            if USE_G:
                b_dh4 *= bg_last_exp
                b_q = b_q * b_g_exp[None, :]
            if USE_GK:
                if STATE_V_FIRST:
                    b_dh4 *= exp2(b_gk_last4)[None, :]
                else:
                    b_dh4 *= exp2(b_gk_last4[:, None])
            if STATE_V_FIRST:
                b_dh4 += tl.trans(
                    tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype))
                )
            else:
                b_dh4 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype))

    if USE_INITIAL_STATE:
        if STATE_V_FIRST:
            p_dh0 = dh0 + o_v[:, None] * K + o_k1[None, :]
            m_dh0 = m_v[:, None] & m_k1[None, :]
        else:
            p_dh0 = dh0 + o_k1[:, None] * V + o_v[None, :]
            m_dh0 = m_k1[:, None] & m_v[None, :]
        tl.store(p_dh0, b_dh1.to(p_dh0.dtype.element_ty), mask=m_dh0)
        if K > 64:
            if STATE_V_FIRST:
                p_dh1 = dh0 + o_v[:, None] * K + o_k2[None, :]
                m_dh1 = m_v[:, None] & m_k2[None, :]
            else:
                p_dh1 = dh0 + o_k2[:, None] * V + o_v[None, :]
                m_dh1 = m_k2[:, None] & m_v[None, :]
            tl.store(p_dh1, b_dh2.to(p_dh1.dtype.element_ty), mask=m_dh1)
        if K > 128:
            if STATE_V_FIRST:
                p_dh2 = dh0 + o_v[:, None] * K + o_k3[None, :]
                m_dh2 = m_v[:, None] & m_k3[None, :]
            else:
                p_dh2 = dh0 + o_k3[:, None] * V + o_v[None, :]
                m_dh2 = m_k3[:, None] & m_v[None, :]
            tl.store(p_dh2, b_dh3.to(p_dh2.dtype.element_ty), mask=m_dh2)
        if K > 192:
            if STATE_V_FIRST:
                p_dh3 = dh0 + o_v[:, None] * K + o_k4[None, :]
                m_dh3 = m_v[:, None] & m_k4[None, :]
            else:
                p_dh3 = dh0 + o_k4[:, None] * V + o_v[None, :]
                m_dh3 = m_k4[:, None] & m_v[None, :]
            tl.store(p_dh3, b_dh4.to(p_dh3.dtype.element_ty), mask=m_dh3)
