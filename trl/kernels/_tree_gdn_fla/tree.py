# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
# Copyright 2026 The HuggingFace Team. All rights reserved.
# Licensed under the MIT license in this directory.

"""FLA 0.5.2 chunk pipeline, with tree-aware state scans. Chunk-local kernels run once for the whole forest."""

import torch
import triton

from .chunk_delta_h import (
    chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64,
    chunk_gated_delta_rule_fwd_kernel_h_blockdim64,
)
from .chunk_fwd import chunk_gated_delta_rule_fwd_kkt_solve_kernel
from .chunk_o import chunk_bwd_dqkwg, chunk_bwd_dv_local, chunk_fwd_o
from .cumsum import chunk_local_cumsum_scalar
from .l2norm import l2norm_bwd, l2norm_fwd
from .wy_fast import prepare_wy_repr_bwd, recompute_w_u_fwd


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
            gk=None,
            h=h,
            h0=final,
            ht=final,
            cu_seqlens=plan.cu_seqlens,
            chunk_offsets=plan.chunk_offsets,
            segments=segments,
            parents=plan.parents,
            child_offsets=plan.child_offsets,
            children=plan.children,
            T=tokens,
            H=heads,
            HV=value_heads,
            K=key_dim,
            V=value_dim,
            BT=64,
            STATE_V_FIRST=False,
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
            gk=None,
            dht=d_initial,
            dh0=d_initial,
            do=do,
            dh=dh,
            dv=dv,
            dv2=du,
            cu_seqlens=plan.cu_seqlens,
            chunk_offsets=plan.chunk_offsets,
            segments=segments,
            parents=plan.parents,
            child_offsets=plan.child_offsets,
            children=plan.children,
            scale=scale,
            T=tokens,
            H=heads,
            HV=value_heads,
            K=key_dim,
            V=value_dim,
            BT=64,
            STATE_V_FIRST=False,
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
        g = chunk_local_cumsum_scalar(g, 64, scale=1.4426950408889634, **layout)
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
        dg = chunk_local_cumsum_scalar(dg, 64, reverse=True, **layout)
        if ctx.normalize:
            dq = l2norm_bwd(q, q_rstd, dq)
            dk = l2norm_bwd(k, k_rstd, dk)
        return dq.to(q), dk.to(k), dv.to(v), dg, db.to(beta), None, None, None
