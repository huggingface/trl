# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Chunk state scan for a segment tree. No token-wise state materialization or branch atomics.

For each chunk: Vnew = U - W H; Hnext = decay H + Kbar^T Vnew. The reverse scan differentiates these two equations and
sums child-state gradients before visiting a parent.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _forward(
    Kbar,
    W,
    U,
    Decay,
    H,
    Vnew,
    Final,
    Offsets,
    Parents,
    Segments,
    NH: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    tile, segment_index, head = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    segment = tl.load(Segments + segment_index)
    parent = tl.load(Parents + segment)
    kk, vv, tt = tl.arange(0, BK), tile * BV + tl.arange(0, BV), tl.arange(0, 64)
    state_index = head * K * V + kk[:, None] * V + vv[None, :]
    state_mask = (kk[:, None] < K) & (vv[None, :] < V)
    state = tl.full((BK, BV), 0, tl.float32)
    if parent >= 0:
        state = tl.load(Final + parent * NH * K * V + state_index, state_mask, 0)
    start, end = tl.load(Offsets + segment), tl.load(Offsets + segment + 1)
    for chunk in range(start, end):
        ch = chunk * NH + head
        tl.store(H + ch * K * V + kk[:, None] * V + vv[None, :], state, state_mask)
        key_index = ch * 64 * K + tt[:, None] * K + kk[None, :]
        value_index = ch * 64 * V + tt[:, None] * V + vv[None, :]
        w = tl.load(W + key_index, kk[None, :] < K, 0)
        u = tl.load(U + value_index, vv[None, :] < V, 0)
        value = u - tl.dot(w, state, input_precision="tf32x3")
        tl.store(Vnew + value_index, value, vv[None, :] < V)
        kb = tl.load(Kbar + key_index, kk[None, :] < K, 0)
        decay = tl.load(Decay + ch)
        state = decay * state + tl.dot(tl.trans(kb), value, input_precision="tf32x3")
    tl.store(Final + segment * NH * K * V + state_index, state, state_mask)


@triton.jit
def _backward(
    Kbar,
    W,
    Decay,
    H,
    Vnew,
    DH,
    DV,
    DKbar,
    DW,
    DU,
    DDecay,
    DInitial,
    Offsets,
    Segments,
    ChildOffsets,
    Children,
    NC: tl.constexpr,
    NH: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    tile, segment_index, head = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    segment = tl.load(Segments + segment_index)
    kk, vv, tt = tl.arange(0, BK), tile * BV + tl.arange(0, BV), tl.arange(0, 64)
    state_index = head * K * V + kk[:, None] * V + vv[None, :]
    state_mask = (kk[:, None] < K) & (vv[None, :] < V)
    dstate = tl.full((BK, BV), 0, tl.float32)
    # Each child's reverse scan completed in an earlier launch. Sum in a fixed order, without atomics.
    first, last = tl.load(ChildOffsets + segment), tl.load(ChildOffsets + segment + 1)
    for i in range(first, last):
        child = tl.load(Children + i)
        dstate += tl.load(DInitial + child * NH * K * V + state_index, state_mask, 0)
    start, end = tl.load(Offsets + segment), tl.load(Offsets + segment + 1)
    for step in range(end - start):
        chunk = end - 1 - step
        ch = chunk * NH + head
        key_index = ch * 64 * K + tt[:, None] * K + kk[None, :]
        value_index = ch * 64 * V + tt[:, None] * V + vv[None, :]
        h_index = ch * K * V + kk[:, None] * V + vv[None, :]
        kb = tl.load(Kbar + key_index, kk[None, :] < K, 0)
        dv = tl.load(DV + value_index, vv[None, :] < V, 0)
        du = dv + tl.dot(kb, dstate, input_precision="tf32x3")
        tl.store(DU + value_index, du, vv[None, :] < V)
        value = tl.load(Vnew + value_index, vv[None, :] < V, 0)
        dk = tl.dot(value, tl.trans(dstate), input_precision="tf32x3")
        # Separate value tiles write separate partials; a reduction below joins them deterministically.
        partial_index = tile * NC * NH * 64 * K + key_index
        tl.store(DKbar + partial_index, dk, kk[None, :] < K)
        h = tl.load(H + h_index, state_mask, 0)
        dw = -tl.dot(du, tl.trans(h), input_precision="tf32x3")
        tl.store(DW + partial_index, dw, kk[None, :] < K)
        da = tl.sum(tl.sum(h * dstate, axis=0), axis=0)
        tl.store(DDecay + tile * NC * NH + ch, da)
        w = tl.load(W + key_index, kk[None, :] < K, 0)
        decay = tl.load(Decay + ch)
        dh = tl.load(DH + h_index, state_mask, 0)
        dstate = dh + decay * dstate - tl.dot(tl.trans(w), du, input_precision="tf32x3")
    tl.store(DInitial + segment * NH * K * V + state_index, dstate, state_mask)


class TreeScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kbar, w, u, decay, plan):
        kbar, w, u, decay = [x.contiguous() for x in (kbar, w, u, decay)]
        nc, heads, _, k = w.shape
        v = u.shape[-1]
        h = w.new_empty((nc, heads, k, v))
        vnew = torch.empty_like(u)
        final = w.new_empty((plan.parents.numel(), heads, k, v))
        for segments in plan.levels:
            _forward[(triton.cdiv(v, 64), segments.numel(), heads)](
                kbar,
                w,
                u,
                decay,
                h,
                vnew,
                final,
                plan.chunk_offsets,
                plan.parents,
                segments,
                heads,
                k,
                v,
                triton.next_power_of_2(k),
                64,
                num_warps=8,
                num_stages=1,
            )
        ctx.save_for_backward(kbar, w, decay, h, vnew)
        ctx.plan = plan
        return h, vnew

    @staticmethod
    def backward(ctx, dh, dv):
        kbar, w, decay, h, vnew = ctx.saved_tensors
        plan = ctx.plan
        nc, heads, _, k = w.shape
        v = vnew.shape[-1]
        tiles = triton.cdiv(v, 64)
        dk = w.new_empty((tiles, *w.shape))
        dw = torch.empty_like(dk)
        du = torch.empty_like(vnew)
        da = decay.new_empty((tiles, *decay.shape))
        d_initial = w.new_empty((plan.parents.numel(), heads, k, v))
        dh, dv = dh.contiguous(), dv.contiguous()
        for segments in reversed(plan.levels):
            _backward[(tiles, segments.numel(), heads)](
                kbar,
                w,
                decay,
                h,
                vnew,
                dh,
                dv,
                dk,
                dw,
                du,
                da,
                d_initial,
                plan.chunk_offsets,
                segments,
                plan.child_offsets,
                plan.children,
                nc,
                heads,
                k,
                v,
                triton.next_power_of_2(k),
                64,
                num_warps=8,
                num_stages=1,
            )
        return dk.sum(0), dw.sum(0), du, da.sum(0), None
