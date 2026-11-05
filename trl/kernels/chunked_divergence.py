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

# Python < 3.14 evaluates annotations at definition time, and `tl` (for `tl.constexpr`) only exists with Triton
from __future__ import annotations

import math

import torch
from transformers.utils import is_triton_available

from ..trainer.utils import maybe_gather_lm_head_ctx
from .chunked_logprob import _BLOCK_SIZE, _MM_OUT_DTYPE, _addmm_fp32, _jit, _torch_transform, _transform, _use_triton


if is_triton_available():
    import triton
    import triton.language as tl
    import triton.language.extra.libdevice as libdevice


# The projections run on `[TOKEN_CHUNK_SIZE, VOCAB_CHUNK_SIZE]` tiles, so neither model's logits exist in full
TOKEN_CHUNK_SIZE = 4096
VOCAB_CHUNK_SIZE = 32768


@_jit
def _logit_gap(zs, zt, s_ref, t_ref):
    # Compensated subtraction preserves both small student-teacher differences and independent logit shifts.
    delta = zs - zt
    error = tl.where(tl.abs(zs) >= tl.abs(zt), (zs - delta) - zt, zs - (delta + zt))
    shift = s_ref - t_ref
    shift_error = tl.where(tl.abs(s_ref) >= tl.abs(t_ref), (s_ref - shift) - t_ref, s_ref - (shift + t_ref))
    return (delta - shift) + (error - shift_error)


def _torch_logit_gap(zs, zt, s_ref, t_ref):
    # Compensated subtraction preserves both small student-teacher differences and independent logit shifts.
    delta = zs - zt
    error = torch.where(zs.abs() >= zt.abs(), (zs - delta) - zt, zs - (delta + zt))
    shift = s_ref - t_ref
    shift_error = torch.where(s_ref.abs() >= t_ref.abs(), (s_ref - shift) - t_ref, s_ref - (shift + t_ref))
    return (delta - shift) + (error - shift_error)


@_jit
def _log_student_mixture(ratio, log_beta, log1m_beta):
    # log(p_s / m) without subtracting nearly equal log-probabilities.
    a = log1m_beta
    b = log_beta - ratio
    top = tl.maximum(a, b)
    return tl.where(
        tl.abs(ratio) <= 0.5,
        -libdevice.log1p(tl.exp(log_beta) * libdevice.expm1(tl.minimum(tl.maximum(-ratio, -0.5), 0.5))),
        -top - tl.log(tl.exp(a - top) + tl.exp(b - top)),
    )


def _torch_log_student_mixture(ratio, log_beta, log1m_beta):
    # log(p_s / m) without subtracting nearly equal log-probabilities.
    return torch.where(
        ratio.abs() <= 0.5,
        -torch.log1p(math.exp(log_beta) * torch.expm1(-ratio.clamp(-0.5, 0.5))),
        -torch.logaddexp(torch.full_like(ratio, log1m_beta), log_beta - ratio),
    )


@_jit
def _stats_kernel(
    s_ptr,
    s_stride,
    t_ptr,
    t_stride,
    s_max_ptr,
    s_sum_ptr,
    s_x_sum_ptr,
    t_max_ptr,
    t_sum_ptr,
    t_weighted_gap_ptr,
    t_at_s_max_ptr,
    n_cols,
    s_scale,
    s_softcap,
    t_scale,
    t_softcap,
    inv_t,
    S_HAS_SOFTCAP: tl.constexpr,
    T_HAS_SOFTCAP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # One program per token: fold one vocabulary chunk of both models' logits into online-logsumexp statistics
    row = tl.program_id(0).to(tl.int64)
    s_row = s_ptr + row * s_stride
    t_row = t_ptr + row * t_stride
    s_m = tl.load(s_max_ptr + row)
    s_s = tl.load(s_sum_ptr + row)
    s_xs = tl.load(s_x_sum_ptr + row)
    t_m = tl.load(t_max_ptr + row)
    t_s = tl.load(t_sum_ptr + row)
    t_gap = tl.load(t_weighted_gap_ptr + row)
    t_at_s_max = tl.load(t_at_s_max_ptr + row)
    for start in range(0, n_cols, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        zs = _transform(
            tl.load(s_row + offsets, mask=mask, other=0.0).to(tl.float32), s_scale, s_softcap, inv_t, S_HAS_SOFTCAP
        )
        zt = _transform(
            tl.load(t_row + offsets, mask=mask, other=0.0).to(tl.float32), t_scale, t_softcap, inv_t, T_HAS_SOFTCAP
        )

        new_s_m = tl.maximum(s_m, tl.max(tl.where(mask, zs, -float("inf")), axis=0))
        s_rescale = tl.exp(s_m - new_s_m)
        es = tl.where(mask, tl.exp(zs - new_s_m), 0.0)
        # Keep one maximum's unit mass out of the sum until log1p.
        tile_s_max = tl.max(tl.where(mask, zs, -float("inf")), axis=0)
        tile_s_argmax = tl.min(tl.where(mask & (zs == tile_s_max), offsets, 2147483647), axis=0)
        t_at_s_max = tl.where(s_m < tile_s_max, tl.sum(tl.where(offsets == tile_s_argmax, zt, 0.0), 0), t_at_s_max)
        tile_s_sum = tl.sum(tl.where((s_m < tile_s_max) & (offsets == tile_s_argmax), 0.0, es), axis=0)
        s_shift = tl.where(s_m == -float("inf"), 0.0, s_m - new_s_m)
        s_xs = (s_xs + (1 + s_s) * s_shift) * s_rescale + tl.sum(es * tl.where(mask, zs - new_s_m, 0.0), axis=0)
        s_s = s_s * s_rescale + tl.where(s_m < tile_s_max, s_rescale, 0.0) + tile_s_sum

        new_t_m = tl.maximum(t_m, tl.max(tl.where(mask, zt, -float("inf")), axis=0))
        t_rescale = tl.exp(t_m - new_t_m)
        et = tl.where(mask, tl.exp(zt - new_t_m), 0.0)
        # Keep one maximum's unit mass out of the sum until log1p.
        tile_t_max = tl.max(tl.where(mask, zt, -float("inf")), axis=0)
        tile_t_argmax = tl.min(tl.where(mask & (zt == tile_t_max), offsets, 2147483647), axis=0)
        tile_t_sum = tl.sum(tl.where((t_m < tile_t_max) & (offsets == tile_t_argmax), 0.0, et), axis=0)
        # Changing either running maximum shifts every previously accumulated logit gap.
        gap_shift = tl.where(t_m == -float("inf"), 0.0, _logit_gap(t_m, s_m, new_t_m, new_s_m))
        gap = tl.where(mask, _logit_gap(zt, zs, new_t_m, new_s_m), 0.0)
        t_gap = (t_gap + (1 + t_s) * gap_shift) * t_rescale + tl.sum(et * gap, axis=0)
        t_s = t_s * t_rescale + tl.where(t_m < tile_t_max, t_rescale, 0.0) + tile_t_sum
        s_m = new_s_m
        t_m = new_t_m
    tl.store(s_max_ptr + row, s_m)
    tl.store(s_sum_ptr + row, s_s)
    tl.store(s_x_sum_ptr + row, s_xs)
    tl.store(t_max_ptr + row, t_m)
    tl.store(t_sum_ptr + row, t_s)
    tl.store(t_weighted_gap_ptr + row, t_gap)
    tl.store(t_at_s_max_ptr + row, t_at_s_max)


@_jit
def _ratio_kernel(
    s_ptr,
    s_stride,
    t_ptr,
    t_stride,
    s_max_ptr,
    t_max_ptr,
    t_log_z_ptr,
    s_log_z_ptr,
    center_ptr,
    correction_ptr,
    near_ptr,
    s_centered_gap_ptr,
    t_at_s_max_ptr,
    n_cols,
    s_scale,
    s_softcap,
    t_scale,
    t_softcap,
    inv_t,
    S_HAS_SOFTCAP: tl.constexpr,
    T_HAS_SOFTCAP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    center = tl.load(center_ptr + row)
    correction = tl.load(correction_ptr + row)
    near = tl.load(near_ptr + row)
    s_centered_gap = tl.load(s_centered_gap_ptr + row)
    for start in range(0, n_cols, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        zs = _transform(
            tl.load(s_ptr + row * s_stride + offsets, mask, 0).to(tl.float32), s_scale, s_softcap, inv_t, S_HAS_SOFTCAP
        )
        zt = _transform(
            tl.load(t_ptr + row * t_stride + offsets, mask, 0).to(tl.float32), t_scale, t_softcap, inv_t, T_HAS_SOFTCAP
        )
        s_max = tl.load(s_max_ptr + row)
        t_max = tl.load(t_max_ptr + row)
        d = _logit_gap(zs, zt, s_max, t_max) - center
        near = near & (tl.max(tl.where(mask, tl.abs(d), 0.0), 0) <= 0.5)
        p = tl.where(mask, tl.exp((zt - t_max) - tl.load(t_log_z_ptr + row)), 0.0)
        correction += tl.sum(p * libdevice.expm1(tl.minimum(d, 0.5)), 0)
        q = tl.where(mask, tl.exp((zs - s_max) - tl.load(s_log_z_ptr + row)), 0.0)
        # Center under the student's maximum so its small gradient never subtracts two large expectations.
        s_gap = _logit_gap(zs, zt, s_max, tl.load(t_at_s_max_ptr + row))
        s_centered_gap += tl.sum(q * tl.where(mask, s_gap, 0.0), 0)
    tl.store(correction_ptr + row, correction)
    tl.store(near_ptr + row, near)
    tl.store(s_centered_gap_ptr + row, s_centered_gap)


def _ratio_tile(
    s_tile,
    s_stride,
    t_tile,
    t_stride,
    s_max,
    t_max,
    t_log_z,
    s_log_z,
    center,
    correction,
    near,
    s_centered_gap,
    t_at_s_max,
    n_cols,
    s_scale,
    s_softcap,
    t_scale,
    t_softcap,
    inv_t,
    S_HAS_SOFTCAP,
    T_HAS_SOFTCAP,
    BLOCK_SIZE,
):
    zs = _torch_transform(s_tile, s_scale, s_softcap, inv_t, S_HAS_SOFTCAP)
    zt = _torch_transform(t_tile, t_scale, t_softcap, inv_t, T_HAS_SOFTCAP)
    d = _torch_logit_gap(zs, zt, s_max.unsqueeze(-1), t_max.unsqueeze(-1)) - center.unsqueeze(-1)
    near.logical_and_(d.abs().amax(-1) <= 0.5)
    p = torch.exp((zt - t_max.unsqueeze(-1)) - t_log_z.unsqueeze(-1))
    correction.add_((p * torch.expm1(d.clamp(max=0.5))).sum(-1))
    q = torch.exp((zs - s_max.unsqueeze(-1)) - s_log_z.unsqueeze(-1))
    # Center under the student's maximum so its small gradient never subtracts two large expectations.
    s_gap = _torch_logit_gap(zs, zt, s_max.unsqueeze(-1), t_at_s_max.unsqueeze(-1))
    s_centered_gap.add_((q * s_gap).sum(-1))


@_jit
def _jsd_kernel(
    s_ptr,
    s_stride,
    t_ptr,
    t_stride,
    s_max_ptr,
    t_max_ptr,
    s_log_z_ptr,
    t_log_z_ptr,
    center_ptr,
    correction_ptr,
    jsd_max_ratio_ptr,
    jsd_ptr,
    kl_student_ptr,
    n_cols,
    log_beta,
    log1m_beta,
    s_scale,
    s_softcap,
    t_scale,
    t_softcap,
    inv_t,
    S_HAS_SOFTCAP: tl.constexpr,
    T_HAS_SOFTCAP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Accumulate generalized JSD using the mixture m = (1 - beta) p_s + beta p_t
    # Before torch 2.15, torch.compile passes Python float arguments as fp64; keep the math in fp32
    log_beta = tl.cast(log_beta, tl.float32)
    log1m_beta = tl.cast(log1m_beta, tl.float32)
    s_scale = tl.cast(s_scale, tl.float32)
    s_softcap = tl.cast(s_softcap, tl.float32)
    t_scale = tl.cast(t_scale, tl.float32)
    t_softcap = tl.cast(t_softcap, tl.float32)
    inv_t = tl.cast(inv_t, tl.float32)
    row = tl.program_id(0).to(tl.int64)
    s_row = s_ptr + row * s_stride
    t_row = t_ptr + row * t_stride
    s_log_z = tl.load(s_log_z_ptr + row)
    t_log_z = tl.load(t_log_z_ptr + row)
    jsd = tl.load(jsd_ptr + row)
    kl_student = tl.load(kl_student_ptr + row)
    for start in range(0, n_cols, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        zs = _transform(
            tl.load(s_row + offsets, mask=mask, other=0.0).to(tl.float32), s_scale, s_softcap, inv_t, S_HAS_SOFTCAP
        )
        zt = _transform(
            tl.load(t_row + offsets, mask=mask, other=0.0).to(tl.float32), t_scale, t_softcap, inv_t, T_HAS_SOFTCAP
        )
        s_max = tl.load(s_max_ptr + row)
        t_max = tl.load(t_max_ptr + row)
        log_ps = (zs - s_max) - s_log_z
        log_pt = (zt - t_max) - t_log_z
        ratio = _logit_gap(zs, zt, s_max, t_max) - tl.load(center_ptr + row) - tl.load(correction_ptr + row)
        log_s_m = _log_student_mixture(ratio, log_beta, log1m_beta)
        a = log1m_beta + ratio
        b = log_beta
        top = tl.maximum(a, b)
        log_t_m = tl.where(
            tl.abs(ratio) <= 0.5,
            -libdevice.log1p(tl.exp(log1m_beta) * libdevice.expm1(tl.minimum(tl.maximum(ratio, -0.5), 0.5))),
            -top - tl.log(tl.exp(a - top) + tl.exp(b - top)),
        )
        ps = tl.where(mask, tl.exp(log_ps), 0.0)
        pt = tl.where(mask, tl.exp(log_pt), 0.0)
        kl_s = ps * tl.where(mask, log_s_m, 0.0)
        jsd += tl.sum(tl.exp(log_beta) * pt * tl.where(mask, log_t_m, 0.0) + tl.exp(log1m_beta) * kl_s, axis=0)
        anchor = _log_student_mixture(tl.load(jsd_max_ratio_ptr + row), log_beta, log1m_beta)
        kl_student += tl.sum(ps * tl.where(mask, log_s_m - anchor, 0.0), axis=0)
    tl.store(jsd_ptr + row, jsd)
    tl.store(kl_student_ptr + row, kl_student)


@_jit
def _backward_kernel(
    s_ptr,
    s_stride,
    t_ptr,
    t_stride,
    s_max_ptr,
    t_max_ptr,
    s_log_z_ptr,
    t_log_z_ptr,
    center_ptr,
    correction_ptr,
    aux_ptr,
    grad_ptr,
    n_cols,
    log_beta,
    log1m_beta,
    s_scale,
    s_softcap,
    t_scale,
    t_softcap,
    inv_t,
    MODE: tl.constexpr,
    S_HAS_SOFTCAP: tl.constexpr,
    T_HAS_SOFTCAP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Overwrite one block of the student tile with the gradient of the divergence with respect to the projection output
    # Before torch 2.15, torch.compile passes Python float arguments as fp64; keep the math in fp32
    log_beta = tl.cast(log_beta, tl.float32)
    log1m_beta = tl.cast(log1m_beta, tl.float32)
    s_scale = tl.cast(s_scale, tl.float32)
    s_softcap = tl.cast(s_softcap, tl.float32)
    t_scale = tl.cast(t_scale, tl.float32)
    t_softcap = tl.cast(t_softcap, tl.float32)
    inv_t = tl.cast(inv_t, tl.float32)
    row = tl.program_id(0).to(tl.int64)
    offsets = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    s_ptrs = s_ptr + row * s_stride + offsets
    y = tl.load(s_ptrs, mask=mask, other=0.0).to(tl.float32) * s_scale
    if S_HAS_SOFTCAP:
        th = 2 * tl.sigmoid(2 * y / s_softcap) - 1
        zs = s_softcap * th * inv_t
    else:
        zs = y * inv_t
    zt = _transform(
        tl.load(t_ptr + row * t_stride + offsets, mask=mask, other=0.0).to(tl.float32),
        t_scale,
        t_softcap,
        inv_t,
        T_HAS_SOFTCAP,
    )
    s_max = tl.load(s_max_ptr + row)
    t_max = tl.load(t_max_ptr + row)
    log_ps = (zs - s_max) - tl.load(s_log_z_ptr + row)
    log_pt = (zt - t_max) - tl.load(t_log_z_ptr + row)
    ps = tl.exp(log_ps)
    ratio = _logit_gap(zs, zt, s_max, t_max) - tl.load(center_ptr + row) - tl.load(correction_ptr + row)
    if MODE == 0:
        # forward KL(p_t || p_s)
        grad = tl.where(
            ratio <= 0,
            tl.exp(log_pt) * libdevice.expm1(tl.minimum(ratio, 0.0)),
            -ps * libdevice.expm1(tl.minimum(-ratio, 0.0)),
        )
    elif MODE == 1:
        # reverse KL(p_s || p_t), `aux` holds the teacher logit at the student maximum and the centered expectation
        gap = _logit_gap(zs, zt, s_max, tl.load(aux_ptr + 2 * row))
        grad = ps * (gap - tl.load(aux_ptr + 2 * row + 1))
    else:
        # generalized JSD, `aux` holds log(p_s / p_t) at the student maximum and the centered expectation
        log_s_m = _log_student_mixture(ratio, log_beta, log1m_beta)
        anchor = _log_student_mixture(tl.load(aux_ptr + 2 * row), log_beta, log1m_beta)
        grad = tl.exp(log1m_beta) * ps * (log_s_m - anchor - tl.load(aux_ptr + 2 * row + 1))
    grad = grad * tl.load(grad_ptr + row) * inv_t
    if S_HAS_SOFTCAP:
        grad = grad * (1 - th * th)
    grad = grad * s_scale
    tl.store(s_ptrs, grad.to(s_ptr.dtype.element_ty), mask=mask)


def _stats_tile(
    s_tile,
    s_stride,
    t_tile,
    t_stride,
    s_max,
    s_sum,
    s_x_sum,
    t_max,
    t_sum,
    t_weighted_gap,
    t_at_s_max,
    n_cols,
    s_scale,
    s_softcap,
    t_scale,
    t_softcap,
    inv_t,
    S_HAS_SOFTCAP,
    T_HAS_SOFTCAP,
    BLOCK_SIZE,
):
    # `_stats_kernel` in torch, for devices without Triton: the whole tile at once instead of one program per token
    zs = _torch_transform(s_tile, s_scale, s_softcap, inv_t, S_HAS_SOFTCAP)
    zt = _torch_transform(t_tile, t_scale, t_softcap, inv_t, T_HAS_SOFTCAP)

    new_s_m = torch.maximum(s_max, zs.amax(-1))
    s_rescale = torch.exp(s_max - new_s_m)
    es = torch.exp(zs - new_s_m.unsqueeze(-1))
    tile_s_max, tile_s_argmax = zs.max(-1)
    new_s_maximum = s_max < tile_s_max
    t_at_s_max.copy_(torch.where(new_s_maximum, zt.gather(1, tile_s_argmax.unsqueeze(-1)).squeeze(-1), t_at_s_max))
    other_es = es.clone()
    other_es.scatter_(
        1,
        tile_s_argmax.unsqueeze(-1),
        torch.where(new_s_maximum.unsqueeze(-1), 0.0, es.gather(1, tile_s_argmax.unsqueeze(-1))),
    )
    s_shift = torch.where(torch.isfinite(s_max), s_max - new_s_m, 0.0)
    s_x_sum.add_((1 + s_sum) * s_shift).mul_(s_rescale).add_((es * (zs - new_s_m.unsqueeze(-1))).sum(-1))
    s_sum.mul_(s_rescale).add_(torch.where(new_s_maximum, s_rescale, 0.0)).add_(other_es.sum(-1))

    new_t_m = torch.maximum(t_max, zt.amax(-1))
    t_rescale = torch.exp(t_max - new_t_m)
    et = torch.exp(zt - new_t_m.unsqueeze(-1))
    tile_t_max, tile_t_argmax = zt.max(-1)
    new_t_maximum = t_max < tile_t_max
    other_et = et.clone()
    other_et.scatter_(
        1,
        tile_t_argmax.unsqueeze(-1),
        torch.where(new_t_maximum.unsqueeze(-1), 0.0, et.gather(1, tile_t_argmax.unsqueeze(-1))),
    )
    # Changing either running maximum shifts every previously accumulated logit gap.
    gap_shift = torch.where(torch.isfinite(t_max), _torch_logit_gap(t_max, s_max, new_t_m, new_s_m), 0.0)
    gap = _torch_logit_gap(zt, zs, new_t_m.unsqueeze(-1), new_s_m.unsqueeze(-1))
    t_weighted_gap.add_((1 + t_sum) * gap_shift).mul_(t_rescale).add_((et * gap).sum(-1))
    t_sum.mul_(t_rescale).add_(torch.where(new_t_maximum, t_rescale, 0.0)).add_(other_et.sum(-1))
    s_max.copy_(new_s_m)
    t_max.copy_(new_t_m)


def _jsd_tile(
    s_tile,
    s_stride,
    t_tile,
    t_stride,
    s_max,
    t_max,
    s_log_z,
    t_log_z,
    center,
    correction,
    jsd_max_ratio,
    jsd,
    kl_student,
    n_cols,
    log_beta,
    log1m_beta,
    s_scale,
    s_softcap,
    t_scale,
    t_softcap,
    inv_t,
    S_HAS_SOFTCAP,
    T_HAS_SOFTCAP,
    BLOCK_SIZE,
):
    # `_jsd_kernel` in torch, for devices without Triton: the whole tile at once instead of one program per token
    zs = _torch_transform(s_tile, s_scale, s_softcap, inv_t, S_HAS_SOFTCAP)
    zt = _torch_transform(t_tile, t_scale, t_softcap, inv_t, T_HAS_SOFTCAP)
    log_ps = (zs - s_max.unsqueeze(-1)) - s_log_z.unsqueeze(-1)
    log_pt = (zt - t_max.unsqueeze(-1)) - t_log_z.unsqueeze(-1)
    ratio = _torch_logit_gap(zs, zt, s_max.unsqueeze(-1), t_max.unsqueeze(-1))
    ratio = ratio - center.unsqueeze(-1) - correction.unsqueeze(-1)
    log_s_m = _torch_log_student_mixture(ratio, log_beta, log1m_beta)
    log_t_m = torch.where(
        ratio.abs() <= 0.5,
        -torch.log1p(math.exp(log1m_beta) * torch.expm1(ratio.clamp(-0.5, 0.5))),
        -torch.logaddexp(log1m_beta + ratio, torch.full_like(ratio, log_beta)),
    )
    kl_s = torch.exp(log_ps) * log_s_m
    jsd += (math.exp(log_beta) * torch.exp(log_pt) * log_t_m + math.exp(log1m_beta) * kl_s).sum(-1)
    anchor = _torch_log_student_mixture(jsd_max_ratio, log_beta, log1m_beta)
    kl_student += (torch.exp(log_ps) * (log_s_m - anchor.unsqueeze(-1))).sum(-1)


def _backward_tile(
    s_tile,
    s_stride,
    t_tile,
    t_stride,
    s_max,
    t_max,
    s_log_z,
    t_log_z,
    center,
    correction,
    aux,
    grad_divergence,
    n_cols,
    log_beta,
    log1m_beta,
    s_scale,
    s_softcap,
    t_scale,
    t_softcap,
    inv_t,
    MODE,
    S_HAS_SOFTCAP,
    T_HAS_SOFTCAP,
    BLOCK_SIZE,
):
    # `_backward_kernel` in torch, for devices without Triton: the whole tile at once instead of one block per program
    y = s_tile * s_scale
    if S_HAS_SOFTCAP:
        th = torch.tanh(y / s_softcap)
        zs = s_softcap * th * inv_t
    else:
        zs = y * inv_t
    zt = _torch_transform(t_tile, t_scale, t_softcap, inv_t, T_HAS_SOFTCAP)
    log_ps = (zs - s_max.unsqueeze(-1)) - s_log_z.unsqueeze(-1)
    log_pt = (zt - t_max.unsqueeze(-1)) - t_log_z.unsqueeze(-1)
    ps = torch.exp(log_ps)
    ratio = _torch_logit_gap(zs, zt, s_max.unsqueeze(-1), t_max.unsqueeze(-1))
    ratio = ratio - center.unsqueeze(-1) - correction.unsqueeze(-1)
    if MODE == 0:
        # forward KL(p_t || p_s)
        grad = torch.where(
            ratio <= 0, torch.exp(log_pt) * torch.expm1(ratio.clamp(max=0)), -ps * torch.expm1((-ratio).clamp(max=0))
        )
    elif MODE == 1:
        # reverse KL(p_s || p_t), `aux` holds the teacher logit at the student maximum and the centered expectation
        gap = _torch_logit_gap(zs, zt, s_max.unsqueeze(-1), aux[:, :1])
        grad = ps * (gap - aux[:, 1:])
    else:
        # generalized JSD, `aux` holds log(p_s / p_t) at the student maximum and the centered expectation
        log_s_m = _torch_log_student_mixture(ratio, log_beta, log1m_beta)
        anchor = _torch_log_student_mixture(aux[:, :1], log_beta, log1m_beta)
        grad = math.exp(log1m_beta) * ps * (log_s_m - anchor - aux[:, 1:])
    grad = grad * grad_divergence.unsqueeze(-1) * inv_t
    if S_HAS_SOFTCAP:
        grad = grad * (1 - th * th)
    s_tile.copy_(grad * s_scale)


def _compute_dtype(hidden: torch.Tensor) -> torch.dtype:
    device_type = hidden.device.type
    return torch.get_autocast_dtype(device_type) if torch.is_autocast_enabled(device_type) else hidden.dtype


def _project(tile, hidden_chunk, weight, bias, start, end, dtype):
    # FP32 projection output avoids rounding logits before computing the divergence.
    w = weight[start:end].to(dtype).t()
    if _MM_OUT_DTYPE and tile.device.type == "cuda" and dtype in (torch.float16, torch.bfloat16):
        torch.mm(hidden_chunk, w, out_dtype=torch.float32, out=tile)
    else:
        torch.mm(hidden_chunk.float(), w.float(), out=tile)
    if bias is not None:
        tile.add_(bias[start:end].float())


class ChunkedDivergenceFunction(torch.autograd.Function):
    """
    Per-token divergence between the student's and the teacher's next-token distributions, and the student's entropy,
    computed from both models' hidden states and LM heads without materializing either `[N, V]` logits.

    `beta` selects the divergence: `0.0` is the forward KL(teacher || student), `1.0` the reverse KL(student ||
    teacher), and anything in between the generalized JSD with mixture `(1 - beta) * p_student + beta * p_teacher`. The
    gradient flows to the student only; the entropy has none.

    Following https://huggingface.co/papers/2610.31642, normalization excludes one maximum and a second pass centers
    student-teacher logit differences before using log1p/expm1. This preserves small gradients near teacher agreement
    and at high confidence, while keeping memory bounded by the projection tiles.
    """

    @staticmethod
    def forward(
        ctx,
        student_hidden: torch.Tensor,  # [N, H_s]
        student_weight: torch.Tensor,  # [V, H_s]
        student_bias: torch.Tensor | None,
        teacher_hidden: torch.Tensor,  # [N, H_t]
        teacher_weight: torch.Tensor,  # [V, H_t]
        teacher_bias: torch.Tensor | None,
        beta: float,
        temperature: float = 1.0,
        student_logit_scale: float = 1.0,
        student_final_logit_softcapping: float | None = None,
        teacher_logit_scale: float = 1.0,
        teacher_final_logit_softcapping: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ctx.set_materialize_grads(False)
        device = student_hidden.device
        N = student_hidden.shape[0]
        s_dtype, t_dtype = _compute_dtype(student_hidden), _compute_dtype(teacher_hidden)
        kernel_args = {
            "s_scale": student_logit_scale,
            "s_softcap": student_final_logit_softcapping or 1.0,
            "t_scale": teacher_logit_scale,
            "t_softcap": teacher_final_logit_softcapping or 1.0,
            "inv_t": 1 / temperature,
            "S_HAS_SOFTCAP": student_final_logit_softcapping is not None,
            "T_HAS_SOFTCAP": teacher_final_logit_softcapping is not None,
        }

        def zeros():
            return torch.zeros((N,), device=device, dtype=torch.float32)

        s_max, t_max = zeros() - math.inf, zeros() - math.inf
        s_sum, s_x_sum, t_sum, t_weighted_gap = zeros(), zeros(), zeros(), zeros()
        jsd, kl_student, jsd_max_ratio = zeros(), zeros(), zeros()
        center, correction = zeros(), zeros()
        t_at_s_max, s_centered_gap = zeros(), zeros()
        near = torch.ones((N,), device=device, dtype=torch.bool)
        rows = min(N, TOKEN_CHUNK_SIZE)
        s_buf = torch.empty((rows, VOCAB_CHUNK_SIZE), device=device, dtype=torch.float32)
        t_buf = torch.empty((rows, VOCAB_CHUNK_SIZE), device=device, dtype=torch.float32)
        is_jsd = 0.0 < beta < 1.0
        # Taken from the float64 beta: near an endpoint, rounding beta to float32 first moves log(1 - beta) by ~1%
        log_beta, log1m_beta = (math.log(beta), math.log1p(-beta)) if is_jsd else (0.0, 0.0)

        with (
            maybe_gather_lm_head_ctx(student_weight, student_bias),
            maybe_gather_lm_head_ctx(teacher_weight, teacher_bias),
        ):
            vocab = student_weight.shape[0]
            for token_start in range(0, N, TOKEN_CHUNK_SIZE):
                token_end = min(token_start + TOKEN_CHUNK_SIZE, N)
                n = token_end - token_start
                h_s = student_hidden[token_start:token_end].to(s_dtype)
                h_t = teacher_hidden[token_start:token_end].to(t_dtype)
                sl = slice(token_start, token_end)
                for start in range(0, vocab, VOCAB_CHUNK_SIZE):
                    end = min(start + VOCAB_CHUNK_SIZE, vocab)
                    s_tile, t_tile = s_buf[:n, : end - start], t_buf[:n, : end - start]
                    _project(s_tile, h_s, student_weight, student_bias, start, end, s_dtype)
                    _project(t_tile, h_t, teacher_weight, teacher_bias, start, end, t_dtype)
                    fold = _stats_kernel[(n,)] if _use_triton(device) else _stats_tile
                    fold(
                        s_tile,
                        s_tile.stride(0),
                        t_tile,
                        t_tile.stride(0),
                        s_max[sl],
                        s_sum[sl],
                        s_x_sum[sl],
                        t_max[sl],
                        t_sum[sl],
                        t_weighted_gap[sl],
                        t_at_s_max[sl],
                        end - start,
                        BLOCK_SIZE=_BLOCK_SIZE,
                        **kernel_args,
                    )
                s_log_z = torch.log1p(s_sum[sl])
                t_log_z = torch.log1p(t_sum[sl])
                # Center the student-teacher logit difference under the teacher. Close distributions need
                # log1p/expm1. Outside the small-gap region, use separate maximum and log1p normalizers instead;
                # bounding expm1's argument avoids overflow for distant distributions.
                center[sl] = -t_weighted_gap[sl] / (1 + t_sum[sl])
                # Traverse backwards to reuse the last projected tile from the statistics pass.
                for start in reversed(range(0, vocab, VOCAB_CHUNK_SIZE)):
                    end = min(start + VOCAB_CHUNK_SIZE, vocab)
                    s_tile, t_tile = s_buf[:n, : end - start], t_buf[:n, : end - start]
                    if end != vocab:
                        _project(s_tile, h_s, student_weight, student_bias, start, end, s_dtype)
                        _project(t_tile, h_t, teacher_weight, teacher_bias, start, end, t_dtype)
                    fold = _ratio_kernel[(n,)] if _use_triton(device) else _ratio_tile
                    fold(
                        s_tile,
                        s_tile.stride(0),
                        t_tile,
                        t_tile.stride(0),
                        s_max[sl],
                        t_max[sl],
                        t_log_z,
                        s_log_z,
                        center[sl],
                        correction[sl],
                        near[sl],
                        s_centered_gap[sl],
                        t_at_s_max[sl],
                        end - start,
                        BLOCK_SIZE=_BLOCK_SIZE,
                        **kernel_args,
                    )
                correction[sl] = torch.where(
                    near[sl], torch.log1p(correction[sl]), torch.log1p(s_sum[sl]) - torch.log1p(t_sum[sl])
                )
                center[sl] = torch.where(near[sl], center[sl], 0.0)
                if is_jsd:
                    # Center log(p_s / m) at the student's maximum before accumulating its expectation.
                    jsd_max_ratio[sl] = (t_max[sl] - t_at_s_max[sl]) - center[sl] - correction[sl]
                    for start in range(0, vocab, VOCAB_CHUNK_SIZE):
                        end = min(start + VOCAB_CHUNK_SIZE, vocab)
                        s_tile, t_tile = s_buf[:n, : end - start], t_buf[:n, : end - start]
                        # The ratio pass leaves the first tile in the buffers.
                        if start != 0:
                            _project(s_tile, h_s, student_weight, student_bias, start, end, s_dtype)
                            _project(t_tile, h_t, teacher_weight, teacher_bias, start, end, t_dtype)
                        fold = _jsd_kernel[(n,)] if _use_triton(device) else _jsd_tile
                        fold(
                            s_tile,
                            s_tile.stride(0),
                            t_tile,
                            t_tile.stride(0),
                            s_max[sl],
                            t_max[sl],
                            s_log_z,
                            t_log_z,
                            center[sl],
                            correction[sl],
                            jsd_max_ratio[sl],
                            jsd[sl],
                            kl_student[sl],
                            end - start,
                            log_beta,
                            log1m_beta,
                            BLOCK_SIZE=_BLOCK_SIZE,
                            **kernel_args,
                        )

        s_log_z = torch.log1p(s_sum)
        t_log_z = torch.log1p(t_sum)
        entropy = s_log_z - s_x_sum / (1 + s_sum)
        if beta == 0.0:
            divergence = (center + t_weighted_gap / (1 + t_sum)) + correction
            aux = divergence
        elif beta == 1.0:
            divergence = ((t_max - t_at_s_max) - center) + s_centered_gap - correction
            aux = torch.stack((t_at_s_max, s_centered_gap), dim=-1)
        else:
            divergence = jsd
            aux = torch.stack((jsd_max_ratio, kl_student), dim=-1)

        ctx.save_for_backward(
            student_hidden,
            student_weight,
            student_bias,
            teacher_hidden,
            teacher_weight,
            teacher_bias,
            s_max,
            t_max,
            s_log_z,
            t_log_z,
            center,
            correction,
            aux,
        )
        ctx.dtypes = s_dtype, t_dtype
        ctx.log_betas = log_beta, log1m_beta
        ctx.mode = 0 if beta == 0.0 else 1 if beta == 1.0 else 2
        ctx.kernel_args = kernel_args
        ctx.mark_non_differentiable(entropy)
        return divergence, entropy

    @staticmethod
    def backward(ctx, grad_divergence: torch.Tensor | None, _):  # type: ignore
        (
            student_hidden,
            student_weight,
            student_bias,
            teacher_hidden,
            teacher_weight,
            teacher_bias,
            s_max,
            t_max,
            s_log_z,
            t_log_z,
            center,
            correction,
            aux,
        ) = ctx.saved_tensors
        s_dtype, t_dtype = ctx.dtypes
        needs_hidden_grad, needs_weight_grad, needs_bias_grad = ctx.needs_input_grad[:3]
        N = student_hidden.shape[0]
        grad_hidden = grad_weight = grad_bias = None
        if grad_divergence is None:
            return (None,) * 12
        grad_divergence = grad_divergence.float().contiguous()
        with (
            maybe_gather_lm_head_ctx(student_weight, student_bias),
            maybe_gather_lm_head_ctx(teacher_weight, teacher_bias),
        ):
            vocab = student_weight.shape[0]
            # Always accumulate in fp32, even when the inputs are not
            if needs_hidden_grad:
                grad_hidden = torch.zeros(student_hidden.shape, device=student_hidden.device, dtype=torch.float32)
            if needs_weight_grad:
                grad_weight = torch.zeros(student_weight.shape, device=student_weight.device, dtype=torch.float32)
            if needs_bias_grad:
                grad_bias = torch.zeros(student_bias.shape, device=student_bias.device, dtype=torch.float32)
            rows = min(N, TOKEN_CHUNK_SIZE)
            s_buf = torch.empty((rows, VOCAB_CHUNK_SIZE), device=student_hidden.device, dtype=torch.float32)
            t_buf = torch.empty((rows, VOCAB_CHUNK_SIZE), device=student_hidden.device, dtype=torch.float32)
            for token_start in range(0, N, TOKEN_CHUNK_SIZE):
                token_end = min(token_start + TOKEN_CHUNK_SIZE, N)
                n = token_end - token_start
                sl = slice(token_start, token_end)
                h_s = student_hidden[sl].to(s_dtype)
                h_t = teacher_hidden[sl].to(t_dtype)
                for start in range(0, vocab, VOCAB_CHUNK_SIZE):
                    end = min(start + VOCAB_CHUNK_SIZE, vocab)
                    s_tile, t_tile = s_buf[:n, : end - start], t_buf[:n, : end - start]
                    _project(s_tile, h_s, student_weight, student_bias, start, end, s_dtype)
                    _project(t_tile, h_t, teacher_weight, teacher_bias, start, end, t_dtype)
                    fold = (
                        _backward_kernel[(n, triton.cdiv(end - start, _BLOCK_SIZE))]
                        if _use_triton(student_hidden.device)
                        else _backward_tile
                    )
                    fold(
                        s_tile,
                        s_tile.stride(0),
                        t_tile,
                        t_tile.stride(0),
                        s_max[sl],
                        t_max[sl],
                        s_log_z[sl],
                        t_log_z[sl],
                        center[sl],
                        correction[sl],
                        aux[sl],
                        grad_divergence[sl],
                        end - start,
                        *ctx.log_betas,
                        MODE=ctx.mode,
                        BLOCK_SIZE=_BLOCK_SIZE,
                        **ctx.kernel_args,
                    )
                    grad_tile = s_tile.to(s_dtype)
                    if grad_hidden is not None:
                        _addmm_fp32(grad_hidden[sl], grad_tile, student_weight[start:end].to(s_dtype))
                    if grad_weight is not None:
                        _addmm_fp32(grad_weight[start:end], grad_tile.t(), h_s)
                    if grad_bias is not None:
                        grad_bias[start:end] += s_tile.sum(dim=0, dtype=torch.float32)

        return (
            grad_hidden.to(student_hidden.dtype) if grad_hidden is not None else None,
            grad_weight.to(student_weight.dtype) if grad_weight is not None else None,
            grad_bias.to(student_bias.dtype) if grad_bias is not None else None,
            *(None,) * 9,
        )
