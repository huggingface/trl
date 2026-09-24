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

import math

import torch
import triton
import triton.language as tl

from ..trainer.utils import maybe_gather_lm_head_ctx
from .chunked_logprob import _BLOCK_SIZE, _transform


# The projections run on `[TOKEN_CHUNK_SIZE, VOCAB_CHUNK_SIZE]` tiles, so neither model's logits exist in full
TOKEN_CHUNK_SIZE = 2048
VOCAB_CHUNK_SIZE = 8192


@triton.jit
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
    s_weighted_gap_ptr,
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
    s_gap = tl.load(s_weighted_gap_ptr + row)
    for start in range(0, n_cols, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        zs = _transform(
            tl.load(s_row + offsets, mask=mask, other=0.0).to(tl.float32), s_scale, s_softcap, inv_t, S_HAS_SOFTCAP
        )
        zt = _transform(
            tl.load(t_row + offsets, mask=mask, other=0.0).to(tl.float32), t_scale, t_softcap, inv_t, T_HAS_SOFTCAP
        )
        gap = tl.where(mask, zt - zs, 0.0)

        new_s_m = tl.maximum(s_m, tl.max(tl.where(mask, zs, -float("inf")), axis=0))
        s_rescale = tl.exp(s_m - new_s_m)
        es = tl.where(mask, tl.exp(zs - new_s_m), 0.0)
        s_s = s_s * s_rescale + tl.sum(es, axis=0)
        s_xs = s_xs * s_rescale + tl.sum(es * tl.where(mask, zs, 0.0), axis=0)
        s_gap = s_gap * s_rescale - tl.sum(es * gap, axis=0)
        s_m = new_s_m

        new_t_m = tl.maximum(t_m, tl.max(tl.where(mask, zt, -float("inf")), axis=0))
        t_rescale = tl.exp(t_m - new_t_m)
        et = tl.where(mask, tl.exp(zt - new_t_m), 0.0)
        t_s = t_s * t_rescale + tl.sum(et, axis=0)
        t_gap = t_gap * t_rescale + tl.sum(et * gap, axis=0)
        t_m = new_t_m
    tl.store(s_max_ptr + row, s_m)
    tl.store(s_sum_ptr + row, s_s)
    tl.store(s_x_sum_ptr + row, s_xs)
    tl.store(t_max_ptr + row, t_m)
    tl.store(t_sum_ptr + row, t_s)
    tl.store(t_weighted_gap_ptr + row, t_gap)
    tl.store(s_weighted_gap_ptr + row, s_gap)


@triton.jit
def _jsd_kernel(
    s_ptr,
    s_stride,
    t_ptr,
    t_stride,
    s_log_z_ptr,
    t_log_z_ptr,
    jsd_ptr,
    kl_student_ptr,
    n_cols,
    beta,
    s_scale,
    s_softcap,
    t_scale,
    t_softcap,
    inv_t,
    S_HAS_SOFTCAP: tl.constexpr,
    T_HAS_SOFTCAP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Second pass of the generalized JSD, which needs both normalizers to form the mixture m = (1 - beta) p_s + beta p_t
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
        log_ps = zs - s_log_z
        log_pt = zt - t_log_z
        a = log_ps + tl.log(1 - beta)
        b = log_pt + tl.log(beta)
        top = tl.maximum(a, b)
        log_m = top + tl.log(tl.exp(a - top) + tl.exp(b - top))
        ps = tl.where(mask, tl.exp(log_ps), 0.0)
        pt = tl.where(mask, tl.exp(log_pt), 0.0)
        kl_s = ps * tl.where(mask, log_ps - log_m, 0.0)
        jsd += tl.sum(beta * pt * tl.where(mask, log_pt - log_m, 0.0) + (1 - beta) * kl_s, axis=0)
        kl_student += tl.sum(kl_s, axis=0)
    tl.store(jsd_ptr + row, jsd)
    tl.store(kl_student_ptr + row, kl_student)


@triton.jit
def _backward_kernel(
    s_ptr,
    s_stride,
    t_ptr,
    t_stride,
    s_log_z_ptr,
    t_log_z_ptr,
    aux_ptr,
    grad_ptr,
    n_cols,
    beta,
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
    log_ps = zs - tl.load(s_log_z_ptr + row)
    log_pt = zt - tl.load(t_log_z_ptr + row)
    ps = tl.exp(log_ps)
    if MODE == 0:
        # forward KL(p_t || p_s)
        grad = ps - tl.exp(log_pt)
    elif MODE == 1:
        # reverse KL(p_s || p_t), `aux` is the per-token KL
        grad = ps * (log_ps - log_pt - tl.load(aux_ptr + row))
    else:
        # generalized JSD, `aux` is the per-token KL(p_s || m)
        a = log_ps + tl.log(1 - beta)
        b = log_pt + tl.log(beta)
        top = tl.maximum(a, b)
        log_m = top + tl.log(tl.exp(a - top) + tl.exp(b - top))
        grad = (1 - beta) * ps * (log_ps - log_m - tl.load(aux_ptr + row))
    grad = grad * tl.load(grad_ptr + row) * inv_t
    if S_HAS_SOFTCAP:
        grad = grad * (1 - th * th)
    grad = grad * s_scale
    tl.store(s_ptrs, grad.to(s_ptr.dtype.element_ty), mask=mask)


def _compute_dtype(hidden: torch.Tensor) -> torch.dtype:
    device_type = hidden.device.type
    return torch.get_autocast_dtype(device_type) if torch.is_autocast_enabled(device_type) else hidden.dtype


def _project(tile, hidden_chunk, weight, bias, start, end, dtype):
    torch.mm(hidden_chunk, weight[start:end].to(dtype).t(), out=tile)
    if bias is not None:
        tile.add_(bias[start:end].to(dtype))


class ChunkedDivergenceFunction(torch.autograd.Function):
    """
    Per-token divergence between the student's and the teacher's next-token distributions, and the student's entropy,
    computed from both models' hidden states and LM heads without materializing either `[N, V]` logits.

    `beta` selects the divergence: `0.0` is the forward KL(teacher || student), `1.0` the reverse KL(student ||
    teacher), and anything in between the generalized JSD with mixture `(1 - beta) * p_student + beta * p_teacher`. The
    gradient flows to the student only; the entropy has none.
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
        vocab = student_weight.shape[0]
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
        s_sum, s_x_sum, t_sum, t_weighted_gap, s_weighted_gap = zeros(), zeros(), zeros(), zeros(), zeros()
        jsd, kl_student = zeros(), zeros()
        rows = min(N, TOKEN_CHUNK_SIZE)
        s_buf = torch.empty((rows, VOCAB_CHUNK_SIZE), device=device, dtype=s_dtype)
        t_buf = torch.empty((rows, VOCAB_CHUNK_SIZE), device=device, dtype=t_dtype)
        is_jsd = 0.0 < beta < 1.0

        with (
            maybe_gather_lm_head_ctx(student_weight, student_bias),
            maybe_gather_lm_head_ctx(teacher_weight, teacher_bias),
        ):
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
                    _stats_kernel[(n,)](
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
                        s_weighted_gap[sl],
                        end - start,
                        BLOCK_SIZE=_BLOCK_SIZE,
                        **kernel_args,
                    )
                s_log_z = s_max[sl] + torch.log(s_sum[sl])
                t_log_z = t_max[sl] + torch.log(t_sum[sl])
                if is_jsd:
                    for start in range(0, vocab, VOCAB_CHUNK_SIZE):
                        end = min(start + VOCAB_CHUNK_SIZE, vocab)
                        s_tile, t_tile = s_buf[:n, : end - start], t_buf[:n, : end - start]
                        _project(s_tile, h_s, student_weight, student_bias, start, end, s_dtype)
                        _project(t_tile, h_t, teacher_weight, teacher_bias, start, end, t_dtype)
                        _jsd_kernel[(n,)](
                            s_tile,
                            s_tile.stride(0),
                            t_tile,
                            t_tile.stride(0),
                            s_log_z,
                            t_log_z,
                            jsd[sl],
                            kl_student[sl],
                            end - start,
                            beta,
                            BLOCK_SIZE=_BLOCK_SIZE,
                            **kernel_args,
                        )

        s_log_z = s_max + torch.log(s_sum)
        t_log_z = t_max + torch.log(t_sum)
        entropy = s_log_z - s_x_sum / s_sum
        if beta == 0.0:
            divergence = t_weighted_gap / t_sum - t_log_z + s_log_z
            aux = divergence
        elif beta == 1.0:
            divergence = s_weighted_gap / s_sum - s_log_z + t_log_z
            aux = divergence
        else:
            divergence = jsd
            aux = kl_student

        ctx.save_for_backward(
            student_hidden,
            student_weight,
            student_bias,
            teacher_hidden,
            teacher_weight,
            teacher_bias,
            s_log_z,
            t_log_z,
            aux,
        )
        ctx.dtypes = s_dtype, t_dtype
        ctx.beta = beta
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
            s_log_z,
            t_log_z,
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
            s_buf = torch.empty((rows, VOCAB_CHUNK_SIZE), device=student_hidden.device, dtype=s_dtype)
            t_buf = torch.empty((rows, VOCAB_CHUNK_SIZE), device=student_hidden.device, dtype=t_dtype)
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
                    _backward_kernel[(n, triton.cdiv(end - start, _BLOCK_SIZE))](
                        s_tile,
                        s_tile.stride(0),
                        t_tile,
                        t_tile.stride(0),
                        s_log_z[sl],
                        t_log_z[sl],
                        aux[sl],
                        grad_divergence[sl],
                        end - start,
                        ctx.beta,
                        MODE=ctx.mode,
                        BLOCK_SIZE=_BLOCK_SIZE,
                        **ctx.kernel_args,
                    )
                    if grad_hidden is not None:
                        grad_hidden[sl] += s_tile @ student_weight[start:end].to(s_dtype)
                    if grad_weight is not None:
                        grad_weight[start:end] += s_tile.t() @ h_s
                    if grad_bias is not None:
                        grad_bias[start:end] += s_tile.sum(dim=0, dtype=torch.float32)

        return (
            grad_hidden.to(student_hidden.dtype) if grad_hidden is not None else None,
            grad_weight.to(student_weight.dtype) if grad_weight is not None else None,
            grad_bias.to(student_bias.dtype) if grad_bias is not None else None,
            *(None,) * 9,
        )
