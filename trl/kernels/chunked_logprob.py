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

import torch
import triton
import triton.language as tl

from ..trainer.utils import maybe_gather_lm_head_ctx


# The projection runs on `[TOKEN_CHUNK_SIZE, VOCAB_CHUNK_SIZE]` tiles, so the logits never exist in full
TOKEN_CHUNK_SIZE = 4096
VOCAB_CHUNK_SIZE = 32768
_BLOCK_SIZE = 1024


@triton.jit
def _transform(z, logit_scale, softcap, inv_t, HAS_SOFTCAP: tl.constexpr):
    z = z * logit_scale
    if HAS_SOFTCAP:
        # tanh(x) = 2 * sigmoid(2x) - 1
        z = softcap * (2 * tl.sigmoid(2 * z / softcap) - 1)
    return z * inv_t


@triton.jit
def _forward_kernel(
    mm_ptr,
    mm_stride,
    targets_ptr,
    max_ptr,
    sum_exp_ptr,
    x_sum_exp_ptr,
    sq_sum_exp_ptr,
    target_logit_ptr,
    rescale_ptr,
    vocab_start,
    n_cols,
    logit_scale,
    softcap,
    inv_t,
    HAS_SOFTCAP: tl.constexpr,
    WRITE_WEIGHTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # One program per token: fold one vocabulary chunk of its logits into the running online-logsumexp statistics
    row = tl.program_id(0).to(tl.int64)
    row_ptr = mm_ptr + row * mm_stride
    m = tl.load(max_ptr + row)
    initial_m = m
    s = tl.load(sum_exp_ptr + row)
    xs = tl.load(x_sum_exp_ptr + row)
    sq = tl.load(sq_sum_exp_ptr + row)
    for start in range(0, n_cols, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        z = tl.load(row_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        z = _transform(z, logit_scale, softcap, inv_t, HAS_SOFTCAP)
        z = tl.where(mask, z, -float("inf"))
        new_m = tl.maximum(m, tl.max(z, axis=0))
        rescale = tl.exp(m - new_m)
        e = tl.where(mask, tl.exp(z - new_m), 0.0)
        s = s * rescale + tl.sum(e, axis=0)
        xs = xs * rescale + tl.sum(e * tl.where(mask, z, 0.0), axis=0)
        sq = sq * rescale * rescale + tl.sum(e * e, axis=0)
        m = new_m
    tl.store(max_ptr + row, m)
    tl.store(sum_exp_ptr + row, s)
    tl.store(x_sum_exp_ptr + row, xs)
    tl.store(sq_sum_exp_ptr + row, sq)

    local = tl.load(targets_ptr + row) - vocab_start
    if (local >= 0) & (local < n_cols):
        z = tl.load(row_ptr + local).to(tl.float32)
        tl.store(target_logit_ptr + row, _transform(z, logit_scale, softcap, inv_t, HAS_SOFTCAP))

    if WRITE_WEIGHTS:
        # Overwrite the tile with exp(z - m) * dz/dy, the weights of this chunk's rows of `weight` in the running
        # expectation of dz/dh; the caller rescales its accumulator by exp(initial_m - m) before adding them
        tl.store(rescale_ptr + row, tl.exp(initial_m - m))
        for start in range(0, n_cols, BLOCK_SIZE):
            offsets = start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_cols
            y = tl.load(row_ptr + offsets, mask=mask, other=0.0).to(tl.float32) * logit_scale
            if HAS_SOFTCAP:
                t = 2 * tl.sigmoid(2 * y / softcap) - 1
                z = softcap * t * inv_t
                dz_dy = logit_scale * inv_t * (1 - t * t)
            else:
                z = y * inv_t
                dz_dy = logit_scale * inv_t
            w = tl.where(mask, tl.exp(z - m) * dz_dy, 0.0)
            tl.store(row_ptr + offsets, w.to(mm_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _backward_kernel(
    mm_ptr,
    mm_stride,
    targets_ptr,
    log_z_ptr,
    entropy_ptr,
    grad_logprobs_ptr,
    grad_entropy_ptr,
    vocab_start,
    n_cols,
    logit_scale,
    softcap,
    inv_t,
    HAS_SOFTCAP: tl.constexpr,
    HAS_LOGPROB_GRAD: tl.constexpr,
    HAS_ENTROPY_GRAD: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Overwrite one block of the logits tile with the gradient of the loss with respect to the projection output
    row = tl.program_id(0).to(tl.int64)
    offsets = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    ptrs = mm_ptr + row * mm_stride + offsets
    y = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32) * logit_scale
    if HAS_SOFTCAP:
        t = 2 * tl.sigmoid(2 * y / softcap) - 1
        z = softcap * t * inv_t
    else:
        z = y * inv_t
    log_p = z - tl.load(log_z_ptr + row)
    p = tl.exp(log_p)

    grad = tl.zeros((BLOCK_SIZE,), tl.float32)
    if HAS_LOGPROB_GRAD:
        # d(log p_target) / dz_j = 1[j = target] - p_j
        g = tl.load(grad_logprobs_ptr + row)
        is_target = (offsets + vocab_start) == tl.load(targets_ptr + row)
        grad += g * (is_target.to(tl.float32) - p)
    if HAS_ENTROPY_GRAD:
        # d(entropy) / dz_j = -p_j * (log p_j + entropy)
        g = tl.load(grad_entropy_ptr + row)
        grad -= g * p * (log_p + tl.load(entropy_ptr + row))
    grad = grad * inv_t
    if HAS_SOFTCAP:
        grad = grad * (1 - t * t)
    grad = grad * logit_scale
    tl.store(ptrs, grad.to(mm_ptr.dtype.element_ty), mask=mask)


class ChunkedLogProbFunction(torch.autograd.Function):
    """
    Per-token log-probabilities, entropy, `log(sum_v p_v^2)` and whether the target is the argmax, of `hidden @
    weight.T`, without materializing the `[N, V]` logits.

    The projection runs in cuBLAS on `[TOKEN_CHUNK_SIZE, VOCAB_CHUNK_SIZE]` tiles; a Triton kernel folds each tile into
    online-logsumexp statistics in one pass. The backward recomputes each tile and turns it into the logits gradient in
    place.

    When the LM head is frozen and only the hidden states need a gradient (PEFT), the forward also accumulates each
    token's `d log p / d hidden`, one extra GEMM per tile, and the backward scales it by the incoming gradient instead
    of recomputing the tiles: two vocabulary-sized GEMMs in total instead of three.
    """

    @staticmethod
    def forward(
        ctx,
        hidden: torch.Tensor,  # [N, H]
        weight: torch.Tensor,  # [V, H]
        bias: torch.Tensor | None,  # [V]
        targets: torch.Tensor,  # [N]
        temperature: float,
        final_logit_softcapping: float | None = None,
        logit_scale: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # entropy is often computed for logging only (no grad required); without this, autograd would
        # materialize its incoming gradient as zeros and backward would waste compute on a no-op term
        ctx.set_materialize_grads(False)
        device = hidden.device
        N = hidden.shape[0]
        vocab = weight.shape[0]
        compute_dtype = (
            torch.get_autocast_dtype(device.type) if torch.is_autocast_enabled(device.type) else hidden.dtype
        )
        kernel_args = {
            "logit_scale": logit_scale,
            "softcap": final_logit_softcapping or 1.0,
            "inv_t": 1 / temperature,
            "HAS_SOFTCAP": final_logit_softcapping is not None,
        }

        running_max = torch.full((N,), float("-inf"), device=device, dtype=torch.float32)
        sum_exp = torch.zeros((N,), device=device, dtype=torch.float32)
        x_sum_exp = torch.zeros((N,), device=device, dtype=torch.float32)
        sq_sum_exp = torch.zeros((N,), device=device, dtype=torch.float32)
        target_logit = torch.zeros((N,), device=device, dtype=torch.float32)
        mm_buf = torch.empty((min(N, TOKEN_CHUNK_SIZE), VOCAB_CHUNK_SIZE), device=device, dtype=compute_dtype)
        needs_hidden_grad, needs_weight_grad, needs_bias_grad = ctx.needs_input_grad[:3]
        # d log p_target / d hidden = dz_target/dy * W[target] - E_p[dz/dy * W], accumulated online like the logsumexp
        jacobian = None
        if needs_hidden_grad and not needs_weight_grad and not needs_bias_grad:
            jacobian = torch.empty(hidden.shape, device=device, dtype=hidden.dtype)
            rescale = torch.empty((min(N, TOKEN_CHUNK_SIZE),), device=device, dtype=torch.float32)

        for token_start in range(0, N, TOKEN_CHUNK_SIZE):
            token_end = min(token_start + TOKEN_CHUNK_SIZE, N)
            n = token_end - token_start
            hidden_chunk = hidden[token_start:token_end].to(compute_dtype)
            if jacobian is not None:
                expected_w = torch.zeros((n, hidden.shape[1]), device=device, dtype=torch.float32)
            for start in range(0, vocab, VOCAB_CHUNK_SIZE):
                end = min(start + VOCAB_CHUNK_SIZE, vocab)
                tile = mm_buf[:n, : end - start]
                torch.mm(hidden_chunk, weight[start:end].to(compute_dtype).t(), out=tile)
                if bias is not None:
                    tile.add_(bias[start:end].to(compute_dtype))
                _forward_kernel[(n,)](
                    tile,
                    tile.stride(0),
                    targets[token_start:token_end],
                    running_max[token_start:token_end],
                    sum_exp[token_start:token_end],
                    x_sum_exp[token_start:token_end],
                    sq_sum_exp[token_start:token_end],
                    target_logit[token_start:token_end],
                    rescale if jacobian is not None else sum_exp,
                    start,
                    end - start,
                    WRITE_WEIGHTS=jacobian is not None,
                    BLOCK_SIZE=_BLOCK_SIZE,
                    **kernel_args,
                )
                if jacobian is not None:
                    expected_w.mul_(rescale[:n, None]).add_(tile @ weight[start:end].to(compute_dtype))
            if jacobian is not None:
                sl = slice(token_start, token_end)
                # dz/dy at the target, recovered from its transformed logit z = softcap * tanh(...) * inv_t
                dz_dy = torch.full((n,), logit_scale / temperature, device=device)
                if final_logit_softcapping is not None:
                    t = target_logit[sl] * temperature / final_logit_softcapping
                    dz_dy = dz_dy * (1 - t * t)
                jacobian[sl] = weight[targets[sl]].float() * dz_dy[:, None] - expected_w / sum_exp[sl, None]

        log_z = running_max + torch.log(sum_exp)
        logprobs = target_logit - log_z
        entropy = log_z - x_sum_exp / sum_exp
        log_sum_sq_probs = torch.log(sq_sum_exp) - 2 * torch.log(sum_exp)
        is_top1 = target_logit >= running_max

        ctx.save_for_backward(hidden, weight, bias, targets, log_z, entropy, jacobian)
        ctx.compute_dtype = compute_dtype
        ctx.kernel_args = kernel_args
        ctx.mark_non_differentiable(log_sum_sq_probs)
        return logprobs, entropy, log_sum_sq_probs, is_top1

    @staticmethod
    def backward(ctx, grad_logprobs: torch.Tensor | None, grad_entropy: torch.Tensor | None, *_):  # type: ignore
        hidden, weight, bias, targets, log_z, entropy, jacobian = ctx.saved_tensors
        compute_dtype = ctx.compute_dtype
        needs_hidden_grad, needs_weight_grad, needs_bias_grad = ctx.needs_input_grad[:3]
        N = hidden.shape[0]
        if jacobian is not None and grad_entropy is None:
            grad_hidden = grad_logprobs[:, None] * jacobian.float() if grad_logprobs is not None else None
            return grad_hidden.to(hidden.dtype) if grad_hidden is not None else None, *(None,) * 6
        with maybe_gather_lm_head_ctx(weight, bias):
            vocab = weight.shape[0]
            # Always accumulate in fp32, even when the inputs are not
            grad_hidden = (
                torch.zeros(hidden.shape, device=hidden.device, dtype=torch.float32) if needs_hidden_grad else None
            )
            grad_weight = (
                torch.zeros(weight.shape, device=weight.device, dtype=torch.float32) if needs_weight_grad else None
            )
            grad_bias = torch.zeros(bias.shape, device=bias.device, dtype=torch.float32) if needs_bias_grad else None
            mm_buf = torch.empty(
                (min(N, TOKEN_CHUNK_SIZE), VOCAB_CHUNK_SIZE), device=hidden.device, dtype=compute_dtype
            )
            grad_logprobs = grad_logprobs.float().contiguous() if grad_logprobs is not None else None
            grad_entropy = grad_entropy.float().contiguous() if grad_entropy is not None else None

            for token_start in range(0, N, TOKEN_CHUNK_SIZE):
                token_end = min(token_start + TOKEN_CHUNK_SIZE, N)
                n = token_end - token_start
                hidden_chunk = hidden[token_start:token_end].to(compute_dtype)
                for start in range(0, vocab, VOCAB_CHUNK_SIZE):
                    end = min(start + VOCAB_CHUNK_SIZE, vocab)
                    w_chunk = weight[start:end].to(compute_dtype)
                    tile = mm_buf[:n, : end - start]
                    torch.mm(hidden_chunk, w_chunk.t(), out=tile)
                    if bias is not None:
                        tile.add_(bias[start:end].to(compute_dtype))
                    _backward_kernel[(n, triton.cdiv(end - start, _BLOCK_SIZE))](
                        tile,
                        tile.stride(0),
                        targets[token_start:token_end],
                        log_z[token_start:token_end],
                        entropy[token_start:token_end],
                        grad_logprobs[token_start:token_end] if grad_logprobs is not None else log_z,
                        grad_entropy[token_start:token_end] if grad_entropy is not None else log_z,
                        start,
                        end - start,
                        HAS_LOGPROB_GRAD=grad_logprobs is not None,
                        HAS_ENTROPY_GRAD=grad_entropy is not None,
                        BLOCK_SIZE=_BLOCK_SIZE,
                        **ctx.kernel_args,
                    )
                    if grad_hidden is not None:
                        grad_hidden[token_start:token_end] += tile @ w_chunk
                    if grad_weight is not None:
                        grad_weight[start:end] += tile.t() @ hidden_chunk
                    if grad_bias is not None:
                        grad_bias[start:end] += tile.sum(dim=0, dtype=torch.float32)

        return (
            grad_hidden.to(hidden.dtype) if grad_hidden is not None else None,
            grad_weight.to(weight.dtype) if grad_weight is not None else None,
            grad_bias.to(bias.dtype) if grad_bias is not None else None,
            None,
            None,
            None,
            None,
        )
