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
from packaging.version import Version
from transformers.utils import is_triton_available


if is_triton_available():
    import triton
    import triton.language as tl


# Tokens per projection tile; the caller sets the vocabulary width with `chunk_size`
TOKEN_CHUNK_SIZE = 4096
# Per-token outputs besides the log-probabilities, each computed only when requested
OPTIONAL_OUTPUTS = ("entropy", "log_sum_sq_probs", "mean_logits", "is_top1")
_BLOCK_SIZE = 1024
# `torch.addmm(..., out_dtype=torch.float32)` from bf16/fp16 inputs, CUDA only
_MM_OUT_DTYPE = Version(torch.__version__) >= Version("2.8.0")


def _jit(fn):
    # Triton ships with PyTorch on Linux only; without it, the tiles are folded in torch
    return triton.jit(fn) if is_triton_available() else fn


# Read once at import: Dynamo warns when it traces a call to the `lru_cache`-wrapped `is_triton_available`
_TRITON_AVAILABLE = is_triton_available()


def _use_triton(device: torch.device) -> bool:
    return _TRITON_AVAILABLE and device.type in ("cuda", "xpu")


def _addmm_fp32(acc: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> None:
    """
    `acc += a @ b` for an fp32 `acc`, without rounding `a @ b` to the inputs' dtype first.

    Rounding every vocabulary chunk's product to bf16 before summing them biases training (entropy drifts lower over a
    few dozen steps), even though each step's gradient error looks the same size as with full logits.
    """
    if _MM_OUT_DTYPE and acc.device.type == "cuda" and a.dtype in (torch.float16, torch.bfloat16):
        torch.addmm(acc, a, b, out_dtype=torch.float32, out=acc)
    else:
        acc.addmm_(a.float(), b.float())


def _support_mask(support_ids: torch.Tensor, vocab_start: int, vocab_end: int) -> torch.Tensor:
    """
    Indicator of the supports over one vocabulary tile: `[n, vocab_end - vocab_start]`, nonzero where the token id is
    in the row's support. `support_ids` is `[n, K]` of token ids, padded with `-1`.
    """
    width = vocab_end - vocab_start
    local = support_ids - vocab_start
    in_tile = (local >= 0) & (local < width)
    support_mask = torch.zeros((local.shape[0], width), device=support_ids.device, dtype=torch.int8)
    support_mask.scatter_add_(1, local.clamp(0, width - 1), in_tile.to(torch.int8))
    return support_mask


def _gather_ids(tile: torch.Tensor, ids: torch.Tensor, vocab_start: int) -> tuple[torch.Tensor, ...]:
    """
    Raw logits of one vocabulary tile at `ids` (`[n, K]`, padded with `-1`): the values, their clamped column in the
    tile and whether each id falls in the tile at all.
    """
    width = tile.shape[1]
    local = ids - vocab_start
    in_tile = (local >= 0) & (local < width)
    local = local.clamp(0, width - 1)
    return tile.gather(1, local).float(), local, in_tile


@_jit
def _transform(z, logit_scale, softcap, inv_t, HAS_SOFTCAP: tl.constexpr):
    # Before torch 2.15, torch.compile passes Python float arguments as fp64; keep the math in fp32
    logit_scale = tl.cast(logit_scale, tl.float32)
    softcap = tl.cast(softcap, tl.float32)
    inv_t = tl.cast(inv_t, tl.float32)
    z = z * logit_scale
    if HAS_SOFTCAP:
        # tanh(x) = 2 * sigmoid(2x) - 1
        z = softcap * (2 * tl.sigmoid(2 * z / softcap) - 1)
    return z * inv_t


@_jit
def _forward_kernel(
    mm_ptr,
    mm_stride,
    targets_ptr,
    max_ptr,
    sum_exp_ptr,
    x_sum_exp_ptr,
    sq_sum_exp_ptr,
    z_sum_ptr,
    max_before_target_ptr,
    target_logit_ptr,
    support_mask_ptr,
    support_mask_stride,
    support_sum_exp_ptr,
    vocab_start,
    n_cols,
    logit_scale,
    softcap,
    inv_t,
    HAS_SOFTCAP: tl.constexpr,
    HAS_SUPPORT: tl.constexpr,
    HAS_ENTROPY: tl.constexpr,
    HAS_LOG_SUM_SQ_PROBS: tl.constexpr,
    HAS_MEAN_LOGITS: tl.constexpr,
    HAS_IS_TOP1: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # One program per token: fold one vocabulary chunk of its logits into the running online-logsumexp statistics
    row = tl.program_id(0).to(tl.int64)
    row_ptr = mm_ptr + row * mm_stride
    m = tl.load(max_ptr + row)
    local = tl.load(targets_ptr + row) - vocab_start
    max_before_target = tl.load(max_before_target_ptr + row)
    s = tl.load(sum_exp_ptr + row)
    xs = tl.load(x_sum_exp_ptr + row)
    sq = tl.load(sq_sum_exp_ptr + row)
    zs = tl.load(z_sum_ptr + row)
    ss = tl.load(support_sum_exp_ptr + row)
    for start in range(0, n_cols, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        z = tl.load(row_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        z = _transform(z, logit_scale, softcap, inv_t, HAS_SOFTCAP)
        if HAS_MEAN_LOGITS:
            zs += tl.sum(tl.where(mask, z, 0.0), axis=0)
        z = tl.where(mask, z, -float("inf"))
        if HAS_IS_TOP1:
            # `argmax` returns the first maximum, so a tie with an earlier token does not count as a top-1 prediction
            max_before_target = tl.maximum(
                max_before_target, tl.max(tl.where(offsets < local, z, -float("inf")), axis=0)
            )
        new_m = tl.maximum(m, tl.max(z, axis=0))
        rescale = tl.exp(m - new_m)
        e = tl.where(mask, tl.exp(z - new_m), 0.0)
        s = s * rescale + tl.sum(e, axis=0)
        if HAS_SUPPORT:
            in_support = tl.load(support_mask_ptr + row * support_mask_stride + offsets, mask=mask, other=0) > 0
            ss = ss * rescale + tl.sum(tl.where(in_support, e, 0.0), axis=0)
        if HAS_ENTROPY:
            xs = xs * rescale + tl.sum(e * tl.where(mask, z, 0.0), axis=0)
        if HAS_LOG_SUM_SQ_PROBS:
            sq = sq * rescale * rescale + tl.sum(e * e, axis=0)
        m = new_m
    tl.store(max_ptr + row, m)
    tl.store(sum_exp_ptr + row, s)
    if HAS_SUPPORT:
        tl.store(support_sum_exp_ptr + row, ss)
    if HAS_ENTROPY:
        tl.store(x_sum_exp_ptr + row, xs)
    if HAS_LOG_SUM_SQ_PROBS:
        tl.store(sq_sum_exp_ptr + row, sq)
    if HAS_MEAN_LOGITS:
        tl.store(z_sum_ptr + row, zs)
    if HAS_IS_TOP1:
        tl.store(max_before_target_ptr + row, max_before_target)

    if (local >= 0) & (local < n_cols):
        z = tl.load(row_ptr + local).to(tl.float32)
        tl.store(target_logit_ptr + row, _transform(z, logit_scale, softcap, inv_t, HAS_SOFTCAP))


@_jit
def _backward_kernel(
    mm_ptr,
    mm_stride,
    targets_ptr,
    log_z_ptr,
    support_log_z_ptr,
    entropy_ptr,
    grad_logprobs_ptr,
    grad_entropy_ptr,
    support_mask_ptr,
    support_mask_stride,
    vocab_start,
    n_cols,
    logit_scale,
    softcap,
    inv_t,
    HAS_SOFTCAP: tl.constexpr,
    HAS_SUPPORT: tl.constexpr,
    HAS_LOGPROB_GRAD: tl.constexpr,
    HAS_ENTROPY_GRAD: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Overwrite one block of the logits tile with the gradient of the loss with respect to the projection output
    # Before torch 2.15, torch.compile passes Python float arguments as fp64; keep the math in fp32
    logit_scale = tl.cast(logit_scale, tl.float32)
    softcap = tl.cast(softcap, tl.float32)
    inv_t = tl.cast(inv_t, tl.float32)
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
        # d(log p_target) / dz_j = 1[j = target] - p_j, with p the softmax over the support when one is given
        g = tl.load(grad_logprobs_ptr + row)
        is_target = (offsets + vocab_start) == tl.load(targets_ptr + row)
        if HAS_SUPPORT:
            in_support = tl.load(support_mask_ptr + row * support_mask_stride + offsets, mask=mask, other=0) > 0
            q = tl.where(in_support, tl.exp(z - tl.load(support_log_z_ptr + row)), 0.0)
        else:
            q = p
        grad += g * (is_target.to(tl.float32) - q)
    if HAS_ENTROPY_GRAD:
        # d(entropy) / dz_j = -p_j * (log p_j + entropy)
        g = tl.load(grad_entropy_ptr + row)
        grad -= g * p * (log_p + tl.load(entropy_ptr + row))
    grad = grad * inv_t
    if HAS_SOFTCAP:
        grad = grad * (1 - t * t)
    grad = grad * logit_scale
    tl.store(ptrs, grad.to(mm_ptr.dtype.element_ty), mask=mask)


def _torch_transform(z, logit_scale, softcap, inv_t, HAS_SOFTCAP):
    z = z * logit_scale
    if HAS_SOFTCAP:
        z = softcap * torch.tanh(z / softcap)
    return z * inv_t


def _forward_tile(
    tile,
    tile_stride,
    targets,
    running_max,
    sum_exp,
    x_sum_exp,
    sq_sum_exp,
    z_sum,
    max_before_target,
    target_logit,
    support_mask,
    support_mask_stride,
    support_sum_exp,
    vocab_start,
    n_cols,
    logit_scale,
    softcap,
    inv_t,
    HAS_SOFTCAP,
    HAS_SUPPORT,
    HAS_ENTROPY,
    HAS_LOG_SUM_SQ_PROBS,
    HAS_MEAN_LOGITS,
    HAS_IS_TOP1,
    BLOCK_SIZE,
):
    # `_forward_kernel` in torch, for devices without Triton: the whole tile at once instead of one program per token
    z = _torch_transform(tile.float(), logit_scale, softcap, inv_t, HAS_SOFTCAP)
    local = targets - vocab_start
    if HAS_MEAN_LOGITS:
        z_sum += z.sum(-1)
    if HAS_IS_TOP1:
        # `argmax` returns the first maximum, so a tie with an earlier token does not count as a top-1 prediction
        before_target = torch.arange(n_cols, device=z.device) < local.unsqueeze(-1)
        max_before_target.copy_(torch.maximum(max_before_target, z.masked_fill(~before_target, -math.inf).amax(-1)))
    new_m = torch.maximum(running_max, z.amax(-1))
    rescale = torch.exp(running_max - new_m)
    e = torch.exp(z - new_m.unsqueeze(-1))
    sum_exp.mul_(rescale).add_(e.sum(-1))
    if HAS_SUPPORT:
        support_sum_exp.mul_(rescale).add_((e * (support_mask > 0)).sum(-1))
    if HAS_ENTROPY:
        x_sum_exp.mul_(rescale).add_((e * z).sum(-1))
    if HAS_LOG_SUM_SQ_PROBS:
        sq_sum_exp.mul_(rescale * rescale).add_((e * e).sum(-1))
    running_max.copy_(new_m)
    in_tile = (local >= 0) & (local < n_cols)
    target_z = z.gather(-1, local.clamp(0, n_cols - 1).unsqueeze(-1)).squeeze(-1)
    target_logit.copy_(torch.where(in_tile, target_z, target_logit))


def _backward_tile(
    tile,
    tile_stride,
    targets,
    log_z,
    support_log_z,
    entropy,
    grad_logprobs,
    grad_entropy,
    support_mask,
    support_mask_stride,
    vocab_start,
    n_cols,
    logit_scale,
    softcap,
    inv_t,
    HAS_SOFTCAP,
    HAS_SUPPORT,
    HAS_LOGPROB_GRAD,
    HAS_ENTROPY_GRAD,
    BLOCK_SIZE,
):
    # `_backward_kernel` in torch, for devices without Triton: the whole tile at once instead of one block per program
    y = tile.float() * logit_scale
    if HAS_SOFTCAP:
        t = torch.tanh(y / softcap)
        z = softcap * t * inv_t
    else:
        z = y * inv_t
    log_p = z - log_z.unsqueeze(-1)
    p = torch.exp(log_p)

    grad = torch.zeros_like(z)
    if HAS_LOGPROB_GRAD:
        # d(log p_target) / dz_j = 1[j = target] - p_j, with p the softmax over the support when one is given
        is_target = torch.arange(vocab_start, vocab_start + n_cols, device=z.device) == targets.unsqueeze(-1)
        q = torch.exp(z - support_log_z.unsqueeze(-1)) * (support_mask > 0) if HAS_SUPPORT else p
        grad += grad_logprobs.unsqueeze(-1) * (is_target.float() - q)
    if HAS_ENTROPY_GRAD:
        # d(entropy) / dz_j = -p_j * (log p_j + entropy)
        grad -= grad_entropy.unsqueeze(-1) * p * (log_p + entropy.unsqueeze(-1))
    grad = grad * inv_t
    if HAS_SOFTCAP:
        grad = grad * (1 - t * t)
    tile.copy_(grad * logit_scale)


class ChunkedLogProbFunction(torch.autograd.Function):
    """
    Per-token log-probabilities of `hidden @ weight.T`, without materializing the `[N, V]` logits, and optionally the
    entropy, `log(sum_v p_v^2)`, the mean logit and whether the target is the argmax (`outputs`, `None` when not
    requested).

    The projection runs in cuBLAS on `[TOKEN_CHUNK_SIZE, chunk_size]` tiles; a Triton kernel (torch ops on devices
    without Triton) folds each tile into online-logsumexp statistics in one pass. The backward recomputes each tile,
    turns it into the logits gradient in place, and accumulates the gradient GEMMs in fp32.

    With `support_ids`, a `[N, K]` tensor of token ids padded with `-1`, the log-probabilities and their gradient are
    normalized over each row's support, the set of tokens the sampler could draw, instead of the whole vocabulary,
    matching a sampler that drew the target from a top-k/top-p truncated distribution. The entropy stays that of the
    whole vocabulary. With `score_ids`, a `[N, K]` tensor of token ids padded with `-1`, the sixth output is the
    log-probability of each of those tokens under the same normalization (`0` at padding), differentiable like the
    first. Score ids are expected inside the support; the value at one outside it is not a log-probability.
    """

    @staticmethod
    def forward(
        ctx,
        hidden: torch.Tensor,  # [N, H]
        weight: torch.Tensor,  # [V, H]
        bias: torch.Tensor | None,  # [V]
        targets: torch.Tensor,  # [N]
        temperature: float,
        chunk_size: int,
        final_logit_softcapping: float | None = None,
        logit_scale: float = 1.0,
        outputs: tuple[str, ...] = OPTIONAL_OUTPUTS,
        support_ids: torch.Tensor | None = None,  # [N, K]
        score_ids: torch.Tensor | None = None,  # [N, K]
    ) -> tuple[torch.Tensor, ...]:
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
            "HAS_SUPPORT": support_ids is not None,
        }
        output_flags = {f"HAS_{name.upper()}": name in outputs for name in OPTIONAL_OUTPUTS}

        running_max = torch.full((N,), float("-inf"), device=device, dtype=torch.float32)
        sum_exp = torch.zeros((N,), device=device, dtype=torch.float32)
        support_sum_exp = torch.zeros((N,), device=device, dtype=torch.float32)
        x_sum_exp = torch.zeros((N,), device=device, dtype=torch.float32)
        sq_sum_exp = torch.zeros((N,), device=device, dtype=torch.float32)
        z_sum = torch.zeros((N,), device=device, dtype=torch.float32)
        max_before_target = torch.full((N,), float("-inf"), device=device, dtype=torch.float32)
        target_logit = torch.zeros((N,), device=device, dtype=torch.float32)
        score_logit = (
            torch.zeros(score_ids.shape, device=device, dtype=torch.float32) if score_ids is not None else None
        )
        mm_buf = torch.empty((min(N, TOKEN_CHUNK_SIZE), chunk_size), device=device, dtype=compute_dtype)

        for token_start in range(0, N, TOKEN_CHUNK_SIZE):
            token_end = min(token_start + TOKEN_CHUNK_SIZE, N)
            n = token_end - token_start
            hidden_chunk = hidden[token_start:token_end].to(compute_dtype)
            for start in range(0, vocab, chunk_size):
                end = min(start + chunk_size, vocab)
                tile = mm_buf[:n, : end - start]
                torch.mm(hidden_chunk, weight[start:end].to(compute_dtype).t(), out=tile)
                if bias is not None:
                    tile.add_(bias[start:end].to(compute_dtype))
                support_mask = (
                    _support_mask(support_ids[token_start:token_end], start, end)
                    if support_ids is not None
                    else running_max
                )
                if score_ids is not None:
                    z, _, in_tile = _gather_ids(tile, score_ids[token_start:token_end], start)
                    z = _torch_transform(
                        z, logit_scale, final_logit_softcapping, 1 / temperature, final_logit_softcapping is not None
                    )
                    chunk = score_logit[token_start:token_end]
                    chunk.copy_(torch.where(in_tile, z, chunk))
                fold = _forward_kernel[(n,)] if _use_triton(device) else _forward_tile
                fold(
                    tile,
                    tile.stride(0),
                    targets[token_start:token_end],
                    running_max[token_start:token_end],
                    sum_exp[token_start:token_end],
                    x_sum_exp[token_start:token_end],
                    sq_sum_exp[token_start:token_end],
                    z_sum[token_start:token_end],
                    max_before_target[token_start:token_end],
                    target_logit[token_start:token_end],
                    support_mask,
                    support_mask.stride(0),
                    support_sum_exp[token_start:token_end],
                    start,
                    end - start,
                    BLOCK_SIZE=_BLOCK_SIZE,
                    **output_flags,
                    **kernel_args,
                )

        log_z = running_max + torch.log(sum_exp)
        support_log_z = log_z if support_ids is None else running_max + torch.log(support_sum_exp)
        logprobs = target_logit - support_log_z
        entropy = log_z - x_sum_exp / sum_exp if "entropy" in outputs else None
        log_sum_sq_probs = torch.log(sq_sum_exp) - 2 * torch.log(sum_exp) if "log_sum_sq_probs" in outputs else None
        mean_logits = z_sum / vocab if "mean_logits" in outputs else None
        is_top1 = (target_logit >= running_max) & (target_logit > max_before_target) if "is_top1" in outputs else None
        score_log_probs = (
            (score_logit - support_log_z.unsqueeze(-1)).masked_fill(score_ids < 0, 0.0)
            if score_ids is not None
            else None
        )

        # Without entropy there is no entropy gradient, so `log_z` only fills its slot
        ctx.save_for_backward(
            hidden,
            weight,
            bias,
            targets,
            log_z,
            support_log_z,
            log_z if entropy is None else entropy,
            support_ids,
            score_ids,
        )
        ctx.compute_dtype = compute_dtype
        ctx.chunk_size = chunk_size
        ctx.kernel_args = kernel_args
        ctx.logit_scale, ctx.temperature, ctx.softcap = logit_scale, temperature, final_logit_softcapping
        ctx.mark_non_differentiable(*(x for x in (log_sum_sq_probs, mean_logits, is_top1) if x is not None))
        return logprobs, entropy, log_sum_sq_probs, mean_logits, is_top1, score_log_probs

    @staticmethod
    def backward(ctx, grad_logprobs, grad_entropy, _grad_sq, _grad_mean, _grad_top1, grad_score):  # type: ignore
        # `trl.trainer.utils` imports this module, so import from it at call time
        from ..trainer.utils import maybe_gather_lm_head_ctx

        hidden, weight, bias, targets, log_z, support_log_z, entropy, support_ids, score_ids = ctx.saved_tensors
        compute_dtype = ctx.compute_dtype
        chunk_size = ctx.chunk_size
        needs_hidden_grad, needs_weight_grad, needs_bias_grad = ctx.needs_input_grad[:3]
        N = hidden.shape[0]
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
            mm_buf = torch.empty((min(N, TOKEN_CHUNK_SIZE), chunk_size), device=hidden.device, dtype=compute_dtype)
            grad_logprobs = grad_logprobs.float().contiguous() if grad_logprobs is not None else None
            if grad_score is not None:
                # d(sum_k g_k log p_k) / dz_j = g_j - (sum_k g_k) p_j. The second term is the target's log-prob gradient
                # scaled by sum_k g_k, which also puts sum_k g_k on the target: an extra column takes it off again.
                grad_score = grad_score.float().masked_fill(score_ids < 0, 0.0)
                grad_score_sum = grad_score.sum(-1)
                grad_logprobs = grad_score_sum if grad_logprobs is None else grad_logprobs + grad_score_sum
                score_ids = torch.cat([score_ids, targets.unsqueeze(-1)], -1)
                grad_score = torch.cat([grad_score, -grad_score_sum.unsqueeze(-1)], -1)
            grad_entropy = grad_entropy.float().contiguous() if grad_entropy is not None else None

            for token_start in range(0, N, TOKEN_CHUNK_SIZE):
                token_end = min(token_start + TOKEN_CHUNK_SIZE, N)
                n = token_end - token_start
                hidden_chunk = hidden[token_start:token_end].to(compute_dtype)
                for start in range(0, vocab, chunk_size):
                    end = min(start + chunk_size, vocab)
                    w_chunk = weight[start:end].to(compute_dtype)
                    tile = mm_buf[:n, : end - start]
                    torch.mm(hidden_chunk, w_chunk.t(), out=tile)
                    if bias is not None:
                        tile.add_(bias[start:end].to(compute_dtype))
                    support_mask = (
                        _support_mask(support_ids[token_start:token_end], start, end)
                        if support_ids is not None
                        else log_z
                    )
                    if grad_score is not None:
                        y, local, in_tile = _gather_ids(tile, score_ids[token_start:token_end], start)
                        y = y * ctx.logit_scale
                        chain = torch.full_like(y, ctx.logit_scale / ctx.temperature)
                        if ctx.softcap is not None:
                            chain = chain * (1 - torch.tanh(y / ctx.softcap) ** 2)
                        sparse_grad = grad_score[token_start:token_end] * chain * in_tile
                    fold = (
                        _backward_kernel[(n, triton.cdiv(end - start, _BLOCK_SIZE))]
                        if _use_triton(hidden.device)
                        else _backward_tile
                    )
                    fold(
                        tile,
                        tile.stride(0),
                        targets[token_start:token_end],
                        log_z[token_start:token_end],
                        support_log_z[token_start:token_end],
                        entropy[token_start:token_end],
                        grad_logprobs[token_start:token_end] if grad_logprobs is not None else log_z,
                        grad_entropy[token_start:token_end] if grad_entropy is not None else log_z,
                        support_mask,
                        support_mask.stride(0),
                        start,
                        end - start,
                        HAS_LOGPROB_GRAD=grad_logprobs is not None,
                        HAS_ENTROPY_GRAD=grad_entropy is not None,
                        BLOCK_SIZE=_BLOCK_SIZE,
                        **ctx.kernel_args,
                    )
                    if grad_score is not None:
                        tile.scatter_add_(1, local, sparse_grad.to(tile.dtype))
                    if grad_hidden is not None:
                        _addmm_fp32(grad_hidden[token_start:token_end], tile, w_chunk)
                    if grad_weight is not None:
                        _addmm_fp32(grad_weight[start:end], tile.t(), hidden_chunk)
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
            None,
            None,
            None,
            None,
        )
