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


_BLOCK_SIZE = 1024


@triton.jit
def _forward_kernel(
    logits_ptr,
    index_ptr,
    row_mask_ptr,
    logprobs_ptr,
    entropy_ptr,
    log_z_ptr,
    expected_logit_ptr,
    logits_batch_stride,
    logits_row_stride,
    index_batch_stride,
    index_row_stride,
    mask_batch_stride,
    mask_row_stride,
    n_cols: tl.constexpr,
    rows_per_batch,
    TEMPERATURE: tl.constexpr,
    HAS_MASK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    batch = row // rows_per_batch
    row_in_batch = row - batch * rows_per_batch
    if HAS_MASK:
        is_valid = tl.load(row_mask_ptr + batch * mask_batch_stride + row_in_batch * mask_row_stride)
        if is_valid == 0:
            tl.store(logprobs_ptr + row, 0.0)
            tl.store(entropy_ptr + row, 0.0)
            tl.store(log_z_ptr + row, 0.0)
            tl.store(expected_logit_ptr + row, 0.0)
            return
    logits_row_ptr = logits_ptr + batch * logits_batch_stride + row_in_batch * logits_row_stride
    index = tl.load(index_ptr + batch * index_batch_stride + row_in_batch * index_row_stride)

    row_max = -float("inf")
    denominator = 0.0
    weighted_sum = 0.0
    for start in range(0, n_cols, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        logits = tl.load(logits_row_ptr + offsets, mask=mask, other=-float("inf")).to(tl.float32) / TEMPERATURE
        block_max = tl.max(logits, axis=0)
        new_max = tl.maximum(row_max, block_max)
        correction = tl.exp(row_max - new_max)
        probabilities = tl.where(mask, tl.exp(logits - new_max), 0.0)
        denominator = denominator * correction + tl.sum(probabilities, axis=0)
        weighted_sum = weighted_sum * correction + tl.sum(probabilities * tl.where(mask, logits, 0.0), axis=0)
        row_max = new_max

    log_z = row_max + tl.log(denominator)
    expected_logit = weighted_sum / denominator
    target_logit = tl.load(logits_row_ptr + index).to(tl.float32) / TEMPERATURE
    tl.store(logprobs_ptr + row, target_logit - log_z)
    tl.store(entropy_ptr + row, log_z - expected_logit)
    tl.store(log_z_ptr + row, log_z)
    tl.store(expected_logit_ptr + row, expected_logit)


@triton.jit
def _backward_kernel(
    logits_ptr,
    index_ptr,
    row_mask_ptr,
    log_z_ptr,
    expected_logit_ptr,
    grad_logprobs_ptr,
    grad_entropy_ptr,
    grad_logits_ptr,
    logits_batch_stride,
    logits_row_stride,
    index_batch_stride,
    index_row_stride,
    mask_batch_stride,
    mask_row_stride,
    grad_logits_batch_stride,
    grad_logits_row_stride,
    n_cols: tl.constexpr,
    rows_per_batch,
    TEMPERATURE: tl.constexpr,
    HAS_MASK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_LOGPROB_GRAD: tl.constexpr,
    HAS_ENTROPY_GRAD: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    block = tl.program_id(1)
    batch = row // rows_per_batch
    row_in_batch = row - batch * rows_per_batch
    offsets = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols

    logits_row_ptr = logits_ptr + batch * logits_batch_stride + row_in_batch * logits_row_stride
    grad_logits_row_ptr = grad_logits_ptr + batch * grad_logits_batch_stride + row_in_batch * grad_logits_row_stride
    if HAS_MASK:
        is_valid = tl.load(row_mask_ptr + batch * mask_batch_stride + row_in_batch * mask_row_stride)
        if is_valid == 0:
            tl.store(grad_logits_row_ptr + offsets, 0.0, mask=mask)
            return
    logits = tl.load(logits_row_ptr + offsets, mask=mask).to(tl.float32) / TEMPERATURE
    index = tl.load(index_ptr + batch * index_batch_stride + row_in_batch * index_row_stride)
    log_z = tl.load(log_z_ptr + row)
    probabilities = tl.exp(logits - log_z)

    grad = tl.zeros((BLOCK_SIZE,), tl.float32)
    if HAS_LOGPROB_GRAD:
        grad_logprob = tl.load(grad_logprobs_ptr + row).to(tl.float32)
        grad += grad_logprob * ((offsets == index).to(tl.float32) - probabilities)
    if HAS_ENTROPY_GRAD:
        grad_entropy = tl.load(grad_entropy_ptr + row).to(tl.float32)
        expected_logit = tl.load(expected_logit_ptr + row)
        grad += grad_entropy * probabilities * (expected_logit - logits)
    tl.store(grad_logits_row_ptr + offsets, grad / TEMPERATURE, mask=mask)


def _layout(tensor: torch.Tensor) -> tuple[int, int, int, int]:
    if tensor.ndim == 1:
        return 1, 1, 0, 0
    if tensor.ndim == 2:
        return 1, tensor.shape[0], 0, tensor.stride(0)
    return tensor.shape[0], tensor.shape[1], tensor.stride(0), tensor.stride(1)


class _LogProbEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        logits: torch.Tensor,
        index: torch.Tensor,
        temperature: float,
        row_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, rows_per_batch, logits_batch_stride, logits_row_stride = _layout(logits)
        _, _, index_batch_stride, index_row_stride = _layout(index.unsqueeze(-1))
        if row_mask is None:
            row_mask_ptr = index
            mask_batch_stride = mask_row_stride = 0
        else:
            row_mask_ptr = row_mask
            _, _, mask_batch_stride, mask_row_stride = _layout(row_mask.unsqueeze(-1))
        n_rows = batch_size * rows_per_batch
        n_cols = logits.shape[-1]
        logprobs = torch.empty(logits.shape[:-1], device=logits.device, dtype=torch.float32)
        entropy = torch.empty_like(logprobs)
        log_z = torch.empty_like(logprobs)
        expected_logit = torch.empty_like(logprobs)

        _forward_kernel[(n_rows,)](
            logits,
            index,
            row_mask_ptr,
            logprobs,
            entropy,
            log_z,
            expected_logit,
            logits_batch_stride,
            logits_row_stride,
            index_batch_stride,
            index_row_stride,
            mask_batch_stride,
            mask_row_stride,
            n_cols=n_cols,
            rows_per_batch=rows_per_batch,
            TEMPERATURE=temperature,
            HAS_MASK=row_mask is not None,
            BLOCK_SIZE=_BLOCK_SIZE,
        )

        ctx.set_materialize_grads(False)
        ctx.save_for_backward(logits, index, log_z, expected_logit, row_mask)
        ctx.layout = batch_size, rows_per_batch, logits_batch_stride, logits_row_stride
        ctx.temperature = temperature
        return logprobs, entropy

    @staticmethod
    def backward(ctx, grad_logprobs: torch.Tensor | None, grad_entropy: torch.Tensor | None):
        logits, index, log_z, expected_logit, row_mask = ctx.saved_tensors
        batch_size, rows_per_batch, logits_batch_stride, logits_row_stride = ctx.layout
        n_rows = batch_size * rows_per_batch
        n_cols = logits.shape[-1]
        _, _, index_batch_stride, index_row_stride = _layout(index.unsqueeze(-1))
        if row_mask is None:
            row_mask_ptr = index
            mask_batch_stride = mask_row_stride = 0
        else:
            row_mask_ptr = row_mask
            _, _, mask_batch_stride, mask_row_stride = _layout(row_mask.unsqueeze(-1))
        grad_logits = torch.empty_like(logits, memory_format=torch.contiguous_format)
        _, _, grad_logits_batch_stride, grad_logits_row_stride = _layout(grad_logits)

        has_logprob_grad = grad_logprobs is not None
        has_entropy_grad = grad_entropy is not None
        if grad_logprobs is None:
            grad_logprobs = log_z
        else:
            grad_logprobs = grad_logprobs.contiguous()
        if grad_entropy is None:
            grad_entropy = expected_logit
        else:
            grad_entropy = grad_entropy.contiguous()
        _backward_kernel[(n_rows, triton.cdiv(n_cols, _BLOCK_SIZE))](
            logits,
            index,
            row_mask_ptr,
            log_z,
            expected_logit,
            grad_logprobs,
            grad_entropy,
            grad_logits,
            logits_batch_stride,
            logits_row_stride,
            index_batch_stride,
            index_row_stride,
            mask_batch_stride,
            mask_row_stride,
            grad_logits_batch_stride,
            grad_logits_row_stride,
            n_cols=n_cols,
            rows_per_batch=rows_per_batch,
            TEMPERATURE=ctx.temperature,
            HAS_MASK=row_mask is not None,
            BLOCK_SIZE=_BLOCK_SIZE,
            HAS_LOGPROB_GRAD=has_logprob_grad,
            HAS_ENTROPY_GRAD=has_entropy_grad,
        )
        return grad_logits, None, None, None


def selective_log_softmax_and_entropy(
    logits: torch.Tensor,
    index: torch.Tensor,
    temperature: float = 1.0,
    row_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute selected log-probabilities and Shannon entropy in one pass over logits."""
    return _LogProbEntropyFunction.apply(logits, index, temperature, row_mask)
