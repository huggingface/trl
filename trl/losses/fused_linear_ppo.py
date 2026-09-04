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

# Copyright 2024 LinkedIn Corporation
# All Rights Reserved.
#
# This source code is licensed under the BSD 2-Clause license found in the
# LICENSE file in the root directory of https://github.com/linkedin/Liger-Kernel.
#
# Adapted from Liger-Kernel v0.8.2 (000be609), src/liger_kernel/chunked_loss/fused_linear_ppo.py.

from abc import abstractmethod
from functools import partial

import torch
import torch._dynamo.config


_SELECTIVE_LOGPROB_VOCAB_CHUNK_SIZE = 4096
_SELECTIVE_LOGPROB_SEQ_CHUNK_SIZE = 2048


def _maybe_mark_dynamic_dim1(tensor):
    if tensor is not None:
        torch._dynamo.maybe_mark_dynamic(tensor, 1)


def _selective_logprob_forward(
    hidden, weight, targets, bias=None, temperature=1.0, vocab_chunk_size=_SELECTIVE_LOGPROB_VOCAB_CHUNK_SIZE
):
    """Compute selective log-probabilities by streaming over sequence and vocab chunks.

    Dual chunking (sequence × vocab) bounds peak temporary memory to
    ``seq_chunk_size × vocab_chunk_size`` regardless of total N or V.
    """
    device = hidden.device
    n_rows, _ = hidden.shape
    vocab_size, _ = weight.shape
    inv_t = 1.0 / temperature
    seq_chunk_size = _SELECTIVE_LOGPROB_SEQ_CHUNK_SIZE

    logprobs = torch.empty((n_rows,), device=device, dtype=torch.float32)
    log_z = torch.empty((n_rows,), device=device, dtype=torch.float32)

    for seq_start in range(0, n_rows, seq_chunk_size):
        seq_end = min(seq_start + seq_chunk_size, n_rows)
        n_chunk = seq_end - seq_start
        hidden_chunk = hidden[seq_start:seq_end]
        targets_chunk = targets[seq_start:seq_end]

        max_old = torch.full((n_chunk,), float("-inf"), device=device, dtype=torch.float32)
        sum_exp = torch.zeros((n_chunk,), device=device, dtype=torch.float32)
        target_logit = torch.zeros((n_chunk,), device=device, dtype=torch.float32)
        row_idx = torch.arange(n_chunk, device=device)

        for vocab_start in range(0, vocab_size, vocab_chunk_size):
            vocab_end = min(vocab_start + vocab_chunk_size, vocab_size)
            weight_chunk = weight[vocab_start:vocab_end]
            logits_chunk = (hidden_chunk @ weight_chunk.to(hidden.dtype).t()).float()
            if bias is not None:
                logits_chunk.add_(bias[vocab_start:vocab_end].to(torch.float32))
            logits_chunk.mul_(inv_t)

            chunk_max = logits_chunk.amax(dim=-1)
            max_new = torch.maximum(max_old, chunk_max)
            rescale = torch.exp(max_old - max_new)
            chunk_exp = torch.exp(logits_chunk - max_new.unsqueeze(-1))

            sum_exp = sum_exp * rescale + chunk_exp.sum(dim=-1)
            max_old = max_new

            in_chunk = (targets_chunk >= vocab_start) & (targets_chunk < vocab_end)
            local_idx = torch.clamp(targets_chunk - vocab_start, 0, vocab_end - vocab_start - 1)
            target_logit += logits_chunk[row_idx, local_idx] * in_chunk

        log_z_chunk = max_old + torch.log(sum_exp)
        log_z[seq_start:seq_end] = log_z_chunk
        logprobs[seq_start:seq_end] = target_logit - log_z_chunk

    return logprobs, log_z


def _selective_logprob_backward(hidden, weight, targets, bias, log_z, grad_logprobs, temperature, vocab_chunk_size):
    """Dual-chunked (sequence × vocab) backward for selective logprob.

    Recomputes logits per chunk for memory efficiency.
    """
    inv_t = 1.0 / temperature
    n_rows, _ = hidden.shape
    vocab_size = weight.shape[0]
    has_bias = bias is not None
    seq_chunk_size = _SELECTIVE_LOGPROB_SEQ_CHUNK_SIZE

    grad_hidden = torch.zeros(hidden.shape, device=hidden.device, dtype=torch.float32)
    grad_weight = torch.zeros(weight.shape, device=weight.device, dtype=torch.float32)
    grad_bias = torch.zeros((vocab_size,), device=weight.device, dtype=torch.float32) if has_bias else None

    grad_logprobs = grad_logprobs.to(torch.float32)

    for seq_start in range(0, n_rows, seq_chunk_size):
        seq_end = min(seq_start + seq_chunk_size, n_rows)
        hidden_chunk = hidden[seq_start:seq_end]
        targets_chunk = targets[seq_start:seq_end]
        grad_chunk = grad_logprobs[seq_start:seq_end]
        logz_chunk = log_z[seq_start:seq_end]
        row_idx = torch.arange(seq_end - seq_start, device=hidden.device)

        for vocab_start in range(0, vocab_size, vocab_chunk_size):
            vocab_end = min(vocab_start + vocab_chunk_size, vocab_size)
            weight_chunk = weight[vocab_start:vocab_end]
            logits_chunk = (hidden_chunk @ weight_chunk.to(hidden.dtype).t()).float()
            if has_bias:
                logits_chunk.add_(bias[vocab_start:vocab_end].to(torch.float32))
            logits_chunk.mul_(inv_t)

            probs = torch.exp(logits_chunk - logz_chunk.unsqueeze(-1))
            grad_logits = (-grad_chunk).unsqueeze(-1) * probs

            in_chunk = (targets_chunk >= vocab_start) & (targets_chunk < vocab_end)
            local_idx = torch.clamp(targets_chunk - vocab_start, 0, vocab_end - vocab_start - 1)
            grad_logits[row_idx, local_idx] += grad_chunk * in_chunk
            grad_logits.mul_(inv_t)

            grad_hidden[seq_start:seq_end].add_(grad_logits @ weight_chunk.float())
            grad_weight[vocab_start:vocab_end].add_(grad_logits.t() @ hidden_chunk.float())
            if has_bias:
                grad_bias[vocab_start:vocab_end].add_(grad_logits.sum(dim=0))

    return grad_hidden, grad_weight, grad_bias


class _ChunkedSelectiveLogProbFunction(torch.autograd.Function):
    """Custom autograd function for memory-efficient selective logprob.

    Forward: streams over vocab chunks, only stores hidden/weight/targets/log_z.
    Backward: recomputes logits per chunk instead of storing all intermediates.
    """

    @staticmethod
    def forward(ctx, hidden, weight, targets, bias, temperature, vocab_chunk_size):
        logprobs, log_z = _selective_logprob_forward(hidden, weight, targets, bias, temperature, vocab_chunk_size)
        if bias is None:
            bias = hidden.new_empty((0,))
        ctx.save_for_backward(hidden, weight, targets, bias, log_z)
        ctx.has_bias = bias.numel() > 0
        ctx.temperature = temperature
        ctx.vocab_chunk_size = vocab_chunk_size
        return logprobs

    @staticmethod
    def backward(ctx, grad_logprobs):
        hidden, weight, targets, bias, log_z = ctx.saved_tensors
        grad_hidden, grad_weight, grad_bias = _selective_logprob_backward(
            hidden=hidden,
            weight=weight,
            targets=targets,
            bias=bias if ctx.has_bias else None,
            log_z=log_z,
            grad_logprobs=grad_logprobs,
            temperature=ctx.temperature,
            vocab_chunk_size=ctx.vocab_chunk_size,
        )
        return (
            grad_hidden.to(hidden.dtype),
            grad_weight.to(weight.dtype),
            None,
            grad_bias.to(bias.dtype) if ctx.has_bias else None,
            None,
            None,
        )


class FusedLinearPPOBase(torch.autograd.Function):
    @abstractmethod
    def ppo_loss_fn(*args, **kwargs):
        """
        To be extended by subclasses.
        """
        raise NotImplementedError("PPO loss function must be implemented.")

    @staticmethod
    def forward(
        cls,
        ctx,
        _input,
        weight,
        selected_token_ids,
        attention_mask,
        advantages,
        bias=None,
        ref_per_token_logps=None,
        old_per_token_logps=None,
        ref_input=None,
        ref_weight=None,
        ref_bias=None,
        epsilon_low=0.2,
        epsilon_high=0.2,
        beta=0.04,
        loss_type="dapo",
        max_completion_length=None,
        importance_sampling_level="token",
        temperature=1.0,
        compiled=True,
        use_ref_model=False,
        chunk_size=1,
        sapo_temperature_pos=1.0,
        sapo_temperature_neg=1.05,
        vllm_is_ratio=None,
        delta=None,
        use_bias_correction_kl=False,
        vespo_k_pos=2.0,
        vespo_lambda_pos=3.0,
        vespo_k_neg=3.0,
        vespo_lambda_neg=2.0,
        num_items_in_batch=None,
    ):
        """Chunked forward pass for PPO loss computation.

        Hybrid approach: chunk_forward (custom autograd, memory-efficient) runs OUTSIDE
        torch.compile; only the loss math (ppo_loss_fn) is compiled.
        """
        if use_ref_model:
            assert ref_per_token_logps is not None or ref_input is not None, (
                "If use_ref_model is True, ref_per_token_logps or ref_input must be provided"
            )
            if ref_per_token_logps is not None and ref_input is not None:
                raise Warning("Both ref_per_token_logps and ref_input are provided. Using ref_per_token_logps.")
        if loss_type == "dr_grpo":
            assert max_completion_length is not None, "max_completion_length must be provided for loss_type 'dr_grpo'"
        if vllm_is_ratio is not None:
            B, T = attention_mask.shape
            assert vllm_is_ratio.dim() in (1, 2), (
                f"vllm_is_ratio must be 1D (B,) or 2D (B, T) / (B, 1), got {vllm_is_ratio.dim()}D"
            )
            if vllm_is_ratio.dim() == 2:
                assert vllm_is_ratio.shape[0] == B and vllm_is_ratio.shape[1] in (1, T), (
                    f"vllm_is_ratio shape must be ({B}, 1) or ({B}, {T}), got {tuple(vllm_is_ratio.shape)}"
                )
            else:
                assert vllm_is_ratio.shape[0] == B, (
                    f"vllm_is_ratio shape must be ({B},), got {tuple(vllm_is_ratio.shape)}"
                )
                vllm_is_ratio = vllm_is_ratio.unsqueeze(-1)  # (B,) -> (B, 1) for broadcasting
        # Initialize accumulators
        loss_acc = torch.zeros((), device=_input.device, dtype=torch.float32)
        grad_weight = torch.zeros_like(weight)  # [V, H]
        grad_inputs = []
        grad_bias = torch.zeros_like(bias) if bias is not None else None  # [V]
        aggregated_metrics = []

        # Only compile the loss math, NOT chunk_forward (which uses custom autograd.Function)
        compute_loss = partial(
            FusedLinearPPOBase._compute_loss_from_logps,
            full_attention_mask=attention_mask,
            epsilon_low=epsilon_low,
            epsilon_high=epsilon_high,
            beta=beta,
            loss_type=loss_type,
            max_completion_length=max_completion_length,
            importance_sampling_level=importance_sampling_level,
            ppo_loss_fn=cls.ppo_loss_fn,
            sapo_temperature_pos=sapo_temperature_pos,
            sapo_temperature_neg=sapo_temperature_neg,
            delta=delta,
            use_bias_correction_kl=use_bias_correction_kl,
            vespo_k_pos=vespo_k_pos,
            vespo_lambda_pos=vespo_lambda_pos,
            vespo_k_neg=vespo_k_neg,
            vespo_lambda_neg=vespo_lambda_neg,
            num_items_in_batch=num_items_in_batch,
        )
        compiled_compute_loss = torch.compile(compute_loss) if compiled else compute_loss

        def fused_fwd_bwd(
            input_chunk,
            selected_token_ids_chunk,
            attention_mask_chunk,
            advantages_chunk,
            ref_per_token_logps_chunk,
            old_per_token_logps_chunk,
            ref_input_chunk,
            vllm_is_ratio_chunk,
        ):
            """Fused forward and backward for a chunk."""
            with torch.enable_grad():
                input_chunk = input_chunk.detach().requires_grad_(True)
                weight_local = weight.detach().requires_grad_(True)
                bias_local = bias.detach().requires_grad_(True) if bias is not None else None

                # Step 1: compute logprobs OUTSIDE compile (custom autograd, memory-efficient)
                per_token_logps = FusedLinearPPOBase.chunk_forward(
                    input_chunk,
                    weight_local,
                    selected_token_ids_chunk,
                    bias=bias_local,
                    temperature=temperature,
                )

                # Compute ref logprobs if needed (also outside compile)
                if use_ref_model and ref_per_token_logps_chunk is None:
                    with torch.no_grad():
                        ref_per_token_logps_chunk = FusedLinearPPOBase.chunk_forward(
                            ref_input_chunk,
                            ref_weight,
                            selected_token_ids_chunk,
                            bias=ref_bias,
                            temperature=temperature,
                        )

                # Step 2: compute loss INSIDE compile (just math, torch.compile-friendly)
                chunk_loss, chunk_metrics = compiled_compute_loss(
                    per_token_logps,
                    attention_mask_chunk,
                    advantages_chunk,
                    ref_per_token_logps_chunk=ref_per_token_logps_chunk,
                    old_per_token_logps_chunk=old_per_token_logps_chunk,
                    vllm_is_ratio_chunk=vllm_is_ratio_chunk,
                )

                grad_targets = [input_chunk, weight_local]
                if bias_local is not None:
                    grad_targets.append(bias_local)
                grads = torch.autograd.grad(chunk_loss, grad_targets)
            return grads, (chunk_loss.detach(), tuple(metric.detach() for metric in chunk_metrics))

        def accumulate_chunk(
            input_chunk,
            selected_token_ids_chunk,
            attention_mask_chunk,
            advantages_chunk,
            ref_per_token_logps_chunk=None,
            old_per_token_logps_chunk=None,
            ref_input_chunk=None,
            vllm_is_ratio_chunk=None,
        ):
            (chunk_grad_input, chunk_grad_weight, *chunk_grad_bias), (chunk_loss, chunk_metrics) = fused_fwd_bwd(
                input_chunk,
                selected_token_ids_chunk,
                attention_mask_chunk,
                advantages_chunk,
                ref_per_token_logps_chunk,
                old_per_token_logps_chunk,
                ref_input_chunk,
                vllm_is_ratio_chunk,
            )
            if bias is not None:
                grad_bias.add_(chunk_grad_bias[0])

            # Accumulate gradients and loss
            grad_weight.add_(chunk_grad_weight)
            grad_inputs.append(chunk_grad_input)
            loss_acc.add_(chunk_loss)
            # Initialize storage for metrics on first chunk
            if len(aggregated_metrics) == 0:
                for metric in chunk_metrics:
                    if metric.ndim == 0:
                        aggregated_metrics.append(torch.zeros((), device=metric.device))
                    else:
                        aggregated_metrics.append([])

            # Accumulate metrics
            for i, metric in enumerate(chunk_metrics):
                if metric.ndim == 0:
                    aggregated_metrics[i].add_(metric)
                else:
                    aggregated_metrics[i].append(metric)

        # Process input in chunks based on chunk_size
        chunks = max(1, _input.shape[0] // chunk_size)
        _input_chunks = torch.chunk(_input, chunks=chunks, dim=0)
        _selected_token_ids_chunks = torch.chunk(selected_token_ids, chunks=chunks, dim=0)
        _attention_mask_chunks = torch.chunk(attention_mask, chunks=chunks, dim=0)
        _advantages_chunks = torch.chunk(advantages, chunks=chunks, dim=0)
        _ref_per_token_logps_chunks = (
            torch.chunk(ref_per_token_logps, chunks=chunks, dim=0)
            if use_ref_model and ref_per_token_logps is not None
            else [None] * chunks
        )
        _old_per_token_logps_chunks = (
            torch.chunk(old_per_token_logps, chunks=chunks, dim=0)
            if old_per_token_logps is not None
            else [None] * chunks
        )
        # If ref_per_token_logps is provided, we don't need ref_input to calculate the reference log probs.
        _ref_input_chunks = (
            torch.chunk(ref_input, chunks=chunks, dim=0)
            if use_ref_model and ref_per_token_logps is None
            else [None] * chunks
        )
        _vllm_is_ratio_chunks = (
            torch.chunk(vllm_is_ratio, chunks=chunks, dim=0) if vllm_is_ratio is not None else [None] * chunks
        )

        for (
            input_chunk,
            selected_token_ids_chunk,
            attention_mask_chunk,
            advantages_chunk,
            ref_per_token_logps_chunk,
            old_per_token_logps_chunk,
            ref_input_chunk,
            vllm_is_ratio_chunk,
        ) in zip(
            _input_chunks,
            _selected_token_ids_chunks,
            _attention_mask_chunks,
            _advantages_chunks,
            _ref_per_token_logps_chunks,
            _old_per_token_logps_chunks,
            _ref_input_chunks,
            _vllm_is_ratio_chunks,
            strict=True,
        ):
            # Allow torch.compile to generalize sequence length without forcing it to be dynamic.
            _maybe_mark_dynamic_dim1(input_chunk)
            _maybe_mark_dynamic_dim1(selected_token_ids_chunk)
            _maybe_mark_dynamic_dim1(attention_mask_chunk)
            _maybe_mark_dynamic_dim1(ref_per_token_logps_chunk)
            _maybe_mark_dynamic_dim1(ref_input_chunk)
            _maybe_mark_dynamic_dim1(old_per_token_logps_chunk)
            _maybe_mark_dynamic_dim1(vllm_is_ratio_chunk)

            accumulate_chunk(
                input_chunk,
                selected_token_ids_chunk,
                attention_mask_chunk,
                advantages_chunk,
                ref_per_token_logps_chunk,
                old_per_token_logps_chunk,
                ref_input_chunk,
                vllm_is_ratio_chunk,
            )

        # Combine gradients
        grad_input = torch.cat(grad_inputs, dim=0)

        # Save for backward
        ctx.save_for_backward(grad_input, grad_weight, grad_bias)

        # Finalize metrics
        final_metrics = []
        for metric in aggregated_metrics:
            if isinstance(metric, list):
                final_metrics.append(torch.cat(metric, dim=0))
            else:
                final_metrics.append(metric)

        return loss_acc, tuple(final_metrics)

    @staticmethod
    def _compute_dapo_normalizer(attention_mask, num_items_in_batch=None):
        """Per-process normalizer for DAPO/CISPO/VESPO.

        When ``num_items_in_batch`` is provided it is used directly, matching
        TRL's ``num_items_in_batch / num_processes`` — the total active tokens
        across the entire generation batch (all grad-accum micro-batches × all
        processes). Falling back to the current micro-batch's mask biases the
        per-token weight by micro-batch size when grad-accum steps have
        unequal completion lengths.
        """
        world_size = 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            import torch.distributed as dist

            world_size = dist.get_world_size()

        if num_items_in_batch is not None:
            if isinstance(num_items_in_batch, torch.Tensor):
                normalizer = num_items_in_batch.to(device=attention_mask.device, dtype=torch.float32)
            else:
                normalizer = torch.as_tensor(
                    float(num_items_in_batch), device=attention_mask.device, dtype=torch.float32
                )
            normalizer = normalizer / world_size
            return torch.clamp(normalizer, min=1.0)

        normalizer = attention_mask.to(torch.float32).sum()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            import torch.distributed as dist

            normalizer = normalizer.clone()
            dist.all_reduce(normalizer, op=dist.ReduceOp.SUM)

        normalizer = normalizer / world_size
        return torch.clamp(normalizer, min=1.0)

    @staticmethod
    def _compute_loss_from_logps(
        per_token_logps,
        attention_mask_chunk,
        advantages_chunk,
        ref_per_token_logps_chunk=None,
        old_per_token_logps_chunk=None,
        vllm_is_ratio_chunk=None,
        full_attention_mask=None,
        epsilon_low=0.2,
        epsilon_high=0.2,
        beta=0.04,
        loss_type="dapo",
        max_completion_length=None,
        importance_sampling_level="token",
        ppo_loss_fn=None,
        sapo_temperature_pos=1.0,
        sapo_temperature_neg=1.05,
        delta=None,
        use_bias_correction_kl=False,
        vespo_k_pos=2.0,
        vespo_lambda_pos=3.0,
        vespo_k_neg=3.0,
        vespo_lambda_neg=2.0,
        num_items_in_batch=None,
    ):
        """Compute loss from pre-computed logprobs. This is the torch.compile-friendly part."""
        chunk_loss, chunk_metrics = ppo_loss_fn(
            per_token_logps=per_token_logps,
            attention_mask=attention_mask_chunk,
            advantages=advantages_chunk,
            full_attention_mask=full_attention_mask,
            ref_per_token_logps=ref_per_token_logps_chunk.float() if ref_per_token_logps_chunk is not None else None,
            old_per_token_logps=old_per_token_logps_chunk.float() if old_per_token_logps_chunk is not None else None,
            epsilon_low=epsilon_low,
            epsilon_high=epsilon_high,
            beta=beta,
            loss_type=loss_type,
            max_completion_length=max_completion_length,
            importance_sampling_level=importance_sampling_level,
            sapo_temperature_pos=sapo_temperature_pos,
            sapo_temperature_neg=sapo_temperature_neg,
            vllm_is_ratio=vllm_is_ratio_chunk,
            delta=delta,
            use_bias_correction_kl=use_bias_correction_kl,
            vespo_k_pos=vespo_k_pos,
            vespo_lambda_pos=vespo_lambda_pos,
            vespo_k_neg=vespo_k_neg,
            vespo_lambda_neg=vespo_lambda_neg,
            num_items_in_batch=num_items_in_batch,
        )
        return chunk_loss, chunk_metrics

    @staticmethod
    def chunk_forward(input_chunk, weight, selected_token_ids, bias=None, temperature=1.0):
        """Compute selected-token log probabilities without materializing full vocab logits.

        Uses _ChunkedSelectiveLogProbFunction for memory-efficient custom backward
        (recomputes logits per vocab chunk instead of storing all intermediates).
        """
        batch_size, seq_len, hidden_size = input_chunk.shape
        hidden = input_chunk.reshape(batch_size * seq_len, hidden_size).contiguous()
        targets = selected_token_ids.reshape(batch_size * seq_len).contiguous()
        per_token_logps = _ChunkedSelectiveLogProbFunction.apply(
            hidden,
            weight,
            targets,
            bias,
            temperature,
            _SELECTIVE_LOGPROB_VOCAB_CHUNK_SIZE,
        )
        return per_token_logps.reshape(batch_size, seq_len)

    @staticmethod
    def backward(ctx, grad_output, *grad_metrics):
        """Backward pass for PPO loss."""
        grad_input, grad_weight, grad_bias = ctx.saved_tensors

        if grad_output != 1.0:
            grad_input = grad_input * grad_output
            grad_weight = grad_weight * grad_output
            if grad_bias is not None:
                grad_bias = grad_bias * grad_output

        return (
            grad_input,
            grad_weight,
            None,  # grad_selected_token_ids
            None,  # grad_attention_mask
            None,  # grad_advantages
            grad_bias,
            None,  # grad_ref_per_token_logps
            None,  # grad_old_per_token_logps
            None,  # grad_ref_input
            None,  # grad_ref_weight
            None,  # grad_ref_bias
            None,  # grad_epsilon_low
            None,  # grad_epsilon_high
            None,  # grad_beta
            None,  # grad_loss_type
            None,  # grad_max_completion_length
            None,  # grad_importance_sampling_level
            None,  # grad_temperature
            None,  # grad_compiled
            None,  # grad_use_ref_model
            None,  # grad_chunk_size
            None,  # grad_sapo_temperature_pos
            None,  # grad_sapo_temperature_neg
            None,  # grad_vllm_is_ratio
            None,  # grad_delta
            None,  # grad_use_bias_correction_kl
            None,  # grad_num_items_in_batch
        )
