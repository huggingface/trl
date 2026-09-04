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
# Adapted from Liger-Kernel v0.8.2 (000be609), src/liger_kernel/chunked_loss/fused_linear_preference.py.

from abc import abstractmethod
from functools import partial

import torch
from torch.nn import functional as F


class FusedLinearPreferenceBase(torch.autograd.Function):
    @abstractmethod
    def preference_loss_fn(*args, **kwargs):
        """
        To be extended by subclasses.
        """
        raise NotImplementedError("Preference loss function must be implemented.")

    @staticmethod
    def forward(
        cls,
        ctx,
        _input,
        weight,
        target,
        bias=None,
        chunk_size=1,
        ignore_index=-100,
        alpha=1.0,
        beta=0.1,
        compute_nll_loss=True,
        nll_target=None,
        compiled=True,
        use_ref_model=False,
        ref_input=None,
        ref_weight=None,
        ref_bias=None,
        average_log_prob=True,
        **loss_kwargs,
    ):
        """
        Base class for fused linear layer with preference loss.
        Expects _input to be stacked with chosen and rejected inputs on the batch dimension.

        The mental model is:

        forward()
        ├── Loop over chunks
            └── compute_loss()
                ├── chunk_forward()  # Compute logits and log probs
                └── prefer_loss()    # Calculate preference loss

        Args:
            _input (torch.Tensor): Input tensor. Shape: (batch_size, seq_len, hidden_size).
            weight (torch.Tensor): Weight tensor. Shape: (vocab_size, hidden_size).
            target (torch.Tensor): Target tensor. Shape: (batch_size, seq_len).
            bias (torch.Tensor, optional): Bias tensor. Shape: (vocab_size,).
            loss_fn (callable): Loss function to compute the loss on a chunk of input/target.
            chunk_size (int): Size of a chunk (# of batches of stacked chosen and rejected inputs).
            ignore_index (int): Index to ignore for loss computation.
            alpha (float): Weight for the NLL loss.
            beta (float): Weight for the preference loss.
            compute_nll_loss (bool): Whether to compute NLL loss.
            nll_target (torch.Tensor, optional): Target tensor for NLL loss. Shape: (batch_size, seq_len). If not provided the target is used.
            compiled (bool): Whether to use torch compile for chunk accumulation.
            use_ref_model (bool): Whether to use a reference model for the alignment loss.
            ref_weight (torch.Tensor): Reference weight tensor. Shape: (vocab_size, hidden_size).
            ref_bias (torch.Tensor, optional): Reference bias tensor. Shape: (vocab_size,).
            average_log_prob (bool): Whether to average log probabilities or to sum them over the completion.
            loss_kwargs (dict): Other possible arguments that a loss function might need
        """
        # TODO: Tune CHUNK_SIZE to fully utilize the GPU
        CHUNK_SIZE = chunk_size

        # Gradients to be accumulated
        grad_weight = torch.zeros_like(weight)
        grad_chosen_inputs = []
        grad_rejected_inputs = []
        grad_bias = torch.zeros_like(bias) if bias is not None else None

        # Loss to be accumulated
        loss_acc = torch.zeros((), device=_input.device)

        # Metrics to be recorded
        policy_chosen_logps = []
        policy_rejected_logps = []
        policy_chosen_logits_mean = torch.zeros((), device=_input.device)
        policy_rejected_logits_mean = torch.zeros((), device=_input.device)
        policy_nll_loss = torch.zeros((), device=_input.device)
        aggregated_aux_outputs = []  # aggregated aux outputs from all chunks

        compute_loss = partial(
            FusedLinearPreferenceBase._compute_loss,
            preference_loss_fn=cls.preference_loss_fn,
            ignore_index=ignore_index,
            alpha=alpha,
            beta=beta,
            compute_nll_loss=compute_nll_loss,
            full_target=target,
            use_ref_model=use_ref_model,
            ref_weight=ref_weight,
            ref_bias=ref_bias,
            full_nll_target=nll_target,
            average_log_prob=average_log_prob,
            **loss_kwargs,
        )

        def fused_fwd_bwd(input_chunk, target_chunk, ref_input_chunk, chosen_nll_target_chunk):
            """
            Fused forward and backward pass for a chunk of input and target.
            """
            if bias is not None:
                return torch.func.grad_and_value(compute_loss, argnums=(0, 1, 3), has_aux=True)(
                    input_chunk,
                    weight,
                    target_chunk,
                    bias,
                    ref_input_chunk=ref_input_chunk,
                    chosen_nll_target_chunk=chosen_nll_target_chunk,
                )
            else:
                return torch.func.grad_and_value(compute_loss, argnums=(0, 1), has_aux=True)(
                    input_chunk,
                    weight,
                    target_chunk,
                    ref_input_chunk=ref_input_chunk,
                    chosen_nll_target_chunk=chosen_nll_target_chunk,
                )

        def accumulate_chunk(input_chunk, target_chunk, ref_input_chunk=None, chosen_nll_target_chunk=None):
            if bias is not None:
                (
                    (chunk_grad_input, chunk_grad_weight, chunk_grad_bias),
                    (
                        chunk_loss,
                        (
                            chunk_chosen_logps,
                            chunk_rejected_logps,
                            chunk_chosen_logits_mean,
                            chunk_rejected_logits_mean,
                            chunk_nll_loss,
                            *aux_outputs,
                        ),
                    ),
                ) = fused_fwd_bwd(input_chunk, target_chunk, ref_input_chunk, chosen_nll_target_chunk)
                grad_bias.add_(chunk_grad_bias)  # accumulate bias gradient
            else:
                (
                    (chunk_grad_input, chunk_grad_weight),
                    (
                        chunk_loss,
                        (
                            chunk_chosen_logps,
                            chunk_rejected_logps,
                            chunk_chosen_logits_mean,
                            chunk_rejected_logits_mean,
                            chunk_nll_loss,
                            *aux_outputs,
                        ),
                    ),
                ) = fused_fwd_bwd(input_chunk, target_chunk, ref_input_chunk, chosen_nll_target_chunk)

            # Accumulate gradients
            grad_weight.add_(chunk_grad_weight)
            grad_chosen_inputs.append(chunk_grad_input[: chosen_target_chunk.shape[0]])
            grad_rejected_inputs.append(chunk_grad_input[chosen_target_chunk.shape[0] :])

            # Accumulate loss
            loss_acc.add_(chunk_loss)

            # Accumulate metrics
            policy_chosen_logps.append(chunk_chosen_logps)
            policy_rejected_logps.append(chunk_rejected_logps)
            policy_chosen_logits_mean.add_(chunk_chosen_logits_mean)
            policy_rejected_logits_mean.add_(chunk_rejected_logits_mean)
            policy_nll_loss.add_(chunk_nll_loss)

            # aux_outputs
            # Initialize storage for aux_outputs
            if len(aggregated_aux_outputs) == 0:
                for aux in aux_outputs:
                    if aux.ndim == 0:
                        aggregated_aux_outputs.append(torch.zeros((), device=aux.device))
                    else:
                        aggregated_aux_outputs.append([])

            # Process each aux_output
            for i, aux in enumerate(aux_outputs):
                if aux.ndim == 0:
                    aggregated_aux_outputs[i].add_(aux)
                else:
                    aggregated_aux_outputs[i].append(aux)

        if compiled:
            fused_fwd_bwd = torch.compile(fused_fwd_bwd)

        len_chosen = target.shape[0] // 2
        chunks = max(1, _input.shape[0] // (2 * CHUNK_SIZE))
        _chosen_input_chunks = torch.chunk(_input[:len_chosen], chunks=chunks, dim=0)
        _chosen_target_chunks = torch.chunk(target[:len_chosen], chunks=chunks, dim=0)
        _rejected_input_chunks = torch.chunk(_input[len_chosen:], chunks=chunks, dim=0)
        _rejected_target_chunks = torch.chunk(target[len_chosen:], chunks=chunks, dim=0)

        if nll_target is not None:
            _chosen_nll_target_chunks = torch.chunk(nll_target[:len_chosen], chunks=chunks, dim=0)

        if use_ref_model:
            _ref_chosen_input_chunks = torch.chunk(ref_input[:len_chosen], chunks=chunks, dim=0)
            _ref_rejected_input_chunks = torch.chunk(ref_input[len_chosen:], chunks=chunks, dim=0)

        for (
            chosen_input_chunk,
            rejected_input_chunk,
            chosen_target_chunk,
            rejected_target_chunk,
            ref_chosen_input_chunk,
            ref_rejected_input_chunk,
            chosen_nll_target_chunk,
        ) in zip(
            _chosen_input_chunks,
            _rejected_input_chunks,
            _chosen_target_chunks,
            _rejected_target_chunks,
            (_ref_chosen_input_chunks if use_ref_model else [None] * len(_chosen_input_chunks)),
            (_ref_rejected_input_chunks if use_ref_model else [None] * len(_rejected_input_chunks)),
            (_chosen_nll_target_chunks if nll_target is not None else [None] * len(_chosen_input_chunks)),
            strict=True,
        ):
            input_chunk = torch.cat([chosen_input_chunk, rejected_input_chunk], dim=0)
            ref_input_chunk = (
                torch.cat([ref_chosen_input_chunk, ref_rejected_input_chunk], dim=0) if use_ref_model else None
            )
            target_chunk = torch.cat([chosen_target_chunk, rejected_target_chunk], dim=0)

            # mark input_chunk, target_chunk, and target dimension 1 as dynamic to prevent torch.compile recompilation
            torch._dynamo.mark_dynamic(input_chunk, 1)
            torch._dynamo.mark_dynamic(target_chunk, 1)
            torch._dynamo.mark_dynamic(target, 1)
            torch._dynamo.mark_dynamic(ref_input_chunk, 1) if use_ref_model else None
            torch._dynamo.mark_dynamic(chosen_nll_target_chunk, 1) if nll_target is not None else None

            # accumulate loss, gradients, and metrics
            accumulate_chunk(input_chunk, target_chunk, ref_input_chunk, chosen_nll_target_chunk)

        # combine grad_chosen_inputs and grad_rejected_inputs
        grad_inputs = grad_chosen_inputs + grad_rejected_inputs
        policy_chosen_logps = torch.cat(policy_chosen_logps, dim=0)
        policy_rejected_logps = torch.cat(policy_rejected_logps, dim=0)

        # Aggregate aux outputs lists into tensors
        for i, aux in enumerate(aggregated_aux_outputs):
            if isinstance(aux, list):
                aggregated_aux_outputs[i] = torch.cat(aux, dim=0)

        ctx.save_for_backward(
            torch.cat(grad_inputs, dim=0),
            grad_weight,
            grad_bias,
        )
        return_vars = (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits_mean,
            policy_rejected_logits_mean,
            policy_nll_loss,
        )
        return loss_acc, (*return_vars, *aggregated_aux_outputs)

    @staticmethod
    def backward(ctx, *grad_output):
        grad_input, grad_weight, grad_bias = ctx.saved_tensors
        if torch.ne(grad_output[0][0], torch.tensor(1.0, device=grad_output[0][0].device)):
            grad_input = grad_input * grad_output[0][0]
            grad_weight = grad_weight * grad_output[0][0]
            grad_bias = grad_bias * grad_output[0][0] if grad_bias is not None else None

        return grad_input, grad_weight, None, grad_bias, None, None, None, None

    @staticmethod
    def chunk_forward(
        input_chunk,
        weight,
        target_chunk,
        bias=None,
        ignore_index=-100,
        compute_nll_loss=True,
        chosen_nll_target_chunk=None,
        average_log_prob=True,
    ):
        len_chosen_chunk = target_chunk.shape[0] // 2
        logits_chunk = input_chunk @ weight.t()
        if bias is not None:
            logits_chunk = logits_chunk + bias
        log_probs_chunk = F.log_softmax(logits_chunk.float(), dim=-1)

        chosen_nll_loss = 0.0
        if compute_nll_loss:
            nll_labels = (
                chosen_nll_target_chunk if chosen_nll_target_chunk is not None else target_chunk[:len_chosen_chunk]
            )
            chosen_nll_loss = F.nll_loss(
                log_probs_chunk[:len_chosen_chunk].view(-1, log_probs_chunk.shape[-1]),
                nll_labels.view(-1),
                reduction="sum",
                ignore_index=ignore_index,
            )

        loss_mask = target_chunk != ignore_index
        label_chunk = torch.where(loss_mask, target_chunk, 0)

        per_token_logps = log_probs_chunk.gather(-1, label_chunk.unsqueeze(-1)).squeeze(-1)
        if average_log_prob:
            log_prob = (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            log_prob = (per_token_logps * loss_mask).sum(-1)

        chosen_logps = log_prob[:len_chosen_chunk]
        rejected_logps = log_prob[len_chosen_chunk:]

        chosen_logits = logits_chunk[:len_chosen_chunk]
        rejected_logits = logits_chunk[len_chosen_chunk:]

        return (
            chosen_logps,
            rejected_logps,
            chosen_logits,
            rejected_logits,
            chosen_nll_loss,
        )

    @staticmethod
    def _compute_loss(
        input_chunk,
        weight,
        target_chunk,
        bias=None,
        preference_loss_fn=None,
        full_target=None,
        ignore_index=-100,
        alpha=1.0,
        beta=0.1,
        compute_nll_loss=True,
        use_ref_model=False,
        ref_input_chunk=None,
        ref_weight=None,
        ref_bias=None,
        full_nll_target=None,
        chosen_nll_target_chunk=None,
        average_log_prob=True,
        **loss_kwargs,
    ):
        """
        Compute the total loss for a chunk of input and target, while using an alignment/preference loss function.
        Args:
            preference_loss_fn (callable): Loss function to compute the loss on a chunk of input/target.
            input_chunk (torch.Tensor): Chunk of input tensor. Shape: (2 * chunk_size, sequence_length, hidden_size).
            weight (torch.Tensor): Weight tensor. Shape: (vocab_size, hidden_size).
            target_chunk (torch.Tensor): Chunk of target tensor. Shape: (2 * chunk_size, sequence_length).
            bias (torch.Tensor, optional): Bias tensor. Shape: (vocab_size,).
            full_target (torch.Tensor): Full target tensor. Shape: (batch_size, sequence_length).
            ignore_index (int): Index to ignore for loss computation.
            alpha (float): Weight for the NLL loss.
            beta (float): Weight for the preference loss.
            compute_nll_loss (bool): Whether to compute NLL loss.
            use_ref_model (bool): Whether to use a reference model for the alignment loss.
            ref_weight (torch.Tensor): Reference weight tensor. Shape: (vocab_size, hidden_size).
            ref_bias (torch.Tensor, optional): Reference bias tensor. Shape: (vocab_size,).
            full_nll_target (torch.Tensor, optional): Full target tensor for NLL loss. Shape: (batch_size, sequence_length).
            chosen_nll_target_chunk (torch.Tensor, optional): Target tensor for NLL loss. Shape: (chunk_size, sequence_length) If not provided the target_chunk is used.
            average_log_prob (bool): Whether to average log probabilities or the sum.
            loss_kwargs (dict): Additional arguments for the loss function.
        """
        (
            chosen_logps,
            rejected_logps,
            chosen_logits,
            rejected_logits,
            chosen_nll_loss,
        ) = FusedLinearPreferenceBase.chunk_forward(
            input_chunk,
            weight,
            target_chunk,
            bias=bias,
            ignore_index=ignore_index,
            compute_nll_loss=compute_nll_loss,
            chosen_nll_target_chunk=chosen_nll_target_chunk,
            average_log_prob=average_log_prob,
        )
        if full_nll_target is not None:
            chosen_nll_loss = (
                chosen_nll_loss / (full_nll_target[: full_nll_target.shape[0] // 2] != ignore_index).sum()
            )
        else:
            chosen_nll_loss = chosen_nll_loss / (full_target[: full_target.shape[0] // 2] != ignore_index).sum()

        chosen_logits_mean = chosen_logits.sum() / (full_target.shape[0] // 2 * input_chunk.shape[1] * weight.shape[0])
        rejected_logits_mean = rejected_logits.sum() / (
            full_target.shape[0] // 2 * input_chunk.shape[1] * weight.shape[0]
        )

        if use_ref_model:
            with torch.no_grad():
                (
                    ref_chosen_logps,
                    ref_rejected_logps,
                    _,
                    _,
                    _,
                ) = FusedLinearPreferenceBase.chunk_forward(
                    ref_input_chunk,
                    ref_weight,
                    target_chunk,
                    ref_bias,
                    ignore_index=ignore_index,
                    compute_nll_loss=False,  # We don't need NLL loss for the reference model
                    chosen_nll_target_chunk=None,
                    average_log_prob=average_log_prob,
                )
            loss_kwargs["ref_chosen_logps"] = ref_chosen_logps
            loss_kwargs["ref_rejected_logps"] = ref_rejected_logps

        preference_loss_outputs = preference_loss_fn(
            chosen_logps, rejected_logps, full_target, beta=beta, **loss_kwargs
        )
        if isinstance(preference_loss_outputs, tuple):
            preference_loss, *aux_outputs = preference_loss_outputs
        else:
            preference_loss, aux_outputs = preference_loss_outputs, []

        loss = alpha * chosen_nll_loss + preference_loss
        return_vars = (
            chosen_logps,
            rejected_logps,
            chosen_logits_mean,
            rejected_logits_mean,
            chosen_nll_loss,
        )
        return loss, (*return_vars, *aux_outputs)
