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
# Adapted from Liger-Kernel v0.8.2 (000be609), src/liger_kernel/chunked_loss/fused_linear_unpaired_preference.py.

from abc import abstractmethod
from functools import partial

import torch
from torch.nn import functional as F


class FusedLinearUnpairedPreferenceBase(torch.autograd.Function):
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
        preference_labels,
        bias=None,
        chunk_size=1,
        ignore_index=-100,
        compiled=True,
        use_ref_model=False,
        ref_input=None,
        ref_weight=None,
        ref_bias=None,
        average_log_prob=False,
        **loss_kwargs,
    ):
        """
        Base class for fused linear layer with unpaired preference loss like KTO
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
            beta (float): Weight for the preference loss.
            compiled (bool): Whether to use torch compile for chunk accumulation.
            use_ref_model (bool): Whether to use a reference model for the alignment loss.
            preference_labels (torch.Tensor): Boolean tensor indicating chosen (True) vs rejected (False) examples.
                Shape: (batch_size,).
            ref_weight (torch.Tensor): Reference weight tensor. Shape: (vocab_size, hidden_size).
            ref_bias (torch.Tensor, optional): Reference bias tensor. Shape: (vocab_size,).
            average_log_prob (bool): Whether to average the log probability per non-masked token.
            loss_kwargs (dict): Other possible arguments that a loss function might need
        """
        # TODO: Tune CHUNK_SIZE to fully utilize the GPU
        CHUNK_SIZE = chunk_size

        # Gradients to be accumulated
        grad_inputs = []
        grad_weight = torch.zeros_like(weight)
        grad_bias = torch.zeros_like(bias) if bias is not None else None

        # Loss to be accumulated
        loss_acc = torch.zeros((), device=_input.device)

        # Metrics to be recorded
        chosen_logps_sum = torch.zeros((), device=_input.device)
        rejected_logps_sum = torch.zeros((), device=_input.device)
        chosen_logits_sum = torch.zeros((), device=_input.device)
        rejected_logits_sum = torch.zeros((), device=_input.device)
        aggregated_aux_outputs = []

        compute_loss = partial(
            FusedLinearUnpairedPreferenceBase._compute_loss,
            preference_loss_fn=cls.preference_loss_fn,
            full_target=target,
            ignore_index=ignore_index,
            use_ref_model=use_ref_model,
            ref_weight=ref_weight,
            ref_bias=ref_bias,
            average_log_prob=average_log_prob,
            **loss_kwargs,
        )

        def fused_fwd_bwd(input_chunk, target_chunk, preference_labels_chunk, ref_input_chunk):
            """
            Fused forward and backward pass for a chunk of input and target.
            """
            argnums = (0, 1, 4) if bias is not None else (0, 1)
            return torch.func.grad_and_value(compute_loss, argnums=argnums, has_aux=True)(
                input_chunk,
                weight,
                target_chunk,
                preference_labels_chunk,
                bias,
                ref_input_chunk=ref_input_chunk,
            )

        def accumulate_chunk(
            input_chunk,
            target_chunk,
            preference_labels_chunk=None,
            ref_input_chunk=None,
        ):
            (
                (chunk_grad_input, chunk_grad_weight, *chunk_grad_bias),
                (
                    chunk_loss,
                    (
                        chunk_chosen_logps_sum,
                        chunk_rejected_logps_sum,
                        chunk_chosen_logits_sum,
                        chunk_rejected_logits_sum,
                        *aux_outputs,
                    ),
                ),
            ) = fused_fwd_bwd(input_chunk, target_chunk, preference_labels_chunk, ref_input_chunk)
            if bias is not None:
                grad_bias.add_(chunk_grad_bias[0])  # accumulate bias gradient

            # Accumulate gradients
            grad_weight.add_(chunk_grad_weight)
            grad_inputs.append(chunk_grad_input)

            # Accumulate loss
            loss_acc.add_(chunk_loss)

            # Accumulate metrics
            chosen_logps_sum.add_(chunk_chosen_logps_sum)
            rejected_logps_sum.add_(chunk_rejected_logps_sum)
            chosen_logits_sum.add_(chunk_chosen_logits_sum)
            rejected_logits_sum.add_(chunk_rejected_logits_sum)

            # aux_outputs
            # Initialize storage for aux_outputs
            if len(aggregated_aux_outputs) == 0:
                for aux in aux_outputs:
                    aggregated_aux_outputs.append(torch.zeros((), device=aux.device))

            # Process each aux_output
            for i, aux in enumerate(aux_outputs):
                if aux.ndim == 0:
                    aggregated_aux_outputs[i].add_(aux)

        if compiled:
            fused_fwd_bwd = torch.compile(fused_fwd_bwd)

        # When not paired, use labels to separate chosen and rejected
        assert preference_labels is not None, "preference_labels must be provided for unpaired preference loss"

        chunks = max(1, _input.shape[0] // CHUNK_SIZE)
        _input_chunks = torch.chunk(_input, chunks=chunks, dim=0)
        _target_chunks = torch.chunk(target, chunks=chunks, dim=0)
        _preference_labels_chunks = torch.chunk(preference_labels, chunks=chunks, dim=0)

        if use_ref_model:
            _ref_input_chunks = torch.chunk(ref_input, chunks=chunks, dim=0)

        for (
            input_chunk,
            target_chunk,
            ref_input_chunk,
            preference_labels_chunk,
        ) in zip(
            _input_chunks,
            _target_chunks,
            (_ref_input_chunks if use_ref_model else [None] * len(_input_chunks)),
            _preference_labels_chunks,
            strict=True,
        ):
            # mark input_chunk, target_chunk, and target dimension 1 (sequence length) as dynamic to prevent torch.compile recompilation
            torch._dynamo.mark_dynamic(input_chunk, 1)
            torch._dynamo.mark_dynamic(target_chunk, 1)
            torch._dynamo.mark_dynamic(target, 1)
            torch._dynamo.mark_dynamic(ref_input_chunk, 1) if use_ref_model else None
            torch._dynamo.mark_dynamic(preference_labels_chunk, 1)

            # accumulate loss, gradients, and metrics
            accumulate_chunk(input_chunk, target_chunk, preference_labels_chunk, ref_input_chunk)

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
            chosen_logps_sum,
            rejected_logps_sum,
            chosen_logits_sum,
            rejected_logits_sum,
        )

        return loss_acc, (*return_vars, *aggregated_aux_outputs)

    @staticmethod
    def backward(ctx, *grad_output):
        grad_input, grad_weight, grad_bias = ctx.saved_tensors
        if torch.ne(grad_output[0][0], torch.tensor(1.0, device=grad_output[0][0].device)):
            grad_input = grad_input * grad_output[0][0]
            grad_weight = grad_weight * grad_output[0][0]
            grad_bias = grad_bias * grad_output[0][0] if grad_bias is not None else None

        return grad_input, grad_weight, None, None, grad_bias

    @staticmethod
    def chunk_forward(
        input_chunk,
        weight,
        target_chunk,
        preference_labels_chunk,
        bias=None,
        ignore_index=-100,
        average_log_prob=False,
    ):
        logits_chunk = input_chunk @ weight.t()
        if bias is not None:
            logits_chunk = logits_chunk + bias
        log_probs_chunk = F.log_softmax(logits_chunk.float(), dim=-1)
        loss_mask_chunk = target_chunk != ignore_index
        label_chunk = torch.where(loss_mask_chunk, target_chunk, 0)

        per_token_logps_chunk = log_probs_chunk.gather(-1, label_chunk.unsqueeze(-1)).squeeze(-1)
        if average_log_prob:
            log_probs = (per_token_logps_chunk * loss_mask_chunk).sum(-1) / loss_mask_chunk.sum(-1)
        else:
            log_probs = (per_token_logps_chunk * loss_mask_chunk).sum(-1)

        chosen_logps_sum = (log_probs * preference_labels_chunk.unsqueeze(1)).sum()
        rejected_logps_sum = (log_probs * (~preference_labels_chunk).unsqueeze(1)).sum()

        chosen_logits_sum = (logits_chunk * preference_labels_chunk.unsqueeze(1)).sum()
        rejected_logits_sum = (logits_chunk * (~preference_labels_chunk).unsqueeze(1)).sum()

        return (
            log_probs,
            chosen_logps_sum,
            rejected_logps_sum,
            chosen_logits_sum,
            rejected_logits_sum,
        )

    @staticmethod
    def _compute_loss(
        input_chunk,
        weight,
        target_chunk,
        preference_labels_chunk,
        bias=None,
        preference_loss_fn=None,
        full_target=None,
        ignore_index=-100,
        use_ref_model=False,
        ref_input_chunk=None,
        ref_weight=None,
        ref_bias=None,
        average_log_prob=False,
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
            use_ref_model (bool): Whether to use a reference model for the alignment loss.
            ref_weight (torch.Tensor): Reference weight tensor. Shape: (vocab_size, hidden_size).
            ref_bias (torch.Tensor, optional): Reference bias tensor. Shape: (vocab_size,).
            average_log_prob (bool): Whether to average the log probability per non-masked token.
            loss_kwargs (dict): Additional arguments for the loss function.
        """
        (
            log_prob_chunk,
            chosen_logps_sum,
            rejected_logps_sum,
            chosen_logits_sum,
            rejected_logits_sum,
        ) = FusedLinearUnpairedPreferenceBase.chunk_forward(
            input_chunk,
            weight,
            target_chunk,
            preference_labels_chunk,
            bias=bias,
            ignore_index=ignore_index,
            average_log_prob=average_log_prob,
        )

        if use_ref_model:
            with torch.no_grad():
                (
                    ref_log_prob_chunk,
                    _,
                    _,
                    _,
                    _,
                ) = FusedLinearUnpairedPreferenceBase.chunk_forward(
                    ref_input_chunk,
                    ref_weight,
                    target_chunk,
                    preference_labels_chunk,
                    ref_bias,
                    ignore_index=ignore_index,
                    average_log_prob=average_log_prob,
                )
            loss_kwargs["ref_log_prob_chunk"] = ref_log_prob_chunk

        preference_loss_outputs = preference_loss_fn(
            log_prob_chunk, preference_labels_chunk, full_target, **loss_kwargs
        )
        if isinstance(preference_loss_outputs, tuple):
            preference_loss_chunk, *aux_outputs = preference_loss_outputs
        else:
            preference_loss_chunk, aux_outputs = preference_loss_outputs, []

        return_vars = (
            chosen_logps_sum,
            rejected_logps_sum,
            chosen_logits_sum,
            rejected_logits_sum,
        )

        return preference_loss_chunk, (*return_vars, *aux_outputs)
