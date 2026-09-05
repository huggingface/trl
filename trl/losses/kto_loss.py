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
# Adapted from Liger-Kernel v0.8.2 (000be609), src/liger_kernel/chunked_loss/kto_loss.py.

import torch
import torch.nn.functional as F

from .fused_linear_unpaired_preference import FusedLinearUnpairedPreferenceBase


class FusedLinearKTOFunction(FusedLinearUnpairedPreferenceBase):
    @staticmethod
    def preference_loss_fn(
        log_prob_chunk,
        preference_labels_chunk,
        full_target,
        ref_log_prob_chunk=None,
        beta=0.1,
        kl=None,
    ):
        """
        Implements the Kahneman-Tversky Optimization (KTO) loss function. Paper: "KTO: Model Alignment as Prospect
        Theory-Guided Optimization" https://arxiv.org/abs/2402.01306

        KTO loss is inspired by prospect theory (https://en.wikipedia.org/wiki/Prospect_theory) from behavioral
        economics, which models how humans make decisions under uncertainty. The loss function is asymmetric, treating
        gains and losses differently, similar to human decision-making patterns.

        Formula: When y is chosen: L_KTO = 1 - σ(β * (log[π(x)/π₀(x)] - KL(π||π₀)_y)) When y is rejected: L_KTO = 1 -
        σ(β * (KL(π||π₀)_y - log[π(x)/π₀(x)]))

        Where:
        - σ: Sigmoid function
        - β: Temperature parameter controlling the strength of the preference signal
        - π(x): Policy (current model)
        - π₀(x): Reference policy (reference model)
        - KL(π||π₀)_y: KL divergence estimated using the rejected response y

        The loss encourages the model to:
        1. Assign higher probability to chosen responses
        2. Assign lower probability to rejected responses
        3. Maintain reasonable distance from the reference model

        Args:
            log_prob_chunk: Log probabilities for the chunk (batch_size,)
            preference_labels_chunk: Preference labels for the chunk (batch_size,)
            full_target: Non chunked full target tensor
            ref_log_prob_chunk: Reference log probs for the chunk (batch_size,)
            beta: Weight for the KTO loss
            kl:
                KL divergence between the policy model and the reference model for the chosen responses. Shape:
                (batch_size,)
        Returns:
            - loss: The KTO loss value
        """
        if ref_log_prob_chunk is not None:
            logratios_chunk = log_prob_chunk - ref_log_prob_chunk
        else:
            logratios_chunk = log_prob_chunk
        multiplier_chunk = torch.where(preference_labels_chunk, 1, -1)
        if kl is not None:
            losses = 1 - F.sigmoid(beta * (logratios_chunk - kl) * multiplier_chunk)
        else:
            losses = 1 - F.sigmoid(beta * logratios_chunk * multiplier_chunk)

        rewards = beta * logratios_chunk
        chosen_rewards_sum = (rewards * preference_labels_chunk).sum()
        rejected_rewards_sum = (rewards * ~preference_labels_chunk).sum()

        return losses.sum() / (full_target.shape[0]), chosen_rewards_sum, rejected_rewards_sum

    @classmethod
    def forward(
        cls,
        ctx,
        _input,
        weight,
        target,
        preference_labels,
        bias=None,
        ref_input=None,
        ref_weight=None,
        ref_bias=None,
        kl=None,
        ignore_index=-100,
        beta=0.1,
        compiled=True,
        use_ref_model=True,
        average_log_prob=False,
        chunk_size=1,
    ):
        """
        Fused linear layer with KTO loss.

        Args:
            _input (torch.Tensor): Input tensor. Shape: (batch_size * seq_len, hidden_size)
            weight (torch.Tensor): Weight tensor. Shape: (vocab_size, hidden_size)
            target (torch.LongTensor): Target tensor. Shape: (batch_size * seq_len,)
            preference_labels (torch.Tensor): Preference labels tensor. Shape: (batch_size,)
            bias (torch.Tensor, optional): Bias tensor. Shape: (vocab_size,)
            ref_input (torch.Tensor, optional):
                Reference model input tensor. Shape: (batch_size * seq_len, hidden_size)
            ref_weight (torch.Tensor, optional): Reference model weight tensor. Shape: (vocab_size, hidden_size)
            ref_bias (torch.Tensor, optional): Reference model bias tensor. Shape: (vocab_size,)
            kl (torch.Tensor, optional): KL divergence tensor. Shape: (batch_size,)
            ignore_index (int): Index to ignore in loss computation
            beta (float): Temperature parameter for the KTO loss
            compiled (bool): Whether to use torch compile
            use_ref_model (bool): Whether to use a reference model
            average_log_prob (bool): Whether to average the log probability per non-masked token
            chunk_size (int): Size of chunks for processing
        Returns:
            torch.Tensor: Computed loss
        """
        return super().forward(
            cls=cls,
            ctx=ctx,
            _input=_input,
            weight=weight,
            target=target,
            preference_labels=preference_labels,
            bias=bias,
            ignore_index=ignore_index,
            beta=beta,
            compiled=compiled,
            use_ref_model=use_ref_model,
            ref_input=ref_input,
            ref_weight=ref_weight,
            ref_bias=ref_bias,
            average_log_prob=average_log_prob,
            kl=kl,
            chunk_size=chunk_size,
        )

    @staticmethod
    def backward(ctx, *grad_output):
        grads = FusedLinearUnpairedPreferenceBase.backward(ctx, grad_output)[:5]
        return (
            *grads,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class FusedLinearKTOLoss(torch.nn.Module):
    """
    Fused linear layer with Kahneman-Tversky Optimization (KTO) loss.
    """

    def __init__(
        self,
        ignore_index: int = -100,
        beta: float = 0.1,
        compiled: bool = True,
        use_ref_model: bool = False,
        average_log_prob: bool = False,
        chunk_size: int = 1,
    ):
        """
        Args:
            ignore_index (int): Index to ignore in the loss calculation
            beta (float): Temperature parameter for the KTO loss
            compiled (bool): Whether to use compiled operations
            use_ref_model (bool): Whether to use a reference model for the DPO loss.
            average_log_prob (bool): Whether to average the log probability per non-masked token
            chunk_size (int): Size of chunks for processing
        """
        super().__init__()
        self.ignore_index = ignore_index
        self.beta = beta
        self.compiled = compiled
        self.use_ref_model = use_ref_model
        self.average_log_prob = average_log_prob
        self.chunk_size = chunk_size

    def forward(
        self,
        _input,
        lin_weight,
        target,
        bias=None,
        preference_labels=None,
        ref_input=None,
        ref_weight=None,
        ref_bias=None,
        kl=None,
    ):
        return FusedLinearKTOFunction.apply(
            _input,
            lin_weight,
            target,
            preference_labels,
            bias,
            ref_input,
            ref_weight,
            ref_bias,
            kl,
            self.ignore_index,
            self.beta,
            self.compiled,
            self.use_ref_model,
            self.average_log_prob,
            self.chunk_size,
        )
