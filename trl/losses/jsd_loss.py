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
# Adapted from Liger-Kernel v0.8.2 (000be609), src/liger_kernel/chunked_loss/jsd_loss.py.

import math

import torch
import torch.nn.functional as F

from .fused_linear_distillation import FusedLinearDistillationBase


class FusedLinearJSDFunction(FusedLinearDistillationBase):
    @staticmethod
    def distillation_loss_fn(student_logits, teacher_logits, beta=0.5, target=None, ignore_index=-100):
        """
        Compute JSD loss (Jensen-Shannon Divergence Loss).
        Args:
            student_logits (torch.Tensor): Logits of student tokens. Shape: (batch_size * seq_len,).
            teacher_logits (torch.Tensor): Logits of teacher tokens. Shape: (batch_size * seq_len,).
            beta (float): Coefficient beta of generalized JSD in the interval [0, 1]. Default: `0.5`.
            target (torch.Tensor): Target labels for masking. Shape: (chunk_size,).
            ignore_index (int): Index to ignore in loss computation.
        Returns:
            torch.Tensor: Jensen-Shannon Divergence loss
        Note:
            - Uses reduction="none" to preserve per-token losses for masking
            - KL divergence requires summing over vocab dimension (not mean)
            - Masking excludes padding/prompt tokens from loss computation
        """
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

        if beta == 0:
            jsd_loss = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True)
        elif beta == 1:
            jsd_loss = F.kl_div(teacher_log_probs, student_log_probs, reduction="none", log_target=True)
        else:
            # Compute probabilities (only required for mean calculation)
            log_mean_probs = torch.logsumexp(
                torch.stack([student_log_probs + math.log(1 - beta), teacher_log_probs + math.log(beta)], dim=0), dim=0
            )

            student_kl = F.kl_div(log_mean_probs, student_log_probs, reduction="none", log_target=True)
            teacher_kl = F.kl_div(log_mean_probs, teacher_log_probs, reduction="none", log_target=True)

            # JSD is the weighted average of the KL divergences
            jsd_loss = beta * teacher_kl + (1 - beta) * student_kl

        # Sum over vocab dimension (KL divergence definition)
        jsd_loss = jsd_loss.sum(dim=-1)  # (chunk_size,)

        # Apply ignore_index mask
        if target is not None:
            mask = target != ignore_index
            jsd_loss = jsd_loss.masked_fill(~mask, 0.0)

        return jsd_loss.sum()

    @classmethod
    def forward(
        cls,
        ctx,
        student_input: torch.Tensor,
        student_weight: torch.Tensor,
        teacher_input: torch.Tensor,
        teacher_weight: torch.Tensor,
        true_labels: torch.LongTensor,
        student_bias: torch.Tensor,
        teacher_bias: torch.Tensor,
        weight_hard_loss: float = 0.5,
        weight_soft_loss: float = 0.5,
        beta: float = 0.5,
        ignore_index: int = -100,
        temperature: float = 1.0,
        compiled: bool = True,
        chunk_size: int = 1024,
        return_soft_hard_loss: bool = False,
    ):
        """
        Fused linear layer with JSD distillation loss.
        Args:
            student_input (torch.Tensor): Student input tensor. Shape: (batch_size * seq_len, hidden_size_student)
            student_weight (torch.Tensor): Student weight tensor. Shape: (vocab_size, hidden_size_student)
            teacher_input (torch.Tensor): Teacher input tensor. Shape: (batch_size * seq_len, hidden_size_teacher)
            teacher_weight (torch.Tensor): Teacher weight tensor. Shape: (vocab_size, hidden_size_teacher)
            true_labels (torch.LongTensor): Target tensor. Shape: (batch_size * seq_len,)
            weight_hard_loss (float): Weight for hard loss.
            weight_soft_loss (float): Weight for soft loss.
            beta (float): Coefficient beta of generalized JSD in the interval [0, 1]. Default: `0.5`.
            ignore_index (int): Index to ignore in loss computation
            temperature (float): Temperature for softening/sharpening distributions
            compiled (bool): Whether to use torch compile
            chunk_size (int): Size of chunks for processing.
            return_soft_hard_loss (bool): Whether to return soft and hard losses separately. Default: False.
        Returns:
            torch.Tensor: Computed loss, or tuple (loss, soft_loss, hard_loss) if return_soft_hard_loss=True
        """
        return super().forward(
            cls=cls,
            ctx=ctx,
            student_input=student_input,
            student_weight=student_weight,
            teacher_input=teacher_input,
            teacher_weight=teacher_weight,
            target=true_labels,
            student_bias=student_bias,
            teacher_bias=teacher_bias,
            chunk_size=chunk_size,
            weight_hard_loss=weight_hard_loss,
            weight_soft_loss=weight_soft_loss,
            beta=beta,
            ignore_index=ignore_index,
            temperature=temperature,
            compiled=compiled,
            return_soft_hard_loss=return_soft_hard_loss,
        )

    @staticmethod
    def backward(ctx, grad_output, *args):
        grads = FusedLinearDistillationBase.backward(ctx, grad_output, *args)[:6]

        return (
            *grads,
            None,  # teacher_bias
            None,  # weight_hard_loss
            None,  # weight_soft_loss
            None,  # beta
            None,  # ignore_index
            None,  # temperature
            None,  # compiled
            None,  # chunk_size
            None,  # return_soft_hard_loss
        )


class FusedLinearJSDLoss(torch.nn.Module):
    """
    Fused linear layer with JSD distillation loss.
    """

    def __init__(
        self,
        weight_hard_loss: float = 0.5,
        weight_soft_loss: float = 0.5,
        beta: float = 0.5,
        ignore_index: int = -100,
        temperature: float = 1.0,
        compiled: bool = True,
        chunk_size: int = 1024,
        return_soft_hard_loss: bool = False,
    ):
        """
        Args:
            weight_hard_loss (float): Weight for hard loss.
            weight_soft_loss (float): Weight for soft loss.
            ignore_index (int): Index to ignore in the loss
            temperature (float): Temperature for softening distributions
            compiled (bool): Whether to use torch compile
            beta (float): Coefficient beta of generalized JSD in the interval [0, 1]. Default: `0.5`.
            chunk_size (int): Size of chunks for processing.
            return_soft_hard_loss (bool): Whether to return soft and hard losses separately. Default: False.
        """
        super().__init__()
        assert temperature != 0, "Temperature cannot be 0."
        self.weight_hard_loss = weight_hard_loss
        self.weight_soft_loss = weight_soft_loss
        self.ignore_index = ignore_index
        self.temperature = temperature
        self.compiled = compiled
        self.beta = beta
        self.chunk_size = chunk_size
        self.return_soft_hard_loss = return_soft_hard_loss

    def forward(
        self,
        student_input: torch.Tensor,
        student_weight: torch.Tensor,
        teacher_input: torch.Tensor,
        teacher_weight: torch.Tensor,
        true_labels: torch.LongTensor,
        student_bias: torch.Tensor = None,
        teacher_bias: torch.Tensor = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the JSD distillation loss.

        Args:
            student_input (torch.Tensor): Student input tensor
            student_weight (torch.Tensor): Student weight tensor
            teacher_input (torch.Tensor): Teacher input tensor
            teacher_weight (torch.Tensor): Teacher weight tensor
            true_labels (torch.LongTensor): Target labels tensor

        Returns:
            torch.Tensor or Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                If return_soft_hard_loss is False: Computed combined loss
                If return_soft_hard_loss is True: Tuple of (combined_loss, soft_loss, hard_loss)
        """
        return FusedLinearJSDFunction.apply(
            student_input,
            student_weight,
            teacher_input,
            teacher_weight,
            true_labels,
            student_bias,
            teacher_bias,
            self.weight_hard_loss,
            self.weight_soft_loss,
            self.beta,
            self.ignore_index,
            self.temperature,
            self.compiled,
            self.chunk_size,
            self.return_soft_hard_loss,
        )
