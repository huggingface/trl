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
# Adapted from Liger-Kernel v0.8.2 (000be609), src/liger_kernel/chunked_loss/grpo_loss.py.

import math

import torch

from .fused_linear_ppo import FusedLinearPPOBase


def k3_loss_fn(log_p, log_q):
    # computes k3 estimate of KL[q, p]
    # ref: http://joschu.net/blog/kl-approx.html
    return torch.exp(log_p - log_q) - (log_p - log_q) - 1.0


@torch.no_grad()
def get_gamma_weights(
    advantages: torch.Tensor,
    log_ratio_per_token: torch.Tensor,
    mask: torch.Tensor,
    importance_sampling_ratio: torch.Tensor | None = None,
    k_pos: float = 2.0,
    lambda_pos: float = 3.0,
    k_neg: float = 3.0,
    lambda_neg: float = 2.0,
) -> torch.Tensor:
    """VESPO gamma weighting: phi(w) = e^lambda * w^k * e^{-lambda*w} (normalized so phi(1)=1).

    Computed in log space and detached (via ``@torch.no_grad``) so ``phi_seq`` acts purely
    as a gradient-scaling coefficient. Returns a (B, 1) tensor.
    TRL reference: ``trl.trainer.grpo_trainer.GRPOTrainer.get_gamma_weights``.
    """
    lower_clamp = math.log(1e-8)

    log_ratio_clamped = torch.clamp(log_ratio_per_token, -20.0, 20.0)
    seq_log_ratio = torch.sum(log_ratio_clamped * mask, dim=-1, keepdim=True)  # (B, 1)

    if importance_sampling_ratio is not None:
        log_is_ratio = torch.clamp(torch.log(importance_sampling_ratio), lower_clamp, 20.0)
        seq_log_ratio = seq_log_ratio + torch.sum(log_is_ratio, dim=-1, keepdim=True)

    log_w_seq = torch.clamp(seq_log_ratio, lower_clamp, 20.0)
    w_seq = torch.exp(log_w_seq)

    is_nonneg_adv = advantages.unsqueeze(-1) >= 0
    k_seq = torch.where(is_nonneg_adv, k_pos, k_neg)
    lambda_seq = torch.where(is_nonneg_adv, lambda_pos, lambda_neg).clamp(min=1e-4)

    log_phi = lambda_seq + k_seq * log_w_seq - lambda_seq * w_seq
    phi_seq = torch.exp(log_phi).nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
    return phi_seq


def sapo_loss_fn(importance_ratio: torch.Tensor, temperature: float) -> torch.Tensor:
    """SAPO (Soft Adaptive Policy Optimization) loss function.

    Replaces hard clipping with a smooth, temperature-controlled gate that
    adaptively attenuates off-policy updates while preserving useful learning signals.

    Reference: https://huggingface.co/papers/2511.20347
    TRL implementation: https://github.com/huggingface/trl/blob/1bd2a52ec2d8344050af736d60cdc735181ae4b8/trl/trainer/grpo_trainer.py#L1913

    Args:
        importance_ratio: The importance sampling ratio (pi_theta / pi_old).
        temperature: Temperature parameter controlling the softness of the gate.

    Returns:
        The SAPO loss value.
    """
    if temperature <= 0:
        raise ValueError("sapo_temperature must be > 0.")
    sigmoid_input = temperature * (importance_ratio - 1)
    sigmoid_smoothed_loss = torch.sigmoid(sigmoid_input)
    return sigmoid_smoothed_loss * 4 / temperature


def clip_coef_fn(coef, epsilon_low, epsilon_high, loss_type):
    if loss_type == "cispo":
        # CISPO: clip and detach the importance weights
        upper_bound = epsilon_high
        lower_bound = None
        clipped_coef = torch.clamp(coef, lower_bound, upper_bound).detach()
        is_lower_clipped = False
        is_upper_clipped = coef > upper_bound
    elif loss_type in ("sapo", "vespo"):
        # SAPO / VESPO don't use clipping metrics
        clipped_coef = None
        is_lower_clipped = torch.zeros_like(coef, dtype=torch.bool)
        is_upper_clipped = torch.zeros_like(coef, dtype=torch.bool)
    else:
        upper_bound = 1 + epsilon_high
        lower_bound = 1 - epsilon_low
        clipped_coef = torch.clamp(coef, lower_bound, upper_bound)
        is_lower_clipped = coef < lower_bound
        is_upper_clipped = coef > upper_bound
    return clipped_coef, is_lower_clipped, is_upper_clipped


class FusedLinearGRPOFunction(FusedLinearPPOBase):
    @staticmethod
    def ppo_loss_fn(
        per_token_logps,
        attention_mask,
        advantages,
        full_attention_mask,
        ref_per_token_logps=None,  # shape: [chunk_size, seq_len]
        old_per_token_logps=None,
        epsilon_low=0.2,
        epsilon_high=0.2,
        beta=0.04,
        loss_type="dapo",  # ["grpo", "bnpo", "dr_grpo", "dapo", "cispo", "sapo", "luspo", "vespo"]
        max_completion_length=None,  # Required for dr_grpo
        importance_sampling_level="token",  # ["token", "sequence"] - new parameter for GSPO
        sapo_temperature_pos=1.0,  # Temperature for positive advantages in SAPO
        sapo_temperature_neg=1.05,  # Temperature for negative advantages in SAPO
        vllm_is_ratio=None,  # vLLM importance sampling ratio (chunk_size, seq_len) or (chunk_size, 1) or None
        delta=None,  # Upper clamp for two-sided clipping (INTELLECT-2)
        use_bias_correction_kl=False,  # Importance-sampling-corrected KL (DeepSeek-V3.2)
        vespo_k_pos=2.0,  # VESPO gamma shape k for non-negative advantages
        vespo_lambda_pos=3.0,  # VESPO gamma rate lambda for non-negative advantages
        vespo_k_neg=3.0,  # VESPO gamma shape k for negative advantages
        vespo_lambda_neg=2.0,  # VESPO gamma rate lambda for negative advantages
        num_items_in_batch=None,  # Total active tokens across the entire generation batch (TRL-compat)
        **kwargs,
    ):
        """GRPO Loss Function matching GRPOTrainer implementation."""
        # Validate sequence-level + loss_type combinations
        if importance_sampling_level == "sequence" and loss_type in ("cispo", "sapo", "vespo"):
            raise ValueError(
                f"Sequence-level importance sampling is not supported for loss_type='{loss_type}'. "
                f"Use importance_sampling_level='token' instead."
            )

        # Get reference model probabilities
        if ref_per_token_logps is None:
            ref_per_token_logps = per_token_logps.detach()

        # Compute policy gradient loss with importance sampling ratio
        old_per_token_logps = old_per_token_logps if old_per_token_logps is not None else per_token_logps.detach()
        log_ratio = per_token_logps - old_per_token_logps

        if importance_sampling_level == "token":
            log_importance_weights = log_ratio
        elif importance_sampling_level == "sequence":
            log_importance_weights = (log_ratio * attention_mask).sum(-1) / attention_mask.sum(-1).clamp(min=1.0)
            log_importance_weights = log_importance_weights.unsqueeze(-1)
        else:
            raise ValueError(
                f"Unknown importance sampling level: {importance_sampling_level}. Possible values are 'token' "
                "and 'sequence'."
            )

        # From here, log_importance_weights (and all subsequent tensors, coef_1, coef_2, etc.) shape depends on
        # importance_sampling_level: "token" level: (B, T); "sequence" level: (B, 1)
        coef_1 = torch.exp(log_importance_weights)
        coef_2, is_lower_clipped, is_upper_clipped = clip_coef_fn(coef_1, epsilon_low, epsilon_high, loss_type)
        if loss_type == "cispo":
            # CISPO: clip and detach the importance weights, multiply by log probs
            # Reference: https://github.com/huggingface/trl/blob/035c3ff151b953ca72cdfe0ee966bc1469a26fde/trl/trainer/grpo_trainer.py#L2030
            per_token_loss = -coef_2 * advantages.unsqueeze(1) * per_token_logps
        elif loss_type == "sapo":
            # SAPO: Soft Adaptive Policy Optimization
            # Uses sigmoid-based soft gating instead of hard clipping
            # Reference: https://huggingface.co/papers/2511.20347
            # TRL implementation: https://github.com/huggingface/trl/blob/1bd2a52ec2d8344050af736d60cdc735181ae4b8/trl/trainer/grpo_trainer.py#L2037-L2046
            advantages_expanded = advantages.unsqueeze(1).expand_as(coef_1)
            positive_advantages_mask = advantages_expanded > 0

            # torch.where over both branches rather than boolean-mask assignment:
            # mask indexing lowers to aten.nonzero, which graph-breaks under
            # torch.compile. sapo_loss_fn is elementwise so this is identical.
            per_token_loss = torch.where(
                positive_advantages_mask,
                sapo_loss_fn(coef_1, sapo_temperature_pos),
                sapo_loss_fn(coef_1, sapo_temperature_neg),
            )
            per_token_loss = -per_token_loss * advantages_expanded
        elif loss_type == "vespo":
            # VESPO: Value-Enhanced Sequence-level Policy Optimization.
            # Uses a detached gamma weighting phi(w) as a gradient scaling coefficient.
            # Reference: TRL grpo_trainer.get_gamma_weights. The vllm correction for
            # distribution mismatch is folded into phi_seq via ``importance_sampling_ratio``
            # rather than multiplying per_token_loss.
            phi_seq = get_gamma_weights(
                advantages=advantages,
                log_ratio_per_token=log_ratio,
                mask=attention_mask,
                importance_sampling_ratio=vllm_is_ratio,
                k_pos=vespo_k_pos,
                lambda_pos=vespo_lambda_pos,
                k_neg=vespo_k_neg,
                lambda_neg=vespo_lambda_neg,
            )
            per_token_loss = -phi_seq * advantages.unsqueeze(1) * per_token_logps
        else:
            # Apply delta (two-sided clipping from INTELLECT-2) to coef_1
            if delta is not None:
                coef_1 = torch.clamp(coef_1, max=delta)
            per_token_loss1 = coef_1 * advantages.unsqueeze(1)
            per_token_loss2 = coef_2 * advantages.unsqueeze(1)
            per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        # Apply vLLM importance sampling correction BEFORE adding KL penalty
        # VESPO folds this correction into phi_seq (in log space), so we skip it here.
        if vllm_is_ratio is not None and loss_type != "vespo":
            per_token_loss = per_token_loss * vllm_is_ratio

        if beta != 0.0:
            # Compute KL penalty (approximates KL[per_token_logps, ref_per_token_logps])
            kl_div = k3_loss_fn(ref_per_token_logps, per_token_logps)
            if use_bias_correction_kl:
                # Importance-sampling-corrected KL (DeepSeek-V3.2): kl *= coef_1.
                # Use exp(log_importance_weights) so the ratio's shape matches
                # importance_sampling_level (token: (B, T); sequence: (B, 1)),
                # mirroring TRL's ``per_token_kl * coef_1`` (un-clamped, before delta).
                kl_div = kl_div * torch.exp(log_importance_weights)
            # Combine losses
            per_token_loss = per_token_loss + beta * kl_div

        # Note: We normalize by the number of tokens in the batch (using full_attention_mask),
        # which is consistent with the DAPO loss implementation (https://arxiv.org/html/2503.14476v1)
        # and TRL GRPO implementation
        # (https://github.com/huggingface/trl/blob/e751a16df56e70190fb94bed4a2035eec3303777/trl/trainer/grpo_trainer.py#L966)
        if loss_type == "grpo" or loss_type == "sapo":
            # Average per-sequence loss (SAPO uses same normalization as GRPO)
            loss = (
                (per_token_loss * attention_mask).sum(-1) / torch.clamp(attention_mask.sum(-1), min=1.0)
            ).sum() / full_attention_mask.shape[0]
        elif loss_type == "bnpo":
            # Batch Normalized Per-token loss (original implementation)
            loss = (per_token_loss * attention_mask).sum() / torch.clamp(full_attention_mask.sum(), min=1.0)
        elif loss_type == "dr_grpo":
            # Dimension-Reduced GRPO (normalize by batch_size * max_completion_length)
            if max_completion_length is None:
                raise ValueError("max_completion_length must be provided for loss_type 'dr_grpo'")
            loss = (per_token_loss * attention_mask).sum() / (full_attention_mask.shape[0] * max_completion_length)
        elif loss_type in ("dapo", "cispo", "vespo"):
            loss_normalizer = FusedLinearPPOBase._compute_dapo_normalizer(
                full_attention_mask, num_items_in_batch=num_items_in_batch
            )
            loss = (per_token_loss * attention_mask).sum() / loss_normalizer
        elif loss_type == "luspo":
            # Match TRL exactly: loss = (per_token_loss * mask.sum(1, keepdim=True)).mean()
            weighted = per_token_loss * attention_mask.sum(1, keepdim=True)
            loss = weighted.sum() / (full_attention_mask.shape[0] * weighted.shape[1])
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        # Calculate metrics
        metrics = []
        if beta != 0.0:
            metrics.append((kl_div * attention_mask).sum() / torch.clamp(full_attention_mask.sum(), min=1.0))

        # Adjust clipping metric calculation based on importance sampling level
        if importance_sampling_level == "token":
            is_clipped = (is_lower_clipped & (advantages.unsqueeze(1) < 0)) | (
                is_upper_clipped & (advantages.unsqueeze(1) > 0)
            )
        else:  # sequence level
            # For sequence level, coef_1 is shape (B, 1), advantages is shape (B,)
            is_clipped = (is_lower_clipped & (advantages.unsqueeze(1) < 0)) | (
                is_upper_clipped & (advantages.unsqueeze(1) > 0)
            )
            is_clipped = is_clipped.expand_as(attention_mask)

        metrics.append((is_clipped * attention_mask).sum() / torch.clamp(full_attention_mask.sum(), min=1.0))
        return loss, metrics

    @classmethod
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
        beta=0.04,
        epsilon_low=0.2,
        epsilon_high=0.2,
        loss_type="dapo",
        max_completion_length=None,
        importance_sampling_level="token",
        sapo_temperature_pos=1.0,
        sapo_temperature_neg=1.05,
        temperature=1.0,
        compiled=True,
        use_ref_model=True,
        chunk_size=1,
        vllm_is_ratio=None,
        delta=None,
        use_bias_correction_kl=False,
        vespo_k_pos=2.0,
        vespo_lambda_pos=3.0,
        vespo_k_neg=3.0,
        vespo_lambda_neg=2.0,
        num_items_in_batch=None,
    ):
        """
        Fused linear layer with GRPO loss.
        Args:
            _input (torch.Tensor): Input tensor. Shape: (batch_size * seq_len, hidden_size)
            weight (torch.Tensor): Weight tensor. Shape: (vocab_size, hidden_size)
            selected_token_ids (torch.Tensor): Selected token ids tensor. Shape: (batch_size, seq_len)
            attention_mask (torch.Tensor): Attention mask tensor. Shape: (batch_size, seq_len)
            advantages (torch.Tensor): Advantages tensor. Shape: (batch_size,)
            bias (torch.Tensor, optional): Bias tensor. Shape: (vocab_size,)
            ref_per_token_logps:  Reference model log probs per token tensor. Shape:(batch_size, seq_len)
            ref_input (torch.Tensor, optional): Reference model input tensor. Shape: (batch_size * seq_len, hidden_size)
            ref_weight (torch.Tensor, optional): Reference model weight tensor. Shape: (vocab_size, hidden_size)
            ref_bias (torch.Tensor, optional): Reference model bias tensor. Shape: (vocab_size,)
            beta (float): Weight for the KL penalty
            loss_type (str): Type of loss calculation ("grpo", "bnpo", "dr_grpo", "dapo", "cispo", "sapo", "luspo").
                Defaults to "dapo".
            max_completion_length (int, optional): Maximum completion length, required for "dr_grpo". Defaults to None.
            importance_sampling_level (str): Level of importance sampling ("token" or "sequence"). Defaults to "token".
            sapo_temperature_pos (float): Temperature for positive advantages in SAPO. Defaults to 1.0.
            sapo_temperature_neg (float): Temperature for negative advantages in SAPO. Defaults to 1.05.
            temperature (float): Temperature for the logits
            compiled (bool): Whether to use torch compile
            use_ref_model (bool): Whether to use a reference model
            chunk_size (int): Size of chunks for processing.
            vllm_is_ratio (torch.Tensor, optional): vLLM importance sampling ratio (batch_size, seq_len) or (batch_size, 1) or None.
                Used to correct for distribution mismatch when using vLLM for generation.
        Returns:
            torch.Tensor: Computed loss
        """
        # Validate before entering torch.compile boundary
        if importance_sampling_level == "sequence" and loss_type in ("cispo", "sapo", "vespo"):
            raise ValueError(
                f"Sequence-level importance sampling is not supported for loss_type='{loss_type}'. "
                f"Use importance_sampling_level='token' instead."
            )

        return super().forward(
            cls=cls,
            ctx=ctx,
            _input=_input,
            weight=weight,
            selected_token_ids=selected_token_ids,
            attention_mask=attention_mask,
            advantages=advantages,
            bias=bias,
            ref_per_token_logps=ref_per_token_logps,
            old_per_token_logps=old_per_token_logps,
            ref_input=ref_input,
            ref_weight=ref_weight,
            ref_bias=ref_bias,
            beta=beta,
            epsilon_low=epsilon_low,
            epsilon_high=epsilon_high,
            loss_type=loss_type,
            max_completion_length=max_completion_length,
            temperature=temperature,
            compiled=compiled,
            use_ref_model=use_ref_model,
            chunk_size=chunk_size,
            importance_sampling_level=importance_sampling_level,
            sapo_temperature_pos=sapo_temperature_pos,
            sapo_temperature_neg=sapo_temperature_neg,
            vllm_is_ratio=vllm_is_ratio,
            delta=delta,
            use_bias_correction_kl=use_bias_correction_kl,
            vespo_k_pos=vespo_k_pos,
            vespo_lambda_pos=vespo_lambda_pos,
            vespo_k_neg=vespo_k_neg,
            vespo_lambda_neg=vespo_lambda_neg,
            num_items_in_batch=num_items_in_batch,
        )

    @staticmethod
    def backward(ctx, grad_output, *grad_metrics):
        """Backward pass for GRPO loss.

        Args:
            grad_output: Gradient of the loss (scalar)
            grad_metrics: Gradients of the metrics (not used in backward computation)
        """
        grads = FusedLinearPPOBase.backward(ctx, grad_output)
        return (
            *grads[
                :6
            ],  # grad_input, grad_weight, grad_selected_token_ids, grad_attention_mask, grad_advantages, grad_bias
            None,  # grad_ref_per_token_logps
            None,  # grad_old_per_token_logps
            None,  # grad_ref_input
            None,  # grad_ref_weight
            None,  # grad_ref_bias
            None,  # grad_beta
            None,  # grad_epsilon_low
            None,  # grad_epsilon_high
            None,  # grad_loss_type (string, not differentiable)
            None,  # grad_max_completion_length (int, not differentiable)
            None,  # grad_importance_sampling_level (string, not differentiable)
            None,  # grad_sapo_temperature_pos (float, not differentiable)
            None,  # grad_sapo_temperature_neg (float, not differentiable)
            None,  # grad_temperature
            None,  # grad_compiled
            None,  # grad_use_ref_model
            None,  # grad_chunk_size
            None,  # grad_vllm_is_ratio
            None,  # grad_delta
            None,  # grad_use_bias_correction_kl
            None,  # grad_vespo_k_pos
            None,  # grad_vespo_lambda_pos
            None,  # grad_vespo_k_neg
            None,  # grad_vespo_lambda_neg
            None,  # grad_num_items_in_batch
        )


class FusedLinearGRPOLoss(torch.nn.Module):
    """Fused linear layer with GRPO loss."""

    def __init__(
        self,
        beta: float = 0.04,
        compiled: bool = True,
        use_ref_model: bool = True,
        chunk_size: int = 1,
        epsilon_low: float = 0.2,
        epsilon_high: float = 0.2,
        loss_type: str = "dapo",
        max_completion_length: int | None = None,
        importance_sampling_level: str = "token",
        sapo_temperature_pos: float = 1.0,
        sapo_temperature_neg: float = 1.05,
        temperature: float = 1.0,
        delta: float | None = None,
        use_bias_correction_kl: bool = False,
        vespo_k_pos: float = 2.0,
        vespo_lambda_pos: float = 3.0,
        vespo_k_neg: float = 3.0,
        vespo_lambda_neg: float = 2.0,
    ):
        """
        Args:
            beta (float): Weight for the KL penalty.
            compiled (bool): Whether to use torch compile.
            use_ref_model (bool): Whether to use a reference model.
            chunk_size (int): Size of chunks for processing.
            epsilon_low (float): Lower bound for the importance sampling ratio.
            epsilon_high (float): Upper bound for the importance sampling ratio.
            loss_type (str): Type of loss calculation ("grpo", "bnpo", "dr_grpo", "dapo", "cispo", "sapo", "luspo").
                Defaults to "dapo". For "cispo", epsilon_high is typically larger (e.g. 5.0) and
                epsilon_low is unused. For "sapo", uses soft gating instead of hard clipping.
            max_completion_length (int, optional): Maximum completion length, required for "dr_grpo". Defaults to None.
            importance_sampling_level (str): Level of importance sampling ("token" or "sequence"). Defaults to "token".
            sapo_temperature_pos (float): Temperature for positive advantages in SAPO. Defaults to 1.0.
            sapo_temperature_neg (float): Temperature for negative advantages in SAPO. Defaults to 1.05.
            temperature (float): Temperature for the logits.
            delta (float, optional): Upper clamp for two-sided clipping (INTELLECT-2). None means disabled.
            use_bias_correction_kl (bool): If True, multiply KL by importance sampling ratio (DeepSeek-V3.2).
        """
        super().__init__()
        # Validate SAPO temperatures to prevent division by zero or numerical instability
        if sapo_temperature_pos <= 0:
            raise ValueError(f"sapo_temperature_pos must be positive, got {sapo_temperature_pos}")
        if sapo_temperature_neg <= 0:
            raise ValueError(f"sapo_temperature_neg must be positive, got {sapo_temperature_neg}")
        if delta is not None and delta <= 0:
            raise ValueError(f"delta must be positive, got {delta}")
        self.beta = beta
        self.compiled = compiled
        self.use_ref_model = use_ref_model
        self.chunk_size = chunk_size
        self.epsilon_low = epsilon_low
        self.epsilon_high = epsilon_high
        self.loss_type = loss_type
        self.max_completion_length = max_completion_length
        self.importance_sampling_level = importance_sampling_level
        self.sapo_temperature_pos = sapo_temperature_pos
        self.sapo_temperature_neg = sapo_temperature_neg
        self.temperature = temperature
        self.delta = delta
        self.use_bias_correction_kl = use_bias_correction_kl
        self.vespo_k_pos = vespo_k_pos
        self.vespo_lambda_pos = vespo_lambda_pos
        self.vespo_k_neg = vespo_k_neg
        self.vespo_lambda_neg = vespo_lambda_neg

    def forward(
        self,
        _input,
        lin_weight,
        selected_token_ids,
        attention_mask,
        advantages,
        bias=None,
        ref_per_token_logps=None,
        old_per_token_logps=None,
        ref_input=None,
        ref_weight=None,
        ref_bias=None,
        vllm_is_ratio=None,
        num_items_in_batch=None,
    ):
        return FusedLinearGRPOFunction.apply(
            _input,
            lin_weight,
            selected_token_ids,
            attention_mask,
            advantages,
            bias,
            ref_per_token_logps,
            old_per_token_logps,
            ref_input,
            ref_weight,
            ref_bias,
            self.beta,
            self.epsilon_low,
            self.epsilon_high,
            self.loss_type,
            self.max_completion_length,
            self.importance_sampling_level,
            self.sapo_temperature_pos,
            self.sapo_temperature_neg,
            self.temperature,
            self.compiled,
            self.use_ref_model,
            self.chunk_size,
            vllm_is_ratio,
            self.delta,
            self.use_bias_correction_kl,
            self.vespo_k_pos,
            self.vespo_lambda_pos,
            self.vespo_k_neg,
            self.vespo_lambda_neg,
            num_items_in_batch,
        )
