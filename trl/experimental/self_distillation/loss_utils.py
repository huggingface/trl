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

"""Pure helper functions for self-distillation loss computation."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def select_token_log_probs(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
) -> torch.Tensor:
    logsumexp = torch.logsumexp(logits, dim=-1, keepdim=True)
    indices = token_ids.unsqueeze(-1)
    return (torch.gather(logits, dim=-1, index=indices) - logsumexp).squeeze(-1)


def compute_divergence(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    if alpha == 0.0:
        kl = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True)
    elif alpha == 1.0:
        kl = F.kl_div(teacher_log_probs, student_log_probs, reduction="none", log_target=True)
    else:
        alpha_t = torch.tensor(alpha, dtype=student_log_probs.dtype, device=student_log_probs.device)
        mixture = torch.logsumexp(
            torch.stack([student_log_probs + torch.log(1 - alpha_t), teacher_log_probs + torch.log(alpha_t)]),
            dim=0,
        )
        kl_teacher = F.kl_div(mixture, teacher_log_probs, reduction="none", log_target=True)
        kl_student = F.kl_div(mixture, student_log_probs, reduction="none", log_target=True)
        kl = torch.lerp(kl_student, kl_teacher, alpha)
    return kl.sum(-1)


def add_tail(log_probs: torch.Tensor) -> torch.Tensor:
    log_s = torch.logsumexp(log_probs, dim=-1, keepdim=True)
    log_s = torch.clamp(log_s, max=-1e-7)
    tail_log = torch.log(-torch.expm1(log_s))
    return torch.cat([log_probs, tail_log], dim=-1)


def renorm_topk_log_probs(log_probs: torch.Tensor) -> torch.Tensor:
    return log_probs - torch.logsumexp(log_probs, dim=-1, keepdim=True)


def compute_token_level_distillation_loss(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
) -> torch.Tensor:
    log_ratio = student_log_probs - teacher_log_probs
    return log_ratio.detach() * student_log_probs


def apply_importance_sampling_clipping(
    per_token_loss: torch.Tensor,
    student_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    clip_coeff: float,
) -> torch.Tensor:
    negative_approx_kl = (student_log_probs - old_log_probs).detach()
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl).clamp(max=clip_coeff)
    return per_token_loss * ratio


def aggregate_loss(
    per_token_loss: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    loss_type: str,
    max_completion_length: int,
) -> torch.Tensor:
    """Reduce a per-token loss tensor according to the configured reduction.

    Args:
        per_token_loss:
            Per-token loss values of shape `(batch_size, seq_len)`.
        response_mask:
            Mask selecting which completion-token positions contribute to the loss.
        loss_type:
            Reduction mode. Uses the same loss-type conventions as the GRPO-family trainers.
        max_completion_length:
            Used by the `dr_grpo` reduction, which normalizes by a fixed completion budget.
    """
    if loss_type == "grpo":
        loss = (per_token_loss * response_mask).sum(-1) / response_mask.sum(-1).clamp(min=1.0)
        return loss.mean()
    if loss_type == "bnpo":
        return (per_token_loss * response_mask).sum() / response_mask.sum().clamp(min=1.0)
    if loss_type == "dr_grpo":
        return (per_token_loss * response_mask).sum() / (per_token_loss.size(0) * max_completion_length)
    if loss_type in ["dapo", "luspo", "cispo", "sapo"]:
        return (per_token_loss * response_mask).sum() / response_mask.sum().clamp(min=1.0)
    raise ValueError(f"Unsupported loss_type: {loss_type}")


def compute_topk_self_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    distillation_topk: int,
    distillation_alpha: float,
    distillation_add_tail: bool,
) -> torch.Tensor:
    """Compute distillation loss on the student's top-k token support.

    The student's top-k logits define the support. The teacher distribution is projected onto the same token indices.
    The selected support is then either renormalized or augmented with a tail bucket before the divergence is computed.
    """
    student_logsumexp = torch.logsumexp(student_logits, dim=-1, keepdim=True)
    topk_student_logits, topk_indices = torch.topk(student_logits, k=distillation_topk, dim=-1)
    topk_student_log_probs = topk_student_logits - student_logsumexp

    teacher_logsumexp = torch.logsumexp(teacher_logits, dim=-1, keepdim=True)
    topk_teacher_logits = torch.gather(teacher_logits, dim=-1, index=topk_indices)
    topk_teacher_log_probs = topk_teacher_logits - teacher_logsumexp

    if distillation_add_tail:
        topk_student_log_probs = add_tail(topk_student_log_probs)
        topk_teacher_log_probs = add_tail(topk_teacher_log_probs)
    else:
        topk_student_log_probs = renorm_topk_log_probs(topk_student_log_probs)
        topk_teacher_log_probs = renorm_topk_log_probs(topk_teacher_log_probs)

    return compute_divergence(topk_student_log_probs, topk_teacher_log_probs, distillation_alpha)


def compute_full_logit_self_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    distillation_alpha: float,
) -> torch.Tensor:
    """Compute full-vocabulary self-distillation loss between student and teacher logits."""
    student_log_probs = torch.log_softmax(student_logits, dim=-1)
    teacher_log_probs = torch.log_softmax(teacher_logits, dim=-1)
    return compute_divergence(student_log_probs, teacher_log_probs, distillation_alpha)


def compute_sampled_token_self_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    completion_ids: torch.Tensor,
    *,
    distillation_alpha: float,
) -> torch.Tensor:
    """Compute token-level self-distillation loss only on the sampled completion tokens.

    This path compares student and teacher log-probabilities on the realized completion tokens rather than over a
    larger token support.
    """
    if distillation_alpha != 1.0:
        raise ValueError(
            "Only reverse KL (alpha=1.0) is supported for token-level distillation when "
            f"`full_logit_distillation=False`, got alpha={distillation_alpha}"
        )

    student_per_token_logps = select_token_log_probs(student_logits, completion_ids)
    teacher_per_token_logps = select_token_log_probs(teacher_logits, completion_ids)
    return compute_token_level_distillation_loss(student_per_token_logps, teacher_per_token_logps)
