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

import torch
import torch.nn.functional as F

from ...trainer.utils import selective_log_softmax


def compute_divergence(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    alpha: float,
    kl_clip: float | None = None,
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
    if kl_clip is not None:
        # Pointwise per-vocabulary-entry clip: caps high-divergence style tokens before the vocabulary sum.
        kl = kl.clamp(max=kl_clip)
    return kl.sum(-1)


def add_tail_bucket(log_probs: torch.Tensor) -> torch.Tensor:
    """Append a bucket holding the leftover probability mass to a top-k log-prob support.

    `log_probs` are true log-probabilities over a top-k subset, so they sum to the captured mass `P_topk <= 1`. This
    appends one extra category equal to the tail mass `1 - P_topk`, yielding a distribution that sums to exactly 1.
    """
    log_s = torch.logsumexp(log_probs, dim=-1, keepdim=True)
    log_s = torch.clamp(log_s, max=-1e-7)
    tail_log = torch.log(-torch.expm1(log_s))
    return torch.cat([log_probs, tail_log], dim=-1)


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


def compute_topk_self_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    distillation_topk: int,
    distillation_alpha: float,
    distillation_add_tail: bool,
    distillation_kl_clip: float | None = None,
    topk_support: str = "student",
) -> torch.Tensor:
    """Compute distillation loss on a top-k token support.

    `topk_support` selects which side's top-k logits define the support: SDFT's convention is `"student"`; some
    methods built on this trainer (e.g. OPSD) use `"teacher"` instead. The other side's distribution is projected
    onto the same token indices. The selected support is then either renormalized or augmented with a tail bucket
    before the divergence is computed.
    """
    if topk_support == "student":
        support_logits, other_logits = student_logits, teacher_logits
    elif topk_support == "teacher":
        support_logits, other_logits = teacher_logits, student_logits
    else:
        raise ValueError(f"topk_support must be 'student' or 'teacher', got {topk_support!r}")

    support_logsumexp = torch.logsumexp(support_logits, dim=-1, keepdim=True)
    topk_support_logits, topk_indices = torch.topk(support_logits, k=distillation_topk, dim=-1)
    topk_support_log_probs = topk_support_logits - support_logsumexp

    other_logsumexp = torch.logsumexp(other_logits, dim=-1, keepdim=True)
    topk_other_logits = torch.gather(other_logits, dim=-1, index=topk_indices)
    topk_other_log_probs = topk_other_logits - other_logsumexp

    if topk_support == "student":
        topk_student_log_probs, topk_teacher_log_probs = topk_support_log_probs, topk_other_log_probs
    else:
        topk_teacher_log_probs, topk_student_log_probs = topk_support_log_probs, topk_other_log_probs

    # Top-k log-probs sum to the captured mass P_topk <= 1; the rest (1 - P_topk) is the "tail".
    if distillation_add_tail:
        # Lump the tail into one bucket so the divergence approximates the full-vocab divergence.
        topk_student_log_probs = add_tail_bucket(topk_student_log_probs)
        topk_teacher_log_probs = add_tail_bucket(topk_teacher_log_probs)
    else:
        # Drop the tail and renormalize the top-k to sum to 1: divergence over the top-k conditional only.
        topk_student_log_probs = topk_student_log_probs - torch.logsumexp(topk_student_log_probs, dim=-1, keepdim=True)
        topk_teacher_log_probs = topk_teacher_log_probs - torch.logsumexp(topk_teacher_log_probs, dim=-1, keepdim=True)

    return compute_divergence(topk_student_log_probs, topk_teacher_log_probs, distillation_alpha, distillation_kl_clip)


def compute_full_logit_self_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    distillation_alpha: float,
    distillation_kl_clip: float | None = None,
) -> torch.Tensor:
    """Compute full-vocabulary self-distillation loss between student and teacher logits."""
    student_log_probs = torch.log_softmax(student_logits, dim=-1)
    teacher_log_probs = torch.log_softmax(teacher_logits, dim=-1)
    return compute_divergence(student_log_probs, teacher_log_probs, distillation_alpha, distillation_kl_clip)


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
            f"`distillation_mode='sampled_token'`, got alpha={distillation_alpha}"
        )

    student_per_token_logps = selective_log_softmax(student_logits, completion_ids)
    teacher_per_token_logps = selective_log_softmax(teacher_logits, completion_ids)
    log_ratio = student_per_token_logps - teacher_per_token_logps
    return log_ratio.detach() * student_per_token_logps


def compute_dopd_routed_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    completion_ids: torch.Tensor,
    *,
    gap_threshold: float,
    confidence_threshold: float,
    light_topk: int,
    self_reg_weight: float,
    student_consistency_weight: float,
) -> torch.Tensor:
    """DOPD-style (https://huggingface.co/papers/2606.30626) advantage-gap routing between four token-level regimes.

    The "advantage gap" is the absolute log-probability difference on the realized token between the privileged
    teacher (``teacher_logits``, e.g. scored with the ground-truth solution in context) and the privileged student
    (``student_logits``, sampled on-policy from the bare problem). Each token is routed into exactly one of:

        1. low gap, either side confident   -> light top-k reverse-KL toward the teacher.
        2. low gap, both sides unsure       -> weak self-regularization, stop-gradient anchor.
        3. high gap, teacher confident      -> full-vocabulary JSD toward the teacher.
        4. high gap, student confident      -> light student-consistency nudge, stop-gradient.

    Tokens where the gap is high but neither side is confident (no reliable regime 3/4 signal) fall back to the
    weak self-regularization of regime 2, matching the paper's intent that ambiguous tokens get the smallest,
    least committal update.
    """
    student_logp_tok = selective_log_softmax(student_logits, completion_ids)
    teacher_logp_tok = selective_log_softmax(teacher_logits, completion_ids)
    gap = (teacher_logp_tok - student_logp_tok).detach().abs()

    student_confidence = student_logits.softmax(dim=-1).amax(dim=-1).detach()
    teacher_confidence = teacher_logits.softmax(dim=-1).amax(dim=-1).detach()
    teacher_confident = teacher_confidence >= confidence_threshold
    student_confident = student_confidence >= confidence_threshold

    low_gap = gap <= gap_threshold
    high_gap = ~low_gap

    regime1 = low_gap & (teacher_confident | student_confident)
    regime3 = high_gap & teacher_confident
    regime4 = high_gap & ~teacher_confident & student_confident
    # Everything not claimed by 1/3/4 (both the plain low-confidence low-gap case and the unresolved
    # high-gap-but-neither-confident case) gets the weak self-regularization fallback.
    regime2 = ~(regime1 | regime3 | regime4)

    loss1 = compute_topk_self_distillation_loss(
        student_logits,
        teacher_logits,
        distillation_topk=light_topk,
        distillation_alpha=1.0,
        distillation_add_tail=True,
        # DOPD's own convention (distinct from SDFT's default "student" support): the teacher's top-k logits define
        # the support for this regime's light signal.
        topk_support="teacher",
    )

    # Entropy-style self-anchor: forward value is ~0 (same distribution on both sides of the KL), but the
    # detached side breaks symmetry so autograd still produces a small regularizing gradient through the live
    # student logits. This is the "stop-gradient" self-regularization the paper uses for low-confidence tokens
    # with no reliable external signal.
    student_log_probs_full = torch.log_softmax(student_logits, dim=-1)
    loss2 = self_reg_weight * compute_divergence(
        student_log_probs_full,
        student_log_probs_full.detach(),
        alpha=1.0,
    )

    loss3 = compute_full_logit_self_distillation_loss(
        student_logits,
        teacher_logits,
        distillation_alpha=0.5,
    )

    # Light, stop-gradient student-consistency nudge: reuses the REINFORCE-style surrogate from
    # `compute_sampled_token_self_distillation_loss` (a detached log-ratio weighting a live student log-prob),
    # scaled down since this regime should not commit as hard as regime 3's full JSD.
    loss4 = student_consistency_weight * compute_sampled_token_self_distillation_loss(
        student_logits,
        teacher_logits,
        completion_ids,
        distillation_alpha=1.0,
    )

    per_token_loss = torch.where(regime1, loss1, torch.zeros_like(loss1))
    per_token_loss = torch.where(regime2, loss2, per_token_loss)
    per_token_loss = torch.where(regime3, loss3, per_token_loss)
    per_token_loss = torch.where(regime4, loss4, per_token_loss)
    return per_token_loss
