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

"""CPU-only unit tests for the DOPD (https://huggingface.co/papers/2606.30626) token-routing loss.

These tests exercise `compute_dopd_routed_loss` in isolation with hand-built logits: no model, tokenizer, vLLM, or
GPU is involved, matching the coverage style of `test_self_distillation_trainer_behavior.py`'s pure loss-util tests.
"""

import pytest
import torch

from trl.experimental.sdft import SDFTConfig
from trl.experimental.sdft.loss_utils import (
    compute_divergence,
    compute_dopd_routed_loss,
    compute_full_logit_self_distillation_loss,
    compute_topk_self_distillation_loss,
)


GAP_THRESHOLD = 1.0
CONFIDENCE_THRESHOLD = 0.5
LIGHT_TOPK = 2
SELF_REG_WEIGHT = 0.01
STUDENT_CONSISTENCY_WEIGHT = 0.1


def _row(probs: list[float]) -> torch.Tensor:
    """Turn a probability vector into a (1, V) logits row via log(p) (already normalized, so log_softmax(log(p)) == log(p))."""
    return torch.tensor(probs, dtype=torch.float32).log().unsqueeze(0)


class TestDOPDRouting:
    def _build_batch(self):
        """Five single-token rows, one per routing regime (see inline comments for the intended regime).

        `privileged_student_logits` are the student's own forward on the privileged context: they differ from the
        bare `student_logits` while preserving each row's routing regime, so regimes 2 and 4 anchor on a genuinely
        different distribution rather than the bare student's detached copy.
        """
        # Row 0 - low gap, both confident -> regime 1 (light top-k reverse-KL)
        student_0 = [0.85, 0.05, 0.05, 0.05]
        privileged_0 = [0.80, 0.10, 0.05, 0.05]  # privileged sees the answer: confident, agrees with teacher
        teacher_0 = [0.80, 0.10, 0.05, 0.05]
        # Row 1 - high gap, teacher confident, student unsure -> regime 3 (full-vocab JSD)
        student_1 = [0.20, 0.30, 0.30, 0.20]
        privileged_1 = [0.25, 0.35, 0.25, 0.15]  # still unsure even with the privileged context
        teacher_1 = [0.90, 0.05, 0.03, 0.02]
        # Row 2 - high gap, student confident, teacher unsure -> regime 4 (light privileged-student consistency)
        student_2 = [0.90, 0.05, 0.03, 0.02]
        privileged_2 = [0.88, 0.05, 0.04, 0.03]
        teacher_2 = [0.20, 0.30, 0.30, 0.20]
        # Row 3 - low gap, neither confident -> regime 2 fallback (weak self-reg)
        student_3 = [0.30, 0.30, 0.20, 0.20]
        privileged_3 = [0.31, 0.29, 0.21, 0.19]
        teacher_3 = [0.28, 0.30, 0.22, 0.20]
        # Row 4 - high gap, neither confident -> regime 2 fallback (ambiguous, least committal update)
        student_4 = [0.02, 0.35, 0.33, 0.30]
        privileged_4 = [0.03, 0.34, 0.33, 0.30]
        teacher_4 = [0.45, 0.30, 0.15, 0.10]

        def _stack(rows):
            return torch.cat([_row(r) for r in rows]).unsqueeze(1)

        student_logits = _stack([student_0, student_1, student_2, student_3, student_4])
        privileged_student_logits = _stack([privileged_0, privileged_1, privileged_2, privileged_3, privileged_4])
        teacher_logits = _stack([teacher_0, teacher_1, teacher_2, teacher_3, teacher_4])
        completion_ids = torch.zeros((5, 1), dtype=torch.long)
        return student_logits, privileged_student_logits, teacher_logits, completion_ids

    def test_each_row_routes_to_its_expected_regime(self):
        student_logits, privileged_student_logits, teacher_logits, completion_ids = self._build_batch()

        routed = compute_dopd_routed_loss(
            student_logits,
            teacher_logits,
            privileged_student_logits,
            completion_ids,
            gap_threshold=GAP_THRESHOLD,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            light_topk=LIGHT_TOPK,
            self_reg_weight=SELF_REG_WEIGHT,
            student_consistency_weight=STUDENT_CONSISTENCY_WEIGHT,
        )
        assert routed.shape == (5, 1)

        expected_regime1 = compute_topk_self_distillation_loss(
            student_logits,
            teacher_logits,
            distillation_topk=LIGHT_TOPK,
            distillation_alpha=1.0,
            distillation_add_tail=True,
            topk_support="teacher",  # matches compute_dopd_routed_loss's paper-faithful convention
        )
        expected_regime3 = compute_full_logit_self_distillation_loss(
            student_logits,
            teacher_logits,
            distillation_alpha=0.5,
        )
        expected_regime4 = STUDENT_CONSISTENCY_WEIGHT * compute_topk_self_distillation_loss(
            student_logits,
            privileged_student_logits.detach(),  # paper eq. 9: sg[ΠS(·|x, p, y<n)]
            distillation_topk=LIGHT_TOPK,
            distillation_alpha=1.0,
            distillation_add_tail=True,
            topk_support="student",
        )
        expected_regime2 = SELF_REG_WEIGHT * compute_topk_self_distillation_loss(
            student_logits,
            privileged_student_logits.detach(),  # paper eq. 7: sg[ΠS(·|x, p, y<n)]
            distillation_topk=LIGHT_TOPK,
            distillation_alpha=1.0,
            distillation_add_tail=True,
            topk_support="student",
        )

        torch.testing.assert_close(routed[0], expected_regime1[0])
        torch.testing.assert_close(routed[1], expected_regime3[1])
        torch.testing.assert_close(routed[2], expected_regime4[2])
        torch.testing.assert_close(routed[3], expected_regime2[3])
        torch.testing.assert_close(routed[4], expected_regime2[4])

    def test_regimes_are_mutually_exclusive_and_exhaustive(self):
        """Every token must be claimed by exactly one regime; rebuild the boolean masks the same way the loss does."""
        student_logits, privileged_student_logits, teacher_logits, completion_ids = self._build_batch()

        # Routing is measured on the privileged pair (the loss routes on the privileged-student gap and
        # confidence, not the bare student's) -- mirror `compute_dopd_routed_loss` exactly.
        privileged_student_logp_tok = torch.gather(
            torch.log_softmax(privileged_student_logits, dim=-1), -1, completion_ids.unsqueeze(-1)
        ).squeeze(-1)
        teacher_logp_tok = torch.gather(
            torch.log_softmax(teacher_logits, dim=-1), -1, completion_ids.unsqueeze(-1)
        ).squeeze(-1)
        gap = (teacher_logp_tok - privileged_student_logp_tok).abs()
        student_conf = privileged_student_logits.softmax(dim=-1).amax(dim=-1)
        teacher_conf = teacher_logits.softmax(dim=-1).amax(dim=-1)

        low_gap = gap <= GAP_THRESHOLD
        high_gap = ~low_gap
        teacher_confident = teacher_conf >= CONFIDENCE_THRESHOLD
        student_confident = student_conf >= CONFIDENCE_THRESHOLD

        regime1 = low_gap & (teacher_confident | student_confident)
        regime3 = high_gap & teacher_confident
        regime4 = high_gap & ~teacher_confident & student_confident
        regime2 = ~(regime1 | regime3 | regime4)

        stacked = torch.stack([regime1, regime2, regime3, regime4], dim=0)
        assert torch.equal(stacked.sum(dim=0), torch.ones_like(gap, dtype=torch.long))
        assert regime1[0, 0] and regime3[1, 0] and regime4[2, 0] and regime2[3, 0] and regime2[4, 0]

    def test_self_reg_regime_anchors_on_privileged_student_not_bare(self):
        """Regime 2 (paper eq. 7) anchors on the privileged student, not the bare student's own detached logits.

        `KL(p || sg(p))` is identically zero in value *and* gradient, so a bare-student anchor would make the
        regime a no-op. With a genuinely different privileged student the loss must be nonzero and must backprop
        a nonzero gradient into the live student.
        """
        # Degenerate case: privileged student == bare student -> the anchor collapses to KL(p || sg(p)) == 0.
        student_logits = _row([0.30, 0.30, 0.20, 0.20]).unsqueeze(1).clone().requires_grad_(True)
        teacher_logits = _row([0.28, 0.30, 0.22, 0.20]).unsqueeze(1)
        completion_ids = torch.zeros((1, 1), dtype=torch.long)

        routed = compute_dopd_routed_loss(
            student_logits,
            teacher_logits,
            student_logits,  # degenerate: privileged == bare student
            completion_ids,
            gap_threshold=GAP_THRESHOLD,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            light_topk=LIGHT_TOPK,
            self_reg_weight=SELF_REG_WEIGHT,
            student_consistency_weight=STUDENT_CONSISTENCY_WEIGHT,
        )
        torch.testing.assert_close(routed, torch.zeros_like(routed), atol=1e-5, rtol=0)
        routed.sum().backward()
        assert student_logits.grad is not None
        # `KL(p || sg(p))` has zero gradient in exact arithmetic, but computing log_softmax twice on the same
        # tensor can leave float32-epsilon-scale residue depending on backend/kernel fusion; use a tolerance
        # instead of an exact-zero check so this isn't hardware-dependent.
        torch.testing.assert_close(student_logits.grad, torch.zeros_like(student_logits.grad), atol=1e-6, rtol=0)

        # Real case: privileged student differs from the bare student -> regime 2 is a live KL with gradient.
        privileged_student_logits = _row([0.35, 0.25, 0.20, 0.20]).unsqueeze(1)
        student_logits = _row([0.30, 0.30, 0.20, 0.20]).unsqueeze(1).clone().requires_grad_(True)
        routed = compute_dopd_routed_loss(
            student_logits,
            teacher_logits,
            privileged_student_logits,
            completion_ids,
            gap_threshold=GAP_THRESHOLD,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            light_topk=LIGHT_TOPK,
            self_reg_weight=SELF_REG_WEIGHT,
            student_consistency_weight=STUDENT_CONSISTENCY_WEIGHT,
        )
        assert not torch.allclose(routed, torch.zeros_like(routed), atol=1e-5)
        routed.sum().backward()
        assert student_logits.grad is not None
        assert torch.any(student_logits.grad.abs() > 0)

    def test_all_regimes_produce_nonzero_gradient_through_student_logits(self):
        """Every regime (1, 2, 3, 4) must backprop into the student, not just regime 2 (checked separately above).

        A silently mis-detached path in any regime's loss formula would zero out that row's gradient while leaving
        the forward value (and the other regime-specific value tests) untouched, so this needs its own check.
        """
        student_logits, privileged_student_logits, teacher_logits, completion_ids = self._build_batch()
        student_logits = student_logits.clone().requires_grad_(True)

        routed = compute_dopd_routed_loss(
            student_logits,
            teacher_logits,
            privileged_student_logits,
            completion_ids,
            gap_threshold=GAP_THRESHOLD,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            light_topk=LIGHT_TOPK,
            self_reg_weight=SELF_REG_WEIGHT,
            student_consistency_weight=STUDENT_CONSISTENCY_WEIGHT,
        )
        routed.sum().backward()

        assert student_logits.grad is not None
        # Row order from `_build_batch`: 0->regime1, 1->regime3, 2->regime4, 3->regime2, 4->regime2 (ambiguous).
        for row, regime_name in enumerate(["regime1", "regime3", "regime4", "regime2", "regime2 (ambiguous)"]):
            row_grad = student_logits.grad[row]
            assert torch.any(row_grad.abs() > 0), (
                f"row {row} ({regime_name}) got a zero gradient through student_logits"
            )

    def test_raising_gap_threshold_moves_high_gap_rows_into_regime_two(self):
        """Sanity check on the threshold's monotonic effect: a huge gap_threshold collapses everything to 'low gap'."""
        student_logits, privileged_student_logits, teacher_logits, completion_ids = self._build_batch()

        routed_permissive = compute_dopd_routed_loss(
            student_logits,
            teacher_logits,
            privileged_student_logits,
            completion_ids,
            gap_threshold=100.0,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            light_topk=LIGHT_TOPK,
            self_reg_weight=SELF_REG_WEIGHT,
            student_consistency_weight=STUDENT_CONSISTENCY_WEIGHT,
        )
        expected_regime1 = compute_topk_self_distillation_loss(
            student_logits,
            teacher_logits,
            distillation_topk=LIGHT_TOPK,
            distillation_alpha=1.0,
            distillation_add_tail=True,
            topk_support="teacher",  # matches compute_dopd_routed_loss's paper-faithful convention
        )
        # Rows 1 and 2 (previously high-gap regimes 3/4) must now fall under regime 1 since every row is "low gap".
        torch.testing.assert_close(routed_permissive[1], expected_regime1[1])
        torch.testing.assert_close(routed_permissive[2], expected_regime1[2])

    def test_routing_uses_privileged_student_not_bare_student(self):
        """The advantage gap is measured between teacher and *privileged* student, not the bare student.

        A token where the bare student agrees with the teacher (low bare gap) but the privileged student is unsure
        and far from the teacher (high privileged gap) must route to regime 3, not regime 1. Routing on the bare
        student would conflate the information-asymmetry gap with the capability gap (the paper's "privilege
        illusion").
        """
        # Bare student: close to the teacher on the realized token -> low bare gap.
        student_logits = _row([0.85, 0.05, 0.05, 0.05]).unsqueeze(1)
        # Privileged student: uniform -> low confidence and a large log-prob gap vs the teacher.
        privileged_student_logits = _row([0.25, 0.25, 0.25, 0.25]).unsqueeze(1)
        teacher_logits = _row([0.90, 0.05, 0.03, 0.02]).unsqueeze(1)
        completion_ids = torch.zeros((1, 1), dtype=torch.long)

        routed = compute_dopd_routed_loss(
            student_logits,
            teacher_logits,
            privileged_student_logits,
            completion_ids,
            gap_threshold=GAP_THRESHOLD,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            light_topk=LIGHT_TOPK,
            self_reg_weight=SELF_REG_WEIGHT,
            student_consistency_weight=STUDENT_CONSISTENCY_WEIGHT,
        )

        # High privileged gap + teacher confident -> regime 3 (full-vocab JSD), NOT regime 1.
        expected_regime3 = compute_full_logit_self_distillation_loss(
            student_logits, teacher_logits, distillation_alpha=0.5
        )
        torch.testing.assert_close(routed, expected_regime3)
        # Regime 1's light top-k loss on this pair is nonzero and different from the regime-3 value.
        expected_regime1 = compute_topk_self_distillation_loss(
            student_logits,
            teacher_logits,
            distillation_topk=LIGHT_TOPK,
            distillation_alpha=1.0,
            distillation_add_tail=True,
            topk_support="teacher",
        )
        assert not torch.allclose(routed, expected_regime1, atol=1e-6), (
            "routing used the bare student's gap instead of the privileged student's gap"
        )


class TestDOPDConfigValidation:
    def test_dopd_rejects_live_teacher(self):
        """`teacher_model_kind='live'` makes the privileged teacher and privileged-student forwards identical,
        collapsing the advantage gap to ~0, so the high-gap routing regimes are unreachable and config must reject it.
        """
        with pytest.raises(ValueError, match="teacher_model_kind='live'"):
            SDFTConfig(
                output_dir="unused",
                distillation_mode="dopd",
                teacher_model_kind="live",
            )

    def test_dopd_allows_base_and_ema_teacher(self):
        SDFTConfig(output_dir="unused", distillation_mode="dopd", teacher_model_kind="base")
        SDFTConfig(output_dir="unused", distillation_mode="dopd", teacher_model_kind="ema")

    def test_teacher_server_aligns_topk_support(self):
        """The server only returns the teacher's top-k, so `distillation_topk_support` is force-aligned to
        `"teacher"` at construction instead of rejecting previously-working default configs.
        """
        config = SDFTConfig(
            output_dir="unused",
            use_teacher_server=True,
            distillation_mode="topk_logits",
            distillation_topk_support="student",  # the default — must not raise
        )
        assert config.distillation_topk_support == "teacher"


class TestTopkSupportParameter:
    """`compute_topk_self_distillation_loss`'s `topk_support` controls whose top-k defines the token support.

    These use logits where the student's and teacher's top-k token sets are disjoint, so switching support must
    change which tokens are even considered, not just reweight the same ones.
    """

    def _build_disjoint_logits(self):
        # Student's top-2 tokens are {0, 1} (masses 0.55, 0.30); teacher's top-2 tokens are {3, 4} (masses 0.35,
        # 0.50). No overlap, and deliberately *not* a plain mirror-image of each other: a mirror-symmetric pair
        # combined with a symmetric divergence (e.g. alpha=0.5 JSD) would coincidentally produce equal losses for
        # both support choices even though the compared token sets differ, which would defeat the point of this test.
        student_logits = _row([0.55, 0.30, 0.08, 0.04, 0.03])
        teacher_logits = _row([0.02, 0.03, 0.10, 0.35, 0.50])
        return student_logits, teacher_logits

    def test_student_and_teacher_support_give_different_losses(self):
        student_logits, teacher_logits = self._build_disjoint_logits()

        loss_student_support = compute_topk_self_distillation_loss(
            student_logits,
            teacher_logits,
            distillation_topk=2,
            distillation_alpha=1.0,
            distillation_add_tail=True,
            topk_support="student",
        )
        loss_teacher_support = compute_topk_self_distillation_loss(
            student_logits,
            teacher_logits,
            distillation_topk=2,
            distillation_alpha=1.0,
            distillation_add_tail=True,
            topk_support="teacher",
        )

        assert not torch.allclose(loss_student_support, loss_teacher_support, atol=1e-4), (
            "topk_support='student' and topk_support='teacher' produced the same loss despite disjoint top-k "
            "token support sets; the parameter is not affecting which tokens are compared."
        )

    def test_invalid_topk_support_raises(self):
        student_logits, teacher_logits = self._build_disjoint_logits()

        with pytest.raises(ValueError, match="topk_support"):
            compute_topk_self_distillation_loss(
                student_logits,
                teacher_logits,
                distillation_topk=2,
                distillation_alpha=0.5,
                distillation_add_tail=True,
                topk_support="nonsense",
            )


class TestKLClip:
    """`compute_divergence`'s `kl_clip` caps the per-position KL (summed over the vocabulary), not individual
    per-vocabulary-entry `F.kl_div` terms, which can be negative on their own and would otherwise let clipping
    flip the summed divergence negative.
    """

    def test_kl_clip_bounds_the_summed_divergence(self):
        # Mismatched enough that the unclipped reverse-KL comfortably exceeds the clip threshold below, and that
        # per-entry `F.kl_div` terms have mixed signs (the low-probability token's term is negative), so a naive
        # per-vocab-entry clip (rather than a post-sum clip) would risk flipping the summed value negative.
        student_log_probs = _row([0.99, 0.01])
        teacher_log_probs = _row([0.5, 0.5])

        unclipped = compute_divergence(student_log_probs, teacher_log_probs, alpha=1.0)
        clip_threshold = unclipped.item() / 2
        clipped = compute_divergence(student_log_probs, teacher_log_probs, alpha=1.0, kl_clip=clip_threshold)

        assert unclipped.item() > clip_threshold, "test setup invalid: clipping never engages"
        assert clipped.item() == pytest.approx(clip_threshold)
        assert clipped.item() >= 0.0

    def test_kl_clip_none_disables_clipping(self):
        student_log_probs = _row([0.99, 0.01])
        teacher_log_probs = _row([0.5, 0.5])

        unclipped = compute_divergence(student_log_probs, teacher_log_probs, alpha=1.0)
        explicit_none = compute_divergence(student_log_probs, teacher_log_probs, alpha=1.0, kl_clip=None)

        torch.testing.assert_close(unclipped, explicit_none)
