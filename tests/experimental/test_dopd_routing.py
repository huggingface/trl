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

from trl.experimental.sdft.loss_utils import (
    compute_divergence,
    compute_dopd_routed_loss,
    compute_full_logit_self_distillation_loss,
    compute_sampled_token_self_distillation_loss,
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
        """Five single-token rows, one per routing regime (see inline comments for the intended regime)."""
        # Row 0 - low gap, both confident -> regime 1 (light top-k reverse-KL)
        student_0 = [0.85, 0.05, 0.05, 0.05]
        teacher_0 = [0.80, 0.10, 0.05, 0.05]
        # Row 1 - high gap, teacher confident, student unsure -> regime 3 (full-vocab JSD)
        student_1 = [0.20, 0.30, 0.30, 0.20]
        teacher_1 = [0.90, 0.05, 0.03, 0.02]
        # Row 2 - high gap, student confident, teacher unsure -> regime 4 (light stop-grad consistency)
        student_2 = [0.90, 0.05, 0.03, 0.02]
        teacher_2 = [0.20, 0.30, 0.30, 0.20]
        # Row 3 - low gap, neither confident -> regime 2 fallback (weak stop-grad self-reg)
        student_3 = [0.30, 0.30, 0.20, 0.20]
        teacher_3 = [0.28, 0.30, 0.22, 0.20]
        # Row 4 - high gap, neither confident -> regime 2 fallback (ambiguous, least committal update)
        student_4 = [0.02, 0.35, 0.33, 0.30]
        teacher_4 = [0.45, 0.30, 0.15, 0.10]

        student_logits = torch.cat(
            [_row(student_0), _row(student_1), _row(student_2), _row(student_3), _row(student_4)]
        ).unsqueeze(1)
        teacher_logits = torch.cat(
            [_row(teacher_0), _row(teacher_1), _row(teacher_2), _row(teacher_3), _row(teacher_4)]
        ).unsqueeze(1)
        completion_ids = torch.zeros((5, 1), dtype=torch.long)
        return student_logits, teacher_logits, completion_ids

    def test_each_row_routes_to_its_expected_regime(self):
        student_logits, teacher_logits, completion_ids = self._build_batch()

        routed = compute_dopd_routed_loss(
            student_logits,
            teacher_logits,
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
        expected_regime4 = STUDENT_CONSISTENCY_WEIGHT * compute_sampled_token_self_distillation_loss(
            student_logits,
            teacher_logits,
            completion_ids,
            distillation_alpha=1.0,
        )
        student_log_probs_full = torch.log_softmax(student_logits, dim=-1)
        expected_regime2 = SELF_REG_WEIGHT * compute_divergence(
            student_log_probs_full,
            student_log_probs_full.detach(),
            alpha=1.0,
        )

        torch.testing.assert_close(routed[0], expected_regime1[0])
        torch.testing.assert_close(routed[1], expected_regime3[1])
        torch.testing.assert_close(routed[2], expected_regime4[2])
        torch.testing.assert_close(routed[3], expected_regime2[3])
        torch.testing.assert_close(routed[4], expected_regime2[4])

    def test_regimes_are_mutually_exclusive_and_exhaustive(self):
        """Every token must be claimed by exactly one regime; rebuild the boolean masks the same way the loss does."""
        student_logits, teacher_logits, completion_ids = self._build_batch()

        student_logp_tok = torch.gather(
            torch.log_softmax(student_logits, dim=-1), -1, completion_ids.unsqueeze(-1)
        ).squeeze(-1)
        teacher_logp_tok = torch.gather(
            torch.log_softmax(teacher_logits, dim=-1), -1, completion_ids.unsqueeze(-1)
        ).squeeze(-1)
        gap = (teacher_logp_tok - student_logp_tok).abs()
        student_conf = student_logits.softmax(dim=-1).amax(dim=-1)
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

    def test_self_regularization_regime_has_near_zero_value_but_nonzero_gradient(self):
        """Regime 2 is a stop-gradient anchor: forward value ~0, but it must still emit a regularizing gradient."""
        student_logits = _row([0.30, 0.30, 0.20, 0.20]).unsqueeze(1).clone().requires_grad_(True)
        teacher_logits = _row([0.28, 0.30, 0.22, 0.20]).unsqueeze(1)
        completion_ids = torch.zeros((1, 1), dtype=torch.long)

        routed = compute_dopd_routed_loss(
            student_logits,
            teacher_logits,
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
        assert torch.any(student_logits.grad.abs() > 0)

    def test_all_regimes_produce_nonzero_gradient_through_student_logits(self):
        """Every regime (1, 2, 3, 4) must backprop into the student, not just regime 2 (checked separately above).

        A silently mis-detached path in any regime's loss formula would zero out that row's gradient while leaving
        the forward value (and the other regime-specific value tests) untouched, so this needs its own check.
        """
        student_logits, teacher_logits, completion_ids = self._build_batch()
        student_logits = student_logits.clone().requires_grad_(True)

        routed = compute_dopd_routed_loss(
            student_logits,
            teacher_logits,
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
        student_logits, teacher_logits, completion_ids = self._build_batch()

        routed_permissive = compute_dopd_routed_loss(
            student_logits,
            teacher_logits,
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
