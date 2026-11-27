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

"""CPU-only unit tests for the SDFT distillation plumbing: top-k support, KL clipping, teacher-server alignment.

Hand-built logits only; no model, tokenizer, vLLM, or GPU is involved, matching the style of
`test_self_distillation_trainer_behavior.py`'s pure loss-util tests.
"""

import pytest
import torch

from trl.experimental.sdft import SDFTConfig
from trl.experimental.sdft.loss_utils import compute_divergence, compute_topk_self_distillation_loss
from trl.experimental.sdpo import SDPOConfig


def _row(probs: list[float]) -> torch.Tensor:
    """Turn a probability vector into a (1, V) logits row via log(p) (already normalized, so log_softmax(log(p)) == log(p))."""
    return torch.tensor(probs, dtype=torch.float32).log().unsqueeze(0)


class TestTopkSupportParameter:
    """`compute_topk_self_distillation_loss`'s `topk_support` controls whose top-k defines the token support.

    These use logits where the student's and teacher's top-k token sets are disjoint, so switching support must change
    which tokens are even considered, not just reweight the same ones.
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
    per-vocabulary-entry `F.kl_div` terms, which can be negative on their own and would otherwise let clipping flip the
    summed divergence negative.
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


class TestTeacherServerTopkSupport:
    def test_use_teacher_server_aligns_topk_support(self):
        """The vLLM server returns only the teacher's top-k, so `distillation_topk_support` is force-aligned to
        `"teacher"` at construction rather than rejecting a config that used to work.
        """
        config = SDFTConfig(
            output_dir="unused",
            use_teacher_server=True,
            distillation_mode="topk_logits",
            distillation_topk_support="student",  # the default — must not raise
        )
        assert config.distillation_topk_support == "teacher"

    def test_kl_clip_rejected_for_modes_without_full_logits(self):
        """`kl_clip` needs a per-position divergence over the vocabulary, which `sampled_token` does not produce."""
        with pytest.raises(ValueError, match="distillation_kl_clip"):
            SDFTConfig(
                output_dir="unused",
                distillation_mode="sampled_token",
                distillation_alpha=1.0,  # sampled_token requires reverse KL; set it so the kl_clip check is reached
                distillation_kl_clip=1.0,
            )


class TestSDPOConfigParity:
    """SDPOConfig duplicates SDFT's distillation config, so the same two knobs must exist and validate the same way."""

    def test_fields_exist_with_matching_defaults(self):
        config = SDPOConfig(output_dir="unused")
        assert config.distillation_topk_support == "student"
        assert config.distillation_kl_clip is None

    def test_kl_clip_rejected_for_modes_without_full_logits(self):
        with pytest.raises(ValueError, match="distillation_kl_clip"):
            SDPOConfig(
                output_dir="unused",
                distillation_mode="sampled_token",
                distillation_alpha=1.0,
                distillation_kl_clip=1.0,
            )

    def test_use_teacher_server_aligns_topk_support(self):
        config = SDPOConfig(
            output_dir="unused",
            use_vllm=True,
            vllm_mode="server",
            use_teacher_server=True,
            teacher_model_kind="live",
            distillation_weight=1.0,
            distillation_mode="topk_logits",
            distillation_topk=2,
            distillation_topk_support="student",  # the default — must not raise
        )
        assert config.distillation_topk_support == "teacher"
