# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
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

# Copyright 2025 The HuggingFace Team. All rights reserved.
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

"""
Validation tests for HICRA implementation.

These tests verify that the HICRA implementation correctly follows the VeRL
reference implementation and the paper's algorithm.
"""

import numpy as np
import torch
from datasets import load_dataset

from trl import HICRAConfig, HICRATrainer

from .testing_utils import TrlTestCase


class TestHICRAAlgorithmCorrectness(TrlTestCase):
    """Tests to validate HICRA algorithm correctness against the paper and VeRL implementation."""

    def test_advantage_amplification_formula_without_planning_tokens(self):
        """
        Test that advantage amplification follows VeRL formula without planning tokens.

        VeRL formula (without planning tokens):
        advantages[is_higher_entropy] *= (1 + alpha)

        This test verifies:
        1. Only high-entropy tokens are amplified
        2. Only correct (advantage > 0) responses are amplified
        3. Only longer-than-average responses are amplified
        4. Amplification factor is exactly (1 + alpha)
        """
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        alpha = 0.2
        config = HICRAConfig(
            output_dir=self.tmp_dir,
            use_hicra=True,
            hicra_alpha=alpha,
            hicra_entropy_topk=0.3,
            use_planning_tokens=False,
            report_to="none",
        )

        trainer = HICRATrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=config,
            train_dataset=dataset,
        )

        # Create controlled test data
        # Group 0: Two sequences, first is correct and longer
        batch_size = 2
        seq_len = 10

        # Sequence 0: Correct (positive advantage), length 8, high entropy
        # Sequence 1: Correct (positive advantage), length 4, high entropy
        advantages = torch.tensor(
            [
                [1.0] * seq_len,  # All positive
                [1.0] * seq_len,  # All positive
            ]
        )

        completion_ids = torch.randint(0, 1000, (batch_size, seq_len))

        # Set entropies: first sequence has high entropy, second has low entropy
        entropies = torch.tensor(
            [
                [0.8] * seq_len,  # High entropy
                [0.1] * seq_len,  # Low entropy
            ]
        )

        response_mask = torch.tensor(
            [
                [1.0] * 8 + [0.0] * 2,  # Length 8
                [1.0] * 4 + [0.0] * 6,  # Length 4
            ]
        )

        group_ids = np.array([0, 0])  # Same group

        original_advantages = advantages.clone()

        modified_advantages = trainer.modify_advantages_hicra(
            advantages=advantages,
            completion_ids=completion_ids,
            entropies=entropies,
            response_mask=response_mask,
            group_ids=group_ids,
            planning_token_mask=None,
        )

        # Verify amplification logic:
        # Sequence 0: correct, longer than avg (6), high entropy -> should be amplified
        # Sequence 1: correct, shorter than avg (6), high entropy -> should NOT be amplified

        # Check sequence 0 is amplified
        seq0_amplified_mask = modified_advantages[0] != original_advantages[0]
        assert seq0_amplified_mask.any(), "Sequence 0 should have some amplified tokens"

        # Check sequence 1 is NOT amplified (shorter than average)
        seq1_unchanged = torch.equal(modified_advantages[1], original_advantages[1])
        assert seq1_unchanged, "Sequence 1 should not be amplified (shorter than average)"

        # Verify amplification factor for sequence 0
        # High-entropy tokens should be amplified by (1 + alpha)
        amplified_tokens = modified_advantages[0][seq0_amplified_mask]
        original_tokens = original_advantages[0][seq0_amplified_mask]
        expected_amplified = original_tokens * (1 + alpha)

        torch.testing.assert_close(amplified_tokens, expected_amplified, rtol=1e-5, atol=1e-5)

    def test_advantage_amplification_formula_with_planning_tokens(self):
        """
        Test that advantage amplification follows VeRL formula with planning tokens.

        VeRL formula (with planning tokens):
        advantages[is_higher_entropy | is_planning_token] *= (1 + alpha * sign(advantages))

        This test verifies:
        1. Both high-entropy and planning tokens are amplified
        2. Positive advantages are amplified: adv * (1 + alpha)
        3. Negative advantages are dampened: adv * (1 - alpha)
        4. Only correct and longer-than-average responses are modified
        """
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        alpha = 0.2
        config = HICRAConfig(
            output_dir=self.tmp_dir,
            use_hicra=True,
            hicra_alpha=alpha,
            hicra_entropy_topk=0.3,
            use_planning_tokens=True,
            strategic_grams=["test phrase"],
            report_to="none",
        )

        trainer = HICRATrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=config,
            train_dataset=dataset,
        )

        # Create controlled test data with mixed positive/negative advantages
        batch_size = 2
        seq_len = 10

        # Sequence 0: Correct overall, mixed advantages, length 8
        # Sequence 1: Correct overall, mixed advantages, length 4
        advantages = torch.tensor(
            [
                [1.0, -0.5, 1.0, -0.5, 1.0, -0.5, 1.0, -0.5, 0.0, 0.0],
                [1.0, -0.5, 1.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )

        completion_ids = torch.randint(0, 1000, (batch_size, seq_len))

        # High entropy for all valid tokens
        entropies = torch.tensor(
            [
                [0.8] * seq_len,
                [0.8] * seq_len,
            ]
        )

        response_mask = torch.tensor(
            [
                [1.0] * 8 + [0.0] * 2,  # Length 8
                [1.0] * 4 + [0.0] * 6,  # Length 4
            ]
        )

        group_ids = np.array([0, 0])

        # Mark some tokens as planning tokens
        planning_token_mask = torch.tensor(
            [
                [True, True, False, False, True, True, False, False, False, False],
                [True, True, False, False, False, False, False, False, False, False],
            ]
        )

        original_advantages = advantages.clone()

        modified_advantages = trainer.modify_advantages_hicra(
            advantages=advantages,
            completion_ids=completion_ids,
            entropies=entropies,
            response_mask=response_mask,
            group_ids=group_ids,
            planning_token_mask=planning_token_mask,
        )

        # Verify signed amplification for sequence 0 (longer than average)
        # Positive advantages should be amplified: adv * (1 + alpha)
        # Negative advantages should be dampened: adv * (1 - alpha)

        # Check sequence 0 is modified
        seq0_modified = not torch.equal(modified_advantages[0], original_advantages[0])
        assert seq0_modified, "Sequence 0 should be modified"

        # Check sequence 1 is NOT modified (shorter than average)
        seq1_unchanged = torch.equal(modified_advantages[1], original_advantages[1])
        assert seq1_unchanged, "Sequence 1 should not be modified (shorter than average)"

        # Verify signed amplification for sequence 0
        # Tokens 0, 1, 4, 5 are planning or high-entropy and should be modified
        for i in [0, 1, 4, 5]:
            if response_mask[0, i] > 0:
                original_adv = original_advantages[0, i].item()
                modified_adv = modified_advantages[0, i].item()

                if original_adv > 0:
                    # Positive advantage: should be amplified
                    expected = original_adv * (1 + alpha)
                    assert abs(modified_adv - expected) < 1e-5, (
                        f"Token {i}: positive advantage not amplified correctly. "
                        f"Expected {expected}, got {modified_adv}"
                    )
                elif original_adv < 0:
                    # Negative advantage: should be dampened
                    expected = original_adv * (1 - alpha)
                    assert abs(modified_adv - expected) < 1e-5, (
                        f"Token {i}: negative advantage not dampened correctly. "
                        f"Expected {expected}, got {modified_adv}"
                    )

    def test_entropy_threshold_computation(self):
        """
        Test that entropy threshold is computed correctly at top-k percentile.

        VeRL uses topk_threshold with k=0.3 to compute the (1-k) = 70th percentile,
        which serves as the baseline for identifying high-entropy tokens.
        """
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        config = HICRAConfig(
            output_dir=self.tmp_dir,
            hicra_entropy_topk=0.3,
            report_to="none",
        )

        trainer = HICRATrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=config,
            train_dataset=dataset,
        )

        # Create test data with known entropy distribution
        # Sequence: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        # 70th percentile (1-0.3) should be around 0.7
        mask = torch.ones(1, 10, dtype=torch.float)
        values = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]])

        threshold = trainer._topk_threshold(mask, values, k=0.3)

        # Threshold should be at 70th percentile
        # For 10 values, 70th percentile is at index 7 (0-indexed), which is 0.7
        expected_threshold = 0.7
        assert abs(threshold[0, 0].item() - expected_threshold) < 0.1, (
            f"Threshold should be around {expected_threshold}, got {threshold[0, 0].item()}"
        )

    def test_length_based_filtering(self):
        """
        Test that only longer-than-average correct responses are amplified.

        VeRL filters by:
        should_amplify = is_correct_response & is_longer_than_average
        """
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        config = HICRAConfig(
            output_dir=self.tmp_dir,
            use_hicra=True,
            hicra_alpha=0.2,
            use_planning_tokens=False,
            report_to="none",
        )

        trainer = HICRATrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=config,
            train_dataset=dataset,
        )

        # Create test data with 4 sequences in same group
        # Lengths: [8, 4, 6, 2]
        # Average length: 5
        # Correct: [True, True, True, True]
        # Should amplify: [True (8>5), False (4<5), True (6>5), False (2<5)]

        batch_size = 4
        seq_len = 10

        advantages = torch.ones(batch_size, seq_len)  # All correct
        completion_ids = torch.randint(0, 1000, (batch_size, seq_len))
        entropies = torch.ones(batch_size, seq_len) * 0.8  # All high entropy

        response_mask = torch.tensor(
            [
                [1.0] * 8 + [0.0] * 2,  # Length 8
                [1.0] * 4 + [0.0] * 6,  # Length 4
                [1.0] * 6 + [0.0] * 4,  # Length 6
                [1.0] * 2 + [0.0] * 8,  # Length 2
            ]
        )

        group_ids = np.array([0, 0, 0, 0])  # All same group

        original_advantages = advantages.clone()

        modified_advantages = trainer.modify_advantages_hicra(
            advantages=advantages,
            completion_ids=completion_ids,
            entropies=entropies,
            response_mask=response_mask,
            group_ids=group_ids,
            planning_token_mask=None,
        )

        # Check which sequences were amplified
        seq0_amplified = not torch.equal(modified_advantages[0], original_advantages[0])
        seq1_amplified = not torch.equal(modified_advantages[1], original_advantages[1])
        seq2_amplified = not torch.equal(modified_advantages[2], original_advantages[2])
        seq3_amplified = not torch.equal(modified_advantages[3], original_advantages[3])

        assert seq0_amplified, "Sequence 0 (length 8 > avg 5) should be amplified"
        assert not seq1_amplified, "Sequence 1 (length 4 < avg 5) should NOT be amplified"
        assert seq2_amplified, "Sequence 2 (length 6 > avg 5) should be amplified"
        assert not seq3_amplified, "Sequence 3 (length 2 < avg 5) should NOT be amplified"

    def test_incorrect_responses_not_amplified(self):
        """
        Test that incorrect responses (advantage <= 0) are never amplified.

        VeRL filters by:
        is_correct_response = group_advantages > 0
        """
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        config = HICRAConfig(
            output_dir=self.tmp_dir,
            use_hicra=True,
            hicra_alpha=0.2,
            use_planning_tokens=False,
            report_to="none",
        )

        trainer = HICRATrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=config,
            train_dataset=dataset,
        )

        # Create test data with incorrect responses
        batch_size = 2
        seq_len = 10

        # Sequence 0: Incorrect (negative advantage), length 8
        # Sequence 1: Incorrect (negative advantage), length 4
        advantages = torch.tensor(
            [
                [-1.0] * seq_len,  # All negative (incorrect)
                [-1.0] * seq_len,  # All negative (incorrect)
            ]
        )

        completion_ids = torch.randint(0, 1000, (batch_size, seq_len))
        entropies = torch.ones(batch_size, seq_len) * 0.8  # All high entropy

        response_mask = torch.tensor(
            [
                [1.0] * 8 + [0.0] * 2,
                [1.0] * 4 + [0.0] * 6,
            ]
        )

        group_ids = np.array([0, 0])

        original_advantages = advantages.clone()

        modified_advantages = trainer.modify_advantages_hicra(
            advantages=advantages,
            completion_ids=completion_ids,
            entropies=entropies,
            response_mask=response_mask,
            group_ids=group_ids,
            planning_token_mask=None,
        )

        # Neither sequence should be amplified (both incorrect)
        torch.testing.assert_close(modified_advantages, original_advantages)

    def test_group_based_processing(self):
        """
        Test that HICRA processes each GRPO group independently.

        VeRL iterates over unique group IDs and processes each group separately.
        """
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        config = HICRAConfig(
            output_dir=self.tmp_dir,
            use_hicra=True,
            hicra_alpha=0.2,
            use_planning_tokens=False,
            report_to="none",
        )

        trainer = HICRATrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=config,
            train_dataset=dataset,
        )

        # Create test data with two groups
        # Group 0: lengths [8, 4], avg = 6
        # Group 1: lengths [6, 2], avg = 4

        batch_size = 4
        seq_len = 10

        advantages = torch.ones(batch_size, seq_len)  # All correct
        completion_ids = torch.randint(0, 1000, (batch_size, seq_len))
        entropies = torch.ones(batch_size, seq_len) * 0.8  # All high entropy

        response_mask = torch.tensor(
            [
                [1.0] * 8 + [0.0] * 2,  # Group 0, length 8 > avg 6
                [1.0] * 4 + [0.0] * 6,  # Group 0, length 4 < avg 6
                [1.0] * 6 + [0.0] * 4,  # Group 1, length 6 > avg 4
                [1.0] * 2 + [0.0] * 8,  # Group 1, length 2 < avg 4
            ]
        )

        group_ids = np.array([0, 0, 1, 1])

        original_advantages = advantages.clone()

        modified_advantages = trainer.modify_advantages_hicra(
            advantages=advantages,
            completion_ids=completion_ids,
            entropies=entropies,
            response_mask=response_mask,
            group_ids=group_ids,
            planning_token_mask=None,
        )

        # Check amplification per group
        seq0_amplified = not torch.equal(modified_advantages[0], original_advantages[0])
        seq1_amplified = not torch.equal(modified_advantages[1], original_advantages[1])
        seq2_amplified = not torch.equal(modified_advantages[2], original_advantages[2])
        seq3_amplified = not torch.equal(modified_advantages[3], original_advantages[3])

        # Group 0: seq0 (8 > 6) should be amplified, seq1 (4 < 6) should not
        assert seq0_amplified, "Group 0, seq 0 should be amplified"
        assert not seq1_amplified, "Group 0, seq 1 should NOT be amplified"

        # Group 1: seq2 (6 > 4) should be amplified, seq3 (2 < 4) should not
        assert seq2_amplified, "Group 1, seq 2 should be amplified"
        assert not seq3_amplified, "Group 1, seq 3 should NOT be amplified"


class TestHICRAPerformance(TrlTestCase):
    """Tests to validate HICRA performance and efficiency."""

    def test_planning_token_identification_performance(self):
        """Test that planning token identification is reasonably fast."""
        import time

        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        config = HICRAConfig(
            output_dir=self.tmp_dir,
            strategic_grams=["let's try"] * 100,  # 100 Strategic Grams
            use_planning_tokens=True,
            report_to="none",
        )

        trainer = HICRATrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=config,
            train_dataset=dataset,
        )

        # Create large batch
        batch_size = 32
        seq_len = 128
        completion_ids = torch.randint(0, 1000, (batch_size, seq_len))

        # Warm-up
        _ = trainer.identify_planning_tokens(completion_ids)

        # Time the operation
        start_time = time.time()
        for _ in range(10):
            _ = trainer.identify_planning_tokens(completion_ids)
        elapsed_time = time.time() - start_time

        # Should complete in reasonable time (< 1 second for 10 iterations)
        assert elapsed_time < 1.0, f"Planning token identification too slow: {elapsed_time:.3f}s for 10 iterations"

    def test_advantage_modification_performance(self):
        """Test that advantage modification is reasonably fast."""
        import time

        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        config = HICRAConfig(
            output_dir=self.tmp_dir,
            use_hicra=True,
            hicra_alpha=0.2,
            report_to="none",
        )

        trainer = HICRATrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=config,
            train_dataset=dataset,
        )

        # Create large batch
        batch_size = 32
        seq_len = 128
        advantages = torch.randn(batch_size, seq_len)
        completion_ids = torch.randint(0, 1000, (batch_size, seq_len))
        entropies = torch.rand(batch_size, seq_len)
        response_mask = torch.ones(batch_size, seq_len)
        group_ids = np.repeat(np.arange(batch_size // 4), 4)

        # Warm-up
        _ = trainer.modify_advantages_hicra(
            advantages=advantages.clone(),
            completion_ids=completion_ids,
            entropies=entropies,
            response_mask=response_mask,
            group_ids=group_ids,
        )

        # Time the operation
        start_time = time.time()
        for _ in range(10):
            _ = trainer.modify_advantages_hicra(
                advantages=advantages.clone(),
                completion_ids=completion_ids,
                entropies=entropies,
                response_mask=response_mask,
                group_ids=group_ids,
            )
        elapsed_time = time.time() - start_time

        # Should complete in reasonable time (< 1 second for 10 iterations)
        assert elapsed_time < 1.0, f"Advantage modification too slow: {elapsed_time:.3f}s for 10 iterations"

    def test_memory_efficiency_with_large_batch(self):
        """Test that HICRA doesn't cause memory issues with large batches."""
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        config = HICRAConfig(
            output_dir=self.tmp_dir,
            use_hicra=True,
            hicra_alpha=0.2,
            use_planning_tokens=True,
            strategic_grams=["let's try"] * 50,
            report_to="none",
        )

        trainer = HICRATrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=config,
            train_dataset=dataset,
        )

        # Create large batch
        batch_size = 64
        seq_len = 256
        completion_ids = torch.randint(0, 1000, (batch_size, seq_len))
        advantages = torch.randn(batch_size, seq_len)
        entropies = torch.rand(batch_size, seq_len)
        response_mask = torch.ones(batch_size, seq_len)
        group_ids = np.repeat(np.arange(batch_size // 4), 4)

        # Should not raise OOM or other memory errors
        planning_mask = trainer.identify_planning_tokens(completion_ids)
        modified_advantages = trainer.modify_advantages_hicra(
            advantages=advantages,
            completion_ids=completion_ids,
            entropies=entropies,
            response_mask=response_mask,
            group_ids=group_ids,
            planning_token_mask=planning_mask,
        )

        assert modified_advantages.shape == advantages.shape
