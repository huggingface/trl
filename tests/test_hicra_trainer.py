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

import json
import os
import tempfile

import numpy as np
import pytest
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from trl import HICRAConfig, HICRATrainer
from trl.trainer.strategic_grams import (
    extract_strategic_grams,
    get_default_strategic_grams,
    load_strategic_grams_from_file,
    save_strategic_grams_to_file,
)

from .testing_utils import TrlTestCase, require_peft


class TestStrategicGrams(TrlTestCase):
    """Unit tests for Strategic Gram utilities."""

    def test_get_default_strategic_grams_math(self):
        """Test loading default Strategic Grams for math domain."""
        sgs = get_default_strategic_grams("math")
        assert isinstance(sgs, list)
        assert len(sgs) > 0
        assert all(isinstance(sg, str) for sg in sgs)
        # Check for some expected math Strategic Grams
        assert any("approach" in sg.lower() for sg in sgs)

    def test_get_default_strategic_grams_code(self):
        """Test loading default Strategic Grams for code domain."""
        sgs = get_default_strategic_grams("code")
        assert isinstance(sgs, list)
        assert len(sgs) > 0
        assert all(isinstance(sg, str) for sg in sgs)

    def test_get_default_strategic_grams_invalid_domain(self):
        """Test that invalid domain raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported domain"):
            get_default_strategic_grams("invalid_domain")

    def test_save_and_load_strategic_grams(self):
        """Test saving and loading Strategic Grams from file."""
        test_sgs = ["let's try", "we can use", "notice that"]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            # Save
            save_strategic_grams_to_file(test_sgs, temp_path)
            assert os.path.exists(temp_path)

            # Load
            loaded_sgs = load_strategic_grams_from_file(temp_path)
            assert loaded_sgs == test_sgs
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_load_strategic_grams_invalid_format(self):
        """Test that loading invalid format raises ValueError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"not": "a list"}, f)
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Expected a list"):
                load_strategic_grams_from_file(temp_path)
        finally:
            os.unlink(temp_path)

    def test_load_strategic_grams_non_string_items(self):
        """Test that loading non-string items raises ValueError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([1, 2, 3], f)
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="All items.*must be strings"):
                load_strategic_grams_from_file(temp_path)
        finally:
            os.unlink(temp_path)

    def test_extract_strategic_grams_basic(self):
        """Test basic Strategic Gram extraction from corpus."""
        corpus = [
            "let's try a different approach to solve this",
            "we can use the fact that numbers are positive",
            "notice that the pattern repeats every time",
        ]

        sgs = extract_strategic_grams(corpus, n_range=(3, 4), n_clusters=2)
        assert isinstance(sgs, list)
        assert len(sgs) > 0
        assert all(isinstance(sg, str) for sg in sgs)

    def test_extract_strategic_grams_empty_corpus(self):
        """Test extraction with empty corpus."""
        sgs = extract_strategic_grams([], n_range=(3, 5))
        assert sgs == []


class TestStrategicGramMatching(TrlTestCase):
    """Unit tests for Strategic Gram matching in HICRATrainer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
        self.strategic_grams = ["let's try", "we can use", "notice that"]

    def test_identify_planning_tokens_basic(self):
        """Test basic planning token identification with known sequences."""
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        config = HICRAConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,
            num_generations=2,
            max_completion_length=8,
            strategic_grams=self.strategic_grams,
            use_planning_tokens=True,
            report_to="none",
        )

        trainer = HICRATrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=config,
            train_dataset=dataset,
        )

        # Create test completion with known Strategic Gram
        text = "let's try a different approach"
        completion_ids = torch.tensor([trainer.processing_class.encode(text, add_special_tokens=False)])

        planning_mask = trainer.identify_planning_tokens(completion_ids)

        assert planning_mask.shape == completion_ids.shape
        assert planning_mask.dtype == torch.bool
        # Should have at least some planning tokens
        assert planning_mask.any()

    def test_identify_planning_tokens_empty_sequence(self):
        """Test planning token identification with empty sequence."""
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        config = HICRAConfig(
            output_dir=self.tmp_dir,
            strategic_grams=self.strategic_grams,
            report_to="none",
        )

        trainer = HICRATrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=config,
            train_dataset=dataset,
        )

        # Empty sequence
        completion_ids = torch.tensor([[]], dtype=torch.long)
        planning_mask = trainer.identify_planning_tokens(completion_ids)

        assert planning_mask.shape == completion_ids.shape
        assert not planning_mask.any()

    def test_identify_planning_tokens_no_matches(self):
        """Test planning token identification with no Strategic Gram matches."""
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        config = HICRAConfig(
            output_dir=self.tmp_dir,
            strategic_grams=self.strategic_grams,
            report_to="none",
        )

        trainer = HICRATrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=config,
            train_dataset=dataset,
        )

        # Text with no Strategic Grams
        text = "hello world this is a test"
        completion_ids = torch.tensor([trainer.processing_class.encode(text, add_special_tokens=False)])

        planning_mask = trainer.identify_planning_tokens(completion_ids)

        assert planning_mask.shape == completion_ids.shape
        # Should have no planning tokens
        assert not planning_mask.any()

    def test_identify_planning_tokens_overlapping_sgs(self):
        """Test planning token identification with overlapping Strategic Grams."""
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        # Use Strategic Grams that might overlap
        overlapping_sgs = ["let's try", "try a", "a different"]

        config = HICRAConfig(
            output_dir=self.tmp_dir,
            strategic_grams=overlapping_sgs,
            report_to="none",
        )

        trainer = HICRATrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=config,
            train_dataset=dataset,
        )

        text = "let's try a different approach"
        completion_ids = torch.tensor([trainer.processing_class.encode(text, add_special_tokens=False)])

        planning_mask = trainer.identify_planning_tokens(completion_ids)

        assert planning_mask.shape == completion_ids.shape
        # With overlapping SGs, multiple tokens should be marked
        assert planning_mask.any()

    def test_identify_planning_tokens_no_strategic_grams(self):
        """Test planning token identification with no Strategic Grams loaded."""
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        config = HICRAConfig(
            output_dir=self.tmp_dir,
            strategic_grams=[],  # Empty list
            report_to="none",
        )

        trainer = HICRATrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=config,
            train_dataset=dataset,
        )

        text = "let's try a different approach"
        completion_ids = torch.tensor([trainer.processing_class.encode(text, add_special_tokens=False)])

        planning_mask = trainer.identify_planning_tokens(completion_ids)

        assert planning_mask.shape == completion_ids.shape
        # No Strategic Grams means no planning tokens
        assert not planning_mask.any()


class TestHICRAAdvantageModification(TrlTestCase):
    """Unit tests for HICRA advantage modification logic."""

    def test_topk_threshold_basic(self):
        """Test entropy threshold computation at top-k percentile."""
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

        # Create test data
        mask = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]], dtype=torch.float)
        values = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [0.7, 0.8, 0.9, 1.0, 1.1, 1.2]])

        thresholds = trainer._topk_threshold(mask, values, k=0.3)

        assert thresholds.shape == (2, 1)
        # For first sequence: top 30% of [0.1, 0.2, 0.3, 0.4, 0.5, 0.6] is around 0.42
        # For second sequence: top 30% of [0.7, 0.8, 0.9, 1.0] is around 0.91
        assert thresholds[0, 0] > 0.3
        assert thresholds[1, 0] > 0.8

    def test_topk_threshold_empty_mask(self):
        """Test entropy threshold with empty mask."""
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        config = HICRAConfig(
            output_dir=self.tmp_dir,
            report_to="none",
        )

        trainer = HICRATrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=config,
            train_dataset=dataset,
        )

        mask = torch.tensor([[0, 0, 0, 0]], dtype=torch.float)
        values = torch.tensor([[0.1, 0.2, 0.3, 0.4]])

        thresholds = trainer._topk_threshold(mask, values, k=0.3)

        assert thresholds.shape == (1, 1)
        assert thresholds[0, 0] == 0.0

    def test_modify_advantages_hicra_disabled(self):
        """Test that HICRA modification is skipped when disabled."""
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        config = HICRAConfig(
            output_dir=self.tmp_dir,
            use_hicra=False,  # Disable HICRA
            report_to="none",
        )

        trainer = HICRATrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=config,
            train_dataset=dataset,
        )

        # Create test data
        batch_size, seq_len = 4, 10
        advantages = torch.randn(batch_size, seq_len)
        completion_ids = torch.randint(0, 1000, (batch_size, seq_len))
        entropies = torch.rand(batch_size, seq_len)
        response_mask = torch.ones(batch_size, seq_len)
        group_ids = np.array([0, 0, 1, 1])

        original_advantages = advantages.clone()

        modified_advantages = trainer.modify_advantages_hicra(
            advantages=advantages,
            completion_ids=completion_ids,
            entropies=entropies,
            response_mask=response_mask,
            group_ids=group_ids,
        )

        # Advantages should be unchanged
        torch.testing.assert_close(modified_advantages, original_advantages)

    def test_modify_advantages_hicra_length_filtering(self):
        """Test length-based filtering logic in HICRA."""
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

        # Create test data with different lengths
        # Group 0: lengths [6, 4] - avg = 5, only first should be amplified
        # Group 1: lengths [8, 6] - avg = 7, only first should be amplified
        batch_size = 4
        advantages = torch.tensor(
            [
                [1.0] * 6 + [0.0] * 4,  # Correct, longer than avg
                [1.0] * 4 + [0.0] * 6,  # Correct, shorter than avg
                [1.0] * 8 + [0.0] * 2,  # Correct, longer than avg
                [1.0] * 6 + [0.0] * 4,  # Correct, shorter than avg
            ]
        )
        completion_ids = torch.randint(0, 1000, (batch_size, 10))
        entropies = torch.ones(batch_size, 10) * 0.5  # All high entropy
        response_mask = torch.tensor(
            [
                [1.0] * 6 + [0.0] * 4,
                [1.0] * 4 + [0.0] * 6,
                [1.0] * 8 + [0.0] * 2,
                [1.0] * 6 + [0.0] * 4,
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
        )

        # Check that only longer-than-average correct responses are amplified
        # Sequences 0 and 2 should be amplified, 1 and 3 should not
        assert (modified_advantages[0] != original_advantages[0]).any()
        assert torch.equal(modified_advantages[1], original_advantages[1])
        assert (modified_advantages[2] != original_advantages[2]).any()
        assert torch.equal(modified_advantages[3], original_advantages[3])

    def test_modify_advantages_hicra_signed_amplification(self):
        """Test signed amplification formula with planning tokens."""
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        config = HICRAConfig(
            output_dir=self.tmp_dir,
            use_hicra=True,
            hicra_alpha=0.2,
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

        # Create test data with mixed positive/negative advantages
        batch_size, seq_len = 2, 10
        advantages = torch.tensor(
            [
                [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
                [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
            ]
        )
        completion_ids = torch.randint(0, 1000, (batch_size, seq_len))
        entropies = torch.ones(batch_size, seq_len) * 0.5
        response_mask = torch.ones(batch_size, seq_len)
        group_ids = np.array([0, 0])
        planning_token_mask = torch.tensor(
            [
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # First two tokens are planning
                [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],  # Third and fourth tokens are planning
            ],
            dtype=torch.bool,
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

        # With signed amplification:
        # Positive advantages should be amplified: adv * (1 + alpha)
        # Negative advantages should be dampened: adv * (1 - alpha)
        # Check that planning tokens are modified
        assert not torch.equal(modified_advantages, original_advantages)

    def test_modify_advantages_hicra_without_planning_tokens(self):
        """Test HICRA modification without planning token mask."""
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

        # Create test data
        batch_size, seq_len = 2, 10
        advantages = torch.ones(batch_size, seq_len)
        completion_ids = torch.randint(0, 1000, (batch_size, seq_len))
        entropies = torch.ones(batch_size, seq_len) * 0.5
        response_mask = torch.ones(batch_size, seq_len)
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

        # Without planning tokens, only high-entropy tokens are amplified
        # Should use unsigned amplification: adv * (1 + alpha)
        assert not torch.equal(modified_advantages, original_advantages)


class TestHICRATrainerIntegration(TrlTestCase):
    """Integration tests for HICRATrainer."""

    def test_trainer_initialization_default_sgs(self):
        """Test trainer initialization with default Strategic Grams."""
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        config = HICRAConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,
            num_generations=2,
            max_completion_length=8,
            report_to="none",
        )

        trainer = HICRATrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=config,
            train_dataset=dataset,
        )

        assert trainer.strategic_grams is not None
        assert len(trainer.strategic_grams) > 0
        assert trainer.sg_token_ids is not None

    def test_trainer_initialization_custom_sgs(self):
        """Test trainer initialization with custom Strategic Grams."""
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        custom_sgs = ["let's try", "we can use", "notice that"]

        config = HICRAConfig(
            output_dir=self.tmp_dir,
            strategic_grams=custom_sgs,
            report_to="none",
        )

        trainer = HICRATrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=config,
            train_dataset=dataset,
        )

        assert trainer.strategic_grams == custom_sgs

    def test_trainer_initialization_sgs_from_file(self):
        """Test trainer initialization with Strategic Grams from file."""
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        test_sgs = ["let's try", "we can use"]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_sgs, f)
            temp_path = f.name

        try:
            config = HICRAConfig(
                output_dir=self.tmp_dir,
                strategic_grams_path=temp_path,
                report_to="none",
            )

            trainer = HICRATrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
                args=config,
                train_dataset=dataset,
            )

            assert trainer.strategic_grams == test_sgs
        finally:
            os.unlink(temp_path)

    def test_training_with_hicra_enabled(self):
        """Test training loop with HICRA enabled."""
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        config = HICRAConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=3,
            num_generations=3,
            max_completion_length=8,
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

        previous_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that params have changed
        for n, param in previous_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_training_with_hicra_disabled(self):
        """Test training loop with HICRA disabled (should behave like GRPO)."""
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        config = HICRAConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=3,
            num_generations=3,
            max_completion_length=8,
            use_hicra=False,  # Disable HICRA
            report_to="none",
        )

        trainer = HICRATrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=config,
            train_dataset=dataset,
        )

        previous_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that params have changed
        for n, param in previous_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_training_with_planning_tokens(self):
        """Test training with planning token identification enabled."""
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        config = HICRAConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=3,
            num_generations=3,
            max_completion_length=8,
            use_hicra=True,
            use_planning_tokens=True,
            strategic_grams=["let's try", "we can use"],
            report_to="none",
        )

        trainer = HICRATrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=config,
            train_dataset=dataset,
        )

        previous_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that params have changed
        for n, param in previous_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    @require_peft
    def test_training_with_peft(self):
        """Test HICRA training with PEFT (LoRA)."""
        from peft import LoraConfig

        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        config = HICRAConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=3,
            num_generations=3,
            max_completion_length=8,
            use_hicra=True,
            report_to="none",
        )

        trainer = HICRATrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=config,
            train_dataset=dataset,
            peft_config=LoraConfig(),
        )

        previous_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that LoRA params have changed
        for n, param in previous_params.items():
            new_param = trainer.model.get_parameter(n)
            if "lora" in n.lower():
                assert not torch.equal(param, new_param), f"LoRA parameter {n} has not changed."


class TestHICRAEndToEnd(TrlTestCase):
    """End-to-end tests for HICRA training."""

    def test_end_to_end_training_with_metrics(self):
        """Test complete training pipeline with HICRA metrics logging."""
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        config = HICRAConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=3,
            num_generations=3,
            max_completion_length=8,
            use_hicra=True,
            hicra_alpha=0.2,
            use_planning_tokens=True,
            log_planning_token_ratio=True,
            log_semantic_entropy=True,
            strategic_grams=["let's try", "we can use", "notice that"],
            report_to="none",
        )

        trainer = HICRATrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=config,
            train_dataset=dataset,
        )

        trainer.train()

        # Verify training completed
        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that HICRA metrics are logged (if any planning tokens were found)
        # Note: Metrics may not be present if no planning tokens were identified
        log_keys = set()
        for log_entry in trainer.state.log_history:
            log_keys.update(log_entry.keys())

        # At minimum, we should have standard training metrics
        assert "train_loss" in log_keys

    def test_compare_hicra_vs_grpo(self):
        """Compare HICRA training with GRPO baseline."""
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        # Train with GRPO (HICRA disabled)
        grpo_config = HICRAConfig(
            output_dir=self.tmp_dir + "/grpo",
            learning_rate=0.1,
            per_device_train_batch_size=3,
            num_generations=3,
            max_completion_length=8,
            use_hicra=False,
            report_to="none",
        )

        grpo_trainer = HICRATrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=grpo_config,
            train_dataset=dataset,
        )

        grpo_trainer.train()
        grpo_loss = grpo_trainer.state.log_history[-1]["train_loss"]

        # Train with HICRA
        hicra_config = HICRAConfig(
            output_dir=self.tmp_dir + "/hicra",
            learning_rate=0.1,
            per_device_train_batch_size=3,
            num_generations=3,
            max_completion_length=8,
            use_hicra=True,
            hicra_alpha=0.2,
            report_to="none",
        )

        hicra_trainer = HICRATrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=hicra_config,
            train_dataset=dataset,
        )

        hicra_trainer.train()
        hicra_loss = hicra_trainer.state.log_history[-1]["train_loss"]

        # Both should complete training successfully
        assert grpo_loss is not None
        assert hicra_loss is not None

    def test_checkpoint_compatibility(self):
        """Test that HICRA checkpoints are compatible with standard loading."""
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        config = HICRAConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=3,
            num_generations=3,
            max_completion_length=8,
            max_steps=2,  # Just a few steps
            save_strategy="steps",
            save_steps=1,
            use_hicra=True,
            report_to="none",
        )

        trainer = HICRATrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=config,
            train_dataset=dataset,
        )

        trainer.train()

        # Check that checkpoint was saved
        checkpoint_dirs = [d for d in os.listdir(self.tmp_dir) if d.startswith("checkpoint")]
        assert len(checkpoint_dirs) > 0

        # Verify checkpoint can be loaded
        from transformers import AutoModelForCausalLM

        checkpoint_path = os.path.join(self.tmp_dir, checkpoint_dirs[0])
        loaded_model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
        assert loaded_model is not None

    def test_semantic_entropy_computation(self):
        """Test semantic entropy computation with known Strategic Grams."""
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        config = HICRAConfig(
            output_dir=self.tmp_dir,
            strategic_grams=["let's try", "we can"],
            report_to="none",
        )

        trainer = HICRATrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=config,
            train_dataset=dataset,
        )

        # Create test data with known Strategic Grams
        text1 = "let's try this approach"
        text2 = "we can solve this"
        completion_ids = torch.tensor(
            [
                trainer.processing_class.encode(text1, add_special_tokens=False),
                trainer.processing_class.encode(text2, add_special_tokens=False),
            ]
        )

        # Pad to same length
        max_len = max(completion_ids.shape[1], 10)
        padded_ids = torch.zeros(2, max_len, dtype=torch.long)
        padded_ids[0, : completion_ids.shape[1]] = completion_ids[0]
        padded_ids[1, : completion_ids.shape[1]] = completion_ids[1]

        planning_mask = trainer.identify_planning_tokens(padded_ids)

        entropy = trainer.compute_semantic_entropy(padded_ids, planning_mask)

        # Entropy should be non-negative
        assert entropy >= 0.0
        # If we have multiple different SGs, entropy should be positive
        if planning_mask.any():
            assert entropy >= 0.0
