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

import gc

import pytest
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from transformers.utils import is_peft_available

from trl.experimental.dppo import DPPOConfig, DPPOTrainer

from ..testing_utils import TrlTestCase, require_peft


if is_peft_available():
    from peft import LoraConfig


class TestDPPOConfig(TrlTestCase):
    """Test DPPO configuration validation and defaults."""

    def test_default_clip_ratios(self):
        """clip_ratio_low and clip_ratio_high should default to cliprange."""
        config = DPPOConfig(
            output_dir=self.tmp_dir,
            cliprange=0.2,
        )
        assert config.clip_ratio_low == 0.2
        assert config.clip_ratio_high == 0.2

    def test_custom_clip_ratios(self):
        """Asymmetric clip ratios should be respected when set explicitly."""
        config = DPPOConfig(
            output_dir=self.tmp_dir,
            clip_ratio_low=0.1,
            clip_ratio_high=0.3,
        )
        assert config.clip_ratio_low == 0.1
        assert config.clip_ratio_high == 0.3

    def test_default_loss_mode(self):
        """Default loss_mode should be dppo_binary_tv."""
        config = DPPOConfig(output_dir=self.tmp_dir)
        assert config.loss_mode == "dppo_binary_tv"

    def test_loss_mode_kl(self):
        """dppo_binary_kl loss mode should be accepted."""
        config = DPPOConfig(output_dir=self.tmp_dir, loss_mode="dppo_binary_kl")
        assert config.loss_mode == "dppo_binary_kl"

    def test_loss_mode_kl_recompute(self):
        """dppo_binary_kl_recompute loss mode should be accepted."""
        config = DPPOConfig(output_dir=self.tmp_dir, loss_mode="dppo_binary_kl_recompute")
        assert config.loss_mode == "dppo_binary_kl_recompute"

    def test_clip_ratio_c_default(self):
        """clip_ratio_c should default to 10.0."""
        config = DPPOConfig(output_dir=self.tmp_dir)
        assert config.clip_ratio_c == 10.0

    def test_dppo_specific_params(self):
        """All DPPO-specific parameters can be set."""
        config = DPPOConfig(
            output_dir=self.tmp_dir,
            loss_mode="dppo_binary_kl",
            clip_ratio_low=0.1,
            clip_ratio_high=0.3,
            clip_ratio_c=5.0,
            num_ppo_epochs=4,
        )
        assert config.loss_mode == "dppo_binary_kl"
        assert config.clip_ratio_low == 0.1
        assert config.clip_ratio_high == 0.3
        assert config.clip_ratio_c == 5.0
        assert config.num_ppo_epochs == 4

    def test_dppo_config_inheritance(self):
        """DPPOConfig should inherit from TrainingArguments."""
        config = DPPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=3e-6,
            num_ppo_epochs=4,
        )
        assert hasattr(config, "learning_rate")
        assert hasattr(config, "output_dir")
        assert hasattr(config, "loss_mode")
        assert hasattr(config, "clip_ratio_low")
        assert hasattr(config, "clip_ratio_high")
        assert hasattr(config, "clip_ratio_c")
        assert hasattr(config, "num_ppo_epochs")


class TestDPPOTrainer(TrlTestCase):
    """Test DPPO trainer functionality."""

    def setup_method(self):
        self.model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, dtype="float32")
        self.ref_model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, padding_side="left")
        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        reward_model_id = "trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5"
        self.value_model = AutoModelForSequenceClassification.from_pretrained(reward_model_id, num_labels=1)
        self.reward_model = AutoModelForSequenceClassification.from_pretrained(reward_model_id, num_labels=1)

        raw_dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only")

        def tokenize(example, tokenizer):
            tokenized = tokenizer(text=example["prompt"])
            if tokenizer.eos_token_id is not None and tokenized["input_ids"][-1] != tokenizer.eos_token_id:
                tokenized["input_ids"] = tokenized["input_ids"] + [tokenizer.eos_token_id]
                tokenized["attention_mask"] = tokenized["attention_mask"] + [1]
            return tokenized

        self.raw_dataset = raw_dataset.map(tokenize, fn_kwargs={"tokenizer": self.tokenizer}, remove_columns="prompt")

    def teardown_method(self):
        gc.collect()

    def _make_trainer(self, **config_kwargs):
        """Helper to build a DPPOTrainer with minimal config for unit tests."""
        training_args = DPPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,
            report_to="none",
            **config_kwargs,
        )
        return DPPOTrainer(
            args=training_args,
            processing_class=self.tokenizer,
            model=self.model,
            ref_model=self.ref_model,
            reward_model=self.reward_model,
            value_model=self.value_model,
            train_dataset=self.raw_dataset["train"],
        )

    # ------------------------------------------------------------------
    # Trainer creation
    # ------------------------------------------------------------------

    def test_dppo_trainer_creation(self):
        """DPPOTrainer should be created without errors."""
        trainer = self._make_trainer()
        assert trainer is not None

    def test_dppo_trainer_tags(self):
        """DPPO trainer should expose correct tag names."""
        trainer = self._make_trainer()
        assert hasattr(trainer, "_tag_names")
        assert "trl" in trainer._tag_names
        assert "dppo" in trainer._tag_names

    def test_dppo_has_single_optimizer(self):
        """DPPO uses a single joint optimizer (policy + value), not a separate one."""
        trainer = self._make_trainer()
        assert trainer.optimizer is not None
        # DPPO does NOT create a separate value optimizer
        assert not hasattr(trainer, "vf_optimizer")

    # ------------------------------------------------------------------
    # Divergence mask unit tests
    # ------------------------------------------------------------------

    def test_dppo_mask_binary_tv_all_inside(self):
        """All tokens should be valid when divergence is below threshold."""
        trainer = self._make_trainer(loss_mode="dppo_binary_tv", cliprange=0.2)
        # Set up nearly identical probabilities
        new_lp = torch.log(torch.tensor([[0.5, 0.3, 0.2]]))
        rollout_lp = torch.log(torch.tensor([[0.5, 0.3, 0.2]]))
        advantages = torch.tensor([[1.0, -1.0, 1.0]])
        response_mask = torch.ones(1, 3, dtype=torch.bool)
        token_mask, invalid_mask = trainer._compute_dppo_mask(new_lp, rollout_lp, rollout_lp, advantages, response_mask)
        assert token_mask.all(), "All tokens should be valid when divergence is 0"

    def test_dppo_mask_binary_tv_positive_exceeds_threshold(self):
        """Tokens with positive advantage whose prob moved up beyond threshold are masked."""
        trainer = self._make_trainer(loss_mode="dppo_binary_tv", cliprange=0.1)
        # prob_current = 0.8, prob_rollout = 0.5 → diff = 0.3 > 0.1
        new_lp = torch.log(torch.tensor([[0.8]]))
        rollout_lp = torch.log(torch.tensor([[0.5]]))
        advantages = torch.tensor([[1.0]])  # positive advantage
        response_mask = torch.ones(1, 1, dtype=torch.bool)
        token_mask, invalid_mask = trainer._compute_dppo_mask(new_lp, rollout_lp, rollout_lp, advantages, response_mask)
        assert not token_mask.any(), "Token should be masked (prob increased too much for positive advantage)"

    def test_dppo_mask_binary_tv_negative_inside_threshold(self):
        """Token with negative advantage whose prob moved down within threshold stays valid."""
        trainer = self._make_trainer(loss_mode="dppo_binary_tv", cliprange=0.2)
        # prob_current = 0.4, prob_rollout = 0.5 → diff = -0.1, threshold = 0.2 → inside
        new_lp = torch.log(torch.tensor([[0.4]]))
        rollout_lp = torch.log(torch.tensor([[0.5]]))
        advantages = torch.tensor([[-1.0]])  # negative advantage
        response_mask = torch.ones(1, 1, dtype=torch.bool)
        token_mask, invalid_mask = trainer._compute_dppo_mask(new_lp, rollout_lp, rollout_lp, advantages, response_mask)
        assert token_mask.all(), "Token should be valid (prob decreased within threshold for negative advantage)"

    def test_dppo_mask_binary_tv_padding_zeroed(self):
        """Padding tokens should always be zeroed regardless of divergence."""
        trainer = self._make_trainer(loss_mode="dppo_binary_tv", cliprange=1.0)  # very wide threshold
        new_lp = torch.log(torch.tensor([[0.5, 0.5]]))
        rollout_lp = torch.log(torch.tensor([[0.5, 0.5]]))
        advantages = torch.tensor([[1.0, 1.0]])
        # Second token is padding
        response_mask = torch.tensor([[True, False]])
        token_mask, invalid_mask = trainer._compute_dppo_mask(new_lp, rollout_lp, rollout_lp, advantages, response_mask)
        assert token_mask[0, 0] == 1.0, "Valid token should be unmasked"
        assert token_mask[0, 1] == 0.0, "Padding token should be zeroed"

    def test_dppo_mask_binary_kl_all_inside(self):
        """dppo_binary_kl: identical distributions give zero KL, all tokens valid."""
        trainer = self._make_trainer(loss_mode="dppo_binary_kl", cliprange=0.01)
        lp = torch.log(torch.tensor([[0.5, 0.3, 0.2]]))
        advantages = torch.tensor([[1.0, -1.0, 1.0]])
        response_mask = torch.ones(1, 3, dtype=torch.bool)
        token_mask, _ = trainer._compute_dppo_mask(lp, lp, lp, advantages, response_mask)
        assert token_mask.all()

    def test_dppo_mask_binary_kl_recompute_uses_old_logprobs(self):
        """dppo_binary_kl_recompute should anchor to old_logprobs, not rollout_logprobs."""
        trainer = self._make_trainer(loss_mode="dppo_binary_kl_recompute", cliprange=0.05)
        # Make new_logprobs close to old_logprobs (inside trust region) but far from rollout
        old_lp = torch.log(torch.tensor([[0.5]]))
        new_lp = torch.log(torch.tensor([[0.49]]))   # very close to old → inside trust region
        rollout_lp = torch.log(torch.tensor([[0.1]]))  # far away → would trigger if anchored to rollout
        advantages = torch.tensor([[1.0]])
        response_mask = torch.ones(1, 1, dtype=torch.bool)
        token_mask, _ = trainer._compute_dppo_mask(new_lp, rollout_lp, old_lp, advantages, response_mask)
        assert token_mask.all(), "Should be valid when anchored to old_logprobs (not rollout_logprobs)"

    def test_dppo_mask_invalid_loss_mode(self):
        """Unknown loss_mode should raise ValueError."""
        trainer = self._make_trainer()
        trainer.args.loss_mode = "invalid_mode"
        lp = torch.log(torch.tensor([[0.5]]))
        adv = torch.tensor([[1.0]])
        mask = torch.ones(1, 1, dtype=torch.bool)
        with pytest.raises(ValueError, match="Unknown loss_mode"):
            trainer._compute_dppo_mask(lp, lp, lp, adv, mask)

    # ------------------------------------------------------------------
    # End-to-end training
    # ------------------------------------------------------------------

    def test_basic_training_tv(self):
        """DPPO-Binary-TV: both policy and value weights should update."""
        initial_value_weights = {n: p.clone().detach() for n, p in self.value_model.named_parameters()}
        initial_policy_weights = {n: p.clone().detach() for n, p in self.model.named_parameters()}

        training_args = DPPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=4,
            loss_mode="dppo_binary_tv",
            num_ppo_epochs=2,
            report_to="none",
        )
        trainer = DPPOTrainer(
            args=training_args,
            processing_class=self.tokenizer,
            model=self.model,
            ref_model=self.ref_model,
            reward_model=self.reward_model,
            value_model=self.value_model,
            train_dataset=self.raw_dataset["train"],
            eval_dataset=self.raw_dataset["test"],
        )
        trainer.train()

        value_updated = any(
            not torch.allclose(initial_value_weights[n], p.cpu(), atol=1e-6)
            for n, p in trainer.model.value_model.named_parameters()
        )
        policy_updated = any(
            not torch.allclose(initial_policy_weights[n], p.cpu(), atol=1e-6)
            for n, p in trainer.model.policy.named_parameters()
        )
        assert value_updated, "Value function weights were not updated"
        assert policy_updated, "Policy weights were not updated"

    def test_basic_training_kl(self):
        """DPPO-Binary-KL: training should complete without errors."""
        training_args = DPPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=4,
            loss_mode="dppo_binary_kl",
            num_ppo_epochs=2,
            num_sample_generations=0,
            report_to="none",
        )
        trainer = DPPOTrainer(
            args=training_args,
            processing_class=self.tokenizer,
            model=self.model,
            ref_model=self.ref_model,
            reward_model=self.reward_model,
            value_model=self.value_model,
            train_dataset=self.raw_dataset["train"],
        )
        trainer.train()  # should not raise

    def test_basic_training_kl_recompute(self):
        """DPPO-Binary-KL-Recompute: training should complete without errors."""
        training_args = DPPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=4,
            loss_mode="dppo_binary_kl_recompute",
            num_ppo_epochs=2,
            num_sample_generations=0,
            report_to="none",
        )
        trainer = DPPOTrainer(
            args=training_args,
            processing_class=self.tokenizer,
            model=self.model,
            ref_model=self.ref_model,
            reward_model=self.reward_model,
            value_model=self.value_model,
            train_dataset=self.raw_dataset["train"],
        )
        trainer.train()  # should not raise

    def test_training_without_ref_model(self):
        """DPPO should train when ref_model=None (uses null-ref context)."""
        training_args = DPPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=4,
            num_ppo_epochs=2,
            report_to="none",
        )
        trainer = DPPOTrainer(
            args=training_args,
            processing_class=self.tokenizer,
            model=self.model,
            ref_model=None,
            reward_model=self.reward_model,
            value_model=self.value_model,
            train_dataset=self.raw_dataset["train"],
        )
        trainer.train()

    def test_training_logs_maskfrac(self):
        """Training logs should include policy/maskfrac_avg (DPPO-specific metric)."""
        training_args = DPPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=4,
            num_ppo_epochs=2,
            logging_steps=1,
            num_sample_generations=0,
            report_to="none",
        )
        trainer = DPPOTrainer(
            args=training_args,
            processing_class=self.tokenizer,
            model=self.model,
            ref_model=self.ref_model,
            reward_model=self.reward_model,
            value_model=self.value_model,
            train_dataset=self.raw_dataset["train"],
        )
        trainer.train()

        has_maskfrac = any("maskfrac" in entry for entry in trainer.state.log_history)
        assert has_maskfrac, "Training logs should include 'policy/maskfrac_avg'"

    def test_asymmetric_clip_ratios(self):
        """Asymmetric clip_ratio_low / clip_ratio_high should be accepted and applied."""
        training_args = DPPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=4,
            loss_mode="dppo_binary_tv",
            clip_ratio_low=0.05,   # strict for negative advantages
            clip_ratio_high=0.3,   # lenient for positive advantages
            num_ppo_epochs=2,
            num_sample_generations=0,
            report_to="none",
        )
        trainer = DPPOTrainer(
            args=training_args,
            processing_class=self.tokenizer,
            model=self.model,
            ref_model=self.ref_model,
            reward_model=self.reward_model,
            value_model=self.value_model,
            train_dataset=self.raw_dataset["train"],
        )
        assert trainer.args.clip_ratio_low == 0.05
        assert trainer.args.clip_ratio_high == 0.3
        trainer.train()

    @require_peft
    def test_peft_training(self):
        """DPPO training with PEFT (LoRA) should update LoRA weights."""
        initial_value_weights = {n: p.clone().detach() for n, p in self.value_model.named_parameters()}

        training_args = DPPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=4,
            num_ppo_epochs=2,
            report_to="none",
        )
        peft_config = LoraConfig(
            r=32,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        trainer = DPPOTrainer(
            args=training_args,
            processing_class=self.tokenizer,
            model=self.model,
            ref_model=None,
            reward_model=self.reward_model,
            value_model=self.value_model,
            train_dataset=self.raw_dataset["train"],
            eval_dataset=self.raw_dataset["test"],
            peft_config=peft_config,
        )
        trainer.train()

        value_updated = any(
            n in initial_value_weights
            and not torch.allclose(initial_value_weights[n], p.cpu(), atol=1e-6)
            for n, p in trainer.model.value_model.named_parameters()
        )
        policy_lora_nonzero = any(
            "lora" in n.lower() and p.requires_grad and not torch.allclose(p, torch.zeros_like(p), atol=1e-6)
            for n, p in trainer.model.policy.named_parameters()
        )
        assert value_updated, "Value function weights were not updated with PEFT"
        assert policy_lora_nonzero, "Policy LoRA weights were not updated with PEFT"


class TestDPPOTrainerIntegration(TrlTestCase):
    """Integration tests for DPPO with various configurations."""

    def setup_method(self):
        self.model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, dtype="float32")
        self.ref_model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, padding_side="left")
        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        reward_model_id = "trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5"
        self.value_model = AutoModelForSequenceClassification.from_pretrained(reward_model_id, num_labels=1)
        self.reward_model = AutoModelForSequenceClassification.from_pretrained(reward_model_id, num_labels=1)

        raw_dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only")

        def tokenize(example, tokenizer):
            tokenized = tokenizer(text=example["prompt"])
            if tokenizer.eos_token_id is not None and tokenized["input_ids"][-1] != tokenizer.eos_token_id:
                tokenized["input_ids"] = tokenized["input_ids"] + [tokenizer.eos_token_id]
                tokenized["attention_mask"] = tokenized["attention_mask"] + [1]
            return tokenized

        self.raw_dataset = raw_dataset.map(tokenize, fn_kwargs={"tokenizer": self.tokenizer}, remove_columns="prompt")

    def teardown_method(self):
        gc.collect()

    def test_all_loss_modes_run(self):
        """All three DPPO loss modes should complete training without errors."""
        for loss_mode in ("dppo_binary_tv", "dppo_binary_kl", "dppo_binary_kl_recompute"):
            model = AutoModelForCausalLM.from_pretrained(self.model_id, dtype="float32")
            training_args = DPPOConfig(
                output_dir=self.tmp_dir,
                per_device_train_batch_size=4,
                loss_mode=loss_mode,
                num_ppo_epochs=2,
                num_sample_generations=0,
                report_to="none",
            )
            trainer = DPPOTrainer(
                args=training_args,
                processing_class=self.tokenizer,
                model=model,
                ref_model=self.ref_model,
                reward_model=self.reward_model,
                value_model=self.value_model,
                train_dataset=self.raw_dataset["train"],
            )
            trainer.train()
            gc.collect()

    def test_tight_threshold_masks_most_tokens(self):
        """A very tight threshold should result in a high mask fraction."""
        training_args = DPPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=4,
            loss_mode="dppo_binary_tv",
            cliprange=1e-6,  # nearly all tokens will be outside trust region
            num_ppo_epochs=1,
            num_sample_generations=0,
            logging_steps=1,
            report_to="none",
        )
        trainer = DPPOTrainer(
            args=training_args,
            processing_class=self.tokenizer,
            model=self.model,
            ref_model=self.ref_model,
            reward_model=self.reward_model,
            value_model=self.value_model,
            train_dataset=self.raw_dataset["train"],
        )
        trainer.train()

        maskfrac_values = [
            entry.get("policy/maskfrac_avg", None)
            for entry in trainer.state.log_history
            if "policy/maskfrac_avg" in entry
        ]
        assert len(maskfrac_values) > 0
        # With a near-zero threshold, most tokens should be masked
        assert max(maskfrac_values) > 0.5, "Expected high mask fraction with very tight threshold"

    def test_wide_threshold_masks_few_tokens(self):
        """A very wide threshold should result in a low mask fraction."""
        training_args = DPPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=4,
            loss_mode="dppo_binary_tv",
            cliprange=1.0,  # threshold = 1 means almost nothing is masked (prob diff ∈ [0, 1])
            num_ppo_epochs=1,
            num_sample_generations=0,
            logging_steps=1,
            report_to="none",
        )
        trainer = DPPOTrainer(
            args=training_args,
            processing_class=self.tokenizer,
            model=self.model,
            ref_model=self.ref_model,
            reward_model=self.reward_model,
            value_model=self.value_model,
            train_dataset=self.raw_dataset["train"],
        )
        trainer.train()

        maskfrac_values = [
            entry.get("policy/maskfrac_avg", None)
            for entry in trainer.state.log_history
            if "policy/maskfrac_avg" in entry
        ]
        assert len(maskfrac_values) > 0
        # With threshold = 1 (max possible prob diff), few tokens should be masked
        assert min(maskfrac_values) < 0.5, "Expected low mask fraction with very wide threshold"
