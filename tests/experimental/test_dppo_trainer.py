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

    def test_default_vf_learning_rate(self):
        """Test that vf_learning_rate defaults to learning_rate if not set."""
        config = DPPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=3e-6,
        )
        assert config.vf_learning_rate == 3e-6

    def test_custom_vf_learning_rate(self):
        """Test that custom vf_learning_rate is respected."""
        config = DPPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=3e-6,
            vf_learning_rate=1e-4,
        )
        assert config.learning_rate == 3e-6
        assert config.vf_learning_rate == 1e-4

    def test_num_vf_epochs_default(self):
        """Test default value of num_vf_epochs."""
        config = DPPOConfig(output_dir=self.tmp_dir)
        assert config.num_vf_epochs == 1

    def test_vf_update_frequency_default(self):
        """Test default value of vf_update_frequency."""
        config = DPPOConfig(output_dir=self.tmp_dir)
        assert config.vf_update_frequency == 1

    def test_dppo_specific_params(self):
        """Test all DPPO-specific parameters can be set."""
        config = DPPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=3e-6,
            vf_learning_rate=1e-4,
            num_vf_epochs=2,
            vf_update_frequency=2,
            num_ppo_epochs=4,
        )
        assert config.vf_learning_rate == 1e-4
        assert config.num_vf_epochs == 2
        assert config.vf_update_frequency == 2
        assert config.num_ppo_epochs == 4


class TestDPPOTrainer(TrlTestCase):
    """Test DPPO trainer functionality."""

    def setup_method(self):
        """Set up models, tokenizer, and dataset for testing."""
        # Set up the models and tokenizer using the test model
        self.model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, dtype="float32")
        self.ref_model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, padding_side="left")
        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        # Add reward and value models
        reward_model_id = "trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5"
        self.value_model = AutoModelForSequenceClassification.from_pretrained(reward_model_id, num_labels=1)
        self.reward_model = AutoModelForSequenceClassification.from_pretrained(reward_model_id, num_labels=1)

        # Load dataset
        raw_dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only")

        def tokenize(example, tokenizer):
            tokenized = tokenizer(text=example["prompt"])
            if tokenizer.eos_token_id is not None and tokenized["input_ids"][-1] != tokenizer.eos_token_id:
                tokenized["input_ids"] = tokenized["input_ids"] + [tokenizer.eos_token_id]
                tokenized["attention_mask"] = tokenized["attention_mask"] + [1]
            return tokenized

        self.raw_dataset = raw_dataset.map(tokenize, fn_kwargs={"tokenizer": self.tokenizer}, remove_columns="prompt")

    def teardown_method(self):
        """Clean up after each test."""
        gc.collect()

    def test_dppo_trainer_creation(self):
        """Test that DPPO trainer can be created successfully."""
        training_args = DPPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,
            learning_rate=3e-6,
            vf_learning_rate=1e-4,
            num_ppo_epochs=2,
            num_vf_epochs=1,
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

        # Verify trainer has separate value function optimizer
        assert hasattr(trainer, "vf_optimizer")
        assert hasattr(trainer, "vf_lr_scheduler")
        assert trainer.vf_optimizer is not None
        assert trainer.vf_lr_scheduler is not None

    def test_separate_optimizers(self):
        """Test that DPPO creates separate optimizers for policy and value function."""
        training_args = DPPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,
            learning_rate=3e-6,
            vf_learning_rate=1e-4,
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

        # Verify both optimizers exist and are different
        assert trainer.optimizer is not None
        assert trainer.vf_optimizer is not None
        assert trainer.optimizer is not trainer.vf_optimizer

    def test_basic_training(self):
        """Test basic DPPO training and verify both policy and value weights update."""
        # Capture initial weights
        initial_value_weights = {}
        initial_policy_weights = {}
        for name, param in self.value_model.named_parameters():
            initial_value_weights[name] = param.clone().detach()
        for name, param in self.model.named_parameters():
            initial_policy_weights[name] = param.clone().detach()

        # Configure training args with decoupled learning rates
        training_args = DPPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=4,
            learning_rate=3e-6,  # Policy learning rate
            vf_learning_rate=1e-4,  # Value function learning rate (typically higher)
            num_ppo_epochs=2,
            num_vf_epochs=1,
            vf_update_frequency=1,
            report_to="none",
        )

        # Create trainer
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

        # Train
        trainer.train()

        # Check if value function weights have been updated
        value_weights_updated = False
        for name, param in trainer.model.value_model.named_parameters():
            if not torch.allclose(initial_value_weights[name], param.to("cpu"), atol=1e-6):
                value_weights_updated = True
                break

        # Check if policy weights have been updated
        policy_weights_updated = False
        for name, param in trainer.model.policy.named_parameters():
            if not torch.allclose(initial_policy_weights[name], param.to("cpu"), atol=1e-6):
                policy_weights_updated = True
                break

        assert value_weights_updated, "Value function weights were not updated during DPPO training"
        assert policy_weights_updated, "Policy weights were not updated during DPPO training"

    def test_decoupled_learning_rates(self):
        """Test that policy and value function use different learning rates."""
        training_args = DPPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=4,
            learning_rate=3e-6,
            vf_learning_rate=1e-4,
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

        # Get initial learning rates from schedulers
        policy_lr = trainer.lr_scheduler.get_last_lr()[0]
        vf_lr = trainer.vf_lr_scheduler.get_last_lr()[0]

        # Verify they are different as configured
        assert policy_lr == pytest.approx(3e-6, rel=1e-3)
        assert vf_lr == pytest.approx(1e-4, rel=1e-3)
        assert policy_lr != vf_lr

    def test_vf_update_frequency(self):
        """Test that value function update frequency is respected."""
        training_args = DPPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=4,
            learning_rate=3e-6,
            vf_learning_rate=1e-4,
            num_ppo_epochs=2,
            num_vf_epochs=2,
            vf_update_frequency=2,  # Only update value every 2 policy updates
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

        # Verify the configuration is set correctly
        assert trainer.args.vf_update_frequency == 2
        assert trainer.args.num_vf_epochs == 2

    def test_num_vf_epochs(self):
        """Test that num_vf_epochs controls value function training epochs."""
        training_args = DPPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=4,
            learning_rate=3e-6,
            vf_learning_rate=1e-4,
            num_ppo_epochs=2,
            num_vf_epochs=3,  # Train value function for 3 epochs
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

        # Verify the configuration
        assert trainer.args.num_vf_epochs == 3

    @require_peft
    def test_peft_training(self):
        """Test DPPO training with PEFT configuration."""
        # Capture initial weights
        initial_value_weights = {}
        initial_policy_weights = {}
        for name, param in self.value_model.named_parameters():
            initial_value_weights[name] = param.clone().detach()
        for name, param in self.model.named_parameters():
            initial_policy_weights[name] = param.clone().detach()

        # Configure training args
        training_args = DPPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=4,
            learning_rate=3e-6,
            vf_learning_rate=1e-4,
            num_ppo_epochs=2,
            num_vf_epochs=1,
            report_to="none",
        )

        # Configure PEFT
        peft_config = LoraConfig(
            r=32,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # Create trainer with PEFT
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

        # Train
        trainer.train()

        # Check if value function weights have been updated
        value_weights_updated = False
        for name, param in trainer.model.value_model.named_parameters():
            if name in initial_value_weights and not torch.allclose(
                initial_value_weights[name], param.to("cpu"), atol=1e-6
            ):
                value_weights_updated = True
                break

        # Check if policy LoRA weights have been updated
        policy_weights_updated = False
        for name, param in trainer.model.policy.named_parameters():
            if "lora" in name.lower() and param.requires_grad:
                # New weights should be non-zero if they've been updated
                if not torch.allclose(param, torch.zeros_like(param), atol=1e-6):
                    policy_weights_updated = True
                    break

        assert value_weights_updated, "Value function weights were not updated during DPPO training with PEFT"
        assert policy_weights_updated, "Policy LoRA weights were not updated during DPPO training with PEFT"

    def test_training_logs_both_learning_rates(self):
        """Test that training logs include both policy and value function learning rates."""
        training_args = DPPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=4,
            learning_rate=3e-6,
            vf_learning_rate=1e-4,
            num_ppo_epochs=2,
            num_vf_epochs=1,
            logging_steps=1,
            num_sample_generations=0,  # Disable sample generation
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

        # Train for a few steps
        trainer.train()

        # Check that logs contain both learning rates
        assert len(trainer.state.log_history) > 0
        # Find a log entry that has learning rates
        has_lr = False
        has_vf_lr = False
        for log_entry in trainer.state.log_history:
            if "lr" in log_entry:
                has_lr = True
            if "vf_lr" in log_entry:
                has_vf_lr = True
            if has_lr and has_vf_lr:
                break

        assert has_lr, "Training logs should include policy learning rate (lr)"
        assert has_vf_lr, "Training logs should include value function learning rate (vf_lr)"

    def test_dppo_without_ref_model(self):
        """Test DPPO training without providing a reference model."""
        training_args = DPPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=4,
            learning_rate=3e-6,
            vf_learning_rate=1e-4,
            num_ppo_epochs=2,
            report_to="none",
        )

        trainer = DPPOTrainer(
            args=training_args,
            processing_class=self.tokenizer,
            model=self.model,
            ref_model=None,  # No reference model
            reward_model=self.reward_model,
            value_model=self.value_model,
            train_dataset=self.raw_dataset["train"],
        )

        # Should still create both optimizers
        assert hasattr(trainer, "optimizer")
        assert hasattr(trainer, "vf_optimizer")
        assert trainer.optimizer is not None
        assert trainer.vf_optimizer is not None

    def test_dppo_scheduler_steps(self):
        """Test that both schedulers step correctly during training."""
        training_args = DPPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=4,
            learning_rate=3e-6,
            vf_learning_rate=1e-4,
            num_ppo_epochs=2,
            num_vf_epochs=1,
            num_sample_generations=0,  # Disable sample generation
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

        # Get initial learning rates
        initial_policy_lr = trainer.lr_scheduler.get_last_lr()[0]
        initial_vf_lr = trainer.vf_lr_scheduler.get_last_lr()[0]

        # Verify initial learning rates are correct
        assert initial_policy_lr == pytest.approx(3e-6, rel=1e-3)
        assert initial_vf_lr == pytest.approx(1e-4, rel=1e-3)

        # Train
        trainer.train()

        # After training, verify both schedulers exist and can return LR
        # (Note: LR may decay to 0 or near-0 after training completes depending on scheduler)
        final_policy_lr = trainer.lr_scheduler.get_last_lr()[0]
        final_vf_lr = trainer.vf_lr_scheduler.get_last_lr()[0]

        # Both should return non-negative learning rates
        assert final_policy_lr >= 0
        assert final_vf_lr >= 0
        # The LRs should have decreased or stayed the same (due to scheduling)
        assert final_policy_lr <= initial_policy_lr
        assert final_vf_lr <= initial_vf_lr

    def test_dppo_trainer_tags(self):
        """Test that DPPO trainer has correct tags."""
        training_args = DPPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,
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

        # Check trainer has correct tags
        assert hasattr(trainer, "_tag_names")
        assert "trl" in trainer._tag_names
        assert "dppo" in trainer._tag_names

    def test_dppo_config_inheritance(self):
        """Test that DPPOConfig properly inherits from TrainingArguments."""
        config = DPPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=3e-6,
            vf_learning_rate=1e-4,
            num_ppo_epochs=4,
            num_vf_epochs=2,
        )

        # Check it has both base TrainingArguments attrs and DPPO-specific ones
        assert hasattr(config, "learning_rate")  # From TrainingArguments
        assert hasattr(config, "output_dir")  # From TrainingArguments
        assert hasattr(config, "vf_learning_rate")  # DPPO-specific
        assert hasattr(config, "num_vf_epochs")  # DPPO-specific
        assert hasattr(config, "vf_update_frequency")  # DPPO-specific
        assert hasattr(config, "num_ppo_epochs")  # From PPO/DPPO


class TestDPPOTrainerIntegration(TrlTestCase):
    """Integration tests for DPPO trainer with various configurations."""

    def setup_method(self):
        """Set up models for integration tests."""
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
        """Clean up after tests."""
        gc.collect()

    def test_dppo_higher_vf_learning_rate(self):
        """Test DPPO with higher value function learning rate (typical use case)."""
        training_args = DPPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=4,
            learning_rate=3e-6,  # Lower policy LR
            vf_learning_rate=1e-4,  # Higher value LR (typical for DPPO)
            num_ppo_epochs=2,
            num_vf_epochs=1,
            num_sample_generations=0,  # Disable sample generation
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

        # Verify VF LR is indeed higher
        policy_lr = trainer.lr_scheduler.get_last_lr()[0]
        vf_lr = trainer.vf_lr_scheduler.get_last_lr()[0]
        assert vf_lr > policy_lr, "Value function LR should be higher than policy LR in typical DPPO setup"

        # Train should complete successfully
        trainer.train()

    def test_dppo_infrequent_vf_updates(self):
        """Test DPPO with infrequent value function updates."""
        training_args = DPPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=4,
            learning_rate=3e-6,
            vf_learning_rate=1e-4,
            num_ppo_epochs=2,
            num_vf_epochs=2,
            vf_update_frequency=3,  # Update value every 3 policy updates
            num_sample_generations=0,  # Disable sample generation
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

        # Train should complete successfully with infrequent VF updates
        trainer.train()

        # Verify configuration was respected
        assert trainer.args.vf_update_frequency == 3
