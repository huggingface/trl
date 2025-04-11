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

import tempfile
import unittest

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from transformers.testing_utils import require_peft
from transformers.utils import is_peft_available

from trl import PPOConfig, PPOTrainer
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE


if is_peft_available():
    from peft import LoraConfig


class TestPPOTrainer(unittest.TestCase):
    def setUp(self):
        # Set up the models and tokenizer using the test model
        self.model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self.ref_model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, padding_side="left")
        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        if self.tokenizer.chat_template is None:
            self.tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE

        # Add reward and value models as in ppo.py
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

    def test_basic_training(self):
        """Test basic PPO training configuration and verify model updates."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Capture initial weights
            initial_critic_weights = {}
            initial_policy_weights = {}
            for name, param in self.value_model.named_parameters():
                initial_critic_weights[name] = param.clone().detach()
            for name, param in self.model.named_parameters():
                initial_policy_weights[name] = param.clone().detach()

            # Configure training args similar to example script
            training_args = PPOConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=4,
                per_device_eval_batch_size=2,
                num_ppo_epochs=2,  # Decrease number of PPO epochs to speed up test
                report_to="none",
            )

            # Create trainer
            trainer = PPOTrainer(
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

            # Check if critic weights have been updated
            critic_weights_updated = False
            for name, param in trainer.model.value_model.named_parameters():
                if not torch.allclose(initial_critic_weights[name], param.to("cpu")):
                    critic_weights_updated = True
                    break

            # Check if policy weights have been updated
            policy_weights_updated = False
            for name, param in trainer.model.policy.named_parameters():
                if not torch.allclose(initial_policy_weights[name], param.to("cpu")):
                    policy_weights_updated = True
                    break

            self.assertTrue(critic_weights_updated, "Critic weights were not updated during training")
            self.assertTrue(policy_weights_updated, "Policy weights were not updated during training")

    @require_peft
    def test_peft_training(self):
        """Test PPO training with PEFT configuration and verify model updates."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Capture initial weights
            initial_critic_weights = {}
            initial_policy_weights = {}
            for name, param in self.value_model.named_parameters():
                initial_critic_weights[name] = param.clone().detach()
            for name, param in self.model.named_parameters():
                initial_policy_weights[name] = param.clone().detach()

            # Configure training args
            training_args = PPOConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=4,
                per_device_eval_batch_size=2,
                num_ppo_epochs=2,  # Decrease number of PPO epochs to speed up test
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
            trainer = PPOTrainer(
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

            # Check if critic weights have been updated
            critic_weights_updated = False
            for name, param in trainer.model.value_model.named_parameters():
                if name in initial_critic_weights and not torch.allclose(
                    initial_critic_weights[name], param.to("cpu")
                ):
                    critic_weights_updated = True
                    break

            # Check if policy weights have been updated - for PEFT we check the LoRA weights
            policy_weights_updated = False
            for name, param in trainer.model.policy.named_parameters():
                if "lora" in name.lower() and param.requires_grad:  # Only check LoRA weights
                    # New weights should be non-zero if they've been updated
                    if not torch.allclose(param, torch.zeros_like(param)):
                        policy_weights_updated = True
                        break

            self.assertTrue(critic_weights_updated, "Critic weights were not updated during training")
            self.assertTrue(policy_weights_updated, "Policy LoRA weights were not updated during training")
