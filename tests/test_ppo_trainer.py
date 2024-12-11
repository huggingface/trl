# Copyright 2024 The HuggingFace Team. All rights reserved.
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

import platform
import subprocess
import tempfile
import unittest

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from transformers.testing_utils import require_peft

from trl import PPOConfig, PPOTrainer
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE


def test():
    command = """\
python examples/scripts/ppo/ppo.py \
    --dataset_name trl-internal-testing/descriptiveness-sentiment-trl-style \
    --dataset_train_split descriptiveness \
    --learning_rate 3e-6 \
    --output_dir models/minimal/ppo \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --total_episodes 10 \
    --model_name_or_path trl-internal-testing/tiny-Qwen2ForCausalLM-2.5 \
    --reward_model_path trl-internal-testing/tiny-Qwen2ForCausalLM-2.5 \
    --sft_model_path trl-internal-testing/tiny-Qwen2ForCausalLM-2.5 \
    --missing_eos_penalty 1.0 \
    --save_strategy no \
    --stop_token eos
"""
    if platform.system() == "Windows":
        # windows CI does not work with subprocesses for some reason
        # e.g., https://github.com/huggingface/trl/actions/runs/9600036224/job/26475286210?pr=1743
        return
    subprocess.run(
        command,
        shell=True,
        check=True,
    )


def test_num_train_epochs():
    command = """\
python examples/scripts/ppo/ppo.py \
    --dataset_name trl-internal-testing/descriptiveness-sentiment-trl-style \
    --dataset_train_split descriptiveness \
    --learning_rate 3e-6 \
    --output_dir models/minimal/ppo \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 0.003 \
    --model_name_or_path trl-internal-testing/tiny-Qwen2ForCausalLM-2.5 \
    --reward_model_path trl-internal-testing/tiny-Qwen2ForCausalLM-2.5 \
    --sft_model_path trl-internal-testing/tiny-Qwen2ForCausalLM-2.5 \
    --missing_eos_penalty 1.0 \
    --save_strategy no \
    --stop_token eos
"""
    if platform.system() == "Windows":
        # windows CI does not work with subprocesses for some reason
        # e.g., https://github.com/huggingface/trl/actions/runs/9600036224/job/26475286210?pr=1743
        return
    subprocess.run(
        command,
        shell=True,
        check=True,
    )


@require_peft
def test_peft_support():
    command = """\
python examples/scripts/ppo/ppo.py \
    --dataset_name trl-internal-testing/descriptiveness-sentiment-trl-style \
    --dataset_train_split descriptiveness \
    --learning_rate 3e-6 \
    --output_dir models/minimal/ppo \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --total_episodes 10 \
    --model_name_or_path EleutherAI/pythia-14m \
    --missing_eos_penalty 1.0 \
    --save_strategy no \
    --stop_token eos \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 16 \
    --lora_target_modules query_key_value dense
"""
    if platform.system() == "Windows":
        # windows CI does not work with subprocesses for some reason
        # e.g., https://github.com/huggingface/trl/actions/runs/9600036224/job/26475286210?pr=1743
        return
    subprocess.run(
        command,
        shell=True,
        check=True,
    )


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
        self.value_model = AutoModelForSequenceClassification.from_pretrained(
            self.model_id, trust_remote_code=True, num_labels=1
        )
        self.reward_model = AutoModelForSequenceClassification.from_pretrained(
            self.model_id, trust_remote_code=True, num_labels=1
        )

        # Load dataset
        raw_dataset = load_dataset(
            "trl-internal-testing/descriptiveness-sentiment-trl-style",
            split="descriptiveness",
        )

        def prepare_dataset(dataset, tokenizer):
            """pre-tokenize the dataset before training"""

            def tokenize(element):
                outputs = tokenizer(
                    element["prompt"],
                    padding=False,
                )
                return {"input_ids": outputs["input_ids"]}

            return dataset.map(
                tokenize,
                batched=True,
                remove_columns=dataset.column_names,
                load_from_cache_file=False,
            )

        # Process the dataset
        # Split into train and eval datasets as in ppo.py
        eval_samples = 100
        train_dataset = raw_dataset.select(range(len(raw_dataset) - eval_samples))
        eval_dataset = raw_dataset.select(range(len(raw_dataset) - eval_samples, len(raw_dataset)))

        # Process both datasets
        self.train_dataset = prepare_dataset(train_dataset, self.tokenizer)
        self.eval_dataset = prepare_dataset(eval_dataset, self.tokenizer)

    def test_basic_training(self):
        """Test basic PPO training configuration from the example script."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Capture initial critic weights
            initial_critic_weights = {}
            for name, param in self.value_model.named_parameters():
                initial_critic_weights[name] = param.clone().detach()

            # Configure training args similar to example script
            training_args = PPOConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=4,  # Reduced from 64 for testing
                gradient_accumulation_steps=1,
                learning_rate=3e-6,
                total_episodes=10,  # Reduced for testing
                save_strategy="no",
                report_to="none",
                missing_eos_penalty=1.0,
            )

            # Create trainer
            trainer = PPOTrainer(
                args=training_args,
                processing_class=self.tokenizer,
                model=self.model,
                ref_model=self.ref_model,
                reward_model=self.reward_model,
                value_model=self.value_model,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
            )

            # Train and verify no exceptions are raised
            trainer.train()

            # Check if critic weights have been updated
            weights_updated = False
            for name, param in self.value_model.named_parameters():
                if not torch.allclose(initial_critic_weights[name], param.to("cpu")):
                    weights_updated = True
                    break

            self.assertTrue(weights_updated, "Critic weights were not updated during training")

    @require_peft
    def test_peft_training(self):
        """Test PPO training with PEFT configuration."""
        from peft import LoraConfig

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Capture initial critic weights
            initial_critic_weights = {}
            for name, param in self.value_model.named_parameters():
                initial_critic_weights[name] = param.clone().detach()

            # Configure training args
            training_args = PPOConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=4,
                gradient_accumulation_steps=1,
                learning_rate=3e-6,
                total_episodes=10,
                save_strategy="no",
                report_to="none",
                missing_eos_penalty=1.0,
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
                ref_model=None,  # No ref_model needed with PEFT
                reward_model=self.reward_model,
                value_model=self.value_model,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                peft_config=peft_config,
            )

            # Train and verify no exceptions are raised
            trainer.train()

            # Check if critic weights have been updated
            weights_updated = False
            for name, param in self.value_model.named_parameters():
                if not torch.allclose(initial_critic_weights[name], param.to("cpu")):
                    weights_updated = True
                    break

            self.assertTrue(weights_updated, "Critic weights were not updated during training")

    def test_deepspeed_config(self):
        """Test PPO training with DeepSpeed-like configuration."""
        if platform.system() == "Windows":
            # Skip on Windows as noted in original tests
            return

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Capture initial critic weights
            initial_critic_weights = {}
            for name, param in self.value_model.named_parameters():
                initial_critic_weights[name] = param.clone().detach()

            # Configure training args similar to deepspeed example
            training_args = PPOConfig(
                output_dir=tmp_dir,
                num_ppo_epochs=1,
                num_mini_batches=1,
                learning_rate=3e-6,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=16,
                total_episodes=10,  # Reduced for testing
                save_strategy="no",
                report_to="none",
                local_rollout_forward_batch_size=1,
                missing_eos_penalty=1.0,
            )

            # Create trainer
            trainer = PPOTrainer(
                args=training_args,
                processing_class=self.tokenizer,
                model=self.model,
                ref_model=self.ref_model,
                reward_model=self.reward_model,
                value_model=self.value_model,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
            )

            # Train and verify no exceptions are raised
            trainer.train()

            # Check if critic weights have been updated
            weights_updated = False
            for name, param in self.value_model.named_parameters():
                if not torch.allclose(initial_critic_weights[name], param.to("cpu")):
                    weights_updated = True
                    break

            self.assertTrue(weights_updated, "Critic weights were not updated during training")

    def test_with_num_train_epochs(self):
        """Test PPO training with num_train_epochs configuration."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Capture initial critic weights
            initial_critic_weights = {}
            for name, param in self.value_model.named_parameters():
                initial_critic_weights[name] = param.clone().detach()

            # Configure training args
            training_args = PPOConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=4,
                gradient_accumulation_steps=1,
                learning_rate=3e-6,
                num_train_epochs=0.003,  # As used in original test
                save_strategy="no",
                report_to="none",
                missing_eos_penalty=1.0,
            )

            # Create trainer
            trainer = PPOTrainer(
                args=training_args,
                processing_class=self.tokenizer,
                model=self.model,
                ref_model=self.ref_model,
                reward_model=self.reward_model,
                value_model=self.value_model,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
            )

            # Train and verify no exceptions are raised
            trainer.train()

            # Check if critic weights have been updated
            weights_updated = False
            for name, param in self.value_model.named_parameters():
                if not torch.allclose(initial_critic_weights[name], param.to("cpu")):
                    weights_updated = True
                    break

            self.assertTrue(weights_updated, "Critic weights were not updated during training")
