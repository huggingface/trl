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

"""Test SFTTrainer with PromptEncoder PEFT configuration"""
import unittest

from datasets import Dataset
from transformers import AutoTokenizer
from transformers.testing_utils import require_peft
from transformers.utils import is_peft_available

from trl import SFTConfig, SFTTrainer


if is_peft_available():
    from peft import PromptEncoderConfig, TaskType


@require_peft
class TestSFTTrainerPromptEncoder(unittest.TestCase):
    """Test SFTTrainer with PromptEncoder configuration"""

    def setUp(self):
        """Set up test fixtures"""
        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m", trust_remote_code=True)

        # Create a simple dataset for testing
        data = [
            {"messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"}
            ]},
            {"messages": [
                {"role": "user", "content": "How are you?"},
                {"role": "assistant", "content": "I'm doing well, thanks!"}
            ]},
        ]

        # Apply chat template
        def apply_template(example):
            text = self.tokenizer.apply_chat_template(example["messages"], tokenize=False)
            return {"text": text}

        dataset = Dataset.from_list(data)
        self.dataset = dataset.map(apply_template)

    def test_prompt_encoder_token_accuracy(self):
        """Test that token accuracy computation works with PromptEncoder"""
        # Create PEFT config with virtual tokens
        peft_config = PromptEncoderConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=16,  # Add 16 virtual tokens
            encoder_hidden_size=128
        )

        # Training config
        training_args = SFTConfig(
            output_dir="./test_output",
            per_device_train_batch_size=1,
            max_steps=2,
            report_to="none",
            dataset_text_field="text"
        )

        # Create trainer - this should not raise an error
        trainer = SFTTrainer(
            model="EleutherAI/pythia-160m",
            args=training_args,
            train_dataset=self.dataset,
            processing_class=self.tokenizer,
            peft_config=peft_config
        )

        # Run a training step - this should work without dimension mismatch errors
        trainer.train()

        # Check that mean_token_accuracy was computed
        self.assertIn("mean_token_accuracy", trainer._metrics["train"])
        self.assertGreater(len(trainer._metrics["train"]["mean_token_accuracy"]), 0)

    def test_prompt_encoder_different_virtual_token_counts(self):
        """Test with different numbers of virtual tokens"""
        for num_virtual_tokens in [8, 32, 64]:
            peft_config = PromptEncoderConfig(
                task_type=TaskType.CAUSAL_LM,
                num_virtual_tokens=num_virtual_tokens,
                encoder_hidden_size=128
            )

            training_args = SFTConfig(
                output_dir=f"./test_output_{num_virtual_tokens}",
                per_device_train_batch_size=1,
                max_steps=1,
                report_to="none",
                dataset_text_field="text"
            )

            trainer = SFTTrainer(
                model="EleutherAI/pythia-160m",
                args=training_args,
                train_dataset=self.dataset,
                processing_class=self.tokenizer,
                peft_config=peft_config
            )

            # Should not raise any errors
            trainer.train()
