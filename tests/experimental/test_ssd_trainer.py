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

import torch
from datasets import Dataset
from transformers.utils import is_peft_available

from trl.experimental.ssd import SSDConfig, SSDTrainer

from ..testing_utils import TrlTestCase, require_peft


if is_peft_available():
    from peft import LoraConfig


class TestSSDTrainer(TrlTestCase):
    def test_vllm_config_defaults_match_reference_trainers(self):
        config = SSDConfig(output_dir=self.tmp_dir)

        assert config.vllm_mode == "colocate"
        assert config.vllm_model_impl == "vllm"

    def test_training_with_string_prompts(self):
        dataset = Dataset.from_dict(
            {
                "prompt": ["Write a function to add two numbers.", "Write a function to check if a number is prime."],
            }
        )

        training_args = SSDConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=1,
            max_completion_length=8,
            max_steps=1,
            num_generations=1,
        )

        trainer = SSDTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=dataset,
        )

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

    def test_training_with_chat_prompts(self):
        dataset = Dataset.from_dict(
            {
                "prompt": [
                    [{"role": "user", "content": "Write a function to add two numbers."}],
                    [{"role": "user", "content": "Write a function to check if a number is prime."}],
                ],
            }
        )

        training_args = SSDConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=1,
            max_completion_length=8,
            max_steps=1,
            num_generations=1,
        )

        trainer = SSDTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=dataset,
        )

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

    def test_training_with_temperature_and_truncation(self):
        """Test with SSD-paper-style hyperparameters: T_train=0.6, top_k=20, top_p=0.95."""
        dataset = Dataset.from_dict(
            {
                "prompt": [
                    "Write a Python function to reverse a string.",
                    "Write a Python function to sort a list.",
                ],
            }
        )

        training_args = SSDConfig(
            output_dir=self.tmp_dir,
            learning_rate=5e-6,
            per_device_train_batch_size=1,
            max_completion_length=16,
            max_steps=1,
            num_generations=1,
            temperature=0.6,
            top_k=20,
            top_p=0.95,
        )

        trainer = SSDTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=dataset,
        )

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

    def test_training_with_multiple_generations(self):
        dataset = Dataset.from_dict(
            {
                "prompt": [
                    "Write a function to add two numbers.",
                    "Write a function to check if a number is prime.",
                ],
            }
        )

        training_args = SSDConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=2,
            max_completion_length=8,
            max_steps=1,
            num_generations=2,
        )

        trainer = SSDTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=dataset,
        )

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

    def test_training_reuses_buffered_generation_batches(self):
        dataset = Dataset.from_dict(
            {
                "prompt": ["Write a function to add two numbers.", "Write a function to check if a number is prime."],
            }
        )

        training_args = SSDConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=1,
            steps_per_generation=2,
            max_completion_length=8,
            max_steps=2,
            num_generations=1,
        )

        trainer = SSDTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=dataset,
        )

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

    def test_training_with_filter_empty_disabled(self):
        dataset = Dataset.from_dict(
            {
                "prompt": ["Write a function to add two numbers.", "Write a function to check if a number is prime."],
            }
        )

        training_args = SSDConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=1,
            max_completion_length=8,
            max_steps=1,
            num_generations=1,
            filter_empty=False,
        )

        trainer = SSDTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=dataset,
        )

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

    def test_training_logs_ssd_metrics(self):
        dataset = Dataset.from_dict(
            {
                "prompt": ["Write a function to add two numbers.", "Write a function to check if a number is prime."],
            }
        )

        training_args = SSDConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=1,
            max_completion_length=8,
            max_steps=1,
            num_generations=1,
            logging_steps=1,
            report_to="none",
        )

        trainer = SSDTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=dataset,
        )

        trainer.train()

        # Check that SSD-specific metrics were logged
        assert len(trainer._metrics["train"]["ssd/cross_entropy_loss"]) > 0
        assert len(trainer._metrics["train"]["ssd/active_sample_ratio"]) > 0
        assert len(trainer._metrics["train"]["completions/mean_length"]) > 0

    @require_peft
    def test_training_with_peft_model(self):
        dataset = Dataset.from_dict(
            {
                "prompt": ["Write a function to add two numbers.", "Write a function to check if a number is prime."],
            }
        )

        training_args = SSDConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=1,
            max_completion_length=8,
            max_steps=1,
            num_generations=1,
        )

        trainer = SSDTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=dataset,
            peft_config=LoraConfig(
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "v_proj"],
            ),
        )

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

    def test_training_with_disable_dropout_false(self):
        dataset = Dataset.from_dict(
            {
                "prompt": ["Write a function to add two numbers.", "Write a function to check if a number is prime."],
            }
        )

        training_args = SSDConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=1,
            max_completion_length=8,
            max_steps=1,
            num_generations=1,
            disable_dropout=False,
        )

        trainer = SSDTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=dataset,
        )

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None
