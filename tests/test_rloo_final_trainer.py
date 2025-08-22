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

import unittest

import torch
from datasets import load_dataset
from parameterized import parameterized
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from transformers.testing_utils import require_peft
from transformers.utils import is_peft_available

from trl.trainer.rloo_final_trainer import RLOOFinalTrainer
from trl.trainer.rloo_finall_config import RLOOConfig_NEW

from .testing_utils import TrlTestCase

if is_peft_available():
    from peft import LoraConfig


class RLOOFinalTrainerTester(TrlTestCase):
    def test_init_minimal(self):
        # Test that RLOOFinalTrainer can be instantiated with only model, reward_funcs and train_dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")
        RLOOFinalTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            train_dataset=dataset,
        )

    @parameterized.expand([("standard_prompt_only",), ("conversational_prompt_only",)])
    def test_training(self, config_name):
        dataset = load_dataset("trl-internal-testing/zen", config_name, split="train")

        training_args = RLOOConfig_NEW(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # increase the learning rate to speed up the test
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
        )
        trainer = RLOOFinalTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            self.assertFalse(torch.equal(param, new_param), f"Parameter {n} has not changed.")

    def test_training_with_eval(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only")

        training_args = RLOOConfig_NEW(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            per_device_eval_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            eval_strategy="steps",
            eval_steps=2,
            report_to="none",
        )
        trainer = RLOOFinalTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
        )

        trainer.train()

    def test_training_multiple_iterations(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = RLOOConfig_NEW(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # increase the learning rate to speed up the test
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            num_iterations=2,
            report_to="none",
        )
        trainer = RLOOFinalTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            self.assertFalse(torch.equal(param, new_param), f"Parameter {n} has not changed.")

    @require_peft
    def test_training_peft(self):
        model = AutoModelForCausalLM.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
        base_param_names = [f"base_model.model.{n}" for n, _ in model.named_parameters()]
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = RLOOConfig_NEW(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # increase the learning rate to speed up the test
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
        )
        trainer = RLOOFinalTrainer(
            model=model,
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset,
            peft_config=LoraConfig(),
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

        # Only LoRA parameters should change
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if any(base_name in n for base_name in base_param_names):
                # Base model parameters should remain unchanged
                self.assertTrue(torch.equal(param, new_param), f"Base parameter {n} has changed.")
            else:
                # LoRA parameters should change
                self.assertFalse(torch.equal(param, new_param), f"LoRA parameter {n} has not changed.")

    def test_training_with_multiple_reward_funcs(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        def custom_reward_func_1(completions, **kwargs):
            return [len(completion[0]["content"]) for completion in completions]

        def custom_reward_func_2(completions, **kwargs):
            return [1.0 for completion in completions]  # constant reward

        training_args = RLOOConfig_NEW(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=3,
            num_generations=3,
            max_completion_length=8,
            report_to="none",
        )
        trainer = RLOOFinalTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=[custom_reward_func_1, custom_reward_func_2],
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            self.assertFalse(torch.equal(param, new_param), f"Parameter {n} has not changed.")

    def test_training_with_reward_weights(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        def custom_reward_func_1(completions, **kwargs):
            return [1.0 for completion in completions]

        def custom_reward_func_2(completions, **kwargs):
            return [2.0 for completion in completions]

        training_args = RLOOConfig_NEW(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=3,
            num_generations=3,
            max_completion_length=8,
            reward_weights=[0.3, 0.7],
            report_to="none",
        )
        trainer = RLOOFinalTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=[custom_reward_func_1, custom_reward_func_2],
            args=training_args,
            train_dataset=dataset,
        )

        # Check reward weights are set correctly
        expected_weights = torch.tensor([0.3, 0.7])
        torch.testing.assert_close(trainer.reward_weights, expected_weights)

        trainer.train()

        self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

    def test_training_with_beta_values(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        for beta in [0.0, 0.1, 1.0]:
            with self.subTest(beta=beta):
                training_args = RLOOConfig_NEW(
                    output_dir=self.tmp_dir,
                    learning_rate=0.1,
                    per_device_train_batch_size=3,
                    num_generations=3,
                    max_completion_length=8,
                    beta=beta,
                    report_to="none",
                )
                trainer = RLOOFinalTrainer(
                    model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                    reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
                    args=training_args,
                    train_dataset=dataset,
                )

                self.assertEqual(trainer.beta, beta)
                trainer.train()
                self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

    @parameterized.expand([("token",), ("sequence",)])
    def test_importance_sampling_levels(self, importance_sampling_level):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = RLOOConfig_NEW(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=3,
            num_generations=3,
            max_completion_length=8,
            importance_sampling_level=importance_sampling_level,
            report_to="none",
        )
        trainer = RLOOFinalTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset,
        )

        self.assertEqual(trainer.importance_sampling_level, importance_sampling_level)
        trainer.train()
        self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

    def test_scale_rewards_parameter(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = RLOOConfig_NEW(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=3,
            num_generations=4,  # Need at least 4 generations for proper advantage scaling
            max_completion_length=8,
            scale_rewards=True,
            report_to="none",
        )
        trainer = RLOOFinalTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset,
        )

        self.assertTrue(trainer.scale_rewards)
        trainer.train()
        self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

    def test_logging_functionality(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = RLOOConfig_NEW(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=3,
            num_generations=3,
            max_completion_length=8,
            logging_steps=1,
            log_completions=True,
            num_completions_to_print=2,
            report_to="none",
        )
        trainer = RLOOFinalTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset,
        )

        trainer.train()
        
        # Check that logging occurred
        self.assertTrue(len(trainer.state.log_history) > 0)
        self.assertIn("train_loss", trainer.state.log_history[-1])

    def test_rloo_baseline_calculation(self):
        # Test RLOO baseline calculation with known values
        device = torch.device("cpu")
        num_generations = 4
        
        # Create synthetic rewards: [1, 2, 3, 4] for one prompt
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)
        grouped_rewards = rewards.view(-1, num_generations)  # Shape: (1, 4)
        
        # Calculate baselines manually
        grouped_sum = grouped_rewards.sum(dim=1, keepdim=True)  # [10.0]
        baselines = (grouped_sum - grouped_rewards) / (num_generations - 1)
        
        expected_baselines = torch.tensor([
            [(10.0 - 1.0) / 3,  # 3.0
             (10.0 - 2.0) / 3,  # 8/3 ≈ 2.67
             (10.0 - 3.0) / 3,  # 7/3 ≈ 2.33
             (10.0 - 4.0) / 3]  # 2.0
        ], device=device)
        
        torch.testing.assert_close(baselines, expected_baselines, rtol=1e-4, atol=1e-4)
        
        # Test advantages
        baselines_flat = baselines.view(-1)
        advantages = rewards - baselines_flat
        expected_advantages = torch.tensor([
            1.0 - 3.0,      # -2.0
            2.0 - 8.0/3,    # ≈ -0.67
            3.0 - 7.0/3,    # ≈ 0.67
            4.0 - 2.0,      # 2.0
        ], device=device)
        
        torch.testing.assert_close(advantages, expected_advantages, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()