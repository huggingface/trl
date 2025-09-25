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
from unittest.mock import patch

import pytest
import torch
from datasets import load_dataset
from parameterized import parameterized
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from transformers.testing_utils import require_liger_kernel, require_peft, require_vision
from transformers.utils import is_peft_available

from trl import GRPOConfig, GRPOTrainer
from trl.experimental.grpo_with_replay_buffer.grpo_with_replay_buffer_config import GRPOWithReplayBufferConfig
from trl.experimental.grpo_with_replay_buffer.grpo_with_replay_buffer_trainer import (
    GRPOWithReplayBufferTrainer,
    ReplayBuffer,
)
from trl.experimental.gspo_token import GRPOTrainer as GSPOTokenTrainer

from .testing_utils import TrlTestCase, require_vllm


if is_peft_available():
    from peft import LoraConfig, PeftModel


class GetHighEntropyMaskTester(TrlTestCase):
    def get_high_entropy_mask(self, entropies, mask, threshold):
        """Helper method to test the get_high_entropy_mask functionality."""
        # Create a mock trainer with minimal setup
        from unittest.mock import Mock

        # Create a mock accelerator
        mock_accelerator = Mock()
        mock_accelerator.num_processes = 1  # Single process for testing

        # Create a minimal trainer instance just to access the method
        trainer = Mock(spec=GRPOTrainer)
        trainer.accelerator = mock_accelerator
        trainer.accelerator.gather = lambda x: x
        trainer.accelerator.pad_across_processes = lambda x, dim, pad_index: x

        # Call the actual method from GRPOTrainer
        return GRPOTrainer.get_high_entropy_mask(trainer, entropies, mask, threshold)

    def test_compute_entropy_mask_0(self):
        # We have a total of 12 tokens out of which 10 are non-pad.
        # for a top_entropy_quantile of 0.8, we expect the top 20% i.e 2 non-pad tokens corresponding to
        # the highest entropy to be unmasked.
        # In our example these will be the tokens corresponding to the entropies 0.9 and 1.0 since 1.1 and 1.2 are pad
        # tokens they are excluded from the entropy threshold calculation.
        entropies = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [0.7, 0.8, 0.9, 1.0, 1.1, 1.2]])
        mask = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]])
        entropy_mask = self.get_high_entropy_mask(entropies, mask, threshold=0.8)
        expected_mask = torch.tensor([[0, 0, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0]], dtype=torch.bool)
        torch.testing.assert_close(entropy_mask, expected_mask)

    def test_compute_entropy_mask_1(self):
        # Another example with a different set of entropies and a different mask.
        entropies = torch.tensor([[0.1, 0.2, 0.3, 1.4, 0.5, 0.14], [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]])
        mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 0, 0]])
        entropy_mask = self.get_high_entropy_mask(entropies, mask, threshold=0.8)
        expected_mask = torch.tensor([[0, 0, 0, 1, 0, 0], [0, 0, 0, 1, 0, 0]], dtype=torch.bool)
        torch.testing.assert_close(entropy_mask, expected_mask)

    def test_compute_entropy_mask_lower_threshold(self):
        # For a threshold of 0.5 we expect the top half of the non-pad tokens to be unmasked.
        entropies = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [0.7, 0.8, 0.9, 1.0, 1.1, 1.2]])
        mask = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]])
        entropy_mask = self.get_high_entropy_mask(entropies, mask, threshold=0.5)
        expected_mask = torch.tensor([[0, 0, 0, 0, 0, 1], [1, 1, 1, 1, 0, 0]], dtype=torch.bool)
        torch.testing.assert_close(entropy_mask, expected_mask)

    def test_compute_entropy_threshold_0(self):
        # If the threshold is 0.0 then we expect the mask to be all ones for non-pad tokens.
        entropies = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [0.7, 0.8, 0.9, 1.0, 1.1, 1.2]])
        mask = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]])
        entropy_mask = self.get_high_entropy_mask(entropies, mask, threshold=0.0)
        expected_mask = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]], dtype=torch.bool)
        torch.testing.assert_close(entropy_mask, expected_mask)

    def test_compute_entropy_threshold_1(self):
        # If the threshold is 1.0 then we expect the mask to be all zeros BUT ONE VALUE.
        entropies = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [0.7, 0.8, 0.9, 1.0, 1.1, 1.2]])
        mask = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]])
        entropy_mask = self.get_high_entropy_mask(entropies, mask, threshold=1.0)
        expected_mask = torch.tensor([[0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]], dtype=torch.bool)
        torch.testing.assert_close(entropy_mask, expected_mask)

    def test_compute_entropy_all_masked(self):
        # If there are no non-pad tokens we expect the mask to be all zeros.
        entropies = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [0.7, 0.8, 0.9, 1.0, 1.1, 1.2]])
        mask = torch.tensor([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]])
        entropy_mask = self.get_high_entropy_mask(entropies, mask, threshold=0.5)
        expected_mask = torch.tensor([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]], dtype=torch.bool)
        torch.testing.assert_close(entropy_mask, expected_mask)


class GRPOTrainerTester(TrlTestCase):
    def test_init_minimal(self):
        # Test that GRPOTrainer can be instantiated with only model, reward_model and train_dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")
        GRPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            train_dataset=dataset,
        )

    @parameterized.expand([("standard_prompt_only",), ("conversational_prompt_only",)])
    def test_training(self, config_name):
        dataset = load_dataset("trl-internal-testing/zen", config_name, split="train")

        training_args = GRPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # increase the learning rate to speed up the test
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
        )
        trainer = GRPOTrainer(
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

    @parameterized.expand([("bnpo",), ("dr_grpo",), ("dapo",)])
    def test_training_loss_types(self, loss_type):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = GRPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # increase the learning rate to speed up the test
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=32,  # reduce the completion length to reduce memory usage
            gradient_accumulation_steps=2,  # set to 2 to test than DAPO can operate with accumulated batch
            loss_type=loss_type,
            report_to="none",
        )
        trainer = GRPOTrainer(
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

        training_args = GRPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            per_device_eval_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            eval_strategy="steps",
            eval_steps=2,
            report_to="none",
        )
        trainer = GRPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
        )

        trainer.train()

    def test_training_multiple_iterations(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = GRPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # increase the learning rate to speed up the test
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            num_iterations=2,
            report_to="none",
        )
        trainer = GRPOTrainer(
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

        training_args = GRPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # increase the learning rate to speed up the test
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
        )
        trainer = GRPOTrainer(
            model=model,
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset,
            peft_config=LoraConfig(),
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

        # Check that the peft params have changed and the base model params have not changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if n in base_param_names:  # We expect the base model params to be the same
                self.assertTrue(torch.allclose(param, new_param), f"Parameter {n} has changed.")
            elif "base_layer" not in n:  # We expect the peft params to be different (except for the base layer)
                self.assertFalse(torch.allclose(param, new_param), f"Parameter {n} has not changed.")

    @require_peft
    def test_training_peft_with_gradient_checkpointing(self):
        """Test that training works with PEFT and gradient checkpointing enabled."""
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        model = AutoModelForCausalLM.from_pretrained(
            "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            dtype=torch.float32,  # Use float32 for testing to avoid precision issues
        )

        lora_config = LoraConfig(
            r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none"
        )

        training_args = GRPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=3,
            num_generations=3,
            max_completion_length=8,
            gradient_checkpointing=True,  # Enable gradient checkpointing
            report_to="none",
        )
        trainer = GRPOTrainer(
            model=model,
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset,
            peft_config=lora_config,
        )

        # Verify gradient checkpointing is enabled
        self.assertIsInstance(trainer.model, PeftModel)

        # Store initial parameters to check which ones change
        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

        # Check that only LoRA parameters have changed, base model parameters remain unchanged
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if "lora" in n.lower():  # LoRA parameters should change
                self.assertFalse(torch.equal(param, new_param), f"LoRA parameter {n} has not changed.")
            else:  # Base model parameters should not change
                self.assertTrue(torch.equal(param, new_param), f"Base parameter {n} has changed.")

    def test_training_different_reward_model(self):
        # Use a reward model different from the model: different chat template, tokenization, etc.
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_only", split="train")
        reward_model_id = "trl-internal-testing/tiny-LlamaForSequenceClassification-3.2"
        reward_model = AutoModelForSequenceClassification.from_pretrained(reward_model_id)
        reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_id)
        # By default, the trainer uses the eos token as the padding token. However, for Llama models, the eos token
        # appears in the chat template. Using it as a pad token disrupts the reward calculation, as the calculation
        # considers the score of the last token before the first pad token. To ensure correct reward calculations,
        # we use a separate pad token instead.
        reward_tokenizer.pad_token = "<|finetune_right_pad_id|>"

        training_args = GRPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # increase the learning rate to speed up the test
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
        )
        trainer = GRPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=reward_model,
            args=training_args,
            train_dataset=dataset,
            reward_processing_classes=reward_tokenizer,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            self.assertFalse(torch.equal(param, new_param), f"Parameter {n} has not changed.")

    def test_training_reward_func_standard(self):
        # Test if trainer can handle reward function with standard format
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        def reward_func(completions, **kwargs):
            """Reward function that rewards longer completions."""
            return [float(len(completion)) for completion in completions]

        training_args = GRPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # increase the learning rate to speed up the test
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
        )
        trainer = GRPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=reward_func,
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

    def test_training_reward_func_conversational(self):
        # Test if trainer can handle reward function with conversational format
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_only", split="train")

        def reward_func(completions, **kwargs):
            """Reward function that gives higher scores to longer completion content."""
            completion_contents = [completion[0]["content"] for completion in completions]
            return [float(len(content)) for content in completion_contents]

        training_args = GRPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # increase the learning rate to speed up the test
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
        )
        trainer = GRPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=reward_func,
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

    def test_training_multiple_reward_funcs(self):
        # Test that GRPOTrainer can be instantiated with multiple reward functions
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        def reward_func1(completions, **kwargs):
            """Reward function that rewards longer completions."""
            return [float(len(completion)) for completion in completions]

        def reward_func2(completions, **kwargs):
            """Reward function that rewards completions with more unique letters."""
            return [float(len(set(completion))) for completion in completions]

        training_args = GRPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # increase the learning rate to speed up the test
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
        )
        trainer = GRPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=[reward_func1, reward_func2],
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

    def test_training_multiple_reward_funcs_with_None_output(self):
        """Test that a valid math reward function is processed correctly while the code reward function returns None."""
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        def applicable_reward_func(completions, **kwargs):
            """A reward function that rewards longer completions."""
            return [float(len(completion)) for completion in completions]

        def non_applicable_reward_func(completions, **kwargs):
            """A reward function that returns None for all inputs, as it is not applicable to this sample."""
            return [None] * len(completions)

        training_args = GRPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=3,
            num_generations=3,
            max_completion_length=8,
            report_to="none",
        )

        trainer = GRPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=[
                applicable_reward_func,
                non_applicable_reward_func,
            ],  # One applicable, one non applicable
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = {
            n: param.clone() for n, param in trainer.model.named_parameters() if param.requires_grad
        }

        trainer.train()

        self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            self.assertFalse(torch.equal(param, new_param), f"Parameter {n} has not changed.")

    def test_training_multiple_reward_funcs_with_weights(self):
        """Test that GRPOTrainer can handle multiple reward functions with weights."""
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        def reward_func1(completions, **kwargs):
            """Reward function that rewards longer completions."""
            return [float(len(completion)) for completion in completions]

        def reward_func2(completions, **kwargs):
            """Reward function that rewards completions with more unique letters."""
            return [float(len(set(completion))) for completion in completions]

        training_args = GRPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # increase the learning rate to speed up the test
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
            reward_weights=[0.7, 0.3],  # weight of reward_func1 and reward_func2 respectively
        )
        trainer = GRPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=[reward_func1, reward_func2],
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        # Check that training logs contain both reward metrics
        self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])
        self.assertIn("rewards/reward_func1/mean", trainer.state.log_history[-1])
        self.assertIn("rewards/reward_func1/std", trainer.state.log_history[-1])
        self.assertIn("rewards/reward_func2/mean", trainer.state.log_history[-1])
        self.assertIn("rewards/reward_func2/std", trainer.state.log_history[-1])

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            self.assertFalse(torch.equal(param, new_param), f"Parameter {n} has not changed.")

    def test_training_multiple_mixed_reward_funcs(self):
        # Test if the trainer can handle a mix of reward functions and reward models
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        def reward_func(completions, **kwargs):
            """Reward function that rewards longer completions."""
            return [float(len(completion)) for completion in completions]

        training_args = GRPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # increase the learning rate to speed up the test
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
        )
        trainer = GRPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=[reward_func, "trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5"],
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

    def test_training_reward_func_additional_column(self):
        # Test if trainer can handle reward function that rely on additional columns in the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        # Add a column to the dataset (dummy example, the column could be anything)
        some_values = list(range(len(dataset)))
        dataset = dataset.add_column("some_values", some_values)

        def reward_func(completions, some_values, **kwargs):
            """Reward function that rewards completions with lengths closer to the values in some_values."""
            return [float(abs(len(completion) - value)) for completion, value in zip(completions, some_values)]

        training_args = GRPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # increase the learning rate to speed up the test
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
        )
        trainer = GRPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=reward_func,
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

    def test_training_with_sync_ref_model(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = GRPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # increase the learning rate to speed up the test
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            sync_ref_model=True,
            ref_model_sync_steps=2,  # reduce sync steps to ensure a sync happens
            report_to="none",
        )
        trainer = GRPOTrainer(
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

    def test_training_beta_non_zero(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")
        training_args = GRPOConfig(
            output_dir=self.tmp_dir,
            beta=0.1,  # set beta to non-zero value to test the case where the reference model is used
            learning_rate=0.1,  # increase the learning rate to speed up the test
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
        )
        trainer = GRPOTrainer(
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

    def test_training_with_entropy_filter(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")
        training_args = GRPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # increase the learning rate to speed up the test
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
            top_entropy_quantile=0.2,
        )
        trainer = GRPOTrainer(
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

    @unittest.skip("We should add a mock for the vLLM server.")
    @require_peft
    @require_vllm
    def test_training_vllm_and_peft(self):
        """Test that training works with vLLM for generation."""
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")  # tiny model is too small for vLLM
        base_param_names = [f"base_model.model.{n}" for n, _ in model.named_parameters()]
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = GRPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # increase the learning rate to speed up the test
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
            use_vllm=True,
        )
        lora_config = LoraConfig(
            target_modules="all-linear",
            # test with non-default modules as it adds extra keys in state_dict that we need to handle
            modules_to_save=["embed_tokens", "lm_head"],
        )
        trainer = GRPOTrainer(
            model=model,
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset,
            peft_config=lora_config,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

        # Check that the peft params have changed and the base model params have not changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if n in base_param_names:  # We expect the base model params to be the same
                self.assertTrue(torch.allclose(param, new_param), f"Parameter {n} has changed.")
            elif "base_layer" not in n and "original_module" not in n:
                # We expect the peft params to be different (except for the base layer)
                self.assertFalse(torch.allclose(param, new_param), f"Parameter {n} has not changed.")

    @require_vllm
    @unittest.skip("We should add a mock for the vLLM server.")
    def test_training_vllm_guided_decoding(self):
        """Test that training works with vLLM for generation with guided decoding."""
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = GRPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # increase the learning rate to speed up the test
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
            use_vllm=True,
            vllm_guided_decoding_regex=r"<reasoning>\n.*\n</reasoning>\n<answer>\n.*\n</answer>",
        )
        trainer = GRPOTrainer(
            model="Qwen/Qwen2.5-0.5B-Instruct",  # tiny model is too small for vLLM
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

    @require_vllm
    @unittest.skip("We should add a mock for the vLLM server.")
    def test_training_vllm_importance_sampling_correction(self):
        """Test that training works with vLLM for generation with guided decoding."""
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = GRPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=3,
            num_generations=3,
            max_completion_length=8,
            report_to="none",
            use_vllm=True,
            vllm_importance_sampling_correction=True,
            vllm_importance_sampling_cap=3.0,
        )
        trainer = GRPOTrainer(
            model="Qwen/Qwen2.5-0.5B-Instruct",  # tiny model is too small for vLLM
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

    def test_training_with_additional_generation_kwargs(self):
        """Test that training works with additional generation kwargs."""
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = GRPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # increase the learning rate to speed up the test
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
            top_p=0.9,
            top_k=10,
            min_p=0.01,
            repetition_penalty=1.1,
        )

        trainer = GRPOTrainer(
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

    @require_vllm
    @unittest.skip("We should add a mock for the vLLM server.")
    def test_training_vllm_with_additional_generation_kwargs(self):
        """Test that training works with vLLM and additional generation kwargs."""
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = GRPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # increase the learning rate to speed up the test
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
            use_vllm=True,
            top_p=0.9,
            top_k=10,
            min_p=0.01,
            repetition_penalty=1.1,
        )

        trainer = GRPOTrainer(
            model="Qwen/Qwen2.5-0.5B-Instruct",  # tiny model is too small for vLLM
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

    @parameterized.expand([(False,), ("group",), ("batch",), (True,), ("none",)])
    def test_training_scale_rewards(self, scale_rewards):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = GRPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # increase the learning rate to speed up the test
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            scale_rewards=scale_rewards,
            report_to="none",
        )
        trainer = GRPOTrainer(
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

    @patch("transformers.generation.utils.GenerationMixin.generate")
    def test_training_with_mask_truncated_completions(self, mock_generate):
        """Test that training works with mask_truncated_completions=True parameter."""

        # We mock the generate method because the model's random weights make it extremely unlikely to produce a
        # sequence containing the EOS token within the allowed max_completion_length. As a result, all tokens are
        # masked in the loss, the model doesn't update, and the final check (which verifies the update) fails.
        def fake_generate(input_ids, **kwargs):
            # pad_token_id = 151643; eos_token_id = 151645
            completions_ids = torch.tensor(
                [
                    [1, 2, 3, 4, 5, 6, 7, 8],  # this one is truncated
                    [9, 10, 11, 151645, 151643, 151643, 151643, 151643],  # this one contains eos
                    [12, 13, 14, 15, 16, 17, 18, 151645],  # particular case, eos is generated just within the limit
                ],
                device=input_ids.device,
            )
            return torch.cat([input_ids, completions_ids], dim=1)

        mock_generate.side_effect = fake_generate

        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = GRPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # increase the learning rate to speed up the test
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            mask_truncated_completions=True,  # Enable masking of truncated completions
            report_to="none",
        )
        trainer = GRPOTrainer(
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

    def test_training_with_mask_truncated_completions_all_masked(self):
        """
        Test that when all generated completions are truncated (i.e., none contain an EOS token), and
        mask_truncated_completions=True, the model receives no effective learning signal and therefore does not update
        its parameters.

        Here, we don't mock the generate method, be we rely on the fact that the model the probability of generating
        the EOS token is extremely low, so all generated completions are truncated.
        """
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = GRPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # increase the learning rate to speed up the test
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            mask_truncated_completions=True,  # Enable masking of truncated completions
            report_to="none",
        )
        trainer = GRPOTrainer(
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
            self.assertTrue(torch.equal(param, new_param), f"Parameter {n} has changed.")

    def test_warning_raised_all_rewards_none(self):
        """Test that a proper warning is raised when all rewards are None."""
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        def always_none_reward_func(completions, **kwargs):
            """Reward function that always returns None."""
            return [None] * len(completions)

        training_args = GRPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # increase the learning rate to speed up the test
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
        )
        trainer = GRPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=always_none_reward_func,
            args=training_args,
            train_dataset=dataset,
        )

        with self.assertLogs("trl.trainer.grpo_trainer", level="WARNING") as cm:
            trainer.train()

        expected_warning = "All reward functions returned None for the following kwargs:"
        self.assertIn(expected_warning, cm.output[0])

    def test_training_num_generations_larger_than_batch_size(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = GRPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # increase the learning rate to speed up the test
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            num_generations=6,  # the number of generations is larger than the batch size, but
            gradient_accumulation_steps=2,  # gradient accumulation should allow that
            report_to="none",
        )
        trainer = GRPOTrainer(
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

    def test_training_delta_clipping(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = GRPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # increase the learning rate to speed up the test
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            delta=2.0,  # set delta to a non-None value
            report_to="none",
        )
        trainer = GRPOTrainer(
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

    def test_training_multiple_dataloader_workers(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = GRPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # increase the learning rate to speed up the test
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            dataloader_num_workers=2,  # use multiple dataloader workers
            report_to="none",
        )
        trainer = GRPOTrainer(
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

    def test_training_with_generation_kwargs(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = GRPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # increase the learning rate to speed up the test
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            generation_kwargs={"do_sample": True, "top_k": 50, "length_penalty": -0.1},  # Add some gen kwargs
            report_to="none",
        )
        trainer = GRPOTrainer(
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

    def test_training_with_reward_func_accessing_trainer_state(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        def reward_func(completions, **kwargs):
            trainer_state = kwargs.get("trainer_state")
            assert trainer_state is not None
            # transformers.TrainerState instance should have a `global_step` property.
            assert hasattr(trainer_state, "global_step")
            return [float(len(set(completion))) for completion in completions]

        training_args = GRPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,
            num_generations=2,
            max_completion_length=8,
            report_to="none",
        )
        trainer = GRPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=reward_func,
            args=training_args,
            train_dataset=dataset,
        )
        trainer.train()

    def test_prepare_input_called_with_correct_data(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")
        training_args = GRPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # increase the learning rate to speed up the test
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            gradient_accumulation_steps=3,  # can be anything in this test
            # steps_per_generation*per_device_train_batch_size=24 is divisible by num_generations=4
            steps_per_generation=4,
            num_generations=4,
            per_device_train_batch_size=6,  # reduce the batch size to reduce memory usage
            num_iterations=2,
            shuffle_dataset=False,
            report_to="none",
        )
        trainer = GRPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset,
        )
        # steps_per_generation=4, per_device_train_batch_size=6 and num_generations=4, so we expect a
        # generation batch of 24 samples (steps_per_generation * per_device_train_batch_size), containing 6
        # different prompts (steps_per_generation * per_device_train_batch_size // num_generations), each repeated
        # 4 times (num_generations).
        expected_first_generation_batch = (
            [{"prompt": "Beautiful is better than"}] * 4
            + [{"prompt": "Explicit is"}] * 4
            + [{"prompt": "Simple is better"}] * 4
            + [{"prompt": "Complex"}] * 4
            + [{"prompt": "Flat is better than"}] * 4
            + [{"prompt": "Sparse is better"}] * 4
        )
        expected_second_generation_batch = (
            [{"prompt": "Readability"}] * 4
            + [{"prompt": "Special cases aren't special"}] * 4
            + [{"prompt": "Although practicality beats"}] * 4
            + [{"prompt": "Errors should never"}] * 4
            + [{"prompt": "Unless explicitly"}] * 4
            + [{"prompt": "In the face of ambiguity, refuse"}] * 4
        )

        with patch.object(GRPOTrainer, "training_step", wraps=trainer.training_step) as mock_prepare:
            trainer.train()
            # 3 epochs * 2 iterations * 2 generation batches to cover the dataset * 4 steps_per_generation
            self.assertEqual(mock_prepare.call_count, 48)
            for i in range(0, 8):  # Generation batch repeated 8 times (steps_per_generation*num_iterations)
                assert mock_prepare.call_args_list[i].args[1] == expected_first_generation_batch
            for i in range(8, 16):
                assert mock_prepare.call_args_list[i].args[1] == expected_second_generation_batch

    @parameterized.expand(
        [
            ("trl-internal-testing/tiny-Gemma3ForConditionalGeneration",),
            ("trl-internal-testing/tiny-LlavaNextForConditionalGeneration",),
            ("trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration",),
            ("trl-internal-testing/tiny-Qwen2VLForConditionalGeneration",),
            # ("trl-internal-testing/tiny-SmolVLMForConditionalGeneration",), seems not to support bf16 properly
        ]
    )
    @require_vision
    def test_training_vlm(self, model_id):
        dataset = load_dataset("trl-internal-testing/zen-image", "conversational_prompt_only", split="train")

        def reward_func(completions, **kwargs):
            """Reward function that rewards longer completions."""
            return [float(len(completion[0]["content"])) for completion in completions]

        training_args = GRPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # increase the learning rate to speed up the test
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            max_prompt_length=None,  # disable prompt truncation, because usually, models don't support it
            report_to="none",
        )
        trainer = GRPOTrainer(
            model=model_id,
            reward_funcs=reward_func,
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

        # Check that the params have changed
        # Because of the way the tiny models are initialized, the gradient does not flow properly through the
        # vision parts of the model, so we skip them. Ideally, we should fix the init of these models.
        params_to_skip = (
            "model.vision_tower.",
            "model.multi_modal_projector.",
            "model.vision_model.",
            "model.visual.",
            "model.image_newline",
        )
        for n, param in previous_trainable_params.items():
            if n.startswith(params_to_skip):
                continue
            new_param = trainer.model.get_parameter(n)
            self.assertFalse(torch.equal(param, new_param), f"Parameter {n} has not changed.")

    @require_vision
    def test_training_vlm_beta_non_zero(self):
        dataset = load_dataset("trl-internal-testing/zen-image", "conversational_prompt_only", split="train")

        def reward_func(completions, **kwargs):
            """Reward function that rewards longer completions."""
            return [float(len(completion[0]["content"])) for completion in completions]

        training_args = GRPOConfig(
            output_dir=self.tmp_dir,
            beta=0.1,  # set beta to non-zero value to test the case where the reference model is used
            learning_rate=0.1,  # increase the learning rate to speed up the test
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
        )
        trainer = GRPOTrainer(
            model="trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration",
            reward_funcs=reward_func,
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

        # Check that the params have changed
        # Because of the way the tiny models are initialized, the gradient does not flow properly through the
        # vision parts of the model, so we skip them. Ideally, we should fix the init of these models.
        params_to_skip = ("model.visual.",)
        for n, param in previous_trainable_params.items():
            if n.startswith(params_to_skip):
                continue
            new_param = trainer.model.get_parameter(n)
            self.assertFalse(torch.equal(param, new_param), f"Parameter {n} has not changed.")

    @require_vision
    @require_peft
    def test_training_vlm_peft(self):
        model = AutoModelForImageTextToText.from_pretrained(
            "trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration"
        )
        base_param_names = [f"base_model.model.{n}" for n, _ in model.named_parameters()]
        dataset = load_dataset("trl-internal-testing/zen-image", "conversational_prompt_only", split="train")

        def reward_func(completions, **kwargs):
            """Reward function that rewards longer completions."""
            return [float(len(completion[0]["content"])) for completion in completions]

        training_args = GRPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # increase the learning rate to speed up the test
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
        )
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=reward_func,
            args=training_args,
            train_dataset=dataset,
            peft_config=LoraConfig(target_modules=["q_proj", "v_proj"]),
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

        # Check that the peft params have changed and the base model params have not changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if n in base_param_names:  # We expect the base model params to be the same
                self.assertTrue(torch.allclose(param, new_param), f"Parameter {n} has changed.")
            elif "base_layer" not in n:  # We expect the peft params to be different (except for the base layer)
                self.assertFalse(torch.allclose(param, new_param), f"Parameter {n} has not changed.")

    @require_vision
    def test_training_vlm_and_importance_sampling(self):
        dataset = load_dataset("trl-internal-testing/zen-image", "conversational_prompt_only", split="train")

        def reward_func(completions, **kwargs):
            """Reward function that rewards longer completions."""
            return [float(len(completion[0]["content"])) for completion in completions]

        training_args = GRPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # increase the learning rate to speed up the test
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            steps_per_generation=2,  # increase the steps per generation to trigger IS
            report_to="none",
        )
        trainer = GRPOTrainer(
            model="trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration",
            reward_funcs=reward_func,
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

        # Check that the params have changed
        # Because of the way the tiny models are initialized, the gradient does not flow properly through the
        # vision parts of the model, so we skip them. Ideally, we should fix the init of these models.
        params_to_skip = ("model.visual.",)
        for n, param in previous_trainable_params.items():
            if n.startswith(params_to_skip):
                continue
            new_param = trainer.model.get_parameter(n)
            self.assertFalse(torch.equal(param, new_param), f"Parameter {n} has not changed.")

    @require_vision
    @require_liger_kernel
    def test_training_vlm_and_liger(self):
        dataset = load_dataset("trl-internal-testing/zen-image", "conversational_prompt_only", split="train")

        def reward_func(completions, **kwargs):
            """Reward function that rewards longer completions."""
            return [float(len(completion[0]["content"])) for completion in completions]

        training_args = GRPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # increase the learning rate to speed up the test
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            use_liger_loss=True,  # enable Liger loss
            loss_type="bnpo",  # default dapo is not supported yet
            report_to="none",
        )
        trainer = GRPOTrainer(
            model="trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration",
            reward_funcs=reward_func,
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

        # Check that the params have changed
        # Because of the way the tiny models are initialized, the gradient does not flow properly through the
        # vision parts of the model, so we skip them. Ideally, we should fix the init of these models.
        params_to_skip = ("model.visual.",)
        for n, param in previous_trainable_params.items():
            if n.startswith(params_to_skip):
                continue
            new_param = trainer.model.get_parameter(n)
            self.assertFalse(torch.equal(param, new_param), f"Parameter {n} has not changed.")

    @require_vision
    def test_training_vlm_and_prompt_truncation(self):
        # If not handled properly, prompt truncation may truncate image token
        dataset = load_dataset("trl-internal-testing/zen-image", "conversational_prompt_only", split="train")

        def reward_func(completions, **kwargs):
            """Reward function that rewards longer completions."""
            return [float(len(completion[0]["content"])) for completion in completions]

        training_args = GRPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # increase the learning rate to speed up the test
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            max_prompt_length=18,
            report_to="none",
        )
        trainer = GRPOTrainer(
            model="trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration",
            reward_funcs=reward_func,
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

        # Check that the params have changed
        # Because of the way the tiny models are initialized, the gradient does not flow properly through the
        # vision parts of the model, so we skip them. Ideally, we should fix the init of these models.
        params_to_skip = ("model.visual.",)
        for n, param in previous_trainable_params.items():
            if n.startswith(params_to_skip):
                continue
            new_param = trainer.model.get_parameter(n)
            self.assertFalse(torch.equal(param, new_param), f"Parameter {n} has not changed.")

    @require_vision
    @require_vllm
    @parameterized.expand(
        [
            ("trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration",),
            ("trl-internal-testing/tiny-Gemma3ForConditionalGeneration",),
        ]
    )
    @unittest.skip("We should add a mock for the vLLM server.")
    def test_training_vlm_and_vllm(self, model_id) -> None:
        dataset = load_dataset("trl-internal-testing/zen-image", "conversational_prompt_only", split="train")

        def reward_func(completions, **kwargs):
            """Reward function that rewards longer completions."""
            return [float(len(completion[0]["content"])) for completion in completions]

        training_args = GRPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=3,
            num_generations=3,
            max_completion_length=8,
            max_prompt_length=18,
            report_to="none",
            use_vllm=True,
            vllm_mode="server",
        )
        trainer = GRPOTrainer(
            model=model_id,
            reward_funcs=reward_func,
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            self.assertFalse(torch.equal(param, new_param), f"Parameter {n} has not changed.")

    @require_vision
    def test_training_vlm_multi_image(self):
        dataset = load_dataset("trl-internal-testing/zen-multi-image", "conversational_prompt_only", split="train")

        # For now, mixing image+text and text-only examples is not supported, so we filter out text-only examples
        dataset = dataset.filter(lambda x: len(x["images"]) > 0)

        def reward_func(completions, **kwargs):
            """Reward function that rewards longer completions."""
            return [float(len(completion[0]["content"])) for completion in completions]

        training_args = GRPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # increase the learning rate to speed up the test
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            max_prompt_length=None,  # disable prompt truncation, because usually, models don't support it
            report_to="none",
        )
        trainer = GRPOTrainer(
            model="trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration",
            reward_funcs=reward_func,
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

        # Check that the params have changed
        # Because of the way the tiny models are initialized, the gradient does not flow properly through the
        # vision parts of the model, so we skip them. Ideally, we should fix the init of these models.
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            self.assertFalse(torch.equal(param, new_param), f"Parameter {n} has not changed.")

    def test_training_sequence_importance_sampling(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = GRPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # increase the learning rate to speed up the test
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            num_iterations=2,  # the importance sampling weights won't be 0 in this case
            importance_sampling_level="sequence",
            report_to="none",
        )
        trainer = GRPOTrainer(
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

    def test_mismatched_reward_processing_classes_length(self):
        """Test that mismatched length between reward_funcs and reward_processing_classes raises error."""
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        # Use two reward models
        reward_models = [
            "trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            "trl-internal-testing/tiny-Qwen3ForSequenceClassification",
        ]

        # Create a single processing class (tokenizer)
        single_processing_class = AutoTokenizer.from_pretrained(
            "trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5"
        )

        training_args = GRPOConfig(output_dir=self.tmp_dir, report_to="none")

        with self.assertRaises(ValueError) as context:
            GRPOTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                reward_funcs=reward_models,
                reward_processing_classes=single_processing_class,  # only one, but need two
                args=training_args,
                train_dataset=dataset,
            )

        self.assertIn("must match", str(context.exception))

    def test_correct_reward_processing_classes_list(self):
        """Test that correct list of reward_processing_classes works properly."""
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        # Use two reward models
        reward_models = [
            "trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            "trl-internal-testing/tiny-Qwen3ForSequenceClassification",
        ]

        # Create processing classes
        processing_class1 = AutoTokenizer.from_pretrained(
            "trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5"
        )
        processing_class2 = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-Qwen3ForSequenceClassification")

        training_args = GRPOConfig(output_dir=self.tmp_dir, report_to="none")

        # Correct list length should work
        correct_processing_classes = [processing_class1, processing_class2]

        trainer = GRPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=reward_models,
            reward_processing_classes=correct_processing_classes,
            args=training_args,
            train_dataset=dataset,
        )

        self.assertEqual(len(trainer.reward_processing_classes), len(reward_models))

    def test_single_reward_model_with_single_processing_class(self):
        """Test that single reward model with single processing class works."""
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        # Use single reward model
        reward_model = "trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5"

        # Create a single processing class (tokenizer)
        single_processing_class = AutoTokenizer.from_pretrained(
            "trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5"
        )

        training_args = GRPOConfig(output_dir=self.tmp_dir, report_to="none")

        trainer = GRPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=reward_model,
            reward_processing_classes=single_processing_class,  # single object for single reward model
            args=training_args,
            train_dataset=dataset,
        )

        self.assertEqual(len(trainer.reward_processing_classes), 1)
        self.assertEqual(trainer.reward_processing_classes[0], single_processing_class)


@pytest.mark.low_priority
class TestReplayBuffer(unittest.TestCase):
    def setUp(self):
        self.replay_buffer = ReplayBuffer(max_size=5)

    def test_add(self):
        # Add elements to the replay buffer
        scores = [0.5, 0.8, 0.3, 0.9, 0.7]
        data = [
            {"id": 1},
            {"id": 2},
            {"id": 3},
            {"id": 4},
            {"id": 5},
        ]
        self.replay_buffer.add(scores, data)

        # Check if the buffer contains the correct number of elements
        self.assertEqual(len(self.replay_buffer.heap), 5)

        # Check if the buffer maintains the min-heap property
        heap_scores = [item[0] for item in self.replay_buffer.heap]
        self.assertEqual(heap_scores[0], min(heap_scores))
        self.assertEqual(heap_scores[0], 0.3)

    def test_add_more_than_maxlen(self):
        # Add elements to the replay buffer
        scores = [0.5, 0.8, 0.3, 0.9, 0.7, 0.6, 0.4]
        data = [
            {"id": 1},
            {"id": 2},
            {"id": 3},
            {"id": 4},
            {"id": 5},
            {"id": 6},
            {"id": 7},
        ]
        self.replay_buffer.add(scores, data)

        # Check if the buffer contains the correct number of elements
        self.assertEqual(len(self.replay_buffer.heap), 5)

        # Check if the buffer maintains the min-heap property
        heap_scores = [item[0] for item in self.replay_buffer.heap]
        self.assertEqual(heap_scores[0], min(heap_scores))
        self.assertEqual(heap_scores[0], 0.5)  # 0.3 and 0.4 should be removed

    def test_sample(self):
        # Add elements to the replay buffer
        scores = [0.5, 0.8, 0.3, 0.9, 0.7]
        data = [
            {"id": 1},
            {"id": 2},
            {"id": 3},
            {"id": 4},
            {"id": 5},
        ]
        self.replay_buffer.add(scores, data)

        # Sample elements from the buffer
        sampled = self.replay_buffer.sample(num_samples=3)

        # Check if the sampled elements are from the buffer
        self.assertEqual(len(sampled), 3)
        for item in sampled:
            self.assertIn(item, [entry[1] for entry in self.replay_buffer.heap])


@pytest.mark.low_priority
class TestUpdateWithReplayBuffer(unittest.TestCase):
    def setUp(self):
        config = GRPOWithReplayBufferConfig(
            replay_buffer_size=5,
        )
        self.trainer = GRPOWithReplayBufferTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=config,
            train_dataset=None,
        )
        self.trainer.replay_buffer = ReplayBuffer(max_size=5)
        self.trainer.num_generations = 2

    def _prepopulate_buffer(self, with_pixels=False, with_logprobs=False):
        scores = [0.1, 0.9]
        data = [
            {
                "prompt_ids": torch.tensor([[100, 101], [102, 103]]),
                "prompt_mask": torch.ones(2, 2, dtype=torch.long),
                "completion_ids": torch.tensor([[5, 6], [7, 8]]),
                "completion_mask": torch.ones(2, 2, dtype=torch.long),
                "advantages": torch.tensor([[0.5, 0.6]]),
                **({"pixel_values": torch.randn(2, 3, 224, 224)} if with_pixels else {}),
                **({"old_per_token_logps": torch.randn(2, 2)} if with_logprobs else {}),
            },
            {
                "prompt_ids": torch.tensor([[104, 105], [106, 107]]),
                "prompt_mask": torch.ones(2, 2, dtype=torch.long),
                "completion_ids": torch.tensor([[13, 14], [15, 16]]),
                "completion_mask": torch.ones(2, 2, dtype=torch.long),
                "advantages": torch.tensor([[0.8, 0.85]]),
                **({"pixel_values": torch.randn(2, 3, 224, 224)} if with_pixels else {}),
                **({"old_per_token_logps": torch.randn(2, 2)} if with_logprobs else {}),
            },
        ]
        self.trainer.replay_buffer.add(scores, data)

    def _make_inputs(self, group_advantages, with_pixels=False, with_logprobs=False):
        inputs = {
            "group_advantages": group_advantages,
            "prompt_ids": torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]]),
            "prompt_mask": torch.ones(4, 2, dtype=torch.long),
            "completion_ids": torch.tensor([[9, 10], [11, 12], [13, 14], [15, 16]]),
            "completion_mask": torch.ones(4, 2, dtype=torch.long),
            "prompt_inputs": {"pixel_values": torch.randn(4, 3, 224, 224)} if with_pixels else {},
            "old_per_token_logps": torch.randn(4, 2) if with_logprobs else None,
        }
        inputs["group_std_rewards"] = group_advantages.std(dim=1).expand_as(group_advantages)
        return inputs

    def test_update_with_replay_buffer_no_variance(self):
        self._prepopulate_buffer(with_pixels=True, with_logprobs=True)
        group_advantages = torch.tensor([[0.5, 0.5], [0.8, 0.8]])  # no variance
        inputs = self._make_inputs(group_advantages, with_pixels=True, with_logprobs=True)
        original_prompt_ids = inputs["prompt_ids"].clone()

        outputs = self.trainer.update_with_replay_buffer(**inputs, num_items_in_batch=4)

        self.assertIsNotNone(outputs)
        self.assertIn("pixel_values", outputs)
        self.assertIn("old_per_token_logps", outputs)
        self.assertEqual(len(self.trainer.replay_buffer.heap), 2)
        for pid in outputs["prompt_ids"]:
            self.assertNotIn(pid.tolist(), original_prompt_ids.tolist())

    def test_update_with_replay_buffer_with_variance(self):
        self._prepopulate_buffer()
        group_advantages = torch.tensor([[0.6, 0.4], [0.7, 1.2]])  # has variance
        inputs = self._make_inputs(group_advantages)

        sampled = self.trainer.update_with_replay_buffer(**inputs, num_items_in_batch=4)

        self.assertEqual(len(self.trainer.replay_buffer.heap), 4)  # grew
        self.assertIsNone(sampled)

    def test_update_with_mixed_variance(self):
        self._prepopulate_buffer()
        group_advantages = torch.tensor([[0.6, 0.6], [0.3, 0.45]])  # one no-variance, one variance
        inputs = self._make_inputs(group_advantages)
        original_prompt_ids = inputs["prompt_ids"].clone().view(-1, self.trainer.num_generations, 2).tolist()

        outputs = self.trainer.update_with_replay_buffer(**inputs, num_items_in_batch=4)

        self.assertEqual(len(self.trainer.replay_buffer.heap), 3)  # grew by 1
        output_prompt_ids = outputs["prompt_ids"].view(-1, self.trainer.num_generations, 2).tolist()

        buffer_ids = [item[1]["prompt_ids"].tolist() for item in self.trainer.replay_buffer.heap]
        found_from_buffer = any(pid in buffer_ids for pid in output_prompt_ids)
        found_from_original = any(pid in original_prompt_ids for pid in output_prompt_ids)

        self.assertTrue(found_from_buffer)
        self.assertTrue(found_from_original)
        self.assertNotIn([[1, 2], [3, 4]], output_prompt_ids)  # excluded no-variance group

    def test_update_with_inputs_different_seq_len(self):
        """
        Test with inputs where the sequence lengths are different from the prepopulated buffer.
        """
        self._prepopulate_buffer()
        pad_token_id = self.trainer.tokenizer.pad_token_id
        group_advantages = torch.tensor([[0.6, 0.6], [0.3, 0.45]])  # one no-variance, one variance
        inputs = {
            "group_advantages": group_advantages,
            "prompt_ids": torch.tensor(
                [
                    [1, 2, pad_token_id],
                    [1, 2, pad_token_id],
                    [3, 4, 5],
                    [3, 4, 5],
                ]
            ),
            "prompt_mask": torch.tensor([[1, 1, 0], [1, 1, 0], [1, 1, 1], [1, 1, 1]], dtype=torch.long),
            "completion_ids": torch.tensor(
                [
                    [1009, 1010, pad_token_id],
                    [1011, 1012, 1013],
                    [1013, 1014, pad_token_id],
                    [1015, 1016, 1017],
                ]
            ),
            "completion_mask": torch.tensor([[1, 1, 0], [1, 1, 1], [1, 1, 0], [1, 1, 1]], dtype=torch.long),
            "prompt_inputs": {},
        }
        inputs["group_std_rewards"] = group_advantages.std(dim=1).expand_as(group_advantages)

        outputs_after_sampling = self.trainer.update_with_replay_buffer(**inputs, num_items_in_batch=4)
        # Seq length of current batch should be preserved
        self.assertEqual(outputs_after_sampling["prompt_ids"].shape[-1], 3)
        self.assertEqual(len(self.trainer.replay_buffer.heap), 3)
        output_prompt_ids = outputs_after_sampling["prompt_ids"].view(-1, self.trainer.num_generations, 3).tolist()

        buffered_prompt_completion_ids = [
            (item[1]["prompt_ids"].tolist(), item[1]["completion_ids"].tolist())
            for item in self.trainer.replay_buffer.heap
        ]
        buffered_prompt_ids, buffered_completion_ids = zip(*buffered_prompt_completion_ids)

        # Check for new entry with seq len 3 in buffer
        self.assertIn([[3, 4, 5], [3, 4, 5]], buffered_prompt_ids)  # excluded no-variance group
        self.assertIn(
            [[1013, 1014, pad_token_id], [1015, 1016, 1017]], buffered_completion_ids
        )  # excluded no-variance group

        # Check that sampled outputs contain one group with prompt_ids starting with a pad token
        self.assertTrue(
            [
                [pad_token_id, 101, 102],
                [pad_token_id, 102, 103],
            ]
            in output_prompt_ids
            or [
                [pad_token_id, 104, 105],
                [pad_token_id, 106, 107],
            ]
            in output_prompt_ids
        )


@pytest.mark.low_priority
class TestGRPOWithReplayBufferTrainer(TrlTestCase):
    def test_training_with_replay_buffer(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        # Guarantee that some rewards have 0 std
        def custom_reward_func(completions, **kwargs):
            if torch.rand(1).item() < 0.25:
                return [0] * len(completions)  # simulate some None rewards
            else:
                return torch.rand(len(completions)).tolist()

        training_args = GRPOWithReplayBufferConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # increase the learning rate to speed up the test
            per_device_train_batch_size=4,  # reduce the batch size to reduce memory usage
            num_generations=4,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            replay_buffer_size=8,
            report_to="none",
        )
        trainer = GRPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=[custom_reward_func],
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


class GSPOTokenTrainerTester(TrlTestCase):
    def test_training(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = GRPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # increase the learning rate to speed up the test
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            num_iterations=2,  # the importance sampling weights won't be 0 in this case
            importance_sampling_level="sequence_token",
            report_to="none",
        )
        trainer = GSPOTokenTrainer(
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


if __name__ == "__main__":
    unittest.main()
