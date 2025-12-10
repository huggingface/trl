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

import gc
import os
import warnings
from unittest.mock import patch

import numpy as np
import pytest
import torch
import transformers
from accelerate.utils.memory import release_memory
from datasets import Dataset, Features, Image, Value, load_dataset
from packaging.version import Version
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from transformers.testing_utils import backend_empty_cache, torch_device
from transformers.utils import is_peft_available

from trl import GRPOConfig, GRPOTrainer
from trl.trainer.utils import get_kbit_device_map

from .testing_utils import (
    TrlTestCase,
    require_ampere_or_newer,
    require_bitsandbytes,
    require_kernels,
    require_liger_kernel,
    require_peft,
    require_torch_accelerator,
    require_vision,
    require_vllm,
)


if is_peft_available():
    from peft import LoraConfig, PeftModel


class TestGetHighEntropyMask(TrlTestCase):
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


class TestGRPOTrainer(TrlTestCase):
    def test_init_minimal(self):
        # Test that GRPOTrainer can be instantiated with only model, reward_model and train_dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")
        GRPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            train_dataset=dataset,
        )

    @pytest.mark.parametrize("config_name", ["standard_prompt_only", "conversational_prompt_only"])
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

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_get_sapo_token_loss(self):
        sample_token_importance_ratio = torch.ones((2, 4))
        sapo_token_loss = GRPOTrainer.get_sapo_token_loss(sample_token_importance_ratio, 1.0)
        # sigmoid(temp * (1-1)) * 4/temp = 0.5 * 4 = 2
        expected_sapo_token_loss = torch.full_like(sample_token_importance_ratio, 2.0)
        torch.testing.assert_close(sapo_token_loss, expected_sapo_token_loss)

    @pytest.mark.parametrize("loss_type", ["bnpo", "dr_grpo", "dapo", "cispo", "sapo"])
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

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

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

    def test_training_with_num_generations_eval(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only")

        training_args = GRPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            per_device_eval_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            num_generations_eval=1,
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

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

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

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the peft params have changed and the base model params have not changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if n in base_param_names:  # We expect the base model params to be the same
                assert torch.allclose(param, new_param), f"Parameter {n} has changed."
            elif "base_layer" not in n:  # We expect the peft params to be different (except for the base layer)
                assert not torch.allclose(param, new_param), f"Parameter {n} has not changed."

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
        assert isinstance(trainer.model, PeftModel)

        # Store initial parameters to check which ones change
        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that only LoRA parameters have changed, base model parameters remain unchanged
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if "lora" in n.lower():  # LoRA parameters should change
                assert not torch.equal(param, new_param), f"LoRA parameter {n} has not changed."
            else:  # Base model parameters should not change
                assert torch.equal(param, new_param), f"Base parameter {n} has changed."

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

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

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

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

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

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

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

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

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

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

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
        assert trainer.state.log_history[-1]["train_loss"] is not None
        assert "rewards/reward_func1/mean" in trainer.state.log_history[-1]
        assert "rewards/reward_func1/std" in trainer.state.log_history[-1]
        assert "rewards/reward_func2/mean" in trainer.state.log_history[-1]
        assert "rewards/reward_func2/std" in trainer.state.log_history[-1]

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

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

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_training_reward_func_additional_column(self):
        # Test if trainer can handle reward function that rely on additional columns in the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        # Add a column to the dataset (dummy example, the column could be anything)
        some_values = list(range(len(dataset)))
        dataset = dataset.add_column("some_values", some_values)

        def reward_func(completions, some_values, **kwargs):
            """Reward function that rewards completions with lengths closer to the values in some_values."""
            return [
                float(abs(len(completion) - value)) for completion, value in zip(completions, some_values, strict=True)
            ]

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

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

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

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

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

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_training_with_bias_correction_kl(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")
        training_args = GRPOConfig(
            output_dir=self.tmp_dir,
            beta=0.1,  # set beta to non-zero value to test the case where the reference model is used
            use_bias_correction_kl=True,
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

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    @pytest.mark.parametrize(
        "model_name",
        ["trl-internal-testing/tiny-Qwen3ForCausalLM", "trl-internal-testing/tiny-Gemma2ForCausalLM"],
        # Gemma2 has the input word embeddings and lm_head tied, Qwen3 does not
    )
    def test_training_with_cast_lm_head_to_fp32(self, model_name):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")
        training_args = GRPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=3,
            num_generations=3,
            max_completion_length=8,
            report_to="none",
            cast_lm_head_to_fp32=True,
        )
        trainer = GRPOTrainer(
            model=model_name,
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset,
        )
        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None
        assert trainer.model.lm_head.weight.dtype == torch.float32

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

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

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    @require_peft
    @require_vllm
    @pytest.mark.skip(reason="We should add a mock for the vLLM server.")
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

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the peft params have changed and the base model params have not changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if n in base_param_names:  # We expect the base model params to be the same
                assert torch.allclose(param, new_param), f"Parameter {n} has changed."
            elif "base_layer" not in n and "original_module" not in n:
                # We expect the peft params to be different (except for the base layer)
                assert not torch.allclose(param, new_param), f"Parameter {n} has not changed."

    @require_vllm
    @pytest.mark.skip(reason="We should add a mock for the vLLM server.")
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

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    @require_vllm
    @pytest.mark.skip(reason="We should add a mock for the vLLM server.")
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

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

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

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    @require_vllm
    @pytest.mark.skip(reason="We should add a mock for the vLLM server.")
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

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    @pytest.mark.parametrize("scale_rewards", [False, "group", "batch", True, "none"])
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

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

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

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

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

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert torch.equal(param, new_param), f"Parameter {n} has changed."

    def test_warning_raised_all_rewards_none(self, caplog):
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

        with caplog.at_level("WARNING", logger="trl.trainer.grpo_trainer"):
            trainer.train()

        expected_warning = "All reward functions returned None for the following kwargs:"
        assert expected_warning in caplog.text

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

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

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

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

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

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

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

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

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
            assert mock_prepare.call_count == 48
            for i in range(0, 8):  # Generation batch repeated 8 times (steps_per_generation*num_iterations)
                assert mock_prepare.call_args_list[i].args[1] == expected_first_generation_batch
            for i in range(8, 16):
                assert mock_prepare.call_args_list[i].args[1] == expected_second_generation_batch

    @pytest.mark.parametrize(
        "model_id",
        [
            "trl-internal-testing/tiny-Gemma3ForConditionalGeneration",
            "trl-internal-testing/tiny-LlavaNextForConditionalGeneration",
            "trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration",
            pytest.param(
                "trl-internal-testing/tiny-Qwen2VLForConditionalGeneration",
                marks=pytest.mark.xfail(reason="Blocked by upstream bug in transformers#42762", strict=True),
            ),
            # "trl-internal-testing/tiny-SmolVLMForConditionalGeneration", seems not to support bf16 properly
        ],
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

        assert trainer.state.log_history[-1]["train_loss"] is not None

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
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    @pytest.mark.parametrize(
        "model_id",
        [
            "trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration",
        ],
    )
    @require_vision
    def test_training_vlm_beta_non_zero(self, model_id):
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
            model=model_id,
            reward_funcs=reward_func,
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        # Because of the way the tiny models are initialized, the gradient does not flow properly through the
        # vision parts of the model, so we skip them. Ideally, we should fix the init of these models.
        params_to_skip = ("model.visual.",)
        for n, param in previous_trainable_params.items():
            if n.startswith(params_to_skip):
                continue
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    @pytest.mark.parametrize(
        "model_id",
        [
            pytest.param(
                "trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration",
                marks=pytest.mark.xfail(reason="Blocked by upstream bug in transformers#42762", strict=True),
            ),
        ],
    )
    @require_vision
    @require_peft
    def test_training_vlm_peft(self, model_id):
        model = AutoModelForImageTextToText.from_pretrained(model_id)
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

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the peft params have changed and the base model params have not changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if n in base_param_names:  # We expect the base model params to be the same
                assert torch.allclose(param, new_param), f"Parameter {n} has changed."
            elif "base_layer" not in n:  # We expect the peft params to be different (except for the base layer)
                assert not torch.allclose(param, new_param), f"Parameter {n} has not changed."

    @pytest.mark.parametrize(
        "model_id",
        [
            "trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration",
        ],
    )
    @require_vision
    def test_training_vlm_and_importance_sampling(self, model_id):
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
            model=model_id,
            reward_funcs=reward_func,
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        # Because of the way the tiny models are initialized, the gradient does not flow properly through the
        # vision parts of the model, so we skip them. Ideally, we should fix the init of these models.
        params_to_skip = ("model.visual.",)
        for n, param in previous_trainable_params.items():
            if n.startswith(params_to_skip):
                continue
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    @pytest.mark.parametrize(
        "model_id",
        [
            "trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration",
        ],
    )
    @require_vision
    @require_liger_kernel
    def test_training_vlm_and_liger(self, model_id):
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
            use_liger_kernel=True,  # enable Liger kernel
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

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        # Because of the way the tiny models are initialized, the gradient does not flow properly through the
        # vision parts of the model, so we skip them. Ideally, we should fix the init of these models.
        params_to_skip = ("model.visual.",)
        for n, param in previous_trainable_params.items():
            if n.startswith(params_to_skip):
                continue
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    @pytest.mark.parametrize(
        "model_id",
        [
            "trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration",
            "trl-internal-testing/tiny-Gemma3ForConditionalGeneration",
        ],
    )
    @require_vision
    @require_vllm
    @pytest.mark.skip(reason="We should add a mock for the vLLM server.")
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

        assert trainer.state.log_history[-1]["train_loss"] is not None

        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    @pytest.mark.parametrize(
        "model_id",
        [
            "trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration",
        ],
    )
    @require_vision
    def test_training_vlm_multi_image(self, model_id):
        dataset = load_dataset("trl-internal-testing/zen-multi-image", "conversational_prompt_only", split="train")

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
            model=model_id,
            reward_funcs=reward_func,
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        # Because of the way the tiny models are initialized, the gradient does not flow properly through the
        # vision parts of the model, so we skip them. Ideally, we should fix the init of these models.
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

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

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_training_with_chat_template_kwargs(self):
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_only", split="train")

        training_args = GRPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=3,
            num_generations=3,
            max_completion_length=8,
            report_to="none",
            chat_template_kwargs={"enable_thinking": False},
        )
        trainer = GRPOTrainer(
            model="trl-internal-testing/tiny-Qwen3ForCausalLM",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    @pytest.mark.xfail(
        condition=Version(transformers.__version__) < Version("5.0.0.dev0"),
        reason="Tool parsing is not supported in transformers versions below 5.0.0",
        strict=True,
    )
    def test_training_with_tools(self):
        # In this test, we define a simple tool that multiplies two integers. Regardless of the input prompt,
        # the model will generate 3 completions, 2 of which will be valid tool calls. Among the 2 tool calls, one will
        # succeed and the other will fail (because of a wrong argument name).
        def multiply(a: int, b: int) -> int:
            """
            Multiplies two integers.

            Args:
                a: The first integer.
                b: The second integer.

            Returns:
                The product of the two integers.
            """
            return a * b

        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_only", split="train")

        training_args = GRPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=3,
            num_generations=3,
            max_completion_length=128,
            report_to="none",
        )
        trainer = GRPOTrainer(
            model="trl-internal-testing/tiny-Qwen3MoeForCausalLM",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset,
            tools=[multiply],
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        def fake_generate(input_ids, **kwargs):
            if input_ids.shape[0] == 3:  # first call
                # fmt: off
                completion_ids = torch.tensor(
                    [
                        # '<tool_call>\n{"name": "multiply", "arguments": {"a": 3, "b": 4}}\n</tool_call><|im_end|>'
                        [151657, 198, 4913, 606, 788, 330, 64648, 497, 330, 16370, 788, 5212, 64, 788, 220, 18, 11, 330, 65, 788, 220, 19, 11248, 151658, 151645],
                        # '<tool_call>\n{"name": "multiply", "arguments": {"a": 3, "c": 4}}\n</tool_call><|im_end|>'
                        [151657, 198, 4913, 606, 788, 330, 64648, 497, 330, 16370, 788, 5212, 64, 788, 220, 18, 11, 330, 66, 788, 220, 19, 11248, 151658, 151645],
                        # "I don't know any tool<|im_end|>"
                        [40, 1513, 944, 1414, 894, 5392, 151645, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643],
                    ],
                    device=input_ids.device,
                )
                # fmt: on
            else:  # second call will only have two inputs in the batch, because two examples have a tool call.
                completion_ids = torch.tensor(
                    [
                        # 'Done!<|im_end|>'
                        [17453, 0, 151645],
                        # 'Done!<|im_end|>'
                        [17453, 0, 151645],
                    ],
                    device=input_ids.device,
                )
            return torch.cat([input_ids, completion_ids], dim=-1)

        with patch.object(trainer.model, "generate", side_effect=fake_generate):
            trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None
        assert trainer.state.log_history[-1]["tools/call_frequency"] is not None
        assert trainer.state.log_history[-1]["tools/call_frequency"] == pytest.approx(2 / 3)
        assert trainer.state.log_history[-1]["tools/failure_frequency"] is not None
        assert trainer.state.log_history[-1]["tools/failure_frequency"] == pytest.approx(1 / 2)

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

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

        with pytest.raises(ValueError, match="must match"):
            GRPOTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                reward_funcs=reward_models,
                reward_processing_classes=single_processing_class,  # only one, but need two
                args=training_args,
                train_dataset=dataset,
            )

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

        assert len(trainer.reward_processing_classes) == len(reward_models)

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

        assert len(trainer.reward_processing_classes) == 1
        assert trainer.reward_processing_classes[0] == single_processing_class


@pytest.mark.slow
@require_torch_accelerator
class TestGRPOTrainerSlow(TrlTestCase):
    def setup_method(self):
        self.train_dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")
        self.eval_dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="test")
        self.max_length = 128

    def teardown_method(self):
        gc.collect()
        backend_empty_cache(torch_device)
        gc.collect()

    @pytest.mark.parametrize(
        "model_name",
        [
            "trl-internal-testing/tiny-LlamaForCausalLM-3.2",
            "trl-internal-testing/tiny-MistralForCausalLM-0.2",
        ],
    )
    @require_liger_kernel
    def test_training_with_liger_grpo_kernel(self, model_name):
        training_args = GRPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=3,
            num_generations=3,
            use_liger_kernel=True,
            max_completion_length=self.max_length,
            report_to="none",
            logging_strategy="no",
        )

        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token

        trainer = GRPOTrainer(
            model=model,
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            processing_class=tokenizer,
        )
        from liger_kernel.chunked_loss import LigerFusedLinearGRPOLoss

        assert isinstance(trainer.liger_grpo_loss, LigerFusedLinearGRPOLoss)

        previous_trainable_params = {n: param.clone() for n, param in model.named_parameters()}

        trainer.train()

        for n, param in previous_trainable_params.items():
            new_param = model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

        release_memory(model, trainer)

    @pytest.mark.parametrize(
        "model_name",
        [
            "trl-internal-testing/tiny-LlamaForCausalLM-3.2",
            "trl-internal-testing/tiny-MistralForCausalLM-0.2",
        ],
    )
    @require_liger_kernel
    @require_peft
    def test_training_with_liger_grpo_kernel_and_peft(self, model_name):
        from peft import LoraConfig, TaskType

        training_args = GRPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=3,
            num_generations=3,
            use_liger_kernel=True,
            max_completion_length=self.max_length,
            report_to="none",
            logging_strategy="no",
        )

        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token

        # Configure PEFT with LoRA
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"],
        )

        trainer = GRPOTrainer(
            model=model,
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            processing_class=tokenizer,
            peft_config=peft_config,
        )
        from liger_kernel.chunked_loss import LigerFusedLinearGRPOLoss

        assert isinstance(trainer.liger_grpo_loss, LigerFusedLinearGRPOLoss)

        # Verify PEFT adapter is properly initialized
        from peft import PeftModel

        assert isinstance(trainer.model, PeftModel), "Model should be wrapped with PEFT"

        # Store adapter weights before training
        previous_trainable_params = {
            n: param.clone() for n, param in trainer.model.named_parameters() if param.requires_grad
        }
        assert len(previous_trainable_params) > 0, "No trainable parameters found in PEFT model"

        trainer.train()

        # Verify adapter weights have changed after training
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

        release_memory(model, trainer)

    @pytest.mark.parametrize(
        "model_name",
        [
            "trl-internal-testing/tiny-LlamaForCausalLM-3.2",
            "trl-internal-testing/tiny-MistralForCausalLM-0.2",
        ],
    )
    def test_training_with_transformers_paged(self, model_name):
        """Test that training works with transformers paged implementation (requires GPU)."""
        if Version(transformers.__version__) < Version("4.57.0"):
            pytest.xfail("Upstream bug in transformers (GH#40692). Fix merged; awaiting release >= 4.57.0")
        training_args = GRPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # increase the learning rate to speed up the test
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            use_transformers_paged=True,  # Enable transformers paged implementation
            report_to="none",
            logging_strategy="no",
        )

        model = AutoModelForCausalLM.from_pretrained(model_name)

        trainer = GRPOTrainer(
            model=model,
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=self.train_dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

        release_memory(model, trainer)

    @pytest.mark.parametrize(
        "model_name",
        [
            "HuggingFaceTB/SmolVLM-Instruct",  # Only test the smaller model to avoid OOM
        ],
    )
    @require_kernels
    @require_ampere_or_newer  # Flash attention 2 requires Ampere or newer GPUs
    @require_bitsandbytes
    @require_peft
    def test_vlm_training(self, model_name):
        """
        Test VLM training with aggressive memory optimization.

        This test uses multiple memory reduction techniques:
        - 4-bit quantization with double quantization
        - LoRA with very low rank (r=4)
        - Minimal batch size (1) with gradient accumulation
        - Small images (64x64 instead of 224x224)
        - Short sequences (max_completion_length=8)
        - Only 4 training samples
        - Only 1 training step
        - Gradient checkpointing and bfloat16
        """

        # Create processor once outside the data generator
        processor = AutoProcessor.from_pretrained(model_name, use_fast=True, padding_side="left")
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is in the image?"},
                ],
            },
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

        def data_gen(num_samples):
            for _ in range(num_samples):
                yield {
                    "prompt": prompt,
                    "image": np.random.uniform(low=0.0, high=255.0, size=(64, 64, 3)).astype(
                        np.uint8
                    ),  # Much smaller images
                }

        dataset = Dataset.from_generator(
            data_gen, gen_kwargs={"num_samples": 4}, features=Features(image=Image(), prompt=Value(dtype="string"))
        )
        # reduce memory requirements as much as possible
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype="bfloat16",
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_storage="bfloat16",
        )
        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            attn_implementation="kernels-community/flash-attn2",
            dtype="bfloat16",
            device_map=get_kbit_device_map(),
            quantization_config=quantization_config,
        )

        def reward_func(prompts, completions, **kwargs):
            # simple nonsensical reward
            return [-((len(c) - 25) ** 2) + 100 for c in completions]

        training_args = GRPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=1,  # Minimal batch size
            gradient_accumulation_steps=2,  # Maintain effective batch size
            num_generations=2,
            max_completion_length=8,  # Much shorter completions
            bf16=True,  # Use bfloat16 precision
            max_steps=1,  # Only do 1 training step to save time and memory
            report_to="none",
            logging_strategy="no",
        )
        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=4,  # Much lower rank for minimal memory
            lora_alpha=8,  # Reduced alpha proportionally
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"],  # Minimal target modules
            # For VLM models, we typically want to freeze the vision encoder
            # and only adapt the language model parameters
            modules_to_save=None,
        )

        try:
            trainer = GRPOTrainer(
                model=model,
                processing_class=processor,
                reward_funcs=[reward_func],
                args=training_args,
                train_dataset=dataset,
                peft_config=lora_config,
            )

            assert isinstance(trainer.model, PeftModel)

            previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

            trainer.train()

            assert trainer.state.log_history[-1]["train_loss"] is not None

            # Check that LoRA parameters have changed
            # For VLM models, we're more permissive about which parameters can change
            lora_params_changed = False
            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                if "lora" in n.lower():  # LoRA parameters should change
                    if not torch.equal(param, new_param):
                        lora_params_changed = True

            # At least some LoRA parameters should have changed during training
            assert lora_params_changed, "No LoRA parameters were updated during training."

        except torch.OutOfMemoryError as e:
            pytest.skip(f"Skipping VLM training test due to insufficient GPU memory: {e}")
        except Exception as e:
            # Check for other memory-related errors
            if any(keyword in str(e).lower() for keyword in ["memory", "cuda", "out of memory", "insufficient"]):
                pytest.skip(f"Skipping VLM training test due to hardware constraints: {e}")
            else:
                raise

        release_memory(model, trainer)

    @require_vllm
    @require_bitsandbytes
    @require_peft
    def test_vlm_processor_vllm_colocate_mode(self):
        """
        Test that VLM processors work with vLLM in colocate mode.

        This test uses multiple memory optimization techniques to ensure it runs on limited hardware:
        - LoRA (Low-Rank Adaptation) with minimal rank (r=4)
        - 4-bit quantization with BitsAndBytesConfig
        - Gradient checkpointing
        - bfloat16 precision
        - Minimal batch sizes and sequence lengths
        - Very low GPU memory utilization (5%)
        """
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        config = GRPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=1,  # Minimal batch size
            gradient_accumulation_steps=2,  # Make effective batch size 2, divisible by num_generations
            num_generations=2,
            max_completion_length=4,  # Very short completions to reduce memory
            use_vllm=True,  # Enable vLLM
            vllm_mode="colocate",  # Use colocate mode to avoid server dependency
            vllm_gpu_memory_utilization=0.05,  # Use minimal GPU memory (5%)
            gradient_checkpointing=True,  # Enable gradient checkpointing to save memory
            bf16=True,  # Use bfloat16 to reduce memory
            report_to="none",
            logging_strategy="no",
        )

        # Create a VLM processor
        processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-Instruct", use_fast=True, padding_side="left")

        # Verify processor has both required attributes for VLM detection
        assert hasattr(processor, "tokenizer")
        assert hasattr(processor, "image_processor")

        def dummy_reward_func(completions, **kwargs):
            return [1.0] * len(completions)

        # Use LoRA configuration for memory efficiency
        lora_config = LoraConfig(
            r=4,  # Very low rank for minimal memory
            lora_alpha=8,
            target_modules=["q_proj", "v_proj"],  # Minimal target modules
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # Use 4-bit quantization for further memory reduction
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        original_env = {}
        required_env_vars = {
            "RANK": "0",
            "LOCAL_RANK": "0",
            "WORLD_SIZE": "1",
            "LOCAL_WORLD_SIZE": "1",
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "12355",
        }

        for key, value in required_env_vars.items():
            original_env[key] = os.environ.get(key)
            os.environ[key] = value

        try:
            # Test VLM processor with vLLM colocate mode
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                try:
                    # Load model with quantization for memory efficiency
                    model = AutoModelForCausalLM.from_pretrained(
                        "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                        quantization_config=quantization_config,
                        dtype=torch.bfloat16,
                    )

                    trainer = GRPOTrainer(
                        model=model,
                        reward_funcs=dummy_reward_func,
                        args=config,
                        train_dataset=dataset,
                        processing_class=processor,  # VLM processor
                        peft_config=lora_config,  # Use LoRA for memory efficiency
                    )

                    # Should detect VLM processor correctly and allow vLLM
                    assert trainer.use_vllm, "vLLM should be enabled for VLM processors in colocate mode"
                    assert trainer.vllm_mode == "colocate", "Should use colocate mode"

                    # Check if signature columns were set properly
                    if trainer._signature_columns is not None:
                        # Should include 'image' in signature columns for VLM processors
                        assert "image" in trainer._signature_columns, (
                            "Should include 'image' in signature columns for VLM"
                        )

                    # Should not emit any warnings about VLM incompatibility
                    incompatibility_warnings = [
                        str(w_item.message)
                        for w_item in w
                        if "does not support VLMs" in str(w_item.message)
                        or "not compatible" in str(w_item.message).lower()
                    ]
                    assert len(incompatibility_warnings) == 0, (
                        f"Should not emit VLM incompatibility warnings, but got: {incompatibility_warnings}"
                    )

                    # Test passes if we get this far without exceptions

                except Exception as e:
                    # If vLLM fails to initialize due to hardware constraints or other issues, that's expected
                    if any(
                        keyword in str(e).lower()
                        for keyword in [
                            "outofmemoryerror",
                            "cuda",
                            "memory",
                            "insufficient",
                            "no such device",
                            "free memory",
                            "gpu memory utilization",
                            "decrease gpu memory",
                        ]
                    ):
                        pytest.skip(f"Skipping vLLM colocate test due to hardware constraints: {e}")
                    elif "KeyError" in str(e) and "RANK" in str(e):
                        pytest.skip(f"Skipping vLLM colocate test due to environment setup issues: {e}")
                    elif "ValueError" in str(e) and "memory" in str(e).lower():
                        pytest.skip(f"Skipping vLLM colocate test due to memory constraints: {e}")
                    else:
                        raise
        finally:
            # Restore original environment variables
            for key, original_value in original_env.items():
                if original_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = original_value

            release_memory(model, trainer)

    @require_vllm
    def test_training_vllm(self):
        """Test that training works with vLLM for generation."""
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = GRPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # increase the learning rate to speed up the test
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
            logging_strategy="no",
            use_vllm=True,
        )

        try:
            trainer = GRPOTrainer(
                model="Qwen/Qwen2.5-0.5B-Instruct",  # tiny models are too small for vLLM
                reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
                args=training_args,
                train_dataset=dataset,
            )

            previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

            trainer.train()

            assert trainer.state.log_history[-1]["train_loss"] is not None

            # Check that the params have changed
            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

        except Exception as e:
            # If vLLM fails to initialize due to hardware constraints or other issues, that's expected
            if any(
                keyword in str(e).lower()
                for keyword in [
                    "outofmemoryerror",
                    "cuda",
                    "memory",
                    "insufficient",
                    "no such device",
                    "free memory",
                    "gpu memory utilization",
                    "decrease gpu memory",
                ]
            ):
                pytest.skip(f"Skipping vLLM training test due to hardware constraints: {e}")
            elif "KeyError" in str(e) and "RANK" in str(e):
                pytest.skip(f"Skipping vLLM training test due to environment setup issues: {e}")
            elif "ValueError" in str(e) and "memory" in str(e).lower():
                pytest.skip(f"Skipping vLLM training test due to memory constraints: {e}")
            else:
                raise

        release_memory(trainer.model, trainer)
