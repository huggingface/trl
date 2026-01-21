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

from unittest.mock import patch

import pytest
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from transformers.utils import is_peft_available

from trl import RLOOConfig, RLOOTrainer

from .testing_utils import TrlTestCase, require_peft, require_vision, require_vllm


if is_peft_available():
    from peft import LoraConfig, PeftModel, get_peft_model


class TestRLOOTrainer(TrlTestCase):
    def test_init_minimal(self):
        # Test that RLOOTrainer can be instantiated with only model, reward_model and train_dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")
        RLOOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            train_dataset=dataset,
        )

    @pytest.mark.parametrize("config_name", ["standard_prompt_only", "conversational_prompt_only"])
    def test_training(self, config_name):
        dataset = load_dataset("trl-internal-testing/zen", config_name, split="train")

        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
        )
        trainer = RLOOTrainer(
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

        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            per_device_eval_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            eval_strategy="steps",
            eval_steps=2,
            report_to="none",
        )
        trainer = RLOOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
        )

        trainer.train()

    def test_training_with_num_generations_eval(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only")

        training_args = RLOOConfig(
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
        trainer = RLOOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
        )

        trainer.train()

    def test_training_multiple_iterations(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            num_iterations=2,
            report_to="none",
        )
        trainer = RLOOTrainer(
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
    def test_training_peft_config(self):
        model = AutoModelForCausalLM.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", dtype="float32")
        base_param_names = [f"base_model.model.{n}" for n, _ in model.named_parameters()]
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
        )
        trainer = RLOOTrainer(
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
    def test_training_peft_model(self):
        model = AutoModelForCausalLM.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", dtype="float32")
        base_param_names = [f"base_model.model.{n}" for n, _ in model.named_parameters()]
        lora_config = LoraConfig()
        model = get_peft_model(model, lora_config)
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # increase the learning rate to speed up the test
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
        )
        trainer = RLOOTrainer(
            model=model,
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the peft params have changed and the base model params have not changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if n in base_param_names:  # We expect the base model params to be the same
                assert torch.allclose(param, new_param), f"Parameter {n} has changed."
            elif "base_layer" not in n and "ref" not in n:  # and the peft params to be different (except base and ref)
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

        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,
            num_generations=3,
            max_completion_length=8,
            gradient_checkpointing=True,  # Enable gradient checkpointing
            report_to="none",
        )
        trainer = RLOOTrainer(
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

        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
        )
        trainer = RLOOTrainer(
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

        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
        )
        trainer = RLOOTrainer(
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

        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
        )
        trainer = RLOOTrainer(
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
        # Test that RLOOTrainer can be instantiated with multiple reward functions
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        def reward_func1(completions, **kwargs):
            """Reward function that rewards longer completions."""
            return [float(len(completion)) for completion in completions]

        def reward_func2(completions, **kwargs):
            """Reward function that rewards completions with more unique letters."""
            return [float(len(set(completion))) for completion in completions]

        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
        )
        trainer = RLOOTrainer(
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

    def test_training_sync_and_async_reward_funcs(self):
        # Test that RLOOTrainer can be instantiated with multiple reward functions one of which is async
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        def sync_reward_func1(completions, **kwargs):
            """Reward function that rewards longer completions."""
            return [float(len(completion)) for completion in completions]

        def sync_reward_func2(completions, **kwargs):
            return [1 for _ in completions]

        async def async_reward_func(completions, **kwargs):
            """Async Reward function that rewards completions with more unique letters."""
            return [float(len(set(completion))) for completion in completions]

        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,
            num_generations=3,
            max_completion_length=8,
            report_to="none",
        )
        trainer = RLOOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=[sync_reward_func1, sync_reward_func2, async_reward_func],
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

        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,
            num_generations=3,
            max_completion_length=8,
            report_to="none",
        )

        trainer = RLOOTrainer(
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
        """Test that RLOOTrainer can handle multiple reward functions with weights."""
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        def reward_func1(completions, **kwargs):
            """Reward function that rewards longer completions."""
            return [float(len(completion)) for completion in completions]

        def reward_func2(completions, **kwargs):
            """Reward function that rewards completions with more unique letters."""
            return [float(len(set(completion))) for completion in completions]

        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
            reward_weights=[0.7, 0.3],  # weight of reward_func1 and reward_func2 respectively
        )
        trainer = RLOOTrainer(
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

        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
        )
        trainer = RLOOTrainer(
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

        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
        )
        trainer = RLOOTrainer(
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

        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            sync_ref_model=True,
            ref_model_sync_steps=2,  # reduce sync steps to ensure a sync happens
            report_to="none",
        )
        trainer = RLOOTrainer(
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

    def test_training_beta_zero(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")
        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            beta=0.0,  # set beta to zero value to test the case where the reference model is not used
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
        )
        trainer = RLOOTrainer(
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

        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
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
        trainer = RLOOTrainer(
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
    def test_training_vllm_structured_outputs(self):
        """Test that training works with vLLM for generation with structured outputs."""
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
            use_vllm=True,
            vllm_structured_outputs_regex=r"<reasoning>\n.*\n</reasoning>\n<answer>\n.*\n</answer>",
        )
        trainer = RLOOTrainer(
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

        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
            top_p=0.9,
            top_k=10,
            min_p=0.01,
            repetition_penalty=1.1,
        )

        trainer = RLOOTrainer(
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

        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
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

        trainer = RLOOTrainer(
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

    def test_training_with_normalized_advantages(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            normalize_advantages=True,
            report_to="none",
        )
        trainer = RLOOTrainer(
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

    def test_training_with_clipped_rewards(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            reward_clip_range=(-1, 1),
            report_to="none",
        )
        trainer = RLOOTrainer(
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
            completion_ids = torch.tensor(
                [
                    [1, 2, 3, 4, 5, 6, 7, 8],  # this one is truncated
                    [9, 10, 11, 151645, 151643, 151643, 151643, 151643],  # this one contains eos
                    [12, 13, 14, 15, 16, 17, 18, 151645],  # particular case, eos is generated just within the limit
                ],
                device=input_ids.device,
            )
            return torch.cat([input_ids, completion_ids], dim=1)

        mock_generate.side_effect = fake_generate

        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            mask_truncated_completions=True,  # Enable masking of truncated completions
            report_to="none",
        )
        trainer = RLOOTrainer(
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

        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            mask_truncated_completions=True,  # Enable masking of truncated completions
            report_to="none",
        )
        trainer = RLOOTrainer(
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

        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
        )
        trainer = RLOOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=always_none_reward_func,
            args=training_args,
            train_dataset=dataset,
        )

        with caplog.at_level("WARNING", logger="trl.trainer.rloo_trainer"):
            trainer.train()

        expected_warning = "All reward functions returned None for the following kwargs:"
        assert expected_warning in caplog.text

    def test_training_num_generations_larger_than_batch_size(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            num_generations=6,  # the number of generations is larger than the batch size, but
            gradient_accumulation_steps=2,  # gradient accumulation should allow that
            report_to="none",
        )
        trainer = RLOOTrainer(
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

        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            dataloader_num_workers=2,  # use multiple dataloader workers
            report_to="none",
        )
        trainer = RLOOTrainer(
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

        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            # Pass gen kwargs
            generation_kwargs={"do_sample": True, "top_k": 50, "num_beams": 2, "length_penalty": -0.1},
            report_to="none",
        )
        trainer = RLOOTrainer(
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

        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,
            num_generations=2,
            max_completion_length=8,
            report_to="none",
        )
        trainer = RLOOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=reward_func,
            args=training_args,
            train_dataset=dataset,
        )
        trainer.train()

    def test_prepare_input_called_with_correct_data(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")
        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
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
        trainer = RLOOTrainer(
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

        with patch.object(RLOOTrainer, "training_step", wraps=trainer.training_step) as mock_prepare:
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
            "trl-internal-testing/tiny-Qwen2VLForConditionalGeneration",
            # "trl-internal-testing/tiny-SmolVLMForConditionalGeneration", seems not to support bf16 properly
        ],
    )
    @require_vision
    def test_training_vlm(self, model_id):
        dataset = load_dataset("trl-internal-testing/zen-image", "conversational_prompt_only", split="train")

        def reward_func(completions, **kwargs):
            """Reward function that rewards longer completions."""
            return [float(len(completion[0]["content"])) for completion in completions]

        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
        )
        trainer = RLOOTrainer(
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

        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            beta=0.1,  # set beta to non-zero value to test the case where the reference model is used
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
        )
        trainer = RLOOTrainer(
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
    @require_peft
    def test_training_vlm_peft(self, model_id):
        model = AutoModelForImageTextToText.from_pretrained(model_id)
        base_param_names = [f"base_model.model.{n}" for n, _ in model.named_parameters()]
        dataset = load_dataset("trl-internal-testing/zen-image", "conversational_prompt_only", split="train")

        def reward_func(completions, **kwargs):
            """Reward function that rewards longer completions."""
            return [float(len(completion[0]["content"])) for completion in completions]

        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
        )
        trainer = RLOOTrainer(
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

        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,
            num_generations=3,
            max_completion_length=8,
            report_to="none",
            use_vllm=True,
            vllm_mode="server",
        )
        trainer = RLOOTrainer(
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

        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
        )
        trainer = RLOOTrainer(
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

    def test_training_with_chat_template_kwargs(self):
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_only", split="train")

        training_args = RLOOConfig(
            bf16=False,
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,
            num_generations=3,
            max_completion_length=8,
            report_to="none",
            chat_template_kwargs={"enable_thinking": False},
        )
        trainer = RLOOTrainer(
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

        training_args = RLOOConfig(output_dir=self.tmp_dir, report_to="none")

        with pytest.raises(ValueError, match="must match"):
            RLOOTrainer(
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

        training_args = RLOOConfig(output_dir=self.tmp_dir, report_to="none")

        # Correct list length should work
        correct_processing_classes = [processing_class1, processing_class2]

        trainer = RLOOTrainer(
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

        training_args = RLOOConfig(output_dir=self.tmp_dir, report_to="none")

        trainer = RLOOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=reward_model,
            reward_processing_classes=single_processing_class,  # single object for single reward model
            args=training_args,
            train_dataset=dataset,
        )

        assert len(trainer.reward_processing_classes) == 1
        assert trainer.reward_processing_classes[0] == single_processing_class
