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
import os
import tempfile
import unittest
import warnings
from unittest.mock import patch

import numpy as np
import torch
from datasets import Dataset, Features, Image, Value, load_dataset
from parameterized import parameterized
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from transformers.testing_utils import (
    require_bitsandbytes,
    require_flash_attn,
    require_peft,
    require_torch_accelerator,
)
from transformers.utils import is_peft_available

from trl import GRPOConfig, GRPOTrainer
from trl.trainer.grpo_trainer import RepeatSampler, shuffle_tensor_dict, split_tensor_dict
from trl.trainer.utils import get_kbit_device_map

from .testing_utils import require_vllm


if is_peft_available():
    from peft import LoraConfig, PeftModel


class SplitTensorDictTester(unittest.TestCase):
    def test_split_equal_chunks(self):
        x = torch.arange(12).reshape(6, 2)
        y = torch.arange(6).reshape(6, 1)
        tensor_dict = {"x": x, "y": y}

        result = split_tensor_dict(tensor_dict, 3)

        expected_x_chunks = torch.chunk(x, 3, dim=0)
        expected_y_chunks = torch.chunk(y, 3, dim=0)
        self.assertEqual(len(result), 3)
        for i in range(3):
            self.assertTrue(torch.equal(result[i]["x"], expected_x_chunks[i]))
            self.assertTrue(torch.equal(result[i]["y"], expected_y_chunks[i]))

    def test_with_none_tensor(self):
        x = torch.arange(12).reshape(6, 2)
        tensor_dict = {"x": x, "y": None}

        result = split_tensor_dict(tensor_dict, 2)

        expected_x_chunks = torch.chunk(x, 2, dim=0)
        self.assertEqual(len(result), 2)
        for i in range(2):
            self.assertTrue(torch.equal(result[i]["x"], expected_x_chunks[i]))
            self.assertIsNone(result[i]["y"])


class ShuffleTensorDictTester(unittest.TestCase):
    def test_shuffle_preserves_shape(self):
        x = torch.arange(6).reshape(3, 2)
        y = torch.arange(3).reshape(3, 1)
        tensor_dict = {"x": x.clone(), "y": y.clone()}

        shuffled = shuffle_tensor_dict(tensor_dict)

        self.assertEqual(shuffled["x"].shape, x.shape)
        self.assertEqual(shuffled["y"].shape, y.shape)

    def test_shuffle_consistent_across_tensors(self):
        # Use known patterns to check alignment
        x = torch.tensor([[10, 11], [20, 21], [30, 31]])
        y = torch.tensor([[1], [2], [3]])
        tensor_dict = {"x": x.clone(), "y": y.clone()}

        shuffled = shuffle_tensor_dict(tensor_dict)

        # Build a reverse map from shuffled x rows to y values
        for i in range(3):
            x_row = shuffled["x"][i]
            y_val = shuffled["y"][i].item()

            if torch.equal(x_row, torch.tensor([10, 11])):
                self.assertEqual(y_val, 1)
            elif torch.equal(x_row, torch.tensor([20, 21])):
                self.assertEqual(y_val, 2)
            elif torch.equal(x_row, torch.tensor([30, 31])):
                self.assertEqual(y_val, 3)
            else:
                self.fail("Unexpected x row in shuffled output.")

    def test_none_tensor_remains_none(self):
        x = torch.arange(6).reshape(3, 2)
        tensor_dict = {"x": x.clone(), "y": None}

        shuffled = shuffle_tensor_dict(tensor_dict)

        self.assertIsNone(shuffled["y"])
        self.assertEqual(shuffled["x"].shape, x.shape)


class RepeatRandomSamplerTester(unittest.TestCase):
    def test_sampler(self):
        dataset = ["a", "b", "c", "d", "e", "f", "g"]
        sampler = RepeatSampler(dataset, mini_repeat_count=2)
        # Should output something like [4, 4, 3, 3, 0, 0, 1, 1, 2, 2, 6, 6, 5, 5]
        sampled = list(sampler)
        # Check that the length is doubled
        assert len(sampled) == 2 * len(dataset)
        # Check that all indexes are present
        assert set(sampled) == set(range(len(dataset)))
        # Check that each element is repeated twice
        assert all(sampled[i] == sampled[i + 1] for i in range(0, len(sampled), 2))

    def test_sampler_no_shuffle(self):
        dataset = ["a", "b", "c", "d", "e", "f", "g"]
        sampler = RepeatSampler(dataset, mini_repeat_count=2, shuffle=False)
        sampled = list(sampler)
        expected = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]
        self.assertEqual(sampled, expected)

    def test_sampler_no_repeat(self):
        dataset = ["a", "b", "c", "d", "e", "f", "g"]
        sampler = RepeatSampler(dataset, mini_repeat_count=1)
        # Should output something like [4, 3, 0, 1, 2, 6, 5]
        sampled = list(sampler)
        # Check that the length is the same
        assert len(sampled) == len(dataset)
        # Check that all indexes are present
        assert set(sampled) == set(range(len(dataset)))

    def test_sampler_with_batch_size(self):
        dataset = ["a", "b", "c", "d", "e", "f", "g", "h"]
        sampler = RepeatSampler(dataset, mini_repeat_count=1, batch_size=2, repeat_count=2)
        # Should output something like [4, 3, 4, 3, 0, 1, 0, 1, 2, 6, 2, 6, 5, 7, 5, 7]
        sampled = list(sampler)
        # Check that the length is doubled
        assert len(sampled) == 2 * len(dataset)
        # Check that all indexes are present
        assert set(sampled) == set(range(len(dataset)))
        # Check that each element is repeated as expected
        assert all(sampled[i : i + 1] == sampled[i + 2 : i + 3] for i in range(0, len(sampled), 4))

    def test_sampler_with_batch_size_and_drop(self):
        dataset = ["a", "b", "c", "d", "e", "f", "g"]
        sampler = RepeatSampler(dataset, mini_repeat_count=1, batch_size=2, repeat_count=2)
        # Should output something like [4, 3, 4, 3, 0, 1, 0, 1, 2, 6, 2, 6]
        sampled = list(sampler)
        # Check that the length is doubled
        assert len(sampled) == 2 * (
            len(dataset) - 1
        )  # one element is dropped, because it's not enough to form a batch
        # Check that the sampled indexes are a subset of the dataset indexes
        assert set(sampled).issubset(set(range(len(dataset))))
        # Check that each element is repeated as expected
        assert all(sampled[i : i + 1] == sampled[i + 2 : i + 3] for i in range(0, len(sampled), 4))

    def test_sampler_with_mini_repeat_count_and_batch_size_1(self):
        dataset = ["a", "b", "c", "d", "e", "f", "g"]
        sampler = RepeatSampler(dataset, mini_repeat_count=2, batch_size=3, repeat_count=2)
        # Should output something like [4, 4, 3, 3, 0, 0, 4, 4, 3, 3, 0, 0,
        #                               1, 1, 2, 2, 6, 6, 1, 1, 2, 2, 6, 6]
        sampled = list(sampler)
        # Check that the length is quadrupled
        assert len(sampled) == 4 * (len(dataset) - 1)  # 1 element is dropped, because it's not enough to form a batch
        # Check that the sampled indexes are a subset of the dataset indexes
        assert set(sampled).issubset(set(range(len(dataset))))
        # Check that each element is repeated as expected
        assert all(sampled[i] == sampled[i + 1] for i in range(0, len(sampled), 2))
        # Check that the batch is repeated as expected
        assert sampled[0:6] == sampled[6:12]
        assert sampled[12:18] == sampled[18:24]

    def test_sampler_with_mini_repeat_count_and_batch_size_2(self):
        dataset = ["a", "b", "c", "d", "e", "f", "g"]
        sampler = RepeatSampler(dataset, mini_repeat_count=3, batch_size=2, repeat_count=2)
        # Should output something like [4, 4, 4, 3, 3, 3, 4, 4, 4, 3, 3, 3,
        #                               0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1,
        #                               2, 2, 2, 6, 6, 6, 2, 2, 2, 6, 6, 6]
        sampled = list(sampler)
        # Check that the length is sextupled
        assert len(sampled) == 6 * (len(dataset) - 1)  # 1 element is dropped, because it's not enough to form a batch
        # Check that the sampled indexes are a subset of the dataset indexes
        assert set(sampled).issubset(set(range(len(dataset))))
        # Check that each element is repeated as expected
        assert all(sampled[i] == sampled[i + 1] == sampled[i + 2] for i in range(0, len(sampled), 3))
        # Check that the batch is repeated as expected
        assert sampled[0:6] == sampled[6:12]
        assert sampled[12:18] == sampled[18:24]
        assert sampled[24:30] == sampled[30:36]

    def test_sampler_with_mini_repeat_count_and_batch_size_3(self):
        dataset = ["a", "b", "c", "d", "e", "f", "g"]
        sampler = RepeatSampler(dataset, mini_repeat_count=2, batch_size=2, repeat_count=3)
        # Should output something like [4, 4, 3, 3, 4, 4, 3, 3, 4, 4, 3, 3,
        #                               0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1,
        #                               2, 2, 6, 6, 2, 2, 6, 6, 2, 2, 6, 6]
        sampled = list(sampler)
        # Check that the length is sextupled
        assert len(sampled) == 6 * (len(dataset) - 1)  # 1 element is dropped, because it's not enough to form a batch
        # Check that the sampled indexes are a subset of the dataset indexes
        assert set(sampled).issubset(set(range(len(dataset))))
        # Check that each element is repeated as expected
        assert all(sampled[i] == sampled[i + 1] for i in range(0, len(sampled), 2))
        # Check that the batch is repeated as expected
        assert sampled[0:4] == sampled[4:8] == sampled[8:12]
        assert sampled[12:16] == sampled[16:20] == sampled[20:24]
        assert sampled[24:28] == sampled[28:32] == sampled[32:36]


class GRPOTrainerTester(unittest.TestCase):
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

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GRPOConfig(
                output_dir=tmp_dir,
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

    @parameterized.expand([("bnpo",), ("dr_grpo",)])
    def test_training_loss_types(self, loss_type):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GRPOConfig(
                output_dir=tmp_dir,
                learning_rate=0.1,  # increase the learning rate to speed up the test
                per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
                num_generations=3,  # reduce the number of generations to reduce memory usage
                max_completion_length=32,  # reduce the completion length to reduce memory usage
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

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GRPOConfig(
                output_dir=tmp_dir,
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

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GRPOConfig(
                output_dir=tmp_dir,
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

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GRPOConfig(
                output_dir=tmp_dir,
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
            torch_dtype=torch.float32,  # Use float32 for testing to avoid precision issues
            use_cache=False,  # Required for gradient checkpointing
        )

        lora_config = LoraConfig(
            r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none"
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GRPOConfig(
                output_dir=tmp_dir,
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

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GRPOConfig(
                output_dir=tmp_dir,
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

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GRPOConfig(
                output_dir=tmp_dir,
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

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GRPOConfig(
                output_dir=tmp_dir,
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

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GRPOConfig(
                output_dir=tmp_dir,
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

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GRPOConfig(
                output_dir=tmp_dir,
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

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GRPOConfig(
                output_dir=tmp_dir,
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

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GRPOConfig(
                output_dir=tmp_dir,
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

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GRPOConfig(
                output_dir=tmp_dir,
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

    @require_vllm
    def test_training_vllm(self):
        """Test that training works with vLLM for generation."""
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GRPOConfig(
                output_dir=tmp_dir,
                learning_rate=0.1,  # increase the learning rate to speed up the test
                per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
                num_generations=3,  # reduce the number of generations to reduce memory usage
                max_completion_length=8,  # reduce the completion length to reduce memory usage
                report_to="none",
                use_vllm=True,
                vllm_mode="colocate",  # Use colocate mode instead of server mode
                vllm_gpu_memory_utilization=0.2,  # Use minimal memory
            )

            original_env = {}
            required_env_vars = {
                "RANK": "0",
                "LOCAL_RANK": "0",
                "WORLD_SIZE": "1",
                "LOCAL_WORLD_SIZE": "1",
                "MASTER_ADDR": "localhost",
                "MASTER_PORT": "12356",
            }

            for key, value in required_env_vars.items():
                original_env[key] = os.environ.get(key)
                os.environ[key] = value

            try:
                trainer = GRPOTrainer(
                    model="Qwen/Qwen2.5-0.5B-Instruct",  # Use a model that works with vLLM
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
            except Exception as e:
                if any(keyword in str(e).lower() for keyword in ["memory", "cuda", "insufficient", "no such device"]):
                    self.skipTest(f"Skipping vLLM test due to hardware constraints: {e}")
                else:
                    raise
            finally:
                # Restore environment variables
                for key, original_value in original_env.items():
                    if original_value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = original_value

    def test_training_with_sync_ref_model(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GRPOConfig(
                output_dir=tmp_dir,
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
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GRPOConfig(
                output_dir=tmp_dir,
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
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GRPOConfig(
                output_dir=tmp_dir,
                beta=0.1,  # set beta to non-zero value to test the case where the reference model is used
                learning_rate=0.1,  # increase the learning rate to speed up the test
                per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
                num_generations=3,  # reduce the number of generations to reduce memory usage
                max_completion_length=8,  # reduce the completion length to reduce memory usage
                report_to="none",
                token_entropy_percentile_threshold=0.8,
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
    @require_vllm
    def test_training_vllm_and_peft(self):
        """Test that training works with vLLM for generation."""
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")  # Use a model that works with vLLM
        base_param_names = [f"base_model.model.{n}" for n, _ in model.named_parameters()]
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GRPOConfig(
                output_dir=tmp_dir,
                learning_rate=0.1,  # increase the learning rate to speed up the test
                per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
                num_generations=3,  # reduce the number of generations to reduce memory usage
                max_completion_length=8,  # reduce the completion length to reduce memory usage
                report_to="none",
                use_vllm=True,
                vllm_mode="colocate",  # Use colocate mode instead of server mode
                vllm_gpu_memory_utilization=0.2,  # Use minimal memory
            )
            lora_config = LoraConfig(
                target_modules="all-linear",
                # test with non-default modules as it add extra keys in state_dict tht we need to handle
                modules_to_save=["embed_tokens", "lm_head"],
            )

            original_env = {}
            required_env_vars = {
                "RANK": "0",
                "LOCAL_RANK": "0",
                "WORLD_SIZE": "1",
                "LOCAL_WORLD_SIZE": "1",
                "MASTER_ADDR": "localhost",
                "MASTER_PORT": "12357",
            }

            for key, value in required_env_vars.items():
                original_env[key] = os.environ.get(key)
                os.environ[key] = value

            try:
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
            except Exception as e:
                if any(keyword in str(e).lower() for keyword in ["memory", "cuda", "insufficient", "no such device"]):
                    self.skipTest(f"Skipping vLLM+PEFT test due to hardware constraints: {e}")
                else:
                    raise
            finally:
                # Restore environment variables
                for key, original_value in original_env.items():
                    if original_value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = original_value

    @require_vllm
    def test_training_vllm_guided_decoding(self):
        """Test that training works with vLLM for generation with guided decoding."""
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GRPOConfig(
                output_dir=tmp_dir,
                learning_rate=0.1,  # increase the learning rate to speed up the test
                per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
                num_generations=3,  # reduce the number of generations to reduce memory usage
                max_completion_length=8,  # reduce the completion length to reduce memory usage
                report_to="none",
                use_vllm=True,
                vllm_mode="colocate",  # Use colocate mode instead of server mode
                vllm_gpu_memory_utilization=0.2,  # Use minimal memory
                vllm_guided_decoding_regex=r"<reasoning>\n.*\n</reasoning>\n<answer>\n.*\n</answer>",
            )

            original_env = {}
            required_env_vars = {
                "RANK": "0",
                "LOCAL_RANK": "0",
                "WORLD_SIZE": "1",
                "LOCAL_WORLD_SIZE": "1",
                "MASTER_ADDR": "localhost",
                "MASTER_PORT": "12358",
            }

            for key, value in required_env_vars.items():
                original_env[key] = os.environ.get(key)
                os.environ[key] = value

            try:
                trainer = GRPOTrainer(
                    model="Qwen/Qwen2.5-0.5B-Instruct",  # Use a model that works with vLLM
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
            except Exception as e:
                if any(keyword in str(e).lower() for keyword in ["memory", "cuda", "insufficient", "no such device"]):
                    self.skipTest(f"Skipping vLLM guided decoding test due to hardware constraints: {e}")
                else:
                    raise
            finally:
                # Restore environment variables
                for key, original_value in original_env.items():
                    if original_value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = original_value

    def test_training_with_additional_generation_kwargs(self):
        """Test that training works with additional generation kwargs."""
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GRPOConfig(
                output_dir=tmp_dir,
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
    def test_training_vllm_with_additional_generation_kwargs(self):
        """Test that training works with vLLM and additional generation kwargs."""
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GRPOConfig(
                output_dir=tmp_dir,
                learning_rate=0.1,  # increase the learning rate to speed up the test
                per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
                num_generations=3,  # reduce the number of generations to reduce memory usage
                max_completion_length=8,  # reduce the completion length to reduce memory usage
                report_to="none",
                use_vllm=True,
                vllm_mode="colocate",  # Use colocate mode instead of server mode
                vllm_gpu_memory_utilization=0.2,  # Use minimal memory
                top_p=0.9,
                top_k=10,
                min_p=0.01,
                repetition_penalty=1.1,
            )

            original_env = {}
            required_env_vars = {
                "RANK": "0",
                "LOCAL_RANK": "0",
                "WORLD_SIZE": "1",
                "LOCAL_WORLD_SIZE": "1",
                "MASTER_ADDR": "localhost",
                "MASTER_PORT": "12359",
            }

            for key, value in required_env_vars.items():
                original_env[key] = os.environ.get(key)
                os.environ[key] = value

            try:
                trainer = GRPOTrainer(
                    model="Qwen/Qwen2.5-0.5B-Instruct",  # Use a model that works with vLLM
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
            except Exception as e:
                if any(keyword in str(e).lower() for keyword in ["memory", "cuda", "insufficient", "no such device"]):
                    self.skipTest(f"Skipping vLLM generation kwargs test due to hardware constraints: {e}")
                else:
                    raise
            finally:
                # Restore environment variables
                for key, original_value in original_env.items():
                    if original_value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = original_value

    def test_training_no_scale_rewards(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GRPOConfig(
                output_dir=tmp_dir,
                learning_rate=0.1,  # increase the learning rate to speed up the test
                per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
                num_generations=3,  # reduce the number of generations to reduce memory usage
                max_completion_length=8,  # reduce the completion length to reduce memory usage
                scale_rewards=False,
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
        def fake_generate(input_ids=None, **kwargs):
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

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GRPOConfig(
                output_dir=tmp_dir,
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

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GRPOConfig(
                output_dir=tmp_dir,
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

    def test_training_num_generations_larger_than_batch_size(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GRPOConfig(
                output_dir=tmp_dir,
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

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GRPOConfig(
                output_dir=tmp_dir,
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

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GRPOConfig(
                output_dir=tmp_dir,
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

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GRPOConfig(
                output_dir=tmp_dir,
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

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GRPOConfig(
                output_dir=tmp_dir,
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

    @require_flash_attn
    @require_bitsandbytes
    @require_peft
    @require_torch_accelerator
    @parameterized.expand(
        [
            ("HuggingFaceTB/SmolVLM-Instruct",),  # Only test the smaller model to avoid OOM
        ]
    )
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

        def data_gen(num_samples):
            processor = AutoProcessor.from_pretrained(model_name)
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
            del processor

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
            attn_implementation="flash_attention_2",
            torch_dtype="bfloat16",
            device_map=get_kbit_device_map(),
            quantization_config=quantization_config,
        )

        processor = AutoProcessor.from_pretrained(model_name)

        def reward_func(prompts, completions, **kwargs):
            # simple nonsensical reward
            return [-((len(c) - 25) ** 2) + 100 for c in completions]

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GRPOConfig(
                output_dir=tmp_dir,
                learning_rate=0.1,
                per_device_train_batch_size=1,  # Minimal batch size
                gradient_accumulation_steps=2,  # Maintain effective batch size
                num_generations=2,
                max_completion_length=8,  # Much shorter completions
                max_prompt_length=128,  # Limit prompt length to save memory
                gradient_checkpointing=True,  # Enable gradient checkpointing
                bf16=True,  # Use bfloat16 precision
                max_steps=1,  # Only do 1 training step to save time and memory
                report_to="none",
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

                self.assertIsInstance(trainer.model, PeftModel)

                previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

                trainer.train()

                self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

                # Check that LoRA parameters have changed
                # For VLM models, we're more permissive about which parameters can change
                lora_params_changed = False
                for n, param in previous_trainable_params.items():
                    new_param = trainer.model.get_parameter(n)
                    if "lora" in n.lower():  # LoRA parameters should change
                        if not torch.equal(param, new_param):
                            lora_params_changed = True

                # At least some LoRA parameters should have changed during training
                self.assertTrue(lora_params_changed, "No LoRA parameters were updated during training.")

            except torch.OutOfMemoryError as e:
                self.skipTest(f"Skipping VLM training test due to insufficient GPU memory: {e}")
            except Exception as e:
                # Check for other memory-related errors
                if any(keyword in str(e).lower() for keyword in ["memory", "cuda", "out of memory", "insufficient"]):
                    self.skipTest(f"Skipping VLM training test due to hardware constraints: {e}")
                else:
                    raise

    @parameterized.expand(
        [
            ("Qwen/Qwen2-VL-2B-Instruct",),
            ("HuggingFaceTB/SmolVLM-Instruct",),
        ]
    )
    def test_vlm_processor_detection(self, processor_model_name):
        """Test that VLM processors are properly detected and vLLM compatibility is handled correctly."""
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        with tempfile.TemporaryDirectory() as tmp_dir:
            config = GRPOConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                num_generations=2,
                max_completion_length=8,
                use_vllm=False,  # Disable vLLM to test only processor detection
                report_to="none",
            )

            # Create a VLM processor (has both tokenizer and image_processor attributes)
            processor = AutoProcessor.from_pretrained(processor_model_name)

            # Verify processor has both required attributes for VLM detection
            self.assertTrue(hasattr(processor, "tokenizer"))
            self.assertTrue(hasattr(processor, "image_processor"))

            def dummy_reward_func(completions, **kwargs):
                return [1.0] * len(completions)

            # Test VLM processor detection logic without vLLM initialization
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                trainer = GRPOTrainer(
                    model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                    reward_funcs=dummy_reward_func,
                    args=config,
                    train_dataset=dataset,
                    processing_class=processor,  # VLM processor
                )

                # Should detect VLM processor correctly
                is_vlm_processor = hasattr(processor, "tokenizer") and hasattr(processor, "image_processor")
                self.assertTrue(is_vlm_processor, "Should detect VLM processor correctly")

                # Should include 'image' in signature columns for VLM processors
                self.assertIn(
                    "image", trainer._signature_columns, "Should include 'image' in signature columns for VLM"
                )

                # Should not emit any compatibility warnings when vLLM is disabled
                vlm_warnings = [str(w_item.message) for w_item in w if "VLM" in str(w_item.message)]
                self.assertEqual(
                    len(vlm_warnings),
                    0,
                    f"Should not emit VLM warnings when vLLM is disabled, but got: {vlm_warnings}",
                )

    @require_torch_accelerator
    @require_peft
    @require_bitsandbytes
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

        with tempfile.TemporaryDirectory() as tmp_dir:
            config = GRPOConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=1,  # Minimal batch size
                gradient_accumulation_steps=2,  # Make effective batch size 2, divisible by num_generations
                num_generations=2,
                max_completion_length=4,  # Very short completions to reduce memory
                max_prompt_length=32,  # Very short prompts to reduce memory
                use_vllm=True,  # Enable vLLM
                vllm_mode="colocate",  # Use colocate mode to avoid server dependency
                vllm_gpu_memory_utilization=0.05,  # Use minimal GPU memory (5%)
                gradient_checkpointing=True,  # Enable gradient checkpointing to save memory
                bf16=True,  # Use bfloat16 to reduce memory
                report_to="none",
            )

            # Create a VLM processor
            processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-Instruct")

            # Verify processor has both required attributes for VLM detection
            self.assertTrue(hasattr(processor, "tokenizer"))
            self.assertTrue(hasattr(processor, "image_processor"))

            def dummy_reward_func(completions, **kwargs):
                return [1.0] * len(completions)

            # Use LoRA configuration for memory efficiency
            from peft import LoraConfig

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
                            torch_dtype=torch.bfloat16,
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
                        self.assertTrue(trainer.use_vllm, "vLLM should be enabled for VLM processors in colocate mode")
                        self.assertEqual(trainer.vllm_mode, "colocate", "Should use colocate mode")

                        # Check if signature columns were set properly
                        if trainer._signature_columns is not None:
                            # Should include 'image' in signature columns for VLM processors
                            self.assertIn(
                                "image",
                                trainer._signature_columns,
                                "Should include 'image' in signature columns for VLM",
                            )

                        # Should not emit any warnings about VLM incompatibility
                        incompatibility_warnings = [
                            str(w_item.message)
                            for w_item in w
                            if "does not support VLMs" in str(w_item.message)
                            or "not compatible" in str(w_item.message).lower()
                        ]
                        self.assertEqual(
                            len(incompatibility_warnings),
                            0,
                            f"Should not emit VLM incompatibility warnings, but got: {incompatibility_warnings}",
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
                            self.skipTest(f"Skipping vLLM colocate test due to hardware constraints: {e}")
                        elif "KeyError" in str(e) and "RANK" in str(e):
                            self.skipTest(f"Skipping vLLM colocate test due to environment setup issues: {e}")
                        elif "ValueError" in str(e) and "memory" in str(e).lower():
                            self.skipTest(f"Skipping vLLM colocate test due to memory constraints: {e}")
                        else:
                            raise
            finally:
                # Restore original environment variables
                for key, original_value in original_env.items():
                    if original_value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = original_value


class GRPOImageProcessingTester(unittest.TestCase):
    """Test cases for image preprocessing and validation in GRPO trainer."""

    def test_validate_and_preprocess_images_pil_images(self):
        """Test validation and preprocessing of PIL images."""
        from PIL import Image

        # Create test PIL images
        img1 = Image.new("RGB", (100, 100), color="red")
        img2 = Image.new("RGBA", (50, 50), color="blue")  # Will be converted to RGB
        img3 = Image.new("L", (75, 75), color=128)  # Grayscale, will be converted to RGB

        images = [img1, img2, img3, None]

        processed = GRPOTrainer._validate_and_preprocess_images(images)

        self.assertEqual(len(processed), 4)

        # First image should remain unchanged
        self.assertEqual(processed[0].mode, "RGB")
        self.assertEqual(processed[0].size, (100, 100))

        # Second image should be converted from RGBA to RGB
        self.assertEqual(processed[1].mode, "RGB")
        self.assertEqual(processed[1].size, (50, 50))

        # Third image should be converted from grayscale to RGB
        self.assertEqual(processed[2].mode, "RGB")
        self.assertEqual(processed[2].size, (75, 75))

        # Fourth image should remain None
        self.assertIsNone(processed[3])

    def test_validate_and_preprocess_images_numpy_arrays(self):
        """Test validation and preprocessing of numpy arrays."""
        # Create test numpy arrays in different formats
        rgb_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        rgba_array = np.random.randint(0, 255, (50, 50, 4), dtype=np.uint8)
        grayscale_array = np.random.randint(0, 255, (75, 75), dtype=np.uint8)
        float_array = np.random.rand(80, 80, 3).astype(np.float32)  # Will be normalized

        images = [rgb_array, rgba_array, grayscale_array, float_array]

        processed = GRPOTrainer._validate_and_preprocess_images(images)

        self.assertEqual(len(processed), 4)

        # All should be converted to PIL RGB images
        for img in processed:
            self.assertEqual(img.mode, "RGB")

        # Check sizes
        self.assertEqual(processed[0].size, (100, 100))
        self.assertEqual(processed[1].size, (50, 50))  # RGBA converted to RGB
        self.assertEqual(processed[2].size, (75, 75))  # Grayscale converted to RGB
        self.assertEqual(processed[3].size, (80, 80))  # Float normalized and converted

    def test_validate_and_preprocess_images_invalid_inputs(self):
        """Test handling of invalid image inputs."""
        # Create invalid inputs
        zero_size_array = np.zeros((0, 0, 3))  # Zero size

        images = [zero_size_array, "non_existent_file.jpg", 123, object()]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            processed = GRPOTrainer._validate_and_preprocess_images(images)

        # Should have warnings for all invalid inputs
        self.assertGreater(len(w), 0)

        # All should be converted to None
        for img in processed:
            self.assertIsNone(img)

    def test_validate_and_preprocess_images_large_images(self):
        """Test handling of very large images."""
        from PIL import Image

        # Create a very large image that should trigger a warning
        large_image = Image.new("RGB", (5000, 5000), color="green")  # 25MP image

        images = [large_image]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            processed = GRPOTrainer._validate_and_preprocess_images(images)

        # Should have a warning about large image size
        large_image_warnings = [warning for warning in w if "very large" in str(warning.message)]
        self.assertGreater(len(large_image_warnings), 0)

        # Image should still be processed
        self.assertEqual(len(processed), 1)
        self.assertEqual(processed[0].mode, "RGB")
        self.assertEqual(processed[0].size, (5000, 5000))

    def test_validate_and_preprocess_images_file_paths(self):
        """Test validation and preprocessing of image file paths."""
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create a test image file
            test_image = Image.new("RGB", (100, 100), color="red")
            test_image_path = os.path.join(tmp_dir, "test_image.jpg")
            test_image.save(test_image_path)

            # Test with valid and invalid file paths
            images = [test_image_path, "non_existent_file.jpg"]

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                processed = GRPOTrainer._validate_and_preprocess_images(images)

            # Should have a warning for the non-existent file
            file_warnings = [warning for warning in w if "Failed to load image from path" in str(warning.message)]
            self.assertGreater(len(file_warnings), 0)

            # First image should be loaded successfully
            self.assertEqual(processed[0].mode, "RGB")
            self.assertEqual(processed[0].size, (100, 100))

            # Second should be None
            self.assertIsNone(processed[1])

    def test_validate_and_preprocess_images_empty_list(self):
        """Test handling of empty image list."""
        images = []
        processed = GRPOTrainer._validate_and_preprocess_images(images)
        self.assertEqual(processed, [])

    def test_validate_and_preprocess_images_all_none(self):
        """Test handling of list with only None values."""
        images = [None, None, None]
        processed = GRPOTrainer._validate_and_preprocess_images(images)
        self.assertEqual(processed, [None, None, None])

    def test_image_validation_integration_with_trainer(self):
        """Test integration of image validation with GRPO trainer."""
        from PIL import Image

        # Create a simple dataset with images
        test_images = [
            Image.new("RGB", (100, 100), color="red"),
            np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8),
            None,
        ]

        def data_gen():
            for i, img in enumerate(test_images):
                yield {
                    "prompt": f"Describe image {i}",
                    "image": img,
                }

        dataset = Dataset.from_generator(data_gen)

        def dummy_reward_func(completions, **kwargs):
            return [1.0] * len(completions)

        with tempfile.TemporaryDirectory() as tmp_dir:
            config = GRPOConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                num_generations=2,
                max_completion_length=8,
                report_to="none",
            )

            # This should work without errors
            trainer = GRPOTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                reward_funcs=dummy_reward_func,
                args=config,
                train_dataset=dataset,
            )

            # Test the image validation method directly
            inputs = [{"image": img} for img in test_images]
            images = [x.get("image", None) for x in inputs]

            processed_images = trainer._validate_and_preprocess_images(images)

            # First two should be valid PIL images, third should be None
            self.assertEqual(processed_images[0].mode, "RGB")
            self.assertEqual(processed_images[1].mode, "RGB")
            self.assertIsNone(processed_images[2])

    def test_image_validation_memory_efficiency(self):
        """Test that image validation doesn't cause memory leaks."""
        import gc

        from PIL import Image

        # Create multiple large images to test memory handling
        large_images = []
        for i in range(5):
            # Create moderately sized images (not too large to avoid test timeouts)
            img = Image.new("RGB", (1000, 1000), color=i * 50)
            large_images.append(img)

        # Process the images
        processed = GRPOTrainer._validate_and_preprocess_images(large_images)

        # Clear references
        del large_images
        gc.collect()

        # Verify all images were processed
        self.assertEqual(len(processed), 5)
        for img in processed:
            self.assertEqual(img.mode, "RGB")
            self.assertEqual(img.size, (1000, 1000))

    def test_image_validation_with_processing_class(self):
        """Test image validation with different processing classes."""
        from PIL import Image

        # Test with None processing class
        test_image = Image.new("RGB", (100, 100), color="blue")
        images = [test_image]

        processed = GRPOTrainer._validate_and_preprocess_images(images, processing_class=None)
        self.assertEqual(len(processed), 1)
        self.assertEqual(processed[0].mode, "RGB")

        # Test with mock processing class (should still work the same)
        class MockProcessor:
            pass

        mock_processor = MockProcessor()
        processed = GRPOTrainer._validate_and_preprocess_images(images, processing_class=mock_processor)
        self.assertEqual(len(processed), 1)
        self.assertEqual(processed[0].mode, "RGB")

    def test_compute_entropy_mask(self):
        """Test the _compute_entropy_mask method."""
        trainer = GRPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=GRPOConfig(token_entropy_percentile_threshold=0.8),
        )

        # Create dummy entropies and completion mask
        entropies = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [0.7, 0.8, 0.9, 1.0, 1.1, 1.2]])
        completion_mask = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]])

        entropy_mask = trainer._compute_entropy_mask(entropies, completion_mask)

        self.assertEqual(entropy_mask.shape, entropies.shape)

        # We have a total of 12 tokens out of which 10 are non-pad,
        # for a token_entropy_percentile_threshold of 0.8,
        # we expect the top 20% i.e 2 non-pad tokens corresponding to the highest entropy to be unmasked.
        # In our example these will be the tokens corresponding to the entropies 0.9 and 1.0
        # since 1.1 and 1.2 are pad tokens they are excluded from the entropy threshold calculation.
        expected_mask = torch.tensor([[0, 0, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0]], dtype=torch.bool)
        self.assertTrue(torch.equal(entropy_mask, expected_mask))

        entropies = torch.tensor([[0.1, 0.2, 0.3, 1.4, 0.5, 0.14], [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]])
        completion_mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 0, 0]])

        expected_mask = torch.tensor([[0, 0, 0, 1, 0, 0], [0, 0, 0, 1, 0, 0]], dtype=torch.bool)
        entropy_mask = trainer._compute_entropy_mask(entropies, completion_mask)

        self.assertTrue(torch.equal(entropy_mask, expected_mask))
