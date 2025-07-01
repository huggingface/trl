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
from unittest.mock import patch

import torch
from datasets import load_dataset
from parameterized import parameterized
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from transformers.testing_utils import require_peft
from transformers.utils import is_peft_available

from trl import GRPOConfig, GRPOTrainer
from trl.trainer.grpo_trainer import RepeatSampler, shuffle_tensor_dict, split_tensor_dict

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
    @unittest.skip("We should add a mock for the vLLM server.")
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
            )
            trainer = GRPOTrainer(
                model="Qwen/Qwen2.5-0.5B-Instruct",  # tiny is too small for vLLM
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

    @unittest.skip("We should add a mock for the vLLM server.")
    @require_peft
    @require_vllm
    def test_training_vllm_and_peft(self):
        """Test that training works with vLLM for generation."""
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")  # tiny model is too small for vLLM
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
            )
            lora_config = LoraConfig(
                target_modules="all-linear",
                # test with non-default modules as it add extra keys in state_dict tht we need to handle
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

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GRPOConfig(
                output_dir=tmp_dir,
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
    @unittest.skip("We should add a mock for the vLLM server.")
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
        def fake_generate(prompt_ids, **kwargs):
            # pad_token_id = 151643; eos_token_id = 151645
            completions_ids = torch.tensor(
                [
                    [1, 2, 3, 4, 5, 6, 7, 8],  # this one is truncated
                    [9, 10, 11, 151645, 151643, 151643, 151643, 151643],  # this one contains eos
                    [12, 13, 14, 15, 16, 17, 18, 151645],  # particular case, eos is generated just within the limit
                ],
                device=prompt_ids.device,
            )
            return torch.cat([prompt_ids, completions_ids], dim=1)

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
                
class PackingTester(unittest.TestCase):
    def setUp(self):
        self.dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")
        self.model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        self.reward_model_id = "trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5"

    def create_trainer(self, num_generations=2, batch_size=8, do_pack_completions=False, beta=0.0, max_steps=None, tmp_dir=None, **kwargs):
        args_kwargs = dict(
            output_dir=tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=batch_size,
            num_generations=num_generations,
            report_to="none",
            do_pack_completions=do_pack_completions,
            beta=beta,
            **kwargs,
        )
        if max_steps is not None:
            args_kwargs["max_steps"] = max_steps
        training_args = GRPOConfig(**args_kwargs)
        return GRPOTrainer(
            model=self.model_id,
            reward_funcs=self.reward_model_id,
            args=training_args,
            train_dataset=self.dataset,
        )

    def test_pack_responses_in_a_group(self):
        num_generations = 2
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = self.create_trainer(num_generations=num_generations, tmp_dir=tmp_dir, do_pack_completions=True)

            prompt_ids = torch.tensor(
                [
                    [100, 101, 102, 103],
                    [100, 101, 102, 103],
                    # Prompt tokens 202, 203
                    [200, 201, 202, 203],
                    [200, 201, 202, 203],
                ]
            )
            prompt_mask = torch.tensor(
                [
                    [True, True, True, True],
                    [True, True, True, True],
                    [False, False, True, True],
                    [False, False, True, True],
                ]
            )
            completion_ids = torch.tensor(
                [
                    [1000, 1001, 1002, 1003, 1004, 1005],
                    [1010, 1011, 1012, 1013, 1014, 1015],
                    [2000, 2001, 2002, 2003, 2004, 2005],
                    [2010, 2011, 2012, 2013, 2014, 2015],
                ]
            )
            completion_mask = torch.tensor(
                [
                    [True, True, True, True, False, False],
                    [True, True, True, True, True, False],
                    [True, True, True, True, True, True],
                    [True, True, False, False, False, False],
                ]
            )

            packed_sequence = trainer._pack_responses_in_a_group(prompt_ids, completion_ids, prompt_mask, completion_mask)
            packed_input_ids = packed_sequence["input_ids"]
            packed_position_ids = packed_sequence["position_ids"]
            packed_attention_mask = packed_sequence["attention_mask"]

            num_unique_prompts = len(prompt_ids) // num_generations
            self.assertEqual(len(packed_input_ids), num_unique_prompts)
            self.assertEqual(len(packed_position_ids), num_unique_prompts)
            self.assertEqual(len(packed_attention_mask), num_unique_prompts)

            pad_token_id = trainer.processing_class.pad_token_id
            # Prompts should be left padded and completions right padded.
            expected_input_ids = torch.tensor(
                [
                    [100, 101, 102, 103, 1000, 1001, 1002, 1003, 1010, 1011, 1012, 1013, 1014],
                    [ pad_token_id, pad_token_id, 202, 203, 2000, 2001, 2002, 2003, 2004, 2005, 2010, 2011, pad_token_id],
                ]
            )
            self.assertTrue(torch.equal(packed_input_ids, expected_input_ids))
            expected_position_ids = torch.tensor(
                [
                    [0, 1, 2, 3, 4, 5, 6, 7, 4, 5, 6, 7, 8],
                    [ pad_token_id, pad_token_id, 0, 1, 2, 3, 4, 5, 6, 7, 2, 3, pad_token_id],
                ]
            )
            self.assertTrue(torch.equal(packed_position_ids, expected_position_ids))
            expected_attention_mask = torch.tensor(
                [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                ]
            )
            self.assertTrue(torch.equal(packed_attention_mask, expected_attention_mask))

    def test_unpack_tensor(self):
        num_generations = 2
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = self.create_trainer(num_generations=num_generations, tmp_dir=tmp_dir, do_pack_completions=True)
            pad_token_id = trainer.processing_class.pad_token_id
            position_ids = torch.tensor(
                [
                    [0, 1, 2, 3, 4, 5, 6, 7, 4, 5, 6, 7, 8],
                    [pad_token_id, pad_token_id, 0, 1, 2, 3, 4, 5, 6, 7, 2, 3, pad_token_id],
                ]
            )
            prompt_length = torch.tensor([4, 2])
            all_logps = torch.tensor(
                [
                    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3],
                    [0.0, 0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 0.0],
                ]
            )
            unpacked_logps = trainer._unpack_tensor(position_ids, all_logps, prompt_length, padding_value=-100.0)
            expected_unpacked_logps = torch.tensor(
                [
                    [0.5, 0.6, 0.7, 0.8, -100.0, -100.0],
                    [0.9, 1.0, 1.1, 1.2, 1.3, -100.0],
                    [0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                    [1.0, 1.1, -100.0, -100.0, -100.0, -100.0],
                ]
            )
            self.assertTrue(torch.allclose(unpacked_logps, expected_unpacked_logps))
    
    def test_forward_pass_with_packing(self):
        num_generations = 2
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = self.create_trainer(
                num_generations=num_generations,
                do_pack_completions=False,
                beta=0.1,
                tmp_dir=tmp_dir,
                model_init_kwargs={"attn_implementation": "flash_attention_2"},
            )
            pad_token_id = trainer.processing_class.pad_token_id
            prompt_ids = torch.tensor(
                [
                    [100, 101, 102, 103],
                    [100, 101, 102, 103],
                    [pad_token_id, pad_token_id, 202, 203],
                    [pad_token_id, pad_token_id, 202, 203],
                ],
                device=trainer.model.device,
            )
            prompt_attention_mask = torch.tensor(
                [
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [0, 0, 1, 1],
                    [0, 0, 1, 1],
                ],
                device=trainer.model.device,
            )
            completion_ids = torch.tensor(
                [
                    [1000, 1001, 1002, 1003, pad_token_id, pad_token_id],
                    [1010, 1011, 1012, 1013, 1014, pad_token_id],
                    [2000, 2001, 2002, 2003, 2004, 2005],
                    [2010, 2011, 2012, pad_token_id, pad_token_id, pad_token_id],
                ],
                device=trainer.model.device,
            )
            completion_attention_mask = torch.tensor(
                [
                    [1, 1, 1, 1, 0, 0],
                    [1, 1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 0, 0, 0],
                ],
                device=trainer.model.device,
            )
            
            input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
            attention_mask = torch.cat([prompt_attention_mask, completion_attention_mask], dim=1)
            
            logps_without_packing = trainer._get_per_token_logps_and_entropies(
                trainer.model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                logits_to_keep=completion_attention_mask.shape[1],
            )["logps"]
            
            logps_without_packing = logps_without_packing * completion_attention_mask
            
            trainer.do_pack_completions = True
            packed_inputs = trainer._pack_responses_in_a_group(
                prompt_ids, completion_ids, prompt_attention_mask, completion_attention_mask
            )
            packed_input_ids = packed_inputs["input_ids"]
            packed_attention_mask = packed_inputs["attention_mask"]
            packed_position_ids = packed_inputs["position_ids"]
            
            logps_with_packing = trainer._get_per_token_logps_and_entropies(
                trainer.model,
                input_ids=packed_input_ids,
                attention_mask=packed_attention_mask,
                position_ids=packed_position_ids,
                logits_to_keep=None,
                prompt_mask=prompt_attention_mask[::trainer.num_generations],
            )["logps"]
            
            logps_with_packing = logps_with_packing * completion_attention_mask            
            assert torch.allclose(logps_without_packing, logps_with_packing, atol=1e-5)
            
       
    def test_training_with_packing(self):
        num_generations = 4
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = self.create_trainer(
                num_generations=num_generations,
                do_pack_completions=True,
                beta=0.1,
                tmp_dir=tmp_dir,
                model_init_kwargs={"attn_implementation": "flash_attention_2"},
            )
            trainer.train()
            previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}
            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])
            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                self.assertFalse(torch.equal(param, new_param), f"Parameter {n} has not changed.")

    def test_train_loss_same_with_and_without_packing(self):
        num_generations = 2
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer_packed = self.create_trainer(
                num_generations=num_generations,
                batch_size=4,
                do_pack_completions=True,
                beta=0.1,
                tmp_dir=tmp_dir,
                max_completion_length=16,
                model_init_kwargs={"attn_implementation": "flash_attention_2"},
            )
            trainer_packed.train()
            train_loss_packed = trainer_packed.state.log_history[-1]["train_loss"]

        with tempfile.TemporaryDirectory() as tmp_dir2:
            trainer_no_packing = self.create_trainer(
                num_generations=num_generations,
                batch_size=4,
                do_pack_completions=False,
                beta=0.1,
                tmp_dir=tmp_dir2,
                max_completion_length=16,
                model_init_kwargs={"attn_implementation": "flash_attention_2"},
            )
            trainer_no_packing.train()
            train_loss_no_packing = trainer_no_packing.state.log_history[-1]["train_loss"]

        self.assertAlmostEqual(train_loss_packed, train_loss_no_packing, places=5)