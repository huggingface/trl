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
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BatchFeature,
)
from transformers.testing_utils import (
    require_peft,
)
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

    def test_split_with_batch_feature(self):
        batch_feature = BatchFeature(
            {
                "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]),
                "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]),
                "pixel_values": torch.randn(4, 3, 224, 224),
                "image_sizes": torch.tensor([[224, 224], [224, 224]]),
                "spatial_merge_size": 2,
            }
        )

        regular_tensor = torch.arange(8).reshape(4, 2)
        tensor_dict = {"visual_inputs": batch_feature, "prompt_ids": regular_tensor}

        result = split_tensor_dict(tensor_dict, 2)

        self.assertEqual(len(result), 2)
        for i in range(2):
            self.assertIsInstance(result[i]["visual_inputs"], BatchFeature)
            expected_keys = {"input_ids", "attention_mask", "pixel_values", "image_sizes", "spatial_merge_size"}
            self.assertEqual(set(result[i]["visual_inputs"].data.keys()), expected_keys)
            self.assertEqual(result[i]["visual_inputs"]["input_ids"].shape, (2, 3))
            self.assertEqual(result[i]["visual_inputs"]["attention_mask"].shape, (2, 3))
            self.assertEqual(result[i]["visual_inputs"]["pixel_values"].shape, (2, 3, 224, 224))
            self.assertEqual(result[i]["prompt_ids"].shape, (2, 2))
            self.assertTrue(torch.equal(result[i]["visual_inputs"]["image_sizes"], batch_feature["image_sizes"]))
            self.assertEqual(result[i]["visual_inputs"]["spatial_merge_size"], batch_feature["spatial_merge_size"])

        self.assertTrue(torch.equal(result[0]["visual_inputs"]["input_ids"], batch_feature["input_ids"][:2]))
        self.assertTrue(torch.equal(result[1]["visual_inputs"]["input_ids"], batch_feature["input_ids"][2:]))
        self.assertTrue(torch.equal(result[0]["prompt_ids"], regular_tensor[:2]))
        self.assertTrue(torch.equal(result[1]["prompt_ids"], regular_tensor[2:]))


class ShuffleTensorDictTester(unittest.TestCase):
    def test_shuffle_preserves_shape(self):
        x = torch.arange(6).reshape(3, 2)
        y = torch.arange(3).reshape(3, 1)
        tensor_dict = {"x": x.clone(), "y": y.clone()}

        shuffled = shuffle_tensor_dict(tensor_dict)

        self.assertEqual(shuffled["x"].shape, x.shape)
        self.assertEqual(shuffled["y"].shape, y.shape)

    def test_shuffle_consistent_across_tensors(self):
        x = torch.tensor([[10, 11], [20, 21], [30, 31]])
        y = torch.tensor([[1], [2], [3]])
        tensor_dict = {"x": x.clone(), "y": y.clone()}

        shuffled = shuffle_tensor_dict(tensor_dict)

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

    def test_shuffle_with_batch_feature(self):
        batch_feature = BatchFeature(
            {
                "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
                "pixel_values": torch.randn(3, 3, 224, 224),
                "image_sizes": torch.tensor([[224, 224], [224, 224]]),
                "spatial_merge_size": 2,
            }
        )

        regular_tensor = torch.tensor([[10, 11], [12, 13], [14, 15]])
        tensor_dict = {"visual_inputs": batch_feature, "prompt_ids": regular_tensor}

        shuffled = shuffle_tensor_dict(tensor_dict)

        self.assertIsInstance(shuffled["visual_inputs"], BatchFeature)
        expected_keys = {"input_ids", "attention_mask", "pixel_values", "image_sizes", "spatial_merge_size"}
        self.assertEqual(set(shuffled["visual_inputs"].data.keys()), expected_keys)
        self.assertEqual(shuffled["visual_inputs"]["input_ids"].shape, (3, 3))
        self.assertEqual(shuffled["visual_inputs"]["attention_mask"].shape, (3, 3))
        self.assertEqual(shuffled["visual_inputs"]["pixel_values"].shape, (3, 3, 224, 224))
        self.assertEqual(shuffled["prompt_ids"].shape, (3, 2))
        self.assertTrue(torch.equal(shuffled["visual_inputs"]["image_sizes"], batch_feature["image_sizes"]))
        self.assertEqual(shuffled["visual_inputs"]["spatial_merge_size"], batch_feature["spatial_merge_size"])

        original_input_ids = batch_feature["input_ids"]
        shuffled_input_ids = shuffled["visual_inputs"]["input_ids"]
        shuffled_prompt_ids = shuffled["prompt_ids"]

        for i in range(3):
            original_pos = None
            for j in range(3):
                if torch.equal(shuffled_input_ids[i], original_input_ids[j]):
                    original_pos = j
                    break
            self.assertIsNotNone(original_pos)
            self.assertTrue(torch.equal(shuffled_prompt_ids[i], regular_tensor[original_pos]))


class RepeatRandomSamplerTester(unittest.TestCase):
    def test_sampler(self):
        dataset = ["a", "b", "c", "d", "e", "f", "g"]
        sampler = RepeatSampler(dataset, mini_repeat_count=2)
        sampled = list(sampler)
        assert len(sampled) == 2 * len(dataset)
        assert set(sampled) == set(range(len(dataset)))
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
        sampled = list(sampler)
        assert len(sampled) == len(dataset)
        assert set(sampled) == set(range(len(dataset)))

    def test_sampler_with_batch_size(self):
        dataset = ["a", "b", "c", "d", "e", "f", "g", "h"]
        sampler = RepeatSampler(dataset, mini_repeat_count=1, batch_size=2, repeat_count=2)
        sampled = list(sampler)
        assert len(sampled) == 2 * len(dataset)
        assert set(sampled) == set(range(len(dataset)))
        assert all(sampled[i : i + 1] == sampled[i + 2 : i + 3] for i in range(0, len(sampled), 4))

    def test_sampler_with_batch_size_and_drop(self):
        dataset = ["a", "b", "c", "d", "e", "f", "g"]
        sampler = RepeatSampler(dataset, mini_repeat_count=1, batch_size=2, repeat_count=2)
        sampled = list(sampler)
        assert len(sampled) == 2 * (len(dataset) - 1)
        assert set(sampled).issubset(set(range(len(dataset))))
        assert all(sampled[i : i + 1] == sampled[i + 2 : i + 3] for i in range(0, len(sampled), 4))

    def test_sampler_with_mini_repeat_count_and_batch_size_1(self):
        dataset = ["a", "b", "c", "d", "e", "f", "g"]
        sampler = RepeatSampler(dataset, mini_repeat_count=2, batch_size=3, repeat_count=2)
        sampled = list(sampler)
        assert len(sampled) == 4 * (len(dataset) - 1)
        assert set(sampled).issubset(set(range(len(dataset))))
        assert all(sampled[i] == sampled[i + 1] for i in range(0, len(sampled), 2))
        assert sampled[0:6] == sampled[6:12]
        assert sampled[12:18] == sampled[18:24]

    def test_sampler_with_mini_repeat_count_and_batch_size_2(self):
        dataset = ["a", "b", "c", "d", "e", "f", "g"]
        sampler = RepeatSampler(dataset, mini_repeat_count=3, batch_size=2, repeat_count=2)
        sampled = list(sampler)
        assert len(sampled) == 6 * (len(dataset) - 1)
        assert set(sampled).issubset(set(range(len(dataset))))
        assert all(sampled[i] == sampled[i + 1] == sampled[i + 2] for i in range(0, len(sampled), 3))
        assert sampled[0:6] == sampled[6:12]
        assert sampled[12:18] == sampled[18:24]
        assert sampled[24:30] == sampled[30:36]

    def test_sampler_with_mini_repeat_count_and_batch_size_3(self):
        dataset = ["a", "b", "c", "d", "e", "f", "g"]
        sampler = RepeatSampler(dataset, mini_repeat_count=2, batch_size=2, repeat_count=3)
        sampled = list(sampler)
        assert len(sampled) == 6 * (len(dataset) - 1)
        assert set(sampled).issubset(set(range(len(dataset))))
        assert all(sampled[i] == sampled[i + 1] for i in range(0, len(sampled), 2))
        assert sampled[0:4] == sampled[4:8] == sampled[8:12]
        assert sampled[12:16] == sampled[16:20] == sampled[20:24]
        assert sampled[24:28] == sampled[28:32] == sampled[32:36]


class GRPOTrainerTester(unittest.TestCase):
    def test_init_minimal(self):
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
                learning_rate=0.1,
                per_device_train_batch_size=3,
                num_generations=3,
                max_completion_length=8,
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

            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                self.assertFalse(torch.equal(param, new_param), f"Parameter {n} has not changed.")

    @parameterized.expand([("bnpo",), ("dr_grpo",)])
    def test_training_loss_types(self, loss_type):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GRPOConfig(
                output_dir=tmp_dir,
                learning_rate=0.1,
                per_device_train_batch_size=3,
                num_generations=3,
                max_completion_length=32,
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

            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                self.assertFalse(torch.equal(param, new_param), f"Parameter {n} has not changed.")

    def test_training_with_eval(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only")

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GRPOConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=3,
                per_device_eval_batch_size=3,
                num_generations=3,
                max_completion_length=8,
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
                learning_rate=0.1,
                per_device_train_batch_size=3,
                num_generations=3,
                max_completion_length=8,
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
                learning_rate=0.1,
                per_device_train_batch_size=3,
                num_generations=3,
                max_completion_length=8,
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

            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                if n in base_param_names:
                    self.assertTrue(torch.allclose(param, new_param), f"Parameter {n} has changed.")
                elif "base_layer" not in n:
                    self.assertFalse(torch.allclose(param, new_param), f"Parameter {n} has not changed.")

    @require_peft
    def test_training_peft_with_gradient_checkpointing(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        model = AutoModelForCausalLM.from_pretrained(
            "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            torch_dtype=torch.float32,
            use_cache=False,
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
                gradient_checkpointing=True,
                report_to="none",
            )
            trainer = GRPOTrainer(
                model=model,
                reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
                args=training_args,
                train_dataset=dataset,
                peft_config=lora_config,
            )

            self.assertIsInstance(trainer.model, PeftModel)
            previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}
            trainer.train()
            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                if "lora" in n.lower():
                    self.assertFalse(torch.equal(param, new_param), f"LoRA parameter {n} has not changed.")
                else:
                    self.assertTrue(torch.equal(param, new_param), f"Base parameter {n} has changed.")

    def test_training_different_reward_model(self):
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_only", split="train")
        reward_model_id = "trl-internal-testing/tiny-LlamaForSequenceClassification-3.2"
        reward_model = AutoModelForSequenceClassification.from_pretrained(reward_model_id)
        reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_id)
        reward_tokenizer.pad_token = "<|finetune_right_pad_id|>"

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
                reward_funcs=reward_model,
                reward_processing_classes=reward_tokenizer,
                args=training_args,
                train_dataset=dataset,
            )

            previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}
            trainer.train()
            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                self.assertFalse(torch.equal(param, new_param), f"Parameter {n} has not changed.")

    def test_training_reward_func_standard(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        def reward_func(completions, **kwargs):
            return [float(len(set(completion))) for completion in completions]

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

    def test_training_reward_func_conversational(self):
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_only", split="train")

        def reward_func(completions, **kwargs):
            return [float(len(set(completion[0]["content"]))) for completion in completions]

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

    def test_training_multiple_reward_funcs(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        def reward_func1(completions, **kwargs):
            return [float(len(completion)) for completion in completions]

        def reward_func2(completions, **kwargs):
            return [float(len(set(completion))) for completion in completions]

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
                reward_funcs=[reward_func1, reward_func2],
                args=training_args,
                train_dataset=dataset,
            )

            previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}
            trainer.train()
            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                self.assertFalse(torch.equal(param, new_param), f"Parameter {n} has not changed.")

    def test_training_multiple_reward_funcs_with_None_output(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        def applicable_reward_func(completions, **kwargs):
            return [float(len(completion)) for completion in completions]

        def non_applicable_reward_func(completions, **kwargs):
            return [None for _ in completions]

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
                reward_funcs=[applicable_reward_func, non_applicable_reward_func],
                args=training_args,
                train_dataset=dataset,
            )

            previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}
            trainer.train()
            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                self.assertFalse(torch.equal(param, new_param), f"Parameter {n} has not changed.")

    def test_training_multiple_reward_funcs_with_weights(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        def reward_func1(completions, **kwargs):
            return [float(len(completion)) for completion in completions]

        def reward_func2(completions, **kwargs):
            return [float(len(set(completion))) for completion in completions]

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GRPOConfig(
                output_dir=tmp_dir,
                learning_rate=0.1,
                per_device_train_batch_size=3,
                num_generations=3,
                max_completion_length=8,
                reward_weights=[0.3, 0.7],
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

            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                self.assertFalse(torch.equal(param, new_param), f"Parameter {n} has not changed.")

    def test_training_multiple_mixed_reward_funcs(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        def reward_func(completions, **kwargs):
            return [float(len(completion)) for completion in completions]

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
                reward_funcs=[reward_func, "trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5"],
                args=training_args,
                train_dataset=dataset,
            )

            previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}
            trainer.train()
            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                self.assertFalse(torch.equal(param, new_param), f"Parameter {n} has not changed.")

    def test_training_reward_func_additional_column(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")
        dataset = dataset.map(lambda x: {"some_values": list(range(10))})

        def reward_func(completions, some_values, **kwargs):
            return [float(len(completion)) + sum(some_values[i]) for i, completion in enumerate(completions)]

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GRPOConfig(
                output_dir=tmp_dir,
                learning_rate=0.1,
                per_device_train_batch_size=3,
                num_generations=3,
                max_completion_length=8,
                remove_unused_columns=False,
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

            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                self.assertFalse(torch.equal(param, new_param), f"Parameter {n} has not changed.")

    @require_vllm
    @unittest.skip("We should add a mock for the vLLM server.")
    def test_training_vllm(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GRPOConfig(
                output_dir=tmp_dir,
                learning_rate=0.1,
                per_device_train_batch_size=3,
                num_generations=3,
                max_completion_length=8,
                report_to="none",
                use_vllm=True,
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

            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                self.assertFalse(torch.equal(param, new_param), f"Parameter {n} has not changed.")

    def test_training_with_sync_ref_model(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GRPOConfig(
                output_dir=tmp_dir,
                learning_rate=0.1,
                per_device_train_batch_size=3,
                num_generations=3,
                max_completion_length=8,
                sync_ref_model=True,
                ref_model_mixup_alpha=0.6,
                ref_model_sync_steps=2,
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

            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                self.assertFalse(torch.equal(param, new_param), f"Parameter {n} has not changed.")

    def test_training_beta_non_zero(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GRPOConfig(
                output_dir=tmp_dir,
                learning_rate=0.1,
                per_device_train_batch_size=3,
                num_generations=3,
                max_completion_length=8,
                beta=0.1,
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

            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                self.assertFalse(torch.equal(param, new_param), f"Parameter {n} has not changed.")

    def test_training_with_entropy_filter(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GRPOConfig(
                output_dir=tmp_dir,
                learning_rate=0.1,
                per_device_train_batch_size=3,
                num_generations=3,
                max_completion_length=8,
                token_entropy_percentile_threshold=0.8,
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

            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                self.assertFalse(torch.equal(param, new_param), f"Parameter {n} has not changed.")

    @unittest.skip("We should add a mock for the vLLM server.")
    @require_peft
    @require_vllm
    def test_training_vllm_and_peft(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GRPOConfig(
                output_dir=tmp_dir,
                learning_rate=0.1,
                per_device_train_batch_size=3,
                num_generations=3,
                max_completion_length=8,
                report_to="none",
                use_vllm=True,
            )

            trainer = GRPOTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
                args=training_args,
                train_dataset=dataset,
                peft_config=LoraConfig(),
            )

            previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}
            trainer.train()
            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                if "lora" in n:
                    self.assertFalse(torch.equal(param, new_param), f"Parameter {n} has not changed.")

    @require_vllm
    @unittest.skip("We should add a mock for the vLLM server.")
    def test_training_vllm_guided_decoding(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GRPOConfig(
                output_dir=tmp_dir,
                learning_rate=0.1,
                per_device_train_batch_size=3,
                num_generations=3,
                max_completion_length=8,
                report_to="none",
                use_vllm=True,
                vllm_guided_decoding_regex=r"^[A-Za-z0-9 ]+$",
            )

            trainer = GRPOTrainer(
                model="Qwen/Qwen2.5-0.5B-Instruct",
                reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
                args=training_args,
                train_dataset=dataset,
            )

            previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}
            trainer.train()
            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                self.assertFalse(torch.equal(param, new_param), f"Parameter {n} has not changed.")

    def test_training_with_additional_generation_kwargs(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GRPOConfig(
                output_dir=tmp_dir,
                learning_rate=0.1,
                per_device_train_batch_size=3,
                num_generations=3,
                max_completion_length=8,
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

            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                self.assertFalse(torch.equal(param, new_param), f"Parameter {n} has not changed.")

    @require_vllm
    @unittest.skip("We should add a mock for the vLLM server.")
    def test_training_vllm_with_additional_generation_kwargs(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GRPOConfig(
                output_dir=tmp_dir,
                learning_rate=0.1,
                per_device_train_batch_size=3,
                num_generations=3,
                max_completion_length=8,
                report_to="none",
                use_vllm=True,
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

            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                self.assertFalse(torch.equal(param, new_param), f"Parameter {n} has not changed.")

    def test_training_no_scale_rewards(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GRPOConfig(
                output_dir=tmp_dir,
                learning_rate=0.1,
                per_device_train_batch_size=3,
                num_generations=3,
                max_completion_length=8,
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

            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                self.assertFalse(torch.equal(param, new_param), f"Parameter {n} has not changed.")

    @patch("transformers.generation.utils.GenerationMixin.generate")
    def test_training_with_mask_truncated_completions(self, mock_generate):
        def fake_generate(*args, **kwargs):
            prompt_ids = kwargs.get("input_ids", args[1] if len(args) > 1 else None)
            completions_ids = torch.tensor(
                [
                    [1, 2, 3, 4, 5, 6, 7, 8],
                    [9, 10, 11, 151645, 151643, 151643, 151643, 151643],
                    [12, 13, 14, 15, 16, 17, 18, 151645],
                ],
                device=prompt_ids.device,
            )
            return torch.cat([prompt_ids, completions_ids], dim=1)

        mock_generate.side_effect = fake_generate

        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GRPOConfig(
                output_dir=tmp_dir,
                learning_rate=0.1,
                per_device_train_batch_size=3,
                num_generations=3,
                max_completion_length=8,
                mask_truncated_completions=True,
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

            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                self.assertFalse(torch.equal(param, new_param), f"Parameter {n} has not changed.")

    def test_training_with_mask_truncated_completions_all_masked(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GRPOConfig(
                output_dir=tmp_dir,
                learning_rate=0.1,
                per_device_train_batch_size=3,
                num_generations=3,
                max_completion_length=8,
                mask_truncated_completions=True,
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

            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                self.assertTrue(torch.equal(param, new_param), f"Parameter {n} has changed.")

    def test_training_num_generations_larger_than_batch_size(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GRPOConfig(
                output_dir=tmp_dir,
                learning_rate=0.1,
                per_device_train_batch_size=3,
                max_completion_length=8,
                num_generations=6,
                gradient_accumulation_steps=2,
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

            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                self.assertFalse(torch.equal(param, new_param), f"Parameter {n} has not changed.")

    def test_training_delta_clipping(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GRPOConfig(
                output_dir=tmp_dir,
                learning_rate=0.1,
                per_device_train_batch_size=3,
                num_generations=3,
                max_completion_length=8,
                delta=2.0,
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

            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                self.assertFalse(torch.equal(param, new_param), f"Parameter {n} has not changed.")

    def test_training_multiple_dataloader_workers(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GRPOConfig(
                output_dir=tmp_dir,
                learning_rate=0.1,
                per_device_train_batch_size=3,
                num_generations=3,
                max_completion_length=8,
                dataloader_num_workers=2,
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

            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                self.assertFalse(torch.equal(param, new_param), f"Parameter {n} has not changed.")

    def test_training_with_generation_kwargs(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GRPOConfig(
                output_dir=tmp_dir,
                learning_rate=0.1,
                per_device_train_batch_size=3,
                num_generations=3,
                max_completion_length=8,
                generation_kwargs={"do_sample": True, "top_k": 50, "length_penalty": -0.1},
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

            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                self.assertFalse(torch.equal(param, new_param), f"Parameter {n} has not changed.")

    def test_training_with_reward_func_accessing_trainer_state(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        def reward_func(completions, **kwargs):
            trainer_state = kwargs.get("trainer_state")
            assert trainer_state is not None
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

    def test_compute_entropy_mask(self):
        trainer = GRPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=GRPOConfig(token_entropy_percentile_threshold=0.8),
        )

        entropies = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [0.7, 0.8, 0.9, 1.0, 1.1, 1.2]])
        completion_mask = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]])

        entropy_mask = trainer._compute_entropy_mask(entropies, completion_mask)

        self.assertEqual(entropy_mask.shape, entropies.shape)
        expected_mask = torch.tensor([[0, 0, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0]], dtype=torch.bool)
        self.assertTrue(torch.equal(entropy_mask, expected_mask))

        entropies = torch.tensor([[0.1, 0.2, 0.3, 1.4, 0.5, 0.14], [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]])
        completion_mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 0, 0]])

        expected_mask = torch.tensor([[0, 0, 0, 1, 0, 0], [0, 0, 0, 1, 0, 0]], dtype=torch.bool)
        entropy_mask = trainer._compute_entropy_mask(entropies, completion_mask)

        self.assertTrue(torch.equal(entropy_mask, expected_mask))


class GRPOImageProcessingTester(unittest.TestCase):
    def test_validate_and_preprocess_images_pil_images(self):
        from PIL import Image

        img1 = Image.new("RGB", (100, 100), color="red")
        img2 = Image.new("RGBA", (50, 50), color="blue")
        img3 = Image.new("L", (75, 75), color=128)
        images = [img1, img2, img3, None]

        processed = GRPOTrainer._validate_and_preprocess_images(images)

        self.assertEqual(len(processed), 4)
        self.assertEqual(processed[0].mode, "RGB")
        self.assertEqual(processed[0].size, (100, 100))
        self.assertEqual(processed[1].mode, "RGB")
        self.assertEqual(processed[1].size, (50, 50))
        self.assertEqual(processed[2].mode, "RGB")
        self.assertEqual(processed[2].size, (75, 75))
        self.assertIsNone(processed[3])


class DualModeBatchingTester(unittest.TestCase):
    def test_has_visual_inputs_detection(self):
        trainer = GRPOTrainer.__new__(GRPOTrainer)

        dict_with_visual = {"visual_inputs": BatchFeature({"pixel_values": torch.randn(2, 3, 224, 224)})}
        dict_without_visual = {"prompt_ids": torch.randn(2, 10)}
        dict_with_none_visual = {"visual_inputs": None}

        self.assertTrue(trainer._has_visual_inputs(dict_with_visual))
        self.assertFalse(trainer._has_visual_inputs(dict_without_visual))
        self.assertFalse(trainer._has_visual_inputs(dict_with_none_visual))

        list_with_visual = [{"prompt": "test", "image": torch.randn(3, 224, 224)}]
        list_without_visual = [{"prompt": "test"}]
        list_empty = []

        self.assertTrue(trainer._has_visual_inputs(list_with_visual))
        self.assertFalse(trainer._has_visual_inputs(list_without_visual))
        self.assertFalse(trainer._has_visual_inputs(list_empty))

    def test_split_vlm_batch_safely_with_realistic_data(self):
        trainer = GRPOTrainer.__new__(GRPOTrainer)

        batch_size = 4
        prompt_length = 50
        completion_length = 30

        visual_inputs = BatchFeature(
            {
                "input_ids": torch.randint(1, 1000, (batch_size, prompt_length)),
                "attention_mask": torch.ones(batch_size, prompt_length),
                "pixel_values": torch.randn(batch_size, 3, 224, 224),
                "image_grid_thw": torch.randint(1, 10, (batch_size, 3)),
            }
        )

        batch_data = {
            "prompt_ids": torch.randint(1, 1000, (batch_size, prompt_length)),
            "prompt_mask": torch.ones(batch_size, prompt_length),
            "completion_ids": torch.randint(1, 1000, (batch_size, completion_length)),
            "completion_mask": torch.ones(batch_size, completion_length),
            "advantages": torch.randn(batch_size),
            "old_per_token_logps": torch.randn(batch_size, completion_length),
            "ref_per_token_logps": torch.randn(batch_size, completion_length),
            "visual_inputs": visual_inputs,
        }

        chunks = trainer._split_vlm_batch_safely(batch_data, num_chunks=2)
        self.assertEqual(len(chunks), 2)

        for chunk in chunks:
            self.assertIn("prompt_ids", chunk)
            self.assertIn("visual_inputs", chunk)
            self.assertIsInstance(chunk["visual_inputs"], BatchFeature)
            expected_keys = ["input_ids", "attention_mask", "pixel_values", "image_grid_thw"]
            for key in expected_keys:
                self.assertIn(key, chunk["visual_inputs"])
            self.assertEqual(chunk["prompt_ids"].shape[0], 2)
            self.assertEqual(chunk["visual_inputs"]["input_ids"].shape[0], 2)

        for num_chunks in [1, 2, 4]:
            chunks = trainer._split_vlm_batch_safely(batch_data, num_chunks=num_chunks)
            self.assertEqual(len(chunks), num_chunks)
            for chunk in chunks:
                self.assertIsInstance(chunk["visual_inputs"], BatchFeature)
                chunk_size = batch_size // num_chunks
                self.assertEqual(chunk["visual_inputs"]["input_ids"].shape[0], chunk_size)
                self.assertEqual(chunk["advantages"].shape[0], chunk_size)

        chunks = trainer._split_vlm_batch_safely(batch_data, num_chunks=2)
        reconstructed_prompt_ids = torch.cat([chunk["prompt_ids"] for chunk in chunks])
        reconstructed_visual_input_ids = torch.cat([chunk["visual_inputs"]["input_ids"] for chunk in chunks])

        self.assertTrue(torch.equal(reconstructed_prompt_ids, batch_data["prompt_ids"]))
        self.assertTrue(torch.equal(reconstructed_visual_input_ids, visual_inputs["input_ids"]))

    def test_vlm_mode_includes_shuffling(self):
        trainer = GRPOTrainer.__new__(GRPOTrainer)
        trainer.is_vision_model = True
        trainer.args = GRPOConfig(per_device_train_batch_size=2, steps_per_generation=2, num_generations=2)
        trainer.num_iterations = 1
        trainer._step = 0
        trainer._buffered_inputs = None

        def mock_generate_and_score_completions(inputs):
            batch_feature = BatchFeature(
                {
                    "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]),
                    "pixel_values": torch.arange(48).reshape(4, 3, 4),
                    "spatial_merge_size": 4,
                }
            )
            return {
                "visual_inputs": batch_feature,
                "prompt_ids": torch.arange(8).reshape(4, 2),
                "completion_ids": torch.arange(20).reshape(4, 5),
                "advantages": torch.arange(4, dtype=torch.float),
            }

        trainer._generate_and_score_completions = mock_generate_and_score_completions
        trainer.model = type("MockModel", (), {"training": True})()

        original_shuffle = shuffle_tensor_dict
        shuffle_called = False

        def mock_shuffle(tensor_dict):
            nonlocal shuffle_called
            shuffle_called = True
            return original_shuffle(tensor_dict)

        split_input = None

        def mock_split(tensor_dict, num_chunks):
            nonlocal split_input
            split_input = tensor_dict
            return [tensor_dict] * num_chunks

        trainer._split_vlm_batch_safely = mock_split

        import trl.trainer.grpo_trainer as grpo_module

        original_module_shuffle = grpo_module.shuffle_tensor_dict
        grpo_module.shuffle_tensor_dict = mock_shuffle

        try:
            vlm_batch = [{"prompt": "test", "image": torch.randn(3, 224, 224)}]
            trainer._prepare_inputs_vlm_mode(vlm_batch)

            self.assertTrue(shuffle_called)
            self.assertIsNotNone(split_input)
            self.assertIn("visual_inputs", split_input)
            self.assertIn("prompt_ids", split_input)
            self.assertIsInstance(split_input["visual_inputs"], BatchFeature)
            self.assertEqual(split_input["visual_inputs"]["spatial_merge_size"], 4)

        finally:
            grpo_module.shuffle_tensor_dict = original_module_shuffle
