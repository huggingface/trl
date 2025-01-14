# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import torch
from datasets import load_dataset
from parameterized import parameterized

from trl import GRPOConfig, GRPOTrainer


class GRPOTrainerTester(unittest.TestCase):
    def test_init_minimal(self):
        # Test that GRPOTrainer can be instantiated with only model, reward_model and train_dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")
        GRPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            train_dataset=dataset,
        )

    @parameterized.expand([("standard_prompt_only",), ("conversational_prompt_only",)])
    def test_training(self, config_name):
        dataset = load_dataset("trl-internal-testing/zen", config_name, split="train")

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GRPOConfig(
                output_dir=tmp_dir,
                learning_rate=0.1,  # increase the learning rate to speed up the test
                per_device_train_batch_size=2,  # reduce the batch size to reduce memory usage
                num_generations=3,  # reduce the number of generations to reduce memory usage
                report_to="none",
            )
            trainer = GRPOTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                reward_model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                args=training_args,
                train_dataset=dataset,
            )

            previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

            # check the params have changed
            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                # check the params have changed - ignore 0 biases
                if param.sum() != 0:
                    self.assertFalse(torch.equal(param, new_param))
