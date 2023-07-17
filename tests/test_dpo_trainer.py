# Copyright 2023 The HuggingFace Team. All rights reserved.
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
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

from trl import DPOTrainer


class DPOTrainerTester(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_id = "trl-internal-testing/dummy-GPT2-correct-vocab"
        cls.model = AutoModelForCausalLM.from_pretrained(cls.model_id)
        cls.ref_model = AutoModelForCausalLM.from_pretrained(cls.model_id)
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_id)
        cls.tokenizer.pad_token = cls.tokenizer.eos_token

    def test_dpo_trainer(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = TrainingArguments(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                max_steps=3,
                remove_unused_columns=False,
                gradient_accumulation_steps=4,
                learning_rate=9e-1,
                evaluation_strategy="steps",
            )

            # fmt: off
            dummy_dataset_dict = {
                "prompt": [
                    "hello",
                    "how are you",
                    "What is your name?",
                    "Which is the best programming language?",
                ],
                "responses": [
                    ["hi nice to meet you", "leave me alone"],
                    ["I am not fine", "I am fine"],
                    ["My name is Mary", "Whats it to you?", "I dont have a name"],
                    ["Python", "Javascript", "C++", "Java"],
                ],
                "pairs": [
                    [(0, 1)],
                    [(1, 0)],
                    [(0, 2), (0, 1)],
                    [(0, 1), (0, 2), (0, 3)],
                ],
            }
            # fmt: on
            dummy_dataset = Dataset.from_dict(dummy_dataset_dict)

            trainer = DPOTrainer(
                model=self.model,
                ref_model=self.ref_model,
                beta=0.1,
                args=training_args,
                tokenizer=self.tokenizer,
                train_dataset=dummy_dataset,
                eval_dataset=dummy_dataset,
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
