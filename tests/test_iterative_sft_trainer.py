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
from functools import partial

import torch
from datasets import Dataset
from parameterized import parameterized
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments

from trl import IterativeSFTTrainer


class IterativeTrainerTester(unittest.TestCase):
    def setUp(self):
        self.model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # get t5 as seq2seq example:
        model_id = "trl-internal-testing/tiny-T5ForConditionalGeneration"
        self.t5_model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        self.t5_tokenizer = AutoTokenizer.from_pretrained(model_id)

    def _init_tensor_dummy_dataset(self):
        dummy_dataset_dict = {
            "input_ids": [
                torch.tensor([5303, 3621, 3666, 1438, 318]),
                torch.tensor([3666, 1438, 318, 3666, 1438, 318]),
                torch.tensor([5303, 3621, 3666, 1438, 318]),
            ],
            "attention_mask": [
                torch.tensor([1, 1, 1, 1, 1]),
                torch.tensor([1, 1, 1, 1, 1, 1]),
                torch.tensor([1, 1, 1, 1, 1]),
            ],
            "labels": [
                torch.tensor([5303, 3621, 3666, 1438, 318]),
                torch.tensor([3666, 1438, 318, 3666, 1438, 318]),
                torch.tensor([5303, 3621, 3666, 1438, 318]),
            ],
        }

        dummy_dataset = Dataset.from_dict(dummy_dataset_dict)
        dummy_dataset.set_format("torch")
        return dummy_dataset

    def _init_textual_dummy_dataset(self):
        dummy_dataset_dict = {
            "texts": ["Testing the IterativeSFTTrainer.", "This is a test of the IterativeSFTTrainer"],
            "texts_labels": ["Testing the IterativeSFTTrainer.", "This is a test of the IterativeSFTTrainer"],
        }

        dummy_dataset = Dataset.from_dict(dummy_dataset_dict)
        dummy_dataset.set_format("torch")
        return dummy_dataset

    @parameterized.expand(
        [
            ["qwen", "tensor"],
            ["qwen", "text"],
            ["t5", "tensor"],
            ["t5", "text"],
        ]
    )
    def test_iterative_step_from_tensor(self, model_name, input_name):
        with tempfile.TemporaryDirectory() as tmp_dir:
            # initialize dataset
            if input_name == "tensor":
                dummy_dataset = self._init_tensor_dummy_dataset()
                inputs = {
                    "input_ids": dummy_dataset["input_ids"],
                    "attention_mask": dummy_dataset["attention_mask"],
                    "labels": dummy_dataset["labels"],
                }
            else:
                dummy_dataset = self._init_textual_dummy_dataset()
                inputs = {
                    "texts": dummy_dataset["texts"],
                    "texts_labels": dummy_dataset["texts_labels"],
                }

            if model_name == "qwen":
                model = self.model
                tokenizer = self.tokenizer
            else:
                model = self.t5_model
                tokenizer = self.t5_tokenizer

            training_args = TrainingArguments(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                max_steps=2,
                learning_rate=1e-3,
                report_to="none",
            )
            iterative_trainer = IterativeSFTTrainer(model=model, args=training_args, processing_class=tokenizer)
            iterative_trainer.optimizer.zero_grad = partial(iterative_trainer.optimizer.zero_grad, set_to_none=False)

            iterative_trainer.step(**inputs)

            for param in iterative_trainer.model.parameters():
                self.assertIsNotNone(param.grad)
