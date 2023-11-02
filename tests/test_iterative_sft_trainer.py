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
from parameterized import parameterized
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments

from trl import IterativeSFTTrainer


class IterativeTrainerTester(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_id = "trl-internal-testing/dummy-GPT2-correct-vocab"
        cls.model = AutoModelForCausalLM.from_pretrained(cls.model_id)
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_id)
        cls.tokenizer.pad_token = cls.tokenizer.eos_token

        # get t5 as seq2seq example:
        model_id = "trl-internal-testing/tiny-T5ForConditionalGeneration-correct-vocab"
        cls.t5_model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        cls.t5_tokenizer = AutoTokenizer.from_pretrained(model_id)

    def _init_tensor_dummy_dataset(self):
        dummy_dataset_dict = {
            "input_ids": [torch.tensor([5303, 3621]), torch.tensor([3666, 1438, 318]), torch.tensor([5303, 3621])],
            "attention_mask": [torch.tensor([1, 1]), torch.tensor([1, 1, 1]), torch.tensor([1, 1])],
            "labels": [torch.tensor([5303, 3621]), torch.tensor([3666, 1438, 318]), torch.tensor([5303, 3621])],
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

    def setUp(self):
        # initialize trainer
        self.model.train()
        return super().setUp()

    @parameterized.expand(
        [
            ["gpt2", "tensor"],
            ["gpt2", "text"],
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

            if model_name == "gpt2":
                model = self.model
                tokenizer = self.tokenizer
            else:
                model = self.t5_model
                tokenizer = self.t5_tokenizer

            args = TrainingArguments(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                max_steps=2,
            )
            iterative_trainer = IterativeSFTTrainer(model=model, args=args, tokenizer=tokenizer)

            iterative_trainer.step(**inputs)

            for param in iterative_trainer.model.parameters():
                assert param.grad is not None
