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
from parameterized import parameterized

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, AutoModelForSeq2SeqLM

from trl import DPOTrainer


class DPOTrainerTester(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_id = "trl-internal-testing/dummy-GPT2-correct-vocab"
        cls.gpt2_model = AutoModelForCausalLM.from_pretrained(cls.model_id)
        cls.gpt2_ref_model = AutoModelForCausalLM.from_pretrained(cls.model_id)
        cls.gpt2_tokenizer = AutoTokenizer.from_pretrained(cls.model_id)
        cls.gpt2_tokenizer.pad_token = cls.gpt2_tokenizer.eos_token

        # get t5 as seq2seq example:
        model_id = "trl-internal-testing/tiny-T5ForConditionalGeneration-correct-vocab"
        cls.t5_model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        cls.t5_ref_model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        cls.t5_tokenizer = AutoTokenizer.from_pretrained(model_id)

    @parameterized.expand(
        [
            ["gpt2"],
            ["t5"],
        ]
    )
    def test_dpo_trainer(self, name):
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
                    "What is your name?",
                    "Which is the best programming language?",
                    "Which is the best programming language?",
                    "Which is the best programming language?",
                ],
                "chosen": [
                    "hi nice to meet you",
                    "I am fine",
                    "My name is Mary",
                    "My name is Mary",
                    "Python",
                    "Python",
                    "Python",
                ],
                "rejected": [
                    "leave me alone",
                    "I am not fine",
                    "Whats it to you?",
                    "I dont have a name",
                    "Javascript",
                    "C++",
                    "Java",
                ],
            }
            # fmt: on
            dummy_dataset = Dataset.from_dict(dummy_dataset_dict)

            if name == "gpt2":
                model = self.gpt2_model
                ref_model = self.gpt2_ref_model
                tokenizer = self.gpt2_tokenizer
            elif name == "t5":
                model = self.t5_model
                ref_model = self.t5_ref_model
                tokenizer = self.t5_tokenizer

            trainer = DPOTrainer(
                model=model,
                ref_model=ref_model,
                beta=0.1,
                args=training_args,
                tokenizer=tokenizer,
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
