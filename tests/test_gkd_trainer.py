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
import copy
import os
import tempfile
import unittest

import numpy as np
import pytest
import torch
from datasets import Dataset, Image, Sequence
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    LlavaForConditionalGeneration,
    TrainingArguments,
)

from trl import GKDConfig, GKDTrainer
from trl.import_utils import is_peft_available, is_pil_available
from trl.trainer import ConstantLengthDataset, DataCollatorForCompletionOnlyLM

from .testing_utils import require_peft, requires_pil


def formatting_prompts_func(example):
    text = f"### Question: {example['question']}\n ### Answer: {example['answer']}"
    return text


def formatting_prompts_func_batched(example):
    output_text = []
    for i, question in enumerate(example["question"]):
        text = f"### Question: {question}\n ### Answer: {example['answer'][i]}"
        output_text.append(text)
    return output_text


if is_peft_available():
    from peft import LoraConfig, PeftModel

class GKDTrainerTester(unittest.TestCase):
    r""" """

    @classmethod
    def setUpClass(cls):
        cls.model_id = "trl-internal-testing/dummy-GPT2-correct-vocab"
        cls.model = AutoModelForCausalLM.from_pretrained(cls.model_id)
        cls.teacher_model = AutoModelForCausalLM.from_pretrained(cls.model_id)
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_id)
        cls.tokenizer.pad_token = cls.tokenizer.eos_token
        cls.dummy_dataset = Dataset.from_dict(
            {
                "question": [
                    "Does llamas know how to code?",
                    "Does llamas know how to fly?",
                    "Does llamas know how to talk?",
                    "Does llamas know how to code?",
                    "Does llamas know how to fly?",
                    "Does llamas know how to talk?",
                    "Does llamas know how to swim?",
                ],
                "answer": [
                    "Yes, llamas are very good at coding.",
                    "No, llamas can't fly.",
                    "Yes, llamas are very good at talking.",
                    "Yes, llamas are very good at coding.",
                    "No, llamas can't fly.",
                    "Yes, llamas are very good at talking.",
                    "No, llamas can't swim.",
                ],
                "text": [
                    "### Question: Does llamas know how to code?\n ### Answer: Yes, llamas are very good at coding.",
                    "### Question: Does llamas know how to fly?\n ### Answer: No, llamas can't fly.",
                    "### Question: Does llamas know how to talk?\n ### Answer: Yes, llamas are very good at talking.",
                    "### Question: Does llamas know how to code?\n ### Answer: Yes, llamas are very good at coding.",
                    "### Question: Does llamas know how to fly?\n ### Answer: No, llamas can't fly.",
                    "### Question: Does llamas know how to talk?\n ### Answer: Yes, llamas are very good at talking.",
                    "### Question: Does llamas know how to swim?\n ### Answer: No, llamas can't swim.",
                ],
            }
        )
        cls.dummy_chatml_dataset = Dataset.from_dict(
            {
                "messages": [
                    [
                        {"role": "system", "content": "You are helpful"},
                        {"role": "user", "content": "Hello"},
                        {"role": "assistant", "content": "Hi, how can I help you?"},
                        {"role": "user", "content": "What is 2+2?"},
                        {"role": "assistant", "content": "4"},
                        {"role": "user", "content": "What is 3+3?"},
                        {"role": "assistant", "content": "6"},
                    ],
                    [
                        {"role": "system", "content": "You are helpful"},
                        {"role": "user", "content": "Hello"},
                        {"role": "assistant", "content": "Hi, how can I help you?"},
                    ],
                ]
            }
        )
        cls.dummy_instruction_dataset = Dataset.from_list(
            [
                {"prompt": "What is 2+2?", "completion": "4"},
                {"prompt": "What is 3+3?", "completion": "6"},
                {"prompt": "What is 4+4?", "completion": "8"},
                {"prompt": "What is 2+2?", "completion": "4"},
                {"prompt": "What is 3+3?", "completion": "6"},
                {"prompt": "What is 4+4?", "completion": "8"},
                {"prompt": "What is 2+2?", "completion": "4"},
                {"prompt": "What is 3+3?", "completion": "6"},
                {"prompt": "What is 4+4?", "completion": "8"},
                {"prompt": "What is 2+2?", "completion": "4"},
                {"prompt": "What is 3+3?", "completion": "6"},
                {"prompt": "What is 4+4?", "completion": "8"},
            ]
        )

        cls.train_dataset = ConstantLengthDataset(
            cls.tokenizer,
            cls.dummy_dataset,
            dataset_text_field=None,
            formatting_func=formatting_prompts_func,
            seq_length=16,
            num_of_sequences=16,
        )

        cls.eval_dataset = ConstantLengthDataset(
            cls.tokenizer,
            cls.dummy_dataset,
            dataset_text_field=None,
            formatting_func=formatting_prompts_func,
            seq_length=16,
            num_of_sequences=16,
        )

    def test_gkd_trainer(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GKDConfig(
                output_dir=tmp_dir,
                dataloader_drop_last=True,
                eval_strategy="steps",
                max_steps=4,
                eval_steps=2,
                save_steps=2,
                per_device_train_batch_size=2,
                packing=True,
            )

            trainer = GKDTrainer(
                model=self.model_id,
                teacher_model=self.model_id,
                args=training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
            )

            trainer.train()

            assert trainer.state.log_history[(-1)]["train_loss"] is not None
            assert trainer.state.log_history[0]["eval_loss"] is not None

            assert "model.safetensors" in os.listdir(tmp_dir + "/checkpoint-2")