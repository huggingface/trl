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
import os
import tempfile
import unittest

import numpy as np
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

from trl import SFTTrainer
from trl.import_utils import is_peft_available
from trl.trainer import ConstantLengthDataset, DataCollatorForCompletionOnlyLM

from .testing_utils import require_peft


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


class SFTTrainerTester(unittest.TestCase):
    r""" """

    @classmethod
    def setUpClass(cls):
        cls.model_id = "trl-internal-testing/dummy-GPT2-correct-vocab"
        cls.model = AutoModelForCausalLM.from_pretrained(cls.model_id)
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

    def test_constant_length_dataset(self):
        formatted_dataset = ConstantLengthDataset(
            self.tokenizer,
            self.dummy_dataset,
            dataset_text_field=None,
            formatting_func=formatting_prompts_func,
        )

        self.assertTrue(len(formatted_dataset) == len(self.dummy_dataset))
        self.assertTrue(len(formatted_dataset) > 0)

        for example in formatted_dataset:
            self.assertTrue("input_ids" in example)
            self.assertTrue("labels" in example)

            self.assertTrue(len(example["input_ids"]) == formatted_dataset.seq_length)
            self.assertTrue(len(example["labels"]) == formatted_dataset.seq_length)

            decoded_text = self.tokenizer.decode(example["input_ids"])
            self.assertTrue(("Question" in decoded_text) and ("Answer" in decoded_text))

    def test_sft_trainer(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = TrainingArguments(
                output_dir=tmp_dir,
                dataloader_drop_last=True,
                evaluation_strategy="steps",
                max_steps=4,
                eval_steps=2,
                save_steps=2,
                per_device_train_batch_size=2,
            )

            trainer = SFTTrainer(
                model=self.model_id,
                args=training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                packing=True,
            )

            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])
            self.assertIsNotNone(trainer.state.log_history[0]["eval_loss"])

            self.assertTrue("pytorch_model.bin" in os.listdir(tmp_dir + "/checkpoint-2"))

    def test_sft_trainer_uncorrect_data(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = TrainingArguments(
                output_dir=tmp_dir,
                dataloader_drop_last=True,
                evaluation_strategy="steps",
                max_steps=2,
                eval_steps=1,
                save_steps=1,
                per_device_train_batch_size=2,
            )

            with self.assertRaises(ValueError):
                _ = SFTTrainer(
                    model=self.model,
                    args=training_args,
                    train_dataset=self.dummy_dataset,
                    packing=True,
                )

            # This should work
            _ = SFTTrainer(
                model=self.model,
                args=training_args,
                train_dataset=self.dummy_dataset,
                formatting_func=formatting_prompts_func,
                packing=True,
            )

            # This should not work as well
            with self.assertRaises(ValueError):
                _ = SFTTrainer(
                    model=self.model,
                    args=training_args,
                    train_dataset=self.dummy_dataset,
                    formatting_func=formatting_prompts_func,
                    packing=False,
                )

            # but this shpuld work
            _ = SFTTrainer(
                model=self.model,
                args=training_args,
                train_dataset=self.dummy_dataset,
                formatting_func=formatting_prompts_func_batched,
                packing=False,
            )

    def test_sft_trainer_with_model_num_train_epochs(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = TrainingArguments(
                output_dir=tmp_dir,
                dataloader_drop_last=True,
                evaluation_strategy="steps",
                max_steps=2,
                eval_steps=1,
                save_steps=1,
                num_train_epochs=2,
                per_device_train_batch_size=2,
            )

            trainer = SFTTrainer(
                model=self.model,
                args=training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                packing=True,
            )

            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])
            self.assertIsNotNone(trainer.state.log_history[0]["eval_loss"])

            self.assertTrue("pytorch_model.bin" in os.listdir(tmp_dir + "/checkpoint-2"))

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = TrainingArguments(
                output_dir=tmp_dir,
                dataloader_drop_last=True,
                evaluation_strategy="steps",
                max_steps=2,
                save_steps=1,
                num_train_epochs=2,
                per_device_train_batch_size=2,
            )

            trainer = SFTTrainer(
                model=self.model,
                args=training_args,
                train_dataset=self.dummy_dataset,
                dataset_text_field="text",
                max_seq_length=16,
                num_of_sequences=16,
                packing=True,
            )

            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

            self.assertTrue("pytorch_model.bin" in os.listdir(tmp_dir + "/checkpoint-2"))

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = TrainingArguments(
                output_dir=tmp_dir,
                dataloader_drop_last=True,
                evaluation_strategy="steps",
                max_steps=2,
                save_steps=1,
                num_train_epochs=2,
                per_device_train_batch_size=2,
            )

            trainer = SFTTrainer(
                model=self.model,
                args=training_args,
                train_dataset=self.dummy_dataset,
                dataset_text_field="text",
                max_seq_length=16,
            )

            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

            self.assertTrue("pytorch_model.bin" in os.listdir(tmp_dir + "/checkpoint-1"))

    def test_sft_trainer_with_model(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = TrainingArguments(
                output_dir=tmp_dir,
                dataloader_drop_last=True,
                evaluation_strategy="steps",
                max_steps=2,
                eval_steps=1,
                save_steps=1,
                per_device_train_batch_size=2,
            )

            trainer = SFTTrainer(
                model=self.model,
                args=training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                packing=True,
            )

            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])
            self.assertIsNotNone(trainer.state.log_history[0]["eval_loss"])

            self.assertTrue("pytorch_model.bin" in os.listdir(tmp_dir + "/checkpoint-2"))

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = TrainingArguments(
                output_dir=tmp_dir,
                dataloader_drop_last=True,
                evaluation_strategy="steps",
                max_steps=2,
                save_steps=1,
                per_device_train_batch_size=2,
            )

            trainer = SFTTrainer(
                model=self.model,
                args=training_args,
                train_dataset=self.dummy_dataset,
                dataset_text_field="text",
                max_seq_length=16,
                num_of_sequences=16,
                packing=True,
            )

            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

            self.assertTrue("pytorch_model.bin" in os.listdir(tmp_dir + "/checkpoint-2"))

        # with formatting_func + packed
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = TrainingArguments(
                output_dir=tmp_dir,
                dataloader_drop_last=True,
                evaluation_strategy="steps",
                max_steps=2,
                save_steps=1,
                per_device_train_batch_size=2,
            )

            trainer = SFTTrainer(
                model=self.model,
                args=training_args,
                train_dataset=self.dummy_dataset,
                formatting_func=formatting_prompts_func,
                max_seq_length=16,
                num_of_sequences=16,
                packing=True,
            )

            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

            self.assertTrue("pytorch_model.bin" in os.listdir(tmp_dir + "/checkpoint-2"))

        # with formatting_func + packed
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = TrainingArguments(
                output_dir=tmp_dir,
                dataloader_drop_last=True,
                evaluation_strategy="steps",
                max_steps=2,
                save_steps=1,
                per_device_train_batch_size=2,
            )

            trainer = SFTTrainer(
                model=self.model,
                args=training_args,
                train_dataset=self.dummy_dataset,
                formatting_func=formatting_prompts_func_batched,
                max_seq_length=16,
            )

            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

            self.assertTrue("pytorch_model.bin" in os.listdir(tmp_dir + "/checkpoint-2"))

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = TrainingArguments(
                output_dir=tmp_dir,
                dataloader_drop_last=True,
                evaluation_strategy="steps",
                max_steps=2,
                save_steps=1,
                per_device_train_batch_size=2,
            )

            trainer = SFTTrainer(
                model=self.model,
                args=training_args,
                train_dataset=self.dummy_dataset,
                dataset_text_field="text",
                max_seq_length=16,
            )

            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

            self.assertTrue("pytorch_model.bin" in os.listdir(tmp_dir + "/checkpoint-1"))

    def test_data_collator_completion_lm(self):
        response_template = "### Response:\n"
        data_collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=self.tokenizer, mlm=False)

        text = """\n\n### Instructions:\nHello all this should be masked\n\n### Response:\nI have not been masked correctly."""
        encoded_text = self.tokenizer(text)
        encoded_text["input_ids"] = encoded_text["input_ids"]

        examples = [encoded_text]

        batch = data_collator(examples)
        labels = batch["labels"]
        last_pad_idx = np.where(labels == -100)[1][-1]
        result_text = self.tokenizer.decode(batch["input_ids"][0, last_pad_idx + 1 :])
        self.assertTrue(result_text == "I have not been masked correctly.")

    def test_sft_trainer_infinite_with_model(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = TrainingArguments(
                output_dir=tmp_dir,
                dataloader_drop_last=True,
                evaluation_strategy="steps",
                max_steps=5,
                eval_steps=1,
                save_steps=1,
                per_device_train_batch_size=2,
            )

            trainer = SFTTrainer(
                model=self.model,
                args=training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                packing=True,
                max_seq_length=500,
            )

            self.assertTrue(trainer.train_dataset.infinite)

            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])
            self.assertIsNotNone(trainer.state.log_history[0]["eval_loss"])

            # make sure the trainer did 5 steps
            self.assertTrue("pytorch_model.bin" in os.listdir(tmp_dir + "/checkpoint-5"))

    def test_sft_trainer_infinite_with_model_epochs(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = TrainingArguments(
                output_dir=tmp_dir,
                dataloader_drop_last=True,
                num_train_epochs=1,
                per_device_train_batch_size=2,
                save_strategy="epoch",
            )

            trainer = SFTTrainer(
                model=self.model,
                args=training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                packing=True,
                max_seq_length=500,
            )

            self.assertFalse(trainer.train_dataset.infinite)

            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

            # make sure the trainer did 5 steps
            self.assertTrue("pytorch_model.bin" in os.listdir(tmp_dir + "/checkpoint-4"))

    @require_peft
    def test_peft_sft_trainer(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = TrainingArguments(
                output_dir=tmp_dir,
                dataloader_drop_last=True,
                evaluation_strategy="steps",
                max_steps=4,
                eval_steps=2,
                save_steps=2,
                per_device_train_batch_size=2,
            )

            peft_config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )

            trainer = SFTTrainer(
                model=self.model_id,
                args=training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                peft_config=peft_config,
                packing=True,
            )

            self.assertTrue(isinstance(trainer.model, PeftModel))

            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])
            self.assertIsNotNone(trainer.state.log_history[0]["eval_loss"])

            self.assertTrue("adapter_model.bin" in os.listdir(tmp_dir + "/checkpoint-2"))
            self.assertTrue("adapter_config.json" in os.listdir(tmp_dir + "/checkpoint-2"))
            self.assertTrue("pytorch_model.bin" not in os.listdir(tmp_dir + "/checkpoint-2"))
