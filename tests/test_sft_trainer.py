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

from trl import SFTConfig, SFTTrainer
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

if is_pil_available():
    from PIL import Image as PILImage


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

        if is_pil_available():
            cls.dummy_vsft_instruction_dataset = Dataset.from_dict(
                {
                    "messages": [
                        [
                            {
                                "role": "user",
                                "content": [{"type": "text", "text": "What is in this image?"}, {"type": "image"}],
                            },
                            {
                                "role": "assistant",
                                "content": [{"type": "text", "text": "It is random noise."}],
                            },
                            {
                                "role": "user",
                                "content": [{"type": "text", "text": "Oh ye, you are right, what is 1+1"}],
                            },
                            {
                                "role": "assistant",
                                "content": [{"type": "text", "text": "2"}],
                            },
                        ],
                        [
                            {
                                "role": "user",
                                "content": [{"type": "text", "text": "What is in this image?"}, {"type": "image"}],
                            },
                            {
                                "role": "assistant",
                                "content": [{"type": "text", "text": "It is random noise."}],
                            },
                        ],
                    ],
                    "images": [
                        [PILImage.fromarray((np.random.rand(40, 50, 3) * 255).astype("uint8")).convert("RGBA")],
                        [PILImage.fromarray((np.random.rand(50, 60, 3) * 255).astype("uint8")).convert("RGBA")],
                    ],
                }
            )
            cls.dummy_vsft_instruction_dataset = cls.dummy_vsft_instruction_dataset.cast_column(
                "images", Sequence(Image())
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

        assert len(formatted_dataset) == len(self.dummy_dataset)
        assert len(formatted_dataset) > 0

        for example in formatted_dataset:
            assert "input_ids" in example
            assert "labels" in example

            assert len(example["input_ids"]) == formatted_dataset.seq_length
            assert len(example["labels"]) == formatted_dataset.seq_length

            decoded_text = self.tokenizer.decode(example["input_ids"])
            assert ("Question" in decoded_text) and ("Answer" in decoded_text)

    def test_sft_trainer_backward_compatibility(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = TrainingArguments(
                output_dir=tmp_dir,
                eval_strategy="steps",
                max_steps=4,
                eval_steps=2,
                save_steps=2,
                per_device_train_batch_size=2,
                hub_token="not_a_real_token",
            )

            trainer = SFTTrainer(
                model=self.model_id,
                args=training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
            )

            assert trainer.args.hub_token == training_args.hub_token

            trainer.train()

            assert trainer.state.log_history[(-1)]["train_loss"] is not None
            assert trainer.state.log_history[0]["eval_loss"] is not None

            assert "model.safetensors" in os.listdir(tmp_dir + "/checkpoint-2")

    def test_sft_trainer(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = SFTConfig(
                output_dir=tmp_dir,
                dataloader_drop_last=True,
                eval_strategy="steps",
                max_steps=4,
                eval_steps=2,
                save_steps=2,
                per_device_train_batch_size=2,
                packing=True,
            )

            trainer = SFTTrainer(
                model=self.model_id,
                args=training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
            )

            trainer.train()

            assert trainer.state.log_history[(-1)]["train_loss"] is not None
            assert trainer.state.log_history[0]["eval_loss"] is not None

            assert "model.safetensors" in os.listdir(tmp_dir + "/checkpoint-2")

    def test_sft_trainer_uncorrect_data(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = SFTConfig(
                output_dir=tmp_dir,
                dataloader_drop_last=True,
                eval_strategy="steps",
                max_steps=2,
                eval_steps=1,
                save_steps=1,
                per_device_train_batch_size=2,
                packing=True,
            )

            with pytest.raises(ValueError):
                _ = SFTTrainer(
                    model=self.model,
                    args=training_args,
                    train_dataset=self.dummy_dataset,
                )
            # this should work since the dummy chatml include the correct format
            training_args = SFTConfig(
                output_dir=tmp_dir,
                dataloader_drop_last=True,
                eval_strategy="steps",
                max_steps=2,
                eval_steps=1,
                save_steps=1,
                per_device_train_batch_size=2,
                max_seq_length=32,  # make sure there is at least 1 packed sequence
                num_of_sequences=32,
                packing=True,
            )
            _ = SFTTrainer(
                model=self.model,
                args=training_args,
                train_dataset=self.dummy_chatml_dataset,
            )

            training_args = SFTConfig(
                output_dir=tmp_dir,
                dataloader_drop_last=True,
                eval_strategy="steps",
                max_steps=2,
                eval_steps=1,
                save_steps=1,
                per_device_train_batch_size=2,
                packing=False,
            )
            _ = SFTTrainer(
                model=self.model,
                args=training_args,
                train_dataset=self.dummy_chatml_dataset,
            )
            # this should work since the dummy instruction dataset is the correct format

            training_args = SFTConfig(
                output_dir=tmp_dir,
                dataloader_drop_last=True,
                eval_strategy="steps",
                max_steps=2,
                eval_steps=1,
                save_steps=1,
                per_device_train_batch_size=2,
                max_seq_length=16,  # make sure there is at least 1 packed sequence
                packing=True,
            )
            _ = SFTTrainer(
                model=self.model,
                args=training_args,
                train_dataset=self.dummy_instruction_dataset,
            )

            training_args = SFTConfig(
                output_dir=tmp_dir,
                dataloader_drop_last=True,
                eval_strategy="steps",
                max_steps=2,
                eval_steps=1,
                save_steps=1,
                per_device_train_batch_size=2,
                packing=False,
            )
            _ = SFTTrainer(
                model=self.model,
                args=training_args,
                train_dataset=self.dummy_instruction_dataset,
            )

            training_args = SFTConfig(
                output_dir=tmp_dir,
                dataloader_drop_last=True,
                eval_strategy="steps",
                max_steps=2,
                eval_steps=1,
                save_steps=1,
                per_device_train_batch_size=2,
                max_seq_length=32,  # make sure there is at least 1 packed sequence
                packing=True,
            )
            # This should work
            _ = SFTTrainer(
                model=self.model,
                args=training_args,
                train_dataset=self.dummy_dataset,
                formatting_func=formatting_prompts_func,
            )

            with pytest.raises(ValueError):
                # This should not work because not enough data for one sample
                training_args = SFTConfig(
                    output_dir=tmp_dir,
                    dataloader_drop_last=True,
                    eval_strategy="steps",
                    max_steps=2,
                    eval_steps=1,
                    save_steps=1,
                    per_device_train_batch_size=2,
                    max_seq_length=1024,  # make sure there is NOT at least 1 packed sequence
                    packing=True,
                )
                _ = SFTTrainer(
                    model=self.model,
                    args=training_args,
                    train_dataset=self.dummy_dataset,
                    formatting_func=formatting_prompts_func,
                )

            # This should not work as well
            with pytest.raises(ValueError):
                training_args = SFTConfig(
                    output_dir=tmp_dir,
                    dataloader_drop_last=True,
                    eval_strategy="steps",
                    max_steps=2,
                    eval_steps=1,
                    save_steps=1,
                    per_device_train_batch_size=2,
                    packing=False,
                )
                _ = SFTTrainer(
                    model=self.model,
                    args=training_args,
                    train_dataset=self.dummy_dataset,
                    formatting_func=formatting_prompts_func,
                )

            # but this should work
            training_args = SFTConfig(
                output_dir=tmp_dir,
                dataloader_drop_last=True,
                eval_strategy="steps",
                max_steps=2,
                eval_steps=1,
                save_steps=1,
                per_device_train_batch_size=2,
                packing=False,
            )
            _ = SFTTrainer(
                model=self.model,
                args=training_args,
                train_dataset=self.dummy_dataset,
                formatting_func=formatting_prompts_func_batched,
            )

    def test_sft_trainer_with_model_num_train_epochs(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = SFTConfig(
                output_dir=tmp_dir,
                dataloader_drop_last=True,
                eval_strategy="steps",
                max_steps=2,
                eval_steps=1,
                save_steps=1,
                num_train_epochs=2,
                per_device_train_batch_size=2,
                packing=True,
            )

            trainer = SFTTrainer(
                model=self.model,
                args=training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
            )

            trainer.train()

            assert trainer.state.log_history[(-1)]["train_loss"] is not None
            assert trainer.state.log_history[0]["eval_loss"] is not None

            assert "model.safetensors" in os.listdir(tmp_dir + "/checkpoint-2")

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = SFTConfig(
                output_dir=tmp_dir,
                dataloader_drop_last=True,
                eval_strategy="steps",
                max_steps=2,
                save_steps=1,
                num_train_epochs=2,
                per_device_train_batch_size=2,
                dataset_text_field="text",
                max_seq_length=16,
                num_of_sequences=16,
                packing=True,
            )

            trainer = SFTTrainer(
                model=self.model,
                args=training_args,
                train_dataset=self.dummy_dataset,
            )

            trainer.train()

            assert trainer.state.log_history[(-1)]["train_loss"] is not None

            assert "model.safetensors" in os.listdir(tmp_dir + "/checkpoint-2")

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = SFTConfig(
                output_dir=tmp_dir,
                dataloader_drop_last=True,
                eval_strategy="steps",
                max_steps=2,
                save_steps=1,
                num_train_epochs=2,
                per_device_train_batch_size=2,
                dataset_text_field="text",
                max_seq_length=16,
            )

            trainer = SFTTrainer(
                model=self.model,
                args=training_args,
                train_dataset=self.dummy_dataset,
            )

            trainer.train()

            assert trainer.state.log_history[(-1)]["train_loss"] is not None

            assert "model.safetensors" in os.listdir(tmp_dir + "/checkpoint-1")

    def test_sft_trainer_with_model(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = SFTConfig(
                output_dir=tmp_dir,
                dataloader_drop_last=True,
                eval_strategy="steps",
                max_steps=2,
                eval_steps=1,
                save_steps=1,
                per_device_train_batch_size=2,
                packing=True,
            )

            trainer = SFTTrainer(
                model=self.model,
                args=training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
            )

            trainer.train()

            assert trainer.state.log_history[(-1)]["train_loss"] is not None
            assert trainer.state.log_history[0]["eval_loss"] is not None

            assert "model.safetensors" in os.listdir(tmp_dir + "/checkpoint-2")

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = SFTConfig(
                output_dir=tmp_dir,
                dataloader_drop_last=True,
                eval_strategy="steps",
                max_steps=2,
                save_steps=1,
                per_device_train_batch_size=2,
                dataset_text_field="text",
                max_seq_length=16,
                num_of_sequences=16,
                packing=True,
            )

            trainer = SFTTrainer(
                model=self.model,
                args=training_args,
                train_dataset=self.dummy_dataset,
            )

            trainer.train()

            assert trainer.state.log_history[(-1)]["train_loss"] is not None

            assert "model.safetensors" in os.listdir(tmp_dir + "/checkpoint-2")

        # with formatting_func + packed
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = SFTConfig(
                output_dir=tmp_dir,
                dataloader_drop_last=True,
                eval_strategy="steps",
                max_steps=2,
                save_steps=1,
                per_device_train_batch_size=2,
                max_seq_length=16,
                num_of_sequences=16,
                packing=True,
            )

            trainer = SFTTrainer(
                model=self.model,
                args=training_args,
                train_dataset=self.dummy_dataset,
                formatting_func=formatting_prompts_func,
            )

            trainer.train()

            assert trainer.state.log_history[(-1)]["train_loss"] is not None

            assert "model.safetensors" in os.listdir(tmp_dir + "/checkpoint-2")

        # with formatting_func + packed
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = SFTConfig(
                output_dir=tmp_dir,
                dataloader_drop_last=True,
                eval_strategy="steps",
                max_steps=2,
                save_steps=1,
                per_device_train_batch_size=2,
                max_seq_length=16,
            )

            trainer = SFTTrainer(
                model=self.model,
                args=training_args,
                train_dataset=self.dummy_dataset,
                formatting_func=formatting_prompts_func_batched,
            )

            trainer.train()

            assert trainer.state.log_history[(-1)]["train_loss"] is not None

            assert "model.safetensors" in os.listdir(tmp_dir + "/checkpoint-2")

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = SFTConfig(
                output_dir=tmp_dir,
                dataloader_drop_last=True,
                eval_strategy="steps",
                max_steps=2,
                save_steps=1,
                per_device_train_batch_size=2,
                dataset_text_field="text",
                max_seq_length=16,
            )

            trainer = SFTTrainer(
                model=self.model,
                args=training_args,
                train_dataset=self.dummy_dataset,
            )

            trainer.train()

            assert trainer.state.log_history[(-1)]["train_loss"] is not None

            assert "model.safetensors" in os.listdir(tmp_dir + "/checkpoint-1")

    def test_sft_trainer_with_multiple_eval_datasets(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = SFTConfig(
                output_dir=tmp_dir,
                dataloader_drop_last=True,
                eval_strategy="steps",
                max_steps=1,
                eval_steps=1,
                save_steps=1,
                per_device_train_batch_size=2,
                packing=True,
            )

            trainer = SFTTrainer(
                model=self.model_id,
                args=training_args,
                train_dataset=self.train_dataset,
                eval_dataset={
                    "data1": self.eval_dataset,
                    "data2": self.eval_dataset,
                },
            )

            trainer.train()

            assert trainer.state.log_history[(-1)]["train_loss"] is not None
            assert trainer.state.log_history[0]["eval_data1_loss"] is not None
            assert trainer.state.log_history[1]["eval_data2_loss"] is not None

            assert "model.safetensors" in os.listdir(tmp_dir + "/checkpoint-1")

    def test_data_collator_completion_lm(self):
        response_template = "### Response:\n"
        data_collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=self.tokenizer, mlm=False)

        text = """\n\n### Instructions:\nHello all this should be masked\n\n### Response:\nI have not been masked correctly."""
        encoded_text = self.tokenizer(text)

        examples = [encoded_text]

        batch = data_collator(examples)
        labels = batch["labels"]
        last_pad_idx = np.where(labels == -100)[1][-1]
        result_text = self.tokenizer.decode(batch["input_ids"][0, last_pad_idx + 1 :])
        assert result_text == "I have not been masked correctly."

    def test_data_collator_completion_lm_with_multiple_text(self):
        tokenizer = copy.deepcopy(self.tokenizer)
        tokenizer.padding_side = "left"

        response_template = "### Response:\n"
        data_collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer, mlm=False)

        text1 = """\n\n### Instructions:\nHello all this should be masked\n\n### Response:\nI have not been masked correctly."""
        text2 = """\n\n### Instructions:\nThis is another longer text that should also be masked. This text is significantly longer than the previous one.\n\n### Response:\nI have not been masked correctly."""

        encoded_text1 = tokenizer(text1)
        encoded_text2 = tokenizer(text2)

        examples = [encoded_text1, encoded_text2]

        batch = data_collator(examples)

        for i in range(2):
            labels = batch["labels"][i]
            last_pad_idx = np.where(labels == -100)[0][-1]
            result_text = tokenizer.decode(batch["input_ids"][i, last_pad_idx + 1 :])
            assert result_text == "I have not been masked correctly."

    def test_data_collator_chat_completion_lm(self):
        instruction_template = "### Human:"
        assistant_template = "### Assistant:"
        data_collator = DataCollatorForCompletionOnlyLM(
            response_template=assistant_template,
            instruction_template=instruction_template,
            tokenizer=self.tokenizer,
            mlm=False,
        )

        text = """### Human: Hello all this should be masked.### Assistant: I should not be masked.### Human: All this should be masked too.### Assistant: I should not be masked too."""
        encoded_text = self.tokenizer(text)

        examples = [encoded_text]

        batch = data_collator(examples)
        labels = batch["labels"]
        non_masked_tokens = batch["input_ids"][labels != -100]
        result_text = self.tokenizer.decode(non_masked_tokens)
        assert result_text == " I should not be masked. I should not be masked too."

    def test_data_collator_chat_completion_lm_with_multiple_text(self):
        tokenizer = copy.deepcopy(self.tokenizer)
        tokenizer.padding_side = "left"

        instruction_template = "### Human:"
        assistant_template = "### Assistant:"
        data_collator = DataCollatorForCompletionOnlyLM(
            response_template=assistant_template,
            instruction_template=instruction_template,
            tokenizer=tokenizer,
            mlm=False,
        )

        text1 = """### Human: Hello all this should be masked.### Assistant: I should not be masked."""
        text2 = """### Human: Hello all this should be masked.### Assistant: I should not be masked.### Human: All this should be masked too.### Assistant: I should not be masked too."""
        encoded_text1 = tokenizer(text1)
        encoded_text2 = tokenizer(text2)

        examples = [encoded_text1, encoded_text2]

        batch = data_collator(examples)
        labels = batch["labels"]
        input_ids = batch["input_ids"]

        non_masked_tokens1 = input_ids[0][labels[0] != -100]
        result_text1 = tokenizer.decode(non_masked_tokens1)
        assert result_text1 == " I should not be masked."

        non_masked_tokens2 = input_ids[1][labels[1] != -100]
        result_text2 = tokenizer.decode(non_masked_tokens2)
        assert result_text2 == " I should not be masked. I should not be masked too."

    def test_sft_trainer_infinite_with_model(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = SFTConfig(
                output_dir=tmp_dir,
                dataloader_drop_last=True,
                eval_strategy="steps",
                max_steps=5,
                eval_steps=1,
                save_steps=1,
                per_device_train_batch_size=2,
                packing=True,
                max_seq_length=500,
            )

            trainer = SFTTrainer(
                model=self.model,
                args=training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
            )

            assert trainer.train_dataset.infinite

            trainer.train()

            assert trainer.state.log_history[(-1)]["train_loss"] is not None
            assert trainer.state.log_history[0]["eval_loss"] is not None

            # make sure the trainer did 5 steps
            assert "model.safetensors" in os.listdir(tmp_dir + "/checkpoint-5")

    def test_sft_trainer_infinite_with_model_epochs(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = SFTConfig(
                output_dir=tmp_dir,
                dataloader_drop_last=True,
                num_train_epochs=1,
                per_device_train_batch_size=2,
                save_strategy="epoch",
                packing=True,
                max_seq_length=500,
            )

            trainer = SFTTrainer(
                model=self.model,
                args=training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
            )

            assert not trainer.train_dataset.infinite

            trainer.train()

            assert trainer.state.log_history[(-1)]["train_loss"] is not None

            # make sure the trainer did 5 steps
            assert "model.safetensors" in os.listdir(tmp_dir + "/checkpoint-4")

    def test_sft_trainer_with_model_neftune(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = SFTConfig(
                output_dir=tmp_dir,
                dataloader_drop_last=True,
                eval_strategy="steps",
                max_steps=2,
                eval_steps=1,
                save_steps=1,
                per_device_train_batch_size=2,
                neftune_noise_alpha=5,
                packing=True,
            )

            trainer = SFTTrainer(
                model=self.model,
                args=training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
            )

            trainer.model = trainer._trl_activate_neftune(trainer.model)

            device = trainer.model.get_input_embeddings().weight.device
            trainer.model.train()

            torch.random.manual_seed(42)
            embeds_neftune = trainer.model.get_input_embeddings()(torch.LongTensor([[1, 0, 1]]).to(device))

            torch.random.manual_seed(24)
            embeds_neftune_2 = trainer.model.get_input_embeddings()(torch.LongTensor([[1, 0, 1]]).to(device))

            assert not torch.allclose(embeds_neftune, embeds_neftune_2)
            assert len(trainer.model.get_input_embeddings()._forward_hooks) > 0

            trainer.neftune_hook_handle.remove()

            trainer.train()

            # Make sure forward pass works fine
            _ = trainer.model(torch.LongTensor([[1, 0, 1]]).to(device))
            assert len(trainer.model.get_input_embeddings()._forward_hooks) == 0

    @require_peft
    def test_peft_sft_trainer_str(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            peft_config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )

            training_args = SFTConfig(packing=True, output_dir=tmp_dir)

            _ = SFTTrainer(
                model=self.model_id,
                args=training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                peft_config=peft_config,
            )

    @require_peft
    def test_peft_sft_trainer(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = SFTConfig(
                output_dir=tmp_dir,
                dataloader_drop_last=True,
                eval_strategy="steps",
                max_steps=4,
                eval_steps=2,
                save_steps=2,
                per_device_train_batch_size=2,
                packing=True,
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
            )

            assert isinstance(trainer.model, PeftModel)

            trainer.train()

            assert trainer.state.log_history[(-1)]["train_loss"] is not None
            assert trainer.state.log_history[0]["eval_loss"] is not None

            assert "adapter_model.safetensors" in os.listdir(tmp_dir + "/checkpoint-2")
            assert "adapter_config.json" in os.listdir(tmp_dir + "/checkpoint-2")
            assert "model.safetensors" not in os.listdir(tmp_dir + "/checkpoint-2")

    @require_peft
    def test_peft_sft_trainer_gc(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = SFTConfig(
                output_dir=tmp_dir,
                dataloader_drop_last=True,
                eval_strategy="steps",
                max_steps=4,
                eval_steps=2,
                save_steps=2,
                per_device_train_batch_size=2,
                gradient_checkpointing=True,
                packing=True,
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
            )

            assert isinstance(trainer.model, PeftModel)

            trainer.train()

            assert trainer.state.log_history[(-1)]["train_loss"] is not None
            assert trainer.state.log_history[0]["eval_loss"] is not None

            assert "adapter_model.safetensors" in os.listdir(tmp_dir + "/checkpoint-2")
            assert "adapter_config.json" in os.listdir(tmp_dir + "/checkpoint-2")
            assert "model.safetensors" not in os.listdir(tmp_dir + "/checkpoint-2")

    @require_peft
    def test_peft_sft_trainer_neftune(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = SFTConfig(
                output_dir=tmp_dir,
                dataloader_drop_last=True,
                eval_strategy="steps",
                max_steps=4,
                eval_steps=2,
                save_steps=2,
                per_device_train_batch_size=2,
                neftune_noise_alpha=5,
                packing=True,
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
            )

            trainer.model = trainer._trl_activate_neftune(trainer.model)

            assert isinstance(trainer.model, PeftModel)

            device = trainer.model.get_input_embeddings().weight.device
            trainer.model.train()

            torch.random.manual_seed(42)
            embeds_neftune = trainer.model.get_input_embeddings()(torch.LongTensor([[1, 0, 1]]).to(device))

            torch.random.manual_seed(24)
            embeds_neftune_2 = trainer.model.get_input_embeddings()(torch.LongTensor([[1, 0, 1]]).to(device))

            assert not torch.allclose(embeds_neftune, embeds_neftune_2)
            assert len(trainer.model.get_input_embeddings()._forward_hooks) > 0

            trainer.neftune_hook_handle.remove()

            trainer.train()

            assert trainer.state.log_history[(-1)]["train_loss"] is not None
            assert trainer.state.log_history[0]["eval_loss"] is not None

            assert "adapter_model.safetensors" in os.listdir(tmp_dir + "/checkpoint-2")
            assert "adapter_config.json" in os.listdir(tmp_dir + "/checkpoint-2")
            assert "model.safetensors" not in os.listdir(tmp_dir + "/checkpoint-2")

            # Make sure forward pass works fine to check if embeddings forward is not broken.
            _ = trainer.model(torch.LongTensor([[1, 0, 1]]).to(device))
            assert len(trainer.model.get_input_embeddings()._forward_hooks) == 0

    @require_peft
    def test_peft_sft_trainer_tag(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = SFTConfig(
                output_dir=tmp_dir,
                dataloader_drop_last=True,
                eval_strategy="steps",
                max_steps=4,
                eval_steps=2,
                save_steps=2,
                per_device_train_batch_size=2,
                gradient_checkpointing=True,
                packing=True,
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
            )

            assert trainer.model.model_tags == trainer._tag_names

    @require_peft
    def test_sft_trainer_tag(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = SFTConfig(
                output_dir=tmp_dir,
                dataloader_drop_last=True,
                eval_strategy="steps",
                max_steps=4,
                eval_steps=2,
                save_steps=2,
                per_device_train_batch_size=2,
                gradient_checkpointing=True,
                packing=True,
            )

            trainer = SFTTrainer(
                model=self.model_id,
                args=training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
            )

            assert trainer.model.model_tags == trainer._tag_names

    def test_sft_trainer_eval_packing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = SFTConfig(
                output_dir=tmp_dir,
                dataloader_drop_last=True,
                eval_strategy="steps",
                max_steps=4,
                eval_steps=2,
                save_steps=2,
                per_device_train_batch_size=2,
                gradient_checkpointing=True,
                packing=True,
                max_seq_length=32,  # make sure there is at least 1 packed sequence
                eval_packing=False,
            )

            trainer = SFTTrainer(
                model=self.model_id,
                args=training_args,
                train_dataset=self.dummy_chatml_dataset,
                eval_dataset=self.dummy_chatml_dataset,
            )

            assert len(trainer.train_dataset["input_ids"]) == 1
            assert len(trainer.eval_dataset["input_ids"]) != 1

            training_args = SFTConfig(
                output_dir=tmp_dir,
                dataloader_drop_last=True,
                eval_strategy="steps",
                max_steps=4,
                eval_steps=2,
                save_steps=2,
                per_device_train_batch_size=2,
                gradient_checkpointing=True,
                max_seq_length=32,  # make sure there is at least 1 packed sequence
                packing=True,
            )
            trainer = SFTTrainer(
                model=self.model_id,
                args=training_args,
                train_dataset=self.dummy_chatml_dataset,
                eval_dataset=self.dummy_chatml_dataset,
            )

            assert len(trainer.train_dataset["input_ids"]) == 1
            assert len(trainer.eval_dataset["input_ids"]) == 1

            training_args = SFTConfig(
                output_dir=tmp_dir,
                dataloader_drop_last=True,
                eval_strategy="steps",
                max_steps=4,
                eval_steps=2,
                save_steps=2,
                per_device_train_batch_size=2,
                gradient_checkpointing=True,
                max_seq_length=32,  # make sure there is at least 1 packed sequence
                packing=False,
            )
            trainer = SFTTrainer(
                model=self.model_id,
                args=training_args,
                train_dataset=self.dummy_chatml_dataset,
                eval_dataset=self.dummy_chatml_dataset,
            )

            assert len(trainer.train_dataset["input_ids"]) != 1
            assert len(trainer.eval_dataset["input_ids"]) != 1

    @requires_pil
    def test_sft_trainer_skip_prepare_dataset(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = SFTConfig(
                output_dir=tmp_dir,
                dataloader_drop_last=True,
                eval_strategy="steps",
                max_steps=4,
                eval_steps=2,
                save_steps=2,
                per_device_train_batch_size=2,
                gradient_checkpointing=True,
                remove_unused_columns=False,
                dataset_text_field="text",  # need a dummy field
                dataset_kwargs={"skip_prepare_dataset": True},
            )

            trainer = SFTTrainer(
                model=self.model_id,
                args=training_args,
                train_dataset=self.dummy_vsft_instruction_dataset,
                eval_dataset=self.dummy_vsft_instruction_dataset,
            )
            assert trainer.train_dataset.features == self.dummy_vsft_instruction_dataset.features
            assert trainer.eval_dataset.features == self.dummy_vsft_instruction_dataset.features

    def test_sft_trainer_skip_prepare_dataset_with_no_packing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = SFTConfig(
                output_dir=tmp_dir,
                dataloader_drop_last=True,
                eval_strategy="steps",
                max_steps=4,
                eval_steps=2,
                save_steps=2,
                per_device_train_batch_size=2,
                gradient_checkpointing=True,
                remove_unused_columns=False,
                packing=False,
                dataset_kwargs={"skip_prepare_dataset": True},
            )

            trainer = SFTTrainer(
                model=self.model_id,
                args=training_args,
                train_dataset=self.dummy_dataset,
            )
            assert trainer.train_dataset.features == self.dummy_dataset.features

    @requires_pil
    def test_sft_trainer_llava(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = SFTConfig(
                output_dir=tmp_dir,
                dataloader_drop_last=True,
                eval_strategy="steps",
                max_steps=4,
                eval_steps=2,
                save_steps=2,
                per_device_train_batch_size=2,
                per_device_eval_batch_size=2,
                remove_unused_columns=False,
                dataset_text_field="text",  # need a dummy field
                dataset_kwargs={"skip_prepare_dataset": True},
            )
            tiny_llava = LlavaForConditionalGeneration.from_pretrained(
                "trl-internal-testing/tiny-random-LlavaForConditionalGeneration"
            )
            processor = AutoProcessor.from_pretrained("trl-internal-testing/tiny-random-LlavaForConditionalGeneration")

            processor.tokenizer.chat_template = """{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. {% for message in messages %}{% if message['role'] == 'user' %}USER: {% else %}ASSISTANT: {% endif %}{% for item in message['content'] %}{% if item['type'] == 'text' %}{{ item['text'] }}{% elif item['type'] == 'image' %}<image>{% endif %}{% endfor %}{% if message['role'] == 'user' %} {% else %}{{eos_token}}{% endif %}{% endfor %}{% if add_generation_prompt %}ASSISTANT: {% endif %}"""

            class LLavaDataCollator:
                def __init__(self, processor):
                    self.processor = processor

                def __call__(self, examples):
                    texts = []
                    images = []
                    for example in examples:
                        if len(example["images"]) > 1:
                            raise ValueError("This collator only supports one image per example")
                        messages = example["messages"]
                        text = self.processor.tokenizer.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=False
                        )
                        texts.append(text)
                        images.append(example["images"][0])

                    batch = self.processor(texts, images, return_tensors="pt", padding=True)

                    labels = batch["input_ids"].clone()
                    if self.processor.tokenizer.pad_token_id is not None:
                        labels[labels == self.processor.tokenizer.pad_token_id] = -100
                    batch["labels"] = labels

                    return batch

            data_collator = LLavaDataCollator(processor)

            trainer = SFTTrainer(
                model=tiny_llava,
                args=training_args,
                train_dataset=self.dummy_vsft_instruction_dataset,
                eval_dataset=self.dummy_vsft_instruction_dataset,
                data_collator=data_collator,
            )

            trainer.train()

            assert trainer.state.log_history[(-1)]["train_loss"] is not None
            assert trainer.state.log_history[0]["eval_loss"] is not None

            assert "model.safetensors" in os.listdir(tmp_dir + "/checkpoint-2")

    def test_sft_trainer_torch_dtype(self):
        # See https://github.com/huggingface/trl/issues/1751
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = SFTConfig(
                output_dir=tmp_dir,
                eval_strategy="steps",
                max_steps=4,
                eval_steps=2,
                save_steps=2,
                per_device_train_batch_size=2,
                model_init_kwargs={"torch_dtype": torch.float16},
            )
            trainer = SFTTrainer(
                model=self.model_id,
                args=training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
            )
            assert trainer.model.config.torch_dtype == torch.float16

        # Now test when `torch_dtype` is provided but is wrong
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = SFTConfig(
                output_dir=tmp_dir,
                eval_strategy="steps",
                max_steps=4,
                eval_steps=2,
                save_steps=2,
                per_device_train_batch_size=2,
                model_init_kwargs={"torch_dtype": -1},
            )
            with pytest.raises(
                ValueError,
                match="Invalid `torch_dtype` passed to the SFTConfig. Expected a string with either `torch.dtype` or 'auto', but got -1.",
            ):
                _ = SFTTrainer(
                    model=self.model_id,
                    args=training_args,
                    train_dataset=self.train_dataset,
                    eval_dataset=self.eval_dataset,
                )
