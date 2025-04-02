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

import copy
import os
import tempfile
import unittest

import numpy as np
import torch
from datasets import Dataset, Image, Sequence, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    LlavaForConditionalGeneration,
    TrainingArguments,
    is_vision_available,
)
from transformers.testing_utils import require_flash_attn, require_peft, require_vision
from transformers.utils import is_peft_available

from trl import SFTConfig, SFTTrainer
from trl.trainer import ConstantLengthDataset, DataCollatorForCompletionOnlyLM
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling


def formatting_prompts_func(example):
    text = f"### Question: {example['question']}\n ### Answer: {example['answer']}"
    return text


def formatting_func_for_pretokenized(example):
    return example["input_ids"]


def formatting_prompts_func_batched(example):
    output_text = []
    for i, question in enumerate(example["question"]):
        text = f"### Question: {question}\n ### Answer: {example['answer'][i]}"
        output_text.append(text)
    return output_text


if is_peft_available():
    from peft import LoraConfig, PeftModel, get_peft_model

if is_vision_available():
    from PIL import Image as PILImage


class TestDataCollatorForLanguageModeling(unittest.TestCase):
    def test_collate_padding(self):
        collator = DataCollatorForLanguageModeling(pad_token_id=0)
        examples = [{"input_ids": [1, 2, 3]}, {"input_ids": [4, 5]}]
        output = collator(examples)

        expected_input_ids = torch.tensor([[1, 2, 3], [4, 5, 0]])
        expected_attention_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
        expected_labels = torch.tensor([[1, 2, 3], [4, 5, -100]])

        self.assertEqual(output["input_ids"].tolist(), expected_input_ids.tolist())
        self.assertEqual(output["attention_mask"].tolist(), expected_attention_mask.tolist())
        self.assertEqual(output["labels"].tolist(), expected_labels.tolist())

    def test_collate_no_padding(self):
        collator = DataCollatorForLanguageModeling(pad_token_id=0)
        examples = [{"input_ids": [1, 2, 3]}, {"input_ids": [4, 5, 6]}]
        output = collator(examples)

        expected_input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
        expected_attention_mask = torch.tensor([[1, 1, 1], [1, 1, 1]])
        expected_labels = torch.tensor([[1, 2, 3], [4, 5, 6]])

        self.assertEqual(output["input_ids"].tolist(), expected_input_ids.tolist())
        self.assertEqual(output["attention_mask"].tolist(), expected_attention_mask.tolist())
        self.assertEqual(output["labels"].tolist(), expected_labels.tolist())


class SFTTrainerTester(unittest.TestCase):
    r""" """

    def setUp(self):
        self.model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.dummy_dataset = Dataset.from_dict(
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
        self.dummy_tokenized_dataset = Dataset.from_dict(
            {
                "input_ids": [
                    self.tokenizer.encode(
                        "TRL is a library to post-train LLMs and diffusion models with methods such as Supervised Fine-tuning (SFT), Proximal Policy Optimization (PPO), and Direct Preference Optimization (DPO)."
                    )
                ]
                * 10
            }
        )

        self.conversational_lm_dataset = load_dataset("trl-internal-testing/zen", "conversational_language_modeling")
        self.standard_prompt_completion_dataset = load_dataset(
            "trl-internal-testing/zen", "standard_prompt_completion"
        )

        if is_vision_available():
            self.dummy_vsft_instruction_dataset = Dataset.from_dict(
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
            self.dummy_vsft_instruction_dataset.cast_column("images", Sequence(Image()))
            self.dummy_vsft_instruction_dataset = self.dummy_vsft_instruction_dataset.cast_column(
                "images", Sequence(Image())
            )

        self.train_dataset = ConstantLengthDataset(
            self.tokenizer,
            self.dummy_dataset,
            formatting_func=formatting_prompts_func,
            seq_length=16,
            num_of_sequences=16,
        )

        self.eval_dataset = ConstantLengthDataset(
            self.tokenizer,
            self.dummy_dataset,
            formatting_func=formatting_prompts_func,
            seq_length=16,
            num_of_sequences=16,
        )

        self.train_dataset_from_pretokenized = ConstantLengthDataset(
            self.tokenizer,
            self.dummy_tokenized_dataset,
            seq_length=16,
            num_of_sequences=16,
            formatting_func=formatting_func_for_pretokenized,
        )

        self.eval_dataset_from_pretokenized = ConstantLengthDataset(
            self.tokenizer,
            self.dummy_tokenized_dataset,
            seq_length=16,
            num_of_sequences=16,
            formatting_func=formatting_func_for_pretokenized,
        )

    def test_constant_length_dataset_with_pretokenized_data(self):
        constant_len_dataset = ConstantLengthDataset(
            self.tokenizer,
            self.dummy_tokenized_dataset,
            formatting_func=formatting_func_for_pretokenized,
        )

        assert len(constant_len_dataset) == len(self.dummy_tokenized_dataset)
        assert len(constant_len_dataset) > 0

        for example in constant_len_dataset:
            assert "input_ids" in example
            assert "labels" in example

            assert len(example["input_ids"]) == constant_len_dataset.seq_length
            assert len(example["labels"]) == constant_len_dataset.seq_length

            decoded_text = self.tokenizer.decode(example["input_ids"])
            assert ("TRL" in decoded_text) and ("(DPO)" in decoded_text)

    def test_constant_length_dataset(self):
        formatted_dataset = ConstantLengthDataset(
            self.tokenizer,
            self.dummy_dataset,
            formatting_func=formatting_prompts_func,
        )

        self.assertEqual(len(formatted_dataset), len(self.dummy_dataset))
        self.assertGreater(len(formatted_dataset), 0)

        for example in formatted_dataset:
            self.assertIn("input_ids", example)
            self.assertIn("labels", example)

            self.assertEqual(len(example["input_ids"]), formatted_dataset.seq_length)
            self.assertEqual(len(example["labels"]), formatted_dataset.seq_length)

            decoded_text = self.tokenizer.decode(example["input_ids"])
            self.assertTrue(("Question" in decoded_text) and ("Answer" in decoded_text))

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
                report_to="none",
            )

            trainer = SFTTrainer(
                model=self.model_id,
                args=training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                formatting_func=formatting_prompts_func,
            )

            self.assertEqual(trainer.args.hub_token, training_args.hub_token)

            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[(-1)]["train_loss"])
            self.assertIsNotNone(trainer.state.log_history[0]["eval_loss"])

            self.assertIn("model.safetensors", os.listdir(tmp_dir + "/checkpoint-2"))

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
                report_to="none",
            )

            trainer = SFTTrainer(
                model=self.model_id,
                args=training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
            )

            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[(-1)]["train_loss"])
            self.assertIsNotNone(trainer.state.log_history[0]["eval_loss"])

            self.assertIn("model.safetensors", os.listdir(tmp_dir + "/checkpoint-2"))

    def test_sft_trainer_with_pretokenized_data_packing(self):
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
                report_to="none",
            )

            trainer = SFTTrainer(
                model=self.model_id,
                args=training_args,
                train_dataset=self.train_dataset_from_pretokenized,
                eval_dataset=self.eval_dataset_from_pretokenized,
            )

            trainer.train()

            assert trainer.state.log_history[(-1)]["train_loss"] is not None
            assert trainer.state.log_history[0]["eval_loss"] is not None

            assert "model.safetensors" in os.listdir(tmp_dir + "/checkpoint-2")

    def test_sft_trainer_uncorrect_data(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Shoud work as SFTTrainer natively supports conversational lm dataset
            training_args = SFTConfig(
                output_dir=tmp_dir,
                dataloader_drop_last=True,
                max_steps=2,
                eval_steps=1,
                save_steps=1,
                per_device_train_batch_size=2,
                max_length=32,  # make sure there is at least 1 packed sequence
                packing=True,
                report_to="none",
            )
            _ = SFTTrainer(
                model=self.model,
                args=training_args,
                train_dataset=self.conversational_lm_dataset["train"],
            )

            # Same, but without packing
            training_args = SFTConfig(
                output_dir=tmp_dir,
                dataloader_drop_last=True,
                max_steps=2,
                eval_steps=1,
                save_steps=1,
                per_device_train_batch_size=2,
                packing=False,
                report_to="none",
            )
            _ = SFTTrainer(
                model=self.model,
                args=training_args,
                train_dataset=self.conversational_lm_dataset["train"],
            )

            # Same, but with packing with `max_length`
            training_args = SFTConfig(
                output_dir=tmp_dir,
                dataloader_drop_last=True,
                max_steps=2,
                eval_steps=1,
                save_steps=1,
                per_device_train_batch_size=2,
                max_length=16,  # make sure there is at least 1 packed sequence
                packing=True,
                report_to="none",
            )
            _ = SFTTrainer(
                model=self.model,
                args=training_args,
                train_dataset=self.standard_prompt_completion_dataset["train"],
            )

            # Same but with prompt completion dataset
            training_args = SFTConfig(
                output_dir=tmp_dir,
                dataloader_drop_last=True,
                max_steps=2,
                eval_steps=1,
                save_steps=1,
                per_device_train_batch_size=2,
                packing=False,
                report_to="none",
            )
            _ = SFTTrainer(
                model=self.model,
                args=training_args,
                train_dataset=self.standard_prompt_completion_dataset["train"],
            )

            # Should work as dummy dataset are supported with a formatting function
            training_args = SFTConfig(
                output_dir=tmp_dir,
                dataloader_drop_last=True,
                max_steps=2,
                eval_steps=1,
                save_steps=1,
                per_device_train_batch_size=2,
                max_length=32,  # make sure there is at least 1 packed sequence
                packing=True,
                report_to="none",
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
                max_steps=2,
                eval_steps=1,
                save_steps=1,
                per_device_train_batch_size=2,
                packing=False,
                report_to="none",
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
                report_to="none",
            )
            trainer = SFTTrainer(
                model=self.model,
                args=training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
            )

            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[(-1)]["train_loss"])
            self.assertIsNotNone(trainer.state.log_history[0]["eval_loss"])

            self.assertIn("model.safetensors", os.listdir(tmp_dir + "/checkpoint-2"))

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = SFTConfig(
                output_dir=tmp_dir,
                dataloader_drop_last=True,
                max_steps=2,
                save_steps=1,
                num_train_epochs=2,
                per_device_train_batch_size=2,
                max_length=16,
                packing=True,
                report_to="none",
            )
            trainer = SFTTrainer(
                model=self.model,
                args=training_args,
                train_dataset=self.dummy_dataset,
            )

            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[(-1)]["train_loss"])

            self.assertIn("model.safetensors", os.listdir(tmp_dir + "/checkpoint-2"))

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = SFTConfig(
                output_dir=tmp_dir,
                dataloader_drop_last=True,
                max_steps=2,
                save_steps=1,
                num_train_epochs=2,
                per_device_train_batch_size=2,
                max_length=16,
                report_to="none",
            )
            trainer = SFTTrainer(
                model=self.model,
                args=training_args,
                train_dataset=self.dummy_dataset,
            )

            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[(-1)]["train_loss"])

            self.assertIn("model.safetensors", os.listdir(tmp_dir + "/checkpoint-1"))

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
                report_to="none",
            )
            trainer = SFTTrainer(
                model=self.model,
                args=training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
            )

            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[(-1)]["train_loss"])
            self.assertIsNotNone(trainer.state.log_history[0]["eval_loss"])

            self.assertIn("model.safetensors", os.listdir(tmp_dir + "/checkpoint-2"))

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = SFTConfig(
                output_dir=tmp_dir,
                dataloader_drop_last=True,
                max_steps=2,
                save_steps=1,
                per_device_train_batch_size=2,
                max_length=16,
                packing=True,
                report_to="none",
            )
            trainer = SFTTrainer(
                model=self.model,
                args=training_args,
                train_dataset=self.dummy_dataset,
            )

            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[(-1)]["train_loss"])

            self.assertIn("model.safetensors", os.listdir(tmp_dir + "/checkpoint-2"))

        # with formatting_func + packed
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = SFTConfig(
                output_dir=tmp_dir,
                dataloader_drop_last=True,
                max_steps=2,
                save_steps=1,
                per_device_train_batch_size=2,
                max_length=16,
                packing=True,
                report_to="none",
            )
            trainer = SFTTrainer(
                model=self.model,
                args=training_args,
                train_dataset=self.dummy_dataset,
                formatting_func=formatting_prompts_func,
            )

            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[(-1)]["train_loss"])

            self.assertIn("model.safetensors", os.listdir(tmp_dir + "/checkpoint-2"))

        # with formatting_func + packed
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = SFTConfig(
                output_dir=tmp_dir,
                dataloader_drop_last=True,
                max_steps=2,
                save_steps=1,
                per_device_train_batch_size=2,
                max_length=16,
                report_to="none",
            )
            trainer = SFTTrainer(
                model=self.model,
                args=training_args,
                train_dataset=self.dummy_dataset,
                formatting_func=formatting_prompts_func_batched,
            )

            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[(-1)]["train_loss"])

            self.assertIn("model.safetensors", os.listdir(tmp_dir + "/checkpoint-2"))

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = SFTConfig(
                output_dir=tmp_dir,
                dataloader_drop_last=True,
                max_steps=2,
                save_steps=1,
                per_device_train_batch_size=2,
                max_length=16,
                report_to="none",
            )
            trainer = SFTTrainer(
                model=self.model,
                args=training_args,
                train_dataset=self.dummy_dataset,
            )

            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[(-1)]["train_loss"])

            self.assertIn("model.safetensors", os.listdir(tmp_dir + "/checkpoint-1"))

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
                report_to="none",
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

            self.assertIsNotNone(trainer.state.log_history[(-1)]["train_loss"])
            self.assertIsNotNone(trainer.state.log_history[0]["eval_data1_loss"])
            self.assertIsNotNone(trainer.state.log_history[1]["eval_data2_loss"])

            self.assertIn("model.safetensors", os.listdir(tmp_dir + "/checkpoint-1"))

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
        self.assertEqual(result_text, "I have not been masked correctly.")

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
            self.assertEqual(result_text, "I have not been masked correctly.")

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
        self.assertEqual(result_text, " I should not be masked. I should not be masked too.")

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
        self.assertEqual(result_text1, " I should not be masked.")

        non_masked_tokens2 = input_ids[1][labels[1] != -100]
        result_text2 = tokenizer.decode(non_masked_tokens2)
        self.assertEqual(result_text2, " I should not be masked. I should not be masked too.")

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
                max_length=500,
                report_to="none",
            )
            trainer = SFTTrainer(
                model=self.model,
                args=training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
            )

            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[(-1)]["train_loss"])
            self.assertIsNotNone(trainer.state.log_history[0]["eval_loss"])

            # make sure the trainer did 5 steps
            self.assertIn("model.safetensors", os.listdir(tmp_dir + "/checkpoint-5"))

    def test_sft_trainer_infinite_with_model_epochs(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = SFTConfig(
                output_dir=tmp_dir,
                dataloader_drop_last=True,
                num_train_epochs=1,
                per_device_train_batch_size=2,
                save_strategy="epoch",
                packing=True,
                max_length=500,
                report_to="none",
            )
            trainer = SFTTrainer(
                model=self.model,
                args=training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
            )

            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[(-1)]["train_loss"])

            # make sure the trainer did 5 steps
            self.assertIn("model.safetensors", os.listdir(tmp_dir + "/checkpoint-4"))

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
                report_to="none",
            )
            trainer = SFTTrainer(
                model=self.model,
                args=training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
            )

            trainer.model = trainer._activate_neftune(trainer.model)

            device = trainer.model.get_input_embeddings().weight.device
            trainer.model.train()

            torch.random.manual_seed(42)
            embeds_neftune = trainer.model.get_input_embeddings()(torch.LongTensor([[1, 0, 1]]).to(device))

            torch.random.manual_seed(24)
            embeds_neftune_2 = trainer.model.get_input_embeddings()(torch.LongTensor([[1, 0, 1]]).to(device))

            self.assertFalse(torch.allclose(embeds_neftune, embeds_neftune_2))
            self.assertGreater(len(trainer.model.get_input_embeddings()._forward_hooks), 0)

            trainer.neftune_hook_handle.remove()

            trainer.train()

            # Make sure forward pass works fine
            _ = trainer.model(torch.LongTensor([[1, 0, 1]]).to(device))
            self.assertEqual(len(trainer.model.get_input_embeddings()._forward_hooks), 0)

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

            training_args = SFTConfig(
                packing=True,
                output_dir=tmp_dir,
                report_to="none",
            )

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
                report_to="none",
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

            self.assertTrue(isinstance(trainer.model, PeftModel))

            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[(-1)]["train_loss"])
            self.assertIsNotNone(trainer.state.log_history[0]["eval_loss"])

            self.assertIn("adapter_model.safetensors", os.listdir(tmp_dir + "/checkpoint-2"))
            self.assertIn("adapter_config.json", os.listdir(tmp_dir + "/checkpoint-2"))
            self.assertNotIn("model.safetensors", os.listdir(tmp_dir + "/checkpoint-2"))

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
                report_to="none",
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

            self.assertIsInstance(trainer.model, PeftModel)

            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[(-1)]["train_loss"])
            self.assertIsNotNone(trainer.state.log_history[0]["eval_loss"])

            self.assertIn("adapter_model.safetensors", os.listdir(tmp_dir + "/checkpoint-2"))
            self.assertIn("adapter_config.json", os.listdir(tmp_dir + "/checkpoint-2"))
            self.assertNotIn("model.safetensors", os.listdir(tmp_dir + "/checkpoint-2"))

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
                report_to="none",
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

            trainer.model = trainer._activate_neftune(trainer.model)

            self.assertIsInstance(trainer.model, PeftModel)

            device = trainer.model.get_input_embeddings().weight.device
            trainer.model.train()

            torch.random.manual_seed(42)
            embeds_neftune = trainer.model.get_input_embeddings()(torch.LongTensor([[1, 0, 1]]).to(device))

            torch.random.manual_seed(24)
            embeds_neftune_2 = trainer.model.get_input_embeddings()(torch.LongTensor([[1, 0, 1]]).to(device))

            self.assertFalse(torch.allclose(embeds_neftune, embeds_neftune_2))
            self.assertGreater(len(trainer.model.get_input_embeddings()._forward_hooks), 0)

            trainer.neftune_hook_handle.remove()

            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[(-1)]["train_loss"])
            self.assertIsNotNone(trainer.state.log_history[0]["eval_loss"])

            self.assertIn("adapter_model.safetensors", os.listdir(tmp_dir + "/checkpoint-2"))
            self.assertIn("adapter_config.json", os.listdir(tmp_dir + "/checkpoint-2"))
            self.assertNotIn("model.safetensors", os.listdir(tmp_dir + "/checkpoint-2"))

            # Make sure forward pass works fine to check if embeddings forward is not broken.
            _ = trainer.model(torch.LongTensor([[1, 0, 1]]).to(device))
            self.assertEqual(len(trainer.model.get_input_embeddings()._forward_hooks), 0)

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
                report_to="none",
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

            for tag in ["sft", "trl"]:
                self.assertIn(tag, trainer.model.model_tags)

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
                report_to="none",
            )

            trainer = SFTTrainer(
                model=self.model_id,
                args=training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
            )

            for tag in ["sft", "trl"]:
                self.assertIn(tag, trainer.model.model_tags)

    def test_sft_trainer_only_train_packing(self):
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
                max_length=16,  # make sure there is at least 1 packed sequence
                eval_packing=False,
                report_to="none",
            )

            trainer = SFTTrainer(
                model=self.model_id,
                args=training_args,
                train_dataset=self.conversational_lm_dataset["train"],
                eval_dataset=self.conversational_lm_dataset["test"],
            )

            self.assertEqual(len(trainer.train_dataset["input_ids"]), 47)  # w/ this dataset, we end up with 46 seqs
            self.assertEqual(len(trainer.eval_dataset["input_ids"]), len(self.conversational_lm_dataset["test"]))

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
                max_length=16,  # make sure there is at least 1 packed sequence
                packing=True,
                report_to="none",
            )
            trainer = SFTTrainer(
                model=self.model_id,
                args=training_args,
                train_dataset=self.conversational_lm_dataset["train"],
                eval_dataset=self.conversational_lm_dataset["test"],
            )

            self.assertEqual(len(trainer.train_dataset["input_ids"]), 47)  # w/ this dataset, we end up with 47 seqs
            self.assertEqual(len(trainer.eval_dataset["input_ids"]), 7)  # w/ this dataset, we end up with 7 seqs

    def test_sft_trainer_no_packing(self):
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
                max_length=16,  # make sure there is at least 1 packed sequence
                packing=False,
                report_to="none",
            )
            trainer = SFTTrainer(
                model=self.model_id,
                args=training_args,
                train_dataset=self.conversational_lm_dataset["train"],
                eval_dataset=self.conversational_lm_dataset["test"],
            )

            self.assertEqual(len(trainer.train_dataset["input_ids"]), len(self.conversational_lm_dataset["train"]))
            self.assertEqual(len(trainer.eval_dataset["input_ids"]), len(self.conversational_lm_dataset["test"]))

    @require_vision
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
                dataset_kwargs={"skip_prepare_dataset": True},
                report_to="none",
            )

            trainer = SFTTrainer(
                model=self.model_id,
                args=training_args,
                train_dataset=self.dummy_vsft_instruction_dataset,
                eval_dataset=self.dummy_vsft_instruction_dataset,
            )
            self.assertEqual(trainer.train_dataset.features, self.dummy_vsft_instruction_dataset.features)
            self.assertEqual(trainer.eval_dataset.features, self.dummy_vsft_instruction_dataset.features)

    def test_sft_trainer_skip_prepare_dataset_with_no_packing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = SFTConfig(
                output_dir=tmp_dir,
                dataloader_drop_last=True,
                max_steps=4,
                eval_steps=2,
                save_steps=2,
                per_device_train_batch_size=2,
                gradient_checkpointing=True,
                remove_unused_columns=False,
                packing=False,
                dataset_kwargs={"skip_prepare_dataset": True},
                report_to="none",
            )

            trainer = SFTTrainer(
                model=self.model_id,
                args=training_args,
                train_dataset=self.dummy_dataset,
            )
            self.assertEqual(trainer.train_dataset.features, self.dummy_dataset.features)

    @require_vision
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
                dataset_kwargs={"skip_prepare_dataset": True},
                report_to="none",
            )
            tiny_llava = LlavaForConditionalGeneration.from_pretrained(
                "trl-internal-testing/tiny-LlavaForConditionalGeneration"
            )
            processor = AutoProcessor.from_pretrained("trl-internal-testing/tiny-LlavaForConditionalGeneration")

            processor.chat_template = """{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. {% for message in messages %}{% if message['role'] == 'user' %}USER: {% else %}ASSISTANT: {% endif %}{% for item in message['content'] %}{% if item['type'] == 'text' %}{{ item['text'] }}{% elif item['type'] == 'image' %}<image>{% endif %}{% endfor %}{% if message['role'] == 'user' %} {% else %}{{eos_token}}{% endif %}{% endfor %}{% if add_generation_prompt %}ASSISTANT: {% endif %}"""

            def collate_fn(examples):
                # Get the texts and images, and apply the chat template
                texts = [processor.apply_chat_template(example["messages"], tokenize=False) for example in examples]
                images = [example["images"][0] for example in examples]

                # Tokenize the texts and process the images
                batch = processor(texts, images, return_tensors="pt", padding=True)

                # The labels are the input_ids, and we mask the padding tokens in the loss computation
                labels = batch["input_ids"].clone()
                labels[labels == processor.tokenizer.pad_token_id] = -100
                batch["labels"] = labels

                return batch

            trainer = SFTTrainer(
                model=tiny_llava,
                args=training_args,
                data_collator=collate_fn,
                train_dataset=self.dummy_vsft_instruction_dataset,
                eval_dataset=self.dummy_vsft_instruction_dataset,
            )

            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[(-1)]["train_loss"])
            self.assertIsNotNone(trainer.state.log_history[0]["eval_loss"])

            self.assertIn("model.safetensors", os.listdir(tmp_dir + "/checkpoint-2"))

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
                report_to="none",
            )
            trainer = SFTTrainer(
                model=self.model_id,
                args=training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                formatting_func=formatting_prompts_func,
            )
            self.assertEqual(trainer.model.config.torch_dtype, torch.float16)

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
                report_to="none",
            )
            with self.assertRaises(ValueError) as context:
                _ = SFTTrainer(
                    model=self.model_id,
                    args=training_args,
                    train_dataset=self.train_dataset,
                    eval_dataset=self.eval_dataset,
                )

            self.assertIn(
                "Invalid `torch_dtype` passed to `SFTConfig`. Expected either 'auto' or a string representing "
                "a `torch.dtype` (e.g., 'float32'), but got -1.",
                str(context.exception),
            )


# This new tester aims to replace the first one at some point
class SFTTrainerTester2(unittest.TestCase):
    def test_train(self):
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Initialize the trainer
            training_args = SFTConfig(output_dir=tmp_dir, report_to="none")
            trainer = SFTTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", args=training_args, train_dataset=dataset
            )

            # Save the initial parameters to compare them later
            previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

            # Train the model
            trainer.train()

            # Check that the training loss is not None
            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

            # Check the params have changed
            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                self.assertFalse(torch.allclose(param, new_param), f"Parameter {n} has not changed")

    @require_peft
    def test_train_peft_model(self):
        # Get the base model
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        model = AutoModelForCausalLM.from_pretrained(model_id)

        # Get the base model parameter names
        base_param_names = [f"base_model.model.{n}" for n, _ in model.named_parameters()]

        # Turn the model into a peft model
        lora_config = LoraConfig()
        model = get_peft_model(model, lora_config)

        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Initialize the trainer
            training_args = SFTConfig(output_dir=tmp_dir, report_to="none")
            trainer = SFTTrainer(model=model, args=training_args, train_dataset=dataset)

            # Save the initial parameters to compare them later
            previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

            # Train the model
            trainer.train()

            # Check that the training loss is not None
            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

            # Check the peft params have changed and the base model params have not changed
            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                if n in base_param_names:  # We expect the base model parameters to be the same
                    self.assertTrue(torch.allclose(param, new_param), f"Parameter {n} has changed")
                elif (
                    "base_layer" not in n
                ):  # We expect the peft parameters to be different (except for the base layer)
                    self.assertFalse(torch.allclose(param, new_param), f"Parameter {n} has not changed")

    def test_train_with_non_chatml_conversational_data(self):
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "conversational_language_modeling", split="train")

        # Rename role/content to from/value to ensure SFT works with non-chatML conversational data
        def rename_fields(example: list[dict]):
            return {"conversations": [{"from": m["role"], "value": m["content"]} for m in example["messages"]]}

        dataset = dataset.map(rename_fields, remove_columns="messages")

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Initialize the trainer
            training_args = SFTConfig(output_dir=tmp_dir, report_to="none")
            trainer = SFTTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", args=training_args, train_dataset=dataset
            )

            # Save the initial parameters to compare them later
            previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

            # Train the model
            trainer.train()

            # Check that the training loss is not None
            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

            # Check the params have changed
            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                self.assertFalse(torch.allclose(param, new_param), f"Parameter {n} has not changed")

    def test_train_with_pretokenized_data(self):
        # Get the dataset
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")

        def tokenize_example(example):
            return tokenizer(example["text"])

        # Apply tokenization
        tokenized_dataset = dataset.map(tokenize_example, remove_columns=["text"])

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Initialize the trainer
            training_args = SFTConfig(output_dir=tmp_dir, report_to="none")
            trainer = SFTTrainer(model=model_id, args=training_args, train_dataset=tokenized_dataset)

            # Save the initial parameters to compare them later
            previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

            # Train the model
            trainer.train()

            # Check that the training loss is not None
            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

            # Check the params have changed
            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                self.assertFalse(torch.allclose(param, new_param), f"Parameter {n} has not changed")

    def test_train_with_iterable_dataset(self):
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train", streaming=True)

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Initialize the trainer
            training_args = SFTConfig(output_dir=tmp_dir, max_steps=3, report_to="none")
            trainer = SFTTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", args=training_args, train_dataset=dataset
            )

            # Save the initial parameters to compare them later
            previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

            # Train the model
            trainer.train()

            # Check that the training loss is not None
            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

            # Check the params have changed
            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                self.assertFalse(torch.allclose(param, new_param), f"Parameter {n} has not changed")

    def test_train_with_data_collator_for_completion_only_and_padding_free(self):
        # Get the dataset
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_completion", split="train")

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        response_template = "<|im_start|>assistant\n"
        collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer, padding_free=True)

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Initialize the trainer
            training_args = SFTConfig(output_dir=tmp_dir, report_to="none")
            trainer = SFTTrainer(model=model_id, args=training_args, train_dataset=dataset, data_collator=collator)

            # Save the initial parameters to compare them later
            previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

            # Train the model
            trainer.train()

            # Check that the training loss is not None
            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

            # Check the params have changed
            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                self.assertFalse(torch.allclose(param, new_param), f"Parameter {n} has not changed")

    @require_flash_attn
    def test_train_padding_free(self):
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Initialize the trainer
            training_args = SFTConfig(
                output_dir=tmp_dir,
                padding_free=True,
                model_init_kwargs={"attn_implementation": "flash_attention_2"},
                bf16=True,  # flash_attention_2 only supports bf16 and fp16
                report_to="none",
            )
            trainer = SFTTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", args=training_args, train_dataset=dataset
            )

            # Save the initial parameters to compare them later
            previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

            # Train the model
            trainer.train()

            # Check that the training loss is not None
            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

            # Check the params have changed
            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                self.assertFalse(torch.allclose(param, new_param), f"Parameter {n} has not changed")
