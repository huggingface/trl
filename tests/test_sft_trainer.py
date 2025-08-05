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

import pathlib
import tempfile
import unittest

import numpy as np
import torch
from datasets import Dataset, Image, Sequence, load_dataset
from parameterized import parameterized
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    LlavaForConditionalGeneration,
    is_vision_available,
)
from transformers.testing_utils import require_flash_attn, require_peft, require_vision
from transformers.utils import is_peft_available

from trl import SFTConfig, SFTTrainer
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling


if is_peft_available():
    from peft import LoraConfig, PeftModel, get_peft_model

if is_vision_available():
    from PIL import Image as PILImage


def formatting_prompts_func(example):
    text = f"### Question: {example['question']}\n ### Answer: {example['answer']}"
    return text


def formatting_func_for_pretokenized(example):
    return example["input_ids"]


class TestDataCollatorForLanguageModeling(unittest.TestCase):
    def test_basic_padding(self):
        """Test basic padding functionality without completion masks."""
        self.collator = DataCollatorForLanguageModeling(pad_token_id=0)
        examples = [{"input_ids": [1, 2, 3]}, {"input_ids": [4, 5]}]

        result = self.collator(examples)

        torch.testing.assert_close(result["input_ids"], torch.tensor([[1, 2, 3], [4, 5, 0]]))
        torch.testing.assert_close(result["attention_mask"], torch.tensor([[1, 1, 1], [1, 1, 0]]))
        torch.testing.assert_close(result["position_ids"], torch.tensor([[0, 1, 2], [0, 1, 0]]))
        torch.testing.assert_close(result["labels"], torch.tensor([[1, 2, 3], [4, 5, -100]]))

    def test_completion_mask(self):
        """Test completion mask functionality."""
        self.collator = DataCollatorForLanguageModeling(pad_token_id=0)
        examples = [
            {"input_ids": [1, 2, 3], "completion_mask": [0, 1, 1]},
            {"input_ids": [4, 5], "completion_mask": [0, 1]},
        ]

        result = self.collator(examples)

        torch.testing.assert_close(result["input_ids"], torch.tensor([[1, 2, 3], [4, 5, 0]]))
        torch.testing.assert_close(result["attention_mask"], torch.tensor([[1, 1, 1], [1, 1, 0]]))
        torch.testing.assert_close(result["position_ids"], torch.tensor([[0, 1, 2], [0, 1, 0]]))
        torch.testing.assert_close(result["labels"], torch.tensor([[-100, 2, 3], [-100, 5, -100]]))

    def test_completion_only_loss_disabled(self):
        """Test behavior when completion_only_loss is disabled."""
        collator = DataCollatorForLanguageModeling(pad_token_id=0, completion_only_loss=False)
        examples = [
            {"input_ids": [1, 2, 3], "completion_mask": [0, 1, 1]},
            {"input_ids": [4, 5], "completion_mask": [0, 1]},
        ]

        result = collator(examples)

        # Labels should not be masked when completion_only_loss=False
        torch.testing.assert_close(result["input_ids"], torch.tensor([[1, 2, 3], [4, 5, 0]]))
        torch.testing.assert_close(result["attention_mask"], torch.tensor([[1, 1, 1], [1, 1, 0]]))
        torch.testing.assert_close(result["position_ids"], torch.tensor([[0, 1, 2], [0, 1, 0]]))
        torch.testing.assert_close(result["labels"], torch.tensor([[1, 2, 3], [4, 5, -100]]))

    def test_padding_free_mode(self):
        """Test padding-free mode where sequences are concatenated."""
        collator = DataCollatorForLanguageModeling(pad_token_id=0, padding_free=True)
        examples = [{"input_ids": [1, 2, 3]}, {"input_ids": [4, 5]}]

        result = collator(examples)

        torch.testing.assert_close(result["input_ids"], torch.tensor([[1, 2, 3, 4, 5]]))
        torch.testing.assert_close(result["attention_mask"], torch.tensor([[1, 1, 1, 1, 1]]))
        torch.testing.assert_close(result["position_ids"], torch.tensor([[0, 1, 2, 0, 1]]))
        torch.testing.assert_close(result["labels"], torch.tensor([[1, 2, 3, 4, 5]]))

    def test_padding_free_with_completion_mask(self):
        """Test padding-free mode with completion masks."""
        collator = DataCollatorForLanguageModeling(pad_token_id=0, padding_free=True)
        examples = [
            {"input_ids": [1, 2, 3], "completion_mask": [0, 1, 1]},
            {"input_ids": [4, 5], "completion_mask": [1, 1]},
        ]

        result = collator(examples)

        torch.testing.assert_close(result["input_ids"], torch.tensor([[1, 2, 3, 4, 5]]))
        torch.testing.assert_close(result["attention_mask"], torch.tensor([[1, 1, 1, 1, 1]]))
        torch.testing.assert_close(result["position_ids"], torch.tensor([[0, 1, 2, 0, 1]]))
        torch.testing.assert_close(result["labels"], torch.tensor([[-100, 2, 3, 4, 5]]))

    def test_packing_drops_attention_mask_for_flash_attention(self):
        """Test that when using packing with position_ids, attention_mask is dropped with fa2."""
        collator = DataCollatorForLanguageModeling(pad_token_id=0, padding_free=True, return_position_ids=True)

        # Simulate packed sequences with position_ids that restart (typical of BFD packing)
        examples = [
            {
                "input_ids": [1, 2, 3, 4, 5, 6, 7, 8],  # Packed: [1,2,3] + [4,5] + [6,7,8]
                "seq_lengths": [3, 2, 3],
            }
        ]

        result = collator(examples)

        # Verify that attention_mask is NOT present - this allows FlashAttention to use position_ids
        self.assertNotIn("attention_mask", result, "attention_mask should be dropped for packing with position_ids")

        # Verify essential keys are present
        self.assertIn("input_ids", result)
        self.assertIn("position_ids", result)
        self.assertIn("labels", result)

        # Verify the data is correctly processed
        torch.testing.assert_close(result["input_ids"], torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]))
        torch.testing.assert_close(result["position_ids"], torch.tensor([[0, 1, 2, 0, 1, 0, 1, 2]]))
        torch.testing.assert_close(result["labels"], torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]))

    def test_padding_free_without_position_ids_keeps_attention_mask(self):
        """
        Test that padding_free mode without explicit position_ids still creates attention_mask.
        """
        collator = DataCollatorForLanguageModeling(pad_token_id=0, padding_free=True, return_position_ids=True)

        # Examples without position_ids (not packed)
        examples = [{"input_ids": [1, 2, 3, 4, 5]}]

        result = collator(examples)

        # Should still have attention_mask since no packed position_ids
        self.assertIn("attention_mask", result, "attention_mask should be present when no packed position_ids")
        self.assertIn("position_ids", result)
        self.assertIn("input_ids", result)

        torch.testing.assert_close(result["input_ids"], torch.tensor([[1, 2, 3, 4, 5]]))
        torch.testing.assert_close(result["attention_mask"], torch.tensor([[1, 1, 1, 1, 1]]))
        torch.testing.assert_close(result["position_ids"], torch.tensor([[0, 1, 2, 3, 4]]))

    def test_pad_to_multiple_of(self):
        """Test padding to multiple of specified value."""
        collator = DataCollatorForLanguageModeling(pad_token_id=0, pad_to_multiple_of=4)
        examples = [{"input_ids": [1, 2, 3]}, {"input_ids": [4, 5]}]

        result = collator(examples)

        torch.testing.assert_close(result["input_ids"], torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]]))
        torch.testing.assert_close(result["attention_mask"], torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]))
        torch.testing.assert_close(result["position_ids"], torch.tensor([[0, 1, 2, 0], [0, 1, 0, 0]]))
        torch.testing.assert_close(result["labels"], torch.tensor([[1, 2, 3, -100], [4, 5, -100, -100]]))

    def test_custom_position_ids(self):
        """Test handling of custom position IDs in examples."""
        self.collator = DataCollatorForLanguageModeling(pad_token_id=0)
        examples = [{"input_ids": [1, 2, 3], "seq_lengths": [1, 2]}, {"input_ids": [4, 5], "seq_lengths": [2]}]

        result = self.collator(examples)

        torch.testing.assert_close(result["input_ids"], torch.tensor([[1, 2, 3], [4, 5, 0]]))
        torch.testing.assert_close(result["attention_mask"], torch.tensor([[1, 1, 1], [1, 1, 0]]))
        torch.testing.assert_close(result["position_ids"], torch.tensor([[0, 0, 1], [0, 1, 0]]))
        torch.testing.assert_close(result["labels"], torch.tensor([[1, 2, 3], [4, 5, -100]]))

    def test_single_example(self):
        """Test collator with a single example."""
        self.collator = DataCollatorForLanguageModeling(pad_token_id=0)
        examples = [{"input_ids": [1, 2, 3, 4]}]

        result = self.collator(examples)

        torch.testing.assert_close(result["input_ids"], torch.tensor([[1, 2, 3, 4]]))
        torch.testing.assert_close(result["attention_mask"], torch.tensor([[1, 1, 1, 1]]))
        torch.testing.assert_close(result["position_ids"], torch.tensor([[0, 1, 2, 3]]))
        torch.testing.assert_close(result["labels"], torch.tensor([[1, 2, 3, 4]]))

    def test_different_pad_token_id(self):
        """Test with different pad token ID."""
        collator = DataCollatorForLanguageModeling(pad_token_id=999)
        examples = [{"input_ids": [1, 2, 3]}, {"input_ids": [4, 5]}]

        result = collator(examples)

        torch.testing.assert_close(result["input_ids"], torch.tensor([[1, 2, 3], [4, 5, 999]]))
        torch.testing.assert_close(result["attention_mask"], torch.tensor([[1, 1, 1], [1, 1, 0]]))
        torch.testing.assert_close(result["position_ids"], torch.tensor([[0, 1, 2], [0, 1, 0]]))
        torch.testing.assert_close(result["labels"], torch.tensor([[1, 2, 3], [4, 5, -100]]))

    def test_assistant_masks(self):
        """Test handling of assistant masks in examples."""
        self.collator = DataCollatorForLanguageModeling(pad_token_id=0)
        examples = [
            {"input_ids": [1, 2, 3], "assistant_masks": [0, 1, 1]},
            {"input_ids": [4, 5], "assistant_masks": [0, 1]},
        ]

        result = self.collator(examples)

        torch.testing.assert_close(result["input_ids"], torch.tensor([[1, 2, 3], [4, 5, 0]]))
        torch.testing.assert_close(result["attention_mask"], torch.tensor([[1, 1, 1], [1, 1, 0]]))
        torch.testing.assert_close(result["position_ids"], torch.tensor([[0, 1, 2], [0, 1, 0]]))
        torch.testing.assert_close(result["labels"], torch.tensor([[-100, 2, 3], [-100, 5, -100]]))


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

    def test_uncorrect_data(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Shoud work as SFTTrainer natively supports conversational lm dataset
            training_args = SFTConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                max_length=32,  # make sure there is at least 1 packed sequence
                packing=True,
                report_to="none",
            )
            _ = SFTTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                args=training_args,
                train_dataset=self.conversational_lm_dataset["train"],
            )

            # Same, but without packing
            training_args = SFTConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                packing=False,
                report_to="none",
            )
            _ = SFTTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                args=training_args,
                train_dataset=self.conversational_lm_dataset["train"],
            )

            # Same, but with packing with `max_length`
            training_args = SFTConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                max_length=16,  # make sure there is at least 1 packed sequence
                packing=True,
                report_to="none",
            )
            _ = SFTTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                args=training_args,
                train_dataset=self.standard_prompt_completion_dataset["train"],
            )

            # Same but with prompt-completion dataset
            training_args = SFTConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                packing=False,
                report_to="none",
            )
            _ = SFTTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                args=training_args,
                train_dataset=self.standard_prompt_completion_dataset["train"],
            )

            # Should work as dummy dataset are supported with a formatting function
            training_args = SFTConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                max_length=32,  # make sure there is at least 1 packed sequence
                packing=True,
                report_to="none",
            )
            _ = SFTTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                args=training_args,
                train_dataset=self.dummy_dataset,
                formatting_func=formatting_prompts_func,
            )

    def test_with_model_(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = SFTConfig(
                output_dir=tmp_dir,
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

            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

        # with formatting_func + packed
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = SFTConfig(
                output_dir=tmp_dir,
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

            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = SFTConfig(
                output_dir=tmp_dir,
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

            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

    def test_only_train_packing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = SFTConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                gradient_checkpointing=True,
                packing=True,
                max_length=128,  # make sure there is at least 1 packed sequence
                eval_packing=False,
                report_to="none",
            )

            trainer = SFTTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                args=training_args,
                train_dataset=self.conversational_lm_dataset["train"],
                eval_dataset=self.conversational_lm_dataset["test"],
            )

            self.assertEqual(len(trainer.train_dataset["input_ids"]), 7)  # w/ this dataset, we end up with 46 seqs
            self.assertEqual(len(trainer.eval_dataset["input_ids"]), len(self.conversational_lm_dataset["test"]))

    def test_eval_packing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = SFTConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                max_length=128,  # make sure there is at least 1 packed sequence
                packing=True,
                report_to="none",
            )
            trainer = SFTTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                args=training_args,
                train_dataset=self.conversational_lm_dataset["train"],
                eval_dataset=self.conversational_lm_dataset["test"],
            )

            self.assertEqual(len(trainer.train_dataset["input_ids"]), 7)  # w/ this dataset, we end up with 46 seqs
            self.assertEqual(len(trainer.eval_dataset["input_ids"]), 1)  # w/ this dataset, we end up with 6 seqs

    def test_no_packing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = SFTConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                max_length=128,  # make sure there is at least 1 packed sequence
                packing=False,
                report_to="none",
            )
            trainer = SFTTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                args=training_args,
                train_dataset=self.conversational_lm_dataset["train"],
                eval_dataset=self.conversational_lm_dataset["test"],
            )

            self.assertEqual(len(trainer.train_dataset["input_ids"]), len(self.conversational_lm_dataset["train"]))
            self.assertEqual(len(trainer.eval_dataset["input_ids"]), len(self.conversational_lm_dataset["test"]))

    @require_vision
    def test_skip_prepare_dataset(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = SFTConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                remove_unused_columns=False,
                dataset_kwargs={"skip_prepare_dataset": True},
                report_to="none",
            )

            trainer = SFTTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                args=training_args,
                train_dataset=self.dummy_vsft_instruction_dataset,
            )
            self.assertEqual(trainer.train_dataset.features, self.dummy_vsft_instruction_dataset.features)

    def test_skip_prepare_dataset_with_no_packing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = SFTConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                remove_unused_columns=False,
                packing=False,
                dataset_kwargs={"skip_prepare_dataset": True},
                report_to="none",
            )

            trainer = SFTTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                args=training_args,
                train_dataset=self.dummy_dataset,
            )
            self.assertEqual(trainer.train_dataset.features, self.dummy_dataset.features)

    @require_vision
    def test_llava(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = SFTConfig(
                output_dir=tmp_dir,
                remove_unused_columns=False,
                dataset_kwargs={"skip_prepare_dataset": True},
                report_to="none",
            )
            tiny_llava = LlavaForConditionalGeneration.from_pretrained(
                "trl-internal-testing/tiny-LlavaForConditionalGeneration"
            )
            processor = AutoProcessor.from_pretrained("trl-internal-testing/tiny-LlavaForConditionalGeneration")

            processor.chat_template = """{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}A chat between a curious
user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's
questions. {% for message in messages %}{% if message['role'] == 'user' %}USER: {% else %}ASSISTANT: {% endif %}{% for
item in message['content'] %}{% if item['type'] == 'text' %}{{ item['text'] }}{% elif item['type'] == 'image'
%}<image>{% endif %}{% endfor %}{% if message['role'] == 'user' %} {% else %}{{eos_token}}{% endif %}{% endfor %}{% if
add_generation_prompt %}ASSISTANT: {% endif %}"""

            def collate_fn(examples):
                # Get the texts and images, and apply the chat template
                texts = [processor.apply_chat_template(example["messages"], tokenize=False) for example in examples]
                images = [example["images"][0] for example in examples]

                # Tokenize the texts and process the images
                batch = processor(images=images, text=texts, return_tensors="pt", padding=True)

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
            )

            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])


# This new tester aims to replace the first one at some point
class SFTTrainerTester2(unittest.TestCase):
    @parameterized.expand(
        [
            ("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",),
            ("trl-internal-testing/tiny-Qwen3MoeForCausalLM",),
            ("trl-internal-testing/tiny-GptOssForCausalLM",),
        ]
    )
    def test_train(self, model_id):
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Initialize the trainer
            training_args = SFTConfig(output_dir=tmp_dir, report_to="none")
            trainer = SFTTrainer(model=model_id, args=training_args, train_dataset=dataset)

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

    # Special case for harmony
    def test_train_gpt_oss(self):
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/harmony", "language_modeling", split="train")

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Initialize the trainer
            training_args = SFTConfig(output_dir=tmp_dir, report_to="none")
            trainer = SFTTrainer(
                model="trl-internal-testing/tiny-GptOssForCausalLM", args=training_args, train_dataset=dataset
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

    def test_train_model(self):
        # Instantiate the model
        model = AutoModelForCausalLM.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")

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

            # Check the params have changed
            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                self.assertFalse(torch.allclose(param, new_param), f"Parameter {n} has not changed")

    def test_train_model_torch_dtype(self):
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Initialize the trainer
            training_args = SFTConfig(
                output_dir=tmp_dir, model_init_kwargs={"torch_dtype": torch.float16}, report_to="none"
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
                # Check the torch dtype
                self.assertEqual(new_param.dtype, torch.float16)
                self.assertFalse(torch.allclose(param, new_param), f"Parameter {n} has not changed")

    @require_peft
    def test_train_dense_with_peft_config(self):
        # Get the base model parameter names
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        model = AutoModelForCausalLM.from_pretrained(model_id)
        base_param_names = [f"base_model.model.{n}" for n, _ in model.named_parameters()]

        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Initialize the trainer
            training_args = SFTConfig(output_dir=tmp_dir, report_to="none")

            trainer = SFTTrainer(
                model=model_id,
                args=training_args,
                train_dataset=dataset,
                peft_config=LoraConfig(),
            )

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

    @require_peft
    def test_train_moe_with_peft_config(self):
        # Get the base model parameter names
        model_id = "trl-internal-testing/tiny-GptOssForCausalLM"
        model = AutoModelForCausalLM.from_pretrained(model_id)
        base_param_names = [f"base_model.model.{n}" for n, _ in model.named_parameters()]

        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Initialize the trainer
            training_args = SFTConfig(output_dir=tmp_dir, report_to="none")

            trainer = SFTTrainer(
                model=model_id,
                args=training_args,
                train_dataset=dataset,
                peft_config=LoraConfig(target_parameters=["mlp.experts.down_proj", "mlp.experts.gate_up_proj"]),
            )

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

    @require_peft
    def test_train_dense_with_peft_config_and_gradient_checkpointing(self):
        # Get the base model parameter names
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        model = AutoModelForCausalLM.from_pretrained(model_id)
        base_param_names = [f"base_model.model.{n}" for n, _ in model.named_parameters()]

        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Initialize the trainer
            training_args = SFTConfig(output_dir=tmp_dir, gradient_checkpointing=True, report_to="none")

            trainer = SFTTrainer(
                model=model_id,
                args=training_args,
                train_dataset=dataset,
                peft_config=LoraConfig(),
            )

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

    @require_peft
    def test_train_moe_with_peft_config_and_gradient_checkpointing(self):
        # Get the base model parameter names
        model_id = "trl-internal-testing/tiny-GptOssForCausalLM"
        model = AutoModelForCausalLM.from_pretrained(model_id)
        base_param_names = [f"base_model.model.{n}" for n, _ in model.named_parameters()]

        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Initialize the trainer
            training_args = SFTConfig(output_dir=tmp_dir, gradient_checkpointing=True, report_to="none")

            trainer = SFTTrainer(
                model=model_id,
                args=training_args,
                train_dataset=dataset,
                peft_config=LoraConfig(target_parameters=["mlp.experts.down_proj", "mlp.experts.gate_up_proj"]),
            )

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

    @require_peft
    def test_train_with_peft_model_and_gradient_checkpointing(self):
        # Get the base model parameter names
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        model = AutoModelForCausalLM.from_pretrained(model_id)
        base_param_names = [f"base_model.model.{n}" for n, _ in model.named_parameters()]
        model = get_peft_model(model, LoraConfig())

        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Initialize the trainer
            training_args = SFTConfig(output_dir=tmp_dir, gradient_checkpointing=True, report_to="none")

            trainer = SFTTrainer(model=model, args=training_args, train_dataset=dataset)

            # Verify model is a PeftModel
            self.assertIsInstance(trainer.model, PeftModel)

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

    @parameterized.expand([("bfd",), ("wrapped",)])
    def test_train_packing(self, packing_strategy):
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Initialize the trainer
            training_args = SFTConfig(
                output_dir=tmp_dir, packing=True, packing_strategy=packing_strategy, max_length=10, report_to="none"
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

    def test_train_with_chat_template_kwargs(self):
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Initialize the trainer
            training_args = SFTConfig(output_dir=tmp_dir, report_to="none")

            tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
            # The following template is a simplified version of the Qwen chat template, where an additional argument
            # `role_capital` is used to control the capitalization of roles.
            tokenizer.chat_template = '{%- if messages[0]["role"] == "system" -%}    {{ "<|im_start|>" + ("SYSTEM" if role_capital else "system") + "\\n" + messages[0]["content"] + "<|im_end|>\\n" }}{%- else -%}    {{ "<|im_start|>" + ("SYSTEM" if role_capital else "system") + "\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n" }}{%- endif -%}{%- for message in messages -%}    {%- if (message.role == "user") or (message.role == "system" and not loop.first) or (message.role == "assistant" and not message.tool_calls) -%}        {{ "<|im_start|>" + (message.role.upper() if role_capital else message.role) + "\\n" + message.content + "<|im_end|>\\n" }}    {%- elif message.role == "assistant" -%}        {{ "<|im_start|>" + ("ASSISTANT" if role_capital else "assistant") }}        {%- if message.content -%}            {{ "\\n" + message.content }}        {%- endif -%}        {{ "<|im_end|>\\n" }}    {%- elif message.role == "tool" -%}        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != "tool") -%}            {{ "<|im_start|>" + ("USER" if role_capital else "user") }}        {%- endif -%}        {{ "\\n<tool_response>\\n" + message.content + "\\n</tool_response>" }}        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") -%}            {{ "<|im_end|>\\n" }}        {%- endif -%}    {%- endif -%}{%- endfor -%}{%- if add_generation_prompt -%}    {{ "<|im_start|>" + ("ASSISTANT" if role_capital else "assistant") + "\\n" }}{%- endif -%}'

            dataset.add_column("chat_template_kwargs", [{"role_capital": bool(i % 2)} for i in range(len(dataset))])

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

    def test_train_assistant_only(self):
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "conversational_language_modeling", split="train")

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Initialize the trainer
            training_args = SFTConfig(output_dir=tmp_dir, assistant_only_loss=True, report_to="none")
            trainer = SFTTrainer(
                model="trl-internal-testing/tiny-Qwen3ForCausalLM", args=training_args, train_dataset=dataset
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

    def test_train_completion_only(self):
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_completion", split="train")

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Initialize the trainer
            training_args = SFTConfig(output_dir=tmp_dir, completion_only_loss=True, report_to="none")
            trainer = SFTTrainer(
                model="trl-internal-testing/tiny-Qwen3ForCausalLM", args=training_args, train_dataset=dataset
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

    def test_train_completion_only_harmony(self):
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/harmony", "prompt_completion", split="train")

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Initialize the trainer
            training_args = SFTConfig(output_dir=tmp_dir, completion_only_loss=True, report_to="none")
            trainer = SFTTrainer(
                model="trl-internal-testing/tiny-GptOssForCausalLM", args=training_args, train_dataset=dataset
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

    def test_train_assistant_only_and_completion_only(self):
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_completion", split="train")

        # To test this case, we need to add user messages in the completion (they'll be masked in the loss)
        def add_to_completion(example):
            example["completion"].append(example["prompt"][0])
            example["completion"].append(example["completion"][0])
            return example

        dataset = dataset.map(add_to_completion)

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Initialize the trainer
            training_args = SFTConfig(
                output_dir=tmp_dir, assistant_only_loss=True, completion_only_loss=True, report_to="none"
            )
            trainer = SFTTrainer(
                model="trl-internal-testing/tiny-Qwen3ForCausalLM", args=training_args, train_dataset=dataset
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

    def test_train_assistant_only_iterable_dataset(self):
        # Get the dataset
        dataset = load_dataset(
            "trl-internal-testing/zen", "conversational_language_modeling", split="train", streaming=True
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Initialize the trainer
            training_args = SFTConfig(output_dir=tmp_dir, assistant_only_loss=True, max_steps=3, report_to="none")
            trainer = SFTTrainer(
                model="trl-internal-testing/tiny-Qwen3ForCausalLM", args=training_args, train_dataset=dataset
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

    def test_train_with_set_chat_template_from_model(self):
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "conversational_language_modeling", split="train")

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Initialize the trainer
            training_args = SFTConfig(output_dir=tmp_dir, chat_template_path="Qwen/Qwen3-4B", report_to="none")
            # trl-internal-testing/tiny-GPTNeoXForCausalLM doesn't have a chat template set by default
            trainer = SFTTrainer(
                model="trl-internal-testing/tiny-GPTNeoXForCausalLM", args=training_args, train_dataset=dataset
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

    def test_train_with_set_chat_template_from_path(self):
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "conversational_language_modeling", split="train")

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Initialize the trainer
            training_args = SFTConfig(
                output_dir=tmp_dir,
                chat_template_path=str(pathlib.Path(__file__).parent / "data" / "template.jinja"),
                report_to="none",
            )
            # trl-internal-testing/tiny-GPTNeoXForCausalLM doesn't have a chat template set by default
            trainer = SFTTrainer(
                model="trl-internal-testing/tiny-GPTNeoXForCausalLM", args=training_args, train_dataset=dataset
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

            # Check that the template saved in the output directory is the same as the one used for training
            template_path = pathlib.Path(tmp_dir) / "checkpoint-9" / "chat_template.jinja"
            self.assertTrue(template_path.exists(), f"Chat template not found at {template_path}")

            with open(template_path) as f:
                template_content = f.read()
            with open(training_args.chat_template_path) as f:
                original_template_content = f.read()
            self.assertEqual(
                template_content, original_template_content, "Chat template content does not match the original"
            )

    def test_train_toolcall_data(self):
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/toolcall", split="train")

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

    def test_train_with_eval(self):
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling")

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Initialize the trainer
            training_args = SFTConfig(output_dir=tmp_dir, eval_strategy="steps", eval_steps=3, report_to="none")
            trainer = SFTTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                args=training_args,
                train_dataset=dataset["train"],
                eval_dataset=dataset["test"],
            )

            # Train the model
            trainer.train()

            # Check that the eval loss is not None
            self.assertIsNotNone(trainer.state.log_history[0]["eval_loss"])

    def test_train_with_multiple_eval_dataset(self):
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling")

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Initialize the trainer
            training_args = SFTConfig(output_dir=tmp_dir, eval_strategy="steps", eval_steps=3, report_to="none")
            trainer = SFTTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                args=training_args,
                train_dataset=dataset["train"],
                eval_dataset={"data1": dataset["test"], "data2": dataset["test"]},
            )
            # Train the model
            trainer.train()

            # Check that the eval losses are not None
            self.assertIsNotNone(trainer.state.log_history[-3]["eval_data1_loss"])
            self.assertIsNotNone(trainer.state.log_history[-2]["eval_data2_loss"])

    def test_train_with_gradient_checkpointing(self):
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Initialize the trainer
            training_args = SFTConfig(output_dir=tmp_dir, gradient_checkpointing=True, report_to="none")
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

    def test_tag_added(self):
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")

        # Initialize the trainer
        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            train_dataset=dataset,
        )

        for tag in ["sft", "trl"]:
            self.assertIn(tag, trainer.model.model_tags)

    @require_peft
    def test_tag_added_peft(self):
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")

        # Initialize the trainer
        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            train_dataset=dataset,
            peft_config=LoraConfig(),
        )

        for tag in ["sft", "trl"]:
            self.assertIn(tag, trainer.model.model_tags)

    def test_train_with_torch_dtype(self):
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Initialize the trainer
            training_args = SFTConfig(
                output_dir=tmp_dir, model_init_kwargs={"torch_dtype": torch.float16}, report_to="none"
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
