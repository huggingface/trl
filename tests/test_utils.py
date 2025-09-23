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

import textwrap
from io import StringIO
from unittest.mock import patch

import numpy as np
import torch
from datasets import load_dataset
from parameterized import parameterized
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers.testing_utils import require_peft
from transformers.utils import is_peft_available

from trl import ModelConfig
from trl.trainer import compute_accuracy
from trl.trainer.utils import (
    DataCollatorForChatML,
    RepeatSampler,
    batch_generation,
    decode_and_strip_padding,
    entropy_from_logits,
    flush_left,
    flush_right,
    generate_model_card,
    get_peft_config,
    pad,
    print_prompt_completions_sample,
    selective_log_softmax,
    shuffle_sequence_dict,
    split_pixel_values_by_grid,
    split_tensor_dict,
    truncate_with_protected_tokens,
    unsplit_pixel_values_by_grid,
)

from .testing_utils import TrlTestCase, require_rich


if is_peft_available():
    from peft import LoraConfig


class TestPad(TrlTestCase):
    def test_pad_1_dim_left(self):
        x = torch.tensor([1, 2, 3])
        y = torch.tensor([4, 5])
        output = pad((x, y), padding_value=0, padding_side="left")
        expected = torch.tensor([[1, 2, 3], [0, 4, 5]])
        self.assertTrue(torch.equal(output, expected))

    def test_pad_1_dim_right(self):
        x = torch.tensor([1, 2, 3])
        y = torch.tensor([4, 5])
        output = pad((x, y), padding_value=0, padding_side="right")
        expected = torch.tensor([[1, 2, 3], [4, 5, 0]])
        self.assertTrue(torch.equal(output, expected))

    def test_pad_2_dim_left(self):
        x = torch.tensor([[1, 2], [3, 4]])
        y = torch.tensor([[5, 6]])
        output = pad((x, y), padding_value=0, padding_side="left")
        expected = torch.tensor(
            [
                [[1, 2], [3, 4]],
                [[0, 0], [5, 6]],
            ]
        )
        self.assertTrue(torch.equal(output, expected))

    def test_pad_2_dim_right(self):
        x = torch.tensor([[1, 2], [3, 4]])
        y = torch.tensor([[5, 6]])
        output = pad((x, y), padding_value=0, padding_side="right")
        expected = torch.tensor(
            [
                [[1, 2], [3, 4]],
                [[5, 6], [0, 0]],
            ]
        )
        self.assertTrue(torch.equal(output, expected))

    def test_pad_2_dim_right_multidim(self):
        x = torch.tensor([[1, 2], [3, 4]])
        y = torch.tensor([[5]])
        output = pad((x, y), padding_value=0, padding_side="right")
        expected = torch.tensor(
            [
                [[1, 2], [3, 4]],
                [[5, 0], [0, 0]],
            ]
        )
        self.assertTrue(torch.equal(output, expected))

    def test_pad_to_multiple_of_1(self):
        x = torch.tensor([1, 2, 3])
        y = torch.tensor([4, 5])
        # Max length is 3, pad to multiple of 4
        output = pad((x, y), padding_value=0, padding_side="right", pad_to_multiple_of=4)
        expected = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]])
        self.assertTrue(torch.equal(output, expected))

    def test_pad_to_multiple_of_2(self):
        x = torch.tensor([1, 2, 3, 4, 5])
        y = torch.tensor([6, 7, 8])
        # Max length is 3, pad to multiple of 4
        output = pad((x, y), padding_value=0, padding_side="right", pad_to_multiple_of=4)
        expected = torch.tensor([[1, 2, 3, 4, 5, 0, 0, 0], [6, 7, 8, 0, 0, 0, 0, 0]])
        self.assertTrue(torch.equal(output, expected))

    def test_pad_to_multiple_of_side_left(self):
        x = torch.tensor([1, 2, 3, 4, 5])
        y = torch.tensor([6, 7, 8])
        # Max length is 3, pad to multiple of 4
        output = pad((x, y), padding_value=0, padding_side="left", pad_to_multiple_of=4)
        expected = torch.tensor([[0, 0, 0, 1, 2, 3, 4, 5], [0, 0, 0, 0, 0, 6, 7, 8]])
        self.assertTrue(torch.equal(output, expected))

    def test_pad_to_multiple_of_no_extra_padding(self):
        x = torch.tensor([1, 2, 3, 4])
        y = torch.tensor([5, 6, 7, 8])
        # Already multiple of 4
        output = pad((x, y), padding_value=0, padding_side="left", pad_to_multiple_of=4)
        expected = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
        self.assertTrue(torch.equal(output, expected))


@require_peft
class TestGetPEFTConfig(TrlTestCase):
    def test_create_peft_config_use_peft_false(self):
        """Test that when use_peft is False, the function returns None."""
        model_args = ModelConfig(use_peft=False)
        peft_config = get_peft_config(model_args)
        self.assertIsNone(peft_config)

    def test_create_peft_config_use_peft_true(self):
        """Test that when use_peft is True, the function returns a LoraConfig object."""
        # Provide non-default values to the model config for testing
        peft_kwargs = {
            "lora_r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.1,
            "lora_task_type": "SEQ_CLS",
            "use_rslora": True,
            "lora_target_modules": ["up_proj", "down_proj"],
            "lora_modules_to_save": ["up_proj"],
        }
        model_args = ModelConfig(use_peft=True, **peft_kwargs)
        peft_config = get_peft_config(model_args)
        self.assertTrue(isinstance(peft_config, LoraConfig))
        for arg, value in peft_kwargs.items():
            # Test that lists of modules are converted to sets
            if arg == "lora_target_modules":
                value = set(value)
            # Rename the argument to match the LoraConfig attribute name
            if arg in ["lora_r", "lora_task_type", "lora_target_modules", "lora_modules_to_save"]:
                arg = arg[len("lora_") :] if arg.startswith("lora_") else arg

            self.assertEqual(getattr(peft_config, arg), value)


class TestDecodeAndStripPadding(TrlTestCase):
    def setUp(self):
        super().setUp()
        self.tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")

    def test_example_with_padding(self):
        inputs = self.tokenizer(["Hello world", "Hello"], padding=True, return_tensors="pt")
        decoded = decode_and_strip_padding(inputs["input_ids"], self.tokenizer)
        self.assertEqual(decoded, ["Hello world", "Hello"])

    def test_example_without_padding(self):
        inputs = self.tokenizer(["Hello", "Hello"], padding=False, return_tensors="pt")
        decoded = decode_and_strip_padding(inputs["input_ids"], self.tokenizer)
        self.assertEqual(decoded, ["Hello", "Hello"])


class TestGenerateModelCard(TrlTestCase):
    def test_full(self):
        model_card = generate_model_card(
            base_model="username/my_base_model",
            model_name="my_model",
            hub_model_id="username/my_hub_model",
            dataset_name="username/my_dataset",
            tags=["trl", "trainer-tag"],
            wandb_url="https://wandb.ai/username/project_id/runs/abcd1234",
            comet_url="https://www.comet.com/username/project_id/experiment_id",
            trainer_name="My Trainer",
            trainer_citation="@article{my_trainer, ...}",
            paper_title="My Paper",
            paper_id="1234.56789",
        )
        card_text = str(model_card)
        self.assertIn("[username/my_base_model](https://huggingface.co/username/my_base_model)", card_text)
        self.assertIn("my_model", card_text)
        self.assertIn('pipeline("text-generation", model="username/my_hub_model", device="cuda")', card_text)
        self.assertIn("datasets: username/my_dataset", card_text)
        self.assertIn("](https://wandb.ai/username/project_id/runs/abcd1234)", card_text)
        self.assertIn("](https://www.comet.com/username/project_id/experiment_id", card_text)
        self.assertIn("My Trainer", card_text)
        self.assertIn("```bibtex\n@article{my_trainer, ...}\n```", card_text)
        self.assertIn("[My Paper](https://huggingface.co/papers/1234.56789)", card_text)

    def test_val_none(self):
        model_card = generate_model_card(
            base_model=None,
            model_name="my_model",
            hub_model_id="username/my_hub_model",
            dataset_name=None,
            tags=[],
            wandb_url=None,
            comet_url=None,
            trainer_name="My Trainer",
            trainer_citation=None,
            paper_title=None,
            paper_id=None,
        )
        card_text = str(model_card)
        self.assertIn("my_model", card_text)
        self.assertIn('pipeline("text-generation", model="username/my_hub_model", device="cuda")', card_text)
        self.assertIn("My Trainer", card_text)


class TestDataCollatorForChatML(TrlTestCase):
    def setUp(self):
        super().setUp()
        # Initialize the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Define token IDs
        self.bos_token_id = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else 1
        self.eos_token_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 2
        # Token ID for "true", the last assistant's response in the example:
        self.ignore_index = -100
        self.max_length = 1024
        self.messages_key = "messages"

        # Example input
        dataset = load_dataset("trl-internal-testing/zen", "conversational_language_modeling", split="train")
        self.examples = dataset.to_list()

        # Initialize the data collator
        self.collator = DataCollatorForChatML(
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            ignore_index=self.ignore_index,
        )

    def test_data_collator_for_chatml(self):
        # Process the data
        data = self.collator(self.examples)

        # Verify basic shapes and types
        self.assertIn("input_ids", data)
        self.assertIn("attention_mask", data)
        self.assertIn("labels", data)
        self.assertIn("prompts", data)
        self.assertIn("prompt_attention_mask", data)

        # Decode input_ids and labels for verification
        input_ids = data["input_ids"][0].tolist()
        labels = data["labels"][0].tolist()
        prompt_only = data["prompts"][0].tolist()

        # Get the last assistant's response for comparison
        last_message = self.examples[0][self.messages_key][-1]
        self.assertEqual(last_message["role"], "assistant", "Last message should be from assistant")
        last_assistant_response = last_message["content"]

        # Verify that input_ids contain both prompt and response
        decoded_input = self.tokenizer.decode(input_ids)
        self.assertIn(last_assistant_response, decoded_input, "Input should contain assistant's response")

        # Verify that prompts only contain the conversation up to the last response
        decoded_prompt = self.tokenizer.decode(prompt_only)
        self.assertNotIn(last_assistant_response, decoded_prompt, "Prompt should not contain assistant's response")

        # Verify labels are -100 for non-assistant parts
        prompt_length = len(prompt_only)
        self.assertTrue(
            all(label == self.ignore_index for label in labels[:prompt_length]),
            "Labels should be ignore_index for prompt tokens",
        )

        # Verify labels match assistant response after prompt
        # Add a filter to remove any trailing tokens after the first <|im_end|>
        last_assistant_response_with_end = last_assistant_response + self.tokenizer.eos_token
        last_assistant_response_tokens = self.tokenizer.encode(
            last_assistant_response_with_end, add_special_tokens=False
        )

        response_labels = []
        for label in labels[prompt_length:]:
            if label == self.ignore_index:
                continue
            response_labels.append(label)
            if label == self.tokenizer.convert_tokens_to_ids("<|im_end|>"):
                break
        self.assertEqual(
            response_labels,
            last_assistant_response_tokens,
            "Labels should match assistant response tokens",
        )

        # Verify there isn't a generation prompt at the end
        generation_prompt = "<|im_start|>assistant"
        self.assertFalse(
            decoded_input.strip().endswith(generation_prompt),
            f"Input should not end with generation prompt '{generation_prompt}'",
        )

        self.assertEqual(
            response_labels,
            last_assistant_response_tokens,
            "Labels should match assistant response tokens",
        )


class TestBatchGeneration(TrlTestCase):
    def setUp(self):
        super().setUp()
        # Initialize the tokenizer
        self.model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        self.generation_config = GenerationConfig(
            max_new_tokens=128,
            temperature=0.5,
            do_sample=True,
            top_k=0,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # Example input
        dataset = load_dataset("trl-internal-testing/zen", "conversational_language_modeling", split="train")
        self.examples = dataset["messages"]
        self.mini_batch_size = 3

    def test_mini_batch_generation(self):
        batch = [
            self.tokenizer.apply_chat_template(example[:-1], add_generation_prompt=True, tokenize=False)
            for example in self.examples
        ]
        queries = self.tokenizer(batch, padding=True, return_tensors="pt")["input_ids"].to(self.device)
        bs, context_length = queries.shape

        query_responses, logits = batch_generation(
            self.model, queries, self.mini_batch_size, self.tokenizer.pad_token_id, self.generation_config
        )

        max_length_query = query_responses.shape[1]
        max_length_logits = max_length_query - context_length

        self.assertGreater(max_length_query, context_length)
        self.assertEqual(query_responses.shape, (bs, max_length_query))
        self.assertEqual(logits.shape, (bs, max_length_logits, self.model.config.vocab_size))

    def test_single_batch_generation(self):
        batch = [
            self.tokenizer.apply_chat_template(example[:-1], add_generation_prompt=True, tokenize=False)
            for example in self.examples
        ]
        queries = self.tokenizer(batch, padding=True, return_tensors="pt")["input_ids"].to(self.device)
        bs, context_length = queries.shape

        query_responses, logits = batch_generation(
            self.model, queries, bs, self.tokenizer.pad_token_id, self.generation_config
        )

        max_length_query = query_responses.shape[1]
        max_length_logits = max_length_query - context_length

        self.assertGreater(max_length_query, context_length)
        self.assertEqual(query_responses.shape, (bs, max_length_query))
        self.assertEqual(logits.shape, (bs, max_length_logits, self.model.config.vocab_size))


class TestComputeAccuracy(TrlTestCase):
    def test_token_classification_task(self):
        eval_pred = (
            np.array(
                [
                    [[0.1, 0.9], [0.8, 0.2]],  # Batch 1
                    [[0.3, 0.7], [0.6, 0.4]],  # Batch 2
                ]
            ),
            np.array([[0, 1], [1, 0]]),
        )
        expected_accuracy = 0.5  # 2 matches, 2 mismatches
        result = compute_accuracy(eval_pred)
        self.assertAlmostEqual(result["accuracy"], expected_accuracy)

    def test_token_classification_task_with_ignored_tokens_0(self):
        eval_pred = (
            np.array(
                [
                    [[0.1, 0.9], [0.8, 0.2]],  # Batch 1
                    [[0.3, 0.7], [0.6, 0.4]],  # Batch 2
                ]
            ),
            np.array([[1, 0], [1, -100]]),
        )
        expected_accuracy = 1.0  # All non-ignored tokens match
        result = compute_accuracy(eval_pred)
        self.assertAlmostEqual(result["accuracy"], expected_accuracy)

    def test_token_classification_task_with_ignored_tokens_1(self):
        eval_pred = (
            np.array(
                [
                    [[0.1, 0.9], [0.8, 0.2]],  # Batch 1
                    [[0.3, 0.7], [0.6, 0.4]],  # Batch 2
                ]
            ),
            np.array([[1, 1], [0, -100]]),
        )
        expected_accuracy = 1 / 3  # 1 match, 2 mismatch, 1 ignored
        result = compute_accuracy(eval_pred)
        self.assertAlmostEqual(result["accuracy"], expected_accuracy)

    def test_rewards_comparison_task(self):
        eval_pred = (
            np.array(
                [
                    [0.9, 0.1],  # Batch 1
                    [0.6, 0.4],  # Batch 2
                    [0.5, 0.5],  # Batch 3 (equal)
                ]
            ),
            np.array([0, 1, 1]),
        )
        expected_accuracy = 0.5  # 1 match, 1 mismatch, 1 equal (ignored)

        with self.assertLogs("trl.trainer.utils", level="WARNING") as cm:
            result = compute_accuracy(eval_pred)

        self.assertAlmostEqual(result["accuracy"], expected_accuracy)
        expected_warning = (
            "There are 1 out of 3 instances where the predictions for both options are equal. "
            "These instances are ignored in the accuracy computation."
        )
        self.assertIn(expected_warning, cm.output[0])


class TestFlushLeft(TrlTestCase):
    def test_basic_case(self):
        mask = torch.tensor([[0, 0, 1, 1, 1], [0, 1, 1, 0, 0]])
        tensor1 = torch.tensor([[0, 0, 2, 3, 4], [0, 5, 6, 0, 0]])
        tensor2 = torch.tensor([[0, 0, 7, 8, 9], [0, 10, 11, 0, 0]])
        new_mask, new_tensor1, new_tensor2 = flush_left(mask, tensor1, tensor2)

        expected_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
        expected_tensor1 = torch.tensor([[2, 3, 4], [5, 6, 0]])
        expected_tensor2 = torch.tensor([[7, 8, 9], [10, 11, 0]])

        self.assertTrue(torch.equal(new_mask, expected_mask))
        self.assertTrue(torch.equal(new_tensor1, expected_tensor1))
        self.assertTrue(torch.equal(new_tensor2, expected_tensor2))

    def test_single_row(self):
        mask = torch.tensor([[0, 0, 1, 1]])
        tensor1 = torch.tensor([[0, 0, 2, 3]])
        new_mask, new_tensor1 = flush_left(mask, tensor1)

        expected_mask = torch.tensor([[1, 1]])
        expected_tensor1 = torch.tensor([[2, 3]])

        self.assertTrue(torch.equal(new_mask, expected_mask))
        self.assertTrue(torch.equal(new_tensor1, expected_tensor1))

    def test_no_shift_needed(self):
        mask = torch.tensor([[1, 1, 0, 0], [1, 0, 0, 0]])
        tensor1 = torch.tensor([[5, 6, 0, 0], [7, 0, 0, 0]])
        new_mask, new_tensor1 = flush_left(mask, tensor1)

        expected_mask = torch.tensor([[1, 1], [1, 0]])
        expected_tensor1 = torch.tensor([[5, 6], [7, 0]])

        self.assertTrue(torch.equal(new_mask, expected_mask))
        self.assertTrue(torch.equal(new_tensor1, expected_tensor1))

    def test_no_tensors(self):
        mask = torch.tensor([[0, 0, 1, 1, 1], [0, 1, 1, 0, 0]])
        new_mask = flush_left(mask)
        expected_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
        self.assertTrue(torch.equal(new_mask, expected_mask))


class TestFlushRight(TrlTestCase):
    def test_basic_case(self):
        mask = torch.tensor([[1, 1, 1, 0, 0], [0, 0, 1, 1, 0]])
        tensor1 = torch.tensor([[2, 3, 4, 0, 0], [0, 0, 5, 6, 0]])
        tensor2 = torch.tensor([[7, 8, 9, 0, 0], [0, 0, 10, 11, 0]])
        new_mask, new_tensor1, new_tensor2 = flush_right(mask, tensor1, tensor2)

        expected_mask = torch.tensor([[1, 1, 1], [0, 1, 1]])
        expected_tensor1 = torch.tensor([[2, 3, 4], [0, 5, 6]])
        expected_tensor2 = torch.tensor([[7, 8, 9], [0, 10, 11]])

        self.assertTrue(torch.equal(new_mask, expected_mask))
        self.assertTrue(torch.equal(new_tensor1, expected_tensor1))
        self.assertTrue(torch.equal(new_tensor2, expected_tensor2))

    def test_single_row(self):
        mask = torch.tensor([[1, 1, 0, 0]])
        tensor1 = torch.tensor([[2, 3, 0, 0]])
        new_mask, new_tensor1 = flush_right(mask, tensor1)

        expected_mask = torch.tensor([[1, 1]])
        expected_tensor1 = torch.tensor([[2, 3]])

        self.assertTrue(torch.equal(new_mask, expected_mask))
        self.assertTrue(torch.equal(new_tensor1, expected_tensor1))

    def test_no_shift_needed(self):
        mask = torch.tensor([[0, 0, 1, 1], [0, 0, 0, 1]])
        tensor1 = torch.tensor([[0, 0, 5, 6], [0, 0, 0, 7]])
        new_mask, new_tensor1 = flush_right(mask, tensor1)

        expected_mask = torch.tensor([[1, 1], [0, 1]])
        expected_tensor1 = torch.tensor([[5, 6], [0, 7]])

        self.assertTrue(torch.equal(new_mask, expected_mask))
        self.assertTrue(torch.equal(new_tensor1, expected_tensor1))

    def test_no_tensors(self):
        mask = torch.tensor([[1, 1, 1, 0, 0], [0, 0, 1, 1, 0]])
        new_mask = flush_right(mask)
        expected_mask = torch.tensor([[1, 1, 1], [0, 1, 1]])
        self.assertTrue(torch.equal(new_mask, expected_mask))


class RepeatRandomSamplerTester(TrlTestCase):
    def test_sampler(self):
        dataset = ["a", "b", "c", "d", "e", "f", "g"]
        sampler = RepeatSampler(dataset, mini_repeat_count=2)
        # Should output something like [4, 4, 3, 3, 0, 0, 1, 1, 2, 2, 6, 6, 5, 5]
        sampled = list(sampler)
        # Check that the length is doubled
        assert len(sampled) == 2 * len(dataset)
        # Check that all indexes are present
        assert set(sampled) == set(range(len(dataset)))
        # Check that each element is repeated twice
        assert all(sampled[i] == sampled[i + 1] for i in range(0, len(sampled), 2))

    def test_sampler_no_shuffle(self):
        dataset = ["a", "b", "c", "d", "e", "f", "g"]
        sampler = RepeatSampler(dataset, mini_repeat_count=2, shuffle=False)
        sampled = list(sampler)
        expected = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]
        self.assertEqual(sampled, expected)

    def test_sampler_no_repeat(self):
        dataset = ["a", "b", "c", "d", "e", "f", "g"]
        sampler = RepeatSampler(dataset, mini_repeat_count=1)
        # Should output something like [4, 3, 0, 1, 2, 6, 5]
        sampled = list(sampler)
        # Check that the length is the same
        assert len(sampled) == len(dataset)
        # Check that all indexes are present
        assert set(sampled) == set(range(len(dataset)))

    def test_sampler_with_batch_size(self):
        dataset = ["a", "b", "c", "d", "e", "f", "g", "h"]
        sampler = RepeatSampler(dataset, mini_repeat_count=1, batch_size=2, repeat_count=2)
        # Should output something like [4, 3, 4, 3, 0, 1, 0, 1, 2, 6, 2, 6, 5, 7, 5, 7]
        sampled = list(sampler)
        # Check that the length is doubled
        assert len(sampled) == 2 * len(dataset)
        # Check that all indexes are present
        assert set(sampled) == set(range(len(dataset)))
        # Check that each element is repeated as expected
        assert all(sampled[i : i + 1] == sampled[i + 2 : i + 3] for i in range(0, len(sampled), 4))

    def test_sampler_with_batch_size_and_drop(self):
        dataset = ["a", "b", "c", "d", "e", "f", "g"]
        sampler = RepeatSampler(dataset, mini_repeat_count=1, batch_size=2, repeat_count=2)
        # Should output something like [4, 3, 4, 3, 0, 1, 0, 1, 2, 6, 2, 6]
        sampled = list(sampler)
        # Check that the length is doubled
        assert len(sampled) == 2 * (
            len(dataset) - 1
        )  # one element is dropped, because it's not enough to form a batch
        assert len(sampler) == len(sampled)  # the length should be the same as the sampled length
        # Check that the sampled indexes are a subset of the dataset indexes
        assert set(sampled).issubset(set(range(len(dataset))))
        # Check that each element is repeated as expected
        assert all(sampled[i : i + 1] == sampled[i + 2 : i + 3] for i in range(0, len(sampled), 4))

    def test_sampler_with_mini_repeat_count_and_batch_size_1(self):
        dataset = ["a", "b", "c", "d", "e", "f", "g"]
        sampler = RepeatSampler(dataset, mini_repeat_count=2, batch_size=3, repeat_count=2)
        # Should output something like [4, 4, 3, 3, 0, 0, 4, 4, 3, 3, 0, 0,
        #                               1, 1, 2, 2, 6, 6, 1, 1, 2, 2, 6, 6]
        sampled = list(sampler)
        # Check that the length is quadrupled
        assert len(sampled) == 4 * (len(dataset) - 1)  # 1 element is dropped, because it's not enough to form a batch
        assert len(sampler) == len(sampled)  # the length should be the same as the sampled length
        # Check that the sampled indexes are a subset of the dataset indexes
        assert set(sampled).issubset(set(range(len(dataset))))
        # Check that each element is repeated as expected
        assert all(sampled[i] == sampled[i + 1] for i in range(0, len(sampled), 2))
        # Check that the batch is repeated as expected
        assert sampled[0:6] == sampled[6:12]
        assert sampled[12:18] == sampled[18:24]

    def test_sampler_with_mini_repeat_count_and_batch_size_2(self):
        dataset = ["a", "b", "c", "d", "e", "f", "g"]
        sampler = RepeatSampler(dataset, mini_repeat_count=3, batch_size=2, repeat_count=2)
        # Should output something like [4, 4, 4, 3, 3, 3, 4, 4, 4, 3, 3, 3,
        #                               0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1,
        #                               2, 2, 2, 6, 6, 6, 2, 2, 2, 6, 6, 6]
        sampled = list(sampler)
        # Check that the length is sextupled
        assert len(sampled) == 6 * (len(dataset) - 1)  # 1 element is dropped, because it's not enough to form a batch
        assert len(sampler) == len(sampled)  # the length should be the same as the sampled length
        # Check that the sampled indexes are a subset of the dataset indexes
        assert set(sampled).issubset(set(range(len(dataset))))
        # Check that each element is repeated as expected
        assert all(sampled[i] == sampled[i + 1] == sampled[i + 2] for i in range(0, len(sampled), 3))
        # Check that the batch is repeated as expected
        assert sampled[0:6] == sampled[6:12]
        assert sampled[12:18] == sampled[18:24]
        assert sampled[24:30] == sampled[30:36]

    def test_sampler_with_mini_repeat_count_and_batch_size_3(self):
        dataset = ["a", "b", "c", "d", "e", "f", "g"]
        sampler = RepeatSampler(dataset, mini_repeat_count=2, batch_size=2, repeat_count=3)
        # Should output something like [4, 4, 3, 3, 4, 4, 3, 3, 4, 4, 3, 3,
        #                               0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1,
        #                               2, 2, 6, 6, 2, 2, 6, 6, 2, 2, 6, 6]
        sampled = list(sampler)
        # Check that the length is sextupled
        assert len(sampled) == 6 * (len(dataset) - 1)  # 1 element is dropped, because it's not enough to form a batch
        # Check that the sampled indexes are a subset of the dataset indexes
        assert set(sampled).issubset(set(range(len(dataset))))
        # Check that each element is repeated as expected
        assert all(sampled[i] == sampled[i + 1] for i in range(0, len(sampled), 2))
        # Check that the batch is repeated as expected
        assert sampled[0:4] == sampled[4:8] == sampled[8:12]
        assert sampled[12:16] == sampled[16:20] == sampled[20:24]
        assert sampled[24:28] == sampled[28:32] == sampled[32:36]


class TestEntropyFromLogits(TrlTestCase):
    @parameterized.expand(
        [
            (dtype, chunk_size, shape)
            for dtype in (torch.float64, torch.float32, torch.float16, torch.bfloat16)
            for chunk_size in (1, 16)
            for shape in [(768,), (32, 768), (8, 16, 768), (2, 4, 8, 768)]
        ]
    )
    def test_entropy_from_logits_2_dims(self, dtype, chunk_size, shape):
        logits = torch.randn(*shape, dtype=dtype)
        if dtype in (torch.float64, torch.float32):
            p = logits.softmax(-1)
            entropy = -torch.sum(p * p.log(), dim=-1)
        else:
            logps = logits.log_softmax(dim=-1)
            entropy = -(torch.exp(logps) * logps).sum(-1)
        predicted_entropy = entropy_from_logits(logits, chunk_size=chunk_size)
        torch.testing.assert_close(predicted_entropy, entropy, rtol=1e-5, atol=1e-5)


@require_rich
class TestPrintPromptCompletionsSample(TrlTestCase):
    @patch("sys.stdout", new_callable=StringIO)
    def test_print_output(self, mock_stdout):
        prompts = ["The sky is", "The sun is"]
        completions = [" blue.", " in the sky."]
        rewards = {"Correctness": [0.123, 0.456], "Format": [0.789, 0.101]}
        advantages = [0.987, 0.654]
        step = 42

        print_prompt_completions_sample(prompts, completions, rewards, advantages, step)

        output = mock_stdout.getvalue()

        # docstyle-ignore
        expected_output = textwrap.dedent("""\
        ╭──────────────────────────── Step 42 ─────────────────────────────╮
        │ ┏━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━┓ │
        │ ┃ Prompt     ┃ Completion   ┃ Correctness ┃ Format ┃ Advantage ┃ │
        │ ┡━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━┩ │
        │ │ The sky is │  blue.       │        0.12 │   0.79 │      0.99 │ │
        │ ├────────────┼──────────────┼─────────────┼────────┼───────────┤ │
        │ │ The sun is │  in the sky. │        0.46 │   0.10 │      0.65 │ │
        │ └────────────┴──────────────┴─────────────┴────────┴───────────┘ │
        ╰──────────────────────────────────────────────────────────────────╯
        """)

        self.assertEqual(output, expected_output)

    @patch("sys.stdout", new_callable=StringIO)
    def test_num_samples(self, mock_stdout):
        prompts = ["A", "B"]
        completions = ["1", "2"]
        rewards = {"Score": [0.1, 0.2]}
        advantages = [0.3, 0.4]
        step = 10

        print_prompt_completions_sample(prompts, completions, rewards, advantages, step, num_samples=1)
        output = mock_stdout.getvalue()

        # docstyle-ignore
        possible_outputs = [
            textwrap.dedent("""\
            ╭────────────────── Step 10 ──────────────────╮
            │ ┏━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━┓ │
            │ ┃ Prompt ┃ Completion ┃ Score ┃ Advantage ┃ │
            │ ┡━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━┩ │
            │ │ A      │ 1          │  0.10 │      0.30 │ │
            │ └────────┴────────────┴───────┴───────────┘ │
            ╰─────────────────────────────────────────────╯
                """),
            # docstyle-ignore
            textwrap.dedent("""\
            ╭────────────────── Step 10 ──────────────────╮
            │ ┏━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━┓ │
            │ ┃ Prompt ┃ Completion ┃ Score ┃ Advantage ┃ │
            │ ┡━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━┩ │
            │ │ B      │ 2          │  0.20 │      0.40 │ │
            │ └────────┴────────────┴───────┴───────────┘ │
            ╰─────────────────────────────────────────────╯
                """),
        ]
        self.assertIn(output, possible_outputs)


class TestSelectiveLogSoftmax(TrlTestCase):
    @parameterized.expand([(torch.float64,), (torch.float32,), (torch.float16,), (torch.bfloat16,)])
    def test_selective_log_softmax(self, dtype):
        """Test selective_log_softmax with logits of different dtypes"""
        vocab_size = 1024
        batch_size = 4
        seq_len = 32

        input_ids = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len))
        logits = torch.randn(batch_size, seq_len, vocab_size, dtype=dtype)

        expected_output = torch.gather(logits.log_softmax(-1), dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)
        actual_output = selective_log_softmax(logits, input_ids)

        if dtype in [torch.float16, torch.bfloat16]:
            # half-precision dtypes fall back to an exact method
            self.assertTrue(torch.equal(actual_output, expected_output))
        else:
            torch.testing.assert_close(actual_output, expected_output, rtol=1e-5, atol=1e-5)


class ShuffleSequenceDictTester(TrlTestCase):
    def test_shuffle_preserves_shape(self):
        x = torch.arange(6).reshape(3, 2)
        y = torch.arange(3).reshape(3, 1)
        tensor_dict = {"x": x.clone(), "y": y.clone()}

        shuffled = shuffle_sequence_dict(tensor_dict)

        self.assertEqual(shuffled["x"].shape, x.shape)
        self.assertEqual(shuffled["y"].shape, y.shape)

    def test_shuffle_consistent_across_tensors(self):
        # Use known patterns to check alignment
        x = torch.tensor([[10, 11], [20, 21], [30, 31]])
        y = torch.tensor([[1], [2], [3]])
        tensor_dict = {"x": x.clone(), "y": y.clone()}

        shuffled = shuffle_sequence_dict(tensor_dict)

        # Build a reverse map from shuffled x rows to y values
        for i in range(3):
            x_row = shuffled["x"][i]
            y_val = shuffled["y"][i].item()

            if torch.equal(x_row, torch.tensor([10, 11])):
                self.assertEqual(y_val, 1)
            elif torch.equal(x_row, torch.tensor([20, 21])):
                self.assertEqual(y_val, 2)
            elif torch.equal(x_row, torch.tensor([30, 31])):
                self.assertEqual(y_val, 3)
            else:
                self.fail("Unexpected x row in shuffled output.")

    def test_none_tensor_remains_none(self):
        x = torch.arange(6).reshape(3, 2)
        tensor_dict = {"x": x.clone(), "y": None}

        shuffled = shuffle_sequence_dict(tensor_dict)

        self.assertIsNone(shuffled["y"])
        self.assertEqual(shuffled["x"].shape, x.shape)

    def test_shuffle_with_list(self):
        x = torch.tensor([[10, 11], [20, 21], [30, 31]])
        y = ["a", "b", "c"]

        sequence_dict = {"x": x.clone(), "y": y}

        shuffled = shuffle_sequence_dict(sequence_dict)

        # Check that the list y is shuffled in the same order as x
        for i in range(3):
            x_row = shuffled["x"][i]
            y_val = shuffled["y"][i]

            if torch.equal(x_row, torch.tensor([10, 11])):
                self.assertEqual(y_val, "a")
            elif torch.equal(x_row, torch.tensor([20, 21])):
                self.assertEqual(y_val, "b")
            elif torch.equal(x_row, torch.tensor([30, 31])):
                self.assertEqual(y_val, "c")
            else:
                self.fail("Unexpected x row in shuffled output.")


class SplitTensorDictTester(TrlTestCase):
    def test_split_equal_chunks(self):
        x = torch.arange(12).reshape(6, 2)
        y = torch.arange(6).reshape(6, 1)
        tensor_dict = {"x": x, "y": y}

        result = split_tensor_dict(tensor_dict, 3)

        expected_x_chunks = torch.chunk(x, 3, dim=0)
        expected_y_chunks = torch.chunk(y, 3, dim=0)
        self.assertEqual(len(result), 3)
        for i in range(3):
            self.assertTrue(torch.equal(result[i]["x"], expected_x_chunks[i]))
            self.assertTrue(torch.equal(result[i]["y"], expected_y_chunks[i]))

    def test_with_none_tensor(self):
        x = torch.arange(12).reshape(6, 2)
        tensor_dict = {"x": x, "y": None}

        result = split_tensor_dict(tensor_dict, 2)

        expected_x_chunks = torch.chunk(x, 2, dim=0)
        self.assertEqual(len(result), 2)
        for i in range(2):
            self.assertTrue(torch.equal(result[i]["x"], expected_x_chunks[i]))
            self.assertIsNone(result[i]["y"])

    def test_with_scalar(self):
        x = torch.arange(12).reshape(6, 2)
        tensor_dict = {"x": x, "y": torch.tensor(1)}

        result = split_tensor_dict(tensor_dict, 2)

        expected_x_chunks = torch.chunk(x, 2, dim=0)
        self.assertEqual(len(result), 2)
        for i in range(2):
            self.assertTrue(torch.equal(result[i]["x"], expected_x_chunks[i]))
            self.assertTrue(torch.equal(result[i]["y"], torch.tensor(1)))


class SplitPixelValuesByGridTester(TrlTestCase):
    def test_split_correctly_0(self):
        batch = {
            "image_grid_thw": torch.tensor([[1, 2, 2], [1, 2, 2]]),
            "num_images": [1, 1],
            "pixel_values": torch.arange(8 * 3).reshape(8, 3),  # Shape: [8, 3]
        }
        result = split_pixel_values_by_grid(batch)
        self.assertIsInstance(result["pixel_values"], list)
        self.assertEqual(len(result["pixel_values"]), 2)
        self.assertTrue(torch.equal(result["pixel_values"][0], batch["pixel_values"][:4]))
        self.assertTrue(torch.equal(result["pixel_values"][1], batch["pixel_values"][4:]))
        self.assertIsInstance(result["image_grid_thw"], list)
        self.assertEqual(len(result["image_grid_thw"]), 2)
        self.assertTrue(torch.equal(result["image_grid_thw"][0], torch.tensor([[1, 2, 2]])))
        self.assertTrue(torch.equal(result["image_grid_thw"][1], torch.tensor([[1, 2, 2]])))

    def test_split_correctly_1(self):
        batch = {
            "image_grid_thw": torch.tensor([[1, 2, 2], [1, 2, 4]]),
            "num_images": [1, 1],
            "pixel_values": torch.arange(12 * 3).reshape(12, 3),  # Shape: [12, 3]
        }
        result = split_pixel_values_by_grid(batch)
        self.assertIsInstance(result["pixel_values"], list)
        self.assertEqual(len(result["pixel_values"]), 2)
        self.assertTrue(torch.equal(result["pixel_values"][0], batch["pixel_values"][:4]))
        self.assertTrue(torch.equal(result["pixel_values"][1], batch["pixel_values"][4:12]))
        self.assertIsInstance(result["image_grid_thw"], list)
        self.assertEqual(len(result["image_grid_thw"]), 2)
        self.assertTrue(torch.equal(result["image_grid_thw"][0], torch.tensor([[1, 2, 2]])))
        self.assertTrue(torch.equal(result["image_grid_thw"][1], torch.tensor([[1, 2, 4]])))

    def test_missing_keys(self):
        batch = {"pixel_values": torch.tensor([1.0])}
        result = split_pixel_values_by_grid(batch)
        self.assertEqual(result, batch)

    def test_mismatched_length(self):
        batch = {
            "image_grid_thw": torch.tensor([[1, 1, 2], [1, 2, 1]]),  # Total = 8
            "num_images": [1, 1],
            "pixel_values": torch.randn(3, 5),  # Only 3 rows
        }
        with self.assertRaises(ValueError):
            split_pixel_values_by_grid(batch)

    def test_multi_images(self):
        batch = {
            "image_grid_thw": torch.tensor([[1, 1, 2], [1, 2, 2], [1, 2, 1]]),  # Total = 8
            "num_images": [1, 2],
            "pixel_values": torch.arange(8 * 3).reshape(8, 3),  # Shape: [8, 3]
        }
        result = split_pixel_values_by_grid(batch)
        self.assertIsInstance(result["pixel_values"], list)
        self.assertEqual(len(result["pixel_values"]), 2)
        self.assertTrue(torch.equal(result["pixel_values"][0], batch["pixel_values"][:2]))
        self.assertTrue(torch.equal(result["pixel_values"][1], batch["pixel_values"][2:]))
        self.assertIsInstance(result["image_grid_thw"], list)
        self.assertEqual(len(result["image_grid_thw"]), 2)
        self.assertTrue(torch.equal(result["image_grid_thw"][0], torch.tensor([[1, 1, 2]])))
        self.assertTrue(torch.equal(result["image_grid_thw"][1], torch.tensor([[1, 2, 2], [1, 2, 1]])))


class TruncateWithProtectedTokensTester(TrlTestCase):
    def test_basic_example(self):
        """Test the basic example from the problem description."""
        prompt_ids = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        prompt_mask = torch.ones_like(prompt_ids)
        protected_tokens = [2, 3, 6]
        target_length = 3

        new_ids, new_mask = truncate_with_protected_tokens(prompt_ids, prompt_mask, target_length, protected_tokens)

        expected_ids = torch.tensor([[2, 3, 5], [6, 9, 10]])
        expected_mask = torch.ones_like(expected_ids)

        self.assertTrue(torch.equal(new_ids, expected_ids))
        self.assertTrue(torch.equal(new_mask, expected_mask))

    def test_no_truncation_needed(self):
        """Test when target length equals current length."""
        prompt_ids = torch.tensor([[1, 2, 3]])
        prompt_mask = torch.ones_like(prompt_ids)
        protected_tokens = [2]
        target_length = 3

        new_ids, new_mask = truncate_with_protected_tokens(prompt_ids, prompt_mask, target_length, protected_tokens)

        self.assertTrue(torch.equal(new_ids, prompt_ids))
        self.assertTrue(torch.equal(new_mask, prompt_mask))

    def test_no_protected_tokens(self):
        """Test truncation with no protected tokens (normal right truncation)."""
        prompt_ids = torch.tensor([[1, 2, 3, 4, 5]])
        prompt_mask = torch.ones_like(prompt_ids)
        protected_tokens = []
        target_length = 3

        new_ids, new_mask = truncate_with_protected_tokens(prompt_ids, prompt_mask, target_length, protected_tokens)

        expected_ids = torch.tensor([[3, 4, 5]])  # Last 3 tokens
        self.assertTrue(torch.equal(new_ids, expected_ids))

    def test_all_tokens_protected(self):
        """Test when all remaining tokens are protected."""
        prompt_ids = torch.tensor([[1, 2, 3, 4, 5]])
        prompt_mask = torch.ones_like(prompt_ids)
        protected_tokens = [3, 4, 5]
        target_length = 3

        new_ids, new_mask = truncate_with_protected_tokens(prompt_ids, prompt_mask, target_length, protected_tokens)

        expected_ids = torch.tensor([[3, 4, 5]])
        self.assertTrue(torch.equal(new_ids, expected_ids))

    def test_too_many_protected_tokens(self):
        """Test error when too many protected tokens for target length."""
        prompt_ids = torch.tensor([[1, 2, 3, 4, 5]])
        prompt_mask = torch.ones_like(prompt_ids)
        protected_tokens = [1, 2, 3, 4]
        target_length = 3

        with self.assertRaises(ValueError):
            truncate_with_protected_tokens(prompt_ids, prompt_mask, target_length, protected_tokens)

    def test_single_batch_single_token(self):
        """Test edge case with single batch and single token."""
        prompt_ids = torch.tensor([[5]])
        prompt_mask = torch.ones_like(prompt_ids)
        protected_tokens = [5]
        target_length = 1

        new_ids, new_mask = truncate_with_protected_tokens(prompt_ids, prompt_mask, target_length, protected_tokens)

        self.assertTrue(torch.equal(new_ids, prompt_ids))

    def test_mask_preservation(self):
        """Test that mask values are correctly preserved."""
        prompt_ids = torch.tensor([[1, 2, 3, 4, 5]])
        prompt_mask = torch.tensor([[1, 0, 1, 0, 1]])  # Mixed mask values
        protected_tokens = [2, 4]
        target_length = 3

        new_ids, new_mask = truncate_with_protected_tokens(prompt_ids, prompt_mask, target_length, protected_tokens)

        expected_ids = torch.tensor([[2, 4, 5]])
        expected_mask = torch.tensor([[0, 0, 1]])  # Corresponding mask values

        self.assertTrue(torch.equal(new_ids, expected_ids))
        self.assertTrue(torch.equal(new_mask, expected_mask))

    def test_multiple_batches_different_protected(self):
        """Test multiple batches where protected tokens appear differently."""
        prompt_ids = torch.tensor([[1, 2, 3, 4, 5], [2, 6, 7, 8, 9], [10, 11, 12, 2, 13]])
        prompt_mask = torch.ones_like(prompt_ids)
        protected_tokens = [2]
        target_length = 3

        new_ids, new_mask = truncate_with_protected_tokens(prompt_ids, prompt_mask, target_length, protected_tokens)

        expected_ids = torch.tensor(
            [
                [2, 4, 5],  # 2 is protected, keep last 2 non-protected (4,5)
                [2, 8, 9],  # 2 is protected, keep last 2 non-protected (8,9)
                [12, 2, 13],  # 2 is protected, keep last 2 non-protected (12,13)
            ]
        )

        self.assertTrue(torch.equal(new_ids, expected_ids))

    def test_order_preservation(self):
        """Test that relative order is preserved."""
        prompt_ids = torch.tensor([[10, 2, 20, 3, 30, 40]])
        prompt_mask = torch.ones_like(prompt_ids)
        protected_tokens = [2, 3]
        target_length = 4

        new_ids, new_mask = truncate_with_protected_tokens(prompt_ids, prompt_mask, target_length, protected_tokens)

        # Should keep protected tokens 2,3 and last 2 non-protected tokens 30,40
        # Order should be: 2, 3, 30, 40 (maintaining original relative positions)
        expected_ids = torch.tensor([[2, 3, 30, 40]])

        self.assertTrue(torch.equal(new_ids, expected_ids))

    def test_empty_protected_tokens_list(self):
        """Test with empty protected tokens list."""
        prompt_ids = torch.tensor([[1, 2, 3, 4, 5]])
        prompt_mask = torch.ones_like(prompt_ids)
        protected_tokens = []
        target_length = 2

        new_ids, new_mask = truncate_with_protected_tokens(prompt_ids, prompt_mask, target_length, protected_tokens)

        expected_ids = torch.tensor([[4, 5]])  # Last 2 tokens
        self.assertTrue(torch.equal(new_ids, expected_ids))


class UnsplitPixelValuesByGridTester(TrlTestCase):
    def test_unsplit_correctly(self):
        pixel_values = [torch.randn(4, 5), torch.randn(2, 5)]
        pixel_values_merged = torch.cat(pixel_values, dim=0)
        image_grid_thw = [torch.tensor([[1, 2, 2]]), torch.tensor([[1, 2, 1]])]
        image_grid_thw_merged = torch.cat(image_grid_thw, dim=0)
        batch = {"pixel_values": pixel_values, "image_grid_thw": image_grid_thw, "other_key": torch.tensor([1])}
        result = unsplit_pixel_values_by_grid(batch)
        self.assertIsInstance(result["pixel_values"], torch.Tensor)
        self.assertTrue(torch.allclose(result["pixel_values"], pixel_values_merged))
        self.assertIsInstance(result["image_grid_thw"], torch.Tensor)
        self.assertTrue(torch.equal(result["image_grid_thw"], image_grid_thw_merged))
        self.assertIn("other_key", result)

    def test_no_op_if_not_list(self):
        original = torch.randn(5, 3)
        batch = {"pixel_values": original}
        result = unsplit_pixel_values_by_grid(batch)
        self.assertTrue(torch.equal(result["pixel_values"], original))
