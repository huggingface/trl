# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import unittest

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers.testing_utils import require_peft
from transformers.utils import is_peft_available

from trl.trainer.model_config import ModelConfig
from trl.trainer.utils import (
    DataCollatorForChatML,
    batch_generation,
    decode_and_strip_padding,
    generate_model_card,
    get_peft_config,
    pad,
)


if is_peft_available():
    from peft import LoraConfig


class TestPad(unittest.TestCase):
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


@require_peft
class TestGetPEFTConfig(unittest.TestCase):
    def test_create_peft_config_use_peft_false(self):
        """Test that when use_peft is False, the function returns None."""
        model_config = ModelConfig(use_peft=False)
        peft_config = get_peft_config(model_config)
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
        model_config = ModelConfig(use_peft=True, **peft_kwargs)
        peft_config = get_peft_config(model_config)
        self.assertTrue(isinstance(peft_config, LoraConfig))
        for arg, value in peft_kwargs.items():
            # Test that lists of modules are converted to sets
            if arg == "lora_target_modules":
                value = set(value)
            # Rename the argument to match the LoraConfig attribute name
            if arg in ["lora_r", "lora_task_type", "lora_target_modules", "lora_modules_to_save"]:
                arg = arg[len("lora_") :] if arg.startswith("lora_") else arg

            self.assertEqual(getattr(peft_config, arg), value)


class TestDecodeAndStripPadding(unittest.TestCase):
    def setUp(self):
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

    def test_example_with_padding(self):
        inputs = self.tokenizer(["Hello world", "Hello"], padding=True, return_tensors="pt")
        decoded = decode_and_strip_padding(inputs["input_ids"], self.tokenizer)
        self.assertEqual(decoded, ["Hello world", "Hello"])

    def test_example_without_padding(self):
        inputs = self.tokenizer(["Hello", "Hello"], padding=False, return_tensors="pt")
        decoded = decode_and_strip_padding(inputs["input_ids"], self.tokenizer)
        self.assertEqual(decoded, ["Hello", "Hello"])


class TestGenerateModelCard(unittest.TestCase):
    def test_full(self):
        model_card = generate_model_card(
            base_model="username/my_base_model",
            model_name="my_model",
            hub_model_id="username/my_hub_model",
            dataset_name="username/my_dataset",
            tags=["trl", "trainer-tag"],
            wandb_url="https://wandb.ai/username/project_id/runs/abcd1234",
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
            trainer_name="My Trainer",
            trainer_citation=None,
            paper_title=None,
            paper_id=None,
        )
        card_text = str(model_card)
        self.assertIn("my_model", card_text)
        self.assertIn('pipeline("text-generation", model="username/my_hub_model", device="cuda")', card_text)
        self.assertIn("My Trainer", card_text)


class TestDataCollatorForChatML(unittest.TestCase):
    def setUp(self):
        # Initialize the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-Instruct-hf")
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

        # Decode input_ids and labels for verification
        input_ids = data["input_ids"][0].tolist()
        labels = data["labels"][0].tolist()
        prompt_only = data["prompts"][0].tolist()

        # Verify that input_ids start with optional padding tokens  and a single BOS token and there are no extra ones
        first_non_pad = next(token for token in input_ids if token != self.tokenizer.pad_token_id)
        self.assertEqual(
            first_non_pad, self.bos_token_id, "The first non-padding token of input_ids should be BOS token."
        )
        self.assertEqual(input_ids.count(self.bos_token_id), 1, "There should be exactly one BOS token in input_ids.")

        # Verify that the assistant's response token is present in input_ids and not in the prompt_only
        last_assistant_response = self.examples[0][self.messages_key][-1]["content"]
        last_assistant_response_tokens = self.tokenizer.encode(last_assistant_response, add_special_tokens=False)
        response_in_input_ids = all(token in input_ids for token in last_assistant_response_tokens)
        self.assertTrue(response_in_input_ids, "The assistant's response should be present in input_ids.")

        # Check if the last assistant's response tokens are not in prompt_only
        response_in_prompt = all(token in prompt_only for token in last_assistant_response_tokens)
        self.assertFalse(response_in_prompt, "The assistant's response should not be present in prompt_only.")

        # Verify that EOS token is at the end of input_ids
        self.assertEqual(input_ids[-1], self.eos_token_id, "The last token of input_ids should be EOS token.")

        # Verify that the labels preserved the target string (last_assistant_response)
        last_assistant_response = self.examples[0][self.messages_key][-1]["content"]
        last_assistant_response_tokens = self.tokenizer.encode(last_assistant_response, add_special_tokens=False)

        # Find the start and end of the last assistant's response in the labels
        response_start = next(i for i, label in enumerate(labels) if label != self.ignore_index)
        response_end = next(i for i in range(len(labels) - 1, -1, -1) if labels[i] != self.ignore_index)

        actual_response = labels[response_start : response_end - 1]
        self.assertEqual(
            actual_response,
            last_assistant_response_tokens,
            "The labels should preserve the last assistant's response tokens.",
        )

        # Verify that EOS token is at the end of labels
        self.assertEqual(labels[-1], self.eos_token_id, "The last token of labels should be EOS token.")


class TestBatchGeneration(unittest.TestCase):
    def setUp(self):
        # Initialize the tokenizer
        self.model_id = "Qwen/Qwen2-0.5B-Instruct"
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
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
        queries = self.tokenizer(batch, padding=True, return_tensors="pt")["input_ids"]
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
        queries = self.tokenizer(batch, padding=True, return_tensors="pt")["input_ids"]
        bs, context_length = queries.shape

        query_responses, logits = batch_generation(
            self.model, queries, bs, self.tokenizer.pad_token_id, self.generation_config
        )

        max_length_query = query_responses.shape[1]
        max_length_logits = max_length_query - context_length

        self.assertGreater(max_length_query, context_length)
        self.assertEqual(query_responses.shape, (bs, max_length_query))
        self.assertEqual(logits.shape, (bs, max_length_logits, self.model.config.vocab_size))
