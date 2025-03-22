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

import itertools
import unittest

from datasets import Dataset, DatasetDict
from parameterized import parameterized
from transformers import AutoProcessor, AutoTokenizer

from trl.data_utils import (
    apply_chat_template,
    extract_prompt,
    is_conversational,
    maybe_apply_chat_template,
    maybe_convert_to_chatml,
    maybe_extract_prompt,
    maybe_unpair_preference_dataset,
    pack_dataset,
    pack_examples,
    truncate_dataset,
    unpair_preference_dataset,
)


class IsConversationalTester(unittest.TestCase):
    conversational_examples = [
        {  # Language modeling
            "messages": [
                {"role": "user", "content": "What color is the sky?"},
                {"role": "assistant", "content": "It is blue."},
            ],
        },
        {  # Prompt only
            "prompt": [{"role": "user", "content": "What color is the sky?"}],
        },
        {  # Prompt-completion
            "prompt": [{"role": "user", "content": "What color is the sky?"}],
            "completion": [{"role": "assistant", "content": "It is blue."}],
        },
        {  # Preference
            "prompt": [{"role": "user", "content": "What color is the sky?"}],
            "chosen": [{"role": "assistant", "content": "It is blue."}],
            "rejected": [{"role": "assistant", "content": "It is green."}],
        },
        {  # Preference with implicit prompt
            "chosen": [
                {"role": "user", "content": "What color is the sky?"},
                {"role": "assistant", "content": "It is blue."},
            ],
            "rejected": [
                {"role": "user", "content": "What color is the sky?"},
                {"role": "assistant", "content": "It is green."},
            ],
        },
        {  # Unpaired preference
            "prompt": [{"role": "user", "content": "What color is the sky?"}],
            "completion": [{"role": "assistant", "content": "It is blue."}],
            "label": True,
        },
    ]

    non_conversational_examples = [
        {"prompt": "The sky is", "completion": " blue."},
        {"text": "The sky is blue."},
        {"prompt": "The sky is"},
        {"prompt": "The sky is", "chosen": " blue.", "rejected": " green."},
        {"prompt": "The sky is", "completion": " blue.", "label": True},
    ]

    @parameterized.expand(itertools.product(conversational_examples))
    def test_conversational(self, example):
        self.assertTrue(is_conversational(example))

    @parameterized.expand(itertools.product(non_conversational_examples))
    def test_non_conversational(self, example):
        self.assertFalse(is_conversational(example))


class ApplyChatTemplateTester(unittest.TestCase):
    tokenizers = [
        "trl-internal-testing/tiny-CohereForCausalLM",
        "trl-internal-testing/tiny-DbrxForCausalLM",
        "trl-internal-testing/tiny-FalconMambaForCausalLM",
        "trl-internal-testing/tiny-Gemma2ForCausalLM",
        "trl-internal-testing/tiny-GemmaForCausalLM",
        "trl-internal-testing/tiny-LlamaForCausalLM-3.1",
        "trl-internal-testing/tiny-LlamaForCausalLM-3.2",
        "trl-internal-testing/tiny-LlamaForCausalLM-3",
        "trl-internal-testing/tiny-MistralForCausalLM-0.1",
        "trl-internal-testing/tiny-MistralForCausalLM-0.2",
        "trl-internal-testing/tiny-Phi3ForCausalLM",
        "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
    ]

    conversational_examples = [
        {  # Language modeling
            "messages": [
                {"role": "user", "content": "What color is the sky?"},
                {"role": "assistant", "content": "It is blue."},
            ],
        },
        {  # Prompt only
            "prompt": [{"role": "user", "content": "What color is the sky?"}],
        },
        {  # Prompt-completion
            "prompt": [{"role": "user", "content": "What color is the sky?"}],
            "completion": [{"role": "assistant", "content": "It is blue."}],
        },
        {  # Preference
            "prompt": [{"role": "user", "content": "What color is the sky?"}],
            "chosen": [{"role": "assistant", "content": "It is blue."}],
            "rejected": [{"role": "assistant", "content": "It is green."}],
        },
        {  # Preference with implicit prompt
            "chosen": [
                {"role": "user", "content": "What color is the sky?"},
                {"role": "assistant", "content": "It is blue."},
            ],
            "rejected": [
                {"role": "user", "content": "What color is the sky?"},
                {"role": "assistant", "content": "It is green."},
            ],
        },
        {  # Unpaired preference
            "prompt": [{"role": "user", "content": "What color is the sky?"}],
            "completion": [{"role": "assistant", "content": "It is blue."}],
            "label": True,
        },
    ]

    non_conversational_examples = [
        {"prompt": "The sky is", "completion": " blue."},
        {"text": "The sky is blue."},
        {"prompt": "The sky is"},
        {"prompt": "The sky is", "chosen": " blue.", "rejected": " green."},
        {"chosen": "The sky is blue.", "rejected": "The sky is green."},
        {"prompt": "The sky is", "completion": " blue.", "label": True},
    ]

    @parameterized.expand(itertools.product(tokenizers, conversational_examples))
    def test_apply_chat_template(self, tokenizer_id, example):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        result = apply_chat_template(example, tokenizer)

        # Checking if the result is a dictionary
        self.assertIsInstance(result, dict)

        # The chat template should be applied to the following keys
        for key in ["prompt", "chosen", "rejected", "completion"]:
            if key in example:
                self.assertIn(key, result)
                self.assertIsInstance(result[key], str)

        # Exception for messages, the key is "text" once the chat template is applied
        if "messages" in example:
            self.assertIn("text", result)
            self.assertIsInstance(result["text"], str)

        # The label should be kept
        if "label" in example:
            self.assertIn("label", result)
            self.assertIsInstance(result["label"], bool)
            self.assertEqual(result["label"], example["label"])

    # both conversational and non-conversational examples
    @parameterized.expand(itertools.product(tokenizers, conversational_examples + non_conversational_examples))
    def test_maybe_apply_chat_template(self, tokenizer_id, example):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        result = maybe_apply_chat_template(example, tokenizer)

        # Checking if the result is a dictionary
        self.assertIsInstance(result, dict)

        # The chat template should be applied to the following keys
        for key in ["prompt", "chosen", "rejected", "completion"]:
            if key in example:
                self.assertIn(key, result)
                self.assertIsInstance(result[key], str)

        # Exception for messages, the key is "text" once the chat template is applied
        if "messages" in example:
            self.assertIn("text", result)
            self.assertIsInstance(result["text"], str)

        # The label should be kept
        if "label" in example:
            self.assertIn("label", result)
            self.assertIsInstance(result["label"], bool)
            self.assertEqual(result["label"], example["label"])

    def test_apply_chat_template_with_tools(self):
        tokenizer = AutoProcessor.from_pretrained("trl-internal-testing/tiny-LlamaForCausalLM-3.2")

        # Define dummy test tools
        def get_current_temperature(location: str):
            """
            Gets the temperature at a given location.

            Args:
                location: The location to get the temperature for
            """
            return 22.0

        # Define test case
        test_case = {
            "prompt": [
                {"content": "Whats the temperature in London?", "role": "user"},
            ]
        }
        # Test with tools
        result_with_tools = apply_chat_template(test_case, tokenizer, tools=[get_current_temperature])

        # Verify tools are included in the output
        self.assertIn("get_current_temperature", result_with_tools["prompt"])

        # Test without tools
        result_without_tools = apply_chat_template(test_case, tokenizer, tools=None)

        # Verify tools are not included in the output
        self.assertNotIn("get_current_temperature", result_without_tools["prompt"])


class UnpairPreferenceDatasetTester(unittest.TestCase):
    paired_dataset = Dataset.from_dict(
        {
            "prompt": ["The sky is", "The sun is"],
            "chosen": [" blue.", " in the sky."],
            "rejected": [" green.", " in the sea."],
        }
    )

    unpaired_dataset = Dataset.from_dict(
        {
            "prompt": ["The sky is", "The sun is", "The sky is", "The sun is"],
            "completion": [" blue.", " in the sky.", " green.", " in the sea."],
            "label": [True, True, False, False],
        }
    )

    def test_unpair_preference_dataset(self):
        # Test that a paired dataset is correctly converted to unpaired
        unpaired_dataset = unpair_preference_dataset(self.paired_dataset)
        self.assertEqual(
            unpaired_dataset.to_dict(),
            self.unpaired_dataset.to_dict(),
            "The paired dataset should be converted to unpaired.",
        )

    def test_unpair_preference_dataset_dict(self):
        # Test that a paired dataset dict is correctly converted to unpaired
        paired_dataset_dict = DatasetDict({"abc": self.paired_dataset})
        unpaired_dataset_dict = unpair_preference_dataset(paired_dataset_dict)
        self.assertEqual(
            unpaired_dataset_dict["abc"].to_dict(),
            self.unpaired_dataset.to_dict(),
            "The paired dataset should be converted to unpaired.",
        )

    def test_maybe_unpair_preference_dataset(self):
        # Test that a paired dataset is correctly converted to unpaired with maybe_unpair_preference_dataset
        unpaired_dataset = maybe_unpair_preference_dataset(self.paired_dataset)
        self.assertEqual(
            unpaired_dataset.to_dict(),
            self.unpaired_dataset.to_dict(),
            "The paired dataset should be converted to unpaired.",
        )

    def test_maybe_unpair_preference_dataset_dict(self):
        # Test that a paired dataset dict is correctly converted to unpaired with maybe_unpair_preference_dataset
        paired_dataset_dict = DatasetDict({"abc": self.paired_dataset})
        unpaired_dataset_dict = maybe_unpair_preference_dataset(paired_dataset_dict)
        self.assertEqual(
            unpaired_dataset_dict["abc"].to_dict(),
            self.unpaired_dataset.to_dict(),
            "The paired dataset should be converted to unpaired.",
        )

    def test_maybe_unpair_preference_dataset_already_paired(self):
        # Test that a paired dataset remains unchanged with maybe_unpair_preference_dataset
        unpaired_dataset = maybe_unpair_preference_dataset(self.unpaired_dataset)
        self.assertEqual(
            unpaired_dataset.to_dict(),
            self.unpaired_dataset.to_dict(),
            "The unpaired dataset should remain unchanged.",
        )

    def test_maybe_unpair_preference_dataset_dict_already_paired(self):
        # Test that a paired dataset dict remains unchanged with maybe_unpair_preference_dataset
        unpaired_dataset_dict = maybe_unpair_preference_dataset(DatasetDict({"abc": self.unpaired_dataset}))
        self.assertEqual(
            unpaired_dataset_dict["abc"].to_dict(),
            self.unpaired_dataset.to_dict(),
            "The unpaired dataset should remain unchanged.",
        )


class ExtractPromptTester(unittest.TestCase):
    example_implicit_prompt_conversational = {
        "chosen": [
            {"role": "user", "content": "What color is the sky?"},
            {"role": "assistant", "content": "It is blue."},
        ],
        "rejected": [
            {"role": "user", "content": "What color is the sky?"},
            {"role": "assistant", "content": "It is green."},
        ],
    }

    example_explicit_prompt_conversational = {
        "prompt": [
            {"role": "user", "content": "What color is the sky?"},
        ],
        "chosen": [
            {"role": "assistant", "content": "It is blue."},
        ],
        "rejected": [
            {"role": "assistant", "content": "It is green."},
        ],
    }

    example_implicit_prompt_standard = {
        "chosen": "The sky is blue.",
        "rejected": "The sky is green.",
    }

    example_explicit_prompt_standard = {
        "prompt": "The sky is",
        "chosen": " blue.",
        "rejected": " green.",
    }

    def test_extract_prompt_conversational(self):
        # Test that the prompt is correctly extracted from the dataset
        example_extracted_prompt = extract_prompt(self.example_implicit_prompt_conversational)
        self.assertEqual(
            example_extracted_prompt,
            self.example_explicit_prompt_conversational,
            "The prompt is not correctly extracted from the dataset.",
        )

    def test_maybe_extract_prompt_conversational(self):
        # Test that the prompt is correctly extracted from the dataset with maybe_extract_prompt
        example_extracted_prompt = maybe_extract_prompt(self.example_implicit_prompt_conversational)
        self.assertEqual(
            example_extracted_prompt,
            self.example_explicit_prompt_conversational,
            "The prompt is not correctly extracted from the dataset.",
        )

    def test_maybe_extract_prompt_conversational_already_explicit(self):
        # Test that the prompt remains unchanged with maybe_extract_prompt
        example_extracted_prompt = maybe_extract_prompt(self.example_explicit_prompt_conversational)
        self.assertEqual(
            example_extracted_prompt,
            self.example_explicit_prompt_conversational,
            "The prompt should remain unchanged.",
        )

    def test_extract_prompt_standard(self):
        # Test that the prompt is correctly extracted from the dataset
        example_extracted_prompt = extract_prompt(self.example_implicit_prompt_standard)
        self.assertEqual(
            example_extracted_prompt,
            self.example_explicit_prompt_standard,
            "The prompt is not correctly extracted from the dataset.",
        )

    def test_maybe_extract_prompt_standard(self):
        # Test that the prompt is correctly extracted from the dataset with maybe_extract_prompt
        example_extracted_prompt = maybe_extract_prompt(self.example_implicit_prompt_standard)
        self.assertEqual(
            example_extracted_prompt,
            self.example_explicit_prompt_standard,
            "The prompt is not correctly extracted from the dataset.",
        )

    def test_maybe_extract_prompt_standard_already_explicit(self):
        # Test that the prompt remains unchanged with maybe_extract_prompt
        example_extracted_prompt = maybe_extract_prompt(self.example_explicit_prompt_standard)
        self.assertEqual(
            example_extracted_prompt,
            self.example_explicit_prompt_standard,
            "The prompt should remain unchanged.",
        )


class TestPackExamples(unittest.TestCase):
    def test_larger_chunks(self):
        examples = {
            "input_ids": [[1, 2, 3], [4, 5, 6, 7], [8]],
            "attention_mask": [[0, 1, 1], [0, 0, 1, 1], [1]],
        }
        seq_length = 5
        expected_output = {
            "input_ids": [[1, 2, 3, 4, 5], [6, 7, 8]],
            "attention_mask": [[0, 1, 1, 0, 0], [1, 1, 1]],
        }
        result = pack_examples(examples, seq_length)
        self.assertEqual(result, expected_output)

    def test_smaller_chunks(self):
        examples = {
            "input_ids": [[1, 2, 3], [4, 5, 6, 7], [8]],
            "attention_mask": [[0, 1, 1], [0, 0, 1, 1], [1]],
        }
        seq_length = 2
        expected_output = {
            "input_ids": [[1, 2], [3, 4], [5, 6], [7, 8]],
            "attention_mask": [[0, 1], [1, 0], [0, 1], [1, 1]],
        }
        result = pack_examples(examples, seq_length)
        self.assertEqual(result, expected_output)

    def test_with_dataset(self):
        examples = {
            "input_ids": [[1, 2, 3], [4, 5, 6, 7], [8]],
            "attention_mask": [[0, 1, 1], [0, 0, 1, 1], [1]],
        }
        dataset = Dataset.from_dict(examples)
        seq_length = 3
        expected_output = {
            "input_ids": [[1, 2, 3], [4, 5, 6], [7, 8]],
            "attention_mask": [[0, 1, 1], [0, 0, 1], [1, 1]],
        }
        dataset = dataset.map(pack_examples, batched=True, fn_kwargs={"seq_length": seq_length})
        self.assertEqual(dataset.to_dict(), expected_output)


class TestPackDataset(unittest.TestCase):
    def test_with_dataset(self):
        examples = {
            "input_ids": [[1, 2, 3], [4, 5, 6, 7], [8]],
            "attention_mask": [[0, 1, 1], [0, 0, 1, 1], [1]],
        }
        dataset = Dataset.from_dict(examples)
        seq_length = 3
        expected_output = {
            "input_ids": [[1, 2, 3], [4, 5, 6], [7, 8]],
            "attention_mask": [[0, 1, 1], [0, 0, 1], [1, 1]],
        }
        dataset = pack_dataset(dataset, seq_length)
        self.assertEqual(dataset.to_dict(), expected_output)

    def test_with_iterable_dataset(self):
        examples = {
            "input_ids": [[1, 2, 3], [4, 5, 6, 7], [8]],
            "attention_mask": [[0, 1, 1], [0, 0, 1, 1], [1]],
        }
        dataset = Dataset.from_dict(examples).to_iterable_dataset()
        seq_length = 3
        expected_output = {
            "input_ids": [[1, 2, 3], [4, 5, 6], [7, 8]],
            "attention_mask": [[0, 1, 1], [0, 0, 1], [1, 1]],
        }
        dataset = pack_dataset(dataset, seq_length)
        num_examples = len(examples[next(iter(examples))])
        self.assertEqual(next(iter(dataset.batch(batch_size=num_examples))), expected_output)


class TestTruncateExamples(unittest.TestCase):
    def test_with_dataset(self):
        examples = {
            "input_ids": [[1, 2, 3], [4, 5, 6, 7], [8]],
            "attention_mask": [[0, 1, 1], [0, 0, 1, 1], [1]],
        }
        dataset = Dataset.from_dict(examples)
        max_length = 2
        expected_output = {
            "input_ids": [[1, 2], [4, 5], [8]],
            "attention_mask": [[0, 1], [0, 0], [1]],
        }
        dataset = truncate_dataset(dataset, max_length)
        self.assertEqual(dataset.to_dict(), expected_output)

    def test_with_iterable_dataset(self):
        examples = {
            "input_ids": [[1, 2, 3], [4, 5, 6, 7], [8]],
            "attention_mask": [[0, 1, 1], [0, 0, 1, 1], [1]],
        }
        dataset = Dataset.from_dict(examples).to_iterable_dataset()
        max_length = 2
        expected_output = {
            "input_ids": [[1, 2], [4, 5], [8]],
            "attention_mask": [[0, 1], [0, 0], [1]],
        }
        dataset = truncate_dataset(dataset, max_length)
        num_examples = len(examples[next(iter(examples))])
        self.assertEqual(next(iter(dataset.batch(batch_size=num_examples))), expected_output)

    def test_with_extra_column(self):
        examples = {
            "input_ids": [[1, 2, 3], [4, 5, 6, 7], [8]],
            "attention_mask": [[0, 1, 1], [0, 0, 1, 1], [1]],
            "my_column": ["a", "b", "c"],
        }
        dataset = Dataset.from_dict(examples)
        max_length = 2
        expected_output = {
            "input_ids": [[1, 2], [4, 5], [8]],
            "attention_mask": [[0, 1], [0, 0], [1]],
            "my_column": ["a", "b", "c"],
        }
        dataset = truncate_dataset(dataset, max_length)
        self.assertEqual(dataset.to_dict(), expected_output)


class TestMaybeConvertToChatML(unittest.TestCase):
    def test_with_conversations_key(self):
        # Particular case where the key is "conversations": we rename it to "messages"
        example = {
            "conversations": [
                {"from": "user", "value": "What color is the sky?"},
                {"from": "assistant", "value": "It is blue."},
            ]
        }
        expected_output = {
            "messages": [
                {"role": "user", "content": "What color is the sky?"},
                {"role": "assistant", "content": "It is blue."},
            ]
        }
        self.assertEqual(maybe_convert_to_chatml(example), expected_output)

    def test_without_conversations_key(self):
        # Same as before, but we don't rename the keys
        example = {
            "prompt": [{"from": "user", "value": "What color is the sky?"}],
            "completion": [{"from": "assistant", "value": "It is blue."}],
        }
        expected_output = {
            "prompt": [{"role": "user", "content": "What color is the sky?"}],
            "completion": [{"role": "assistant", "content": "It is blue."}],
        }
        self.assertEqual(maybe_convert_to_chatml(example), expected_output)

    def test_not_conversional(self):
        # When not needed, the example should remain unchanged
        example = {"text": "The sky is blue."}
        self.assertEqual(maybe_convert_to_chatml(example), example)

    def test_already_chatml(self):
        # When the example is already in ChatML format, it should remain unchanged
        example = {
            "messages": [
                {"role": "user", "content": "What color is the sky?"},
                {"role": "assistant", "content": "It is blue."},
            ]
        }
        self.assertEqual(maybe_convert_to_chatml(example), example)


# Run the tests
if __name__ == "__main__":
    unittest.main()
