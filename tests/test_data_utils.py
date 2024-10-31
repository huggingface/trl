# Copyright 2024 The HuggingFace Team. All rights reserved.
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
from transformers import AutoTokenizer

from trl.data_utils import (
    apply_chat_template,
    extract_prompt,
    is_conversational,
    maybe_apply_chat_template,
    maybe_extract_prompt,
    maybe_unpair_preference_dataset,
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
        {  # Pompt-completion
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
        "trl-internal-testing/tiny-random-Qwen2-7B-Instruct",
        "trl-internal-testing/tiny-random-Meta-Llama-3.1-8B-Instruct",
        "trl-internal-testing/tiny-random-Meta-Llama-3-8B-Instruct",
        "trl-internal-testing/tiny-random-DeepSeek-Coder-V2-Instruct",
        "trl-internal-testing/tiny-random-Phi-3-mini-128k-instruct",
        "trl-internal-testing/tiny-random-gemma-2-9b-it",
        "trl-internal-testing/tiny-random-Mistral-7B-Instruct-v0.1",
        "trl-internal-testing/tiny-random-Mistral-7B-Instruct-v0.2",
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
        {  # Pompt-completion
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

        # The chat template should be applied to the the following keys
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

        # The chat template should be applied to the the following keys
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


# Run the tests
if __name__ == "__main__":
    unittest.main()
