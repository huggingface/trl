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
    is_conversational,
    maybe_unpair_preference_dataset,
    unpair_preference_dataset,
)


class TestIsConversational(unittest.TestCase):
    conversational_examples = [
        {  # Name to find
            "messages": [
                [
                    {"role": "user", "content": "What color is the sky?"},
                    {"role": "assistant", "content": "It is blue."},
                ]
            ],
        },
        {  # Prompt only
            "prompt": [[{"role": "user", "content": "What color is the sky?"}]],
        },
        {  # Pompt-completion
            "prompt": [[{"role": "user", "content": "What color is the sky?"}]],
            "completion": [[{"role": "assistant", "content": "It is blue."}]],
        },
        {  # Preference
            "prompt": [[{"role": "user", "content": "What color is the sky?"}]],
            "chosen": [[{"role": "assistant", "content": "It is blue."}]],
            "rejected": [[{"role": "assistant", "content": "It is green."}]],
        },
        {  # Unpaired preference
            "prompt": [[{"role": "user", "content": "What color is the sky?"}]],
            "completion": [[{"role": "assistant", "content": "It is blue."}]],
            "label": [True],
        },
    ]

    non_conversational_examples = [
        {"prompt": ["The sky is"], "completion": [" blue."]},
        {"text": ["The sky is blue."]},
        {"prompt": ["The sky is"]},
        {"prompt": ["The sky is"], "chosen": [" blue."], "rejected": [" green."]},
        {"prompt": ["The sky is"], "completion": [" blue."], "label": [True]},
    ]

    @parameterized.expand(itertools.product(conversational_examples))
    def test_conversational(self, example):
        dataset = Dataset.from_dict(example)
        self.assertTrue(is_conversational(dataset))

    @parameterized.expand(itertools.product(non_conversational_examples))
    def test_non_conversational(self, example):
        dataset = Dataset.from_dict(example)
        self.assertFalse(is_conversational(dataset))

    @parameterized.expand(itertools.product(conversational_examples))
    def test_conversational_datasetdict(self, example):
        dataset_dict = DatasetDict({"train": Dataset.from_dict(example)})
        self.assertTrue(is_conversational(dataset_dict))

    @parameterized.expand(itertools.product(non_conversational_examples))
    def test_non_conversational_datasetdict(self, example):
        dataset_dict = DatasetDict({"train": Dataset.from_dict(example)})
        self.assertFalse(is_conversational(dataset_dict))


class TestApplyChatTemplate(unittest.TestCase):
    tokenizers = [
        "Qwen/Qwen2-7B-Instruct",
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "deepseek-ai/DeepSeek-Coder-V2-Instruct",
        "microsoft/Phi-3-mini-128k-instruct",
        "google/gemma-2-9b-it",
        "mistralai/Mistral-7B-Instruct-v0.1",
        "mistralai/Mistral-7B-Instruct-v0.2",
        "mistralai/Mistral-7B-Instruct-v0.3",
    ]

    examples = [
        {  # Name to find
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
        {  # Unpaired preference
            "prompt": [{"role": "user", "content": "What color is the sky?"}],
            "completion": [{"role": "assistant", "content": "It is blue."}],
            "label": True,
        },
    ]

    @parameterized.expand(itertools.product(tokenizers, examples))
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

        # The label should be kept
        if "label" in example:
            self.assertIn("label", result)
            self.assertIsInstance(result["label"], bool)
            self.assertEqual(result["label"], example["label"])


class UnpairPreferenceDatasetTester(unittest.TestCase):
    def setUp(self):
        # Create a sample DPO-formatted dataset for testing
        self.dpo_data = {
            "prompt": ["What is AI?", "Define machine learning."],
            "chosen": ["AI is artificial intelligence.", "Machine learning is a subset of AI."],
            "rejected": ["AI is a computer.", "Machine learning is a program."],
        }
        self.dpo_dataset = DatasetDict({"train": Dataset.from_dict(self.dpo_data)})

        # Create a sample KTO-formatted dataset for testing
        self.kto_data = {
            "prompt": ["What is AI?", "Define machine learning.", "What is AI?", "Define machine learning."],
            "completion": [
                "AI is artificial intelligence.",
                "Machine learning is a subset of AI.",
                "AI is a computer.",
                "Machine learning is a program.",
            ],
            "label": [True, True, False, False],
        }
        self.kto_dataset = DatasetDict({"train": Dataset.from_dict(self.kto_data)})

    def test_dpo_to_kto_conversion(self):
        # Test that a DPO-formatted dataset is correctly reformatted to KTO format
        reformatted_dataset = unpair_preference_dataset(self.dpo_dataset)
        self.assertEqual(
            reformatted_dataset["train"].to_dict(),
            self.kto_dataset["train"].to_dict(),
            "The DPO-formatted dataset was not correctly reformatted to KTO format.",
        )

    def test_already_kto_format(self):
        # Test that a KTO-formatted dataset remains unchanged
        reformatted_dataset = maybe_unpair_preference_dataset(self.kto_dataset)
        self.assertEqual(
            reformatted_dataset["train"].to_dict(),
            self.kto_dataset["train"].to_dict(),
            "The KTO-formatted dataset should remain unchanged.",
        )


# Run the tests
if __name__ == "__main__":
    unittest.main()
