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
import unittest

from datasets import Dataset, DatasetDict

from trl.data_utils import maybe_reformat_dpo_to_kto


class MaybeReformatDPOToKTOTester(unittest.TestCase):
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
        reformatted_dataset = maybe_reformat_dpo_to_kto(self.dpo_dataset)
        self.assertEqual(
            reformatted_dataset["train"].to_dict(),
            self.kto_dataset["train"].to_dict(),
            "The DPO-formatted dataset was not correctly reformatted to KTO format.",
        )

    def test_already_kto_format(self):
        # Test that a KTO-formatted dataset remains unchanged
        reformatted_dataset = maybe_reformat_dpo_to_kto(self.kto_dataset)
        self.assertEqual(
            reformatted_dataset["train"].to_dict(),
            self.kto_dataset["train"].to_dict(),
            "The KTO-formatted dataset should remain unchanged.",
        )

    def test_invalid_format(self):
        # Test that a dataset with an incompatible format raises a ValueError
        invalid_data = {
            "input": ["What is AI?", "Define machine learning."],
            "output": ["AI is artificial intelligence.", "Machine learning is a subset of AI."],
        }
        invalid_dataset = DatasetDict({"train": Dataset.from_dict(invalid_data)})

        with self.assertRaises(ValueError):
            maybe_reformat_dpo_to_kto(invalid_dataset)
