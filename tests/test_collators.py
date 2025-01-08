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

import unittest

import torch

from trl.trainer.dpo_trainer import DataCollatorForPreference


class TestDataCollatorForPreference(unittest.TestCase):
    def setUp(self):
        self.collator = DataCollatorForPreference(pad_token_id=0)

    def assertTensorEqual(self, tensor1, tensor2):
        self.assertTrue(torch.equal(tensor1, tensor2), f"Tensors are not equal:\n{tensor1}\n{tensor2}")

    def test_padding_behavior(self):
        examples = [
            {"prompt_input_ids": [1, 2, 3], "chosen_input_ids": [4, 5], "rejected_input_ids": [6]},
            {"prompt_input_ids": [7, 8], "chosen_input_ids": [9, 10], "rejected_input_ids": [11, 12, 13]},
        ]
        output = self.collator.torch_call(examples)

        expected_prompt_input_ids = torch.tensor([[1, 2, 3], [0, 7, 8]])
        expected_prompt_attention_mask = torch.tensor([[1, 1, 1], [0, 1, 1]])
        expected_chosen_input_ids = torch.tensor([[4, 5], [9, 10]])
        expected_chosen_attention_mask = torch.tensor([[1, 1], [1, 1]])
        expected_rejected_input_ids = torch.tensor([[6, 0, 0], [11, 12, 13]])
        expected_rejected_attention_mask = torch.tensor([[1, 0, 0], [1, 1, 1]])

        self.assertTensorEqual(output["prompt_input_ids"], expected_prompt_input_ids)
        self.assertTensorEqual(output["prompt_attention_mask"], expected_prompt_attention_mask)
        self.assertTensorEqual(output["chosen_input_ids"], expected_chosen_input_ids)
        self.assertTensorEqual(output["chosen_attention_mask"], expected_chosen_attention_mask)
        self.assertTensorEqual(output["rejected_input_ids"], expected_rejected_input_ids)
        self.assertTensorEqual(output["rejected_attention_mask"], expected_rejected_attention_mask)

    def test_optional_fields(self):
        examples = [
            {
                "prompt_input_ids": [1],
                "chosen_input_ids": [2],
                "rejected_input_ids": [3],
                "pixel_values": [[[0.1, 0.2], [0.3, 0.4]]],  # Example 3D tensor (1x2x2)
            },
            {
                "prompt_input_ids": [4],
                "chosen_input_ids": [5],
                "rejected_input_ids": [6],
                "pixel_values": [[[0.5, 0.6], [0.7, 0.8]]],  # Example 3D tensor (1x2x2)
            },
        ]
        output = self.collator.torch_call(examples)

        expected_pixel_values = torch.tensor(
            [
                [[[0.1, 0.2], [0.3, 0.4]]],
                [[[0.5, 0.6], [0.7, 0.8]]],
            ]
        )  # Shape: (2, 1, 2, 2)

        self.assertTensorEqual(output["pixel_values"], expected_pixel_values)
