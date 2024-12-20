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

import torch

from trl.trainer.kto_trainer import DataCollatorForUnpairedPreference


class TestDataCollatorForUnpairedPreference(unittest.TestCase):
    def setUp(self):
        self.collator = DataCollatorForUnpairedPreference(pad_token_id=0)

    def assertTensorEqual(self, tensor1, tensor2):
        self.assertTrue(torch.equal(tensor1, tensor2), f"Tensors are not equal:\n{tensor1}\n{tensor2}")

    def test_padding_behavior(self):
        examples = [
            {"prompt_input_ids": [1, 2, 3], "completion_input_ids": [4, 5], "label": True},
            {"prompt_input_ids": [6, 7], "completion_input_ids": [8, 9, 10], "label": False},
        ]
        output = self.collator.torch_call(examples)

        expected_prompt_input_ids = torch.tensor([[1, 2, 3], [0, 6, 7]])
        expected_prompt_attention_mask = torch.tensor([[1, 1, 1], [0, 1, 1]])
        expected_completion_input_ids = torch.tensor([[4, 5, 0], [8, 9, 10]])
        expected_completion_attention_mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
        expected_labels = torch.tensor([True, False])

        self.assertTensorEqual(output["prompt_input_ids"], expected_prompt_input_ids)
        self.assertTensorEqual(output["prompt_attention_mask"], expected_prompt_attention_mask)
        self.assertTensorEqual(output["completion_input_ids"], expected_completion_input_ids)
        self.assertTensorEqual(output["completion_attention_mask"], expected_completion_attention_mask)
        self.assertTensorEqual(output["labels"], expected_labels)
