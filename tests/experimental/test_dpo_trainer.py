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

import torch

from trl.experimental.dpo.dpo_trainer import DataCollatorForPreference

from ..testing_utils import TrlTestCase


class TestDataCollatorForPreference(TrlTestCase):
    def test_padding_and_masks(self):
        collator = DataCollatorForPreference(pad_token_id=0)
        examples = [
            {"prompt_ids": [1, 2, 3], "chosen_ids": [4, 5], "rejected_ids": [6]},
            {"prompt_ids": [7, 8], "chosen_ids": [9, 10], "rejected_ids": [11, 12, 13]},
        ]

        output = collator(examples)

        expected_input_ids = torch.tensor(
            [
                [1, 2, 3, 4, 5],  # prompt + chosen (example 1)
                [7, 8, 9, 10, 0],  # prompt + chosen (example 2, padded)
                [1, 2, 3, 6, 0],  # prompt + rejected (example 1, padded)
                [7, 8, 11, 12, 13],  # prompt + rejected (example 2)
            ]
        )
        expected_attention_mask = torch.tensor(
            [
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 0],
                [1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1],
            ]
        )
        expected_completion_mask = torch.tensor(
            [
                [0, 0, 0, 1, 1],  # chosen completion (example 1)
                [0, 0, 1, 1, 0],  # chosen completion (example 2, padded)
                [0, 0, 0, 1, 0],  # rejected completion (example 1, padded)
                [0, 0, 1, 1, 1],  # rejected completion (example 2)
            ]
        )

        assert torch.equal(output["input_ids"], expected_input_ids)
        assert torch.equal(output["attention_mask"], expected_attention_mask)
        assert torch.equal(output["completion_mask"], expected_completion_mask)

    def test_optional_reference_logps(self):
        collator = DataCollatorForPreference(pad_token_id=0)
        examples = [
            {
                "prompt_ids": [1, 2],
                "chosen_ids": [3],
                "rejected_ids": [4],
                "ref_chosen_logps": 0.1,
                "ref_rejected_logps": 0.2,
            },
            {
                "prompt_ids": [5],
                "chosen_ids": [6, 7],
                "rejected_ids": [8, 9],
                "ref_chosen_logps": 0.3,
                "ref_rejected_logps": 0.4,
            },
        ]

        output = collator(examples)

        expected_ref_chosen_logps = torch.tensor([0.1, 0.3])
        expected_ref_rejected_logps = torch.tensor([0.2, 0.4])

        assert torch.equal(output["ref_chosen_logps"], expected_ref_chosen_logps)
        assert torch.equal(output["ref_rejected_logps"], expected_ref_rejected_logps)

    def test_with_pad_to_multiple_of(self):
        collator = DataCollatorForPreference(pad_token_id=0, pad_to_multiple_of=5)
        examples = [
            {"prompt_ids": [1], "chosen_ids": [2], "rejected_ids": [3]},
            {"prompt_ids": [4, 5], "chosen_ids": [6, 7], "rejected_ids": [8, 9]},
        ]

        output = collator(examples)

        expected_input_ids = torch.tensor(
            [
                [1, 2, 0, 0, 0],  # prompt + chosen (example 1, padded to multiple of 5)
                [4, 5, 6, 7, 0],  # prompt + chosen (example 2)
                [1, 3, 0, 0, 0],  # prompt + rejected (example 1, padded to multiple of 5)
                [4, 5, 8, 9, 0],  # prompt + rejected (example 2)
            ]
        )

        assert torch.equal(output["input_ids"], expected_input_ids)
