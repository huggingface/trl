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

from copy import deepcopy

import torch

from trl.trainer.grpo_replay_buffer import repad


PAD_TOKEN_ID = 123


def test_repad_basic_padding():
    sample = [
        {
            "prompt_ids": torch.LongTensor([1, 2, 3]),
            "prompt_mask": torch.LongTensor([1, 1, 0]),
            "completion_ids": torch.LongTensor([5, 6, 7, 8]),
            "completion_mask": torch.LongTensor([1, 1, 1, 0]),
            "old_per_token_logps": torch.tensor([0.1, 0.2, 0.3, 0.4]),
            "ref_per_token_logps": torch.tensor([0.0, -0.1, -0.2, -0.3]),
        },
        {
            "prompt_ids": torch.LongTensor([4, 5]),
            "prompt_mask": torch.LongTensor([1, 1]),
            "completion_ids": torch.LongTensor([9, 10]),
            "completion_mask": torch.LongTensor([1, 1]),
            "old_per_token_logps": torch.tensor([-0.5, -0.6]),
            "ref_per_token_logps": torch.tensor([0.5, 0.6]),
        },
    ]

    padded = repad(deepcopy(sample), padding_value=PAD_TOKEN_ID)

    assert len(padded[0]["prompt_ids"]) == 2
    assert len(padded[0]["completion_ids"]) == 3

    for ex in padded:
        # All sequences in same batch should have same length
        assert len(ex["prompt_ids"]) == len(padded[0]["prompt_ids"])
        assert len(ex["prompt_mask"]) == len(padded[0]["prompt_mask"])
        assert len(ex["completion_ids"]) == len(padded[0]["completion_ids"])
        assert len(ex["completion_mask"]) == len(padded[0]["completion_mask"])

        # Mask and ids should match in shape
        assert ex["prompt_ids"].shape == ex["prompt_mask"].shape
        assert ex["completion_ids"].shape == ex["completion_mask"].shape


def test_repad_logps_padding():
    sample = [
        {
            "prompt_ids": torch.LongTensor([1]),
            "prompt_mask": torch.LongTensor([1]),
            "completion_ids": torch.LongTensor([2, 3, 4]),
            "completion_mask": torch.LongTensor([1, 1, 0]),
            "old_per_token_logps": torch.tensor([-0.1, -0.2, -0.3]),
            "ref_per_token_logps": torch.tensor([-0.5, -0.6, -0.7]),
        },
        {
            "prompt_ids": torch.LongTensor([5, 6]),
            "prompt_mask": torch.LongTensor([1, 1]),
            "completion_ids": torch.LongTensor([7, 8]),
            "completion_mask": torch.LongTensor([1, 1]),
            "old_per_token_logps": torch.tensor([0.4, 0.5]),
            "ref_per_token_logps": torch.tensor([0.6, 0.7]),
        },
    ]

    padded = repad(deepcopy(sample), padding_value=PAD_TOKEN_ID)

    for logps in ["old_per_token_logps", "ref_per_token_logps"]:
        for ex in padded:
            assert len(ex[logps]) == len(padded[0][logps])
            assert isinstance(ex[logps], torch.Tensor)


def test_repad_empty_masks():
    sample = [
        {
            "prompt_ids": torch.tensor([0]),
            "prompt_mask": torch.tensor([0]),
            "completion_ids": torch.tensor([0]),
            "completion_mask": torch.tensor([0]),
            "old_per_token_logps": torch.tensor([0.0]),
            "ref_per_token_logps": torch.tensor([0.0]),
        },
        {
            "prompt_ids": torch.tensor([1]),
            "prompt_mask": torch.tensor([0]),
            "completion_ids": torch.tensor([1]),
            "completion_mask": torch.tensor([0]),
            "old_per_token_logps": torch.tensor([0.0]),
            "ref_per_token_logps": torch.tensor([0.0]),
        },
        {
            "prompt_ids": torch.tensor([1, 1]),
            "prompt_mask": torch.tensor([0, 1]),
            "completion_ids": torch.tensor([1, 2]),
            "completion_mask": torch.tensor([1, 0]),
            "old_per_token_logps": torch.tensor([0.0, 1.0]),
            "ref_per_token_logps": torch.tensor([0.0, 1.0]),
        },
        {
            "prompt_ids": torch.tensor([1, 1]),
            "prompt_mask": torch.tensor([1, 1]),
            "completion_ids": torch.tensor([1, 2]),
            "completion_mask": torch.tensor([1, 0]),
            "old_per_token_logps": torch.tensor([0.0, 1.0]),
            "ref_per_token_logps": torch.tensor([0.0, 1.0]),
        },
    ]
    padded = repad(deepcopy(sample), padding_value=999)

    assert len(padded[0]["prompt_ids"]) == 2
    assert len(padded[0]["completion_ids"]) == 1

    assert padded[0]["prompt_ids"].eq(999).all()
    assert padded[0]["completion_ids"].eq(999).all()
