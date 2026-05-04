# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
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

import pytest
import torch
from datasets import load_dataset

from trl.experimental.grpo_with_replay_buffer import (
    GRPOWithReplayBufferConfig,
    GRPOWithReplayBufferTrainer,
    ReplayBuffer,
)

from ..testing_utils import TrlTestCase


@pytest.mark.low_priority
class TestReplayBuffer:
    def setup_method(self):
        self.replay_buffer = ReplayBuffer(max_size=5)

    def test_add(self):
        # Add elements to the replay buffer
        scores = [0.5, 0.8, 0.3, 0.9, 0.7]
        data = [
            {"id": 1},
            {"id": 2},
            {"id": 3},
            {"id": 4},
            {"id": 5},
        ]
        self.replay_buffer.add(scores, data)

        # Check if the buffer contains the correct number of elements
        assert len(self.replay_buffer.heap) == 5

        # Check if the buffer maintains the min-heap property
        heap_scores = [item[0] for item in self.replay_buffer.heap]
        assert heap_scores[0] == min(heap_scores)
        assert heap_scores[0] == 0.3

    def test_add_more_than_maxlen(self):
        # Add elements to the replay buffer
        scores = [0.5, 0.8, 0.3, 0.9, 0.7, 0.6, 0.4]
        data = [
            {"id": 1},
            {"id": 2},
            {"id": 3},
            {"id": 4},
            {"id": 5},
            {"id": 6},
            {"id": 7},
        ]
        self.replay_buffer.add(scores, data)

        # Check if the buffer contains the correct number of elements
        assert len(self.replay_buffer.heap) == 5

        # Check if the buffer maintains the min-heap property
        heap_scores = [item[0] for item in self.replay_buffer.heap]
        assert heap_scores[0] == min(heap_scores)
        assert heap_scores[0] == 0.5  # 0.3 and 0.4 should be removed

    def test_sample(self):
        # Add elements to the replay buffer
        scores = [0.5, 0.8, 0.3, 0.9, 0.7]
        data = [
            {"id": 1},
            {"id": 2},
            {"id": 3},
            {"id": 4},
            {"id": 5},
        ]
        self.replay_buffer.add(scores, data)

        # Sample elements from the buffer
        sampled = self.replay_buffer.sample(num_samples=3)

        # Check if the sampled elements are from the buffer
        assert len(sampled) == 3
        for item in sampled:
            assert item in [entry[1] for entry in self.replay_buffer.heap]


@pytest.mark.low_priority
class TestUpdateWithReplayBuffer:
    def setup_method(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")
        config = GRPOWithReplayBufferConfig(
            replay_buffer_size=5,
        )
        self.trainer = GRPOWithReplayBufferTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=config,
            train_dataset=dataset,
        )
        self.trainer.replay_buffer = ReplayBuffer(max_size=5)
        self.trainer.num_generations = 2

    def _prepopulate_buffer(self, with_pixels=False, with_logprobs=False):
        scores = [0.1, 0.9]
        data = [
            {
                "prompt_ids": torch.tensor([[100, 101], [102, 103]]),
                "prompt_mask": torch.ones(2, 2, dtype=torch.long),
                "completion_ids": torch.tensor([[5, 6], [7, 8]]),
                "completion_mask": torch.ones(2, 2, dtype=torch.long),
                "advantages": torch.tensor([[0.5, 0.6]]),
                **({"pixel_values": torch.randn(2, 3, 224, 224)} if with_pixels else {}),
                **({"old_per_token_logps": torch.randn(2, 2)} if with_logprobs else {}),
            },
            {
                "prompt_ids": torch.tensor([[104, 105], [106, 107]]),
                "prompt_mask": torch.ones(2, 2, dtype=torch.long),
                "completion_ids": torch.tensor([[13, 14], [15, 16]]),
                "completion_mask": torch.ones(2, 2, dtype=torch.long),
                "advantages": torch.tensor([[0.8, 0.85]]),
                **({"pixel_values": torch.randn(2, 3, 224, 224)} if with_pixels else {}),
                **({"old_per_token_logps": torch.randn(2, 2)} if with_logprobs else {}),
            },
        ]
        self.trainer.replay_buffer.add(scores, data)

    def _make_inputs(self, group_advantages, with_pixels=False, with_logprobs=False):
        inputs = {
            "group_advantages": group_advantages,
            "prompt_ids": torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]]),
            "prompt_mask": torch.ones(4, 2, dtype=torch.long),
            "completion_ids": torch.tensor([[9, 10], [11, 12], [13, 14], [15, 16]]),
            "completion_mask": torch.ones(4, 2, dtype=torch.long),
            "forward_kwargs": {"pixel_values": torch.randn(4, 3, 224, 224)} if with_pixels else {},
            "old_per_token_logps": torch.randn(4, 2) if with_logprobs else None,
        }
        inputs["group_std_rewards"] = group_advantages.std(dim=1).expand_as(group_advantages)
        return inputs

    def test_update_with_replay_buffer_no_variance(self):
        self._prepopulate_buffer(with_pixels=True, with_logprobs=True)
        group_advantages = torch.tensor([[0.5, 0.5], [0.8, 0.8]])  # no variance
        inputs = self._make_inputs(group_advantages, with_pixels=True, with_logprobs=True)
        original_prompt_ids = inputs["prompt_ids"].clone()

        outputs = self.trainer.update_with_replay_buffer(**inputs, num_items_in_batch=4)

        assert outputs is not None
        assert "pixel_values" in outputs
        assert "old_per_token_logps" in outputs
        assert len(self.trainer.replay_buffer.heap) == 2
        for pid in outputs["prompt_ids"]:
            assert pid.tolist() not in original_prompt_ids.tolist()

    def test_update_with_replay_buffer_with_variance(self):
        self._prepopulate_buffer()
        group_advantages = torch.tensor([[0.6, 0.4], [0.7, 1.2]])  # has variance
        inputs = self._make_inputs(group_advantages)

        sampled = self.trainer.update_with_replay_buffer(**inputs, num_items_in_batch=4)

        assert len(self.trainer.replay_buffer.heap) == 4  # grew
        assert sampled is None

    def test_update_with_mixed_variance(self):
        self._prepopulate_buffer()
        group_advantages = torch.tensor([[0.6, 0.6], [0.3, 0.45]])  # one no-variance, one variance
        inputs = self._make_inputs(group_advantages)
        original_prompt_ids = inputs["prompt_ids"].clone().view(-1, self.trainer.num_generations, 2).tolist()

        outputs = self.trainer.update_with_replay_buffer(**inputs, num_items_in_batch=4)

        assert len(self.trainer.replay_buffer.heap) == 3  # grew by 1
        output_prompt_ids = outputs["prompt_ids"].view(-1, self.trainer.num_generations, 2).tolist()

        buffer_ids = [item[1]["prompt_ids"].tolist() for item in self.trainer.replay_buffer.heap]
        found_from_buffer = any(pid in buffer_ids for pid in output_prompt_ids)
        found_from_original = any(pid in original_prompt_ids for pid in output_prompt_ids)

        assert found_from_buffer
        assert found_from_original
        assert [[1, 2], [3, 4]] not in output_prompt_ids  # excluded no-variance group

    def test_update_with_inputs_different_seq_len(self):
        """
        Test with inputs where the sequence lengths are different from the prepopulated buffer.
        """
        self._prepopulate_buffer()
        pad_token_id = self.trainer.processing_class.pad_token_id
        group_advantages = torch.tensor([[0.6, 0.6], [0.3, 0.45]])  # one no-variance, one variance
        inputs = {
            "group_advantages": group_advantages,
            "prompt_ids": torch.tensor(
                [
                    [1, 2, pad_token_id],
                    [1, 2, pad_token_id],
                    [3, 4, 5],
                    [3, 4, 5],
                ]
            ),
            "prompt_mask": torch.tensor([[1, 1, 0], [1, 1, 0], [1, 1, 1], [1, 1, 1]], dtype=torch.long),
            "completion_ids": torch.tensor(
                [
                    [1009, 1010, pad_token_id],
                    [1011, 1012, 1013],
                    [1013, 1014, pad_token_id],
                    [1015, 1016, 1017],
                ]
            ),
            "completion_mask": torch.tensor([[1, 1, 0], [1, 1, 1], [1, 1, 0], [1, 1, 1]], dtype=torch.long),
            "forward_kwargs": {},
        }
        inputs["group_std_rewards"] = group_advantages.std(dim=1).expand_as(group_advantages)

        outputs_after_sampling = self.trainer.update_with_replay_buffer(**inputs, num_items_in_batch=4)
        # Seq length of current batch should be preserved
        assert outputs_after_sampling["prompt_ids"].shape[-1] == 3
        assert len(self.trainer.replay_buffer.heap) == 3
        output_prompt_ids = outputs_after_sampling["prompt_ids"].view(-1, self.trainer.num_generations, 3).tolist()

        buffered_prompt_completion_ids = [
            (item[1]["prompt_ids"].tolist(), item[1]["completion_ids"].tolist())
            for item in self.trainer.replay_buffer.heap
        ]
        buffered_prompt_ids, buffered_completion_ids = zip(*buffered_prompt_completion_ids, strict=True)

        # Check for new entry with seq len 3 in buffer
        assert [[3, 4, 5], [3, 4, 5]] in buffered_prompt_ids  # excluded no-variance group
        assert [
            [1013, 1014, pad_token_id],
            [1015, 1016, 1017],
        ] in buffered_completion_ids  # excluded no-variance group

        # Check that sampled outputs contain one group with prompt_ids starting with a pad token
        assert [
            [pad_token_id, 101, 102],
            [pad_token_id, 102, 103],
        ] in output_prompt_ids or [
            [pad_token_id, 104, 105],
            [pad_token_id, 106, 107],
        ] in output_prompt_ids


@pytest.mark.low_priority
@pytest.mark.parametrize("scale_rewards", ["batch", "group"])
class TestGRPOWithReplayBufferTrainer(TrlTestCase):
    def test_training_with_replay_buffer(self, scale_rewards):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        # Guarantee that some rewards have 0 std
        def custom_reward_func(completions, **kwargs):
            if torch.rand(1).item() < 0.25:
                return [0] * len(completions)  # simulate some None rewards
            else:
                return torch.rand(len(completions)).tolist()

        training_args = GRPOWithReplayBufferConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=4,  # reduce the batch size to reduce memory usage
            num_generations=4,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            replay_buffer_size=8,
            report_to="none",
            scale_rewards=scale_rewards,
        )
        trainer = GRPOWithReplayBufferTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=[custom_reward_func],
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."
