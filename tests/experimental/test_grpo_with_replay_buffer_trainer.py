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
from datasets import DatasetDict, load_dataset

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
        heap_scores = [entry[0] for entry in self.replay_buffer.heap]
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
        heap_scores = [entry[0] for entry in self.replay_buffer.heap]
        assert heap_scores[0] == min(heap_scores)
        assert heap_scores[0] == 0.5  # 0.3 and 0.4 should be removed

    def test_add_with_equal_scores(self):
        # Equal scores must not crash: heap entries carry a unique tiebreaker, so the data dicts (which do not
        # support comparison) are never compared. Binary rewards make score ties common in practice.
        scores = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        data = [{"id": i} for i in range(6)]
        self.replay_buffer.add(scores, data)

        assert len(self.replay_buffer.heap) == 5

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
            assert item in [entry[2] for entry in self.replay_buffer.heap]

    def test_sample_empty_buffer(self):
        assert self.replay_buffer.sample(num_samples=2) is None


@pytest.mark.low_priority
class TestUpdateWithReplayBuffer(TrlTestCase):
    """Unit tests for the replay machinery, called with the exact tensor layout the trainer produces: flat
    row-major batches of shape (num_groups * num_generations, seq_len), left-padded prompts, right-padded
    completions, and a per-sample (batch_size,) group-std vector."""

    @pytest.fixture(autouse=True)
    def make_trainer(self, set_tmp_dir):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")
        config = GRPOWithReplayBufferConfig(
            output_dir=self.tmp_dir,
            replay_buffer_size=5,
            report_to="none",
        )
        self.trainer = GRPOWithReplayBufferTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=config,
            train_dataset=dataset,
        )
        self.trainer.num_generations = 2
        self.pad_token_id = self.trainer.processing_class.pad_token_id

    def _prepopulate_buffer(self, with_logprobs=False, completion_len=2):
        # Buffer items follow the `_slice_group` format: trimmed tensors, (num_generations,) advantages
        scores = [0.1, 0.9]
        data = [
            {
                "prompt_ids": torch.tensor([[100, 101], [102, 103]]),
                "prompt_mask": torch.ones(2, 2, dtype=torch.long),
                "completion_ids": torch.arange(5, 5 + 2 * completion_len).reshape(2, completion_len),
                "completion_mask": torch.ones(2, completion_len, dtype=torch.long),
                "advantages": torch.tensor([0.5, -0.5]),
                **({"old_per_token_logps": torch.randn(2, completion_len)} if with_logprobs else {}),
            },
            {
                "prompt_ids": torch.tensor([[104, 105], [106, 107]]),
                "prompt_mask": torch.ones(2, 2, dtype=torch.long),
                "completion_ids": torch.arange(13, 13 + 2 * completion_len).reshape(2, completion_len),
                "completion_mask": torch.ones(2, completion_len, dtype=torch.long),
                "advantages": torch.tensor([0.8, -0.8]),
                **({"old_per_token_logps": torch.randn(2, completion_len)} if with_logprobs else {}),
            },
        ]
        self.trainer.replay_buffer.add(scores, data)

    def _make_output(self, advantages, with_logprobs=False):
        output = {
            "prompt_ids": torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]]),
            "prompt_mask": torch.ones(4, 2, dtype=torch.long),
            "completion_ids": torch.tensor([[9, 10], [11, 12], [13, 14], [15, 16]]),
            "completion_mask": torch.ones(4, 2, dtype=torch.long),
            "advantages": advantages,
        }
        if with_logprobs:
            output["old_per_token_logps"] = torch.randn(4, 2)
        return output

    def test_no_variance_groups_replaced_from_buffer(self):
        self._prepopulate_buffer(with_logprobs=True)
        # Two zero-variance groups: both must be replaced with buffered groups
        output = self._make_output(advantages=torch.zeros(4), with_logprobs=True)
        original_prompt_ids = output["prompt_ids"].clone()
        group_std_rewards = torch.zeros(4)

        output = self.trainer.update_with_replay_buffer(output, group_std_rewards)

        assert len(self.trainer.replay_buffer.heap) == 2  # nothing buffered (no variance)
        assert "old_per_token_logps" in output
        for pid in output["prompt_ids"]:
            assert pid.tolist() not in original_prompt_ids.tolist()
        # Replayed advantages must cover every row of the replaced groups (not just the first one)
        assert {round(a, 4) for a in output["advantages"].abs().tolist()} <= {0.5, 0.8}

    def test_variance_groups_buffered_not_replaced(self):
        self._prepopulate_buffer()
        output = self._make_output(advantages=torch.tensor([0.6, -0.6, 0.7, -0.7]))
        original = {k: v.clone() for k, v in output.items()}
        group_std_rewards = torch.tensor([0.5, 0.5, 0.7, 0.7])

        output = self.trainer.update_with_replay_buffer(output, group_std_rewards)

        assert len(self.trainer.replay_buffer.heap) == 4  # both groups buffered
        for key, value in original.items():
            assert torch.equal(output[key], value)  # nothing replaced

    def test_mixed_variance(self):
        self._prepopulate_buffer()
        # Group 0 has variance (buffered, kept), group 1 has none (replaced)
        output = self._make_output(advantages=torch.tensor([0.6, -0.6, 0.0, 0.0]))
        original_prompt_ids = output["prompt_ids"].clone()
        group_std_rewards = torch.tensor([0.5, 0.5, 0.0, 0.0])

        output = self.trainer.update_with_replay_buffer(output, group_std_rewards)

        assert len(self.trainer.replay_buffer.heap) == 3  # grew by one
        assert torch.equal(output["prompt_ids"][:2], original_prompt_ids[:2])  # variance group kept
        assert output["prompt_ids"][2:].tolist() != original_prompt_ids[2:].tolist()  # no-variance group replaced
        buffered_prompt_ids = [entry[2]["prompt_ids"].tolist() for entry in self.trainer.replay_buffer.heap]
        assert output["prompt_ids"][2:].tolist() in buffered_prompt_ids

    def test_empty_buffer_leaves_batch_unchanged(self):
        # Zero-variance groups with an empty buffer: nothing to replay, and it must not crash
        output = self._make_output(advantages=torch.zeros(4))
        original = {k: v.clone() for k, v in output.items()}
        group_std_rewards = torch.zeros(4)

        output = self.trainer.update_with_replay_buffer(output, group_std_rewards)

        assert len(self.trainer.replay_buffer.heap) == 0
        for key, value in original.items():
            assert torch.equal(output[key], value)

    def test_left_padded_prompts_are_trimmed_from_the_left(self):
        # Regression test: prompts are left-padded, so trimming a buffered group must keep the *last* columns.
        # Group 0 (variance) has a 2-token prompt inside a width-3 batch: [pad, 3, 4]
        output = {
            "prompt_ids": torch.tensor([[self.pad_token_id, 3, 4], [self.pad_token_id, 3, 4], [5, 6, 7], [5, 6, 7]]),
            "prompt_mask": torch.tensor([[0, 1, 1], [0, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=torch.long),
            "completion_ids": torch.tensor([[9, 10], [11, 12], [13, 14], [15, 16]]),
            "completion_mask": torch.ones(4, 2, dtype=torch.long),
            "advantages": torch.tensor([0.6, -0.6, 0.7, -0.7]),
        }
        group_std_rewards = torch.tensor([0.5, 0.5, 0.7, 0.7])

        self.trainer.update_with_replay_buffer(output, group_std_rewards)

        buffered = [entry[2] for entry in self.trainer.replay_buffer.heap]
        short_prompt = next(item for item in buffered if item["prompt_ids"].size(1) == 2)
        # The end of the prompt must be preserved, not the padding
        assert short_prompt["prompt_ids"].tolist() == [[3, 4], [3, 4]]
        assert short_prompt["prompt_mask"].tolist() == [[1, 1], [1, 1]]

    def test_longer_buffered_completions_widen_the_batch(self):
        # A buffered completion longer than the current batch must widen the batch tensors (including logps)
        # rather than crash or truncate
        self._prepopulate_buffer(with_logprobs=True, completion_len=4)
        output = self._make_output(advantages=torch.zeros(4), with_logprobs=True)
        group_std_rewards = torch.zeros(4)

        output = self.trainer.update_with_replay_buffer(output, group_std_rewards)

        assert output["completion_ids"].size(1) == 4
        assert output["completion_mask"].size(1) == 4
        assert output["old_per_token_logps"].size(1) == 4

    def test_importance_sampling_ratio_roundtrip(self):
        # The vLLM IS ratio must be buffered per group and restored on replay, re-padded to the batch width
        self.trainer.vllm_importance_sampling_mode = "token_truncate"
        self.trainer.replay_buffer.add(
            [0.9],
            [
                {
                    "prompt_ids": torch.tensor([[100, 101], [102, 103]]),
                    "prompt_mask": torch.ones(2, 2, dtype=torch.long),
                    "completion_ids": torch.tensor([[5, 6, 7], [8, 9, 10]]),
                    "completion_mask": torch.ones(2, 3, dtype=torch.long),
                    "advantages": torch.tensor([0.5, -0.5]),
                    "importance_sampling_ratio": torch.full((2, 3), 0.7),
                }
            ],
        )
        output = {
            "prompt_ids": torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]]),
            "prompt_mask": torch.ones(4, 2, dtype=torch.long),
            "completion_ids": torch.tensor([[9, 10], [11, 12], [13, 14], [15, 16]]),
            "completion_mask": torch.ones(4, 2, dtype=torch.long),
            "advantages": torch.tensor([0.6, -0.6, 0.0, 0.0]),
            "importance_sampling_ratio": torch.ones(4, 2),
        }
        group_std_rewards = torch.tensor([0.5, 0.5, 0.0, 0.0])

        output = self.trainer.update_with_replay_buffer(output, group_std_rewards)

        # Batch widened to the buffered completion length; replaced rows carry the buffered ratio
        assert output["importance_sampling_ratio"].shape == (4, 3)
        assert torch.allclose(output["importance_sampling_ratio"][2:], torch.full((2, 3), 0.7))
        # Non-replaced rows keep their own ratio, padded with the neutral value 1.0
        assert torch.allclose(output["importance_sampling_ratio"][:2], torch.ones(2, 3))

    def test_token_level_is_ratio_widened_even_when_batch_width_is_one(self):
        # Regression test: a token-level IS ratio is also (B, 1) when every completion in the batch is a single
        # token (a degenerate batch, which is exactly when replay tends to trigger). Sequence-level handling must
        # be decided by the configured mode, not the tensor width, so the ratio is widened with the completions.
        self.trainer.vllm_importance_sampling_mode = "token_truncate"
        self.trainer.replay_buffer.add(
            [0.9],
            [
                {
                    "prompt_ids": torch.tensor([[100, 101], [102, 103]]),
                    "prompt_mask": torch.ones(2, 2, dtype=torch.long),
                    "completion_ids": torch.tensor([[5, 6, 7], [8, 9, 10]]),
                    "completion_mask": torch.ones(2, 3, dtype=torch.long),
                    "advantages": torch.tensor([0.5, -0.5]),
                    "importance_sampling_ratio": torch.full((2, 3), 0.7),
                }
            ],
        )
        # Degenerate batch: every completion is one token, so the token-level ratio is (4, 1)
        output = {
            "prompt_ids": torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]]),
            "prompt_mask": torch.ones(4, 2, dtype=torch.long),
            "completion_ids": torch.tensor([[9], [11], [13], [15]]),
            "completion_mask": torch.ones(4, 1, dtype=torch.long),
            "advantages": torch.zeros(4),
            "importance_sampling_ratio": torch.full((4, 1), 0.9),
        }
        group_std_rewards = torch.zeros(4)

        output = self.trainer.update_with_replay_buffer(output, group_std_rewards)

        # Both zero-variance groups replaced with the buffered one; ratio widened alongside the completions
        assert output["completion_ids"].shape == (4, 3)
        assert output["importance_sampling_ratio"].shape == (4, 3)
        assert torch.allclose(output["importance_sampling_ratio"], torch.full((4, 3), 0.7))

    def test_sequence_level_is_ratio_stays_narrow(self):
        # Sequence-level modes carry one ratio per sequence; it must stay (B, 1) while completions widen
        self.trainer.vllm_importance_sampling_mode = "sequence_truncate"
        self.trainer.replay_buffer.add(
            [0.9],
            [
                {
                    "prompt_ids": torch.tensor([[100, 101], [102, 103]]),
                    "prompt_mask": torch.ones(2, 2, dtype=torch.long),
                    "completion_ids": torch.tensor([[5, 6, 7], [8, 9, 10]]),
                    "completion_mask": torch.ones(2, 3, dtype=torch.long),
                    "advantages": torch.tensor([0.5, -0.5]),
                    "importance_sampling_ratio": torch.full((2, 1), 0.7),
                }
            ],
        )
        output = {
            "prompt_ids": torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]]),
            "prompt_mask": torch.ones(4, 2, dtype=torch.long),
            "completion_ids": torch.tensor([[9, 10], [11, 12], [13, 14], [15, 16]]),
            "completion_mask": torch.ones(4, 2, dtype=torch.long),
            "advantages": torch.zeros(4),
            "importance_sampling_ratio": torch.full((4, 1), 0.9),
        }
        group_std_rewards = torch.zeros(4)

        output = self.trainer.update_with_replay_buffer(output, group_std_rewards)

        assert output["completion_ids"].shape == (4, 3)  # widened to the buffered length
        assert output["importance_sampling_ratio"].shape == (4, 1)  # stays sequence-level
        assert torch.allclose(output["importance_sampling_ratio"], torch.full((4, 1), 0.7))

    def test_multimodal_batches_pass_through(self):
        # The replay machinery cannot slice vision tensors; multimodal batches must pass through unchanged
        self._prepopulate_buffer()
        output = self._make_output(advantages=torch.zeros(4))
        output["pixel_values"] = torch.randn(4, 3, 8, 8)
        original_prompt_ids = output["prompt_ids"].clone()
        group_std_rewards = torch.zeros(4)

        output = self.trainer.update_with_replay_buffer(output, group_std_rewards)

        assert torch.equal(output["prompt_ids"], original_prompt_ids)  # no replacement
        assert len(self.trainer.replay_buffer.heap) == 2  # no buffering either


@pytest.mark.low_priority
@pytest.mark.parametrize("scale_rewards", ["batch", "group"])
class TestGRPOWithReplayBufferTrainer(TrlTestCase):
    def test_train_with_replay_buffer(self, scale_rewards):
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
        # The replay hook runs through the real trainer path, so groups with variance must have been buffered
        assert len(trainer.replay_buffer.heap) > 0

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_train_with_replay_buffer_disabled(self, scale_rewards):
        # replay_buffer_size=0 disables the buffer entirely; training must work as plain GRPO
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = GRPOWithReplayBufferConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=3,
            num_generations=3,
            max_completion_length=8,
            max_steps=2,
            replay_buffer_size=0,
            report_to="none",
            scale_rewards=scale_rewards,
        )
        trainer = GRPOWithReplayBufferTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset,
        )

        assert trainer.replay_buffer is None

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

    @pytest.mark.parametrize("eval_dataset_type", ["dataset", "dataset_dict", "dict_of_dataset", "none"])
    def test_init_with_eval_dataset(self, scale_rewards, eval_dataset_type):
        # Streaming datasets are not yet supported in GRPO with replay buffer
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only")

        if eval_dataset_type == "none":
            eval_dataset = None
        elif eval_dataset_type == "dataset":
            eval_dataset = dataset["test"]
        elif eval_dataset_type == "dataset_dict":
            eval_dataset = DatasetDict({"data1": dataset["test"], "data2": dataset["test"]})
        else:  # "dict_of_dataset"
            eval_dataset = {"data1": dataset["test"], "data2": dataset["test"]}

        training_args = GRPOWithReplayBufferConfig(
            output_dir=self.tmp_dir, scale_rewards=scale_rewards, report_to="none"
        )
        trainer = GRPOWithReplayBufferTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=eval_dataset,
        )

        if eval_dataset_type == "none":
            assert trainer.eval_dataset is None
        elif isinstance(trainer.eval_dataset, dict):
            assert set(trainer.eval_dataset.keys()) == {"data1", "data2"}
        else:
            assert trainer.eval_dataset is eval_dataset
