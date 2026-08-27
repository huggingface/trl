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

from trl.experimental.zero_sync_grpo import ZeroSyncGRPOConfig, ZeroSyncGRPOTrainer
from trl.experimental.zero_sync_grpo.zero_sync_grpo_trainer import TurnRecord, _chain_to_sequences

from ..testing_utils import TrlTestCase, require_torch_accelerator


def dummy_reward_func(completions, **kwargs):
    return [float(len(completion[0]["content"])) for completion in completions]


class TestChainToSequences(TrlTestCase):
    """Reconciliation of a multi-turn conversation into training rows."""

    def test_clean_append_stays_one_row(self):
        # The second turn's prompt starts with everything held so far: the new part is context.
        turns = [TurnRecord([1, 2], [3, 4], [-0.1, -0.2]), TurnRecord([1, 2, 3, 4, 5], [6], [-0.3])]
        rows, forks = _chain_to_sequences(turns)
        assert forks == 0
        assert len(rows) == 1
        assert rows[0]["input_ids"] == [1, 2, 3, 4, 5, 6]
        assert rows[0]["completion_mask"] == [0, 0, 1, 1, 0, 1]
        assert rows[0]["logprobs"] == [0.0, 0.0, -0.1, -0.2, 0.0, -0.3]

    def test_single_token_drift_forks(self):
        # The template re-tokenized one token of the last answer: that answer keeps its own row.
        turns = [TurnRecord([1, 2], [3, 4], [-0.1, -0.2]), TurnRecord([1, 2, 3, 9, 5], [6], [-0.3])]
        rows, forks = _chain_to_sequences(turns)
        assert forks == 1
        assert [row["input_ids"] for row in rows] == [[1, 2, 3, 4], [1, 2, 3, 9, 5, 6]]
        assert rows[1]["completion_mask"] == [0, 0, 0, 0, 0, 1]

    def test_drift_before_the_last_answer_forks(self):
        # The template rewrote the prompt itself: the trained tokens of turn 1 keep their own row.
        turns = [TurnRecord([1, 2], [3, 4], [-0.1, -0.2]), TurnRecord([7, 7, 7], [8], [-0.4])]
        rows, forks = _chain_to_sequences(turns)
        assert forks == 1
        assert [row["input_ids"] for row in rows] == [[1, 2, 3, 4], [7, 7, 7, 8]]

    def test_row_without_trained_token_is_dropped(self):
        rows, _ = _chain_to_sequences([TurnRecord([1, 2], [], [])])
        assert rows == []


class TestZeroSyncGRPOConfig(TrlTestCase):
    def test_batch_size_must_be_divisible_by_num_generations(self):
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_only", split="train")
        args = ZeroSyncGRPOConfig(output_dir=self.tmp_dir, per_device_train_batch_size=3, num_generations=2)
        with pytest.raises(ValueError, match="must be evenly divisible"):
            ZeroSyncGRPOTrainer(
                model="trl-internal-testing/small-Qwen2ForCausalLM-2.5",
                reward_funcs=dummy_reward_func,
                args=args,
                train_dataset=dataset,
            )


@require_torch_accelerator
class TestZeroSyncGRPOTrainer(TrlTestCase):
    def test_train(self):
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_only", split="train")
        training_args = ZeroSyncGRPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            generation_ahead=1,  # keep few requests in flight, the test model is tiny
            max_steps=2,
            report_to="none",
        )
        trainer = ZeroSyncGRPOTrainer(
            model="trl-internal-testing/small-Qwen2ForCausalLM-2.5",
            reward_funcs=dummy_reward_func,
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
