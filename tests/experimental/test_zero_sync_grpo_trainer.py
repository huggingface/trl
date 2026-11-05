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

import os

import pytest
import torch
from datasets import load_dataset
from torch.distributed.tensor import DTensor

from trl.experimental.zero_sync_grpo import ZeroSyncGRPOConfig, ZeroSyncGRPOTrainer
from trl.experimental.zero_sync_grpo.zero_sync_grpo_trainer import (
    DriftKind,
    TurnRecord,
    _chain_to_sequences,
    _common_prefix_len,
    _SampleBuilder,
)

from ..testing_utils import TrlTestCase, require_torch_accelerator


def dummy_reward_func(completions, **kwargs):
    return [float(len(completion[0]["content"])) for completion in completions]


def _finalize(turns, rollout_id="r0", fork_threshold=1024):
    rows, _tally = _chain_to_sequences(turns, rollout_id, fork_threshold)
    return rows


class TestReconciler(TrlTestCase):
    @pytest.mark.parametrize("mask", [None, [1, 0, 1], [0, 0, 0]])
    def test_completion_masks_are_generic(self, mask):
        rows = _finalize([TurnRecord([1, 2], [3, 4, 5], [-0.1, -0.2, -0.3], mask)])
        if mask == [0, 0, 0]:
            assert rows == []
        else:
            assert rows[0].completion_mask == [0, 0] + (mask if mask is not None else [1, 1, 1])
            assert rows[0].input_ids == [1, 2, 3, 4, 5]
            assert rows[0].old_log_probs[-3:] == [-0.1, -0.2, -0.3]

    def test_common_prefix_len(self):
        assert _common_prefix_len([1, 2, 3], [1, 2, 3]) == 3  # identical
        assert _common_prefix_len([1, 2], [1, 2, 3, 4]) == 2  # old is a prefix of new
        assert _common_prefix_len([], [1, 2, 3]) == 0  # empty
        # chunk=2 exercises both the whole-chunk fast path and the per-token fallback
        assert _common_prefix_len([1, 2, 3, 4, 5], [1, 2, 3, 9, 5], chunk=2) == 3
        assert _common_prefix_len([1, 2, 3, 4], [1, 2, 3, 4, 5], chunk=2) == 4

    def test_single_turn(self):
        rows = _finalize([TurnRecord([1, 2, 3], [10, 11], [-0.1, -0.2])])
        assert len(rows) == 1
        assert rows[0].input_ids == [1, 2, 3, 10, 11]
        assert rows[0].completion_mask == [0, 0, 0, 1, 1]
        assert rows[0].old_log_probs == [0.0, 0.0, 0.0, -0.1, -0.2]
        assert rows[0].sampling_mask == [[], [], [], [], []]
        assert rows[0].rollout_id == "r0"

    def test_sampling_mask_follows_the_sampled_tokens(self):
        # Candidate sets ride with the sampled tokens; the prompt and a realigned tail are context and carry none.
        turn1 = TurnRecord([1, 2, 3], [10, 11], [-0.1, -0.2], output_sampling_mask=[[10, 12], [11]])
        turn2 = TurnRecord([1, 2, 3, 10, 99, 20], [30], [-0.3], output_sampling_mask=[[30, 31, 32]])
        rows = _finalize([turn1, turn2])
        assert len(rows) == 1
        assert rows[0].input_ids == [1, 2, 3, 10, 99, 20, 30]
        assert rows[0].completion_mask == [0, 0, 0, 1, 0, 0, 1]
        assert rows[0].sampling_mask == [[], [], [], [10, 12], [], [], [30, 31, 32]]

    def test_clean_chain_stays_one_row(self):
        # Turn 2's re-tokenized prompt starts with the held tokens (gen [10,11] + tool [20,21]) -> CLEAN.
        turn1 = TurnRecord([1, 2, 3], [10, 11], [-0.1, -0.2])
        turn2 = TurnRecord([1, 2, 3, 10, 11, 20, 21], [30, 31], [-0.3, -0.4])
        rows = _finalize([turn1, turn2])
        assert len(rows) == 1
        assert rows[0].input_ids == [1, 2, 3, 10, 11, 20, 21, 30, 31]
        assert rows[0].completion_mask == [0, 0, 0, 1, 1, 0, 0, 1, 1]  # prompt=0 gen=1 tool=0 gen=1
        assert rows[0].old_log_probs == [0.0, 0.0, 0.0, -0.1, -0.2, 0.0, 0.0, -0.3, -0.4]

    def test_rewrite_forks_into_two_rows(self):
        # Divergence inside turn 1's answer + a turn >= fork_threshold -> FORK. Every generated token is
        # trained in exactly one row (turn 1's tokens are context in row 2).
        turn1 = TurnRecord([1, 2, 3], [10, 11, 12, 13], [-0.2] * 4)
        turn2 = TurnRecord([1, 2, 3, 10, 99, 88, 77], [30, 31, 32], [-0.2] * 3)
        rows = _finalize([turn1, turn2], fork_threshold=2)
        assert len(rows) == 2
        assert rows[0].input_ids == [1, 2, 3, 10, 11, 12, 13]
        assert rows[0].completion_mask == [0, 0, 0, 1, 1, 1, 1]
        assert rows[1].input_ids == [1, 2, 3, 10, 99, 88, 77, 30, 31, 32]
        assert rows[1].completion_mask == [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]

    def test_fork_when_divergence_precedes_last_response(self):
        # matched < last_response_start_idx -> FORK regardless of threshold (distinct from the length trigger).
        builder = _SampleBuilder(fork_threshold=1024)
        builder.append_turn(
            TurnRecord([1, 2, 3], [10, 11], [-0.2] * 2), DriftKind.CLEAN
        )  # last_response_start_idx == 3
        # 5 held tokens, only the first matches -> 4 tokens of drift, reported alongside the kind.
        assert builder.classify_token_drift(TurnRecord([1, 9, 3, 10, 11], [30], [-0.2] * 1)) == (DriftKind.FORK, 4)

    def test_drift_tally_counts_transitions(self):
        # The tally is what makes `fork_threshold_tokens` tunable: it reports the drift the threshold is compared
        # against. It counts turn TRANSITIONS, so a single-turn rollout has none.
        turn1 = TurnRecord([1, 2, 3], [10, 11, 12, 13], [-0.2] * 4)
        clean = TurnRecord([1, 2, 3, 10, 11, 12, 13], [20], [-0.2] * 1)
        forked = TurnRecord([1, 2, 3, 10, 99, 88, 77, 20], [30], [-0.2] * 1)

        _rows, tally = _chain_to_sequences([turn1], "r0", 2)
        assert tally == {"clean": 0, "realign": 0, "fork": 0, "transitions": 0, "drift_tokens": 0, "drift_max": 0}

        _rows, tally = _chain_to_sequences([turn1, clean, forked], "r0", 2)
        assert (tally["transitions"], tally["clean"], tally["fork"]) == (2, 1, 1)
        assert tally["drift_max"] == 4  # the forking transition invalidated 4 held tokens

    def test_tail_wobble_realigns_to_context(self):
        # Last generated token re-renders (11 -> 12): a short wobble in the last answer -> REALIGN. Only the drifted
        # token is masked to context; the matched part of the answer (10) keeps its training signal and logprob.
        turn1 = TurnRecord([1, 2, 3], [10, 11], [-0.1, -0.2])
        turn2 = TurnRecord([1, 2, 3, 10, 12], [30, 31], [-0.3, -0.4])
        rows = _finalize([turn1, turn2])
        assert len(rows) == 1
        assert rows[0].input_ids == [1, 2, 3, 10, 12, 30, 31]
        assert rows[0].completion_mask == [0, 0, 0, 1, 0, 1, 1]
        assert rows[0].old_log_probs == [0.0, 0.0, 0.0, -0.1, 0.0, -0.3, -0.4]

    def test_large_drift_forks_even_when_next_turn_is_short(self):
        # The Qwen3 case: turn 1 is a long trained answer (a reasoning block); the template drops it on turn 2, so
        # the prompt diverges at the answer start (large drift) while turn 2's own answer is tiny. The decision must
        # key on the drift (6 tokens invalidated), not the incoming turn's length (1) -> FORK, so turn 1 keeps its
        # training signal. (fork_threshold sits between the two: gating on output length would wrongly REALIGN.)
        turn1 = TurnRecord([1, 2, 3], [10, 11, 12, 13, 14, 15], [-0.2] * 6)  # long trained answer
        turn2 = TurnRecord([1, 2, 3, 99, 88], [30], [-0.2] * 1)  # answer dropped by the template; short next turn
        rows = _finalize([turn1, turn2], fork_threshold=3)
        assert len(rows) == 2
        assert rows[0].input_ids == [1, 2, 3, 10, 11, 12, 13, 14, 15]
        assert rows[0].completion_mask == [0, 0, 0, 1, 1, 1, 1, 1, 1]  # long answer stays trained
        assert rows[1].input_ids == [1, 2, 3, 99, 88, 30]
        assert rows[1].completion_mask == [0, 0, 0, 0, 0, 1]


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
            rollouts_in_flight=3,  # keep few rollouts generating, the test model is tiny
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

    def test_train_bf16_steps_fp32_copies(self):
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_only", split="train")
        training_args = ZeroSyncGRPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=1e-6,  # a step bf16 rounds away on most weights
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            rollouts_in_flight=3,  # keep few rollouts generating, the test model is tiny
            max_steps=2,
            model_init_kwargs={"dtype": "bfloat16"},
            report_to="none",
        )
        trainer = ZeroSyncGRPOTrainer(
            model="trl-internal-testing/small-Qwen2ForCausalLM-2.5",
            reward_funcs=dummy_reward_func,
            args=training_args,
            train_dataset=dataset,
        )

        trainer.train()

        # The optimizer steps fp32 copies of the bf16 parameters, which keep what bf16 cannot represent, and keeps them
        # in host memory between steps
        for group in trainer.optimizer.param_groups:
            for master in group["params"]:
                assert master.dtype == torch.float32
                assert master.is_pinned()
                assert not torch.equal(master, master.bfloat16().float())

    @pytest.mark.skipif("TRL_TEST_ZERO_SYNC_TP_OUTPUT_DIR" not in os.environ, reason="launched by tests/distributed")
    def test_train_tensor_parallel_body(self):
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_only", split="train")
        training_args = ZeroSyncGRPOConfig(
            output_dir=os.environ["TRL_TEST_ZERO_SYNC_TP_OUTPUT_DIR"],
            learning_rate=1e-6,  # a step bf16 rounds away on most weights
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            rollouts_in_flight=3,  # keep few rollouts generating, the test model is tiny
            max_steps=2,
            model_init_kwargs={"dtype": "bfloat16"},
            tp_size=int(os.environ["WORLD_SIZE"]),
            report_to="none",
        )
        trainer = ZeroSyncGRPOTrainer(
            model="trl-internal-testing/small-Qwen2ForCausalLM-2.5",
            reward_funcs=dummy_reward_func,
            args=training_args,
            train_dataset=dataset,
        )

        trainer.train()

        # Each rank's shard of the fp32 copies goes back to host memory after the step
        for group in trainer.optimizer.param_groups:
            for master in group["params"]:
                local = master.to_local() if isinstance(master, DTensor) else master
                assert local.dtype == torch.float32
                assert local.is_pinned()
