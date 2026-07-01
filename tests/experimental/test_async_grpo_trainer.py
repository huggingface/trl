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

import itertools
import json
import multiprocessing as mp
import os
import queue
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from accelerate import PartialState
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
from transformers.testing_utils import torch_device

from trl.experimental.async_grpo import AsyncGRPOConfig, AsyncGRPOTrainer
from trl.experimental.async_grpo.async_grpo_trainer import (
    DataCollatorForRollout,
    FixedCountBatcher,
    TokenBudgetBatcher,
    _balance_by_squared_length,
    _SaveRolloutStateCallback,
)
from trl.experimental.async_grpo.async_rollout_worker import AsyncRolloutWorker, RolloutSample, _AsyncRolloutLoop

from ..testing_utils import TrlTestCase, is_ampere_or_newer


def dummy_reward_func(completions, **kwargs):
    return [float(hash(c[0]["content"]) % 100) / 100.0 for c in completions]


class _StubRolloutWorker:
    """Minimal rollout worker stub for testing the trainer in isolation."""

    def __init__(self, tokenizer, dataset, num_generations: int = 8, samples_per_weight_sync: int = 10):
        self.rollout_buffer = queue.Queue()
        self._samples_per_weight_sync = samples_per_weight_sync
        self._model_version = 0
        self._sample_iter = self._make_sample_iter(tokenizer, dataset, num_generations)

    def _make_sample_iter(self, tokenizer, dataset, num_generations):
        for row in itertools.cycle(dataset):
            completions = [
                [{"role": "assistant", "content": f"{row['completion'][0]['content']} {idx}"}]
                for idx in range(num_generations)
            ]
            prompt_completions = [row["prompt"] + completion for completion in completions]
            prompt_ids = tokenizer.apply_chat_template(
                row["prompt"], tokenize=True, add_generation_prompt=True, return_dict=False
            )
            prompt_completion_ids = tokenizer.apply_chat_template(
                prompt_completions, tokenize=True, add_generation_prompt=False, return_dict=False
            )
            rewards = np.array(dummy_reward_func(completions))
            advantages = (rewards - rewards.mean()) / rewards.std()
            for idx in range(num_generations):
                completion_ids = prompt_completion_ids[idx][len(prompt_ids) :]
                yield RolloutSample(
                    prompt=row["prompt"],
                    completion=completions[idx],
                    input_ids=prompt_ids + completion_ids,
                    completion_mask=[0] * len(prompt_ids) + [1] * len(completion_ids),
                    old_log_probs=[0.0] * len(prompt_ids) + [-0.5] * len(completion_ids),
                    advantage=float(advantages[idx]),
                    model_version=self._model_version,
                    metrics={"reward": float(rewards[idx]), "reward_std": float(rewards.std())},
                )

    def _fill_queue(self):
        for _ in range(self._samples_per_weight_sync):
            self.rollout_buffer.put(next(self._sample_iter))

    def start(self):
        self._fill_queue()

    def update_model_version(self, version):
        self._model_version = version
        self._fill_queue()

    def stop(self):
        pass

    def check_health(self, stale_after_s):
        pass


@pytest.mark.skipif(
    not is_ampere_or_newer() and torch_device != "xpu",
    reason="Flash Attention 2 requires Ampere or newer GPU, or XPU",
)
class TestAsyncGRPOTrainer(TrlTestCase):
    def test_init_minimal(self):
        # Test that AsyncGRPOTrainer can be instantiated with only model, reward_model and train_dataset
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_completion", split="train")
        AsyncGRPOTrainer(
            model=model_id,
            reward_funcs=dummy_reward_func,
            train_dataset=dataset,
            rollout_worker=_StubRolloutWorker(AutoTokenizer.from_pretrained(model_id), dataset, num_generations=3),
        )

    def test_train(self):
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_completion", split="train")

        training_args = AsyncGRPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            vllm_server_timeout=5.0,  # short timeout so test fails fast if queue runs dry
            report_to="none",
        )
        trainer = AsyncGRPOTrainer(
            model=model_id,
            reward_funcs=dummy_reward_func,  # unused: the stub pre-computes rewards, but the trainer requires this argument
            args=training_args,
            train_dataset=dataset,
            rollout_worker=_StubRolloutWorker(AutoTokenizer.from_pretrained(model_id), dataset, num_generations=3),
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_resume_from_checkpoint(self):
        # ignore_data_skip must always be True for AsyncGRPO, this is because the base Trainer's skip-and-replay
        # loop doesn't apply to a live rollout queue and would trigger unnecessary vLLM inference.
        # This test also verifies that training resumes from a checkpoint without errors.
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_completion", split="train")
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        training_args = AsyncGRPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=3,
            num_generations=3,
            max_steps=2,
            save_steps=1,
            max_completion_length=8,
            vllm_server_timeout=5.0,
            report_to="none",
        )

        # First run: train for 2 steps, which saves a checkpoint at step 1.
        trainer = AsyncGRPOTrainer(
            model=model_id,
            reward_funcs=dummy_reward_func,
            args=training_args,
            train_dataset=dataset,
            rollout_worker=_StubRolloutWorker(tokenizer, dataset, num_generations=3),
        )
        assert trainer.args.ignore_data_skip is True
        trainer.train()

        # Second run: resume from the step-1 checkpoint.
        checkpoint_dir = os.path.join(self.tmp_dir, "checkpoint-1")
        assert os.path.isfile(os.path.join(checkpoint_dir, "trainer_state.json"))

        training_args2 = AsyncGRPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=3,
            num_generations=3,
            max_steps=3,
            max_completion_length=8,
            vllm_server_timeout=5.0,
            report_to="none",
        )
        trainer2 = AsyncGRPOTrainer(
            model=model_id,
            reward_funcs=dummy_reward_func,
            args=training_args2,
            train_dataset=dataset,
            rollout_worker=_StubRolloutWorker(tokenizer, dataset, num_generations=3),
        )
        assert trainer2.args.ignore_data_skip is True
        trainer2.train(resume_from_checkpoint=checkpoint_dir)


class TestRolloutStateCheckpoint(TrlTestCase):
    """Prompt-index checkpoint/resume logic — no GPU or vLLM required."""

    def _make_rollout_loop(self, dataset, dataset_start_index=0, num_generations=2, prompt_index_value=None):
        ctx = mp.get_context("spawn")
        kwargs = dict(
            model_name="test",
            dataset=dataset,
            reward_funcs=[],
            processing_class=MagicMock(),
            rollout_buffer=ctx.Queue(),
            model_version_value=ctx.Value("i", 0),
            heartbeat_value=ctx.Value("d", 0.0),
            failed_event=ctx.Event(),
            exception_info_queue=ctx.Queue(),
            num_generations=num_generations,
            dataset_start_index=dataset_start_index,
            prompt_index_value=prompt_index_value,
        )
        with patch("trl.experimental.async_grpo.async_rollout_worker.add_response_schema", side_effect=lambda x: x):
            return _AsyncRolloutLoop(**kwargs)

    def test_save_rollout_state_callback_writes_json(self):
        class _MockWorker(AsyncRolloutWorker):
            # isinstance check requires AsyncRolloutWorker. skip __init__ (needs vLLM)
            def __init__(self, index):
                self._index = index

            @property
            def prompt_index(self):
                return self._index

        checkpoint_dir = os.path.join(self.tmp_dir, "checkpoint-5")
        os.makedirs(checkpoint_dir)

        trainer = MagicMock()
        trainer.accelerator.is_main_process = True
        trainer.rollout_worker = _MockWorker(42)

        args = MagicMock()
        args.output_dir = self.tmp_dir
        state = MagicMock()
        state.global_step = 5

        _SaveRolloutStateCallback(trainer).on_save(args, state, None)

        with open(os.path.join(checkpoint_dir, "rollout_state.json")) as f:
            data = json.load(f)
        assert data["prompt_index"] == 42

    def test_rollout_loop_skips_to_start_index(self):
        dataset = Dataset.from_dict({"prompt": [f"row_{i}" for i in range(10)]})
        ctx = mp.get_context("spawn")
        prompt_index_value = ctx.Value("i", 0)

        loop = self._make_rollout_loop(dataset, dataset_start_index=3, prompt_index_value=prompt_index_value)

        assert prompt_index_value.value == 3

        it = loop._repeat_iterator()
        _group_id, row = next(it)
        assert row["prompt"] == "row_3"

    def test_rollout_loop_tracks_prompt_index_value(self):
        dataset = Dataset.from_dict({"prompt": [f"row_{i}" for i in range(10)]})
        ctx = mp.get_context("spawn")
        prompt_index_value = ctx.Value("i", 0)

        loop = self._make_rollout_loop(
            dataset, dataset_start_index=0, num_generations=2, prompt_index_value=prompt_index_value
        )

        it = loop._repeat_iterator()

        # row_0 is yielded twice (num_generations=2) but the counter only ticks once per row
        next(it)
        assert prompt_index_value.value == 1
        next(it)
        assert prompt_index_value.value == 1

        # row_1 is now pulled
        next(it)
        assert prompt_index_value.value == 2

    def test_inner_training_loop_sets_dataset_start_index_from_file(self):
        class _MockWorker(AsyncRolloutWorker):
            # isinstance check requires AsyncRolloutWorker. skip __init__ (needs vLLM)
            def __init__(self):
                self._loop_kwargs = {}

        checkpoint_dir = os.path.join(self.tmp_dir, "checkpoint-10")
        os.makedirs(checkpoint_dir)
        with open(os.path.join(checkpoint_dir, "rollout_state.json"), "w") as f:
            json.dump({"prompt_index": 77}, f)

        # __new__ skips __init__ (requires GPU + model)
        trainer = AsyncGRPOTrainer.__new__(AsyncGRPOTrainer)
        trainer.rollout_worker = _MockWorker()
        trainer.train_dataset = Dataset.from_dict({"prompt": list(range(100))})
        trainer.accelerator = MagicMock()
        trainer.accelerator.is_main_process = False  # skip finally-block teardown

        from trl.trainer.base_trainer import _BaseTrainer

        with patch.object(_BaseTrainer, "_inner_training_loop", return_value=None):
            trainer._inner_training_loop(resume_from_checkpoint=checkpoint_dir)

        assert trainer.rollout_worker._loop_kwargs["dataset_start_index"] == 77


def _rollout_sample(length: int, advantage: float = 0.0, reward: float = 0.0) -> dict:
    # First token is a prompt token (completion_mask 0); the rest are completion tokens.
    return {
        "input_ids": list(range(length)),
        "completion_mask": [0] + [1] * (length - 1),
        "old_log_probs": [0.0] * length,
        "advantage": advantage,
        "metrics": {"reward": reward},
    }


def _squared_load(group: list[dict]) -> int:
    return sum(len(sample["input_ids"]) ** 2 for sample in group)


def _row_tokens(group: list[dict]) -> int:
    return sum(len(sample["input_ids"]) for sample in group)


class TestPackingAwareBatching(TrlTestCase):
    """Packing-aware dynamic batching is pure scheduling/tensorization, so these run without a GPU."""

    def test_balance_partitions_all_samples_into_non_empty_rows(self):
        samples = [_rollout_sample(length) for length in (5, 1, 1, 1, 1)]
        groups = _balance_by_squared_length(samples, num_groups=2)

        assert len(groups) == 2
        assert all(len(group) > 0 for group in groups)
        # Every sample is placed exactly once.
        flat = [sample for group in groups for sample in group]
        assert sorted(len(s["input_ids"]) for s in flat) == sorted(len(s["input_ids"]) for s in samples)

    def test_balance_equalizes_squared_length(self):
        # [3, 3, 2, 2] across two rows packs as [3, 2] | [3, 2] -> equal Σ Lᵢ², not equal counts by accident.
        samples = [_rollout_sample(length) for length in (3, 3, 2, 2)]
        groups = _balance_by_squared_length(samples, num_groups=2)

        loads = [_squared_load(group) for group in groups]
        assert loads[0] == loads[1]

    def test_balance_prefers_squared_length_over_count(self):
        # One long sample alone balances Σ Lᵢ² better than splitting by count: [4] | [3, 2, 1].
        samples = [_rollout_sample(length) for length in (4, 3, 2, 1)]
        groups = _balance_by_squared_length(samples, num_groups=2)

        counts = sorted(len(group) for group in groups)
        assert counts == [1, 3]
        loads = sorted(_squared_load(group) for group in groups)
        assert loads == [14, 16]  # [3,2,1] -> 9+4+1=14, [4] -> 16

    def test_token_budget_batcher_respects_budget_and_fills_every_row(self):
        source = (_rollout_sample(3) for _ in range(100))
        batcher = TokenBudgetBatcher(source, num_processes=2, token_budget=8)

        micro_batches = list(itertools.islice(iter(batcher), 5))
        assert len(micro_batches) == 5
        for groups in micro_batches:
            assert len(groups) == 2
            assert all(len(group) > 0 for group in groups)  # no rank forwards zero tokens
            assert all(_row_tokens(group) <= 8 for group in groups)  # peak memory bounded

    def test_token_budget_batcher_sizes_rows_dynamically(self):
        # Long samples pack few per row, short samples pack many — same budget, different counts.
        long_batcher = TokenBudgetBatcher((_rollout_sample(5) for _ in range(100)), num_processes=2, token_budget=8)
        short_batcher = TokenBudgetBatcher((_rollout_sample(2) for _ in range(100)), num_processes=2, token_budget=8)

        long_mb = next(iter(long_batcher))
        short_mb = next(iter(short_batcher))

        assert all(len(group) == 1 for group in long_mb)  # 5 + 5 > 8 -> one per row
        assert all(len(group) == 4 for group in short_mb)  # 2 * 4 = 8 -> four per row

    def test_token_budget_batcher_drops_oversized_sample(self):
        # A sample longer than the whole budget (12 > 8) fits in no row, so it is dropped, never emptying a row.
        PartialState()  # the drop path logs via accelerate's logger, which needs an initialized state
        source = (_rollout_sample(n) for n in ([12] + [3] * 60))
        batcher = TokenBudgetBatcher(source, num_processes=2, token_budget=8)
        for groups in itertools.islice(iter(batcher), 5):
            assert len(groups) == 2
            assert all(len(group) > 0 for group in groups)  # every row stays non-empty
            lengths = [len(sample["input_ids"]) for group in groups for sample in group]
            assert all(length <= 8 for length in lengths)  # the oversized sample (12) was dropped

    def test_token_budget_defaults_to_per_device_bs_times_completion_length(self):
        # Unset token_budget resolves to per_device_train_batch_size * max_completion_length = 3 * 8.
        args = AsyncGRPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=3,
            max_completion_length=8,
            bf16=False,
            report_to="none",
        )
        assert args.token_budget == 24
        # An explicit <= 0 value is left untouched and disables budgeting (-> FixedCountBatcher).
        args = AsyncGRPOConfig(output_dir=self.tmp_dir, token_budget=-1, bf16=False, report_to="none")
        assert args.token_budget == -1

    def test_fixed_count_batcher_yields_balanced_fixed_count_micro_batches(self):
        source = (_rollout_sample(length) for length in itertools.cycle((4, 3, 2, 1)))
        batcher = FixedCountBatcher(source, num_processes=2, microbatch_size=4)

        micro_batches = list(itertools.islice(iter(batcher), 3))
        assert len(micro_batches) == 3
        for groups in micro_batches:
            assert len(groups) == 2
            assert all(len(group) > 0 for group in groups)  # no rank forwards zero tokens
            assert sum(len(group) for group in groups) == 4  # fixed sample count per micro-batch

    def test_collator_pads_unequal_rows(self):
        # The planner hands the collator a pre-partitioned micro-batch, wrapped in a length-1 list by the dataloader.
        collator = DataCollatorForRollout(pad_token_id=0, num_processes=2)
        a = _rollout_sample(3, advantage=1.0, reward=0.5)  # input_ids [0, 1, 2]
        b = _rollout_sample(2, advantage=-1.0, reward=0.25)  # input_ids [0, 1]
        groups = [[a], [b]]

        batch = collator([groups])

        assert batch["input_ids"].tolist() == [[0, 1, 2], [0, 1, 0]]  # row b right-padded with pad_token_id
        assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]
        assert batch["position_ids"].tolist() == [[0, 1, 2], [0, 1, 0]]  # resets per sequence, 0-padded
        assert batch["completion_mask"].tolist() == [[0, 1, 1], [0, 1, 0]]
        assert batch["advantages"].tolist() == [[1.0, 1.0, 1.0], [-1.0, -1.0, 0.0]]  # per-token, 0-padded
        assert batch["global_n_tokens"].tolist() == [3.0, 3.0]  # a: 2 + b: 1 completion tokens
        assert batch["metrics"]["reward"].tolist() == [[0.5], [0.25]]  # float32-exact values

    def test_collator_packs_multiple_samples_per_row(self):
        # Two samples per row: position_ids reset at each sequence start and advantages expand per token.
        collator = DataCollatorForRollout(pad_token_id=0, num_processes=2)
        a = _rollout_sample(3, advantage=1.0, reward=0.5)  # input_ids [0, 1, 2]
        c = _rollout_sample(2, advantage=2.0, reward=0.25)  # input_ids [0, 1]
        b = _rollout_sample(2, advantage=-1.0, reward=0.75)  # input_ids [0, 1]
        d = _rollout_sample(3, advantage=-2.0, reward=0.5)  # input_ids [0, 1, 2]
        groups = [[a, c], [b, d]]  # both rows pack to 5 tokens -> no inter-rank padding

        batch = collator([groups])

        assert batch["input_ids"].tolist() == [[0, 1, 2, 0, 1], [0, 1, 0, 1, 2]]
        assert batch["position_ids"].tolist() == [[0, 1, 2, 0, 1], [0, 1, 0, 1, 2]]  # resets at each sequence start
        assert batch["completion_mask"].tolist() == [[0, 1, 1, 0, 1], [0, 1, 0, 1, 1]]
        assert batch["advantages"].tolist() == [[1.0, 1.0, 1.0, 2.0, 2.0], [-1.0, -1.0, -2.0, -2.0, -2.0]]
        assert batch["global_n_tokens"].tolist() == [6.0, 6.0]  # a:2 + c:1 + b:1 + d:2 completion tokens
        assert batch["metrics"]["reward"].tolist() == [[0.5, 0.25], [0.75, 0.5]]  # one row per rank, per sample
