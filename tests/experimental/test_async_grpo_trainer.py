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
import multiprocessing as mp
import queue

import numpy as np
import pytest
import torch
from accelerate import PartialState
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers.testing_utils import torch_device

from trl.experimental.async_grpo import AsyncGRPOConfig, AsyncGRPOTrainer
from trl.experimental.async_grpo.async_grpo_trainer import (
    DataCollatorForRollout,
    FixedCountBatcher,
    TokenBudgetBatcher,
    _balance_by_squared_length,
)
from trl.experimental.async_grpo.async_rollout_worker import RolloutSample, _AsyncRolloutLoop

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
            token_budget=256,  # set explicitly; the stub worker has no real vLLM server to query for max_model_len
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

    def test_dataset_required_without_environment(self):
        # The data has to come from somewhere: an external `train_dataset`, or an environment that owns it. With
        # neither, construction fails fast.
        training_args = AsyncGRPOConfig(output_dir=self.tmp_dir, max_steps=5, report_to="none")
        with pytest.raises(ValueError, match="`train_dataset` is required"):
            AsyncGRPOTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                reward_funcs=dummy_reward_func,
                args=training_args,
            )

    def test_environment_owned_data_requires_max_steps(self):
        # When the environment owns the data and no external `train_dataset` is provided, `max_steps` must set the
        # training length. The guard fires before the rollout worker is built, so no worker/dataset is needed.
        class DummyEnvironment:
            def reset(self, **kwargs):
                return "Guess the 5-letter word."

        args = AsyncGRPOConfig(output_dir=self.tmp_dir, report_to="none")  # max_steps unset
        with pytest.raises(ValueError, match="max_steps"):
            AsyncGRPOTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                reward_funcs=dummy_reward_func,
                args=args,
                environment_factory=DummyEnvironment,
            )


class TestAsyncRolloutWorkerEnvironments(TrlTestCase):
    """Unit tests for the rollout worker's environment/tool wiring (no vLLM required)."""

    def _make_loop(self, environment_factory, dataset=None):
        model_id = "trl-internal-testing/tiny-Qwen3MoeForCausalLM"
        if dataset is None:
            dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_only", split="train")
        # `_AsyncRolloutLoop.__init__` only sets up state (no vLLM connection / generation happens here).
        return _AsyncRolloutLoop(
            model_name=model_id,
            dataset=dataset,
            reward_funcs=[dummy_reward_func],
            processing_class=AutoTokenizer.from_pretrained(model_id),
            rollout_buffer=mp.Queue(),
            model_version_value=mp.Value("i", 0),
            heartbeat_value=mp.Value("d", 0.0),
            failed_event=mp.Event(),
            exception_info_queue=mp.Queue(),
            environment_factory=environment_factory,
            num_generations=2,
            max_inflight_tasks=4,
        )

    def test_multiple_environments_expose_only_their_own_tools(self):
        class CounterEnvironment:
            def reset(self, **kwargs): ...

            def increment(self, step: int) -> int:
                """Increment the counter.

                Args:
                    step: Value to add.

                Returns:
                    The updated value.
                """
                return step

        class EchoEnvironment:
            def reset(self, **kwargs): ...

            def shout(self, text: str) -> str:
                """Shout the text.

                Args:
                    text: Text to shout.

                Returns:
                    The text in upper case.
                """
                return text.upper()

        loop = self._make_loop({"counter": CounterEnvironment, "echo": EchoEnvironment})
        try:
            assert loop._multi_environment is True
            # Each environment exposes only its own tool, used to render that example's prompt schema.
            assert [tool.__name__ for tool in loop._env_tools["counter"]] == ["increment"]
            assert [tool.__name__ for tool in loop._env_tools["echo"]] == ["shout"]
            # `self.tools` is the union, used only to decide whether a training chat template is needed.
            assert sorted(tool.__name__ for tool in loop.tools) == ["increment", "shout"]
            # The probe instances seed the reuse pool, so they are not wasted.
            assert len(loop._environment_pool["counter"]) == 1
            assert len(loop._environment_pool["echo"]) == 1
        finally:
            loop._loop.close()

    def test_unknown_environment_raises(self):
        # An example whose `environment` field doesn't match any configured environment should fail with a clear error
        # rather than a bare KeyError mid-rollout. The check fires before any generation, so no vLLM is needed here.
        class CounterEnvironment:
            def reset(self, **kwargs): ...

            def increment(self, step: int) -> int:
                """Increment the counter.

                Args:
                    step: Value to add.

                Returns:
                    The updated value.
                """
                return step

        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_only", split="train")
        dataset = dataset.map(lambda example: {"environment": "unknown"})

        loop = self._make_loop({"counter": CounterEnvironment}, dataset=dataset)
        try:
            with pytest.raises(ValueError, match="not among the environments"):
                loop._loop.run_until_complete(loop._generate_loop(stop_event=loop._stop_event))
        finally:
            loop._loop.close()


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
