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

import asyncio
import dataclasses
import math
import os
import threading
import time
from collections import defaultdict

import pytest
import torch
from accelerate import PartialState
from datasets import Dataset, load_dataset
from torch.distributed.tensor import DTensor
from transformers import AutoTokenizer

from trl.experimental.zero_sync_grpo import ZeroSyncGRPOConfig, ZeroSyncGRPOTrainer, ZeroSyncRolloutLoop
from trl.experimental.zero_sync_grpo.zero_sync_grpo_trainer import (
    Rollout,
    RolloutGroup,
    TrainingSequence,
    TurnRecord,
    _chain_to_sequences,
    _common_prefix_len,
    _reduce_metric,
)

from ..testing_utils import TrlTestCase, require_response_parsing, require_torch_accelerator


def dummy_reward_func(completions, **kwargs):
    return [float(len(completion[0]["content"])) for completion in completions]


def _finalize(turns, rollout_id="r0"):
    rows, _drifts = _chain_to_sequences(Rollout([], [], turns, rollout_id=rollout_id))
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
        # Candidate sets ride with the sampled tokens; the prompt and the tool result are context and carry none.
        turn1 = TurnRecord([1, 2, 3], [10, 11], [-0.1, -0.2], output_sampling_mask=[[10, 12], [11]])
        turn2 = TurnRecord([1, 2, 3, 10, 11, 20], [30], [-0.3], output_sampling_mask=[[30, 31, 32]])
        rows = _finalize([turn1, turn2])
        assert len(rows) == 1
        assert rows[0].input_ids == [1, 2, 3, 10, 11, 20, 30]
        assert rows[0].completion_mask == [0, 0, 0, 1, 1, 0, 1]
        assert rows[0].sampling_mask == [[], [], [], [10, 12], [11], [], [30, 31, 32]]

    def test_clean_chain_stays_one_row(self):
        # Turn 2's re-tokenized prompt starts with the held tokens (gen [10,11] + tool [20,21]), so it extends the row.
        turn1 = TurnRecord([1, 2, 3], [10, 11], [-0.1, -0.2])
        turn2 = TurnRecord([1, 2, 3, 10, 11, 20, 21], [30, 31], [-0.3, -0.4])
        rows = _finalize([turn1, turn2])
        assert len(rows) == 1
        assert rows[0].input_ids == [1, 2, 3, 10, 11, 20, 21, 30, 31]
        assert rows[0].completion_mask == [0, 0, 0, 1, 1, 0, 0, 1, 1]  # prompt=0 gen=1 tool=0 gen=1
        assert rows[0].old_log_probs == [0.0, 0.0, 0.0, -0.1, -0.2, 0.0, 0.0, -0.3, -0.4]

    def test_rewrite_forks_into_two_rows(self):
        # Turn 2's prompt rewrites turn 1's answer, so turn 2 starts a new row. Every generated token is trained in
        # exactly one row (turn 1's tokens are context in row 2).
        turn1 = TurnRecord([1, 2, 3], [10, 11, 12, 13], [-0.2] * 4)
        turn2 = TurnRecord([1, 2, 3, 10, 99, 88, 77], [30, 31, 32], [-0.2] * 3)
        rows = _finalize([turn1, turn2])
        assert len(rows) == 2
        assert rows[0].input_ids == [1, 2, 3, 10, 11, 12, 13]
        assert rows[0].completion_mask == [0, 0, 0, 1, 1, 1, 1]
        assert rows[1].input_ids == [1, 2, 3, 10, 99, 88, 77, 30, 31, 32]
        assert rows[1].completion_mask == [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]

    def test_reports_the_drift_of_each_fork(self):
        turn1 = TurnRecord([1, 2, 3], [10, 11, 12, 13], [-0.2] * 4)
        clean = TurnRecord([1, 2, 3, 10, 11, 12, 13], [20], [-0.2] * 1)
        forked = TurnRecord([1, 2, 3, 10, 99, 88, 77, 20], [30], [-0.2] * 1)
        assert _chain_to_sequences(Rollout([], [], [turn1]))[1] == []
        # The fork diverges at the 5th held token (11 -> 99), so 4 of the 8 held tokens re-rendered.
        assert _chain_to_sequences(Rollout([], [], [turn1, clean, forked]))[1] == [4]

    def test_tail_wobble_forks(self):
        # Last generated token re-renders (11 -> 12). Turn 1 keeps its row and both its trained tokens; turn 2 starts
        # a new row where the re-rendered answer is context.
        turn1 = TurnRecord([1, 2, 3], [10, 11], [-0.1, -0.2])
        turn2 = TurnRecord([1, 2, 3, 10, 12], [30, 31], [-0.3, -0.4])
        rows = _finalize([turn1, turn2])
        assert len(rows) == 2
        assert rows[0].completion_mask == [0, 0, 0, 1, 1]
        assert rows[1].input_ids == [1, 2, 3, 10, 12, 30, 31]
        assert rows[1].completion_mask == [0, 0, 0, 0, 0, 1, 1]
        assert rows[1].old_log_probs == [0.0, 0.0, 0.0, 0.0, 0.0, -0.3, -0.4]


def _rollout_loop(loop_cls=ZeroSyncRolloutLoop, **kwargs):
    # `ZeroSyncRolloutLoop.__init__` only sets up state (no engine request / generation happens here).
    model_id = "trl-internal-testing/tiny-Qwen3MoeForCausalLM"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    config_fields = {f.name for f in dataclasses.fields(ZeroSyncGRPOConfig)}
    config_kwargs = {k: kwargs.pop(k) for k in list(kwargs) if k in config_fields}
    config_kwargs.setdefault("max_completion_length", 32)
    config_kwargs.setdefault("max_concurrent_rollouts", 128)
    config = ZeroSyncGRPOConfig(report_to="none", **config_kwargs)
    kwargs = {
        "manager": None,
        "reward_funcs": [dummy_reward_func],
        "processing_class": tokenizer,
        "eos_token_ids": [tokenizer.eos_token_id],
        "metrics": defaultdict(list),
        "model_version_fn": lambda: 7,
        **kwargs,
    }
    return loop_cls(config, **kwargs)


class TestMetricReduction(TrlTestCase):
    """The value's shape and the key's suffix are the whole reduction API; nothing is registered or configured."""

    def test_gauge_is_meaned(self):
        assert _reduce_metric("perf/fwd_bwd_s", [1.0, 2.0, 6.0]) == 3.0

    def test_counter_is_summed(self):
        assert _reduce_metric("sample/dropped_stale_total", [1.0, 1.0, 1.0]) == 3.0

    def test_extrema(self):
        assert _reduce_metric("sample/forwarded_tokens_max", [3.0, 9.0, 5.0]) == 9.0
        assert _reduce_metric("clip_ratio/low_min", [3.0, 9.0, 5.0]) == 3.0
        # A `max`/`min` WORD, not a suffix, so the upstream spellings reduce correctly too.
        assert _reduce_metric("completions/max_length", [3.0, 9.0, 5.0]) == 9.0
        assert _reduce_metric("completions/min_length", [3.0, 9.0, 5.0]) == 3.0
        # ... and a name that merely contains those letters is still a gauge.
        assert _reduce_metric("perf/maximal_s", [2.0, 4.0]) == 3.0

    def test_rate_is_sum_over_sum_not_mean_of_ratios(self):
        # The two windows have rates 1 tok/s and 100 tok/s. Meaning the ratios would give 50.5; the honest rate is
        # total tokens over total seconds. This is the defect that made `training_tok/s` unusable.
        assert _reduce_metric("perf/forwarded_tok_s_e2e", [(100.0, 100.0), (100.0, 1.0)]) == pytest.approx(200 / 101)

    def test_rate_with_zero_denominator_is_nan(self):
        assert math.isnan(_reduce_metric("perf/forwarded_tok_s_e2e", [(0.0, 0.0)]))


class TestMetricPush(TrlTestCase):
    """The loop's payload has the same shape as the trainer's sink, so pushing it is an append."""

    def test_counters_and_rates_ride_along_and_reset(self):
        loop = _rollout_loop()
        loop._counters["tools/search/call_total"] += 2
        loop._rates["tools/latency_s"][0] += 3.0
        loop._rates["tools/latency_s"][1] += 2
        loop._push_metrics({"rollout/score_s": 0.5})

        assert loop.metrics == {
            "rollout/score_s": [0.5],
            "tools/search/call_total": [2.0],
            "tools/latency_s": [(3.0, 2.0)],
        }
        # Counters carry deltas: whatever was pushed must not be pushed again.
        loop._push_metrics({})
        assert loop.metrics == {
            "rollout/score_s": [0.5],
            "tools/search/call_total": [2.0],
            "tools/latency_s": [(3.0, 2.0)],
        }


class TestToolExecution(TrlTestCase):
    @staticmethod
    def _call(name, **arguments):
        return {"type": "function", "function": {"name": name, "arguments": arguments}}

    def test_every_tool_shape_runs_and_calls_keep_their_order(self):
        def sync_tool(value: int) -> str:
            return f"sync:{value}"

        async def async_tool(value: int) -> str:
            await asyncio.sleep(0)
            return f"async:{value}"

        async def failing_tool(value: int) -> str:
            raise RuntimeError(f"boom:{value}")

        loop = _rollout_loop(max_concurrent_rollouts=4)
        tool_dict = {
            "sync_tool": sync_tool,
            "async_tool": async_tool,
            "failing_tool": failing_tool,
        }
        calls = [
            self._call("sync_tool", value=1),
            self._call("async_tool", value=2),
            self._call("failing_tool", value=3),
            self._call("missing_tool", value=4),
        ]
        messages, n_calls, n_failures = asyncio.run(loop._execute_tool_calls(calls, tool_dict))

        assert n_calls == 4
        assert n_failures == 2
        assert [m["name"] for m in messages] == ["sync_tool", "async_tool", "failing_tool", "missing_tool"]
        assert messages[0]["content"] == "sync:1"
        assert messages[1]["content"] == "async:2"
        assert "boom:3" in messages[2]["content"]
        assert "unknown tool" in messages[3]["content"]
        assert loop._counters["tools/failing_tool/failure_total"] == 1
        assert loop._counters["tools/unknown_name_total"] == 1
        assert loop._rates["tools/latency_s"][1] == 3

    def test_slow_sync_tool_does_not_block_the_event_loop(self):
        def slow_tool() -> str:
            time.sleep(0.5)
            return "done"

        loop = _rollout_loop(max_concurrent_rollouts=4)

        async def scenario():
            ticks = 0

            async def ticker():
                nonlocal ticks
                while True:
                    ticks += 1
                    await asyncio.sleep(0.01)

            task = asyncio.create_task(ticker())
            messages, _, _ = await loop._execute_tool_calls([self._call("slow_tool")], {"slow_tool": slow_tool})
            task.cancel()
            return messages, ticks

        messages, ticks = asyncio.run(scenario())
        assert messages[0]["content"] == "done"
        assert ticks > 2  # inline, the ticker would have advanced once at most


# The engine's turns, opened by Qwen3's empty thinking block: a tool call continues the loop, an answer ends it
_TOOL_CALL = '<think>\n\n</think>\n\n<tool_call>\n{"name": "t", "arguments": {}}\n</tool_call><|im_end|>'
_FINAL = "<think>\n\n</think>\n\ndone<|im_end|>"


def t() -> str:
    """
    Return ok.
    """
    return "ok"


def _run(*, turns, max_iters=None):
    """
    Drive `ZeroSyncRolloutLoop.generate_one` with the real tokenizer, chat template and response parser, scripting only
    the engine, which generates the text of `turns` one turn after the other. Returns the rollout and the training rows
    the loop builds from it.
    """
    loop = _rollout_loop(tools=[t], max_tool_calling_iterations=max_iters)
    outputs = [loop.tokenizer.encode(text, add_special_tokens=False) for text in turns]

    async def generate_one_turn(prompt_ids):
        ids = outputs.pop(0)
        return TurnRecord(prompt_ids, ids, [-0.1] * len(ids))

    loop.generate_one_turn = generate_one_turn
    rollout = asyncio.run(loop.generate_one([{"role": "user", "content": "hi"}], [t]))
    rows, _drifts = _chain_to_sequences(rollout)
    return rollout, rows


class TestRolloutLoop(TrlTestCase):
    def test_single_turn_no_tool_call(self):
        rollout, rows = _run(turns=[_FINAL])
        assert len(rows) == 1
        n = len(rollout.completion_ids)
        assert rows[0].input_ids[-n:] == rollout.completion_ids
        assert rows[0].completion_mask == [0] * (len(rows[0].input_ids) - n) + [1] * n
        assert rows[0].old_log_probs[-n:] == [-0.1] * n
        assert [m["role"] for m in rollout.completion] == ["assistant"]
        assert rollout.tool_call_count == 0 and rollout.tool_failure_count == 0

    def test_clean_two_turns_stay_one_row(self):
        # The training chat template is prefix-preserving, so turn 2's re-tokenized prompt extends turn 1 -> one row
        rollout, rows = _run(turns=[_TOOL_CALL, _FINAL])
        assert len(rows) == 1
        assert sum(rows[0].completion_mask) == len(rollout.completion_ids)  # every generated token is trained once
        assert [m["role"] for m in rollout.completion] == ["assistant", "tool", "assistant"]
        assert rollout.completion[1]["content"] == "ok"
        assert rollout.tool_call_count == 1 and rollout.tool_failure_count == 0

    def test_max_tool_calling_iterations_caps_turns(self):
        # max_iters=0: even though turn 1 is a tool call, the loop breaks before executing it.
        rollout, rows = _run(turns=[_TOOL_CALL], max_iters=0)
        assert len(rows) == 1
        assert [m["role"] for m in rollout.completion] == ["assistant"]  # no tool message appended
        assert rollout.tool_call_count == 0


class TestRolloutLoopEnvironments(TrlTestCase):
    """Unit tests for the rollout loop's environment/tool wiring (no engine required)."""

    def _make_loop(self, environment_factory, dataset=None):
        if dataset is None:
            dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_only", split="train")
        loop = _rollout_loop(environment_factory=environment_factory, num_generations=2, max_concurrent_rollouts=4)
        loop.prompts = iter([list(dataset)])
        return loop

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
            # `self.tools` is the union, used only to decide whether a training chat template is needed.
            assert sorted(tool.__name__ for tool in loop.tools) == ["increment", "shout"]
            # The probe instances seed the reuse pool, so they are not wasted.
            assert len(loop._environment_pool["counter"]) == 1
            assert len(loop._environment_pool["echo"]) == 1
        finally:
            loop._loop.close()

    def test_tool_discovery_does_not_evaluate_properties(self):
        # The init-time probe lists an environment's tool methods. `inspect.getmembers` calls `getattr` on every name
        # before applying the predicate, so listing the instance would evaluate this property; listing the class
        # leaves it inert.
        class PropertyEnvironment:
            def reset(self, **kwargs): ...

            @property
            def reward(self) -> float:
                raise RuntimeError("`reward` must not be evaluated while discovering tools")

            def echo(self, text: str) -> str:
                """Echo the text back.

                Args:
                    text: Text to echo.

                Returns:
                    The text, unchanged.
                """
                return text

        loop = self._make_loop(PropertyEnvironment)
        try:
            assert [tool.__name__ for tool in loop.tools] == ["echo"]
        finally:
            loop._loop.close()

    @require_response_parsing
    def test_environment_calls_run_off_the_event_loop_and_are_timed(self):
        reset_threads = []

        class SlowEnvironment:
            def reset(self, **kwargs):
                reset_threads.append(threading.current_thread())
                time.sleep(0.02)

            def step(self) -> str:
                """
                Take a step.
                """
                time.sleep(0.02)
                return "ok"

            def get_reward(self):
                time.sleep(0.02)
                loop._loop.call_soon_threadsafe(loop._stop_event.set)
                return 1.0

        loop = self._make_loop(SlowEnvironment)
        tool_call = loop.tokenizer.encode(_TOOL_CALL.replace('"t"', '"step"'), add_special_tokens=False)
        final = loop.tokenizer.encode(_FINAL, add_special_tokens=False)

        async def generate_one_turn(prompt_ids):
            ids = final if "<tool_response>" in loop.tokenizer.decode(prompt_ids) else tool_call
            return TurnRecord(prompt_ids, ids, [-0.1] * len(ids))

        loop.generate_one_turn = generate_one_turn
        PartialState()
        try:
            loop._loop.run_until_complete(loop._run_loops(stop_event=loop._stop_event))
        finally:
            loop._loop.close()

        rates = defaultdict(lambda: [0.0, 0.0])
        for key, values in loop.metrics.items():
            for value in values:
                if isinstance(value, tuple):
                    rates[key][0] += value[0]
                    rates[key][1] += value[1]
        assert reset_threads and all(t is not threading.main_thread() for t in reset_threads)
        for key in ("env/reset_s", "env/reward_s", "tools/step/latency_s"):
            assert rates[key][0] / rates[key][1] >= 0.02

    def test_unknown_environment_raises(self):
        # An example whose `environment` field doesn't match any configured environment should fail with a clear error
        # rather than a bare KeyError mid-rollout. The check fires before any generation, so no engine is needed here.
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


class _FixedRolloutLoop(ZeroSyncRolloutLoop):
    """Answers every prompt with the same tokens, without the engine."""

    async def generate_one(self, prompt, tools, group_id=0):
        return Rollout([{"role": "assistant", "content": "x"}], [2, 3], [TurnRecord([1], [2, 3], [-0.1, -0.1])])


class _EvenRowsLoop(_FixedRolloutLoop):
    def filter_prompt(self, row_index, row, epoch):
        self.prompt_calls.append((row_index, epoch))
        return row_index % 2 == 0

    async def generate_one(self, prompt, tools, group_id=0):
        self.generated_rows.append(prompt[-1]["content"])
        return await super().generate_one(prompt, tools, group_id)


class _RejectAllLoop(_FixedRolloutLoop):
    def filter_prompt(self, row_index, row, epoch):
        return False


class _SelfScoredLoop(_FixedRolloutLoop):
    async def generate_one(self, prompt, tools, group_id=0):
        rollout = await super().generate_one(prompt, tools, group_id)
        self.calls += 1
        return dataclasses.replace(rollout, reward=float(self.calls % 2))


class _LifecycleLoop(_FixedRolloutLoop):
    async def setup(self):
        self.events.append("setup")

    async def generate_one(self, prompt, tools, group_id=0):
        self.events.append("generate")
        return await super().generate_one(prompt, tools, group_id)

    async def close(self):
        self.events.append("close")


class TestCustomRolloutLoop(TrlTestCase):
    @pytest.fixture(autouse=True)
    def _init_accelerate_state(self):
        PartialState()

    def _run(self, loop, num_samples, dataset=None):
        """Run the loop as the trainer does until `num_samples` samples reach the rollout buffer, over one epoch of
        `dataset`; return everything the buffer received."""
        if dataset is None:
            dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_only", split="train")
        loop.prompts = iter([list(dataset)])
        loop.run_until(lambda: len(loop.rollout_buffer) >= num_samples)
        loop.stop()
        return list(loop.rollout_buffer)

    def test_filter_prompt_skips_rows(self):
        dataset = Dataset.from_dict({"prompt": [[{"role": "user", "content": f"row_{i}"}] for i in range(4)]})
        loop = _rollout_loop(_EvenRowsLoop, num_generations=2, reward_funcs=[two_reward])
        loop.prompt_calls, loop.generated_rows = [], []
        samples = self._run(loop, 4, dataset)
        assert loop.prompt_calls == [(0, 0), (1, 0), (2, 0), (3, 0)]
        assert set(loop.generated_rows) == {"row_0", "row_2"}
        assert {sample.group_id for sample in samples} == {0, 2}

    def test_filter_prompt_rejecting_a_whole_epoch_raises(self):
        loop = _rollout_loop(_RejectAllLoop, reward_funcs=[two_reward])
        loop.prompts = iter(
            [list(load_dataset("trl-internal-testing/zen", "conversational_prompt_only", split="train"))]
        )
        with pytest.raises(RuntimeError, match="rejected every prompt of epoch 0"):
            loop.run_until(lambda: False)

    def test_the_built_in_generate_one_needs_a_reward_source(self):
        with pytest.raises(ValueError, match="No reward source provided"):
            _rollout_loop(reward_funcs=[])

    def test_generate_one_can_score_its_own_rollouts(self):
        loop = _rollout_loop(_SelfScoredLoop, num_generations=2, reward_funcs=[])
        loop.calls = 0
        samples = self._run(loop, 4)
        assert {sample.metrics["rewards/rollout/mean"] for sample in samples} == {0.0, 1.0}
        assert {round(sample.advantage, 4) for sample in samples} == {-1.0, 1.0}

    def test_setup_runs_before_generation_and_close_after(self):
        loop = _rollout_loop(_LifecycleLoop, num_generations=2, reward_funcs=[two_reward])
        loop.events = []
        self._run(loop, 2)
        assert loop.events[0] == "setup"
        assert loop.events[-1] == "close"
        assert "generate" in loop.events[1:-1]

    def test_each_rollout_gets_the_tools_of_its_environment(self):
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

        class ToolsLoop(_FixedRolloutLoop):
            async def generate_one(self, prompt, tools, group_id=0):
                self.seen_tools.append(tuple(tool.__name__ for tool in tools))
                return await super().generate_one(prompt, tools, group_id)

        prompt = [{"role": "user", "content": "hi"}]
        dataset = Dataset.from_dict({"prompt": [prompt, prompt], "environment": ["counter", "echo"]})
        loop = _rollout_loop(
            ToolsLoop,
            environment_factory={"counter": CounterEnvironment, "echo": EchoEnvironment},
            num_generations=2,
            reward_funcs=[two_reward],
        )
        loop.seen_tools = []
        self._run(loop, 4, dataset)
        assert set(loop.seen_tools) == {("increment",), ("shout",)}


def two_reward(completions, **kwargs):
    # Deterministic per-conversation reward: two conversations get rewards 1.0 and 3.0.
    return [1.0, 3.0]


class AsyncTwoReward:
    # Same reward as `two_reward`, as an async callable class: the picklable form the docs recommend, and the one
    # `inspect.iscoroutinefunction` does not see as asynchronous on its own.
    async def __call__(self, completions, **kwargs):
        return [1.0, 3.0]


def _group(completions_sequences, completions_ids):
    n = len(completions_sequences)
    return RolloutGroup(
        prompts=[[{"role": "user", "content": "hi"}] for _ in range(n)],
        reward_kwargs={},
        completions=[[{"role": "assistant", "content": f"c{i}"}] for i in range(n)],
        completions_ids=completions_ids,
        completions_sequences=completions_sequences,
        tool_call_counts=[0] * n,
        tool_failure_counts=[0] * n,
        model_version=7,
        group_id=0,
        row_index=0,
        env_rewards=[None] * n,
        rollout_rewards=[None] * n,
    )


class TestScoreGroupOptionThree(TrlTestCase):
    @pytest.fixture(autouse=True)
    def _init_accelerate_state(self):
        # _score_group logs via accelerate's logger, which requires PartialState() to be initialized (the trainer
        # does this before it builds the loop).
        PartialState()

    def test_one_advantage_per_conversation_stamped_on_every_row(self):
        # conv 0: one row; conv 1: two rows (a fork).
        seq_a = TrainingSequence([1, 2, 3, 10, 11], [0, 0, 0, 1, 1], [0, 0, 0, -0.1, -0.2], [[]] * 5, [{}] * 5, "c0")
        seq_b1 = TrainingSequence([1, 2, 3, 20, 21], [0, 0, 0, 1, 1], [0, 0, 0, -0.3, -0.4], [[]] * 5, [{}] * 5, "c1")
        seq_b2 = TrainingSequence(
            [1, 2, 3, 20, 99, 30], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, -0.5], [[]] * 6, [{}] * 6, "c1"
        )
        group = _group([[seq_a], [seq_b1, seq_b2]], completions_ids=[[10, 11], [20, 21, 30]])

        samples = asyncio.run(_rollout_loop(reward_funcs=[two_reward])._score_group(group))

        # 1 row for conv 0 + 2 rows for conv 1 = 3 samples.
        assert len(samples) == 3

        # rewards [1, 3] -> mean 2, std 1 -> advantages [-1, +1].
        assert samples[0].advantage == pytest.approx(-1.0)
        assert samples[1].advantage == pytest.approx(1.0)
        assert samples[2].advantage == pytest.approx(1.0)
        # The fork's two rows carry the identical (conversation-level) advantage — no split.
        assert samples[1].advantage == samples[2].advantage

        # Each sample maps 1:1 onto its source TrainingSequence.
        assert (
            samples[0].input_ids.tolist() == seq_a.input_ids
            and samples[0].completion_mask.tolist() == seq_a.completion_mask
        )
        assert samples[0].old_log_probs.tolist() == pytest.approx(seq_a.old_log_probs)
        assert samples[1].input_ids.tolist() == seq_b1.input_ids
        assert samples[2].input_ids.tolist() == seq_b2.input_ids

        assert all(s.model_version == 7 for s in samples)

    def test_async_callable_class_reward_func_is_awaited(self):
        seq_a = TrainingSequence([1, 2, 10], [0, 0, 1], [0, 0, -0.1], [[]] * 3, [{}] * 3, "c0")
        seq_b = TrainingSequence([1, 2, 20], [0, 0, 1], [0, 0, -0.2], [[]] * 3, [{}] * 3, "c1")
        group = _group([[seq_a], [seq_b]], completions_ids=[[10], [20]])

        samples = asyncio.run(_rollout_loop(reward_funcs=[AsyncTwoReward()])._score_group(group))

        # Run as a synchronous function, the reward is an un-awaited coroutine and never a number.
        assert samples[0].metrics["rewards/AsyncTwoReward/mean"] == 1.0
        assert samples[1].metrics["rewards/AsyncTwoReward/mean"] == 3.0
        assert samples[0].advantage == pytest.approx(-1.0)
        assert samples[1].advantage == pytest.approx(1.0)

    def test_metrics_are_per_conversation_and_independent(self):
        seq_a = TrainingSequence([1, 2, 10], [0, 0, 1], [0, 0, -0.1], [[]] * 3, [{}] * 3, "c0")
        seq_b1 = TrainingSequence([1, 2, 20], [0, 0, 1], [0, 0, -0.2], [[]] * 3, [{}] * 3, "c1")
        seq_b2 = TrainingSequence([1, 2, 20, 30], [0, 0, 0, 1], [0, 0, 0, -0.3], [[]] * 4, [{}] * 4, "c1")
        group = _group([[seq_a], [seq_b1, seq_b2]], completions_ids=[[10], [20, 30]])

        samples = asyncio.run(_rollout_loop(reward_funcs=[two_reward])._score_group(group))

        assert samples[0].metrics["reward_kept"] == 1.0
        assert samples[1].metrics["reward_kept"] == 3.0 and samples[2].metrics["reward_kept"] == 3.0
        assert samples[0].metrics["reward_group_std"] == pytest.approx(1.0)
        assert samples[1].metrics["rewards/two_reward/mean"] == 3.0
        assert samples[1].metrics["rewards/two_reward/group_std"] == pytest.approx(1.0)
        # The fork's two rows must not share a metrics dict (the score loop mutates it per sample).
        assert samples[1].metrics is not samples[2].metrics

    def test_all_none_reward_conversation_is_unscorable(self):
        # A conversation for which every reward func returns None gets advantage 0 and NaN reward.
        def maybe_none(completions, **kwargs):
            return [None, 2.0, 4.0]

        seq_a = TrainingSequence([1, 2, 10], [0, 0, 1], [0, 0, -0.1], [[]] * 3, [{}] * 3, "c0")
        seq_b = TrainingSequence([1, 2, 20], [0, 0, 1], [0, 0, -0.2], [[]] * 3, [{}] * 3, "c1")
        seq_c = TrainingSequence([1, 2, 30], [0, 0, 1], [0, 0, -0.3], [[]] * 3, [{}] * 3, "c2")
        group = _group([[seq_a], [seq_b], [seq_c]], completions_ids=[[10], [20], [30]])

        samples = asyncio.run(_rollout_loop(reward_funcs=[maybe_none])._score_group(group))

        assert len(samples) == 3
        assert samples[0].advantage == 0.0  # unscorable -> advantage 0
        assert math.isnan(samples[0].metrics["reward_kept"])
        assert samples[1].advantage < 0 < samples[2].advantage  # advantage over the scorable rows only
        assert samples[1].metrics["reward_kept"] == 2.0


def constant_reward(completions, **kwargs):
    return [1.0] * len(completions)


def all_none_reward(completions, **kwargs):
    return [None] * len(completions)


def one_scored_reward(completions, **kwargs):
    return [None, 1.0]


class TestDropZeroAdvantageGroups(TrlTestCase):
    @pytest.fixture(autouse=True)
    def _init_accelerate_state(self):
        PartialState()

    def _score(self, loop, reward_funcs):
        for group_id, reward_func in enumerate(reward_funcs):
            seq_a = TrainingSequence([1, 2, 10], [0, 0, 1], [0, 0, -0.1], [[]] * 3, [{}] * 3, "c0")
            seq_b = TrainingSequence([1, 2, 20], [0, 0, 1], [0, 0, -0.2], [[]] * 3, [{}] * 3, "c1")
            group = _group([[seq_a], [seq_b]], completions_ids=[[10], [20]])
            group.group_id = group_id
            group.queued_at = time.monotonic()
            loop.reward_funcs = [reward_func]
            loop._groups_to_score.put_nowait(group)
            loop._groups_to_score.put_nowait(None)
            loop._loop.run_until_complete(loop._score_loop(asyncio.Event()))
        return list(loop.rollout_buffer), loop.metrics

    @pytest.mark.parametrize(
        "reward_func, reward",
        [(constant_reward, (2.0, 2)), (all_none_reward, (0.0, 0)), (one_scored_reward, (1.0, 1))],
    )
    def test_dropped_group_puts_nothing_in_the_buffer(self, reward_func, reward):
        items, metrics = self._score(_rollout_loop(), [reward_func, two_reward])
        assert [s.group_id for s in items] == [1, 1]
        assert metrics["frac_reward_zero_std"] == [(1.0, 1), (0.0, 1)]
        assert metrics["reward"] == [reward, (4.0, 2)]  # the dropped group still counts in the policy's reward


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
            max_concurrent_rollouts=3,  # keep few rollouts generating, the test model is tiny
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

    def test_train_with_a_custom_loop_cls(self):
        class FixedTrainer(ZeroSyncGRPOTrainer):
            loop_cls = _FixedRolloutLoop

        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_only", split="train")
        trainer = FixedTrainer(
            model="trl-internal-testing/small-Qwen2ForCausalLM-2.5",
            reward_funcs=two_reward,
            args=ZeroSyncGRPOConfig(
                output_dir=self.tmp_dir,
                per_device_train_batch_size=2,
                num_generations=2,
                max_steps=1,
                report_to="none",
            ),
            train_dataset=dataset,
        )
        trainer.train()
        assert trainer.state.global_step == 1

    def test_train_bf16_steps_fp32_copies(self):
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_only", split="train")
        training_args = ZeroSyncGRPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=1e-6,  # a step bf16 rounds away on most weights
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            max_concurrent_rollouts=3,  # keep few rollouts generating, the test model is tiny
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

    @pytest.mark.skipif("TRL_TEST_ZERO_SYNC_DP_OUTPUT_DIR" not in os.environ, reason="launched by tests/distributed")
    def test_train_replicas_body(self):
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_only", split="train")
        training_args = ZeroSyncGRPOConfig(
            output_dir=os.environ["TRL_TEST_ZERO_SYNC_DP_OUTPUT_DIR"],
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            gradient_accumulation_steps=2,  # the replicas wait for every micro-batch of the step before pausing
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            max_concurrent_rollouts=3,  # keep few rollouts generating, the test model is tiny
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

        # Every parameter has moved, and the replicas, which each update a share of them, hold the same weights
        params = {n: param.detach().cpu() for n, param in trainer.model.named_parameters()}
        for n, param in previous_trainable_params.items():
            assert not torch.equal(param.cpu(), params[n]), f"Parameter {n} has not changed."
        gathered = [None] * int(os.environ["WORLD_SIZE"])
        torch.distributed.all_gather_object(gathered, params)
        for other in gathered[1:]:
            for n, param in params.items():
                assert torch.equal(param, other[n]), f"Parameter {n} differs between replicas."

    @pytest.mark.skipif("TRL_TEST_ZERO_SYNC_TP_OUTPUT_DIR" not in os.environ, reason="launched by tests/distributed")
    def test_train_tensor_parallel_body(self):
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_only", split="train")
        scored = []

        def counting_reward_func(completions, **kwargs):
            scored.extend(completions)
            return dummy_reward_func(completions, **kwargs)

        training_args = ZeroSyncGRPOConfig(
            output_dir=os.environ["TRL_TEST_ZERO_SYNC_TP_OUTPUT_DIR"],
            learning_rate=1e-6,  # a step bf16 rounds away on most weights
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            gradient_accumulation_steps=2,  # the batch the first rank sends holds every micro-batch of the step
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            max_concurrent_rollouts=3,  # keep few rollouts generating, the test model is tiny
            max_steps=2,
            model_init_kwargs={"dtype": "bfloat16"},
            tp_size=int(os.environ["WORLD_SIZE"]),
            save_strategy="no",  # a model sharded at load time cannot be checkpointed for resuming
            report_to="none",
        )
        trainer = ZeroSyncGRPOTrainer(
            model="trl-internal-testing/small-Qwen2ForCausalLM-2.5",
            reward_funcs=counting_reward_func,
            args=training_args,
            train_dataset=dataset,
        )

        trainer.train()

        # The group's first rank runs the rollouts and scores them; the others train on the batch it sends them
        if int(os.environ["RANK"]) == 0:
            assert len(scored) >= 2 * 3 * 2  # two steps of two micro-batches of three samples
        else:
            assert scored == []

        # Each rank's shard of the fp32 copies goes back to host memory after the step
        for group in trainer.optimizer.param_groups:
            for master in group["params"]:
                local = master.to_local() if isinstance(master, DTensor) else master
                assert local.dtype == torch.float32
                assert local.is_pinned()
