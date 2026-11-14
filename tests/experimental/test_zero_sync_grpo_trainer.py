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
from collections.abc import Callable
from unittest.mock import patch

import pytest
import torch
import transformers
from accelerate import PartialState
from datasets import Dataset, DatasetDict, load_dataset
from packaging.version import Version
from torch.distributed.tensor import DTensor
from transformers import AutoModelForCausalLM, AutoTokenizer

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


def multiply_tool(a: int, b: int) -> int:
    """
    Multiplies two integers.

    Args:
        a: The first integer.
        b: The second integer.

    Returns:
        The product of the two integers.
    """
    return a * b


async def async_multiply_tool(a: int, b: int) -> int:
    """
    Asynchronously multiplies two integers.

    Args:
        a: The first integer.
        b: The second integer.

    Returns:
        The product of the two integers.
    """
    return a * b


def unique_letters_reward(completions, **kwargs):
    """Reward function that rewards completions with more unique letters."""
    return [float(len(set(completion[0]["content"]))) for completion in completions]


async def async_unique_letters_reward(completions, **kwargs):
    """Async Reward function that rewards completions with more unique letters."""
    return [float(len(set(completion[0]["content"]))) for completion in completions]


class AsyncCallableReward:
    """Async reward function written as a callable class."""

    async def __call__(self, completions, **kwargs):
        return [float(completion[0]["content"].count(" ")) for completion in completions]


def distance_to_value_reward(completions, some_values, **kwargs):
    """Reward function that rewards completions with lengths closer to the values in some_values."""
    return [
        float(abs(len(completion[0]["content"]) - value))
        for completion, value in zip(completions, some_values, strict=True)
    ]


def wrong_length_reward(completions, **kwargs):
    return [1.0]  # too few: one reward is expected per completion


def first_none_reward(completions, **kwargs):
    # None for the first completion of each group only: a group whose rewards are all None is dropped, and with every
    # group dropped no step completes
    return [None] + [float(i) for i in range(1, len(completions))]


def str_length_reward(completions, **kwargs):
    """Reward function that rewards longer completions, whatever their format."""
    return [float(len(str(completion))) for completion in completions]


class IncrementEnvironment:
    def reset(self, **kwargs):
        self._counter = 0

    def increment(self, step: int) -> int:
        """
        Increment the internal counter.

        Args:
            step: Value to add to the counter.

        Returns:
            The updated counter value.
        """
        self._counter += step
        return self._counter


class GuessTheWordEnvironment:
    def reset(self, **kwargs):
        return "Guess the 5-letter word."  # the env owns the data and returns the prompt


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
        "reward_funcs": [str_length_reward],
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
        assert _reduce_metric("sample/dropped_stale_count", [1.0, 1.0, 1.0]) == 3.0

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
        assert _reduce_metric(
            "perf/wall_clock_forwarded_tokens_per_s", [(100.0, 100.0), (100.0, 1.0)]
        ) == pytest.approx(200 / 101)

    def test_rate_with_zero_denominator_is_nan(self):
        assert math.isnan(_reduce_metric("perf/wall_clock_forwarded_tokens_per_s", [(0.0, 0.0)]))


class TestMetricPush(TrlTestCase):
    """The loop's payload has the same shape as the trainer's sink, so pushing it is an append."""

    def test_counters_and_rates_ride_along_and_reset(self):
        loop = _rollout_loop()
        loop._counters["tools/search/call_count"] += 2
        loop._rates["tools/latency_s"][0] += 3.0
        loop._rates["tools/latency_s"][1] += 2
        loop._push_metrics({"rollout/score_s": 0.5})

        assert loop.metrics == {
            "rollout/score_s": [0.5],
            "tools/search/call_count": [2.0],
            "tools/latency_s": [(3.0, 2.0)],
        }
        # Counters carry deltas: whatever was pushed must not be pushed again.
        loop._push_metrics({})
        assert loop.metrics == {
            "rollout/score_s": [0.5],
            "tools/search/call_count": [2.0],
            "tools/latency_s": [(3.0, 2.0)],
        }

    @pytest.mark.parametrize(
        ("ends_on_eos", "clipped"),
        [
            (True, 0.0),  # ends on eos: the model stopped on its own
            (False, 1.0),  # ends mid-sentence: cut off by max_completion_length
        ],
    )
    def test_rollout_push_reports_the_completion(self, ends_on_eos, clipped):
        loop = _rollout_loop()
        completion_ids = [7, 8, loop.eos_token_ids[0] if ends_on_eos else 9]
        loop._push_rollout_metrics(turns=1, sequences=1, completion_ids=completion_ids, drifts=[], duration_s=1.5)
        metrics = loop.metrics

        assert metrics["completions/mean_length"] == [(3.0, 1.0)]
        assert metrics["completions/max_length"] == [3.0]
        assert metrics["completions/clipped_ratio"] == [(clipped, 1.0)]
        # Only a completion that ended on EOS counts in the terminated lengths, as in GRPO
        assert ("completions/mean_terminated_length" in metrics) is ends_on_eos
        if ends_on_eos:
            assert metrics["completions/mean_terminated_length"] == [(3.0, 1.0)]
            assert (
                metrics["completions/min_terminated_length"] == metrics["completions/max_terminated_length"] == [3.0]
            )


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
        assert loop._counters["tools/failing_tool/failure_count"] == 1
        assert loop._counters["tools/unknown_name_count"] == 1
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

    def test_chat_template_without_response_parser(self):
        # TRL has no response parser for this chat template. Only tool calls need one, so without tools the turn is
        # decoded as plain text, as in GRPO
        tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-Qwen3ForCausalLM")
        loop = _rollout_loop(processing_class=tokenizer, eos_token_ids=[tokenizer.eos_token_id])
        output_ids = tokenizer.encode("Hello!", add_special_tokens=False) + [tokenizer.eos_token_id]

        async def generate_one_turn(prompt_ids):
            return TurnRecord(prompt_ids, output_ids, [-0.1] * len(output_ids))

        loop.generate_one_turn = generate_one_turn
        rollout = asyncio.run(loop.generate_one([{"role": "user", "content": "hi"}], []))
        assert rollout.completion == [{"role": "assistant", "content": "Hello!"}]

    def test_known_chat_template_is_parsed_without_tools(self):
        # TRL has a response parser for this chat template, so the turn is parsed even without tools: the thinking goes
        # to `reasoning_content`, and `content` holds the answer
        loop = _rollout_loop()
        output_ids = loop.tokenizer.encode("<think>\nLet me add.\n</think>\n\n4<|im_end|>", add_special_tokens=False)

        async def generate_one_turn(prompt_ids):
            return TurnRecord(prompt_ids, output_ids, [-0.1] * len(output_ids))

        loop.generate_one_turn = generate_one_turn
        rollout = asyncio.run(loop.generate_one([{"role": "user", "content": "2+2?"}], []))
        assert rollout.completion == [{"role": "assistant", "reasoning_content": "Let me add.", "content": "4"}]

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

        loop = _rollout_loop(reward_funcs=[AsyncTwoReward()])
        samples = asyncio.run(loop._score_group(group))

        # Run as a synchronous function, the reward is an un-awaited coroutine and never a number.
        assert loop._rates["rewards/AsyncTwoReward/mean"] == [4.0, 2]  # rewards 1.0 and 3.0
        assert samples[0].advantage == pytest.approx(-1.0)
        assert samples[1].advantage == pytest.approx(1.0)

    def test_metrics_are_per_conversation_and_independent(self):
        seq_a = TrainingSequence([1, 2, 10], [0, 0, 1], [0, 0, -0.1], [[]] * 3, [{}] * 3, "c0")
        seq_b1 = TrainingSequence([1, 2, 20], [0, 0, 1], [0, 0, -0.2], [[]] * 3, [{}] * 3, "c1")
        seq_b2 = TrainingSequence([1, 2, 20, 30], [0, 0, 0, 1], [0, 0, 0, -0.3], [[]] * 4, [{}] * 4, "c1")
        group = _group([[seq_a], [seq_b1, seq_b2]], completions_ids=[[10], [20, 30]])

        loop = _rollout_loop(reward_funcs=[two_reward])
        samples = asyncio.run(loop._score_group(group))

        assert samples[0].metrics["reward_trained"] == 1.0
        assert samples[1].metrics["reward_trained"] == 3.0 and samples[2].metrics["reward_trained"] == 3.0
        assert samples[0].metrics["reward_group_std"] == pytest.approx(1.0)
        # Each reward function's mean counts the forked conversation once, as GRPO counts a completion once
        assert loop._rates["rewards/two_reward/mean"] == [4.0, 2]  # rewards 1.0 and 3.0
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
        assert math.isnan(samples[0].metrics["reward_trained"])
        assert samples[1].advantage < 0 < samples[2].advantage  # advantage over the scorable rows only
        assert samples[1].metrics["reward_trained"] == 2.0


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
            # Every tool call of the dropped group fails: it must still count in the tool metrics.
            group.tool_call_counts = [2, 2] if group_id == 0 else [1, 0]
            group.tool_failure_counts = [2, 2] if group_id == 0 else [0, 0]
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
        assert metrics["frac_zero_std_groups"] == [(1.0, 1), (0.0, 1)]
        assert metrics["reward"] == [reward, (4.0, 2)]  # the dropped group still counts in the policy's reward
        # So does each reward function's mean, except for a group the function didn't score
        assert metrics["rewards/str_length_reward/mean"] == ([reward, (4.0, 2)] if reward[1] else [(4.0, 2)])
        assert metrics["tools/call_frequency"] == [(4, 2), (1, 2)]
        assert metrics["tools/failure_frequency"] == [(4, 4), (0, 1)]

    def test_reward_zero_std_groups_and_duplicate_frac(self):
        # The first token of each completion is its reward, and 0 means the reward func cannot score it.
        def first_token_reward(completion_ids, **kwargs):
            return [float(ids[0]) if ids[0] else None for ids in completion_ids]

        loop = _rollout_loop(reward_funcs=[first_token_reward])
        seqs = [[TrainingSequence([1, 2, 10], [0, 0, 1], [0, 0, -0.1], [[]] * 3, [{}] * 3, f"c{i}")] for i in range(2)]
        asyncio.run(loop._score_group(_group(seqs, completions_ids=[[1, 10], [2, 10]])))  # kept
        asyncio.run(loop._score_group(_group(seqs, completions_ids=[[3, 10], [3, 10]])))  # dropped, duplicated
        asyncio.run(loop._score_group(_group(seqs, completions_ids=[[0, 10], [4, 10]])))  # dropped, one scored
        asyncio.run(loop._score_group(_group(seqs, completions_ids=[[0, 10], [0, 20]])))  # dropped, unscorable

        assert loop._rates["frac_zero_std_groups"] == [3, 4]
        # Only the dropped groups with a scored completion contribute the reward they are stuck at.
        assert loop._rates["reward_zero_std_groups"] == [3.0 + 4.0, 2]
        # One of the eight completions repeats an earlier one of its group.
        assert loop._rates["completions/duplicate_frac"] == [1, 8]


@require_torch_accelerator
class TestZeroSyncGRPOTrainer(TrlTestCase):
    def test_init_minimal(self):
        # Test that ZeroSyncGRPOTrainer can be instantiated with only model, reward function and train_dataset
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_only", split="train")
        ZeroSyncGRPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=str_length_reward,
            train_dataset=dataset,
        )

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
            reward_funcs=str_length_reward,
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None
        logged = trainer.state.log_history[0]
        for key in [
            "sampling/importance_sampling_ratio/min",
            "sampling/importance_sampling_ratio/mean",
            "sampling/importance_sampling_ratio/max",
            "sampling/sampling_logp_difference/mean",
            "sampling/sampling_logp_difference/max",
        ]:
            assert key in logged
        assert logged["num_tokens"] > 0

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    @pytest.mark.parametrize(
        "model_id, expect_coef, expect_aux_loss",
        [
            # MoE whose forward returns an auxiliary loss: the architecture's own coefficient is applied
            ("trl-internal-testing/tiny-Qwen3MoeForCausalLM", 0.001, True),
            # MoE that balances its experts with a router bias: it declares no coefficient, so the term stays off
            ("trl-internal-testing/tiny-DeepseekV3ForCausalLM", 0.0, False),
            # Dense model: no coefficient to inherit
            ("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", 0.0, False),
        ],
    )
    def test_router_aux_loss_coef_defaults_to_the_architecture(self, model_id, expect_coef, expect_aux_loss):
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_only", split="train[:2]")
        training_args = ZeroSyncGRPOConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = ZeroSyncGRPOTrainer(
            model=model_id, reward_funcs=str_length_reward, args=training_args, train_dataset=dataset
        )

        assert trainer.router_aux_loss_coef == expect_coef
        assert trainer.aux_loss_enabled == expect_aux_loss

    def test_router_aux_loss_coef_fails_without_router_logits(self):
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_only", split="train[:2]")
        training_args = ZeroSyncGRPOConfig(output_dir=self.tmp_dir, router_aux_loss_coef=0.5, report_to="none")

        # Dense model: no `output_router_logits` on its config, so there is nothing to compute the term from
        with pytest.raises(ValueError, match="not a Mixture-of-Experts model"):
            ZeroSyncGRPOTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                reward_funcs=str_length_reward,
                args=training_args,
                train_dataset=dataset,
            )

    def test_router_aux_loss_coef_explicit_value_overrides_the_architecture(self):
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_only", split="train[:2]")
        training_args = ZeroSyncGRPOConfig(output_dir=self.tmp_dir, router_aux_loss_coef=0.5, report_to="none")

        trainer = ZeroSyncGRPOTrainer(
            model="trl-internal-testing/tiny-Qwen3MoeForCausalLM",
            reward_funcs=str_length_reward,
            args=training_args,
            train_dataset=dataset,
        )

        assert trainer.router_aux_loss_coef == 0.5
        assert trainer.aux_loss_enabled

    def test_trust_remote_code(self):
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_only", split="train")
        model_id = "trl-internal-testing/tiny-RemoteForCausalLM"

        with pytest.raises(ValueError, match="custom code"):
            ZeroSyncGRPOTrainer(
                model=model_id,
                args=ZeroSyncGRPOConfig(output_dir=self.tmp_dir, report_to="none"),
                reward_funcs=str_length_reward,
                train_dataset=dataset,
            )

        trainer = ZeroSyncGRPOTrainer(
            model=model_id,
            args=ZeroSyncGRPOConfig(output_dir=self.tmp_dir, report_to="none", trust_remote_code=True),
            reward_funcs=str_length_reward,
            train_dataset=dataset,
        )
        assert type(trainer.model).__name__ == "RemoteForCausalLM"

    def test_logps_match_plain_forward(self):
        # The fused LM head scores the completion tokens like a plain forward of the model, at the sampling temperature
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_only", split="train")
        training_args = ZeroSyncGRPOConfig(
            output_dir=self.tmp_dir,
            bf16=False,  # compare in full precision
            temperature=0.7,
            model_init_kwargs={"dtype": "float32"},
            report_to="none",
        )
        trainer = ZeroSyncGRPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=str_length_reward,
            args=training_args,
            train_dataset=dataset,
        )
        # One unpacked sequence, so the plain attention of the reference forward serves both
        trainer.model.set_attn_implementation("sdpa")
        input_ids = torch.randint(0, trainer.model.config.vocab_size, (1, 16), device=trainer.accelerator.device)
        shift_labels = torch.cat([input_ids[:, 1:], torch.full_like(input_ids[:, :1], -100)], dim=1)

        with torch.no_grad():
            outputs = trainer.model(input_ids=input_ids, shift_labels=shift_labels, fused_lm_head=True)
            logits = trainer.model(input_ids=input_ids).logits[:, :-1] / 0.7
        ref_logps = logits.log_softmax(-1)

        expected_logps = ref_logps.gather(-1, input_ids[:, 1:, None]).squeeze(-1)
        expected_entropies = -(ref_logps.exp() * ref_logps).sum(-1)
        torch.testing.assert_close(outputs.log_probs[:, :-1], expected_logps, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(outputs.entropy[:, :-1], expected_entropies, atol=1e-4, rtol=1e-4)

    def test_train_with_iterable_dataset(self):
        # Iterable (streaming) datasets have no length, so `max_steps` is required.
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_only", split="train", streaming=True)

        training_args = ZeroSyncGRPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            max_concurrent_rollouts=3,  # keep few rollouts generating, the test model is tiny
            max_steps=4,
            report_to="none",
        )
        trainer = ZeroSyncGRPOTrainer(
            model="trl-internal-testing/small-Qwen2ForCausalLM-2.5",
            reward_funcs=str_length_reward,
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

    @pytest.mark.parametrize("train_dataset_type", ["dataset", "iterable_dataset", "none", "unsupported_dataset_dict"])
    def test_init_with_train_dataset(self, train_dataset_type):
        streaming = "iterable" in train_dataset_type
        if train_dataset_type == "none":
            train_dataset = None
        else:
            train_dataset = load_dataset(
                "trl-internal-testing/zen", "conversational_prompt_only", split="train", streaming=streaming
            )
            if train_dataset_type == "unsupported_dataset_dict":
                # `DatasetDict` is representative of any unsupported type here; not exhaustive
                train_dataset = DatasetDict({"train": train_dataset})

        # Iterable (streaming) datasets have no length, so `max_steps` is required.
        training_args = ZeroSyncGRPOConfig(output_dir=self.tmp_dir, max_steps=4 if streaming else -1, report_to="none")

        if train_dataset_type == "none":
            with pytest.raises(ValueError, match="`train_dataset` is required"):
                ZeroSyncGRPOTrainer(
                    model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                    reward_funcs=str_length_reward,
                    args=training_args,
                    train_dataset=train_dataset,
                )
        elif train_dataset_type == "unsupported_dataset_dict":
            with pytest.raises(TypeError, match="`train_dataset` must be a `Dataset` or `IterableDataset`"):
                ZeroSyncGRPOTrainer(
                    model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                    reward_funcs=str_length_reward,
                    args=training_args,
                    train_dataset=train_dataset,
                )
        else:
            trainer = ZeroSyncGRPOTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                reward_funcs=str_length_reward,
                args=training_args,
                train_dataset=train_dataset,
            )
            assert trainer.train_dataset is train_dataset

    def test_train_model(self):
        # The engine decodes from the instance's own weights, so it keeps the checkpoint dtype
        model = AutoModelForCausalLM.from_pretrained("trl-internal-testing/small-Qwen2ForCausalLM-2.5", dtype="auto")
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
            model=model, reward_funcs=str_length_reward, args=training_args, train_dataset=dataset
        )
        # The instance is trained in place
        assert trainer.model is model

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    @pytest.mark.xfail(
        reason="The engine decodes through the mixed-precision forward, whose keys and values do not match a KV cache "
        "in another dtype than the training precision",
        strict=True,
    )
    def test_train_model_dtype(self):
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_only", split="train")

        training_args = ZeroSyncGRPOConfig(
            output_dir=self.tmp_dir,
            model_init_kwargs={"dtype": torch.float16},
            # Keeps the fp16 model's completions diverse: once they all match, every group is dropped and no step completes
            learning_rate=1e-3,
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            max_concurrent_rollouts=3,  # keep few rollouts generating, the test model is tiny
            max_steps=2,
            report_to="none",
        )
        trainer = ZeroSyncGRPOTrainer(
            model="trl-internal-testing/small-Qwen2ForCausalLM-2.5",
            reward_funcs=str_length_reward,
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            # Check the torch dtype
            assert new_param.dtype == torch.float16
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    @pytest.mark.xfail(
        reason="The engine decodes through the mixed-precision forward, whose keys and values do not match a KV cache "
        "in another dtype than the training precision",
        strict=True,
    )
    def test_train_with_fp16(self):
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_only", split="train")

        training_args = ZeroSyncGRPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            max_concurrent_rollouts=3,  # keep few rollouts generating, the test model is tiny
            max_steps=2,
            fp16=True,
            report_to="none",
        )
        trainer = ZeroSyncGRPOTrainer(
            model="trl-internal-testing/small-Qwen2ForCausalLM-2.5",
            reward_funcs=str_length_reward,
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

    def test_train_multiple_reward_funcs(self):
        # Test that ZeroSyncGRPOTrainer can be instantiated with multiple reward functions
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
            reward_funcs=[str_length_reward, unique_letters_reward],
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

    def test_train_sync_and_async_reward_funcs(self):
        # Test that ZeroSyncGRPOTrainer can be instantiated with multiple reward functions, one of which is async, in
        # both the function and the callable class form.
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
            reward_funcs=[str_length_reward, constant_reward, async_unique_letters_reward, AsyncCallableReward()],
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

    def test_train_multiple_reward_funcs_with_None_output(self):
        """Test that a valid math reward function is processed correctly while the code reward function returns None."""
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
            reward_funcs=[str_length_reward, all_none_reward],  # One applicable, one non applicable
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = {
            n: param.clone() for n, param in trainer.model.named_parameters() if param.requires_grad
        }

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_train_reward_func_additional_column(self):
        # Test if trainer can handle reward function that rely on additional columns in the dataset
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_only", split="train")

        # Add a column to the dataset (dummy example, the column could be anything)
        some_values = list(range(len(dataset)))
        dataset = dataset.add_column("some_values", some_values)

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
            reward_funcs=distance_to_value_reward,
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

    def test_reward_func_wrong_number_of_rewards(self):
        # A reward function that returns the wrong number of rewards should raise a clear error instead of silently
        # broadcasting (when it returns a single value) or failing later with an opaque shape error.
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_only", split="train")

        training_args = ZeroSyncGRPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            max_concurrent_rollouts=3,  # keep few rollouts generating, the test model is tiny
            report_to="none",
        )
        trainer = ZeroSyncGRPOTrainer(
            model="trl-internal-testing/small-Qwen2ForCausalLM-2.5",
            reward_funcs=wrong_length_reward,
            args=training_args,
            train_dataset=dataset,
        )
        # The rollout loop scores in the training process, so its error is raised as is
        with pytest.raises(ValueError, match="returned 1 rewards"):
            trainer.train()

    def test_train_with_gradient_checkpointing(self):
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_only", split="train")

        training_args = ZeroSyncGRPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            max_concurrent_rollouts=3,  # keep few rollouts generating, the test model is tiny
            max_steps=2,
            gradient_checkpointing=True,
            report_to="none",
        )
        trainer = ZeroSyncGRPOTrainer(
            model="trl-internal-testing/small-Qwen2ForCausalLM-2.5",
            reward_funcs=str_length_reward,
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

    def test_train_with_gradient_accumulation(self):
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_only", split="train")

        training_args = ZeroSyncGRPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            gradient_accumulation_steps=2,  # accumulate over 2 micro-batches
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            max_concurrent_rollouts=3,  # keep few rollouts generating, the test model is tiny
            max_steps=2,
            report_to="none",
        )
        trainer = ZeroSyncGRPOTrainer(
            model="trl-internal-testing/small-Qwen2ForCausalLM-2.5",
            reward_funcs=str_length_reward,
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

    def test_warning_raised_all_rewards_none(self, caplog):
        """Test that a proper warning is raised when all rewards are None."""
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
            reward_funcs=first_none_reward,
            args=training_args,
            train_dataset=dataset,
        )

        # The rollout loop scores in the training process, as in GRPO
        with caplog.at_level("WARNING", logger="trl.experimental.zero_sync_grpo.zero_sync_grpo_trainer"):
            trainer.train()

        expected_warning = "All reward functions returned None for the following kwargs:"
        assert expected_warning in caplog.text

    def test_train_num_generations_larger_than_batch_size(self):
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_only", split="train")

        training_args = ZeroSyncGRPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            num_generations=6,  # the number of generations is larger than the batch size, but
            gradient_accumulation_steps=2,  # gradient accumulation should allow that
            max_concurrent_rollouts=6,  # keep few rollouts generating, the test model is tiny
            max_steps=2,
            report_to="none",
        )
        trainer = ZeroSyncGRPOTrainer(
            model="trl-internal-testing/small-Qwen2ForCausalLM-2.5",
            reward_funcs=str_length_reward,
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

    def test_train_with_chat_template_kwargs(self):
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
            chat_template_kwargs={"enable_thinking": False},
        )
        trainer = ZeroSyncGRPOTrainer(
            model="trl-internal-testing/small-Qwen3ForCausalLM",
            reward_funcs=str_length_reward,
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

    @require_response_parsing
    @pytest.mark.parametrize("tools", [[multiply_tool], [async_multiply_tool]])
    def test_train_with_tools(self, tools: list[Callable]):
        # The engine samples from the model, so unlike AsyncGRPO's test no tool call is scripted: this checks that
        # training runs with tools. The tool loop itself is covered by `TestRolloutLoop`.
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
            model="trl-internal-testing/tiny-Qwen3MoeForCausalLM",
            reward_funcs=str_length_reward,
            args=training_args,
            train_dataset=dataset,
            tools=tools,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    @pytest.mark.skipif(
        Version(transformers.__version__) < Version("5.13.0.dev0"),
        reason="Tool parsing relies on the legacy jmespath-based `response_schema` parser below transformers 5.13.0",
    )
    def test_tools_without_jmespath(self):
        # jmespath is only used by the legacy `response_schema` parser (transformers < 5.13); the new-style
        # `response_template` parser doesn't need it. The guard must therefore not fire on newer transformers, where
        # jmespath is never imported. Guards against an over-broad regression: none of the openenv examples install
        # jmespath, so a spurious guard breaks them before training starts.
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_only", split="train")
        training_args = ZeroSyncGRPOConfig(output_dir=self.tmp_dir, report_to="none")
        with patch("trl.experimental.zero_sync_grpo.zero_sync_grpo_trainer.is_jmespath_available", return_value=False):
            ZeroSyncGRPOTrainer(  # must construct without raising
                model="trl-internal-testing/tiny-Qwen3MoeForCausalLM",
                reward_funcs=str_length_reward,
                args=training_args,
                train_dataset=dataset,
                tools=[multiply_tool],
            )

    @require_response_parsing
    def test_train_with_unknown_environment_raises(self):
        # With a dict `environment_factory`, an example whose `environment` field doesn't match any configured
        # environment should fail with a clear error rather than a bare KeyError mid-rollout.
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_only", split="train")
        dataset = dataset.map(lambda example: {"environment": "unknown"})

        training_args = ZeroSyncGRPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_concurrent_rollouts=3,  # keep few rollouts generating, the test model is tiny
            report_to="none",
        )

        trainer = ZeroSyncGRPOTrainer(
            model="trl-internal-testing/tiny-Qwen3MoeForCausalLM",
            reward_funcs=str_length_reward,
            args=training_args,
            train_dataset=dataset,
            environment_factory={"counter": IncrementEnvironment},
        )

        # The rollout loop runs in the training process, so its error is raised as is
        with pytest.raises(ValueError, match="not among the environments"):
            trainer.train()

    @require_response_parsing
    def test_train_with_environment_factory(self):
        # The engine samples from the model, so unlike AsyncGRPO's test no tool call is scripted: this checks that
        # training runs with an environment. The environment's tools are covered by `TestRolloutLoopEnvironments`.
        training_args = ZeroSyncGRPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            max_steps=2,  # required with environment_factory
            max_concurrent_rollouts=3,  # keep few rollouts generating, the test model is tiny
            report_to="none",
        )

        trainer = ZeroSyncGRPOTrainer(
            model="trl-internal-testing/tiny-Qwen3MoeForCausalLM",
            reward_funcs=str_length_reward,
            args=training_args,
            environment_factory=IncrementEnvironment,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_no_reward_source_raises(self):
        # `reward_funcs` is optional, but a reward must come from somewhere: with neither `reward_funcs` nor an
        # environment-owned `get_reward`, construction fails fast instead of training on an all-zero reward.
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_only", split="train")
        training_args = ZeroSyncGRPOConfig(output_dir=self.tmp_dir, report_to="none")
        with pytest.raises(ValueError, match="No reward source provided"):
            ZeroSyncGRPOTrainer(
                model="trl-internal-testing/tiny-Qwen3ForCausalLM",
                args=training_args,
                train_dataset=dataset,
            )

    def test_dataset_required_without_environment(self):
        # The data has to come from somewhere: an external `train_dataset`, or an environment that owns it. With
        # neither, construction fails fast.
        training_args = ZeroSyncGRPOConfig(output_dir=self.tmp_dir, report_to="none")
        with pytest.raises(ValueError, match="`train_dataset` is required"):
            ZeroSyncGRPOTrainer(
                model="trl-internal-testing/tiny-Qwen3ForCausalLM",
                reward_funcs=str_length_reward,
                args=training_args,
            )

    def test_remove_unused_columns_raises(self):
        # Every dataset column is passed to the reward functions, so unneeded columns are removed from the dataset,
        # not by the trainer
        with pytest.raises(ValueError, match="remove_unused_columns"):
            ZeroSyncGRPOConfig(output_dir=self.tmp_dir, remove_unused_columns=True)

    @require_response_parsing
    def test_environment_owned_data_requires_max_steps(self):
        # When the environment owns the data and no external `train_dataset` is provided, `max_steps` must set the
        # training length.
        class DummyEnvironment:
            def reset(self, **kwargs):
                return "Guess the 5-letter word."

        training_args = ZeroSyncGRPOConfig(output_dir=self.tmp_dir, report_to="none")  # max_steps unset
        with pytest.raises(ValueError, match="max_steps"):
            ZeroSyncGRPOTrainer(
                model="trl-internal-testing/tiny-Qwen3ForCausalLM",
                reward_funcs=str_length_reward,
                args=training_args,
                environment_factory=DummyEnvironment,
            )

    @require_response_parsing
    def test_multiple_environments_without_dataset_raises(self):
        # Multiple environments (a `dict` factory) need a `train_dataset` with an `environment` column to route each
        # example. Without one, there is no per-rollout selector, so construction fails fast with a clear message.
        class EnvA:
            def reset(self, **kwargs): ...

        class EnvB:
            def reset(self, **kwargs): ...

        training_args = ZeroSyncGRPOConfig(output_dir=self.tmp_dir, max_steps=1, report_to="none")
        with pytest.raises(ValueError, match="requires a `train_dataset`"):
            ZeroSyncGRPOTrainer(
                model="trl-internal-testing/tiny-Qwen3ForCausalLM",
                reward_funcs=str_length_reward,
                args=training_args,
                environment_factory={"a": EnvA, "b": EnvB},
            )

    @require_response_parsing
    def test_train_with_environment_owned_data(self):
        # The environment owns the data — no external `train_dataset` is needed. `reset()` returns the prompt, and
        # `max_steps` drives the training loop. A placeholder dataset is built internally.
        training_args = ZeroSyncGRPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_steps=2,  # the environment owns the data, so `max_steps` sets the training length
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            max_concurrent_rollouts=3,  # keep few rollouts generating, the test model is tiny
            report_to="none",
        )

        trainer = ZeroSyncGRPOTrainer(
            model="trl-internal-testing/tiny-Qwen3ForCausalLM",
            reward_funcs=str_length_reward,
            args=training_args,
            environment_factory=GuessTheWordEnvironment,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_init_auto_processing_class_uses_model_revision(self):
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_only", split="train")
        trainer = ZeroSyncGRPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=str_length_reward,
            args=ZeroSyncGRPOConfig(
                output_dir=self.tmp_dir,
                report_to="none",
                model_init_kwargs={"revision": "8913f5819566"},
            ),
            train_dataset=dataset,
        )
        # This revision's chat template is 2558 chars; the one on `main` is 2507. Comparing the length is enough to
        # catch the tokenizer being loaded from the default branch instead of the pinned revision.
        assert len(trainer.processing_class.chat_template) == 2558

    def test_pad_token_id_synced_with_model_config(self):
        # This model's tokenizer has no pad token, so the trainer falls back to the eos token. The model configs must
        # follow: otherwise `Trainer` realigns them at train time and reports it as a change the user did not make.
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_only", split="train")

        training_args = ZeroSyncGRPOConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = ZeroSyncGRPOTrainer(
            model="trl-internal-testing/tiny-MistralForCausalLM-0.2",
            reward_funcs=str_length_reward,
            args=training_args,
            train_dataset=dataset,
        )

        pad_token_id = trainer.processing_class.pad_token_id
        assert pad_token_id is not None
        assert trainer.model.config.pad_token_id == pad_token_id
        assert trainer.model.generation_config.pad_token_id == pad_token_id

    def test_tag_added(self):
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_only", split="train")

        trainer = ZeroSyncGRPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=str_length_reward,
            train_dataset=dataset,
        )

        for tag in ["zero-sync-grpo", "trl"]:
            assert tag in trainer.model.model_tags

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
            reward_funcs=str_length_reward,
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
            reward_funcs=str_length_reward,
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
            return str_length_reward(completions, **kwargs)

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
