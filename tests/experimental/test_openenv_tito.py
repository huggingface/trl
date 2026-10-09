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

"""Captured contexts and eligible sampled tokens must survive trainer ingestion exactly."""

import asyncio
import itertools
import queue
import threading
import time
from collections import defaultdict
from functools import partial
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from accelerate import PartialState


pytest.importorskip("openenv.core.harness.training")

from openenv.core.harness import HarnessRolloutResult, ToolResult, ToolTraceEntry, TrainingTrace

from trl.experimental.async_grpo import openenv_harness
from trl.experimental.async_grpo.async_rollout_worker import _chain_to_sequences, _SampleBuilder


def entry(prompt, completion, logprobs, mask=None):
    return {
        "prompt_token_ids": prompt,
        "completion_token_ids": completion,
        "per_token_logps": logprobs,
        "loss_mask": mask if mask is not None else [0] * len(prompt) + [1] * len(completion),
    }


def convert(entries):
    for index, row in enumerate(entries):
        row.setdefault("metadata", {})["node_id"] = str(index)
    return openenv_harness._turns_from_training_trace(TrainingTrace.from_entries(entries))


def test_lossless_capture_preserves_rewritten_tail_and_partial_mask():
    entries = [
        entry([1, 2], [3, 4, 5], [-0.1, -0.2, -0.3], [0, 0, 1, 0, 1]),
        entry([1, 2, 3, 4, 99, 6], [7], [-0.4], [0, 0, 0, 0, 0, 0, 1]),
    ]
    turns = convert(entries)
    rows, tally = _chain_to_sequences(turns, "rollout", 0)
    assert len(rows) == 2 and tally["realign"] == 0
    assert rows[0].input_ids == [1, 2, 3, 4, 5]
    assert rows[0].completion_mask == [0, 0, 1, 0, 1]
    assert rows[0].old_log_probs[-3:] == [-0.1, -0.2, -0.3]
    assert rows[1].input_ids == [1, 2, 3, 4, 99, 6, 7]
    assert rows[1].completion_mask == [0, 0, 0, 0, 0, 0, 1]
    assert sum(sum(row.completion_mask) for row in rows) == 3


@pytest.mark.parametrize("logprobs", [[], None, [float("nan")], [float("inf")], [0.1], [True]])
def test_missing_or_invalid_sample_logprobs_cannot_be_filled_with_zeros(logprobs):
    with pytest.raises(ValueError):
        convert([entry([1], [2], logprobs)])


@pytest.mark.parametrize("mask,logprobs", [(1, None), (1, []), ([1, 0], [-0.1]), (0, [])])
def test_builder_rejects_misaligned_arrays_before_mutation(mask, logprobs):
    builder = _SampleBuilder(fork_threshold=0)
    with pytest.raises(ValueError):
        builder._append([2], mask=mask, logprobs=logprobs)
    assert builder.tokens == builder.loss_mask == builder.logprobs == []


@pytest.mark.parametrize("mask", [[1, 1], [0], [0, 2], [0, True]])
def test_consumer_rejects_invalid_full_masks(mask):
    with pytest.raises(ValueError):
        convert([entry([1], [2], [-0.1], mask)])


@pytest.mark.parametrize(
    "adapter,lossless,threshold", [(None, True, 0), (None, False, 1024), (SimpleNamespace(), True, 1024)]
)
def test_loop_owning_capture_defaults_to_lossless_reconciliation(monkeypatch, adapter, lossless, threshold):
    def init(self, **kwargs):
        self._fork_threshold_tokens = kwargs.get("fork_threshold_tokens", 1024)
        self.max_tool_calling_iterations = 8
        self.temperature = 0.8
        self.top_p = 1.0
        self.top_k = -1
        self.min_p = 0.0
        self.repetition_penalty = 1.0
        self.max_tokens = 32
        self.reward_func_names = []
        self.max_inflight_tasks = 1

    monkeypatch.setattr(openenv_harness._AsyncRolloutLoop, "__init__", init)
    factory = MagicMock(return_value=SimpleNamespace())
    loop = openenv_harness._HarnessRolloutLoop(
        harness_session_factory=factory, harness_adapter=adapter, lossless_capture=lossless
    )
    try:
        assert loop._fork_threshold_tokens == threshold
        if adapter is not None:
            factory.assert_called_once_with()
        else:
            factory.assert_called_once_with(sampling=openenv_harness.training_sampling({"temperature": 0.8}))
    finally:
        loop._session_pool.shutdown()


@pytest.fixture
def make_loop():
    PartialState()
    loops = []

    def make(factory=None, *, factory_builder=None, **kwargs):
        kwargs.setdefault("temperature", 0.8)
        tokenizer = MagicMock(eos_token_id=0, pad_token_id=0)
        loop = openenv_harness._HarnessRolloutLoop(
            harness_session_factory=factory_builder or (lambda **kwargs: factory),
            model_name="test",
            dataset=[{"prompt": [{"role": "user", "content": "task"}]}],
            reward_funcs=[],
            processing_class=tokenizer,
            eos_token_ids=[0],
            rollout_buffer=queue.Queue(),
            model_version_value=SimpleNamespace(value=0),
            heartbeat_value=SimpleNamespace(value=0.0),
            failed_event=threading.Event(),
            exception_info_queue=queue.Queue(),
            metrics_queue=queue.Queue(),
            max_inflight_tasks=8,
            **kwargs,
        )
        loops.append(loop)
        return loop

    yield make
    for loop in loops:
        loop._session_pool.shutdown(wait=True)
        loop._loop.close()


def captured_entry():
    row = entry([1], [2, 3], [-0.1, -0.2], [0, 1, 0])
    row.update(
        request={"messages": [{"role": "user", "content": "task"}]},
        response={"choices": [{"message": {"role": "assistant", "content": "answer"}}]},
        metadata={"node_id": "0", "sampling_params": openenv_harness.training_sampling({"temperature": 0.8})},
    )
    return row


class Session:
    def __init__(self, row):
        self.row = row
        self.closed = threading.Event()

    def wait_for_completion(self):
        return 0

    def fetch_training_trace(self):
        row = {**self.row, "metadata": {**self.row.get("metadata", {}), "node_id": "0"}}
        return TrainingTrace.from_entries([row])

    def verify(self, completion):
        return SimpleNamespace(env_reward=0.25)

    def close(self):
        self.closed.set()


@pytest.mark.parametrize("temperature", [0.3, 0.8, 1.2])
def test_worker_supplies_factory_sampling_before_session_start(make_loop, monkeypatch, temperature):
    harness = pytest.importorskip("harbor_env.harness")
    policy = openenv_harness.training_sampling({"temperature": temperature})
    row = captured_entry()
    # Token ingestion no longer depends on redundant policy metadata.
    row.pop("metadata")
    session = Session(row)
    construct_session = MagicMock(return_value=session)
    monkeypatch.setattr(harness, "HarborSession", construct_session)
    monkeypatch.setattr(harness.HarborSessionFactory, "new_client", lambda self: MagicMock())
    builder = partial(harness.HarborSessionFactory, "http://unused", sampling={"temperature": 1.0})
    loop = make_loop(factory_builder=builder, temperature=temperature)
    assert loop._factory.sampling == policy
    loop._factory._tasks = [{"instruction": "task", "index": 0}]
    loop._factory._by_instruction = {harness.instruction_id("task"): 0}
    result, _ = loop._run_session([{"role": "user", "content": "task"}])
    assert construct_session.call_args.kwargs["sampling"] == policy
    assert result != loop._EMPTY_ROLLOUT
    assert session.closed.is_set()


@pytest.mark.parametrize("field,value", [("prompt_token_ids", []), ("per_token_logps", []), ("loss_mask", [1, 1, 1])])
def test_invalid_capture_reaches_worker_failure_channel(make_loop, field, value):
    row = captured_entry()
    row[field] = value
    session = Session(row)
    loop = make_loop(SimpleNamespace(create=lambda *args, **kwargs: session))
    with pytest.raises(openenv_harness.CaptureContractError):
        loop.run()
    assert loop._failed_event.is_set()
    name, message, traceback = loop._exception_info_queue.get_nowait()
    assert name == "CaptureContractError" and message and "fetch_training_trace" in traceback
    assert session.closed.is_set()
    assert loop.rollout_buffer.empty()


def test_producer_validation_error_is_fatal(make_loop):
    session = Session(captured_entry())
    session.fetch_training_trace = MagicMock(
        side_effect=ValueError("cannot train a capture with fatal validation findings")
    )
    loop = make_loop(SimpleNamespace(create=lambda *args, **kwargs: session))
    with pytest.raises(openenv_harness.CaptureContractError, match="fatal validation"):
        loop._run_session([])
    assert session.closed.is_set()


@pytest.mark.parametrize("phase", ["create", "wait_for_completion", "fetch_training_trace", "verify"])
def test_transport_failures_remain_unscorable(make_loop, phase):
    session = Session(captured_entry())
    factory = SimpleNamespace(create=lambda *args, **kwargs: session)
    target = factory if phase == "create" else session
    setattr(target, phase, MagicMock(side_effect=ConnectionError("sandbox unavailable")))
    loop = make_loop(factory)
    assert loop._run_session([]) == (loop._EMPTY_ROLLOUT, None)
    if phase != "create":
        assert session.closed.is_set()


def test_timeout_preserves_verifier_reward_and_tokens(make_loop):
    session = Session(captured_entry())
    session.wait_for_completion = MagicMock(side_effect=TimeoutError("agent budget"))
    loop = make_loop(SimpleNamespace(create=lambda *args, **kwargs: session))
    result, metrics = loop._run_session([])
    assert result[-1] == 0.25
    assert result[2][0].input_ids == [1, 2, 3]
    assert result[2][0].completion_mask == [0, 1, 0]
    assert metrics["loop_exhausted"]


def test_capture_error_closes_other_inflight_sessions(make_loop):
    started = threading.Event()
    blocking = Session(captured_entry())
    released = []

    def wait():
        started.set()
        released.append(blocking.closed.wait(3))

    blocking.wait_for_completion = wait
    broken = Session(captured_entry())
    broken.row["prompt_token_ids"] = []
    broken.wait_for_completion = lambda: started.wait(3)
    sessions = iter([blocking, broken])
    loop = make_loop(SimpleNamespace(create=lambda *args, **kwargs: next(sessions)))
    with pytest.raises(openenv_harness.CaptureContractError):
        loop.run()
    assert blocking.closed.is_set() and broken.closed.is_set()
    assert released == [True]
    assert loop._stop_event.is_set()


def test_concurrent_correctness_metrics_stay_on_event_loop(make_loop):
    owner = threading.get_ident()

    class Rates(defaultdict):
        def __getitem__(self, key):
            assert threading.get_ident() == owner, "metric accumulator accessed from a session thread"
            return super().__getitem__(key)

    loop = make_loop(
        SimpleNamespace(create=lambda *args, **kwargs: Session(captured_entry())),
        rollout_reward_fn=lambda outcome: outcome.env_reward + 0.3,
    )
    loop._rates = Rates(lambda: [0.0, 0.0])

    async def run():
        return await asyncio.gather(*(loop._generate_one([], {}, [], group_id=i) for i in range(32)))

    results = asyncio.run(run())
    assert all(result[-1] == 0.55 and result[2] for result in results)
    metrics = [loop._metrics_queue.get_nowait() for _ in results]
    assert sum(item["rollout/correctness_mean"][0] for item in metrics) == 8.0
    assert sum(item["rollout/correctness_mean"][1] for item in metrics) == 32


def test_white_box_tool_metrics_reach_the_queue_per_tool(make_loop, monkeypatch):
    """The adapter runs the tools between two model turns; that gap is logged as their latency, by name."""
    call = {"type": "function", "function": {"name": "bash", "arguments": {}}}
    replies = iter([{"role": "assistant", "tool_calls": [call, {**call, "function": {"name": "edit"}}]}, {}])
    monkeypatch.setattr(openenv_harness, "parse_response", lambda *args, **kwargs: next(replies))

    class Adapter:
        def run_white_box(self, step, session, limits):
            trace = []
            for tc in step([{"role": "user", "content": "task"}], [], {}).response.tool_calls:
                time.sleep(0.02)
                trace.append(ToolTraceEntry(tc.name, tc.args, ToolResult(error="boom" if tc.name == "edit" else None)))
            step([{"role": "user", "content": "task"}], [], {})
            return HarnessRolloutResult(messages=[{"role": "assistant", "content": "done"}], tool_trace=trace)

    loop = make_loop(
        SimpleNamespace(create=lambda *args, **kwargs: Session(captured_entry())), harness_adapter=Adapter()
    )
    loop.tokenizer.apply_chat_template = MagicMock(return_value=[1])

    async def generate_one_turn(prompt_ids):
        return [2, 3], [-0.1, -0.2]

    loop._generate_one_turn = generate_one_turn
    result = loop._loop.run_until_complete(loop._generate_one([], {}, [], group_id=0))
    assert result[3:5] == (2, 1)
    payload = loop._metrics_queue.get_nowait()
    assert payload["tools/bash_call_total"] == 1 and payload["tools/edit_call_total"] == 1
    assert payload["tools/edit_failure_total"] == 1 and "tools/bash_failure_total" not in payload
    assert payload["tools/latency_s"][1] == 2 and payload["tools/latency_s"][0] >= 0.04
    assert payload["tools/bash_latency_s"][1] == 1 and payload["tools/edit_latency_s"][0] >= 0.04
    assert payload["rollout/tool_s"] == (payload["tools/latency_s"][0] / 2, 1)
    assert payload["rollout/generate_s"][1] == 1


def test_loop_owning_tool_counts_resolve_names_through_call_ids(make_loop):
    """Captured tool results carry `tool_call_id`, not a name; failures still land on the right tool."""
    row = captured_entry()
    row["request"]["messages"] = [
        {"role": "user", "content": "task"},
        {
            "role": "assistant",
            "tool_calls": [{"id": "a", "function": {"name": "bash"}}, {"id": "b", "function": {"name": "edit"}}],
        },
        {"role": "tool", "tool_call_id": "a", "content": "ok"},
        {"role": "tool", "tool_call_id": "b", "content": "Traceback: boom"},
    ]
    row["response"]["choices"][0]["message"]["tool_calls"] = [{"id": "c", "function": {"name": "bash"}}]
    loop = make_loop(SimpleNamespace(create=lambda *args, **kwargs: Session(row)))
    result = loop._loop.run_until_complete(loop._generate_one([], {}, [], group_id=0))
    assert result[3:5] == (1, 1)
    payload = loop._metrics_queue.get_nowait()
    assert payload["tools/bash_call_total"] == 1 and payload["tools/edit_failure_total"] == 1
    assert "tools/latency_s" not in payload and "rollout/tool_s" not in payload


def test_harness_worker_produces_scored_training_samples(make_loop):
    indices = itertools.count()

    def create(*args, **kwargs):
        reward = float(next(indices) % 2)
        session = Session(captured_entry())
        session.verify = lambda completion: SimpleNamespace(env_reward=reward)
        return session

    loop = make_loop(SimpleNamespace(create=create))

    async def run():
        runner = asyncio.create_task(loop._run_loops(loop._stop_event))

        async def wait_for_group():
            while loop.rollout_buffer.qsize() < 8:
                if runner.done():
                    await runner
                await asyncio.sleep(0.01)

        try:
            await asyncio.wait_for(wait_for_group(), timeout=5)
        finally:
            loop._stop_event.set()
            await asyncio.wait_for(runner, timeout=5)

    loop._loop.run_until_complete(run())
    rows = [loop.rollout_buffer.get_nowait() for _ in range(8)]
    assert {row.group_id for row in rows} == {0}
    assert all(row.input_ids == [1, 2, 3] and row.completion_mask == [0, 1, 0] for row in rows)
    assert all(row.old_log_probs == [0.0, -0.1, -0.2] for row in rows)
    assert min(row.advantage for row in rows) < 0 < max(row.advantage for row in rows)


def test_harbor_producer_defaults_and_partial_masks_round_trip():
    harness = pytest.importorskip("harbor_env.harness")
    from openenv.harbor.contract import to_training_trace
    from openenv.harbor.models import HarborRolloutResult, HarborTurn

    factory = harness.HarborSessionFactory("http://unused", sampling={"temperature": 0.8, "top_p": 1.0, "top_k": -1})
    result = HarborRolloutResult(
        turns=[
            HarborTurn(
                turn=0,
                node_id="call-0",
                prompt_token_ids=[10, 11],
                completion_token_ids=[12, 13, 14],
                per_token_logps=[-0.1, -0.2, -0.3],
                loss_mask=[0, 0, 1, 0, 1],
                request_messages=[{"role": "user", "content": "task"}],
                sampling_params=factory.sampling,
            )
        ]
    )
    policy = openenv_harness.training_sampling(
        {"temperature": 0.8, "top_p": 1.0, "top_k": 0, "repetition_penalty": 1.0}
    )
    trace = to_training_trace(result)
    assert factory.sampling == policy
    turns = openenv_harness._turns_from_training_trace(trace)
    rows, _ = _chain_to_sequences(turns, "test", 0)
    assert rows[0].input_ids == [10, 11, 12, 13, 14]
    assert rows[0].completion_mask == [0, 0, 1, 0, 1]
    assert rows[0].old_log_probs[-3:] == [-0.1, -0.2, -0.3]


def test_reconciliation_error_reaches_worker_failure_channel(make_loop, monkeypatch):
    session = Session(captured_entry())
    loop = make_loop(SimpleNamespace(create=lambda *args, **kwargs: session))
    monkeypatch.setattr(
        openenv_harness, "_chain_to_sequences", MagicMock(side_effect=ValueError("invalid training sequence"))
    )
    with pytest.raises(openenv_harness.CaptureContractError, match="invalid training sequence"):
        loop.run()
    assert loop._failed_event.is_set()
    assert session.closed.is_set()
    assert loop.rollout_buffer.empty()


@pytest.mark.parametrize("reward", [None, 0.0, 1.0])
def test_harbor_session_to_worker_preserves_masks_and_outcome(make_loop, reward):
    HarborSession = pytest.importorskip("harbor_env.harness").HarborSession
    from openenv.harbor.models import HarborRolloutResult, HarborTurn

    turns = [
        HarborTurn(
            turn=0,
            node_id="a",
            prompt_token_ids=[10],
            completion_token_ids=[11, 12],
            per_token_logps=[-0.1, -0.2],
            loss_mask=[0, 1, 0],
            request_messages=[{"role": "user", "content": "task"}],
            tool_calls=[{"name": "bash", "arguments": "{}"}],
        ),
        HarborTurn(
            turn=1,
            node_id="b",
            prompt_token_ids=[20],
            completion_token_ids=[21],
            per_token_logps=[-0.3],
            loss_mask=[0, 0],
            trainable=False,
            request_messages=[{"role": "user", "content": "task"}],
            tool_calls=[{"name": "bash", "arguments": "{}"}],
        ),
        HarborTurn(turn=2, node_id="title", role="auxiliary"),
        HarborTurn(turn=3, node_id="retry", discarded=True),
    ]
    result = HarborRolloutResult.model_validate_json(HarborRolloutResult(turns=turns, reward=reward).model_dump_json())
    env = MagicMock()
    env.run_rollout.return_value = result
    session = HarborSession(
        env=env,
        split="tasks",
        task_index=0,
        instruction="task",
        harness="opencode",
        sandbox="docker",
        llm_url="http://unused",
        model="test",
        owns_env=True,
    )
    loop = make_loop(SimpleNamespace(create=lambda *args, **kwargs: session))
    actual, _ = loop._run_session([])
    assert actual[-1] == reward
    assert actual[3] == 2  # Usage includes the zero-masked agent call.
    assert len(actual[2]) == 1
    assert actual[2][0].input_ids == [10, 11, 12]
    assert actual[2][0].completion_mask == [0, 1, 0]
    assert actual[2][0].old_log_probs == [0.0, -0.1, -0.2]
    env.close.assert_called_once()


def test_shared_prefix_is_supervised_once_across_both_branches():
    from openenv.core.harness.capture.contract import to_training_trace
    from openenv.core.harness.capture.graph import RolloutGraph, TurnNode

    graph = RolloutGraph()
    for node_id, prompt, output in [("root", [1], [2]), ("left", [1, 2], [3]), ("right", [1, 2], [4])]:
        graph.add_turn(TurnNode(node_id=node_id, prompt_ids=prompt, sampled_ids=output, sampled_logprobs=[-0.1]))
    document = {
        "sequences": [
            {"role": "agent", "node_ids": ["root", "left"], "loss_mask": [0, 1, 1]},
            {"role": "agent", "node_ids": ["root", "right"], "loss_mask": [0, 1, 1]},
        ]
    }
    turns = openenv_harness._turns_from_training_trace(to_training_trace(graph, document))
    rows, _ = _chain_to_sequences(turns, "rollout", 0)
    supervised = [
        token for row in rows for token, mask in zip(row.input_ids, row.completion_mask, strict=True) if mask
    ]
    assert supervised == [2, 3, 4]


def test_legacy_session_fails_before_starting_agent(make_loop):
    session = SimpleNamespace(wait_for_completion=MagicMock(), fetch_proxy_trace=MagicMock(), close=MagicMock())

    # Sessions must be hashable because the worker tracks them for shutdown.
    class LegacySession:
        wait_for_completion = session.wait_for_completion
        fetch_proxy_trace = session.fetch_proxy_trace
        close = session.close

    loop = make_loop(SimpleNamespace(create=lambda *args, **kwargs: LegacySession()))
    with pytest.raises(openenv_harness.CaptureContractError, match="fetch_training_trace"):
        loop._run_session([])
    session.wait_for_completion.assert_not_called()
    session.close.assert_called_once()
