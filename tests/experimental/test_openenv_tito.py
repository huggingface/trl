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
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from accelerate import PartialState


pytest.importorskip("openenv.core.harness.capture.validate")

from trl.experimental.async_grpo import openenv_harness
from trl.experimental.async_grpo.async_rollout_worker import _chain_to_sequences, _SampleBuilder


def entry(prompt, completion, logprobs, mask=None):
    return {
        "prompt_token_ids": prompt,
        "completion_token_ids": completion,
        "per_token_logps": logprobs,
        "loss_mask": mask,
    }


def test_lossless_capture_preserves_rewritten_tail_and_partial_mask():
    entries = [
        entry([1, 2], [3, 4, 5], [-0.1, -0.2, -0.3], [0, 0, 1, 0, 1]),
        entry([1, 2, 3, 4, 99, 6], [7], [-0.4], [0, 0, 0, 0, 0, 0, 1]),
    ]
    turns = openenv_harness._turns_from_trace(entries)
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
        openenv_harness._turns_from_trace([entry([1], [2], logprobs)])


@pytest.mark.parametrize("mask,logprobs", [(1, None), (1, []), ([1, 0], [-0.1]), (0, [])])
def test_builder_rejects_misaligned_arrays_before_mutation(mask, logprobs):
    builder = _SampleBuilder(fork_threshold=0)
    with pytest.raises(ValueError):
        builder._append([2], mask=mask, logprobs=logprobs)
    assert builder.tokens == builder.loss_mask == builder.logprobs == []


@pytest.mark.parametrize("mask", [[1, 1], [0], [0, 2], [0, True]])
def test_consumer_rejects_invalid_full_masks(mask):
    with pytest.raises(ValueError):
        openenv_harness._turns_from_trace([entry([1], [2], [-0.1], mask)])


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
    loop = openenv_harness._HarnessRolloutLoop(
        harness_session_factory=SimpleNamespace(), harness_adapter=adapter, lossless_capture=lossless
    )
    try:
        assert loop._fork_threshold_tokens == threshold
    finally:
        loop._session_pool.shutdown()


def test_captured_policy_must_match_training_temperature():
    row = entry([1], [2], [-0.1])
    with pytest.raises(ValueError, match="sampling"):
        openenv_harness._turns_from_trace([row], sampling={"temperature": 0.8})
    row["metadata"] = {"sampling_params": {"temperature": 1.0}}
    with pytest.raises(ValueError, match="sampling"):
        openenv_harness._turns_from_trace([row], sampling={"temperature": 0.8})
    row["metadata"]["sampling_params"]["temperature"] = 0.8
    assert len(openenv_harness._turns_from_trace([row], sampling={"temperature": 0.8})) == 1


@pytest.fixture
def make_loop():
    PartialState()
    loops = []

    def make(factory, **kwargs):
        tokenizer = MagicMock(eos_token_id=0, pad_token_id=0)
        loop = openenv_harness._HarnessRolloutLoop(
            harness_session_factory=factory,
            model_name="test",
            dataset=[{"prompt": [{"role": "user", "content": "task"}]}],
            reward_funcs=[],
            processing_class=tokenizer,
            rollout_buffer=queue.Queue(),
            model_version_value=SimpleNamespace(value=0),
            heartbeat_value=SimpleNamespace(value=0.0),
            failed_event=threading.Event(),
            exception_info_queue=queue.Queue(),
            metrics_queue=queue.Queue(),
            temperature=0.8,
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
        metadata={"sampling_params": openenv_harness.training_sampling({"temperature": 0.8})},
    )
    return row


class Session:
    def __init__(self, row):
        self.row = row
        self.closed = threading.Event()

    def wait_for_completion(self):
        return 0

    def fetch_proxy_trace(self):
        return [self.row]

    def verify(self, completion):
        return SimpleNamespace(env_reward=0.25)

    def close(self):
        self.closed.set()


def test_factory_temperature_can_differ_from_trainer(make_loop, monkeypatch):
    harness = pytest.importorskip("harbor_env.harness")
    factory = harness.HarborSessionFactory("http://unused", sampling={"temperature": 1.0})
    row = captured_entry()
    row["metadata"]["sampling_params"] = factory.sampling
    session = Session(row)
    monkeypatch.setattr(factory, "create", lambda *args, **kwargs: session)
    loop = make_loop(factory)  # Trainer temperature is 0.8.
    with pytest.raises(openenv_harness.CaptureContractError, match="sampling"):
        loop._run_session([])
    assert session.closed.is_set()
    assert loop.rollout_buffer.empty()


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
    assert name == "CaptureContractError" and message and "_turns_from_trace" in traceback
    assert session.closed.is_set()
    assert loop.rollout_buffer.empty()


def test_producer_validation_error_is_fatal(make_loop):
    session = Session(captured_entry())
    session.fetch_proxy_trace = MagicMock(
        side_effect=ValueError("cannot train a capture with fatal validation findings")
    )
    loop = make_loop(SimpleNamespace(create=lambda *args, **kwargs: session))
    with pytest.raises(openenv_harness.CaptureContractError, match="fatal validation"):
        loop._run_session([])
    assert session.closed.is_set()


@pytest.mark.parametrize("phase", ["create", "wait_for_completion", "fetch_proxy_trace", "verify"])
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
    from openenv.harbor.contract import to_trace_entries
    from openenv.harbor.models import HarborRolloutResult, HarborTurn

    factory = harness.HarborSessionFactory("http://unused", sampling={"temperature": 0.8, "top_p": 1.0, "top_k": -1})
    result = HarborRolloutResult(
        turns=[
            HarborTurn(
                turn=0,
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
    trace = to_trace_entries(result)
    turns = openenv_harness._turns_from_trace(trace, sampling=policy)
    rows, _ = _chain_to_sequences(turns, "test", 0)
    assert rows[0].input_ids == [10, 11, 12, 13, 14]
    assert rows[0].completion_mask == [0, 0, 1, 0, 1]
    assert rows[0].old_log_probs[-3:] == [-0.1, -0.2, -0.3]
    for key in ("temperature", "repetition_penalty"):
        original = trace[0]["metadata"]["sampling_params"].pop(key)
        with pytest.raises(ValueError, match="sampling"):
            openenv_harness._turns_from_trace(trace, sampling=policy)
        trace[0]["metadata"]["sampling_params"][key] = original + 0.1
        with pytest.raises(ValueError, match="sampling"):
            openenv_harness._turns_from_trace(trace, sampling=policy)
        trace[0]["metadata"]["sampling_params"][key] = original


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
