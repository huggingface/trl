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

from types import SimpleNamespace

import pytest


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
    builder = _SampleBuilder(fork_threshold=0)
    with pytest.raises(ValueError):
        builder._append([2], mask=1, logprobs=logprobs)
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
