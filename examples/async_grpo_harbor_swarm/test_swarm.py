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

"""CPU checks using the real msgboard, capture contract, and TRL packing objects."""

import asyncio
import json
import os
import subprocess
import sys
import tarfile

import httpx
import pytest
import torch
from board import TeamBoard, tool_bundle
from msgboard.client import BoardError, Client
from openenv.harbor.models import HarborRolloutResult, HarborTurn
from runtime import validate_team
from training import SwarmConfig, pack_team, score_teams

from trl.trainer.async_grpo_trainer import (
    DataCollatorForRollout,
    Rollout,
    RolloutGroup,
    TurnRecord,
    _chain_to_sequences,
)


def group_for(rewards, fork=False):
    rollouts = []
    for i, reward in enumerate(rewards):
        # Distinct agent prompts, and one rewritten history when requested.
        first = TurnRecord([10, 11], [12, 13], [-0.5, -0.7])
        second = TurnRecord([10, 11, 12, 13, 14] if not fork else [10, 99, 14], [15], [-0.8])
        third = TurnRecord([20, 21], [22, 23, 24, 25], [-0.1] * 4, [1, 0, 1, 1])
        turns = [first, second, third] if i == 0 else [first]
        rollouts.append(
            Rollout(
                completion=[{"role": "assistant", "content": json.dumps({"team_id": str(i), "policy_version": 7})}],
                completion_ids=[],
                turns=turns,
                reward=reward,
            )
        )
    return RolloutGroup(
        prompts=[[] for _ in rewards],
        reward_kwargs={},
        completions=[r.completion for r in rollouts],
        completions_ids=[[] for _ in rewards],
        completions_sequences=[_chain_to_sequences(r)[0] for r in rollouts],
        tool_call_counts=[0] * len(rewards),
        tool_failure_counts=[0] * len(rewards),
        model_version=6,
        group_id=42,
        row_index=0,
        env_rewards=[None] * len(rewards),
        rollout_rewards=rewards,
    )


def test_team_advantage_and_unscorable():
    packets = score_teams(group_for([1, 0, None]))
    assert len(packets) == 2
    assert all(row["advantage"] == pytest.approx(1) for row in packets[0].rows)
    assert all(row["advantage"] == pytest.approx(-1) for row in packets[1].rows)
    assert all(packet.model_version == 7 for packet in packets)


def test_swarm_defaults_disable_token_score_centering(tmp_path):
    config = SwarmConfig(output_dir=str(tmp_path), use_cpu=True, bf16=False, report_to="none")
    assert config.score_centering is False


def test_equal_rewards_are_kept_with_zero_advantage():
    packets = score_teams(group_for([0, 0]))
    assert len(packets) == 2
    assert all(row["advantage"] == 0 for packet in packets for row in packet.rows)
    assert score_teams(group_for([None, None])) == []


@pytest.mark.parametrize("budget", [8, 64])
@pytest.mark.parametrize("fork", [False, True])
def test_team_loss_and_gradient_are_independent_of_packing(budget, fork):
    packets = score_teams(group_for([1, 0], fork))
    parameter = torch.tensor(0.13, requires_grad=True)
    reference = []
    for packet in packets:
        tokens = []
        for row in packet.rows:
            tokens.extend(row["input_ids"][1:][row["completion_mask"][1:].astype(bool)])
        # A nonconstant differentiable loss makes incorrect weighting visible in both loss and gradient.
        reference.append(((parameter * torch.tensor(tokens)).square()).mean() * packet.rows[0]["advantage"])
    expected = torch.stack(reference).mean()
    (expected_grad,) = torch.autograd.grad(expected, parameter, retain_graph=True)
    actual = torch.tensor(0.0)
    for packet in packets:
        chunks = pack_team(packet, budget, DataCollatorForRollout(0))["chunks"]
        for chunk in chunks:
            selected = chunk["token_index"] >= 0
            labels = chunk["shift_labels"][chunk["attention_mask"].bool()]
            tokens = labels[chunk["token_index"][selected]]
            actual = actual + ((parameter * tokens).square() * chunk["advantages"][selected]).sum() / (
                chunk["team_tokens"][0] * len(packets)
            )
    (actual_grad,) = torch.autograd.grad(actual, parameter)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_grad, expected_grad)


def test_capture_masks_and_logprobs_survive_forks():
    packet = score_teams(group_for([1, 0], fork=True))[0]
    chunks = pack_team(packet, 8, DataCollatorForRollout(0))["chunks"]
    logps = torch.cat([chunk["old_log_probs"][chunk["token_index"] >= 0] for chunk in chunks])
    assert sorted(logps.tolist()) == pytest.approx(sorted([-0.5, -0.7, -0.8, -0.1, -0.1, -0.1]))
    assert all(chunk["team_tokens"].item() == 6 for chunk in chunks)
    with pytest.raises(ValueError, match="exceeds"):
        pack_team(packet, 2, DataCollatorForRollout(0))


def test_real_msgboard_and_container_tool_bundle(tmp_path):
    async def run():
        first = TeamBoard(tmp_path / "first", ["agent000", "agent001"])
        second = TeamBoard(tmp_path / "second", ["agent000"])
        try:
            await first.start()
            await second.start()
            with Client(
                "http://board", first.credentials["agent000"], transport=httpx.HTTPTransport(uds=str(first.socket))
            ) as a:
                a.post(first.channel, "verified clue", request_id="post-1")
                a.post(first.channel, "verified clue", request_id="post-1")
            with Client(
                "http://board", first.credentials["agent001"], transport=httpx.HTTPTransport(uds=str(first.socket))
            ) as b:
                messages = b.read(first.channel)["messages"]
                assert [message["body"] for message in messages] == ["verified clue"]
                assert b.read(first.channel, after=messages[-1]["seq"])["messages"] == []
            with Client(
                "http://board", first.credentials["agent000"], transport=httpx.HTTPTransport(uds=str(second.socket))
            ) as wrong_board:
                with pytest.raises(BoardError):
                    wrong_board.me()
            bundle = tool_bundle(tmp_path / "tools.tgz")
            unpacked = tmp_path / "tools"
            with tarfile.open(bundle) as archive:
                archive.extractall(unpacked, filter="data")
            result = subprocess.run(
                [
                    sys.executable,
                    "-S",
                    str(unpacked / "board_tool.py"),
                    "message_board.read",
                    "--json",
                    '{"channel":"general","after":0,"limit":12}',
                ],
                env={
                    **os.environ,
                    "MSGBOARD_URL": "http://board",
                    "MSGBOARD_TOKEN": first.credentials["agent001"],
                    "MSGBOARD_SOCKET": str(first.socket),
                },
                capture_output=True,
                text=True,
                check=True,
            )
            assert json.loads(result.stdout)["messages"][0]["body"] == "verified clue"
        finally:
            await first.close()
            await second.close()
        assert first.process.returncode is not None
        assert not first.socket.exists()
        assert "verified clue" in (tmp_path / "first/messages.jsonl").read_text()

    asyncio.run(run())


def test_stable_disjoint_scenario_split():
    from async_grpo_harbor_swarm import split_tasks

    names = [f"task-{i}" for i in range(60)]
    split = split_tasks(names)
    assert len(split["train"]) == 48
    assert len(split["eval"]) == 12
    assert not set(split["train"]) & set(split["eval"])
    assert split == split_tasks(reversed(names))


def captured_result(**kwargs):
    return HarborRolloutResult(
        turns=[
            HarborTurn(
                turn=0,
                node_id="call-1",
                prompt_token_ids=[10, 11],
                completion_token_ids=[12, 13],
                per_token_logps=[-0.1, -0.2],
                loss_mask=[0, 0, 1, 1],
            )
        ],
        **kwargs,
    )


def test_native_success_and_budget_exhaustion():
    success = captured_result(reward=1)
    timeout = captured_result(reward=0, ok=False, exception_type="AgentTimeoutError")
    reward, traces = validate_team([success, timeout], max_context=16, winner=0)
    assert reward == 1
    assert traces[1].turns[0].completion_token_ids == [12, 13]
    assert validate_team([timeout, timeout], 16)[0] == 0
    with pytest.raises(ValueError, match="confirm"):
        validate_team([success, timeout], 16, winner=1)


@pytest.mark.parametrize(
    "failure",
    [
        {"reward": None},
        {"reward": 0, "ok": False, "exception_type": "ConnectionError"},
        {"reward": 1, "findings": ["[FATAL] missing sampling logprobs"]},
    ],
)
def test_infrastructure_and_invalid_capture_are_not_zero_rewards(failure):
    with pytest.raises(ValueError):
        validate_team([captured_result(reward=1), captured_result(**failure)], 16)
