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
from runtime import control_command, validate_team
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


def test_two_difficulty_split_is_balanced_and_disjoint():
    from async_grpo_harbor_swarm import split_difficulties

    names = [f"agentescape-{level}-task-{i:03}" for level in (5, 10, 15) for i in range(60)]
    splits = split_difficulties(names, [10, 15])
    assert len(splits["train"]) == 96
    assert len(splits["eval"]) == 24
    assert not set(splits["train"]) & set(splits["eval"])
    assert all(name.startswith(f"agentescape-{10 if i % 2 == 0 else 15}-") for i, name in enumerate(splits["train"]))
    assert splits == split_difficulties(reversed(names), [10, 15])


def test_call_gate_drains_and_versions_inflight_requests():
    from policy_gateway import PolicyGate

    async def run():
        gate = PolicyGate()
        entered, release = asyncio.Event(), asyncio.Event()
        versions = []

        async def first():
            async with gate.call() as version:
                versions.append(version)
                entered.set()
                await release.wait()

        async def second():
            async with gate.call() as version:
                versions.append(version)

        active = asyncio.create_task(first())
        await entered.wait()
        drain = asyncio.create_task(gate.drain())
        await asyncio.sleep(0)
        queued = asyncio.create_task(second())
        await asyncio.sleep(0)
        assert not drain.done() and not queued.done()
        release.set()
        await active
        await drain
        assert not queued.done()
        await gate.publish(1)
        await queued
        assert versions == [0, 1]
        await gate.drain()
        await gate.fail()
        from fastapi import HTTPException

        with pytest.raises(HTTPException):
            async with gate.call():
                pytest.fail("failed policy must not serve")
        with pytest.raises(HTTPException):
            await gate.publish(2)

    asyncio.run(run())


def test_gateway_preserves_capture_and_rejects_ambiguous_versions(tmp_path):
    from policy_gateway import create_app, match_versions

    async def upstream(request):
        body = json.loads(request.content)
        assert body["cache_salt"] in {"swarm-policy-0", "swarm-policy-1"}
        return httpx.Response(
            200,
            json={
                "prompt_token_ids": [10, 11],
                "choices": [{"token_ids": [12, 13], "logprobs": {"content": [{"logprob": -0.1}, {"logprob": -0.2}]}}],
            },
        )

    async def run():
        app = create_app("http://fake", tmp_path, "test-token", httpx.MockTransport(upstream))
        headers = {"x-swarm-control": "test-token"}
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://gateway") as client:
            assert (await client.post("/control/drain")).status_code == 403
            response = await client.post("/agents/team-agent/v1/chat/completions", json={"stream": False})
            assert response.json()["choices"][0]["logprobs"]["content"][1]["logprob"] == -0.2
            records = (await client.get("/records/team-agent", headers=headers)).json()
            assert match_versions(captured_result().turns, records) == [{"node_id": "call-1", "policy_version": 0}]
            await client.post("/control/drain", headers=headers)
            await client.post("/control/publish", headers=headers, json={"version": 1})
            await client.post("/agents/team-agent/v1/chat/completions", json={})
            records = (await client.get("/records/team-agent", headers=headers)).json()
            with pytest.raises(ValueError, match="ambiguous"):
                match_versions(captured_result().turns, records)
            await client.delete("/records/team-agent", headers=headers)
            assert (await client.get("/records/team-agent", headers=headers)).json() == []

    asyncio.run(run())


def test_staleness_rejects_whole_team_and_boundary_is_inclusive():
    import queue
    from collections import defaultdict
    from types import SimpleNamespace

    from training import TeamDataset

    packets = score_teams(group_for([1, 0]))
    packets[0].model_version = 6  # At model 11, even one version-6 call invalidates the entire team.
    packets[1].model_version = 7
    q = queue.Queue()
    for packet in packets:
        q.put(packet)
    trainer = SimpleNamespace(
        args=SimpleNamespace(max_staleness=4, gradient_accumulation_steps=1, token_budget=64, collection_deadline=0),
        accelerator=SimpleNamespace(num_processes=1),
        rollout_queue=q,
        model_version=11,
        rollout_worker=SimpleNamespace(token_allowance=SimpleNamespace(value=0)),
        _metrics={"train": defaultdict(list)},
    )
    assert next(iter(TeamDataset(trainer))) is packets[1]
    assert trainer._metrics["train"]["team/dropped_stale"] == [1.0]


@pytest.mark.parametrize("budget", [8, 64])
def test_distributed_padding_preserves_equal_team_gradients(budget):
    from training import equalize_chunks

    packets = score_teams(group_for([1, 0]))
    batches = equalize_chunks([pack_team(packet, budget, DataCollatorForRollout(0)) for packet in packets])
    assert len(batches[0]["chunks"]) == len(batches[1]["chunks"])
    parameter = torch.tensor(0.13, requires_grad=True)
    reference = []
    for packet in packets:
        tokens = torch.tensor(
            [token for row in packet.rows for token in row["input_ids"][1:][row["completion_mask"][1:].astype(bool)]]
        )
        reference.append((parameter * tokens).square().mean() * packet.rows[0]["advantage"])
    expected = torch.stack(reference).mean()
    actual = torch.tensor(0.0)
    for batch in batches:
        for chunk in batch["chunks"]:
            selected = chunk["token_index"] >= 0
            labels = chunk["shift_labels"][chunk["attention_mask"].bool()]
            values = labels[chunk["token_index"][selected]]
            # Mirrors rank averaging, with the denominator passed to TRL's world-scaled loss.
            global_denominator = chunk["team_tokens"][0] * len(batches)
            rank_loss = ((parameter * values).square() * chunk["advantages"][selected]).sum()
            actual += rank_loss / (global_denominator / len(batches)) / len(batches)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual, parameter)[0], torch.autograd.grad(expected, parameter)[0])


class ScalarPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.002))

    def forward(self, input_ids, shift_labels, **kwargs):
        from types import SimpleNamespace

        return SimpleNamespace(
            log_probs=-0.5 + self.scale * shift_labels.float(), entropy=torch.ones_like(input_ids, dtype=torch.float32)
        )


class CPUCollectives:
    def __init__(self, world=1, rank=0):
        self.num_processes = world
        self.is_main_process = rank == 0

    def reduce(self, value, reduction="sum"):
        value = value.clone()
        if self.num_processes > 1:
            torch.distributed.all_reduce(value)
        return value

    def gather(self, value):
        if self.num_processes == 1:
            return value
        values = [torch.empty_like(value) for _ in range(self.num_processes)]
        torch.distributed.all_gather(values, value)
        return torch.cat([item.reshape(-1) for item in values])


def loss_trainer(world, rank, rounds):
    from collections import defaultdict
    from types import SimpleNamespace

    from training import SwarmTrainer

    trainer = object.__new__(SwarmTrainer)
    trainer.accelerator = CPUCollectives(world, rank)
    trainer.args = SimpleNamespace(
        vllm_importance_sampling_clip_min=0.2,
        vllm_importance_sampling_clip_max=5.0,
        score_centering=False,
        gradient_accumulation_steps=rounds,
        token_budget=8,
    )
    trainer.epsilon_low, trainer.epsilon_high = 1.0, 1e9
    trainer.aux_loss_enabled = False
    trainer.current_gradient_accumulation_steps = rounds
    trainer._metrics = {"train": defaultdict(list)}
    for name in (
        "_step_forward_tokens",
        "_step_trained_tokens",
        "_step_seq_len_weighted",
        "_step_samples",
        "_step_forward_s",
    ):
        setattr(trainer, name, 0.0)
    return trainer


def distributed_packets():
    packets = score_teams(group_for([1, 0, 0, 1], fork=True))
    for row in packets[2].rows:
        row["advantage"] = 0.0
    return packets


def distributed_gradient_worker(rank, init_method, expected):
    from training import DistributedTeams

    torch.distributed.init_process_group("gloo", rank=rank, world_size=2, init_method=init_method)
    try:
        trainer = loss_trainer(2, rank, 2)
        trainer._rollout_dataset = iter(distributed_packets()) if rank == 0 else None
        trainer.team_collator = DataCollatorForRollout(0)
        model = ScalarPolicy()
        rounds = 0
        for batch in DistributedTeams(trainer):
            rounds += 1
            for chunk in batch["chunks"]:
                trainer.compute_loss(model, chunk).backward()
        assert rounds == 2
        torch.distributed.all_reduce(model.scale.grad)
        model.scale.grad /= 2
        torch.testing.assert_close(model.scale.grad, torch.tensor(expected))
    finally:
        torch.distributed.destroy_process_group()


def test_two_process_real_loss_matches_single_process_reference():
    import tempfile
    from pathlib import Path

    trainer = loss_trainer(1, 0, 4)
    model = ScalarPolicy()
    for packet in distributed_packets():
        for chunk in pack_team(packet, 8, DataCollatorForRollout(0))["chunks"]:
            trainer.compute_loss(model, chunk).backward()
    with tempfile.TemporaryDirectory(prefix="swarm-gloo-test-") as directory:
        torch.multiprocessing.start_processes(
            distributed_gradient_worker,
            args=(f"file://{Path(directory) / 'rendezvous'}", model.scale.grad.item()),
            nprocs=2,
            join=True,
            start_method="fork",
        )


@pytest.mark.parametrize("failures", [0, 1, 2])
def test_container_control_timeout_retry(failures):
    # Stand in only for the external Harbor container command.
    calls = []

    async def container_command(command, *, service, timeout_sec):
        calls.append((command, service, timeout_sec))
        if len(calls) <= failures:
            raise RuntimeError(f"Command timed out after {timeout_sec} seconds")
        return "trusted state"

    async def run():
        return await control_command(
            container_command, "read state", service="puzzle", operation="agent003: read trusted puzzle state"
        )

    if failures == 2:
        with pytest.raises(RuntimeError, match="agent003: read trusted puzzle state: timed out twice"):
            asyncio.run(run())
    else:
        assert asyncio.run(run()) == "trusted state"
    assert calls == [("read state", "puzzle", 30)] * min(failures + 1, 2)


@pytest.mark.parametrize("error", [RuntimeError("container disappeared"), asyncio.CancelledError()])
def test_container_control_does_not_retry_errors_or_cancellation(error):
    calls = []

    async def container_command(*args, **kwargs):
        calls.append(1)
        raise error

    with pytest.raises(type(error)):
        asyncio.run(control_command(container_command, operation="signal Pi to stop"))
    assert len(calls) == 1
