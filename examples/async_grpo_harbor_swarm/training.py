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

"""Team-atomic queueing and exact team means, without changing TRL's public trainer."""

import copy
import json
import os
import queue
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
from runtime import Settings, TeamRunner
from torch.utils.data import DataLoader, IterableDataset

from trl import AsyncGRPOConfig, AsyncGRPOTrainer
from trl.trainer.async_grpo_trainer import AsyncRolloutLoop, DataCollatorForRollout, Rollout, TreeRow, TurnRecord


@dataclass
class SwarmConfig(AsyncGRPOConfig):
    swarm_settings: dict | None = None
    max_groups: int = 32
    score_centering: bool = False
    collection_deadline: float = 0


@dataclass
class TeamPacket:
    team_id: str
    rows: list[dict]
    model_version: int
    group_id: int
    enqueued_at: float = 0

    @property
    def forwarded_tokens(self):
        return sum(len(row["input_ids"]) for row in self.rows)


def score_teams(group) -> list[TeamPacket]:
    """Keep complete valid teams, including equal-reward groups, with one advantage per team."""
    rewards = np.array([np.nan if reward is None else reward for reward in group.rollout_rewards])
    valid = np.isfinite(rewards)
    advantages = np.zeros(len(rewards))
    if valid.any():
        advantages[valid] = (rewards[valid] - rewards[valid].mean()) / (rewards[valid].std() + 1e-8)
    packets = []
    for index in np.flatnonzero(valid):
        meta = json.loads(group.completions[index][0]["content"])
        rows = []
        for seq in group.completions_sequences[index]:
            if not any(seq.completion_mask[1:]):
                continue
            rows.append(
                {
                    "input_ids": np.asarray(seq.input_ids, dtype=np.int64),
                    "completion_mask": np.asarray(seq.completion_mask, dtype=np.int64),
                    "old_log_probs": np.asarray(seq.old_log_probs, dtype=np.float32),
                    "sampling_mask": seq.sampling_mask,
                    "sampler_ids": np.full((len(seq.input_ids), 0), -1, dtype=np.int64),
                    "sampler_log_probs": np.full((len(seq.input_ids), 0), -np.inf, dtype=np.float32),
                    "advantage": float(advantages[index]),
                    "group_id": group.group_id,
                    "metrics": {"reward_trained": float(rewards[index])},
                }
            )
        if rows:
            packets.append(TeamPacket(meta["team_id"], rows, meta["policy_version"], group.group_id))
    return packets


def pack_team(packet: TeamPacket, budget: int, collator: DataCollatorForRollout):
    """Every chunk uses the full team's trained-token denominator, including forked histories."""
    total = sum(int(row["completion_mask"][1:].sum()) for row in packet.rows)
    if not total:
        raise ValueError("A team must contain supervised tokens")
    chunks, row = [], TreeRow()
    for sample in packet.rows:
        if len(sample["input_ids"]) > budget:
            raise ValueError("A captured row exceeds token_budget; increase it to at least max_context")
        if row.samples and not row.fits(sample, budget):
            chunks.append(collator([[row]]))
            row = TreeRow()
        row.add(sample)
    if row.samples:
        chunks.append(collator([[row]]))
    for chunk in chunks:
        chunk["team_tokens"] = torch.tensor([total], dtype=torch.float32)
    return {"chunks": chunks}


class TeamDataset(IterableDataset):
    """Consume/discard whole teams; retain TRL's queue heartbeat and token backpressure."""

    def __init__(self, trainer):
        self.trainer = trainer
        self.token_allowance = trainer.rollout_worker.token_allowance
        self.wait_s = self.consumed_tokens = self.step_tokens = 0
        self.lookahead_tokens = (
            max(trainer.args.max_staleness, 1)
            * trainer.args.gradient_accumulation_steps
            * trainer.accelerator.num_processes
            * trainer.args.token_budget
        )
        self.token_allowance.value = self.lookahead_tokens

    def __iter__(self):
        trainer = self.trainer
        while True:
            if trainer.args.collection_deadline and time.time() >= trainer.args.collection_deadline:
                return
            start = time.monotonic()
            try:
                packet = trainer.rollout_queue.get(timeout=5)
            except queue.Empty:
                self.wait_s += time.monotonic() - start
                trainer.rollout_worker.check_health(trainer.args.rollout_worker_timeout)
                continue
            self.wait_s += time.monotonic() - start
            if isinstance(packet, int):
                trainer._dropped_groups.add(packet)
                continue
            self.consumed_tokens += packet.forwarded_tokens
            self.step_tokens += packet.forwarded_tokens
            self.token_allowance.value = self.consumed_tokens + self.lookahead_tokens
            if trainer.model_version - packet.model_version > trainer.args.max_staleness:
                trainer._metrics["train"]["team/dropped_stale"].append(1.0)
                continue
            trainer._metrics["train"]["team/policy_staleness"].append(trainer.model_version - packet.model_version)
            trainer._metrics["train"]["team/reward"].append(packet.rows[0]["metrics"]["reward_trained"])
            yield packet


def equalize_chunks(batches):
    """Match forward/backward counts across ranks without adding any policy gradient."""
    count = max(len(batch["chunks"]) for batch in batches)
    for batch in batches:
        while len(batch["chunks"]) < count:
            dummy = copy.deepcopy(batch["chunks"][0])
            dummy["advantages"].zero_()
            dummy["global_n_tokens"].zero_()
            dummy["global_n_forward_tokens"].zero_()
            batch["chunks"].append(dummy)
    return batches


class DistributedTeams(IterableDataset):
    """Stage a whole optimizer batch before yielding any accumulation rounds."""

    def __init__(self, trainer, group=None):
        self.trainer, self.group = trainer, group

    def __iter__(self):
        trainer = self.trainer
        world = trainer.accelerator.num_processes
        main = trainer.accelerator.is_main_process
        source = iter(trainer._rollout_dataset) if main else None
        rounds = trainer.args.gradient_accumulation_steps
        while True:
            batches, error = [], None
            if main:
                try:
                    # No partial optimizer updates when collection reaches its deadline.
                    packets = [next(source) for _ in range(world * rounds)]
                    for start in range(0, len(packets), world):
                        batches.append(
                            equalize_chunks(
                                [
                                    pack_team(packet, trainer.args.token_budget, trainer.team_collator)
                                    for packet in packets[start : start + world]
                                ]
                            )
                        )
                except StopIteration:
                    batches = []
                except Exception as exc:
                    error = repr(exc)
            status = [bool(batches), error]
            if world > 1:
                torch.distributed.broadcast_object_list(status, src=0, group=self.group)
            if status[1]:
                raise RuntimeError(f"Team batch collection failed: {status[1]}")
            if not status[0]:
                return
            for index in range(rounds):
                if world == 1:
                    yield batches[index][0]
                else:
                    result = [None]
                    torch.distributed.scatter_object_list(
                        result, batches[index] if main else None, src=0, group=self.group
                    )
                    yield result[0]


def identity(value):
    return value


class SwarmRolloutLoop(AsyncRolloutLoop):
    async def setup(self):
        if self.config.log_completions:
            raise ValueError("Use log_completions=False; swarm outcomes are recorded in teams.jsonl")
        if self.sampling_replay or self.config.score_centering:
            raise ValueError("This example requires full-vocabulary sampling and score_centering=False")
        if self.lora_name is None and not self.config.swarm_settings.get("gateway_url"):
            raise ValueError("Full-model training requires a versioned policy gateway")
        self.runner = TeamRunner(Settings(**self.config.swarm_settings))
        await self.runner.start()

    def _fits_token_allowance(self):
        deadline = self.config.swarm_settings.get("admission_deadline", 0)
        return (not deadline or time.time() < deadline) and super()._fits_token_allowance()

    async def close(self):
        await self.runner.close()

    async def generate_one(self, prompt, tools, group_id=0):
        task = Path(prompt[-1]["content"])
        version = self.model_version
        deadline = self.config.swarm_settings.get("admission_deadline", 0)
        if deadline and time.time() >= deadline:
            return Rollout(
                completion=[
                    {
                        "role": "assistant",
                        "content": json.dumps({"team_id": "admission-closed", "policy_version": version}),
                    }
                ],
                completion_ids=[],
                turns=[],
                reward=None,
            )
        model = f"{self.lora_name}-v{version}" if self.lora_name else self.model_name
        state, traces = await self.runner.run(task, model, version)
        turns = [
            TurnRecord(
                list(turn.prompt_token_ids),
                list(turn.completion_token_ids),
                list(turn.per_token_logps),
                list(turn.loss_mask[len(turn.prompt_token_ids) :]),
            )
            for trace in traces
            for turn in trace.turns
        ]
        self._push_metrics(
            {"team/unscorable": float(state["reward"] is None), "team/wall_seconds": state["wall_seconds"]}
        )
        return Rollout(
            completion=[{"role": "assistant", "content": json.dumps(state)}],
            completion_ids=[token for turn in turns for token in turn.output_ids],
            turns=turns,
            reward=state["reward"],
            rollout_id=state["team_id"],
        )

    async def _score_group(self, group):
        if self.config.max_groups and self._total_groups_scored >= self.config.max_groups:
            raise RuntimeError("Collection budget exhausted; inspect teams.jsonl before increasing --max-groups")
        packets = score_teams(group)
        rewards = [packet.rows[0]["metrics"]["reward_trained"] for packet in packets]
        mixed = len(set(rewards)) > 1
        with (self.runner.root / "groups.jsonl").open("a") as output:
            output.write(
                json.dumps(
                    {
                        "group": group.group_id,
                        "rewards": rewards,
                        "mixed": mixed,
                        "valid_teams": len(packets),
                        "expected_teams": self.config.num_generations,
                    }
                )
                + "\n"
            )
        self._push_metrics({"team/mixed_group": float(mixed)})
        return packets

    def _put_in_buffer(self, item):
        if isinstance(item, TeamPacket):
            item.enqueued_at = time.time()
            self._produced_tokens += item.forwarded_tokens
            self.rollout_buffer.put_nowait(item)
        else:
            super()._put_in_buffer(item)


class SwarmTrainer(AsyncGRPOTrainer):
    loop_cls = SwarmRolloutLoop

    def get_train_dataloader(self):
        if self.args.deepspeed:
            raise ValueError("Use FSDP2 for distributed swarm training")
        world = self.accelerator.num_processes
        if world == 1 and self.args.n_gpu > 1:
            raise ValueError("Use FSDP2 for multiple GPUs, or expose only one training GPU")
        if world > 1 and not self.is_fsdp_enabled:
            raise ValueError("Distributed swarm training requires FSDP2")
        if self.args.token_budget < self.args.swarm_settings["max_context"]:
            raise ValueError("token_budget must be at least max_context")
        self.team_collator = DataCollatorForRollout(
            self.processing_class.pad_token_id,
            1,
            groups_trained=self._trained_groups,
            metrics=self._metrics["train"],
            token_budget=self.args.token_budget,
        )
        self._rollout_dataset = TeamDataset(self) if self.accelerator.is_main_process else None
        group = torch.distributed.new_group(backend="gloo", timeout=timedelta(hours=2)) if world > 1 else None
        # Scatter complete teams ourselves; Accelerate's tensor dispatcher cannot split ragged chunks.
        return DataLoader(DistributedTeams(self, group), batch_size=None, collate_fn=identity, num_workers=0)

    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = torch.zeros((), device=self.accelerator.device)
        for chunk in inputs["chunks"]:
            loss += super().training_step(model, chunk, num_items_in_batch)
        return loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        actual = inputs["global_n_tokens"]
        inputs = {**inputs, "global_n_tokens": inputs["team_tokens"] * self.accelerator.num_processes}
        loss = super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
        self._step_trained_tokens += float(actual[0] - inputs["global_n_tokens"][0])
        return loss

    def _log_step_metrics(self):
        if torch.cuda.is_available():
            memory = torch.tensor(
                [torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated()],
                device=self.accelerator.device,
                dtype=torch.float64,
            )
            memory = self.accelerator.gather(memory).reshape(-1, 2)
            self._metrics["train"]["gpu/allocated_gb_mean"].append(float(memory[:, 0].mean() / 1e9))
            self._metrics["train"]["gpu/peak_allocated_gb_max"].append(float(memory[:, 1].max() / 1e9))
        if self.accelerator.is_main_process:
            self._metrics["train"]["team/queue_depth"].append(self.rollout_queue.qsize())
            self._metrics["train"]["policy/version"].append(self.model_version)
        return super()._log_step_metrics()

    def _sync_weight_merged(self, t0):
        """Gate whole inference calls, not just tokens, around TRL's NCCL transfer."""
        import httpx

        gateway = self.args.swarm_settings.get("gateway_url")
        if not gateway:
            raise ValueError("Full-model sync requires the call-boundary gateway")
        headers = {"x-swarm-control": os.environ["SWARM_GATEWAY_TOKEN"]}
        status = [None]
        if self.accelerator.is_main_process:
            try:
                response = httpx.post(f"{gateway}/control/drain", headers=headers, timeout=620)
                response.raise_for_status()
            except Exception as exc:
                status[0] = repr(exc)
        from accelerate.utils import broadcast_object_list

        status = broadcast_object_list(status)
        if status[0]:
            raise RuntimeError(f"Cannot drain inference calls: {status[0]}")
        try:
            times = super()._sync_weight_merged(t0)
            if self.accelerator.is_main_process:
                response = httpx.post(f"{self.args.vllm_server_base_url}/reset_prefix_cache", timeout=60)
                response.raise_for_status()
                if not response.json().get("success"):
                    raise RuntimeError("Could not reset the serving prefix cache after full-model sync")
                response = httpx.post(
                    f"{gateway}/control/publish", json={"version": self.model_version}, headers=headers, timeout=30
                )
                response.raise_for_status()
        except Exception:
            if self.accelerator.is_main_process:
                try:
                    httpx.post(f"{gateway}/control/fail", headers=headers, timeout=10)
                except httpx.HTTPError:
                    pass
            raise
        self.accelerator.wait_for_everyone()
        return times
