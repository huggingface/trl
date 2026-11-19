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

import json
import queue
import time
from dataclasses import dataclass
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
            max(trainer.args.max_staleness, 1) * trainer.args.gradient_accumulation_steps * trainer.args.token_budget
        )
        self.token_allowance.value = self.lookahead_tokens

    def __iter__(self):
        trainer = self.trainer
        while True:
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
            trainer._metrics["train"]["team/reward"].append(packet.rows[0]["metrics"]["reward_trained"])
            yield pack_team(packet, trainer.args.token_budget, trainer.team_collator)


def identity(value):
    return value


class SwarmRolloutLoop(AsyncRolloutLoop):
    async def setup(self):
        if self.config.log_completions:
            raise ValueError("Use log_completions=False; swarm outcomes are recorded in teams.jsonl")
        if self.sampling_replay or self.config.score_centering:
            raise ValueError("This example requires full-vocabulary sampling and score_centering=False")
        if self.lora_name is None:
            raise ValueError("Use LoRA so each running team can pin one immutable policy version")
        self.runner = TeamRunner(Settings(**self.config.swarm_settings))
        await self.runner.start()

    async def close(self):
        await self.runner.close()

    async def generate_one(self, prompt, tools, group_id=0):
        task = Path(prompt[-1]["content"])
        version = self.model_version
        state, traces = await self.runner.run(task, f"{self.lora_name}-v{version}", version)
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
            output.write(json.dumps({"group": group.group_id, "rewards": rewards, "mixed": mixed}) + "\n")
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
        if self.accelerator.num_processes != 1 or self.args.n_gpu > 1:
            raise ValueError("The swarm example supports one training GPU; use a separate GPU for vLLM")
        if self.args.deepspeed or self.args.fsdp:
            raise ValueError("Use plain single-GPU LoRA training for this example")
        if self.args.token_budget < self.args.swarm_settings["max_context"]:
            raise ValueError("token_budget must be at least max_context")
        self.team_collator = DataCollatorForRollout(
            self.processing_class.pad_token_id,
            1,
            groups_trained=self._trained_groups,
            metrics=self._metrics["train"],
            token_budget=self.args.token_budget,
        )
        self._rollout_dataset = TeamDataset(self)
        # No dispatcher: this v1 intentionally supports one training rank and one logical team per item.
        return DataLoader(self._rollout_dataset, batch_size=None, collate_fn=identity, num_workers=0)

    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = torch.zeros((), device=self.accelerator.device)
        for chunk in inputs["chunks"]:
            loss += super().training_step(model, chunk, num_items_in_batch)
        return loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        actual = inputs["global_n_tokens"]
        inputs = {**inputs, "global_n_tokens": inputs["team_tokens"]}
        loss = super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
        self._step_trained_tokens += float(actual[0] - inputs["team_tokens"][0])
        return loss
