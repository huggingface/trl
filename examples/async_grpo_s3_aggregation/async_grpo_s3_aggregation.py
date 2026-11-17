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

# /// script
# dependencies = [
#     "trl[kernels]",
#     "datasets",
#     "httpx",
# ]
# ///

"""Multi-agent team RL with `AsyncGRPOTrainer`: information aggregation with Qwen3-0.6B, a replication of s3
(https://huggingface.co/papers/2611.08615).

One TRL rollout is one whole team episode: `generate_one` awaits `harness.run_team` (N token-level agents talking to
the vLLM server, with an in-memory message board) on the rollout loop's event loop, and returns every agent's sampled
stretches as token turns. Each agent's turns chain into one training row, so a team gives N rows, all with the
team's advantage.

The paper's objective for information aggregation:
- Team reward (Eq. 1): R = S * (1 - 0.25 * min(1, sum_i |tau_i| / (N L))), S = any agent's final list is correct,
  |tau_i| = tokens agent i generated, L = 8,192 work + 1,024 final tokens.
- Team-relative advantage (Eq. 2): `num_generations` = G = 8 team attempts per problem,
  A_g = (R_g - mean R) / (std R + 1e-8), TRL's default. A group whose teams all score the same is kept as
  advantage-0 rows (App. C: it still counts toward collection).
- Loss (Eq. 3): -1/|B| sum_g 1/|T_g| sum_{t in T_g} m_t rho_t A_g, with rho_t = pi_theta / mu (mu is the vLLM
  sampler's logprob) and m_t = 1[0.2 <= rho_t <= 5.0]. Here the importance mask is
  `vllm_importance_sampling_clip_min/max` = 0.2 / 5.0 and PPO clipping is off (`epsilon` = 1, `epsilon_high` = 1e9,
  so the min of the clipped and unclipped terms is rho * A). `TeamBatcher` replaces TRL's token-budget packing so
  that a micro-batch is exactly one team (its N agent rows) per training process. TRL's per-micro-batch token mean is
  then the team mean 1/|T_g| sum_t, and `gradient_accumulation_steps` = teams per update averages over teams
  (1/|B| sum_g). This is exact on one training process; with more, each micro-batch averages over one team per rank.
- Not implemented: the KL penalty listed for aggregation in the paper's Table 4 (coefficient and reference model are
  not published).

Run on 2 GPUs (vLLM on GPU 0, the trainer on GPU 1), from the TRL repository root:

    CUDA_VISIBLE_DEVICES=0 VLLM_SERVER_DEV_MODE=1 vllm serve Qwen/Qwen3-0.6B --port 8000 \
        --max-model-len 40960 --max-num-seqs 512 --enable-prefix-caching \
        --logprobs-mode processed_logprobs --weight-transfer-config '{"backend":"nccl"}'

    CUDA_VISIBLE_DEVICES=1 accelerate launch --num_processes 1 --mixed_precision bf16 \
        examples/async_grpo_s3_aggregation/async_grpo_s3_aggregation.py \
        --vllm-url http://127.0.0.1:8000 --teams-per-update 128 --updates 50 --concurrent-teams 128

See README.md for the results of this run and how to evaluate the checkpoint.
"""

import argparse
import logging
import math
import os
import random
from dataclasses import dataclass, field

import numpy as np
from datasets import Dataset
from harness import Settings, VLLMSampler, run_team

from trl import AsyncGRPOConfig, AsyncGRPOTrainer
from trl.trainer import async_grpo_trainer
from trl.trainer.async_grpo_trainer import (
    AsyncRolloutLoop,
    Rollout,
    RolloutSample,
    TokenBudgetBatcher,
    TreeRow,
    TurnRecord,
)


logger = logging.getLogger(__name__)


@dataclass
class TeamConfig(AsyncGRPOConfig):
    team_agents: int = field(default=4, metadata={"help": "agents per team (N)"})
    team_work_tokens: int = field(default=8192, metadata={"help": "per-agent work-phase token allowance"})
    team_final_tokens: int = field(default=1024, metadata={"help": "per-agent final-answer token allowance"})
    team_model: str = field(default="Qwen/Qwen3-0.6B", metadata={"help": "served model name on the vLLM server"})


def agent_turns(state) -> list[TurnRecord]:
    """One TurnRecord per contiguous stretch of model-sampled tokens; the context before it is the prompt."""
    toks, mask, lps = state.tokens, state.mask, state.logprobs
    turns, i = [], 0
    while i < len(toks):
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < len(toks) and mask[j]:
            j += 1
        turns.append(
            TurnRecord(
                prompt_ids=list(toks[:i]),
                output_ids=list(toks[i:j]),
                output_log_probs=[float(x) for x in lps[i:j]],
            )
        )
        i = j
    return turns


class TeamRolloutLoop(AsyncRolloutLoop):
    """Rollouts are `harness.run_team` episodes (see the module docstring)."""

    async def setup(self) -> None:
        c = self.config
        self._sampler = VLLMSampler(c.vllm_server_base_url, c.team_model)
        self._settings = Settings(work_tokens=c.team_work_tokens, final_tokens=c.team_final_tokens)
        self._rng = random.Random(os.getpid())

    async def _init_kv_cap(self) -> None:
        """Start at `max_concurrent_rollouts` teams. TRL's default (KV cache tokens / max_model_len) assumes one
        full-length sequence per rollout, about 14 here; a team is N shorter agent contexts, many of them idle at a
        time. The control loop still lowers the cap when KV cache usage passes 0.8."""
        self._kv_cap = max(1, self.config.max_concurrent_rollouts)

    async def close(self) -> None:
        await self._sampler.aclose()

    async def generate_one(self, prompt, tools, group_id: int = 0) -> Rollout:
        problem = int(prompt[-1]["content"].split("problem:", 1)[1])
        n = self.config.team_agents
        try:
            ep = await run_team(self._sampler, problem, self._rng.randrange(1 << 31), n, self._settings)
        except Exception as e:  # an infrastructure failure: return it unscored, so it gives no rows
            logger.warning("team episode for problem %d failed: %s: %s", problem, type(e).__name__, e)
            self._push_metrics({"team/episode_errors": 1.0})
            return Rollout(
                completion=[{"role": "assistant", "content": f"error {type(e).__name__}"}],
                completion_ids=[],
                turns=[],
                reward=None,
            )
        turns, completion_ids = [], []
        for st in ep.agents:
            t = agent_turns(st)
            turns += t
            completion_ids += [x for tr in t for x in tr.output_ids]
        msgs = sum(e["kind"] == "message" for st in ep.agents for e in st.events)
        self._push_metrics(
            {"team/success": float(ep.success), "team/tokens": float(ep.tokens), "team/messages": float(msgs)}
        )
        summary = [
            f"problem {problem} seed {ep.problem.seed} target {ep.problem.target} success {ep.success} "
            f"reward {ep.reward:.3f}"
        ]
        summary += [
            f"{st.name}: {st.work_generated}+{st.final_generated} tok, "
            f"{sum(e['kind'] == 'message' for e in st.events)} msgs, answer {st.answer}"
            for st in ep.agents
        ]
        return Rollout(
            completion=[{"role": "assistant", "content": "\n".join(summary)}],
            completion_ids=completion_ids,
            turns=turns,
            reward=ep.reward,
        )

    async def _score_group(self, group):
        """Keep a group whose teams all got the same reward, as advantage-0 rows (App. C: such groups "have zero
        advantage; they still count toward the collection budget"). TRL drops them and generates more groups until an
        update has `gradient_accumulation_steps` teams with signal; kept, they add no gradient but count in the 1/|B|
        team average, so an update is exactly `teams_per_update` teams collected, as in the paper."""
        samples = await super()._score_group(group)
        if samples or group.rewards is None or not len(group.completions_sequences):
            return _tag_teams(samples)
        kept = float(np.nanmean(group.rewards)) if not np.isnan(group.rewards).all() else 0.0
        out = []
        for prompt, completion, sequences in zip(
            group.prompts, group.completions, group.completions_sequences, strict=True
        ):
            for seq in sequences:
                n = len(seq.input_ids)
                out.append(
                    RolloutSample(
                        prompt=prompt,
                        completion=completion,
                        input_ids=np.asarray(seq.input_ids, dtype=np.int64),
                        completion_mask=np.asarray(seq.completion_mask, dtype=np.int64),
                        old_log_probs=np.asarray(seq.old_log_probs, dtype=np.float32),
                        sampling_mask=seq.sampling_mask,
                        sampler_ids=np.full((n, 0), -1, dtype=np.int64),
                        sampler_log_probs=np.full((n, 0), -math.inf, dtype=np.float32),
                        advantage=0.0,
                        model_version=group.model_version,
                        group_id=group.group_id,
                        metrics={"reward_trained": kept, "reward_group_std": 0.0},
                    )
                )
        return _tag_teams(out)


def _tag_teams(samples: list[RolloutSample]) -> list[RolloutSample]:
    """Mark which team each row came from (the episode summary is unique per team), for `TeamBatcher`."""
    for s in samples:
        s.metrics["team"] = s.completion[0]["content"]
    return samples


class TeamBatcher(TokenBudgetBatcher):
    """One whole team per row: each micro-batch is `num_processes` teams, one per rank.

    A team's rows arrive consecutively (a group's samples are queued together, team by team), so a team is complete
    when the next sample belongs to another team. The team tag is removed here, so it is never logged as a metric."""

    def __iter__(self):
        teams, team, key = [], [], None
        for sample in self.dataset:
            k = sample["metrics"].pop("team")
            if team and k != key:
                teams.append(team)
                team = []
                if len(teams) == self.num_processes:
                    yield [self._row(t) for t in teams]
                    teams = []
            team.append(sample)
            key = k

    @staticmethod
    def _row(team: list[dict]) -> TreeRow:
        row = TreeRow()
        for sample in team:
            row.add(sample)
        return row


class TeamTrainer(AsyncGRPOTrainer):
    loop_cls = TeamRolloutLoop

    def get_train_dataloader(self):
        """TRL's dataloader with `TeamBatcher` in place of `TokenBudgetBatcher`."""
        async_grpo_trainer.TokenBudgetBatcher = TeamBatcher
        try:
            return super().get_train_dataloader()
        finally:
            async_grpo_trainer.TokenBudgetBatcher = TokenBudgetBatcher


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--agents", type=int, default=4)
    ap.add_argument("--teams-per-update", type=int, default=128)
    ap.add_argument("--updates", type=int, default=50)
    ap.add_argument("--problems", type=int, default=1024, help="training problem ids 0..problems-1")
    ap.add_argument("--vllm-url", default=os.environ.get("VLLM_URL", "http://127.0.0.1:8000"))
    ap.add_argument("--output-dir", default="Qwen3-0.6B-S3-Aggregation")
    ap.add_argument("--concurrent-teams", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-7)
    ap.add_argument("--report-to", default="none")
    a = ap.parse_args()
    rows = [{"prompt": [{"role": "user", "content": f"problem:{i}"}]} for i in range(a.problems)]
    config = TeamConfig(
        output_dir=a.output_dir,
        team_agents=a.agents,
        team_model=a.model,
        num_generations=8,  # G team attempts per problem
        # TeamBatcher forms one-team micro-batches; the token budget only sizes how far generation may run ahead
        token_budget=a.agents * 40960,
        gradient_accumulation_steps=a.teams_per_update,
        max_steps=a.updates,
        learning_rate=a.lr,
        lr_scheduler_type="constant",
        warmup_steps=0,
        weight_decay=0.0,
        max_grad_norm=1.0,
        epsilon=1.0,
        epsilon_high=1e9,  # no PPO clipping: the term is m * rho * A
        vllm_importance_sampling_clip_min=0.2,
        vllm_importance_sampling_clip_max=5.0,
        score_centering=False,
        max_staleness=4,
        max_concurrent_rollouts=a.concurrent_teams,
        vllm_server_base_url=a.vllm_url,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=1,
        save_strategy="steps",
        save_steps=10,
        save_total_limit=1,
        report_to=a.report_to,
        log_completions=True,
        num_completions_to_print=1,
        rollout_worker_timeout=1800.0,
    )
    trainer = TeamTrainer(model=a.model, args=config, train_dataset=Dataset.from_list(rows), reward_funcs=[])
    trainer.train()
    trainer.end()


if __name__ == "__main__":
    main()
