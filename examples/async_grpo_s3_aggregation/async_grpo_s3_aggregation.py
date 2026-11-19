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

One TRL rollout is one whole team episode: `generate_one` awaits `harness.run_team` (N token-level agents with an
in-memory message board) on the rollout loop's event loop, and returns every agent's sampled stretches as token turns.
The agents sample through the loop's own `generate_one_turn` (so model versions, LoRA adapters, retries and the
prefix-cache salt behave as in any TRL rollout) and render prompts with the loop's tokenizer and chat template. Each
agent's turns chain into one training row, so a team gives N rows, all with the team's advantage.

The paper's objective for information aggregation:
- Team reward (Eq. 1): R = S * (1 - 0.25 * min(1, sum_i |tau_i| / (N L))), S = any agent's final list is correct,
  |tau_i| = tokens agent i generated, L = 8,192 work + 1,024 final tokens.
- Team-relative advantage (Eq. 2): `num_generations` = G = 8 team attempts per problem,
  A_g = (R_g - mean R) / (std R + 1e-8), TRL's default. A group whose teams all score the same is kept as
  advantage-0 rows (App. C: it still counts toward collection).
- Loss (Eq. 3): -1/|B| sum_g 1/|T_g| sum_{t in T_g} m_t rho_t A_g, with rho_t = pi_theta / mu (mu is the vLLM
  sampler's logprob) and m_t = 1[0.2 <= rho_t <= 5.0]. Here the importance mask is
  `vllm_importance_sampling_clip_min/max` = 0.2 / 5.0 and PPO clipping is off (`epsilon` = 1, `epsilon_high` = 1e9,
  so the min of the clipped and unclipped terms is rho * A). TRL's loss is a token mean per micro-batch; here it is a
  team mean: `_score_group` scales each agent row's advantage by N / |T_g| (|T_g| = its team's trained tokens), and
  `TeamTrainer.compute_loss` divides the micro-batch's summed token losses by its agent rows instead of its tokens.
  Each team then contributes its token mean 1/|T_g| sum_t, and a micro-batch averages over its teams (rows / N). With
  TRL's token-budget packing a team can be split across micro-batches; the mean over micro-batches stands in for
  1/|B|, exact when the micro-batches of an update hold the same number of teams.
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
import itertools
import logging
import math
import os
import random
from dataclasses import dataclass, field

import numpy as np
import torch
from datasets import Dataset
from harness import Chat, Settings, run_team

from trl import AsyncGRPOConfig, AsyncGRPOTrainer
from trl.trainer.async_grpo_trainer import (
    AsyncRolloutLoop,
    Rollout,
    RolloutSample,
    TurnRecord,
)


logger = logging.getLogger(__name__)


@dataclass
class TeamConfig(AsyncGRPOConfig):
    team_agents: int = field(default=4, metadata={"help": "agents per team (N)"})
    team_work_tokens: int = field(default=8192, metadata={"help": "per-agent work-phase token allowance"})
    team_final_tokens: int = field(default=1024, metadata={"help": "per-agent final-answer token allowance"})


def team_metrics(ep) -> dict[str, float]:
    """Per-episode team metrics, averaged over the episodes of a logging step: counts are per team, `frac` ones are
    over the team's agents."""
    events = [e for st in ep.agents for e in st.events]
    sent = [e for e in events if e["kind"] == "message"]
    ends = [e["reason"] for e in events if e["kind"] == "work_end"]
    n = len(ep.agents)
    return {
        "team/success": float(ep.success),
        "team/reward": ep.reward,
        "team/tokens": float(ep.tokens),
        "team/final_tokens": float(sum(st.final_generated for st in ep.agents)),
        "team/messages": float(len(sent)),
        "team/broadcasts": float(sum(e["to"].lower() == "all" for e in sent)),
        "team/bad_recipients": float(sum(not e["ok"] for e in sent)),
        "team/reads": float(sum(e["kind"] == "read" for e in events)),
        "team/notifications": float(sum(e["kind"] == "notified" for e in events)),
        "team/silent_agent_frac": sum(not any(e["kind"] == "message" for e in st.events) for st in ep.agents) / n,
        "team/correct_agent_frac": sum(st.answer == ep.problem.target for st in ep.agents) / n,
        **{f"team/work_end_{r}_frac": ends.count(r) / n for r in ("answered", "idle", "allowance", "context")},
    }


def team_mean_advantages(samples: list, n_agents: int) -> list:
    """Scale each agent row's advantage by N / |T_g|, with |T_g| its team's trained tokens (all its agents' rows),
    for `TeamTrainer.compute_loss`. A group's samples are its teams' rows; the rollout summary in `completion` is
    unique per team."""
    tokens: dict[str, int] = {}
    for s in samples:
        team = s.completion[0]["content"]
        tokens[team] = tokens.get(team, 0) + int(np.asarray(s.completion_mask)[1:].sum())  # TRL's trained positions
    for s in samples:
        s.advantage *= n_agents / max(1, tokens[s.completion[0]["content"]])
    return samples


class LoopSampler:
    """The harness's sampler on `AsyncRolloutLoop.generate_one_turn`, which samples up to `max_completion_length`
    (the harness's request size) with the config's sampling settings. A request for fewer tokens (the end of an
    agent's allowance) keeps the first `max_tokens`: a sample cut short is a sample with a smaller limit."""

    def __init__(self, loop: AsyncRolloutLoop):
        self.loop = loop

    async def generate(self, prompt_ids, max_tokens):
        turn = await self.loop.generate_one_turn(prompt_ids)
        ids, lps = list(turn.output_ids), list(turn.output_log_probs)
        cut = len(ids) > max_tokens
        ids, lps = ids[:max_tokens], lps[:max_tokens]
        # vLLM stopped by itself when it returned fewer than max_completion_length tokens or ended on an EOS token
        stopped = not cut and (len(ids) < self.loop.config.max_completion_length or ids[-1] in self.loop.eos_token_ids)
        return ids, lps, "stop" if stopped else "length"


class TeamRolloutLoop(AsyncRolloutLoop):
    """Rollouts are `harness.run_team` episodes (see the module docstring)."""

    async def setup(self) -> None:
        c = self.config
        if self.sampling_replay:
            raise ValueError("the team harness keeps no sampling masks: use top_p=1.0 and top_k=0")
        if c.team_final_tokens > c.max_completion_length:
            raise ValueError("team_final_tokens must fit in one request of max_completion_length tokens")
        self._sampler = LoopSampler(self)
        self._settings = Settings(
            work_tokens=c.team_work_tokens, final_tokens=c.team_final_tokens, chunk=c.max_completion_length
        )
        kwargs = {"enable_thinking": self._settings.enable_thinking, **self.chat_template_kwargs}
        self._chat = Chat(self.tokenizer, chat_template=self.chat_template, **kwargs)
        self._rng = random.Random(os.getpid())

    async def generate_one(self, prompt, tools, group_id: int = 0) -> Rollout:
        problem = int(prompt[-1]["content"].split("problem:", 1)[1])
        n = self.config.team_agents
        try:
            ep = await run_team(self._sampler, self._chat, problem, self._rng.randrange(1 << 31), n, self._settings)
        except Exception as e:  # an infrastructure failure: return it unscored, so it gives no rows
            logger.warning("team episode for problem %d failed: %s: %s", problem, type(e).__name__, e)
            self._push_metrics({"team/episode_errors": 1.0})
            return Rollout(
                completion=[{"role": "assistant", "content": f"error {type(e).__name__}"}],
                completion_ids=[],
                turns=[],
                reward=None,
            )
        # one turn per stretch of model-sampled tokens; the context before it is its prompt
        turns = []
        for st in ep.agents:
            i = 0
            for sampled, run in itertools.groupby(st.mask):
                k = len(list(run))
                if sampled:
                    turns.append(
                        TurnRecord(
                            prompt_ids=st.tokens[:i],
                            output_ids=st.tokens[i : i + k],
                            output_log_probs=st.logprobs[i : i + k],
                        )
                    )
                i += k
        completion_ids = [x for tr in turns for x in tr.output_ids]
        self._push_metrics(team_metrics(ep))
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
        """Scale advantages for the team-mean loss (`team_mean_advantages`), and keep a group whose teams all got the
        same reward as advantage-0 rows (App. C: such groups "have zero advantage; they still count toward the
        collection budget"). TRL drops them and generates more groups instead; kept, they add no gradient but count in
        the average over teams."""
        samples = await super()._score_group(group)
        if not samples and group.rewards is not None and len(group.completions_sequences):
            kept = float(np.nanmean(group.rewards)) if not np.isnan(group.rewards).all() else 0.0
            samples = [
                RolloutSample(
                    prompt=prompt,
                    completion=completion,
                    input_ids=np.asarray(seq.input_ids, dtype=np.int64),
                    completion_mask=np.asarray(seq.completion_mask, dtype=np.int64),
                    old_log_probs=np.asarray(seq.old_log_probs, dtype=np.float32),
                    sampling_mask=seq.sampling_mask,
                    sampler_ids=np.full((len(seq.input_ids), 0), -1, dtype=np.int64),
                    sampler_log_probs=np.full((len(seq.input_ids), 0), -math.inf, dtype=np.float32),
                    advantage=0.0,
                    model_version=group.model_version,
                    group_id=group.group_id,
                    metrics={"reward_trained": kept, "reward_group_std": 0.0},
                )
                for prompt, completion, sequences in zip(
                    group.prompts, group.completions, group.completions_sequences, strict=True
                )
                for seq in sequences
            ]
        return team_mean_advantages(samples, self.config.team_agents)


class TeamTrainer(AsyncGRPOTrainer):
    loop_cls = TeamRolloutLoop

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """TRL divides a micro-batch's summed token losses by its trained tokens (summed over ranks). Divide by its
        agent rows instead: with advantages scaled by N / |T_g|, that is the mean over the micro-batch's teams of each
        team's token mean."""
        tokens = inputs["global_n_tokens"]
        segments = inputs["segment_id"][inputs["token_index"] >= 0]
        rows = torch.tensor(float(segments.max() + 1) if segments.numel() else 0.0, device=tokens.device)
        rows = float(self.accelerator.reduce(rows, reduction="sum"))
        inputs["global_n_tokens"] = torch.full_like(tokens, rows)
        loss = super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
        self._step_trained_tokens += float(tokens[0]) - rows  # TRL counts trained tokens from the same input
        return loss


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
        num_generations=8,  # G team attempts per problem
        max_completion_length=1024,  # tokens per generation request, so the harness's chunk between notifications
        # a 40,960-token micro-batch holds about two teams (a team forwards about 17k tokens with shared prefixes)
        token_budget=40960,
        gradient_accumulation_steps=max(1, a.teams_per_update // 2),
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
