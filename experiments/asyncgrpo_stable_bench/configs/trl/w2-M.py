"""
W2-M (agentic: TermGrade on HF Sandboxes, terminus-2, Qwen3.5-9B) shared config as AsyncGRPOTrainer arguments. See
configs/SHARED_W2.md for the spec and configs/trl/EQUIVALENCE.md (W2 section) for the knob-by-knob mapping.

Launched by `launch/trl/w2.sbatch` with `accelerate launch --config_file configs/trl/w1-S.accelerate.yaml` (FSDP2), next
to a vLLM server and an OpenEnv Harbor server started there. Every rollout is one terminus-2 episode run by the Harbor
server in an HF sandbox; its model calls go through OpenEnv's capture proxy to vLLM, and TRL trains on the captured token
ids, logprobs and masks (the agent owns the loop). Reads from the environment:

    RUN_ID, RUN_DIR       run id (trackio run name) and output dir
    MODEL                 defaults to Qwen/Qwen3.5-9B
    VLLM_URL, HARBOR_URL  the vLLM server and the OpenEnv Harbor server
    SPLIT_DIR             the prepared Harbor task set (workloads/agentic/prepare.py)
    MAX_STEPS (60), SAVE_STEPS (20)
    NUM_GENERATIONS (8)   episodes per task (GRPO group)
    GRAD_ACCUM (6), TOKEN_BUDGET (65536): shared-arm batching, one packed row of <= TOKEN_BUDGET unique tokens per rank
                          per micro-batch, GRAD_ACCUM micro-batches per optimizer step (sized for ~64 episodes per step,
                          see EQUIVALENCE.md)
    CONCURRENCY (64)      episodes in flight = live sandboxes (hard cap: the harness session thread pool)
    AGENT_TIMEOUT (600), STEP_LIMIT (30)  per-episode agent wall clock and model-call budget
"""

import json
import os
import time
from functools import partial

import numpy as np
import torch
from datasets import Dataset
from termgrade_env import HARNESS, SANDBOX, TermGradeSessionFactory
from transformers import AutoTokenizer, TrainerCallback
from transformers.trainer_utils import get_last_checkpoint

from trl.experimental.async_grpo import AsyncGRPOConfig
from trl.experimental.async_grpo.openenv_harness import HarnessAsyncGRPOTrainer, _HarnessRolloutLoop


MODEL = os.environ.get("MODEL", "Qwen/Qwen3.5-9B")
RUN_ID = os.environ.get("RUN_ID", "trl-w2-M-local")
RUN_DIR = os.environ.get("RUN_DIR", f"/fsx/amine_dirhoussi/stable-bench-runs/trl/{RUN_ID}")
# Per-episode records (wall time, Harbor phases, turns, tokens), written by the sessions in the rollout process.
os.environ.setdefault("W2_EPISODES_FILE", os.path.join(RUN_DIR, "episodes.jsonl"))


class W2RolloutLoop(_HarnessRolloutLoop):
    """Group-mean advantage without std normalization (SHARED_W2 = SHARED_W1's loss)."""

    def compute_advantages(self, group):
        rewards = group.rewards
        scored = ~np.isnan(rewards)
        advantages = np.zeros_like(rewards)
        if scored.any():
            advantages[scored] = rewards[scored] - rewards[scored].mean()
        return advantages


class W2Trainer(HarnessAsyncGRPOTrainer):
    loop_cls = W2RolloutLoop


class BenchCallback(TrainerCallback):
    """Writes every log dict (metrics.jsonl), step spans (spans-train.jsonl) and per-rank memory stats (mem-rank*.jsonl).
    Same as configs/trl/w1-S.py."""

    def __init__(self, rank):
        self.rank = rank
        self.t_begin = None
        self.t_prev_end = time.time()
        self.t_end = None
        self.mem = open(os.path.join(RUN_DIR, f"mem-rank{rank}.jsonl"), "a", buffering=1)
        if rank == 0:
            self.metrics = open(os.path.join(RUN_DIR, "metrics.jsonl"), "a", buffering=1)
            self.spans = open(os.path.join(RUN_DIR, "spans-train.jsonl"), "a", buffering=1)

    def on_step_begin(self, args, state, control, **kwargs):
        if self.t_begin is None:
            self.t_begin = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        self.t_end = time.time()
        stats = torch.cuda.memory_stats()
        self.mem.write(
            json.dumps(
                {
                    "step": state.global_step,
                    "rank": self.rank,
                    "time": self.t_end,
                    "allocated_gb": stats.get("allocated_bytes.all.current", 0) / 1e9,
                    "reserved_gb": stats.get("reserved_bytes.all.current", 0) / 1e9,
                    "peak_allocated_gb": stats.get("allocated_bytes.all.peak", 0) / 1e9,
                    "peak_reserved_gb": stats.get("reserved_bytes.all.peak", 0) / 1e9,
                    "num_alloc_retries": stats.get("num_alloc_retries", 0),
                }
            )
            + "\n"
        )
        torch.cuda.reset_peak_memory_stats()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.rank != 0 or logs is None:
            return
        self.metrics.write(json.dumps({"step": state.global_step, "time": time.time(), **logs}) + "\n")
        if self.t_begin is not None and self.t_end is not None:
            sync = logs.get("perf/weight_sync_s") or 0.0
            step = state.global_step
            for kind, start, end in (
                ("wait", self.t_prev_end, self.t_begin),
                ("train_step", self.t_begin, self.t_end - sync),
                ("weight_sync", self.t_end - sync, self.t_end),
            ):
                self.spans.write(
                    json.dumps({"kind": kind, "rank": 0, "start": start, "end": end, "step": step}) + "\n"
                )
            self.t_prev_end, self.t_begin, self.t_end = self.t_end, None, None


def main():
    os.makedirs(RUN_DIR, exist_ok=True)
    factory = partial(
        TermGradeSessionFactory,
        os.environ["HARBOR_URL"],
        split=os.environ["SPLIT_DIR"],
        harness=HARNESS,
        sandbox=SANDBOX,
        llm_url=os.environ["VLLM_URL"],
        model=MODEL,
        agent_timeout_sec=float(os.environ.get("AGENT_TIMEOUT", "600")),
        agent_step_limit=int(os.environ.get("STEP_LIMIT", "30")),
    )
    # The dataset is the server's task list, so every prompt maps back to its task (prompt_rows), shuffled with seed 42.
    metadata = factory()
    dataset = Dataset.from_list(metadata.prompt_rows()).shuffle(seed=42)
    metadata._env.close()

    grad_accum = int(os.environ.get("GRAD_ACCUM", "6"))
    num_processes = int(os.environ.get("WORLD_SIZE", "4"))
    num_generations = int(os.environ.get("NUM_GENERATIONS", "8"))
    rollouts_per_step = int(os.environ.get("PROMPTS_PER_STEP", "8")) * num_generations
    config = AsyncGRPOConfig(
        output_dir=RUN_DIR,
        run_name=RUN_ID,
        report_to="trackio",
        project="asyncgrpo-stable-bench",
        trackio_space_id="aminediroHF/asyncgrpo-stable-bench",
        seed=42,
        # model: fp32 master weights + optimizer state under FSDP2 bf16 mixed precision (the trainer's default, as W1)
        dtype="float32",
        bf16=True,
        gradient_checkpointing=True,
        # generation: the agent samples through the capture proxy with these params (full vocabulary, T=1);
        # the per-turn token cap is the Harbor server's --max-output-tokens (8192)
        num_generations=num_generations,
        max_completion_length=8192,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        min_p=None,
        chat_template_kwargs={"enable_thinking": False},
        vllm_server_base_url=os.environ["VLLM_URL"],
        # batch (shared arm): rollouts_per_step only sizes bookkeeping; the step is GRAD_ACCUM packed micro-batches of
        # one <= TOKEN_BUDGET row per rank (see EQUIVALENCE.md W2)
        per_device_train_batch_size=max(1, rollouts_per_step // (grad_accum * num_processes)),
        gradient_accumulation_steps=grad_accum,
        token_budget=int(os.environ.get("TOKEN_BUDGET", "65536")),
        # optimizer: as W1
        learning_rate=1e-6,
        lr_scheduler_type="constant",
        warmup_steps=0,
        weight_decay=0.0,
        adam_beta1=0.9,
        adam_beta2=0.999,
        max_grad_norm=1.0,
        # loss: clipped GRPO objective, eps 0.2 / 0.28, no KL; masked IS against the vLLM sampler at the defaults
        score_centering=False,
        epsilon=0.2,
        epsilon_high=0.28,
        # async: trainer default staleness; in-flight episodes capped at the sandbox budget
        max_staleness=int(os.environ.get("MAX_STALENESS", "8")),  # 8 = prime-rl's default (user, 2026-11-11: match it)
        max_concurrent_rollouts=int(os.environ.get("CONCURRENCY", "64")),
        # duration and checkpoints
        max_steps=int(os.environ.get("MAX_STEPS", "60")),
        save_strategy="steps",
        save_steps=int(os.environ.get("SAVE_STEPS", "20")),
        logging_steps=1,
        log_completions=False,
        # FSDP: rank 0 waits for whole groups of multi-minute episodes before the first batch
        ddp_timeout=4 * 3600,
    )
    trainer = W2Trainer(
        model=MODEL,
        harness_session_factory=factory,
        args=config,
        train_dataset=dataset,
        processing_class=AutoTokenizer.from_pretrained(MODEL),
    )
    trainer.add_callback(BenchCallback(trainer.accelerator.process_index))
    trainer.train(resume_from_checkpoint=get_last_checkpoint(RUN_DIR) if os.path.isdir(RUN_DIR) else None)


if __name__ == "__main__":
    main()
