"""
W2 (agentic: TermGrade, terminus-2 via OpenEnv Harbor) on a MoE model sharded at load time by transformers'
`DistributedConfig` (expert parallelism with token dispatch + FSDP2), LoRA, launched with `torchrun` by
`launch/trl/moe_w2.sbatch`. Same AsyncGRPO arguments as `configs/trl/w2-M.py` (the 9B W2 run), except:

    MODEL         defaults to Qwen/Qwen3.5-35B-A3B
    DTYPE         bfloat16 (default): bf16 params, grads and Adam states; float32: fp32 master + bf16 compute
    EP            ep_size, defaults to the number of trainer processes
    LORA_R (32), LR (1e-5)  LoRA on attention, GDN and shared-expert projections, as `launch/trl/moe_w1.py`
    SAVE_STEPS (20)  PEFT adapter checkpoints (`save_only_model`: resuming is not supported under load-time sharding);
                  `launch/trl/merge_lora.py` turns one into a full HF model for vLLM

Other knobs as in `configs/trl/w2-M.py`: RUN_ID, RUN_DIR, VLLM_URL, HARBOR_URL (+ HARBOR_URLS for sharded servers),
SPLIT_DIR, MAX_STEPS (60), NUM_GENERATIONS (8), PROMPTS_PER_STEP (8), GRAD_ACCUM, TOKEN_BUDGET, CONCURRENCY (64),
AGENT_TIMEOUT (600), STEP_LIMIT (30), MAX_STALENESS (4).
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
from transformers.distributed import DistributedConfig

import trl

from trl.experimental.async_grpo import AsyncGRPOConfig
from trl.experimental.async_grpo.openenv_harness import HarnessAsyncGRPOTrainer, _HarnessRolloutLoop


MODEL = os.environ.get("MODEL", "Qwen/Qwen3.5-35B-A3B")
DTYPE = os.environ.get("DTYPE", "bfloat16")
LORA_R = int(os.environ.get("LORA_R", "32"))
LORA_TARGETS = (
    r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|out_proj|gate_proj|"
    r"up_proj|down_proj)"
)
RUN_ID = os.environ.get("RUN_ID", "trl-moe-w2-local")
RUN_DIR = os.environ.get("RUN_DIR", f"/fsx/amine_dirhoussi/stable-bench-runs/trl/{RUN_ID}")
# Per-episode records (wall time, Harbor phases, turns, tokens), written by the sessions in the rollout process.
os.environ.setdefault("W2_EPISODES_FILE", os.path.join(RUN_DIR, "episodes.jsonl"))
assert "trl-wt-asyncgrpo-ep/" in trl.__file__, f"trl imported from {trl.__file__}, not the EP worktree"


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
        # model: sharded at load time (experts split EP ways, everything else FSDP over all ranks), as launch/trl/moe_w1.py
        model_init_kwargs={
            "dtype": DTYPE,
            "distributed_config": DistributedConfig(
                fsdp_size=num_processes,
                ep_size=int(os.environ.get("EP", num_processes)),
                fsdp_mixed_precision=DTYPE == "float32",
            ),
        },
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
        learning_rate=float(os.environ.get("LR", "1e-5" if LORA_R else "1e-6")),
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
        max_staleness=int(os.environ.get("MAX_STALENESS", "4")),
        max_concurrent_rollouts=int(os.environ.get("CONCURRENCY", "64")),
        # duration and checkpoints
        max_steps=int(os.environ.get("MAX_STEPS", "60")),
        save_strategy="steps" if int(os.environ.get("SAVE_STEPS", "20")) else "no",
        save_steps=int(os.environ.get("SAVE_STEPS", "20")) or 500,
        # resuming is not supported for models sharded at load time
        save_only_model=True,
        logging_steps=1,
        log_completions=False,
        # FSDP: rank 0 waits for whole groups of multi-minute episodes before the first batch
        ddp_timeout=4 * 3600,
    )
    peft_config = None
    if LORA_R:
        from peft import LoraConfig

        peft_config = LoraConfig(r=LORA_R, lora_alpha=2 * LORA_R, target_modules=LORA_TARGETS)
    trainer = W2Trainer(
        model=MODEL,
        harness_session_factory=factory,
        args=config,
        train_dataset=dataset,
        processing_class=AutoTokenizer.from_pretrained(MODEL),
        peft_config=peft_config,
    )
    trainer.add_callback(BenchCallback(trainer.accelerator.process_index))
    print(f"[rank{trainer.accelerator.process_index}] trl from {trl.__file__}, "
          f"alloc_after_init={torch.cuda.memory_allocated() / 1e9:.1f}GB", flush=True)
    trainer.train()


if __name__ == "__main__":
    main()
