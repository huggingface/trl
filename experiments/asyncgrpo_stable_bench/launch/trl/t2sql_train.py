"""
miles' text-to-SQL recipe (examples/text_to_sql) with TRL's AsyncGRPOTrainer on a MoE model sharded at load time by
transformers' `DistributedConfig` (EP + FSDP2), LoRA, launched with `torchrun` by `t2sql_train.sbatch`. The MoE pieces
are `launch/trl/moe_w2.py`'s; the episode and reward are `t2sql_episode.py` (miles' sql_episode.py on River's recipe modules).

    MODEL (Qwen/Qwen3.5-35B-A3B), DTYPE (bfloat16), EP (= trainer processes), LORA_R (32)
    TRAIN_FILE    convert.py jsonl (train.jsonl, or a pass-rate-filtered one)
    LR (5e-5), WARMUP (10)  River's LoRA r32 optimizer: linear warmup, cosine decay to 10% over MAX_STEPS
    MAX_STEPS, SAVE_STEPS (20), MAX_STALENESS (8), GRAD_ACCUM, TOKEN_BUDGET (32768), CONCURRENCY (256)
"""

import json
import os
import time

import torch
from datasets import Dataset
from t2sql_episode import MAX_TOKENS_PER_TURN, Text2SQLLoop
from transformers import AutoTokenizer, TrainerCallback
from transformers.distributed import DistributedConfig

import trl

from trl.experimental.async_grpo import AsyncGRPOConfig, AsyncGRPOTrainer


MODEL = os.environ.get("MODEL", "Qwen/Qwen3.5-35B-A3B")
DTYPE = os.environ.get("DTYPE", "bfloat16")
LORA_R = int(os.environ.get("LORA_R", "32"))
LORA_TARGETS = (
    r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|out_proj|gate_proj|"
    r"up_proj|down_proj)"
)
RUN_ID = os.environ.get("RUN_ID", "trl-t2sql-local")
RUN_DIR = os.environ.get("RUN_DIR", f"/fsx/amine_dirhoussi/stable-bench-runs/trl-moe-ep/{RUN_ID}")
os.environ.setdefault("T2SQL_EPISODES_FILE", os.path.join(RUN_DIR, "episodes.jsonl"))
assert "trl-wt-asyncgrpo-ep/" in trl.__file__, f"trl imported from {trl.__file__}, not the EP worktree"


class T2SQLTrainer(AsyncGRPOTrainer):
    loop_cls = Text2SQLLoop


class BenchCallback(TrainerCallback):
    """metrics.jsonl (every log dict) and mem-rank*.jsonl (per-rank memory per step), as launch/trl/moe_w2.py."""

    def __init__(self, rank):
        self.rank = rank
        self.mem = open(os.path.join(RUN_DIR, f"mem-rank{rank}.jsonl"), "a", buffering=1)
        if rank == 0:
            self.metrics = open(os.path.join(RUN_DIR, "metrics.jsonl"), "a", buffering=1)

    def on_step_end(self, args, state, control, **kwargs):
        stats = torch.cuda.memory_stats()
        self.mem.write(json.dumps({
            "step": state.global_step,
            "rank": self.rank,
            "time": time.time(),
            "allocated_gb": stats.get("allocated_bytes.all.current", 0) / 1e9,
            "reserved_gb": stats.get("reserved_bytes.all.current", 0) / 1e9,
            "peak_allocated_gb": stats.get("allocated_bytes.all.peak", 0) / 1e9,
            "peak_reserved_gb": stats.get("reserved_bytes.all.peak", 0) / 1e9,
            "num_alloc_retries": stats.get("num_alloc_retries", 0),
        }) + "\n")
        torch.cuda.reset_peak_memory_stats()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.rank == 0 and logs is not None:
            self.metrics.write(json.dumps({"step": state.global_step, "time": time.time(), **logs}) + "\n")


def main():
    os.makedirs(RUN_DIR, exist_ok=True)
    rows = [json.loads(line) for line in open(os.environ["TRAIN_FILE"])]
    # miles: --rollout-shuffle
    dataset = Dataset.from_list([{"prompt": r["prompt"], "task": r["task"], "id": r["id"]} for r in rows]).shuffle(seed=42)

    num_processes = int(os.environ.get("WORLD_SIZE", "4"))
    grad_accum = int(os.environ.get("GRAD_ACCUM", "32"))
    max_steps = int(os.environ.get("MAX_STEPS", "100"))
    save_steps = int(os.environ.get("SAVE_STEPS", "20"))
    config = AsyncGRPOConfig(
        output_dir=RUN_DIR,
        run_name=RUN_ID,
        report_to="trackio",
        project="asyncgrpo-stable-bench",
        trackio_space_id="aminediroHF/asyncgrpo-stable-bench",
        seed=42,
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
        # generation: miles' 16 samples per question at T=1.0 over the full vocabulary; the episode sets each
        # request's max_tokens itself (3,072 per turn, 29,696 per episode)
        num_generations=16,
        max_completion_length=MAX_TOKENS_PER_TURN,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        min_p=None,
        vllm_server_base_url=os.environ["VLLM_URL"],
        vllm_server_timeout=3600,
        # batch: each step is GRAD_ACCUM micro-batches of one <= TOKEN_BUDGET-token row per rank; TOKEN_BUDGET >= the
        # 29,696-token episode cap so no episode is dropped
        per_device_train_batch_size=1,
        gradient_accumulation_steps=grad_accum,
        token_budget=int(os.environ.get("TOKEN_BUDGET", "32768")),
        # optimizer: River's LoRA r32 settings (lr 5e-5, warmup 10, cosine to 10%, adam beta2 0.95)
        learning_rate=float(os.environ.get("LR", "5e-5")),
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr_rate": 0.1},
        warmup_steps=int(os.environ.get("WARMUP", "10")),
        weight_decay=0.0,
        adam_beta1=0.9,
        adam_beta2=0.95,
        max_grad_norm=1.0,
        # loss: miles' clipped GRPO objective (eps 0.2 / 0.28, no KL, no entropy bonus); masked IS against the sampler
        score_centering=False,
        epsilon=0.2,
        epsilon_high=0.28,
        max_staleness=int(os.environ.get("MAX_STALENESS", "8")),
        # the recipe preset's max_concurrent_rollouts for this model
        max_concurrent_rollouts=int(os.environ.get("CONCURRENCY", "256")),
        max_steps=max_steps,
        save_strategy="steps" if save_steps else "no",
        save_steps=save_steps or 500,
        save_only_model=True,
        logging_steps=1,
        log_completions=False,
        ddp_timeout=4 * 3600,
    )
    from peft import LoraConfig

    peft_config = LoraConfig(r=LORA_R, lora_alpha=2 * LORA_R, target_modules=LORA_TARGETS)
    trainer = T2SQLTrainer(
        model=MODEL,
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
