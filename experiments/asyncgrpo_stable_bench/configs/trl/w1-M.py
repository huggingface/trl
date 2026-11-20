"""
W1-M (math, Qwen3.5-9B) shared config as AsyncGRPOTrainer arguments, identical to w1-S.py except the model. See EQUIVALENCE.md ("M").

Launched by `launch/trl/w1.sbatch` with `accelerate launch --config_file configs/trl/w1-M.accelerate.yaml`, next to a
vLLM server started there. Reads from the environment:

    RUN_ID        run id, also the trackio run name
    RUN_DIR       where checkpoints, metrics.jsonl, spans and mem snapshots go
    VLLM_URL      the vLLM server, e.g. http://localhost:8040
    MAX_STEPS     defaults to 200
    SAVE_STEPS    defaults to 50
    GRAD_ACCUM    micro-batches per optimizer step (each micro-batch is one packed row of TOKEN_BUDGET tokens per rank),
                  defaults to 16
    TOKEN_BUDGET  unique tokens per packed row, defaults to 49152; empty = the trainer default (vLLM max_model_len)
    ARM           shared (default) or recommended (TRL's own defaults for loss, advantage and batching, see EQUIVALENCE.md)
    MAX_COMPLETION  defaults to 16384 (SHARED_W1 rev. 2)
    ENABLE_THINKING defaults to false (SHARED_W1 rev. 2), passed to the chat template; empty = template default
    ABLATION      optional arm, see ABLATIONS.md: score_centering, std_adv, fixed_concurrency (with MAX_CONCURRENT)
    MAX_CONCURRENT  optional max_concurrent_rollouts
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, TrainerCallback
from transformers.trainer_utils import get_last_checkpoint

from trl.experimental.async_grpo import AsyncGRPOConfig, AsyncGRPOTrainer
from trl.experimental.async_grpo.async_rollout_worker import AsyncRolloutLoop


BENCH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(BENCH, "workloads", "math"))
from reward import PROMPT, compute_score  # noqa: E402


MODEL = "Qwen/Qwen3.5-9B"
RUN_ID = os.environ.get("RUN_ID", "trl-w1-M-local")
RUN_DIR = os.environ.get("RUN_DIR", f"/fsx/amine_dirhoussi/stable-bench-runs/trl/{RUN_ID}")
ABLATION = os.environ.get("ABLATION", "")
RECOMMENDED = os.environ.get("ARM", "shared") == "recommended"

_tokenizer = None


async def math_reward(completions, answer, completion_ids, **kwargs):
    """The shared reward on the raw completion text (thinking included), decoded from the sampled ids.

    Async so that the rollout worker runs it on its event loop, which is the main thread of its process: there
    math_verify's SIGALRM timeouts are on (see `compute_score`). A sync reward runs in a worker thread, where they are
    off, and an answer like `\\boxed{10^{10^{10}}}` then keeps sympy computing a huge integer while holding the GIL,
    which froze the whole rollout worker in run 364684.
    """
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL)
    texts = _tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
    return [compute_score(text, gold) for text, gold in zip(texts, answer, strict=True)]


class W1RolloutLoop(AsyncRolloutLoop):
    """One `generate` span per sampling request (measurement only; advantages use the trainer default, SHARED_W1 rev. 8)."""


    # ABLATION=fixed_concurrency: keep the in-flight cap at max_concurrent_rollouts (overrides private methods, see
    # ABLATIONS.md: there is no public switch)
    async def _init_concurrency_cap(self):
        if ABLATION != "fixed_concurrency":
            await super()._init_concurrency_cap()

    async def _control_loop(self, stop_event):
        if ABLATION != "fixed_concurrency":
            await super()._control_loop(stop_event)

    async def setup(self):
        await super().setup()
        self._spans = open(os.path.join(RUN_DIR, f"spans-generate-{os.getpid()}.jsonl"), "a", buffering=1)

    async def close(self):
        self._spans.close()
        await super().close()

    async def generate_one_turn(self, prompt_ids):
        start = time.time()
        turn = await super().generate_one_turn(prompt_ids)
        self._spans.write(
            json.dumps(
                {
                    "kind": "generate",
                    "worker": "vllm",
                    "start": start,
                    "end": time.time(),
                    "version": self.model_version,
                    "prompt_tokens": len(prompt_ids),
                    "tokens": len(turn.output_ids),
                }
            )
            + "\n"
        )
        return turn


class W1Trainer(AsyncGRPOTrainer):
    loop_cls = W1RolloutLoop


class BenchCallback(TrainerCallback):
    """Writes every log dict (metrics.jsonl), step spans (spans-train.jsonl) and per-rank memory stats (mem-rank*.jsonl)."""

    def __init__(self, rank):
        self.rank = rank
        self.t_begin = None
        self.t_prev_end = time.time()
        self.t_opt = None
        self.t_end = None
        self.mem = open(os.path.join(RUN_DIR, f"mem-rank{rank}.jsonl"), "a", buffering=1)
        if rank == 0:
            self.metrics = open(os.path.join(RUN_DIR, "metrics.jsonl"), "a", buffering=1)
            self.spans = open(os.path.join(RUN_DIR, "spans-train.jsonl"), "a", buffering=1)

    def on_step_begin(self, args, state, control, **kwargs):
        if self.t_begin is None:
            self.t_begin = time.time()

    def on_optimizer_step(self, args, state, control, **kwargs):
        self.t_opt = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        # Runs after the trainer's own on_step_end callbacks, so the weight sync is already done.
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
        now = time.time()
        self.metrics.write(json.dumps({"step": state.global_step, "time": now, **logs}) + "\n")
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


def kernel_bindings(model):
    """The functions the gated DeltaNet layers really call, found by walking the wrappers' closures."""
    import inspect

    modeling = sys.modules[type(model).__module__]
    found = {}
    for name in ("torch_chunk_gated_delta_rule", "causal_conv1d_fn"):
        chain, fn = [], vars(modeling).get(name)
        while callable(fn) and len(chain) < 6:
            if isinstance(fn, torch.nn.Module):  # transformers' hub-kernel wrapper turns the function into a module
                chain.append(f"{type(fn).__module__}.{type(fn).__qualname__}")
                fn = fn.forward.__func__ if inspect.ismethod(fn.forward) else fn.forward
                continue
            chain.append(f"{fn.__module__}.{fn.__qualname__}")
            nonlocals = inspect.getclosurevars(fn).nonlocals if inspect.isfunction(fn) else {}
            fn = next(
                (nonlocals[k] for k in ("func", "implementation", "conv", "chunk") if callable(nonlocals.get(k))), None
            )
        found[name] = chain
    return found


def main():
    os.makedirs(RUN_DIR, exist_ok=True)
    df = pd.read_parquet(os.path.join(BENCH, "workloads", "math", "data", "train.parquet"))
    dataset = Dataset.from_dict(
        {
            "prompt": [[{"role": "user", "content": PROMPT.format(problem=p)}] for p in df["problem"]],
            "answer": list(df["answer"]),
        }
    )

    grad_accum = int(os.environ.get("GRAD_ACCUM", "16"))
    num_processes = int(os.environ.get("WORLD_SIZE", "4"))
    config = AsyncGRPOConfig(
        output_dir=RUN_DIR,
        run_name=RUN_ID,
        report_to="trackio",
        project="asyncgrpo-stable-bench",
        trackio_space_id="aminediroHF/asyncgrpo-stable-bench",
        seed=42,
        # model: fp32 master weights + optimizer state under FSDP2 bf16 mixed precision (the trainer's default)
        dtype="float32",
        bf16=True,
        gradient_checkpointing=True,
        # generation
        num_generations=8,
        max_completion_length=int(os.environ.get("MAX_COMPLETION", "16384")),
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        min_p=None,
        chat_template_kwargs=(
            {"enable_thinking": thinking == "true"}
            if (thinking := os.environ.get("ENABLE_THINKING", "false"))
            else None
        ),
        vllm_server_base_url=os.environ["VLLM_URL"],
        # batch: 256 rollouts per step is what sizes the in-flight pool; the optimizer step itself is GRAD_ACCUM
        # packed micro-batches (see EQUIVALENCE.md, "prompts per optimizer step")
        per_device_train_batch_size=256 // (grad_accum * num_processes),
        gradient_accumulation_steps=grad_accum,
        token_budget=int(os.environ.get("TOKEN_BUDGET", "49152") or 0) or None,  # empty: trainer default
        # optimizer
        learning_rate=1e-6,
        lr_scheduler_type="constant",
        warmup_steps=0,
        weight_decay=0.0,
        adam_beta1=0.9,
        adam_beta2=0.999,
        max_grad_norm=1.0,
        # loss: clipped GRPO objective, eps 0.2 / 0.28, no KL; masked IS against the vLLM sampler at the defaults
        score_centering=RECOMMENDED or ABLATION == "score_centering",
        epsilon=0.2,
        epsilon_high=None if RECOMMENDED else 0.28,
        # sampler top-k logprobs for score centering (trainer default 8; 2 removes the vLLM pause stall, PAUSE_STALL.md)
        **(
            {"num_sampler_logprobs": int(os.environ["NUM_SAMPLER_LOGPROBS"])}
            if os.environ.get("NUM_SAMPLER_LOGPROBS")
            else {}
        ),
        # async: trainer defaults
        max_staleness=4,
        max_concurrent_rollouts=int(os.environ["MAX_CONCURRENT"]) if os.environ.get("MAX_CONCURRENT") else None,
        # duration and checkpoints
        max_steps=int(os.environ.get("MAX_STEPS", "200")),
        save_strategy="steps",
        save_steps=int(os.environ.get("SAVE_STEPS", "50")),
        logging_steps=1,
        log_completions=False,
    )
    trainer = W1Trainer(model=MODEL, args=config, train_dataset=dataset, reward_funcs=math_reward)
    trainer.add_callback(BenchCallback(trainer.accelerator.process_index))
    if trainer.accelerator.is_main_process:
        try:  # a diagnostic: never let it kill the run
            bindings = kernel_bindings(trainer.accelerator.unwrap_model(trainer.model))
        except Exception as e:
            bindings = {"error": repr(e)}
        print("kernel bindings:", json.dumps(bindings, indent=1))
        json.dump(bindings, open(os.path.join(RUN_DIR, "kernels.json"), "w"), indent=1)
    trainer.train(resume_from_checkpoint=get_last_checkpoint(RUN_DIR) if os.path.isdir(RUN_DIR) else None)


if __name__ == "__main__":
    main()
