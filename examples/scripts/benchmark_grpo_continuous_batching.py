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
#     "trl",
#     "kernels",
# ]
# ///

"""
Benchmark: default generate() vs transformers continuous batching (CB) in GRPOTrainer.

== KEY FINDINGS (A100 80GB, kernels-community/flash-attn2, Llama-3.2-1B-Instruct) ==

  Dataset        N   Max tok  Default           CB                Speedup  VRAM delta
  ─────────────────────────────────────────────────────────────────────────────────────
  Uniform        8   512      23.51 ± 0.12 s    30.42 ± 0.50 s    0.78x    +1.23 GB
  GSM8K          8   2048     18.24 ± 12.81 s   18.16 ± 8.12 s    1.00x    +17.86 GB
  GSM8K         32   2048     37.06 ± 8.63 s    29.91 ± 10.09 s   1.24x    +7.39 GB   ← CB wins
  GSM8K         64   2048     41.12 ± 12.69 s   32.84 ± 7.00 s    1.25x    -16.66 GB  ← CB wins + less VRAM

== WHY ==

CB's advantage comes from sequence removal: when a sequence finishes, CB reclaims its KV slots
immediately and retires it from the batch. Default generate() keeps all N sequences alive until
the slowest one finishes, running full forward passes over padded "done" sequences for every
remaining decode step.

  N=8:  sequence-removal savings are too small to overcome the ~0.5-1.5 s per-step
        ContinuousBatchingManager reinit overhead → CB is slower or at parity.

  N=32: ~25-30 of 32 GSM8K sequences finish within 500 decode steps (mean ~360 tokens,
        max 2048). CB drops them immediately; default runs the full batch to 2048. The ~4-5x
        KV bandwidth reduction in the tail overcomes the reinit overhead → 1.24x speedup.

  N=64: same mechanism, larger effect. Additionally, default generate() eagerly allocates KV
        cache for 64 × 2048 sequences (~41 GB) while CB pre-allocates a fixed fraction of
        free VRAM (~25 GB with max_memory_percent=0.3) regardless of N. The VRAM delta
        inverts: CB uses 16.66 GB *less* than default. CB also has lower step-time variance
        (7.35 s vs 12.89 s stdev) because long-tail sequences don't block the whole batch.

  Rule of thumb: CB pulls ahead for GRPO at N≥32 with variable completion lengths (e.g.,
  math reasoning). For small batches or uniform prompts, use default generate().

== MEMORY GUIDANCE ==

  TRL defaults max_memory_percent to 0.5 instead of transformers' 0.9. Tune it down further
  for large batches. Recommended values:

    N≤16:   max_memory_percent=0.4  (leave 60% for training + activations)
    N≥32:   max_memory_percent=0.3  (training + gradient checkpointing need ~20-25 GB)

  Gradient checkpointing is strongly recommended at N≥32 to reduce activation memory
  (~43 GB → ~11 GB for a 64-sequence training forward pass on Llama-3.2-1B).

== USAGE ==

  # Small batch (N=8, worst case for CB — shows baseline parity):
  python examples/scripts/benchmark_grpo_continuous_batching.py

  # Large batch (N=32, CB break-even point):
  python examples/scripts/benchmark_grpo_continuous_batching.py --large-batch --num-generations 32

  # Large batch (N=64, CB wins throughput + VRAM — recommended starting point):
  python examples/scripts/benchmark_grpo_continuous_batching.py --large-batch

  # Single backend only:
  python examples/scripts/benchmark_grpo_continuous_batching.py --large-batch --backend continuous_batching

  # Different model:
  python examples/scripts/benchmark_grpo_continuous_batching.py --large-batch --model meta-llama/Llama-3.2-3B-Instruct
"""

import argparse
import logging
import statistics
import time

import torch
from datasets import Dataset, load_dataset

from trl import GRPOConfig, GRPOTrainer


logging.getLogger("ContinuousBatchingLogger").setLevel(logging.ERROR)


WARMUP_GENS = 3
MEASURE_GENS = 10

DEFAULT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"

LARGE_BATCH_NUM_GENERATIONS = 64
LARGE_BATCH_MAX_COMPLETION_LENGTH = 2048
# Training micro-batch for large-batch mode; gradient accumulation decouples it from num_generations.
LARGE_BATCH_TRAIN_BATCH_SIZE = 16


def dummy_reward(completions, **kwargs):
    return [0.0] * len(completions)


def make_uniform_dataset(n_samples: int = 256) -> Dataset:
    prompt = "Solve the following problem step by step and explain your reasoning. " * 10
    return Dataset.from_dict({"prompt": [prompt] * n_samples})


def make_gsm8k_dataset(n_samples: int = 256) -> Dataset:
    dataset = load_dataset("openai/gsm8k", "socratic", split="test")
    dataset = dataset.select(range(min(n_samples, len(dataset))))
    return Dataset.from_dict({"prompt": dataset["question"]})


def run_benchmark(
    backend: str,
    model_name: str,
    num_generations: int,
    max_completion_length: int,
    attn_impl: str,
    large_batch: bool,
) -> dict:
    print(f"\n{'=' * 60}")
    print(f"Backend: {backend}  |  N: {num_generations}  |  max_completion_length: {max_completion_length}")
    print(f"attn_impl: {attn_impl}  |  large_batch: {large_batch}")
    print(f"{'=' * 60}")

    use_cb = backend == "continuous_batching"

    if large_batch:
        train_batch_size = LARGE_BATCH_TRAIN_BATCH_SIZE
        grad_accum = num_generations // train_batch_size
    else:
        train_batch_size = num_generations
        grad_accum = 1

    # default max_memory_percent=0.9 leaves no VRAM for the training backward pass
    if use_cb:
        max_memory_percent = 0.3 if large_batch else 0.4
        cb_config = {"use_cuda_graph": False, "max_memory_percent": max_memory_percent}
    else:
        cb_config = None

    config = GRPOConfig(
        output_dir=f"/tmp/grpo_bench_{backend}",
        max_steps=WARMUP_GENS + MEASURE_GENS,
        per_device_train_batch_size=train_batch_size,
        gradient_accumulation_steps=grad_accum,
        num_generations=num_generations,
        max_completion_length=max_completion_length,
        temperature=0.9,
        use_transformers_continuous_batching=use_cb,
        transformers_continuous_batching_config=cb_config,
        model_init_kwargs={"attn_implementation": attn_impl},
        gradient_checkpointing=large_batch,
        save_strategy="no",
        logging_strategy="no",
        report_to="none",
        bf16=True,
        disable_dropout=False,
    )

    dataset = make_gsm8k_dataset() if large_batch else make_uniform_dataset()

    trainer = GRPOTrainer(
        model=model_name,
        reward_funcs=dummy_reward,
        args=config,
        train_dataset=dataset,
    )

    original_compute_loss = trainer.compute_loss

    def fast_compute_loss(model, inputs, **kwargs):
        return original_compute_loss(model, inputs, **kwargs).detach().requires_grad_()

    trainer.compute_loss = fast_compute_loss

    generation_times = []
    original_generate = trainer._generate

    def timed_generate(prompts):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        result = original_generate(prompts)
        torch.cuda.synchronize()
        generation_times.append(time.perf_counter() - t0)
        return result

    trainer._generate = timed_generate
    trainer.train()

    measured = generation_times[WARMUP_GENS:]

    # upper bound: assumes every sequence reaches max_completion_length
    total_tokens = train_batch_size * num_generations * max_completion_length * len(measured)
    peak_vram_gb = torch.cuda.max_memory_allocated() / 1024**3

    return {
        "backend": backend,
        "mean_gen_time_s": statistics.mean(measured),
        "stdev_gen_time_s": statistics.stdev(measured) if len(measured) > 1 else 0.0,
        "tokens_per_sec": total_tokens / sum(measured),
        "peak_vram_gb": peak_vram_gb,
    }


def print_results(results: list[dict]) -> None:
    print(f"\n{'=' * 60}")
    print("RESULTS")
    print(f"{'=' * 60}")
    header = f"{'Backend':<25} {'Gen time (s)':<22} {'Tokens/s':<15} {'Peak VRAM (GB)':<15}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['backend']:<25} "
            f"{r['mean_gen_time_s']:.2f} ± {r['stdev_gen_time_s']:.2f}{'':>5}"
            f"{r['tokens_per_sec']:<15.1f} "
            f"{r['peak_vram_gb']:<15.2f}"
        )

    if len(results) == 2:
        baseline = next(r for r in results if r["backend"] == "default")
        cb = next(r for r in results if r["backend"] == "continuous_batching")
        speedup = baseline["mean_gen_time_s"] / cb["mean_gen_time_s"]
        vram_delta = cb["peak_vram_gb"] - baseline["peak_vram_gb"]
        sign = "CB uses LESS VRAM" if vram_delta < 0 else "CB uses more VRAM"
        print(f"\nSpeedup (CB vs default): {speedup:.2f}x")
        print(f"VRAM delta (CB - default): {vram_delta:+.2f} GB  ({sign})")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark default generate() vs transformers continuous batching in GRPOTrainer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HuggingFace model name or path.")
    parser.add_argument(
        "--num-generations",
        type=int,
        default=None,
        help="Completions per prompt. Defaults to 64 with --large-batch, 8 otherwise.",
    )
    parser.add_argument(
        "--max-completion-length",
        type=int,
        default=None,
        help="Max completion tokens. Defaults to 2048 with --large-batch, 512 otherwise.",
    )
    parser.add_argument(
        "--backend",
        choices=["default", "continuous_batching", "both"],
        default="both",
        help="Which backend(s) to benchmark.",
    )
    parser.add_argument(
        "--attn-impl",
        default="kernels-community/flash-attn2",
        help="Attention implementation passed to model_init_kwargs.",
    )
    parser.add_argument(
        "--large-batch",
        action="store_true",
        help="GSM8K dataset with num_generations=64 and max_completion_length=2048. This is the "
        "scenario where CB has a structural advantage due to variable completion lengths and "
        "sequence removal. Training uses gradient accumulation (16 seqs × 4 steps).",
    )
    args = parser.parse_args()

    num_generations = args.num_generations or (LARGE_BATCH_NUM_GENERATIONS if args.large_batch else 8)
    max_completion_length = args.max_completion_length or (
        LARGE_BATCH_MAX_COMPLETION_LENGTH if args.large_batch else 512
    )
    backends = ["default", "continuous_batching"] if args.backend == "both" else [args.backend]

    results = []
    for backend in backends:
        result = run_benchmark(
            backend=backend,
            model_name=args.model,
            num_generations=num_generations,
            max_completion_length=max_completion_length,
            attn_impl=args.attn_impl,
            large_batch=args.large_batch,
        )
        results.append(result)
        print(
            f"  [{backend}] mean: {result['mean_gen_time_s']:.2f}s  "
            f"tok/s: {result['tokens_per_sec']:.1f}  "
            f"peak VRAM: {result['peak_vram_gb']:.2f} GB"
        )

    print_results(results)


if __name__ == "__main__":
    main()
