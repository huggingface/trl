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
#     "trackio",
# ]
# ///

"""
Benchmark `loss_type="nll"` against `loss_type="chunked_nll"` in [`SFTTrainer`].

Runs two SFT training loops from the same seed — one with the standard NLL loss and one with
the memory-efficient chunked NLL loss — then:

- Logs both runs to the same trackio space under different run names (so you can overlay the
  loss curves visually).
- Prints a side-by-side table of per-step losses and peak GPU memory.
- Fails with a non-zero exit code if the per-step losses diverge beyond `--tolerance` (sanity
  check that `chunked_nll` is numerically equivalent to `nll`).

Example:

```
python examples/scripts/benchmark_chunked_nll.py \
    --model Qwen/Qwen2.5-0.5B \
    --dataset trl-lib/Capybara \
    --max_steps 20 \
    --max_length 2048 \
    --batch_size 2
```
"""

import argparse
import gc
import time
import uuid

import torch
from datasets import load_dataset

from trl import SFTConfig, SFTTrainer


def _to_prompt_completion(example: dict) -> dict:
    messages = example["messages"]
    return {"prompt": [messages[0]], "completion": [messages[1]]}


def _run(args, loss_type: str) -> tuple[list[tuple[int, float]], float, float]:
    """Run one SFT training and return (per-step losses, peak GPU memory in GB, train time in s)."""
    dataset = load_dataset(args.dataset)
    # Keep only single-turn rows and split into a prompt-completion dataset so `completion_only_loss`
    # kicks in — this is the path `chunked_nll` is designed to accelerate (ignored prompt tokens are
    # skipped entirely before the lm_head matmul).
    dataset = dataset.filter(lambda ex: len(ex["messages"]) == 2)
    dataset = dataset.map(_to_prompt_completion, remove_columns=["messages"])

    training_args = SFTConfig(
        output_dir=f"{args.output_dir}/{args.benchmark_id}/{loss_type}",
        loss_type=loss_type,
        run_name=f"{args.benchmark_id}-{loss_type}",
        max_length=args.max_length,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        logging_steps=1,
        save_strategy="no",
        report_to="trackio",
        trackio_space_id=args.trackio_space_id,
        seed=args.seed,
        data_seed=args.seed,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        packing=args.packing,
        model_init_kwargs={"attn_implementation": args.attn_implementation} if args.attn_implementation else None,
    )

    trainer = SFTTrainer(
        model=args.model,
        args=training_args,
        train_dataset=dataset["train"],
    )

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    trainer.train()
    torch.cuda.synchronize()
    train_time = time.perf_counter() - start
    peak_mem = torch.cuda.max_memory_allocated() / 1e9

    losses = [
        (int(entry["step"]), float(entry["loss"]))
        for entry in trainer.state.log_history
        if "loss" in entry and "step" in entry
    ]

    # Free everything before the next run so memory stats are clean.
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    return losses, peak_mem, train_time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--dataset", default="trl-lib/Capybara")
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--packing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--attn_implementation",
        default=None,
        help="e.g. 'sdpa', 'eager', 'flash_attention_2'. If None, the model default is used.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="/tmp/chunked_nll_bench")
    parser.add_argument("--trackio_space_id", default="chunked-nll-benchmark")
    parser.add_argument(
        "--benchmark_id",
        default=None,
        help=(
            "Shared identifier for both runs in this invocation. Used as the trackio run-name "
            "prefix and the output subdirectory, so the two configs can be grouped together "
            "(and separated from earlier runs). Auto-generated if not provided."
        ),
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=5e-3,
        help="Max allowed |Δloss| between the two runs at any logged step (bf16 tolerance).",
    )
    args = parser.parse_args()

    suffix = uuid.uuid4().hex[:8]
    args.benchmark_id = f"{args.benchmark_id}-{suffix}" if args.benchmark_id else f"bench-{suffix}"
    print(f"benchmark_id: {args.benchmark_id}")

    print(f"=== run 1/2: loss_type='nll' ===")
    losses_nll, peak_nll, time_nll = _run(args, "nll")
    print(f"=== run 2/2: loss_type='chunked_nll' ===")
    losses_chunked, peak_chunked, time_chunked = _run(args, "chunked_nll")

    print()
    print(f"peak GPU memory — nll: {peak_nll:.2f} GB, chunked_nll: {peak_chunked:.2f} GB")
    if peak_nll > 0:
        print(f"peak memory change: {(peak_chunked / peak_nll - 1) * 100:+.1f}%")
    print(f"training time — nll: {time_nll:.2f} s, chunked_nll: {time_chunked:.2f} s")
    if time_nll > 0:
        print(f"training time change: {(time_chunked / time_nll - 1) * 100:+.1f}%")

    if len(losses_nll) != len(losses_chunked):
        raise SystemExit(
            f"mismatched number of logged steps: nll={len(losses_nll)} vs chunked_nll={len(losses_chunked)}"
        )

    print()
    print(f"{'step':>6} {'nll':>12} {'chunked_nll':>12} {'|Δ|':>10}")
    max_diff = 0.0
    for (step_n, loss_n), (step_c, loss_c) in zip(losses_nll, losses_chunked):
        if step_n != step_c:
            raise SystemExit(f"step mismatch at entry: {step_n} vs {step_c}")
        diff = abs(loss_n - loss_c)
        max_diff = max(max_diff, diff)
        print(f"{step_n:>6} {loss_n:>12.6f} {loss_c:>12.6f} {diff:>10.2e}")

    print()
    print(f"max |Δloss| = {max_diff:.3e}  (tolerance = {args.tolerance:.3e})")

    if max_diff > args.tolerance:
        raise SystemExit(f"FAIL: losses diverge beyond tolerance ({max_diff:.3e} > {args.tolerance:.3e})")
    print("OK — losses match within tolerance.")


if __name__ == "__main__":
    main()
