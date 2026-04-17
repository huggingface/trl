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
# ]
# ///

"""
Sweep the `chunked_nll` chunk size in [`SFTTrainer`] to find the sweet spot between memory
savings and throughput.

Runs a short SFT training for each chunk size and records peak GPU memory plus wall-clock
training time. Prints a table at the end so you can eyeball the knee of the memory/time curve.

Example:

```
python examples/scripts/benchmark_chunked_nll_chunk_size.py \
    --model Qwen/Qwen3-1.7B \
    --max_steps 20 \
    --chunk_sizes 128,256,512,1024,2048,4096,8192
```
"""

import argparse
import gc
import time

import torch
from datasets import load_dataset

from trl import SFTConfig, SFTTrainer
from trl.trainer import sft_trainer


def _to_prompt_completion(example: dict) -> dict:
    messages = example["messages"]
    return {"prompt": [messages[0]], "completion": [messages[1]]}


def _run(args, chunk_size: int) -> tuple[float, float]:
    """Run one SFT training at `chunk_size` and return (peak GPU memory GB, train time s)."""
    dataset = load_dataset(args.dataset)
    # Keep single-turn rows and split into prompt/completion so `completion_only_loss` kicks in —
    # this is the path `chunked_nll` is designed to accelerate.
    dataset = dataset.filter(lambda ex: len(ex["messages"]) == 2)
    dataset = dataset.map(_to_prompt_completion, remove_columns=["messages"])

    training_args = SFTConfig(
        output_dir=f"{args.output_dir}/chunk_{chunk_size}",
        loss_type="chunked_nll",
        max_length=args.max_length,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        packing=args.packing,
        model_init_kwargs={"attn_implementation": args.attn_implementation} if args.attn_implementation else None,
    )

    # Override the module-level constant so the patching inside SFTTrainer picks up our chunk size.
    original = sft_trainer._CHUNKED_LM_HEAD_CHUNK_SIZE
    sft_trainer._CHUNKED_LM_HEAD_CHUNK_SIZE = chunk_size
    try:
        trainer = SFTTrainer(model=args.model, args=training_args, train_dataset=dataset["train"])
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        trainer.train()
        torch.cuda.synchronize()
        train_time = time.perf_counter() - start
        peak_mem = torch.cuda.max_memory_allocated() / 1e9
    finally:
        sft_trainer._CHUNKED_LM_HEAD_CHUNK_SIZE = original

    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    return peak_mem, train_time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--dataset", default="trl-lib/Capybara")
    parser.add_argument("--max_steps", type=int, default=20)
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
    parser.add_argument("--output_dir", default="/tmp/chunked_nll_chunk_size")
    parser.add_argument(
        "--chunk_sizes",
        default="128,256,512,1024,2048,4096,8192",
        help="Comma-separated chunk sizes to sweep.",
    )
    args = parser.parse_args()

    chunk_sizes = [int(s) for s in args.chunk_sizes.split(",")]

    results: list[tuple[int, float | None, float | None]] = []
    for i, cs in enumerate(chunk_sizes, 1):
        print(f"=== run {i}/{len(chunk_sizes)}: chunk_size={cs} ===")
        try:
            peak_mem, train_time = _run(args, cs)
        except torch.cuda.OutOfMemoryError:
            print(f"  OOM at chunk_size={cs}")
            results.append((cs, None, None))
            gc.collect()
            torch.cuda.empty_cache()
            continue
        results.append((cs, peak_mem, train_time))
        print(f"  peak memory: {peak_mem:.2f} GB, time: {train_time:.2f} s")

    print()
    print(f"{'chunk_size':>12} {'peak_mem_gb':>14} {'time_s':>10}")
    for cs, peak, t in results:
        if peak is None:
            print(f"{cs:>12} {'OOM':>14} {'—':>10}")
        else:
            print(f"{cs:>12} {peak:>14.2f} {t:>10.2f}")

    ok = [(cs, p, t) for cs, p, t in results if p is not None]
    if ok:
        best_mem = min(ok, key=lambda r: r[1])
        best_time = min(ok, key=lambda r: r[2])
        print()
        print(f"min peak memory: chunk_size={best_mem[0]} → {best_mem[1]:.2f} GB, {best_mem[2]:.2f} s")
        print(f"min train time:  chunk_size={best_time[0]} → {best_time[1]:.2f} GB, {best_time[2]:.2f} s")


if __name__ == "__main__":
    main()
