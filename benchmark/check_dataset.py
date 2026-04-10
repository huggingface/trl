# ruff: noqa: T201
"""Check token length distribution of a dataset for benchmarking."""

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


ds = load_dataset("trl-lib/Capybara", split="train")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")

print(f"Dataset size: {len(ds)}")
sample = ds[0]
print(f"Keys: {list(sample.keys())}")
turns = sample.get("num_turns", len(sample["messages"]))
print(f"Sample ({turns} turns):")
for m in sample["messages"]:
    print(f"  {m['role']}: {m['content'][:100]}...")

lengths = []
for i in range(min(2000, len(ds))):
    text = tok.apply_chat_template(ds[i]["messages"], tokenize=False)
    toks = tok(text)["input_ids"]
    lengths.append(len(toks))

lengths = np.array(lengths)
print(f"\nToken length stats (first {len(lengths)} samples):")
print(f"  min:    {lengths.min()}")
print(f"  median: {np.median(lengths):.0f}")
print(f"  mean:   {lengths.mean():.0f}")
print(f"  p90:    {np.percentile(lengths, 90):.0f}")
print(f"  p99:    {np.percentile(lengths, 99):.0f}")
print(f"  max:    {lengths.max()}")
print(f"  >8k:    {(lengths > 8192).sum()}")
print(f"  >16k:   {(lengths > 16384).sum()}")
print(f"  >32k:   {(lengths > 32768).sum()}")
