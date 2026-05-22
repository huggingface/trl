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
#     "datasets",
# ]
# ///

"""
Diagnose a trained reward model on a preference dataset.

Reports chosen/rejected accuracy, reward statistics, margin distribution,
length-bias correlation, and dumps failure cases to CSV. Intended as a
sanity check before using the reward model in PPO/RLOO/etc.

Example (published TRL reward model on TL;DR preferences):

python examples/scripts/ppo/evaluate_reward_model.py \
    --reward_model_path trl-lib/pythia-1b-deduped-tldr-rm \
    --dataset_name trl-lib/tldr-preference \
    --split validation \
    --num_samples 1000 \
    --output_dir reward_model_eval

Outputs `summary.json` (aggregate stats) and `failures.csv` (mispredictions
plus near-zero-margin cases) inside --output_dir.
"""

import argparse
import csv
import json
import os
from statistics import mean

import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reward_model_path", required=True, help="Path or Hub ID of the reward model.")
    parser.add_argument("--dataset_name", default="trl-lib/tldr-preference", help="Preference dataset name.")
    parser.add_argument("--split", default="validation", help="Dataset split to evaluate on.")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of examples to evaluate.")
    parser.add_argument("--max_length", type=int, default=1024, help="Max tokens per (prompt + completion).")
    parser.add_argument(
        "--margin_threshold",
        type=float,
        default=0.5,
        help="Below this absolute margin a pair is flagged as 'near-zero'.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Inference device. 'auto' picks cuda > mps > cpu.",
    )
    parser.add_argument("--output_dir", default="reward_model_eval", help="Directory for summary.json + failures.csv.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    return parser.parse_args()


def pick_device(choice: str) -> str:
    if choice != "auto":
        return choice
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def score_pair(model, tokenizer, prompt: str, completion: str, max_length: int, device: str) -> float:
    # Mirror RewardTrainer's `add_eos` (trl/trainer/reward_trainer.py): training sequences always end in
    # `eos_token`, so the causal sequence-classification head learns from EOS hidden states. Inference must
    # match, otherwise scores read a different last-token representation than the head was trained on.
    eos = tokenizer.eos_token or ""
    text = prompt + completion
    if eos and not text.endswith(eos):
        text = text + eos
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    with torch.no_grad():
        logits = model(**encoded).logits
    # AutoModelForSequenceClassification with num_labels=1 returns shape (batch, 1).
    return logits.squeeze(-1).item()


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mx, my = mean(xs), mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=False))
    denom_x = sum((x - mx) ** 2 for x in xs) ** 0.5
    denom_y = sum((y - my) ** 2 for y in ys) ** 0.5
    if denom_x == 0 or denom_y == 0:
        return 0.0
    return num / (denom_x * denom_y)


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = pick_device(args.device)
    print(f"Using device: {device}")

    print(f"Loading tokenizer + reward model from: {args.reward_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.reward_model_path)
    # Pythia and many GPT-NeoX-style tokenizers ship without pad_token; fall through to eos_token, mirroring
    # what SFTTrainer / RewardTrainer do internally (see trl/trainer/sft_trainer.py:1157 and
    # trl/trainer/reward_trainer.py:471). Inference here doesn't strictly need padding, but we set it so
    # the reader can compare with the training setup.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(args.reward_model_path, num_labels=1)
    model.to(device)
    model.eval()

    print(f"Loading dataset: {args.dataset_name} [split={args.split}]")
    ds = load_dataset(args.dataset_name, split=args.split)
    if args.num_samples and args.num_samples < len(ds):
        ds = ds.shuffle(seed=args.seed).select(range(args.num_samples))
    print(f"Evaluating {len(ds)} pairs")

    chosen_rewards: list[float] = []
    rejected_rewards: list[float] = []
    chosen_lengths: list[int] = []
    rejected_lengths: list[int] = []
    margins: list[float] = []
    failures: list[dict] = []

    for i, row in enumerate(ds):
        prompt = row["prompt"]
        chosen = row["chosen"]
        rejected = row["rejected"]
        r_c = score_pair(model, tokenizer, prompt, chosen, args.max_length, device)
        r_r = score_pair(model, tokenizer, prompt, rejected, args.max_length, device)
        margin = r_c - r_r
        chosen_rewards.append(r_c)
        rejected_rewards.append(r_r)
        chosen_lengths.append(len(tokenizer.encode(chosen)))
        rejected_lengths.append(len(tokenizer.encode(rejected)))
        margins.append(margin)
        flagged = margin <= 0 or abs(margin) < args.margin_threshold
        if flagged:
            failures.append(
                {
                    "index": i,
                    "reward_chosen": r_c,
                    "reward_rejected": r_r,
                    "margin": margin,
                    "len_chosen": chosen_lengths[-1],
                    "len_rejected": rejected_lengths[-1],
                    "category": "mispreferred" if margin <= 0 else "near_zero",
                    "prompt": prompt[:200],
                    "chosen": chosen[:200],
                    "rejected": rejected[:200],
                }
            )
        if (i + 1) % 50 == 0:
            running_acc = sum(1 for m in margins if m > 0) / len(margins)
            print(f"  [{i + 1}/{len(ds)}] running accuracy={running_acc:.3f}, mean margin={mean(margins):.3f}")

    n = len(margins)
    acc = sum(1 for m in margins if m > 0) / n
    # Note: `near_zero_frac` and `mispreferred_frac` are not disjoint. `near_zero_frac` is `|margin| < threshold`
    # (i.e., margins in (-threshold, threshold)), while `mispreferred_frac` is `margin <= 0`. They overlap on
    # (-threshold, 0]; do not add them when reporting.
    near_zero_frac = sum(1 for m in margins if abs(m) < args.margin_threshold) / n
    mispreferred_frac = sum(1 for m in margins if m <= 0) / n

    length_diff = [c - r for c, r in zip(chosen_lengths, rejected_lengths, strict=False)]
    length_bias = pearson(length_diff, margins)

    summary = {
        "reward_model_path": args.reward_model_path,
        "dataset_name": args.dataset_name,
        "split": args.split,
        "n_pairs": n,
        "accuracy": acc,
        "mean_reward_chosen": mean(chosen_rewards),
        "mean_reward_rejected": mean(rejected_rewards),
        "mean_margin": mean(margins),
        "median_margin": sorted(margins)[n // 2],
        "min_margin": min(margins),
        "max_margin": max(margins),
        "mispreferred_fraction": mispreferred_frac,
        "near_zero_fraction": near_zero_frac,
        "margin_threshold": args.margin_threshold,
        "length_bias_pearson": length_bias,
        "mean_chosen_length_tokens": mean(chosen_lengths),
        "mean_rejected_length_tokens": mean(rejected_lengths),
    }

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote summary: {summary_path}")

    failures_path = os.path.join(args.output_dir, "failures.csv")
    fieldnames = [
        "index",
        "category",
        "margin",
        "reward_chosen",
        "reward_rejected",
        "len_chosen",
        "len_rejected",
        "prompt",
        "chosen",
        "rejected",
    ]
    with open(failures_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in failures:
            writer.writerow(row)
    print(f"Wrote failures ({len(failures)} rows): {failures_path}")

    print()
    print("=== Reward model diagnostic summary ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
