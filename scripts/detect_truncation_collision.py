#!/usr/bin/env python3
"""
Minimal reproduction: preference collision after DataCollatorForPreference truncation.

Loads a public preference dataset, tokenizes with a public tokenizer, then simulates
the exact truncation that `DataCollatorForPreference` applies. Detects samples whose
chosen/rejected preference would reverse after truncation.

Usage:
    python scripts/detect_truncation_collision.py [--dry-run]

Reproducibility:
    - Dataset:  Anthropic/hh-rlhf (revision pinned)
    - Tokenizer: gpt2 (revision pinned)
    - Collision IDs saved to results/truncation_collision_ids.json for audit
"""

import argparse
import json
import os
import sys

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

# ── Fixed revisions for reproducibility ──────────────────────────────────────
DATASET_NAME = "Anthropic/hh-rlhf"
DATASET_REVISION = "09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa"
TOKENIZER_NAME = "gpt2"
TOKENIZER_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
NUM_SAMPLES = 5000
SEED = 42
BETA = 0.1  # standard DPO beta, only used for classifying preference direction


def load_data(split: str = "train", max_samples: int = NUM_SAMPLES):
    """Load and shuffle the harmless subset of hh-rlhf."""
    ds = load_dataset(
        DATASET_NAME,
        revision=DATASET_REVISION,
        split=split,
        trust_remote_code=False,
    )
    ds = ds.shuffle(seed=SEED).select(range(min(max_samples, len(ds))))
    return ds


def tokenize_sample(tokenizer: AutoTokenizer, sample: dict) -> dict:
    """Tokenize a single (chosen, rejected) pair. Returns prompt + completion ids."""
    chosen_text = sample["chosen"]
    rejected_text = sample["rejected"]

    # Split prompt from response: hh-rlhf uses "\n\nAssistant: " as delimiter
    chosen_parts = chosen_text.split("\n\nAssistant: ", 1)
    rejected_parts = rejected_text.split("\n\nAssistant: ", 1)

    if len(chosen_parts) != 2 or len(rejected_parts) != 2:
        return None

    prompt, chosen_resp = chosen_parts
    _, rejected_resp = rejected_parts

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    chosen_ids = tokenizer.encode(chosen_resp, add_special_tokens=False)
    rejected_ids = tokenizer.encode(rejected_resp, add_special_tokens=False)

    return {
        "prompt_ids": prompt_ids,
        "chosen_ids": chosen_ids,
        "rejected_ids": rejected_ids,
        "chosen_text": chosen_text,
        "rejected_text": rejected_text,
    }


def collator_truncate(tokenized: dict, max_length: int, mode: str = "keep_start") -> dict:
    """Apply the exact same truncation as DataCollatorForPreference.torch_call."""
    prompt_chosen = tokenized["prompt_ids"] + tokenized["chosen_ids"]
    prompt_rejected = tokenized["prompt_ids"] + tokenized["rejected_ids"]
    chosen_mask = [0] * len(tokenized["prompt_ids"]) + [1] * len(tokenized["chosen_ids"])
    rejected_mask = [0] * len(tokenized["prompt_ids"]) + [1] * len(tokenized["rejected_ids"])

    if mode == "keep_start":
        sl = slice(None, max_length)
    elif mode == "keep_end":
        sl = slice(-max_length, None)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return {
        "prompt_chosen_ids": prompt_chosen[sl],
        "prompt_rejected_ids": prompt_rejected[sl],
        "chosen_mask": chosen_mask[sl],
        "rejected_mask": rejected_mask[sl],
        "chosen_len": len(tokenized["chosen_ids"]),
        "rejected_len": len(tokenized["rejected_ids"]),
        "truncated_chosen_comp_len": sum(chosen_mask[sl]),
        "truncated_rejected_comp_len": sum(rejected_mask[sl]),
    }


def compute_preference(
    model, tokenizer, tokenized: dict, device: str = "cpu"
) -> dict:
    """Compute pre-truncation log-prob difference using the model.

    Clips to model max_position_embeddings to avoid OOB errors.
    """
    model.eval()
    max_ctx = model.config.max_position_embeddings
    with torch.no_grad():
        pc = torch.tensor([tokenized["prompt_ids"] + tokenized["chosen_ids"]], device=device)
        pr = torch.tensor([tokenized["prompt_ids"] + tokenized["rejected_ids"]], device=device)
        if pc.size(1) > max_ctx:
            pc = pc[:, :max_ctx]
        if pr.size(1) > max_ctx:
            pr = pr[:, :max_ctx]

        cm = torch.tensor(
            [[0] * len(tokenized["prompt_ids"]) + [1] * len(tokenized["chosen_ids"])],
            device=device,
        )
        rm = torch.tensor(
            [[0] * len(tokenized["prompt_ids"]) + [1] * len(tokenized["rejected_ids"])],
            device=device,
        )
        if cm.size(1) > max_ctx:
            cm = cm[:, :max_ctx]
        if rm.size(1) > max_ctx:
            rm = rm[:, :max_ctx]

        chosen_logp = _sequence_logprob(model, pc, cm)
        rejected_logp = _sequence_logprob(model, pr, rm)

    return {
        "chosen_logp": float(chosen_logp),
        "rejected_logp": float(rejected_logp),
        "delta": float(chosen_logp - rejected_logp),
    }


def compute_preference_truncated(
    model, tokenized: dict, truncated: dict, device: str = "cpu"
) -> dict:
    """Compute post-truncation log-prob difference using the model."""
    model.eval()
    with torch.no_grad():
        pc = torch.tensor([truncated["prompt_chosen_ids"]], device=device)
        pr = torch.tensor([truncated["prompt_rejected_ids"]], device=device)
        cm = torch.tensor([truncated["chosen_mask"]], device=device)
        rm = torch.tensor([truncated["rejected_mask"]], device=device)

        chosen_logp = _sequence_logprob(model, pc, cm)
        rejected_logp = _sequence_logprob(model, pr, rm)

    return {
        "chosen_logp": float(chosen_logp),
        "rejected_logp": float(rejected_logp),
        "delta": float(chosen_logp - rejected_logp),
    }


def _sequence_logprob(model, input_ids, completion_mask) -> float:
    """Sum of per-token log-probs over completion tokens only."""
    # gpt2 max_position_embeddings = 1024; truncate if needed
    max_ctx = model.config.max_position_embeddings
    if input_ids.size(1) > max_ctx:
        input_ids = input_ids[:, :max_ctx]
        completion_mask = completion_mask[:, :max_ctx]
    logits = model(input_ids).logits  # (1, seq_len, vocab)
    shift_logits = logits[:, :-1, :]  # predict next token
    shift_labels = input_ids[:, 1:]
    shift_mask = completion_mask[:, 1:]
    per_token_logps = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    token_logps = per_token_logps.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
    masked_logps = token_logps * shift_mask
    return masked_logps.sum().item()


def surface_collision_check(tokenized: dict, truncated: dict) -> dict:
    """Fast, model-free collision indicators based on token statistics."""
    chosen_full = tokenized["chosen_ids"]
    rejected_full = tokenized["rejected_ids"]
    chosen_trunc = truncated["chosen_mask"]
    rejected_trunc = truncated["rejected_mask"]

    # Did chosen/rejected completion length ratio flip?
    full_ratio = len(chosen_full) / max(len(rejected_full), 1)
    trunc_chosen_len = sum(chosen_trunc)
    trunc_rejected_len = sum(rejected_trunc)
    trunc_ratio = trunc_chosen_len / max(trunc_rejected_len, 1)

    # Did prompt get completely consumed by truncation?
    prompt_len = len(tokenized["prompt_ids"])
    chosen_prompt_remain = sum(1 for m in chosen_trunc[:prompt_len] if m == 0)
    rejected_prompt_remain = sum(1 for m in rejected_trunc[:prompt_len] if m == 0)

    return {
        "full_ratio": round(full_ratio, 3),
        "trunc_ratio": round(trunc_ratio, 3),
        "ratio_flipped": (full_ratio > 1 and trunc_ratio < 1)
        or (full_ratio < 1 and trunc_ratio > 1),
        "chosen_prompt_kept": chosen_prompt_remain,
        "rejected_prompt_kept": rejected_prompt_remain,
        "chosen_fully_dropped": trunc_chosen_len == 0,
        "rejected_fully_dropped": trunc_rejected_len == 0,
    }


def detect_original_duplicates(tokenized_batch: list[dict]) -> list[int]:
    """Count pairs where chosen_ids == rejected_ids BEFORE truncation."""
    dup_indices = []
    for i, t in enumerate(tokenized_batch):
        if t is None:
            continue
        if t["chosen_ids"] == t["rejected_ids"]:
            dup_indices.append(i)
    return dup_indices


def detect_truncated_duplicates(
    tokenized_batch: list[dict], max_length: int, mode: str
) -> list[int]:
    """Count pairs where chosen_ids == rejected_ids AFTER truncation."""
    dup_indices = []
    for i, t in enumerate(tokenized_batch):
        if t is None:
            continue
        tr = collator_truncate(t, max_length, mode)
        if tr["prompt_chosen_ids"] == tr["prompt_rejected_ids"]:
            dup_indices.append(i)
    return dup_indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run surface-level scan only, skip model verification.",
    )
    parser.add_argument(
        "--max-lengths",
        type=int,
        nargs="+",
        default=[256, 512, 1024],
        help="max_length values to test.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["keep_start", "keep_end"],
        help="Truncation modes to test.",
    )
    parser.add_argument(
        "--num-model-samples",
        type=int,
        default=200,
        help="Number of samples to verify with model log-probs.",
    )
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 72)
    print("Minimal Reproduction: DPO Truncation Preference Collision")
    print("=" * 72)
    print(f"  Dataset:    {DATASET_NAME} (revision: {DATASET_REVISION[:8]}...)")
    print(f"  Tokenizer:  {TOKENIZER_NAME} (revision: {TOKENIZER_REVISION[:8]}...)")
    print(f"  Samples:    {NUM_SAMPLES}")
    print(f"  Seed:       {SEED}")
    print("-" * 72)

    # ── Step 1: Load & tokenize ──────────────────────────────────────────
    print("\n[1/4] Loading & tokenizing dataset...")
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_NAME, revision=TOKENIZER_REVISION
    )
    tokenizer.pad_token = tokenizer.eos_token

    ds = load_data(max_samples=NUM_SAMPLES)
    tokenized_batch = []
    for sample in ds:
        tokenized_batch.append(tokenize_sample(tokenizer, sample))
    valid = [t for t in tokenized_batch if t is not None]
    print(f"  Tokenized: {len(tokenized_batch)} samples ({len(valid)} valid)")

    # ── Step 2: Surface-level collision scan ──────────────────────────────
    print("\n[2/4] Surface collision scan (token-level, no model)...")

    collision_ids = {}  # mode -> max_length -> list of sample indices
    for mode in args.modes:
        collision_ids[mode] = {}
        for ml in args.max_lengths:
            collisions = []
            dup_full = detect_original_duplicates(tokenized_batch)
            dup_trunc = detect_truncated_duplicates(tokenized_batch, ml, mode)

            for i, t in enumerate(tokenized_batch):
                if t is None:
                    continue
                tr = collator_truncate(t, ml, mode)
                surface = surface_collision_check(t, tr)
                if surface["chosen_fully_dropped"] or surface["rejected_fully_dropped"]:
                    collisions.append(
                        {
                            "index": i,
                            "reason": "completion_dropped"
                            if surface["chosen_fully_dropped"]
                            or surface["rejected_fully_dropped"]
                            else "ratio_flipped",
                            "surface": surface,
                        }
                    )
                elif surface["ratio_flipped"]:
                    collisions.append(
                        {"index": i, "reason": "ratio_flipped", "surface": surface}
                    )

            collision_ids[mode][str(ml)] = collisions
            print(
                f"  {mode:>10s} @ max_length={ml:>4d}: "
                f"{len(collisions)} collisions / {NUM_SAMPLES} "
                f"({100*len(collisions)/NUM_SAMPLES:.2f}%)"
            )
            print(
                f"  {'':>10s} {'':>10s}  "
                f"orig_dup={len(dup_full):>3d}  "
                f"candidate_dup={len(dup_trunc):>3d} "
                f"(retained candidates after truncation)"
            )

    # ── Step 3: Model-based verification ──────────────────────────────────
    if not args.dry_run:
        print(
            f"\n[3/4] Model verification on top {args.num_model_samples} samples..."
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Using device: {device}")

        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            TOKENIZER_NAME, revision=TOKENIZER_REVISION
        ).to(device)
        model.eval()

        # Pick most interesting collision candidates across modes
        verify_indices = set()
        for mode_collisions in collision_ids.values():
            for collisions in mode_collisions.values():
                for c in collisions[: args.num_model_samples // len(args.modes)]:
                    verify_indices.add(c["index"])
        # Also sample non-collision controls
        all_indices = set(i for i, t in enumerate(tokenized_batch) if t is not None)
        non_collision = list(all_indices - verify_indices)
        import random

        random.seed(SEED)
        controls = random.sample(non_collision, min(50, len(non_collision)))
        verify_indices |= set(controls)
        verify_indices = sorted(verify_indices)[: args.num_model_samples]

        model_results = []
        # Verify on surface-flagged collision candidates AND a control sample
        candidate_indices = sorted(verify_indices)
        for mode in args.modes:
            for ml_str, collisions in collision_ids[mode].items():
                ml = int(ml_str)
                # Process surface-flagged candidates
                for c in collisions:
                    if c["index"] in verify_indices:
                        idx = c["index"]
                        t = tokenized_batch[idx]
                        tr = collator_truncate(t, ml, mode)

                        pref_full = compute_preference(model, tokenizer, t, device)
                        pref_trunc = compute_preference_truncated(model, t, tr, device)

                        is_collision = (
                            pref_full["delta"] > 0 and pref_trunc["delta"] < 0
                        ) or (pref_full["delta"] < 0 and pref_trunc["delta"] > 0)

                        model_results.append({
                            "index": idx, "mode": mode, "max_length": ml,
                            "full": pref_full, "truncated": pref_trunc,
                            "is_collision": is_collision,
                        })

                # Also verify random non-flagged samples for unbiased estimate
                remaining = [i for i in candidate_indices
                            if i not in {c["index"] for c in collisions}]
                import random as _random
                _random.seed(SEED)
                extras = _random.sample(remaining, min(20, len(remaining)))
                for idx in extras:
                    t = tokenized_batch[idx]
                    tr = collator_truncate(t, ml, mode)
                    pref_full = compute_preference(model, tokenizer, t, device)
                    pref_trunc = compute_preference_truncated(model, t, tr, device)
                    is_collision = (
                        pref_full["delta"] > 0 and pref_trunc["delta"] < 0
                    ) or (pref_full["delta"] < 0 and pref_trunc["delta"] > 0)
                    model_results.append({
                        "index": idx, "mode": mode, "max_length": ml,
                        "full": pref_full, "truncated": pref_trunc,
                        "is_collision": is_collision,
                    })

        # ── Step 4: Report ─────────────────────────────────────────────────
        print("\n[4/4] Model verification results:")
        model_collisions = [r for r in model_results if r["is_collision"]]
        print(f"  Model-verified collisions: {len(model_collisions)}")
        total_checked = len(model_results)
        if total_checked > 0:
            print(f"  Collision rate (model): {len(model_collisions)}/{total_checked}")
        for r in model_collisions[:10]:
            print(
                f"    idx={r['index']:>4d}  {r['mode']:>10s}  "
                f"max_length={r['max_length']:>4d}  "
                f"full_delta={r['full']['delta']:.3f}  "
                f"trunc_delta={r['truncated']['delta']:.3f}"
            )

        # Save results
        out_path = os.path.join(OUT_DIR, "truncation_collision_ids.json")
        with open(out_path, "w") as f:
            json.dump(
                {
                    "metadata": {
                        "dataset": DATASET_NAME,
                        "dataset_revision": DATASET_REVISION,
                        "tokenizer": TOKENIZER_NAME,
                        "tokenizer_revision": TOKENIZER_REVISION,
                        "num_samples": NUM_SAMPLES,
                        "seed": SEED,
                    },
                    "collision_ids": {
                        m: {str(ml): [c["index"] for c in colls]}
                        for m, mode_data in collision_ids.items()
                        for ml, colls in mode_data.items()
                    },
                    "model_verified": [
                        {
                            k: v
                            for k, v in r.items()
                            if k not in ("full", "truncated")
                        }
                        | {"full": r["full"], "truncated": r["truncated"]}
                        for r in model_collisions
                    ],
                },
                f,
                indent=2,
            )
        print(f"\n  Collision IDs saved to: {out_path}")
    else:
        # Save surface results without model
        out_path = os.path.join(OUT_DIR, "truncation_collision_ids.json")
        with open(out_path, "w") as f:
            json.dump(
                {
                    "metadata": {
                        "dataset": DATASET_NAME,
                        "dataset_revision": DATASET_REVISION,
                        "tokenizer": TOKENIZER_NAME,
                        "tokenizer_revision": TOKENIZER_REVISION,
                        "num_samples": NUM_SAMPLES,
                        "seed": SEED,
                    },
                    "collision_ids": {
                        m: {
                            str(ml): [c["index"] for c in colls]
                            for ml, colls in mode_data.items()
                        }
                        for m, mode_data in collision_ids.items()
                    },
                },
                f,
                indent=2,
            )
        print(f"\n  Surface collision IDs saved to: {out_path}")

    print("\n" + "=" * 72)
    print("Done.")
    print("=" * 72)


if __name__ == "__main__":
    main()