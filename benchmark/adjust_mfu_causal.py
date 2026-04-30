# ruff: noqa: T201
"""
Compute causal-corrected MFU from reported (uncausal) MFU.

Usage:
    # One-off:
    python adjust_mfu_causal.py --model Qwen/Qwen3-30B-A3B --ctx 65536 --mfu 66.46

    # Per-context table for a model:
    python adjust_mfu_causal.py --model Qwen/Qwen3-30B-A3B --table

    # Apply to a CSV column (in-place, adds 'mfu_adjusted'):
    python adjust_mfu_causal.py --csv benchmark/results.csv --model-col model --ctx-col context_length --mfu-col mfu
"""

import argparse


def adj_factor(cfg, seq_len: int) -> float:
    """Adjustment factor: reported MFU × adj_factor = causal-corrected MFU.

    The (unmodified) `compute_flops_per_token` formula uses the non-causal convention,
    which double-counts attention scores. Causal masking halves the attention-score
    contribution (`Q·K^T` + `attn·V`), so we subtract `L * 3 * 2 * n_heads * head_dim * seq_len`
    from the full FLOPs and take the ratio.
    """
    from trl.trainer.utils import compute_flops_per_token

    f_full = compute_flops_per_token(cfg, seq_len)
    n_heads = cfg.num_attention_heads
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // n_heads)
    L = cfg.num_hidden_layers
    half_attn_per_layer = 3 * 2 * n_heads * head_dim * seq_len  # 3x for fwd+bwd
    half_attn_total = L * half_attn_per_layer
    f_causal = f_full - half_attn_total
    return f_causal / f_full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    ap.add_argument("--ctx", type=int, default=None)
    ap.add_argument("--mfu", type=float, default=None, help="reported MFU (%)")
    ap.add_argument("--table", action="store_true", help="print full per-ctx adj table")
    ap.add_argument("--csv", default=None, help="CSV path with `ctx` and `mfu` columns")
    ap.add_argument("--ctx-col", default="context_length")
    ap.add_argument("--mfu-col", default="mfu")
    args = ap.parse_args()

    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)

    if args.csv:
        import csv

        rows = []
        with open(args.csv) as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames + ["mfu_adjusted"]
            for r in reader:
                ctx = int(r[args.ctx_col])
                mfu = float(r[args.mfu_col])
                a = adj_factor(cfg, ctx)
                r["mfu_adjusted"] = f"{mfu * a:.2f}"
                rows.append(r)
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"Updated {args.csv} with `mfu_adjusted` column.")
        return

    if args.table:
        print(f"# {args.model}: causal MFU adjustment factors")
        print(f"{'ctx':>9s}  {'GFLOPs/tok':>11s}  {'adj GFLOPs':>11s}  {'adj_factor':>10s}")
        from trl.trainer.utils import compute_flops_per_token

        for ctx in [16384, 32768, 65536, 131072, 262144, 524288, 1048576]:
            full = compute_flops_per_token(cfg, ctx) / 1e9
            a = adj_factor(cfg, ctx)
            print(f"{ctx:>9d}  {full:>10.2f}   {full * a:>10.2f}   {a:>10.4f}")
        return

    if args.ctx is not None:
        a = adj_factor(cfg, args.ctx)
        print(f"adj_factor({args.model}, ctx={args.ctx}) = {a:.4f}")
        if args.mfu is not None:
            print(f"reported MFU = {args.mfu:.2f}%  →  adjusted MFU = {args.mfu * a:.2f}%")
        return

    ap.print_help()


if __name__ == "__main__":
    main()
