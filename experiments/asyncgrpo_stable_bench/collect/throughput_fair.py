"""
results/raw/*-w1-S-{shared,recommended}-*/ -> results/tables/throughput_fair.csv (see results/tables/throughput_fair.md).

    uv run --no-project --python 3.12 --with pandas python collect/throughput_fair.py

Steady state = optimizer steps > 5: tokens of those steps / (wall clock of the last step - wall clock of the last step <= 5).
Per run, one common definition:
  all_trained     prompt + completion of every sample that went through a backward, padding excluded
  useful_trained  the same, restricted to samples whose group had nonzero reward spread (nonzero advantage)
  generated       completion tokens produced by the inference engine
  zero_adv_frac   share of generated groups whose 8 rewards are all equal
"""

import glob
import json
import os

import numpy as np
import pandas as pd


BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP = 5


def load(run_dir):
    rows = [json.loads(line) for line in open(os.path.join(run_dir, "steps.jsonl"))]
    meta = json.load(open(os.path.join(run_dir, "meta.json")))
    return rows, meta


def nat(r, k):
    v = (r.get("native") or {}).get(k)
    return float(v) if isinstance(v, (int, float)) else None


runs = {}
for d in sorted(glob.glob(os.path.join(BENCH, "results/raw/*-w1-S-*"))):
    run_id = os.path.basename(d)
    if "smoke" in run_id:
        continue
    rows, meta = load(d)
    runs[run_id] = (meta["lib"], "recommended" if "-recommended-" in run_id else "shared", rows, meta)

# ---- zero-advantage fits, for the libs that don't log it (verl, OpenRLHF fully; SkyRL's all-correct share) ----------
# Per-step points (step reward mean, share of all-equal groups) from the libs that log the share for the step's groups.
pts_zero, pts_all1 = [], []
for run_id, (lib, arm, rows, meta) in runs.items():
    for r in rows:
        if r["step"] <= SKIP or r.get("reward_mean") is None:
            continue
        if lib in ("prime-rl", "miles", "trl") and r.get("frac_zero_adv_groups") is not None:
            pts_zero.append((lib, r["reward_mean"], r["frac_zero_adv_groups"]))
        if lib == "prime-rl":
            pts_all1.append((lib, r["reward_mean"], nat(r, "orchestrator:train/agg/all/agent/solved_all")))
pz = pd.DataFrame(pts_zero, columns=["lib", "r", "f"])
pa = pd.DataFrame(pts_all1, columns=["lib", "r", "f"])
fit_zero = np.polyfit(pz.r, pz.f, 2)
fit_all1 = np.polyfit(pa.r, pa.f, 2)
loo = {}
for lib in pz.lib.unique():
    tr, te = pz[pz.lib != lib], pz[pz.lib == lib]
    c = np.polyfit(tr.r, tr.f, 2)
    loo[lib] = (float(np.polyval(c, te.r).mean()), float(te.f.mean()))
print("fit zero-adv vs reward:", fit_zero, "points", len(pz))
print("leave-one-lib-out (predicted mean, actual mean):", loo)
print("fit all-correct vs reward (prime-rl):", fit_all1)
clip = lambda x: float(min(max(x, 0.0), 1.0))  # noqa: E731

# Mean prompt tokens per sample (SkyRL logs none): prime-rl shared's mean over all arrived rollouts, same parquet order
# and same rendered prompt (thinking off).
prl = next(v for k, v in runs.items() if k.startswith("prime-rl-w1-S-shared"))
PROMPT_MEAN = float(np.mean([nat(r, "orchestrator:train/agg/all/num_input_tokens/mean") for r in prl[2]]))
print("prompt mean tokens (prime-rl shared):", PROMPT_MEAN)

out, bias = [], []
for run_id, (lib, arm, rows, meta) in runs.items():
    rows = sorted(rows, key=lambda r: r["step"])
    start = [r for r in rows if r["step"] <= SKIP][-1]
    ss = [r for r in rows if r["step"] > SKIP]
    wall = ss[-1]["time"] - start["time"]
    n_opt = ss[-1]["step"] - start["step"]
    all_t, useful_t, gen_t, zf, extra = [], [], [], [], {}
    pad_fwd, dummy = [], []
    for r in ss:
        if lib == "trl":
            a = nat(r, "batch/forwarded_tokens_per_step") * nat(r, "batch/packing_ratio")
            f = r["frac_zero_adv_groups"]
            u = a
        elif lib == "prime-rl":
            a = nat(r, "orchestrator:progress/input_tokens") + nat(r, "orchestrator:progress/output_tokens")
            dummy.append(r["trained_tokens"] - a)
            f = r["frac_zero_adv_groups"]
            u = a
            arrived = nat(r, "orchestrator:progress/tokens")
            if f is not None and f < 1:
                bias.append((a / arrived) / (1 - f))
            r = dict(
                r,
                gen_tokens=nat(r, "orchestrator:progress/rollouts")
                * nat(r, "orchestrator:train/agg/all/num_output_tokens/mean"),
            )
        elif lib == "skyrl":
            a = r["trained_tokens"] + r["num_rollouts"] * PROMPT_MEAN
            pad_fwd.append(nat(r, "generate/batch_padded_seq_len") * r["num_rollouts"])
            none = 1 - nat(r, "reward/avg_pass_at_8")
            f = clip(none + clip(np.polyval(fit_all1, r["reward_mean"])))
            u = a * (1 - f)
        elif lib in ("miles",):
            a = r["trained_tokens"]
            f = r["frac_zero_adv_groups"]
            u = a * (1 - f)
        else:  # verl, openrlhf
            a = r["trained_tokens"]
            f = clip(np.polyval(fit_zero, r["reward_mean"]))
            u = a * (1 - f)
        all_t.append(a)
        useful_t.append(u)
        gen_t.append(r["gen_tokens"])
        zf.append(f)
    lib_reported = sum(r["trained_tokens"] for r in ss) / wall
    row = {
        "run_id": run_id,
        "lib": lib,
        "arm": arm,
        "steps_used": n_opt,
        "step_time_s": wall / n_opt,
        "all_trained_tok_s": sum(all_t) / wall,
        "useful_trained_tok_s": sum(useful_t) / wall,
        "generated_tok_s": sum(gen_t) / wall,
        "zero_adv_frac": float(np.mean(zf)),
        "zero_adv_in_trained_batch": lib not in ("trl", "prime-rl"),
        "zero_adv_frac_source": {
            "trl": "logged (frac_reward_zero_std, groups scored in the step window)",
            "prime-rl": "logged (solved_none + solved_all over rollouts arrived in the window)",
            "miles": "logged (zero_std all_zero + all_one over the step's groups)",
            "skyrl": "partial: all-wrong = 1 - pass@8 (logged); all-correct estimated from reward mean (prime-rl fit)",
            "verl": "estimated from reward mean (pooled fit over prime-rl/miles/TRL steps)",
            "openrlhf": "estimated from reward mean (pooled fit over prime-rl/miles/TRL steps)",
        }[lib],
        "useful_is_exact": lib in ("trl", "prime-rl"),
        "lib_reported_trained_tok_s": lib_reported,
        "padded_forward_tok_s": sum(pad_fwd) / wall if pad_fwd else None,
        "dummy_microbatch_tok_s": sum(dummy) / wall if dummy else None,
    }
    out.append(row)

df = pd.DataFrame(out).sort_values(["arm", "useful_trained_tok_s"], ascending=[False, False])
os.makedirs(os.path.join(BENCH, "results/tables"), exist_ok=True)
df.to_csv(os.path.join(BENCH, "results/tables/throughput_fair.csv"), index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
print(df.drop(columns=["zero_adv_frac_source"]).round(3).to_string(index=False))
print(
    "prime-rl count-share bias, (shipped token share) / (1 - zero frac): mean %.3f, p10 %.3f, p90 %.3f"
    % (np.mean(bias), np.percentile(bias, 10), np.percentile(bias, 90))
)
