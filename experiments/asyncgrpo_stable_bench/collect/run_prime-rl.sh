#!/bin/bash
# hopper-cpu wrapper: python collect/prime-rl.py <run_id> [args] with the bench-prime-rl venv, plus a short summary.
#SBATCH --job-name=prl-collect
#SBATCH --partition=hopper-cpu
#SBATCH --qos=low
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/fsx/amine_dirhoussi/stable-bench-runs/prime-rl/logs/%x-%j.out
set -uo pipefail
B=/fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench
PY=/fsx/amine_dirhoussi/envs/bench-prime-rl/bin/python
$PY $B/collect/prime-rl.py "$@"
$PY - "$B/results/raw/$1" <<'P'
import json, sys, statistics as st
d = sys.argv[1]
rows = [json.loads(l) for l in open(d + "/steps.jsonl")]
for r in rows: r.pop("native")
def m(k, rs):
    v = [r[k] for r in rs if r[k] is not None]
    return round(st.mean(v), 4) if v else None
keys = ["step_time", "train_compute_time", "wait_for_rollouts_time", "weight_sync_time", "num_rollouts", "gen_tokens",
        "trained_tokens", "reward_mean", "frac_zero_adv_groups", "completion_len_mean", "truncated_fraction",
        "staleness_mean", "staleness_max", "logprob_mismatch_kl", "grad_norm", "peak_mem_gb"]
print("steps", len(rows), "last", rows[-1]["step"] if rows else None)
for lo, hi in [(1, 10), (11, 50), (51, 100), (101, 150), (151, 200)]:
    rs = [r for r in rows if lo <= r["step"] <= hi]
    print(f"steps {lo}-{hi}:", {k: m(k, rs) for k in keys})
print("max staleness", max((r["staleness_max"] or 0) for r in rows), "max peak mem", max((r["peak_mem_gb"] or 0) for r in rows))
meta = json.load(open(d + "/meta.json")); print({k: meta[k] for k in ("status", "start_time", "end_time", "gpu_hours", "config_sha256")}); print(meta["notes"][:120])
P
