#!/bin/bash
# hopper-cpu: every 5 min, rewrite <run>-aux/steps.csv from the prime-rl file monitor, one row per step, for charting next
# to TRL's metrics.jsonl: step time split (fwd_bwd vs rollout wait), episode length (output tokens, turns), reward,
# rollouts per step and the zero-advantage fraction (rollout/no_signal / progress/rollouts; pruned and refilled).
# Requeue-safe (it only rewrites the CSV); exits 10 min after the run job ends.
#   sbatch -p hopper-cpu --qos=low -c 1 --mem=4G -t 2-0 -J prl-t2sql-steps \
#       -o /fsx/amine_dirhoussi/stable-bench-runs/prime-rl/logs/%x-%j.out envs/prime-rl/t2sql_steps.sh <run job id>
#SBATCH --requeue
J=$1
R=/fsx/amine_dirhoussi/stable-bench-runs/prime-rl/prime-rl-t2sql-Qwen3.5-35B-A3B-lora32${RUN_TAG:+-$RUN_TAG}-$J
while true; do
    grep -F '"producer"' $R/monitors/file/metrics.jsonl | python3 -c '
import csv, json, sys
t, o = {}, {}
for line in sys.stdin:
    try:
        r = json.loads(line)
    except json.JSONDecodeError:
        continue
    if r.get("step") is None:
        continue
    (t if r["producer"] == "trainer" else o).setdefault(r["step"], {}).update(r)
P = "train/t2sql/all/agent/"
cols = {"time": ("t", "time"), "step_s": ("t", "time/step"), "fwd_bwd_s": ("t", "time/forward_backward"),
        "rollout_wait_s": ("t", "time/wait_for_batch"), "broadcast_s": ("t", "time/broadcast_weights"),
        "peak_mem_gib": ("t", "perf/peak_memory"), "mismatch_kl": ("t", "mismatch_kl/all/mean"),
        "grad_norm": ("t", "optim/grad_norm"), "lr": ("t", "optim/lr"),
        "reward": ("o", P + "reward/mean"), "out_tokens_mean": ("o", P + "num_output_tokens/mean"),
        "out_tokens_p90": ("o", P + "num_output_tokens/p90"), "total_tokens_mean": ("o", P + "num_total_tokens/mean"),
        "turns_mean": ("o", P + "num_turns/mean"), "episode_s_mean": ("o", P + "timing/total/mean"),
        "model_s_mean": ("o", P + "timing/agent/model/mean"), "truncated": ("o", P + "is_truncated/mean"),
        "errored": ("o", "train/t2sql/all/has_error/mean"), "rollouts": ("o", "progress/rollouts"),
        "staleness_max": ("o", "staleness/rollout/max"),
        "no_signal": ("o", "rollout/no_signal"), "solved_all": ("o", P + "solved_all"), "solved_none": ("o", P + "solved_none")}
w = csv.writer(open(sys.argv[1], "w"))
w.writerow(["step", *cols, "zero_adv_frac"])
for s in sorted(k for k in t if k in o):
    n, z = o[s].get("progress/rollouts"), o[s].get("rollout/no_signal")
    w.writerow([s, *[{"t": t, "o": o}[src][s].get(key) for src, key in cols.values()], z / n if n and z is not None else None])
' $R-aux/steps.csv.tmp && mv $R-aux/steps.csv.tmp $R-aux/steps.csv
    if ! squeue -h -j $J | grep -q .; then [ -n "${ended:-}" ] && exit 0; ended=1; fi
    sleep 300
done
