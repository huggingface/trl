#!/bin/bash
#SBATCH --job-name=miles-w2-collect
#SBATCH --partition=hopper-cpu
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/fsx/amine_dirhoussi/stable-bench-runs/miles/logs/%x-%j.out
cd /fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench
for j in 372222 372145; do
  R=/fsx/amine_dirhoussi/stable-bench-runs/miles/miles-w2-M-shared-$j
  uv run --no-project python collect/miles.py $R --status failed
  cp $R/watch.log $R/boxes.csv $R/versions.txt results/raw/miles-w2-M-shared-$j/ 2>/dev/null
  uv run --no-project python - <<PY
import json, collections
eps=[json.loads(l) for l in open("results/raw/miles-w2-M-shared-$j/episodes.jsonl")]
st=collections.Counter(e["status"] for e in eps); sc=[e for e in eps if e["reward"] is not None]
m=lambda k:[e[k] for e in sc if e.get(k) is not None]
avg=lambda x: round(sum(x)/len(x),2) if x else None
ph=collections.defaultdict(list)
for e in sc:
    for k,v in (e["phases"] or {}).items(): ph[k].append(v)
print("$j episodes", len(eps), dict(st), "reward_mean", avg(m("reward")), "wall", avg(m("wall_s")), "turns", avg(m("turns")),
      "model_s", avg(m("model_s")), "sandbox_s", avg(m("sandbox_s")), "seq_tokens", avg(m("seq_tokens")),
      "phases", {k: avg(v) for k,v in ph.items()}, "errors", collections.Counter((e["error"] or "")[:60] for e in eps if e["status"]=="failed").most_common(3))
PY
done
uv run --no-project python results/robustness/summary.py 2>&1 | tail -5
