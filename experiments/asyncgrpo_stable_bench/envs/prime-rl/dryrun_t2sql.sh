#!/bin/bash
# hopper-cpu: dry-run configs/prime-rl/t2sql-moe.toml and print the resolved knobs that matter.
#SBATCH --job-name=prl-dry-t2sql
#SBATCH --partition=hopper-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:40:00
#SBATCH --output=/fsx/amine_dirhoussi/stable-bench-runs/prime-rl/logs/%x-%j.out
set -uo pipefail
BENCH=/fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench
source /fsx/amine_dirhoussi/envs/bench-prime-rl/bin/activate
export HF_HUB_CACHE=/fsx/amine_dirhoussi/stable-bench-hub HF_HUB_OFFLINE=1
export PYTHONPATH=/fsx/amine_dirhoussi/stable-bench-src/text2sql/river-recipes/text-to-sql/src:$BENCH/configs/prime-rl
DRY=/fsx/amine_dirhoussi/stable-bench-runs/prime-rl/dryrun
cd /fsx/amine_dirhoussi/stable-bench-src/prime-rl
name=${NAME:-t2sql-moe}
rl @ "$BENCH/configs/prime-rl/$name.toml" --output-dir "$DRY" --run.name "$name" --clean --no-monitors.prime --dry-run \
    --inference.server.port 8123 --orchestrator.model.client.base-url http://localhost:8123/v1 \
    --rollout-transport.type zmq --rollout-transport.port 8126 2>&1 | tail -n 15
python - "$DRY/$name" <<'PY'
import json, sys, pathlib
d = pathlib.Path(sys.argv[1]) / "configs/latest/resolved"
print(sorted(p.relative_to(d).as_posix() for p in d.rglob("*.json")))
o = json.loads((d / "orchestrator.json").read_text()); t = json.loads((d / "trainer.json").read_text())
i = json.loads((d / "inference.json").read_text()) if (d / "inference.json").exists() else {}
print("max_steps", o.get("max_steps"), t.get("max_steps"), "seq_len", o["seq_len"], "batch", o["batch_size"], "constant", o["constant_trainer_batch_size"], "staleness", o["max_staleness"], "concurrency", o["concurrency"])
print("renderer", o["renderer"], "weight_broadcast", o["weight_broadcast"])
src = o["train"]["source"][0]
print("source", json.dumps({k: src[k] for k in ("name", "group_size", "sampling", "algo", "select") if k in src}))
print("env", json.dumps(src["env"]))
print("trainer seq_len", t["model"]["seq_len"], "ep", t["model"].get("ep"), "lora", t["model"].get("lora"), "loss", t["loss"], "optim", t["optim"], "sched", t["scheduler"])
print("vllm", json.dumps(i.get("vllm") or i))
PY
