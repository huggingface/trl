#!/bin/bash
# hopper-cpu: dry-run the W2 config (plus the smoke overrides) and print the resolved knobs that matter, and the env
# worker pool config rl writes for the W2 source.
#SBATCH --job-name=prl-dry-w2
#SBATCH --partition=hopper-cpu
#SBATCH --qos=low
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:40:00
#SBATCH --output=/fsx/amine_dirhoussi/stable-bench-runs/prime-rl/logs/%x-%j.out
set -uo pipefail
BENCH=/fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench
source /fsx/amine_dirhoussi/envs/bench-prime-rl/bin/activate
export HF_HUB_CACHE=/fsx/amine_dirhoussi/stable-bench-hub
DRY=/fsx/amine_dirhoussi/stable-bench-runs/prime-rl/dryrun
mkdir -p ~/.cache/harbor
ln -sfn /fsx/amine_dirhoussi/stable-bench-src/termgrade/harbor-local/train/tasks ~/.cache/harbor/termgrade-train
cd /fsx/amine_dirhoussi/stable-bench-src/prime-rl
for name in w2-M w2-M-smoke; do
    extra=(); [ $name = w2-M-smoke ] && extra=(--max-steps 3 --orchestrator.concurrency.initial-inflight 16 --orchestrator.concurrency.max-inflight 16)
    echo "=== dry-run $name"
    rl @ "$BENCH/configs/prime-rl/w2-M.toml" --output-dir "$DRY" --run.name "$name" --clean --no-monitors.prime --dry-run \
        --inference.server.port 8123 --orchestrator.model.client.base-url http://localhost:8123/v1 \
        --rollout-transport.type zmq --rollout-transport.port 8126 --weight-broadcast.type nccl --weight-broadcast.port 8128 "${extra[@]}" 2>&1 | tail -n 6
    python - "$DRY/$name" <<'EOF'
import json, sys, pathlib
d = pathlib.Path(sys.argv[1]) / "configs/latest/resolved"
print(sorted(p.relative_to(d).as_posix() for p in d.rglob("*.json")))
o = json.loads((d / "orchestrator.json").read_text()); t = json.loads((d / "trainer.json").read_text())
i = json.loads((d / "inference.json").read_text()) if (d / "inference.json").exists() else {}
print("model", o["model"]["name"], t["model"]["name"], "max_steps", o.get("max_steps"), t.get("max_steps"), "ckpt", t.get("ckpt"))
print("orch seq_len", o["seq_len"], "batch", o["batch_size"], "constant", o["constant_trainer_batch_size"], "max_staleness", o["max_staleness"], "concurrency", o["concurrency"])
print("renderer", o["renderer"])
print("train group", {k: v for k, v in o["train"].items() if k != "source"})
src = o["train"]["source"][0]
print("source", json.dumps({k: src[k] for k in ("name", "group_size", "sampling", "algo", "select") if k in src})[:1500])
print("env", json.dumps(src["env"])[:3000])
print("trainer seq_len", t["model"]["seq_len"], "loss", t["loss"], "optim", t["optim"], "fused_lm_head", t["model"].get("fused_lm_head_token_chunk_size"), "offload", t["model"].get("offload"))
print("vllm", json.dumps(i.get("vllm") or i)[:800])
EOF
done
# Does the Harbor taskset load the local task copies as verifiers would?
python - <<'EOF'
import verifiers.v1 as vf
from verifiers.v1.tasksets.harbor.taskset import HarborConfig, HarborTaskset
ts = HarborTaskset(HarborConfig(dataset="termgrade-train"))
tasks = list(ts.load())
t = tasks[0].data
print("tasks", len(tasks), "| first", t.name, "| image", getattr(t, "image", None) or getattr(t, "docker_image", None), "| upload_environment", t.upload_environment, "| healthcheck", t.healthcheck, "| resources", getattr(t, "resources", None))
EOF
