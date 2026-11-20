#!/bin/bash
# hopper-cpu: dry-run both W1 arms and print the resolved knobs that matter, plus the rendered prompt tail.
#SBATCH --job-name=prl-dry
#SBATCH --partition=hopper-cpu
#SBATCH --qos=low
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:40:00
#SBATCH --output=/fsx/amine_dirhoussi/stable-bench-runs/prime-rl/logs/%x-%j.out
set -uo pipefail
BENCH=/fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench
source /fsx/amine_dirhoussi/envs/bench-prime-rl/bin/activate
export HF_HUB_CACHE=/fsx/amine_dirhoussi/stable-bench-hub PYTHONPATH=$BENCH/configs/prime-rl
DRY=/fsx/amine_dirhoussi/stable-bench-runs/prime-rl/dryrun
cd /fsx/amine_dirhoussi/stable-bench-src/prime-rl
for arm in ${ARMS:-S S-recommended}; do
    echo "=== dry-run w1-$arm"
    rl @ "$BENCH/configs/prime-rl/w1-$arm.toml" --output-dir "$DRY" --run.name "w1-$arm" --clean --no-monitors.prime --dry-run \
        --inference.server.port 8123 --orchestrator.model.client.base-url http://localhost:8123/v1 \
        --rollout-transport.type zmq --rollout-transport.port 8126 --weight-broadcast.type nccl --weight-broadcast.port 8128 ${EXTRA:-} 2>&1 | tail -n 4
    python - "$DRY/w1-$arm" <<'EOF'
import json, sys, pathlib
from transformers import AutoTokenizer
from renderers import create_renderer
from renderers.configs import Qwen35RendererConfig
d = pathlib.Path(sys.argv[1]) / "configs/latest/resolved"
o = json.loads((d / "orchestrator.json").read_text()); t = json.loads((d / "trainer.json").read_text())
print("model", o["model"]["name"], t["model"]["name"], "max_steps", o.get("max_steps"), t.get("max_steps"), "ckpt", t.get("ckpt"))
print("orch seq_len", o["seq_len"], "batch", o["batch_size"], "constant", o["constant_trainer_batch_size"], "max_staleness", o["max_staleness"])
print("renderer", o["renderer"])
print("trainer seq_len", t["model"]["seq_len"], "loss", t["loss"], "lr", t["optim"]["lr"])
print("file monitor orch", o["monitors"].get("file"), "trainer", t["monitors"].get("file"))
rc = {k: v for k, v in o["renderer"].items() if k in ("enable_thinking",)}
tok = AutoTokenizer.from_pretrained(o["model"]["name"])
msgs = [{"role": "user", "content": "What is $1+1$?"}]
ids = list(create_renderer(tok, Qwen35RendererConfig(**rc)).render_ids(msgs, add_generation_prompt=True))
kw = {} if rc.get("enable_thinking") is None else {"enable_thinking": rc["enable_thinking"]}
ref = tok(tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, **kw), add_special_tokens=False)["input_ids"]
print("renderer kwargs", rc, "| match apply_chat_template", ids == ref, "| tail", repr(tok.decode(ids[-8:])))
EOF
done
