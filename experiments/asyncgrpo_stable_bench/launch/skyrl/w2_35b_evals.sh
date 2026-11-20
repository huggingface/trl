#!/bin/bash
# W2 35B variant of w2rec_evals.sh: the Megatron export has no visual.* tensors, so the step dir gets the base vision tower
# (t2sql_evals.sh), and 35B evals get VLLM_ENGINE_READY_TIMEOUT_S=2700 as in the x12 driver.
# Env: KEY (state/eval key prefix, default skyrl9b), MAX_B200_JOBS (skip submitting while I already hold that many b200 jobs).
# Queues the 12-sample local held-out W2 eval (the x12 setup of results/tables/w2_heldout_x12.md, stable-bench-runs/queue/x12/
# driver.sh) for each HF export of a SkyRL recommended W2 run as it lands. Idempotent (state/<key>.waiter), requeue-safe.
#   sbatch -p hopper-cpu --qos=low -c 1 --mem=2G -t 7-0 -J skyrl-w2rec-evals -o /fsx/amine_dirhoussi/stable-bench-runs/skyrl/logs/%x-%j.out \
#       launch/skyrl/w2rec_evals.sh <run id> "20 40 60"
for v in $(compgen -e | grep -E '^(SLURM|SBATCH|SRUN)_'); do unset "$v"; done
BENCH=/fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench
RUN=$1; STEPS=${2:-20 40 60}
RUN_DIR=/fsx/amine_dirhoussi/stable-bench-runs/skyrl/$RUN
ST=/fsx/amine_dirhoussi/stable-bench-runs/skyrl/w2rec/state; mkdir -p $ST
PY=/fsx/amine_dirhoussi/envs/bench-trl-w2/bin/python
export HF_HUB_CACHE=/fsx/amine_dirhoussi/stable-bench-hub HF_HUB_OFFLINE=1
log() { echo "$(date -u +%FT%TZ) $*"; }
while true; do
  left=0
  for N in $STEPS; do
    k=${KEY:-skyrl35b}-$N
    [ -s $ST/$k.waiter ] && continue
    left=1
    EXP=$RUN_DIR/global_step_$N/policy
    # complete: config + weights present, nothing written for 3 min
    $PY - "$EXP" <<'PY' || continue
import os, sys, time
d = sys.argv[1]
fs = [f"{d}/{f}" for f in os.listdir(d)] if os.path.isdir(d) else []
assert os.path.exists(f"{d}/config.json") and any(f.endswith(".safetensors") for f in fs)
assert time.time() - max(os.path.getmtime(f) for f in fs) > 180
PY
    # node budget (audit reruns): at most 2 of my hopper-b200 jobs at once, so an eval waits for a free slot
    [ -n "${MAX_B200_JOBS:-}" ] && [ "$(squeue -u $USER -p hopper-b200 -h | wc -l)" -ge "$MAX_B200_JOBS" ] && continue
    MODEL=$($PY - "$EXP" "$RUN_DIR/eval-model/step-$N" <<'PY'
import json, os, sys
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file
exp, dst = sys.argv[1], sys.argv[2]
idx = json.load(open(f"{exp}/model.safetensors.index.json"))
arch = json.load(open(f"{exp}/config.json"))["architectures"][0]
if "ConditionalGeneration" not in arch or any("visual" in k for k in idx["weight_map"]):
    print(exp); sys.exit()
os.makedirs(dst, exist_ok=True)
for f in os.listdir(exp):
    if f != "model.safetensors.index.json" and not os.path.exists(f"{dst}/{f}"):
        os.symlink(f"{exp}/{f}", f"{dst}/{f}")
base = snapshot_download("Qwen/Qwen3.5-35B-A3B")
bidx = json.load(open(f"{base}/model.safetensors.index.json"))
vis = {}
for shard in sorted({f for k, f in bidx["weight_map"].items() if "visual" in k}):
    with safe_open(f"{base}/{shard}", "pt") as f:
        vis.update({k: f.get_tensor(k) for k in f.keys() if "visual" in k})
save_file(vis, f"{dst}/visual-from-base.safetensors", metadata={"format": "pt"})
idx["weight_map"].update({k: "visual-from-base.safetensors" for k in vis})
json.dump(idx, open(f"{dst}/model.safetensors.index.json", "w"), indent=2)
print(dst)
PY
) || { log "step $N: could not prepare model dir"; continue; }
    out=$ST/$k.submit.log
    ( cd $BENCH && env STAGE=probe SIZE=M NVLLM=4 SPLIT=eval SANDBOX=local HARBOR_SHARDS=4 TERMINUS_MAX_TURNS=1 SAMPLES=12 \
        EVAL_MODEL="$MODEL" VLLM_ENGINE_READY_TIMEOUT_S=2700 HF_HUB_CACHE=/fsx/amine_dirhoussi/stable-bench-hub TAG="x12-$k" PRIO_TAG=PRIO_TRAIN \
        /fsx/amine_dirhoussi/bin/b200-queue --gres=gpu:4 -t 4:00:00 -J "w2local-evalx12-$k" launch/trl/w2.sbatch > "$out" 2>&1 ) &
    pid=$!; until grep -q 'Submitted batch job' "$out" || ! kill -0 $pid 2>/dev/null; do sleep 2; done
    grep -oP 'Submitted batch job \K[0-9]+' "$out" > $ST/$k.waiter
    log "queued $k (waiter $(cat $ST/$k.waiter)) $MODEL"
  done
  [ $left = 0 ] && { log "all queued"; wait; break; }
  sleep 300
done
