#!/bin/bash
# Greedy evals (TRL's text-to-SQL eval: workloads/text2sql/eval.sbatch, T=0, 1 sample, validation + arcwise
# + hard) of a SkyRL text-to-SQL run's HF exports, queued as each export lands. Idempotent: a step with
# t2sql-eval/<run>-checkpoint-N/queue.log is never queued again, so rerunning (or a requeued helper) is safe.
#   sbatch -p hopper-cpu -c 1 --mem=4G -t 2-0 -J skyrl-t2sql-evals \
#       -o /fsx/amine_dirhoussi/stable-bench-runs/skyrl/logs/%x-%j.out launch/skyrl/t2sql_evals.sh <waiter id> "<steps>"
# SkyRL exports text-only weights for a VLM checkpoint (language_model_only, strict=False). If the export keeps the VLM
# config but lacks the vision tower, the helper builds <run>/eval-model/step-N: symlinks to the export plus the base
# model's visual.* tensors, so vLLM can load it (the eval never sends images).
BENCH=/fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench
STATE=/fsx/amine_dirhoussi/stable-bench-runs/queue/state/$1.jid; STEPS=$2; EVALS=${3:-validation hard arcwise}
D=/fsx/amine_dirhoussi/stable-bench-src/text2sql/data
PY=/fsx/amine_dirhoussi/envs/bench-trl-ep/bin/python
export HF_HUB_CACHE=/fsx/amine_dirhoussi/stable-bench-hub HF_HUB_OFFLINE=1
until [ -s "$STATE" ]; do sleep 120; done
J=$(cat "$STATE")
RUN=skyrl-t2sql-Qwen3.5-35B-A3B-lora32${RUN_SUFFIX:-}-$J
RUN_DIR=/fsx/amine_dirhoussi/stable-bench-runs/skyrl/$RUN
echo "$(date -u +%FT%TZ) run job $J ($RUN)"
SPECS=""; for e in $EVALS; do SPECS="$SPECS eval-$e=$D/$e.jsonl:1:0"; done
declare -A done
n=$(echo $STEPS | wc -w)
while [ ${#done[@]} -lt $n ]; do
  for N in $STEPS; do
    [ -n "${done[$N]:-}" ] && continue
    EXP=$RUN_DIR/global_step_$N/policy
    OUT=/fsx/amine_dirhoussi/stable-bench-runs/trl-moe-ep/t2sql-eval/$RUN-checkpoint-$N
    [ -f $OUT/queue.log ] && { done[$N]=1; continue; }
    # complete = index + config present, every listed shard present, nothing written for 3 min
    $PY - "$EXP" <<'PY' || continue
import json, os, sys, time
d = sys.argv[1]
idx = json.load(open(f"{d}/model.safetensors.index.json"))
files = set(idx["weight_map"].values()) | {"config.json", "model.safetensors.index.json"}
assert all(os.path.exists(f"{d}/{f}") for f in files)
assert time.time() - max(os.path.getmtime(f"{d}/{f}") for f in files) > 180
PY
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
) || { echo "$(date -u +%FT%TZ) step $N: could not prepare model dir"; continue; }
    mkdir -p $OUT
    echo "$(date -u +%FT%TZ) step $N ready ($MODEL), queueing eval -> $OUT"
    (cd $BENCH && PRIO_TAG=PRIO_TRAIN HF_HUB_CACHE=/fsx/amine_dirhoussi/stable-bench-hub nohup /fsx/amine_dirhoussi/bin/b200-queue -J t2sql-eval --gres=gpu:4 -t 4:00:00 \
      --export=ALL,MODEL=$MODEL,OUT=$OUT,NGPU=4,SPECS="$SPECS" $BENCH/workloads/text2sql/eval.sbatch > $OUT/queue.log 2>&1 &)
    done[$N]=1
  done
  [ -n "${ended:-}" ] && { echo "$(date -u +%FT%TZ) run $J ended; queued: ${!done[*]}"; break; }
  squeue -h -j "$J" | grep -q . || ended=1
  sleep 120
done
wait
