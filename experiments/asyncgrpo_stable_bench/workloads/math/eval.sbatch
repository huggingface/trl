#!/bin/bash
# Evaluate every checkpoint-<step> under a run dir, one checkpoint per GPU in parallel, rows appended to OUT.
#   b200-queue --gres=gpu:4 workloads/math/eval.sbatch <run dir with checkpoint-*> <eval.jsonl> [python]
# The python must have vllm, pandas and math-verify; it defaults to the TRL bench venv.
# Env: MAX_TOKENS (16384), ENABLE_THINKING (false); SHARED_W1 rev. 2.
#SBATCH -J math-eval
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH -t 4:00:00
#SBATCH -o /fsx/amine_dirhoussi/stable-bench-runs/trl/eval/%x-%j.out
set -uo pipefail
RUN_DIR=$1; OUT=$2; PY=${3:-/fsx/amine_dirhoussi/envs/bench-trl/bin/python}
export PATH=$(dirname "$PY"):$PATH HF_HUB_CACHE=/fsx/amine_dirhoussi/stable-bench-hub
cd /fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench/workloads/math
NGPU=$(nvidia-smi -L | wc -l)
i=0
for ckpt in $(ls -d "$RUN_DIR"/checkpoint-* | sort -t- -k2 -n); do
    step=${ckpt##*-}
    CUDA_VISIBLE_DEVICES=$((i % NGPU)) $PY eval.py --model "$ckpt" --step "$step" --out "$OUT" \
        --max-tokens ${MAX_TOKENS:-16384} --enable-thinking ${ENABLE_THINKING:-false} \
        --samples-out "$RUN_DIR/eval-samples-$step.jsonl" > "$RUN_DIR/eval-$step.log" 2>&1 &
    i=$((i + 1))
    (( i % NGPU == 0 )) && wait
done
wait
cat "$OUT"
