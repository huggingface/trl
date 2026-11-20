#!/bin/bash
# Evaluate the pending checkpoints of several runs in one 8-GPU job, one checkpoint per GPU, 8 at a time.
#   b200-queue --gres=gpu:N [-w node] workloads/math/eval_pack.sbatch <run dir>...  (N parallel evals, N <= 8)
# A checkpoint is pending when its step is not yet in results/raw/<run id>/eval.jsonl. Same eval.py call as eval.sbatch.
#SBATCH -J eval-pack
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH -t 8:00:00
#SBATCH -o /fsx/amine_dirhoussi/stable-bench-runs/trl/eval/%x-%j.out
set -uo pipefail
BENCH=/fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench
PY=${EVAL_PY:-/fsx/amine_dirhoussi/envs/bench-trl/bin/python}
export PATH=$(dirname "$PY"):$PATH HF_HUB_CACHE=/fsx/amine_dirhoussi/stable-bench-hub
cd $BENCH/workloads/math
jobs_list=()
for run in "$@"; do
    run=$(readlink -f "$run"); out=$BENCH/results/raw/$(basename "$run")/eval.jsonl; mkdir -p "$(dirname "$out")"
    for ckpt in $(ls -d "$run"/checkpoint-* 2>/dev/null | sort -t- -k2 -n); do
        step=${ckpt##*-}
        grep -q "\"step\": $step[,}]" "$out" 2>/dev/null && continue
        jobs_list+=("$ckpt|$step|$out|$run")
    done
done
echo "pending: ${#jobs_list[@]}"; printf '%s\n' "${jobs_list[@]}"
# GPU pool over the GPUs Slurm gave this job: worker g takes items g, g+NG, g+2NG, ...
GPUS=(${CUDA_VISIBLE_DEVICES//,/ }); NG=${#GPUS[@]}
for ((g = 0; g < NG; g++)); do
    (
        for ((k = g; k < ${#jobs_list[@]}; k += NG)); do
            IFS='|' read -r ckpt step out run <<<"${jobs_list[$k]}"
            echo "$(date -u +%T) gpu$g start $ckpt"
            CUDA_VISIBLE_DEVICES=${GPUS[$g]} $PY eval.py --model "$ckpt" --step "$step" --out "$out" \
                --max-tokens ${MAX_TOKENS:-16384} --enable-thinking ${ENABLE_THINKING:-false} \
                --samples-out "$run/eval-samples-$step.jsonl" > "$run/eval-$step.log" 2>&1
            echo "$(date -u +%T) gpu$g done $ckpt rc=$?"
        done
    ) &
done
wait
