#!/bin/bash
# vLLM-only probe / eval of miles' text-to-SQL episode (probe_eval.py) on one node: vLLM DP=NGPU + EP serves MODEL,
# the driver runs the episodes (SQLite on the node's CPUs, snapshots on /scratch).
#   PRIO_TAG=PRIO_TRAIN b200-queue --gres=gpu:8 --exclusive -J t2sql-eval -t 6:00:00 \
#       --export=ALL,MODEL=...,OUT=...,SPECS="name=file:n:T ..." $PWD/eval.sbatch
#SBATCH -J t2sql-eval
#SBATCH --nodes=1
#SBATCH -o /fsx/amine_dirhoussi/stable-bench-runs/trl-moe-ep/jobs/%x-%j.out
set -uo pipefail
HERE=/fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench/workloads/text2sql
D=/fsx/amine_dirhoussi/stable-bench-src/text2sql
VENV=/fsx/amine_dirhoussi/envs/bench-trl-ep
MODEL=${MODEL:-Qwen/Qwen3.5-35B-A3B}; NGPU=${NGPU:-8}
OUT=${OUT:?}; mkdir -p "$OUT"
echo "$(date -u) job $SLURM_JOB_ID on $(hostname): MODEL=$MODEL SPECS=$SPECS" | tee -a "$OUT/job.txt"
export HF_HUB_CACHE=/fsx/amine_dirhoussi/stable-bench-hub HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export VLLM_ENGINE_READY_TIMEOUT_S=${VLLM_ENGINE_READY_TIMEOUT_S:-3600}
export PYTHONPATH=$D/river-recipes/text-to-sql/src:$HERE:/fsx/amine_dirhoussi/trl-wt-asyncgrpo-ep
S=/scratch/$USER/t2sql-$SLURM_JOB_ID; mkdir -p -m 700 $S $S/snapshots $S/tmp; chmod 700 $S/snapshots
export VLLM_CACHE_ROOT=$S/vllm-cache
PORT=$((8000 + (SLURM_JOB_ID % 90) * 10))
FI_WS=/fsx/amine_dirhoussi/flashinfer-ws/$(basename $VENV)
PATH=$VENV/bin:$PATH CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NGPU - 1))) VLLM_SERVER_DEV_MODE=1 FLASHINFER_WORKSPACE_BASE=$FI_WS \
    setsid vllm serve "$MODEL" --port $PORT --dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization 0.85 \
    --data-parallel-size $NGPU --enable-expert-parallel --api-server-count $NGPU --seed 42 \
    --logprobs-mode processed_logprobs --generation-config vllm --return-tokens-as-token-ids \
    --limit-mm-per-prompt '{"image":0,"video":0}' > "$OUT/vllm-$SLURM_JOB_ID.log" 2>&1 &
VLLM_PID=$!
trap 'kill -- -$VLLM_PID 2>/dev/null; sleep 10; kill -9 -- -$VLLM_PID 2>/dev/null; rm -rf $S' EXIT
T0=$(date +%s)
until curl -sf "http://localhost:$PORT/health" > /dev/null; do
    kill -0 $VLLM_PID 2>/dev/null || { echo "vLLM died"; tail -n 40 "$OUT/vllm-$SLURM_JOB_ID.log"; exit 1; }
    sleep 5
done
echo "$(date -u) vLLM ready after $(( $(date +%s) - T0 )) s" | tee -a "$OUT/job.txt"
for i in $(seq 240); do [ -f $D/data/READY ] && break; sleep 30; done  # convert.py output (hopper-cpu prep job)
[ -f $D/data/READY ] || { echo "no $D/data/READY"; exit 1; }
cd $HERE
TEXT_TO_SQL_SQLITE_SNAPSHOT_ROOT=$S/snapshots TMPDIR=$S/tmp CONCURRENCY=${CONCURRENCY:-512} \
    $VENV/bin/python probe_eval.py "http://localhost:$PORT" "$MODEL" "$OUT" $SPECS 2>&1 | tee -a "$OUT/driver-$SLURM_JOB_ID.log"
rc=${PIPESTATUS[0]}
echo "$(date -u) driver exit $rc" | tee -a "$OUT/job.txt"
exit $rc
