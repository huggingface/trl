#!/bin/bash
# vLLM exactly as launch/trl/w1.sbatch serves it (DP=4, 8 API servers, GPUs 0-3), loaded like the rollout worker at
# 688 in flight, with TRL's /pause?mode=keep&clear_cache=false + /resume every 20 s. One arm after another:
# comp8 (run 364705), gen8 (proposed fix), comp0 (run 364684). Only addition to the w1 serve args: --enable-scale-out,
# which mounts /inference/v1/generate (and /inference/v1/abort_requests, used between arms).
# Knobs: ARMS (comp8 gen8 comp0, also comp4/comp2/comp1), MEASURE_S, FILL_S, API_SERVERS (--api-server-count, default 8).
#   b200-queue --nodes=1 --gres=gpu:8 -t 30 envs/trl/pause_stall/repro.sbatch
#SBATCH -J pause-repro
#SBATCH --nodes=1
#SBATCH -t 00:30:00
#SBATCH -o /fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench/envs/trl/pause_stall/logs/%x-%j.out
set -uo pipefail
D=/fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench/envs/trl/pause_stall
VENV=/fsx/amine_dirhoussi/envs/bench-trl
OUT=$D/logs/$SLURM_JOB_ID${OUT_SUFFIX:-}; mkdir -p "$OUT"
export PATH=$VENV/bin:$PATH HF_HUB_CACHE=/fsx/amine_dirhoussi/stable-bench-hub PYTHONUNBUFFERED=1
PORT=$((8000 + (SLURM_JOB_ID % 90) * 10))
URL=http://localhost:$PORT
ARMS=${ARMS:-comp8 gen8 comp0}

CUDA_VISIBLE_DEVICES=0,1,2,3 VLLM_SERVER_DEV_MODE=1 vllm serve Qwen/Qwen3.5-2B \
    --port $PORT --dtype bfloat16 --max-model-len 18432 --gpu-memory-utilization 0.9 \
    --data-parallel-size 4 --api-server-count ${API_SERVERS:-8} --seed 42 \
    --logprobs-mode processed_logprobs --generation-config vllm \
    --limit-mm-per-prompt '{"image":0,"video":0}' \
    --weight-transfer-config '{"backend":"nccl"}' --enable-scale-out > "$OUT/vllm.log" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID $CPU_PID 2>/dev/null; pkill -P $$ 2>/dev/null' EXIT
until curl -sf "$URL/health" > /dev/null; do
    kill -0 $VLLM_PID 2>/dev/null || { echo "vLLM died during startup"; tail -n 60 "$OUT/vllm.log"; exit 1; }
    sleep 5
done
echo "vllm ready $(date -u +%T) api_servers=${API_SERVERS:-8} arms=$ARMS measure_s=${MEASURE_S:-170}"
python $D/cpu.py $VLLM_PID "$OUT/cpu.jsonl" &
CPU_PID=$!
for arm in $ARMS; do
    echo "arm $arm start $(date -u +%T)"
    python $D/load.py "$URL" "$arm" "$OUT/load.jsonl" ${FILL_S:-120} ${MEASURE_S:-170}
    grep "\"arm\": \"$arm\"" "$OUT/load.jsonl" | grep -E '"kind": "(pause|measure_end|check|error)"' | cut -c1-300
done
echo "done $(date -u +%T)"
