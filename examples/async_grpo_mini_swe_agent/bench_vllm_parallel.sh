#!/usr/bin/env bash
# Measure how vLLM serves the MoE policy under different parallelism modes, on the workload the agent actually
# produces: long prompts, short completions, many concurrent rollouts. One Job per mode, each serving and
# benchmarking itself, then exiting. Prints the Job IDs.
#
#   ./bench_vllm_parallel.sh            # all four modes
#   MODES="tp1 ep2" ./bench_vllm_parallel.sh
#
# Read the result with: uvx hf jobs logs <id> | grep -E "Output token throughput|Request throughput|MODE"
set -euo pipefail

MODEL=${MODEL:-Qwen/Qwen3-Coder-30B-A3B-Instruct}
VLLM_TAG=${VLLM_TAG:-v0.27.1}
TIMEOUT=${BENCH_TIMEOUT:-40m}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-65536}
# The agent's shape: a long conversation resent every turn, a short command back, many rollouts at once.
IN_LEN=${IN_LEN:-8000}
OUT_LEN=${OUT_LEN:-500}
CONCURRENCY=${CONCURRENCY:-48}
NUM_PROMPTS=${NUM_PROMPTS:-96}
MODES=${MODES:-tp1 tp2 ep2 dp2ep}

for mode in $MODES; do
    case "$mode" in
        tp1)   flavor=h200;   serve_args="--tensor-parallel-size 1" ;;
        tp2)   flavor=h200x2; serve_args="--tensor-parallel-size 2" ;;
        ep2)   flavor=h200x2; serve_args="--tensor-parallel-size 2 --enable-expert-parallel" ;;
        dp2ep) flavor=h200x2; serve_args="--data-parallel-size 2 --enable-expert-parallel" ;;
        *) echo "unknown mode $mode" >&2; exit 1 ;;
    esac
    uvx hf jobs run \
        --name "vllm-bench-${mode}" --flavor "$flavor" --timeout "$TIMEOUT" --detach --secrets HF_TOKEN \
        -e "MODEL=${MODEL}" -e "MODE=${mode}" -e "SERVE_ARGS=${serve_args}" \
        -e "MAX_MODEL_LEN=${MAX_MODEL_LEN}" -e "IN_LEN=${IN_LEN}" -e "OUT_LEN=${OUT_LEN}" \
        -e "CONCURRENCY=${CONCURRENCY}" -e "NUM_PROMPTS=${NUM_PROMPTS}" \
        -e HF_HOME=/tmp/hf -e PYTHONUNBUFFERED=1 -e VLLM_SERVER_DEV_MODE=1 \
        -- "vllm/vllm-openai:${VLLM_TAG}" bash -c '
set -uo pipefail
echo "=== MODE $MODE: $SERVE_ARGS on $(nvidia-smi -L | wc -l) GPU(s) ==="
vllm serve "$MODEL" --host 127.0.0.1 --port 8000 \
    --dtype bfloat16 --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization 0.9 \
    --logprobs-mode processed_logprobs --generation-config vllm \
    --weight-transfer-config "{\"backend\":\"nccl\"}" \
    $SERVE_ARGS > /tmp/vllm.log 2>&1 &
for i in $(seq 1 240); do
    curl -sf http://127.0.0.1:8000/health > /dev/null && { echo "MODE $MODE server up after $((i * 5))s"; break; }
    sleep 5
done
curl -sf http://127.0.0.1:8000/health > /dev/null || { echo "MODE $MODE server never came up"; tail -40 /tmp/vllm.log; exit 1; }

vllm bench serve --backend openai-chat --endpoint /v1/chat/completions \
    --model "$MODEL" --dataset-name random \
    --random-input-len "$IN_LEN" --random-output-len "$OUT_LEN" \
    --max-concurrency "$CONCURRENCY" --num-prompts "$NUM_PROMPTS" --ignore-eos 2>&1 | tail -40
echo "=== MODE $MODE done ==="
' 2>&1 | grep -vE 'Warning|warnings\.warn|Request ID' | grep -oE '[0-9a-f]{24}' | tail -1
done
