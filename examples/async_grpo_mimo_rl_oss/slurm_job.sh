#!/bin/bash
# One Slurm node runs the whole loop: a vLLM server on the first GPUs, the trainer on the rest, rollouts in Hugging
# Face sandboxes reached over the internet, the judge on the Hugging Face router. A full fine-tune streams the
# merged weights to vLLM over NCCL; with `LORA_RANK` set, the adapter each weight sync writes to
# `<output_dir>/.vllm_lora/` is a path the server can open directly since both see the same `/fsx`.
#
# Submitted by `run_slurm.sh`, which sets everything below through the environment. Run it directly only to rerun a
# single configuration:
#
#   PACKING=tree OUTPUT_DIR=/fsx/$USER/runs/tree sbatch --output=/fsx/$USER/runs/logs/%x-%j.out slurm_job.sh
set -uo pipefail

# Slurm runs a *copy* of this script out of its spool directory, so `$0` is not where the example lives.
EXAMPLE_DIR=${EXAMPLE_DIR:-$SLURM_SUBMIT_DIR}
VENV=${VENV:-/fsx/$USER/envs/trl-agent}
HF_HUB=${HF_HUB:-/fsx/$USER/hf-hub}
MODEL=${MODEL:-Qwen/Qwen3-8B}
PACKING=${PACKING:-sequence}
OUTPUT_DIR=${OUTPUT_DIR:?set OUTPUT_DIR}
PROJECT=${PROJECT:-async-grpo-mimo-general-packing}
RUN_NAME=${RUN_NAME:-$PACKING}
VLLM_GPUS=${VLLM_GPUS:-1}
TRAIN_GPUS=${TRAIN_GPUS:-7}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-40960}
LORA_RANK=${LORA_RANK:-0}
TRAIN_ARGS=${TRAIN_ARGS:-}

# `venv-load` prestages the venv onto the node's NVMe; on a partition without one it passes the path through. The
# hub cache stays on /fsx: it holds a handful of large weight files, not the thousands of small ones that make an
# unstaged venv an eight-minute import.
source "$(venv-load "$VENV")"
export HF_HUB_CACHE=$HF_HUB
export PYTHONUNBUFFERED=1
export TRL_EXPERIMENTAL_SILENCE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Node-local: the ranks race on the cache and /fsx does not give Triton the atomic renames it relies on.
export TRITON_CACHE_DIR=/tmp/triton-$SLURM_JOB_ID
# cuDNN and the CUDA runtime ship inside the venv rather than on the node.
SITE_PACKAGES=$(echo "$VIRTUAL_ENV"/lib/python*/site-packages)
export LD_LIBRARY_PATH="$SITE_PACKAGES/nvidia/cudnn/lib:$SITE_PACKAGES/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "$OUTPUT_DIR"
# Jobs share nodes, so never hardcode a port; vLLM also grabs a few adjacent ones for its engine core.
PORT=$((8000 + (SLURM_JOB_ID % 90) * 10))
RDZV_PORT=$((29500 + SLURM_JOB_ID % 1000))
VLLM_LOG=$OUTPUT_DIR/vllm-$SLURM_JOB_ID.log
echo "=== $PACKING | $MODEL | vLLM on $VLLM_GPUS GPU(s) port $PORT | trainer on $TRAIN_GPUS | $OUTPUT_DIR"

# --max-loras >= max_staleness + 2: the trainer keeps `max_staleness + 1` versions servable and loads the next
# before unloading the oldest. 12 covers a staleness of up to 10.
LORA_ARGS=""
[ "$LORA_RANK" -gt 0 ] && LORA_ARGS="--enable-lora --max-lora-rank $LORA_RANK --max-loras 12 --max-cpu-loras 16"

# --logprobs-mode processed_logprobs: the PPO denominator comes from these logprobs.
# --generation-config vllm: ignore the model card's sampling defaults, which would apply to every rollout.
CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((VLLM_GPUS - 1))) \
VLLM_SERVER_DEV_MODE=1 VLLM_ALLOW_RUNTIME_LORA_UPDATING=1 \
    vllm serve "$MODEL" \
        --host 0.0.0.0 --port "$PORT" \
        --tensor-parallel-size "$VLLM_GPUS" \
        --dtype bfloat16 \
        --max-model-len "$MAX_MODEL_LEN" \
        --gpu-memory-utilization 0.85 \
        --logprobs-mode processed_logprobs \
        --generation-config vllm \
        --weight-transfer-config '{"backend":"nccl"}' \
        $LORA_ARGS \
        --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3 \
        > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null' EXIT

echo "=== waiting for vLLM (up to 30 min), log: $VLLM_LOG"
for _ in $(seq 1 360); do
    curl -sf "http://localhost:$PORT/health" > /dev/null && break
    kill -0 $VLLM_PID 2>/dev/null || { echo "!!! vLLM died"; tail -40 "$VLLM_LOG"; exit 1; }
    sleep 5
done
curl -sf "http://localhost:$PORT/health" > /dev/null || { echo "!!! vLLM never came up"; tail -40 "$VLLM_LOG"; exit 1; }
echo "=== vLLM ready"

CUDA_VISIBLE_DEVICES=$(seq -s, "$VLLM_GPUS" $((VLLM_GPUS + TRAIN_GPUS - 1))) \
    accelerate launch --config_file "$EXAMPLE_DIR/fsdp2.yaml" --num_processes "$TRAIN_GPUS" \
        --main_process_port "$RDZV_PORT" \
        "$EXAMPLE_DIR/async_grpo_mimo_general.py" \
        --model "$MODEL" \
        --packing "$PACKING" \
        --lora-rank "$LORA_RANK" \
        --gradient-checkpointing \
        --vllm-url "http://localhost:$PORT" \
        --output-dir "$OUTPUT_DIR" \
        --project "$PROJECT" --run-name "$RUN_NAME" --trackio-space-id "$PROJECT" \
        $TRAIN_ARGS
