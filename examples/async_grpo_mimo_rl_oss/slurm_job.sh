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
MODEL=${MODEL:-XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B}
DOMAINS=${DOMAINS:-general}
PACKING=${PACKING:-sequence}
# MiMo's own reasoning and tool-call formats; Qwen3 wants `hermes` and `qwen3`.
TOOL_PARSER=${TOOL_PARSER:-mimo}
REASONING_PARSER=${REASONING_PARSER:-mimo}
OUTPUT_DIR=${OUTPUT_DIR:?set OUTPUT_DIR}
PROJECT=${PROJECT:-async-grpo-mimo}
RUN_NAME=${RUN_NAME:-$DOMAINS}
VLLM_GPUS=${VLLM_GPUS:-1}
# Replicas, not a wider shard: KV cache is what a 9B runs out of when hundreds of agents are in flight, and each
# replica brings its own. Tensor parallelism only helps a model too big for one GPU.
VLLM_DP=${VLLM_DP:-$VLLM_GPUS}
VLLM_TP=${VLLM_TP:-1}
TRAIN_GPUS=${TRAIN_GPUS:-2}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-40960}
LORA_RANK=${LORA_RANK:-0}
TRAIN_ARGS=${TRAIN_ARGS:-}
# A full fine-tune reshards after the forward pass; an adapter run keeps the gathered parameters resident.
FSDP_CONFIG=${FSDP_CONFIG:-$([ "$LORA_RANK" -gt 0 ] && echo fsdp2.yaml || echo fsdp2_fullft.yaml)}

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
# Every data-parallel worker compiles the model and reads its weights off /fsx at the same time, which takes a
# multiple of the single-GPU start that the 600 s default was sized for.
export VLLM_ENGINE_READY_TIMEOUT_S=${VLLM_ENGINE_READY_TIMEOUT_S:-2400}
# cuDNN and the CUDA runtime ship inside the venv rather than on the node.
SITE_PACKAGES=$(echo "$VIRTUAL_ENV"/lib/python*/site-packages)
export LD_LIBRARY_PATH="$SITE_PACKAGES/nvidia/cudnn/lib:$SITE_PACKAGES/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "$OUTPUT_DIR"
# Jobs share nodes, so never hardcode a port; vLLM also grabs a few adjacent ones for its engine core.
PORT=$((8000 + (SLURM_JOB_ID % 90) * 10))
RDZV_PORT=$((29500 + SLURM_JOB_ID % 1000))
VLLM_LOG=$OUTPUT_DIR/vllm-$SLURM_JOB_ID.log

# On one node the server and the trainer split its GPUs. On two, the server takes the second node whole and the
# trainer the first, which is what feeding hundreds of concurrent sandboxes needs: KV cache, not compute, is what
# runs out first, and it scales with replicas.
NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
if [ "${#NODES[@]}" -gt 1 ]; then
    VLLM_HOST=${NODES[1]}
    VLLM_DEVICES=$(seq -s, 0 $((VLLM_GPUS - 1)))
    TRAIN_DEVICES=$(seq -s, 0 $((TRAIN_GPUS - 1)))
    # `--overlap --exact` because a step otherwise claims the whole allocation and the second one would wait for
    # the first to exit instead of running beside it.
    VLLM_LAUNCH=(srun --overlap --exact --nodes=1 --ntasks=1 --gres=gpu:"$VLLM_GPUS" -w "$VLLM_HOST")
    TRAIN_LAUNCH=(srun --overlap --exact --nodes=1 --ntasks=1 --gres=gpu:"$TRAIN_GPUS" -w "${NODES[0]}")
else
    VLLM_HOST=localhost
    VLLM_DEVICES=$(seq -s, 0 $((VLLM_GPUS - 1)))
    TRAIN_DEVICES=$(seq -s, "$VLLM_GPUS" $((VLLM_GPUS + TRAIN_GPUS - 1)))
    VLLM_LAUNCH=()
    TRAIN_LAUNCH=()
fi
echo "=== $DOMAINS | $PACKING | $MODEL | vLLM $VLLM_GPUS GPU(s) on $VLLM_HOST:$PORT | trainer $TRAIN_GPUS on ${NODES[0]} | $OUTPUT_DIR"
# The resolved configuration, one `key=value` a line, so a run can be read back from its own log instead of from
# whatever the launcher happened to be set to at the time.
{
    echo "CONFIG model=$MODEL"
    echo "CONFIG domains=$DOMAINS"
    echo "CONFIG packing=$PACKING"
    echo "CONFIG lora_rank=$LORA_RANK"
    echo "CONFIG max_model_len=$MAX_MODEL_LEN"
    echo "CONFIG nodes=${#NODES[@]}"
    echo "CONFIG vllm_gpus=$VLLM_GPUS vllm_tp=$VLLM_TP vllm_dp=$VLLM_DP"
    echo "CONFIG train_gpus=$TRAIN_GPUS fsdp=$FSDP_CONFIG"
    echo "CONFIG parsers=$TOOL_PARSER/$REASONING_PARSER"
    echo "CONFIG train_args=$TRAIN_ARGS"
} | tee -a "$OUTPUT_DIR/config-$SLURM_JOB_ID.txt"

# --max-loras >= max_staleness + 2: the trainer keeps `max_staleness + 1` versions servable and loads the next
# before unloading the oldest. 12 covers a staleness of up to 10.
LORA_ARGS=""
[ "$LORA_RANK" -gt 0 ] && [ "$VLLM_DP" -eq 1 ] && LORA_ARGS="--enable-lora --max-lora-rank $LORA_RANK --max-loras 12 --max-cpu-loras 16"

# --logprobs-mode processed_logprobs: the PPO denominator comes from these logprobs.
# --generation-config vllm: ignore the model card's sampling defaults, which would apply to every rollout.
# Runtime LoRA updating is what lets the trainer push an adapter instead of the merged weights, and vLLM refuses it
# once data parallelism gives the server more than one API process. A full fine-tune never pushes an adapter, so the
# variable is only set when there is one.
# Only for a single-replica adapter run: vLLM refuses the flag once data parallelism gives the server more than one
# API process, and with several replicas the trainer merges the adapter and streams the full weights anyway -- which
# NCCL does in about a second, so the adapter fast path is not worth arranging the server around.
LORA_ENV=()
[ "$LORA_RANK" -gt 0 ] && [ "$VLLM_DP" -eq 1 ] && LORA_ENV=(VLLM_ALLOW_RUNTIME_LORA_UPDATING=1)
# Through `env`, not as a bare prefix: bash resolves assignment prefixes before it expands anything, so an expanded
# `VAR=1` is run as a command rather than exported.
CUDA_VISIBLE_DEVICES=$VLLM_DEVICES VLLM_SERVER_DEV_MODE=1 \
    env "${LORA_ENV[@]}" "${VLLM_LAUNCH[@]}" vllm serve "$MODEL" \
        --host 0.0.0.0 --port "$PORT" \
        --tensor-parallel-size "$VLLM_TP" \
        --data-parallel-size "$VLLM_DP" \
        --dtype bfloat16 \
        --max-model-len "$MAX_MODEL_LEN" \
        --gpu-memory-utilization 0.85 \
        --logprobs-mode processed_logprobs \
        --generation-config vllm \
        --weight-transfer-config '{"backend":"nccl"}' \
        $LORA_ARGS \
        --enable-auto-tool-choice --tool-call-parser "$TOOL_PARSER" --reasoning-parser "$REASONING_PARSER" \
        > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null' EXIT

echo "=== waiting for vLLM (up to 30 min), log: $VLLM_LOG"
for _ in $(seq 1 360); do
    curl -sf "http://$VLLM_HOST:$PORT/health" > /dev/null && break
    kill -0 $VLLM_PID 2>/dev/null || { echo "!!! vLLM died"; tail -40 "$VLLM_LOG"; exit 1; }
    sleep 5
done
curl -sf "http://$VLLM_HOST:$PORT/health" > /dev/null || { echo "!!! vLLM never came up"; tail -40 "$VLLM_LOG"; exit 1; }
echo "=== vLLM ready"

CUDA_VISIBLE_DEVICES=$TRAIN_DEVICES \
    "${TRAIN_LAUNCH[@]}" accelerate launch --config_file "$EXAMPLE_DIR/$FSDP_CONFIG" --num_processes "$TRAIN_GPUS" \
        --main_process_port "$RDZV_PORT" \
        "$EXAMPLE_DIR/async_grpo_mimo.py" \
        --model "$MODEL" \
        --domains "$DOMAINS" \
        --packing "$PACKING" \
        --lora-rank "$LORA_RANK" \
        --gradient-checkpointing \
        --vllm-url "http://$VLLM_HOST:$PORT" \
        --output-dir "$OUTPUT_DIR" \
        --project "$PROJECT" --run-name "$RUN_NAME" --trackio-space-id "$PROJECT" \
        $TRAIN_ARGS
