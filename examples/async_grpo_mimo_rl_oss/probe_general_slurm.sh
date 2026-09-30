#!/bin/bash
# Probe a policy on the General domain of MiMo-V2.6-RL-oss on one Slurm node: a vLLM server on the GPUs and
# `probe_general.py` next to it, rollouts in Hugging Face sandboxes reached over the internet, the judge on the
# Hugging Face router.
#
#   mkdir -p /fsx/$USER/runs/mimo-general-probe/logs
#   sbatch examples/async_grpo_mimo_rl_oss/probe_general_slurm.sh
#   MODEL=Qwen/Qwen3-32B VLLM_GPUS=2 sbatch --gres=gpu:2 examples/async_grpo_mimo_rl_oss/probe_general_slurm.sh
#
# Both venvs have to be staged first, once, from a login node: `venv-ship /fsx/$USER/envs/trl-agent` (vLLM) and
# `venv-ship /fsx/$USER/envs/mimo-probe` (the probe: mimoagent, python 3.12). Download the task directories from a
# login node too: a task is ~50 small files, and from hopper-atl the hub cache on /fsx takes a second per file.
#
#   HF_HUB_CACHE=/fsx/$USER/hf-hub python -c "from probe_general import load_tasks; load_tasks(1000, 0, language='en')"
#SBATCH --job-name=mimo-general-probe
#SBATCH --partition=hopper-atl
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=/fsx/%u/runs/mimo-general-probe/logs/%x-%j.out
set -uo pipefail

EXAMPLE_DIR=${EXAMPLE_DIR:-$SLURM_SUBMIT_DIR/examples/async_grpo_mimo_rl_oss}
VLLM_VENV=${VLLM_VENV:-/fsx/$USER/envs/trl-agent}
PROBE_VENV=${PROBE_VENV:-/fsx/$USER/envs/mimo-probe}
HF_HUB=${HF_HUB:-/fsx/$USER/hf-hub}
MODEL=${MODEL:-Qwen/Qwen3-8B}
VLLM_GPUS=${VLLM_GPUS:-1}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-40960}
N_TASKS=${N_TASKS:-64}
SAMPLES=${SAMPLES:-4}
MAX_INFLIGHT=${MAX_INFLIGHT:-32}
SEED=${SEED:-0}
PROBE_ARGS=${PROBE_ARGS:-}
RUN_DIR=${RUN_DIR:-/fsx/$USER/runs/mimo-general-probe/$SLURM_JOB_ID}

export HF_HUB_CACHE=$HF_HUB
export PYTHONUNBUFFERED=1
mkdir -p "$RUN_DIR"
PORT=$((8000 + (SLURM_JOB_ID % 90) * 10))
VLLM_LOG=$RUN_DIR/vllm.log
echo "=== $MODEL | vLLM on $VLLM_GPUS GPU(s) port $PORT | $N_TASKS tasks x $SAMPLES samples | $RUN_DIR"

# The vLLM venv ships its own cuDNN and CUDA runtime.
source "$(venv-load "$VLLM_VENV")"
SITE_PACKAGES=$(echo "$VIRTUAL_ENV"/lib/python*/site-packages)
export LD_LIBRARY_PATH="$SITE_PACKAGES/nvidia/cudnn/lib:$SITE_PACKAGES/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((VLLM_GPUS - 1))) \
    vllm serve "$MODEL" \
        --host 0.0.0.0 --port "$PORT" \
        --tensor-parallel-size "$VLLM_GPUS" \
        --dtype bfloat16 \
        --max-model-len "$MAX_MODEL_LEN" \
        --gpu-memory-utilization 0.85 \
        --generation-config vllm \
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

source "$(venv-load "$PROBE_VENV")"
python "$EXAMPLE_DIR/probe_general.py" \
    --model "$MODEL" --base-url "http://localhost:$PORT/v1" \
    --n-tasks "$N_TASKS" --samples-per-task "$SAMPLES" --max-inflight "$MAX_INFLIGHT" --seed "$SEED" \
    --output "$RUN_DIR/probe.jsonl" \
    $PROBE_ARGS
echo "=== done: $RUN_DIR/probe.jsonl"
