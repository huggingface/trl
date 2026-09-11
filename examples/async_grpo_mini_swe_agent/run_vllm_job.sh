#!/usr/bin/env bash
# Start the vLLM Jobs: `VLLM_REPLICAS` identical Jobs (default 1), LoRA serving on, the shared bucket mounted read-only
# at /lora, port 8000 exposed. Prints one Job ID per line on stdout and nothing else, so `run_all.sh` can capture
# them. `lora_proxy.py` on the trainer side routes rollouts across the replicas and broadcasts every adapter load to
# all of them, so from trl's point of view this is one server.
#
#   ./run_vllm_job.sh
#   VLLM_REPLICAS=2 ./run_vllm_job.sh
#
# This side installs nothing: on the adapter-only sync path the trainer hands vLLM a *path* over
# `/v1/load_lora_adapter`, and vLLM's own filesystem LoRA loader reads it off the bucket mount.
set -euo pipefail

BUCKET=${BUCKET:-aminediroHF/async-grpo-mini-swe-agent}
MODEL=${MODEL:-Qwen/Qwen3-32B}
VLLM_TAG=${VLLM_TAG:-v0.27.1}
FLAVOR=${VLLM_FLAVOR:-h200x2}
TIMEOUT=${VLLM_TIMEOUT:-8h}
# Prompt + completion. An agent trajectory re-sends the whole conversation every turn, and tool outputs run up to
# 10k characters each, so this is what bounds how many steps a rollout can take.
MAX_MODEL_LEN=${MAX_MODEL_LEN:-40960}
# A capacity bound, not the rank served; must be one of 1, 8, 16, 32, 64, 128, 256, 320, 512.
MAX_LORA_RANK=${MAX_LORA_RANK:-32}
# At least `max_staleness + 2`: the trainer keeps `max_staleness + 1` adapter versions servable, and each sync loads
# the next one before unloading the oldest.
MAX_LORAS=${MAX_LORAS:-6}
# Tool calling is required (mini-swe-agent's bash tool); the reasoning parser is for Qwen3 hybrid-thinking models.
VLLM_EXTRA_ARGS=${VLLM_EXTRA_ARGS:---enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3}
VLLM_REPLICAS=${VLLM_REPLICAS:-1}

for replica in $(seq 1 "$VLLM_REPLICAS"); do
uvx hf jobs run \
    --name "async-grpo-mini-swe-agent-vllm-${replica}" --flavor "$FLAVOR" --timeout "$TIMEOUT" --detach --secrets HF_TOKEN \
    --expose 8000 \
    -v "hf://buckets/${BUCKET}:/lora:ro" \
    -e "MODEL=${MODEL}" \
    -e "MAX_MODEL_LEN=${MAX_MODEL_LEN}" \
    -e "MAX_LORA_RANK=${MAX_LORA_RANK}" \
    -e "MAX_LORAS=${MAX_LORAS}" \
    -e "VLLM_EXTRA_ARGS=${VLLM_EXTRA_ARGS}" \
    -e HF_HOME=/tmp/hf \
    -e PYTHONUNBUFFERED=1 \
    -e VLLM_SERVER_DEV_MODE=1 \
    -e VLLM_ALLOW_RUNTIME_LORA_UPDATING=1 \
    -- "vllm/vllm-openai:${VLLM_TAG}" bash -c '
set -euo pipefail
grep -q /lora /proc/mounts || { echo "no /lora mount!"; exit 1; }
# --logprobs-mode processed_logprobs: the PPO denominator comes from these logprobs.
# --generation-config vllm: ignore the model card sampling defaults, which would apply to every rollout.
exec vllm serve "$MODEL" \
    --host 0.0.0.0 --port 8000 \
    --tensor-parallel-size "$(nvidia-smi -L | wc -l)" \
    --dtype bfloat16 \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization 0.9 \
    --logprobs-mode processed_logprobs \
    --generation-config vllm \
    --weight-transfer-config "{\"backend\":\"nccl\"}" \
    --enable-lora --max-lora-rank "$MAX_LORA_RANK" --max-loras "$MAX_LORAS" --max-cpu-loras 8 \
    $VLLM_EXTRA_ARGS
' 2>&1 | grep -oE '[0-9a-f]{24}' | head -1
done
