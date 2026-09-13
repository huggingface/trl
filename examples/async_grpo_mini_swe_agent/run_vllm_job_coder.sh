#!/usr/bin/env bash
# Start the vLLM Job for the mixture-of-experts policy in a network group: no exposed port, no bucket, no LoRA.
# The trainer Job in the same group reaches it at http://${HF_NETWORK_GROUP_PREFIX}vllm:8000 and streams the merged
# weights over NCCL between the two pods. Prints the Job ID.
#
#   GROUP=my-run ./run_vllm_job_coder.sh
#
# Qwen3-Coder's tool calls are XML, which only its own parser reads, and it has no think block, so there is no
# reasoning parser. The fused 3D expert tensors the trainer sends are loaded as they are: vLLM's Qwen3-MoE loader
# recognizes a fused checkpoint by rank and splits it per expert itself.
#
# The network-group flags are in huggingface_hub main, not yet in a release, hence `uvx --from git+...`.
set -euo pipefail

HF=(uvx --from "huggingface_hub @ git+https://github.com/huggingface/huggingface_hub.git" hf)
GROUP=${GROUP:?set GROUP to the network group name}
MODEL=${MODEL:-Qwen/Qwen3-Coder-30B-A3B-Instruct}
VLLM_TAG=${VLLM_TAG:-v0.27.1}
FLAVOR=${VLLM_FLAVOR:-h200}
TIMEOUT=${VLLM_TIMEOUT:-12h}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-65536}
VLLM_EXTRA_ARGS=${VLLM_EXTRA_ARGS:---enable-auto-tool-choice --tool-call-parser qwen3_coder}

"${HF[@]}" jobs run \
    --name "async-grpo-mini-swe-agent-coder-vllm-${GROUP}" --flavor "$FLAVOR" --timeout "$TIMEOUT" --detach --secrets HF_TOKEN \
    --network-group "$GROUP" --network-alias vllm \
    -e "MODEL=${MODEL}" \
    -e "MAX_MODEL_LEN=${MAX_MODEL_LEN}" \
    -e "VLLM_EXTRA_ARGS=${VLLM_EXTRA_ARGS}" \
    -e HF_HOME=/tmp/hf \
    -e PYTHONUNBUFFERED=1 \
    ` # The weight-transfer endpoints the trainer calls live behind the dev-mode flag. ` \
    -e VLLM_SERVER_DEV_MODE=1 \
    ` # Pods have no InfiniBand; without these NCCL probes IB first and may pick the wrong interface. ` \
    -e NCCL_IB_DISABLE=1 \
    -e NCCL_SOCKET_IFNAME=eth0 \
    -- "vllm/vllm-openai:${VLLM_TAG}" bash -c '
set -euo pipefail
env | grep HF_NETWORK; hostname -i
exec vllm serve "$MODEL" \
    --host 0.0.0.0 --port 8000 \
    --tensor-parallel-size "$(nvidia-smi -L | wc -l)" \
    --dtype bfloat16 \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization 0.9 \
    --logprobs-mode processed_logprobs \
    --generation-config vllm \
    --weight-transfer-config "{\"backend\":\"nccl\"}" \
    $VLLM_EXTRA_ARGS
' 2>&1 | grep -vE 'Warning|warnings\.warn|Request ID' | grep -oE '[0-9a-f]{24}' | tail -1
