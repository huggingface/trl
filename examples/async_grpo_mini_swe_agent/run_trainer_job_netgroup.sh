#!/usr/bin/env bash
# Start the trainer Job in the network group of `run_vllm_job_netgroup.sh`: full fine-tuning by default (`--lora-rank 0`),
# weights synced to vLLM over NCCL between the two pods, no proxy and no bucket. Prints the Job ID.
#
#   GROUP=my-run ./run_trainer_job_netgroup.sh
set -euo pipefail

cd "$(dirname "$0")"

HF=(uvx --from "huggingface_hub @ git+https://github.com/huggingface/huggingface_hub.git" hf)
GROUP=${GROUP:?set GROUP to the network group name}
MODEL=${MODEL:-Qwen/Qwen3-8B}
TRL_REF=${TRL_REF:-agent-rl-example}
VLLM_TAG=${VLLM_TAG:-v0.27.1}
FLAVOR=${TRAIN_FLAVOR:-h200x4}
TIMEOUT=${TRAIN_TIMEOUT:-8h}
RUN_TAG=${RUN_TAG:-fullft}
PROJECT=${PROJECT:-async-grpo-mini-swe-agent}
TRAIN_ARGS=${TRAIN_ARGS:---lora-rank 0}
# The NCCL the vLLM image ships; the trainer's pip installs pull torch's own, and the two versions cannot bootstrap together.
NCCL_VERSION=${NCCL_VERSION:-2.30.7}

"${HF[@]}" jobs run \
    --name "async-grpo-mini-swe-agent-train-${GROUP}" --flavor "$FLAVOR" --timeout "$TIMEOUT" --detach --secrets HF_TOKEN \
    --network-group "$GROUP" --network-alias trainer \
    -v "$PWD:/work" \
    -e "MODEL=${MODEL}" \
    -e "TRL_REF=${TRL_REF}" \
    -e "RUN_TAG=${RUN_TAG}" \
    -e "PROJECT=${PROJECT}" \
    -e "TRAIN_ARGS=${TRAIN_ARGS}" \
    -e "NCCL_VERSION=${NCCL_VERSION}" \
    -e HF_HOME=/tmp/hf \
    -e PYTHONUNBUFFERED=1 \
    -e TRL_EXPERIMENTAL_SILENCE=1 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e NCCL_IB_DISABLE=1 \
    -e NCCL_SOCKET_IFNAME=eth0 \
    -- "vllm/vllm-openai:${VLLM_TAG}" bash -c '
set -euo pipefail
env | grep HF_NETWORK; hostname -i
for attempt in 1 2 3 4 5; do
    pip install -q "https://codeload.github.com/huggingface/trl/tar.gz/${TRL_REF}" \
        "https://codeload.github.com/huggingface/OpenEnv/tar.gz/main" \
        "https://codeload.github.com/SWE-Gym/SWE-Bench-Package/tar.gz/main" \
        peft bitsandbytes "kernels>=0.16,<0.17" trackio mini-swe-agent openai "huggingface_hub>=1.22" && break
    echo "pip install failed (attempt $attempt), retrying in 30s"; sleep 30
done
pip install -q "nvidia-nccl-cu13==${NCCL_VERSION}" || pip install -q "nvidia-nccl-cu12==${NCCL_VERSION}"
python3 -c "import torch; print(\"torch\", torch.__version__, \"nccl\", torch.cuda.nccl.version())"

V="${HF_NETWORK_GROUP_PREFIX}vllm"
echo "=== waiting for http://$V:8000/health (up to 1800s) ==="
for i in $(seq 1 360); do
    curl -sf "http://$V:8000/health" > /dev/null && { echo "vLLM reachable after $((i * 5))s"; break; }
    sleep 5
done
curl -sf "http://$V:8000/health" > /dev/null || { echo "!!! vLLM unreachable at $V"; getent hosts "$V"; exit 1; }

# The trainer hands vLLM the address it must dial back for the NCCL group: this pod.
export VLLM_HOST_IP="$(hostname -i | awk "{print \$1}")"
NPROC=$(nvidia-smi -L | wc -l)
echo "=== trainer: FSDP2 on $NPROC rank(s), NCCL master $VLLM_HOST_IP ==="
accelerate launch --config_file /work/fsdp2.yaml --num_processes "$NPROC" /work/async_grpo_mini_swe_agent.py \
    --model "$MODEL" --vllm-url "http://$V:8000" --output-dir "/tmp/${RUN_TAG}" \
    --project "$PROJECT" --run-name "$RUN_TAG" --trackio-space-id "$PROJECT" \
    $TRAIN_ARGS
' 2>&1 | grep -oE '[0-9a-f]{24}' | head -1
