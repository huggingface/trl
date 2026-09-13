#!/usr/bin/env bash
# Start the trainer Job for `async_grpo_mini_swe_agent_coder.py` in the network group of `run_vllm_job_coder.sh`:
# every weight of the MoE trains, the experts are sharded across the ranks of an expert-parallel mesh, and the
# merged weights reach vLLM over NCCL. No proxy, no bucket. Prints the Job ID.
#
#   GROUP=my-run ./run_trainer_job_coder.sh
#
# Two things differ from the other trainer Jobs. Transformers comes from the `ep-fsdp-2d-mesh` branch, which is what
# builds the (fsdp, tp) mesh and dispatches tokens to the rank that owns their expert. And the launcher is torchrun,
# not accelerate: the model is already sharded when the Trainer sees it, so an FSDP plugin would try to shard it a
# second time.
set -euo pipefail

cd "$(dirname "$0")"

HF=(uvx --from "huggingface_hub @ git+https://github.com/huggingface/huggingface_hub.git" hf)
GROUP=${GROUP:?set GROUP to the network group name}
MODEL=${MODEL:-Qwen/Qwen3-Coder-30B-A3B-Instruct}
TRL_REF=${TRL_REF:-agent-rl-example}
TRANSFORMERS_REF=${TRANSFORMERS_REF:-ep-fsdp-2d-mesh}
VLLM_TAG=${VLLM_TAG:-v0.27.1}
FLAVOR=${TRAIN_FLAVOR:-h200x4}
TIMEOUT=${TRAIN_TIMEOUT:-12h}
RUN_TAG=${RUN_TAG:-coder-ep}
PROJECT=${PROJECT:-async-grpo-mini-swe-agent-coder}
TRAIN_ARGS=${TRAIN_ARGS:---tp-size 4}
# The NCCL the vLLM image ships; the trainer's pip installs pull torch's own, and the two versions cannot bootstrap together.
NCCL_VERSION=${NCCL_VERSION:-2.30.7}

"${HF[@]}" jobs run \
    --name "async-grpo-mini-swe-agent-coder-train-${GROUP}" --flavor "$FLAVOR" --timeout "$TIMEOUT" --detach --secrets HF_TOKEN \
    --network-group "$GROUP" --network-alias trainer \
    -v "$PWD:/work" \
    -e "MODEL=${MODEL}" \
    -e "TRL_REF=${TRL_REF}" \
    -e "TRANSFORMERS_REF=${TRANSFORMERS_REF}" \
    -e "RUN_TAG=${RUN_TAG}" \
    -e "PROJECT=${PROJECT}" \
    -e "TRAIN_ARGS=${TRAIN_ARGS}" \
    -e "NCCL_VERSION=${NCCL_VERSION}" \
    -e HF_HOME=/tmp/hf \
    -e PYTHONUNBUFFERED=1 \
    -e TRL_EXPERIMENTAL_SILENCE=1 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    ` # Prefetch depth for the sharded checkpoint load, which reads 61 GB of weights straight into their shards. ` \
    -e HF_SHARD_PREFETCH=4 \
    -e NCCL_IB_DISABLE=1 \
    -e NCCL_SOCKET_IFNAME=eth0 \
    -- "vllm/vllm-openai:${VLLM_TAG}" bash -c '
set -euo pipefail
env | grep HF_NETWORK; hostname -i

# Codeload tarballs rather than git+https: the vLLM image ships no git. GitHub rate-limits datacenter IPs, so retry.
for attempt in 1 2 3 4 5; do
    pip install -q "https://codeload.github.com/huggingface/trl/tar.gz/${TRL_REF}" \
        "https://codeload.github.com/huggingface/transformers/tar.gz/${TRANSFORMERS_REF}" \
        "https://codeload.github.com/huggingface/OpenEnv/tar.gz/main" \
        "https://codeload.github.com/SWE-Gym/SWE-Bench-Package/tar.gz/main" \
        peft "kernels>=0.16,<0.17" trackio mini-swe-agent openai "huggingface_hub>=1.22" && break
    echo "pip install failed (attempt $attempt), retrying in 30s"; sleep 30
done
pip install -q "nvidia-nccl-cu13==${NCCL_VERSION}" || pip install -q "nvidia-nccl-cu12==${NCCL_VERSION}"
python3 -c "import torch, transformers; print(\"torch\", torch.__version__, \"nccl\", torch.cuda.nccl.version(), \"transformers\", transformers.__version__)"

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
echo "=== trainer: expert-parallel on $NPROC rank(s), NCCL master $VLLM_HOST_IP ==="
torchrun --nnodes 1 --nproc_per_node "$NPROC" /work/async_grpo_mini_swe_agent_coder.py \
    --model "$MODEL" --vllm-url "http://$V:8000" --output-dir "/tmp/${RUN_TAG}" \
    --project "$PROJECT" --run-name "$RUN_TAG" --trackio-space-id "$PROJECT" \
    $TRAIN_ARGS
' 2>&1 | grep -vE 'Warning|warnings\.warn|Request ID' | grep -oE '[0-9a-f]{24}' | tail -1
