#!/usr/bin/env bash
# Score a fixed set of SWE-Gym instances against the vLLM Job of a network group, from inside that group.
#
#   GROUP=coder-ep-2 TAG=before ./run_eval_job_netgroup.sh
#
# Why a Job rather than a local process: the group's server has no exposed port, so nothing outside the group can
# reach it. Why the same server the trainer uses: it serves whatever the policy currently is, so running this once
# before training and once after gives two numbers on the same problems, the same harness and the same sampler,
# which is the only comparison that isolates the policy. The instance list is fixed for the same reason.
set -euo pipefail

cd "$(dirname "$0")"

HF=(uvx --from "huggingface_hub @ git+https://github.com/huggingface/huggingface_hub.git" hf)
GROUP=${GROUP:?set GROUP to the network group name}
TAG=${TAG:-eval}
MODEL=${MODEL:-Qwen/Qwen3-Coder-30B-A3B-Instruct}
TRL_REF=${TRL_REF:-agent-rl-example}
VLLM_TAG=${VLLM_TAG:-v0.27.1}
FLAVOR=${EVAL_FLAVOR:-h200}
TIMEOUT=${EVAL_TIMEOUT:-2h}
N_INSTANCES=${N_INSTANCES:-48}
CONCURRENCY=${CONCURRENCY:-48}
SEED=${SEED:-123}

"${HF[@]}" jobs run \
    --name "async-grpo-mini-swe-agent-eval-${GROUP}-${TAG}" --flavor "$FLAVOR" --timeout "$TIMEOUT" --detach --secrets HF_TOKEN \
    --network-group "$GROUP" --network-alias "eval-${TAG}" \
    -v "$PWD:/work" \
    -e "MODEL=${MODEL}" -e "TRL_REF=${TRL_REF}" -e "TAG=${TAG}" \
    -e "N_INSTANCES=${N_INSTANCES}" -e "CONCURRENCY=${CONCURRENCY}" -e "SEED=${SEED}" \
    -e HF_HOME=/tmp/hf -e PYTHONUNBUFFERED=1 -e TRL_EXPERIMENTAL_SILENCE=1 \
    -- "vllm/vllm-openai:${VLLM_TAG}" bash -c '
set -euo pipefail
for attempt in 1 2 3 4 5; do
    pip install -q "https://codeload.github.com/huggingface/trl/tar.gz/${TRL_REF}" \
        "https://codeload.github.com/huggingface/OpenEnv/tar.gz/main" \
        "https://codeload.github.com/SWE-Gym/SWE-Bench-Package/tar.gz/main" \
        peft datasets mini-swe-agent openai "huggingface_hub>=1.22" && break
    echo "pip install failed (attempt $attempt), retrying in 30s"; sleep 30
done

V="${HF_NETWORK_GROUP_PREFIX}vllm"
for i in $(seq 1 360); do
    curl -sf "http://$V:8000/health" > /dev/null && { echo "vLLM reachable after $((i * 5))s"; break; }
    sleep 5
done
curl -sf "http://$V:8000/health" > /dev/null || { echo "!!! vLLM unreachable at $V"; exit 1; }

echo "=== eval $TAG: $N_INSTANCES instances, seed $SEED, against $V ==="
python3 /work/eval_mini_swe_agent.py \
    --model "$MODEL" --vllm-url "http://$V:8000" \
    --n-instances "$N_INSTANCES" --samples-per-instance 2 --max-inflight "$CONCURRENCY" --seed "$SEED" \
    --max-turn-tokens 4096 --step-limit 250 --step-timeout 300 --agent-timeout 3600 --eval-timeout 900 \
    --output "/tmp/eval-${TAG}.jsonl"
echo "=== eval $TAG done ==="
' 2>&1 | tee /tmp/lastjob.raw | grep -vE 'Warning|warnings\.warn|Request ID' | grep -oE '[0-9a-f]{24}' | tail -1
