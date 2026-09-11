#!/usr/bin/env bash
# Start the trainer Job: FSDP2 across the flavor's GPUs, the shared bucket mounted read-write at /lora, rollouts in
# Hugging Face sandboxes, generation served by the vLLM Jobs whose IDs are the arguments. Prints the Job ID on stdout.
#
#   ./run_trainer_job.sh <vllm_job_id> [<vllm_job_id> ...]
#
# The trainer never talks to `https://<id>--8000.hf.jobs` directly: that URL needs a bearer token on every request
# and nothing in the AsyncGRPO stack adds one. So `lora_proxy.py` runs alongside the trainer: it adds the header,
# routes each rollout to the replica most likely to hold its KV prefix, broadcasts adapter loads to every replica,
# and answers `/health` only when all of them do, while `vllm_server_base_url` stays `http://localhost:8000`.
set -euo pipefail

cd "$(dirname "$0")"

[ $# -ge 1 ] || { echo "usage: run_trainer_job.sh <vllm_job_id> [<vllm_job_id> ...]" >&2; exit 1; }
UPSTREAM_URLS=$(for id in "$@"; do printf 'https://%s--8000.hf.jobs,' "$id"; done)
UPSTREAM_URLS=${UPSTREAM_URLS%,}
BUCKET=${BUCKET:-aminediroHF/async-grpo-mini-swe-agent}
MODEL=${MODEL:-Qwen/Qwen3-32B}
TRL_REF=${TRL_REF:-main}
VLLM_TAG=${VLLM_TAG:-v0.27.1}
FLAVOR=${TRAIN_FLAVOR:-h200x4}
TIMEOUT=${TRAIN_TIMEOUT:-8h}
RUN_TAG=${RUN_TAG:-r32}
PROJECT=${PROJECT:-async-grpo-mini-swe-agent}
# Every knob of the training script, forwarded as is. Both Jobs mount the bucket at /lora, so `--output-dir` resolves
# to the same adapter directory on the server.
TRAIN_ARGS=${TRAIN_ARGS:-}

uvx hf jobs run \
    --name async-grpo-mini-swe-agent-train --flavor "$FLAVOR" --timeout "$TIMEOUT" --detach --secrets HF_TOKEN \
    -v "hf://buckets/${BUCKET}:/lora" \
    -v "$PWD:/work" \
    -e "UPSTREAM_URLS=${UPSTREAM_URLS}" \
    ` # A replica may hold this many more in-flight rollouts than the least-loaded one before prefix affinity yields. ` \
    -e "PROXY_IMBALANCE=${PROXY_IMBALANCE:-8}" \
    -e "MODEL=${MODEL}" \
    -e "TRL_REF=${TRL_REF}" \
    -e "OUTPUT_DIR=/lora/${RUN_TAG}" \
    -e "RUN_TAG=${RUN_TAG}" \
    -e "PROJECT=${PROJECT}" \
    -e "TRAIN_ARGS=${TRAIN_ARGS}" \
    -e HF_HOME=/tmp/hf \
    -e PYTHONUNBUFFERED=1 \
    -e TRL_EXPERIMENTAL_SILENCE=1 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -- "vllm/vllm-openai:${VLLM_TAG}" bash -c '
set -euo pipefail
grep -q /lora /proc/mounts || { echo "no /lora mount!"; exit 1; }

# Codeload tarballs rather than git+https: the vLLM image ships no git. GitHub rate-limits datacenter IPs, so retry.
for attempt in 1 2 3 4 5; do
    pip install -q "https://codeload.github.com/huggingface/trl/tar.gz/${TRL_REF}" \
        "https://codeload.github.com/huggingface/OpenEnv/tar.gz/main" \
        "https://codeload.github.com/SWE-Gym/SWE-Bench-Package/tar.gz/main" \
        peft trackio mini-swe-agent openai "huggingface_hub>=1.22" && break
    echo "pip install failed (attempt $attempt), retrying in 30s"; sleep 30
done

# Its stdout goes to the Job log, prefixed, so routing stats and adapter broadcasts are visible from outside.
python3 /work/lora_proxy.py > >(sed -u "s/^/[lora_proxy] /") 2>&1 &
sleep 3
echo "=== waiting for every vLLM replica through the proxy (up to 900s) ==="
for i in $(seq 1 180); do
    curl -sf http://localhost:8000/health > /dev/null && { echo "vLLM reachable after $((i * 5))s"; break; }
    sleep 5
done
curl -sf http://localhost:8000/health > /dev/null || { echo "!!! vLLM unreachable through the proxy"; exit 1; }

# Replicas reused across runs still hold the previous run'"'"'s `trl-policy-v*` adapters, and the trainer numbers them
# from v1 every run, so a leftover is a name collision. Unload them through the proxy, which broadcasts.
for u in ${UPSTREAM_URLS//,/ }; do
    curl -s -H "Authorization: Bearer $HF_TOKEN" "$u/v1/models" \
        | python3 -c "import sys, json; [print(m[\"id\"]) for m in json.load(sys.stdin).get(\"data\", []) if m[\"id\"].startswith(\"trl-policy-\")]" || true
done | sort -u | while read -r name; do
    [ -n "$name" ] || continue
    echo "unloading stale adapter $name"
    curl -s -X POST -H "Content-Type: application/json" -d "{\"lora_name\": \"$name\"}" http://localhost:8000/v1/unload_lora_adapter > /dev/null || true
done

NPROC=$(nvidia-smi -L | wc -l)
echo "=== trainer: FSDP2 on $NPROC rank(s) ==="
accelerate launch --config_file /work/fsdp2.yaml --num_processes "$NPROC" /work/async_grpo_mini_swe_agent.py \
    --model "$MODEL" --output-dir "$OUTPUT_DIR" --project "$PROJECT" --run-name "$RUN_TAG" --trackio-space-id "$PROJECT" \
    $TRAIN_ARGS
' 2>&1 | grep -oE '[0-9a-f]{24}' | head -1
