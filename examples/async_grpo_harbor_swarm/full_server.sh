#!/bin/bash
set -euo pipefail
root=${SWARM_RUN_DIR:?}
model=${SWARM_MODEL_SNAPSHOT:?}
local_root=$(mktemp -d /tmp/trl-swarm-full-vllm.XXXXXX)
mkdir -p "$local_root/model" "$local_root/cache"
cp -aL "$model/." "$local_root/model/"
port=$((20000 + SLURM_JOB_ID % 10000))
printf 'http://%s:%s\n' "$(hostname -I | awk '{print $1}')" "$port" > "$root/server-url"
export VLLM_SERVER_DEV_MODE=1 VLLM_CACHE_ROOT="$local_root/cache" HF_HUB_OFFLINE=1
exec vllm serve "$local_root/model" --served-model-name Qwen/Qwen3.5-4B --host 0.0.0.0 --port "$port" --tensor-parallel-size 8 --dtype bfloat16 --max-model-len 16384 --max-num-seqs 256 --gpu-memory-utilization 0.80 --enable-prefix-caching --enable-auto-tool-choice --tool-call-parser qwen3_coder --language-model-only --default-chat-template-kwargs '{"enable_thinking":false}' --logprobs-mode processed_logprobs --return-tokens-as-token-ids --weight-transfer-config '{"backend":"nccl"}'
