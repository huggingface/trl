#!/usr/bin/env bash
# Both Jobs of the network-group setup. The vLLM Job has no exposed port, so the trainer Job waits for it itself.
#
#   ./run_all_netgroup.sh            # launch both and return; Job IDs land in .last_run_netgroup
set -euo pipefail

cd "$(dirname "$0")"

export GROUP=${GROUP:-mswe-$(date +%m%d%H%M)}
echo "=== network group $GROUP ==="
VLLM_ID=$(./run_vllm_job_netgroup.sh)
[ -n "$VLLM_ID" ] || { echo "!!! could not start the vLLM Job"; exit 1; }
echo "vllm job: $VLLM_ID"
TRAIN_ID=$(./run_trainer_job_netgroup.sh)
[ -n "$TRAIN_ID" ] || { echo "!!! could not start the trainer Job"; uvx hf jobs cancel "$VLLM_ID"; exit 1; }
echo "train job: $TRAIN_ID"
printf 'VLLM_IDS="%s"\nTRAIN_ID=%s\n' "$VLLM_ID" "$TRAIN_ID" > .last_run_netgroup
echo "trackio  https://huggingface.co/spaces/$(uvx hf auth whoami 2>/dev/null | grep -oE 'user=[^ ]+' | cut -d= -f2)/${PROJECT:-async-grpo-mini-swe-agent}"
echo "logs     uvx hf jobs logs -f $TRAIN_ID"
