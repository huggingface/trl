#!/usr/bin/env bash
# Both Jobs of the expert-parallel MoE setup. The vLLM Job has no exposed port, so the trainer Job waits for it
# itself; Job IDs land in `.last_run_coder`.
#
#   ./run_all_coder.sh
#
# The trainer is the one that pends: an h200x4 can take an hour to schedule, while the server is up in minutes. So
# the trainer goes first and the server follows, and the trainer's own 30-minute health wait covers the difference.
set -euo pipefail

cd "$(dirname "$0")"

export GROUP=${GROUP:-coder-$(date +%m%d%H%M)}
echo "=== network group $GROUP ==="
TRAIN_ID=$(./run_trainer_job_coder.sh)
[ -n "$TRAIN_ID" ] || { echo "!!! could not start the trainer Job"; exit 1; }
echo "train job: $TRAIN_ID"
VLLM_ID=$(./run_vllm_job_coder.sh)
[ -n "$VLLM_ID" ] || { echo "!!! could not start the vLLM Job"; uvx hf jobs cancel "$TRAIN_ID"; exit 1; }
echo "vllm job:  $VLLM_ID"
printf 'GROUP=%s\nVLLM_IDS="%s"\nTRAIN_ID=%s\n' "$GROUP" "$VLLM_ID" "$TRAIN_ID" > .last_run_coder
echo "trackio  https://huggingface.co/spaces/$(uvx hf auth whoami 2>/dev/null | grep -oE 'user=[^ ]+' | cut -d= -f2)/${PROJECT:-async-grpo-mini-swe-agent-coder}"
echo "logs     uvx hf jobs logs -f $TRAIN_ID"
echo "stop     uvx hf jobs cancel $TRAIN_ID $VLLM_ID"
