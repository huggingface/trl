#!/usr/bin/env bash
# Cancel every Job of the last `run_all.sh`. The vLLM Jobs serve until cancelled or until their timeout, so this is
# what stops them billing once the trainer is done or has died.
set -euo pipefail
cd "$(dirname "$0")"
[ -f .last_run ] || { echo "no .last_run here"; exit 1; }
# shellcheck disable=SC1091
source .last_run
# shellcheck disable=SC2086
for id in "$TRAIN_ID" $VLLM_IDS; do
    echo "cancelling $id"
    uvx hf jobs cancel "$id" 2>/dev/null || true
done
