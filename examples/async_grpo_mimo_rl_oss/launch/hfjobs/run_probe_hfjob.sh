#!/usr/bin/env bash
# Probe a policy on one domain as a Hugging Face Job. No training, so one GPU serves and the rollouts run beside it.
#
#   DOMAIN=code ./run_probe_hfjob.sh
#   DOMAIN=cyber MODEL=XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B TOOL_PARSER=mimo REASONING_PARSER=mimo ./run_probe_hfjob.sh
#
# The result lands in the output bucket as a jsonl, so the distribution survives the job.
set -euo pipefail
cd "$(dirname "$0")"
EXAMPLE_DIR=$(cd ../.. && pwd)

FLAVOR=${FLAVOR:-h200}
# A model too large for one card is sharded rather than replicated: a probe has one server.
VLLM_TP=${VLLM_TP:-1}
# `bfloat16` is wrong for a checkpoint that ships quantized; `auto` takes what the config says.
VLLM_DTYPE=${VLLM_DTYPE:-bfloat16}
DOMAIN=${DOMAIN:-code}
MODEL=${MODEL:-Qwen/Qwen3-8B}
TOOL_PARSER=${TOOL_PARSER:-hermes}
REASONING_PARSER=${REASONING_PARSER:-qwen3}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-40960}
# Qwen3 is trained to 32768 and reaches further only with YaRN, which vLLM takes as an entry in
# `--hf-overrides` rather than a flag of its own. Code rollouts sit at a p50 of 58k, so a window below
# that ends half of them in a 400 rather than an answer.
HF_OVERRIDES=${HF_OVERRIDES:-}
N_TASKS=${N_TASKS:-16}
SAMPLES=${SAMPLES:-4}
MAX_INFLIGHT=${MAX_INFLIGHT:-48}
TIMEOUT=${TIMEOUT:-4h}
OUT_BUCKET=${OUT_BUCKET:-aminediroHF/mimo-rl-adapters}
PROBE_ARGS=${PROBE_ARGS:-}
RUN_NAME=${RUN_NAME:-probe-$DOMAIN-$(date +%m%d-%H%M)}
TRL_SHA=${TRL_SHA:-$(git -C "$EXAMPLE_DIR" rev-parse HEAD)}
MIMOAGENT_SHA=${MIMOAGENT_SHA:-467f0a19016f0ac4d63b8d17a1f0da9ba07f232c}
OPENENV_SHA=${OPENENV_SHA:-49aa302ba5c6}
# The job installs this commit as a tarball, so it has to be on the remote: an unpushed SHA 404s inside the
# job, minutes after it was scheduled and with the flavor already paid for.
curl -fsI "https://codeload.github.com/huggingface/trl/tar.gz/$TRL_SHA" >/dev/null \
    || { echo "TRL_SHA $TRL_SHA is not on GitHub -- push the branch first"; exit 1; }

VLLM_TAG=${VLLM_TAG:-v0.27.1}

echo "=== probe $DOMAIN | $MODEL | $FLAVOR | $N_TASKS tasks x $SAMPLES samples"

uvx hf jobs run --name "$RUN_NAME" \
    --flavor "$FLAVOR" --timeout "$TIMEOUT" --detach --secrets HF_TOKEN \
    -v "$EXAMPLE_DIR:/work" -v "hf://buckets/${OUT_BUCKET}:/out:rw" \
    -e "TRL_SHA=$TRL_SHA" -e "MIMOAGENT_SHA=$MIMOAGENT_SHA" -e "OPENENV_SHA=$OPENENV_SHA" \
    -e "VLLM_TP=$VLLM_TP" -e "VLLM_DTYPE=$VLLM_DTYPE" -e "DOMAIN=$DOMAIN" -e "MODEL=$MODEL" -e "MAX_MODEL_LEN=$MAX_MODEL_LEN" -e "HF_OVERRIDES=$HF_OVERRIDES" \
    -e "TOOL_PARSER=$TOOL_PARSER" -e "REASONING_PARSER=$REASONING_PARSER" \
    -e "N_TASKS=$N_TASKS" -e "SAMPLES=$SAMPLES" -e "MAX_INFLIGHT=$MAX_INFLIGHT" -e "PROBE_ARGS=$PROBE_ARGS" \
    -- "vllm/vllm-openai:${VLLM_TAG}" bash -c '
set -euo pipefail
export HF_HOME=/tmp/hf PYTHONUNBUFFERED=1 TRL_EXPERIMENTAL_SILENCE=1

pip install "https://github.com/huggingface/trl/archive/${TRL_SHA}.tar.gz" \
    trackio "huggingface_hub>=1.31" pandas pyarrow openai \
    "https://github.com/XiaomiMiMo/mimoagent/archive/${MIMOAGENT_SHA}.tar.gz" \
    "https://github.com/huggingface/OpenEnv/archive/${OPENENV_SHA}.tar.gz"

python3 - <<"PYCHECK"
import mimoagent, pandas, huggingface_hub
from mimoagent.agents.cc import CCAgent
from mimoagent.environments.datasets import ARVOEnvironment, OpenSourceCodeEnvironment
print("deps ok")
PYCHECK

OVERRIDE_ARGS=()
if [ -n "$HF_OVERRIDES" ]; then OVERRIDE_ARGS=(--hf-overrides "$HF_OVERRIDES"); fi
vllm serve "$MODEL" --port 8000 --dtype "$VLLM_DTYPE" --tensor-parallel-size "$VLLM_TP" \
    --max-model-len "$MAX_MODEL_LEN" "${OVERRIDE_ARGS[@]}" \
    --gpu-memory-utilization 0.85 --generation-config vllm \
    --enable-auto-tool-choice --tool-call-parser "$TOOL_PARSER" --reasoning-parser "$REASONING_PARSER" \
    > /tmp/vllm.log 2>&1 &
VLLM_PID=$!
until curl -sf localhost:8000/health > /dev/null; do
    kill -0 $VLLM_PID 2>/dev/null || { echo "vLLM died:"; tail -60 /tmp/vllm.log; exit 1; }
    sleep 5
done
echo "vLLM ready"

OUT=/out/probe-${DOMAIN}-$(date +%m%d-%H%M).jsonl
python3 /work/probe.py --domain "$DOMAIN" --model "$MODEL" --base-url http://localhost:8000/v1 \
    --n-tasks "$N_TASKS" --samples-per-task "$SAMPLES" --max-inflight "$MAX_INFLIGHT" \
    --output "$OUT" $PROBE_ARGS
echo "=== probe written to $OUT"
'
