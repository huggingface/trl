#!/usr/bin/env bash
# The trainer and the inference server as two Hugging Face Jobs, so generation gets a whole node.
#
#   ./run_split.sh                                  # 8 serving GPUs, 2 training GPUs
#   SERVE_FLAVOR=a100x8 TRAIN_FLAVOR=h200x2 ./run_split.sh
#
# Why this exists next to `run_hfjob.sh`: in one job the two share a node, and Code measures the trainer busy
# 25% of the step while generation runs at 177 tok/s. Splitting them lets the server have every GPU of its own
# node and the trainer a smaller one, instead of carving one node into two halves that fit neither.
#
# They find each other through a network group: members resolve `${HF_NETWORK_GROUP_PREFIX}<alias>` by DNS. The
# trainer advertises its own address for the NCCL weight-transfer group and the server dials back, so the two
# have to be in the same group for the sync to connect at all.
set -euo pipefail

cd "$(dirname "$0")"
EXAMPLE_DIR=$(cd ../.. && pwd)

SERVE_FLAVOR=${SERVE_FLAVOR:-a100x8}
TRAIN_FLAVOR=${TRAIN_FLAVOR:-a100x8}
VLLM_GPUS=${VLLM_GPUS:-${SERVE_FLAVOR##*x}}
TRAIN_GPUS=${TRAIN_GPUS:-2}
VLLM_TP=${VLLM_TP:-1}
VLLM_DP=${VLLM_DP:-$VLLM_GPUS}

TRL_SHA=${TRL_SHA:-$(git -C "$EXAMPLE_DIR" rev-parse HEAD)}
VLLM_TAG=${VLLM_TAG:-v0.27.1}
MIMOAGENT_SHA=${MIMOAGENT_SHA:-467f0a19016f0ac4d63b8d17a1f0da9ba07f232c}
OPENENV_SHA=${OPENENV_SHA:-49aa302ba5c6}

MODEL=${MODEL:-Qwen/Qwen3-8B}
DOMAINS=${DOMAINS:-code}
PACKING=${PACKING:-tree}
LORA_RANK=${LORA_RANK:-32}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-98304}
HF_OVERRIDES=${HF_OVERRIDES:-'{"rope_scaling":{"rope_type":"yarn","factor":3.0,"original_max_position_embeddings":32768}}'}
TOOL_PARSER=${TOOL_PARSER:-hermes}
REASONING_PARSER=${REASONING_PARSER:-qwen3}
PROJECT=${PROJECT:-async-grpo-mimo-abl}
RUN_NAME=${RUN_NAME:-split-$DOMAINS-$(date +%m%d-%H%M)}
TIMEOUT=${TIMEOUT:-12h}
OUT_BUCKET=${OUT_BUCKET:-aminediroHF/mimo-rl-adapters}
# Unique per launch: an alias is claimed for the lifetime of the group, so a retry must not collide with a job
# that is still shutting down.
GROUP=${GROUP:-rl-$RUN_NAME}
TRAIN_ARGS=${TRAIN_ARGS:---max-turn-tokens 16384 --max-observation-length 40000 --token-budget 65536 --num-generations 16 --max-staleness 4 --max-steps 1000 --reward score --max-inflight 192 --verify-timeout 900 --save-steps 10 --save-total-limit 3 --learning-rate 1e-6 --weight-decay 0.01 --n-prompts 96 --gradient-accumulation-steps 12}

curl -fsI "https://codeload.github.com/huggingface/trl/tar.gz/$TRL_SHA" >/dev/null \
    || { echo "TRL_SHA $TRL_SHA is not on GitHub -- push the branch first"; exit 1; }

PIP_INSTALL='pip install "https://github.com/huggingface/trl/archive/${TRL_SHA}.tar.gz" \
    "kernels>=0.16,<0.17" trackio "huggingface_hub>=1.31" pandas pyarrow openai peft flash-linear-attention \
    "https://github.com/XiaomiMiMo/mimoagent/archive/${MIMOAGENT_SHA}.tar.gz" \
    "https://github.com/huggingface/OpenEnv/archive/${OPENENV_SHA}.tar.gz"'

echo "=== group $GROUP"
echo "=== serve  $SERVE_FLAVOR | $VLLM_GPUS GPUs (tp=$VLLM_TP dp=$VLLM_DP) | $MODEL | len=$MAX_MODEL_LEN"
echo "=== train  $TRAIN_FLAVOR | $TRAIN_GPUS GPUs | $DOMAINS | $PACKING | lora=$LORA_RANK"
echo "=== trl @ $TRL_SHA -> trackio project '$PROJECT', run '$RUN_NAME'"

uvx hf jobs run --name "$RUN_NAME-vllm" \
    --flavor "$SERVE_FLAVOR" --timeout "$TIMEOUT" --detach --secrets HF_TOKEN \
    --network-group "$GROUP" --network-alias vllm \
    -e "TRL_SHA=$TRL_SHA" -e "MODEL=$MODEL" -e "MAX_MODEL_LEN=$MAX_MODEL_LEN" -e "HF_OVERRIDES=$HF_OVERRIDES" \
    -e "VLLM_TP=$VLLM_TP" -e "VLLM_DP=$VLLM_DP" -e "LORA_RANK=$LORA_RANK" \
    -e "TOOL_PARSER=$TOOL_PARSER" -e "REASONING_PARSER=$REASONING_PARSER" \
    -- "vllm/vllm-openai:${VLLM_TAG}" bash -c "
set -euo pipefail
export HF_HOME=/tmp/hf PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OVERRIDE_ARGS=(); [ -n \"\$HF_OVERRIDES\" ] && OVERRIDE_ARGS=(--hf-overrides \"\$HF_OVERRIDES\")
LORA_ARGS=\"\"; LORA_ENV=()
if [ \"\$LORA_RANK\" -gt 0 ] && [ \"\$VLLM_DP\" -eq 1 ]; then
    LORA_ARGS=\"--enable-lora --max-lora-rank \$LORA_RANK --max-loras 12 --max-cpu-loras 16\"
    LORA_ENV=(VLLM_ALLOW_RUNTIME_LORA_UPDATING=1)
fi
# 0.0.0.0, not localhost: the trainer reaches this server from another job in the network group.
VLLM_SERVER_DEV_MODE=1 env \"\${LORA_ENV[@]}\" vllm serve \"\$MODEL\" \\
    --host 0.0.0.0 --port 8000 --dtype bfloat16 --max-model-len \"\$MAX_MODEL_LEN\" \"\${OVERRIDE_ARGS[@]}\" \\
    --gpu-memory-utilization 0.85 --tensor-parallel-size \"\$VLLM_TP\" --data-parallel-size \"\$VLLM_DP\" \\
    --logprobs-mode processed_logprobs --generation-config vllm \\
    --enable-prefix-caching \\
    --weight-transfer-config '{\"backend\":\"nccl\"}' \$LORA_ARGS \\
    --enable-auto-tool-choice --tool-call-parser \"\$TOOL_PARSER\" --reasoning-parser \"\$REASONING_PARSER\"
"

uvx hf jobs run --name "$RUN_NAME-train" \
    --flavor "$TRAIN_FLAVOR" --timeout "$TIMEOUT" --detach --secrets HF_TOKEN \
    --network-group "$GROUP" --network-alias trainer \
    -v "$EXAMPLE_DIR:/work" -v "hf://buckets/${OUT_BUCKET}:/out:rw" \
    -e "TRL_SHA=$TRL_SHA" -e "MODEL=$MODEL" -e "DOMAINS=$DOMAINS" -e "PACKING=$PACKING" \
    -e "LORA_RANK=$LORA_RANK" -e "MAX_MODEL_LEN=$MAX_MODEL_LEN" -e "TRAIN_GPUS=$TRAIN_GPUS" \
    -e "PROJECT=$PROJECT" -e "RUN_NAME=$RUN_NAME" -e "TRAIN_ARGS=$TRAIN_ARGS" \
    -e "MIMOAGENT_SHA=$MIMOAGENT_SHA" -e "OPENENV_SHA=$OPENENV_SHA" \
    -- "vllm/vllm-openai:${VLLM_TAG}" bash -c "
set -euo pipefail
export HF_HOME=/tmp/hf PYTHONUNBUFFERED=1 TRL_EXPERIMENTAL_SILENCE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
$PIP_INSTALL
python3 - <<\"PYCHECK\"
import datasets, huggingface_hub, mimoagent, openai, openenv, pandas, peft, transformers, trl
from mimoagent.agents.cc import CCAgent
from openenv.core.harness import ResourceSession
from trl.experimental.async_grpo import AsyncGRPOConfig
print('deps ok:', trl.__version__, transformers.__version__, peft.__version__)
PYCHECK

VLLM_HOST=\${HF_NETWORK_GROUP_PREFIX}vllm
echo \"waiting for \$VLLM_HOST to resolve\"
until getent hosts \"\$VLLM_HOST\" >/dev/null; do sleep 5; done
echo \"waiting for \$VLLM_HOST:8000 to serve\"
until curl -sf \"http://\$VLLM_HOST:8000/health\" >/dev/null; do sleep 10; done
echo \"vLLM ready at \$VLLM_HOST:8000\"

# The trainer opens the NCCL weight-transfer group on an address it advertises to the server, and the address it
# picks by itself is the container's, which the server cannot route to. Hand it the one the network group resolves.
export VLLM_HOST_IP=\$(getent hosts \"\${HF_NETWORK_GROUP_PREFIX}trainer\" | awk '{print \$1}' | head -1)
echo \"advertising \$VLLM_HOST_IP for the weight-transfer group\"

echo \"CONFIG model=\$MODEL\"
echo \"CONFIG domains=\$DOMAINS\"
echo \"CONFIG packing=\$PACKING\"
echo \"CONFIG lora_rank=\$LORA_RANK\"
echo \"CONFIG max_model_len=\$MAX_MODEL_LEN\"
echo \"CONFIG scheduler=hf-jobs-split train_gpus=\$TRAIN_GPUS\"
echo \"CONFIG train_args=\$TRAIN_ARGS\"

accelerate launch --num_machines 1 --num_processes \"\$TRAIN_GPUS\" --mixed_precision bf16 \\
    --use_fsdp --fsdp_version 2 --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \\
    --fsdp_state_dict_type SHARDED_STATE_DICT --fsdp_cpu_ram_efficient_loading true \\
    /work/async_grpo_mimo.py \\
        --model \"\$MODEL\" --domains \"\$DOMAINS\" --packing \"\$PACKING\" --lora-rank \"\$LORA_RANK\" \\
        --gradient-checkpointing --vllm-url \"http://\$VLLM_HOST:8000\" \\
        --output-dir \"/out/\$RUN_NAME\" --project \"\$PROJECT\" --run-name \"\$RUN_NAME\" --trackio-space-id \"\$PROJECT\" \\
        \$TRAIN_ARGS
"
