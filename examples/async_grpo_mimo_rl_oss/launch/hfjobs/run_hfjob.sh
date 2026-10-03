#!/usr/bin/env bash
# The same ablation arms as `launch/slurm`, run as a Hugging Face Job on H200s.
#
#   ./run_hfjob.sh                                  # tree + LoRA on the General domain
#   FLAVOR=h200x8 TRAIN_GPUS=6 VLLM_GPUS=2 ./run_hfjob.sh
#   PACKING=sequence LORA_RANK=0 ./run_hfjob.sh     # a full fine-tune, if the flavor has the memory
#
# Why this exists next to the Slurm launcher: `hopper-atl` is a Capacity Block that ends on 2026-10-15 and is
# usually full, so an arm that only needs one node runs here instead of waiting for one. An H200 carries 141 GB
# against the H100's 80, which is what makes an 8B adapter run comfortable on a single machine.
#
# Both launchers report to the same trackio project, so the arms land in one Space however they were scheduled.
#
# The rollouts still start Hugging Face sandboxes over the internet and the General judge still runs on the
# router, so the only thing that changes is where the trainer and the server live.
set -euo pipefail

cd "$(dirname "$0")"
EXAMPLE_DIR=$(cd ../.. && pwd)

FLAVOR=${FLAVOR:-h200x4}
NGPU=${NGPU:-${FLAVOR##*x}}; [ "$NGPU" = "$FLAVOR" ] && NGPU=1
VLLM_GPUS=${VLLM_GPUS:-2}
TRAIN_GPUS=${TRAIN_GPUS:-$((NGPU - VLLM_GPUS))}
VLLM_TP=${VLLM_TP:-1}
VLLM_DP=${VLLM_DP:-$VLLM_GPUS}

# The branch this example lives on. Pinned to a commit so the job runs exactly the code under review rather than
# whatever the branch points at when it happens to start.
TRL_SHA=${TRL_SHA:-$(git -C "$EXAMPLE_DIR" rev-parse HEAD)}
VLLM_TAG=${VLLM_TAG:-v0.27.1}
# Tarballs rather than `git+https`: the vLLM image ships no git and pip's VCS installer shells out to it. The URL
# has to carry a real `.tar.gz` and the repository's own name: pip exits 0 without installing anything when the
# extension is missing, and `MiMo-Agent` is a redirect that codeload answers with a different project.
MIMOAGENT_SHA=${MIMOAGENT_SHA:-467f0a19016f0ac4d63b8d17a1f0da9ba07f232c}
OPENENV_SHA=${OPENENV_SHA:-49aa302ba5c6}
# The job installs this commit as a tarball, so it has to be on the remote: an unpushed SHA 404s inside the
# job, minutes after it was scheduled and with the flavor already paid for.
curl -fsI "https://codeload.github.com/huggingface/trl/tar.gz/$TRL_SHA" >/dev/null \
    || { echo "TRL_SHA $TRL_SHA is not on GitHub -- push the branch first"; exit 1; }


MODEL=${MODEL:-Qwen/Qwen3-8B}
DOMAINS=${DOMAINS:-general}
PACKING=${PACKING:-tree}
LORA_RANK=${LORA_RANK:-32}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-40960}
TOOL_PARSER=${TOOL_PARSER:-hermes}
REASONING_PARSER=${REASONING_PARSER:-qwen3}
# The same Space the Slurm launcher reports to, so an arm lands in one place however it was scheduled.
PROJECT=${PROJECT:-async-grpo-mimo}
RUN_NAME=${RUN_NAME:-hfjob-$PACKING-$(date +%m%d-%H%M)}
TIMEOUT=${TIMEOUT:-8h}
# Checkpoints and adapters go to a bucket: a job's own filesystem is gone the moment it ends, so anything written
# to /tmp is lost even on a clean finish.
OUT_BUCKET=${OUT_BUCKET:-aminediroHF/mimo-rl-adapters}
TRAIN_ARGS=${TRAIN_ARGS:---max-turn-tokens 4096 --token-budget 45056 --num-generations 16 --max-staleness 4 --max-steps 1000 --reward score --max-inflight 128 --verify-timeout 900 --save-steps 25 --save-total-limit 3 --learning-rate 1e-5 --n-prompts 96 --gradient-accumulation-steps 1 --instances-file /work/trainable_instances.json}

echo "=== $FLAVOR | $MODEL | $PACKING | lora=$LORA_RANK | vLLM $VLLM_GPUS (tp=$VLLM_TP dp=$VLLM_DP) | trainer $TRAIN_GPUS"
echo "=== trl @ $TRL_SHA -> trackio project '$PROJECT', run '$RUN_NAME'"

uvx hf jobs run --name "$RUN_NAME" \
    --flavor "$FLAVOR" --timeout "$TIMEOUT" --detach --secrets HF_TOKEN \
    -v "$EXAMPLE_DIR:/work" \
    -v "hf://buckets/${OUT_BUCKET}:/out:rw" \
    -e "TRL_SHA=$TRL_SHA" -e "MODEL=$MODEL" -e "DOMAINS=$DOMAINS" -e "PACKING=$PACKING" -e "FLAVOR=$FLAVOR" \
    -e "LORA_RANK=$LORA_RANK" -e "MAX_MODEL_LEN=$MAX_MODEL_LEN" \
    -e "TOOL_PARSER=$TOOL_PARSER" -e "REASONING_PARSER=$REASONING_PARSER" \
    -e "VLLM_GPUS=$VLLM_GPUS" -e "TRAIN_GPUS=$TRAIN_GPUS" -e "VLLM_TP=$VLLM_TP" -e "VLLM_DP=$VLLM_DP" \
    -e "PROJECT=$PROJECT" -e "RUN_NAME=$RUN_NAME" -e "TRAIN_ARGS=$TRAIN_ARGS" \
    -e "MIMOAGENT_SHA=$MIMOAGENT_SHA" -e "OPENENV_SHA=$OPENENV_SHA" \
    -- "vllm/vllm-openai:${VLLM_TAG}" bash -c '
set -euo pipefail
export HF_HOME=/tmp/hf PYTHONUNBUFFERED=1 TRL_EXPERIMENTAL_SILENCE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# GitHub rate-limits datacenter IPs, so this may need a retry.
pip install "https://github.com/huggingface/trl/archive/${TRL_SHA}.tar.gz" \
    "kernels>=0.16,<0.17" trackio "huggingface_hub>=1.31" pandas pyarrow openai peft flash-linear-attention \
    "https://github.com/XiaomiMiMo/mimoagent/archive/${MIMOAGENT_SHA}.tar.gz" \
    "https://github.com/huggingface/OpenEnv/archive/${OPENENV_SHA}.tar.gz"
# pip can report success while installing nothing, so check what the trainer actually needs before paying for a
# vLLM start. A heredoc, not `python -c`: the job body is single-quoted, so a quote in here would end it.
python3 - <<"PYCHECK"
# Every third-party module the trainer imports, so a missing one costs seconds here rather than a model load.
import datasets, huggingface_hub, mimoagent, openai, openenv, pandas, peft, transformers, trl
from mimoagent.agents.cc import CCAgent
from mimoagent.environments.datasets import ARVOEnvironment, OpenSourceCodeEnvironment
from openenv.core.harness import ResourceSession
from peft import LoraConfig
from trl.experimental.async_grpo import AsyncGRPOConfig
print("deps ok:", trl.__version__, transformers.__version__, peft.__version__)
PYCHECK

SERVE_IDS=$(seq -s, "$TRAIN_GPUS" $((TRAIN_GPUS + VLLM_GPUS - 1)))
TRAIN_IDS=$(seq -s, 0 $((TRAIN_GPUS - 1)))
echo "serve gpus=$SERVE_IDS  train gpus=$TRAIN_IDS"
echo "CONFIG model=$MODEL"
echo "CONFIG domains=$DOMAINS"
echo "CONFIG packing=$PACKING"
echo "CONFIG lora_rank=$LORA_RANK"
echo "CONFIG max_model_len=$MAX_MODEL_LEN"
echo "CONFIG scheduler=hf-jobs flavor=$FLAVOR"
echo "CONFIG vllm_gpus=$VLLM_GPUS vllm_tp=$VLLM_TP vllm_dp=$VLLM_DP"
echo "CONFIG train_gpus=$TRAIN_GPUS"
echo "CONFIG parsers=$TOOL_PARSER/$REASONING_PARSER"
echo "CONFIG train_args=$TRAIN_ARGS"

# The adapter fast path needs a flag vLLM refuses once data parallelism gives the server several API processes;
# with replicas the trainer merges and streams the full weights instead, which NCCL does in about a second.
LORA_ARGS=""; LORA_ENV=()
if [ "$LORA_RANK" -gt 0 ] && [ "$VLLM_DP" -eq 1 ]; then
    LORA_ARGS="--enable-lora --max-lora-rank $LORA_RANK --max-loras 12 --max-cpu-loras 16"
    LORA_ENV=(VLLM_ALLOW_RUNTIME_LORA_UPDATING=1)
fi

CUDA_VISIBLE_DEVICES="$SERVE_IDS" VLLM_SERVER_DEV_MODE=1 \
    env "${LORA_ENV[@]}" vllm serve "$MODEL" \
        --port 8000 --dtype bfloat16 --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization 0.85 \
        --tensor-parallel-size "$VLLM_TP" --data-parallel-size "$VLLM_DP" \
        --logprobs-mode processed_logprobs --generation-config vllm \
        --weight-transfer-config "{\"backend\":\"nccl\"}" \
        $LORA_ARGS \
        --enable-auto-tool-choice --tool-call-parser "$TOOL_PARSER" --reasoning-parser "$REASONING_PARSER" \
        > /tmp/vllm.log 2>&1 &
VLLM_PID=$!
until curl -sf localhost:8000/health > /dev/null; do
    kill -0 $VLLM_PID 2>/dev/null || { echo "vLLM died during startup:"; tail -80 /tmp/vllm.log; exit 1; }
    sleep 5
done
echo "vLLM ready"

CUDA_VISIBLE_DEVICES="$TRAIN_IDS" accelerate launch \
    --num_machines 1 --num_processes "$TRAIN_GPUS" --mixed_precision bf16 \
    --use_fsdp --fsdp_version 2 --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
    --fsdp_state_dict_type SHARDED_STATE_DICT --fsdp_cpu_ram_efficient_loading true \
    /work/async_grpo_mimo.py \
        --model "$MODEL" --domains "$DOMAINS" --packing "$PACKING" --lora-rank "$LORA_RANK" \
        --gradient-checkpointing --vllm-url http://localhost:8000 \
        --output-dir "/out/$RUN_NAME" --project "$PROJECT" --run-name "$RUN_NAME" --trackio-space-id "$PROJECT" \
        $TRAIN_ARGS || {
    echo "=== trainer failed; last 120 lines of the vLLM server log ==="
    tail -120 /tmp/vllm.log
    exit 1
}
'
