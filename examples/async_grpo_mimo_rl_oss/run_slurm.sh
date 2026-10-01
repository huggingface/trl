#!/usr/bin/env bash
# Submit one training run per domain set, so they land in one trackio Space and their reward curves can be read
# side by side.
#
#   ./run_slurm.sh                                   # general, code, and the three mixed
#   DOMAIN_SETS=code ./run_slurm.sh                  # just the Code run
#   TRAIN_ARGS="--max-steps 50 --instances-file trainable_instances.json" ./run_slurm.sh
#
# The venv (trl + vllm + mimoagent + openenv) has to be staged first, once, from a login node, and so do the General
# task directories (~50 small files each, a second per file from hopper-atl):
#
#   venv-ship /fsx/$USER/envs/trl-agent
#   HF_HUB_CACHE=/fsx/$USER/hf-hub python -c "import general_domain; general_domain.load_tasks(1000, 0, language='en')"
set -euo pipefail

cd "$(dirname "$0")"

PARTITION=${PARTITION:-hopper-atl}
TIME=${TIME:-12:00:00}
MODEL=${MODEL:-XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B}
VLLM_GPUS=${VLLM_GPUS:-1}
TRAIN_GPUS=${TRAIN_GPUS:-2}
LORA_RANK=${LORA_RANK:-32}
# Sequence, not tree: three of every four layers of Qwen3.5 are linear-attention layers, whose recurrent state
# cannot branch, so a packed prefix forest has no mask to express. They do take `cu_seqlens`, which is what
# sequence packing needs.
PACKING=${PACKING:-sequence}
RUN_TAG=${RUN_TAG:-$(date +%m%d-%H%M)}
RUNS_DIR=${RUNS_DIR:-/fsx/$USER/runs/mimo-multi/$RUN_TAG}
PROJECT=${PROJECT:-async-grpo-mimo}
# The reconciler folds a rollout's turns into one row of its final context, so the budget has to hold
# `max_model_len + max_turn_tokens`.
TRAIN_ARGS=${TRAIN_ARGS:---max-steps 200 --num-generations 8 --max-staleness 2 --token-budget 45056}
DOMAIN_SETS=${DOMAIN_SETS:-"general code general,code,cyber"}

mkdir -p "$RUNS_DIR/logs"

for domains in $DOMAIN_SETS; do
    name=${domains//,/-}
    # Untyped `--gres=gpu:N`: `gpu:h100:N` matches nothing on hopper-atl or hopper-extra and pends forever.
    sbatch \
        --job-name="mimo-$name" \
        --partition="$PARTITION" \
        --nodes=1 \
        --gres=gpu:$((VLLM_GPUS + TRAIN_GPUS)) \
        --time="$TIME" \
        --qos=low \
        --output="$RUNS_DIR/logs/%x-%j.out" \
        --export=ALL,EXAMPLE_DIR="$PWD",LORA_RANK="$LORA_RANK",PACKING="$PACKING",MODEL="$MODEL",DOMAINS="$domains",VLLM_GPUS="$VLLM_GPUS",TRAIN_GPUS="$TRAIN_GPUS",OUTPUT_DIR="$RUNS_DIR/$name",PROJECT="$PROJECT",RUN_NAME="$RUN_TAG-$name",TRAIN_ARGS="$TRAIN_ARGS" \
        slurm_job.sh
done

echo
echo "runs:     $RUNS_DIR"
echo "trackio:  the '$PROJECT' Space under your account"
echo "watch:    squeue -u $USER;  tail -f $RUNS_DIR/logs/*.out"
