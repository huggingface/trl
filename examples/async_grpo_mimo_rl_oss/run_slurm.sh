#!/usr/bin/env bash
# Submit the same run twice on one Slurm partition, once with each packing strategy, so the two land in one trackio
# Space as two runs and `batch/packing_ratio`, `perf/fwd_bwd_s` and the reward curve can be read side by side.
#
#   ./run_slurm.sh                                   # both, Qwen3-8B, hopper-atl
#   PACKINGS=tree ./run_slurm.sh                     # just the tree run
#   TRAIN_ARGS="--max-steps 50 --instances-file trainable_instances.json" ./run_slurm.sh
#
# The venv (trl + vllm + mimoagent + openenv) has to be staged first, once, from a login node, and so do the task
# directories (~50 small files each, a second per file from hopper-atl):
#
#   venv-ship /fsx/$USER/envs/trl-agent
#   HF_HUB_CACHE=/fsx/$USER/hf-hub python -c "from probe_general import load_tasks; load_tasks(1000, 0, language='en')"
set -euo pipefail

cd "$(dirname "$0")"

PARTITION=${PARTITION:-hopper-atl}
TIME=${TIME:-12:00:00}
MODEL=${MODEL:-Qwen/Qwen3-8B}
VLLM_GPUS=${VLLM_GPUS:-1}
TRAIN_GPUS=${TRAIN_GPUS:-7}
LORA_RANK=${LORA_RANK:-0}  # 0 = full fine-tune, which is what packing should be measured on
# A full node per run: a 45k-token row of the 8B full fine-tune only fits next to the fp32 master weights, gradients
# and Adam states when those are sharded over seven ranks.
RUN_TAG=${RUN_TAG:-$(date +%m%d-%H%M)}
RUNS_DIR=${RUNS_DIR:-/fsx/$USER/runs/mimo-general-packing/$RUN_TAG}
PROJECT=${PROJECT:-async-grpo-mimo-general-packing}
# The reconciler folds a rollout's turns into one row of its final context, so the budget has to hold
# `max_model_len + max_turn_tokens` or the long rollouts are dropped.
TRAIN_ARGS=${TRAIN_ARGS:---max-steps 50 --num-generations 8 --max-staleness 2 --token-budget 45056}
PACKINGS=${PACKINGS:-"sequence tree"}

mkdir -p "$RUNS_DIR/logs"

for packing in $PACKINGS; do
    # A tree row holds tens of samples where a sequence row holds two or three, so a tree step is one row per rank:
    # both runs then update every ~70 samples and their reward curves line up per sample.
    packing_args=""
    [ "$packing" = tree ] && packing_args="--gradient-accumulation-steps 1"
    # Untyped `--gres=gpu:N`: `gpu:h100:N` matches nothing on hopper-atl or hopper-extra and pends forever.
    sbatch \
        --job-name="mimo-general-$packing" \
        --partition="$PARTITION" \
        --nodes=1 \
        --gres=gpu:$((VLLM_GPUS + TRAIN_GPUS)) \
        --time="$TIME" \
        --output="$RUNS_DIR/logs/%x-%j.out" \
        --export=ALL,EXAMPLE_DIR="$PWD",LORA_RANK="$LORA_RANK",PACKING="$packing",MODEL="$MODEL",VLLM_GPUS="$VLLM_GPUS",TRAIN_GPUS="$TRAIN_GPUS",OUTPUT_DIR="$RUNS_DIR/$packing",PROJECT="$PROJECT",RUN_NAME="$RUN_TAG-$packing",TRAIN_ARGS="$packing_args $TRAIN_ARGS" \
        slurm_job.sh
done

echo
echo "runs:     $RUNS_DIR"
echo "trackio:  the '$PROJECT' Space under your account, both runs in it"
echo "watch:    squeue -u $USER;  tail -f $RUNS_DIR/logs/*.out"
