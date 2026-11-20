#!/bin/bash
# W1 (math) on SkyRL: 1 node exclusive, 4 vLLM engines + 4 FSDP2 trainer GPUs, fully async.
# Submit with: /fsx/amine_dirhoussi/bin/b200-queue --nodes=1 --gres=gpu:8 --exclusive launch/skyrl/w1.sbatch
# Env: SIZE (default S; M = Qwen3.5-9B), TAG (default shared; "recommended" selects w1-<SIZE>-recommended.overrides),
#      MAX_STEPS (smoke: stop after N steps), SAVE_STEPS (smoke: HF export + resumable checkpoint every N steps),
#      EXTRA (extra key=value overrides, e.g. trainer.max_training_steps=5). Unset knobs leave the config untouched.
#SBATCH --job-name=skyrl-w1
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --output=/fsx/amine_dirhoussi/stable-bench-runs/skyrl/logs/%x-%j.out
set -uo pipefail

BENCH=/fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench
SIZE=${SIZE:-S}
TAG=${TAG:-shared}
CONFIG=$BENCH/configs/skyrl/w1-$SIZE$([ "$TAG" = recommended ] && echo -recommended).overrides
RUN_ID=skyrl-w1-$SIZE-$TAG-$SLURM_JOB_ID
RUN_DIR=/fsx/amine_dirhoussi/stable-bench-runs/skyrl/$RUN_ID
export TENSORBOARD_DIR=$RUN_DIR/tb
# The W1 env module and the shared reward must be importable in every Ray worker (workers inherit the raylet env).
export PYTHONPATH=$BENCH/launch/skyrl:$BENCH/workloads/math${PYTHONPATH:+:$PYTHONPATH}
source $BENCH/launch/skyrl/common.sh

mapfile -t ARGS < <(grep -v '^\s*#' "$CONFIG" | grep -v '^\s*$')
cp "$CONFIG" "$RUN_DIR/config.overrides"
echo "RUN_ID=$RUN_ID CONFIG=$CONFIG SHA256=$(sha256sum "$CONFIG" | cut -d' ' -f1) MAX_STEPS=${MAX_STEPS:-} SAVE_STEPS=${SAVE_STEPS:-} EXTRA=${EXTRA:-}" | tee "$RUN_DIR/run_info.txt"

python -m skyrl_w1_main \
  "${ARGS[@]}" \
  data.train_data="['/fsx/amine_dirhoussi/stable-bench-runs/skyrl/data/w1/train.parquet']" \
  data.val_data="[]" \
  trainer.project_name=asyncgrpo-stable-bench \
  trainer.run_name=$RUN_ID \
  trainer.log_path=$RUN_DIR/skyrl-logs \
  trainer.ckpt_path=$RUN_DIR/ckpts \
  trainer.export_path=$RUN_DIR \
  ${MAX_STEPS:+trainer.max_training_steps=$MAX_STEPS} \
  ${SAVE_STEPS:+trainer.hf_save_interval=$SAVE_STEPS trainer.ckpt_interval=$SAVE_STEPS} \
  ${EXTRA:-} 2>&1 | tee "$RUN_DIR/train.log"
status=${PIPESTATUS[0]}
echo "EXIT=$status" >>"$RUN_DIR/run_info.txt"
exit $status
