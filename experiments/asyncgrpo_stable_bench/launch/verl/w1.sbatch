#!/bin/bash
#SBATCH --job-name=verl-w1
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=/fsx/amine_dirhoussi/stable-bench-runs/verl/logs/%x-%j.out
# W1 math with verl (V1 separate_async, 4 vLLM + 4 FSDP2 GPUs). Run with:
#   SIZE=S TAG=shared b200-queue launch/verl/w1.sbatch [extra hydra overrides]
#   SIZE=S TAG=recommended CONFIG=w1-S-recommended b200-queue launch/verl/w1.sbatch
#   SIZE=M TAG=shared b200-queue launch/verl/w1.sbatch    (Qwen3.5-9B, configs/verl/w1-M.yaml)
#   SIZE=M TAG=recommended CONFIG=w1-M-recommended b200-queue launch/verl/w1.sbatch
# Smoke knobs: MAX_STEPS=<n> (trainer.total_training_steps, in verl global steps), SAVE_STEPS=<n> (trainer.save_freq).
# Native logs, gpu.csv and checkpoints go to /fsx/amine_dirhoussi/stable-bench-runs/verl/<run_id>/;
# collect/verl.py turns them into results/raw/<run_id>/ at the end. A requeued job resumes from its last checkpoint.
set -uo pipefail
SIZE=${SIZE:-S}
TAG=${TAG:-shared}
CONFIG=${CONFIG:-w1-$SIZE}
# Smoke knobs; unset = the config's values.
OVR=()
[ -n "${MAX_STEPS:-}" ] && OVR+=(trainer.total_training_steps=$MAX_STEPS)
[ -n "${SAVE_STEPS:-}" ] && OVR+=(trainer.save_freq=$SAVE_STEPS)
# M: expandable segments (S reserved 170 GB per trainer GPU for 32 GB allocated). Exported before `ray start` in
# common.sh so every Ray worker inherits it; verl's vLLM server turns it on anyway without sleep mode, and the NCCL
# checkpoint engine supports it (cupy buffers on the master).
[ "$SIZE" = M ] && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# verl's own log-level knob (every verl module reads it). At INFO, v0.9.1's separate_async with hybrid_engine=False dies at
# trainer init on `logger.info(..., flush=True)` (TypeError); at WARN that INFO call returns before the bad kwarg is
# checked, so the shipped code runs unchanged. Metrics are unaffected (console/file loggers print). Exported before
# `ray start` in common.sh so the TaskRunner inherits it. envs/verl/NOTES.md "As shipped (2026-11-10)".
export VERL_LOGGING_LEVEL=${VERL_LOGGING_LEVEL:-WARN}
RUN_ID=verl-w1-$SIZE-$TAG-$SLURM_JOB_ID
RUN_DIR=/fsx/amine_dirhoussi/stable-bench-runs/verl/$RUN_ID
mkdir -p "$RUN_DIR"
# verl's file logger opens its file with "wb": keep the metrics of an earlier (preempted) attempt.
[ -s "$RUN_DIR/metrics.jsonl" ] && mv "$RUN_DIR/metrics.jsonl" "$RUN_DIR/metrics.$(date +%s).jsonl"
export VERL_FILE_LOGGER_PATH=$RUN_DIR/metrics.jsonl
source /fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench/launch/verl/common.sh

DATA=/fsx/amine_dirhoussi/stable-bench-runs/verl/data/w1
[ -s "$DATA/train.parquet" ] || python "$BENCH/configs/verl/w1_data.py" "$DATA"
cp "$BENCH/configs/verl/$CONFIG.yaml" "$RUN_DIR/config.yaml"; cp "$BENCH/configs/verl/w1-$SIZE.yaml" "$RUN_DIR/config_base.yaml"; echo "$CONFIG" > "$RUN_DIR/config_name"
git -C "$SRC" rev-parse HEAD > "$RUN_DIR/verl_sha"
echo "$SLURM_JOB_NODELIST" > "$RUN_DIR/nodes"

# Every stdout line gets a unix timestamp prefix: verl's metrics carry no wall clock, the collector takes it from here.
python -m verl.trainer.main_ppo --config-path="$BENCH/configs/verl" --config-name="$CONFIG" \
  trainer.experiment_name="$RUN_ID" \
  trainer.default_local_dir="$RUN_DIR/ckpt" \
  "${OVR[@]}" \
  "$@" 2>&1 | perl -MTime::HiRes=time -ne '$|=1; printf "%.3f %s", time, $_' | tee -a "$RUN_DIR/stdout.log"
RC=${PIPESTATUS[0]}
echo "EXIT $RC"
cleanup
python "$BENCH/collect/verl.py" "$RUN_DIR" --job-log "/fsx/amine_dirhoussi/stable-bench-runs/verl/logs/verl-w1-$SLURM_JOB_ID.out" --status $([ $RC = 0 ] && echo completed || echo failed) || true
exit $RC
