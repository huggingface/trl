#!/bin/bash
# W1 (math) shared-config run for OpenRLHF on one 8 x B200 node: 4 vLLM engines + 4 DeepSpeed ranks, async.
#   b200-queue --nodes=1 --gres=gpu:8 --exclusive launch/openrlhf/w1.sbatch          (SIZE=S by default)
#   SIZE=M [CONFIG_NAME=w1-M-recommended] picks Qwen3.5-9B, its own data file and configs/openrlhf/w1-M*.sh.
#   Smoke knobs: MAX_STEPS (prompt budget, shared arm), STOP_AT_STEP (any arm), SAVE_STEPS (M configs only).
# Then: python collect/openrlhf.py <run dir>   (in a hopper-cpu job)
#SBATCH --job-name=orlhf-w1
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --output=/fsx/amine_dirhoussi/stable-bench-runs/openrlhf/logs/%x-%j.out
set -uo pipefail

SIZE=${SIZE:-S}
TAG=${TAG:-shared}
RUN_ID=openrlhf-w1-${SIZE}-${TAG}-${SLURM_JOB_ID}
RUN_DIR=/fsx/amine_dirhoussi/stable-bench-runs/openrlhf/$RUN_ID
source /fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench/launch/openrlhf/common.sh

case $SIZE in
    S) MODEL=$(ls -d $HF_HUB_CACHE/models--Qwen--Qwen3.5-2B/snapshots/*/ | head -1) ;;
    M) MODEL=$(ls -d $HF_HUB_CACHE/models--Qwen--Qwen3.5-9B/snapshots/*/ | head -1) ;;
esac
THINKING=${THINKING:-false}   # chat template enable_thinking (SHARED_W1 rev. 2: false): default | true | false
DATA=/fsx/amine_dirhoussi/stable-bench-runs/openrlhf/data/w1_train_${SIZE}_think-${THINKING}.parquet
NGPU_INF=${NGPU_INF:-$((NGPU / 2))}
NGPU_TRAIN=${NGPU_TRAIN:-$((NGPU - NGPU_INF))}
CONFIG=$BENCH/configs/openrlhf/${CONFIG_NAME:-w1-$SIZE}.sh   # CONFIG_NAME=w1-S-recommended for the recommended arm
[ -f "$CONFIG" ] || CONFIG=$BENCH/configs/openrlhf/w1-S.sh   # M reuses S unless a w1-M.sh exists
source "$CONFIG"
echo "${CONFIG#$BENCH/}" > "$RUN_DIR/config_path.txt"
printf '%s\n' "${W1_ARGS[@]}" "$@" > "$RUN_DIR/args.txt"   # the resolved flags; config_sha256 is computed on this
echo "MODEL=$MODEL DATA=$DATA CONFIG=$CONFIG NGPU_INF=$NGPU_INF NGPU_TRAIN=$NGPU_TRAIN"

python3 -m openrlhf.cli.train_ppo_ray "${W1_ARGS[@]}" "$@" &
TRAIN=$!
# OpenRLHF has no max-steps flag: the run ends when --data.max_samples prompts are used. Dynamic filtering (recommended
# arm) consumes extra prompts, so it gets the whole pool and is stopped once the step-STOP_AT_STEP HF checkpoint is written.
if [ -n "${STOP_AT_STEP:-}" ]; then
    (CK=$RUN_DIR/ckpt/global_step${STOP_AT_STEP}_hf
     until ls "$CK"/*.safetensors >/dev/null 2>&1 && [ -f "$CK/config.json" ]; do sleep 30; done
     sleep 120; echo "STOP_AT_STEP $STOP_AT_STEP reached, stopping the trainer"; kill -INT $TRAIN) &
fi
# Watchdog: a stuck Ray actor leaves every process idle with no error (NOTES.md items 19-20), so kill the trainer when
# the job log stops growing for STALL_MIN minutes and let Slurm record a failure instead of hanging until the time limit.
LOG=$(scontrol show job "$SLURM_JOB_ID" | sed -n 's/^ *StdOut=//p')
(while sleep 60; do
     [ $(( $(date +%s) - $(stat -c %Y "$LOG") )) -gt $(( ${STALL_MIN:-20} * 60 )) ] || continue
     echo "WATCHDOG: no log output for ${STALL_MIN:-20} min, killing the trainer"; touch "$RUN_DIR/WATCHDOG_KILLED"
     pkill -KILL -P $TRAIN; kill -KILL $TRAIN; break
 done) &
wait $TRAIN
RC=$?
[ -f "$RUN_DIR/WATCHDOG_KILLED" ] && RC=124
[ -n "${STOP_AT_STEP:-}" ] && [ -f "$RUN_DIR/ckpt/global_step${STOP_AT_STEP}_hf/config.json" ] && RC=0
echo "JOB_END $(date +%s.%N) rc=$RC" >> "$RUN_DIR/job_times.txt"
exit $RC
