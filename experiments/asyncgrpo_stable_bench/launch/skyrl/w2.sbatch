#!/bin/bash
# W2 (agentic, TermGrade on HF Sandboxes, terminus-2) on SkyRL: 1 node exclusive, 4 vLLM engines + 4 FSDP2 trainer GPUs,
# fully async. Harbor runs in-process (SkyRL's HarborGenerator), terminus-2 calls SkyRL's vLLM router through LiteLLM.
#   smoke:     TAG=smoke CONCURRENCY=16 EXTRA="trainer.max_training_steps=3 ..." b200-queue --nodes=1 --gres=gpu:8 --exclusive launch/skyrl/w2.sbatch
#   full run:  b200-queue --nodes=1 --gres=gpu:8 --exclusive launch/skyrl/w2.sbatch
# Env: SIZE (M), TAG (shared), SPLIT (train), CONCURRENCY (live sandboxes = in-flight episodes, 64; must be a multiple of 8),
# TASK_REPEATS (18: the split's task dir is listed that many times, one epoch of 28 x 18 / 8 = 63 steps), EXTRA (key=value overrides)
#SBATCH --job-name=skyrl-w2
#SBATCH --nodes=1
#SBATCH --time=96:00:00
#SBATCH --output=/fsx/amine_dirhoussi/stable-bench-runs/skyrl/logs/%x-%j.out
set -uo pipefail

BENCH=/fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench
SIZE=${SIZE:-M}
TAG=${TAG:-shared}
SPLIT=${SPLIT:-train}
CONCURRENCY=${CONCURRENCY:-64}
TASK_REPEATS=${TASK_REPEATS:-18}
CONFIG=$BENCH/configs/skyrl/w2-$SIZE.overrides
HARBOR_CONFIG=$BENCH/configs/skyrl/w2-$SIZE.harbor.yaml
RUN_ID=skyrl-w2-$SIZE-$TAG-$SLURM_JOB_ID
RUN_DIR=/fsx/amine_dirhoussi/stable-bench-runs/skyrl/$RUN_ID
TASKS=/fsx/amine_dirhoussi/stable-bench-src/termgrade/harbor/$SPLIT/tasks
export TENSORBOARD_DIR=$RUN_DIR/tb
export SKYRL_VENV=/fsx/amine_dirhoussi/envs/bench-skyrl-w2
# The entrypoint (launch/skyrl) and SkyRL's examples package must be importable in every Ray worker.
export PYTHONPATH=$BENCH/launch/skyrl:/fsx/amine_dirhoussi/stable-bench-src/skyrl${PYTHONPATH:+:$PYTHONPATH}
# Harbor writes one directory per trial (agent logs, verifier output, result.json): node-local, archived at the end.
TRIALS_DIR=/scratch/$USER/skyrl-w2-trials-$SLURM_JOB_ID
mkdir -p "$TRIALS_DIR"
[ -d /scratch ] && export VLLM_CACHE_ROOT=/scratch/$USER/vllm-cache-$SLURM_JOB_ID
source $BENCH/launch/skyrl/common.sh

w2_cleanup() {
    local rc=$?
    # Sandboxes outlive a killed driver: sweep (twice) every run, whatever its exit. Lists only while another W2 job runs.
    "$BENCH/workloads/agentic/sweep.sh" /fsx/amine_dirhoussi/envs/bench-trl-w2/bin/python "$RUN_DIR/sweep.log"
    # one archive per attempt, so a requeued (resumed) job doesn't overwrite the first attempt's trials
    tar czf "$RUN_DIR/trials$([ "${SLURM_RESTART_COUNT:-0}" != 0 ] && echo "-r$SLURM_RESTART_COUNT").tgz" -C "$TRIALS_DIR" --exclude='*/agent/*.cast' . 2>/dev/null && rm -rf "$TRIALS_DIR"
    (exit $rc); skyrl_cleanup
}
trap w2_cleanup EXIT

mapfile -t ARGS < <(grep -v '^\s*#' "$CONFIG" | grep -v '^\s*$')
cp "$CONFIG" "$RUN_DIR/config.overrides"
cp "$HARBOR_CONFIG" "$RUN_DIR/harbor.yaml"
cp "$0" "$BENCH/launch/skyrl/skyrl_w2_main.py" "$RUN_DIR/" 2>/dev/null
uv pip freeze --python "$SKYRL_VENV/bin/python" > "$RUN_DIR/freeze.txt"
echo "RUN_ID=$RUN_ID CONFIG=$CONFIG SHA256=$(sha256sum "$CONFIG" | cut -d' ' -f1) HARBOR_SHA256=$(sha256sum "$HARBOR_CONFIG" | cut -d' ' -f1) CONCURRENCY=$CONCURRENCY EXTRA=${EXTRA:-}" | tee "$RUN_DIR/run_info.txt"

DATA=$(python -c "import sys; print('[' + ','.join([\"'$TASKS'\"] * $TASK_REPEATS) + ']')")
python -m skyrl_w2_main \
  "${ARGS[@]}" \
  --harbor-config="$HARBOR_CONFIG" \
  data.train_data="$DATA" \
  data.val_data="[]" \
  harbor_trial_config.trials_dir=$TRIALS_DIR \
  trainer.fully_async.num_parallel_generation_workers=$((CONCURRENCY / 8)) \
  generator.rate_limit.max_concurrency=$CONCURRENCY \
  trainer.project_name=asyncgrpo-stable-bench \
  trainer.run_name=$RUN_ID \
  trainer.log_path=$RUN_DIR/skyrl-logs \
  trainer.ckpt_path=$RUN_DIR/ckpts \
  trainer.export_path=$RUN_DIR \
  ${EXTRA:-} 2>&1 | tee "$RUN_DIR/train.log"
status=${PIPESTATUS[0]}
echo "EXIT=$status" >>"$RUN_DIR/run_info.txt"
exit $status
