#!/bin/bash
# Audit F2 rerun: W2-M on SkyRL's stock Harbor path (stock entrypoint main_harbor_fully_async, stock HarborGenerator, Harbor
# from SkyRL's lock pin, local Apptainer boxes) with SkyRL's default loss (dppo + prompt_mean) and SHARED_W2's shape
# (8 tasks x 8 episodes, 64 in flight, 4 x TP1 engines). Config configs/skyrl/w2-M-default-loss.overrides.
#   PRIO_TAG=PRIO_TRAIN /fsx/amine_dirhoussi/bin/b200-queue --gres=gpu:8 --exclusive -t 40:00:00 launch/skyrl/w2-dppo.sbatch
# Resumes from the latest resumable checkpoint (every 5 steps) when RUN_ID names an earlier run.
# Env: RUN_ID (new: skyrl-w2-M-dppo-<job>), TASK_REPEATS (18: one epoch = 28 x 18 / 8 = 63 steps), EXTRA.
#SBATCH --job-name=skyrl-w2-dppo
#SBATCH --nodes=1
#SBATCH --output=/fsx/amine_dirhoussi/stable-bench-runs/skyrl/logs/%x-%j.out
set -uo pipefail

BENCH=/fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench
CONFIG=$BENCH/configs/skyrl/w2-M-default-loss.overrides
TASK_REPEATS=${TASK_REPEATS:-18}
RUN_ID=${RUN_ID:-skyrl-w2-M-dppo-$SLURM_JOB_ID}
RUN_DIR=/fsx/amine_dirhoussi/stable-bench-runs/skyrl/$RUN_ID
TASKS=/fsx/amine_dirhoussi/stable-bench-src/termgrade/harbor-local-sif/train/tasks
export SKYRL_VENV=/fsx/amine_dirhoussi/envs/bench-skyrl-w2local
mkdir -p $RUN_DIR
echo "$(date -u +%FT%TZ) job $SLURM_JOB_ID on $(hostname) restart ${SLURM_RESTART_COUNT:-0}" | tee -a $RUN_DIR/job.txt

export HF_HUB_OFFLINE=1
export SKYRL_WAIT_UNTIL_INFERENCE_SERVER_HEALTHY_TIMEOUT_S=1800
export LD_LIBRARY_PATH=$SKYRL_VENV/lib/python3.12/site-packages/nvidia/cudnn/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export TENSORBOARD_DIR=$RUN_DIR/tb
S=/scratch/$USER/skyrl-w2dppo-$SLURM_JOB_ID; mkdir -p -m 700 $S/trials $S/tmp $S/apptainer-cache $S/apptainer-tmp
export VLLM_CACHE_ROOT=$S/vllm-cache
# Apptainer as `singularity` on PATH for Harbor inside the Ray workers (they inherit the raylet's env, set before ray start)
source /fsx/amine_dirhoussi/stable-bench-src/container-tools/apptainer-env.sh > $RUN_DIR/apptainer-env.log 2>&1
export LD_LIBRARY_PATH=$AP_LIBS:$LD_LIBRARY_PATH APPTAINER_BIND=/etc/resolv.conf TMPDIR=$S/tmp
export APPTAINER_CACHEDIR=$S/apptainer-cache APPTAINER_TMPDIR=$S/apptainer-tmp
apptainer --version >> $RUN_DIR/job.txt 2>&1
export PYTHONPATH=/fsx/amine_dirhoussi/stable-bench-src/skyrl${PYTHONPATH:+:$PYTHONPATH}
# Ray port block below the ephemeral range (t2sql item 9): first free 1000-port block in 12000-31999
for b in $(seq $((12000 + (SLURM_JOB_ID % 20) * 1000)) 1000 31000) $(seq 12000 1000 31000); do
    ss -Htan "sport >= :$b and sport <= :$((b + 999))" | grep -q . || { export RAY_BASE=$b; break; }
done
echo "RAY_BASE=$RAY_BASE" >> $RUN_DIR/job.txt
source $BENCH/launch/skyrl/common.sh
w2_cleanup() {
    local rc=$?
    tar czf "$RUN_DIR/trials-$SLURM_JOB_ID.tgz" -C "$S/trials" --exclude='*/agent/*.cast' . 2>/dev/null
    kill ${WATCH:-} 2>/dev/null; rm -rf $S
    (exit $rc); skyrl_cleanup
}
trap w2_cleanup EXIT

mapfile -t ARGS < <(grep -v '^\s*#' "$CONFIG" | grep -v '^\s*$')
cp "$CONFIG" "$RUN_DIR/config.overrides"; cp "$0" "$RUN_DIR/" 2>/dev/null
uv pip freeze --python "$SKYRL_VENV/bin/python" > "$RUN_DIR/freeze.txt"
echo "RUN_ID=$RUN_ID JOB=$SLURM_JOB_ID CONFIG=$CONFIG SHA256=$(sha256sum "$CONFIG" | cut -d' ' -f1) EXTRA=${EXTRA:-}" | tee -a "$RUN_DIR/run_info.txt"

python $BENCH/launch/skyrl/w2rec_watch.py $RUN_DIR $S/trials &
WATCH=$!
DATA=$(python -c "print('[' + ','.join([\"'$TASKS'\"] * $TASK_REPEATS) + ']')")
python -m examples.train_integrations.harbor.entrypoints.main_harbor_fully_async \
  "${ARGS[@]}" \
  data.train_data="$DATA" \
  data.val_data="[]" \
  harbor_trial_config.trials_dir=$S/trials \
  trainer.project_name=asyncgrpo-stable-bench \
  trainer.run_name=$RUN_ID \
  trainer.log_path=$RUN_DIR/skyrl-logs \
  trainer.ckpt_path=$RUN_DIR/ckpts \
  trainer.export_path=$RUN_DIR \
  ${EXTRA:-} 2>&1 | tee -a "$RUN_DIR/train.log"
status=${PIPESTATUS[0]}
echo "JOB=$SLURM_JOB_ID EXIT=$status" >>"$RUN_DIR/run_info.txt"
exit $status
