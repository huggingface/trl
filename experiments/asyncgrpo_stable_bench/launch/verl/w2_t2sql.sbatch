#!/bin/bash
#SBATCH --job-name=t2sql-verl
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=14:00:00
#SBATCH --output=/fsx/amine_dirhoussi/stable-bench-runs/verl/logs/%x-%j.out
# Text-to-SQL (miles' River recipe, TRL's port) on Qwen3.5-35B-A3B + LoRA with verl v0.9.1 as shipped: V1 separate_async,
# 4 vLLM + 4 FSDP2 GPUs, verl's recommended settings (configs/verl/t2sql-moe.yaml), the episode in
# configs/verl/t2sql_agent_loop.py. Named w2_* so the queue's PRIORITY entry launch/verl/w2 covers it.
#   cd <bench>; PRIO_TAG=PRIO_TRAIN TAG=rec /fsx/amine_dirhoussi/bin/b200-queue launch/verl/w2_t2sql.sbatch
# In-job checks: launch/verl/t2sql_watch.py (WATCHDOG in the run dir stops training). Checkpoints are merged on this
# node's CPU as they land (launch/verl/t2sql_merge.sh) and evaluated by launch/verl/t2sql_evals.sh (hopper-cpu helper).
set -uo pipefail
TAG=${TAG:-rec}
OVR=()
[ -n "${MAX_STEPS:-}" ] && OVR+=(trainer.total_training_steps=$MAX_STEPS)
SAVE_STEPS=${SAVE_STEPS:-20}; OVR+=(trainer.save_freq=$SAVE_STEPS)
# No PYTORCH_CUDA_ALLOC_CONF=expandable_segments: it breaks verl's CUDA-IPC weight sync (NOTES.md W2 item 3).
# v0.9.1 separate_async + hybrid_engine=false needs WARN (NOTES.md "As shipped").
export VERL_LOGGING_LEVEL=${VERL_LOGGING_LEVEL:-WARN}
RUN_ID=verl-t2sql-Qwen3.5-35B-A3B-lora32-$TAG-$SLURM_JOB_ID
RUN_DIR=/fsx/amine_dirhoussi/stable-bench-runs/verl/$RUN_ID
mkdir -p "$RUN_DIR/episodes" /fsx/amine_dirhoussi/stable-bench-runs/verl/merged
export VERL_FILE_LOGGER_PATH=$RUN_DIR/metrics.jsonl
BENCH=/fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench
T2=$BENCH/workloads/text2sql
D=/fsx/amine_dirhoussi/stable-bench-src/text2sql

# What the agent-loop workers (Ray actors, which inherit this environment from `ray start`) need: the River recipe
# (stdlib only), the episode module, the agent loop; SQLite snapshots and temp files on local NVMe.
export PYTHONPATH=$D/river-recipes/text-to-sql/src:$T2:$BENCH/configs/verl
export T2SQL_EPISODES_DIR=$RUN_DIR/episodes
S=/scratch/$USER/t2sql-verl-$SLURM_JOB_ID; mkdir -p -m 700 $S $S/snapshots $S/tmp; chmod 700 $S/snapshots
export TEXT_TO_SQL_SQLITE_SNAPSHOT_ROOT=$S/snapshots TMPDIR=$S/tmp

source $BENCH/launch/verl/common.sh   # venv, Ray head, gpu.csv, `cleanup`; cd's into the verl checkout
MERGE_PID=""; WATCH_PID=""
t2_cleanup() { kill $MERGE_PID $WATCH_PID 2>/dev/null; cleanup; rm -rf "$S"; date +%s.%N > "$RUN_DIR/job_end"; }
trap t2_cleanup EXIT

DATA=$RUN_DIR/data; mkdir -p "$DATA"
python "$BENCH/configs/verl/t2sql_agent_loop.py" data "$D/data/train.jsonl" "$DATA/train.parquet" || exit 1
cp "$BENCH/configs/verl/t2sql-moe.yaml" "$RUN_DIR/config.yaml"
cp "$BENCH/configs/verl/t2sql_agent_loop.py" "$T2/t2sql_episode.py" "$0" "$RUN_DIR/" 2>/dev/null
git -C "$SRC" rev-parse HEAD > "$RUN_DIR/verl_sha"; git -C "$SRC" status --short > "$RUN_DIR/verl_status"
git -C "$D/river-recipes" rev-parse HEAD > "$RUN_DIR/river_recipes_sha"
echo "$SLURM_JOB_NODELIST" > "$RUN_DIR/nodes"
env | grep -E '^(TAG|MAX_STEPS|SAVE_STEPS|VERL_LOGGING_LEVEL)=' > "$RUN_DIR/launch.env"

python "$BENCH/launch/verl/t2sql_watch.py" "$RUN_DIR" "$SAVE_STEPS" > "$RUN_DIR/watchdog.log" 2>&1 &
WATCH_PID=$!
( while kill -0 $WATCH_PID 2>/dev/null; do sleep 30; done
  [ -f "$RUN_DIR/WATCHDOG" ] && { cat "$RUN_DIR/WATCHDOG"; pkill -u $USER -f verl.trainer.main_ppo; } ) &
bash "$BENCH/launch/verl/t2sql_merge.sh" "$RUN_DIR" loop > "$RUN_DIR/merge.log" 2>&1 &
MERGE_PID=$!

python -m verl.trainer.main_ppo --config-path="$BENCH/configs/verl" --config-name=t2sql-moe \
  data.train_files="$DATA/train.parquet" data.val_files="$DATA/train.parquet" \
  trainer.experiment_name="$RUN_ID" \
  trainer.default_local_dir="$RUN_DIR/ckpt" \
  "${OVR[@]}" \
  "$@" 2>&1 | perl -MTime::HiRes=time -ne '$|=1; printf "%.3f %s", time, $_' | tee -a "$RUN_DIR/stdout.log"
RC=${PIPESTATUS[0]}
echo "EXIT $RC"
kill $MERGE_PID 2>/dev/null; wait $MERGE_PID 2>/dev/null; MERGE_PID=""
ray stop --force > /dev/null 2>&1
# last checkpoint(s): merge before the job ends (the eval helper waits for merge_check.json)
bash "$BENCH/launch/verl/t2sql_merge.sh" "$RUN_DIR" once >> "$RUN_DIR/merge.log" 2>&1
python "$BENCH/collect/verl.py" "$RUN_DIR" --job-log "/fsx/amine_dirhoussi/stable-bench-runs/verl/logs/t2sql-verl-$SLURM_JOB_ID.out" --status $([ $RC = 0 ] && echo completed || echo failed) || true
exit $RC
