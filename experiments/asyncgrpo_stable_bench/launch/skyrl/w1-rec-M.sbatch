#!/bin/bash
# Audit F1 rerun: W1-M (Qwen3.5-9B math) with SkyRL's default loss (configs/skyrl/w1-M-recommended.overrides: dppo +
# prompt_mean, everything else the shared W1-M config). Same body as w1.sbatch plus the launcher fixes from the 35B/W2 runs
# (Ray port block in 12000-31999, 1800 s vLLM health wait), the in-job watcher (launch/skyrl/w1_watch.py), and the shared
# math eval (workloads/math/eval.sbatch, AIME24/AIME25 avg@16 + MATH-500) on every HF export once training ends, on this node.
#   PRIO_TAG=PRIO_TRAIN /fsx/amine_dirhoussi/bin/b200-queue --gres=gpu:8 --exclusive -t 12:00:00 launch/skyrl/w1-rec-M.sbatch
#SBATCH --job-name=skyrl-w1-rec-M
#SBATCH --nodes=1
#SBATCH --output=/fsx/amine_dirhoussi/stable-bench-runs/skyrl/logs/%x-%j.out
set -uo pipefail

BENCH=/fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench
CONFIG=$BENCH/configs/skyrl/w1-M-recommended.overrides
RUN_ID=${RUN_ID:-skyrl-w1-M-recommended-$SLURM_JOB_ID}
RUN_DIR=/fsx/amine_dirhoussi/stable-bench-runs/skyrl/$RUN_ID
mkdir -p $RUN_DIR
echo "$(date -u +%FT%TZ) job $SLURM_JOB_ID on $(hostname) restart ${SLURM_RESTART_COUNT:-0}" | tee -a $RUN_DIR/job.txt
export TENSORBOARD_DIR=$RUN_DIR/tb
export SKYRL_WAIT_UNTIL_INFERENCE_SERVER_HEALTHY_TIMEOUT_S=1800
export PYTHONPATH=$BENCH/launch/skyrl:$BENCH/workloads/math${PYTHONPATH:+:$PYTHONPATH}
for b in $(seq $((12000 + (SLURM_JOB_ID % 20) * 1000)) 1000 31000) $(seq 12000 1000 31000); do
    ss -Htan "sport >= :$b and sport <= :$((b + 999))" | grep -q . || { export RAY_BASE=$b; break; }
done
echo "RAY_BASE=$RAY_BASE" >> $RUN_DIR/job.txt
source $BENCH/launch/skyrl/common.sh
w1_cleanup() { local rc=$?; kill ${WATCH:-} 2>/dev/null; (exit $rc); skyrl_cleanup; }
trap w1_cleanup EXIT

mapfile -t ARGS < <(grep -v '^\s*#' "$CONFIG" | grep -v '^\s*$')
cp "$CONFIG" "$RUN_DIR/config.overrides"; cp "$0" "$RUN_DIR/" 2>/dev/null
echo "RUN_ID=$RUN_ID JOB=$SLURM_JOB_ID CONFIG=$CONFIG SHA256=$(sha256sum "$CONFIG" | cut -d' ' -f1) EXTRA=${EXTRA:-}" | tee -a "$RUN_DIR/run_info.txt"

python $BENCH/launch/skyrl/w1_watch.py $RUN_DIR &
WATCH=$!
python -m skyrl_w1_main \
  "${ARGS[@]}" \
  data.train_data="['/fsx/amine_dirhoussi/stable-bench-runs/skyrl/data/w1/train.parquet']" \
  data.val_data="[]" \
  trainer.project_name=asyncgrpo-stable-bench \
  trainer.run_name=$RUN_ID \
  trainer.log_path=$RUN_DIR/skyrl-logs \
  trainer.ckpt_path=$RUN_DIR/ckpts \
  trainer.export_path=$RUN_DIR \
  ${EXTRA:-} 2>&1 | tee -a "$RUN_DIR/train.log"
status=${PIPESTATUS[0]}
echo "JOB=$SLURM_JOB_ID EXIT=$status" >>"$RUN_DIR/run_info.txt"
kill $WATCH 2>/dev/null
[ $status -ne 0 ] && exit $status

# eval on this node: exclusive allocation, so stopping Ray only touches this job
ray stop --force >/dev/null 2>&1; sleep 10
for d in $RUN_DIR/global_step_*/policy; do n=$(basename $(dirname $d)); ln -sfn $n/policy $RUN_DIR/checkpoint-${n#global_step_}; done
echo "$(date -u +%FT%TZ) eval start" >> $RUN_DIR/job.txt
bash $BENCH/workloads/math/eval.sbatch $RUN_DIR $RUN_DIR/eval.jsonl
echo "$(date -u +%FT%TZ) eval end" >> $RUN_DIR/job.txt
