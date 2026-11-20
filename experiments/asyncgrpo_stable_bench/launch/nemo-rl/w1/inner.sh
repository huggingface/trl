#!/bin/bash
# Inside the NeMo-RL container: driver + watcher.
set -uo pipefail
HERE=$(dirname "$(readlink -f "$0")")
B=/fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench
cd /opt/nemo-rl
export PATH=/opt/nemo_rl_venv/bin:$PATH
{ git log -1 --format='nemo-rl %h %cd' 2>/dev/null; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1; } | tee -a $RUN_DIR/times.txt
export HF_HUB_CACHE=/fsx/amine_dirhoussi/stable-bench-hub HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=$HERE:$B/workloads/math${PYTHONPATH:+:$PYTHONPATH}
export TMPDIR=/tmp/n$SLURM_JOB_ID RAY_TMPDIR=/tmp/n$SLURM_JOB_ID
mkdir -p $TMPDIR
CONFIG=${CONFIG:-$B/configs/nemo-rl/w1-S.yaml}
cp $CONFIG $RUN_DIR/config.yaml
echo "$(date -u +%FT%TZ) train" | tee -a $RUN_DIR/times.txt
setsid python $HERE/nemo_w1.py --config $CONFIG checkpointing.checkpoint_dir=$RUN_DIR/ckpt logger.log_dir=$RUN_DIR/logs \
    ${OVERRIDES:-} >> $RUN_DIR/train.log 2>&1 &
PID=$!
python $HERE/watch.py $RUN_DIR $PID >> $RUN_DIR/watch.out 2>&1 &
wait $PID; rc=$?
echo "$(date -u +%FT%TZ) train exit $rc" | tee -a $RUN_DIR/times.txt
tail -n 60 $RUN_DIR/train.log
exit $rc
