#!/bin/bash
# Inside the NeMo-RL container: driver + watcher.
set -uo pipefail
HERE=$(dirname "$(readlink -f "$0")")
D=/fsx/amine_dirhoussi/stable-bench-src/text2sql
T2=/fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench/workloads/text2sql
cd /opt/nemo-rl
# the container venv, not the login uv/PATH leaking in through --export=ALL (uv run would resync it)
export PATH=/opt/nemo_rl_venv/bin:$PATH
{ git log -1 --format='nemo-rl %h %cd' 2>/dev/null; cat /opt/nemo-rl/pyproject.toml | grep -m1 '^version'; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1; } | tee -a $RUN_DIR/times.txt
export HF_HUB_CACHE=/fsx/amine_dirhoussi/stable-bench-hub HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=$D/river-recipes/text-to-sql/src:$T2:$HERE:$HERE/trl_stub${PYTHONPATH:+:$PYTHONPATH}
mkdir -p -m 700 $WORK/snapshots
export TEXT_TO_SQL_SQLITE_SNAPSHOT_ROOT=$WORK/snapshots
# one node: a private Ray session under a short path (AF_UNIX limit); two nodes: attach to ray.sub's cluster (/tmp/ray)
[ -z "${MULTINODE:-}" ] && export TMPDIR=/tmp/n$SLURM_JOB_ID RAY_TMPDIR=/tmp/n$SLURM_JOB_ID
CONFIG=${CONFIG:-$HERE/t2sql.yaml}
export T2SQL_TRAIN=$D/data/train.jsonl T2SQL_EPISODES_FILE=$RUN_DIR/episodes.jsonl
cp $CONFIG $RUN_DIR/config.yaml
echo "$(date -u +%FT%TZ) train" | tee -a $RUN_DIR/times.txt
setsid python $HERE/nemo_t2sql.py $CONFIG ${OVERRIDES:-} > $RUN_DIR/train.log 2>&1 &
PID=$!
python $HERE/watch.py $RUN_DIR $PID > $RUN_DIR/watch.out 2>&1 &
wait $PID; rc=$?
echo "$(date -u +%FT%TZ) train exit $rc" | tee -a $RUN_DIR/times.txt
tail -n 60 $RUN_DIR/train.log
exit $rc
