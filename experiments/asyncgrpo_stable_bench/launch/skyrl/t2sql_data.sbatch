#!/bin/bash
# hopper-cpu: SkyRL parquet for text-to-SQL + parity check vs the TRL port (t2sql_data.py), with the bench-skyrl venv
# (same SkyRL checkout; the generator/env code paths are backend-independent).
#SBATCH -J skyrl-t2sql-data
#SBATCH -o /fsx/amine_dirhoussi/stable-bench-runs/skyrl/logs/%x-%j.out
set -euo pipefail
B=/fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench
D=/fsx/amine_dirhoussi/stable-bench-src/text2sql
export HF_HUB_CACHE=/fsx/amine_dirhoussi/stable-bench-hub HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=$D/river-recipes/text-to-sql/src:$B/workloads/text2sql:$B/launch/skyrl
S=/scratch/$USER/t2sql-data-$SLURM_JOB_ID; [ -d /scratch ] || S=/tmp/$USER-t2sql-data-$SLURM_JOB_ID
mkdir -p -m 700 $S/snapshots $S/tmp; trap 'rm -rf $S' EXIT
export TEXT_TO_SQL_SQLITE_SNAPSHOT_ROOT=$S/snapshots TMPDIR=$S/tmp
echo "$(date -u) start on $(hostname)"
/fsx/amine_dirhoussi/envs/bench-skyrl/bin/python $B/launch/skyrl/t2sql_data.py $D/data/train.jsonl /fsx/amine_dirhoussi/stable-bench-runs/skyrl/data/t2sql ${NPARITY:-20}
echo "$(date -u) done"
