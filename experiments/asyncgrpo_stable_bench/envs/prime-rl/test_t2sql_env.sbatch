#!/bin/bash
# hopper-cpu: CPU check of the prime-rl text-to-SQL env (test_t2sql_env.py) in prime-rl's venv.
#SBATCH -J prl-t2sql-test
#SBATCH -p hopper-cpu
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH -t 2:00:00
#SBATCH -o /fsx/amine_dirhoussi/stable-bench-runs/prime-rl/logs/%x-%j.out
set -uo pipefail
B=/fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench
D=/fsx/amine_dirhoussi/stable-bench-src/text2sql
export PYTHONPATH=$D/river-recipes/text-to-sql/src:$B/configs/prime-rl
export HF_HUB_CACHE=/fsx/amine_dirhoussi/stable-bench-hub HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
S=$D/scratch-cpu-$SLURM_JOB_ID; mkdir -p -m 700 $S $S/snapshots $S/tmp; chmod 700 $S/snapshots
export TEXT_TO_SQL_SQLITE_SNAPSHOT_ROOT=$S/snapshots TMPDIR=$S/tmp
echo "start $(date -u)"
cd $B/envs/prime-rl && /fsx/amine_dirhoussi/envs/bench-prime-rl/bin/python test_t2sql_env.py $D/data ${NTEST:-100}
echo "end $(date -u) rc=$?"
rm -rf $S
