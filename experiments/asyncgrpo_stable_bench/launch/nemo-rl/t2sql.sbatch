#!/bin/bash
# NeMo-RL v0.8.0 (NGC container) on text-to-SQL, Qwen3.5-35B-A3B, one exclusive 8xB200 node. Config t2sql/t2sql.yaml
# (NeMo-RL's validated Qwen3.5-35B-A3B Automodel recipe + the workload), env/driver t2sql/nemo_t2sql.py, in-job
# watcher t2sql/watch.py (no separate smoke run).
#   cd .../asyncgrpo_stable_bench && PRIO_TAG=PRIO_TRAIN /fsx/amine_dirhoussi/bin/b200-queue --gres=gpu:8 --exclusive \
#       -t 16:00:00 launch/nemo-rl/t2sql.sbatch
#SBATCH -J nemo-rl-t2sql
#SBATCH --nodes=1
#SBATCH -o /fsx/amine_dirhoussi/stable-bench-runs/nemo-rl/logs/%x-%j.out
set -uo pipefail
HERE=/fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench/launch/nemo-rl
IMG=/fsx/amine_dirhoussi/stable-bench-src/images/nemo-rl-v0.8.0.sqsh
export RUN_DIR=/fsx/amine_dirhoussi/stable-bench-runs/nemo-rl/t2sql/nemo-rl-t2sql-Qwen3.5-35B-A3B-$SLURM_JOB_ID
export WORK=/scratch/$USER/nemo-rl-t2sql-$SLURM_JOB_ID
mkdir -p $RUN_DIR $WORK
echo "$(date -u +%FT%TZ) job $SLURM_JOB_ID on $(hostname)" | tee $RUN_DIR/times.txt
# b200-queue's hopper-cpu waiter leaks its 1-CPU/2G limits into srun steps (envs/miles/NOTES.md item 9)
unset SLURM_MEM_PER_NODE SLURM_MEM_PER_CPU SLURM_CPUS_PER_TASK SLURM_TRES_PER_TASK
srun --ntasks=1 --mem=0 --cpus-per-task=$SLURM_CPUS_ON_NODE --container-image=$IMG \
    --container-mounts=/fsx:/fsx,/scratch:/scratch --container-workdir=/opt/nemo-rl \
    --export=ALL,RUN_DIR=$RUN_DIR,WORK=$WORK bash $HERE/t2sql/inner.sh
rc=$?
echo "$(date -u +%FT%TZ) exit $rc" | tee -a $RUN_DIR/times.txt
rm -rf $WORK
exit $rc
