#!/bin/bash
# miles' examples/text_to_sql as shipped (miles v0.2.0 image, its launcher `prepare` then `train`), one 8xB200 node,
# on Qwen3.5-35B-A3B (the example's default is Qwen3.6-35B-A3B, same architecture; --model-name/--megatron-model-type
# switch it) with `--attention-backend unfused` appended (the shipped backend that is correct on B200, envs/miles/NOTES.md).
#   PRIO_TAG=PRIO_TRAIN b200-queue --gres=gpu:8 --exclusive -t 12:00:00 -J t2sql-miles $PWD/t2sql.sbatch
#SBATCH -J t2sql-miles
#SBATCH --nodes=1
#SBATCH -o /fsx/amine_dirhoussi/stable-bench-runs/miles/t2sql/%x-%j.out
set -uo pipefail
HERE=/fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench/launch/miles
IMG=/fsx/amine_dirhoussi/stable-bench-src/images/miles-v0.2.0.sqsh
export RUN=/fsx/amine_dirhoussi/stable-bench-runs/miles/t2sql/miles-t2sql-$SLURM_JOB_ID
export WORK=/scratch/$USER/miles-t2sql-$SLURM_JOB_ID
mkdir -p $RUN $WORK
echo "$(date -u) job $SLURM_JOB_ID on $(hostname)" | tee $RUN/times.txt
unset SLURM_MEM_PER_NODE SLURM_MEM_PER_CPU SLURM_CPUS_PER_TASK SLURM_TRES_PER_TASK
srun --ntasks=1 --mem=0 --cpus-per-task=$SLURM_CPUS_ON_NODE --container-image=$IMG \
    --container-mounts=/fsx:/fsx,/scratch:/scratch --container-workdir=/root/miles \
    --export=ALL,RUN=$RUN,WORK=$WORK bash $HERE/t2sql_inner.sh
rc=$?
echo "$(date -u) exit $rc" | tee -a $RUN/times.txt
rm -rf $WORK
exit $rc
