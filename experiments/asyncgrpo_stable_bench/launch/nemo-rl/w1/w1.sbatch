#!/bin/bash
# W1-S (math, Qwen3.5-2B) on NeMo-RL v0.8.0 (NGC container), one exclusive 8xB200 node: 4 vLLM + 4 Megatron GPUs,
# async GRPO. Config configs/nemo-rl/w1-S.yaml, driver nemo_w1.py (= examples/run_grpo.py + the w1_math env),
# in-job watcher watch.py. A requeue (same job id) resumes from the latest checkpoint in the run dir (NeMo-RL resumes
# from checkpointing.checkpoint_dir by itself).
#   cd <bench> && PRIO_TAG=PRIO_TRAIN /fsx/amine_dirhoussi/bin/b200-queue --gres=gpu:8 --exclusive -t 24:00:00 launch/nemo-rl/w1/w1.sbatch
#SBATCH -J nemo-rl-w1
#SBATCH --nodes=1
#SBATCH --requeue
#SBATCH -o /fsx/amine_dirhoussi/stable-bench-runs/nemo-rl/logs/%x-%j.out
set -uo pipefail
HERE=/fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench/launch/nemo-rl/w1
IMG=/fsx/amine_dirhoussi/stable-bench-src/images/nemo-rl-v0.8.0.sqsh
export RUN_DIR=/fsx/amine_dirhoussi/stable-bench-runs/nemo-rl/w1/nemo-rl-w1-S-shared-$SLURM_JOB_ID
mkdir -p $RUN_DIR
echo "$(date -u +%FT%TZ) job $SLURM_JOB_ID on $(hostname) restart ${SLURM_RESTART_COUNT:-0}" | tee -a $RUN_DIR/times.txt
unset SLURM_MEM_PER_NODE SLURM_MEM_PER_CPU SLURM_CPUS_PER_TASK SLURM_TRES_PER_TASK
srun --ntasks=1 --mem=0 --cpus-per-task=$SLURM_CPUS_ON_NODE --container-image=$IMG \
    --container-mounts=/fsx:/fsx,/scratch:/scratch --container-workdir=/opt/nemo-rl \
    --export=ALL,RUN_DIR=$RUN_DIR bash $HERE/inner.sh
rc=$?
echo "$(date -u +%FT%TZ) exit $rc" | tee -a $RUN_DIR/times.txt
exit $rc
