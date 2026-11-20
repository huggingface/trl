#!/bin/bash -l
#SBATCH --job-name=miles-convdbg
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --time=00:40:00
#SBATCH --output=/fsx/amine_dirhoussi/stable-bench-runs/miles/logs/%x-%j.out
# Debug the silent HF->torch_dist hang seen in smoke 364156 (0.8B, >15 min, no CUDA context, single-threaded CPU).
IMG=/fsx/amine_dirhoussi/stable-bench-src/images/miles-v0.1.3.sqsh
# b200-queue submits from a -c 1 --mem=2G waiter with --export=ALL: its SLURM_MEM_PER_NODE / SLURM_CPUS_PER_TASK leak in
# and would cap this srun step at 2 GiB and 1 CPU. Drop them and give the step the whole allocation.
unset SLURM_MEM_PER_NODE SLURM_MEM_PER_CPU SLURM_CPUS_PER_TASK SLURM_TRES_PER_TASK
srun --ntasks=1 --mem=0 --cpus-per-task=$SLURM_CPUS_ON_NODE --container-image=$IMG --container-mounts=/fsx:/fsx --container-workdir=/root/miles \
  --export=ALL,PORT_BASE=$((11000 + (SLURM_JOB_ID % 90) * 10)) bash -c '
source /fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench/launch/miles/common.sh
echo "PYTHONPATH=$PYTHONPATH"
T=$(date +%s); cp -rL $MODELS/Qwen3.5-0.8B /tmp/Qwen3.5-0.8B; echo "COPY_S $(( $(date +%s) - T ))"; du -sh /tmp/Qwen3.5-0.8B
export CONVERT_STACK_EVERY=120; timeout 900 bash -c "source /fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench/launch/miles/common.sh; convert_ckpt qwen3.5-0.8B /tmp/Qwen3.5-0.8B /tmp/q08_torch_dist 2"; echo "LOCAL_EXIT $?"
'
