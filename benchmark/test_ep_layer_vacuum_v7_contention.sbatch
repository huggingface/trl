#!/bin/bash
#SBATCH --job-name=ep-vacuum-v7-contention
#SBATCH --partition=hopper-prod
#SBATCH --qos=normal
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:8
#SBATCH --exclusive
#SBATCH --time=00:45:00
#SBATCH --output=/fsx/amine_dirhoussi/trl/benchmark/logs/ep-vacuum-v7-contention-%j.out
#SBATCH --error=/fsx/amine_dirhoussi/trl/benchmark/logs/ep-vacuum-v7-contention-%j.err

source /fsx/amine_dirhoussi/trl/.venv/bin/activate

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export NCCL_DEBUG=WARN
export TORCH_NCCL_TRACE_BUFFER_SIZE=2048
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=600
export NCCL_HEARTBEAT_TIMEOUT_SEC=600
export TRITON_CACHE_DIR="/tmp/triton-rank-${SLURM_PROCID:-0}-${HOSTNAME}"
mkdir -p "$TRITON_CACHE_DIR"
mkdir -p /fsx/amine_dirhoussi/nccl-traces
export TORCH_NCCL_DEBUG_INFO_TEMP_FILE="/fsx/amine_dirhoussi/nccl-traces/vacuum_v5_${SLURM_JOB_ID}_rank_${SLURM_PROCID}_${HOSTNAME}"

srun --jobid $SLURM_JOB_ID bash -c '
torchrun \
  --nnodes=2 \
  --nproc-per-node=8 \
  --rdzv-backend=c10d \
  --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
  --node-rank=$SLURM_PROCID \
  /fsx/amine_dirhoussi/trl/benchmark/test_ep_layer_vacuum_v7_contention.py
'
