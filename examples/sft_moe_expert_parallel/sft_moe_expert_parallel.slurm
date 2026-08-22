#!/bin/bash
# Multi-node launcher; submit from this directory:
#   sbatch sft_moe_expert_parallel.slurm --model zai-org/GLM-5.2 --ep 32 --fsdp 2 --lora 16
#SBATCH --nodes=8
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=04:00:00

MASTER=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
# ahostsv4 avoids link-local IPv6 addresses, which torchrun cannot rendezvous on
MASTER_IP=$(getent ahostsv4 "$MASTER" | awk 'NR==1{print $1}')

srun torchrun \
    --nnodes "$SLURM_NNODES" \
    --nproc_per_node 8 \
    --node_rank "$SLURM_NODEID" \
    --rdzv_backend c10d \
    --rdzv_endpoint "$MASTER_IP:29700" \
    "$SLURM_SUBMIT_DIR/sft_moe_expert_parallel.py" "$@"
