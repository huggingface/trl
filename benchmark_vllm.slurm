#!/bin/bash
#SBATCH --job-name=trl-vllm-math
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --partition=hopper-prod  # Adjust this for your cluster
#SBATCH --output=./logs/%x-%j.out
#SBATCH --err=./logs/%x-%j.err
#SBATCH --requeue

# Specific configuration optimized for the Hugging Face Compute Cluster
module load cuda/12.4
set -x -e

source ~/.bashrc
source trl-dev2/bin/activate
echo "START TIME: $(date)"

MODEL=$1
USE_VLLM_LOGPROBS=$2

CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model $MODEL &


CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 accelerate launch --config_file=examples/accelerate_configs/deepspeed_zero3.yaml --num_processes=7 run_grpo.py --model $MODEL --use_vllm_logprobs $USE_VLLM_LOGPROBS