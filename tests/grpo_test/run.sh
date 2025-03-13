#!/bin/bash

# Make only GPUs 3,4,5,6 visible to the application
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Set specific environment variables needed for distributed training
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=INFO

# Launch the training using accelerate
accelerate launch --config_file=trl/scripts/grpo_test/grpo_sgl_test.yaml trl/scripts/grpo_test/grpo_sgl_test.py

# Under the ~/trl/tests/grpo_test
# python manual_grpo_launch.py --config_file=grpo_sgl_test.yaml