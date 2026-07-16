#!/usr/bin/env bash
# Launch the LoRA async GRPO trainer on GPU 1.
#
# Usage:  bash ./trl/experimental/async_grpo/lora/example/run_trainer.sh

set -a; source ../.env; set +a;
wandb login

export UV_CACHE_DIR=/root/.cache/uv
export HF_HOME=/root/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface/hub
export HF_DATASETS_CACHE=/root/.cache/huggingface/datasets
export TRANSFORMERS_CACHE=/root/.cache/huggingface/hub 

export LD_LIBRARY_PATH=/root/test_trl/.venv/lib/python3.12/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH

CUDA_VISIBLE_DEVICES=1 \
accelerate launch -m trl.experimental.async_grpo.lora.example.train \
    &> /tmp/lora_train.log 2>&1
