#!/bin/bash
set -e
export HF_ENDPOINT=https://hf-mirror.com
# 设置 Transformers/Datasets/Accelerate 的缓存目录
export HF_HOME=/root/autodl-tmp/huggingface
export TRANSFORMERS_CACHE=/root/autodl-tmp/huggingface/models
export DATASETS_CACHE=/root/autodl-tmp/huggingface/datasets
export HF_DATASETS_CACHE=/root/autodl-tmp/huggingface/datasets

export OMP_NUM_THREADS=8
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export CUDA_VISIBLE_DEVICES=0
export WANDB_PROJECT=trl-sft
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_API_KEY='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'
wandb login


accelerate launch --config_file examples/accelerate_configs/deepspeed_zero2.yaml \
    examples/scripts/ppo/ppo.py \
    --dataset_name openai/summarize_from_feedback \
    --dataset_train_split train \
    --output_dir /root/autodl-tmp/pythia-1b-deduped-summary-trl-style-ppo \
    --num_ppo_epochs 1 \
    --num_mini_batches 1 \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --total_episodes 10000 \
    --model_name_or_path cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr \
    --sft_model_path cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr \
    --reward_model_path cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr \
    --local_rollout_forward_batch_size 2 \
    --missing_eos_penalty 1.0 \
    --lr_scheduler_type linear \
    --batch_size 512 \
    --whiten_rewards True \
