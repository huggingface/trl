#!/bin/bash
set -e
source /etc/network_turbo
# 设置 Transformers/Datasets/Accelerate 的缓存目录
export HF_HOME=/root/autodl-tmp/huggingface
export TRANSFORMERS_CACHE=/root/autodl-tmp/huggingface/models
export DATASETS_CACHE=/root/autodl-tmp/huggingface/datasets
export HF_DATASETS_CACHE=/root/autodl-tmp/huggingface/datasets

export OMP_NUM_THREADS=8
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export CUDA_VISIBLE_DEVICES=0
export WANDB_PROJECT=huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
wandb login


accelerate launch --config_file examples/accelerate_configs/deepspeed_zero2.yaml \
    examples/scripts/ppo/ppo_ip.py \
    --dataset_name BAAI/Infinity-Preference \
    --output_dir /root/autodl-tmp/Qwen2.5-1.5B-PPO-IP \
    --num_mini_batches 1 \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 512 \
    --total_episodes 200000 \
    --model_name_or_path /root/autodl-tmp/Qwen2-1.5B-SFT-IF/checkpoint-1392 \
    --sft_model_path /root/autodl-tmp/Qwen2-1.5B-SFT-IF/checkpoint-1392 \
    --reward_model_path /root/autodl-tmp/Qwen2-1.5B-RM-IP/checkpoint-796 \
    --local_rollout_forward_batch_size 4 \
    --missing_eos_penalty 1.0 \
    --lr_scheduler_type linear \
    --whiten_rewards True \
    --report_to wandb \
    --run_name Qwen2.5-1.5B-PPO-IP \
    --num_ppo_epochs 2 \
    --push_to_hub True \
    --hub_model_id chenyongxi/Qwen2.5-1.5B-PPO-IP \
    --logging_steps 10 \
    --bf16 True \
    --dtype bfloat16 \
    --eval_steps 100 \
    --eval_strategy steps \
    --attn_implementation flash_attention_2 \
    --save_steps 652 \
    --save_strategy steps \
    --save_only_model True \
    --gradient_checkpointing True \
    --response_length 256 \

