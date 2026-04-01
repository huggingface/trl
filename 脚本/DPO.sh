#!/bin/bash
set -e

# 设置 Transformers/Datasets/Accelerate 的缓存目录
export HF_HOME=/root/autodl-tmp/huggingface
export TRANSFORMERS_CACHE=/root/autodl-tmp/huggingface/hub
export DATASETS_CACHE=/root/autodl-tmp/huggingface/datasets
export HF_DATASETS_CACHE=/root/autodl-tmp/huggingface/datasets

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0
source /etc/network_turbo

MODEL_PATH="/root/autodl-tmp/Qwen2-1.5B-SFT-IF/checkpoint-1392"
DATA_PATH="BAAI/Infinity-Preference"
OUTPUT_DIR="/root/autodl-tmp/Qwen2.5-1.5B-DPO-1.5B"



python dpo_ip.py \
    --model_name_or_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --dataset_name $DATA_PATH \
    --bf16 True \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --learning_rate 3e-6 \
    --gradient_checkpointing True \
    --attn_implementation "flash_attention_2" \
    --run_name "Qwen2.5-1.5B-DPO" \
    --report_to wandb \
    --push_to_hub True \
    --eval_strategy steps \
    --eval_steps 50 \
    --num_train_epochs 2 \
    --save_only_model True \
    --save_steps 500 \
    --gradient_checkpointing False \
    --beta 0.05 \
    --dtype bfloat16 \