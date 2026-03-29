#!/bin/bash
set -e
export HF_ENDPOINT=https://hf-mirror.com
# 设置 Transformers/Datasets/Accelerate 的缓存目录
export HF_HOME=/root/autodl-tmp/huggingface
export TRANSFORMERS_CACHE=/root/autodl-tmp/huggingface/hub
export DATASETS_CACHE=/root/autodl-tmp/huggingface/datasets
export HF_DATASETS_CACHE=/root/autodl-tmp/huggingface/datasets

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0

MODEL_PATH="cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr"
DATA_PATH="openai/summarize_from_feedback"
OUTPUT_DIR="/root/autodl-tmp/TLDR"



python step3DPO.py \
    --model_name_or_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --bf16 True \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 32 \
    --learning_rate 5e-6 \
    --dataset_name "tldr" \
    --gradient_checkpointing True \
    --attn_implementation "flash_attention_2" \
    --run_name "DPO-TLDR-1B-2st" \
    --save_only_model True \
    --save_steps 500