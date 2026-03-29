set -e
export HF_ENDPOINT=https://hf-mirror.com
# 设置 Transformers/Datasets/Accelerate 的缓存目录
export HF_HOME=/root/autodl-tmp/huggingface

export DATASETS_CACHE=/root/autodl-tmp/huggingface/datasets
export HF_DATASETS_CACHE=/root/autodl-tmp/huggingface/datasets

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0

MODEL_PATH=Qwen/Qwen2.5-0.5B
DATA_PATH=Anthropic/hh-rlhf
OUTPUT_DIR=/root/autodl-tmp/HH-SFT-EXP

python trl/scripts/sft.py \
    --model_name_or_path Qwen/Qwen2.5-0.5B \
    --dataset_name "$DATA_PATH" \
    --learning_rate 2.0e-5 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 16 \
    --eval_strategy steps \
    --eval_steps 500 \
    --output_dir /root/autodl-tmp/Qwen2-0.5B-SFT-HH \
    --push_to_hub \
    --attn_implementation "flash_attention_2" \
    --report_to wandb \
    --dtype bfloat16 \
    --run_name HH-SFT-Qwen2.5-0.5B \
    --packing True \
    --save_strategy no \
