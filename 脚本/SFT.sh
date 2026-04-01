set -e
export HF_ENDPOINT=https://hf-mirror.com
# 设置 Transformers/Datasets/Accelerate 的缓存目录
export HF_HOME=/root/autodl-tmp/huggingface

export DATASETS_CACHE=/root/autodl-tmp/huggingface/datasets
export HF_DATASETS_CACHE=/root/autodl-tmp/huggingface/datasets
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/root/trl:${PYTHONPATH}
export TRL_DEBUG_PREPARED_SAMPLES=8
export TRL_DEBUG_COLLATOR_BATCHES=4
export TRL_DEBUG_COLLATOR_EXAMPLES=4
export TRL_DEBUG_JSON_PATH=/root/trl/debug/sft_debug.jsonl

mkdir -p /root/trl/debug
rm -f "$TRL_DEBUG_JSON_PATH"




MODEL_PATH=Qwen/Qwen3.5-0.8B
DATA_PATH=BAAI/Infinity-Preference
OUTPUT_DIR=/root/autodl-tmp/Qwen2.5-1.5B-SFT-IP

python -m trl.scripts.sft \
    --model_name_or_path /root/autodl-tmp/Qwen2.5-1.5B-SFT-IF/checkpoint-464 \
    --dataset_name "$DATA_PATH" \
    --learning_rate 2.0e-5 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 32 \
    --eval_strategy steps \
    --eval_steps 50 \
    --output_dir /root/autodl-tmp/Qwen2.5-1.5B-SFT-IP \
    --push_to_hub \
    --report_to wandb \
    --dtype bfloat16 \
    --run_name IB-SFT-Qwen2.5-1.5B \
    --save_strategy steps \
    --save_steps 500 \
    --gradient_checkpointing False \
    --eos_token '<|im_end|>' \
    --completion_only_loss True \
    --lr_scheduler_type "cosine" \
    --packing True \
    --attn_implementation flash_attention_2 \
    --project SFT \

    
