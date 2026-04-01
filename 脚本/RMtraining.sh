set -e
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/huggingface

# 性能加速环境变量
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 定义路径
MODEL_PATH="/root/autodl-tmp/Qwen2-1.5B-SFT-IF/checkpoint-1392"
DATA_PATH="BAAI/Infinity-Preference"
OUTPUT_DIR="/root/autodl-tmp/Qwen2-1.5B-RM-IP"



# 提速核心参数调优：
# 1. per_device_train_batch_size: 0.5B模型在32GB下可以直接拉到 32 甚至 64。
# 2. gradient_accumulation_steps: 配合上面的BS，设为 4，达到总 Batch 128。
# 3. max_length: 1024 覆盖 HH 数据集绝大部分样本。
# 4. remove_unused_columns: 必须设为 False，否则自定义的 chosen/rejected 会被丢弃。

python /root/trl/trl/scripts/Qwen2.5-1.5B-reward.py \
    --model_name_or_path "$MODEL_PATH" \
    --dataset_name "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 32 \
    --learning_rate 3e-6 \
    --lr_scheduler_type cosine \
    --num_train_epochs 1 \
    --bf16 True \
    --attn_implementation "flash_attention_2" \
    --gradient_checkpointing False \
    --logging_steps 10 \
    --eval_strategy "steps" \
    --eval_steps 50 \
    --save_strategy "epoch" \
    --report_to wandb \
    --project graduate_exp \
    --run_name "Qwen2.5-1.5B-RM" \
    --push_to_hub True \
    --hub_model_id "chenyongxi/Qwen2.5-1.5B-RM-IP" \
    --dataset_train_split "train" \
    --dataset_test_split "test" \
    --eos_token "<|im_end|>" \
    