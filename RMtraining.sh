set -e
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/huggingface

# 性能加速环境变量
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 定义路径
MODEL_PATH="Qwen/Qwen2.5-0.5B"
DATA_PATH="Anthropic/hh-rlhf"
OUTPUT_DIR="/root/autodl-tmp/Qwen2-0.5B-RM-HH"

# 提速核心参数调优：
# 1. per_device_train_batch_size: 0.5B模型在32GB下可以直接拉到 32 甚至 64。
# 2. gradient_accumulation_steps: 配合上面的BS，设为 4，达到总 Batch 128。
# 3. max_length: 1024 覆盖 HH 数据集绝大部分样本。
# 4. remove_unused_columns: 必须设为 False，否则自定义的 chosen/rejected 会被丢弃。

python /root/trl/trl/scripts/Qwen2.5_Reward_HH.py \
    --model_name_or_path "$MODEL_PATH" \
    --dataset_name "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --learning_rate 3e-6 \
    --lr_scheduler_type cosine \
    --num_train_epochs 2 \
    --max_length 1024 \
    --bf16 True \
    --attn_implementation "flash_attention_2" \
    --gradient_checkpointing False \
    --remove_unused_columns False \
    --logging_steps 10 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 800 \
    --report_to wandb \
    --run_name "Qwen2.5-0.5B-RM-FAST" \
    --push_to_hub True \
    --hub_model_id "chenyongxi/Qwen2.5-0.5B-RM-HH" \