export HF_ENDPOINT=https://hf-mirror.com
# 设置 Transformers/Datasets/Accelerate 的缓存目录
export HF_HOME=/root/autodl-tmp/huggingface
export TRANSFORMERS_CACHE=/root/autodl-tmp/huggingface/models
export DATASETS_CACHE=/root/autodl-tmp/huggingface/datasets
export HF_DATASETS_CACHE=/root/autodl-tmp/huggingface/datasets
export WANDB_PROJECT=trl-sft


accelerate launch --config_file examples/accelerate_configs/deepspeed_zero2.yaml \
    /root/trl/step1SFT.py \
    --dataset_name Anthropic/hh-rlhf \
    --model_name_or_path Qwen/Qwen2.5-1.5B \
    --output_path /root/autodl-tmp/HH-SFTOUT \
    --exp_name HH-SFT031901 \
    --seed 42

