export HF_ENDPOINT=https://hf-mirror.com
# 设置 Transformers/Datasets/Accelerate 的缓存目录
export HF_HOME=/root/autodl-tmp/huggingface
export TRANSFORMERS_CACHE=/root/autodl-tmp/huggingface/models
export DATASETS_CACHE=/root/autodl-tmp/huggingface/datasets
export HF_DATASETS_CACHE=/root/autodl-tmp/huggingface/datasets

python3 step2DataGeneration.py \
    --output_path /root/autodl-tmp/imdb_pref_tokenized \
    --model_name_or_path /root/autodl-tmp/SFTOUT/checkpoint-9375 \