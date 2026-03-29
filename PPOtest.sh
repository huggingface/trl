#!/bin/bash
set -e
export HF_ENDPOINT=https://hf-mirror.com
# 设置 Transformers/Datasets/Accelerate 的缓存目录
export HF_HOME=/root/autodl-tmp/huggingface
export TRANSFORMERS_CACHE=/root/autodl-tmp/huggingface/models
export DATASETS_CACHE=/root/autodl-tmp/huggingface/datasets
export HF_DATASETS_CACHE=/root/autodl-tmp/huggingface/datasets




python test.py