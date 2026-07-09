#!/usr/bin/env bash
# Launch the LoRA vLLM server on GPU 0.
#
# Usage:  bash ./trl/experimental/async_grpo/lora/example/run_vllm.sh

export UV_CACHE_DIR=/root/.cache/uv
export HF_HOME=/root/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface/hub
export HF_DATASETS_CACHE=/root/.cache/huggingface/datasets
export TRANSFORMERS_CACHE=/root/.cache/huggingface/hub 

export LD_LIBRARY_PATH=/root/test_trl/.venv/lib/python3.12/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
CUDA_VISIBLE_DEVICES=0 \
VLLM_SERVER_DEV_MODE=1 \
VLLM_ALLOW_RUNTIME_LORA_UPDATING=1 \
python -m trl.experimental.async_grpo.lora.run_vllm \
    --model Qwen/Qwen3-4B \
    --adapter /root/outputs/lora_adapter \
    --port 8800 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.9
