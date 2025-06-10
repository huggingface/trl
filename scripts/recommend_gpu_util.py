# ------------------------------------------------------------------------------
# Script: recommend_gpu_mem_util.py
# Description:
#     Estimates the GPU mem reqs for GRPO experiment w/ vllm 
#     It parses the model config (e.g., downloadable link https://huggingface.co/Qwen/Qwen2.5-Math-1.5B/resolve/main/config.json) and GRPO experiment config,
#     estimates model size, KV cache size, and buffer.
#     Outputs recommended GPU memory utilization.
#
# Usage:
#     python recommend_gpu_util.py \
#         --model_config path/to/model.json \
#         --exp_config path/to/experiment.yaml \
#         [--gpu_memory <in billions - e.g., 79>] \
#         [--model_params <in billions - e.g., 7.62>] \
#         [--precision_in_bytes 2] \
#         [--kv_multiplier 2]
#
# Required Parameters in model config:
#     - num_hidden_layers
#     - hidden_size
#     - num_attention_heads
#     - num_key_value_heads
#
# Required Parameters in experiment config (YAML):
#     - per_device_train_batch_size
#     - max_prompt_length
#     - max_completion_length
#     - (optional: vllm_tensor_parallel_size; defaults to 1)
#
# Dependencies:
#     - transformers (tested on version==4.45.2)
#     - accelerate (tested on version==0.28.0)
#     - pyyaml (tested on version==6.0.1)
# ------------------------------------------------------------------------------

import math
import yaml
import argparse
from pathlib import Path
from transformers import AutoConfig, AutoModelForCausalLM 
from accelerate import init_empty_weights

parser = argparse.ArgumentParser(description="Estimate GPU memory utilization for vLLM colocate mode")
parser.add_argument("--model_config", type=str, default="model.json", help="Path to model config JSON")
parser.add_argument("--exp_config", type=str, default="experiment.yaml", help="Path to experiment config YAML")
parser.add_argument("--gpu_memory", type=float, default=79, help="Total memory of a GPU in GB (default: 79)")
parser.add_argument("--model_params", type=float, default=None, help="Model parameters in billions (leave empty to estimate from model config)")
parser.add_argument("--precision_in_bytes", type=int, default=2, help="Precision in bytes (e.g., 2 for bfloat16/float16)")
parser.add_argument("--kv_multiplier", type=int, default=2, help="KV multiplier (usually 2 for K and V)")
args = parser.parse_args()

# Helper to require config values 
def must_get(source, key):
    if isinstance(source, dict):
        value = source.get(key)
    elif hasattr(source, key):
        value = getattr(source, key)
    else:
        value = None

    if value is None:
        raise ValueError(f"Required config '{key}' is missing!!!")
    return value


# Load model config
try:
    config = AutoConfig.from_pretrained(args.model_config)
except Exception as e:
    raise RuntimeError(f"Failed to load model config from {args.model_config}: {e}")

# Estimate no of params of a model if not provided (needed for model size estimation)
if args.model_params is None:
    try:
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config)
        num_params = sum(p.numel() for p in model.parameters())
        args.model_params = num_params / 1e9
        print(f"Estimated model_params from config: {args.model_params:.2f}B")
    except Exception as e:
        raise RuntimeError(f"Failed to estimate model parameters: {e}")

# Load experiment YAML
try:
    with open(Path(args.exp_config), "r") as f:
        exp_config = yaml.safe_load(f)
except Exception as e:
    raise RuntimeError(f"Failed to load experiment config from {args.exp_config}: {e}")

# Extract required config values
try:
    # model config
    num_hidden_layers = must_get(config, "num_hidden_layers")
    hidden_size = must_get(config, "hidden_size")
    num_attention_heads = must_get(config, "num_attention_heads")
    # in some models, each attention head has its own KV
    num_key_value_heads = getattr(config, "num_key_value_heads", num_attention_heads)

    # experiment config
    batch_size = must_get(exp_config, "per_device_train_batch_size")
    max_prompt_length = must_get(exp_config, "max_prompt_length")
    max_completion_length = must_get(exp_config, "max_completion_length")
    tp_size = exp_config.get("vllm_tensor_parallel_size", 1)
    seq_len = max_prompt_length + max_completion_length

    # Compute memory requirements
    model_size = args.model_params * 1024**3 * args.precision_in_bytes / tp_size
    
    # KV_cache_per_token = kv_multiplier (K and V) * num_hidden_layers * (num_key_value_heads * hidden_size / num_attention_heads) * precision_in_bytes
    kv_cache_per_token = (
        args.kv_multiplier
        * num_hidden_layers
        * (num_key_value_heads * hidden_size / num_attention_heads)
        * args.precision_in_bytes
    )

    # KV_cache_total = KV_cache_per_token * Batch_size * Seq_len (max_prompt_length + max_completion_length)
    kv_cache_total = kv_cache_per_token * batch_size * seq_len
    # Buffer = (Model + KV_cache) * 0.2  # generous 20% buffer
    buffer_size = 0.2 * (model_size + kv_cache_total)
    # Total = Model + KV_cache + Buffer
    total_required = model_size + kv_cache_total + buffer_size
    # GPU utilization = Total_reqd / Total_gpu
    gpu_memory_bytes = args.gpu_memory * 1024**3
    gpu_utilization_ratio = total_required / gpu_memory_bytes
    # Round up to nearest 0.05 - this generous estimate works much better than actual prediction!
    rounded_utilization = math.ceil(gpu_utilization_ratio * 20) / 20

    # All estimates
    ans = {
        "KV_cache_per_token_MB": kv_cache_per_token / 1024**2,
        "KV_cache_total_GB": kv_cache_total / 1024**3,
        "Model_size_GB": model_size / 1024**3,
        "Buffer_GB": buffer_size / 1024**3,
        "Total_required_GB": total_required / 1024**3,
        "GPU_mem_util": gpu_utilization_ratio,
        "GPU_mem_util_recommended": rounded_utilization,
    }

    for key, value in ans.items():
        print(f"{key}: {value:.2f}")

    # Final recommendation
    print(f"-------\nRecommended vLLM GPU memory utilization: {rounded_utilization:.2f}")


except Exception as e:
    raise RuntimeError(f"Error during calculation: {e}")
