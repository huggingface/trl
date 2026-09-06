# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Notes:
# - Qwen3.6 reuses the Qwen3_5Moe class with extra MoE config fields
#   (num_experts, num_experts_per_tok, moe_intermediate_size, shared_expert_intermediate_size).
# - Same layer_types/full_attention_interval workaround as Qwen3.5: tiny models (2 layers) need
#   one full-attention layer to keep the dynamic cache happy.
# - The vision config expects `depth`/`num_heads` (not `num_hidden_layers`/`num_attention_heads`).
# - Unlike Qwen3.5, Qwen3.6 stores linear-attn weights in bf16, so no float32 cast is needed.

import torch
from transformers import AutoConfig, AutoProcessor, GenerationConfig, Qwen3_5MoeForConditionalGeneration

from .._common import check_dtype_pattern, check_transformers_version, print_config_diff, push_to_hub, smoke_test


TRANSFORMERS_VERSION = "5.2.0"
check_transformers_version(TRANSFORMERS_VERSION)

MODEL_ID = "Qwen/Qwen3.6-35B-A3B"

processor = AutoProcessor.from_pretrained(MODEL_ID)
generation_config = GenerationConfig.from_pretrained(MODEL_ID)

text_config = {
    "num_hidden_layers": 2,
    "hidden_size": 16,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "layer_types": ["linear_attention", "full_attention"],
    "full_attention_interval": 2,
    "num_experts": 4,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 32,
    "shared_expert_intermediate_size": 32,
}
vision_config = {
    "hidden_size": 16,
    "depth": 2,
    "num_heads": 4,
    "intermediate_size": 32,
    "out_hidden_size": 16,
}

config = AutoConfig.from_pretrained(MODEL_ID, text_config=text_config, vision_config=vision_config)
model = Qwen3_5MoeForConditionalGeneration(config).to(dtype=torch.bfloat16)

smoke_test(model, processor)
check_dtype_pattern(MODEL_ID, model)
print_config_diff(MODEL_ID, model)
push_to_hub(model, processor, generation_config, "tiny", "3.6")
