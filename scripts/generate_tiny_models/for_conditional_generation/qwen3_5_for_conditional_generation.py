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
# - Qwen3.5 auto-builds layer_types from num_hidden_layers with default interval 4, so tiny models
#   (2 layers) end up all-linear-attention, which breaks dynamic cache. Force one full-attention layer.
# - The vision config expects `depth`/`num_heads` (not `num_hidden_layers`/`num_attention_heads`).
# - Qwen3.5 has no published generation_config on the Hub yet.
# - Qwen3.5 keeps some linear-attn weights in float32; we cast them back after the bfloat16 conversion.

import torch
from transformers import AutoConfig, AutoProcessor, Qwen3_5ForConditionalGeneration

from .._common import check_dtype_pattern, check_transformers_version, print_config_diff, push_to_hub, smoke_test


TRANSFORMERS_VERSION = "5.2.0"
check_transformers_version(TRANSFORMERS_VERSION)

MODEL_ID = "Qwen/Qwen3.5-0.8B"

processor = AutoProcessor.from_pretrained(MODEL_ID)

text_config = {
    "num_hidden_layers": 2,
    "hidden_size": 16,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "layer_types": ["linear_attention", "full_attention"],
    "full_attention_interval": 2,
}
vision_config = {
    "hidden_size": 16,
    "depth": 2,
    "num_heads": 4,
    "intermediate_size": 32,
    "out_hidden_size": 16,
}

config = AutoConfig.from_pretrained(MODEL_ID, text_config=text_config, vision_config=vision_config)
model = Qwen3_5ForConditionalGeneration(config).to(dtype=torch.bfloat16)

# Restore float32 for linear-attn weights that the upstream model keeps in fp32.
for i, layer_type in enumerate(config.text_config.layer_types):
    if layer_type == "linear_attention":
        linear_attn = model.model.language_model.layers[i].linear_attn
        linear_attn.A_log.data = linear_attn.A_log.data.float()
        linear_attn.norm.weight.data = linear_attn.norm.weight.data.float()

smoke_test(model, processor)
check_dtype_pattern(MODEL_ID, model)
print_config_diff(MODEL_ID, model)
push_to_hub(model, processor, None, "tiny")
