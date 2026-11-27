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

import torch
from transformers import AutoConfig, AutoProcessor, GenerationConfig, Qwen4ExpForConditionalGeneration

from .._common import (
    check_dtype_pattern,
    check_transformers_version,
    print_config_diff,
    push_to_hub,
    set_seed,
    smoke_test,
)


check_transformers_version("5.17.0")

set_seed()

MODEL_ID = "Qwen/Qwen3.8-Flash-Next"

processor = AutoProcessor.from_pretrained(MODEL_ID)
generation_config = GenerationConfig.from_pretrained(MODEL_ID)

text_config = {
    "hidden_size": 32,
    # 4 layers, so the reference pattern (3 linear attention layers, then a sparse attention one) appears once
    "num_hidden_layers": 4,
    "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "linear_num_key_heads": 2,
    "linear_num_value_heads": 4,
    "linear_key_head_dim": 16,
    "linear_value_head_dim": 16,
    "num_experts": 4,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 32,
    "shared_expert_intermediate_size": 32,
    "hc_lowrank": 8,
    "ple_embed_dim": 32,
    "ngram_vocab_size_base": 1000,
    "split_ngram_parts": 2,  # keeps the reference's n-gram table split into shards
    "indexer_n_heads": 2,
    "indexer_head_dim": 64,  # must hold the attention's rotary dims: head_dim 256 * partial_rotary_factor 0.25
    # Smaller than the test sequences, so the sparse attention path actually drops tokens
    "indexer_budget": 8,
}
vision_config = {
    "hidden_size": 16,
    "depth": 2,
    "num_heads": 2,  # head size 16 // 2 = 8, as Flash Attention requires
    "intermediate_size": 32,
    "out_hidden_size": 32,
}

config = AutoConfig.from_pretrained(MODEL_ID, text_config=text_config, vision_config=vision_config)
model = Qwen4ExpForConditionalGeneration(config).to(dtype=torch.bfloat16)
smoke_test(model, processor)
check_dtype_pattern(MODEL_ID, model)
print_config_diff(MODEL_ID, model)
push_to_hub(model, processor, generation_config, "tiny")
