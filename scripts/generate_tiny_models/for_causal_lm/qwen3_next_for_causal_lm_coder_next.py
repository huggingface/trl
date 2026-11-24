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

# Qwen3-Coder-Next ships the Qwen3-Coder chat template, with XML tool calls, on the Qwen3-Next hybrid architecture.

import torch
from transformers import AutoTokenizer, GenerationConfig, Qwen3NextConfig, Qwen3NextForCausalLM

from .._common import (
    check_dtype_pattern,
    check_transformers_version,
    init_weights_tiny_model,
    print_config_diff,
    push_to_hub,
    set_seed,
    smoke_test,
)


check_transformers_version("4.57.0")

set_seed()

MODEL_ID = "Qwen/Qwen3-Coder-Next"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
generation_config = GenerationConfig.from_pretrained(MODEL_ID)
config = Qwen3NextConfig(
    vocab_size=151936,
    hidden_size=8,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=8,  # a multiple of 8, as Flash Attention requires
    # 4 layers, so that the reference full_attention_interval of 4 gives 3 linear attention layers and 1 full attention
    num_hidden_layers=4,
    intermediate_size=32,
    linear_num_key_heads=2,
    linear_num_value_heads=4,
    linear_key_head_dim=8,
    linear_value_head_dim=8,
    num_experts=4,
    num_experts_per_tok=2,
    moe_intermediate_size=32,
    shared_expert_intermediate_size=32,
    max_position_embeddings=262144,
    rope_theta=5000000,
    bos_token_id=151643,
    eos_token_id=151645,
    # Forwarded via kwargs (not Qwen3NextConfig fields, but PretrainedConfig accepts arbitrary kwargs):
    full_attention_interval=4,
    use_sliding_window=False,
)
model = Qwen3NextForCausalLM(config).to(dtype=torch.bfloat16)
init_weights_tiny_model(model)
smoke_test(model, tokenizer)
check_dtype_pattern(MODEL_ID, model)
print_config_diff(MODEL_ID, model)
push_to_hub(model, tokenizer, generation_config, "tiny", "Coder-Next")
