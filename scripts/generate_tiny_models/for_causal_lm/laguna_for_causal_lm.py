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
from transformers import AutoTokenizer, GenerationConfig, LagunaConfig, LagunaForCausalLM

from .._common import (
    check_dtype_pattern,
    check_transformers_version,
    init_weights_tiny_model,
    print_config_diff,
    push_to_hub,
    set_seed,
    smoke_test,
)


# Laguna was added in transformers 5.7.0, but building it fails at random before 5.16.0 (huggingface/transformers#48162)
check_transformers_version("5.16.0")

set_seed()

MODEL_ID = "poolside/Laguna-XS.2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
generation_config = GenerationConfig.from_pretrained(MODEL_ID)
config = LagunaConfig(
    vocab_size=100352,
    hidden_size=8,
    intermediate_size=32,
    # 2 layers: the reference starts with a full attention + dense MLP layer, followed by sliding attention + MoE layers
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=8,  # a multiple of 8, as Flash Attention requires
    num_attention_heads_per_layer=[4, 8],
    layer_types=["full_attention", "sliding_attention"],
    mlp_layer_types=["dense", "sparse"],
    max_position_embeddings=262144,
    attention_bias=False,
    attention_dropout=0.0,
    rms_norm_eps=1e-6,
    num_experts=4,
    num_experts_per_tok=2,
    moe_intermediate_size=32,
    shared_expert_intermediate_size=32,
    router_aux_loss_coef=0.0,
    moe_apply_router_weight_on_input=False,
    moe_routed_scaling_factor=2.5,
    bos_token_id=2,
    eos_token_id=[2, 24],
    pad_token_id=9,
    tie_word_embeddings=False,
    use_cache=True,
    gating=True,
    sliding_window=512,
    rope_parameters={
        "full_attention": {
            "rope_theta": 500000.0,
            "rope_type": "yarn",
            "factor": 64.0,
            "original_max_position_embeddings": 4096,
            "beta_slow": 1.0,
            "beta_fast": 64.0,
            "attention_factor": 1.4158883083359672,
            "partial_rotary_factor": 0.5,
        },
        "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 1.0},
    },
)
model = LagunaForCausalLM(config).to(dtype=torch.bfloat16)
init_weights_tiny_model(model)
smoke_test(model, tokenizer)
check_dtype_pattern(MODEL_ID, model)
print_config_diff(MODEL_ID, model)
push_to_hub(model, tokenizer, generation_config, "tiny")
