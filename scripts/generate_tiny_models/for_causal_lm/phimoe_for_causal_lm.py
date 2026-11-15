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
from transformers import AutoTokenizer, GenerationConfig, PhimoeConfig, PhimoeForCausalLM

from .._common import (
    check_dtype_pattern,
    check_transformers_version,
    init_weights_tiny_model,
    print_config_diff,
    push_to_hub,
    set_seed,
    smoke_test,
)


check_transformers_version()

set_seed()

MODEL_ID = "microsoft/Phi-3.5-MoE-instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
generation_config = GenerationConfig.from_pretrained(MODEL_ID)
config = PhimoeConfig(
    vocab_size=32064,
    hidden_size=8,
    num_attention_heads=4,
    num_key_value_heads=2,
    num_hidden_layers=2,
    intermediate_size=32,
    num_local_experts=4,
    num_experts_per_tok=2,
    router_aux_loss_coef=0.0,
    max_position_embeddings=131072,
    rope_theta=10000.0,
    original_max_position_embeddings=4096,
    sliding_window=131072,
    # One value per frequency band: length is `head_dim // 2`, values are the reference's first band.
    rope_scaling={
        "type": "longrope",
        "short_factor": [1.0],
        "long_factor": [1.0199999809265137],
        "short_mscale": 1.243163121016122,
        "long_mscale": 1.243163121016122,
        "original_max_position_embeddings": 4096,
    },
    input_jitter_noise=0.01,
    attention_bias=True,
    lm_head_bias=True,
    eos_token_id=32000,
    # Reference-only key: `PhimoeConfig` does not model it.
    hidden_dropout=0.0,
)
model = PhimoeForCausalLM(config).to(dtype=torch.bfloat16)
init_weights_tiny_model(model)
smoke_test(model, tokenizer)
check_dtype_pattern(MODEL_ID, model)
print_config_diff(MODEL_ID, model)
push_to_hub(model, tokenizer, generation_config, "tiny")
