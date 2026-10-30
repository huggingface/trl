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
from transformers import AutoTokenizer, GenerationConfig, SmolLM3Config, SmolLM3ForCausalLM

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

MODEL_ID = "HuggingFaceTB/SmolLM3-3B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
generation_config = GenerationConfig.from_pretrained(MODEL_ID)
config = SmolLM3Config(
    vocab_size=128256,
    hidden_size=8,
    num_attention_heads=4,
    num_key_value_heads=2,
    # Four layers so that the last one skips RoPE (no_rope_layer_interval=4), as every fourth layer of the reference
    num_hidden_layers=4,
    intermediate_size=32,
    tie_word_embeddings=True,
    max_position_embeddings=65536,
    rope_theta=5000000.0,
    rms_norm_eps=1e-06,
    pretraining_tp=2,
    use_cache=False,
    bos_token_id=128000,
    eos_token_id=128012,
    pad_token_id=128004,
)
model = SmolLM3ForCausalLM(config).to(dtype=torch.bfloat16)
init_weights_tiny_model(model)
smoke_test(model, tokenizer)
check_dtype_pattern(MODEL_ID, model)
print_config_diff(MODEL_ID, model)
push_to_hub(model, tokenizer, generation_config, "tiny")
