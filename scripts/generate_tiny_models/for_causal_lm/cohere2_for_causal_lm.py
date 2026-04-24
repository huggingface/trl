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
from transformers import AutoConfig, AutoTokenizer, Cohere2Config, Cohere2ForCausalLM, GenerationConfig

from .._common import (
    check_dtype_pattern,
    check_transformers_version,
    init_weights_tiny_model,
    print_config_diff,
    push_to_hub,
    smoke_test,
)


TRANSFORMERS_VERSION = "4.56.2"
check_transformers_version(TRANSFORMERS_VERSION)

MODEL_ID = "CohereLabs/tiny-aya-earth"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
generation_config = GenerationConfig.from_pretrained(MODEL_ID)
config = Cohere2Config(
    vocab_size=AutoConfig.from_pretrained(MODEL_ID).vocab_size,
    hidden_size=8,
    num_attention_heads=4,
    num_key_value_heads=2,
    num_hidden_layers=2,
    intermediate_size=32,
    bos_token_id=2,
    eos_token_id=3,
    logit_scale=1.0,
    max_position_embeddings=500000,
    rope_theta=50000,
    cache_implementation="hybrid",
    layer_switch=4,
    order_of_interleaved_layers="local_attn_first",
    position_embedding_type="rope_gptj",
    rotary_pct=1.0,
    use_embedding_sharing=True,
    use_gated_activation=True,
    use_parallel_block=True,
    use_parallel_embedding=False,
    use_qk_norm=False,
)
model = Cohere2ForCausalLM(config).to(dtype=torch.bfloat16)
init_weights_tiny_model(model)
smoke_test(model, tokenizer)
check_dtype_pattern(MODEL_ID, model)
print_config_diff(MODEL_ID, model)
push_to_hub(model, tokenizer, generation_config, "tiny")
