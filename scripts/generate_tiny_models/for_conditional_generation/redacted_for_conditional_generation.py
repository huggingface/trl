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
from transformers import AutoConfig, AutoProcessor, GenerationConfig
from transformers.models.redacted import REDACTEDForConditionalGeneration

from .._common import (
    check_dtype_pattern,
    check_transformers_version,
    init_weights_tiny_model,
    print_config_diff,
    push_to_hub,
    set_seed,
    smoke_test,
)


check_transformers_version("5.22.0")

set_seed()

MODEL_ID = "REDACTED_HUB_ID"

processor = AutoProcessor.from_pretrained(MODEL_ID)
generation_config = GenerationConfig.from_pretrained(MODEL_ID)

text_config = {
    # 8 layers, so that each per-layer pattern of the reference appears at least once: a dense first layer, NoPE every
    # 4th layer, a layer reading an earlier layer's KV, and indexer layers that reuse the previous layer's top-k.
    "num_hidden_layers": 8,
    "kv_source_layers": [0, 1, 2, 3, 4, 5, 6, 3],
    "indexer_types": ["full", "full", "full", "shared", "shared", "full", "full", "shared"],
    # Derived from `num_hidden_layers` when unset
    "layer_types": None,
    "mlp_layer_types": None,
    "no_rope_layers": None,
    "hidden_size": 32,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "intermediate_size": 32,
    "kv_lora_rank": 16,
    "qk_rope_head_dim": 8,
    "qk_nope_head_dim": 8,
    "v_head_dim": 8,
    "n_routed_experts": 8,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 16,
    "index_n_heads": 2,
    "index_head_dim": 16,
    # Smaller than the test sequences, so the sparse attention path actually drops keys.
    "index_topk": 4,
}
vision_config = {
    "num_hidden_layers": 2,
    "hidden_size": 16,
    "num_attention_heads": 4,
    "head_dim": 4,
    "intermediate_size": 32,
}

config = AutoConfig.from_pretrained(MODEL_ID, text_config=text_config, vision_config=vision_config)
# The reference checkpoint is FP8; the tiny model is a plain bfloat16 one
del config.quantization_config
model = REDACTEDForConditionalGeneration(config).to(dtype=torch.bfloat16)
init_weights_tiny_model(model)

# Restore float32 for the router and the learned RoPE scaling, which the reference keeps in fp32
for layer in model.model.language_model.layers[config.text_config.first_k_dense_replace :]:
    layer.mlp.gate.weight.data = layer.mlp.gate.weight.data.float()
model.model.language_model.rotary_emb.alpha.data = model.model.language_model.rotary_emb.alpha.data.float()
smoke_test(model, processor)
check_dtype_pattern(MODEL_ID, model)
print_config_diff(MODEL_ID, model)
push_to_hub(model, processor, generation_config, "tiny")
