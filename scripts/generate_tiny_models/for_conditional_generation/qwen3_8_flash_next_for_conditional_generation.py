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
# - Qwen3.8-Flash-Next runs on transformers' Qwen4Exp classes.
# - Each step that reads earlier tokens is present: gated DeltaNet, the n-gram shift and PLE short conv of the
#   per-layer embeddings, and the sparse token selection of the indexed-attention layer.
# - `indexer_budget` is smaller than the test sequences, so the sparse token selection actually drops blocks.
# - Pinned to 5.18.0, the first release that names the indexed layers `indexed_attention` (`qwen_sparse_attention`
#   before it), the name that later releases read. It saves the experts one by one, where the reference fuses them, so
#   the dtype check finds no counterpart for the expert weights.

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


check_transformers_version("5.18.0")

set_seed()

MODEL_ID = "Qwen/Qwen3.8-Flash-Next"

processor = AutoProcessor.from_pretrained(MODEL_ID)
generation_config = GenerationConfig.from_pretrained(MODEL_ID)

text_config = {
    "hidden_size": 32,
    # 4 layers, so the reference's per-layer pattern appears: 3 linear-attention layers then an indexed-attention one,
    # with the per-layer embeddings (PLE) on the second (`ple_layer_ids` is one-indexed)
    "num_hidden_layers": 4,
    "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
    "num_attention_heads": 4,
    "linear_num_key_heads": 2,
    "linear_num_value_heads": 4,
    "hc_lowrank": 8,
    "ple_embed_dim": 32,
    "ngram_vocab_size_base": 30,
    "split_ngram_parts": 1,
    "num_experts": 4,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 32,
    "shared_expert_intermediate_size": 32,
    "indexer_budget": 8,
}
vision_config = {
    "hidden_size": 16,
    "depth": 2,
    "num_heads": 4,
    "intermediate_size": 32,
    "out_hidden_size": 32,
}

config = AutoConfig.from_pretrained(MODEL_ID, text_config=text_config, vision_config=vision_config)
model = Qwen4ExpForConditionalGeneration(config).to(dtype=torch.bfloat16)

smoke_test(model, processor)
check_dtype_pattern(MODEL_ID, model)
print_config_diff(MODEL_ID, model)
push_to_hub(model, processor, generation_config, "tiny", "3.8-Flash-Next")
