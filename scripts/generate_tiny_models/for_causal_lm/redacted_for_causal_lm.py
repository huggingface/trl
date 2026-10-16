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
from transformers import AutoConfig, AutoTokenizer, GenerationConfig
from transformers.models.redacted import REDACTEDConfig, REDACTEDForCausalLM

from .._common import (
    check_dtype_pattern,
    check_transformers_version,
    init_weights_tiny_model,
    print_config_diff,
    push_to_hub,
    set_seed,
    smoke_test,
)


check_transformers_version("5.20.1")

set_seed()

MODEL_ID = "REDACTED_HUB_ID"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
generation_config = GenerationConfig.from_pretrained(MODEL_ID)
reference = AutoConfig.from_pretrained(MODEL_ID)
config = REDACTEDConfig(
    vocab_size=reference.vocab_size,
    hidden_size=16,
    intermediate_size=32,
    moe_intermediate_size=8,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    num_experts=8,
    num_experts_per_tok=2,
    # Group-limited routing needs at least 2 experts per group (the group score sums the top 2)
    n_group=2,
    topk_group=1,
    sliding_window=4,
    # One dense layer then sparse ones, and both attention types, with a sliding window shorter than the test inputs
    mlp_layer_types=["dense", "sparse", "sparse", "sparse"],
    layer_types=["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"],
    rope_parameters=reference.rope_parameters,
    final_logit_softcapping=reference.final_logit_softcapping,
    routed_scaling_factor=reference.routed_scaling_factor,
    max_position_embeddings=reference.max_position_embeddings,
    tie_word_embeddings=reference.tie_word_embeddings,
)
model = REDACTEDForCausalLM(config).to(dtype=torch.bfloat16)
init_weights_tiny_model(model)
# The routers stay in float32, as in the reference checkpoint
for name, module in model.named_modules():
    if name.endswith("mlp.gate"):
        module.float()
smoke_test(model, tokenizer)
check_dtype_pattern(MODEL_ID, model)
print_config_diff(MODEL_ID, model)
push_to_hub(model, tokenizer, generation_config, "tiny")
