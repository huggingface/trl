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
from transformers import AutoTokenizer, GenerationConfig, Lfm2MoeConfig, Lfm2MoeForCausalLM

from .._common import (
    check_dtype_pattern,
    check_transformers_version,
    init_weights_tiny_model,
    print_config_diff,
    push_to_hub,
    set_seed,
    smoke_test,
)


# LFM2-MoE was introduced in transformers 5.0.0, which is also the version that introduced the `TokenizersBackend`
# tokenizer LFM2.5 ships. This is above TRL's transformers floor, so unlike the other tiny models this one can't be
# loaded by the floor CI job, and the tests using it are skipped there.
check_transformers_version("5.0.0")

set_seed()

MODEL_ID = "LiquidAI/LFM2.5-24B-A2B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
generation_config = GenerationConfig.from_pretrained(MODEL_ID)
config = Lfm2MoeConfig(
    vocab_size=128000,
    hidden_size=8,
    num_attention_heads=4,
    num_key_value_heads=2,
    num_hidden_layers=2,
    intermediate_size=32,
    moe_intermediate_size=32,
    num_experts=4,
    num_experts_per_tok=2,
    # LFM2 interleaves short convolution layers with full attention layers, and LFM2-MoE keeps its first
    # `num_dense_layers` feed-forward blocks dense. Keep one layer of each so the conv, attention, dense and sparse
    # paths are all exercised.
    layer_types=["conv", "full_attention"],
    num_dense_layers=1,
    # Non-size fields kept aligned with the reference so the tiny config only differs in what we scale down.
    max_position_embeddings=131072,
    norm_eps=1e-05,
    rope_parameters={"rope_theta": 10000000.0, "rope_type": "default"},
    conv_bias=False,
    conv_L_cache=3,
    norm_topk_prob=True,
    routed_scaling_factor=1.0,
    use_expert_bias=True,
    bos_token_id=124894,
    eos_token_id=124900,
    pad_token_id=124893,
    tie_word_embeddings=True,
    use_cache=True,
)
model = Lfm2MoeForCausalLM(config).to(dtype=torch.bfloat16)
init_weights_tiny_model(model)
smoke_test(model, tokenizer)
check_dtype_pattern(MODEL_ID, model)
print_config_diff(MODEL_ID, model)
push_to_hub(model, tokenizer, generation_config, "tiny", "2.5")
