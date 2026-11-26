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
from transformers import AutoTokenizer, GenerationConfig, GlmMoeDsaConfig, GlmMoeDsaForCausalLM

from .._common import (
    check_dtype_pattern,
    check_transformers_version,
    init_weights_tiny_model,
    print_config_diff,
    push_to_hub,
    set_seed,
    smoke_test,
)


check_transformers_version("5.11.0")

set_seed()

MODEL_ID = "zai-org/GLM-5.2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
generation_config = GenerationConfig.from_pretrained(MODEL_ID)
# Unset: save_pretrained rejects top_p without do_sample (the reference decodes greedily by default anyway)
generation_config.top_p = None

config = GlmMoeDsaConfig(
    vocab_size=154880,
    hidden_size=32,
    intermediate_size=32,
    moe_intermediate_size=32,
    # 4 layers, so the reference's per-layer patterns (3 dense then sparse MLPs, 3 full then shared indexers) appear
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=4,
    n_shared_experts=1,
    n_routed_experts=4,
    num_experts=4,
    num_experts_per_tok=2,
    kv_lora_rank=32,
    q_lora_rank=32,
    qk_rope_head_dim=8,
    qk_nope_head_dim=8,
    v_head_dim=8,
    index_head_dim=16,
    index_n_heads=2,
    # Smaller than the test sequences, so the sparse attention path actually drops keys.
    index_topk=8,
    index_topk_freq=4,
    index_topk_pattern=None,
    index_skip_topk_offset=3,
    index_share_for_mtp_iteration=True,
    indexer_rope_interleave=True,
    rope_interleave=True,
    moe_router_dtype="float32",
    first_k_dense_replace=3,
    max_position_embeddings=1048576,
    rope_parameters={"rope_theta": 8000000, "rope_type": "default"},
    num_nextn_predict_layers=1,
    ep_size=1,
    moe_layer_freq=1,
    pretraining_tp=1,
    scoring_func="sigmoid",
    topk_method="noaux_tc",
    eos_token_id=[154820, 154827, 154829],
    pad_token_id=154820,
)
model = GlmMoeDsaForCausalLM(config).to(dtype=torch.bfloat16)
init_weights_tiny_model(model)
# Restore float32 for the router's correction bias, which the reference keeps in fp32
gate = model.model.layers[3].mlp.gate
gate.e_score_correction_bias.data = gate.e_score_correction_bias.data.float()
smoke_test(model, tokenizer)
check_dtype_pattern(MODEL_ID, model)
print_config_diff(MODEL_ID, model)
push_to_hub(model, tokenizer, generation_config, "tiny")
