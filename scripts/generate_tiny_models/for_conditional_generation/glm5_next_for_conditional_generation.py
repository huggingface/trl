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

import tempfile

import torch
from transformers import (
    AutoProcessor,
    GenerationConfig,
    Glm5NextConfig,
    Glm5NextForConditionalGeneration,
    Glm5NextTextConfig,
    Glm5NextVisionConfig,
)

from .._common import (
    check_dtype_pattern,
    check_transformers_version,
    init_weights_tiny_model,
    print_config_diff,
    push_to_hub,
    set_seed,
    smoke_test,
)


check_transformers_version("5.17.0")

set_seed()

MODEL_ID = "zai-org/GLM-5.3-Flash"

processor = AutoProcessor.from_pretrained(MODEL_ID)
generation_config = GenerationConfig.from_pretrained(MODEL_ID)
# Unset: save_pretrained rejects top_p without do_sample (the reference decodes greedily by default anyway)
generation_config.top_p = None

text_config = Glm5NextTextConfig(
    vocab_size=154880,
    hidden_size=32,
    intermediate_size=32,
    # Not 32: the FP8 experts kernels return wrong values for 32x32 expert weights
    moe_intermediate_size=64,
    # 4 layers, so that each per-layer pattern of the reference appears at least once: 3 linear-attention layers then
    # a sparse-attention one, and 3 dense MLP layers then a sparse one.
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=4,
    n_shared_experts=1,
    n_routed_experts=4,
    num_experts_per_tok=2,
    kv_lora_rank=32,
    q_lora_rank=32,
    qk_rope_head_dim=0,
    qk_nope_head_dim=8,
    v_head_dim=8,
    index_head_dim=4,
    index_n_heads=2,
    # Smaller than the test sequences, so the sparse attention path actually drops keys.
    index_topk=8,
    index_kpool=4,
    linear_head_dim=8,
    linear_num_heads=4,
    eos_token_id=[154820, 154827, 154829],
    pad_token_id=154820,
)
vision_config = Glm5NextVisionConfig(
    depth=2,
    hidden_size=16,
    num_heads=4,
    intermediate_size=32,
    out_hidden_size=32,
    projection_intermediate_size=32,
    image_size=448,
)
config = Glm5NextConfig(text_config=text_config, vision_config=vision_config)
model = Glm5NextForConditionalGeneration(config).to(dtype=torch.bfloat16)
init_weights_tiny_model(model)

# Restore float32 for the router's correction bias, which the reference keeps in fp32
for layer in model.model.language_model.layers[3:]:
    layer.mlp.gate.e_score_correction_bias.data = layer.mlp.gate.e_score_correction_bias.data.float()
with tempfile.TemporaryDirectory() as tmp_dir:
    model.save_pretrained(tmp_dir)
    model = Glm5NextForConditionalGeneration.from_pretrained(tmp_dir, device_map="cuda")
smoke_test(model, processor)
check_dtype_pattern(MODEL_ID, model)
print_config_diff(MODEL_ID, model)
push_to_hub(model, processor, generation_config, "tiny")
