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
from transformers.models.muse_spark import MuseSparkForConditionalGeneration

from .._common import (
    check_dtype_pattern,
    check_transformers_version,
    init_weights_tiny_model,
    print_config_diff,
    push_to_hub,
    set_seed,
    smoke_test,
)


check_transformers_version("5.20.0")

set_seed()

MODEL_ID = "meta-models/Muse-Spark-Open-816B-A42B"

processor = AutoProcessor.from_pretrained(MODEL_ID)
generation_config = GenerationConfig.from_pretrained(MODEL_ID)

text_config = {
    "num_hidden_layers": 2,
    "hidden_size": 64,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "intermediate_size": 32,
    # The reference alternates sliding and full attention; keep one of each so both are exercised.
    "layer_types": ["full_attention", "sliding_attention"],
    "sliding_window": 32,
    # Both MoE, so every layer goes through the expert path. 16 experts divide every expert-parallel
    # width up to 16, which is what the tensor/expert parallel tests shard over.
    "mlp_layer_types": ["sparse", "sparse"],
    "num_local_experts": 16,
    "num_experts_per_tok": 4,
    "moe_intermediate_size": 32,
    "moe_hidden_size": 16,
}
# Only depth and width shrink here: the patch and merge sizes have to match what the image processor produces.
vision_config = {
    "num_hidden_layers": 2,
    "hidden_size": 32,
    "num_attention_heads": 4,
    "intermediate_size": 64,
    "layer_types": ["window_attention", "full_attention"],
}
# `hidden_sizes` and `depths` take one entry per layer.
audio_config = {
    "num_hidden_layers": 4,
    # `patch_sizes` stays at the reference: the processor depends on it.
    "hidden_sizes": (16, 16, 32, 32),
    "depths": (1, 1, 1, 1),
    "mlp_ratio": 2.0,
    "conv_kernel_size": 3,
    # Has to equal the text hidden size.
    "out_hidden_size": 64,
}

config = AutoConfig.from_pretrained(
    MODEL_ID,
    text_config=text_config,
    vision_config=vision_config,
    audio_config=audio_config,
    # Has to stay `vision hidden_size * merge_size ** 2`.
    out_hidden_size=128,
    projector_hidden_size=32,
)
model = MuseSparkForConditionalGeneration(config).to(dtype=torch.bfloat16)
init_weights_tiny_model(model)

# Restore float32 for the router, which the reference keeps in fp32: bf16 router logits tie often enough
# that the chosen experts would differ from the reference model's.
for layer in model.model.language_model.layers:
    layer.mlp.gate.weight.data = layer.mlp.gate.weight.data.float()
    layer.mlp.gate.e_score_correction_bias.data = layer.mlp.gate.e_score_correction_bias.data.float()
smoke_test(model, processor)
check_dtype_pattern(MODEL_ID, model)
print_config_diff(MODEL_ID, model)
push_to_hub(model, processor, generation_config, "tiny")
