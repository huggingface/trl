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
from transformers import AutoConfig, AutoProcessor, GenerationConfig, MuseGlimmerForConditionalGeneration

from .._common import (
    check_dtype_pattern,
    check_transformers_version,
    init_weights_tiny_model,
    print_config_diff,
    push_to_hub,
    smoke_test,
)


check_transformers_version("5.15.0")

MODEL_ID = "meta-models/Muse-Glimmer-30B"

processor = AutoProcessor.from_pretrained(MODEL_ID)
generation_config = GenerationConfig.from_pretrained(MODEL_ID)

text_config = {
    "num_hidden_layers": 2,
    "hidden_size": 16,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 8,
    "intermediate_size": 32,
    # The reference interleaves three sliding-attention layers with one full-attention layer, and the full-attention
    # layers are the ones without rope (`layer_rope_theta` 0); keep one of each so both are exercised.
    "layer_types": ["sliding_attention", "full_attention"],
    "layer_rope_theta": [500000.0, 0],
}
vision_config = {
    "num_hidden_layers": 2,
    "hidden_size": 16,
    "num_attention_heads": 4,
    "intermediate_size": 32,
    # Same idea on the vision side, where the window/full alternation plays the role of the sliding/full one.
    "layer_types": ["window_attention", "full_attention"],
}

# The vision adapter consumes the pixel-shuffled vision output, so `out_hidden_size` must stay
# `vision hidden_size * merge_size ** 2` (6144 = 1536 * 2 ** 2 in the reference).
config = AutoConfig.from_pretrained(
    MODEL_ID, text_config=text_config, vision_config=vision_config, out_hidden_size=64, projector_hidden_size=32
)
model = MuseGlimmerForConditionalGeneration(config).to(dtype=torch.bfloat16)
init_weights_tiny_model(model)
smoke_test(model, processor)
check_dtype_pattern(MODEL_ID, model)
print_config_diff(MODEL_ID, model)
push_to_hub(model, processor, generation_config, "tiny")
