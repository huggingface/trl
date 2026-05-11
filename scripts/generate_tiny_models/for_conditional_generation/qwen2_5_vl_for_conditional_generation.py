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

# Note: Qwen2.5-VL requires out_hidden_size on the vision config, plus root-level num_hidden_layers/hidden_size/
# num_attention_heads (distinct from the text_config fields). See GH-4101 and transformers#41020.

import torch
from transformers import AutoConfig, AutoProcessor, GenerationConfig, Qwen2_5_VLForConditionalGeneration

from .._common import check_dtype_pattern, check_transformers_version, print_config_diff, push_to_hub, smoke_test


check_transformers_version()

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

processor = AutoProcessor.from_pretrained(MODEL_ID)
generation_config = GenerationConfig.from_pretrained(MODEL_ID)

text_config = {
    "num_hidden_layers": 2,
    "hidden_size": 16,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "layer_types": None,
    "rope_scaling": {"type": "default", "mrope_section": [1, 1], "rope_type": "default"},
}
vision_config = {
    "num_hidden_layers": 2,
    "hidden_size": 16,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "embed_dim": 64,
    "depth": 2,
    "out_hidden_size": 16,
}

config = AutoConfig.from_pretrained(
    MODEL_ID,
    text_config=text_config,
    vision_config=vision_config,
    rope_scaling={"type": "default", "mrope_section": [1, 1], "rope_type": "default"},
    num_hidden_layers=2,
    hidden_size=16,
    num_attention_heads=4,
)
model = Qwen2_5_VLForConditionalGeneration(config).to(dtype=torch.bfloat16)
smoke_test(model, processor)
check_dtype_pattern(MODEL_ID, model)
print_config_diff(MODEL_ID, model)
push_to_hub(model, processor, generation_config, "tiny")
