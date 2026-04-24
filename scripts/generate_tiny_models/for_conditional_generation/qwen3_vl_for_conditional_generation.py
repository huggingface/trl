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
# - "layer_types" is intentionally omitted from text_config: qwen3_vl's modeling code checks
#   `hasattr(config, "layer_types")` and uses a different path when absent
#   (see transformers/models/qwen3_vl/modeling_qwen3_vl.py).
# - mrope_section needs 3 elements (for dim, offset in enumerate((1, 2), start=1): mrope_section[dim]).

import torch
from transformers import AutoConfig, AutoProcessor, GenerationConfig, Qwen3VLForConditionalGeneration

from .._common import check_dtype_pattern, check_transformers_version, print_config_diff, push_to_hub, smoke_test


TRANSFORMERS_VERSION = "4.57.0"
check_transformers_version(TRANSFORMERS_VERSION)

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"

processor = AutoProcessor.from_pretrained(MODEL_ID)
generation_config = GenerationConfig.from_pretrained(MODEL_ID)

text_config = {
    "num_hidden_layers": 2,
    "hidden_size": 16,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "rope_scaling": {"mrope_interleaved": True, "mrope_section": [2, 2, 2], "rope_type": "default"},
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

config = AutoConfig.from_pretrained(MODEL_ID, text_config=text_config, vision_config=vision_config)
model = Qwen3VLForConditionalGeneration(config).to(dtype=torch.bfloat16)
smoke_test(model, processor)
check_dtype_pattern(MODEL_ID, model)
print_config_diff(MODEL_ID, model)
push_to_hub(model, processor, generation_config, "tiny")
