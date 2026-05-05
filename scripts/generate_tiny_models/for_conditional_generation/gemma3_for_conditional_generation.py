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
from transformers import AutoConfig, AutoProcessor, Gemma3ForConditionalGeneration, GenerationConfig

from .._common import check_dtype_pattern, check_transformers_version, print_config_diff, push_to_hub, smoke_test


check_transformers_version()

MODEL_ID = "google/gemma-3-4b-it"

processor = AutoProcessor.from_pretrained(MODEL_ID)
generation_config = GenerationConfig.from_pretrained(MODEL_ID)

# Gemma3 SigLIP defaults to image_size=896 → 4,096 patches. With production-scale intermediate_size
# (~4,304), the vision FFN saves [batch, 4,096, 4,304] activations for backprop (~134 MB/layer).
# Use image_size=224 → 256 patches (matching mm_tokens_per_image=256, so the projector's AvgPool2d
# gets kernel_size=1, i.e. identity) and scale down intermediate_size to keep activations tiny.
text_config = {
    "num_hidden_layers": 2,
    "hidden_size": 16,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "layer_types": None,  # Set it automatically from num_hidden_layers
    "intermediate_size": 32,
}
vision_config = {
    "num_hidden_layers": 2,
    "hidden_size": 16,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "embed_dim": 64,
    "intermediate_size": 32,
    "image_size": 224,
}
processor.image_processor.size = {"height": 224, "width": 224}

config = AutoConfig.from_pretrained(MODEL_ID, text_config=text_config, vision_config=vision_config)
model = Gemma3ForConditionalGeneration(config).to(dtype=torch.bfloat16)
smoke_test(model, processor)
check_dtype_pattern(MODEL_ID, model)
print_config_diff(MODEL_ID, model)
push_to_hub(model, processor, generation_config, "tiny")
