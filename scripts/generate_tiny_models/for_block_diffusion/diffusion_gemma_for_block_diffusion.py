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
from transformers import AutoConfig, AutoProcessor, DiffusionGemmaForBlockDiffusion
from transformers.models.diffusion_gemma.generation_diffusion_gemma import DiffusionGemmaGenerationConfig

from .._common import check_dtype_pattern, check_transformers_version, print_config_diff, push_to_hub, smoke_test


TRANSFORMERS_VERSION = "5.11.0"  # DiffusionGemma was added in transformers 5.11.0
check_transformers_version(TRANSFORMERS_VERSION)

MODEL_ID = "google/diffusiongemma-26B-A4B-it"

processor = AutoProcessor.from_pretrained(MODEL_ID)
generation_config = DiffusionGemmaGenerationConfig.from_pretrained(MODEL_ID)

# The vision tower is Gemma4's; the scale-down mirrors gemma4_for_conditional_generation.py.
IMAGE_TOKENS = 70  # minimum supported max_soft_tokens
MAX_PATCHES = IMAGE_TOKENS * 3**2  # 630

text_config = {
    "num_hidden_layers": 2,
    "hidden_size": 16,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 8,
    "intermediate_size": 32,
    "num_global_key_value_heads": 2,
    "global_head_dim": 8,
    "num_experts": 4,
    "top_k_experts": 2,
    "moe_intermediate_size": 16,
}
vision_config = {
    "num_hidden_layers": 2,
    "hidden_size": 16,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "embed_dim": 64,
    "intermediate_size": 32,
    "position_embedding_size": MAX_PATCHES,  # 630
    "default_output_length": IMAGE_TOKENS,  # 70
}

processor.image_seq_length = IMAGE_TOKENS
processor.image_processor.image_seq_length = IMAGE_TOKENS
processor.image_processor.max_soft_tokens = IMAGE_TOKENS

config = AutoConfig.from_pretrained(MODEL_ID)
for k, v in text_config.items():
    setattr(config.text_config, k, v)
for k, v in vision_config.items():
    setattr(config.vision_config, k, v)
config.text_config.layer_types = ["sliding_attention", "full_attention"]
config.canvas_length = 32  # keep block-diffusion forwards/denoising loops small in tests

model = DiffusionGemmaForBlockDiffusion(config).to(dtype=torch.bfloat16)
# Text-only smoke test: the canvas is sampled internally when `decoder_input_ids` is unset
smoke_test(model)
check_dtype_pattern(MODEL_ID, model)
print_config_diff(MODEL_ID, model)
push_to_hub(model, processor, generation_config, "tiny")
