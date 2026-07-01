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
from transformers import AutoConfig, AutoProcessor, GenerationConfig, VoxtralForConditionalGeneration

from .._common import check_dtype_pattern, check_transformers_version, print_config_diff, push_to_hub, smoke_test


check_transformers_version()

MODEL_ID = "mistralai/Voxtral-Mini-3B-2507"

processor = AutoProcessor.from_pretrained(MODEL_ID)
generation_config = GenerationConfig.from_pretrained(MODEL_ID)

# Pass canonical field names for audio_config (not the `attribute_map` aliases like
# `encoder_layers` / `d_model`): the Voxtral reference config serializes only canonical names, and
# the merge logic in `PretrainedConfig.from_dict` would otherwise leave the canonical values from
# disk intact and silently drop the alias override. num_mel_bins and max_source_positions stay at
# production values (locked to the feature extractor / positional embedding shapes).
text_config = {
    "num_hidden_layers": 2,
    "hidden_size": 16,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "intermediate_size": 32,
    "head_dim": 4,  # production default is 128; must match hidden_size / num_attention_heads
    "layer_types": None,
}
audio_config = {
    "num_hidden_layers": 2,
    "hidden_size": 16,
    "num_attention_heads": 4,
    # Must stay at 4 * hidden_size: `get_audio_embeds` reshapes encoder output to width
    # `intermediate_size`, and the tokenizer hardcodes the per-chunk audio token count assuming the
    # production 4:1 ratio. A different ratio causes a feature/token count mismatch at inference.
    "intermediate_size": 64,
}

config = AutoConfig.from_pretrained(MODEL_ID, text_config=text_config, audio_config=audio_config)
model = VoxtralForConditionalGeneration(config).to(dtype=torch.bfloat16)
# Voxtral declares `_keep_in_fp32_modules_strict = ["embed_positions"]`; the bulk bf16 cast above
# overrides that, so restore the strict fp32 modules to match the reference checkpoint.
model.audio_tower.embed_positions.to(dtype=torch.float32)
smoke_test(model, processor)
check_dtype_pattern(MODEL_ID, model)
print_config_diff(MODEL_ID, model)
push_to_hub(model, processor, generation_config, "tiny")
