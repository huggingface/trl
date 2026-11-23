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

import os
import re
import tempfile

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoProcessor, GenerationConfig
from transformers.models.mistral_large4 import MistralLarge4ForConditionalGeneration

from .._common import (
    check_dtype_pattern,
    check_transformers_version,
    init_weights_tiny_model,
    print_config_diff,
    push_dir_to_hub,
    set_seed,
    smoke_test,
)


check_transformers_version("5.22.0")

set_seed()

MODEL_ID = "mistralai/Mistral-Large-4-1T-A52B"

processor = AutoProcessor.from_pretrained(MODEL_ID)
generation_config = GenerationConfig.from_pretrained(MODEL_ID)

text_config = {
    # 8 layers, so that each per-layer pattern of the reference appears at least once: a dense first layer, NoPE every
    # 4th layer, a layer reading an earlier layer's KV, and indexer layers that reuse the previous layer's top-k.
    "num_hidden_layers": 8,
    "kv_source_layers": [0, 1, 2, 3, 4, 5, 6, 3],
    "indexer_types": ["full", "full", "full", "shared", "shared", "full", "full", "shared"],
    # Derived from `num_hidden_layers` when unset
    "layer_types": None,
    "mlp_layer_types": None,
    "no_rope_layers": None,
    "hidden_size": 32,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "intermediate_size": 32,
    "kv_lora_rank": 16,
    "qk_rope_head_dim": 8,
    "qk_nope_head_dim": 8,
    "v_head_dim": 8,
    "n_routed_experts": 8,
    "num_experts_per_tok": 2,
    # Not 32: the FP8 experts kernels return wrong values for 32x32 expert weights
    "moe_intermediate_size": 64,
    "index_n_heads": 2,
    "index_head_dim": 16,
    # Smaller than the test sequences, so the sparse attention path actually drops keys.
    "index_topk": 4,
}
vision_config = {
    "num_hidden_layers": 2,
    "hidden_size": 16,
    "num_attention_heads": 2,  # head size 16 // 2 = 8, as Flash Attention requires
    "head_dim": 8,
    "intermediate_size": 32,
}

config = AutoConfig.from_pretrained(MODEL_ID, text_config=text_config, vision_config=vision_config)
# Set aside: the model is built and saved in bfloat16, then quantized below
quantization_config = config.quantization_config
del config.quantization_config
model = MistralLarge4ForConditionalGeneration(config).to(dtype=torch.bfloat16)
init_weights_tiny_model(model)

# FP8 weights, like the reference checkpoint: quantize the linears the reference stores in FP8 (attention, MLPs,
# experts) in blocks scaled down from the reference's 128x128 to 32x32, each block scaled to the e4m3 range by a power
# of two (the reference's UE8M0 scales), with the scale of each block stored in `qscale_weight`
fp8_weights = re.compile(
    r"layers\.\d+\.(attention\.(wq|wo|indexer\.wq_b)|(feed_forward|shared_experts|experts\.\d+)\.w[123])\.weight"
)
with tempfile.TemporaryDirectory() as tmp_dir:
    # Saved with the checkpoint names of the reference (consolidated format, unfused experts)
    model.save_pretrained(tmp_dir)
    state_dict = load_file(os.path.join(tmp_dir, "model.safetensors"))
    for name, weight in list(state_dict.items()):
        if fp8_weights.fullmatch(name):
            rows, cols = weight.shape
            blocks = weight.float().view(rows // 32, 32, cols // 32, 32)
            scale = torch.exp2(
                torch.ceil(torch.log2(blocks.abs().amax(dim=(1, 3)) / torch.finfo(torch.float8_e4m3fn).max))
            )
            state_dict[name] = (blocks / scale[:, None, :, None]).view(rows, cols).to(torch.float8_e4m3fn)
            state_dict[name.removesuffix("weight") + "qscale_weight"] = scale
    config.quantization_config = {**quantization_config, "weight_block_size": [32, 32]}
    check_dtype_pattern(MODEL_ID, state_dict)
    print_config_diff(MODEL_ID, model)
    config.save_pretrained(tmp_dir)
    save_file(state_dict, os.path.join(tmp_dir, "model.safetensors"), metadata={"format": "pt"})
    model = MistralLarge4ForConditionalGeneration.from_pretrained(tmp_dir, device_map="cuda")
    smoke_test(model, processor)
    push_dir_to_hub(tmp_dir, model, processor, generation_config, "tiny")
