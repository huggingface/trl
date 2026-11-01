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

import math
import os
import tempfile

import torch
from safetensors.torch import save_file
from torch import nn
from transformers import AutoTokenizer, DeepseekV3Config, DeepseekV3ForCausalLM, GenerationConfig

from .._common import (
    check_dtype_pattern,
    check_transformers_version,
    init_weights_tiny_model,
    print_config_diff,
    push_dir_to_hub,
    set_seed,
    smoke_test,
)


check_transformers_version()

set_seed()

MODEL_ID = "deepseek-ai/DeepSeek-R1"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
generation_config = GenerationConfig.from_pretrained(MODEL_ID)
config = DeepseekV3Config(
    vocab_size=129280,
    # 128, not 8: the FP8 kernel quantizes each linear's input in blocks of 128 along its last dimension
    hidden_size=128,
    # DeepSeek-V3 uses MLA, not GQA: kv_b_proj always reconstructs K/V with the full
    # num_attention_heads, so num_key_value_heads must equal num_attention_heads (i.e.
    # num_key_value_groups == 1). A mismatch (e.g. 4/2) is silently ignored by the KV
    # projection but makes SDPA repeat_kv the already-full K/V heads, crashing with a
    # head-count mismatch (see transformers#46960).
    num_attention_heads=2,
    num_key_value_heads=2,
    num_hidden_layers=2,
    intermediate_size=128,
    max_position_embeddings=163840,
    rope_scaling={
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "factor": 40.0,
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
        "original_max_position_embeddings": 4096,
        "type": "yarn",
    },
    ep_size=1,
    moe_layer_freq=1,
    num_nextn_predict_layers=1,
    scoring_func="sigmoid",
    topk_method="noaux_tc",
)
model = DeepseekV3ForCausalLM(config).to(dtype=torch.bfloat16)
init_weights_tiny_model(model)

# FP8 weights, like the reference checkpoint: quantize every linear but lm_head in 128x128 blocks, each block scaled to
# the e4m3 range, with the scale of each block stored in `weight_scale_inv` (needs a GPU with FP8 support to load)
config.quantization_config = {
    "activation_scheme": "dynamic",
    "fmt": "e4m3",
    "quant_method": "fp8",
    "weight_block_size": [128, 128],
}
config.dtype = torch.bfloat16
state_dict = model.state_dict()
for name, module in model.named_modules():
    if isinstance(module, nn.Linear) and name != "lm_head":
        weight = module.weight.float()
        rows, cols = weight.shape
        # The last blocks are partial where a dimension is not a multiple of 128 (kv_a_proj_with_mqa's 576 rows)
        padded = torch.zeros(math.ceil(rows / 128) * 128, math.ceil(cols / 128) * 128)
        padded[:rows, :cols] = weight
        blocks = padded.view(padded.shape[0] // 128, 128, padded.shape[1] // 128, 128)
        scale = blocks.abs().amax(dim=(1, 3)) / torch.finfo(torch.float8_e4m3fn).max
        quantized = (blocks / scale[:, None, :, None]).view(padded.shape)[:rows, :cols]
        state_dict[f"{name}.weight"] = quantized.to(torch.float8_e4m3fn).contiguous()
        state_dict[f"{name}.weight_scale_inv"] = scale

check_dtype_pattern(MODEL_ID, state_dict)
print_config_diff(MODEL_ID, model)
with tempfile.TemporaryDirectory() as tmp_dir:
    config.save_pretrained(tmp_dir)
    save_file(state_dict, os.path.join(tmp_dir, "model.safetensors"), metadata={"format": "pt"})
    model = DeepseekV3ForCausalLM.from_pretrained(tmp_dir, dtype=torch.bfloat16, device_map="cuda")
    smoke_test(model, tokenizer)
    push_dir_to_hub(tmp_dir, model, tokenizer, generation_config, "tiny")
