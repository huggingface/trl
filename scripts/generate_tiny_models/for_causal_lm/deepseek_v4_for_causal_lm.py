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

# WIP: targets the in-review transformers branch that adds DeepSeek-V4 and the in-review model PR on
# the Hub. This script exists to provide early feedback on the integration; expect the pinned version
# and revision to change once V4 lands upstream.

import torch
from torch import nn
from transformers import AutoTokenizer, DeepseekV4Config, DeepseekV4ForCausalLM, GenerationConfig
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4Experts

from .._common import (
    check_dtype_pattern,
    check_transformers_version,
    init_weights_tiny_model,
    print_config_diff,
    push_to_hub,
    smoke_test,
)


TRANSFORMERS_VERSION = "5.8.0"
check_transformers_version(TRANSFORMERS_VERSION)

MODEL_ID = "deepseek-ai/DeepSeek-V4-Flash"
REVISION = "refs/pr/16"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
generation_config = GenerationConfig.from_pretrained(MODEL_ID, revision=REVISION)
config = DeepseekV4Config(
    vocab_size=len(tokenizer.vocab),
    hidden_size=8,
    num_attention_heads=4,
    num_key_value_heads=1,  # V4 MLA: single KV head broadcast to all Q heads.
    # 4 layers so the tiny mirrors the reference's layer-position dispatch:
    # layers 0-1 match ref's first two non-compressor layers; layer 2 = compressor+indexer; layer 3
    # = compressor-only. Also covers the three dense/MoE + hash/learned-gate dispatch axes.
    num_hidden_layers=4,
    intermediate_size=32,
    layer_types=[
        "sliding_attention",
        "sliding_attention",
        "compressed_sparse_attention",
        "heavily_compressed_attention",
    ],
    rope_parameters={
        "main": {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 65536,
            "type": "yarn",
            "rope_theta": 10000,
            "rope_type": "yarn",
            "partial_rotary_factor": 0.125,
        },
        "compress": {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 65536,
            "type": "yarn",
            "rope_theta": 160000,
            "rope_type": "yarn",
            "partial_rotary_factor": 0.125,
        },
    },
    topk_method="noaux_tc",
    quantization_config={
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "fp8",
        "scale_fmt": "ue8m0",
        "weight_block_size": [128, 128],
    },
)
model = DeepseekV4ForCausalLM(config).to(dtype=torch.bfloat16)
init_weights_tiny_model(model)
smoke_test(model, tokenizer)

# Reference checkpoint is FP8-quantized: every Linear (except lm_head) and the MoE Experts module
# carry a `*_scale_inv` companion holding per-block (128x128) dequantization scales. Our tiny stays
# bf16 for usability, but we register the same scale_inv parameters so the on-disk checkpoint shape
# matches the reference and re-loading produces no MISSING entries. Done after smoke_test because
# attaching parameters has no effect on forward (FP8 paths only trigger when the weight itself is fp8).
_FP8_BLOCK = (128, 128)


def _cdiv(a, b):
    return (a + b - 1) // b


for name, module in model.named_modules():
    if isinstance(module, nn.Linear) and not name.endswith("lm_head"):
        out_f, in_f = module.out_features, module.in_features
        scale = torch.ones(_cdiv(out_f, _FP8_BLOCK[0]), _cdiv(in_f, _FP8_BLOCK[1]), dtype=torch.float32)
        module.register_parameter("weight_scale_inv", nn.Parameter(scale, requires_grad=False))
    elif isinstance(module, DeepseekV4Experts):
        n_experts, gu_out, gu_in = module.gate_up_proj.shape
        _, d_out, d_in = module.down_proj.shape
        gu_scale = torch.ones(
            n_experts, _cdiv(gu_out, _FP8_BLOCK[0]), _cdiv(gu_in, _FP8_BLOCK[1]), dtype=torch.float32
        )
        d_scale = torch.ones(n_experts, _cdiv(d_out, _FP8_BLOCK[0]), _cdiv(d_in, _FP8_BLOCK[1]), dtype=torch.float32)
        module.register_parameter("gate_up_proj_scale_inv", nn.Parameter(gu_scale, requires_grad=False))
        module.register_parameter("down_proj_scale_inv", nn.Parameter(d_scale, requires_grad=False))

check_dtype_pattern(MODEL_ID, model, revision=REVISION)
print_config_diff(MODEL_ID, model, revision=REVISION)
push_to_hub(model, tokenizer, generation_config, "tiny")
