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

import tempfile

import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DeepseekV3Config,
    DeepseekV3ForCausalLM,
    FineGrainedFP8Config,
    GenerationConfig,
)

from .._common import (
    check_dtype_pattern,
    check_transformers_version,
    init_weights_tiny_model,
    print_config_diff,
    push_to_hub,
    smoke_test,
)


TRANSFORMERS_VERSION = "4.56.2"
check_transformers_version(TRANSFORMERS_VERSION)

MODEL_ID = "deepseek-ai/DeepSeek-R1"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
generation_config = GenerationConfig.from_pretrained(MODEL_ID)

# DeepSeek-R1 uses weight_block_size=[128,128] upstream. We use [32,32] for the tiny so that smaller
# hidden dims still tile cleanly (every projection dim divisible by 32, ≥ 2 blocks per dim to avoid
# a scalar weight_scale_inv shape). Trade-off: drops out of the DeepGEMM fast path onto Triton; fine
# for a tiny used in tests.
config = DeepseekV3Config(
    vocab_size=AutoConfig.from_pretrained(MODEL_ID).vocab_size,
    hidden_size=64,
    num_attention_heads=4,
    num_key_value_heads=2,
    num_hidden_layers=2,
    intermediate_size=64,
    max_position_embeddings=163840,
    rope_scaling={
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "factor": 40.0,
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
        "original_max_position_embeddings": 4096,
        "rope_type": "yarn",
        "type": "yarn",
    },
    ep_size=1,
    moe_layer_freq=1,
    num_nextn_predict_layers=1,
    scoring_func="sigmoid",
    topk_method="noaux_tc",
)

# Build a random bf16 model, then round-trip through disk with `quantization_config=FineGrainedFP8Config(...)`
# so transformers' FP8 quantizer rewrites Linear layers to FP8Linear and writes FP8 weights + scales.
# Needs a GPU with compute capability >= 8.9 (H100+); on older hardware the quantizer auto-dequantizes to bf16.
with tempfile.TemporaryDirectory() as tmpdir:
    bf16_model = DeepseekV3ForCausalLM(config).to(dtype=torch.bfloat16, device="cuda")
    init_weights_tiny_model(bf16_model)
    bf16_model.save_pretrained(tmpdir)
    tokenizer.save_pretrained(tmpdir)
    del bf16_model
    torch.cuda.empty_cache()

    quantization_config = FineGrainedFP8Config(activation_scheme="dynamic", weight_block_size=[32, 32])
    model = DeepseekV3ForCausalLM.from_pretrained(
        tmpdir,
        quantization_config=quantization_config,
        dtype=torch.bfloat16,
        device_map="cuda",
    )

# `dtype=torch.bfloat16` casts the whole model, including the FP32 per-block scales the quantizer
# created. Restore them to FP32 to match the reference's dtype pattern.
for module in model.modules():
    if hasattr(module, "weight_scale_inv") and module.weight_scale_inv is not None:
        module.weight_scale_inv.data = module.weight_scale_inv.data.float()

smoke_test(model, tokenizer)
check_dtype_pattern(MODEL_ID, model)
print_config_diff(MODEL_ID, model)
push_to_hub(model, tokenizer, generation_config, "tiny")
