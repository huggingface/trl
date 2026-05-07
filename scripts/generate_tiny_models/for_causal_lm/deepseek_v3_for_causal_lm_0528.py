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

# Note: R1-0528 is kept in addition to R1 because it has a different chat template.

import torch
from torch import nn
from transformers import AutoTokenizer, DeepseekV3Config, DeepseekV3ForCausalLM, GenerationConfig

from .._common import (
    check_dtype_pattern,
    check_transformers_version,
    init_weights_tiny_model,
    print_config_diff,
    push_to_hub,
    smoke_test,
)


check_transformers_version()

MODEL_ID = "deepseek-ai/DeepSeek-R1-0528"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
generation_config = GenerationConfig.from_pretrained(MODEL_ID)
config = DeepseekV3Config(
    vocab_size=129280,
    # Every FP8Linear's `in_features` must be a multiple of 128 (the activation-quant kernel asserts
    # `x.shape[-1] % 128 == 0`). That forces hidden_size, q_lora_rank, kv_lora_rank, intermediate_size,
    # and `num_attention_heads * v_head_dim` (the o_proj input) to all be multiples of 128.
    hidden_size=128,
    num_attention_heads=4,
    # MLA already produces keys at `num_attention_heads`; setting `num_kv_heads < num_heads` makes
    # `sdpa_attention_forward` call `repeat_kv` with groups>1 and double the key heads, breaking
    # the matmul. The reference config keeps them equal — mirror that here.
    num_key_value_heads=4,
    num_hidden_layers=2,
    intermediate_size=128,
    q_lora_rank=128,
    kv_lora_rank=128,
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
    quantization_config={
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "fp8",
        "weight_block_size": [128, 128],
    },
    # Forwarded through to match the reference. These fields belong to the upstream remote-code
    # `DeepseekV3Config` (trust_remote_code) and are not part of transformers' in-tree config; we
    # carry them as arbitrary kwargs so the saved config.json mirrors the reference. `auto_map`
    # is intentionally NOT forwarded — it points to remote-code modules (`configuration_deepseek`,
    # `modeling_deepseek`) that this tiny repo does not ship.
    ep_size=1,
    moe_layer_freq=1,
    num_nextn_predict_layers=1,
    scoring_func="sigmoid",
    topk_method="noaux_tc",
)
model = DeepseekV3ForCausalLM(config).to(dtype=torch.bfloat16)
init_weights_tiny_model(model)
smoke_test(model, tokenizer)

# Mirror the reference's FP8 layout: each Linear (except lm_head) gets a per-block (128x128)
# `weight_scale_inv` companion, weight cast to F8_E4M3. Done after smoke_test — the cast severs
# `nn.Linear.forward` (FP8 GEMM only runs via `FP8Linear`, installed by the quantizer on reload).
_FP8_BLOCK = (128, 128)


def _cdiv(a, b):
    return (a + b - 1) // b


for name, module in model.named_modules():
    if not isinstance(module, nn.Linear) or name.endswith("lm_head"):
        continue
    out_f, in_f = module.out_features, module.in_features
    scale = torch.ones(_cdiv(out_f, _FP8_BLOCK[0]), _cdiv(in_f, _FP8_BLOCK[1]), dtype=torch.float32)
    module.register_parameter("weight_scale_inv", nn.Parameter(scale, requires_grad=False))
    # int8 isn't a float dtype, so we drop grad before swapping storage.
    module.weight.requires_grad_(False)
    module.weight.data = module.weight.data.to(torch.float8_e4m3fn)

check_dtype_pattern(MODEL_ID, model)
print_config_diff(MODEL_ID, model)
push_to_hub(model, tokenizer, generation_config, "tiny", "0528")
