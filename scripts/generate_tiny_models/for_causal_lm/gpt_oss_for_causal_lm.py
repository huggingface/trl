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
import tempfile

import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer, GenerationConfig, GptOssConfig, GptOssForCausalLM
from transformers.integrations.mxfp4 import quantize_to_mxfp4
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssExperts

from kernels import get_kernel

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

MODEL_ID = "openai/gpt-oss-20b"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
generation_config = GenerationConfig.from_pretrained(MODEL_ID)
config = GptOssConfig(
    vocab_size=201088,
    # 32, not 8: MXFP4 quantizes the expert weights in blocks of 32 along their input dimension
    hidden_size=32,
    num_attention_heads=4,
    num_key_value_heads=2,
    num_hidden_layers=2,
    intermediate_size=32,
    num_local_experts=4,
    num_experts_per_tok=2,
    eos_token_id=200002,
    pad_token_id=199999,
    # Reference-only keys: `GptOssConfig` does not model any of these.
    initial_context_length=4096,
    swiglu_limit=7.0,
    # Alias the reference carries for `num_experts_per_tok`, so it takes the scaled-down value.
    experts_per_token=2,
)
model = GptOssForCausalLM(config).to(dtype=torch.bfloat16)
init_weights_tiny_model(model)

# MXFP4 experts, like the reference checkpoint: quantize the expert weights in blocks of 32 along their input dimension
# with transformers' own quantizer, and store them in the checkpoint layout: the e2m1 values packed two per byte in
# `*_blocks` and the block exponents in `*_scales` (needs a GPU and the `kernels` package to load without dequantizing)
config.quantization_config = {
    "modules_to_not_convert": [
        "model.layers.*.self_attn",
        "model.layers.*.mlp.router",
        "model.embed_tokens",
        "lm_head",
    ],
    "quant_method": "mxfp4",
}
config.architectures = [model.__class__.__name__]
config.dtype = torch.bfloat16
triton_kernels = get_kernel("kernels-community/triton_kernels")
state_dict = model.state_dict()
for name, module in model.named_modules():
    if isinstance(module, GptOssExperts):
        for proj in ("gate_up_proj", "down_proj"):
            weight = state_dict.pop(f"{name}.{proj}")
            num_experts, in_features, out_features = weight.shape
            blocks, scales = quantize_to_mxfp4(weight, triton_kernels)
            state_dict[f"{name}.{proj}_blocks"] = blocks.transpose(1, 2).reshape(
                num_experts, out_features, in_features // 32, 16
            )
            state_dict[f"{name}.{proj}_scales"] = scales.transpose(1, 2).contiguous()

check_dtype_pattern(MODEL_ID, state_dict)
print_config_diff(MODEL_ID, model)
with tempfile.TemporaryDirectory() as tmp_dir:
    config.save_pretrained(tmp_dir)
    save_file(state_dict, os.path.join(tmp_dir, "model.safetensors"), metadata={"format": "pt"})
    model = GptOssForCausalLM.from_pretrained(tmp_dir, dtype=torch.bfloat16, device_map="cuda")
    smoke_test(model, tokenizer)
    push_dir_to_hub(tmp_dir, model, tokenizer, generation_config, "tiny")
