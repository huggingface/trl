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

import json
import os
import tempfile

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoProcessor, GenerationConfig, Inkling2ForConditionalGeneration

from .._common import (
    check_dtype_pattern,
    check_transformers_version,
    init_weights_tiny_model,
    print_config_diff,
    push_dir_to_hub,
    set_seed,
    smoke_test,
)


check_transformers_version("5.23.0")

set_seed()

# Transformers only supports the MXFP8 checkpoint of Inkling 2 Small
MODEL_ID = "thinkingmachines/Inkling-2-Small-MXFP8"

processor = AutoProcessor.from_pretrained(MODEL_ID)
generation_config = GenerationConfig.from_pretrained(MODEL_ID)

# One layer of each type, in attention residual blocks of two (the reference has 48 layers in blocks of six)
text_config = {
    "num_hidden_layers": 2,
    "layer_types": ["hybrid_sliding", "hybrid"],
    "n_layers_per_attn_res_block": 2,
    "hidden_size": 32,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 8,
    "swa_num_attention_heads": 4,
    "swa_num_key_value_heads": 2,
    "swa_head_dim": 8,
    "sliding_window_size": 16,
    "d_rel": 4,
    "rel_extent": 32,
    "moe_intermediate_size": 32,
    "n_routed_experts": 8,
    "num_experts_per_tok": 2,
}
vision_config = {
    "num_hidden_layers": 2,
    "hidden_size": 32,
    "num_attention_heads": 2,
    "projection_hidden_dim": 64,
    "text_hidden_size": text_config["hidden_size"],
}

config = AutoConfig.from_pretrained(MODEL_ID)
for k, v in text_config.items():
    setattr(config.text_config, k, v)
for k, v in vision_config.items():
    setattr(config.vision_config, k, v)
quantization_config = config.quantization_config
del config.quantization_config  # the model is built in bf16, its experts are quantized below

model = Inkling2ForConditionalGeneration(config).to(dtype=torch.bfloat16)
init_weights_tiny_model(model)
# The modeling zero-initializes the attention residual queries and the static convolution weights, the reference's
# aren't zero
with torch.no_grad():
    for name, param in model.named_parameters():
        if name.endswith(".query"):
            param.normal_(0, 0.5)
        elif name.endswith(".static_weight"):
            param.normal_(0, 0.02)
# The modeling keeps these parameters in float32 (router, convolutions, attention residual inputs)
for name, param in model.named_parameters():
    if any(key in name for key in model._keep_in_fp32_modules_strict):
        param.data = param.data.float()
# Before quantization: transformers 5.23 can't load the quantized checkpoint back, since it doesn't map the reference's
# `modules_to_not_convert` to its own module names and quantizes the vision tower too (#8497)
smoke_test(model, processor)
print_config_diff(MODEL_ID, model)

with tempfile.TemporaryDirectory() as tmp_dir:
    # Save in the reference layout, then store each tensor like the reference: the attention residual queries and the
    # router bias in float32, the routed experts in MXFP8 (e4m3 values with a ue8m0 scale for every 32 values of a
    # row), everything else in bf16
    model.save_pretrained(tmp_dir)
    state_dict = load_file(os.path.join(tmp_dir, "model.safetensors"))
    for name, tensor in list(state_dict.items()):
        if ".mlp.experts.w" in name:
            blocks = tensor.float().unflatten(-1, (-1, 32))
            amax = blocks.abs().amax(dim=-1).clamp(min=torch.finfo(torch.float32).tiny)
            exponent = torch.ceil(torch.log2(amax / torch.finfo(torch.float8_e4m3fn).max))
            state_dict[name] = (blocks / torch.exp2(exponent)[..., None]).flatten(-2).to(torch.float8_e4m3fn)
            state_dict[f"{name}_scale"] = (exponent + 127).to(torch.uint8)
        elif "attn_res" in name or name.endswith("mlp.gate.bias"):
            state_dict[name] = tensor.float()
        else:
            state_dict[name] = tensor.to(torch.bfloat16)
    check_dtype_pattern(MODEL_ID, state_dict)
    save_file(state_dict, os.path.join(tmp_dir, "model.safetensors"), metadata={"format": "pt"})

    config.quantization_config = quantization_config
    config.save_pretrained(tmp_dir)
    # The reference also lists the dtype of every tensor in `hf_quant_config.json`, used by inference engines
    with open(hf_hub_download(MODEL_ID, "hf_quant_config.json")) as f:
        hf_quant_config = json.load(f)
    for group in hf_quant_config["model"]["parameter_groups"]:
        group["tensors"] = [name for name in group["tensors"] if name in state_dict]
    hf_quant_config["model"]["parameter_groups"] = [
        g for g in hf_quant_config["model"]["parameter_groups"] if g["tensors"]
    ]
    del hf_quant_config["mtp"]  # the tiny model has no multi-token prediction head
    with open(os.path.join(tmp_dir, "hf_quant_config.json"), "w") as f:
        json.dump(hf_quant_config, f, indent=2)

    push_dir_to_hub(tmp_dir, model, processor, generation_config, "tiny")
