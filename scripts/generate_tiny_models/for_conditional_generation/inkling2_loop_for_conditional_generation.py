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
from transformers import AutoConfig, AutoProcessor, GenerationConfig, Inkling2LoopForConditionalGeneration

from .._common import (
    check_dtype_pattern,
    check_transformers_version,
    init_weights_tiny_model,
    print_config_diff,
    push_dir_to_hub,
    set_seed,
    smoke_test,
)


check_transformers_version("5.24.0")

set_seed()

MODEL_ID = "thinkingmachines/Inkling-2-Mini"

processor = AutoProcessor.from_pretrained(MODEL_ID)
generation_config = GenerationConfig.from_pretrained(MODEL_ID)

# Attention residual blocks of one layer of each type: one before the looped core, two in it (a pass reads the
# previous pass's blocks after the first one) and one after it (the reference has 24 layers in blocks of six, layers 6
# to 17 looped three times)
text_config = {
    "num_hidden_layers": 8,
    "layer_types": ["hybrid_sliding", "hybrid"] * 4,
    "n_layers_per_attn_res_block": 2,
    "loop_config": {"core_start": 2, "core_end": 6, "num_passes": 3, "gate_rank": 8, "is_varloop": False},
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
# The config derives the loop range from `loop_config` when it's built, so rebuild it
config.text_config = type(config.text_config)(**config.text_config.to_dict())
for k, v in vision_config.items():
    setattr(config.vision_config, k, v)

model = Inkling2LoopForConditionalGeneration(config).to(dtype=torch.bfloat16)
init_weights_tiny_model(model)
# The modeling zero-initializes the attention residual queries and the static convolution weights, the reference's
# aren't zero
with torch.no_grad():
    for name, param in model.named_parameters():
        if name.endswith(".query"):
            param.normal_(0, 0.5)
        elif name.endswith(".static_weight"):
            param.normal_(0, 0.02)
# The modeling keeps these parameters in float32 (router, convolutions, residual and message inputs)
for name, param in model.named_parameters():
    if any(key in name for key in model._keep_in_fp32_modules_strict):
        param.data = param.data.float()
print_config_diff(MODEL_ID, model)

E2M1_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def quantize_nvfp4(weight):
    # Two E2M1 codes per byte (even element in the low nibble), one float8 scale per 16 values, and a per-expert factor
    # `amax / outer_scale_denominator`
    amax = weight.float().abs().amax(dim=(1, 2))
    denominator = torch.tensor([448.0 * 6.0])
    factor = (amax / denominator)[:, None, None]
    blocks = weight.float().unflatten(-1, (-1, 16))
    scale = (blocks.abs().amax(dim=-1) / 6 / factor).clamp(min=2**-9).to(torch.float8_e4m3fn)
    values = blocks / (scale.float() * factor)[..., None]
    codes = (values.abs()[..., None] - E2M1_VALUES).abs().argmin(dim=-1) + 8 * (values < 0)
    codes = codes.flatten(-2)
    return (codes[..., 0::2] | codes[..., 1::2] << 4).to(torch.uint8), scale, amax, denominator


def quantize_mxfp8(weight):
    # Float8 values with one power-of-two scale `2 ** (scale - 127)` per 32 values
    blocks = weight.float().unflatten(-1, (-1, 32))
    amax = blocks.abs().amax(dim=-1).clamp(min=torch.finfo(torch.float32).tiny)
    exponent = torch.ceil(torch.log2(amax / torch.finfo(torch.float8_e4m3fn).max))
    return (blocks / torch.exp2(exponent)[..., None]).flatten(-2).to(torch.float8_e4m3fn), (exponent + 127).to(
        torch.uint8
    )


with tempfile.TemporaryDirectory() as tmp_dir:
    # Save in the reference layout, then store each tensor like the reference: the experts of the last attention
    # residual block in MXFP8 and the others in NVFP4 (the reference: layers 20 to 23 in MXFP8, so the dtype check
    # flags the tiny model's last block), the residual and loop queries and the router bias in float32, everything
    # else in bf16
    model.save_pretrained(tmp_dir)
    state_dict = load_file(os.path.join(tmp_dir, "model.safetensors"))
    last_block = config.text_config.num_hidden_layers - config.text_config.n_layers_per_attn_res_block
    for name, tensor in list(state_dict.items()):
        if ".mlp.experts.w" in name:
            if int(name.split(".layers.")[1].split(".")[0]) >= last_block:
                state_dict[name], state_dict[f"{name}_scale"] = quantize_mxfp8(tensor)
            else:
                packed, scale, amax, denominator = quantize_nvfp4(tensor)
                state_dict[name], state_dict[f"{name}_scale"] = packed, scale
                state_dict[f"{name}_amax"], state_dict[f"{name}_outer_scale_denominator"] = amax, denominator
        elif "attn_res" in name or "cross_pass" in name or name.endswith("mlp.gate.bias"):
            state_dict[name] = tensor.float()
        else:
            state_dict[name] = tensor.to(torch.bfloat16)
    check_dtype_pattern(MODEL_ID, state_dict)
    save_file(state_dict, os.path.join(tmp_dir, "model.safetensors"), metadata={"format": "pt"})

    # The reference also lists the format of every tensor in `hf_quant_config.json`, used by inference engines
    with open(hf_hub_download(MODEL_ID, "hf_quant_config.json")) as f:
        hf_quant_config = json.load(f)
    expert_weights = [name for name in state_dict if name.endswith(("experts.w13_weight", "experts.w2_weight"))]
    for group in hf_quant_config["model"]["parameter_groups"]:
        if group["compute_dtype"] == "nvfp4":
            group["tensors"] = [name for name in expert_weights if state_dict[name].dtype == torch.uint8]
        elif group["compute_dtype"] == "mxfp8":
            group["tensors"] = [name for name in expert_weights if state_dict[name].dtype == torch.float8_e4m3fn]
        else:
            group["tensors"] = [name for name in group["tensors"] if name in state_dict]
    hf_quant_config["model"]["parameter_groups"] = [
        g for g in hf_quant_config["model"]["parameter_groups"] if g["tensors"]
    ]
    del hf_quant_config["mtp"]  # the tiny model has no multi-token prediction head
    with open(os.path.join(tmp_dir, "hf_quant_config.json"), "w") as f:
        json.dump(hf_quant_config, f, indent=2)

    model = Inkling2LoopForConditionalGeneration.from_pretrained(tmp_dir, device_map="cuda")
    smoke_test(model, processor)
    push_dir_to_hub(tmp_dir, model, processor, generation_config, "tiny")
