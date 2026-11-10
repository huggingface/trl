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
import re
import tempfile

import torch
from safetensors.torch import load_file, save_file
from transformers import (
    AutoProcessor,
    GenerationConfig,
    Glm5NextConfig,
    Glm5NextForConditionalGeneration,
    Glm5NextTextConfig,
    Glm5NextVisionConfig,
)

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

MODEL_ID = "zai-org/GLM-5.3-Flash"

processor = AutoProcessor.from_pretrained(MODEL_ID)
generation_config = GenerationConfig.from_pretrained(MODEL_ID)
# Unset: save_pretrained rejects top_p without do_sample (the reference decodes greedily by default anyway)
generation_config.top_p = None

text_config = Glm5NextTextConfig(
    vocab_size=154880,
    # 128 for the dimensions that FP8 linears take as input: the FP8 activations are quantized in blocks of 128
    hidden_size=128,
    intermediate_size=128,
    moe_intermediate_size=128,
    # 4 layers, so that each per-layer pattern of the reference appears at least once: 3 linear-attention layers then
    # a sparse-attention one, and 3 dense MLP layers then a sparse one.
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=4,
    n_shared_experts=1,
    n_routed_experts=4,
    num_experts_per_tok=2,
    kv_lora_rank=32,
    q_lora_rank=128,
    qk_rope_head_dim=0,
    qk_nope_head_dim=32,
    v_head_dim=32,
    index_head_dim=4,
    index_n_heads=2,
    # Smaller than the test sequences, so the sparse attention path actually drops keys.
    index_topk=8,
    index_kpool=4,
    linear_attn_config={
        "num_heads": 4,
        "gate_lower_bound": -5.0,
        "head_dim": 8,
        "short_conv_kernel_size": 4,
        "kda_layers": [0, 1, 2],
        "full_attn_layers": [3],
    },
    first_k_dense_replace=3,
    index_kpool_compress=True,
    index_share_for_mtp_iteration=True,
    indexer_rope_interleave=True,
    mhc=True,
    mla_use_nope=True,
    moe_router_dtype="float32",
    num_nextn_predict_layers=1,
    scoring_func="sigmoid",
    topk_method="noaux_tc",
    eos_token_id=[154820, 154827, 154829],
    pad_token_id=154820,
)
vision_config = Glm5NextVisionConfig(
    depth=2,
    hidden_size=16,
    num_heads=4,
    intermediate_size=32,
    out_hidden_size=128,
    projection_intermediate_size=32,
    image_size=448,
)
config = Glm5NextConfig(text_config=text_config, vision_config=vision_config)
model = Glm5NextForConditionalGeneration(config).to(dtype=torch.bfloat16)
init_weights_tiny_model(model)

# Restore float32 for the parameters the reference keeps in fp32: the mHC base and scale, the linear attention's decay
# parameters and the router's correction bias
for layer in model.model.language_model.layers:
    for hc in (layer.attn_hc, layer.ffn_hc):
        hc.base.data = hc.base.data.float()
        hc.scale.data = hc.scale.data.float()
for layer in model.model.language_model.layers[:3]:
    layer.self_attn.forget_gate.A_log.data = layer.self_attn.forget_gate.A_log.data.float()
    layer.self_attn.forget_gate.dt_bias.data = layer.self_attn.forget_gate.dt_bias.data.float()
for layer in model.model.language_model.layers[3:]:
    layer.mlp.gate.e_score_correction_bias.data = layer.mlp.gate.e_score_correction_bias.data.float()

# FP8 weights, like the reference checkpoint: quantize the linears the reference stores in FP8 (the MLPs, the experts and
# the sparse-attention projections) in 128x128 blocks, each block scaled to the e4m3 range, with the scale of each block
# stored in `weight_scale_inv`. The modules the reference keeps unquantized are listed with its checkpoint names, scaled
# down to the 4 layers (plus the MTP layer 4) and 2 vision blocks of the tiny model.
linear_attention_modules = ["A_log", "b_proj", "dt_bias", "f_a_proj", "f_b_proj", "fused_qkvbfg_a_proj", "g_a_proj"]
linear_attention_modules += ["g_b_proj", "k_conv1d", "k_proj", "o_norm", "o_proj", "q_conv1d", "q_proj", "qkv_proj"]
linear_attention_modules += ["v_conv1d", "v_proj"]
sparse_layer_modules = ["mlp.gate", "mlp.gate.e_score_correction_bias", "self_attn.kv_a_layernorm"]
sparse_layer_modules += ["self_attn.kv_b_proj", "self_attn.q_a_layernorm"]
sparse_layer_modules += [
    f"self_attn.indexer.{name}"
    for name in ["index_kpool_compress_ape", "index_kpool_compress_gate", "k_norm", "k_norm.bias", "weights_proj"]
    + ["wk", "wq_b"]
]
vision_block_modules = ["attn.k_norm", "attn.proj", "attn.proj.bias", "attn.q_norm", "attn.qkv", "attn.qkv.bias"]
vision_block_modules += [
    f"mlp.{proj}{suffix}" for proj in ["down_proj", "gate_proj", "up_proj"] for suffix in ["", ".bias"]
]
vision_block_modules += ["norm1", "norm2"]
modules_to_not_convert = ["attn_mha", "attn_mqa", "dt_bias", "hyper_connection", "lm_head", "mapping_proj"]
modules_to_not_convert += ["model.embed_tokens", "model.norm", "model.visual", "router", "visual", "visual.downsample"]
modules_to_not_convert += ["visual.downsample.bias", "visual.merger.down_proj", "visual.merger.gate_proj"]
modules_to_not_convert += ["visual.merger.post_projection_norm", "visual.merger.post_projection_norm.bias"]
modules_to_not_convert += ["visual.merger.proj", "visual.merger.up_proj", "visual.patch_embed.proj"]
modules_to_not_convert += ["visual.patch_embed.proj.bias", "visual.post_layernorm", "weights_proj"]
modules_to_not_convert += [
    f"model.layers.{i}.hc_{hc}_{name}" for i in range(4) for hc in ["attn", "ffn"] for name in ["base", "fn", "scale"]
]
modules_to_not_convert += [
    f"model.layers.{i}.{norm}" for i in range(5) for norm in ["input_layernorm", "post_attention_layernorm"]
]
modules_to_not_convert += [f"model.layers.{i}.self_attn.{name}" for i in range(3) for name in linear_attention_modules]
modules_to_not_convert += [f"model.layers.{i}.{name}" for i in [3, 4] for name in sparse_layer_modules]
modules_to_not_convert += [f"model.layers.4.{name}" for name in ["eh_proj", "enorm", "hnorm", "shared_head.norm"]]
modules_to_not_convert += [f"visual.blocks.{i}.{name}" for i in range(2) for name in vision_block_modules]
fp8_weights = re.compile(
    r"model\.language_model\.layers\.(\d+\.mlp\.(experts\.\d+\.|shared_experts\.)?(gate|up|down)_proj"
    r"|3\.self_attn\.(q_a_proj|q_b_proj|kv_a_proj_with_mqa|o_proj))\.weight"
)
with tempfile.TemporaryDirectory() as tmp_dir:
    # Saved with the checkpoint names of the reference (unfused experts, one short convolution per projection)
    model.save_pretrained(tmp_dir)
    state_dict = load_file(os.path.join(tmp_dir, "model.safetensors"))
    for name, weight in list(state_dict.items()):
        if fp8_weights.fullmatch(name):
            weight = weight.float()
            rows, cols = weight.shape
            # The last blocks are partial where a dimension is not a multiple of 128 (kv_a_proj_with_mqa's 32 rows)
            padded = torch.zeros(math.ceil(rows / 128) * 128, math.ceil(cols / 128) * 128)
            padded[:rows, :cols] = weight
            blocks = padded.view(padded.shape[0] // 128, 128, padded.shape[1] // 128, 128)
            scale = blocks.abs().amax(dim=(1, 3)) / torch.finfo(torch.float8_e4m3fn).max
            quantized = (blocks / scale[:, None, :, None]).view(padded.shape)[:rows, :cols]
            state_dict[name] = quantized.to(torch.float8_e4m3fn).contiguous()
            state_dict[name.removesuffix("weight") + "weight_scale_inv"] = scale
    config.quantization_config = {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "modules_to_not_convert": sorted(modules_to_not_convert),
        "quant_method": "fp8",
        "weight_block_size": [128, 128],
    }
    check_dtype_pattern(MODEL_ID, state_dict)
    print_config_diff(MODEL_ID, model)
    config.save_pretrained(tmp_dir)
    save_file(state_dict, os.path.join(tmp_dir, "model.safetensors"), metadata={"format": "pt"})
    model = Glm5NextForConditionalGeneration.from_pretrained(tmp_dir, device_map="cuda")
    smoke_test(model, processor)
    push_dir_to_hub(tmp_dir, model, processor, generation_config, "tiny")
