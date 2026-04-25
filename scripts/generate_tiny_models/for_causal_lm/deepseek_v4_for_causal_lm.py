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

# WIP: targets the in-review transformers branch that adds DeepseekV4 and the in-review model PR on
# the Hub. This script exists to provide early feedback on the integration; expect the pinned version
# and revision to change once V4 lands upstream.

import torch
from transformers import AutoTokenizer, DeepseekV4Config, DeepseekV4ForCausalLM, GenerationConfig
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4HyperConnection, DeepseekV4HyperHead

from .._common import (
    check_dtype_pattern,
    check_transformers_version,
    init_weights_tiny_model,
    print_config_diff,
    push_to_hub,
    smoke_test,
)


TRANSFORMERS_VERSION = "5.7.0.dev0"
check_transformers_version(TRANSFORMERS_VERSION)

MODEL_ID = "deepseek-ai/DeepSeek-V4-Flash"
REVISION = "refs/pr/16"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
generation_config = GenerationConfig.from_pretrained(MODEL_ID, revision=REVISION)
_yarn_rope = {
    "beta_fast": 32,
    "beta_slow": 1,
    "factor": 16,
    "original_max_position_embeddings": 65536,
    "rope_type": "yarn",
    "type": "yarn",
}
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
    compress_ratios=[0, 0, 4, 128],
    first_k_dense_replace=2,  # layers 0-1 dense, layers 2-3 MoE
    num_hash_layers=3,  # layer 2 hash-routed (MoE), layer 3 learned-gate (MoE)
    n_routed_experts=4,
    num_experts_per_tok=2,
    topk_method="noaux_tc",
    rope_parameters=_yarn_rope,
    compress_rope_parameters=_yarn_rope,
)
model = DeepseekV4ForCausalLM(config).to(dtype=torch.bfloat16)
init_weights_tiny_model(model)

# Hyper-connection params (hc_head/attn_hc/ffn_hc base/fn/scale), attention sinks, and compressor
# absolute-positional-embedding parameters are kept in fp32 in the reference. Restore them after
# the global bf16 cast.
def _should_be_fp32(name):
    return name.endswith(".sinks") or name.endswith(".ape") or name.endswith(".mlp.gate.bias")


for name, param in model.named_parameters():
    if _should_be_fp32(name):
        param.data = param.data.float()
# Covers buffers (e.g. mlp.gate.bias — DeepSeek's non-learnable routing-balance correction,
# registered as a buffer upstream, not a parameter).
for name, buf in model.named_buffers():
    if _should_be_fp32(name):
        buf.data = buf.data.float()
for module in model.modules():
    if isinstance(module, (DeepseekV4HyperConnection, DeepseekV4HyperHead)):
        for p in module.parameters(recurse=False):
            p.data = p.data.float()

smoke_test(model, tokenizer)

# Reference checkpoint is FP8-quantized on attention projections and shared/routed experts. The
# upstream serialization labels those tensors with safetensors `dtype=I8` (the bytes are FP8 E4M3,
# but the metadata field is int8); the runtime quantizer reinterprets them via
# `quantization_config.fmt=e4m3`. To match that on-disk label we cast to torch.int8 here — the saved
# tiny is byte-equivalent FP8 but reports the same I8 dtype as the reference. Done after smoke_test
# because the cast severs the nn.Linear forward path; this tiny is a storage artefact, not runnable.
_FP8_WEIGHT_SUFFIXES = (
    ".self_attn.wq_a.weight",
    ".self_attn.wq_b.weight",
    ".self_attn.wkv.weight",
    ".self_attn.wo_a.weight",
    ".self_attn.wo_b.weight",
    ".self_attn.compressor.indexer.wq_b.weight",
    ".mlp.shared_experts.gate_proj.weight",
    ".mlp.shared_experts.up_proj.weight",
    ".mlp.shared_experts.down_proj.weight",
    ".mlp.experts.gate_up_proj",
    ".mlp.experts.down_proj",
)
for name, param in model.named_parameters():
    if name.endswith(_FP8_WEIGHT_SUFFIXES):
        # int8 isn't a float/complex dtype, so PyTorch won't let us reassign `.data` on a
        # gradient-requiring parameter. Drop the grad requirement, then swap the storage.
        param.requires_grad_(False)
        param.data = param.data.to(torch.int8)

check_dtype_pattern(MODEL_ID, model, revision=REVISION)
print_config_diff(MODEL_ID, model, revision=REVISION)
push_to_hub(model, tokenizer, generation_config, "tiny")
