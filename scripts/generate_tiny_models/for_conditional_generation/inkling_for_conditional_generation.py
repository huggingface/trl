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

import copy

import torch
from transformers import AutoProcessor, InklingConfig, InklingForConditionalGeneration

from .._common import (
    check_dtype_pattern,
    check_transformers_version,
    init_weights_tiny_model,
    print_config_diff,
    push_to_hub,
    set_seed,
    smoke_test,
)


check_transformers_version("5.15.0")

set_seed()

MODEL_ID = "thinkingmachines/Inkling"

processor = AutoProcessor.from_pretrained(MODEL_ID)

# The sub-configs are passed with the keys of the reference checkpoint, which `InklingConfig` converts the same way when
# it loads the reference (`dense_mlp_idx`, `dense_intermediate_size`, `mtp_config`, ...).
text_config = {
    "model_max_length": 1048576,
    "hidden_size": 16,
    # 4 layers, so that each per-layer pattern of the reference appears at least once: sliding-window layers then a
    # global one (the reference makes every 6th layer global), and dense MLP layers then sparse ones.
    "num_hidden_layers": 4,
    "local_layer_ids": [0, 1, 2],
    "dense_mlp_idx": 2,
    "vocab_size": 201024,
    "unpadded_vocab_size": 200058,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 8,
    "swa_num_attention_heads": 4,
    "swa_num_key_value_heads": 2,
    "swa_head_dim": 8,
    "d_rel": 16,
    "rel_extent": 1024,
    "q_bias": False,
    "o_bias": False,
    "log_scaling_n_floor": 128000,
    "log_scaling_alpha": 0.1,
    "rms_norm_eps": 1e-06,
    "use_embed_norm": True,
    "use_sconv": True,
    "sconv_kernel_size": 4,
    "logits_mup_width_multiplier": 24.0,
    "final_logit_softcapping": None,
    "sliding_window_size": 512,
    "n_routed_experts": 4,
    "num_experts_per_tok": 2,
    "n_shared_experts": 2,
    "shared_expert_sink": True,
    "dense_intermediate_size": 32,
    # The reference's MoE expert size is the config default, which its `intermediate_size` key repeats
    "intermediate_size": 32,
    "moe_intermediate_size": 32,
    "route_scale": 8.0,
    "use_gate_bias": True,
    "gate_activation": "sigmoid",
    "norm_after_topk": True,
    "use_global_scale": True,
}
audio_config = {
    "decoder_dmodel": 16,
    "n_mel_bins": 80,
    "mel_vocab_size": 16,
    "bias": False,
    "dmel_min_value": -7.0,
    "dmel_max_value": 2.0,
    "use_audio_norm": True,
    "audio_mode": "dmel",
}
# The hierarchical-MLP vision encoder derives its widths from the patch size and its 4 layers, so both are kept
vision_config = {
    "vision_encoder_type": "hmlp",
    "decoder_dmodel": 16,
    "patch_size": 40,
    "temporal_patch_size": 2,
    "n_channels": 3,
    "n_layers": 4,
    "use_vision_norm": True,
}
mtp_config = {"num_nextn_predict_layers": 8, "chain_hidden_post_norm": False, "local_layer_ids": [0, 2, 4, 5, 6, 7]}
config = InklingConfig(
    text_config=text_config,
    audio_config=audio_config,
    vision_config=vision_config,
    mtp_config=mtp_config,
    eos_token_id=200006,
)
model = InklingForConditionalGeneration(config)
init_weights_tiny_model(model)
# The reference tokenizer has no pad token, which smoke_test needs to batch its two conversations: set one on a copy, so
# the pushed tokenizer stays as the reference's. The test runs before the cast to bfloat16 because the vision tower does
# not cast the float32 `pixel_values` to the model dtype.
smoke_processor = copy.deepcopy(processor)
smoke_processor.tokenizer.pad_token = "<|endoftext|>"
smoke_test(model, smoke_processor)
model = model.to(dtype=torch.bfloat16)
# Restore float32 for the router parameters the reference keeps in fp32
for layer in model.model.language_model.layers[2:]:
    layer.mlp.gate.e_score_correction_bias.data = layer.mlp.gate.e_score_correction_bias.data.float()
    layer.mlp.gate.global_scale.data = layer.mlp.gate.global_scale.data.float()
check_dtype_pattern(MODEL_ID, model)
print_config_diff(MODEL_ID, model)
push_to_hub(model, processor, None, "tiny")
