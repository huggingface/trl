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

import torch
from transformers import AutoConfig, AutoProcessor, Inkling2ForConditionalGeneration

from .._common import (
    check_dtype_pattern,
    check_transformers_version,
    print_config_diff,
    push_to_hub,
    set_seed,
    smoke_test,
)


check_transformers_version("5.23.0")

set_seed()

MODEL_ID = "thinkingmachines/Inkling-2-Small"

processor = AutoProcessor.from_pretrained(MODEL_ID)
# The reference tokenizer sets no eos or pad token (#8497): use the end-of-turn token and an unused one
processor.tokenizer.eos_token = "<|content_model_end_sampling|>"
processor.tokenizer.pad_token = "<|endoftext|>"

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
    "moe_intermediate_size": 16,
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
del config.quantization_config  # the tiny model is stored in bf16

model = Inkling2ForConditionalGeneration(config).to(dtype=torch.bfloat16)
# Restore float32 for the parameters the reference keeps in fp32 (router, convolutions, attention residual inputs)
for name, param in model.named_parameters():
    if any(key in name for key in model._keep_in_fp32_modules_strict):
        param.data = param.data.float()
smoke_test(model, processor)
check_dtype_pattern(MODEL_ID, model)
print_config_diff(MODEL_ID, model)
# The reference repo ships no generation config
push_to_hub(model, processor, None, "tiny")
