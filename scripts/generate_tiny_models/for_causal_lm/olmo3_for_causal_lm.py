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
from transformers import AutoTokenizer, GenerationConfig, Olmo3Config, Olmo3ForCausalLM

from .._common import (
    check_dtype_pattern,
    check_transformers_version,
    init_weights_tiny_model,
    print_config_diff,
    push_to_hub,
    smoke_test,
)


# 4.57.0 (the release that introduced Olmo 3) was yanked for a packaging issue, so pin the first non-yanked patch.
check_transformers_version("4.57.1")

MODEL_ID = "allenai/Olmo-3-7B-Think"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
generation_config = GenerationConfig.from_pretrained(MODEL_ID)
config = Olmo3Config(
    vocab_size=len(tokenizer.vocab),
    hidden_size=8,
    num_attention_heads=4,
    num_key_value_heads=2,
    num_hidden_layers=2,
    intermediate_size=32,
    # Non-size fields kept aligned with the reference so the tiny config only differs in what we scale down.
    max_position_embeddings=65536,
    rms_norm_eps=1e-06,
    rope_theta=500000,
    rope_scaling={
        "attention_factor": 1.2079441541679836,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "factor": 8.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "yarn",
    },
    bos_token_id=None,
    eos_token_id=100257,
    pad_token_id=100277,
    use_cache=False,
)
model = Olmo3ForCausalLM(config).to(dtype=torch.bfloat16)
init_weights_tiny_model(model)
smoke_test(model, tokenizer)
check_dtype_pattern(MODEL_ID, model)
print_config_diff(MODEL_ID, model)
push_to_hub(model, tokenizer, generation_config, "tiny")
