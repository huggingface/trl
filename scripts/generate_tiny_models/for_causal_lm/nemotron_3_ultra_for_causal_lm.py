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
from transformers import AutoTokenizer, GenerationConfig, NemotronHConfig, NemotronHForCausalLM

from .._common import (
    check_dtype_pattern,
    check_transformers_version,
    init_weights_tiny_model,
    print_config_diff,
    push_to_hub,
    smoke_test,
)


check_transformers_version("5.3.0")

MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
generation_config = GenerationConfig.from_pretrained(MODEL_ID)
config = NemotronHConfig(
    vocab_size=len(tokenizer.vocab),
    hidden_size=16,
    num_attention_heads=4,
    num_key_value_heads=2,
    intermediate_size=32,
    layers_block_type=["mamba", "attention", "moe"],  # one of each block type
    mamba_num_heads=8,
    mamba_head_dim=4,
    mamba_n_groups=1,
    ssm_state_size=16,
    mamba_d_conv=4,
    mamba_expand=2,
    n_routed_experts=4,
    num_experts_per_tok=2,
    moe_intermediate_size=32,
    moe_shared_expert_intermediate_size=32,
    use_mamba_kernels=False,  # CPU-friendly for testing
)
# Unlike the Nano checkpoint, the Ultra checkpoint keeps the Mamba mixer weights in bfloat16, so no fp32 restore here.
model = NemotronHForCausalLM(config).to(dtype=torch.bfloat16)
init_weights_tiny_model(model)
smoke_test(model, tokenizer)
check_dtype_pattern(MODEL_ID, model)
print_config_diff(MODEL_ID, model)
push_to_hub(model, tokenizer, generation_config, "tiny", "ultra")
