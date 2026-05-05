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

# Slightly bigger than the "tiny" variant: vLLM requires hidden_size // num_attention_heads = 32.

import torch
from transformers import AutoTokenizer, GenerationConfig, Qwen3Config, Qwen3ForCausalLM

from .._common import check_dtype_pattern, check_transformers_version, print_config_diff, push_to_hub, smoke_test


check_transformers_version()

MODEL_ID = "Qwen/Qwen3-4B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
generation_config = GenerationConfig.from_pretrained(MODEL_ID)
config = Qwen3Config(
    vocab_size=len(tokenizer.vocab),
    hidden_size=128,
    num_attention_heads=4,
    num_key_value_heads=2,
    num_hidden_layers=2,
    intermediate_size=32,
)
model = Qwen3ForCausalLM(config).to(dtype=torch.bfloat16)
smoke_test(model, tokenizer)
check_dtype_pattern(MODEL_ID, model)
print_config_diff(MODEL_ID, model)
push_to_hub(model, tokenizer, generation_config, "small")
