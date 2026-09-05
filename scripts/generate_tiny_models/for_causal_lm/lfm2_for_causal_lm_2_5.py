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
from transformers import AutoTokenizer, GenerationConfig, Lfm2Config, Lfm2ForCausalLM

from .._common import (
    check_dtype_pattern,
    check_transformers_version,
    init_weights_tiny_model,
    print_config_diff,
    push_to_hub,
    smoke_test,
)


# LFM2.5 ships a `TokenizersBackend` tokenizer, which was introduced in transformers 5.0.0. This is above TRL's
# transformers floor, so unlike the other tiny models this one can't be loaded by the floor CI job, and the tests
# using it are skipped there.
check_transformers_version("5.0.0")

MODEL_ID = "LiquidAI/LFM2.5-230M"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
generation_config = GenerationConfig.from_pretrained(MODEL_ID)
config = Lfm2Config(
    vocab_size=65536,
    hidden_size=8,
    num_attention_heads=4,
    num_key_value_heads=2,
    num_hidden_layers=2,
    intermediate_size=32,
    # LFM2 interleaves short convolution layers with full attention layers. The reference spells the pattern out with
    # `layer_types` rather than `full_attn_idxs`; keep one of each so both layer types are exercised.
    layer_types=["conv", "full_attention"],
    # Non-size fields kept aligned with the reference so the tiny config only differs in what we scale down.
    max_position_embeddings=128000,
    norm_eps=1e-05,
    rope_parameters={"rope_theta": 1000000.0, "rope_type": "default"},
    conv_bias=False,
    conv_L_cache=3,
    block_auto_adjust_ff_dim=False,
    block_multiple_of=256,
    block_ffn_dim_multiplier=1.0,
    bos_token_id=1,
    # The reference tokenizer's EOS is <|im_end|> (7), not <|endoftext|> (2) which `Lfm2Config` defaults to.
    eos_token_id=7,
    pad_token_id=0,
    use_cache=False,
)
model = Lfm2ForCausalLM(config).to(dtype=torch.bfloat16)
init_weights_tiny_model(model)
smoke_test(model, tokenizer)
check_dtype_pattern(MODEL_ID, model)
print_config_diff(MODEL_ID, model)
push_to_hub(model, tokenizer, generation_config, "tiny", "2.5")
