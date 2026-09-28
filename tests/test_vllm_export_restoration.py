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
from transformers.utils import is_bitsandbytes_available, is_peft_available

from .testing_utils import require_bitsandbytes, require_peft


if is_peft_available():
    from peft import LoraConfig, get_peft_model

if is_bitsandbytes_available():
    import bitsandbytes as bnb


@require_peft
def test_merged_export_restores_exact_base_weights(vllm_generation):
    vllm_generation.model = get_peft_model(
        torch.nn.Sequential(torch.nn.Linear(1, 1, bias=False)), LoraConfig(r=1, lora_alpha=1, target_modules=["0"])
    )
    layer = vllm_generation.model.base_model.model[0]
    with torch.no_grad():
        layer.base_layer.weight.fill_(0.1)  # 0.1 + 0.04 - 0.04 != 0.1 in float32
        layer.lora_A["default"].weight.fill_(0.2)
        layer.lora_B["default"].weight.fill_(0.2)
    before = layer.base_layer.weight.detach().clone()

    vllm_generation.sync_weights()

    torch.testing.assert_close(layer.base_layer.weight, before, rtol=0, atol=0)


@require_peft
@require_bitsandbytes
def test_merged_export_restores_exact_quantized_base_weights(vllm_generation):
    torch.manual_seed(0)
    model = torch.nn.Sequential(bnb.nn.Linear4bit(64, 64, bias=False, compute_dtype=torch.float32).to("cpu"))
    model.is_loaded_in_4bit = True  # Use PEFT's bitsandbytes LoRA layer
    vllm_generation.model = get_peft_model(model, LoraConfig(r=1, target_modules=["0"], init_lora_weights=False))
    layer = vllm_generation.model.base_model.model[0]
    weight = layer.base_layer.weight
    before = weight.detach().clone()

    vllm_generation.sync_weights()
    assert layer.base_layer.weight is weight
    torch.testing.assert_close(layer.base_layer.weight, before, rtol=0, atol=0)
