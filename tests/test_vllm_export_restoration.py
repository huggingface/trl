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
from transformers.utils import is_peft_available

from .testing_utils import require_peft


if is_peft_available():
    from peft import LoraConfig, get_peft_model


@require_peft
def test_merged_export_restores_exact_base_weights(generation):
    generation.model = get_peft_model(
        torch.nn.Sequential(torch.nn.Linear(1, 1, bias=False)), LoraConfig(r=1, lora_alpha=1, target_modules=["0"])
    )
    layer = generation.model.base_model.model[0]
    with torch.no_grad():
        layer.base_layer.weight.fill_(0.1)  # 0.1 + 0.04 - 0.04 rounds away from 0.1 in float32
        layer.lora_A["default"].weight.fill_(0.2)
        layer.lora_B["default"].weight.fill_(0.2)
    before = layer.base_layer.weight.detach().clone()

    generation.sync_weights()

    torch.testing.assert_close(layer.base_layer.weight, before, rtol=0, atol=0)
