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

import os
import tempfile

import torch
from safetensors.torch import save_file
from torch import nn
from transformers import AutoConfig, AutoProcessor, GenerationConfig, Mistral3ForConditionalGeneration

from .._common import (
    check_dtype_pattern,
    check_transformers_version,
    print_config_diff,
    push_dir_to_hub,
    set_seed,
    smoke_test,
)


check_transformers_version("5.20.1")

set_seed()

MODEL_ID = "mistralai/Mistral-Medium-3.5-128B"

processor = AutoProcessor.from_pretrained(MODEL_ID)
generation_config = GenerationConfig.from_pretrained(MODEL_ID)

text_config = {
    "num_hidden_layers": 2,
    "hidden_size": 16,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 4,
    "intermediate_size": 32,
}
vision_config = {
    "num_hidden_layers": 2,
    "hidden_size": 16,
    "num_attention_heads": 4,
    "head_dim": 4,
    "intermediate_size": 32,
}

config = AutoConfig.from_pretrained(MODEL_ID)
config.text_config.update(text_config)
config.vision_config.update(vision_config)
# Set aside: the model is built in bfloat16, then quantized below
quantization_config = config.quantization_config
del config.quantization_config
model = Mistral3ForConditionalGeneration(config).to(dtype=torch.bfloat16)

# Its activation scheme is static: calibrate each linear's input scale on a short conversation
inputs_absmax = {}


def record_absmax(name):
    def hook(module, args):
        inputs_absmax[name] = max(inputs_absmax.get(name, 0.0), args[0].abs().max().item())

    return hook


hooks = [m.register_forward_pre_hook(record_absmax(n)) for n, m in model.named_modules() if isinstance(m, nn.Linear)]
conversation = [{"role": "user", "content": [{"type": "text", "text": "What is the capital of France?"}]}]
model(**processor.apply_chat_template(conversation, tokenize=True, return_dict=True, return_tensors="pt"))
for hook in hooks:
    hook.remove()

# FP8 weights, like the reference checkpoint: quantize the language model's linears (the reference keeps the vision
# tower, the projector and lm_head in bfloat16) per tensor, with the weight and input scales stored in bfloat16
fp8_max = torch.finfo(torch.float8_e4m3fn).max
state_dict = model.state_dict()
for name, module in model.named_modules():
    if isinstance(module, nn.Linear) and name.startswith("model.language_model."):
        weight = module.weight.float()
        scale = weight.abs().max() / fp8_max
        state_dict[f"{name}.weight"] = (weight / scale).to(torch.float8_e4m3fn)
        state_dict[f"{name}.weight_scale_inv"] = scale.to(torch.bfloat16)
        state_dict[f"{name}.activation_scale"] = torch.tensor(inputs_absmax[name] / fp8_max, dtype=torch.bfloat16)
config.quantization_config = quantization_config
check_dtype_pattern(MODEL_ID, state_dict)
print_config_diff(MODEL_ID, model)

with tempfile.TemporaryDirectory() as tmp_dir:
    # Written with the in-memory names, which are the reference's (`save_pretrained` writes the legacy layout)
    config.save_pretrained(tmp_dir)
    save_file(state_dict, os.path.join(tmp_dir, "model.safetensors"), metadata={"format": "pt"})
    model = Mistral3ForConditionalGeneration.from_pretrained(tmp_dir, device_map="cuda")
    smoke_test(model, processor)
    push_dir_to_hub(tmp_dir, model, processor, generation_config, "tiny")
