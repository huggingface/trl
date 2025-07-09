# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
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

import config
import torch
from torch.utils import benchmark
from transformers import AutoModelForCausalLM, AutoTokenizer


def generate_tokens(model, inputs):
    outputs = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=64,
    )
    return outputs


def generate_tokens_with_assistance(model, inputs, assistant_early_exit):
    outputs = model.generate(
        **inputs,
        assistant_early_exit=assistant_early_exit,
        do_sample=False,
        max_new_tokens=64,
    )
    return outputs


if __name__ == "__main__":
    ckpt = config.hub_model_id

    model = AutoModelForCausalLM.from_pretrained(ckpt, device_map="auto", torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(ckpt)

    prompt = "### Instruction: What are my alarms for the rest of the day?\n ### Response: "

    results = []
    label = "Generation Times"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    results.append(
        benchmark.Timer(
            stmt="generate_tokens(model, inputs)",
            setup="from __main__ import generate_tokens",
            globals={"model": model, "inputs": inputs},
            num_threads=torch.get_num_threads(),
            label=label,
            sub_label="no layer skip",
            description="generation",
        ).blocked_autorange()
    )

    for i in range(1, model.config.num_hidden_layers):
        results.append(
            benchmark.Timer(
                stmt="generate_tokens_with_assistance(model, inputs, assistant_early_exit)",
                setup="from __main__ import generate_assistant_tokens",
                globals={"model": model, "assistant_early_exit": i, "inputs": inputs},
                num_threads=torch.get_num_threads(),
                label=label,
                sub_label=f"layer skip {i}",
                description="generation",
            ).blocked_autorange()
        )

    benchmark.Compare(results).print()
