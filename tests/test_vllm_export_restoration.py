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

import pytest
import torch


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("use_dora", [False, True])
def test_bf16_repeated_exports_preserve_policy_and_publish_adapter_updates(generation, use_dora, dtype):
    import copy

    peft = pytest.importorskip("peft")
    from transformers import LlamaConfig, LlamaForCausalLM

    with torch.random.fork_rng():
        torch.manual_seed(917)
        model = LlamaForCausalLM(
            LlamaConfig(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
            )
        ).to(dtype)
        model = peft.get_peft_model(
            model, peft.LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"], use_dora=use_dora)
        ).eval()
        with torch.no_grad():
            for name, param in model.named_parameters():
                if "lora_B" in name:
                    param.normal_(std=0.2)
    generation.model = model
    ids = torch.tensor([[1, 5, 9, 3]])
    previous = None
    with torch.no_grad():
        for update in range(2):
            if update:
                for name, param in model.named_parameters():
                    if "lora_B" in name:
                        param.add_(0.07)
            training_state = {name: value.clone() for name, value in model.state_dict().items()}
            logprobs = model(ids).logits.float().log_softmax(-1)
            # A separately merged copy is the reference for exported inference weights. Merged BF16 inference
            # and the unmerged adapter forward need not be bit-identical: their arithmetic differs.
            reference = copy.deepcopy(model).merge_and_unload()
            expected = reference.state_dict()
            for _ in range(8):
                exported = {name: value.clone() for name, value in generation._iter_named_params()}
                torch.testing.assert_close(exported, expected, rtol=0, atol=0)
                torch.testing.assert_close(model.state_dict(), training_state, rtol=0, atol=0)
                torch.testing.assert_close(model(ids).logits.float().log_softmax(-1), logprobs, rtol=0, atol=0)
            if update:
                assert not torch.equal(exported["model.layers.0.self_attn.v_proj.weight"], previous)
            previous = exported["model.layers.0.self_attn.v_proj.weight"]


def test_bf16_export_preserves_merged_adapter_bias(generation):
    peft = pytest.importorskip("peft", minversion="0.21.0")
    model = peft.get_peft_model(
        torch.nn.Sequential(torch.nn.Linear(4, 4)).to(torch.bfloat16),
        peft.LoraConfig(r=2, lora_alpha=4, target_modules=["0"], lora_bias=True),
    )
    with torch.random.fork_rng(), torch.no_grad():
        torch.manual_seed(917)
        for param in model.parameters():
            param.normal_(std=0.2)
    generation.model = model
    expected = {name: value.clone() for name, value in model.state_dict().items()}
    for _ in range(8):
        list(generation._iter_named_params())
        torch.testing.assert_close(model.state_dict(), expected, rtol=0, atol=0)
