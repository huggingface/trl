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

# Builds a tiny model loaded via `trust_remote_code=True`. The architecture is
# a thin `LlamaForCausalLM` subclass under a custom `model_type="remote"`, so
# the resulting Hub repo can only be loaded through the remote-code path.

import torch
from transformers import AutoTokenizer, GenerationConfig

from .._common import check_transformers_version, init_weights_tiny_model, push_to_hub, smoke_test
from ._remote_code.configuration_remote import RemoteConfig
from ._remote_code.modeling_remote import RemoteForCausalLM, RemoteForSequenceClassification, RemoteModel


check_transformers_version()

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
generation_config = GenerationConfig.from_pretrained(MODEL_ID)

RemoteConfig.register_for_auto_class()
RemoteModel.register_for_auto_class("AutoModel")
RemoteForCausalLM.register_for_auto_class("AutoModelForCausalLM")
RemoteForSequenceClassification.register_for_auto_class("AutoModelForSequenceClassification")

config = RemoteConfig(
    vocab_size=len(tokenizer.vocab),
    hidden_size=8,
    num_attention_heads=4,
    num_key_value_heads=2,
    num_hidden_layers=2,
    intermediate_size=32,
)
# `model.save_pretrained` only records auto-map entries for the saved class's own auto-classes.
# Seed the AutoModel and SeqCls entries on the config so the Hub repo can also be loaded as
# `AutoModel` (used by `AutoModel.from_config` inside `LlamaForSequenceClassification.__init__`)
# and `AutoModelForSequenceClassification` (used by e.g. RewardTrainer).
config.auto_map = {
    "AutoModel": "modeling_remote.RemoteModel",
    "AutoModelForSequenceClassification": "modeling_remote.RemoteForSequenceClassification",
}
model = RemoteForCausalLM(config).to(dtype=torch.bfloat16)
init_weights_tiny_model(model)
smoke_test(model, tokenizer)
push_to_hub(model, tokenizer, generation_config, "tiny")
