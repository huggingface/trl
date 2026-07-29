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

# Unlike its siblings, this script does not build a model: it re-publishes an existing tiny model with a
# `response_template` baked into `tokenizer_config.json`, as if the checkpoint shipped one natively.
#
# Most models on the Hub do not (yet) ship a response template, so `add_response_schema` has to supply one
# from its table of known chat templates. This fixture covers the other case — a model that already carries
# its own — which callers must detect and leave alone rather than trying to add a second one.

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from trl import add_response_schema

from .._common import check_transformers_version, push_to_hub, smoke_test


# The new-style `response_template` attribute (as opposed to the legacy `response_schema`) is only set, and
# only serialized to `tokenizer_config.json`, from this release on.
check_transformers_version("5.13.0")

MODEL_ID = "trl-internal-testing/tiny-Qwen3MoeForCausalLM"

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
generation_config = GenerationConfig.from_pretrained(MODEL_ID)

# Take the response template `add_response_schema` would have supplied, so the fixture parses exactly like
# its source model.
tokenizer = add_response_schema(tokenizer)

# Then make the chat template unrecognizable to `add_response_schema`, which matches templates by exact
# string equality. Without this the fixture would prove nothing: `add_response_schema` would still find a
# match and quietly re-set the same template. The marker is a Jinja comment, so rendering is byte-for-byte unchanged.
tokenizer.chat_template = "{# trl-internal-testing fixture: ships its own response template #}" + (
    tokenizer.chat_template
)

smoke_test(model, tokenizer)
push_to_hub(model, tokenizer, generation_config, "tiny", suffix="ResponseTemplate")
