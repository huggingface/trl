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

from peft import LoraConfig, get_peft_model
from transformers import Qwen3ForCausalLM

from .._common import check_transformers_version, push_to_hub, smoke_test


check_transformers_version()

BASE = "trl-internal-testing/tiny-Qwen3ForCausalLM"

model = Qwen3ForCausalLM.from_pretrained(BASE, dtype="auto")
model = get_peft_model(model, LoraConfig())
smoke_test(model, None)
push_to_hub(model, None, None, "tiny")
