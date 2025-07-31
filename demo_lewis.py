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

from transformers import AutoModelForCausalLM, AutoTokenizer

from trl.models.utils import clone_chat_template


# This tokenizer doesn't have a chat_template by default
tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-BloomForCausalLM")
model = AutoModelForCausalLM.from_pretrained("trl-internal-testing/tiny-BloomForCausalLM")
# This one has a chat_template by default
source = "trl-internal-testing/tiny-Qwen3ForCausalLM"
modified_model, modified_tokenizer, _ = clone_chat_template(model, tokenizer, source, resize_to_multiple_of=123)
modified_model, modified_tokenizer, _ = clone_chat_template(
    modified_model, modified_tokenizer, source, resize_to_multiple_of=124
)
