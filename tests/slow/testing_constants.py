# Copyright 2024 The HuggingFace Team. All rights reserved.
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

# TODO: push them under trl-org
MODELS_TO_TEST = [
    "trl-internal-testing/tiny-random-LlamaForCausalLM",
    "HuggingFaceM4/tiny-random-MistralForCausalLM",
]

# We could have also not declared these variables but let's be verbose
PACKING_OPTIONS = [True, False]
GRADIENT_CHECKPOINTING_KWARGS = [None, {"use_reentrant": False}, {"use_reentrant": True}]
DEVICE_MAP_OPTIONS = [{"": 0}, "auto"]

DPO_LOSS_TYPES = ["sigmoid", "ipo"]
DPO_PRECOMPUTE_LOGITS = [True, False]
