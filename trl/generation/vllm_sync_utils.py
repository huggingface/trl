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

"""Utilities shared by vLLM weight synchronization call sites."""

QWEN3_HF_TO_VLLM_PREFIX_MAPS = {
    "Qwen3_5ForCausalLM": (
        ("lm_head.", "language_model.lm_head."),
        ("model.", "language_model.model."),
    ),
    "Qwen3_5ForConditionalGeneration": (
        ("model.visual.", "visual."),
        ("lm_head.", "language_model.lm_head."),
        ("model.language_model.", "language_model.model."),
    ),
    "Qwen3VLForConditionalGeneration": (
        ("model.visual.", "visual."),
        ("lm_head.", "language_model.lm_head."),
        ("model.language_model.", "language_model.model."),
    ),
}


def fix_param_name_to_vllm(
    name: str, architectures: list[str] | tuple[str, ...] | None, extra_prefixes: list[str] | None = None
) -> str:
    """Normalize Hugging Face parameter names to the vLLM runtime namespace."""
    extra_prefixes = extra_prefixes or []
    prefixes = ["_checkpoint_wrapped_module."] + extra_prefixes
    for prefix in prefixes:
        name = name.replace(prefix, "")

    for architecture in architectures or []:
        prefix_map = QWEN3_HF_TO_VLLM_PREFIX_MAPS.get(architecture)
        if prefix_map is None:
            continue

        for hf_prefix, vllm_prefix in prefix_map:
            if name.startswith(hf_prefix):
                return name.replace(hf_prefix, vllm_prefix, 1)
        break

    return name
