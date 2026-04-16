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

from __future__ import annotations

from typing import Any


def _split_prompt_and_privileged_context(inputs: list[dict[str, Any]]) -> tuple[list[Any], list[Any]]:
    prompts = [example["prompt"] for example in inputs]
    privileged_contexts = [example.get("privileged_context") for example in inputs]
    return prompts, privileged_contexts


def extract_last_user_text(prompt: list[dict[str, Any]]) -> str:
    """Extract the text content from the last user message in a conversational prompt."""
    last_message = prompt[-1]
    if last_message.get("role") != "user":
        raise ValueError(
            f"Self-distillation teacher prompt construction expects the conversation to end with a user turn, "
            f"but the last message has role '{last_message.get('role')}'. "
            f"Prompts ending with assistant prefills or tool turns are not supported."
        )
    content = last_message.get("content", "")
    if isinstance(content, list):
        return " ".join(part.get("text", "") for part in content if part.get("type") == "text")
    return content
