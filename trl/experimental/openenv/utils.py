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

from typing import Any


def generate_rollout_completions(
    trainer,
    prompts: list,
    *,
    generation_overrides: dict[str, Any] | None = None,
    as_chat: bool | None = None,
) -> list[dict[str, Any]]:
    """
    Generate completions for custom rollouts via the trainer generation backend.

    Returns one result per prompt, containing prompt and completion token ids along with per-token log probabilities
    and the generated text.
    """

    if not prompts:
        return []

    results = trainer.generation_backend.generate_rollout_completions(
        prompts=prompts,
        processing_class=trainer.processing_class,
        generation_overrides=generation_overrides,
        as_chat=as_chat,
    )

    return [
        {
            "prompt_ids": result.prompt_ids,
            "completion_ids": result.completion_ids,
            "logprobs": result.logprobs,
            "text": result.text,
        }
        for result in results
    ]
