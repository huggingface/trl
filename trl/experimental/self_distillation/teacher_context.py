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

from dataclasses import dataclass
from typing import Any

import torch

from ...data_utils import maybe_apply_chat_template
from ...trainer.base_trainer import _BaseTrainer
from ...trainer.utils import pad


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


@dataclass
class TokenizedPromptBatch:
    prompt_ids: torch.Tensor
    prompt_mask: torch.Tensor


class PromptTokenizer:
    """Internal helper to tokenize prompt-like inputs consistently across self-distillation trainers."""

    def __init__(self, trainer):
        self.trainer = trainer

    def apply_prompt_template(self, prompts: list[Any]) -> list[str]:
        return [
            maybe_apply_chat_template(
                {"prompt": prompt},
                self.trainer.processing_class,
                **getattr(self.trainer, "chat_template_kwargs", {}),
            )["prompt"]
            for prompt in prompts
        ]

    def tokenize_prompts(self, prompts: list[Any]) -> TokenizedPromptBatch:
        prompt_text = self.apply_prompt_template(prompts)
        prompt_inputs = self.trainer.processing_class(
            text=prompt_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            max_length=self.trainer.max_prompt_length,
            truncation=True,
            add_special_tokens=False,
        )
        prompt_inputs = super(_BaseTrainer, self.trainer)._prepare_inputs(prompt_inputs)
        prompt_ids = [
            p[m].tolist()
            for p, m in zip(prompt_inputs["input_ids"], prompt_inputs["attention_mask"].bool(), strict=False)
        ]
        prompt_ids = [torch.tensor(ids, device=self.trainer.accelerator.device) for ids in prompt_ids]
        prompt_mask = [torch.ones_like(ids, dtype=torch.long) for ids in prompt_ids]
        return TokenizedPromptBatch(
            prompt_ids=pad(prompt_ids, padding_value=self.trainer.pad_token_id, padding_side="left"),
            prompt_mask=pad(prompt_mask, padding_value=0, padding_side="left"),
        )
