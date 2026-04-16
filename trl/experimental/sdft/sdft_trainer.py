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

import textwrap
from typing import Any

import torch
from accelerate.logging import get_logger
from datasets import Dataset, IterableDataset
from torch import nn
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainerCallback,
)
from transformers.utils import is_peft_available

from ..self_distillation.base_self_distillation_trainer import (
    BaseSelfDistillationTrainer,
    RolloutBatch,
    TrainingBatch,
)
from ..self_distillation.teacher_context import _split_prompt_and_privileged_context, extract_last_user_text
from .sdft_config import SDFTConfig


if is_peft_available():
    from peft import PeftConfig


logger = get_logger(__name__)


class DemonstrationTeacherContextBuilder:
    """Builds student and teacher contexts from prompts plus privileged context, as in SDFT."""

    def __init__(self, trainer):
        self.trainer = trainer

    def _stringify_privileged_context(self, privileged_context: Any) -> str:
        if privileged_context is None:
            raise ValueError(
                "`privileged_context` must not be None for self-distillation teacher prompt construction."
            )
        if isinstance(privileged_context, str):
            return privileged_context
        if isinstance(privileged_context, list) and privileged_context and isinstance(privileged_context[0], dict):
            chunks = []
            for message in privileged_context:
                content = message.get("content", "")
                if isinstance(content, list):
                    text = " ".join(part.get("text", "") for part in content if part.get("type") == "text")
                else:
                    text = str(content)
                if text:
                    chunks.append(text)
            return "\n".join(chunks)
        return str(privileged_context)

    def _compose_teacher_prompt(self, prompt: Any, privileged_context: Any) -> Any:
        privileged_text = self._stringify_privileged_context(privileged_context)
        if isinstance(prompt, list):
            system_messages = prompt[:-1]
            prompt_text = extract_last_user_text(prompt)
            teacher_text = self.trainer.args.teacher_prompt_template.format(
                prompt=prompt_text,
                privileged_context=privileged_text,
            )
            return system_messages + [{"role": "user", "content": teacher_text}]
        return self.trainer.args.teacher_prompt_template.format(prompt=prompt, privileged_context=privileged_text)

    def select_generation_prompts(self, prompts: list[Any], privileged_contexts: list[Any]) -> list[Any]:
        if not self.trainer.generate_from_teacher:
            return prompts
        return [
            self._compose_teacher_prompt(prompt, privileged_context)
            for prompt, privileged_context in zip(prompts, privileged_contexts, strict=True)
        ]

    def build(
        self,
        prompts: list[Any],
        privileged_contexts: list[Any],
        completion_ids: torch.Tensor,
        completion_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        teacher_prompts = [
            self._compose_teacher_prompt(prompt, privileged_context)
            for prompt, privileged_context in zip(prompts, privileged_contexts, strict=True)
        ]
        teacher_batch = self.trainer._tokenize_prompts(teacher_prompts)
        teacher_input_ids = torch.cat([teacher_batch["prompt_ids"], completion_ids], dim=1)
        teacher_attention_mask = torch.cat([teacher_batch["prompt_mask"], completion_mask], dim=1)
        return {
            "teacher_input_ids": teacher_input_ids,
            "teacher_attention_mask": teacher_attention_mask,
        }


class SDFTTrainer(BaseSelfDistillationTrainer):
    """Trainer for SDFT-style on-policy self-distillation with explicit teacher prompts."""

    _tag_names = ["trl", "sdft"]
    _name = "SDFT"
    config_cls = SDFTConfig
    # docstyle-ignore
    _paper = {
        "title": "Self-Training with On-Policy Self-Distillation for Language Model Alignment",
        "id": "2601.19897",
        "citation": textwrap.dedent("""\
            @article{hubotter2026selftraining,
                title        = {{Self-Training with On-Policy Self-Distillation for Language Model Alignment}},
                author       = {Jonas H\\"ubotter and Frederike L\\"ubeck and Lejs Behric and Anton Baumann and Marco Bagatella and Daniel Marta and Ido Hakimi and Idan Shenfeld and Thomas Kleine Buening and Carlos Guestrin and Andreas Krause},
                year         = 2026,
                eprint       = {arXiv:2601.19897}
            }"""),
    }

    def __init__(
        self,
        model: str | PreTrainedModel | nn.Module,
        args: SDFTConfig | None = None,
        train_dataset: Dataset | IterableDataset | None = None,
        eval_dataset: Dataset | IterableDataset | dict[str, Dataset | IterableDataset] | None = None,
        processing_class: PreTrainedTokenizerBase | ProcessorMixin | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None] = (None, None),
        peft_config: PeftConfig | None = None,
    ):
        if isinstance(train_dataset, IterableDataset):
            raise NotImplementedError("Iterable datasets are not yet supported in SDFTTrainer.")
        if isinstance(eval_dataset, IterableDataset) or (
            isinstance(eval_dataset, dict) and any(isinstance(ds, IterableDataset) for ds in eval_dataset.values())
        ):
            raise NotImplementedError("Iterable eval datasets are not yet supported in SDFTTrainer.")

        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
        )

        self.num_loss_tokens_to_skip = args.num_loss_tokens_to_skip
        self.teacher_context_builder = DemonstrationTeacherContextBuilder(self)

    def finalize_batch(
        self,
        inputs: list[dict[str, Any]],
        rollout_batch: RolloutBatch,
    ) -> TrainingBatch:
        prompts, privileged_contexts = _split_prompt_and_privileged_context(inputs)
        teacher_batch = self.teacher_context_builder.build(
            prompts,
            privileged_contexts,
            rollout_batch.completion_ids,
            rollout_batch.completion_mask,
        )

        batch = super().finalize_batch(inputs, rollout_batch)
        batch.update(
            {
                "teacher_input_ids": teacher_batch["teacher_input_ids"],
                "teacher_attention_mask": teacher_batch["teacher_attention_mask"],
            }
        )
        return batch

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The SDFTTrainer does not support returning outputs")

        if self.num_loss_tokens_to_skip > 0:
            inputs = dict(inputs)
            completion_mask = inputs["completion_mask"].clone()
            token_positions = torch.arange(completion_mask.size(1), device=completion_mask.device).unsqueeze(0)
            completion_mask = completion_mask * (token_positions >= self.num_loss_tokens_to_skip).long()
            inputs["completion_mask"] = completion_mask

        loss = self._compute_self_distillation_loss(model, inputs)
        accumulation_scale = self.current_gradient_accumulation_steps if self.model.training else 1.0
        return loss / accumulation_scale
