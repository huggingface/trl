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

import copy
import textwrap
from typing import Any

import torch
from accelerate.logging import get_logger
from accelerate.utils import is_peft_model
from datasets import Dataset, IterableDataset
from torch import nn
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainerCallback,
)
from transformers.utils import is_peft_available

from ...models import prepare_deepspeed, prepare_fsdp
from ...trainer.callbacks import SyncRefModelCallback
from ...trainer.utils import (
    use_adapter,
)
from ..self_distillation.teacher_context import PromptTokenizer, extract_last_user_text
from ..self_distillation.unified_base_self_distillation_trainer import (
    SelfDistillationBatch,
    SelfDistillationRolloutBatch,
    UnifiedBaseSelfDistillationTrainer,
)
from .sdft_config import SDFTConfig


if is_peft_available():
    from peft import PeftConfig
    from peft.peft_model import PeftModel

    from ..self_distillation.peft_adapter_ema_callback import PEFTAdapterEMACallback


logger = get_logger(__name__)


class DemonstrationTeacherContextBuilder:
    """Builds student and teacher contexts from prompts plus privileged context, as in SDFT."""

    def __init__(self, trainer):
        self.trainer = trainer
        self.prompt_tokenizer = PromptTokenizer(trainer)

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
        student_batch = self.prompt_tokenizer.tokenize_prompts(prompts)
        teacher_prompts = [
            self._compose_teacher_prompt(prompt, privileged_context)
            for prompt, privileged_context in zip(prompts, privileged_contexts, strict=True)
        ]
        teacher_batch = self.prompt_tokenizer.tokenize_prompts(teacher_prompts)
        teacher_input_ids = torch.cat([teacher_batch.prompt_ids, completion_ids], dim=1)
        teacher_attention_mask = torch.cat([teacher_batch.prompt_mask, completion_mask], dim=1)
        return {
            "prompt_ids": student_batch.prompt_ids,
            "prompt_mask": student_batch.prompt_mask,
            "teacher_input_ids": teacher_input_ids,
            "teacher_attention_mask": teacher_attention_mask,
        }


class SDFTTrainer(UnifiedBaseSelfDistillationTrainer):
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

        super().init(
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
        self.generate_from_teacher = args.generate_from_teacher
        self.teacher_context_builder = DemonstrationTeacherContextBuilder(self)

        # In self-distillation the teacher is always derived from the student:
        # - PEFT: base model with adapter disabled (or EMA teacher adapter when sync_ref_model=True)
        # - Non-PEFT: same model (or deep-copied EMA model when sync_ref_model=True)
        self.teacher_model = None

        if args.sync_ref_model:
            if is_peft_available() and is_peft_model(self.model):
                self.add_callback(
                    PEFTAdapterEMACallback(
                        model=self.model,
                        teacher_adapter_name="teacher",
                        update_rate=args.ref_model_mixup_alpha,
                        sync_steps=args.ref_model_sync_steps,
                        accelerator=self.accelerator,
                    )
                )
            else:
                student_model = self.accelerator.unwrap_model(self.model)
                self.teacher_model = copy.deepcopy(student_model)
                self.teacher_model.requires_grad_(False)
                self.teacher_model.eval()
                if self.is_deepspeed_enabled:
                    self.teacher_model = prepare_deepspeed(self.teacher_model, self.accelerator)
                elif self.is_fsdp_enabled:
                    self.teacher_model = prepare_fsdp(self.teacher_model, self.accelerator)
                else:
                    self.teacher_model = self.accelerator.prepare_model(self.teacher_model, evaluation_mode=True)
                self.add_callback(SyncRefModelCallback(ref_model=self.teacher_model, accelerator=self.accelerator))

        self.model_accepts_loss_kwargs = False

    def augment_training_batch(
        self,
        inputs: list[dict[str, Any]],
        rollout_batch: SelfDistillationRolloutBatch,
    ) -> SelfDistillationBatch:
        prompts, privileged_contexts = self._split_prompt_and_privileged_context(inputs)
        teacher_batch = self.teacher_context_builder.build(
            prompts,
            privileged_contexts,
            rollout_batch.completion_ids,
            rollout_batch.completion_mask,
        )

        old_per_token_logps = None if self.generate_from_teacher else rollout_batch.old_per_token_logps
        return SelfDistillationBatch(
            prompt_ids=teacher_batch["prompt_ids"],
            prompt_mask=teacher_batch["prompt_mask"],
            completion_ids=rollout_batch.completion_ids,
            completion_mask=rollout_batch.completion_mask,
            teacher_input_ids=teacher_batch["teacher_input_ids"],
            teacher_attention_mask=teacher_batch["teacher_attention_mask"],
            old_per_token_logps=old_per_token_logps,
        )

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

    def _get_teacher_context_for_self_distillation(self, model):
        if is_peft_available() and isinstance(self.model, PeftModel):
            model = self.accelerator.unwrap_model(self.model)
            if self.args.sync_ref_model and "teacher" in model.peft_config:
                return use_adapter(model, adapter_name="teacher")
            return use_adapter(model, adapter_name=None)
        return super()._get_teacher_context_for_self_distillation(model)
