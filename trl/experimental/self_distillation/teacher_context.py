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

import re
from dataclasses import dataclass
from typing import Any

import torch
from accelerate.utils import gather_object

from ...data_utils import maybe_apply_chat_template
from ...trainer.base_trainer import _BaseTrainer
from ...trainer.utils import pad


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


class DemonstrationTeacherContextBuilder:
    """Builds student and teacher contexts from prompts plus privileged context, as in SDFT."""

    def __init__(self, trainer):
        self.trainer = trainer
        self.prompt_tokenizer = PromptTokenizer(trainer)

    def _extract_last_user_text(self, prompt: list[dict[str, Any]]) -> str:
        last_message = prompt[-1]
        content = last_message.get("content", "")
        if isinstance(content, list):
            return " ".join(part.get("text", "") for part in content if part.get("type") == "text")
        return content

    def _stringify_privileged_context(self, privileged_context: Any) -> str:
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
            prompt_text = self._extract_last_user_text(prompt)
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


class SuccessfulRolloutTeacherContextBuilder:
    """Builds SDPO teacher contexts from successful rollouts, following the official online implementation."""

    def __init__(self, trainer):
        self.trainer = trainer
        self.last_metrics: dict[str, float] = {}

    def _extract_last_user_text(self, prompt: list[dict[str, Any]]) -> str:
        last_message = prompt[-1]
        content = last_message.get("content", "")
        if isinstance(content, list):
            return " ".join(part.get("text", "") for part in content if part.get("type") == "text")
        return content

    def _build_reprompt_text(self, prompt_text: str, solution_text: str, feedback_text: str) -> str:
        return self.trainer.args.reprompt_template.format(
            prompt=prompt_text,
            solution=solution_text,
            feedback=feedback_text,
        )

    def _tokenize_teacher_messages(
        self, teacher_messages_list: list[str | list[dict[str, Any]]]
    ) -> TokenizedPromptBatch:
        teacher_prompt_ids_list = []
        device = self.trainer.accelerator.device
        for msg in teacher_messages_list:
            if isinstance(msg, list) and isinstance(msg[0], dict):
                tokenized = self.trainer.processing_class.apply_chat_template(
                    msg,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
                if isinstance(tokenized, torch.Tensor):
                    ids = tokenized.squeeze(0)
                else:
                    ids = tokenized["input_ids"].squeeze(0)
            else:
                ids = self.trainer.processing_class.encode(msg, return_tensors="pt").squeeze(0)

            if ids.shape[0] > self.trainer.args.max_reprompt_len:
                ids = ids[-self.trainer.args.max_reprompt_len :]
            teacher_prompt_ids_list.append(ids)

        teacher_prompt_ids = [ids.to(device) for ids in teacher_prompt_ids_list]
        teacher_prompt_mask = [torch.ones(len(ids), dtype=torch.long, device=device) for ids in teacher_prompt_ids]
        return TokenizedPromptBatch(
            prompt_ids=pad(teacher_prompt_ids, padding_value=self.trainer.pad_token_id, padding_side="left"),
            prompt_mask=pad(teacher_prompt_mask, padding_value=0, padding_side="left"),
        )

    def build(
        self,
        output: dict[str, torch.Tensor | Any],
        prompts: list[Any],
        rewards: torch.Tensor,
        feedbacks: list[Any] | None = None,
    ) -> dict[str, torch.Tensor]:
        device = self.trainer.accelerator.device
        mode = "train" if self.trainer.model.training else "eval"
        num_generations = self.trainer.num_generations if mode == "train" else self.trainer.num_generations_eval
        total_samples = rewards.shape[0]
        completion_ids = output["completion_ids"]
        completion_mask = output["completion_mask"]

        num_local = len(prompts)
        process_start = self.trainer.accelerator.process_index * num_local
        process_slice = slice(process_start, process_start + num_local)

        # Rewards are already globally gathered before this builder runs, but prompts and completions are still local.
        # Gather only the pieces needed to mine successful rollouts across generation groups; the returned teacher
        # tensors remain local to the current process.
        all_completion_ids = self.trainer.accelerator.gather(completion_ids)
        all_prompts = gather_object(prompts)
        all_feedbacks = gather_object(feedbacks) if feedbacks is not None else [None] * total_samples

        threshold = self.trainer.args.success_reward_threshold
        dont_reprompt_self = self.trainer.args.dont_reprompt_on_self_success
        feedback_only_without_solution = self.trainer.args.environment_feedback_only_without_solution
        self_distillation_mask = torch.zeros(total_samples, device=device)
        num_with_solution = 0
        num_with_feedback_available = 0
        num_with_feedback_used = 0
        success_group_count = 0

        for i in range(total_samples):
            group_start = (i // num_generations) * num_generations
            group_end = group_start + num_generations
            original_prompt = all_prompts[i]

            successful = []
            if self.trainer.args.use_successful_as_teacher:
                for j in range(group_start, group_end):
                    if dont_reprompt_self and j == i:
                        continue
                    if rewards[j].item() >= threshold:
                        successful.append(j)

            if i % num_generations == 0 and len(successful) > 0:
                success_group_count += 1

            raw_feedback = all_feedbacks[i]
            has_feedback = isinstance(raw_feedback, str) and raw_feedback.strip() != ""
            if has_feedback:
                num_with_feedback_available += 1

            has_solution = len(successful) > 0
            use_feedback = (
                self.trainer.args.include_environment_feedback
                and has_feedback
                and (not feedback_only_without_solution or not has_solution)
            )
            if use_feedback:
                num_with_feedback_used += 1
            if has_solution or use_feedback:
                self_distillation_mask[i] = 1.0
            if has_solution:
                num_with_solution += 1

        local_teacher_messages = []
        local_self_distillation_mask = self_distillation_mask[process_slice]
        for global_idx in range(process_start, process_start + num_local):
            original_prompt = all_prompts[global_idx]
            raw_feedback = all_feedbacks[global_idx]
            group_start = (global_idx // num_generations) * num_generations
            group_end = group_start + num_generations

            successful = []
            if self.trainer.args.use_successful_as_teacher:
                for j in range(group_start, group_end):
                    if dont_reprompt_self and j == global_idx:
                        continue
                    if rewards[j].item() >= threshold:
                        successful.append(j)

            has_solution = len(successful) > 0
            has_feedback = isinstance(raw_feedback, str) and raw_feedback.strip() != ""
            use_feedback = (
                self.trainer.args.include_environment_feedback
                and has_feedback
                and (not feedback_only_without_solution or not has_solution)
            )

            if not has_solution and not use_feedback:
                local_teacher_messages.append(original_prompt)
                continue

            solution_text = ""
            if has_solution:
                demo_idx = successful[0]
                demo_ids = all_completion_ids[demo_idx]
                demo_ids = demo_ids[demo_ids != self.trainer.processing_class.pad_token_id]
                demo_text = self.trainer.processing_class.decode(demo_ids, skip_special_tokens=True)

                if self.trainer.args.remove_thinking_from_demonstration:
                    demo_text = re.sub(r"<think>.*?</think>", "", demo_text, flags=re.DOTALL).strip()

                solution_text = self.trainer.args.solution_template.format(successful_previous_attempt=demo_text)

            feedback_text = ""
            if use_feedback:
                feedback_text = self.trainer.args.feedback_template.format(feedback_raw=raw_feedback)

            if isinstance(original_prompt, list):
                system_messages = original_prompt[:-1]
                prompt_text = self._extract_last_user_text(original_prompt)
                reprompt_text = self._build_reprompt_text(prompt_text, solution_text, feedback_text)
                local_teacher_messages.append(system_messages + [{"role": "user", "content": reprompt_text}])
            else:
                local_teacher_messages.append(self._build_reprompt_text(original_prompt, solution_text, feedback_text))

        teacher_batch = self._tokenize_teacher_messages(local_teacher_messages)
        teacher_input_ids = torch.cat([teacher_batch.prompt_ids, completion_ids], dim=1)
        teacher_attention_mask = torch.cat([teacher_batch.prompt_mask, completion_mask], dim=1)

        batch_size = total_samples if total_samples > 0 else 1
        num_groups = max(1, total_samples // max(1, num_generations))
        self.last_metrics = {
            "self_distillation/success_group_fraction": success_group_count / num_groups,
            "self_distillation/success_sample_fraction": num_with_solution / batch_size,
            "self_distillation/feedback_available_fraction": num_with_feedback_available / batch_size,
            "self_distillation/feedback_used_fraction": num_with_feedback_used / batch_size,
            "self_distillation/reprompt_sample_fraction": self_distillation_mask.float().mean().item(),
        }

        return {
            "teacher_input_ids": teacher_input_ids,
            "teacher_attention_mask": teacher_attention_mask,
            "self_distillation_mask": local_self_distillation_mask,
        }
