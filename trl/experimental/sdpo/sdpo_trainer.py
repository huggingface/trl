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

import copy
import re
import textwrap
from typing import Any

import torch
from accelerate.utils import gather_object
from datasets import Dataset, IterableDataset
from torch import nn
from transformers import PreTrainedModel, PreTrainedTokenizerBase, ProcessorMixin, TrainerCallback

from ...trainer.callbacks import SyncRefModelCallback
from ...trainer.utils import pad
from ..self_distillation.base_self_distillation_trainer import BaseSelfDistillationTrainer
from ..self_distillation.teacher_context import TokenizedPromptBatch, extract_last_user_text
from .sdpo_config import SDPOConfig


class EMATeacherSyncCallback(SyncRefModelCallback):
    """Synchronize an EMA teacher model with the student model on each step."""

    def __init__(self, teacher_model, update_rate: float, accelerator=None):
        super().__init__(ref_model=teacher_model, accelerator=accelerator)
        self.update_rate = update_rate

    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs["model"]
        if self.accelerator is not None:
            model = self.accelerator.unwrap_model(model)
        self.sync_target_model(model, self.ref_model, self.update_rate)


class SuccessfulRolloutTeacherContextBuilder:
    """Builds SDPO teacher contexts from successful rollouts, following the official online implementation."""

    def __init__(self, trainer):
        self.trainer = trainer
        self.last_metrics: dict[str, float] = {}

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
        chat_template_kwargs = getattr(self.trainer, "chat_template_kwargs", {})
        for msg in teacher_messages_list:
            if isinstance(msg, list) and isinstance(msg[0], dict):
                tokenized = self.trainer.processing_class.apply_chat_template(
                    msg,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    **chat_template_kwargs,
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
        completion_ids = output["completion_ids"]
        completion_mask = output["completion_mask"]

        num_local = len(prompts)
        process_start = self.trainer.accelerator.process_index * num_local
        process_slice = slice(process_start, process_start + num_local)

        # Rewards arrive already locally sliced (per-process) from the rollout mixin; re-gather them so
        # the mining loop can find successful rollouts across all processes within each generation group.
        all_rewards = self.trainer.accelerator.gather(rewards)
        # Completion tensors are padded to the local max length per rank; align shapes before gathering.
        # Use separate variables so the original completion_ids/completion_mask stay unpadded for the
        # teacher concat (they must match the student's sequence length for logits_to_keep alignment).
        padded_completion_ids = self.trainer.accelerator.pad_across_processes(
            completion_ids, dim=1, pad_index=self.trainer.pad_token_id
        )
        all_completion_ids = self.trainer.accelerator.gather(padded_completion_ids)
        all_prompts = gather_object(prompts)
        total_samples = all_rewards.shape[0]
        all_feedbacks = gather_object(feedbacks) if feedbacks is not None else [None] * total_samples

        threshold = self.trainer.args.success_reward_threshold
        dont_reprompt_self = self.trainer.args.dont_reprompt_on_self_success
        feedback_only_without_solution = self.trainer.args.environment_feedback_only_without_solution
        self_distillation_mask = torch.zeros(total_samples, device=device)
        num_with_solution = 0
        num_with_feedback_available = 0
        num_with_feedback_used = 0
        success_group_count = 0
        successful_demo_indices: list[int | None] = [None] * total_samples
        use_feedback_flags: list[bool] = [False] * total_samples
        has_solution_flags: list[bool] = [False] * total_samples

        for i in range(total_samples):
            group_start = (i // num_generations) * num_generations
            group_end = group_start + num_generations

            successful = []
            if self.trainer.args.use_successful_as_teacher:
                for j in range(group_start, group_end):
                    if dont_reprompt_self and j == i:
                        continue
                    if all_rewards[j].item() >= threshold:
                        successful.append(j)

            if i % num_generations == 0:
                # Count groups with any successful rollout, ignoring self-exclusion which only
                # affects per-sample teacher assignment, not whether the group has successes.
                group_has_success = any(all_rewards[j].item() >= threshold for j in range(group_start, group_end))
                if group_has_success:
                    success_group_count += 1

            raw_feedback = all_feedbacks[i]
            has_feedback = isinstance(raw_feedback, str) and raw_feedback.strip() != ""
            if has_feedback:
                num_with_feedback_available += 1

            has_solution = len(successful) > 0
            has_solution_flags[i] = has_solution
            if has_solution:
                successful_demo_indices[i] = successful[0]
            use_feedback = (
                self.trainer.args.include_environment_feedback
                and has_feedback
                and (not feedback_only_without_solution or not has_solution)
            )
            use_feedback_flags[i] = use_feedback
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
            has_solution = has_solution_flags[global_idx]
            use_feedback = use_feedback_flags[global_idx]

            if not has_solution and not use_feedback:
                local_teacher_messages.append(original_prompt)
                continue

            solution_text = ""
            if has_solution:
                demo_idx = successful_demo_indices[global_idx]
                if demo_idx is None:
                    raise RuntimeError("Expected a successful demonstration index for an active SDPO teacher prompt.")
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
                prompt_text = extract_last_user_text(original_prompt)
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


class SDPOTrainer(BaseSelfDistillationTrainer):
    """
    Trainer for Self-Distillation Policy Optimization (SDPO).

    SDPO augments on-policy optimization with self-distillation from the model's own high-reward trajectories. It
    converts tokenized feedback into a dense learning signal without any external teacher or explicit reward model.
    SDPO treats the current model conditioned on feedback as a self-teacher and distills its feedback-informed
    next-token predictions back into the policy.
    """

    config_cls = SDPOConfig
    _tag_names = ["trl", "sdpo"]
    _name = "SDPO"
    # docstyle-ignore
    _paper = {
        "title": "Reinforcement Learning via Self-Distillation",
        "id": "2601.20802",
        "citation": textwrap.dedent("""\
            @article{hubotter2026sdpo,
                title        = {{Reinforcement Learning via Self-Distillation}},
                author       = {Jonas H\\"ubotter and Frederike L\\"ubeck and Lejs Behric and Anton Baumann and Marco Bagatella and Daniel Marta and Ido Hakimi and Idan Shenfeld and Thomas Kleine Buening and Carlos Guestrin and Andreas Krause},
                year         = 2026,
                eprint       = {arXiv:2601.20802}
            }"""),
    }

    def __init__(
        self,
        model: str | PreTrainedModel | nn.Module,
        reward_funcs: Any | list[Any] | None = None,
        args: SDPOConfig | None = None,
        train_dataset: Dataset | IterableDataset | None = None,
        eval_dataset: Dataset | IterableDataset | dict[str, Dataset | IterableDataset] | None = None,
        processing_class: PreTrainedTokenizerBase | ProcessorMixin | None = None,
        reward_processing_classes: PreTrainedTokenizerBase | list[PreTrainedTokenizerBase] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None] = (None, None),
        peft_config=None,
    ):
        if reward_funcs is None or (isinstance(reward_funcs, list) and len(reward_funcs) == 0):
            raise ValueError("`reward_funcs` is required for SDPOTrainer because SDPO must score rollouts.")
        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
        )
        self.teacher_context_builder = SuccessfulRolloutTeacherContextBuilder(self)
        if self.args.teacher_regularization == "ema":
            # `self.model` may already be accelerator-wrapped after the shared base constructor. Build the EMA
            # teacher from the unwrapped student model first, then prepare it as an auxiliary eval-only module.
            student_model = self.accelerator.unwrap_model(self.model)
            self.teacher_model = copy.deepcopy(student_model)
            self.teacher_model.requires_grad_(False)
            self.teacher_model.eval()
            self.teacher_model = self._prepare_auxiliary_model_for_eval(self.teacher_model)
            self.add_callback(
                EMATeacherSyncCallback(
                    teacher_model=self.teacher_model,
                    update_rate=self.args.teacher_update_rate,
                    accelerator=self.accelerator,
                )
            )

    def _allow_topk_without_full_logit_distillation(self) -> bool:
        return False

    def _generate_and_score_completions(
        self, inputs: list[dict[str, torch.Tensor | Any]]
    ) -> dict[str, torch.Tensor | Any]:
        prompts, privileged_contexts = self._split_prompt_and_privileged_context(inputs)

        output = super()._generate_and_score_completions(inputs)
        output.update(
            self.teacher_context_builder.build(output, prompts, output["rewards"], feedbacks=privileged_contexts)
        )

        mode = "train" if self.model.training else "eval"
        for key, value in self.teacher_context_builder.last_metrics.items():
            self._metrics[mode][key].append(value)
        self._warn_on_inactive_self_distillation(mode)

        self._dispatch_self_distillation_callback(
            "on_teacher_context_built",
            teacher_input_ids=output["teacher_input_ids"],
            teacher_attention_mask=output["teacher_attention_mask"],
            completion_mask=output["completion_mask"],
            self_distillation_mask=output["self_distillation_mask"],
        )

        return output

    def _warn_on_inactive_self_distillation(self, mode: str) -> None:
        metrics = self.teacher_context_builder.last_metrics
        tolerance = self.args.diagnostics_flat_tolerance

        reprompt_fraction = metrics.get("self_distillation/reprompt_sample_fraction", 0.0)
        success_fraction = metrics.get("self_distillation/success_group_fraction", 0.0)

        if reprompt_fraction <= tolerance:
            self._warn_on_degenerate_diagnostics(
                mode=mode,
                counter_key="inactive_self_distillation",
                message=(
                    "SDPO self-distillation is inactive because no reprompted samples were constructed. "
                    "This usually means no rollout exceeded `success_reward_threshold` and no usable privileged "
                    "feedback was available."
                ),
            )
        else:
            self._diagnostic_counters[mode]["inactive_self_distillation"] = 0

        if success_fraction <= tolerance:
            self._warn_on_degenerate_diagnostics(
                mode=mode,
                counter_key="no_successful_rollouts",
                message=(
                    "SDPO did not find any successful rollouts in the current generation groups. "
                    "If this persists, reduce task difficulty, adjust reward shaping, or lower "
                    "`success_reward_threshold`."
                ),
            )
        else:
            self._diagnostic_counters[mode]["no_successful_rollouts"] = 0

    def _compute_loss(
        self,
        model,
        inputs,
    ) -> torch.Tensor:
        accumulation_scale = self.current_gradient_accumulation_steps if self.model.training else 1.0

        if self.args.sdpo_policy_loss_mode == "hybrid":
            base_policy_loss = super()._compute_loss(model, inputs)
            if self.args.distillation_weight <= 0.0:
                return base_policy_loss

            sdpo_loss = self._compute_self_distillation_loss(model, inputs) / accumulation_scale
            return base_policy_loss + self.args.distillation_weight * sdpo_loss

        if self.args.distillation_weight <= 0.0:
            return super()._compute_loss(model, inputs)

        sdpo_loss = self._compute_self_distillation_loss(model, inputs) / accumulation_scale
        return self.args.distillation_weight * sdpo_loss
