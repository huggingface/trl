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

import re
import textwrap
from typing import Any

import torch
from accelerate.utils import gather_object
from datasets import Dataset, IterableDataset
from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainerCallback,
)
from transformers.utils import logging

from ...data_utils import apply_chat_template, is_conversational
from ...models import prepare_deepspeed, prepare_fsdp
from ...trainer.base_trainer import _BaseTrainer
from ...trainer.utils import get_config_model_id, pad
from ..self_distillation.base_self_distillation_trainer import (
    BaseSelfDistillationTrainer,
    SelfDistillationBatch,
    SelfDistillationRolloutBatch,
)
from ..self_distillation.teacher_context import TokenizedPromptBatch, extract_last_user_text
from .sdpo_config import SDPOConfig


logger = logging.get_logger(__name__)


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
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
        )
        self.importance_sampling_level = args.importance_sampling_level
        self.scale_rewards = args.scale_rewards
        self.epsilon_low = args.epsilon
        self.epsilon_high = args.epsilon_high
        self.beta = args.beta

        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        self.reward_func_names = []
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_model_init_kwargs = args.model_init_kwargs or {}
                if args.distributed_state.distributed_type in ["MULTI_GPU", "DEEPSPEED"]:
                    reward_model_init_kwargs["device_map"] = None
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func,
                    num_labels=1,
                    **reward_model_init_kwargs,
                )
            if isinstance(reward_funcs[i], nn.Module):
                self.reward_func_names.append(get_config_model_id(reward_funcs[i].config).split("/")[-1])
            else:
                self.reward_func_names.append(reward_funcs[i].__name__)
        self.reward_funcs = reward_funcs

        if args.reward_weights is not None:
            if len(args.reward_weights) != len(self.reward_funcs):
                raise ValueError("Number of reward weights must match number of reward functions")
            self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32)
        else:
            self.reward_weights = torch.ones(len(self.reward_funcs), dtype=torch.float32)

        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(self.reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        if len(reward_processing_classes) != len(self.reward_funcs):
            raise ValueError("Number of reward processing classes must match number of reward functions")

        for i, (reward_processing_class, reward_func) in enumerate(
            zip(reward_processing_classes, self.reward_funcs, strict=True)
        ):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(get_config_model_id(reward_func.config))
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, nn.Module):
                if self.is_deepspeed_enabled:
                    self.reward_funcs[i] = prepare_deepspeed(reward_func, self.accelerator)
                elif self.is_fsdp_enabled:
                    self.reward_funcs[i] = prepare_fsdp(reward_func, self.accelerator)
                else:
                    self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

        self.teacher_context_builder = SuccessfulRolloutTeacherContextBuilder(self)

    def _allow_topk_without_full_logit_distillation(self) -> bool:
        return False

    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        device = self.accelerator.device
        if len(self.reward_funcs) == 0:
            return torch.zeros((len(prompts), 0), device=device)

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        keys = [key for key in inputs[0] if key not in ["prompt", "completion", "completion_ids"]]
        reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
        reward_kwargs["trainer_state"] = self.state

        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes, strict=True)
        ):
            if isinstance(reward_func, nn.Module):
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions, strict=True)]
                    texts = [
                        apply_chat_template(x, reward_processing_class, **self.chat_template_kwargs)["text"]
                        for x in messages
                    ]
                else:
                    texts = [p + c for p, c in zip(prompts, completions, strict=True)]
                reward_inputs = reward_processing_class(
                    text=texts,
                    return_tensors="pt",
                    padding=True,
                    padding_side="right",
                    add_special_tokens=False,
                )
                reward_inputs = _BaseTrainer._prepare_inputs(self, reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]
            else:
                output_reward_func = reward_func(
                    prompts=prompts,
                    completions=completions,
                    completion_ids=completion_ids_list,
                    **reward_kwargs,
                )
                output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        return self.accelerator.gather(rewards_per_func)

    def augment_training_batch(
        self,
        inputs: list[dict[str, Any]],
        rollout_batch: SelfDistillationRolloutBatch,
    ) -> SelfDistillationBatch:
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"
        prompts, privileged_contexts = self._split_prompt_and_privileged_context(inputs)
        raw_completion_lengths = rollout_batch.metadata["raw_completion_lengths"].detach().cpu().tolist()
        completion_ids_list = [
            ids[:length].tolist()
            for ids, length in zip(rollout_batch.completion_ids.detach().cpu(), raw_completion_lengths, strict=True)
        ]
        if is_conversational({"prompt": prompts[0]}):
            completions_text = self.processing_class.batch_decode(
                rollout_batch.completion_ids, skip_special_tokens=True
            )
            completions = [[{"role": "assistant", "content": content}] for content in completions_text]
        else:
            completions = self.processing_class.batch_decode(rollout_batch.completion_ids, skip_special_tokens=True)

        rewards_per_func = self._calculate_rewards(inputs, prompts, completions, completion_ids_list)
        if rewards_per_func.numel() == 0:
            rewards = torch.zeros(self.accelerator.num_processes * len(prompts), device=device)
        else:
            rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

        num_generations = self.num_generations if mode == "train" else self.num_generations_eval
        mean_grouped_rewards = rewards.view(-1, num_generations).mean(dim=1).repeat_interleave(num_generations, dim=0)
        if self.scale_rewards == "batch":
            std_rewards = rewards.std().expand_as(rewards) if rewards.numel() > 1 else torch.zeros_like(rewards)
            group_std_rewards = rewards.view(-1, num_generations).std(dim=1)
        elif self.scale_rewards == "none":
            std_rewards = torch.ones_like(rewards)
            group_std_rewards = torch.ones(rewards.numel() // num_generations, device=device, dtype=rewards.dtype)
        else:
            group_std_rewards = rewards.view(-1, num_generations).std(dim=1)
            std_rewards = group_std_rewards.repeat_interleave(num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_rewards + 1e-4)
        self._record_reward_diagnostics(mode, rewards, rewards_per_func, group_std_rewards)

        local_batch_size = rollout_batch.completion_ids.size(0)
        process_start = self.accelerator.process_index * local_batch_size
        process_slice = slice(process_start, process_start + local_batch_size)
        local_rewards = rewards[process_slice]
        local_advantages = advantages[process_slice]

        agg_completion_lengths = self.accelerator.gather(
            torch.tensor([len(ids) for ids in completion_ids_list], device=device)
        )
        self._metrics[mode]["completions/mean_length"].append(agg_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_lengths.float().max().item())

        eos_and_pad = [self.eos_token_id, self.pad_token_id]
        is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids_list], device=device)
        agg_is_truncated = self.accelerator.gather(is_truncated)
        self._metrics[mode]["completions/clipped_ratio"].append(agg_is_truncated.float().mean().item())
        term_completion_lengths = agg_completion_lengths[~agg_is_truncated]
        if len(term_completion_lengths) == 0:
            term_completion_lengths = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(term_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_terminated_length"].append(term_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_terminated_length"].append(term_completion_lengths.float().max().item())

        rollout_dict = rollout_batch.to_dict()
        rollout_dict["rewards"] = local_rewards
        rollout_dict["advantages"] = local_advantages
        rollout_dict["num_items_in_batch"] = rollout_batch.completion_mask.sum().detach()
        teacher_context = self.teacher_context_builder.build(
            rollout_dict,
            prompts,
            rollout_dict["rewards"],
            feedbacks=privileged_contexts,
        )

        mode = "train" if self.model.training else "eval"
        for key, value in self.teacher_context_builder.last_metrics.items():
            self._metrics[mode][key].append(value)
        self._warn_on_inactive_self_distillation(mode)

        self._dispatch_self_distillation_callback(
            "on_teacher_context_built",
            teacher_input_ids=teacher_context["teacher_input_ids"],
            teacher_attention_mask=teacher_context["teacher_attention_mask"],
            completion_mask=rollout_batch.completion_mask,
            self_distillation_mask=teacher_context["self_distillation_mask"],
        )

        return SelfDistillationBatch(
            prompt_ids=rollout_batch.prompt_ids,
            prompt_mask=rollout_batch.prompt_mask,
            completion_ids=rollout_batch.completion_ids,
            completion_mask=rollout_batch.completion_mask,
            teacher_input_ids=teacher_context["teacher_input_ids"],
            teacher_attention_mask=teacher_context["teacher_attention_mask"],
            old_per_token_logps=rollout_batch.old_per_token_logps,
            self_distillation_mask=teacher_context["self_distillation_mask"],
            metadata={
                "rewards": local_rewards,
                "advantages": local_advantages,
            },
        )

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

    def _record_reward_diagnostics(
        self,
        mode: str,
        rewards: torch.Tensor,
        rewards_per_func: torch.Tensor,
        group_std_rewards: torch.Tensor,
    ) -> None:
        tolerance = self.args.diagnostics_flat_tolerance

        reward_mean = rewards.mean() if rewards.numel() > 0 else torch.tensor(0.0, device=self.accelerator.device)
        reward_std = rewards.std() if rewards.numel() > 1 else torch.tensor(0.0, device=self.accelerator.device)
        reward_min = rewards.min() if rewards.numel() > 0 else torch.tensor(0.0, device=self.accelerator.device)
        reward_max = rewards.max() if rewards.numel() > 0 else torch.tensor(0.0, device=self.accelerator.device)
        flat_group_fraction = (
            (group_std_rewards <= tolerance).float().mean()
            if group_std_rewards.numel() > 0
            else torch.tensor(1.0, device=self.accelerator.device)
        )

        self._metrics[mode]["self_distillation/reward_mean"].append(self.accelerator.gather(reward_mean).mean().item())
        self._metrics[mode]["self_distillation/reward_std"].append(self.accelerator.gather(reward_std).mean().item())
        self._metrics[mode]["self_distillation/reward_min"].append(self.accelerator.gather(reward_min).min().item())
        self._metrics[mode]["self_distillation/reward_max"].append(self.accelerator.gather(reward_max).max().item())
        self._metrics[mode]["self_distillation/group_reward_std_mean"].append(
            self.accelerator.gather(group_std_rewards.mean() if group_std_rewards.numel() > 0 else reward_std)
            .mean()
            .item()
        )
        self._metrics[mode]["self_distillation/flat_group_fraction"].append(
            self.accelerator.gather(flat_group_fraction).mean().item()
        )

        if rewards_per_func.numel() > 0:
            reward_func_means = rewards_per_func.nanmean(dim=0)
            gathered_means = self.accelerator.gather(reward_func_means).view(-1, reward_func_means.numel()).mean(dim=0)
            for reward_name, reward_func_mean in zip(self.reward_func_names, gathered_means.tolist(), strict=True):
                self._metrics[mode][f"self_distillation/rewards/{reward_name}"].append(reward_func_mean)

        reward_is_flat = reward_std.item() <= tolerance
        grouped_rewards_are_flat = flat_group_fraction.item() >= 1.0 - tolerance
        if reward_is_flat and grouped_rewards_are_flat:
            self._warn_on_degenerate_diagnostics(
                mode=mode,
                counter_key="flat_rewards",
                message=(
                    "Observed flat SDPO rewards across all sampled generations. "
                    "Policy advantages will collapse to zero, and SDPO will not learn. "
                    "Check reward density, reward shaping, or `success_reward_threshold`."
                ),
            )
        else:
            self._diagnostic_counters[mode]["flat_rewards"] = 0

    def _warn_on_degenerate_diagnostics(self, mode: str, counter_key: str, message: str) -> None:
        interval = self.args.diagnostics_warning_interval
        if interval == 0:
            return

        self._diagnostic_counters[mode][counter_key] += 1
        count = self._diagnostic_counters[mode][counter_key]
        if count == 1 or count % interval == 0:
            logger.warning("%s Consecutive degenerate steps: %s.", message, count)

    def _compute_policy_loss(self, model, inputs) -> torch.Tensor:
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        per_token_logps, _ = self._get_per_token_logps_and_entropies(
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            compute_entropy=False,
        )
        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps
        advantages = inputs["advantages"]
        if advantages.dim() == 1:
            advantages = advantages.unsqueeze(1)
        log_ratio = per_token_logps - old_per_token_logps
        if self.importance_sampling_level == "sequence":
            log_ratio = (log_ratio * completion_mask).sum(-1, keepdim=True) / completion_mask.sum(
                -1, keepdim=True
            ).clamp(min=1.0)
        coef_1 = torch.exp(log_ratio)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        per_token_loss = -torch.min(coef_1 * advantages, coef_2 * advantages)

        loss = self._aggregate_self_distillation_loss(per_token_loss, completion_mask)

        mode = "train" if self.model.training else "eval"
        self._metrics[mode]["self_distillation/policy_loss"].append(
            self.accelerator.gather(loss.detach()).mean().item()
        )

        accumulation_scale = self.current_gradient_accumulation_steps if mode == "train" else 1.0
        return loss / accumulation_scale

    def _compute_weighted_self_distillation_loss(self, model, inputs) -> torch.Tensor | None:
        if self.args.distillation_weight <= 0.0:
            return None

        accumulation_scale = self.current_gradient_accumulation_steps if self.model.training else 1.0
        distillation_loss = self._compute_self_distillation_loss(model, inputs) / accumulation_scale
        return self.args.distillation_weight * distillation_loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The SDPOTrainer does not support returning outputs")

        if self.args.sdpo_policy_loss_mode == "hybrid":
            policy_loss = self._compute_policy_loss(model, inputs)
            weighted_distillation_loss = self._compute_weighted_self_distillation_loss(model, inputs)
            return policy_loss if weighted_distillation_loss is None else policy_loss + weighted_distillation_loss

        weighted_distillation_loss = self._compute_weighted_self_distillation_loss(model, inputs)
        if weighted_distillation_loss is not None:
            return weighted_distillation_loss
        return self._compute_policy_loss(model, inputs)
