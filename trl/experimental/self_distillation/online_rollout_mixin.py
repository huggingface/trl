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

"""Online rollout helpers for experimental self-distillation trainers.

This mixin owns generation, reward scoring, grouped reward normalization, and online policy-loss plumbing. It is paired
with `BaseSelfDistillationTrainer` for SDPO-style methods and intentionally kept separate from the generic distillation
loss logic in `self_distillation_mixin.py`.
"""

from __future__ import annotations

import torch
from torch import nn
from transformers.utils import logging

from ...data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from ...models import unwrap_model_for_generation
from ...trainer.base_trainer import _BaseTrainer
from ...trainer.utils import pad


logger = logging.get_logger(__name__)


class OnlineRolloutMixin:
    """Online rollout, reward, and policy-loss utilities shared by SDPO-like trainers."""

    def _apply_prompt_template(self, prompts):
        return [
            maybe_apply_chat_template({"prompt": prompt}, self.processing_class, **self.chat_template_kwargs)["prompt"]
            for prompt in prompts
        ]

    def _build_buffered_batch(self, generation_batch):
        return self._generate_and_score_completions(generation_batch)

    def _generate(self, prompts):
        if self.use_vllm:
            return self._generate_vllm(prompts)
        return self._generate_transformers(prompts)

    def _generate_vllm(self, prompts):
        # Sync weights if training step changed
        if self.state.global_step != self._last_loaded_step:
            self.vllm_generation.sync_weights()
            self._last_loaded_step = self.state.global_step

        # Tokenize prompts to token IDs
        prompts_text = self._apply_prompt_template(prompts)
        tokenized = self.processing_class(
            text=prompts_text,
            return_tensors=None,
            padding=False,
            max_length=self.max_prompt_length,
            truncation=True,
            add_special_tokens=False,
        )
        prompt_ids = tokenized["input_ids"]  # list of list[int]

        # Generate via vLLM — it deduplicates repeated prompts from RepeatSampler internally
        mode = "train" if self.model.training else "eval"
        num_generations = self.num_generations if mode == "train" else self.num_generations_eval
        prompt_ids_out, completion_ids_list, _, _ = self.vllm_generation.generate(
            prompts=prompt_ids,
            images=None,
            num_generations=num_generations,
        )
        return prompt_ids_out, completion_ids_list

    def _generate_transformers(self, prompts):
        # Keep the generation path aligned with the reference trainers: generate from left-padded prompts,
        # then recover completion token spans by trimming prompt tokens and stopping at the first EOS.
        prompts_text = self._apply_prompt_template(prompts)
        generate_inputs = self.processing_class(
            text=prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            max_length=self.max_prompt_length,
            truncation=True,
            add_special_tokens=False,
        )
        # This path already receives tokenized model inputs. Bypass the buffered trainer hook and use the plain
        # tensor/device preparation from `_BaseTrainer`.
        generate_inputs = _BaseTrainer._prepare_inputs(self, generate_inputs)
        with (
            unwrap_model_for_generation(
                self.model_wrapped,
                self.accelerator,
                gather_deepspeed3_params=self.args.ds3_gather_for_generation,
            ) as unwrapped_model,
            torch.no_grad(),
        ):
            prompt_completion_ids = unwrapped_model.generate(
                **generate_inputs, generation_config=self.generation_config
            )
        prompt_ids = generate_inputs["input_ids"]
        prompt_mask = generate_inputs["attention_mask"]
        prompt_length = prompt_ids.size(1)
        completion_ids = prompt_completion_ids[:, prompt_length:]
        is_eos = completion_ids == self.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=completion_ids.device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        seq_idx = torch.arange(is_eos.size(1), device=completion_ids.device).expand(is_eos.size(0), -1)
        completion_mask = (seq_idx <= eos_idx.unsqueeze(1)).int()
        prompt_ids_list = [p[m].tolist() for p, m in zip(prompt_ids, prompt_mask.bool(), strict=False)]
        completion_ids_list = [c[m].tolist() for c, m in zip(completion_ids, completion_mask.bool(), strict=False)]
        return prompt_ids_list, completion_ids_list

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
                # Reward functions operate on tokenized tensors too, so they need the base Trainer input preparation
                # rather than the outer buffered generation hook.
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

    def _generate_and_score_completions(self, inputs):
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"
        prompts = [x["prompt"] for x in inputs]
        prompt_ids_list, completion_ids_list = self._generate(prompts)

        prompt_ids = [torch.tensor(ids) for ids in prompt_ids_list]
        prompt_mask = [torch.ones_like(ids, dtype=torch.long) for ids in prompt_ids]
        prompt_ids = pad(prompt_ids, padding_value=self.pad_token_id, padding_side="left").to(device=device)
        prompt_mask = pad(prompt_mask, padding_value=0, padding_side="left").to(device=device)
        completion_ids = [torch.tensor(ids) for ids in completion_ids_list]
        completion_mask = [torch.ones_like(ids, dtype=torch.long) for ids in completion_ids]
        completion_ids = pad(completion_ids, padding_value=self.pad_token_id, padding_side="right").to(device=device)
        completion_mask = pad(completion_mask, padding_value=0, padding_side="right").to(device=device)

        if self.mask_truncated_completions:
            eos_and_pad = [self.eos_token_id, self.pad_token_id]
            is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids_list], device=device)
            completion_mask = completion_mask * (~is_truncated).unsqueeze(1).int()

        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        with torch.no_grad():
            generate_every = self.args.steps_per_generation * self.num_iterations
            if self.args.gradient_accumulation_steps % generate_every != 0:
                old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    compute_entropy=False,
                )
            else:
                old_per_token_logps = None

        if is_conversational({"prompt": prompts[0]}):
            completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
            completions = [[{"role": "assistant", "content": content}] for content in completions_text]
        else:
            completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

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

        local_batch_size = completion_ids.size(0)
        process_start = self.accelerator.process_index * local_batch_size
        process_slice = slice(process_start, process_start + local_batch_size)
        rewards = rewards[process_slice]
        advantages = advantages[process_slice]

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

        output = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "rewards": rewards,
            "advantages": advantages,
            "num_items_in_batch": completion_mask.sum().detach(),
        }
        if old_per_token_logps is not None:
            output["old_per_token_logps"] = old_per_token_logps

        self._dispatch_self_distillation_callback(
            "on_self_distillation_batch_prepared",
            old_per_token_logps=old_per_token_logps,
            prompt_ids=prompt_ids,
            completion_ids=completion_ids,
        )
        return output

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

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError(f"The {self.__class__.__name__} does not support returning outputs")
        return self._compute_loss(model, inputs)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        if not isinstance(inputs, dict):
            inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
        return loss.detach(), None, None

    def _compute_loss(self, model, inputs):
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
