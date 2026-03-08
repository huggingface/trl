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
from typing import Any

import torch

from ...trainer.callbacks import SyncRefModelCallback
from ..self_distillation.base_self_distillation_trainer import BaseSelfDistillationTrainer
from ..self_distillation.teacher_context import SuccessfulRolloutTeacherContextBuilder
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


class SDPOTrainer(BaseSelfDistillationTrainer):
    """
    Trainer for Self-Distillation Policy Optimization (SDPO).

    SDPO augments on-policy optimization with self-distillation from the model's own high-reward trajectories. It
    converts tokenized feedback into a dense learning signal without any external teacher or explicit reward model.
    SDPO treats the current model conditioned on feedback as a self-teacher and distills its feedback-informed
    next-token predictions back into the policy.
    """

    config_cls = SDPOConfig

    def __init__(self, *args, **kwargs):
        kwargs["args"] = self._coerce_self_distillation_args(kwargs.get("args"))
        super().__init__(*args, **kwargs)
        self._last_rewards_per_func = None
        self.teacher_context_builder = SuccessfulRolloutTeacherContextBuilder(self)
        if self.args.teacher_regularization == "ema":
            self.teacher_model = copy.deepcopy(self.model)
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

    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        rewards_per_func = super()._calculate_rewards(inputs, prompts, completions, completion_ids_list)
        self._last_rewards_per_func = rewards_per_func
        return rewards_per_func

    def _generate_and_score_completions(
        self, inputs: list[dict[str, torch.Tensor | Any]]
    ) -> dict[str, torch.Tensor | Any]:
        prompts, privileged_contexts = self._split_prompt_and_privileged_context(inputs)

        output = super()._generate_and_score_completions(inputs)

        device = self.accelerator.device
        rewards_per_func = self._last_rewards_per_func
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

        output.update(self.teacher_context_builder.build(output, prompts, rewards, feedbacks=privileged_contexts))

        mode = "train" if self.model.training else "eval"
        for key, value in self.teacher_context_builder.last_metrics.items():
            self._metrics[mode][key].append(value)

        self._dispatch_self_distillation_callback(
            "on_teacher_context_built",
            teacher_input_ids=output["teacher_input_ids"],
            teacher_attention_mask=output["teacher_attention_mask"],
            self_distillation_mask=output["self_distillation_mask"],
        )

        return output

    def _compute_loss(
        self,
        model,
        inputs,
    ) -> torch.Tensor:
        base_policy_loss = super()._compute_loss(model, inputs)
        accumulation_scale = self.current_gradient_accumulation_steps if self.model.training else 1.0

        if self.args.distillation_weight <= 0.0:
            return base_policy_loss

        sdpo_loss = self._compute_self_distillation_loss(model, inputs) / accumulation_scale
        if self.args.sdpo_policy_loss_mode == "hybrid":
            return base_policy_loss + self.args.distillation_weight * sdpo_loss
        return self.args.distillation_weight * sdpo_loss
