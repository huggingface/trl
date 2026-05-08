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

"""Shared self-distillation loss utilities used by experimental trainers.

This module intentionally holds only the reusable distillation mechanics: callback dispatch, common prompt/context
helpers, and the student-vs-teacher loss computation. Trainer lifecycle and online rollout concerns live in the trainer
classes or their online-specific base.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
import torch.nn.functional as F

from ...trainer.utils import entropy_from_logits, selective_log_softmax
from .self_distillation_config import SelfDistillationConfig


class SelfDistillationMixin:
    """Reusable self-distillation helpers shared across experimental trainers."""

    config_cls = SelfDistillationConfig

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = ["prompt", "privileged_context"]

    def _dispatch_self_distillation_callback(self, event_name: str, **payload) -> None:
        for callback in self.callback_handler.callbacks:
            callback_fn = getattr(callback, event_name, None)
            if callback_fn is not None:
                callback_fn(
                    args=self.args,
                    state=self.state,
                    control=self.control,
                    model=self.model,
                    processing_class=self.processing_class,
                    **payload,
                )

    @staticmethod
    def _split_prompt_and_privileged_context(inputs: list[dict[str, Any]]) -> tuple[list[Any], list[Any]]:
        prompts = [example["prompt"] for example in inputs]
        privileged_contexts = [example.get("privileged_context") for example in inputs]
        return prompts, privileged_contexts

    def _allow_topk_without_full_logit_distillation(self) -> bool:
        return True

    def _get_per_token_logps_and_entropies(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        compute_entropy=False,
    ):
        model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask, "use_cache": False}
        if "logits_to_keep" in self.model_kwarg_keys:
            model_inputs["logits_to_keep"] = logits_to_keep + 1
        logits = model(**model_inputs).logits
        logits = logits[:, :-1, :]
        logits = logits[:, -logits_to_keep:, :]
        logits = logits / self.temperature
        completion_ids = input_ids[:, -logits_to_keep:]
        selected_logps = selective_log_softmax(logits, completion_ids)
        entropies = entropy_from_logits(logits) if compute_entropy else None
        return selected_logps, entropies

    def _compute_self_distillation_loss(
        self,
        model,
        inputs: dict[str, Any],
    ) -> torch.Tensor:
        # Expected batch contract:
        # - required: `prompt_ids`, `prompt_mask`, `completion_ids`, `completion_mask`,
        #   `teacher_input_ids`, `teacher_attention_mask`
        # - optional: `self_distillation_mask` to zero-out samples without teacher supervision,
        #   `old_per_token_logps` to enable IS clipping when generation and optimization are misaligned
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        logits_to_keep = completion_ids.size(1)

        self_distillation_mask = inputs.get("self_distillation_mask")
        if self_distillation_mask is not None:
            response_mask = completion_mask * self_distillation_mask.unsqueeze(1)
        else:
            response_mask = completion_mask

        if response_mask.sum() == 0:
            mode = "train" if model.training else "eval"
            self._log_self_distillation_metric(mode, "distillation_loss", 0.0)
            return torch.tensor(0.0, device=completion_ids.device, requires_grad=True)

        student_input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        student_attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        student_model_inputs = {
            "input_ids": student_input_ids,
            "attention_mask": student_attention_mask,
            "use_cache": False,
        }
        if "logits_to_keep" in self.model_kwarg_keys:
            student_model_inputs["logits_to_keep"] = logits_to_keep + 1

        student_logits = model(**student_model_inputs).logits
        student_logits = student_logits[:, :-1, :]
        student_logits = student_logits[:, -logits_to_keep:, :]
        student_logits = student_logits / self.temperature

        teacher_input_ids = inputs["teacher_input_ids"]
        teacher_attention_mask = inputs["teacher_attention_mask"]
        teacher_model_inputs = {
            "input_ids": teacher_input_ids,
            "attention_mask": teacher_attention_mask,
            "use_cache": False,
        }
        if "logits_to_keep" in self.model_kwarg_keys:
            teacher_model_inputs["logits_to_keep"] = logits_to_keep + 1

        teacher_model = self._get_teacher_model_for_self_distillation(model)
        with torch.no_grad(), self._get_teacher_context_for_self_distillation(model):
            teacher_logits = teacher_model(**teacher_model_inputs).logits
            teacher_logits = teacher_logits[:, :-1, :]
            teacher_logits = teacher_logits[:, -logits_to_keep:, :]
            teacher_logits = teacher_logits / self.temperature

        use_topk_distillation = self.args.distillation_topk is not None and (
            self.args.full_logit_distillation or self._allow_topk_without_full_logit_distillation()
        )
        if use_topk_distillation:
            student_logsumexp = torch.logsumexp(student_logits, dim=-1, keepdim=True)
            topk_student_logits, topk_indices = torch.topk(student_logits, k=self.args.distillation_topk, dim=-1)
            topk_student_log_probs = topk_student_logits - student_logsumexp

            teacher_logsumexp = torch.logsumexp(teacher_logits, dim=-1, keepdim=True)
            topk_teacher_logits = torch.gather(teacher_logits, dim=-1, index=topk_indices)
            topk_teacher_log_probs = topk_teacher_logits - teacher_logsumexp

            if self.args.distillation_add_tail:
                topk_student_log_probs = self._add_tail(topk_student_log_probs)
                topk_teacher_log_probs = self._add_tail(topk_teacher_log_probs)
            else:
                topk_student_log_probs = self._renorm_topk_log_probs(topk_student_log_probs)
                topk_teacher_log_probs = self._renorm_topk_log_probs(topk_teacher_log_probs)

            per_token_loss = self._compute_divergence(
                topk_student_log_probs, topk_teacher_log_probs, self.args.distillation_alpha
            )
        elif self.args.full_logit_distillation:
            student_log_probs = F.log_softmax(student_logits, dim=-1)
            teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
            per_token_loss = self._compute_divergence(
                student_log_probs, teacher_log_probs, self.args.distillation_alpha
            )
        else:
            if self.args.distillation_alpha != 1.0:
                raise ValueError(
                    "Only reverse KL (alpha=1.0) is supported for token-level distillation when "
                    "`full_logit_distillation=False`, "
                    f"got alpha={self.args.distillation_alpha}"
                )
            student_logsumexp = torch.logsumexp(student_logits, dim=-1, keepdim=True)
            teacher_logsumexp = torch.logsumexp(teacher_logits, dim=-1, keepdim=True)
            idx = completion_ids.unsqueeze(-1)
            student_per_token_logps = (torch.gather(student_logits, dim=-1, index=idx) - student_logsumexp).squeeze(-1)
            teacher_per_token_logps = (torch.gather(teacher_logits, dim=-1, index=idx) - teacher_logsumexp).squeeze(-1)
            per_token_loss = self._compute_token_level_distillation_loss(
                student_per_token_logps, teacher_per_token_logps
            )

        if self.args.distillation_is_clip is not None:
            old_log_probs = inputs.get("old_per_token_logps")
            if old_log_probs is not None:
                with torch.no_grad():
                    student_lse = torch.logsumexp(student_logits, dim=-1, keepdim=True)
                    idx = completion_ids.unsqueeze(-1)
                    student_per_token_logps = (torch.gather(student_logits, dim=-1, index=idx) - student_lse).squeeze(
                        -1
                    )
                per_token_loss = self._apply_importance_sampling_clipping(
                    per_token_loss, student_per_token_logps, old_log_probs, self.args.distillation_is_clip
                )

        loss = self._aggregate_self_distillation_loss(per_token_loss, response_mask)

        mode = "train" if model.training else "eval"
        mean_distill_loss = (per_token_loss * response_mask).sum() / response_mask.sum().clamp(min=1.0)
        self._log_self_distillation_metric(
            mode,
            "distillation_loss",
            self.accelerator.gather(mean_distill_loss).mean().item(),
        )

        return loss

    def _get_teacher_model_for_self_distillation(self, model):
        teacher_model = getattr(self, "teacher_model", None)
        if teacher_model is None:
            return model
        return teacher_model

    def _get_teacher_context_for_self_distillation(self, model):
        return nullcontext()

    def _log_self_distillation_metric(self, mode: str, metric_name: str, value: float) -> None:
        metric_prefix = getattr(self, "_name", "self_distillation").lower().replace(" ", "_")
        self._metrics[mode][f"self_distillation/{metric_name}"].append(value)
        self._metrics[mode][f"{metric_prefix}/{metric_name}"].append(value)

    @staticmethod
    def _compute_divergence(
        student_log_probs: torch.Tensor,
        teacher_log_probs: torch.Tensor,
        alpha: float,
    ) -> torch.Tensor:
        if alpha == 0.0:
            kl = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True)
        elif alpha == 1.0:
            kl = F.kl_div(teacher_log_probs, student_log_probs, reduction="none", log_target=True)
        else:
            alpha_t = torch.tensor(alpha, dtype=student_log_probs.dtype, device=student_log_probs.device)
            mixture = torch.logsumexp(
                torch.stack([student_log_probs + torch.log(1 - alpha_t), teacher_log_probs + torch.log(alpha_t)]),
                dim=0,
            )
            kl_teacher = F.kl_div(mixture, teacher_log_probs, reduction="none", log_target=True)
            kl_student = F.kl_div(mixture, student_log_probs, reduction="none", log_target=True)
            kl = torch.lerp(kl_student, kl_teacher, alpha)
        return kl.sum(-1)

    @staticmethod
    def _add_tail(log_probs: torch.Tensor) -> torch.Tensor:
        log_s = torch.logsumexp(log_probs, dim=-1, keepdim=True)
        log_s = torch.clamp(log_s, max=-1e-7)
        tail_log = torch.log(-torch.expm1(log_s))
        return torch.cat([log_probs, tail_log], dim=-1)

    @staticmethod
    def _renorm_topk_log_probs(log_probs: torch.Tensor) -> torch.Tensor:
        return log_probs - torch.logsumexp(log_probs, dim=-1, keepdim=True)

    @staticmethod
    def _compute_token_level_distillation_loss(
        student_log_probs: torch.Tensor,
        teacher_log_probs: torch.Tensor,
    ) -> torch.Tensor:
        # This is the token-level reverse-KL surrogate used by the official SDPO implementation for
        # `full_logit_distillation=False`. It intentionally treats the teacher log-probs as fixed targets
        # and keeps only the score-function term for the sampled student tokens.
        log_ratio = student_log_probs - teacher_log_probs
        return log_ratio.detach() * student_log_probs

    @staticmethod
    def _apply_importance_sampling_clipping(
        per_token_loss: torch.Tensor,
        student_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        clip_coeff: float,
    ) -> torch.Tensor:
        negative_approx_kl = (student_log_probs - old_log_probs).detach()
        negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
        ratio = torch.exp(negative_approx_kl).clamp(max=clip_coeff)
        return per_token_loss * ratio

    def _aggregate_self_distillation_loss(
        self,
        per_token_loss: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> torch.Tensor:
        loss_type = self.loss_type
        if loss_type == "grpo":
            loss = (per_token_loss * response_mask).sum(-1) / response_mask.sum(-1).clamp(min=1.0)
            return loss.mean()
        if loss_type == "bnpo":
            return (per_token_loss * response_mask).sum() / response_mask.sum().clamp(min=1.0)
        if loss_type == "dr_grpo":
            return (per_token_loss * response_mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
        if loss_type in ["dapo", "luspo", "cispo", "sapo"]:
            return (per_token_loss * response_mask).sum() / response_mask.sum().clamp(min=1.0)
        raise ValueError(f"Unsupported loss_type for self-distillation: {loss_type}")
