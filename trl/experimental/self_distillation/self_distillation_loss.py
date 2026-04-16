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

"""Shared self-distillation loss computation.

This module intentionally contains only the reusable distillation loss mechanics. Trainer lifecycle concerns,
callback dispatch, and batch construction live in the trainer classes.
"""

from __future__ import annotations

from typing import Any, Protocol

import torch
import torch.nn.functional as F


class SelfDistillationRuntime(Protocol):
    """Minimal trainer surface required by `SelfDistillationLossComputer`."""

    args: Any
    accelerator: Any
    model_kwarg_keys: Any
    temperature: float
    loss_type: str
    max_completion_length: int
    _metrics: dict[str, Any]
    _name: str

    def _allow_topk_without_full_logit_distillation(self) -> bool: ...

    def _get_teacher_model_for_self_distillation(self, model): ...

    def _get_teacher_context_for_self_distillation(self): ...


class SelfDistillationLossComputer:
    """Computes the shared student-vs-teacher self-distillation loss."""

    def __init__(self, runtime: SelfDistillationRuntime):
        self.runtime = runtime

    def compute_loss(self, model, inputs: dict[str, Any]) -> torch.Tensor:
        prompt_ids = inputs["prompt_ids"]
        prompt_mask = inputs["prompt_mask"]
        completion_ids = inputs["completion_ids"]
        completion_mask = inputs["completion_mask"]
        logits_to_keep = completion_ids.size(1)

        response_mask = self._build_response_mask(completion_mask, inputs.get("self_distillation_mask"))
        if response_mask.sum() == 0:
            mode = "train" if model.training else "eval"
            self._log_distillation_metric(mode, 0.0)
            return torch.tensor(0.0, device=completion_ids.device, requires_grad=True)

        student_logits = self._compute_student_logits(
            model=model,
            prompt_ids=prompt_ids,
            prompt_mask=prompt_mask,
            completion_ids=completion_ids,
            completion_mask=completion_mask,
            logits_to_keep=logits_to_keep,
        )
        teacher_logits = self._compute_teacher_logits(
            model=model,
            teacher_input_ids=inputs["teacher_input_ids"],
            teacher_attention_mask=inputs["teacher_attention_mask"],
            logits_to_keep=logits_to_keep,
        )

        per_token_loss = self._compute_per_token_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            completion_ids=completion_ids,
        )

        old_log_probs = inputs.get("old_per_token_logps")
        if self.runtime.args.distillation_is_clip is not None and old_log_probs is not None:
            student_per_token_logps = self._select_token_log_probs(student_logits, completion_ids)
            per_token_loss = self._apply_importance_sampling_clipping(
                per_token_loss,
                student_per_token_logps,
                old_log_probs,
                self.runtime.args.distillation_is_clip,
            )

        loss = self._aggregate_loss(per_token_loss, response_mask)

        mode = "train" if model.training else "eval"
        mean_distill_loss = (per_token_loss * response_mask).sum() / response_mask.sum().clamp(min=1.0)
        self._log_distillation_metric(mode, self.runtime.accelerator.gather(mean_distill_loss).mean().item())
        return loss

    @staticmethod
    def _build_response_mask(
        completion_mask: torch.Tensor,
        self_distillation_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if self_distillation_mask is None:
            return completion_mask
        return completion_mask * self_distillation_mask.unsqueeze(1)

    def _compute_student_logits(
        self,
        model,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        completion_ids: torch.Tensor,
        completion_mask: torch.Tensor,
        logits_to_keep: int,
    ) -> torch.Tensor:
        student_input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        student_attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        return self._forward_logits(
            model=model,
            input_ids=student_input_ids,
            attention_mask=student_attention_mask,
            logits_to_keep=logits_to_keep,
        )

    def _compute_teacher_logits(
        self,
        model,
        teacher_input_ids: torch.Tensor,
        teacher_attention_mask: torch.Tensor,
        logits_to_keep: int,
    ) -> torch.Tensor:
        teacher_model = self.runtime._get_teacher_model_for_self_distillation(model)
        with torch.no_grad(), self.runtime._get_teacher_context_for_self_distillation():
            return self._forward_logits(
                model=teacher_model,
                input_ids=teacher_input_ids,
                attention_mask=teacher_attention_mask,
                logits_to_keep=logits_to_keep,
            )

    def _forward_logits(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        logits_to_keep: int,
    ) -> torch.Tensor:
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "use_cache": False,
        }
        if "logits_to_keep" in self.runtime.model_kwarg_keys:
            model_inputs["logits_to_keep"] = logits_to_keep + 1

        logits = model(**model_inputs).logits
        logits = logits[:, :-1, :]
        logits = logits[:, -logits_to_keep:, :]
        return logits / self.runtime.temperature

    def _compute_per_token_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        completion_ids: torch.Tensor,
    ) -> torch.Tensor:
        args = self.runtime.args
        use_topk_distillation = args.distillation_topk is not None and (
            args.full_logit_distillation or self.runtime._allow_topk_without_full_logit_distillation()
        )

        if use_topk_distillation:
            return self._compute_topk_distillation_loss(student_logits, teacher_logits, args.distillation_topk)
        if args.full_logit_distillation:
            return self._compute_full_logit_distillation_loss(student_logits, teacher_logits)
        return self._compute_sampled_token_distillation_loss(student_logits, teacher_logits, completion_ids)

    def _compute_topk_distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        topk: int,
    ) -> torch.Tensor:
        args = self.runtime.args
        student_logsumexp = torch.logsumexp(student_logits, dim=-1, keepdim=True)
        topk_student_logits, topk_indices = torch.topk(student_logits, k=topk, dim=-1)
        topk_student_log_probs = topk_student_logits - student_logsumexp

        teacher_logsumexp = torch.logsumexp(teacher_logits, dim=-1, keepdim=True)
        topk_teacher_logits = torch.gather(teacher_logits, dim=-1, index=topk_indices)
        topk_teacher_log_probs = topk_teacher_logits - teacher_logsumexp

        if args.distillation_add_tail:
            topk_student_log_probs = self._add_tail(topk_student_log_probs)
            topk_teacher_log_probs = self._add_tail(topk_teacher_log_probs)
        else:
            topk_student_log_probs = self._renorm_topk_log_probs(topk_student_log_probs)
            topk_teacher_log_probs = self._renorm_topk_log_probs(topk_teacher_log_probs)

        return self._compute_divergence(topk_student_log_probs, topk_teacher_log_probs, args.distillation_alpha)

    def _compute_full_logit_distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        args = self.runtime.args
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
        return self._compute_divergence(student_log_probs, teacher_log_probs, args.distillation_alpha)

    def _compute_sampled_token_distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        completion_ids: torch.Tensor,
    ) -> torch.Tensor:
        if self.runtime.args.distillation_alpha != 1.0:
            raise ValueError(
                "Only reverse KL (alpha=1.0) is supported for token-level distillation when "
                f"`full_logit_distillation=False`, got alpha={self.runtime.args.distillation_alpha}"
            )

        student_per_token_logps = self._select_token_log_probs(student_logits, completion_ids)
        teacher_per_token_logps = self._select_token_log_probs(teacher_logits, completion_ids)
        return self._compute_token_level_distillation_loss(student_per_token_logps, teacher_per_token_logps)

    @staticmethod
    def _select_token_log_probs(
        logits: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        logsumexp = torch.logsumexp(logits, dim=-1, keepdim=True)
        indices = token_ids.unsqueeze(-1)
        return (torch.gather(logits, dim=-1, index=indices) - logsumexp).squeeze(-1)

    def _aggregate_loss(
        self,
        per_token_loss: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> torch.Tensor:
        loss_type = self.runtime.loss_type
        if loss_type == "grpo":
            loss = (per_token_loss * response_mask).sum(-1) / response_mask.sum(-1).clamp(min=1.0)
            return loss.mean()
        if loss_type == "bnpo":
            return (per_token_loss * response_mask).sum() / response_mask.sum().clamp(min=1.0)
        if loss_type == "dr_grpo":
            return (per_token_loss * response_mask).sum() / (
                per_token_loss.size(0) * self.runtime.max_completion_length
            )
        if loss_type in ["dapo", "luspo", "cispo", "sapo"]:
            return (per_token_loss * response_mask).sum() / response_mask.sum().clamp(min=1.0)
        raise ValueError(f"Unsupported loss_type for self-distillation: {loss_type}")

    def _log_distillation_metric(self, mode: str, value: float) -> None:
        metric_prefix = getattr(self.runtime, "_name", "self_distillation").lower().replace(" ", "_")
        self.runtime._metrics[mode]["self_distillation/distillation_loss"].append(value)
        self.runtime._metrics[mode][f"{metric_prefix}/distillation_loss"].append(value)

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
