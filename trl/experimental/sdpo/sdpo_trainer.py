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

from typing import Any

import torch
import torch.nn.functional as F

from trl.trainer.grpo_trainer import GRPOTrainer

from .sdpo_config import SDPOConfig


class SDPOTrainer(GRPOTrainer):
    """
    Trainer for Self-Distillation Policy Optimization (SDPO).

    SDPO augments on-policy optimization with self-distillation from the model's own high-reward trajectories. It
    converts tokenized feedback into a dense learning signal without any external teacher or explicit reward model.
    SDPO treats the current model conditioned on feedback as a self-teacher and distills its feedback-informed
    next-token predictions back into the policy.

    Args:
        model (`transformers.PreTrainedModel` or `str`):
            The model to train, either a pre-trained model instance or a string model identifier.
        reward_funcs (`list[Callable]` or `Callable`):
            Reward function(s) to compute rewards for generated completions.
        args (`SDPOConfig`, *optional*):
            Configuration for SDPO training. If not provided, a default configuration is used.
        train_dataset (`datasets.Dataset`):
            The training dataset. Each item should have a "prompt" column.
        eval_dataset (`datasets.Dataset`, *optional*):
            The evaluation dataset. Each item should have a "prompt" column.
        processing_class (`transformers.PreTrainedTokenizer` or `transformers.PreTrainedProcessor`, *optional*):
            The tokenizer or processor to use for preprocessing. If not provided, the one associated with the model is
            used.
        peft_config (`dict`, *optional*):
            Configuration for Parameter-Efficient Fine-Tuning (PEFT).
        callbacks (`list[transformers.TrainerCallback]`, *optional*):
            Custom callbacks to use during training.
        **kwargs:
            Additional keyword arguments to pass to the parent `GRPOTrainer` class.

    Example:

    ```python
    from trl import SDPOTrainer
    from trl.rewards import accuracy_reward
    from datasets import load_dataset

    dataset = load_dataset("trl-lib/DeepMath-103K", split="train")

    trainer = SDPOTrainer(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        reward_funcs=accuracy_reward,
        train_dataset=dataset,
        distillation_alpha=0.5,  # JSD (recommended)
        distillation_topk=100,
        use_successful_as_teacher=True,
    )
    trainer.train()
    ```
    """

    def __init__(self, *args, **kwargs):
        # Ensure we're using SDPOConfig
        if not isinstance(kwargs.get("args", None), SDPOConfig):
            # If args is not provided or not SDPOConfig, use default SDPOConfig
            if "args" in kwargs:
                kwargs["args"] = SDPOConfig(**kwargs["args"].__dict__)
            else:
                kwargs["args"] = SDPOConfig()

        super().__init__(*args, **kwargs)

        # SDPO-specific attributes
        self.teacher_model = None
        self.teacher_ema = self.args.ema_update_rate

    def _compute_loss(
        self,
        model,
        inputs,
    ) -> torch.Tensor:
        """
        Compute the loss for SDPO training. This combines the GRPO loss with the self-distillation loss.

        Args:
            model: The model to compute loss for.
            inputs: The inputs dict containing prompts, completions, rewards, etc.

        Returns:
            The computed loss tensor.
        """
        # First, compute the standard GRPO loss
        grpo_loss = super()._compute_loss(model, inputs)

        # Then, compute the self-distillation loss
        if self.args.distillation_weight > 0.0:
            sdpo_loss = self._compute_self_distillation_loss(model, inputs)
            total_loss = grpo_loss + self.args.distillation_weight * sdpo_loss
        else:
            total_loss = grpo_loss

        return total_loss

    def _compute_self_distillation_loss(
        self,
        model,
        inputs: dict[str, Any],
    ) -> torch.Tensor:
        """
        Compute the self-distillation loss via separate forward passes for student and teacher logits.

        This implements the paper's generalized JSD divergence with optional top-K distillation.

        Args:
            model: The student model.
            inputs: The inputs dict containing prompts, completions, rewards, etc.

        Returns:
            The self-distillation loss tensor.
        """
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        response_mask = completion_mask

        # Build model inputs
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "use_cache": False,
        }
        if "logits_to_keep" in self.model_kwarg_keys:
            model_inputs["logits_to_keep"] = logits_to_keep + 1

        # Student forward pass
        student_logits = model(**model_inputs).logits
        student_logits = student_logits[:, :-1, :]
        student_logits = student_logits[:, -logits_to_keep:, :]
        student_logits = student_logits / self.temperature

        # Teacher forward pass (no grad)
        with torch.no_grad():
            teacher_logits = model(**model_inputs).logits
            teacher_logits = teacher_logits[:, :-1, :]
            teacher_logits = teacher_logits[:, -logits_to_keep:, :]
            teacher_logits = teacher_logits / self.temperature

        if self.args.full_logit_distillation:
            # Full-vocabulary divergence: need full (B, T, V) log_softmax
            student_log_probs = F.log_softmax(student_logits, dim=-1)
            teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
            per_token_loss = self._compute_divergence(
                student_log_probs, teacher_log_probs, self.args.distillation_alpha
            )
        elif self.args.distillation_topk is not None:
            # Memory-efficient top-K: compute logsumexp (B, T, 1) and topk on raw logits
            # to avoid materializing full (B, T, V) log_softmax tensors
            student_logsumexp = torch.logsumexp(student_logits, dim=-1, keepdim=True)  # (B, T, 1)
            topk_student_logits, topk_indices = torch.topk(
                student_logits, k=self.args.distillation_topk, dim=-1
            )  # (B, T, K)
            topk_student_log_probs = topk_student_logits - student_logsumexp  # (B, T, K)

            teacher_logsumexp = torch.logsumexp(teacher_logits, dim=-1, keepdim=True)  # (B, T, 1)
            topk_teacher_logits = torch.gather(teacher_logits, dim=-1, index=topk_indices)  # (B, T, K)
            topk_teacher_log_probs = topk_teacher_logits - teacher_logsumexp  # (B, T, K)

            if self.args.distillation_add_tail:
                topk_student_log_probs = self._add_tail(topk_student_log_probs)
                topk_teacher_log_probs = self._add_tail(topk_teacher_log_probs)
            else:
                topk_student_log_probs = self._renorm_topk_log_probs(topk_student_log_probs)
                topk_teacher_log_probs = self._renorm_topk_log_probs(topk_teacher_log_probs)

            per_token_loss = self._compute_divergence(
                topk_student_log_probs, topk_teacher_log_probs, self.args.distillation_alpha
            )
        else:
            # Fallback: token-level reverse KL using only the chosen-token log probs
            if self.args.distillation_alpha != 1.0:
                raise ValueError(
                    f"Only reverse KL (alpha=1.0) is supported for token-level distillation without top-K, "
                    f"got alpha={self.args.distillation_alpha}"
                )
            # Gather log p(chosen token) without full log_softmax
            student_logsumexp = torch.logsumexp(student_logits, dim=-1, keepdim=True)
            teacher_logsumexp = torch.logsumexp(teacher_logits, dim=-1, keepdim=True)
            idx = completion_ids.unsqueeze(-1)
            student_per_token_logps = (torch.gather(student_logits, dim=-1, index=idx) - student_logsumexp).squeeze(-1)
            teacher_per_token_logps = (torch.gather(teacher_logits, dim=-1, index=idx) - teacher_logsumexp).squeeze(-1)
            per_token_loss = self._compute_token_level_distillation_loss(
                student_per_token_logps, teacher_per_token_logps
            )

        # Apply importance sampling clipping if enabled
        if self.args.distillation_is_clip is not None:
            old_log_probs = inputs.get("old_per_token_logps")
            if old_log_probs is not None:
                # Compute per-token log probs for IS ratio without full log_softmax
                with torch.no_grad():
                    student_lse = torch.logsumexp(student_logits, dim=-1, keepdim=True)
                    idx = completion_ids.unsqueeze(-1)
                    student_per_token_logps = (torch.gather(student_logits, dim=-1, index=idx) - student_lse).squeeze(
                        -1
                    )
                per_token_loss = self._apply_importance_sampling_clipping(
                    per_token_loss, student_per_token_logps, old_log_probs, self.args.distillation_is_clip
                )

        # Mask and aggregate
        per_token_loss = per_token_loss * response_mask
        loss = self._aggregate_loss(per_token_loss, response_mask)

        # Log metrics
        mode = "train" if model.training else "eval"
        mean_distill_loss = (per_token_loss * response_mask).sum() / response_mask.sum().clamp(min=1.0)
        self._metrics[mode]["sdpo/distillation_loss"].append(
            self.accelerator.gather(mean_distill_loss).mean().item()
        )

        return loss

    @staticmethod
    def _compute_divergence(
        student_log_probs: torch.Tensor,
        teacher_log_probs: torch.Tensor,
        alpha: float,
    ) -> torch.Tensor:
        """
        Compute generalized divergence between student and teacher distributions.

        Args:
            student_log_probs: Student log probabilities, shape (..., K).
            teacher_log_probs: Teacher log probabilities, shape (..., K).
            alpha: Interpolation parameter. 0=forward KL, 1=reverse KL, 0<alpha<1=JSD.

        Returns:
            Per-token divergence, shape (...) with last dim summed out.
        """
        if alpha == 0.0:
            # Forward KL: KL(teacher || student)
            kl = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True)
        elif alpha == 1.0:
            # Reverse KL: KL(student || teacher)
            kl = F.kl_div(teacher_log_probs, student_log_probs, reduction="none", log_target=True)
        else:
            # Generalized JSD
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
        """
        Add a tail term representing the probability mass of non-top-K tokens.

        Args:
            log_probs: Top-K log probabilities, shape (..., K).

        Returns:
            Log probabilities with tail appended, shape (..., K+1).
        """
        log_s = torch.logsumexp(log_probs, dim=-1, keepdim=True)
        log_s = torch.clamp(log_s, max=-1e-7)
        tail_log = torch.log(-torch.expm1(log_s))
        return torch.cat([log_probs, tail_log], dim=-1)

    @staticmethod
    def _renorm_topk_log_probs(log_probs: torch.Tensor) -> torch.Tensor:
        """
        Renormalize top-K log probabilities to sum to 1.

        Args:
            log_probs: Top-K log probabilities, shape (..., K).

        Returns:
            Renormalized log probabilities, shape (..., K).
        """
        return log_probs - torch.logsumexp(log_probs, dim=-1, keepdim=True)

    def _compute_token_level_distillation_loss(
        self,
        student_log_probs: torch.Tensor,
        teacher_log_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute token-level distillation loss using reverse KL.

        Args:
            student_log_probs: Student model's log probabilities.
            teacher_log_probs: Teacher model's log probabilities.

        Returns:
            The per-token loss.
        """
        # Reverse KL: D_KL(teacher || student)
        log_ratio = student_log_probs - teacher_log_probs
        per_token_loss = log_ratio.detach() * student_log_probs
        return per_token_loss

    def _apply_importance_sampling_clipping(
        self,
        per_token_loss: torch.Tensor,
        student_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        clip_coeff: float,
    ) -> torch.Tensor:
        """
        Apply importance sampling clipping to stabilize training.

        Args:
            per_token_loss: The per-token loss.
            student_log_probs: Student model's per-token log probabilities.
            old_log_probs: Old per-token log probabilities.
            clip_coeff: Clipping coefficient.

        Returns:
            The clipped per-token loss.
        """
        negative_approx_kl = (student_log_probs - old_log_probs).detach()
        negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
        ratio = torch.exp(negative_approx_kl).clamp(max=clip_coeff)
        per_token_loss = per_token_loss * ratio
        return per_token_loss

    def _aggregate_loss(
        self,
        per_token_loss: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Aggregate the per-token loss to a scalar loss.

        Args:
            per_token_loss: The per-token loss.
            mask: Mask indicating valid tokens.

        Returns:
            The aggregated loss.
        """
        num_items_in_batch = (
            self.current_train_batch_size if hasattr(self, "current_train_batch_size") else mask.sum().clamp(min=1.0)
        )
        normalizer = num_items_in_batch / self.accelerator.num_processes
        loss = (per_token_loss * mask).sum() / normalizer
        return loss
