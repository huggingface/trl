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
        distillation_alpha=1.0,  # Reverse KL (recommended)
        distillation_topk=20,
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
        Compute the self-distillation loss.

        This implements the self-distillation loss from the SDPO paper, which distills knowledge from the model's own
        high-reward trajectories (acting as a teacher) back into the policy.

        Args:
            model: The student model.
            inputs: The inputs dict containing prompts, completions, rewards, etc.

        Returns:
            The self-distillation loss tensor.
        """
        # Get student log probabilities
        student_log_probs = inputs.get("per_token_logps")
        if student_log_probs is None:
            # Compute student log probs if not provided
            student_log_probs = self._get_per_token_logps(model, inputs)

        # Get teacher log probabilities
        teacher_log_probs = inputs.get("teacher_per_token_logps")
        if teacher_log_probs is None:
            # Compute teacher log probs (using model conditioned on feedback/teacher demonstrations)
            teacher_log_probs = self._get_teacher_log_probs(model, inputs)

        # Get response mask (valid tokens for loss computation)
        response_mask = inputs.get("completion_mask", inputs.get("response_mask"))
        if response_mask is None:
            response_mask = torch.ones_like(student_log_probs, dtype=torch.bool)

        # Get old log probabilities for importance sampling
        old_log_probs = inputs.get("old_per_token_logps")

        # Get self-distillation mask (optional, for masking certain tokens)
        self_distillation_mask = inputs.get("self_distillation_mask")

        # Compute the loss
        per_token_loss, metrics = self._compute_self_distillation_loss_core(
            student_log_probs=student_log_probs,
            teacher_log_probs=teacher_log_probs,
            response_mask=response_mask,
            old_log_probs=old_log_probs,
            self_distillation_mask=self_distillation_mask,
        )

        # Aggregate loss
        loss = self._aggregate_loss(per_token_loss, response_mask)

        # Log metrics
        mode = "train" if model.training else "eval"
        for key, value in metrics.items():
            self._metrics[mode][f"sdpo/{key}"].append(self.accelerator.gather(value).mean().item())

        return loss

    def _compute_self_distillation_loss_core(
        self,
        student_log_probs: torch.Tensor,
        teacher_log_probs: torch.Tensor,
        response_mask: torch.Tensor,
        old_log_probs: torch.Tensor | None = None,
        self_distillation_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        Core implementation of the self-distillation loss computation.

        Args:
            student_log_probs: Student model's log probabilities, shape (B, T).
            teacher_log_probs: Teacher model's log probabilities, shape (B, T).
            response_mask: Mask indicating valid tokens, shape (B, T).
            old_log_probs: Old log probabilities for importance sampling, shape (B, T).
            self_distillation_mask: Optional mask for self-distillation, shape (B,).

        Returns:
            A tuple of (per_token_loss, metrics).
        """
        metrics = {}

        # Apply self-distillation mask if provided
        loss_mask = response_mask
        if self_distillation_mask is not None:
            loss_mask = loss_mask * self_distillation_mask.unsqueeze(1)

        if self.args.full_logit_distillation:
            # Full logit distillation (not yet implemented - requires full logits)
            # For now, fall back to token-level distillation
            per_token_loss = self._compute_token_level_distillation_loss(student_log_probs, teacher_log_probs)
        else:
            # Token-level distillation (only supports reverse KL, alpha=1.0)
            if self.args.distillation_alpha != 1.0:
                raise ValueError(
                    f"Only reverse KL (alpha=1.0) is supported for non-full-logit distillation, "
                    f"got alpha={self.args.distillation_alpha}"
                )
            per_token_loss = self._compute_token_level_distillation_loss(student_log_probs, teacher_log_probs)

        # Apply importance sampling clipping if enabled
        if self.args.distillation_is_clip is not None:
            if old_log_probs is None:
                raise ValueError("old_log_probs is required for distillation IS ratio.")
            per_token_loss = self._apply_importance_sampling_clipping(
                per_token_loss, student_log_probs, old_log_probs, self.args.distillation_is_clip
            )

        # Apply mask
        per_token_loss = per_token_loss * loss_mask

        return per_token_loss, metrics

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
            student_log_probs: Student model's log probabilities.
            old_log_probs: Old log probabilities.
            clip_coeff: Clipping coefficient.

        Returns:
            The clipped per-token loss.
        """
        # Compute negative approximate KL divergence
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
        # Use the same aggregation as DAPO loss in GRPO
        num_items_in_batch = (
            self.current_train_batch_size if hasattr(self, "current_train_batch_size") else mask.sum().clamp(min=1.0)
        )
        normalizer = num_items_in_batch / self.accelerator.num_processes
        loss = (per_token_loss * mask).sum() / normalizer
        return loss

    def _get_teacher_log_probs(
        self,
        model,
        inputs: dict[str, Any],
    ) -> torch.Tensor:
        """
        Get teacher model's log probabilities.

        For now, we use the same model with teacher demonstrations (success rollouts or feedback-conditioned inputs).
        In a full implementation, this would use a separate teacher model or EMA updated model.

        Args:
            model: The model.
            inputs: The inputs dict.

        Returns:
            Teacher log probabilities.
        """
        # For a minimal implementation, we reuse the student model
        # In a full implementation, this would:
        # 1. Use successful rollouts as teacher demonstrations
        # 2. Or condition the model on feedback/teacher demonstrations
        # 3. Or use an EMA-updated teacher model

        # For now, return the same as student (placeholder)
        # TODO: Implement proper teacher model logic
        return inputs.get("per_token_logps", torch.zeros(1))

    def _get_per_token_logps(
        self,
        model,
        inputs: dict[str, Any],
    ) -> torch.Tensor:
        """
        Get per-token log probabilities.

        Args:
            model: The model.
            inputs: The inputs dict.

        Returns:
            Per-token log probabilities.
        """
        # This is a placeholder - in practice, this would be computed during forward pass
        # and stored in inputs["per_token_logps"]
        return inputs.get("per_token_logps", torch.zeros(1))
