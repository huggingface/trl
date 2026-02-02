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
from typing import Any

import torch
import torch.nn.functional as F

from trl.trainer.grpo_trainer import GRPOTrainer
from trl.trainer.utils import pad

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

        # Stash for per-func rewards from _calculate_rewards
        self._last_rewards_per_func = None

    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        rewards_per_func = super()._calculate_rewards(inputs, prompts, completions, completion_ids_list)
        self._last_rewards_per_func = rewards_per_func
        return rewards_per_func

    def _generate_and_score_completions(
        self, inputs: list[dict[str, torch.Tensor | Any]]
    ) -> dict[str, torch.Tensor | Any]:
        # Stash prompts before super() consumes inputs
        prompts = [x["prompt"] for x in inputs]

        output = super()._generate_and_score_completions(inputs)

        # Compute weighted rewards from stashed per-func rewards (globally gathered)
        device = self.accelerator.device
        rewards_per_func = self._last_rewards_per_func  # shape: (total_samples, num_funcs)
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

        # Build teacher inputs and add to output
        self._build_teacher_inputs(output, prompts, rewards)

        return output

    def _build_teacher_inputs(
        self,
        output: dict[str, torch.Tensor | Any],
        prompts: list,
        rewards: torch.Tensor,
    ):
        """Build teacher-conditioned inputs by reprompting with successful demonstrations."""
        device = self.accelerator.device
        num_generations = self.num_generations
        total_samples = rewards.shape[0]  # globally gathered count

        completion_ids = output["completion_ids"]  # local process, padded (B_local, T_comp)

        # Process slice for this process (same logic as parent)
        num_local = len(prompts)  # prompts per process
        process_start = self.accelerator.process_index * num_local
        process_slice = slice(process_start, process_start + num_local)

        # We need global completion_ids to decode demonstrations from other generations in the group.
        # Gather completion_ids across processes.
        all_completion_ids = self.accelerator.gather(completion_ids)  # (total_samples, T_comp)

        threshold = self.args.success_reward_threshold
        dont_reprompt_self = self.args.dont_reprompt_on_self_success

        # Gather all prompts across processes to map global indices to prompt text
        from accelerate.utils import gather_object

        all_prompts = gather_object(prompts)  # list of all prompts across processes

        # Build per-sample teacher messages
        teacher_messages_list = []
        self_distillation_mask = torch.ones(total_samples, device=device)

        for i in range(total_samples):
            group_idx = i // num_generations
            group_start = group_idx * num_generations
            group_end = group_start + num_generations

            if self_distillation_mask[i].item() == 0.0:
                # No successful demo found; use original prompt (loss will be masked)
                original_prompt = all_prompts[group_idx]
                teacher_messages_list.append(original_prompt)
                continue

            # Find successful demo
            successful = []
            for j in range(group_start, group_end):
                if dont_reprompt_self and j == i:
                    continue
                if rewards[j].item() >= threshold:
                    successful.append(j)

            demo_idx = successful[0]
            demo_ids = all_completion_ids[demo_idx]
            demo_ids = demo_ids[demo_ids != self.processing_class.pad_token_id]
            demo_text = self.processing_class.decode(demo_ids, skip_special_tokens=True)

            if self.args.remove_thinking_from_demonstration:
                demo_text = re.sub(r"<think>.*?</think>", "", demo_text, flags=re.DOTALL).strip()

            original_prompt = all_prompts[group_idx]

            # Format the solution text
            solution_text = self.args.solution_template.format(successful_previous_attempt=demo_text)

            # Build the reprompted message
            # original_prompt is a list of message dicts (conversational format)
            # Extract the text content from the last user message
            if isinstance(original_prompt, list):
                # Conversational format - extract text from last user message
                prompt_text = ""
                for msg in original_prompt:
                    if msg.get("role") == "user":
                        content = msg.get("content", "")
                        if isinstance(content, list):
                            prompt_text = " ".join(
                                part.get("text", "") for part in content if part.get("type") == "text"
                            )
                        else:
                            prompt_text = content

                reprompted_text = self.args.reprompt_template.format(prompt=prompt_text, solution=solution_text)
                # Build new conversational message
                teacher_messages_list.append([{"role": "user", "content": reprompted_text}])
            else:
                reprompted_text = self.args.reprompt_template.format(prompt=original_prompt, solution=solution_text)
                teacher_messages_list.append(reprompted_text)

        # Tokenize teacher messages
        teacher_prompt_ids_list = []
        for msg in teacher_messages_list:
            if isinstance(msg, list) and isinstance(msg[0], dict):
                # Conversational format
                tokenized = self.processing_class.apply_chat_template(
                    msg, tokenize=True, add_generation_prompt=True, return_tensors="pt"
                )
                if isinstance(tokenized, dict):
                    ids = tokenized["input_ids"].squeeze(0)
                else:
                    ids = tokenized.squeeze(0)
                # Truncate to max_reprompt_len
                if ids.shape[0] > self.args.max_reprompt_len:
                    ids = ids[-self.args.max_reprompt_len :]
                teacher_prompt_ids_list.append(ids)
            else:
                ids = self.processing_class.encode(msg, return_tensors="pt").squeeze(0)
                if ids.shape[0] > self.args.max_reprompt_len:
                    ids = ids[-self.args.max_reprompt_len :]
                teacher_prompt_ids_list.append(ids)

        # Pad teacher prompt ids (left-padded like student prompts)
        teacher_prompt_ids = [ids.to(device) for ids in teacher_prompt_ids_list]
        teacher_prompt_mask = [torch.ones(len(ids), dtype=torch.long, device=device) for ids in teacher_prompt_ids]
        teacher_prompt_ids = pad(teacher_prompt_ids, padding_value=self.pad_token_id, padding_side="left")
        teacher_prompt_mask = pad(teacher_prompt_mask, padding_value=0, padding_side="left")

        # Concatenate with completion_ids (global) to form teacher_input_ids
        teacher_input_ids = torch.cat([teacher_prompt_ids, all_completion_ids], dim=1)
        teacher_attention_mask = torch.cat(
            [teacher_prompt_mask, (all_completion_ids != self.pad_token_id).long()], dim=1
        )

        # Slice to local process portion
        teacher_input_ids = teacher_input_ids[process_slice]
        teacher_attention_mask = teacher_attention_mask[process_slice]
        self_distillation_mask = self_distillation_mask[process_slice]

        output["teacher_input_ids"] = teacher_input_ids
        output["teacher_attention_mask"] = teacher_attention_mask
        output["self_distillation_mask"] = self_distillation_mask

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

        The teacher sees reprompted inputs containing a successful demonstration, making the same model a better
        teacher through conditioning.

        Args:
            model: The student model.
            inputs: The inputs dict containing prompts, completions, teacher_input_ids, etc.

        Returns:
            The self-distillation loss tensor.
        """
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        logits_to_keep = completion_ids.size(1)

        # Apply self_distillation_mask to response_mask
        self_distillation_mask = inputs.get("self_distillation_mask")
        if self_distillation_mask is not None:
            response_mask = completion_mask * self_distillation_mask.unsqueeze(1)
        else:
            response_mask = completion_mask

        # If all masked out, return zero loss
        if response_mask.sum() == 0:
            mode = "train" if model.training else "eval"
            self._metrics[mode]["sdpo/distillation_loss"].append(0.0)
            return torch.tensor(0.0, device=completion_ids.device, requires_grad=True)

        # Student forward pass: standard prompt + completion
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

        # Teacher forward pass: reprompted input + same completion
        teacher_input_ids = inputs["teacher_input_ids"]
        teacher_attention_mask = inputs["teacher_attention_mask"]
        teacher_model_inputs = {
            "input_ids": teacher_input_ids,
            "attention_mask": teacher_attention_mask,
            "use_cache": False,
        }
        if "logits_to_keep" in self.model_kwarg_keys:
            teacher_model_inputs["logits_to_keep"] = logits_to_keep + 1

        with torch.no_grad():
            teacher_logits = model(**teacher_model_inputs).logits
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
        self._metrics[mode]["sdpo/distillation_loss"].append(self.accelerator.gather(mean_distill_loss).mean().item())

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
