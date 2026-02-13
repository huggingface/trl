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

import textwrap
from typing import TYPE_CHECKING

import torch
from datasets import Dataset, IterableDataset

from ...trainer.grpo_trainer import GRPOTrainer, RewardFunc
from ...trainer.utils import nanmax, nanmin
from .dppo_config import DPPOConfig


if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase, ProcessorMixin


class DPPOTrainer(GRPOTrainer):
    """
    Trainer for Divergence Proximal Policy Optimization (DPPO).

    DPPO replaces PPO/GRPO's heuristic ratio-clipping with a principled trust region based on direct policy divergence
    estimates. Instead of clipping the probability ratio, DPPO computes a binary mask based on whether the actual
    divergence (TV or KL) exceeds a threshold δ, only masking tokens that are both moving away from the trust region
    AND have high divergence.

    Four divergence approximations are supported:
    - `binary_tv`: Absolute probability difference |π(a) - μ(a)| (simplest)
    - `binary_kl`: Bernoulli KL divergence between old and new token probabilities
    - `topk_tv`: Total variation over the top-K tokens of the distribution
    - `topk_kl`: KL divergence over the top-K tokens of the distribution

    Paper: "Rethinking the Trust Region in LLM Reinforcement Learning" (arXiv:2602.04879)

    Example:

    ```python
    from datasets import load_dataset
    from trl.experimental.dppo import DPPOTrainer, DPPOConfig

    dataset = load_dataset("your-dataset", split="train")


    def reward_func(completions, **kwargs):
        return [compute_reward(c) for c in completions]


    config = DPPOConfig(
        divergence_type="binary_tv",
        epsilon=0.2,       # δ_low
        epsilon_high=0.28, # δ_high
    )

    trainer = DPPOTrainer(
        model="Qwen/Qwen2.5-0.5B",
        reward_funcs=reward_func,
        args=config,
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions for computing rewards.
        args ([`DPPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training.
        eval_dataset: Same requirements as train_dataset.
        processing_class: Processing class (tokenizer/processor) for the model.
        reward_processing_classes: Processing classes for reward models.
        callbacks: Training callbacks.
        optimizers: Optimizer and scheduler tuple.
        peft_config: PEFT configuration if using parameter-efficient fine-tuning.
    """

    _tag_names = ["trl", "dppo"]
    _name = "DPPO"
    _paper = {
        "title": "Rethinking the Trust Region in LLM Reinforcement Learning",
        "id": "2602.04879",
        # docstyle-ignore
        "citation": textwrap.dedent(
            """\
            @misc{zhang2025rethinkingtrust,
                title        = {{Rethinking the Trust Region in LLM Reinforcement Learning}},
                author       = {Yan Zhang and others},
                year         = 2025,
                url          = {https://arxiv.org/abs/2602.04879},
                archivePrefix= {arXiv},
                eprint       = {2602.04879},
                primaryClass = {cs.LG}
            }"""
        ),
    }

    def __init__(
        self,
        model: str | PreTrainedModel,
        reward_funcs: RewardFunc | list[RewardFunc],
        args: DPPOConfig | None = None,
        train_dataset: Dataset | IterableDataset | None = None,
        eval_dataset: Dataset | IterableDataset | dict[str, Dataset | IterableDataset] | None = None,
        processing_class: PreTrainedTokenizerBase | ProcessorMixin | None = None,
        reward_processing_classes: PreTrainedTokenizerBase | list[PreTrainedTokenizerBase] | None = None,
        callbacks=None,
        optimizers=(None, None),
        peft_config=None,
    ):
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = DPPOConfig(f"{model_name}-DPPO")

        self.divergence_type = args.divergence_type
        self.divergence_topk = args.divergence_topk
        self.clip_ratio_c = args.clip_ratio_c

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

    def _get_per_token_logps_and_entropies(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        batch_size=None,
        compute_entropy=False,
        return_topk=False,
        pixel_values=None,
        image_grid_thw=None,
        num_images=None,
        pixel_attention_mask=None,
        image_sizes=None,
        token_type_ids=None,
    ):
        """Compute log-probs, entropies, and optionally top-K token IDs and log-probs."""
        if not return_topk:
            return super()._get_per_token_logps_and_entropies(
                model,
                input_ids,
                attention_mask,
                logits_to_keep,
                batch_size=batch_size,
                compute_entropy=compute_entropy,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                num_images=num_images,
                pixel_attention_mask=pixel_attention_mask,
                image_sizes=image_sizes,
                token_type_ids=token_type_ids,
            )

        # Top-K path: we need access to logits to extract top-K info
        from ...trainer.utils import entropy_from_logits, selective_log_softmax

        batch_size = batch_size or input_ids.size(0)
        all_logps = []
        all_entropies = []
        all_topk_ids = []
        all_topk_logps = []

        for start in range(0, input_ids.size(0), batch_size):
            input_ids_batch = input_ids[start : start + batch_size]
            attention_mask_batch = attention_mask[start : start + batch_size]

            model_inputs = {"input_ids": input_ids_batch, "attention_mask": attention_mask_batch}
            if image_grid_thw is not None and pixel_values is not None:
                rows_per_image = image_grid_thw.prod(dim=-1)
                rows_per_sample = torch.split(rows_per_image, num_images)
                rows_per_sample = torch.stack([s.sum() for s in rows_per_sample])
                cum_rows = torch.cat([torch.tensor([0], device=rows_per_sample.device), rows_per_sample.cumsum(0)])
                row_start, row_end = cum_rows[start].item(), cum_rows[start + batch_size].item()
                model_inputs["pixel_values"] = pixel_values[row_start:row_end]
                cum_imgs = torch.tensor([0] + num_images).cumsum(0)
                img_start, img_end = cum_imgs[start], cum_imgs[start + batch_size]
                model_inputs["image_grid_thw"] = image_grid_thw[img_start:img_end]
            elif pixel_values is not None:
                model_inputs["pixel_values"] = pixel_values[start : start + batch_size]
            if pixel_attention_mask is not None:
                model_inputs["pixel_attention_mask"] = pixel_attention_mask[start : start + batch_size]
            if image_sizes is not None:
                model_inputs["image_sizes"] = image_sizes[start : start + batch_size]
            if token_type_ids is not None:
                model_inputs["token_type_ids"] = token_type_ids[start : start + batch_size]

            if "logits_to_keep" in self.model_kwarg_keys:
                model_inputs["logits_to_keep"] = logits_to_keep + 1

            model_inputs["use_cache"] = False

            logits = model(**model_inputs).logits
            logits = logits[:, :-1, :]
            logits = logits[:, -logits_to_keep:, :]
            logits = logits / self.temperature

            completion_ids = input_ids_batch[:, -logits_to_keep:]
            logps = selective_log_softmax(logits, completion_ids)
            all_logps.append(logps)

            if compute_entropy:
                with torch.no_grad():
                    entropies = entropy_from_logits(logits)
                all_entropies.append(entropies)

            # Extract top-K
            with torch.no_grad():
                log_probs_full = logits.log_softmax(dim=-1)
                topk_logps, topk_ids = log_probs_full.topk(self.divergence_topk, dim=-1)
            all_topk_ids.append(topk_ids)
            all_topk_logps.append(topk_logps)

        logps = torch.cat(all_logps, dim=0)
        entropies = torch.cat(all_entropies, dim=0) if compute_entropy else None
        topk_ids = torch.cat(all_topk_ids, dim=0)
        topk_logps = torch.cat(all_topk_logps, dim=0)
        return logps, entropies, topk_ids, topk_logps

    def _generate_and_score_completions(self, inputs):
        output = super()._generate_and_score_completions(inputs)

        if self.divergence_type.startswith("topk_"):
            # Re-score completions to get top-K token IDs and log-probs from the rollout policy
            prompt_ids, prompt_mask = output["prompt_ids"], output["prompt_mask"]
            completion_ids, completion_mask = output["completion_ids"], output["completion_mask"]
            input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
            attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
            logits_to_keep = completion_ids.size(1)

            mode = "train" if self.model.training else "eval"
            batch_size = (
                self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size
            )

            forward_kwargs = {}
            for key in ["pixel_values", "image_grid_thw", "pixel_attention_mask", "image_sizes", "token_type_ids"]:
                if key in output:
                    forward_kwargs[key] = output[key]

            from ...models.utils import disable_gradient_checkpointing

            with torch.no_grad(), disable_gradient_checkpointing(self.model, self.args.gradient_checkpointing_kwargs):
                _, _, topk_ids, topk_logps = self._get_per_token_logps_and_entropies(
                    self.model,
                    input_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size=batch_size,
                    return_topk=True,
                    num_images=output.get("num_images"),
                    **forward_kwargs,
                )

            output["rollout_topk_ids"] = topk_ids
            output["rollout_topk_logps"] = topk_logps

        return output

    def _compute_divergence_mask(self, per_token_logps, old_per_token_logps, advantages, completion_mask, inputs):
        """Compute the DPPO trust-region mask based on divergence type.

        Returns a mask tensor where 1 = keep, 0 = masked out.
        """
        # Current and rollout probabilities (per-token, for the selected action)
        prob = per_token_logps.detach().exp()
        rollout_prob = old_per_token_logps.exp()

        delta_low = self.epsilon_low
        delta_high = self.epsilon_high

        if self.divergence_type == "binary_tv":
            # D = |π(a_t) - μ(a_t)|
            divergence = (prob - rollout_prob).abs()
            # Mask tokens where divergence > threshold AND policy moves away from trust region
            invalid_pos = (divergence > delta_high) & (prob > rollout_prob)  # increasing prob too much
            invalid_neg = (divergence > delta_low) & (prob < rollout_prob)  # decreasing prob too much
            # For positive advantages: mask if increasing prob too much (invalid_pos)
            # For negative advantages: mask if decreasing prob too much (invalid_neg)
            mask = torch.where(advantages > 0, ~invalid_pos, ~invalid_neg)

        elif self.divergence_type == "binary_kl":
            # Bernoulli KL: D = μ log(μ/π) + (1-μ) log((1-μ)/(1-π))
            # Clamp for numerical stability
            prob_c = prob.clamp(1e-7, 1 - 1e-7)
            rollout_prob_c = rollout_prob.clamp(1e-7, 1 - 1e-7)
            kl = (
                rollout_prob_c * (rollout_prob_c / prob_c).log()
                + (1 - rollout_prob_c) * ((1 - rollout_prob_c) / (1 - prob_c)).log()
            )
            # Direction: is the policy moving away?
            moving_away_pos = prob > rollout_prob  # increasing prob
            moving_away_neg = prob < rollout_prob  # decreasing prob
            invalid_pos = (kl > delta_high) & moving_away_pos
            invalid_neg = (kl > delta_low) & moving_away_neg
            mask = torch.where(advantages > 0, ~invalid_pos, ~invalid_neg)

        elif self.divergence_type in ("topk_tv", "topk_kl"):
            rollout_topk_ids = inputs["rollout_topk_ids"]  # (B, T, K)
            rollout_topk_logps = inputs["rollout_topk_logps"]  # (B, T, K)

            # Get current model's log-probs for the top-K tokens from rollout
            # We need full logits for this, so do a forward pass
            prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
            completion_ids = inputs["completion_ids"]
            input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
            attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
            logits_to_keep = completion_ids.size(1)

            forward_kwargs = {}
            for key in ["pixel_values", "image_grid_thw", "pixel_attention_mask", "image_sizes", "token_type_ids"]:
                if key in inputs:
                    forward_kwargs[key] = inputs[key]

            # Forward pass to get current logits
            model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
            model_inputs.update(forward_kwargs)
            if "logits_to_keep" in self.model_kwarg_keys:
                model_inputs["logits_to_keep"] = logits_to_keep + 1
            model_inputs["use_cache"] = False

            # Use the model that's already available (from the training step)
            # We need logits here - they were already computed in the forward pass but not saved.
            # We'll use the per_token_logps we already have plus gather from full logits.
            # Actually, we need to do a separate forward to get full logits for top-K gathering.
            # This is the cost of top-K divergence methods.
            with torch.no_grad():
                logits = self.model(**model_inputs).logits
                logits = logits[:, :-1, :]
                logits = logits[:, -logits_to_keep:, :]
                logits = logits / self.temperature
                current_log_probs = logits.log_softmax(dim=-1)  # (B, T, V)

            # Gather current log-probs at rollout top-K positions
            current_topk_logps = current_log_probs.gather(-1, rollout_topk_ids)  # (B, T, K)

            # Convert to probabilities for divergence computation
            rollout_topk_probs = rollout_topk_logps.exp()  # (B, T, K)
            current_topk_probs = current_topk_logps.exp()  # (B, T, K)

            if self.divergence_type == "topk_tv":
                # TV = 0.5 * sum_k |π_k - μ_k| over top-K
                divergence = 0.5 * (current_topk_probs - rollout_topk_probs).abs().sum(dim=-1)  # (B, T)
            else:
                # KL = sum_k μ_k * log(μ_k / π_k) over top-K
                ratio = (rollout_topk_probs / current_topk_probs.clamp(min=1e-7)).clamp(min=1e-7)
                divergence = (rollout_topk_probs * ratio.log()).sum(dim=-1)  # (B, T)

            # Direction check: overall probability mass shift
            moving_away_pos = prob > rollout_prob
            moving_away_neg = prob < rollout_prob
            invalid_pos = (divergence > delta_high) & moving_away_pos
            invalid_neg = (divergence > delta_low) & moving_away_neg
            mask = torch.where(advantages > 0, ~invalid_pos, ~invalid_neg)

        else:
            raise ValueError(f"Unknown divergence_type: {self.divergence_type}")

        return mask.float() * completion_mask

    def _compute_loss(self, model, inputs):
        # Compute per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        mask = completion_mask if "tool_mask" not in inputs else completion_mask * inputs["tool_mask"]

        per_token_logps, entropies = self._get_per_token_logps_and_entropies(
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            compute_entropy=True,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            num_images=inputs.get("num_images"),
            pixel_attention_mask=inputs.get("pixel_attention_mask"),
            image_sizes=inputs.get("image_sizes"),
            token_type_ids=inputs.get("token_type_ids"),
        )

        if self.top_entropy_quantile < 1.0:
            entropy_mask = self.get_high_entropy_mask(entropies, mask, 1 - self.top_entropy_quantile)
        else:
            entropy_mask = None

        # Compute the loss
        advantages = inputs["advantages"]
        if advantages.dim() == 1:
            advantages = advantages.unsqueeze(1)

        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps

        # Off-policy mask (same as GRPO)
        if self.off_policy_mask_threshold is not None:
            sampling_per_token_logps = inputs.get("sampling_per_token_logps", old_per_token_logps)
            off_policy_mask = self.get_off_policy_mask(
                advantages=advantages,
                per_token_logps=per_token_logps,
                sampling_per_token_logps=sampling_per_token_logps,
                mask=mask,
                off_policy_threshold=self.off_policy_mask_threshold,
            )

        # KL divergence with reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        # DPPO: compute IS ratio (clamped, detached) and divergence mask
        ratio = torch.clamp(torch.exp(per_token_logps - old_per_token_logps), max=self.clip_ratio_c).detach()
        divergence_mask = self._compute_divergence_mask(per_token_logps, old_per_token_logps, advantages, mask, inputs)

        # DPPO loss: -advantages * ratio * mask * log_prob
        per_token_loss = -advantages * ratio * divergence_mask * per_token_logps

        if self.off_policy_mask_threshold is not None:
            per_token_loss = per_token_loss * off_policy_mask

        if entropy_mask is not None:
            per_token_loss = per_token_loss * entropy_mask

        if self.use_vllm and self.vllm_importance_sampling_correction:
            per_token_loss = per_token_loss * inputs["importance_sampling_ratio"]

        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        # Normalize loss (reuse GRPO's normalization branches)
        mode = "train" if self.model.training else "eval"
        if self.loss_type in ["grpo", "sapo"]:
            loss = ((per_token_loss * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()
            normalizer = self.current_gradient_accumulation_steps if mode == "train" else 1.0
            loss = loss / normalizer
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * mask).sum() / mask.sum().clamp(min=1.0)
            normalizer = self.current_gradient_accumulation_steps if mode == "train" else 1.0
            loss = loss / normalizer
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
            normalizer = self.current_gradient_accumulation_steps if mode == "train" else 1.0
            loss = loss / normalizer
        elif self.loss_type in ["cispo", "dapo"]:
            normalizer = inputs["num_items_in_batch"] / self.accelerator.num_processes
            loss = (per_token_loss * mask).sum() / normalizer
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Log metrics
        completion_token_count = mask.sum().clamp(min=1.0)

        def masked_batch_mean(x):
            if x.shape[1] == 1:
                return x.mean()
            return (x * mask).sum() / completion_token_count

        if self.beta != 0.0:
            mean_kl = masked_batch_mean(per_token_kl)
            self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).nanmean().item())

        mean_entropy = masked_batch_mean(entropies)
        self._metrics[mode]["entropy"].append(self.accelerator.gather(mean_entropy).nanmean().item())

        # Log divergence mask statistics (analogous to clip_ratio in GRPO)
        is_masked = (divergence_mask == 0) & (mask > 0)
        is_masked_pos = is_masked & (advantages > 0)
        is_masked_neg = is_masked & (advantages < 0)

        mask_ratio_pos = masked_batch_mean(is_masked_pos.float())
        mask_ratio_neg = masked_batch_mean(is_masked_neg.float())
        mask_ratio = masked_batch_mean(is_masked.float())

        gathered_mask_ratio_neg = self.accelerator.gather(mask_ratio_neg)
        self._metrics[mode]["clip_ratio/low_mean"].append(gathered_mask_ratio_neg.nanmean().item())
        self._metrics[mode]["clip_ratio/low_min"].append(nanmin(gathered_mask_ratio_neg).item())
        gathered_mask_ratio_pos = self.accelerator.gather(mask_ratio_pos)
        self._metrics[mode]["clip_ratio/high_mean"].append(gathered_mask_ratio_pos.nanmean().item())
        self._metrics[mode]["clip_ratio/high_max"].append(nanmax(gathered_mask_ratio_pos).item())
        gathered_mask_ratio = self.accelerator.gather(mask_ratio)
        self._metrics[mode]["clip_ratio/region_mean"].append(gathered_mask_ratio.nanmean().item())

        return loss
