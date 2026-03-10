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

import random
import textwrap

import torch
from datasets import Dataset, IterableDataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase, ProcessorMixin

from ...trainer.grpo_trainer import GRPOTrainer, RewardFunc
from ...trainer.utils import nanmax, nanmin
from .papo_config import PAPOConfig


class PAPOTrainer(GRPOTrainer):
    """
    Trainer for Perception-Aware Policy Optimization (PAPO).

    PAPO extends GRPO/DAPO for multimodal reasoning by adding an implicit perception loss that encourages the model to
    better utilize visual information. The key innovation is computing KL divergence between model outputs on original
    vs. corrupted (masked) images.

    Two variants are supported:
    - PAPO-G: PAPO + GRPO (use loss_type="grpo")
    - PAPO-D: PAPO + DAPO (use loss_type="dapo")

    Example:

    ```python
    from datasets import load_dataset
    from trl.experimental.papo import PAPOTrainer, PAPOConfig

    dataset = load_dataset("your-vlm-dataset", split="train")


    def reward_func(completions, **kwargs):
        # Your reward function for multimodal reasoning
        return [compute_reward(c) for c in completions]


    # PAPO-G
    config = PAPOConfig(
        loss_type="grpo",  # Use GRPO as base
        perception_loss_weight=0.1,
        mask_ratio=0.3,
    )

    # PAPO-G
    config = PAPOConfig(
        loss_type="dapo",  # Use DAPO as base
        perception_loss_weight=0.1,
        mask_ratio=0.3,
    )

    trainer = PAPOTrainer(
        model="Qwen/Qwen2-VL-2B-Instruct",
        reward_funcs=reward_func,
        args=config,
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained (must be a vision-language model).
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions for computing rewards (same as GRPO).
        args ([`PAPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. Must include "prompt" and "image" columns.
        eval_dataset: Same requirements as train_dataset.
        processing_class: Processing class (tokenizer/processor) for the model.
        reward_processing_classes: Processing classes for reward models.
        callbacks: Training callbacks.
        optimizers: Optimizer and scheduler tuple.
        peft_config: PEFT configuration if using parameter-efficient fine-tuning.
    """

    _tag_names = ["trl", "papo"]
    _name = "PAPO"
    _paper = {
        "title": "Perception-Aware Policy Optimization for Multimodal Reasoning",
        "id": "2507.06448",
        # docstyle-ignore
        "citation": textwrap.dedent(
            """\
            @misc{wang2025perceptionawarepolicyoptimizationmultimodal,
                title        = {{Perception-Aware Policy Optimization for Multimodal Reasoning}},
                author       = {Zhenhailong Wang and Xuehang Guo and Sofia Stoica and Haiyang Xu and Hongru Wang and Hyeonjeong Ha and Xiusi Chen and Yangyi Chen and Ming Yan and Fei Huang and Heng Ji},
                year         = 2025,
                url          = {https://arxiv.org/abs/2507.06448},
                archivePrefix= {arXiv},
                eprint       = {2507.06448},
                primaryClass = {cs.CL}
            }"""
        ),
    }

    def __init__(
        self,
        model: str | PreTrainedModel,
        reward_funcs: RewardFunc | list[RewardFunc],
        args: PAPOConfig | None = None,
        train_dataset: Dataset | IterableDataset | None = None,
        eval_dataset: Dataset | IterableDataset | dict[str, Dataset | IterableDataset] | None = None,
        processing_class: PreTrainedTokenizerBase | ProcessorMixin | None = None,
        reward_processing_classes: PreTrainedTokenizerBase | list[PreTrainedTokenizerBase] | None = None,
        callbacks=None,
        optimizers=(None, None),
        peft_config=None,
    ):
        # Initialize with default PAPO config if not provided
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = PAPOConfig(f"{model_name}-PAPO")

        # Store PAPO-specific parameters
        self.perception_loss_weight = args.perception_loss_weight
        self.mask_ratio = args.mask_ratio
        self.mask_type = args.mask_type
        self.der_loss_weight1 = args.der_loss_weight1
        self.der_loss_weight2 = args.der_loss_weight2

        # Initialize parent GRPO trainer
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

    def _mask_image(self, pixel_values: torch.Tensor, mask_ratio: float = None) -> torch.Tensor:
        """
        Apply masking to image pixel values.

        Args:
            pixel_values: Image tensor of shape (B, C, H, W) or (B, N, C, H, W) for multi-image
            mask_ratio: Ratio of image to mask (defaults to self.mask_ratio)

        Returns:
            Masked pixel values tensor
        """
        if mask_ratio is None:
            mask_ratio = self.mask_ratio

        masked_pixel_values = pixel_values.clone()

        if self.mask_type == "random":
            # Random pixel masking
            mask = torch.rand_like(pixel_values) > mask_ratio
            masked_pixel_values = masked_pixel_values * mask

        elif self.mask_type == "patch":
            # Patch-based masking (mask contiguous regions)
            B = pixel_values.shape[0]
            if pixel_values.ndim == 4:  # (B, C, H, W)
                C, H, W = pixel_values.shape[1:]
                for i in range(B):
                    # Calculate patch size to mask
                    patch_h = int(H * mask_ratio**0.5)
                    patch_w = int(W * mask_ratio**0.5)
                    # Random starting position
                    start_h = random.randint(0, max(0, H - patch_h))
                    start_w = random.randint(0, max(0, W - patch_w))
                    # Apply mask
                    masked_pixel_values[i, :, start_h : start_h + patch_h, start_w : start_w + patch_w] = 0

            elif pixel_values.ndim == 5:  # (B, N, C, H, W) for multi-image
                N, C, H, W = pixel_values.shape[1:]
                for i in range(B):
                    for n in range(N):
                        patch_h = int(H * mask_ratio**0.5)
                        patch_w = int(W * mask_ratio**0.5)
                        start_h = random.randint(0, max(0, H - patch_h))
                        start_w = random.randint(0, max(0, W - patch_w))
                        masked_pixel_values[i, n, :, start_h : start_h + patch_h, start_w : start_w + patch_w] = 0

        elif self.mask_type == "grid":
            # Grid-based masking (mask regular grid cells)
            if pixel_values.ndim == 4:
                C, H, W = pixel_values.shape[1:]
                grid_size = int((1 / mask_ratio) ** 0.5)
                cell_h, cell_w = H // grid_size, W // grid_size

                for i in range(grid_size):
                    for j in range(grid_size):
                        if random.random() < mask_ratio:
                            masked_pixel_values[:, :, i * cell_h : (i + 1) * cell_h, j * cell_w : (j + 1) * cell_w] = 0

        return masked_pixel_values

    def _compute_loss(self, model, inputs):
        # >>> 1. GRPO loss
        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        # Compute the per_token_logps and the entropy at each position in the completion
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
        )

        if self.top_entropy_quantile < 1.0:
            entropy_mask = self.get_high_entropy_mask(entropies, completion_mask, 1 - self.top_entropy_quantile)
        else:
            entropy_mask = None

        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        # Compute the loss
        advantages = inputs["advantages"]
        # When using num_iterations == 1 and steps_per_generation <= gradient_accumulation_steps
        # old_per_token_logps == per_token_logps, so we can skip it's computation
        # (see _generate_and_score_completions) and use per_token_logps.detach() instead.
        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps

        log_ratio = per_token_logps - old_per_token_logps
        if self.importance_sampling_level == "token":
            log_importance_weights = log_ratio
        elif self.importance_sampling_level == "sequence":
            log_importance_weights = (log_ratio * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)
            log_importance_weights = log_importance_weights.unsqueeze(-1)
        else:
            raise ValueError(
                f"Unknown importance sampling level: {self.importance_sampling_level}. Possible values are 'token' "
                "and 'sequence'."
            )
        # From here, log_importance_weights (and all subsequent tensors, coef_1, coef_2, etc.) shape depends on
        # importance_sampling_level: "token" level: (B, T); "sequence" level: (B, 1)

        coef_1 = torch.exp(log_importance_weights)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)

        # Two-sided clipping
        if self.args.delta is not None:
            coef_1 = torch.clamp(coef_1, max=self.args.delta)

        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if entropy_mask is not None:
            per_token_loss = per_token_loss * entropy_mask
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        if self.loss_type == "grpo":
            loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "dapo":
            normalizer = inputs["num_items_in_batch"] / self.accelerator.num_processes
            loss = (per_token_loss * completion_mask).sum() / normalizer
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        # >>> 2. Implicit Perception Loss
        inputs["pixel_values"] = self._mask_image(inputs["pixel_values"], self.mask_ratio)
        mask_img_per_token_logps, mask_img_entropies = self._get_per_token_logps_and_entropies(
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
        )
        perception_kl = (
            torch.exp(mask_img_per_token_logps - per_token_logps) - (mask_img_per_token_logps - per_token_logps) - 1
        )
        perception_kl = torch.clamp(perception_kl, min=0.0, max=0.2)
        perception_loss = self.perception_loss_weight * perception_kl

        # >>> 3. Double Entropy Loss
        der_loss = self.der_loss_weight1 * entropies + self.der_loss_weight2 * mask_img_entropies

        # PAPO Loss
        loss = (loss - perception_loss + der_loss).mean()
        # Log the metrics
        mode = "train" if self.model.training else "eval"

        completion_token_count = completion_mask.sum().clamp(min=1.0)

        def masked_batch_mean(x):
            if x.shape[1] == 1:  # when importance_sampling_level == "sequence"
                return x.mean()
            else:
                return (x * completion_mask).sum() / completion_token_count

        if self.beta != 0.0:
            mean_kl = masked_batch_mean(per_token_kl)
            self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).nanmean().item())

        mean_entropy = masked_batch_mean(entropies)
        self._metrics[mode]["entropy"].append(self.accelerator.gather(mean_entropy).nanmean().item())

        # Compute the clipped probability ratios
        is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)
        is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages.unsqueeze(1) > 0)
        is_region_clipped = is_low_clipped | is_high_clipped

        low_clip = masked_batch_mean(is_low_clipped.float())
        high_clip = masked_batch_mean(is_high_clipped.float())
        clip_ratio = masked_batch_mean(is_region_clipped.float())

        gathered_low_clip = self.accelerator.gather(low_clip)
        self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/low_min"].append(nanmin(gathered_low_clip).item())
        gathered_high_clip = self.accelerator.gather(high_clip)
        self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/high_max"].append(nanmax(gathered_high_clip).item())
        gathered_clip_ratio = self.accelerator.gather(clip_ratio)
        self._metrics[mode]["clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean().item())
        return loss
