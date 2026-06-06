# Copyright 2025 The HuggingFace Team. All rights reserved.
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

"""
HICRA (Hierarchy-Aware Credit Assignment) Trainer for TRL.

This module implements the HICRA algorithm from the paper "Emergent Hierarchical
Reasoning in LLMs through Reinforcement Learning" (arXiv:2509.03646). HICRA extends
GRPO by amplifying learning signals for strategic planning tokens, enabling LLMs to
develop hierarchical reasoning capabilities more efficiently.
"""

import textwrap
from collections import defaultdict
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import torch
from accelerate.logging import get_logger
from datasets import Dataset, IterableDataset
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainerCallback,
)


if TYPE_CHECKING:
    from peft import PeftConfig

from ..extras.profiling import profiling_decorator
from .grpo_trainer import GRPOTrainer
from .hicra_config import HICRAConfig
from .strategic_grams import get_default_strategic_grams, load_strategic_grams_from_file



logger = get_logger(__name__)


class HICRATrainer(GRPOTrainer):
    """
    Trainer for the HICRA (Hierarchy-Aware Credit Assignment) method.

    HICRA extends GRPO by amplifying learning signals for strategic planning tokens,
    enabling LLMs to develop hierarchical reasoning capabilities more efficiently.
    The algorithm was proposed in the paper [Emergent Hierarchical Reasoning in LLMs
    through Reinforcement Learning](https://huggingface.co/papers/2509.03646).

    Example:

    ```python
    from trl import HICRATrainer, HICRAConfig
    from trl.rewards import accuracy_reward
    from datasets import load_dataset

    dataset = load_dataset("trl-lib/DeepMath-103K", split="train")

    config = HICRAConfig(
        learning_rate=1e-6,
        per_device_train_batch_size=4,
        num_train_epochs=1,
        hicra_alpha=0.2,
        use_hicra=True,
    )

    trainer = HICRATrainer(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        reward_funcs=accuracy_reward,
        args=config,
        train_dataset=dataset,
    )
    trainer.train()
    ```

    Args:
        model (`str | PreTrainedModel`):
            Model to be trained. See [`GRPOTrainer`] for details.
        reward_funcs (`RewardFunc | list[RewardFunc]`):
            Reward functions to be used for computing the rewards. See [`GRPOTrainer`] for details.
        args ([`HICRAConfig`], *optional*):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. See [`GRPOTrainer`] for details.
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Dataset | IterableDataset]`):
            Dataset to use for evaluation. See [`GRPOTrainer`] for details.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], [`~transformers.ProcessorMixin`], *optional*):
            Processing class used to process the data. See [`GRPOTrainer`] for details.
        reward_processing_classes ([`~transformers.PreTrainedTokenizerBase`] or `list[PreTrainedTokenizerBase]`, *optional*):
            Processing classes corresponding to the reward functions. See [`GRPOTrainer`] for details.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*):
            List of callbacks to customize the training loop. See [`GRPOTrainer`] for details.
        optimizers (`tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. See [`GRPOTrainer`] for details.
        peft_config ([`~peft.PeftConfig`], *optional*):
            PEFT configuration used to wrap the model. See [`GRPOTrainer`] for details.
        tools (list of `Callable`, *optional*):
            A list of callable tool functions. See [`GRPOTrainer`] for details.
        rollout_func (`RolloutFunc`, *optional*):
            Function to use for generating completions. See [`GRPOTrainer`] for details.
    """

    _tag_names = ["trl", "hicra"]
    _name = "HICRA"
    _paper = {
        "title": "Emergent Hierarchical Reasoning in LLMs through Reinforcement Learning",
        "id": "2509.03646",
        # docstyle-ignore
        "citation": textwrap.dedent("""\
            @article{wang2025emergent,
                title        = {{Emergent Hierarchical Reasoning in LLMs through Reinforcement Learning}},
                author       = {Zihan Wang and Yunxuan Li and Yuzhong Hong and Hao Zhang and Zhihan Liu and Jianyi Yang and Quanquan Gu},
                year         = 2025,
                eprint       = {arXiv:2509.03646},
            }
            """),
    }

    def __init__(
        self,
        model: str | PreTrainedModel,
        reward_funcs: "Callable | list[Callable]",
        args: HICRAConfig | None = None,
        train_dataset: Dataset | IterableDataset | None = None,
        eval_dataset: Dataset | IterableDataset | dict[str, Dataset | IterableDataset] | None = None,
        processing_class: PreTrainedTokenizerBase | ProcessorMixin | None = None,
        reward_processing_classes: PreTrainedTokenizerBase | list[PreTrainedTokenizerBase] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None] = (None, None),
        peft_config: "PeftConfig | None" = None,
        tools: list[Callable] | None = None,
        rollout_func: "Callable | None" = None,
    ):
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
            tools=tools,
            rollout_func=rollout_func,
        )

        # Load Strategic Grams for planning token identification
        self.strategic_grams = self._load_strategic_grams()
        self.sg_token_ids = self._convert_sgs_to_token_ids()

        # Cache for tokenized Strategic Grams as tensors (optimization)
        self._sg_tensor_cache = {}

        # Cache used by _compute_loss to pass forward-pass results to
        # _get_per_token_logps_and_entropies, avoiding a second model forward pass.
        self._hicra_cached_logps = None
        self._hicra_cached_entropies = None

        logger.info(f"Loaded {len(self.strategic_grams)} Strategic Grams for HICRA training")

    def _load_strategic_grams(self) -> list[str]:
        """
        Load Strategic Grams from config or default set.

        Returns:
            List of Strategic Gram strings.
        """
        if self.args.strategic_grams is not None:
            logger.info("Using Strategic Grams provided directly in config")
            return self.args.strategic_grams
        elif self.args.strategic_grams_path is not None:
            logger.info(f"Loading Strategic Grams from file: {self.args.strategic_grams_path}")
            try:
                return load_strategic_grams_from_file(self.args.strategic_grams_path)
            except Exception as e:
                logger.warning(
                    f"Failed to load Strategic Grams from {self.args.strategic_grams_path}: {e}. "
                    f"Falling back to '{self.args.strategic_grams_preset}' preset."
                )
                return get_default_strategic_grams(self.args.strategic_grams_preset)
        else:
            logger.info(f"Using '{self.args.strategic_grams_preset}' Strategic Grams preset")
            return get_default_strategic_grams(self.args.strategic_grams_preset)

    def _convert_sgs_to_token_ids(self) -> dict[int, list[list[int]]]:
        """
        Convert Strategic Grams to token ID sequences for efficient matching.

        This method tokenizes each Strategic Gram and groups them by length (number of tokens)
        to enable efficient sliding window matching during training.

        Returns:
            Dictionary mapping n-gram length to list of token ID sequences.
            Format: {length: [[token_ids], [token_ids], ...]}
        """
        sg_token_ids = {}

        for sg in self.strategic_grams:
            try:
                # Tokenize the Strategic Gram without special tokens
                tokens = self.processing_class.encode(sg, add_special_tokens=False)
                n = len(tokens)

                if n == 0:
                    logger.warning(f"Strategic Gram '{sg}' tokenized to empty sequence, skipping")
                    continue

                if n not in sg_token_ids:
                    sg_token_ids[n] = []

                sg_token_ids[n].append(tokens)
            except Exception as e:
                logger.warning(f"Failed to tokenize Strategic Gram '{sg}': {e}, skipping")
                continue

        if not sg_token_ids:
            logger.warning("No valid Strategic Grams after tokenization")

        return sg_token_ids

    @profiling_decorator
    def identify_planning_tokens(
        self,
        completion_ids: torch.Tensor,  # [batch_size, seq_len]
    ) -> torch.Tensor:  # [batch_size, seq_len] boolean mask
        """
        Identify which tokens are planning tokens (part of Strategic Grams).

        This method uses an optimized vectorized sliding window approach to match Strategic
        Gram token sequences in the completion. A token is marked as a planning token if it
        is part of any Strategic Gram in the current context.

        Optimizations:
        - Caches Strategic Gram tensors to avoid repeated tensor creation
        - Uses vectorized operations for batch processing
        - Processes all Strategic Grams of the same length together
        - Returns early if HICRA is disabled or no planning tokens are needed

        Args:
            completion_ids: Tensor of token IDs with shape [batch_size, seq_len].

        Returns:
            Boolean mask with shape [batch_size, seq_len] where True indicates a planning token.

        Example:
            >>> completion_ids = torch.tensor([[1, 2, 3, 4, 5]])
            >>> planning_mask = trainer.identify_planning_tokens(completion_ids)
            >>> print(planning_mask.shape)
            torch.Size([1, 5])
        """
        # Early return if HICRA is disabled or planning tokens are not used
        if not self.args.use_hicra or not self.args.use_planning_tokens:
            return torch.zeros_like(completion_ids, dtype=torch.bool)

        batch_size, seq_len = completion_ids.shape
        planning_mask = torch.zeros_like(completion_ids, dtype=torch.bool)

        # Handle edge case: empty sequences
        if seq_len == 0:
            return planning_mask

        # Handle edge case: no Strategic Grams
        if not self.sg_token_ids:
            logger.debug("No Strategic Grams available for planning token identification")
            return planning_mask

        device = completion_ids.device

        # For each n-gram length, process all SGs of that length together
        for n, sg_list in self.sg_token_ids.items():
            if n > seq_len or not sg_list:
                continue

            # Get or create cached tensor for all SGs of this length
            cache_key = (n, device)
            if cache_key not in self._sg_tensor_cache:
                # Stack all SG token sequences into a single tensor [num_sgs, n]
                self._sg_tensor_cache[cache_key] = torch.tensor(sg_list, device=device, dtype=completion_ids.dtype)

            sg_tensor = self._sg_tensor_cache[cache_key]  # [num_sgs, n]

            # Create sliding windows for all positions at once
            # Use unfold to create windows: [batch_size, num_windows, n]
            windows = completion_ids.unfold(dimension=1, size=n, step=1)  # [batch_size, seq_len-n+1, n]
            num_windows = windows.shape[1]

            # Reshape for broadcasting: [batch_size, num_windows, 1, n]
            windows_expanded = windows.unsqueeze(2)

            # Reshape SG tensor for broadcasting: [1, 1, num_sgs, n]
            sg_expanded = sg_tensor.unsqueeze(0).unsqueeze(0)

            # Vectorized comparison: [batch_size, num_windows, num_sgs]
            matches = (windows_expanded == sg_expanded).all(dim=3)

            # Check if any SG matches at each position: [batch_size, num_windows]
            any_match = matches.any(dim=2)

            # Mark all tokens in matching windows as planning tokens
            for i in range(num_windows):
                # For each position where there's a match, mark all n tokens
                match_mask = any_match[:, i]  # [batch_size]
                if match_mask.any():
                    planning_mask[:, i : i + n] |= match_mask.unsqueeze(1)

        return planning_mask

    def _topk_threshold(
        self,
        mask: torch.Tensor,  # [batch_size, seq_len]
        values: torch.Tensor,  # [batch_size, seq_len]
        k: float = 0.3,
    ) -> torch.Tensor:
        """
        Compute threshold value at top-k percentile for each sequence.

        This helper function computes a per-sequence threshold at the (1-k) percentile,
        which serves as a baseline for identifying high-entropy tokens. The VeRL
        implementation uses this to compute entropy mean for the top 30% of tokens (k=0.3).

        Reference: VeRL implementation uses this to compute entropy mean for the top 30%
        of tokens, which serves as a baseline for identifying high-entropy tokens.

        Args:
            mask: Boolean mask indicating valid tokens, shape [batch_size, seq_len].
            values: Values to compute threshold from, shape [batch_size, seq_len].
            k: Top-k percentile (default: 0.3 for top 30%).

        Returns:
            Tensor of threshold values with shape [batch_size, 1].
        """
        batch_size = mask.shape[0]
        thresholds = []

        for i in range(batch_size):
            valid_values = values[i][mask[i] > 0]
            if len(valid_values) == 0:
                thresholds.append(0.0)
                continue

            # Get value at (1-k) percentile (top k%)
            k_index = int((1 - k) * len(valid_values))
            k_index = max(0, min(k_index, len(valid_values) - 1))
            sorted_values = torch.sort(valid_values)[0]
            thresholds.append(sorted_values[k_index].item())

        return torch.tensor(thresholds, device=values.device).unsqueeze(1)

    @profiling_decorator
    def modify_advantages_hicra(
        self,
        advantages: torch.Tensor,  # [batch_size, seq_len] - per-token advantages
        completion_ids: torch.Tensor,  # [batch_size, seq_len]
        entropies: torch.Tensor,  # [batch_size, seq_len] - token-level entropies
        response_mask: torch.Tensor,  # [batch_size, seq_len]
        group_ids: np.ndarray,  # [batch_size] - group identifiers for GRPO
        planning_token_mask: torch.Tensor | None = None,  # [batch_size, seq_len]
    ) -> torch.Tensor:
        """
        Apply HICRA advantage modification following VeRL implementation.

        This method implements the HICRA algorithm as described in the VeRL codebase
        (TIGER-AI-Lab/Hierarchical-Reasoner). The VeRL implementation amplifies advantages for:
        1. Tokens with higher-than-average entropy (within correct responses)
        2. Planning tokens (if provided)
        3. Only for responses that are:
           - Correct (advantage > 0)
           - Longer than the average length of correct responses in the group

        Reference: TIGER-AI-Lab/Hierarchical-Reasoner
        verl/trainer/ppo/ray_trainer.py, AdvantageEstimator.HICRA branch

        Key implementation details from VeRL:
        - Uses topk_threshold with k=0.3 to compute entropy mean
        - Normalizes entropies: normalized_entropys = entropys - entropys_mean
        - Filters by: is_correct_response & is_longer_than_average
        - Without planning tokens: advantages *= (1 + target_scaler)
        - With planning tokens: advantages *= (1 + target_scaler * signs)

        Args:
            advantages: Per-token advantages with shape [batch_size, seq_len].
            completion_ids: Token IDs of completions with shape [batch_size, seq_len].
            entropies: Token-level entropies with shape [batch_size, seq_len].
            response_mask: Mask indicating valid tokens with shape [batch_size, seq_len].
            group_ids: Group identifiers for GRPO with shape [batch_size].
            planning_token_mask: Optional mask for planning tokens with shape [batch_size, seq_len].

        Returns:
            Modified advantages with shape [batch_size, seq_len].
        """
        if not self.args.use_hicra:
            return advantages

        alpha = self.args.hicra_alpha  # VeRL uses target_scaler = 0.2
        entropy_topk = self.args.hicra_entropy_topk  # VeRL uses k=0.3

        # VeRL: response_lengths = grpo_calculation_mask.sum(dim=1)
        response_lengths = response_mask.sum(dim=1)  # [batch_size]

        # VeRL: entropys_mean = topk_threshold(grpo_calculation_mask, data.batch[entropy_key], k=0.3)
        entropies_mean = self._topk_threshold(response_mask, entropies, k=entropy_topk)

        # VeRL: normalized_entropys = data.batch[entropy_key] - entropys_mean
        normalized_entropies = entropies - entropies_mean  # [batch_size, seq_len]

        # VeRL: unique_uids = np.unique(uids)
        unique_groups = np.unique(group_ids)

        # VeRL: Iterate over each group
        for group_id in unique_groups:
            # VeRL: is_current_group = torch.from_numpy(uids == uid)
            is_current_group = torch.from_numpy(group_ids == group_id).to(advantages.device)

            # VeRL: group_advantages = advantages[is_current_group, 0]
            # Note: In VeRL, advantages are per-sequence [batch_size, 1], but we have per-token [batch_size, seq_len]
            # We compute per-sequence advantage by taking the mean over valid tokens
            group_advantages_per_token = advantages[is_current_group]  # [group_size, seq_len]
            group_mask = response_mask[is_current_group]  # [group_size, seq_len]
            group_advantages = (group_advantages_per_token * group_mask).sum(dim=1) / group_mask.sum(dim=1).clamp(
                min=1.0
            )  # [group_size]

            # VeRL: group_lengths = response_lengths[is_current_group]
            group_lengths = response_lengths[is_current_group]  # [group_size]

            # VeRL: is_correct_response = group_advantages > 0
            is_correct = group_advantages > 0

            # VeRL: if is_correct_response.any()
            if not is_correct.any():
                continue

            # VeRL: avg_length_of_correct = group_lengths.to(float).mean()
            avg_length = group_lengths.float().mean()

            # VeRL: is_longer_than_average = group_lengths > avg_length_of_correct
            is_longer = group_lengths > avg_length

            # VeRL: should_amplify = is_correct_response & is_longer_than_average
            should_amplify = is_correct & is_longer  # [group_size]

            # VeRL: is_higher_entropy = normalized_entropys[is_current_group]>0
            # VeRL: is_higher_entropy[~should_amplify] = False
            is_high_entropy = normalized_entropies[is_current_group] > 0  # [group_size, seq_len]
            is_high_entropy[~should_amplify] = False

            # VeRL: if planning_token_mask is None
            if planning_token_mask is None:
                # Build a full [total_batch, seq_len] mask so we can use a single-level index
                # operation. Chained indexing (advantages[is_current_group][is_high_entropy])
                # creates an intermediate copy and the in-place *= is silently discarded.
                # VeRL: advantages[is_current_group][is_higher_entropy] *= 1 + target_scaler
                full_mask = torch.zeros_like(advantages, dtype=torch.bool)
                full_mask[is_current_group] = is_high_entropy
                advantages[full_mask] *= 1 + alpha
            else:
                # VeRL: is_planning_token = planning_token_mask[is_current_group]>0
                is_planning = planning_token_mask[is_current_group] > 0  # [group_size, seq_len]

                # VeRL: prev_values = advantages[is_current_group][is_higher_entropy|is_planning_token]
                # VeRL: signs = torch.sign(prev_values)
                amplify_mask = is_high_entropy | is_planning  # [group_size, seq_len]
                full_amplify_mask = torch.zeros_like(advantages, dtype=torch.bool)
                full_amplify_mask[is_current_group] = amplify_mask
                signs = torch.sign(advantages[full_amplify_mask])

                # VeRL: advantages[is_current_group][is_higher_entropy|is_planning_token] *= 1 + target_scaler*signs
                advantages[full_amplify_mask] *= 1 + alpha * signs

        return advantages

    def _get_per_token_logps_and_entropies(self, model, input_ids, attention_mask, logits_to_keep, **kwargs):
        """Return cached logps/entropies when available to avoid a second forward pass."""
        if self._hicra_cached_logps is not None:
            return self._hicra_cached_logps, self._hicra_cached_entropies
        return super()._get_per_token_logps_and_entropies(
            model, input_ids, attention_mask, logits_to_keep, **kwargs
        )

    def _compute_loss(self, model, inputs):
        """
        Compute loss with HICRA-modified advantages.

        Runs one forward pass to obtain per-token entropies, uses them to amplify
        advantages for high-entropy / planning tokens (HICRA modification), then
        delegates the full loss computation and metric logging to the parent
        GRPOTrainer._compute_loss.  The forward-pass result is cached on ``self``
        so that the parent's call to ``_get_per_token_logps_and_entropies`` returns
        immediately without a second pass through the model.

        Args:
            model: The model being trained.
            inputs: Dictionary containing training inputs.

        Returns:
            Loss tensor.
        """
        if not self.args.use_hicra:
            return super()._compute_loss(model, inputs)

        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([inputs["prompt_ids"], completion_ids], dim=1)
        attention_mask = torch.cat([inputs["prompt_mask"], completion_mask], dim=1)

        # Forward pass — results are cached so super()._compute_loss() reuses them.
        per_token_logps, entropies = self._get_per_token_logps_and_entropies(
            model,
            input_ids,
            attention_mask,
            completion_ids.size(1),
            compute_entropy=True,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            num_images=inputs.get("num_images"),
            pixel_attention_mask=inputs.get("pixel_attention_mask"),
            image_sizes=inputs.get("image_sizes"),
            token_type_ids=inputs.get("token_type_ids"),
        )

        # Expand advantages to [batch_size, seq_len] for token-level HICRA modification.
        advantages = inputs["advantages"]
        if advantages.dim() == 1:
            advantages = advantages.unsqueeze(1).expand(-1, completion_ids.size(1)).clone()
        elif advantages.dim() == 2 and advantages.size(1) == 1:
            advantages = advantages.expand(-1, completion_ids.size(1)).clone()

        batch_size = completion_ids.size(0)
        num_generations = self.num_generations if self.model.training else self.num_generations_eval
        group_ids = np.repeat(np.arange(batch_size // num_generations), num_generations)

        planning_token_mask = None
        if self.args.use_planning_tokens and self.sg_token_ids:
            planning_token_mask = self.identify_planning_tokens(completion_ids)

        advantages = self.modify_advantages_hicra(
            advantages=advantages,
            completion_ids=completion_ids,
            entropies=entropies,
            response_mask=completion_mask,
            group_ids=group_ids,
            planning_token_mask=planning_token_mask,
        )
        inputs["advantages"] = advantages

        # Cache forward-pass results; _get_per_token_logps_and_entropies returns them
        # immediately when super()._compute_loss() calls it, avoiding a second pass.
        self._hicra_cached_logps = per_token_logps
        self._hicra_cached_entropies = entropies
        try:
            loss = super()._compute_loss(model, inputs)
        finally:
            self._hicra_cached_logps = None
            self._hicra_cached_entropies = None

        if planning_token_mask is not None:
            self.log_hicra_metrics(
                completion_ids=completion_ids,
                advantages=advantages,
                planning_mask=planning_token_mask,
                response_mask=completion_mask,
            )

        return loss

    def log_hicra_metrics(
        self,
        completion_ids: torch.Tensor,
        advantages: torch.Tensor,
        planning_mask: torch.Tensor,
        response_mask: torch.Tensor,
    ):
        """
        Log HICRA-specific metrics including planning token statistics.

        This method logs metrics related to planning tokens and their advantages,
        providing insights into how HICRA is affecting the training process.
        Timing metrics are automatically logged via the @profiling_decorator on
        identify_planning_tokens and modify_advantages_hicra methods.

        Args:
            completion_ids: Token IDs of completions with shape [batch_size, seq_len].
            advantages: Per-token advantages with shape [batch_size, seq_len].
            planning_mask: Boolean mask for planning tokens with shape [batch_size, seq_len].
            response_mask: Mask indicating valid tokens with shape [batch_size, seq_len].
        """
        mode = "train" if self.model.training else "eval"

        # Only log if configured to do so
        if not self.args.log_planning_token_ratio and not self.args.log_semantic_entropy:
            return

        # Ensure response_mask is bool so that & with planning_mask (bool) stays bool.
        # If response_mask is int/long, PyTorch promotes the result to int and
        # tensor[int_mask] performs integer advanced indexing instead of boolean masking.
        response_mask = response_mask.bool()

        # Planning token ratio: percentage of tokens that are planning tokens
        if self.args.log_planning_token_ratio:
            # Count planning tokens among valid response tokens
            valid_planning_tokens = (planning_mask & response_mask).sum().float()
            valid_total_tokens = response_mask.sum().float().clamp(min=1.0)
            planning_ratio = (valid_planning_tokens / valid_total_tokens).item()

            # Gather across all processes for distributed training
            planning_ratio_tensor = torch.tensor(planning_ratio, device=advantages.device)
            gathered_planning_ratio = self.accelerator.gather(planning_ratio_tensor)
            mean_planning_ratio = gathered_planning_ratio.nanmean().item()

            self._metrics[mode]["hicra/planning_token_ratio"] = self._metrics[mode].get(
                "hicra/planning_token_ratio", []
            )
            self._metrics[mode]["hicra/planning_token_ratio"].append(mean_planning_ratio)

            # Log average advantage amplification for planning vs execution tokens
            if planning_mask.any():
                # Planning token advantages
                planning_advantages = advantages[planning_mask & response_mask]
                if len(planning_advantages) > 0:
                    planning_advantage_mean = planning_advantages.mean()
                    gathered_planning_adv = self.accelerator.gather(planning_advantage_mean)
                    mean_planning_adv = gathered_planning_adv.nanmean().item()

                    self._metrics[mode]["hicra/planning_advantage_mean"] = self._metrics[mode].get(
                        "hicra/planning_advantage_mean", []
                    )
                    self._metrics[mode]["hicra/planning_advantage_mean"].append(mean_planning_adv)

                # Execution token advantages (non-planning tokens)
                execution_mask = (~planning_mask) & response_mask
                execution_advantages = advantages[execution_mask]
                if len(execution_advantages) > 0:
                    execution_advantage_mean = execution_advantages.mean()
                    gathered_execution_adv = self.accelerator.gather(execution_advantage_mean)
                    mean_execution_adv = gathered_execution_adv.nanmean().item()

                    self._metrics[mode]["hicra/execution_advantage_mean"] = self._metrics[mode].get(
                        "hicra/execution_advantage_mean", []
                    )
                    self._metrics[mode]["hicra/execution_advantage_mean"].append(mean_execution_adv)

        # Semantic entropy: diversity of Strategic Grams
        if self.args.log_semantic_entropy and planning_mask.any():
            semantic_entropy = self.compute_semantic_entropy(completion_ids, planning_mask)

            # Gather across all processes
            semantic_entropy_tensor = torch.tensor(semantic_entropy, device=advantages.device)
            gathered_semantic_entropy = self.accelerator.gather(semantic_entropy_tensor)
            mean_semantic_entropy = gathered_semantic_entropy.nanmean().item()

            self._metrics[mode]["hicra/semantic_entropy"] = self._metrics[mode].get("hicra/semantic_entropy", [])
            self._metrics[mode]["hicra/semantic_entropy"].append(mean_semantic_entropy)

    def compute_semantic_entropy(
        self,
        completion_ids: torch.Tensor,
        planning_mask: torch.Tensor,
    ) -> float:
        """
        Compute semantic entropy of Strategic Grams in batch.

        Semantic entropy measures the diversity of strategic reasoning patterns
        by computing Shannon entropy over the frequency distribution of Strategic
        Grams found in the completions.

        Memory optimization: Processes sequences one at a time to avoid creating
        large intermediate tensors.

        Args:
            completion_ids: Token IDs of completions with shape [batch_size, seq_len].
            planning_mask: Boolean mask for planning tokens with shape [batch_size, seq_len].

        Returns:
            Semantic entropy value (float). Returns 0.0 if no Strategic Grams are found.

        Example:
            >>> completion_ids = torch.tensor([[1, 2, 3, 4, 5]])
            >>> planning_mask = torch.tensor([[True, True, False, False, False]])
            >>> entropy = trainer.compute_semantic_entropy(completion_ids, planning_mask)
            >>> print(f"Semantic entropy: {entropy:.4f}")
        """
        # Extract all Strategic Grams from completions
        sg_counts = defaultdict(int)

        batch_size, seq_len = completion_ids.shape

        # For each n-gram length
        for n, sg_list in self.sg_token_ids.items():
            if n > seq_len:
                continue

            # Process each sequence in the batch
            for batch_idx in range(batch_size):
                # Extract the sequence for this batch item (memory efficient)
                seq = completion_ids[batch_idx]
                mask = planning_mask[batch_idx]

                # Slide window over the sequence
                for i in range(seq_len - n + 1):
                    # Only count if the first token of the window is marked as planning
                    if not mask[i]:
                        continue

                    # Extract window (convert to list for comparison)
                    window = seq[i : i + n].tolist()

                    # Check if window matches any Strategic Gram
                    for sg_idx, sg_tokens in enumerate(sg_list):
                        if window == sg_tokens:
                            # Use a unique key for each Strategic Gram
                            sg_key = f"{n}_{sg_idx}"
                            sg_counts[sg_key] += 1
                            break  # Only count once per position

        # Compute Shannon entropy over SG frequency distribution
        if not sg_counts:
            return 0.0

        total = sum(sg_counts.values())
        probs = [count / total for count in sg_counts.values()]

        # Shannon entropy: H = -Σ p(x) * log(p(x))
        entropy = -sum(p * np.log(p) for p in probs if p > 0)

        return entropy
