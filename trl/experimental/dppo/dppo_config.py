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

from dataclasses import dataclass, field
from typing import Literal

from ...trainer.grpo_config import GRPOConfig


@dataclass
class DPPOConfig(GRPOConfig):
    """
    Configuration class for DPPOTrainer.

    DPPO (Divergence Proximal Policy Optimization) replaces PPO/GRPO's heuristic ratio-clipping with a principled
    trust region based on direct policy divergence estimates.

    Paper: "Rethinking the Trust Region in LLM Reinforcement Learning" (arXiv:2602.04879)

    Args:
        divergence_type (`Literal["binary_tv", "binary_kl", "topk_tv", "topk_kl"]`, *optional*, defaults to `"binary_tv"`):
            Divergence approximation used for the trust-region mask. Binary variants use only per-token log-probs;
            top-K variants require storing top-K token IDs and log-probs during rollout generation plus full logits
            during training.

        divergence_topk (`int`, *optional*, defaults to `20`):
            K for top-K divergence approximations. Only used when `divergence_type` starts with `"topk_"`.

        clip_ratio_c (`float`, *optional*, defaults to `10.0`):
            Upper bound on the importance-sampling ratio for stability. The IS ratio is clamped to [0, clip_ratio_c].

        epsilon (`float`, inherited from GRPOConfig, default overridden to `0.2`):
            Divergence threshold δ_low. Tokens whose divergence exceeds this when the policy moves in the
            advantage-decreasing direction are masked.

        epsilon_high (`float`, inherited from GRPOConfig, default overridden to `0.28`):
            Divergence threshold δ_high. Tokens whose divergence exceeds this when the policy moves in the
            advantage-increasing direction are masked. The paper recommends asymmetric thresholds.
    """

    divergence_type: Literal["binary_tv", "binary_kl", "topk_tv", "topk_kl"] = field(
        default="binary_tv",
        metadata={
            "help": "Divergence approximation for the trust-region mask. 'binary_tv': absolute probability "
            "difference. 'binary_kl': Bernoulli KL divergence. 'topk_tv': TV over top-K tokens. "
            "'topk_kl': KL over top-K tokens."
        },
    )
    divergence_topk: int = field(
        default=20,
        metadata={
            "help": "K for top-K divergence approximations. Only used when divergence_type starts with 'topk_'."
        },
    )
    clip_ratio_c: float = field(
        default=10.0,
        metadata={"help": "Upper bound on the importance-sampling ratio for stability."},
    )
    epsilon: float = field(
        default=0.2,
        metadata={"help": "Divergence threshold δ_low for the trust-region mask."},
    )
    epsilon_high: float = field(
        default=0.28,
        metadata={"help": "Divergence threshold δ_high for the trust-region mask (asymmetric)."},
    )

    def __post_init__(self):
        super().__post_init__()

        if self.divergence_type not in ("binary_tv", "binary_kl", "topk_tv", "topk_kl"):
            raise ValueError(
                f"divergence_type must be one of 'binary_tv', 'binary_kl', 'topk_tv', 'topk_kl', "
                f"got {self.divergence_type!r}"
            )

        if self.divergence_topk < 1:
            raise ValueError(f"divergence_topk must be >= 1, got {self.divergence_topk}")

        if self.clip_ratio_c <= 0:
            raise ValueError(f"clip_ratio_c must be > 0, got {self.clip_ratio_c}")

        if self.loss_type != "dapo":
            raise ValueError("loss_type {self.loss_type} is not supported for DPPO")

        if self.top_entropy_quantile != 1.0:
            raise ValueError("top_entropy_quantile is not supported for DPPO")

        if self.off_policy_mask_threshold is not None:
            raise ValueError("off_policy_mask_threshold is not supported for DPPO")

        if self.use_transformers_paged:
            raise ValueError(
                "DPPO requires sampled token logprobs from the generation backend. "
                "Transformers paged (`use_transformers_paged=True`) does not support logprob extraction."
            )
