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

from ...trainer.base_config import _BaseConfig
from ...trainer.grpo_config import GRPOConfig


@dataclass
class KLPOConfig(GRPOConfig):
    r"""
    Configuration class for the [`experimental.klpo.KLPOTrainer`].

    [`experimental.klpo.KLPOConfig`] inherits every parameter from [`GRPOConfig`]. KLPO
    (https://yifanzhang-pro.github.io/KLPO/) is a critic-free, single-rollout method: one complete response per prompt
    is sufficient, so `num_generations` defaults to `1` and, unlike GRPO, values below 2 are allowed. KLPO consumes the
    raw terminal reward directly — the group-relative advantages are not used, so `scale_rewards` has no effect on the
    loss.

    The KL regularization is toward the *sampler* (behavior) policy, controlled by `klpo_beta`; it does not involve a
    reference model, so leave the inherited `beta` at its default of `0.0` (setting it would needlessly load a
    reference model and compute unused reference log-probabilities).

    Parameters:
        num_generations (`int`, *optional*, defaults to `1`):
            Number of generations to sample per prompt. KLPO needs only a single complete rollout per prompt.
        klpo_route (`str`, *optional*, defaults to `"token"`):
            Regression route. `"token"` (default) uses a separate detached feedback coefficient `R - klpo_beta * ell_u`
            for each token; `"sequence"` shares one detached trajectory residual `R - klpo_beta * sum(ell_u + k_u)`
            across all tokens of a response.
        kl_estimator (`str`, *optional*, defaults to `"mc"`):
            Conditional-KL estimator supplying the sampler-conditioned score correction. Supported values are:

            - `"mc"` (default): Monte Carlo KL. Averages `mc_samples` independent auxiliary token draws from the
              sampler at each visited prefix. Sequence regression requires `mc_samples >= 2` (leave-one-out residuals).
            - `"topk"`: Top-K Aggregated KL. Keeps the sampler's `kl_top_k` highest-probability tokens per prefix and
              aggregates all remaining tokens into one tail bucket.
            - `"binary"`: Binary KL. Groups the sampled action against its complement; needs only sampled-action
              log-probabilities (no extra records).
            - `"full"`: exact KL over the entire vocabulary. Stores the sampler's full `(B, T, vocab_size)`
              conditionals per generation batch, which is memory-intensive for real vocabularies; prefer `"topk"` with
              a large `kl_top_k` in that case.
        klpo_beta (`float`, *optional*, defaults to `0.1`):
            KL regularization coefficient toward the sampler policy (β in the KLPO report). Each token's feedback
            coefficient is `R - klpo_beta * (log p(a) - log q(a))`, where `p` is the current policy and `q` the
            sampler.
        mc_samples (`int`, *optional*, defaults to `128`):
            Number of independent auxiliary token draws per visited prefix (M in the KLPO report) used by the Monte
            Carlo KL (MC-KL) score correction. Token regression supports any `M >= 1`; sequence regression requires `M
            >= 2`. The report's launchers default to `128`. Only used when `kl_estimator="mc"`.
        kl_top_k (`int`, *optional*, defaults to `128`):
            Head size K of the Top-K Aggregated KL estimator: the sampler's K highest-probability tokens are stored per
            prefix and the rest aggregated into one tail bucket. The report uses `128`. Only used when
            `kl_estimator="topk"`.
        mask_truncated_completions (`bool`, *optional*, defaults to `True`):
            Whether to exclude truncated completions from loss calculation. KLPO expects each row to be a complete
            trajectory carrying a terminal reward, so this defaults to `True` (unlike GRPO).
    """

    num_generations: int | None = field(
        default=1,
        metadata={
            "help": "Number of generations to sample per prompt. KLPO needs only a single complete rollout per prompt."
        },
    )
    klpo_route: str = field(
        default="token",
        metadata={
            "help": "Regression route. 'token' (default) uses a separate detached feedback coefficient per token; "
            "'sequence' shares one detached trajectory residual across all tokens of a response."
        },
    )
    kl_estimator: str = field(
        default="mc",
        metadata={
            "help": "Conditional-KL estimator. Supported values are 'mc' (default, Monte Carlo KL from mc_samples "
            "auxiliary draws per prefix), 'topk' (Top-K Aggregated KL with head size kl_top_k), 'binary' (sampled "
            "action against its complement, no extra records), and 'full' (exact KL over the entire vocabulary, "
            "memory-intensive for real vocabularies)."
        },
    )
    klpo_beta: float = field(
        default=0.1,
        metadata={
            "help": "KL regularization coefficient toward the sampler policy (β in the KLPO report). Each token's "
            "feedback coefficient is `R - klpo_beta * (log p(a) - log q(a))`."
        },
    )
    mc_samples: int = field(
        default=128,
        metadata={
            "help": "Number of independent auxiliary token draws per visited prefix (M in the KLPO report) used by "
            "the Monte Carlo KL (MC-KL) score correction. Must be >= 1 (>= 2 with klpo_route='sequence'). Only used "
            "when kl_estimator='mc'."
        },
    )
    kl_top_k: int = field(
        default=128,
        metadata={
            "help": "Head size K of the Top-K Aggregated KL estimator: the sampler's K highest-probability tokens "
            "are stored per prefix and the rest aggregated into one tail bucket. Only used when kl_estimator='topk'."
        },
    )
    mask_truncated_completions: bool = field(
        default=True,
        metadata={
            "help": "Whether to exclude truncated completions from loss calculation. KLPO expects each row to be a "
            "complete trajectory carrying a terminal reward, so this defaults to `True`."
        },
    )

    def __post_init__(self):
        # We do not use the post_init of GRPOConfig because:
        # 1. num_generations can be < 2 in KLPOConfig: KLPO is a single-rollout method and does not compute
        #    group-relative advantages.
        _BaseConfig.__post_init__(self)

        self.scale_rewards = {True: "group", False: "none"}.get(self.scale_rewards, self.scale_rewards)
        if self.num_generations == 1:
            self.scale_rewards = "none"

        if self.klpo_route not in ("token", "sequence"):
            raise ValueError(f"klpo_route must be 'token' or 'sequence', got {self.klpo_route!r}.")

        if self.kl_estimator not in ("mc", "topk", "binary", "full"):
            raise ValueError(f"kl_estimator must be 'mc', 'topk', 'binary', or 'full', got {self.kl_estimator!r}.")

        if self.mc_samples < 1:
            raise ValueError(f"mc_samples must be >= 1, got {self.mc_samples}.")

        if self.klpo_route == "sequence" and self.kl_estimator == "mc" and self.mc_samples < 2:
            raise ValueError(
                "sequence MC-KL needs mc_samples >= 2 for leave-one-out residuals, got "
                f"{self.mc_samples}. Increase mc_samples or use klpo_route='token'."
            )

        if self.kl_top_k < 1:
            raise ValueError(f"kl_top_k must be >= 1, got {self.kl_top_k}.")

        if self.use_liger_kernel:
            raise ValueError(
                "KLPO is not compatible with `use_liger_kernel=True`: the MC-KL correction gathers current-policy "
                "log-probabilities at auxiliary token IDs, which requires the full logits rather than the chunked "
                "sampled-token path."
            )

        num_processes = self.world_size
        # The current default effective batch size
        if self.generation_batch_size is None and self.steps_per_generation is None:
            self.steps_per_generation = self.gradient_accumulation_steps
            self.generation_batch_size = self.per_device_train_batch_size * num_processes * self.steps_per_generation
        elif self.generation_batch_size is not None and self.steps_per_generation is None:
            # Just ensure the value is divisible by the global batch size
            if self.generation_batch_size % (self.per_device_train_batch_size * num_processes) != 0:
                raise ValueError(
                    f"generation_batch_size ({self.generation_batch_size}) must be divisible by the global batch size "
                    f"({self.per_device_train_batch_size * num_processes})."
                )
            self.steps_per_generation = self.generation_batch_size // (
                self.per_device_train_batch_size * num_processes
            )
        elif self.generation_batch_size is None and self.steps_per_generation is not None:
            self.generation_batch_size = self.per_device_train_batch_size * num_processes * self.steps_per_generation
        else:
            raise ValueError(
                "'generation_batch_size' and 'steps_per_generation' can not be both configured at the same time"
            )

        if self.do_eval and self.eval_strategy != "no":
            # Determine the number of generations to use for evaluation
            num_generations = self.num_generations_eval or self.num_generations

            # Just ensure the value is divisible by the global batch size
            if (self.per_device_eval_batch_size * num_processes) % num_generations != 0:
                raise ValueError(
                    f"The global eval batch size ({self.per_device_eval_batch_size} * {num_processes}) must be "
                    f"divisible by the number of generations used for evaluation ({num_generations})."
                )

        # The generation batch must contain full prompt groups (no partials), so it must be divisible by
        # num_generations.
        if self.generation_batch_size % self.num_generations != 0:
            raise ValueError(
                f"generation_batch_size ({self.generation_batch_size}) must be divisible by num_generations "
                f"({self.num_generations})."
            )
