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

from collections import Counter
from dataclasses import dataclass
import random
from typing import Callable

import torch


@dataclass(frozen=True)
class RandOptCandidate:
    """A single perturbation candidate evaluated by RandOpt."""

    seed: int
    sigma: float
    score: float


@dataclass
class RandOptConfig:
    r"""
    Configuration for random parameter-space search.

    Args:
        population_size (`int`, *optional*, defaults to `32`):
            Number of perturbations to evaluate.
        sigma_values (`list[float]`, *optional*, defaults to `[1e-4, 5e-4, 1e-3]`):
            Candidate standard deviations used for Gaussian perturbations.
        top_k (`int`, *optional*, defaults to `4`):
            Number of top perturbations kept after evaluation.
        base_seed (`int`, *optional*, defaults to `0`):
            Base random seed used to derive perturbation seeds.
    """

    population_size: int = 32
    sigma_values: list[float] | None = None
    top_k: int = 4
    base_seed: int = 0

    def __post_init__(self):
        if self.sigma_values is None:
            self.sigma_values = [1e-4, 5e-4, 1e-3]
        if self.population_size < 1:
            raise ValueError("`population_size` must be >= 1.")
        if not self.sigma_values:
            raise ValueError("`sigma_values` must contain at least one value.")
        if any(sigma <= 0 for sigma in self.sigma_values):
            raise ValueError("All `sigma_values` must be > 0.")
        if self.top_k < 1:
            raise ValueError("`top_k` must be >= 1.")
        if self.top_k > self.population_size:
            raise ValueError("`top_k` must be <= `population_size`.")


class RandOptSearch:
    r"""
    Random parameter perturbation search with deterministic restore semantics.

    The algorithm evaluates `population_size` perturbed model copies (in-place perturb, score, restore) and returns the
    top-k perturbation descriptors.

    Args:
        model (`torch.nn.Module`):
            Model whose parameters are perturbed in-place.
        score_fn (`Callable[[torch.nn.Module], float]`):
            Scoring function where higher is better.
        config ([`RandOptConfig`], *optional*):
            RandOpt configuration.
        parameter_filter (`Callable[[str, torch.nn.Parameter], bool]`, *optional*):
            Predicate selecting parameters to perturb. Defaults to all trainable floating-point parameters.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        score_fn: Callable[[torch.nn.Module], float],
        config: RandOptConfig | None = None,
        parameter_filter: Callable[[str, torch.nn.Parameter], bool] | None = None,
    ):
        self.model = model
        self.score_fn = score_fn
        self.config = config or RandOptConfig()
        self.parameter_filter = parameter_filter
        self._params = self._collect_params()
        if not self._params:
            raise ValueError("No eligible parameters found for perturbation.")

    def _collect_params(self) -> list[tuple[str, torch.nn.Parameter]]:
        params: list[tuple[str, torch.nn.Parameter]] = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if not torch.is_floating_point(param):
                continue
            if self.parameter_filter is not None and not self.parameter_filter(name, param):
                continue
            params.append((name, param))
        return params

    @torch.no_grad()
    def _apply_noise(self, seed: int, sigma: float):
        generators: dict[torch.device, torch.Generator] = {}
        for _, param in self._params:
            device = param.device
            if device not in generators:
                generator = torch.Generator(device=device)
                generator.manual_seed(seed)
                generators[device] = generator
            noise = torch.randn(
                param.shape,
                generator=generators[device],
                device=device,
                dtype=param.dtype,
            )
            param.add_(noise, alpha=sigma)

    @torch.no_grad()
    def _save_state(self) -> dict[str, torch.Tensor]:
        return {name: param.detach().clone() for name, param in self._params}

    @torch.no_grad()
    def _restore_state(self, state: dict[str, torch.Tensor]):
        for name, param in self._params:
            param.copy_(state[name])

    def run(
        self,
        progress_callback: Callable[[int, int, RandOptCandidate, float], None] | None = None,
        log_every: int | None = None,
    ) -> tuple[list[RandOptCandidate], list[RandOptCandidate]]:
        """
        Returns:
            tuple:
                - all candidates sorted by score (descending)
                - top-k candidates
        """
        rng = random.Random(self.config.base_seed)
        base_state = self._save_state()
        candidates: list[RandOptCandidate] = []

        try:
            for i in range(self.config.population_size):
                seed = self.config.base_seed + i
                sigma = rng.choice(self.config.sigma_values)
                self._apply_noise(seed=seed, sigma=sigma)
                score = float(self.score_fn(self.model))
                candidate = RandOptCandidate(seed=seed, sigma=sigma, score=score)
                candidates.append(candidate)
                best_score = max(c.score for c in candidates)
                if progress_callback is not None:
                    progress_callback(i + 1, self.config.population_size, candidate, best_score)
                elif log_every is not None and log_every > 0 and (i + 1) % log_every == 0:
                    print(
                        f"[RANDOPT] {i + 1}/{self.config.population_size} "
                        f"seed={seed} sigma={sigma} score={score:.4f} best={best_score:.4f}",
                        flush=True,
                    )
                self._restore_state(base_state)
        finally:
            # Defensive restore for exception safety.
            self._restore_state(base_state)

        candidates.sort(key=lambda candidate: candidate.score, reverse=True)
        return candidates, candidates[: self.config.top_k]


def majority_vote(
    model_answers: list[list[str]],
    k: int | None = None,
    empty_answer: str = "",
) -> list[str]:
    """
    Aggregate answer strings with per-example majority voting.

    Args:
        model_answers (`list[list[str]]`):
            `model_answers[m][i]` is answer from model `m` on example `i`.
        k (`int` or `None`, *optional*, defaults to `None`):
            Number of leading models to include. Uses all models when `None`.
        empty_answer (`str`, *optional*, defaults to `""`):
            Placeholder for missing/invalid answers.
    """
    if not model_answers:
        return []
    num_examples = len(model_answers[0])
    if any(len(answers) != num_examples for answers in model_answers):
        raise ValueError("All model answer lists must have the same length.")

    if k is None:
        selected = model_answers
    else:
        if k < 1:
            raise ValueError("`k` must be >= 1 when provided.")
        selected = model_answers[: min(k, len(model_answers))]

    aggregated: list[str] = []
    for idx in range(num_examples):
        votes = [answers[idx] for answers in selected if answers[idx] != empty_answer]
        if not votes:
            aggregated.append(empty_answer)
            continue
        aggregated.append(Counter(votes).most_common(1)[0][0])
    return aggregated
