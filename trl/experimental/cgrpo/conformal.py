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

"""Split-conformal utilities for C-GRPO. Pure Python + NumPy, no torch."""

import math
import random
from collections import Counter
from collections.abc import Sequence

import numpy as np


def conformal_quantile(scores: Sequence[float], delta: float) -> float:
    """
    Split-conformal quantile: the `ceil((n + 1) * (1 - delta))`-th smallest score.

    The `n + 1` correction is what gives coverage `>= 1 - delta` at finite `n`. When the index overruns `n` (very small
    `delta` or `n`), the largest score is returned.
    """
    s = np.sort(np.asarray(list(scores), dtype=float))
    n = s.size
    if n == 0:
        return 1.0
    index = math.ceil((n + 1) * (1.0 - delta))
    return float(s[min(max(index, 1), n) - 1])


def _answer_distribution(answers: Sequence[str], k: int, rng: random.Random) -> list[tuple[str, float]]:
    # Empty strings are failed extractions and carry no mass. Ties are broken uniformly at random: a deterministic
    # order (e.g. alphabetical) makes equal-frequency answers non-interchangeable and breaks exchangeability.
    counts = Counter(a for a in answers[:k] if a != "")
    probs = [(answer, count / k) for answer, count in counts.items()]
    probs.sort(key=lambda item: (-item[1], rng.random()))
    return probs


def aps_score(gold: str, answers: Sequence[str], k: int, rng: random.Random) -> float:
    """
    Randomized APS nonconformity score of `gold` under the empirical distribution of the first `k` answers.

    The score is the mass of every answer ranked ahead of `gold` (ties broken at random) plus `U * p(gold)` with `U ~
    Uniform(0, 1)`, which makes it continuous. It is 1.0 if `gold` was never sampled.
    """
    mass_ahead = 0.0
    for answer, p in _answer_distribution(answers, k, rng):
        if answer == gold:
            return min(mass_ahead + rng.random() * p, 1.0)
        mass_ahead += p
    return 1.0


def prediction_set(answers: Sequence[str], k: int, qhat: float, rng: random.Random) -> list[str]:
    """
    APS prediction set: answers in decreasing frequency until their cumulative mass reaches `qhat`.

    The answer that crosses the threshold is included; excluding it would return the empty set whenever the top answer
    alone exceeds `qhat`, and a singleton could never form.
    """
    out, cumulative = [], 0.0
    for answer, p in _answer_distribution(answers, k, rng):
        out.append(answer)
        cumulative += p
        if cumulative >= qhat:
            break
    return out


def first_success_score(passes: Sequence[bool], k: int) -> float:
    """Execution score `j* / k`, where `j*` is the 1-based position of the first passing completion (1.0 if none)."""
    for j, passed in enumerate(passes[:k], start=1):
        if passed:
            return j / k
    return 1.0


def pass_rate_score(passes: Sequence[bool], k: int) -> float:
    """Execution score `1 - n_pass / k`. Discrete, so ties make coverage conservative rather than exact."""
    return 1.0 - sum(bool(p) for p in passes[:k]) / k


def select_delta_auto(
    scores_at_kmax: Sequence[float], margin: float = 0.05, low: float = 0.05, high: float = 0.95
) -> tuple[float, float]:
    """
    Choose `delta = (1 - solve_rate) + margin`, clipped to `[low, high]`, where `solve_rate` is the fraction of
    calibration examples the policy solves within the largest budget.

    Returns `(delta, solve_rate)`. A fixed stringent `delta` on a weak policy pins every threshold at 1.0, which
    disables early stopping entirely.
    """
    s = np.asarray(list(scores_at_kmax), dtype=float)
    solve_rate = float((s < 1.0 - 1e-9).mean()) if s.size else 0.0
    return float(np.clip(1.0 - solve_rate + margin, low, high)), solve_rate
