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

from collections.abc import Callable


def get_repetition_penalty_reward(ngram_size: int = 3, max_penalty: float = -1.0) -> Callable:
    # docstyle-ignore
    r"""
    Reward function that penalizes repeated n-grams in a completion, used to discourage degenerate, repetitive text
    (a common failure mode and reward-hacking strategy when length- or format-shaping rewards are used). Reference:
    Appendix C.2 of the "Demystifying Long Chain-of-Thought Reasoning" paper (https://huggingface.co/papers/2502.03373).

    The penalty is proportional to the fraction of repeated n-grams in the completion:

    $$
    R_{\text{repetition}}(y) = \left(1 - \frac{\#\,\text{unique } n\text{-grams}}{\#\,\text{total } n\text{-grams}}\right) \times p
    $$

    where $p$ is `max_penalty`. A completion with no repeated n-gram gets a reward of `0.0`, while a fully repetitive
    one approaches `max_penalty`. The n-grams are computed over the completion token ids (the paper applies the penalty
    to repeated tokens), so the reward is tokenizer-defined and language-agnostic. Completions with fewer than
    `ngram_size` tokens get a reward of `0.0`.

    Args:
        ngram_size (`int`, *optional*, defaults to `3`):
            Size of the token n-grams to consider.
        max_penalty (`float`, *optional*, defaults to `-1.0`):
            Most negative penalty, applied to a fully repetitive completion. Must be non-positive.

    Returns:
        `Callable`:
            A reward function that takes a list of completion token ids and returns a list of penalties (each in
            `[max_penalty, 0.0]`).

    Example:
    ```python
    >>> from trl.rewards import get_repetition_penalty_reward

    >>> repetition_penalty = get_repetition_penalty_reward(ngram_size=2, max_penalty=-1.0)
    >>> completion_ids = [[1, 2, 3, 4], [5, 5, 5, 5, 5]]
    >>> repetition_penalty(completion_ids)
    [0.0, -0.75]
    ```
    """
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")
    return _RepetitionPenalty(ngram_size, max_penalty)


class _RepetitionPenalty:
    # Callable class rather than a closure so the reward stays picklable: the async GRPO rollout
    # worker forwards reward funcs to a spawned child process, and closures can't be pickled.
    __name__ = "repetition_penalty_reward"

    def __init__(self, ngram_size: int, max_penalty: float):
        self.ngram_size = ngram_size
        self.max_penalty = max_penalty

    def __call__(self, completion_ids: list[list[int]], **kwargs) -> list[float]:
        rewards = []
        for ids in completion_ids:
            if len(ids) < self.ngram_size:
                rewards.append(0.0)
                continue
            ngrams = list(zip(*[ids[i:] for i in range(self.ngram_size)], strict=False))
            scaling = 1 - len(set(ngrams)) / len(ngrams)
            rewards.append(scaling * self.max_penalty if scaling else 0.0)
        return rewards


def get_soft_overlong_punishment(max_completion_len: int, soft_punish_cache: int) -> Callable:
    # docstyle-ignore
    r"""
    Reward function that penalizes overlong completions. It is used to penalize overlong completions, but not to reward
    shorter completions. Reference: Eq. (13) from the DAPO paper (https://huggingface.co/papers/2503.14476)

    $$
    R_{\text{length}}(y) = \begin{cases}
    0, & |y| \le L_{\max} - L_{\text{cache}} \\
    \dfrac{(L_{\max} - L_{\text{cache}}) - |y|}{L_{\text{cache}}}, & L_{\max} - L_{\text{cache}} < |y| \le L_{\max} \\
    -1, & L_{\max} < |y|
    \end{cases}
    $$

    Args:
        max_completion_len (`int`):
            Maximum length of the completion,  \( L_{\max} \).
        soft_punish_cache (`int`):
            Minimum length of the completion,  \( L_{\text{cache}} \). If set to `0`, no minimum length is applied.

    Example:
    ```python
    >>> from trl.rewards import get_soft_overlong_punishment

    >>> soft_overlong_punishment = get_soft_overlong_punishment(max_completion_len=100, soft_punish_cache=20)
    >>> completion_ids = [[1] * 90]  # simulating a completion with 90 tokens. 90 is between 80 and 100.
    >>> soft_overlong_punishment(completion_ids)
    >>> [-0.5]
    ```
    """
    return _SoftOverlongPunishment(max_completion_len, soft_punish_cache)


class _SoftOverlongPunishment:
    # Callable class rather than a closure so the reward stays picklable: the async GRPO rollout
    # worker forwards reward funcs to a spawned child process, and closures can't be pickled.
    # `__name__` mirrors the old inner-function name so metric logging keys are unchanged.
    __name__ = "soft_overlong_punishment_reward"

    def __init__(self, max_completion_len: int, soft_punish_cache: int):
        self.max_completion_len = max_completion_len
        self.soft_punish_cache = soft_punish_cache

    def __call__(self, completion_ids: list[list[int]], **kwargs) -> list[float]:
        rewards = []
        for ids in completion_ids:
            completion_length = len(ids)
            if completion_length <= self.max_completion_len - self.soft_punish_cache:
                rewards.append(0.0)
            elif self.max_completion_len - self.soft_punish_cache < completion_length <= self.max_completion_len:
                rewards.append(
                    (self.max_completion_len - self.soft_punish_cache - completion_length) / self.soft_punish_cache
                )
            else:
                rewards.append(-1.0)
        return rewards
