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

import logging
import math
import statistics
import threading
from collections import defaultdict
from collections.abc import Callable

from ..import_utils import is_math_verify_available


if is_math_verify_available():
    from latex2sympy2_extended import NormalizationConfig
    from math_verify import LatexExtractionConfig, parse, verify


def accuracy_reward(
    completions: list[list[dict[str, str]]],
    solution: list[str],
    log_extra: Callable[[str, list], None] | None = None,
    **kwargs,
) -> list[float | None]:
    r"""
    Reward function that checks if the completion matches the ground truth.
        - If both gold and prediction are parseable → use math verification.
        - If gold is not parseable → return `None` to skip the example.

    Args:
        completions (`list[list[dict[str, str]]]`):
            List of completions to be evaluated. Each completion must be a list of one message, i.e. a dictionary
            containing the key `"content"` with the value being the text of the completion.
        solution (`list[str]`):
            List of the raw-text solutions to the questions/problems/prompts.
        log_extra (`callable`, *optional*):
            Callable to log extra columns to the completions table, provided automatically by the trainer. Defaults to
            `None` to allow calling the function directly outside of a trainer (e.g., for testing).
        **kwargs:
            Additional keyword arguments. This function does not use them, but they are required in the function
            signature to ensure compatibility with trainers like [`GRPOTrainer`].
    Example:
    ```python
    >>> from trl.rewards import accuracy_reward

    >>> solutions = [r"\frac{1}{3}", r"\frac{1}{3}"]
    >>> completions = [
    ...     [{"role": "assistant", "content": r"My answer is \boxed{\frac{1}{3}}"}],
    ...     [{"role": "assistant", "content": r"My answer is \boxed{\frac{1}{2}}"}],
    ... ]
    >>> accuracy_reward(completions, solutions)
    [1.0, 0.0]
    ```
    """
    if not is_math_verify_available():
        raise ImportError("Please install the `math_verify` package to use accuracy_reward")

    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    gold_parsed_strs = []
    answer_parsed_strs = []

    # math_verify uses signal.alarm() for timeouts, which only works in the main thread.
    # Disable timeouts when running in a non-main thread to avoid ValueError.
    is_main_thread = threading.current_thread() is threading.main_thread()
    parsing_timeout = None if not is_main_thread else 10
    verify_timeout = None if not is_main_thread else 5

    # Suppress the "Timeout is disabled" warnings from math_verify when we intentionally disable timeouts
    if not is_main_thread:
        logging.getLogger("math_verify.parser").setLevel(logging.ERROR)
        logging.getLogger("math_verify.grader").setLevel(logging.ERROR)

    for content, sol in zip(contents, solution, strict=True):
        gold_parsed = parse(sol, parsing_timeout=parsing_timeout)
        if len(gold_parsed) != 0:
            # We require the answer to be provided in correct latex (no malformed operators)
            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(units=True),
                        # Ensures that boxed is tried first
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
                parsing_timeout=parsing_timeout,
            )
            reward = float(verify(gold_parsed, answer_parsed, timeout_seconds=verify_timeout))
            gold_parsed_strs.append(str(gold_parsed))
            answer_parsed_strs.append(str(answer_parsed) if answer_parsed else "[unparseable]")
        else:
            # If the gold solution cannot be parsed, we assign `None` to skip this example
            reward = None
            gold_parsed_strs.append("[unparseable]")
            answer_parsed_strs.append("[skipped]")
        rewards.append(reward)

    if log_extra is not None:
        log_extra("solution", list(solution))
        log_extra("gold_parsed", gold_parsed_strs)
        log_extra("answer_parsed", answer_parsed_strs)

    return rewards


def get_cosine_scaled_reward(
    max_len: int,
    min_value_wrong: float = -1.0,
    max_value_wrong: float = -0.5,
    min_value_correct: float = 0.5,
    max_value_correct: float = 1.0,
) -> Callable:
    # docstyle-ignore
    r"""
    Reward function that scales a correctness reward by the completion length following a cosine schedule, to favor
    concise reasoning. Reference: Appendix C.1 of the "Demystifying Long Chain-of-Thought Reasoning" paper
    (https://huggingface.co/papers/2502.03373).

    Correctness is determined by math verification (as in [`~rewards.accuracy_reward`]), and the length is the number
    of completion tokens. The reward interpolates along a cosine schedule between a short-completion and a
    long-completion bound:

    $$
    R_{\text{cosine}}(y) = v_{\min} + \frac{1}{2}(v_{\max} - v_{\min})\left(1 + \cos\left(\frac{|y|}{L_{\max}}\pi\right)\right)
    $$

    For a **correct** completion, $(v_{\min}, v_{\max}) = (\texttt{min\_value\_correct}, \texttt{max\_value\_correct})$,
    so a shorter completion is rewarded more. For a **wrong** completion, the bounds are swapped to
    $(v_{\min}, v_{\max}) = (\texttt{max\_value\_wrong}, \texttt{min\_value\_wrong})$, so a shorter completion is
    penalized more (a longer wrong completion is penalized less, preserving exploration). When the gold solution is not
    parseable, the example is skipped (reward `None`), as in [`~rewards.accuracy_reward`].

    Args:
        max_len (`int`):
            Maximum completion length (in tokens) used to normalize the cosine schedule, $L_{\max}$.
        min_value_wrong (`float`, *optional*, defaults to `-1.0`):
            Reward of a wrong completion at the shortest length.
        max_value_wrong (`float`, *optional*, defaults to `-0.5`):
            Reward of a wrong completion at the longest length.
        min_value_correct (`float`, *optional*, defaults to `0.5`):
            Reward of a correct completion at the longest length.
        max_value_correct (`float`, *optional*, defaults to `1.0`):
            Reward of a correct completion at the shortest length.

    Returns:
        `Callable`:
            A reward function that takes completions, their solutions and token ids, and returns a list of rewards
            (`None` for examples with an unparseable gold solution).

    Example:
    ```python
    >>> from trl.rewards import get_cosine_scaled_reward

    >>> cosine_scaled_reward = get_cosine_scaled_reward(max_len=100)
    >>> completions = [[{"content": r"\boxed{\frac{1}{3}}"}], [{"content": r"\boxed{\frac{1}{2}}"}]]
    >>> solution = [r"\frac{1}{3}", r"\frac{1}{3}"]
    >>> completion_ids = [[1] * 50, [1] * 50]  # both completions are 50 tokens, half of max_len
    >>> cosine_scaled_reward(completions, solution, completion_ids)
    [0.75, -0.75]
    ```
    """
    return _CosineScaledReward(max_len, min_value_wrong, max_value_wrong, min_value_correct, max_value_correct)


class _CosineScaledReward:
    # Callable class rather than a closure so the reward stays picklable: the async GRPO rollout
    # worker forwards reward funcs to a spawned child process, and closures can't be pickled.
    __name__ = "cosine_scaled_reward"

    def __init__(
        self,
        max_len: int,
        min_value_wrong: float,
        max_value_wrong: float,
        min_value_correct: float,
        max_value_correct: float,
    ):
        self.max_len = max_len
        self.min_value_wrong = min_value_wrong
        self.max_value_wrong = max_value_wrong
        self.min_value_correct = min_value_correct
        self.max_value_correct = max_value_correct

    def __call__(
        self,
        completions: list[list[dict[str, str]]],
        solution: list[str],
        completion_ids: list[list[int]],
        **kwargs,
    ) -> list[float | None]:
        is_correct = accuracy_reward(completions, solution)
        rewards = []
        for correct, ids in zip(is_correct, completion_ids, strict=True):
            if correct is None:
                # Gold solution was not parseable; skip the example, as in accuracy_reward.
                rewards.append(None)
                continue
            # Clamp to 1.0 so completions longer than max_len stay at the long-length bound: cos is periodic, so
            # without clamping the schedule would climb back up past max_len and reward very long completions.
            progress = min(len(ids) / self.max_len, 1.0)
            cosine = math.cos(progress * math.pi)
            if correct:
                min_value, max_value = self.min_value_correct, self.max_value_correct
            else:
                # Swap the bounds so that a shorter wrong completion is penalized more than a longer one.
                min_value, max_value = self.max_value_wrong, self.min_value_wrong
            rewards.append(float(min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)))
        return rewards


def reasoning_accuracy_reward(
    completions: list[list[dict[str, str]]],
    solution: list[str],
    reasoning_delimiters: list[str] | None = None,
    log_extra: Callable[[str, list], None] | None = None,
    **kwargs,
) -> list[float | None]:
    r"""
    Reward function that removes the reasoning content and checks if the final answer matches the ground truth.
        - If both gold and prediction are parseable → use math verification.
        - If gold is not parseable → return `None` to skip the example.

    Args:
        completions (`list[list[dict[str, str]]]`):
            List of completions to be evaluated. Each completion must be a list of one message, i.e. a dictionary
            containing the key `"content"` with the value being the text of the completion.
        solution (`list[str]`):
            List of the raw-text solutions to the questions/problems/prompts.
        reasoning_delimiters (`list[str]]`, *optional*):
            List of strings indicating where the reasoning content ends. The final answer is assumed to be after the
            last occurrence of any of these delimiters. If `None`, defaults to `["</think>"]`.
        log_extra (`callable`, *optional*):
            Callable to log extra columns to the completions table, provided automatically by the trainer. Defaults to
            `None` to allow calling the function directly outside of a trainer (e.g., for testing).
        **kwargs:
            Additional keyword arguments. This function does not use them, but they are required in the function
            signature to ensure compatibility with trainers like [`GRPOTrainer`].
    Example:
        ```python
        >>> from trl.rewards import reasoning_accuracy_reward

        >>> reasoning_delimiters = ["</think>"]
        >>> solutions = [r"\frac{1}{3}", r"\frac{1}{3}", r"\frac{1}{3}"]
        >>> completions = [
        ...     [
        ...         {
        ...             "role": "assistant",
        ...             "content": r"<think> Reasoning content </think> The final answer is \boxed{\frac{1}{3}}",
        ...         }
        ...     ],
        ...     [
        ...         {
        ...             "role": "assistant",
        ...             "content": r"<think> Reasoning content </think> The final answer is \boxed{\frac{1}{2}}",
        ...         }
        ...     ],
        ...     [
        ...         {
        ...             "role": "assistant",
        ...             "content": r"<think> Reasoning content with partial answers \boxed{\frac{1}{3}} but no final answer",
        ...         }
        ...     ],
        ... ]
        >>> reasoning_accuracy_reward(completions, solutions, reasoning_delimiters=reasoning_delimiters)
        [1.0, 0.0, 0.0]
        ```
    """
    if not is_math_verify_available():
        raise ImportError("Please install the `math_verify` package to use reasoning_accuracy_reward")

    if reasoning_delimiters is None:
        # Use sensible defaults for majority of reasoning models
        reasoning_delimiters = ["</think>"]

    rewards = []
    contents = [completion[0]["content"] for completion in completions]
    gold_parsed_strs = []
    answer_parsed_strs = []

    # math_verify uses signal.alarm() for timeouts, which only works in the main thread.
    # Disable timeouts when running in a non-main thread to avoid ValueError.
    is_main_thread = threading.current_thread() is threading.main_thread()
    parsing_timeout = None if not is_main_thread else 10
    verify_timeout = None if not is_main_thread else 5

    # Suppress the "Timeout is disabled" warnings from math_verify when we intentionally disable timeouts
    if not is_main_thread:
        logging.getLogger("math_verify.parser").setLevel(logging.ERROR)
        logging.getLogger("math_verify.grader").setLevel(logging.ERROR)

    for content, sol in zip(contents, solution, strict=True):
        # Split final answer from reasoning content
        is_reasoning_complete = False
        for delim in reasoning_delimiters:
            if delim in content:
                content = content.split(delim)[-1]
                is_reasoning_complete = True
                break
        if not is_reasoning_complete:
            # We assign zero reward instead of `None` to penalize incomplete reasoning
            rewards.append(0.0)
            gold_parsed_strs.append("[incomplete reasoning]")
            answer_parsed_strs.append("[incomplete reasoning]")
            continue

        gold_parsed = parse(sol, parsing_timeout=parsing_timeout)
        if len(gold_parsed) != 0:
            # We require the answer to be provided in correct latex (no malformed operators)
            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        boxed_match_priority=0,
                        normalization_config=NormalizationConfig(
                            units=True,
                        ),
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
                parsing_timeout=parsing_timeout,
            )
            reward = float(verify(gold_parsed, answer_parsed, timeout_seconds=verify_timeout))
            gold_parsed_strs.append(str(gold_parsed))
            answer_parsed_strs.append(str(answer_parsed) if answer_parsed else "[unparseable]")
        else:
            # If the gold solution cannot be parsed, we assign `None` to skip this example
            reward = None
            gold_parsed_strs.append("[unparseable]")
            answer_parsed_strs.append("[skipped]")
        rewards.append(reward)

    if log_extra is not None:
        log_extra("solution", list(solution))
        log_extra("gold_parsed", gold_parsed_strs)
        log_extra("answer_parsed", answer_parsed_strs)

    return rewards


def get_length_scaled_accuracy_reward(
    alpha: float = 0.5,
    incorrect_reward: float = -1.0,
    reasoning_delimiters: list[str] | None = None,
) -> Callable:
    r"""
    Reward function factory that scales [`~rewards.reasoning_accuracy_reward`] by completion length to discourage
    overthinking. Reference: GRPO-LEAD (https://huggingface.co/papers/2504.09696).

    Within each group of completions sharing the same prompt, the length (in tokens if `completion_ids` is available,
    otherwise characters) of every *correct* completion is standardized to a z-score `z`. The reward for a correct
    completion is then `exp(-alpha * z)`, so shorter-than-average correct completions receive a larger reward than
    longer-than-average ones (the rewards of a group's correct completions have geometric mean `1.0`). Incorrect
    completions receive `incorrect_reward` (default `-1.0`), shifting the break-even above zero so that guessing only
    pays off once the model is confident. Unparseable gold solutions are passed through as `None`.

    Unlike [`~rewards.get_cosine_scaled_reward`], which maps an *absolute* token length onto a fixed cosine schedule
    via a `max_len` hyperparameter, this reward is *group-relative*: it standardizes length against the other correct
    completions of the same prompt, so it adapts to each prompt's difficulty without a length budget.

    Args:
        alpha (`float`, *optional*, defaults to `0.5`):
            Strength of the length modulation for correct completions. Larger values give shorter correct completions
            disproportionately more reward. `alpha=0` disables scaling and recovers a `{-1, +1}` accuracy reward.
        incorrect_reward (`float`, *optional*, defaults to `-1.0`):
            Reward assigned to incorrect completions (including completions with incomplete reasoning).
        reasoning_delimiters (`list[str]`, *optional*):
            Forwarded to [`~rewards.reasoning_accuracy_reward`]. List of strings marking the end of the reasoning
            block; the final answer is assumed to follow the last occurrence. If `None`, defaults to `["</think>"]`.

    Returns:
        `Callable`:
            A reward function with the signature expected by [`GRPOTrainer`].

    Example:

    ```python
    >>> from trl.rewards import get_length_scaled_accuracy_reward

    >>> reward_fn = get_length_scaled_accuracy_reward(alpha=0.5)
    >>> solution = [r"\frac{1}{3}", r"\frac{1}{3}", r"\frac{1}{3}"]
    >>> completions = [
    ...     [{"role": "assistant", "content": r"<think> short </think> \boxed{\frac{1}{3}}"}],
    ...     [{"role": "assistant", "content": r"<think> long reasoning... </think> \boxed{\frac{1}{3}}"}],
    ...     [{"role": "assistant", "content": r"<think> reasoning </think> \boxed{\frac{1}{2}}"}],
    ... ]
    >>> prompts = ["same prompt"] * 3
    >>> rewards = reward_fn(completions=completions, solution=solution, prompts=prompts)
    ```
    """
    return _LengthScaledAccuracyReward(alpha, incorrect_reward, reasoning_delimiters)


class _LengthScaledAccuracyReward:
    # Callable class rather than a closure so the reward stays picklable: the async GRPO rollout
    # worker forwards reward funcs to a spawned child process, and closures can't be pickled.
    __name__ = "length_scaled_accuracy_reward"

    def __init__(self, alpha: float, incorrect_reward: float, reasoning_delimiters: list[str] | None):
        self.alpha = alpha
        self.incorrect_reward = incorrect_reward
        self.reasoning_delimiters = reasoning_delimiters

    def __call__(
        self,
        completions: list[list[dict[str, str]]],
        solution: list[str],
        prompts: list,
        completion_ids: list[list[int]] | None = None,
        log_extra: Callable[[str, list], None] | None = None,
        **kwargs,
    ) -> list[float | None]:
        base_rewards = reasoning_accuracy_reward(
            completions=completions,
            solution=solution,
            reasoning_delimiters=self.reasoning_delimiters,
            log_extra=log_extra,
        )

        if completion_ids is not None:
            lengths = [len(ids) for ids in completion_ids]
        else:
            lengths = [len(completion[0]["content"]) for completion in completions]

        groups: dict[str, list[int]] = defaultdict(list)
        for idx, prompt in enumerate(prompts):
            groups[str(prompt)].append(idx)

        rewards: list[float | None] = [None] * len(completions)
        length_z_scores: list[float | None] = [None] * len(completions)

        for indices in groups.values():
            correct_indices = [i for i in indices if base_rewards[i] == 1.0]
            if len(correct_indices) >= 2:
                correct_lengths = [lengths[i] for i in correct_indices]
                mean = statistics.mean(correct_lengths)
                stdev = statistics.stdev(correct_lengths)
            else:
                mean, stdev = 0.0, 0.0

            for i in indices:
                base = base_rewards[i]
                if base is None:
                    rewards[i] = None
                elif base == 1.0:
                    if len(correct_indices) >= 2 and stdev > 0:
                        z = (lengths[i] - mean) / stdev
                        length_z_scores[i] = z
                        rewards[i] = math.exp(-self.alpha * z)
                    else:
                        length_z_scores[i] = 0.0
                        rewards[i] = 1.0
                else:
                    rewards[i] = self.incorrect_reward

        if log_extra is not None:
            log_extra("length_z", length_z_scores)

        return rewards
