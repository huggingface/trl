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

import math
import re
from collections.abc import Callable

from .accuracy_rewards import accuracy_reward
from .format_rewards import rlcr_format_reward


def _extract_last_tag(content: str, tag: str) -> str | None:
    matches = re.findall(rf"<{tag}>(.*?)</{tag}>", content, re.DOTALL)
    return matches[-1].strip() if matches else None


def _parse_confidence(value: str | None) -> float | None:
    if value is None:
        return None

    try:
        confidence = float(value.strip())
    except ValueError:
        return None

    if not math.isfinite(confidence) or confidence < 0.0 or confidence > 1.0:
        return None

    return confidence


def _validate_correctness_rewards(rewards: list[float | None], expected_length: int) -> list[float | None]:
    if len(rewards) != expected_length:
        raise ValueError(
            f"`correctness_reward_func` must return one reward per completion. Expected {expected_length} rewards, "
            f"but got {len(rewards)}."
        )

    validated_rewards = []
    for reward in rewards:
        if reward is None:
            validated_rewards.append(None)
            continue

        try:
            reward = float(reward)
        except (TypeError, ValueError) as exc:
            raise ValueError("`correctness_reward_func` must return floats in [0.0, 1.0] or None.") from exc

        if not math.isfinite(reward) or reward < 0.0 or reward > 1.0:
            raise ValueError("`correctness_reward_func` must return floats in [0.0, 1.0] or None.")

        validated_rewards.append(reward)

    return validated_rewards


def brier_score_reward(
    completions: list[list[dict[str, str]]],
    solution: list[str],
    correctness_reward_func: Callable | None = None,
    log_extra: Callable[[str, list], None] | None = None,
    **kwargs,
) -> list[float | None]:
    r"""
    Reward function that scores how well a model's confidence matches answer correctness.

    This function checks the RLCR format, extracts the last `"<answer>...</answer>"` and
    `"<confidence>...</confidence>"` blocks from each completion, computes correctness on the extracted answer, then
    returns `1 - (confidence - correctness) ** 2`. This is a calibration reward, not a complete RLCR objective by
    itself. For RLCR-style training, combine it with an answer-correctness reward such as [`accuracy_reward`] and a
    format reward such as [`rlcr_format_reward`].

    Args:
        completions (`list[list[dict[str, str]]]`):
            List of completions to be evaluated. Each completion must be a list of one message, i.e. a dictionary
            containing the key `"content"` with the value being the text of the completion.
        solution (`list[str]`):
            List of the raw-text solutions to the questions/problems/prompts.
        correctness_reward_func (`Callable`, *optional*):
            Function used to score the extracted answers. If `None`, defaults to [`accuracy_reward`]. The function must
            return one float in `[0.0, 1.0]` or `None` per completion.
        log_extra (`Callable`, *optional*):
            Callable to log extra columns to the completions table, provided automatically by the trainer. Defaults to
            `None` to allow calling the function directly outside of a trainer (e.g., for testing).
        **kwargs:
            Additional keyword arguments. Dataset columns are forwarded to `correctness_reward_func`; trainer-only
            values such as `trainer_state`, `log_extra`, `log_metric`, and `completion_ids` are not forwarded.

    Returns:
        `list[float | None]`:
            A list of calibration rewards. Invalid RLCR formatting receives `0.0`. If the correctness function returns
            `None`, `None` is propagated for that completion.

    Example:
    ```python
    >>> from trl.rewards import accuracy_reward, brier_score_reward, rlcr_format_reward

    >>> completions = [
    ...     [
    ...         {
    ...             "content": (
    ...                 "<think>Reasoning.</think>"
    ...                 "<answer>42</answer>"
    ...                 "<analysis>Direct arithmetic.</analysis>"
    ...                 "<confidence>0.9</confidence>"
    ...             )
    ...         }
    ...     ]
    ... ]
    >>> solution = ["42"]
    >>> brier_score_reward(completions, solution, correctness_reward_func=lambda **kwargs: [1.0])
    [0.99]
    ```
    """
    if correctness_reward_func is None:
        correctness_reward_func = accuracy_reward

    contents = [completion[0]["content"] for completion in completions]
    answers = [_extract_last_tag(content, "answer") for content in contents]
    confidences = [_parse_confidence(_extract_last_tag(content, "confidence")) for content in contents]
    format_rewards = rlcr_format_reward(completions)
    invalid_mask = [
        format_reward == 0.0 or answer is None or confidence is None
        for format_reward, answer, confidence in zip(format_rewards, answers, confidences, strict=True)
    ]

    rewards: list[float | None] = [0.0] * len(completions)
    correctness_rewards: list[float | None] = [None] * len(completions)

    if not all(invalid_mask):
        answer_completions = [[{"content": answer or ""}] for answer in answers]
        correctness_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key not in ("trainer_state", "log_extra", "log_metric", "completion_ids")
        }
        raw_correctness_rewards = correctness_reward_func(
            completions=answer_completions, solution=solution, **correctness_kwargs
        )
        correctness_rewards = _validate_correctness_rewards(raw_correctness_rewards, len(completions))

        for i, (confidence, correctness_reward, is_invalid) in enumerate(
            zip(confidences, correctness_rewards, invalid_mask, strict=True)
        ):
            if is_invalid:
                continue
            if correctness_reward is None:
                rewards[i] = None
                continue
            rewards[i] = 1.0 - (confidence - correctness_reward) ** 2

    if log_extra is not None:
        log_extra("rlcr_answer", answers)
        log_extra("rlcr_confidence", confidences)
        log_extra("rlcr_correctness", correctness_rewards)
        log_extra("brier_score_reward", rewards)

    return rewards
