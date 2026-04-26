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


_RLCR_TAGS = ("think", "answer", "analysis", "confidence")
_RLCR_PATTERN = re.compile(
    r"\A\s*<think>(?P<think>.*?)</think>\s*"
    r"<answer>(?P<answer>.*?)</answer>\s*"
    r"<analysis>(?P<analysis>.*?)</analysis>\s*"
    r"<confidence>(?P<confidence>.*?)</confidence>\s*\Z",
    re.DOTALL,
)


def _has_rlcr_tag(content: str) -> bool:
    return any(f"<{tag}>" in content or f"</{tag}>" in content for tag in _RLCR_TAGS)


def _parse_confidence(content: str) -> float | None:
    try:
        confidence = float(content.strip())
    except ValueError:
        return None

    if not math.isfinite(confidence) or confidence < 0.0 or confidence > 1.0:
        return None

    return confidence


def think_format_reward(completions: list[list[dict[str, str]]], **kwargs) -> list[float]:
    r"""
    Reward function that checks if the reasoning process is enclosed within `"<think>"` and `"</think>"` tags. The
    function returns a reward of 1.0 if the format is correct, otherwise 0.0.

    Args:
        completions (`list[list[dict[str, str]]]`):
            List of completions to be evaluated. Each completion must be a list of one message, i.e. a dictionary
            containing the key `"content"` with the value being the text of the completion.
        **kwargs:
            Additional keyword arguments. This function does not use them, but they are required in the function
            signature to ensure compatibility with trainers like [`GRPOTrainer`].

    Returns:
        `list[float]`:
            A list of rewards, where each reward is 1.0 if the completion matches the expected format, otherwise 0.0.

    Example:
    ```python
    >>> from trl.rewards import think_format_reward

    >>> completions = [
    ...     [{"content": "<think>\nThis is my reasoning.\n</think>\nThis is my answer."}],
    ...     [{"content": "<think>\nThis is my reasoning.\nThis is my answer."}],
    ... ]
    >>> think_format_reward(completions)
    [1.0, 0.0]
    ```
    """
    pattern = r"^<think>(?!.*<think>)(.*?)</think>.*$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


def rlcr_format_reward(completions: list[list[dict[str, str]]], **kwargs) -> list[float]:
    r"""
    Reward function that checks whether completions follow the RLCR response format.

    The expected format is exactly:
    `"<think>...</think><answer>...</answer><analysis>...</analysis><confidence>...</confidence>"`.
    The function returns 1.0 if the format is correct and the confidence is a finite number in `[0.0, 1.0]`, otherwise
    0.0.

    Args:
        completions (`list[list[dict[str, str]]]`):
            List of completions to be evaluated. Each completion must be a list of one message, i.e. a dictionary
            containing the key `"content"` with the value being the text of the completion.
        **kwargs:
            Additional keyword arguments. This function does not use them, but they are required in the function
            signature to ensure compatibility with trainers like [`GRPOTrainer`].

    Returns:
        `list[float]`:
            A list of rewards, where each reward is 1.0 if the completion matches the expected RLCR format, otherwise
            0.0.

    Example:
    ```python
    >>> from trl.rewards import rlcr_format_reward

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
    ...     ],
    ...     [{"content": "<think>Reasoning.</think><answer>42</answer><confidence>not sure</confidence>"}],
    ... ]
    >>> rlcr_format_reward(completions)
    [1.0, 0.0]
    ```
    """
    rewards = []
    for completion in completions:
        content = completion[0]["content"]
        match = _RLCR_PATTERN.fullmatch(content)
        if match is None:
            rewards.append(0.0)
            continue

        sections = match.groupdict()
        if any(_has_rlcr_tag(sections[tag]) for tag in _RLCR_TAGS):
            rewards.append(0.0)
            continue

        rewards.append(1.0 if _parse_confidence(sections["confidence"]) is not None else 0.0)

    return rewards
