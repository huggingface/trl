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

import re


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


def graduated_format_reward(completions: list[list[dict[str, str]]], **kwargs) -> list[float]:
    r"""
    Graduated version of [`think_format_reward`] that assigns partial credit for partial structure, so the signal is
    denser during cold-start training when strict-format matches are rare.

    Reward tiers:

    - `1.0` if the completion matches the strict pattern of [`think_format_reward`].
    - `0.5` if both `<think>` and `</think>` are present but the completion does not match the strict pattern (e.g.
      nested tags or closer before opener).
    - `0.25` if exactly one of `<think>` and `</think>` is present.
    - `0.0` if neither tag is present.

    Useful when [`think_format_reward`] is too sparse during early training: a model that produces only the opening
    `<think>` tag, or both tags in the wrong order, still receives a small positive signal pulling it toward the
    target structure.

    Args:
        completions (`list[list[dict[str, str]]]`):
            List of completions to be evaluated. Each completion must be a list of one message, i.e. a dictionary
            containing the key `"content"` with the value being the text of the completion.
        **kwargs:
            Additional keyword arguments. This function does not use them, but they are required in the function
            signature to ensure compatibility with trainers like [`GRPOTrainer`].

    Returns:
        `list[float]`:
            A list of rewards in `{0.0, 0.25, 0.5, 1.0}`, one per completion.

    Example:

    ```python
    >>> from trl.rewards import graduated_format_reward

    >>> completions = [
    ...     [{"content": "<think>\nReasoning.\n</think>\nAnswer."}],  # Strict match
    ...     [{"content": "<think>nested <think>reasoning</think></think> answer"}],  # Both tags, malformed
    ...     [{"content": "<think>unclosed reasoning"}],  # Opener only
    ...     [{"content": "No tags here."}],  # Nothing
    ... ]
    >>> graduated_format_reward(completions)
    [1.0, 0.5, 0.25, 0.0]
    ```
    """
    strict_pattern = r"^<think>(?!.*<think>)(.*?)</think>.*$"
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content in contents:
        has_open = "<think>" in content
        has_close = "</think>" in content
        if has_open and has_close:
            if re.match(strict_pattern, content, re.DOTALL | re.MULTILINE):
                rewards.append(1.0)
            else:
                rewards.append(0.5)
        elif has_open or has_close:
            rewards.append(0.25)
        else:
            rewards.append(0.0)
    return rewards
