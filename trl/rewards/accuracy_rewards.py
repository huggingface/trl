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

from ..import_utils import is_math_verify_available


if is_math_verify_available():
    from latex2sympy2_extended import NormalizationConfig
    from math_verify import LatexExtractionConfig, parse, verify


def accuracy_reward(completions: list[list[dict[str, str]]], solution: list[str], **kwargs) -> list[float | None]:
    r"""
    Reward function that checks if the completion matches the ground truth.
        - If both gold and prediction are parseable → use math verification.
        - If gold is not parseable → return `None` to skip the example.

    Args:
        completions (`list[list[dict[str, str]]]`):
            List of completions to be evaluated. Each completion must be a list of one message, i.e. a dictionary
            containing the key `"content"` with the value being the text of the completion.
        solution: (`list[str]`):
            List of the raw-text solutions to the questions/problems/prompts.
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
    for content, sol in zip(contents, solution, strict=True):
        gold_parsed = parse(sol)
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
            )
            reward = float(verify(gold_parsed, answer_parsed))
        else:
            # If the gold solution cannot be parsed, we assign `None` to skip this example
            reward = None
        rewards.append(reward)

    return rewards


def reasoning_accuracy_reward(
    completions: list[list[dict[str, str]]],
    solution: list[str],
    reasoning_delimiters: list[str] | None = None,
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
        solution: (`list[str]`):
            List of the raw-text solutions to the questions/problems/prompts.
        reasoning_delimiters (`list[str]]`, *optional*):
            List of strings indicating where the reasoning content ends. The final answer is assumed to be after the
            last occurrence of any of these delimiters. If `None`, defaults to `["</think>"]`.
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
            continue

        gold_parsed = parse(sol)
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
            )
            reward = float(verify(gold_parsed, answer_parsed))
        else:
            # If the gold solution cannot be parsed, we assign `None` to skip this example
            reward = None
        rewards.append(reward)

    return rewards
