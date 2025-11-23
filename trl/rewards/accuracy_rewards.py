# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
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
    Reward function that checks if the completion is the same as the ground truth.
        - If both gold and prediction are parseable → use math verification.
        - If not parseable → compare as normalized text.

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

    >>> solution = [r"\frac{1}{3}", r"\frac{1}{3}"]
    >>> completion = [
    ...     [{"role": "assistant", "content": r"My answer is \boxed{\frac{1}{3}}"}],
    ...     [{"role": "assistant", "content": r"My answer is \boxed{\frac{1}{2}}"}],
    ... ]
    >>> accuracy_reward(completion, solution)
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
            # Compute binary rewards if verifiable, `None` otherwise to skip this example
            reward = float(verify(gold_parsed, answer_parsed))
        else:
            # If the gold solution is not parseable, we assign `None` to skip this example
            reward = float(content.strip().lower() == sol.strip().lower())
        rewards.append(reward)

    return rewards


def _remove_reasoning_content(text: str, tag_pairs: list[tuple[str, str]]) -> str:
    """Removes all reasoning content from text.

    Iteratively removes content between specified start and end tag pairs. This is useful for cleaning model outputs
    that contain reasoning sections that should be excluded from evaluation or computing rewards.

    Args:
        text (str): The input text containing reasoning tags to remove.
        tag_pairs (list[tuple[str, str]]): List of (start_tag, end_tag) pairs to remove.

    Returns:
        str: The text with all reasoning tag content removed.

    Examples:
        >>> text = "<think> Reasoning section </think> Answer section" >>> tag_pairs = [("<think>", "</think>")] >>>
        _remove_reasoning_content(text, tag_pairs) ' Answer section'

        >>> text = "<reasoning>Step 1</reasoning>Answer<reasoning>Step 2</reasoning>" >>> tag_pairs = [("<reasoning>",
        "</reasoning>")] >>> _remove_reasoning_content(text, tag_pairs) 'Answer'
    """
    result = text

    for start_tag, end_tag in tag_pairs:
        while start_tag in result and end_tag in result:
            start = result.find(start_tag)
            end = result.find(end_tag, start)
            if start != -1 and end != -1:
                result = result[:start] + result[end + len(end_tag) :]
            else:
                break

    return result


def reasoning_accuracy_reward(
    completions: list[list[dict[str, str]]],
    solution: list[str],
    reasoning_tags: list[tuple[str, str]] = None,
    **kwargs,
) -> list[float | None]:
    r"""
    Reward function that removes the reasoning content and checks if the completion is the same as the ground truth.
        - If both gold and prediction are parseable → use math verification.
        - If not parseable → compare as normalized text.

    Args:
        completions (`list[list[dict[str, str]]]`):
            List of completions to be evaluated. Each completion must be a list of one message, i.e. a dictionary
            containing the key `"content"` with the value being the text of the completion.
        solution: (`list[str]`):
            List of the raw-text solutions to the questions/problems/prompts.
        reasoning_tags (`list[tuple[str, str]]`, *optional*, defaults to `("<think>", "</think>")`):
            The opening and closing tags that should enclose the reasoning process.
        **kwargs:
            Additional keyword arguments. This function does not use them, but they are required in the function
            signature to ensure compatibility with trainers like [`GRPOTrainer`].
    Example:
        ```python
        >>> from trl.rewards import reasoning_accuracy_reward

        >>> solution = [r"\frac{1}{3}", r"\frac{1}{3}", r"\frac{1}{3}"]
        >>> completion = [
        ...     [
        ...         {
        ...             "role": "assistant",
        ...             "content": r"<think> Reasoning content </think> My answer is \boxed{\frac{1}{3}}",
        ...         }
        ...     ],
        ...     [{"role": "assistant", "content": r"Reasoning content </think> My answer is \boxed{\frac{1}{2}}"}],
        ...     [{"role": "assistant", "content": r"<think> My answer is \boxed{\frac{1}{3}} </think> I don't know."}],
        ... ]
        >>> reasoning_accuracy_reward(completion, solution)
        [1.0, 0.0, 0.0]
        ```
    """
    if not is_math_verify_available():
        raise ImportError("Please install the `math_verify` package to use accuracy_reward")

    if reasoning_tags is None:
        # Use sensible defaults for majority of reasoning models. The second case captures models that prefill the think tag.
        reasoning_tags = [("<think>", "</think>"), ("", "</think>")]
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(contents, solution, strict=True):
        content = _remove_reasoning_content(content, reasoning_tags)
        gold_parsed = parse(
            sol, extraction_config=[LatexExtractionConfig(boxed_match_priority=0, try_extract_without_anchor=True)]
        )
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
            # Compute binary rewards if verifiable, `None` otherwise to skip this example
            try:
                reward = float(verify(gold_parsed, answer_parsed))
            except Exception as _:
                reward = None
        else:
            # If the gold solution is not parseable, we assign `None` to skip this example
            reward = None
        rewards.append(reward)

    return rewards
