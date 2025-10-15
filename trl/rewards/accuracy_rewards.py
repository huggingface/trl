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

from typing import Optional

from trl.import_utils import is_math_verify_available


if is_math_verify_available():
    from latex2sympy2_extended import NormalizationConfig
    from math_verify import LatexExtractionConfig, parse, verify


def accuracy_reward(completions: list[list[dict[str, str]]], solution: list[str], **kwargs) -> list[Optional[float]]:
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
    for content, sol in zip(contents, solution):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
        )
        if len(gold_parsed) != 0:
            # We require the answer to be provided in correct latex (no malformed operators)
            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            boxed="all",
                            units=True,
                        ),
                        # Ensures that boxed is tried first
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
            # Compute binary rewards if verifiable, `None` otherwise to skip this example
            try:
                reward = float(verify(gold_parsed, answer_parsed))
            except Exception:
                reward = None
        else:
            # If the gold solution is not parseable, we assign `None` to skip this example
            reward = float(content.strip().lower() == sol.strip().lower())
        rewards.append(reward)

    return rewards
