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
