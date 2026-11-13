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

import json
import logging
import math
import os
import queue
import subprocess
import sys
import threading
from collections.abc import Callable

from ..import_utils import is_math_verify_available


if is_math_verify_available():
    from latex2sympy2_extended import NormalizationConfig
    from math_verify import LatexExtractionConfig, parse, verify


# Seconds one answer may take to grade before the grading process is killed and the answer scores 0.
GRADE_TIMEOUT = 30


def _grade(solution: str, content: str) -> tuple[float | None, str, str]:
    """Parse and verify one answer against its solution, returning the reward and the parsed strings to log."""
    gold_parsed = parse(solution, parsing_timeout=10)
    if len(gold_parsed) == 0:
        # If the gold solution cannot be parsed, we assign `None` to skip this example
        return None, "[unparseable]", "[skipped]"
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
        parsing_timeout=10,
    )
    reward = float(verify(gold_parsed, answer_parsed, timeout_seconds=5))
    return reward, str(gold_parsed), str(answer_parsed) if answer_parsed else "[unparseable]"


def _grading_loop() -> None:
    """Grade the `[solution, content]` lines read from stdin, one result line per answer on stdout."""
    for line in sys.stdin:
        solution, content = json.loads(line)
        try:
            result = {"result": _grade(solution, content)}
        except Exception as e:
            result = {"error": repr(e)}
        sys.stdout.write("GRADE " + json.dumps(result) + "\n")
        sys.stdout.flush()


class _Grader:
    """
    Grades answers in a separate process. math-verify's timeouts use `signal.alarm()`, which neither works outside the
    main thread nor interrupts sympy arithmetic running in C: an answer like `5^{5^{5^{5^5}}}` makes `parse` or
    `verify` run forever (huggingface/Math-Verify#65). In the grading process they run on the main thread, so the
    timeouts work for the slow cases they can stop, and a process stuck past `GRADE_TIMEOUT` is killed and restarted.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._process = None
        self._pid = None

    def _start(self):
        self._process = subprocess.Popen(
            [sys.executable, "-c", "from trl.rewards.accuracy_rewards import _grading_loop; _grading_loop()"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._pid = os.getpid()
        self._lines = queue.Queue()

        def read_results(stdout, lines):
            # Libraries may print to stdout, so only the lines starting with "GRADE " are results
            for line in stdout:
                if line.startswith("GRADE "):
                    lines.put(line.removeprefix("GRADE "))

        threading.Thread(target=read_results, args=(self._process.stdout, self._lines), daemon=True).start()

    def grade(self, solution: str, content: str) -> tuple[float | None, str, str]:
        with self._lock:
            # A forked trainer process can't share its parent's pipes, so it starts its own grading process
            if self._process is None or self._process.poll() is not None or self._pid != os.getpid():
                self._start()
            request = json.dumps([solution, content]) + "\n"
            try:
                self._process.stdin.write(request)
                self._process.stdin.flush()
            except OSError:
                # The grading process died since the check above (e.g. out of memory), so start a new one
                self._start()
                self._process.stdin.write(request)
                self._process.stdin.flush()
            try:
                line = self._lines.get(timeout=GRADE_TIMEOUT)
            except queue.Empty:
                self._process.kill()
                self._process = None
                logging.getLogger(__name__).warning(
                    "Grading an answer took more than %s s, so it scores 0 and the grading process was restarted.",
                    GRADE_TIMEOUT,
                )
                return 0.0, "[timeout]", "[timeout]"
        output = json.loads(line)
        if "error" in output:
            raise RuntimeError(f"Grading failed: {output['error']}")
        return tuple(output["result"])


_grader = _Grader()


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
        raise ImportError(
            "`accuracy_reward` requires the `math-verify` library. Install it with `pip install trl[math_verify]`."
        )

    rewards = []
    gold_parsed_strs = []
    answer_parsed_strs = []
    for completion, sol in zip(completions, solution, strict=True):
        reward, gold_parsed_str, answer_parsed_str = _grader.grade(sol, completion[0]["content"])
        rewards.append(reward)
        gold_parsed_strs.append(gold_parsed_str)
        answer_parsed_strs.append(answer_parsed_str)

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
        raise ImportError(
            "`reasoning_accuracy_reward` requires the `math-verify` library. Install it with `pip install trl[math_verify]`."
        )

    if reasoning_delimiters is None:
        # Use sensible defaults for majority of reasoning models
        reasoning_delimiters = ["</think>"]

    rewards = []
    contents = [completion[0]["content"] for completion in completions]
    gold_parsed_strs = []
    answer_parsed_strs = []

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

        reward, gold_parsed_str, answer_parsed_str = _grader.grade(sol, content)
        gold_parsed_strs.append(gold_parsed_str)
        answer_parsed_strs.append(answer_parsed_str)
        rewards.append(reward)

    if log_extra is not None:
        log_extra("solution", list(solution))
        log_extra("gold_parsed", gold_parsed_strs)
        log_extra("answer_parsed", answer_parsed_strs)

    return rewards
