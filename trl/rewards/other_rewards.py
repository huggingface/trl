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
    from trl.rewards import get_soft_overlong_punishment

    soft_overlong_punishment = get_soft_overlong_punishment(max_completion_len=100, soft_punish_cache=20)
    completion_ids = [[1] * 90]  # simulating a completion with 90 tokens. 90 is between 80 and 100.
    rewards = soft_overlong_punishment(completion_ids)
    print(rewards)  # [-0.5]
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
