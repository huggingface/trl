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

import inspect
from typing import Any, Callable, List, Optional, Union

import torch


def conditioned_reward(
    primary_reward_func: Callable,
    secondary_reward_func: Callable,
    condition: float = 1.0,
    primary_reward_name: Optional[str] = None,
) -> Callable:
    """
    Creates a conditioned reward function that only applies the secondary reward if the primary reward meets a threshold.

    This utility enforces a hierarchy where secondary rewards are only granted if a primary reward threshold is met.
    This corresponds to the Eq. 8 of the GDPO paper (https://huggingface.co/papers/2601.05242).

    Args:
        primary_reward_func (`Callable`):
            The primary reward function that determines if the condition is met.
        secondary_reward_func (`Callable`):
            The secondary reward function to be applied if the condition is met.
        condition (`float`, *optional*, defaults to 1.0):
            The threshold value for the primary reward to enable the secondary reward.
        primary_reward_name (`str`, *optional*):
            The name of the primary reward function. Used to query the context for cached results.
            If None, it defaults to `primary_reward_func.__name__`.

    Returns:
        `Callable`:
            A new reward function that wraps the primary and secondary rewards.
            If either input function is asynchronous, the returned function will be asynchronous.
    """
    if primary_reward_name is None:
        primary_reward_name = getattr(primary_reward_func, "__name__", None)

    is_async = inspect.iscoroutinefunction(primary_reward_func) or inspect.iscoroutinefunction(secondary_reward_func)

    def _convert_to_tensor(data, device):
        if isinstance(data, torch.Tensor):
            return data.to(device)
        return torch.tensor([x if x is not None else float("nan") for x in data], device=device, dtype=torch.float32)

    def _process_rewards(primary_rewards, secondary_rewards):
        # Check for tensors or mixed interactions
        if isinstance(primary_rewards, torch.Tensor) or isinstance(secondary_rewards, torch.Tensor):
            # Resolve device: prefer primary if tensor, else secondary
            device = (
                primary_rewards.device
                if isinstance(primary_rewards, torch.Tensor)
                else secondary_rewards.device
            )

            primary_tensor = _convert_to_tensor(primary_rewards, device)
            secondary_tensor = _convert_to_tensor(secondary_rewards, device)

            # Logic:
            # 1. If primary is NaN (was None), result is NaN
            # 2. If primary >= condition, result is secondary
            # 3. Else, result is 0.0

            # Start with 0.0
            results = torch.zeros_like(primary_tensor)

            # Apply secondary where condition is met
            mask_cond = primary_tensor >= condition
            results = torch.where(mask_cond, secondary_tensor, results)

            # Propagate NaNs (Nones)
            mask_nan = torch.isnan(primary_tensor)
            results = torch.where(mask_nan, float("nan"), results)

            return results

        final_rewards = []
        for r_p, r_s in zip(primary_rewards, secondary_rewards):
            if r_p is None:
                final_rewards.append(None)
            elif r_p >= condition:
                final_rewards.append(r_s)
            else:
                final_rewards.append(0.0)
        return final_rewards

    if is_async:

        async def conditioned_reward_func(prompts, completions, **kwargs) -> Union[List[Optional[float]], torch.Tensor]:
            # Try to get primary reward from Context (Avoid Redundancy)
            context = kwargs.get("context")
            primary_rewards = None
            if context is not None and primary_reward_name in context:
                primary_rewards = context[primary_reward_name]

            # Execute primary reward if not found in context
            if primary_rewards is None:
                if inspect.iscoroutinefunction(primary_reward_func):
                    primary_rewards = await primary_reward_func(prompts, completions, **kwargs)
                else:
                    primary_rewards = primary_reward_func(prompts, completions, **kwargs)

            # Execute secondary reward
            if inspect.iscoroutinefunction(secondary_reward_func):
                secondary_rewards = await secondary_reward_func(prompts, completions, **kwargs)
            else:
                secondary_rewards = secondary_reward_func(prompts, completions, **kwargs)

            return _process_rewards(primary_rewards, secondary_rewards)

    else:

        def conditioned_reward_func(prompts, completions, **kwargs) -> Union[List[Optional[float]], torch.Tensor]:
            # Try to get primary reward from Context (Avoid Redundancy)
            context = kwargs.get("context")
            primary_rewards = None
            if context is not None and primary_reward_name in context:
                primary_rewards = context[primary_reward_name]

            if primary_rewards is None:
                primary_rewards = primary_reward_func(prompts, completions, **kwargs)
            secondary_rewards = secondary_reward_func(prompts, completions, **kwargs)

            return _process_rewards(primary_rewards, secondary_rewards)

    return conditioned_reward_func
