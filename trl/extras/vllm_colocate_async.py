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

"""
Async utilities for vLLM colocate mode.

This module provides async wrappers around vLLM's synchronous LLM class for use in async/await contexts without
blocking the event loop.
"""

import asyncio
from typing import Any, Optional


class AsyncVLLMColocateWrapper:
    """
    Async wrapper for vLLM colocate mode.

    This wrapper makes vLLM's synchronous `generate()` method non-blocking by running it in a thread pool. This allows
    for concurrent execution of multiple generation requests and weight updates.

    Args:
        llm: vLLM LLM instance (from trainer.llm)

    Examples:
        ```python
        >>> import asyncio
        >>> from vllm import LLM, SamplingParams
        >>> from trl.extras.vllm_colocate_async import AsyncVLLMColocateWrapper


        >>> async def main():
        ...     # Initialize vLLM
        ...     llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct", gpu_memory_utilization=0.3)
        ...     async_llm = AsyncVLLMColocateWrapper(llm)

        ...     # Run multiple generations in parallel
        ...     sampling_params = SamplingParams(temperature=0.7, max_tokens=32)
        ...     tasks = [
        ...         async_llm.generate_async(["Hello!"], sampling_params),
        ...         async_llm.generate_async(["Hi there!"], sampling_params),
        ...         async_llm.generate_async(["Greetings!"], sampling_params),
        ...     ]
        ...     results = await asyncio.gather(*tasks)  # All run concurrently!


        >>> asyncio.run(main())
        ```
    """

    def __init__(self, llm):
        """
        Initialize the async wrapper.

        Args:
            llm: vLLM LLM instance (from trainer.llm)
        """
        self.llm = llm
        self._lock = asyncio.Lock()  # Ensure sequential access to vLLM engine

    async def generate_async(
        self,
        prompts: list[str],
        sampling_params,
        use_tqdm: bool = False,
    ) -> list:
        """
        Asynchronously generate completions for the provided prompts.

        This method runs vLLM's synchronous generate() in a thread pool to avoid blocking the event loop. Multiple
        calls are serialized using a lock to ensure thread-safe access to the vLLM engine.

        Args:
            prompts (`list[str]` or `list[dict]`):
                List of text prompts or vLLM input dictionaries (for multimodal).
            sampling_params (`SamplingParams`):
                vLLM sampling parameters for generation.
            use_tqdm (`bool`, *optional*, defaults to `False`):
                Whether to show a progress bar.

        Returns:
            `list`: List of vLLM RequestOutput objects containing generated tokens.

        Examples:
            ```python
            >>> from vllm import SamplingParams

            >>> sampling_params = SamplingParams(temperature=0.7, max_tokens=32)
            >>> outputs = await async_llm.generate_async(
            ...     prompts=["Hello, world!"],
            ...     sampling_params=sampling_params,
            ... )
            >>> print(outputs[0].outputs[0].text)
            ```
        """
        # Acquire lock to ensure thread-safe access to vLLM engine
        async with self._lock:
            # Run blocking generate() in thread pool - doesn't block event loop!
            loop = asyncio.get_event_loop()
            outputs = await loop.run_in_executor(
                None,  # Use default ThreadPoolExecutor
                lambda: self.llm.generate(prompts, sampling_params, use_tqdm=use_tqdm),
            )
            return outputs


def make_async_vllm_colocate(llm) -> AsyncVLLMColocateWrapper:
    """
    Convenience function to wrap a vLLM LLM instance for async usage.

    Args:
        llm: vLLM LLM instance

    Returns:
        AsyncVLLMColocateWrapper: Async wrapper around the vLLM instance

    Examples:
        ```python
        >>> from vllm import LLM
        >>> from trl.extras.vllm_colocate_async import make_async_vllm_colocate

        >>> llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct")
        >>> async_llm = make_async_vllm_colocate(llm)

        >>> # Now use async_llm.generate_async() instead of llm.generate()
        ```
    """
    return AsyncVLLMColocateWrapper(llm)
