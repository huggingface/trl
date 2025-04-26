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

# Copyright 2025 The HuggingFace Team. All rights reserved.
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
# from __future__ import annotations

import asyncio
from inspect import getsource
from pathlib import Path
from typing import Callable

from ..import_utils import is_e2b_available, is_langchain_experimental_available


if is_e2b_available():
    from e2b_code_interpreter import AsyncSandbox  # used for E2B async code execution

if is_langchain_experimental_available():
    from langchain_experimental.utilities import PythonREPL  # used for local code execution

default_system_prompt = "You can answer questions and solve problems. If running code helps, write it inside <code> </code>, and you will see the result. Example: To calculate 2 + 2, write <code> print(2 + 2) </code>."

default_environment_prompt = "This is a user-provided script containing tools that you can use to help complete tasks. It will be added to the sandbox environment so you can call its functions. If you are unsure how to use the available tools, you can use the `help()` function to inspect them."


def read_script(user_script_path: str) -> str:
    """
    Reads and returns the content of the provided script.

    Args:
        user_script_path (`str`):
            Path to the user-provided script.

    Returns:
        `str`:
            The content of the script.
    """
    return Path(user_script_path).read_text()


class Localexecuter:
    def execute(self, codes: list[str]) -> list[str]:
        """
        Executes multiple code snippets using PythonREPL sequentially.

        Args:
            codes (list[str]): list of code snippets to execute.

        Returns:
            list[str]: list of execution results in same order as input snippets.
        """
        results = []
        repl = PythonREPL()
        for code in codes:
            try:
                result = repl.run(code)
                results.append(str(result))
            except Exception as e:
                results.append(f"Error executing code: {str(e)}")
        return results


class E2BExecuter:
    def __init__(self, api_key: str, template: str = None, max_concurrent: int = 20):
        """
        Initialize the E2BExecuter for parallel code execution.
        Tests the connection by running a simple print statement.

        Args:
            api_key (`str`): Your E2B API Key.
            template (`str`, *optional*, defaults to None): Template ID for the sandbox environment.
            max_concurrent (`int`, *optional*, defaults to 5): Maximum number of concurrent executions.
        """
        self.api_key = api_key
        self.template = template
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)

        # Run a test code to validate connection
        validation_code = "print('E2B connection test successful')"
        result = self.execute([validation_code])[0]
        # Check if the result contains the expected output
        if "successful" not in result:
            raise ConnectionError(f"E2B connection test failed. Response: {result}")

    async def _execute_single(self, code: str) -> str:
        """Execute a single code snippet in a sandbox"""
        async with self.semaphore:
            try:
                sandbox = await AsyncSandbox.create(api_key=self.api_key, template=self.template)
                try:
                    response = await sandbox.run_code(code)
                    return str(response)
                finally:
                    await sandbox.kill()
            except Exception as e:
                return f"Error: {str(e)}"

    def execute(self, codes: list[str]) -> list[str]:
        """
        Executes multiple code snippets in parallel.

        Args:
            codes (list[str]): list of code snippets to execute.

        Returns:
            list[str]: list of execution results in same order as input snippets.
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        async def run_batch():
            tasks = [self._execute_single(code) for code in codes]
            return await asyncio.gather(*tasks)

        if loop.is_running():
            # We're in a notebook or similar environment
            import nest_asyncio

            nest_asyncio.apply()
            return loop.run_until_complete(run_batch())
        else:
            # We're in a regular Python environment
            return loop.run_until_complete(run_batch())


def prepare_data_for_e2b_agent(
    dataset,
    tokenizer,
    prompt_column: str = "prompt",
    system_prompt: str = default_system_prompt,
    environment_prompt: str = default_environment_prompt,
    tools_script_path: str = None,
) -> list:
    """
    Prepares the Hugging Face dataset for the e2b agent by constructing conversations
    and applying the system prompt.

    Args:
        dataset (`Dataset`):
            A Hugging Face dataset object
        tokenizer (`PreTrainedTokenizer`):
            A tokenizer with an `apply_chat_template` method to process conversations
        prompt_column (`str`, *optional*, defaults to `"prompt"`):
            Name of the column containing prompts.
        system_prompt (`str`, *optional*, defaults to `default_system_prompt`):
            The base system prompt.
        environment_prompt (`str`, *optional*, defaults to `default_environment_prompt`):
            Description to prepend before the tools script
        tools_script_path (`str` or `None`, *optional*, defaults to `None`):
            File path to a tools script to include in system prompt

    Returns:
        `Dataset`:
            Modified dataset with processed prompts in the prompt column
    """
    # Convert dataset prompts to conversational format
    conversations = [[{"role": "user", "content": prompt}] for prompt in dataset[prompt_column]]

    # If a tools script path is provided, read its content and append with the environment prompt
    if tools_script_path:
        try:
            tool_script = read_script(tools_script_path)
            system_prompt += "\n" + environment_prompt + "\n" + tool_script
        except Exception as e:
            raise RuntimeError(f"Error reading the tools script: {e}") from e

    # Create the system prompt message
    system_message = {"role": "system", "content": system_prompt}

    # Prepend the system message to each conversation
    for convo in conversations:
        if not convo or convo[0].get("role") != "system":
            convo.insert(0, system_message)

    # Apply the tokenizer's chat template
    processed_prompts = tokenizer.apply_chat_template(conversations, tokenize=False, add_generation_prompt=True)

    # Create a new dataset with processed prompts
    return dataset.map(lambda x, idx: {prompt_column: processed_prompts[idx]}, with_indices=True)


def prepare_data_for_local_agent(
    dataset,
    tokenizer,
    prompt_column: str = "prompt",
    system_prompt: str = default_system_prompt,
    tools: list[Callable] = None,
    include_source_code: bool = True,
) -> list:
    """
    Prepares the Hugging Face dataset for the local agent by constructing conversations
    and applying the system prompt.

    Args:
        dataset (`Dataset`):
            A Hugging Face dataset object
        tokenizer (`PreTrainedTokenizer`):
            A tokenizer with an `apply_chat_template` method to process conversations
        prompt_column (`str`, *optional*, defaults to `"prompt"`):
            Name of the column containing prompts.
        system_prompt (`str`, *optional*, defaults to `default_system_prompt`):
            The base system prompt.
        tools (`list[Callable]` or `None`, *optional*, defaults to `None`):
            list of callable functions to include in system prompt
        include_source_code (`bool`, *optional*, defaults to `True`):
            If True, includes source code of tools, else includes docstrings

    Returns:
        `Dataset`:
            Modified dataset with processed prompts in the prompt column
    """
    # Convert dataset prompts to conversational format
    conversations = [[{"role": "user", "content": prompt}] for prompt in dataset[prompt_column]]

    # Process tools if provided
    if tools:
        tools_documentation = "\nAvailable tools:\n\n"
        for tool in tools:
            tools_documentation += f"Tool: {tool.__name__}\n"
            if include_source_code:
                try:
                    tools_documentation += f"Source code:\n{getsource(tool)}\n\n"
                except Exception as e:
                    tools_documentation += f"Error getting source: {str(e)}\n\n"
            else:
                doc = tool.__doc__ or "No documentation available"
                tools_documentation += f"Documentation:\n{doc}\n\n"

        system_prompt += tools_documentation

    # Create the system prompt message
    system_message = {"role": "system", "content": system_prompt}

    # Prepend the system message to each conversation
    for convo in conversations:
        if not convo or convo[0].get("role") != "system":
            convo.insert(0, system_message)

    # Apply the tokenizer's chat template
    processed_prompts = tokenizer.apply_chat_template(conversations, tokenize=False, add_generation_prompt=True)

    # Create a new dataset with processed prompts
    return dataset.map(lambda x, idx: {prompt_column: processed_prompts[idx]}, with_indices=True)
