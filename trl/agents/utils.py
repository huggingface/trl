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
from typing import TYPE_CHECKING, Callable, List

# from dataclasses import dataclass
# from e2b_code_interpreter import AsyncSandbox
from ..import_utils import is_e2b_available, is_langchain_experimental_available


if is_e2b_available():
    from e2b_code_interpreter import AsyncSandbox

if is_langchain_experimental_available():
    from langchain_experimental.utilities import PythonREPL

if TYPE_CHECKING:
    from vllm import LLM, SamplingParams

default_system_prompt = "You can answer questions and solve problems. If running code helps, write it inside <code> </code>, and you will see the result. Example: To calculate 2 + 2, write <code> print(2 + 2) </code>."

default_environment_prompt = "This is a user-provided script containing tools that you can use to help complete tasks. It will be added to the sandbox environment so you can call its functions. If you are unsure how to use the available tools, you can use the `help()` function to inspect them."


def get_code(chat: str, tools_script: str = None, parsing_string: str = "<code>") -> str:
    """
    Extracts and optionally prepends a tools script to a code snippet from a chat message.

    Args:
        chat (`str`):
            Chat message containing the code snippet.
        tools_script (`str` or `None`, *optional*, defaults to `None`):
            A script to prepend to the extracted code snippet.
        parsing_string (`str`, *optional*, defaults to `"<code>"`):
            String used to identify the start of the code snippet in the chat message.

    Returns:
        `str`:
            Extracted code snippet, optionally prepended with the tools script.
    """
    code = chat.split(parsing_string)[-1]
    if tools_script:
        code = f"{tools_script}\n{code}"
    return code


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


class LocalExecutor:
    def execute(self, codes: List[str]) -> List[str]:
        """
        Executes multiple code snippets using PythonREPL sequentially.

        Args:
            codes (List[str]): List of code snippets to execute.

        Returns:
            List[str]: List of execution results in same order as input snippets.
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


class E2BExecutor:
    def __init__(self, api_key: str, template: str = None, max_concurrent: int = 5):
        """
        Initialize the E2BExecutor for parallel code execution.

        Args:
            api_key (`str`): Your E2B API Key.
            template (`str`, *optional*, defaults to None): Template ID for the sandbox environment.
            max_concurrent (`int`, *optional*, defaults to 5): Maximum number of concurrent executions.
        """
        self.api_key = api_key
        self.template = template
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)

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

    def execute(self, codes: List[str]) -> List[str]:
        """
        Executes multiple code snippets in parallel.

        Args:
            codes (List[str]): List of code snippets to execute.

        Returns:
            List[str]: List of execution results in same order as input snippets.
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
    tools: List[Callable] = None,
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
        tools (`List[Callable]` or `None`, *optional*, defaults to `None`):
            List of callable functions to include in system prompt
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


def generate_agent_responses(
    dataset: list,
    llm: "LLM",
    sampling_params: "SamplingParams",
    tools_script_path: str = None,
    parsing_string: str = "<code>",
    stop_string: str = "</code>",
    code_executer=None,
) -> list:
    """
    Generates responses for the agent with potential code execution.

    Args:
        dataset (`list`):
            List of preprocessed prompts (strings).
        llm (`LLM`):
            The language model to use for generation.
        sampling_params (`SamplingParams`):
            Sampling parameters for the llm.generate method.
        tools_script_path (`str` or `None`, *optional*, defaults to `None`):
            Path to script to prepend to code extracted.
        parsing_string (`str`, *optional*, defaults to `"<code>"`):
            String used to locate the code in the conversation.
        stop_string (`str`, *optional*, defaults to `"</code>"`):
            String used to stop generation.
        code_executer (`LocalExecutor`, *optional*, defaults to `LocalExecutor()`):
            Executor to run the code.

    Returns:
        `list`:
            A list of complete conversations (strings).
    """
    if code_executer is None:
        code_executer = LocalExecutor()

    # adding stop string to sampling params
    sampling_params.stop = [stop_string]
    # Read the tools script if provided.
    tools_script = read_script(tools_script_path) if tools_script_path else None

    completed_chats = []  # Chats that are fully complete.
    current_batch = dataset  # Start with your initial batch of prompts.

    while current_batch:
        # Generate outputs for the current batch.
        outputs = llm.generate(current_batch, sampling_params, use_tqdm=False)
        next_batch = []  # To store chats that still need code execution.
        code_batch = []  # To collect code snippets for batch execution
        conversations = []  # To keep track of conversations for each code

        # First pass: collect all codes that need execution
        for output in outputs:
            conversation = output.prompt + output.outputs[0].text
            if output.outputs[0].stop_reason == stop_string:
                code = get_code(conversation, tools_script=tools_script, parsing_string=parsing_string)
                code_batch.append(code)
                conversations.append(conversation)
            else:
                completed_chats.append(conversation)

        # Execute all collected codes in one batch
        if code_batch:
            executed_results = code_executer.execute(code_batch)

            # Process results and update conversations
            for conv, result in zip(conversations, executed_results):
                updated_conversation = conv + f"{stop_string}<output>" + result + "</output>"
                next_batch.append(updated_conversation)

        # Process next batch.
        current_batch = next_batch

    return completed_chats
