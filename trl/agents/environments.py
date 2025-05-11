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

import abc
from typing import Any, Optional

from transformers import PreTrainedTokenizerBase

from ..extras.vllm_client import VLLMClientGenerationConfig


class Environment(abc.ABC):
    """Base environment that implements standard VLLM generation"""

    @abc.abstractmethod
    def generate(
        self, vllm_client: Any, generation_config: VLLMClientGenerationConfig, prompts: list[str]
    ) -> tuple[list, list]:
        """Generate responses using VLLM

        Args:
            vllm_client: VLLM client instance
            generation_config: Configuration for generation parameters
            prompts: Input prompts for generation

        Returns:
            tuple: (completion_ids, completion_mask) where:
                - completion_ids: Generated token ids (list of lists)
                - completion_mask: Mask indicating which tokens to consider (list of lists)
        """
        pass


class DefaultEnvironment(Environment):
    """Default environment that implements standard VLLM generation"""

    def generate(
        self, vllm_client: Any, generation_config: VLLMClientGenerationConfig, prompts: list[str]
    ) -> tuple[list, list]:
        """Generate responses using VLLM
        Args:
            vllm_client: VLLM client instance
            generation_config: Configuration for generation parameters
            prompts: Input prompts for generation

        Returns:
            tuple: (completion_ids, completion_mask) where:
                - completion_ids: Generated token ids (list of lists)
                - completion_mask: Mask with 1s for all tokens (list of lists with same shape)
        """
        if generation_config is None:
            raise ValueError("Generation config must be provided to the generate method")

        completion_ids = vllm_client.generate(
            prompts=prompts,
            **vars(generation_config),
        )

        # Create a mask with all 1s matching the shape of completion_ids
        completion_mask = [[1] * len(ids) for ids in completion_ids]

        return completion_ids, completion_mask


class CodeAgentEnvironment(Environment):
    """
    Environment that supports code execution during generation.

    This environment enables an agent to generate text that include code blocks,
    execute those code blocks using a provided code executer (such as Localexecuter or E2BExecuter),
    and insert the execution results back into the conversation. It is designed to work with
    conversational models that can output code delimited by specific tags (e.g., <code>...</code>).

    The environment repeatedly generates text, detects code blocks using configurable delimiters,
    executes the code, and appends the output (delimited by output tags) to the conversation.
    This process continues until no further code execution is requested in the generated text.

    Args:
        code_executer: An object with an `execute` method that takes a list of code strings and returns a list of results.
            This is used to run the extracted code blocks (e.g., Localexecuter or E2BExecuter).
        tokenizer: A PreTrainedTokenizerBase instance for encoding and decoding text and completions.
        parsing_string: String that marks the beginning of code blocks in the generated text (default: " <code> ").
        stop_string: String that marks the end of code blocks in the generated text (default: " </code> ").
        tools_script: Optional script to prepend to each extracted code block before execution.
        output_string_start: String marking the beginning of code output to be inserted into the conversation (default: " <output> ").
        output_string_end: String marking the end of code output (default: " </output> ").
    """

    def __init__(
        self,
        code_executer: Any,
        tokenizer: PreTrainedTokenizerBase,
        parsing_string: str = " <code> ",
        stop_string: str = " </code> ",
        tools_script: Optional[str] = None,
        output_string_start: str = " <output> ",
        output_string_end: str = " </output> ",
    ):
        """Initialize the code agent environment

        Args:
            code_executer: The executer to run code (like Localexecuter or E2BExecuter)
            tokenizer: Tokenizer for encoding/decoding text
            parsing_string: String that marks the beginning of code blocks
            stop_string: String that marks the end of code blocks
            tools_script: Optional script to prepend to extracted code
            output_string_start: String marking the beginning of code output.
            output_string_end: String marking the end of code output.
        """
        if not hasattr(code_executer, "execute"):
            raise ValueError("code_executer must have an 'execute' method.")

        self.code_executer = code_executer
        self.tokenizer = tokenizer
        self.parsing_string = parsing_string
        self.stop_string = stop_string
        self.tools_script = tools_script
        self.output_string_start = output_string_start
        self.output_string_end = output_string_end

    def extract_code(self, text: str) -> Optional[str]:
        """Extract code from the *last* code block in the generated text."""
        if self.parsing_string not in text:
            return None

        # Find the last occurrence of the parsing string
        last_code_start_index = text.rfind(self.parsing_string)
        if last_code_start_index == -1:
            return None  # Should not happen if parsing_string is in text, but safety check

        code_segment = text[last_code_start_index + len(self.parsing_string) :]

        # Find the first occurrence of the stop string *after* the last parsing string
        stop_index = code_segment.find(self.stop_string)
        if stop_index != -1:
            code_parts = code_segment[:stop_index]
        else:
            # If stop string is not found after the last parsing string,
            # maybe the generation stopped exactly at the stop string.
            # Or maybe the stop string is missing. Assume it's the end for now.
            # This might need adjustment based on typical model behavior.
            code_parts = code_segment  # Or handle error?

        # Prepend tools script if available
        if self.tools_script:
            code_parts = f"{self.tools_script}\n{code_parts}"

        # Basic cleaning
        code_parts = code_parts.strip()

        return code_parts if code_parts else None

    def run_agent(
        self, vllm_client: Any, generation_config: VLLMClientGenerationConfig, prompts: list[str]
    ) -> list[str]:
        """Run the agent with code execution and return completed text responses.

        Args:
            vllm_client: VLLM client instance.
            generation_config: Configuration for generation parameters.
            prompts: Input prompts for generation.

        Returns:
            list[str]: Completed text responses with code execution results.
        """
        completed_conversations = []
        active_conversations = []

        # Expand initial prompts based on n
        for prompt in prompts:
            active_conversations.extend([prompt] * generation_config.n)

        # Ensure stop_string is always in the stop sequences for generation
        stop_sequences = [self.stop_string]
        if generation_config.stop:
            stop_sequences = list(set(stop_sequences + generation_config.stop))

        # Create a generation config for individual steps (n=1)
        step_gen_config = VLLMClientGenerationConfig(
            n=1,
            repetition_penalty=generation_config.repetition_penalty,
            temperature=generation_config.temperature,
            top_p=generation_config.top_p,
            top_k=generation_config.top_k,
            min_p=generation_config.min_p,
            max_tokens=generation_config.max_tokens,
            guided_decoding_regex=generation_config.guided_decoding_regex,
            stop=stop_sequences,
        )

        while active_conversations:
            outputs = vllm_client.generate(prompts=active_conversations, **vars(step_gen_config))

            next_active_conversations = []
            code_batch = []
            conversations_pending_code = []

            for i, generated_token_ids in enumerate(outputs):  # Assumes 'output' is directly the list of token IDs
                current_prompt = active_conversations[i]

                # Check if the generated_token_ids list is valid
                if not isinstance(generated_token_ids, list):
                    completed_conversations.append(current_prompt)
                    continue

                # Decode the newly generated part
                generated_text = self.tokenizer.decode(generated_token_ids, skip_special_tokens=True)

                full_conversation_segment = current_prompt + generated_text

                # Check if the generation stopped because of our specific code stop string
                # Rely solely on the text ending with the stop string
                stopped_by_code_tag = generated_text.rstrip().endswith(self.stop_string.rstrip())

                # Check if code execution is requested IN THE NEWLY GENERATED TEXT
                # Use rfind on the full segment to find the *last* code block start
                last_code_start_in_segment = full_conversation_segment.rfind(self.parsing_string)
                # Check if the *last* code block start is within the *newly generated* part
                # Ensure last_code_start_in_segment is found before comparing index
                is_code_in_new_text = last_code_start_in_segment != -1 and last_code_start_in_segment >= len(
                    current_prompt
                )

                if is_code_in_new_text and stopped_by_code_tag:
                    # Extract code from the full segment, as extract_code finds the last block
                    code = self.extract_code(full_conversation_segment)
                    if code:
                        code_batch.append(code)
                        # Store the conversation *including* the generated code block ending with stop_string
                        conversations_pending_code.append(full_conversation_segment)
                    else:
                        # Parsing string found, but extraction failed. Treat as complete.
                        completed_conversations.append(full_conversation_segment)
                else:
                    # Generation finished (max tokens, other stop word) or no code detected in the new part
                    completed_conversations.append(full_conversation_segment)

            # Execute code batch if any code was extracted
            if code_batch:
                try:
                    execution_results = self.code_executer.execute(code_batch)
                    if len(execution_results) != len(conversations_pending_code):
                        raise ValueError(
                            f"Mismatch between code batch size ({len(code_batch)}) and results ({len(execution_results)})"
                        )

                    # Append results and add back to active conversations for the next round
                    for conversation, result in zip(conversations_pending_code, execution_results):
                        updated_conversation = (
                            conversation + f"{self.output_string_start}{result}{self.output_string_end}"
                        )
                        next_active_conversations.append(updated_conversation)
                except Exception:
                    completed_conversations.extend(conversations_pending_code)  # Add pending as completed on error

            # Update the list of conversations for the next iteration
            active_conversations = next_active_conversations

        return completed_conversations

    def mask_tool_output(
        self,
        completion_ids_list: list[list[int]],
        output_string_start: Optional[str] = None,
        output_string_end: Optional[str] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
    ) -> list[list[int]]:
        """
        Create a mask for token sequences, marking tokens that are part of tool outputs with 0s.
        This version assumes the effective output_string_start and output_string_end each represent a single token.
        It masks from the start token ID to the end token ID, inclusive.

        Args:
            completion_ids_list: List of token ID sequences.
            output_string_start: Optional string marking the beginning of code output.
                                    If None, defaults to self.output_string_start.
            output_string_end: Optional string marking the end of code output.
                                If None, defaults to self.output_string_end.
            tokenizer: Tokenizer to convert strings to token IDs.
                        Defaults to self.tokenizer.

        Returns:
            List of mask lists (1s for tokens to keep, 0s for tokens to mask).
        """
        if tokenizer is None:
            tokenizer = self.tokenizer

        current_output_string_start = (
            output_string_start if output_string_start is not None else self.output_string_start
        )
        current_output_string_end = output_string_end if output_string_end is not None else self.output_string_end

        if not current_output_string_start.strip():
            raise ValueError("Effective output_string_start is empty or whitespace.")
        encoded_start = tokenizer.encode(current_output_string_start, add_special_tokens=False)
        if len(encoded_start) != 1:
            raise ValueError(
                f"Effective output_string_start '{current_output_string_start}' does not correspond to a single token. "
                f"Encoded to: {encoded_start}. Ensure it's a single token."
            )
        output_start_token_id = encoded_start[0]

        if not current_output_string_end.strip():
            raise ValueError("Effective output_string_end is empty or whitespace.")
        encoded_end = tokenizer.encode(current_output_string_end, add_special_tokens=False)
        if len(encoded_end) != 1:
            raise ValueError(
                f"Effective output_string_end '{current_output_string_end}' does not correspond to a single token. "
                f"Encoded to: {encoded_end}. Ensure it's a single token."
            )
        output_end_token_id = encoded_end[0]

        completion_masks = []
        for completion_ids in completion_ids_list:
            mask = [1] * len(completion_ids)
            if not completion_ids:
                completion_masks.append(mask)
                continue

            i = 0
            while i < len(completion_ids):
                if completion_ids[i] == output_start_token_id:
                    # Found the start token
                    start_marker_idx = i
                    # Search for the end token
                    j = i + 1
                    found_end_marker = False
                    while j < len(completion_ids):
                        if completion_ids[j] == output_end_token_id:
                            # Found the end token
                            end_marker_idx = j
                            # Mask from start_marker_idx to end_marker_idx (inclusive)
                            for k_mask in range(start_marker_idx, end_marker_idx + 1):
                                if k_mask < len(mask):  # Ensure index is within bounds
                                    mask[k_mask] = 0
                            i = end_marker_idx + 1  # Continue search after the masked segment
                            found_end_marker = True
                            break
                        j += 1
                    if not found_end_marker:
                        # End marker not found after a start marker, advance past the start marker
                        i += 1
                else:
                    i += 1
            completion_masks.append(mask)
        return completion_masks

    def generate(
        self, vllm_client: Any, generation_config: VLLMClientGenerationConfig, prompts: list[str]
    ) -> list[list[int]]:
        """Generate responses with code execution and return token IDs of the completions.
        The Generate method is used for the training loop.

        Args:
            vllm_client: VLLM client instance.
            generation_config: Configuration for generation parameters.
            prompts: Input prompts for generation.

        Returns:
            list[list[int]]: list of generated token ID lists (completions only).
        """
        # Get completed text responses from the agent logic
        completed_conversations = self.run_agent(vllm_client, generation_config, prompts)

        # Recreate the list of original prompts expanded by 'n' to match the output count
        expanded_prompts = []
        for prompt in prompts:
            expanded_prompts.extend([prompt] * generation_config.n)

        if len(expanded_prompts) != len(completed_conversations):
            # This indicates a potential issue in run_agent or prompt expansion
            # Adjust the shorter list to match the longer one? Or raise error?
            # For robustness, let's process based on the number of completed conversations
            expanded_prompts = expanded_prompts[: len(completed_conversations)]

        completion_ids = []
        for original_prompt, final_output in zip(expanded_prompts, completed_conversations):
            # Ensure the final output actually starts with the prompt
            if final_output.startswith(original_prompt):
                completion_text = final_output[len(original_prompt) :]
            else:
                # Handle cases where the output might not perfectly match the start (e.g., due to tokenization differences)
                # Or if the conversation somehow got corrupted. Fallback to using the whole output as completion?
                completion_text = final_output  # Or potentially try a fuzzy match / diff?

            # Encode the completion text to get token IDs
            # add_special_tokens=False is typical for training completions
            completion_token_ids = self.tokenizer.encode(completion_text, add_special_tokens=False)
            completion_ids.append(completion_token_ids)

        # Generate the mask for the completions
        completion_mask = self.mask_tool_output(
            completion_ids_list=completion_ids,
        )

        return completion_ids, completion_mask
