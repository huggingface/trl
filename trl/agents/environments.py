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
        parsing_string: String that marks the beginning of code blocks in the generated text (default: "<code>").
        stop_string: String that marks the end of code blocks in the generated text (default: "</code>").
        tools_script: Optional script to prepend to each extracted code block before execution.
        output_string_start: String marking the beginning of code output to be inserted into the conversation (default: "<output>").
        output_string_end: String marking the end of code output (default: "</output>").
    """

    def __init__(
        self,
        code_executer: Any,
        tokenizer: PreTrainedTokenizerBase,
        parsing_string: str = "<code>",
        stop_string: str = "</code>",
        tools_script: Optional[str] = None,
        output_string_start: str = "<output>",
        output_string_end: str = "</output>",
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
        # Track conversations by prompt index and copy index
        conversation_map = {}
        active_conversations = []
        active_conversations_metadata = []  # Store (prompt_idx, copy_idx) for each active conversation
        
        # Store original prompt lengths to track completion length
        prompt_lengths = {}
    
        # Expand initial prompts based on n
        for prompt_idx, prompt in enumerate(prompts):
            for copy_idx in range(generation_config.n):
                active_conversations.append(prompt)
                active_conversations_metadata.append((prompt_idx, copy_idx))
                # Store original prompt length in tokens
                prompt_lengths[(prompt_idx, copy_idx)] = len(self.tokenizer.encode(prompt, add_special_tokens=False))
    
        # Maximum total tokens across all steps (use max_tokens as the limit for entire conversation)
        total_max_tokens = generation_config.max_tokens
    
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
            next_active_conversations_metadata = []
            code_batch = []
            conversations_pending_code = []
            pending_code_metadata = []  # Track metadata for conversations pending code execution
    
            for i, generated_token_ids in enumerate(outputs):
                current_prompt = active_conversations[i]
                current_metadata = active_conversations_metadata[i]
    
                # Check if the generated_token_ids list is valid
                if not isinstance(generated_token_ids, list):
                    conversation_map[current_metadata] = current_prompt
                    continue
    
                # Decode the newly generated part
                generated_text = self.tokenizer.decode(generated_token_ids, skip_special_tokens=True)
    
                full_conversation_segment = current_prompt + generated_text
                
                # Check if the current completion exceeds the total token limit
                completion_tokens_length = len(self.tokenizer.encode(full_conversation_segment, add_special_tokens=False)) - prompt_lengths[current_metadata]
                
                if completion_tokens_length > total_max_tokens:
                    # Truncate the completion to stay within the limit
                    # Get original prompt
                    original_prompt = ""
                    for prompt in prompts:
                        if full_conversation_segment.startswith(prompt):
                            original_prompt = prompt
                            break
                    
                    # Encode prompt and full conversation
                    prompt_tokens = self.tokenizer.encode(original_prompt, add_special_tokens=False)
                    full_tokens = self.tokenizer.encode(full_conversation_segment, add_special_tokens=False)
                    
                    # Calculate where to truncate
                    truncate_at = len(prompt_tokens) + total_max_tokens
                    truncated_tokens = full_tokens[:truncate_at]
                    
                    # Decode back to text
                    truncated_conversation = self.tokenizer.decode(truncated_tokens, skip_special_tokens=False)
                    
                    # Add to completed conversations and skip further processing
                    conversation_map[current_metadata] = truncated_conversation
                    continue
    
                # Check if the generation stopped because of our specific code stop string
                stopped_by_code_tag = generated_text.rstrip().endswith(self.stop_string.rstrip())
    
                # Check if code execution is requested IN THE NEWLY GENERATED TEXT
                last_code_start_in_segment = full_conversation_segment.rfind(self.parsing_string)
                is_code_in_new_text = last_code_start_in_segment != -1 and last_code_start_in_segment >= len(
                    current_prompt
                )
    
                if is_code_in_new_text and stopped_by_code_tag:
                    # Extract code from the full segment, as extract_code finds the last block
                    code = self.extract_code(full_conversation_segment)
                    if code:
                        code_batch.append(code)
                        conversations_pending_code.append(full_conversation_segment)
                        pending_code_metadata.append(current_metadata)
                    else:
                        # Parsing string found, but extraction failed. Treat as complete.
                        conversation_map[current_metadata] = full_conversation_segment
                else:
                    # Generation finished (max tokens, other stop word) or no code detected in the new part
                    conversation_map[current_metadata] = full_conversation_segment
    
            # Execute code batch if any code was extracted
            if code_batch:
                try:
                    execution_results = self.code_executer.execute(code_batch)
                    if len(execution_results) != len(conversations_pending_code):
                        raise ValueError(
                            f"Mismatch between code batch size ({len(code_batch)}) and results ({len(execution_results)})"
                        )
    
                    # Append results and add back to active conversations for the next round
                    for i, (conversation, result, metadata) in enumerate(
                        zip(conversations_pending_code, execution_results, pending_code_metadata)
                    ):
                        updated_conversation = (
                            conversation + f"{self.output_string_start}{result}{self.output_string_end}"
                        )
                        next_active_conversations.append(updated_conversation)
                        next_active_conversations_metadata.append(metadata)
                except Exception:
                    # Add pending as completed on error
                    for conv, metadata in zip(conversations_pending_code, pending_code_metadata):
                        conversation_map[metadata] = conv
    
            # Update the list of conversations for the next iteration
            active_conversations = next_active_conversations
            active_conversations_metadata = next_active_conversations_metadata
    
        # Reconstruct ordered list of completed conversations
        completed_conversations = []
        for prompt_idx in range(len(prompts)):
            for copy_idx in range(generation_config.n):
                metadata = (prompt_idx, copy_idx)
                if metadata in conversation_map:
                    completed_conversations.append(conversation_map[metadata])
    
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

        completion_ids = []
        extracted_completions = []

        for final_output in completed_conversations:
            # Check if any prompt from prompts exists in the completion
            found_prompt = False
            for prompt in prompts:
                if prompt in final_output:
                    # Extract everything after the prompt
                    start_index = final_output.find(prompt) + len(prompt)
                    completion_text = final_output[start_index:]
                    found_prompt = True
                    break
            
            if not found_prompt:
                print(f"Warning: No matching prompt found in completion. Using full output.")
                completion_text = final_output
            
            # Store the actual text completion for debugging
            extracted_completions.append(completion_text)
            
            # Convert to token IDs
            completion_token_ids = self.tokenizer.encode(completion_text, add_special_tokens=True)
            completion_ids.append(completion_token_ids)

        # Generate the mask for the completions
        completion_mask = self.mask_tool_output(
            completion_ids_list=completion_ids,
        )

        return completion_ids, completion_mask
