from dataclasses import dataclass 
from typing import List, Any, Optional, Union, Dict
import abc
from transformers import PreTrainedTokenizerBase

@dataclass
class VLLMClientGenerationConfig:
    """Configuration for VLLM client generation parameters"""
    n: int
    repetition_penalty: float
    temperature: float
    top_p: float
    top_k: Optional[int]
    min_p: float
    max_tokens: int
    guided_decoding_regex: Optional[str] = None
    stop: Optional[List[str]] = None

class Environment(abc.ABC):
    """Base environment that implements standard VLLM generation"""
    
    @abc.abstractmethod
    def generate(self, vllm_client: Any, generation_config: VLLMClientGenerationConfig, prompts: List[str]) -> List:
        """Generate responses using VLLM

        Args:
            vllm_client: VLLM client instance
            generation_config: Configuration for generation parameters
            prompts: Input prompts for generation
            
        Returns:
            completion_ids: Generated token ids
        """
        pass

class DefaultEnvironment(Environment):
    """Default environment that implements standard VLLM generation"""
    
    def generate(self, vllm_client: Any, generation_config: VLLMClientGenerationConfig, prompts: List[str]) -> List:
        """Generate responses using VLLM

        Args:
            vllm_client: VLLM client instance
            generation_config: Configuration for generation parameters
            prompts: Input prompts for generation
            
        Returns:
            completion_ids: Generated token ids
        """
        if generation_config is None:
            raise ValueError("Generation config must be provided to the generate method")
            
        return vllm_client.generate(
            prompts=prompts,
            **vars(generation_config),
        )

class CodeAgentEnvironment(Environment):
    """Environment that supports code execution during generation"""

    def __init__(
        self,
        code_executor: Any,
        tokenizer: PreTrainedTokenizerBase,
        parsing_string: str = "<code>",
        stop_string: str = "</code>",
        tools_script: Optional[str] = None,
        output_parsing_string: str = "<output>",
        output_stop_string: str = "</output>",
    ):
        """Initialize the code agent environment

        Args:
            code_executor: The executor to run code (like LocalExecutor or E2BExecutor)
            tokenizer: Tokenizer for encoding/decoding text
            parsing_string: String that marks the beginning of code blocks
            stop_string: String that marks the end of code blocks
            tools_script: Optional script to prepend to extracted code
            output_parsing_string: String marking the beginning of code output.
            output_stop_string: String marking the end of code output.
        """
        if not hasattr(code_executor, "execute"):
             raise ValueError("code_executor must have an 'execute' method.")

        self.code_executor = code_executor
        self.tokenizer = tokenizer
        self.parsing_string = parsing_string
        self.stop_string = stop_string
        self.tools_script = tools_script
        self.output_parsing_string = output_parsing_string
        self.output_stop_string = output_stop_string

    def extract_code(self, text: str) -> Optional[str]:
        """Extract code from the *last* code block in the generated text."""
        if self.parsing_string not in text:
            return None

        # Find the last occurrence of the parsing string
        last_code_start_index = text.rfind(self.parsing_string)
        if last_code_start_index == -1:
             return None # Should not happen if parsing_string is in text, but safety check

        code_segment = text[last_code_start_index + len(self.parsing_string):]

        # Find the first occurrence of the stop string *after* the last parsing string
        stop_index = code_segment.find(self.stop_string)
        if stop_index != -1:
            code_parts = code_segment[:stop_index]
        else:
             # If stop string is not found after the last parsing string,
             # maybe the generation stopped exactly at the stop string.
             # Or maybe the stop string is missing. Assume it's the end for now.
             # This might need adjustment based on typical model behavior.
             code_parts = code_segment # Or handle error?

        # Prepend tools script if available
        if self.tools_script:
            code_parts = f"{self.tools_script}\n{code_parts}"

        # Basic cleaning
        code_parts = code_parts.strip()

        return code_parts if code_parts else None

    def run_agent(self, vllm_client: Any, generation_config: VLLMClientGenerationConfig, prompts: List[str]) -> List[str]:
        """Run the agent with code execution and return completed text responses.

        Args:
            vllm_client: VLLM client instance.
            generation_config: Configuration for generation parameters.
            prompts: Input prompts for generation.

        Returns:
            List[str]: Completed text responses with code execution results.
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
            stop=stop_sequences
        )

        while active_conversations:
            outputs = vllm_client.generate(
                prompts=active_conversations,
                **vars(step_gen_config)
            )

            next_active_conversations = []
            code_batch = []
            conversations_pending_code = []

            # Check if the structure of outputs is as expected (list of lists)
            if not isinstance(outputs, list) or (outputs and not all(isinstance(item, list) for item in outputs)):
                    print(f"Warning: Unexpected output structure from vllm_client.generate. Expected List[List[int]], got: {type(outputs)}. Attempting to proceed.")
                    # Depending on the actual structure, this might still fail later.

            for i, generated_token_ids in enumerate(outputs): # Assumes 'output' is directly the list of token IDs
                current_prompt = active_conversations[i]

                # Check if the generated_token_ids list is valid
                if not isinstance(generated_token_ids, list):
                    print(f"Warning: Invalid token list received for prompt index {i}. Skipping. Got: {type(generated_token_ids)}")
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
                is_code_in_new_text = last_code_start_in_segment != -1 and last_code_start_in_segment >= len(current_prompt)

                if is_code_in_new_text and stopped_by_code_tag:
                    # Extract code from the full segment, as extract_code finds the last block
                    code = self.extract_code(full_conversation_segment)
                    if code:
                        code_batch.append(code)
                        # Store the conversation *including* the generated code block ending with stop_string
                        conversations_pending_code.append(full_conversation_segment)
                    else:
                        # Parsing string found, but extraction failed. Treat as complete.
                        print(f"Warning: Code parsing string found but extraction failed for segment: ...{generated_text[-50:]}")
                        completed_conversations.append(full_conversation_segment)
                else:
                    # Generation finished (max tokens, other stop word) or no code detected in the new part
                    completed_conversations.append(full_conversation_segment)

            # Execute code batch if any code was extracted
            if code_batch:
                try:
                    execution_results = self.code_executor.execute(code_batch)
                    if len(execution_results) != len(conversations_pending_code):
                            raise ValueError(f"Mismatch between code batch size ({len(code_batch)}) and results ({len(execution_results)})")

                    # Append results and add back to active conversations for the next round
                    for conversation, result in zip(conversations_pending_code, execution_results):
                        updated_conversation = conversation + f"{self.output_parsing_string}{result}{self.output_stop_string}"
                        next_active_conversations.append(updated_conversation)
                except Exception as e:
                        print(f"Error during code execution batch: {e}")
                        completed_conversations.extend(conversations_pending_code) # Add pending as completed on error

            # Update the list of conversations for the next iteration
            active_conversations = next_active_conversations

        return completed_conversations


    def generate(self, vllm_client: Any, generation_config: VLLMClientGenerationConfig, prompts: List[str]) -> List[List[int]]:
        """Generate responses with code execution and return token IDs of the completions.

        Args:
            vllm_client: VLLM client instance.
            generation_config: Configuration for generation parameters.
            prompts: Input prompts for generation.

        Returns:
            List[List[int]]: List of generated token ID lists (completions only).
        """
        # Get completed text responses from the agent logic
        completed_conversations = self.run_agent(vllm_client, generation_config, prompts)

        # Recreate the list of original prompts expanded by 'n' to match the output count
        expanded_prompts = []
        for prompt in prompts:
            expanded_prompts.extend([prompt] * generation_config.n)

        if len(expanded_prompts) != len(completed_conversations):
             # This indicates a potential issue in run_agent or prompt expansion
             print(f"Warning: Mismatch between expanded prompts ({len(expanded_prompts)}) and completed conversations ({len(completed_conversations)}). Returning completions based on available conversations.")
             # Adjust the shorter list to match the longer one? Or raise error?
             # For robustness, let's process based on the number of completed conversations
             expanded_prompts = expanded_prompts[:len(completed_conversations)]


        completion_ids = []
        for original_prompt, final_output in zip(expanded_prompts, completed_conversations):
            # Ensure the final output actually starts with the prompt
            if final_output.startswith(original_prompt):
                completion_text = final_output[len(original_prompt):]
            else:
                # Handle cases where the output might not perfectly match the start (e.g., due to tokenization differences)
                # Or if the conversation somehow got corrupted. Fallback to using the whole output as completion?
                print(f"Warning: Final output does not start with the original prompt. Using entire final output as completion.")
                completion_text = final_output # Or potentially try a fuzzy match / diff?

            # Encode the completion text to get token IDs
            # add_special_tokens=False is typical for training completions
            completion_token_ids = self.tokenizer.encode(
                completion_text,
                add_special_tokens=False
            )
            completion_ids.append(completion_token_ids)

        return completion_ids