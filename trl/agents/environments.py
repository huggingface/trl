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
        tools_script: Optional[str] = None
    ):
        """Initialize the code agent environment
        
        Args:
            code_executor: The executor to run code (like LocalExecutor or E2BExecutor)
            tokenizer: Tokenizer for encoding/decoding text
            parsing_string: String that marks the beginning of code blocks
            stop_string: String that marks the end of code blocks
            tools_script: Optional script to prepend to extracted code
        """
        self.code_executor = code_executor
        self.tokenizer = tokenizer
        self.parsing_string = parsing_string
        self.stop_string = stop_string
        self.tools_script = tools_script
    
    def extract_code(self, text: str) -> str:
        """Extract code from generated text"""
        if self.parsing_string not in text:
            return None
            
        code_parts = text.split(self.parsing_string)[-1]
        if self.stop_string in code_parts:
            code_parts = code_parts.split(self.stop_string)[0]
        
        if self.tools_script:
            code_parts = f"{self.tools_script}\n{code_parts}"
            
        return code_parts
    
    def run_agent(self, vllm_client: Any, generation_config: VLLMClientGenerationConfig, prompts: List[str]) -> List[str]:
        """Run the agent with code execution and return completed text responses

        Args:
            vllm_client: VLLM client instance
            generation_config: Configuration for generation parameters
            prompts: Input prompts for generation
            
        Returns:
            List[str]: Completed text responses with code execution results
        """
        # Store original prompts for later use
        original_prompts = prompts.copy()
        # a buncha debug print statements
        print(f"Original prompts: {original_prompts}")
        # Duplicate prompts to match the requested number of generations (n)
        expanded_prompts = []
        for prompt in prompts:
            expanded_prompts.extend([prompt] * generation_config.n)
        print(f"Expanded prompts: {expanded_prompts}")
        # Create a modified generation config with n=1 for individual generations
        single_gen_config = VLLMClientGenerationConfig(
            n=1,
            repetition_penalty=generation_config.repetition_penalty,
            temperature=generation_config.temperature,
            top_p=generation_config.top_p,
            top_k=generation_config.top_k,
            min_p=generation_config.min_p,
            max_tokens=generation_config.max_tokens,
            guided_decoding_regex=generation_config.guided_decoding_regex,
            # Ensure stop string is always included for code parsing
            stop=[self.stop_string] if generation_config.stop is None else 
                    list(set(generation_config.stop + [self.stop_string])) # Use set to avoid duplicates
        )
        print(f"Single generation config: {single_gen_config}")
        # Track current conversations for each prompt
        current_conversations = expanded_prompts.copy()
        completed_conversations = []
        print(f"Initial Current conversations: {current_conversations}")
        print(f"Initial Completed conversations: {completed_conversations}")
        # Continue code execution loop until all conversations are completed
        while current_conversations:
            print(f"\n--- Starting Generation Loop ---")
            print(f"Current conversations ({len(current_conversations)}): {current_conversations}")
            # Generate next response segment
            outputs = vllm_client.generate(
                prompts=current_conversations,
                n=1, # Already handled expansion, generate 1 per conversation
                repetition_penalty=single_gen_config.repetition_penalty,
                temperature=single_gen_config.temperature,
                top_p=single_gen_config.top_p,
                top_k=single_gen_config.top_k,
                min_p=single_gen_config.min_p,
                max_tokens=single_gen_config.max_tokens,
                guided_decoding_regex=single_gen_config.guided_decoding_regex,
                stop=single_gen_config.stop
            )
            print(f"Generated outputs (token ids): {outputs}")
            
            next_conversations_for_llm = [] # Conversations needing more generation
            code_batch = [] # Code snippets to execute
            code_conversations_map = {} # Map index to conversation needing code exec

            # Process outputs
            for idx, (conversation_prompt, output_ids) in enumerate(zip(current_conversations, outputs)):
                # output is a list of token ids; decode it
                # Important: Decode carefully, handle potential stop tokens like stop_string
                generated_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
                
                # Check if generation stopped exactly at the stop_string
                stopped_at_code_end = False
                if single_gen_config.stop:
                    # Check if the last decoded token corresponds to any stop sequence
                    # This is tricky, a simpler check might be needed depending on tokenizer behavior
                    # For now, check if the decoded text ends with the stop_string
                    if generated_text.endswith(self.stop_string.strip()): # Strip potential whitespace
                            stopped_at_code_end = True
                            # Remove the stop string itself if needed, or keep it based on desired format
                            # generated_text = generated_text[:-len(self.stop_string)] # Optional: remove stop string

                print(f"\nProcessing conversation index {idx}:")
                print(f"  Prompt: '{conversation_prompt}'")
                print(f"  Generated text: '{generated_text}'")
                print(f"  Stopped at code end: {stopped_at_code_end}")

                # Check if code execution is indicated IN THE NEWLY GENERATED TEXT
                # And if the generation actually stopped because of the stop_string
                if self.parsing_string in generated_text and stopped_at_code_end:
                    # Construct the full conversation up to this point
                    full_conversation_segment = conversation_prompt + generated_text 
                    
                    # Extract code ONLY from the generated part for safety
                    code = self.extract_code(generated_text) # Pass only generated text
                    print(f"  Extracted code: {code}")
                    if code:
                        code_batch.append(code)
                        # Store the conversation *before* adding output tag, map by original index
                        code_conversations_map[idx] = full_conversation_segment
                    else:
                            # Code tags present but extraction failed? Treat as complete for now.
                            print(f"  Warning: Code tags found but extraction failed. Completing conversation.")
                            completed_conversations.append(full_conversation_segment)

                else:
                    # No code block detected in the new text OR didn't stop at stop_string
                    # Conversation is considered complete for this agent logic
                    full_conversation = conversation_prompt + generated_text
                    completed_conversations.append(full_conversation)
                    print(f"  No code detected or not stopped correctly. Conversation completed.")
            
            print(f"\n--- After Processing Generations ---")
            print(f"Code batch ({len(code_batch)}): {code_batch}")
            print(f"Code conversations map ({len(code_conversations_map)}): {code_conversations_map.keys()}")
            print(f"Completed conversations ({len(completed_conversations)}): {[c[-100:]+'...' for c in completed_conversations]}") # Show ends

            # Execute code batch if any code was extracted
            if code_batch:
                print(f"\n--- Executing Code Batch ---")
                execution_results = self.code_executor.execute(code_batch)
                print(f"Execution results: {execution_results}")
                
                # Add execution results back to the corresponding conversations
                result_idx = 0
                for original_idx, conversation_segment in code_conversations_map.items():
                    result = execution_results[result_idx]
                    # Append the output tag and result
                    updated_conversation = conversation_segment + f"<output>{result}</output>"
                    next_conversations_for_llm.append(updated_conversation) # Add to list for next LLM call
                    print(f"  Appended output to conversation index {original_idx}. Ready for next generation.")
                    result_idx += 1
            else:
                    print(f"\n--- No Code to Execute ---")

            # Update current conversations for the next iteration of the loop
            current_conversations = next_conversations_for_llm
            print(f"\n--- End of Loop Iteration ---")
            print(f"Next conversations for LLM ({len(current_conversations)}): {[c[-100:]+'...' for c in current_conversations]}") # Show ends

        # Return the final completed conversations
        print(f"\n--- Agent Run Finished ---")
        print(f"Final Completed conversations ({len(completed_conversations)}): {completed_conversations}")
        return completed_conversations
         
    def generate(self, vllm_client: Any, generation_config: VLLMClientGenerationConfig, prompts: List[str]) -> List:
        """Generate responses with code execution and return token IDs

        Args:
            vllm_client: VLLM client instance
            generation_config: Configuration for generation parameters
            prompts: Input prompts for generation
            
        Returns:
            completion_ids: Generated token ids
        """
        # Get completed text responses from the agent
        completed_conversations = self.run_agent(vllm_client, generation_config, prompts)
        
        # Recreate expanded prompts for completion extraction
        expanded_prompts = []
        for prompt in prompts:
            expanded_prompts.extend([prompt] * generation_config.n)
        
        # Extract completion IDs (just the generated part without the original prompt)
        completion_ids = []
        for original_prompt, final_output in zip(expanded_prompts, completed_conversations):
            # Extract just the completion (everything after the original prompt)
            completion_text = final_output[len(original_prompt):]
            
            # Encode to get token IDs
            completion_token_ids = self.tokenizer.encode(
                completion_text, 
                add_special_tokens=False
            )
            completion_ids.append(completion_token_ids)
        
        return completion_ids