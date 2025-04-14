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
            n=generation_config.n,
            repetition_penalty=generation_config.repetition_penalty,
            temperature=generation_config.temperature,
            top_p=generation_config.top_p,
            top_k=generation_config.top_k,
            min_p=generation_config.min_p,
            max_tokens=generation_config.max_tokens,
            guided_decoding_regex=generation_config.guided_decoding_regex,
            stop=generation_config.stop
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
        
        # Duplicate prompts to match the requested number of generations (n)
        expanded_prompts = []
        for prompt in prompts:
            expanded_prompts.extend([prompt] * generation_config.n)
        
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
            stop=[self.stop_string] if generation_config.stop is None else 
                 generation_config.stop + [self.stop_string]
        )
        
        # Track current conversations for each prompt
        current_conversations = expanded_prompts.copy()
        completed_conversations = []
        
        # Continue code execution loop until all conversations are completed
        while current_conversations:
            # Generate next response segment
            outputs = vllm_client.generate(
                prompts=current_conversations,
                n=1,
                repetition_penalty=single_gen_config.repetition_penalty,
                temperature=single_gen_config.temperature,
                top_p=single_gen_config.top_p,
                top_k=single_gen_config.top_k,
                min_p=single_gen_config.min_p,
                max_tokens=single_gen_config.max_tokens,
                guided_decoding_regex=single_gen_config.guided_decoding_regex,
                stop=single_gen_config.stop
            )
            
            next_conversations = []
            code_batch = []
            code_conversations = []
            
            # Process outputs
            for conversation, output in zip(current_conversations, outputs):
                # Get the full conversation by combining original + generated text
                full_conversation = output["prompt"] + output["text"]
                
                # Check if code execution is needed
                if self.parsing_string in full_conversation and self.stop_string in full_conversation:
                    # Extract code for execution
                    code = self.extract_code(full_conversation)
                    if code:
                        code_batch.append(code)
                        code_conversations.append(full_conversation)
                else:
                    # No code to execute, conversation is complete
                    completed_conversations.append(full_conversation)
            
            # Execute code if needed
            if code_batch:
                execution_results = self.code_executor.execute(code_batch)
                
                # Add execution results to conversations
                for conversation, result in zip(code_conversations, execution_results):
                    updated_conversation = conversation + f"<output>{result}</output>"
                    next_conversations.append(updated_conversation)
            
            # Update current conversations for next iteration
            current_conversations = next_conversations
        
        # Return the final completed conversations
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