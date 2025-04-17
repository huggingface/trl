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
        """Initialize the code agent environment"""
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
        """Run the agent with code execution and return completed text responses"""
        # Configure stop tokens to include the code stop string
        modified_gen_config = VLLMClientGenerationConfig(
            **{k: v for k, v in vars(generation_config).items()},
        )
        modified_gen_config.stop = [self.stop_string] if modified_gen_config.stop is None else list(set(modified_gen_config.stop + [self.stop_string]))
        
        # Handle multiple generations per prompt (n>1)
        expanded_prompts = []
        for prompt in prompts:
            expanded_prompts.extend([prompt] * generation_config.n)
        
        completed_conversations = []  # Fully completed conversations
        current_batch = expanded_prompts  # Current batch of prompts/conversations
        
        # Continue until all conversations are complete
        while current_batch:
            # Generate outputs for current batch
            outputs = vllm_client.generate(
                prompts=current_batch,
                n=1,  # We already expanded the prompts
                repetition_penalty=modified_gen_config.repetition_penalty,
                temperature=modified_gen_config.temperature,
                top_p=modified_gen_config.top_p,
                top_k=modified_gen_config.top_k,
                min_p=modified_gen_config.min_p,
                max_tokens=modified_gen_config.max_tokens,
                guided_decoding_regex=modified_gen_config.guided_decoding_regex,
                stop=modified_gen_config.stop
            )
            
            next_batch = []  # For conversations needing more processing
            code_batch = []  # Code snippets to execute
            conversations = []  # To track conversations for code execution
            
            # Process all outputs
            for prompt, output_ids in zip(current_batch, outputs):
                generated_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
                full_conversation = prompt + generated_text
                
                # Check if generation stopped at code block
                has_code_block = self.parsing_string in generated_text
                stopped_at_code = generated_text.endswith(self.stop_string.strip())
                
                if has_code_block and stopped_at_code:
                    # Extract code for execution
                    code = self.extract_code(generated_text)
                    if code:
                        code_batch.append(code)
                        conversations.append(full_conversation)
                    else:
                        # Code tags present but extraction failed
                        completed_conversations.append(full_conversation)
                else:
                    # No code block or didn't stop at code end - conversation is complete
                    completed_conversations.append(full_conversation)
            
            # Execute all code snippets in batch if any
            if code_batch:
                execution_results = self.code_executor.execute(code_batch)
                
                # Add results back to conversations and continue
                for conversation, result in zip(conversations, execution_results):
                    updated_conversation = conversation + f"<output>{result}</output>"
                    next_batch.append(updated_conversation)
            
            # Update current batch for next iteration
            current_batch = next_batch
        
        return completed_conversations
    
    def generate(self, vllm_client: Any, generation_config: VLLMClientGenerationConfig, prompts: List[str]) -> List:
        """Generate responses with code execution and return token IDs"""
        # Get completed text responses
        completed_conversations = self.run_agent(vllm_client, generation_config, prompts)
        
        # Recreate expanded prompts for proper extraction
        expanded_prompts = []
        for prompt in prompts:
            expanded_prompts.extend([prompt] * generation_config.n)
        
        # Extract completion IDs (just the generated part)
        completion_ids = []
        for original_prompt, final_output in zip(expanded_prompts, completed_conversations):
            completion_text = final_output[len(original_prompt):]
            completion_token_ids = self.tokenizer.encode(
                completion_text, 
                add_special_tokens=False
            )
            completion_ids.append(completion_token_ids)
        
        return completion_ids