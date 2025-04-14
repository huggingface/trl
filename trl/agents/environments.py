from dataclasses import dataclass 
from typing import List, Any, Optional
import abc

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