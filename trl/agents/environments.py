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

class Environment(abc.ABC):
    """Base environment that implements standard VLLM generation"""
    
    def __init__(self, vllm_client: Any):
        """Initialize environment with VLLM client
        
        Args:
            vllm_client: VLLM client instance
        """
        self.vllm_client = vllm_client
    
    @abc.abstractmethod
    def generate(
        self,
        prompts: List[str],
        generation_config: VLLMClientGenerationConfig, 
    ) -> List:
        """Generate responses using VLLM

        Args:
            prompts: Input prompts for generation
            generation_config: VLLM generation parameters
            
        Returns:
            completion_ids: Generated token ids
        """
        pass

class DefaultEnvironment(Environment):
    """Default environment that implements standard VLLM generation"""
    
    def generate(
        self,
        prompts: List[str],
        generation_config: VLLMClientGenerationConfig, 
    ) -> List:
        """Generate responses using VLLM

        Args:
            prompts: Input prompts for generation
            generation_config: VLLM generation parameters
            
        Returns:
            completion_ids: Generated token ids
        """
        return self.vllm_client.generate(
            prompts=prompts,
            **vars(generation_config)
        )