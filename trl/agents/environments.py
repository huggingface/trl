from dataclasses import dataclass 
from typing import List, Any, Optional

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

class Environment:
    """Base environment that implements standard VLLM generation"""
    
    def generate(
        self,
        vllm_client: Any,
        prompts: List[str],
        generation_config: VLLMClientGenerationConfig, 
    ) -> List:
        """Generate responses using VLLM

        Args:
            vllm_client: VLLM client instance
            prompts: Input prompts for generation
            generation_config: VLLM generation parameters
            
        Returns:
            completion_ids: Generated token ids
        """
        return vllm_client.generate(
            prompts=prompts,
            **vars(generation_config)
        )