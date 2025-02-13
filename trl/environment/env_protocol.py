from typing import Any, Dict, List, Protocol


class Environment(Protocol):
    """
    A protocol describing the minimal interface needed for integration 
    with the trainer. Your environment can run any multi-step logic, 
    but must ultimately return token sequences akin to selecting token_ids from 
    vllm.LLM's generate() output. https://docs.vllm.ai/en/stable/api/offline_inference/llm.html

    See the GRPOTrainer 
    [docs](https://github.com/huggingface/trl/blob/main/docs/source/grpo_trainer.md#Customization) 
    for an example implementation.
    """
    def generate(self, prompts: List[List[Dict[str, Any]]], llm: Any, sampling_params: Any) -> List[Any]:
        ...
