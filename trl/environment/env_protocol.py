from typing import Any, List, Protocol


class Environment(Protocol):
    """
    A protocol describing the minimal interface needed for integration
    with the trainer. Your environment can run any multi-step logic,
    but must ultimately return token sequences akin to selecting token_ids from
    vllm.LLM's generate() output. https://docs.vllm.ai/en/stable/api/offline_inference/llm.html
    """

    def generate(self, vllm_inputs, processing_class, vlm, sampling_params) -> List[Any]: ...
