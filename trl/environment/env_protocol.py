from typing import Any, Dict, List, Protocol

class Environment(Protocol):
    llm: Any
    processing_class: Any
    sampling_params: Any
    def generate(self, prompts: List[List[Dict[str, Any]]]) -> List[Any]:
        ...
