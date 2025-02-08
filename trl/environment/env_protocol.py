from typing import Any, Dict, List, Protocol

class Environment(Protocol):
    def generate(self,
                 prompts: List[List[Dict[str, Any]]],
                 llm: Any,
                 sampling_params: Any) -> List[Any]:
        ...
