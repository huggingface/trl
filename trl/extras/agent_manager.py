import os
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import requests
import concurrent.futures


logger = logging.getLogger(__name__)


class AgentManager(ABC):
    """
    Base class for agent managers that coordinate ephemeral agents during GRPO training.

    The AgentManager replaces vllm_client.generate() in the GRPO trainer by deploying
    agents that process prompts and return completions. Each agent exists only for the
    duration of a single training example.
    """
    def __init__(self, vllm_url: str = "http://localhost:8000/v1"):
        self.vllm_url = vllm_url
    
    @abstractmethod
    def deploy(
        self,
        data: List[Dict[str, Any]],  # not just prompts, e.g. for SWE-GYM related tasks we could use repo / commit_hash / problem_statement etc.
        timeout: int = 300,
        # sampling params
        repetition_penalty: float = 1.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        min_p: float = 0.0,
        max_tokens: int = 16,
        guided_decoding_regex: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Deploy parallel agents to process the given data, returns the data after adding the answers.
        """
        ...
        
class SimpleAgentManager(AgentManager):
    """
    Behaves the same as the synchronous, batched, vllm_client.generate() except it uses the asynchronous
    vllm_serve_openai_compatible API.
    """
    def deploy(
        self,
        data: List[Dict[str, Any]],
        timeout: int = 300,
        # sampling params
        repetition_penalty: float = 1.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        min_p: float = 0.0,
        max_tokens: int = 16,
        guided_decoding_regex: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Deploy parallel agents to process the given data, returns the data after adding the answers.
        """
        url = f"{self.vllm_url}/chat/completions"
        headers = {"Authorization": "Bearer dummy"}

        def get_answer(item):
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": item["prompt"]}
            ]
            payload = {
                "model": "deployed_model",
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "repetition_penalty": repetition_penalty,
                "top_p": top_p,
                "top_k": top_k,
                "min_p": min_p,
                "stream": False
            }
            if guided_decoding_regex is not None:
                payload["guided_decoding_regex"] = guided_decoding_regex
                
            resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
            resp.raise_for_status()
            resp_data = resp.json()
            return resp_data["choices"][0]["message"]["content"]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(get_answer, item) for item in data]
            for item, future in zip(data, futures):
                item["answer"] = future.result()

        return data
        