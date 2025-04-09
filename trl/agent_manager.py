import os
import uuid
import logging
import multiprocessing
from functools import partial
from abc import ABC, abstractmethod
from typing import Any, Dict, List
from contextlib import redirect_stdout, redirect_stderr

import requests

from aider.coders import Coder
from aider.io import InputOutput
from aider.models import Model

from .git_utils import clone_repo_at_commit, clean_repo_dir
from .api_utils import APIProxy

logger = logging.getLogger(__name__)
    

class AgentManager(ABC):
    """
    Base class for agent managers that coordinate ephemeral agents during GRPO training.
    
    The AgentManager replaces vllm_client.generate() in the GRPO trainer by deploying
    agents that process prompts and return completions. Each agent exists only for the
    duration of a single training example.
    """
    
    def __init__(self, vllm_url: str = "http://localhost:8000"):
        """
        Initialize the agent manager.
        
        Args:
            vllm_url: URL of the vLLM server (including protocol, host and port)
        """
        self.vllm_url = vllm_url
        
        # Start the API proxy
        self.api_proxy = APIProxy(vllm_url)
        self.proxy_url = self.api_proxy.start()
        logger.info(f"API Proxy started at {self.proxy_url}")
        
    @abstractmethod
    def process_one(self, data: Dict[str, Any]) -> str:
        """Process a single prompt and return a completion"""
        ...
        
    def _process_one(self, agent_id: str, data: Dict[str, Any]) -> None:
        """
        Calls the abstract .process_one() while adding a custom 
        
        Args:
            agent_prompt_tuple: Tuple of (agent_id, prompt)
        """
        # Add a custom header for all requests from this process
        # Monkey patch requests to add our custom header
        # This happens inside a multiprocessing pool, so my best guess is that 
        original_request = requests.request
        
        def patched_request(method, url, **kwargs):
            headers = kwargs.get("headers", {})
            headers["X-Agent-ID"] = agent_id
            kwargs["headers"] = headers
            return original_request(method, url, **kwargs)
        
        requests.request = patched_request
        
        return self.process_one(data)
            
    def deploy(self, prompts: List[Dict[str, Any]], timeout: int = 300) -> List[str]:
        """
        Deploy parallel agents to process the given prompts, returning completion IDs.
        
        Args:
            prompts: List of prompts to process
            timeout: Maximum time in seconds to wait for all prompts to complete
            
        Returns:
            List of completion IDs that can be used by GRPO
        """
        # Generate agent IDs for tracking
        agent_ids = [str(uuid.uuid4()) for _ in range(len(prompts))]
        
        # Prepare prompts with agent IDs
        agent_prompts = [(agent_id, prompt) for agent_id, prompt in zip(agent_ids, prompts)]
        
        # Process all prompts in parallel
        with multiprocessing.Pool() as pool:
            # Start async processing of all prompts
            result = pool.map_async(partial(self._process_one), agent_prompts)  # partial for pickling reasons
            
            try:
                # Wait for results with timeout
                result.get(timeout=timeout)
            except multiprocessing.TimeoutError:
                # Log warning if timeout occurs
                logger.warning(f"Agent timeout reached after {timeout} seconds.")
        
        # Collect conversation histories from the proxy
        completions = []
        histories = self.api_proxy.get_conversation_histories()
        for agent_id in agent_ids:
            history = histories[agent_id]
            # For agents that didn't complete or had no history, use a placeholder
            if not history:
                raise ValueError(f"No history for agent {agent_id}")
           
            # Join all completions for this agent
            completions.append(history)  # TODO: probably better to return a {role:, content:} list
        
        return completions
    

class AiderAgentManager(AgentManager):
    """Example implementation for Aider coding agents"""
    
    def __init__(self, vllm_url: str = "http://localhost:8000"):
        super().__init__(vllm_url)
    
        os.environ["OPENAI_API_BASE"] = self.vllm_url  # Aider uses this
        os.environ["OPENAI_API_KEY"] = "dummy-key"
    
    def process_one(self, prompt: Dict[str, Any]) -> str:
        """Process a single prompt and return a completion"""
        
        assert "repo_url" in prompt and "repo_commit_hash" in prompt and "description" in prompt, "Data should contain repo_url, repo_commit_hash and description"
        
        try:
            # Clone the repo into a temporary folder
            temp_folder = clone_repo_at_commit(prompt["repo_url"], prompt["repo_commit_hash"])
            
            # Change to the repo's root directory so Aider can compute the repo-map
            original_dir = os.getcwd()
            os.chdir(temp_folder)
            
            # Redirect Aider's terminal output to the void
            with open(os.devnull, 'w') as devnull, redirect_stdout(devnull), redirect_stderr(devnull):
                coder = Coder.create(
                    main_model = Model("gpt-4o-mini"),  # Just a placeholder, the spoofed OPENAI_API_BASE will make it use vLLM
                    io = InputOutput(yes=True)
                )
                coder.run(prompt["description"])
            
            # The API proxy tracks all the exchanges, so we don't need to return anything here
            return ""
        
        finally:
            clean_repo_dir(temp_folder)
            os.chdir(original_dir)