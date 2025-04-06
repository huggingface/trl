import os
import logging
import multiprocessing
from functools import partial
from abc import ABC, abstractmethod
from typing import Any, Dict, List
from contextlib import redirect_stdout, redirect_stderr

from aider.aider import Coder, InputOutput, Model

from .git_utils import clone_repo_at_commit, clean_repo_dir

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

    @abstractmethod
    def _process_prompt(self, prompt: Dict[str, Any]) -> str:
        """Process a single prompt and return a completion"""
        ...
    
    def deploy(self, prompts: List[Dict[str, Any]], timeout: int = 300) -> List[str]:
        """
        Deploy parallel agents to process the given prompts, returning completion IDs.
        
        Args:
            prompts: List of prompts to process
            timeout: Maximum time in seconds to wait for all prompts to complete
            
        Returns:
            List of completion IDs that can be used by GRPO
        """
        with multiprocessing.Pool() as pool:
            # Start async processing of all prompts
            result = pool.map_async(partial(self._process_prompt), prompts)  # partial for pickling reasons
            
            try:
                # Wait for results with timeout
                completions = result.get(timeout=timeout)
                return completions
            except multiprocessing.TimeoutError:
                # Log warning and return partial results if timeout occurs
                logger.warning(f"Agent timeout reached after {timeout} seconds. Returning partial results.")
                # Get whatever results are ready
                return result._value if hasattr(result, '_value') else []
        

class AiderAgentManager(AgentManager):
    """Example implementation for Aider coding agents"""
    
    def __init__(self, vllm_url: str = "http://localhost:8000"):
        super().__init__(vllm_url)
    
        os.environ["OPENAI_API_BASE"] = self.vllm_url  # Aider uses this
        os.environ["OPENAI_API_KEY"] = "dummy-key"
    
    def _process_prompt(self, prompt: Dict[str, Any]) -> str:
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

            return "TODO, get the actual completion history"  # TODO: actually get the completion history
        finally:
            clean_repo_dir(temp_folder)
            os.chdir(original_dir)