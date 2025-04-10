import os
import uuid
import logging
import multiprocessing
from abc import ABC, abstractmethod
from typing import Any, Dict, List
from contextlib import redirect_stdout, redirect_stderr

import requests
from aider.coders import Coder
from aider.io import InputOutput
from aider.models import Model

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
        self.vllm_url = vllm_url
    
    @abstractmethod
    def deploy(self, prompts: List[Dict[str, Any]], timeout: int = 300) -> List[List[Dict[str, str]]]:
        """
        Deploy parallel agents to process the given prompts, returning histories.
        """
        ...
        
class ApptainerAgentManager(AgentManager):
    """Agent manager that uses apptainer to parallelize agent deployments."""
    def __init__(self, vllm_url: str = "http://localhost:8000"):
        super().__init__(vllm_url)

class MultiProcessAgentManager(AgentManager):
    """Agent manager that uses multiple processes to parallelize agent deployments."""
    def __init__(self, vllm_url: str = "http://localhost:8000"):
        """
        Initialize the agent manager.
        
        Args:
            vllm_url: URL of the vLLM server (including protocol, host and port)
        """
        super().__init__(vllm_url)
        
    @abstractmethod
    def process_one(self, data: Dict[str, Any]):
        """Process a single prompt and return a completion"""
        ...
    
    def _process_one_wrapper(self, agent_id: str, data: Dict[str, Any], shared_conversations: Dict) -> None:
        """
        Calls the abstract .process_one() while adding conversation tracking
        
        Args:
            agent_id: Unique identifier for this agent
            data: Data containing the prompt and any additional information
            shared_conversations: Multiprocessing-safe shared dictionary for conversations
        """        
        # Monkey patch requests to track conversations
        original_request = requests.requestest
        
        def patched_request(method, url, **kwargs):
            logger.info("MONKEY REQUESTS")
            # Make the original request
            response = original_request(method, url, **kwargs)
            
            # If it was a chat completion request, and the response is ok, track the conversation
            if url.endswith("/v1/chat/completions") or url.endswith("/v1/messages") and response.status_code == 200:
                request_data = kwargs.get("json", {})
                response_data = response.json()
                
                # Update conversation tracker
                shared_conversations[agent_id] = (request_data, response_data)
            
            return response
    
        requests.request = patched_request
        
        try:
            return self.process_one(data)
        finally:
            # Restore original request functions
            requests.request = original_request
    
    def _get_histories_from_shared(self, agent_ids: List[str], shared_conversations: Dict) -> List[List[Dict[str, str]]]:
        """Get conversation histories from the shared dictionary. Uses agent_ids to match the order of the prompts."""
        histories = []
        for agent_id in agent_ids:
            if agent_id in shared_conversations:
                latest_request, latest_response = shared_conversations[agent_id]
                messages = latest_request["messages"] + [{'role': 'assistant', 'content': latest_response["choices"][0]["message"]["content"]}]
                histories.append(messages)
            else:
                # Handle case where agent didn't produce a conversation
                logger.warning(f"Agent {agent_id} did not produce a conversation")
                histories.append([])
        
        return histories
            
    def deploy(self, prompts: List[Dict[str, Any]], timeout: int = 300) -> List[List[Dict[str, str]]]:
        """
        Deploy parallel agents to process the given prompts, returning histories.
        
        Args:
            prompts: List of prompts to process
            timeout: Maximum time in seconds to wait for all prompts to complete
            
        Returns:
            List of histories (same order as prompts)
        """
        # Generate agent IDs for tracking
        agent_ids = [str(uuid.uuid4()) for _ in range(len(prompts))]
        
        # Create a manager for sharing data between processes
        with multiprocessing.Manager() as manager:
            # Shared dictionary for conversations
            shared_conversations = manager.dict()
            
            # Process all prompts in parallel
            with multiprocessing.Pool() as pool:
                # Start async processing of all prompts
                result = pool.starmap_async(
                    self._process_one_wrapper, 
                    [(agent_id, prompt, shared_conversations) for agent_id, prompt in zip(agent_ids, prompts)]
                )
                
                try:
                    # Wait for results with timeout
                    result.get(timeout=timeout)
                except multiprocessing.TimeoutError:
                    logger.warning(f"Agent timeout reached after {timeout} seconds.")
            
            # Get histories using the helper function
            return self._get_histories_from_shared(agent_ids, shared_conversations)

class MultiProcessAider(MultiProcessAgentManager):
    """Example implementation for Aider coding agents"""
    
    def __init__(self, vllm_url: str = "http://localhost:8000"):
        super().__init__(vllm_url)
        # os.environ["OPENAI_API_BASE"] = self.vllm_url  # Aider uses this
        # os.environ["OPENAI_API_KEY"] = "dummy-key"
    
    def process_one(self, prompt: Dict[str, Any]):
        """Process a single prompt and return a completion"""
        
        assert "repo" in prompt and "base_commit" in prompt and "problem_statement" in prompt, "Data should contain repo, base_commit and problem_statement"
        
        temp_folder = None  # Initialize to avoid UnboundLocalError
        original_dir = os.getcwd()  # Save current directory
        
        try:
            # Clone the repo into a temporary folder
            temp_folder = clone_repo_at_commit(prompt["repo"], prompt["base_commit"])
            
            # Change to the repo's root directory so Aider can compute the repo-map
            os.chdir(temp_folder)
            
            # Redirect Aider's terminal output to the void
            with open(os.devnull, 'w') as devnull, redirect_stdout(devnull), redirect_stderr(devnull):
                coder = Coder.create(
                    main_model = Model("haiku"),  # Just a placeholder, the spoofed OPENAI_API_BASE will make it use vLLM
                    io = InputOutput(yes=True)
                )
                coder.run(prompt["problem_statement"])
                
        
        finally:
            if temp_folder: clean_repo_dir(temp_folder)
            os.chdir(original_dir)
            
class ApptainerAider(ApptainerAgentManager):
    """Agent manager that uses apptainer containers to parallelize agent deployments."""
    ...
