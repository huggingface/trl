import os
import uuid
import logging
import threading
import multiprocessing
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple
from contextlib import redirect_stdout, redirect_stderr

import requests
from aider.coders import Coder
from aider.io import InputOutput
from aider.models import Model
from vllm.entrypoints.openai.protocol import ChatCompletionRequest, ChatCompletionResponse

from .git_utils import clone_repo_at_commit, clean_repo_dir

logger = logging.getLogger(__name__)


class AgentManager(ABC):
    """
    Base class for agent managers that coordinate ephemeral agents during GRPO training.
    
    The AgentManager replaces vllm_client.generate() in the GRPO trainer by deploying
    agents that process prompts and return completions. Each agent exists only for the
    duration of a single training example.
    """
    @abstractmethod
    def deploy(self, prompts: List[Dict[str, Any]], timeout: int = 300) -> List[List[Dict[str, str]]]:
        """
        Deploy parallel agents to process the given prompts, returning histories.
        """
        ...
        
class ApptainerAgentManager(AgentManager):
    """Agent manager that uses apptainer to parallelize agent deployments."""
    ...

class MultiProcessAgentManager(AgentManager):
    """Agent manager that uses multiple processes to parallelize agent deployments."""
    def __init__(self, vllm_url: str = "http://localhost:8000"):
        """
        Initialize the agent manager.
        
        Args:
            vllm_url: URL of the vLLM server (including protocol, host and port)
        """
        self.vllm_url = vllm_url
        self.conversations: Dict[str, Tuple[ChatCompletionRequest, ChatCompletionResponse]] = {}  # for now we assume that request T, contains request and response at T-1
        self.lock = threading.RLock()  # For thread-safe access to conversations
        
    @abstractmethod
    def process_one(self, data: Dict[str, Any]):
        """Process a single prompt and return a completion"""
        ...
        
    def _process_one(self, agent_id: str, data: Dict[str, Any]) -> None:
        """
        Calls the abstract .process_one() while adding conversation tracking
        
        Args:
            agent_id: Unique identifier for this agent
            data: Data containing the prompt and any additional information
        """        
        # Monkey patch requests to track conversations
        original_request = requests.request
        
        def patched_request(method, url, **kwargs):
            # Make the original request
            response = original_request(method, url, **kwargs)
            
            # If it was a chat completion request, and the response is ok, track the conversation
            if url.endswith("/v1/chat/completions") and response.status_code == 200:
                request_data = kwargs.get("json", {})
                response_data = response.json()
                
                # Update conversation tracker
                with self.lock:
                    self.conversations[agent_id] = (request_data, response_data)
            
            return response
        
        requests.request = patched_request
        
        try:
            return self.process_one(data)
        finally:
            # Restore original request function
            requests.request = original_request
    
    def _get_and_clear_histories(self) -> List[List[Dict[str, str]]]:
        """Get all conversation histories and clear the tracking data. Uses self.agent_ids to match the order of the prompts."""
        with self.lock:
            histories = []
            for agent_id in self.agent_ids:
                latest_request, latest_response = self.conversations[agent_id]
                messages = latest_request.messages + [{'role': 'assistant', 'content': latest_response.choices[0].message.content}]
                histories.append(messages)
            
            # Clear the conversations
            self.conversations = {}
            
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
        self.agent_ids = [str(uuid.uuid4()) for _ in range(len(prompts))]
        
        # Process all prompts in parallel
        with multiprocessing.Pool() as pool:
            # Start async processing of all prompts
            result = pool.starmap_async(self._process_one, [(agent_id, prompt) for agent_id, prompt in zip(self.agent_ids, prompts)])
            
            try:
                # Wait for results with timeout
                result.get(timeout=timeout)
            except multiprocessing.TimeoutError:
                logger.warning(f"Agent timeout reached after {timeout} seconds.")
        
        return self._get_and_clear_histories()

class MultiProcessAider(MultiProcessAgentManager):
    """Example implementation for Aider coding agents"""
    
    def __init__(self, vllm_url: str = "http://localhost:8000"):
        super().__init__(vllm_url)
    
        os.environ["OPENAI_API_BASE"] = self.vllm_url  # Aider uses this
        os.environ["OPENAI_API_KEY"] = "dummy-key"
    
    def process_one(self, prompt: Dict[str, Any]):
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
        
        finally:
            clean_repo_dir(temp_folder)
            os.chdir(original_dir)
            
class ApptainerAider(ApptainerAgentManager):
    """Agent manager that uses apptainer containers to parallelize agent deployments."""
    ...
