import os
import uuid
import time
import logging
import threading
import multiprocessing
from functools import partial
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from contextlib import redirect_stdout, redirect_stderr

import requests
import uvicorn
from fastapi import FastAPI, Request, BackgroundTasks
from pydantic import BaseModel
from starlette.responses import JSONResponse

from aider.aider import Coder, InputOutput, Model

from .git_utils import clone_repo_at_commit, clean_repo_dir

logger = logging.getLogger(__name__)


class ChatMessage(BaseModel):
    role: str
    content: str
    
class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int = int(time.time())
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


class ConversationTracker:
    """Tracks conversation history for a single agent across multiple API calls"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.exchanges = []  # List of (request, response) pairs
        self.token_counts = {"prompt": 0, "completion": 0}
        self.last_activity = time.time()
    
    def add_exchange(self, request: Dict, response: Dict):
        """Add a request-response pair to the conversation history"""
        self.exchanges.append((request, response))
        self.last_activity = time.time()
        
        # Track approximate token counts
        if "usage" in response:
            self.token_counts["prompt"] += response["usage"].get("prompt_tokens", 0)
            self.token_counts["completion"] += response["usage"].get("completion_tokens", 0)
            
    def is_self_contained(self) -> bool:
        """Check if it always holds true that request_t + response_t is in request_t+1"""
        ...
    
    def get_completion_history(self) -> str:
        """Return the last request and response, log an error if it is not self-contained"""
        ...

    @property
    def is_inactive(self, timeout_seconds=120):
        """Check if this conversation has been inactive for a while"""
        return (time.time() - self.last_activity) > timeout_seconds

# TODO: no reason to restart this proxy every time
class APIProxy:
    """
    A proxy server that intercepts API calls between agents and the vLLM server,
    capturing the full conversation history.
    """
    
    def __init__(self, vllm_url: str, host: str = "127.0.0.1", port: int = 0):
        """
        Initialize the API proxy.
        
        Args:
            vllm_url: URL of the vLLM server
            host: Host to bind the proxy to
            port: Port to bind the proxy to (0 means find an available port)
        """
        self.vllm_url = vllm_url
        self.host = host
        self.port = self._find_free_port() if port == 0 else port
        self.url = f"http://{host}:{self.port}"
        
        self.app = FastAPI(title="API Proxy")
        self.setup_routes()
        
        self.conversations = {}  # agent_id -> ConversationTracker
        self.lock = threading.RLock()  # For thread-safe access to conversations
    
    def _find_free_port(self) -> int:
        """Find an available port to bind to"""
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            return s.getsockname()[1]
    
    def setup_routes(self):
        """Set up the FastAPI routes"""
        
        @self.app.post("/v1/chat/completions")
        async def chat_completions(request: Request, background_tasks: BackgroundTasks):
            """Handle chat completion requests"""
            # Extract agent ID from headers (or generate a new one)
            agent_id = request.headers.get("X-Agent-ID", str(uuid.uuid4()))
            
            # Get the request body
            body = await request.json()
            
            # Forward the request to vLLM
            response = requests.post(
                f"{self.vllm_url}/v1/chat/completions",
                json=body,
                headers={"Content-Type": "application/json"}
            )
            
            # Extract the response data
            response_data = response.json()
            
            # Track the conversation
            background_tasks.add_task(self.track_conversation, agent_id, body, response_data)
            
            # Add a header with the agent ID for tracking
            response_headers = {"X-Agent-ID": agent_id}
            
            return JSONResponse(content=response_data, headers=response_headers)
    
    def track_conversation(self, agent_id: str, request: Dict, response: Dict):
        """Track a conversation exchange between an agent and vLLM"""
        with self.lock:
            if agent_id not in self.conversations:
                self.conversations[agent_id] = ConversationTracker(agent_id)
            
            self.conversations[agent_id].add_exchange(request, response)
    
    # TODO: we are relying on race conditions and "dictionary ordering"
    def get_conversation_histories(self) -> Dict[str, str]:
        """Get and reset the conversation histories for all agents (should be called after .deploy() finishes"""
        histories = {agent_id: self.conversations[agent_id].get_completion_history() for agent_id in self.conversations}
        self.conversations.clear()
        return histories
    
    def start(self):
        """Start the proxy server in a background thread"""
        self.server_thread = threading.Thread(
            target=uvicorn.run,
            kwargs={
                "app": self.app,
                "host": self.host,
                "port": self.port,
                "log_level": "error"
            },
            daemon=True
        )
        self.server_thread.start()
        # Allow time for the server to start
        time.sleep(0.5)
        return self.url
    

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