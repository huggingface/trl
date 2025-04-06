import uuid
import os
import logging
import socket
import contextlib
import tempfile
import multiprocessing
from typing import Any, Dict, List, Optional, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass

import uvicorn
import requests
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from starlette.responses import JSONResponse

from .git_utils import clone_repo_at_commit
from .scripts.vllm_serve import OpenAIResponse

logger = logging.getLogger(__name__)


@dataclass
class AgentManagerConfig:
    ...

class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request format"""
    model: str
    messages: List[Dict[str, str]]
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    
    
class AgentConversation:
    """Tracks a conversation history for a single agent"""
    def __init__(self):
        self.requests = []
        self.responses = []
        
    def add_exchange(self, request: Dict[str, Any], response: Dict[str, Any]):
        """Add a request-response pair to the conversation history"""
        self.requests.append(request)
        self.responses.append(response)
        
        # Check for context discontinuity
        if len(self.requests) > 1:
            prev_messages = self.requests[-2].get("messages", [])
            curr_messages = request.get("messages", [])
            
            # Simple heuristic: check if previous context is substantially contained in new context
            prev_content = " ".join([m.get("content", "") for m in prev_messages])
            curr_content = " ".join([m.get("content", "") for m in curr_messages])
            
            if len(prev_content) > 100 and prev_content not in curr_content:
                logger.warning(
                    "Possible context discontinuity detected: previous context not found in current request."
                )
    
    @property
    def completion_ids(self) -> List[str]:
        """Extract completion IDs from responses"""
        return [
            response.get("id", f"completion-{i}")
            for i, response in enumerate(self.responses)
        ]


def _find_free_port() -> int:
    """Find a free port to use for an agent endpoint"""
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def _run_endpoint(port: int, agent_id: str, vllm_url: str, queue: multiprocessing.Queue):
    """Run a FastAPI endpoint in a separate process"""
    app = FastAPI(title=f"Agent {agent_id}")
    conversation = AgentConversation()
    
    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest, background_tasks: BackgroundTasks):
        # Forward request to vLLM server
        vllm_response = requests.post(
            f"{vllm_url}/v1/chat/completions",
            json=request.dict(),
        ).json()
        
        # Record in conversation history
        conversation.add_exchange(request.dict(), vllm_response)
        
        # Send updates to the conversation history queue
        queue.put(("exchange", agent_id, request.dict(), vllm_response))
        
        return JSONResponse(content=vllm_response)
    
    # Start the server
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="error")


class AgentManager(ABC):
    """
    Base class for agent managers that coordinate ephemeral agents during GRPO training.
    
    The AgentManager replaces vllm_client.generate() in the GRPO trainer by deploying
    agents that process prompts and return completions. Each agent exists only for the
    duration of a single training example.
    """
    
    def __init__(self, vllm_port: int = 8000, vllm_host: str = "localhost"):
        """
        Initialize the agent manager.
        
        Args:
            vllm_port: Port of the vLLM server
            vllm_host: Host of the vLLM server
        """
        self.vllm_host = vllm_host
        self.vllm_port = vllm_port
        self.vllm_url = f"http://{vllm_host}:{vllm_port}"
        self.active_processes = {}  # {agent_id: (process, port, queue)}
        self.conversations = {}     # {agent_id: AgentConversation}
        
        # Create a multiprocessing manager for shared state
        self.mp_manager = multiprocessing.Manager()
        
        # Setup shared communication queue
        self.queue = self.mp_manager.Queue()
        
        # Start queue listener process
        self.listener_process = multiprocessing.Process(
            target=self._process_queue, 
            args=(self.queue,)
        )
        self.listener_process.daemon = True
        self.listener_process.start()
    
    def _process_queue(self, queue):
        """Process messages from the agents"""
        while True:
            try:
                message = queue.get()
                if message[0] == "exchange":
                    _, agent_id, request, response = message
                    if agent_id in self.conversations:
                        self.conversations[agent_id].add_exchange(request, response)
                elif message[0] == "exit":
                    break
            except Exception as e:
                logger.error(f"Error processing queue message: {e}")
    
    def _create_endpoint(self, agent_id: str) -> int:
        """
        Create a FastAPI endpoint for an agent that proxies to vLLM and records history.
        
        Args:
            agent_id: Unique ID for the agent
            
        Returns:
            Port number of the created endpoint
        """
        port = _find_free_port()
        
        # Create conversation tracker
        self.conversations[agent_id] = AgentConversation()
        
        # Start server in a separate process
        queue = self.mp_manager.Queue()
        process = multiprocessing.Process(
            target=_run_endpoint,
            args=(port, agent_id, self.vllm_url, self.queue)
        )
        process.daemon = True
        process.start()
        
        # Store process reference
        self.active_processes[agent_id] = (process, port, queue)
        
        # Wait a moment for the server to start
        process.join(0.5)
        if not process.is_alive():
            raise RuntimeError(f"Failed to start endpoint for agent {agent_id}")
        
        return port
    
    def _cleanup_endpoint(self, agent_id: str):
        """Stop and clean up an endpoint"""
        if agent_id in self.active_processes:
            process, _, _ = self.active_processes[agent_id]
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
                if process.is_alive():
                    process.kill()
            del self.active_processes[agent_id]
    
    def _cleanup_all_endpoints(self):
        """Stop and clean up all endpoints"""
        for agent_id in list(self.active_processes.keys()):
            self._cleanup_endpoint(agent_id)
    
    @abstractmethod
    def deploy(self, prompts: List[Dict[str, Any]]) -> List[str]:
        """
        Deploy agents to process the given prompts, returning completion IDs.
        
        Args:
            prompts: List of prompts to process
            
        Returns:
            List of completion IDs that can be used by GRPO
        """
        pass
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup"""
        self._cleanup_all_endpoints()
        # Signal the listener to exit
        if self.listener_process.is_alive():
            self.queue.put(("exit",))
            self.listener_process.join(timeout=2)
            if self.listener_process.is_alive():
                self.listener_process.terminate()
    
    def get_conversation_history(self, agent_id: str) -> AgentConversation:
        """Get the conversation history for an agent"""
        return self.conversations.get(agent_id)
    
    def get_completion_ids(self) -> List[str]:
        """Get all completion IDs from all conversations"""
        result = []
        for conversation in self.conversations.values():
            result.extend(conversation.completion_ids)
        return result


class AiderAgentManager(AgentManager):
    """Example implementation for Aider coding agents"""
    
    def deploy(self, prompts: List[Dict[str, Any]]) -> List[str]:
        ...
 