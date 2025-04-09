import uuid
import time
import logging
import threading
from typing import Any, Dict, List, Optional

import requests
import uvicorn
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from pydantic import BaseModel
from starlette.responses import JSONResponse


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
            agent_id = request.headers.get("X-Agent-ID", None)
            if not agent_id:
                raise HTTPException(status_code=400, detail="X-Agent-ID header not found")
            
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