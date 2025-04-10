import time
import logging
from typing import Optional
import uuid

import torch
import torch.distributed as dist

import uvicorn
from pydantic import BaseModel
from fastapi import BackgroundTasks, FastAPI

from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ModelCard,
    ModelList,
    ChatMessage,
    ChatCompletionResponseChoice,
    UsageInfo
)


logger = logging.getLogger(__name__)


app = FastAPI()

# Define the endpoints for the model server
@app.get("/health/")
async def health():
    """
    Health check endpoint to verify that the server is running.
    """
    return {"status": "ok"}

@app.get("/v1/models")
async def list_models():
    """
    OpenAI-compatible models endpoint that returns the available models.
    """
    model = ModelCard(
        id="Qwen/Qwen2.5-Coder-32B-Instruct",
        object="model",
        created=int(time.time()),
        owned_by="user",
    )
    return ModelList(data=[model], object="list")

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def openai_chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint.
    
    This endpoint emulates the OpenAI API format while using vLLM for generation.
    """
    # if request.stream if hasattr(request, 'stream') else False:
    #     logger.warning("Streaming mode requested but not supported in this implementation.")
        
    # if request.tools if hasattr(request, 'tools') else None:
    #     logger.warning("Tools requested but not supported in this implementation.")
        
    response = ChatCompletionResponse(
        id=str(uuid.uuid4()),
        object="chat.completion",
        created=int(time.time()),
        model="Qwen/Qwen2.5-Coder-32B-Instruct",
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content="To remove all the logging crap in test.py. First add a comment to the top of the file to disable all logging. Then, remove all the logging calls in the file."),
                finish_reason="stop"
            )
        ],
        usage=UsageInfo(prompt_tokens=0, total_tokens=0, completion_tokens=0)
    )
    
    print(request.stream)
    print("########################")
    print(request.messages[0])
    print("########################")
    print(request.messages[1])
    print("########################")
    print(response)
    return response

# Start the server
uvicorn.run(app, host="0.0.0.0", port=8000)

dist.destroy_process_group()