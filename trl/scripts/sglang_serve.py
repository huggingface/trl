# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import base64
import logging
import os
import subprocess
import time
from collections.abc import Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from io import BytesIO
from itertools import chain
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection
from typing import Optional

import torch
from transformers import is_vision_available

from trl import TrlParser
from trl.import_utils import (
    is_fastapi_available,
    is_pydantic_available,
    is_requests_available,
    is_sglang_available,
    is_uvicorn_available,
)


if is_fastapi_available():
    from fastapi import FastAPI


if is_pydantic_available():
    from pydantic import BaseModel


if is_uvicorn_available():
    import uvicorn


if is_requests_available():
    import requests


if is_vision_available():
    from PIL import Image


if is_sglang_available():
    import sglang as sgl
    from sglang.srt.server_args import ServerArgs
    # We'll use subprocess to launch SGLang server similar to slime


logger = logging.getLogger(__name__)

# We use CUDA with multiprocessing, so we must use the 'spawn' start method
os.environ["SGLANG_WORKER_MULTIPROC_METHOD"] = "spawn"


class WeightSyncWorkerExtension:
    """
    An SGLang worker extension that enables weight synchronization between a client and multiple server workers.

    This worker uses a distributed process group to establish communication and handle efficient GPU-based
    communication using NCCL. The primary purpose is to receive updated model weights from a client process
    and distribute them to all worker processes participating in model inference.
    """

    def __init__(self):
        self.process_group = None
        self.rank = None
        self.world_size = None
        self.group_name = None

    def init_weights_update_group(self, master_address, master_port, rank_offset, world_size, group_name, backend):
        """
        Initialize the weight update communication group.

        Args:
            master_address (`str`): Master node hostname or IP address
            master_port (`int`): Port number for communication
            rank_offset (`int`): Offset to add to the rank
            world_size (`int`): Total number of participating processes
            group_name (`str`): Name of the process group
            backend (`str`): Backend to use (e.g., "nccl")
        """
        import torch.distributed as dist

        if not dist.is_initialized():
            dist.init_process_group(
                backend=backend,
                init_method=f"tcp://{master_address}:{master_port}",
                rank=rank_offset,
                world_size=world_size,
            )

        self.rank = rank_offset
        self.world_size = world_size
        self.group_name = group_name

    def update_weights_from_distributed(self, names, dtypes, shapes, group_name, flush_cache=False):
        """
        Receive updated weights from the client process and update the model.

        Args:
            names (`list[str]`): Names of the weight tensors
            dtypes (`list[str]`): Data types of the weight tensors
            shapes (`list[list[int]]`): Shapes of the weight tensors
            group_name (`str`): Name of the process group
            flush_cache (`bool`): Whether to flush the cache after updating weights
        """
        import torch.distributed as dist

        if group_name != self.group_name:
            raise ValueError(f"Group name mismatch: expected {self.group_name}, got {group_name}")

        weights = []
        for name, dtype_str, shape in zip(names, dtypes, shapes):
            dtype = getattr(torch, dtype_str.split(".")[-1])
            weight = torch.empty(shape, dtype=dtype, device="cuda")
            
            # Receive weight from the client (last rank)
            dist.broadcast(weight, src=self.world_size - 1)
            weights.append((name, weight))

        # Update model weights
        # This will be implemented in the actual SGLang integration
        # For now, we just store the weights
        self.updated_weights = weights

        if flush_cache:
            torch.cuda.empty_cache()


@dataclass
class ScriptArguments:
    r"""
    Arguments for the SGLang serve script.

    Args:
        model (`str`):
            Model name or path to load the model from.
        revision (`str` or `None`, *optional*, defaults to `None`):
            Revision to use for the model.
        tensor_parallel_size (`int`, *optional*, defaults to `1`):
            Number of tensor parallel workers to use.
        data_parallel_size (`int`, *optional*, defaults to `1`):
            Number of data parallel workers to use.
        host (`str`, *optional*, defaults to `"0.0.0.0"`):
            Host address to run the server on.
        port (`int`, *optional*, defaults to `8001`):
            Port to run the server on.
        gpu_memory_utilization (`float`, *optional*, defaults to `0.9`):
            Ratio of GPU memory to reserve for the model.
        dtype (`str`, *optional*, defaults to `"auto"`):
            Data type to use for SGLang generation.
        max_model_len (`int` or `None`, *optional*, defaults to `None`):
            Maximum model length to use.
        enable_prefix_caching (`bool` or `None`, *optional*, defaults to `None`):
            Whether to enable prefix caching in SGLang.
        enforce_eager (`bool`, *optional*, defaults to `False`):
            Whether to enforce eager execution.
        trust_remote_code (`bool`, *optional*, defaults to `False`):
            Whether to trust remote code when loading models.
        log_level (`str`, *optional*, defaults to `"info"`):
            Log level for uvicorn.
    """

    model: str = field(
        metadata={"help": "Model name or path to load the model from."},
    )
    revision: Optional[str] = field(
        default=None,
        metadata={"help": "Revision to use for the model."},
    )
    tensor_parallel_size: int = field(
        default=1,
        metadata={"help": "Number of tensor parallel workers to use."},
    )
    data_parallel_size: int = field(
        default=1,
        metadata={"help": "Number of data parallel workers to use."},
    )
    host: str = field(
        default="0.0.0.0",
        metadata={"help": "Host address to run the server on."},
    )
    port: int = field(
        default=8001,
        metadata={"help": "Port to run the server on."},
    )
    gpu_memory_utilization: float = field(
        default=0.9,
        metadata={
            "help": "Ratio of GPU memory to reserve for the model weights, activations, and KV cache."
        },
    )
    dtype: str = field(
        default="auto",
        metadata={
            "help": "Data type to use for SGLang generation."
        },
    )
    max_model_len: Optional[int] = field(
        default=None,
        metadata={
            "help": "Maximum model length to use."
        },
    )
    enable_prefix_caching: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether to enable prefix caching in SGLang."
        },
    )
    enforce_eager: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to enforce eager execution."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": "Whether to trust remote code when loading models."
        },
    )
    log_level: str = field(
        default="info",
        metadata={
            "help": "Log level for uvicorn."
        },
    )


def sglang_worker(
    script_args: ScriptArguments, data_parallel_rank: int, master_port: int, connection: Connection
) -> None:
    """
    Worker process for SGLang engine.
    
    Args:
        script_args: Configuration arguments
        data_parallel_rank: Rank of this worker in data parallel group
        master_port: Port for distributed communication
        connection: Pipe connection to parent process
    """
    # Set environment variables for data parallelism
    os.environ["SGLANG_DP_RANK"] = str(data_parallel_rank)
    os.environ["SGLANG_DP_SIZE"] = str(script_args.data_parallel_size)
    os.environ["SGLANG_DP_MASTER_PORT"] = str(master_port)

    # Prepare server arguments
    server_kwargs = {
        "model_path": script_args.model,
        "trust_remote_code": script_args.trust_remote_code,
        "tp_size": script_args.tensor_parallel_size,
        "dp_size": script_args.data_parallel_size,
        "host": script_args.host,
        "port": script_args.port + data_parallel_rank,  # Different port for each DP rank
        "mem_fraction_static": script_args.gpu_memory_utilization,
        "dtype": script_args.dtype,
        "context_length": script_args.max_model_len,
        "enable_prefix_caching": script_args.enable_prefix_caching,
        "disable_cuda_graph": script_args.enforce_eager,
    }

    # Filter out None values
    server_kwargs = {k: v for k, v in server_kwargs.items() if v is not None}
    
    # Create ServerArgs
    server_args = ServerArgs(**server_kwargs)
    
    # Create worker extension
    worker_extension = WeightSyncWorkerExtension()
    
    # Launch server using subprocess (similar to slime's approach)
    # Build command line arguments for SGLang server
    cmd = [
        "python", "-m", "sglang.launch_server",
        "--model-path", server_kwargs["model_path"],
        "--host", server_kwargs["host"],
        "--port", str(server_kwargs["port"]),
        "--tp-size", str(server_kwargs["tp_size"]),
    ]
    
    # Add optional arguments
    if "mem_fraction_static" in server_kwargs:
        cmd.extend(["--mem-fraction-static", str(server_kwargs["mem_fraction_static"])])
    if "dtype" in server_kwargs and server_kwargs["dtype"] != "auto":
        cmd.extend(["--dtype", server_kwargs["dtype"]])
    if "context_length" in server_kwargs:
        cmd.extend(["--context-length", str(server_kwargs["context_length"])])
    if server_kwargs.get("trust_remote_code"):
        cmd.append("--trust-remote-code")
    if server_kwargs.get("disable_cuda_graph"):
        cmd.append("--disable-cuda-graph")
    
    # Start the server process
    server_process = subprocess.Popen(cmd)

    # Wait for server to be ready
    base_url = f"http://{script_args.host}:{script_args.port + data_parallel_rank}"
    max_retries = 60
    for i in range(max_retries):
        try:
            response = requests.get(f"{base_url}/health")
            if response.status_code == 200:
                break
        except:
            pass
        time.sleep(2)
        if not server_process.is_alive():
            raise RuntimeError("SGLang server process died during startup")
    else:
        raise RuntimeError(f"SGLang server failed to start after {max_retries * 2} seconds")

    # Send ready signal
    connection.send({"status": "ready", "url": base_url})

    # Main loop to handle commands
    while True:
        try:
            command = connection.recv()
        except KeyboardInterrupt:
            break

        if command["type"] == "init_communicator":
            worker_extension.init_weights_update_group(**command["kwargs"])
            connection.send({"status": "ok"})
        elif command["type"] == "update_weights":
            worker_extension.update_weights_from_distributed(**command["kwargs"])
            # Forward weight updates to SGLang server
            response = requests.post(
                f"{base_url}/update_weights_from_distributed",
                json=command["kwargs"]
            )
            connection.send({"status": "ok", "response": response.json()})
        elif command["type"] == "generate":
            # Forward generation request to SGLang server
            response = requests.post(
                f"{base_url}/generate",
                json=command["kwargs"]
            )
            connection.send(response.json())
        elif command["type"] == "flush_cache":
            response = requests.get(f"{base_url}/flush_cache")
            connection.send({"status": "ok"})
        elif command["type"] == "shutdown":
            break

    # Cleanup
    server_process.terminate()
    server_process.wait()


def chunk_list(lst: list, n: int) -> list[list]:
    """Split list into n evenly distributed sublists."""
    k, r = divmod(len(lst), n)
    return [lst[i * k + min(i, r) : (i + 1) * k + min(i + 1, r)] for i in range(n)]


def main(script_args: ScriptArguments):
    if not is_fastapi_available():
        raise ImportError(
            "FastAPI is required to run the SGLang serve script. Please install it using `pip install fastapi`."
        )

    if not is_pydantic_available():
        raise ImportError(
            "Pydantic is required to run the SGLang serve script. Please install it using `pip install pydantic`."
        )

    if not is_uvicorn_available():
        raise ImportError(
            "Uvicorn is required to run the SGLang serve script. Please install it using `pip install uvicorn`."
        )

    if not is_sglang_available():
        raise ImportError("SGLang is required to run the SGLang serve script. Please install it using `pip install sglang`.")

    # Spawn data parallel workers
    master_port = 29500  # Fixed port for DP communication
    connections = []
    processes = []
    worker_urls = []
    
    for data_parallel_rank in range(script_args.data_parallel_size):
        parent_connection, child_connection = Pipe()
        process = Process(target=sglang_worker, args=(script_args, data_parallel_rank, master_port, child_connection))
        process.start()
        connections.append(parent_connection)
        processes.append(process)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Wait for all workers to be ready
        for connection in connections:
            msg = connection.recv()
            if msg.get("status") == "ready":
                worker_urls.append(msg["url"])
            else:
                raise RuntimeError(f"Worker failed to start: {msg}")

        yield

        # Shutdown workers
        for connection in connections:
            connection.send({"type": "shutdown"})
        
        # Wait for processes to terminate
        for process in processes:
            process.join(timeout=10)
            if process.is_alive():
                logger.warning(f"Process {process} is still alive, terminating...")
                process.terminate()
                process.join()

    app = FastAPI(lifespan=lifespan)

    # Define endpoints
    @app.get("/health/")
    async def health():
        """Health check endpoint."""
        return {"status": "ok"}

    @app.get("/get_world_size/")
    async def get_world_size():
        """Get the total world size."""
        return {"world_size": script_args.tensor_parallel_size * script_args.data_parallel_size}

    class GenerateRequest(BaseModel):
        prompts: list[str]
        images: Optional[list[str]] = None
        sampling_params: dict = field(default_factory=dict)

    class GenerateResponse(BaseModel):
        completion_ids: list[list[int]]

    @app.post("/generate/", response_model=GenerateResponse)
    async def generate(request: GenerateRequest):
        """Generate completions for the provided prompts."""
        # Distribute prompts across DP workers
        chunked_prompts = chunk_list(request.prompts, script_args.data_parallel_size)
        
        # Send to workers
        for connection, prompts in zip(connections, chunked_prompts):
            if not prompts:
                prompts = ["<placeholder>"]  # SGLang requires at least one prompt
            
            kwargs = {
                "text": prompts,
                "sampling_params": request.sampling_params,
            }
            connection.send({"type": "generate", "kwargs": kwargs})

        # Collect results
        all_outputs = []
        for connection, prompts in zip(connections, chunked_prompts):
            if prompts:  # Only collect from workers that had real prompts
                output = connection.recv()
                all_outputs.append(output)

        # Combine results
        completion_ids = []
        for output in all_outputs:
            for item in output.get("text_outputs", []):
                # Extract token IDs from the output
                # This will need adjustment based on SGLang's actual output format
                completion_ids.append(item.get("token_ids", []))

        return {"completion_ids": completion_ids}

    class InitCommunicatorRequest(BaseModel):
        host: str
        port: int
        world_size: int
        client_device_uuid: str

    @app.post("/init_communicator/")
    async def init_communicator(request: InitCommunicatorRequest):
        """Initialize the weight update communicator."""
        world_size = script_args.tensor_parallel_size * script_args.data_parallel_size + 1
        
        # Initialize communicator on all workers
        for i, connection in enumerate(connections):
            kwargs = {
                "master_address": request.host,
                "master_port": request.port,
                "rank_offset": i,
                "world_size": world_size,
                "group_name": "weight_sync",
                "backend": "nccl",
            }
            connection.send({"type": "init_communicator", "kwargs": kwargs})
        
        # Wait for all to complete
        for connection in connections:
            connection.recv()
        
        return {"message": "Communicator initialized"}

    class UpdateWeightsRequest(BaseModel):
        names: list[str]
        dtypes: list[str]
        shapes: list[list[int]]

    @app.post("/update_weights/")
    async def update_weights(request: UpdateWeightsRequest):
        """Update model weights."""
        kwargs = {
            "names": request.names,
            "dtypes": request.dtypes,
            "shapes": request.shapes,
            "group_name": "weight_sync",
            "flush_cache": True,
        }
        
        # Send to all workers
        for connection in connections:
            connection.send({"type": "update_weights", "kwargs": kwargs})
        
        # Wait for all to complete
        for connection in connections:
            connection.recv()
        
        return {"message": "Weights updated"}

    @app.post("/flush_cache/")
    async def flush_cache():
        """Flush the cache."""
        for connection in connections:
            connection.send({"type": "flush_cache"})
        
        for connection in connections:
            connection.recv()
        
        return {"message": "Cache flushed"}

    @app.post("/close_communicator/")
    async def close_communicator():
        """Close the weight update communicator."""
        # SGLang doesn't need explicit communicator closing
        return {"message": "Communicator closed"}

    # Start the server
    uvicorn.run(app, host=script_args.host, port=script_args.port, log_level=script_args.log_level)


def make_parser(subparsers: argparse._SubParsersAction = None):
    if subparsers is not None:
        parser = subparsers.add_parser("sglang-serve", help="Run the SGLang serve script", dataclass_types=ScriptArguments)
    else:
        parser = TrlParser(ScriptArguments)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    (script_args,) = parser.parse_args_and_config()
    main(script_args)