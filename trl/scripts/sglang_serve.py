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
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection
from typing import Optional

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
    pass


if is_vision_available():
    pass


if is_sglang_available():
    from ..extras.sglang_engine_adapter import SGLangEngine


logger = logging.getLogger(__name__)

# We use CUDA with multiprocessing, so we must use the 'spawn' start method
os.environ["SGLANG_WORKER_MULTIPROC_METHOD"] = "spawn"


# WeightSyncWorkerExtension is now handled by SGLangEngine in sglang_engine_adapter.py


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
        metadata={"help": "Ratio of GPU memory to reserve for the model weights, activations, and KV cache."},
    )
    dtype: str = field(
        default="auto",
        metadata={"help": "Data type to use for SGLang generation."},
    )
    max_model_len: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum model length to use."},
    )
    enable_prefix_caching: Optional[bool] = field(
        default=None,
        metadata={"help": "Whether to enable prefix caching in SGLang."},
    )
    enforce_eager: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to enforce eager execution."},
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Whether to trust remote code when loading models."},
    )
    log_level: str = field(
        default="info",
        metadata={"help": "Log level for uvicorn."},
    )


def sglang_worker(
    script_args: ScriptArguments, data_parallel_rank: int, master_port: int, connection: Connection
) -> None:
    """
    Worker process for SGLang engine using improved architecture.

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

    # Create SGLang engine using improved adapter
    port = script_args.port + data_parallel_rank
    nccl_port = master_port + 1000 + data_parallel_rank  # Separate NCCL ports
    dist_init_addr = f"{script_args.host}:{master_port}"

    # Add required attributes to script_args for SGLangEngine
    script_args.sglang_model_path = script_args.model
    script_args.sglang_host = script_args.host
    script_args.sglang_tensor_parallel_size = script_args.tensor_parallel_size
    script_args.sglang_data_parallel_size = script_args.data_parallel_size
    script_args.sglang_pipeline_parallel_size = 1
    script_args.sglang_expert_parallel_size = 1
    script_args.sglang_num_gpus_per_node = 8  # Default
    script_args.colocate = True
    script_args.offload = False

    try:
        # Create SGLang engine
        sglang_engine = SGLangEngine(
            args=script_args,
            rank=data_parallel_rank,
            dist_init_addr=dist_init_addr,
            port=port,
            nccl_port=nccl_port,
        )

        # Send ready signal
        connection.send({"status": "ready", "url": f"http://{script_args.host}:{port}"})

        # Main loop to handle commands
        while True:
            try:
                command = connection.recv()
            except KeyboardInterrupt:
                break

            if command["type"] == "init_communicator":
                sglang_engine.init_process_group(**command["kwargs"])
                connection.send({"status": "ok"})
            elif command["type"] == "init_weights_update_group":
                sglang_engine.init_process_group(**command["kwargs"])
                connection.send({"status": "ok"})
            elif command["type"] == "update_weights":
                sglang_engine.update_weights_from_distributed(**command["kwargs"])
                connection.send({"status": "ok"})
            elif command["type"] == "generate":
                result = sglang_engine.generate(**command["kwargs"])
                connection.send(result)
            elif command["type"] == "flush_cache":
                sglang_engine.reset_prefix_cache()
                connection.send({"status": "ok"})
            elif command["type"] == "shutdown":
                break

    except Exception as e:
        connection.send({"status": "error", "message": str(e)})
        raise
    finally:
        # Cleanup
        if "sglang_engine" in locals():
            sglang_engine.shutdown()


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
        raise ImportError(
            "SGLang is required to run the SGLang serve script. Please install it using `pip install sglang`."
        )

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

    class InitWeightsUpdateGroupRequest(BaseModel):
        master_address: str
        master_port: int
        rank_offset: int
        world_size: int
        group_name: str
        backend: str

    @app.post("/init_weights_update_group/")
    async def init_weights_update_group(request: InitWeightsUpdateGroupRequest):
        """Initialize the weight update group for distributed training."""
        kwargs = {
            "master_address": request.master_address,
            "master_port": request.master_port,
            "rank_offset": request.rank_offset,
            "world_size": request.world_size,
            "group_name": request.group_name,
            "backend": request.backend,
        }

        # Send to all workers
        for connection in connections:
            connection.send({"type": "init_weights_update_group", "kwargs": kwargs})

        # Wait for all to complete
        for connection in connections:
            connection.recv()

        return {"message": "Weight update group initialized"}

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

    class UpdateWeightsFromDistributedRequest(BaseModel):
        names: list[str]
        dtypes: list[str]
        shapes: list[list[int]]
        group_name: str = "weight_sync"
        flush_cache: bool = False

    @app.post("/update_weights_from_distributed/")
    async def update_weights_from_distributed(request: UpdateWeightsFromDistributedRequest):
        """Update model weights from distributed training using NCCL broadcast."""
        kwargs = {
            "names": request.names,
            "dtypes": request.dtypes,
            "shapes": request.shapes,
            "group_name": request.group_name,
            "flush_cache": request.flush_cache,
        }

        # Send to all workers
        for connection in connections:
            connection.send({"type": "update_weights", "kwargs": kwargs})

        # Wait for all to complete
        for connection in connections:
            connection.recv()

        return {"message": "Distributed weights updated"}

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
        parser = subparsers.add_parser(
            "sglang-serve", help="Run the SGLang serve script", dataclass_types=ScriptArguments
        )
    else:
        parser = TrlParser(ScriptArguments)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    (script_args,) = parser.parse_args_and_config()
    main(script_args)
