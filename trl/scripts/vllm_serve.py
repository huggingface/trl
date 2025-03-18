# Copyright 2025 The HuggingFace Team. All rights reserved.
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
from dataclasses import dataclass, field
from typing import Sequence

import torch
import uvicorn
from fastapi import BackgroundTasks, FastAPI
from pydantic import BaseModel
from vllm import LLM
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.parallel_state import get_world_group
from vllm.distributed.utils import StatelessProcessGroup
from vllm.worker.worker import Worker

from trl import TrlParser


logger = logging.getLogger(__name__)

# We use CUDA with multiprocessing, so we must use the 'spawn' start method. Otherwise, we will get the following
# error: RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use
# the 'spawn' start method
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


class WeightSyncWorker(Worker):
    """
    A vLLM worker that enables weight synchronization between a client and multiple server workers.

    This worker uses a `StatelessProcessGroup` to establish communication and a `PyNcclCommunicator` to handle
    efficient GPU-based communication using NCCL. The primary purpose of this class is to receive updated model weights
    from a client process and distribute them to all worker processes participating in model inference.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # The following attributes are initialized when `init_weight_update_group` method is called.
        self.model_update_group = None  # Communicator for weight updates
        self.client_rank = None  # Source rank for broadcasting updated weights

    def init_weight_update_group(self, host: str, port: int, world_size: int) -> None:
        """
        Initializes the weight update communication group using a stateless process group.

        This method creates a `StatelessProcessGroup` that allows external training processes to
        communicate with vLLM workers without interfering with the global torch distributed group.

        Args:
            host (str): The hostname or IP address of the master node.
            port (int): The port number to be used for communication.
            world_size (int): The total number of participating processes in the update group.

        Raises:
            RuntimeError: If `get_world_group()` fails to retrieve the worker's rank.
        """
        if self.model_update_group is not None:
            raise RuntimeError("Weight update group already initialized. Call close_weight_update_group first.")

        # Get the rank of the current worker in the global world group.
        rank = get_world_group().rank

        # Create a stateless process group to manage communication between training processes and vLLM workers.
        pg = StatelessProcessGroup.create(host=host, port=port, rank=rank, world_size=world_size)

        # Initialize the NCCL-based communicator for weight synchronization.
        self.model_update_group = PyNcclCommunicator(pg, device=self.device)

        # The client process that sends updated weights has the highest rank (world_size - 1).
        self.client_rank = world_size - 1

    def update_weights(self, name: str, dtype: torch.dtype, shape: Sequence[int]) -> None:
        """
        Receives updated model weights from the client and applies them to the model.

        Args:
            name (`str`):
                Name of the weight tensor being updated.
            dtype (`torch.dtype`):
                Data type of the weight tensor (e.g., `torch.float32`).
            shape (`Sequence[int]`):
                Shape of the weight tensor.
        """

        if self.model_update_group is None:
            raise RuntimeError("Weight update group not initialized. Call `init_weight_update_group` first.")

        # Allocate memory for the incoming weight tensor on the correct device.
        weight = torch.empty(shape, dtype=dtype, device=self.device)

        # Use NCCL to broadcast the updated weights from the client (src) to all workers.
        self.model_update_group.broadcast(weight, src=self.client_rank, stream=torch.cuda.current_stream())

        # Load the received weights into the model.
        self.model_runner.model.load_weights(weights=[(name, weight)])

        # Explicitly delete the temporary weight tensor to free up memory.
        del weight

    def close_weight_update_group(self) -> None:
        """
        Cleans up the weight update communication group when weight synchronization is no longer needed.

        This method deletes the NCCL communicator to release associated resources.
        """

        if self.model_update_group is not None:
            del self.model_update_group
            self.model_update_group = None  # Ensure attribute is reset to None
            self.client_rank = None  # Ensure attribute is reset to None


@dataclass
class ScriptArguments:
    r"""
    Arguments for the script.

    Args:
        model (`str`):
            Model name or path to load the model from.
        tensor_parallel_size (`int`, *optional*, defaults to `1`):
            Number of tensor parallel workers to use.
        host (`str`, *optional*, defaults to `"0.0.0.0"`):
            Host address to run the server on.
        port (`int`, *optional*, defaults to `8000`):
            Port to run the server on.
    """

    model: str = field(metadata={"help": "Model name or path to load the model from."})
    tensor_parallel_size: int = field(
        default=1,
        metadata={"help": "Number of tensor parallel workers to use."},
    )
    host: str = field(
        default="0.0.0.0",
        metadata={"help": "Host address to run the server on."},
    )
    port: int = field(
        default=8000,
        metadata={"help": "Port to run the server on."},
    )


def main(script_args: ScriptArguments):
    llm = LLM(
        model=script_args.model,
        tensor_parallel_size=script_args.tensor_parallel_size,
        worker_cls=WeightSyncWorker,
    )

    app = FastAPI()

    # Define the endpoints for the model server
    @app.get("/get_tensor_parallel_size/")
    def get_tensor_parallel_size():
        """
        Retrieves the tensor parallel size from the LLM engine.

        Returns:
            `dict`:
                A dictionary containing the tensor parallel size.

        Example response:
        ```json
        {"tensor_parallel_size": 8}
        ```
        """
        return {"tensor_parallel_size": llm.llm_engine.parallel_config.tensor_parallel_size}

    class GenerateRequest(BaseModel):
        prompts: list[str]

    class GenerateResponse(BaseModel):
        completion_ids: list[list[int]]

    @app.post("/generate/", response_model=GenerateResponse)
    def generate(request: GenerateRequest):
        """
        Generates completions for the provided prompts.

        Args:
            request (`GenerateRequest`):
                - `prompts` (list of `str`): A list of prompts (text strings) for the model to generate completions.

        Returns:
            `GenerateResponse`:
                - `completion_ids` (list of list of `int`): A list of lists of token IDs for each generated completion.

        Example request:
        ```json
        {"prompts": ["Hello world", "What is AI?"]}
        ```

        Example response:
        ```json
        {"completion_ids": [[101, 102, 103], [201, 202, 203]]}
        ```
        """
        all_outputs = llm.generate(request.prompts)
        completion_ids = [list(output.token_ids) for outputs in all_outputs for output in outputs.outputs]
        return {"completion_ids": completion_ids}

    class InitWeightUpdateGroupRequest(BaseModel):
        host: str
        port: int
        world_size: int

    @app.post("/init_weight_update_group/")
    def init_weight_update_group(request: InitWeightUpdateGroupRequest, background_tasks: BackgroundTasks):
        """
        Initializes the weight update group for synchronizing model weights between a client and multiple server
        workers.

        Args:
            request (`InitWeightUpdateGroupRequest`):
                - `host` (`str`): Hostname or IP address of the master node.
                - `port` (`int`): Port number to be used for communication.
                - `world_size` (`int`): Total number of participating processes in the update group.
        """
        background_tasks.add_task(
            llm.collective_rpc,
            "init_weight_update_group",
            args=(request.host, request.port, script_args.tensor_parallel_size + 1),
        )
        return {"message": "Request received, initializing weight update group"}

    class UpdateWeightsRequest(BaseModel):
        name: str
        dtype: str
        shape: list[int]

    @app.post("/update_weights/")
    def update_weights(request: UpdateWeightsRequest, background_tasks: BackgroundTasks):
        """
        Updates the model weights with the provided tensor.

        Once this endpoint is called, the client process should broadcast the updated weights to all server workers.

        Args:
            request (`UpdateWeightsRequest`):
                - `name` (`str`): Name of the weight tensor being updated.
                - `dtype` (`str`): Data type of the weight tensor (e.g., `"torch.float32"`).
                - `shape` (list of `int`): Shape of the weight

        """
        # The function is called this way: update_weight(name="name", dtype=torch.float32, shape=(10, 10))
        # So with collect_rpc we need to call it this way:
        # llm.collective_rpc("update_weight", args=("name", torch.float32, (10, 10)))
        # And with background_tasks.add_task we need to call it this way:
        # background_tasks.add_task(llm.collective_rpc, "update_weight", args=("name", torch.float32, (10, 10)))
        dtype = torch.__getattribute__(request.dtype.split(".")[-1])
        background_tasks.add_task(llm.collective_rpc, "update_weights", args=(request.name, dtype, request.shape))
        return {"message": "Request received, initializing weight update group"}

    @app.post("/close_weight_update_group/")
    def close_weight_update_group():
        """
        Closes the weight update group and cleans up associated resources.
        """
        llm.collective_rpc("close_weight_update_group")
        return {"message": "Request received, closing weight update group"}

    # Start the server
    uvicorn.run(app, host=script_args.host, port=script_args.port)


def make_parser(subparsers: argparse._SubParsersAction = None):
    dataclass_types = ScriptArguments
    if subparsers is not None:
        parser = subparsers.add_parser("vllm-serve", help="Run the vLLM serve script", dataclass_types=dataclass_types)
    else:
        parser = TrlParser(dataclass_types)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    (script_args,) = parser.parse_args_and_config()
    main(script_args)
