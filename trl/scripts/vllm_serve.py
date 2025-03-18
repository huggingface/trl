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
import os
from dataclasses import dataclass, field
from typing import List, Sequence

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


# We use CUDA with multiprocessing, so we must use the 'spawn' start method. Otherwise, we will get the following
# error: RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use
# the 'spawn' start method
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


class InitWeightUpdateGroupRequest(BaseModel):
    host: str
    port: int
    world_size: int


class GenerateRequest(BaseModel):
    prompts: List[str]


class GenerateResponse(BaseModel):
    completion_ids: list[list[int]]


class UpdateWeightsRequest(BaseModel):
    name: str
    dtype: str
    shape: List[int]


class WeightSyncWorker(Worker):
    """
    vLLM worker that allows for weight synchronization between the client and the server.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_update_group = None

    def init_weight_update_group(self, host, port, world_size) -> None:
        # Get the rank of the current worker
        rank = get_world_group().rank

        # Create a stateless process group. vLLM provides `StatelessProcessGroup` to create a process group
        # without considering the global process group in torch.distributed. It is recommended to create
        # `StatelessProcessGroup`, and then initialize the data-plane communication (NCCL) between external
        # (train processes) and vLLM workers.
        pg = StatelessProcessGroup.create(host=host, port=port, rank=rank, world_size=world_size)
        self.model_update_group = PyNcclCommunicator(pg, device=self.device)

        # The src corresponds to the rank of the client that sends the updated weights.
        self.src = world_size - 1

    def update_weights(self, name: str, dtype: torch.dtype, shape: Sequence[int]) -> None:
        if self.model_update_group is None:
            raise RuntimeError("Weight update group not initialized. Call init_weight_update_group first.")

        # Create an empty tensor with the specified dtype and shape. It will be used to receive the updated weights.
        weight = torch.empty(shape, dtype=dtype, device=self.device)

        # Broadcast the updated weights from the client to all workers.
        self.model_update_group.broadcast(weight, src=self.src, stream=torch.cuda.current_stream())

        # Load the updated weights to the model and delete the weight tensor.
        self.model_runner.model.load_weights(weights=[(name, weight)])
        del weight


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
    @app.post("/generate/", response_model=GenerateResponse)
    def generate(request: GenerateRequest):
        all_outputs = llm.generate(request.prompts)
        completion_ids = [list(output.token_ids) for outputs in all_outputs for output in outputs.outputs]
        return {"completion_ids": completion_ids}

    @app.post("/init_weight_update_group/")
    def init_weight_update_group(request: InitWeightUpdateGroupRequest, background_tasks: BackgroundTasks):
        background_tasks.add_task(
            llm.collective_rpc, "init_weight_update_group", args=(request.host, request.port, request.world_size)
        )
        return {"message": "Request received, initializing weight update group"}

    @app.post("/update_weights/")
    def update_weights(request: UpdateWeightsRequest, background_tasks: BackgroundTasks):
        # The function is called this way: update_weight(name="name", dtype=torch.float32, shape=(10, 10))
        # So with collect_rpc we need to call it this way:
        # llm.collective_rpc("update_weight", args=("name", torch.float32, (10, 10)))
        # And with background_tasks.add_task we need to call it this way:
        # background_tasks.add_task(llm.collective_rpc, "update_weight", args=("name", torch.float32, (10, 10)))
        dtype = torch.__getattribute__(request.dtype.split(".")[-1])
        background_tasks.add_task(llm.collective_rpc, "update_weights", args=(request.name, dtype, request.shape))
        return {"message": "Request received, initializing weight update group"}

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
