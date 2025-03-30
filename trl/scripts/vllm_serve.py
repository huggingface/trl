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
import asyncio
import torch
import threading
import torch.distributed as dist

from trl import TrlParser
from transformers import AutoTokenizer
from multiprocessing import Queue, Lock
from dataclasses import dataclass, field
from typing import Optional, Sequence, Union, Any
from trl.import_utils import is_fastapi_available, is_pydantic_available, is_uvicorn_available, is_vllm_available


if is_fastapi_available():
    from fastapi import BackgroundTasks, FastAPI


if is_pydantic_available():
    from pydantic import BaseModel


if is_uvicorn_available():
    import uvicorn


if is_vllm_available():
    from vllm import LLM, SamplingParams, AsyncLLMEngine
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.parallel_state import get_world_group
    from vllm.distributed.utils import StatelessProcessGroup
    from vllm.sampling_params import GuidedDecodingParams
    from vllm.worker.worker import Worker
    from vllm.engine.arg_utils import (AsyncEngineArgs, HfOverrides, PoolerConfig, TaskOption)
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.usage.usage_lib import UsageContext
    from vllm.utils import (Counter, Device, deprecate_args, deprecate_kwargs, is_list_of)
    
else:
    Worker = object

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
        if not is_vllm_available():
            raise ImportError(
                "vLLM is required to use the WeightSyncWorker. Please install it using `pip install vllm`."
            )

        super().__init__(*args, **kwargs)

        # The following attributes are initialized when `init_communicator` method is called.
        self.pynccl_comm = None  # Communicator for weight updates
        self.client_rank = None  # Source rank for broadcasting updated weights

    def init_communicator(self, host: str, port: int, world_size: int) -> None:
        """
        Initializes the weight update communicator using a stateless process group.

        This method creates a `StatelessProcessGroup` that allows external training processes to
        communicate with vLLM workers without interfering with the global torch distributed group.

        Args:
            host (`str`):
                Hostname or IP address of the master node.
            port (`int`):
                Port number to be used for communication.
            world_size (`int`):
                Total number of participating processes in the update group.
        """
        if self.pynccl_comm is not None:
            raise RuntimeError("Weight update group already initialized. Call close_communicator first.")

        # Get the rank of the current worker in the global world group.
        rank = get_world_group().rank

        # Create a stateless process group to manage communication between training processes and vLLM workers.
        pg = StatelessProcessGroup.create(host=host, port=port, rank=rank, world_size=world_size)

        # Initialize the NCCL-based communicator for weight synchronization.
        self.pynccl_comm = PyNcclCommunicator(pg, device=self.device)

        # The client process that sends updated weights has the highest rank (world_size - 1).
        self.client_rank = world_size - 1

    def update_named_param(self, name: str, dtype: torch.dtype, shape: Sequence[int]) -> None:
        """
        Receives updated weights from the client process and updates the named parameter in the model.

        Args:
            name (`str`):
                Name of the weight tensor being updated.
            dtype (`torch.dtype`):
                Data type of the weight tensor (e.g., `torch.float32`).
            shape (`Sequence[int]`):
                Shape of the weight tensor.
        """
        if self.pynccl_comm is None:
            raise RuntimeError("Communicator not initialized. Call `init_communicator` first.")

        # Allocate memory for the incoming weight tensor on the correct device.
        weight = torch.empty(shape, dtype=dtype, device=self.device)

        # Use NCCL to broadcast the updated weights from the client (src) to all workers.
        self.pynccl_comm.broadcast(weight, src=self.client_rank, stream=torch.cuda.current_stream())
        self.pynccl_comm.group.barrier()

        # Load the received weights into the model.
        self.model_runner.model.load_weights(weights=[(name, weight)])

    def close_communicator(self) -> None:
        """
        Closes the communicator when weight synchronization is no longer needed.

        This method deletes the NCCL communicator to release associated resources.
        """

        if self.pynccl_comm is not None:
            del self.pynccl_comm
            self.pynccl_comm = None  # Ensure attribute is reset to None
            self.client_rank = None  # Ensure attribute is reset to None


class RolloutEngine():
    def __init__(self, async_llm_engine, tokenizer):
        self.async_llm_engine = async_llm_engine
        self.working_tasks: dict[asyncio.Task] = {}
        self.task = self.run_async_in_new_thread_and_new_event_loop(self.print_working_tasks)
        self.tokenizer = tokenizer

    @staticmethod
    def run_async_in_new_thread_and_new_event_loop(func, *args, **kwargs):
        """ Run an async function in a new thread and a new event loop.
        """
        def job():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            task = loop.create_task(func(*args, **kwargs))
            loop.run_until_complete(task)
        threading.Thread(target=job, daemon=True).start()

    async def print_working_tasks(self):
        """ print number of working tasks every second
        """
        while True:
            print(f"Async working tasks counter: {len(self.working_tasks.keys())}")
            await asyncio.sleep(1)

    async def get_output(self, results_generator):
        """ get async output from async generator
        """
        final_output = None
        async for request_output in results_generator:
            final_output = request_output
            # print(request_output.outputs[0].text[-1])
        return final_output

    async def submit_llm_generate(self, prompts, sampling_params):
        """ submit llm generate task
        """
        import uuid
        event_loop = asyncio.get_event_loop()
        tasks = []
        for prompt in prompts:
            tasks.append(event_loop.create_task(self.get_output(self.async_llm_engine.generate(
                prompt=prompt,
                sampling_params=sampling_params,
                request_id=uuid.uuid4().hex,
                priority=0
            ))))
        all_outputs = await asyncio.gather(*tasks)
        return all_outputs

    async def submit_new_task(self, req_json, prompts, sampling_params, multi_round):
        """
        create async task
        """
        if req_json in self.working_tasks: return {"completion_ids": []}
        event_loop = asyncio.get_event_loop()
        task = event_loop.create_task(self.llm_rollout_task_main(prompts=prompts, sampling_params=sampling_params, multi_round=multi_round))
        self.working_tasks[req_json] = task

    async def get_future(self, req_json):
        """
        return async task result
        """
        # check cache
        assert req_json in self.working_tasks
        # wait until complete
        while (not self.working_tasks[req_json].done()):
            await asyncio.sleep(0.2)
        completion_ids = self.working_tasks.pop(req_json).result()
        return {"completion_ids": completion_ids}

    def running_task_count(self):
        """
        return number of running tasks
        """
        return len(self.working_tasks.keys())

    async def llm_rollout_task_main(self, prompts, sampling_params, multi_round):
        """ main function for llm rollout task
        """
        if not multi_round:
            all_outputs = await self.submit_llm_generate(prompts=prompts, sampling_params=sampling_params)
            completion_ids = [list(output.token_ids) for outputs in all_outputs for output in outputs.outputs]
        else:
            raise NotImplementedError("Multi-round rollout with tool-call is coming soon.")
        return completion_ids


@dataclass
class ScriptArguments:
    r"""
    Arguments for the script.

    Args:
        model (`str`):
            Model name or path to load the model from.
        revision (`str` or `None`, *optional*, defaults to `None`):
            Revision to use for the model. If not specified, the default branch will be used.
        tensor_parallel_size (`int`, *optional*, defaults to `1`):
            Number of tensor parallel workers to use.
        host (`str`, *optional*, defaults to `"0.0.0.0"`):
            Host address to run the server on.
        port (`int`, *optional*, defaults to `8000`):
            Port to run the server on.
        gpu_memory_utilization (`float`, *optional*, defaults to `0.9`):
            Ratio (between 0 and 1) of GPU memory to reserve for the model weights, activations, and KV cache on the
            device dedicated to generation powered by vLLM. Higher values will increase the KV cache size and thus
            improve the model's throughput. However, if the value is too high, it may cause out-of-memory (OOM) errors
            during initialization.
        dtype (`str`, *optional*, defaults to `"auto"`):
            Data type to use for vLLM generation. If set to `"auto"`, the data type will be automatically determined
            based on the model configuration. Find the supported values in the vLLM documentation.
        max_model_len (`int` or `None`, *optional*, defaults to `None`):
            If set, the `max_model_len` to use for vLLM. This can be useful when running with reduced
            `vllm_gpu_memory_utilization`, leading to a reduced KV cache size. If not set, vLLM will use the model
            context size, which might be much larger than the KV cache, leading to inefficiencies.
        enable_prefix_caching (`bool` or `None`, *optional*, defaults to `None`):
            Whether to enable prefix caching in vLLM. If set to `True`, ensure that the model and the hardware support
            this feature.
    """

    model: str = field(metadata={"help": "Model name or path to load the model from."})
    revision: Optional[str] = field(
        default=None,
        metadata={"help": "Revision to use for the model. If not specified, the default branch will be used."},
    )
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
    gpu_memory_utilization: float = field(
        default=0.9,
        metadata={
            "help": "Ratio (between 0 and 1) of GPU memory to reserve for the model weights, activations, and KV "
            "cache on the device dedicated to generation powered by vLLM. Higher values will increase the KV cache "
            "size and thus improve the model's throughput. However, if the value is too high, it may cause "
            "out-of-memory (OOM) errors during initialization."
        },
    )
    dtype: str = field(
        default="auto",
        metadata={
            "help": "Data type to use for vLLM generation. If set to 'auto', the data type will be automatically "
            "determined based on the model configuration. Find the supported values in the vLLM documentation."
        },
    )
    max_model_len: Optional[int] = field(
        default=None,
        metadata={
            "help": "If set, the `max_model_len` to use for vLLM. This can be useful when running with reduced "
            "`vllm_gpu_memory_utilization`, leading to a reduced KV cache size. If not set, vLLM will use the model "
            "context size, which might be much larger than the KV cache, leading to inefficiencies."
        },
    )
    enable_prefix_caching: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether to enable prefix caching in vLLM. If set to `True`, ensure that the model and the "
            "hardware support this feature."
        },
    )


def main(script_args: ScriptArguments):
    if not is_fastapi_available():
        raise ImportError(
            "FastAPI is required to run the vLLM serve script. Please install it using `pip install fastapi`."
        )

    if not is_pydantic_available():
        raise ImportError(
            "Pydantic is required to run the vLLM serve script. Please install it using `pip install pydantic`."
        )

    if not is_uvicorn_available():
        raise ImportError(
            "Uvicorn is required to run the vLLM serve script. Please install it using `pip install uvicorn`."
        )

    if not is_vllm_available():
        raise ImportError("vLLM is required to run the vLLM serve script. Please install it using `pip install vllm`.")

    class AsyncVllmEngine(LLM):
        """This class is copied and modified from: python/site-packages/vllm/entrypoints/llm.py (Tested under vllm 0.8.2)
        """
        def __init__(
            self,
            model: str,
            tokenizer: Optional[str] = None,
            tokenizer_mode: str = "auto",
            skip_tokenizer_init: bool = False,
            trust_remote_code: bool = False,
            allowed_local_media_path: str = "",
            tensor_parallel_size: int = 1,
            dtype: str = "auto",
            quantization: Optional[str] = None,
            revision: Optional[str] = None,
            tokenizer_revision: Optional[str] = None,
            seed: Optional[int] = 42,
            gpu_memory_utilization: float = 0.9,
            swap_space: float = 4,
            cpu_offload_gb: float = 0,
            enforce_eager: Optional[bool] = None,
            max_seq_len_to_capture: int = 8192,
            disable_custom_all_reduce: bool = False,
            disable_async_output_proc: bool = False,
            hf_overrides = None,
            mm_processor_kwargs = None,
            # After positional args are removed, move this right below `model`
            task = "auto",
            override_pooler_config = None,
            compilation_config = None,
            **kwargs,
        ) -> None:

            if "disable_log_stats" not in kwargs:
                kwargs["disable_log_stats"] = True

            if "worker_cls" in kwargs:
                worker_cls = kwargs["worker_cls"]

            compilation_config_instance = None

            engine_args = AsyncEngineArgs(
                model=model,
                task=task,
                tokenizer=tokenizer,
                tokenizer_mode=tokenizer_mode,
                skip_tokenizer_init=skip_tokenizer_init,
                trust_remote_code=trust_remote_code,
                allowed_local_media_path=allowed_local_media_path,
                tensor_parallel_size=tensor_parallel_size,
                dtype=dtype,
                quantization=quantization,
                revision=revision,
                tokenizer_revision=tokenizer_revision,
                seed=seed,
                gpu_memory_utilization=gpu_memory_utilization,
                swap_space=swap_space,
                cpu_offload_gb=cpu_offload_gb,
                enforce_eager=enforce_eager,
                max_seq_len_to_capture=max_seq_len_to_capture,
                disable_custom_all_reduce=disable_custom_all_reduce,
                disable_async_output_proc=disable_async_output_proc,
                hf_overrides=hf_overrides,
                mm_processor_kwargs=mm_processor_kwargs,
                override_pooler_config=override_pooler_config,
                compilation_config=compilation_config_instance,
                **kwargs,
            )

            self.llm_engine = AsyncLLMEngine.from_engine_args(engine_args=engine_args, usage_context=UsageContext.LLM_CLASS)
            self.engine_class = type(self.llm_engine)

            self.request_counter = Counter()
            self.default_sampling_params: Union[dict[str, Any], None] = None

            self.async_llm_engine: AsyncLLMEngine  = self.llm_engine
            self.llm_engine = self.llm_engine.engine

        @staticmethod
        def get_engine_class():
            return AsyncLLMEngine

    llm = AsyncVllmEngine(
        model=script_args.model,
        revision=script_args.revision,
        tensor_parallel_size=script_args.tensor_parallel_size,
        gpu_memory_utilization=script_args.gpu_memory_utilization,
        dtype=script_args.dtype,
        # Automatic Prefix Caching caches the KV cache of existing queries, so that a new query can
        # directly reuse the KV cache if it shares the same prefix with one of the existing queries.
        # This is particularly useful here because we generate completions from the same prompts.
        enable_prefix_caching=script_args.enable_prefix_caching,
        max_model_len=script_args.max_model_len,
        worker_cls='trl.scripts.vllm_serve.WeightSyncWorker',
    )

    app = FastAPI()

    # Define the endpoints for the model server
    @app.get("/health/")
    async def health():
        """
        Health check endpoint to verify that the server is running.
        """
        return {"status": "ok"}

    @app.get("/get_tensor_parallel_size/")
    async def get_tensor_parallel_size():
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
        n: int = 1
        repetition_penalty: float = 1.0
        temperature: float = 1.0
        top_p: float = 1.0
        top_k: int = -1
        min_p: float = 0.0
        max_tokens: int = 16
        guided_decoding_regex: Optional[str] = None
        multi_round: Optional[bool] = False
        is_async: Optional[bool] = False
        version: Optional[int] = 0

    class GenerateResponse(BaseModel):
        completion_ids: list[list[int]]

    rollout_engine = RolloutEngine(async_llm_engine=llm.async_llm_engine, tokenizer=AutoTokenizer.from_pretrained(script_args.model))

    @app.post("/generate/", response_model=GenerateResponse)
    async def generate(request: GenerateRequest):
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

        # Guided decoding, if enabled
        if request.guided_decoding_regex is not None:
            guided_decoding = GuidedDecodingParams(backend="outlines", regex=request.guided_decoding_regex)
        else:
            guided_decoding = None

        # Sampling parameters
        sampling_params = SamplingParams(
            n=request.n,
            repetition_penalty=request.repetition_penalty,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            min_p=request.min_p,
            max_tokens=request.max_tokens,
            guided_decoding=guided_decoding,
        )
        multi_round = request.multi_round
        is_async = request.is_async
        # discard `is_async`, so as to unify the future query key
        request.is_async = None; req_json = request.model_dump_json()
        await rollout_engine.submit_new_task(
            req_json=req_json, 
            prompts=request.prompts, 
            sampling_params=sampling_params, 
            multi_round=multi_round
        )
        if is_async:
            return {"completion_ids": []}
        else:
            return await rollout_engine.get_future(req_json)

    @app.post("/get_future/", response_model=GenerateResponse)
    async def get_future(request: GenerateRequest):
        # discard `is_async`, so as to unify the future query key
        request.is_async = None; req_json = request.model_dump_json()
        return await rollout_engine.get_future(req_json)

    class InitCommunicatorRequest(BaseModel):
        host: str
        port: int
        world_size: int

    @app.post("/init_communicator/")
    async def init_communicator(request: InitCommunicatorRequest, background_tasks: BackgroundTasks):
        """
        Initializes the communicator for synchronizing model weights between a client and multiple server
        workers.

        Args:
            request (`InitCommunicatorRequest`):
                - `host` (`str`): Hostname or IP address of the master node.
                - `port` (`int`): Port number to be used for communication.
                - `world_size` (`int`): Total number of participating processes in the group.
        """
        background_tasks.add_task(
            llm.collective_rpc,
            "init_communicator",
            args=(request.host, request.port, script_args.tensor_parallel_size + 1),
        )
        return {"message": "Request received, initializing communicator"}

    class UpdateWeightsRequest(BaseModel):
        name: str
        dtype: str
        shape: list[int]

    @app.post("/update_named_param/")
    async def update_named_param(request: UpdateWeightsRequest, background_tasks: BackgroundTasks):
        """
        Updates the model weights with the provided tensor.

        Once this endpoint is called, the client process should broadcast the updated weights to all server workers.

        Args:
            request (`UpdateWeightsRequest`):
                - `name` (`str`): Name of the weight tensor being updated.
                - `dtype` (`str`): Data type of the weight tensor (e.g., `"torch.float32"`).
                - `shape` (list of `int`): Shape of the weight

        """
        # The function is called this way: update_named_param(name="name", dtype=torch.float32, shape=(10, 10))
        # So with collect_rpc we need to call it this way:
        # llm.collective_rpc("update_named_param", args=("name", torch.float32, (10, 10)))
        # And with background_tasks.add_task we need to call it this way:
        # background_tasks.add_task(llm.collective_rpc, "update_named_param", args=("name", torch.float32, (10, 10)))
        while (rollout_engine.running_task_count() > 0):
            print("[vllm-serve warning]: You cannot update weights while some generation tasks are running! Waiting for all tasks to finish.")
            await asyncio.sleep(1)

        dtype = torch.__getattribute__(request.dtype.split(".")[-1])
        background_tasks.add_task(llm.collective_rpc, "update_named_param", args=(request.name, dtype, request.shape))

        return {"message": "Request received, updating named parameter"}

    @app.post("/reset_prefix_cache/")
    async def reset_prefix_cache():
        """
        Resets the prefix cache for the model.
        """
        success = llm.llm_engine.reset_prefix_cache()
        return {"message": "Request received, resetting prefix cache status: " + str(success)}

    @app.post("/close_communicator/")
    async def close_communicator():
        """
        Closes the weight update group and cleans up associated resources.
        """
        llm.collective_rpc("close_communicator")
        return {"message": "Request received, closing communicator"}

    # Start the server
    uvicorn.run(app, host=script_args.host, port=script_args.port)

    dist.destroy_process_group()


def make_parser(subparsers: argparse._SubParsersAction = None):
    if subparsers is not None:
        parser = subparsers.add_parser("vllm-serve", help="Run the vLLM serve script", dataclass_types=ScriptArguments)
    else:
        parser = TrlParser(ScriptArguments)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    (script_args,) = parser.parse_args_and_config()
    main(script_args)
