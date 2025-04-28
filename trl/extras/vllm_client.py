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

import atexit
import logging
import time
from typing import Optional

import torch
from torch import nn

from ..import_utils import is_requests_available, is_vllm_ascend_available, is_vllm_available

from ..trainer.grpo_config import GRPOConfig

if is_requests_available():
    import requests
    from requests import ConnectionError

if is_vllm_available():
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup
    from vllm import SamplingParams, LLM
    from vllm.sampling_params import GuidedDecodingParams
    if is_vllm_ascend_available():
        from vllm_ascend.distributed.device_communicators.pyhccl import PyHcclCommunicator as PyNcclCommunicator

from accelerate import Accelerator
from accelerate.utils import broadcast_object_list, gather_object


logger = logging.getLogger(__name__)

class VLLMNoOpClient:
    """
    A no-op vLLM client used in distributed training when the process is neither the main process
    nor running in vLLM colocation mode.

    This stub client ensures compatibility in distributed setups without performing actual
    inference or model updates. 

    Methods like `generate` and `update_named_param` are implemented as no-ops or return default
    values to maintain consistent interfaces across processes.

    This class should only be used internally by `get_vllm_client`.
    """

    def __init__(self, process_index: int):
        self.process_index = process_index

    def generate(
        self,
        prompts: list[str],
        n: int = 1,
        repetition_penalty: float = 1.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        min_p: float = 0.0,
        max_tokens: int = 16,
        guided_decoding_regex: Optional[str] = None,
    ) -> list[list[str]]:
        orig_size = len(prompts)
        prompts = gather_object(prompts) 
        completion_ids = [None] * len(prompts)
        return self._broadcast_and_slice(completion_ids, orig_size)

    def update_named_param(self, name: str, weights: torch.Tensor):
        pass

    def reset_prefix_cache(self):
        pass

    def _gather(self, prompts):
        return gather_object(prompts) 

    def _broadcast_and_slice(self, completion_ids: list, slice_size: int):
        # Broadcast the completions from the main process to all processes, ensuring each process receives its
        # corresponding slice

        completion_ids = broadcast_object_list(completion_ids, from_process=0)
        process_slice = slice(
            self.process_index * slice_size,
            (self.process_index + 1) * slice_size,
        )
        return completion_ids[process_slice]

class VLLMClient(VLLMNoOpClient):
    """
    A client class to interact with a vLLM server.

    This class provides methods to generate completions, initialize and manage weight update groups, and update model
    weights in a distributed setting. Before using it, start the vLLM server with `trl vllm-serve`.

    Args:
        host (`str`, *optional*, defaults to `"0.0.0.0"`):
            IP address of the vLLM server.
        server_port (`int`, *optional*, defaults to `8000`):
            Port number of the vLLM server.
        group_port (`int`, *optional*, defaults to `51216`):
            Port number for the weight update group.
        connection_timeout (`float`, *optional*, defaults to `0.0`):
            Total timeout duration in seconds to wait for the server to be up. If the server is not up after the
            timeout, a `ConnectionError` is raised.

    Examples:
        Run the vLLM server with the model `Qwen/Qwen2.5-7B`:

        ```
        $ trl vllm-serve --model Qwen/Qwen2.5-7B
        ...
        INFO:     Application startup complete.
        INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
        ```

        Use the client to generate completions and update model weights:

        ```python
        >>> from trl.extras.vllm_client import VLLMClient
        >>> client = VLLMClient()
        >>> client.generate(["Hello, AI!", "Tell me a joke"])
        [[2980, 498, 1492, 752, 448, 264, 13027, 8645, 30, 358, 2776, 4460, 311, 3270, 264, 2025],
         [911, 7988, 1251, 382, 3838, 653, 498, 1618, 4325, 879, 2581, 20027, 264, 21428, 30, 362]]

        >>> from transformers import AutoModelForCausalLM
        >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B", device_map="cuda")
        >>> client.init_communicator()
        >>> client.update_model_params(model)
        ```
    """

    def __init__(
        self, host: str = "0.0.0.0", server_port: int = 8000, group_port: int = 51216, connection_timeout: float = 0.0,
        distributed: bool = False
    ):
        super().__init__(process_index=0)

        if not is_requests_available():
            raise ImportError("requests is not installed. Please install it with `pip install requests`.")
        if not is_vllm_available():
            raise ImportError("vLLM is not installed. Please install it with `pip install vllm`.")

        self.distributed = distributed
        self.session = requests.Session()
        self.host = host
        self.server_port = server_port
        self.group_port = group_port
        self.check_server(connection_timeout)  # check server and fail after timeout

    def check_server(self, total_timeout: float = 0.0, retry_interval: float = 2.0):
        """
        Check server availability with retries on failure, within a total timeout duration. If the server is not up
        after the total timeout duration, raise a `ConnectionError`.

        Args:
            retry_interval (`float`, *optional*, defaults to `2.0`):
                Interval in seconds between retries.
            total_timeout (`float`, *optional*, defaults to `0.0`):
                Total timeout duration in seconds.
        """
        url = f"http://{self.host}:{self.server_port}/health/"
        start_time = time.time()  # Record the start time

        while True:
            try:
                response = requests.get(url)
            except requests.exceptions.RequestException as exc:
                # Check if the total timeout duration has passed
                elapsed_time = time.time() - start_time
                if elapsed_time >= total_timeout:
                    raise ConnectionError(
                        f"The vLLM server can't be reached at {self.host}:{self.server_port} after {total_timeout} "
                        "seconds. Make sure the server is running by running `trl vllm-serve`."
                    ) from exc
            else:
                if response.status_code == 200:
                    logger.info("Server is up!")
                    return None

            # Retry logic: wait before trying again
            logger.info(f"Server is not up yet. Retrying in {retry_interval} seconds...")
            time.sleep(retry_interval)

    def generate(
        self,
        prompts: list[str],
        n: int = 1,
        repetition_penalty: float = 1.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        min_p: float = 0.0,
        max_tokens: int = 16,
        guided_decoding_regex: Optional[str] = None,
    ) -> list[list[int]]:
        """
        Generates model completions for the provided prompts.

        Args:
            prompts (`list[str]`):
                List of text prompts for which the model will generate completions.
            n (`int`, *optional*, defaults to `1`):
                Number of completions to generate for each prompt.
            repetition_penalty (`float`, *optional*, defaults to `1.0`):
                Parameter for repetition penalty. 1.0 means no penalty.
            temperature (`float`, *optional*, defaults to `1.0`):
                Temperature parameter for sampling. Higher values increase diversity.
            top_p (`float`, *optional*, defaults to `1.0`):
                Top-p sampling parameter.`1.0` means no truncation.
            top_k (`int`, *optional*, defaults to `-1`):
                Top-k sampling parameter. `-1` means no truncation.
            min_p (`float`, *optional*, defaults to `0.0`):
                Minimum probability for sampling.
            max_tokens (`int`, *optional*, defaults to `16`):
                Maximum number of tokens to generate for each prompt.
            guided_decoding_regex (`str` or `None`, *optional*, defaults to `None`):
                Regular expression to guide the decoding process.

        Returns:
            `list[list[int]]`:
                List of lists of token IDs representing the model-generated completions for each prompt.
        """

        if self.distributed:
            orig_size = len(prompts)
            prompts = self._gather(prompts) 

        # Since 'prompts' contains 'num_generations' duplicates, we first take unique prompts, and generate
        # num_generations outputs for each one. This is faster than generating outputs for each duplicate
        # prompt individually 
        prompts = prompts[::n]

        url = f"http://{self.host}:{self.server_port}/generate/"
        response = self.session.post(
            url,
            json={
                "prompts": prompts,
                "n": n,
                "repetition_penalty": repetition_penalty,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "min_p": min_p,
                "max_tokens": max_tokens,
                "guided_decoding_regex": guided_decoding_regex,
            },
        )
        if response.status_code == 200:
            completion_ids = response.json()["completion_ids"]
        else:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

        if self.distributed:
            completion_ids = self._broadcast_and_slice(completion_ids, orig_size)

        return completion_ids

    def init_communicator(self):
        """
        Initializes the weight update group in a distributed setup for model synchronization.
        """
        # Get the world size from the server
        url = f"http://{self.host}:{self.server_port}/get_world_size/"
        response = requests.get(url)
        if response.status_code == 200:
            vllm_world_size = response.json()["world_size"]
        else:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

        world_size = vllm_world_size + 1  # add the client to the world
        self.rank = vllm_world_size  # the client's rank is the last process

        # Initialize weight update group
        url = f"http://{self.host}:{self.server_port}/init_communicator/"
        # In the server side, the host is set to 0.0.0.0
        response = self.session.post(url, json={"host": "0.0.0.0", "port": self.group_port, "world_size": world_size})
        if response.status_code != 200:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

        # Brief delay to allow server initialization. While not strictly required (client socket will retry on
        # connection failure), this prevents log warnings like:
        # [W416 23:24:57.460001114 socket.cpp:204] [c10d] The hostname of the client socket cannot be retrieved. err=-3
        time.sleep(0.1)

        # Set up the communication group for weight broadcasting
        pg = StatelessProcessGroup.create(host=self.host, port=self.group_port, rank=self.rank, world_size=world_size)
        self.pynccl_comm = PyNcclCommunicator(pg, device=0)

        # When the client object is deleted, close the weight update group
        atexit.register(self.close_communicator)

    def update_named_param(self, name: str, weights: torch.Tensor):
        """
        Updates a specific named parameter in the model and broadcasts it to other processes.

        Args:
            name (`str`):
                Name of the layer whose weights are being updated.
            weights (`torch.Tensor`):
                Tensor containing the updated weights.
        """
        dtype, shape = str(weights.dtype), tuple(weights.shape)
        url = f"http://{self.host}:{self.server_port}/update_named_param/"
        response = self.session.post(url, json={"name": name, "dtype": dtype, "shape": shape})
        if response.status_code != 200:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

        # Broadcast the weights to the other processes
        self.pynccl_comm.broadcast(weights, src=self.rank)
        self.pynccl_comm.group.barrier()

    def update_model_params(self, model: nn.Module):
        """
        Updates all parameters of the given model by calling `update_named_param` for each parameter in the model.

        Args:
            model (`nn.Module`):
                Model whose parameters (weights/biases) are to be updated.
        """
        for name, param in model.named_parameters():
            # Update each parameter individually
            self.update_named_param(name, param.data)

    def reset_prefix_cache(self):
        """
        Resets the prefix cache for the model.
        """
        url = f"http://{self.host}:{self.server_port}/reset_prefix_cache/"
        response = self.session.post(url)
        if response.status_code != 200:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

    def close_communicator(self):
        """
        Closes the weight update group and cleans up the communication group.
        """
        url = f"http://{self.host}:{self.server_port}/close_communicator/"

        try:
            response = self.session.post(url)
        except ConnectionError:
            # The server might be already down, so we don't need to close the communicator
            pass
        else:
            if response.status_code != 200:
                raise Exception(f"Request failed: {response.status_code}, {response.text}")

class VLLMColocationClient:
    """
    A client class to interact with vLLM processes colocated with the training process.

    This client bypasses remote communication and directly interacts with the in-process vLLM engine.
    It supports weight updates and text generation functionalities similar to `VLLMClient`, but is optimized
    for scenarios where vLLM is running in the same process or node as training.

    Args:
        args (`GRPOConfig`): Configuration object containing vLLM parameters.
        model (`transformers.PreTrainedModel`): The model being used.
        vllm_device (`torch.device` or `str`): Device on which the model is loaded (e.g., "cuda:0").
    """

    def __init__(self, args: GRPOConfig, model, vllm_device):
        self.args: GRPOConfig = args
        self.model = model
        self.vllm_device = vllm_device

        self.llm = LLM(
            model=self.model.name_or_path,
            device=self.vllm_device,
            gpu_memory_utilization=self.args.vllm_gpu_memory_utilization,
            dtype=self.args.vllm_dtype,
            max_model_len=self.args.vllm_max_model_len,
            distributed_executor_backend="external_launcher",
            seed=0
        )
        
    def update_named_param(self, name: str, weights: torch.Tensor):
        """
        Updates a specific named parameter in the model.

        Args:
            name (`str`):
                Name of the layer whose weights are being updated.
            weights (`torch.Tensor`):
                Tensor containing the updated weights.
        """
        llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
        llm_model.load_weights([(name,weights)])

    def generate(
        self,
        prompts: list[str],
        n: int = 1,
        repetition_penalty: float = 1.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        min_p: float = 0.0,
        max_tokens: int = 16,
        guided_decoding_regex: Optional[str] = None,
    ) -> list[list[str]]:
        """
        Generates model completions for the provided prompts.

        Args:
            prompts (`list[str]`):
                List of text prompts for which the model will generate completions.
            n (`int`, *optional*, defaults to `1`):
                Number of completions to generate for each prompt.
            repetition_penalty (`float`, *optional*, defaults to `1.0`):
                Parameter for repetition penalty. 1.0 means no penalty.
            temperature (`float`, *optional*, defaults to `1.0`):
                Temperature parameter for sampling. Higher values increase diversity.
            top_p (`float`, *optional*, defaults to `1.0`):
                Top-p sampling parameter.`1.0` means no truncation.
            top_k (`int`, *optional*, defaults to `-1`):
                Top-k sampling parameter. `-1` means no truncation.
            min_p (`float`, *optional*, defaults to `0.0`):
                Minimum probability for sampling.
            max_tokens (`int`, *optional*, defaults to `16`):
                Maximum number of tokens to generate for each prompt.
            guided_decoding_regex (`str` or `None`, *optional*, defaults to `None`):
                Regular expression to guide the decoding process.

        Returns:
            `list[list[int]]`:
                List of lists of token IDs representing the model-generated completions for each prompt.
        """
        # Guided decoding, if enabled
        if guided_decoding_regex is not None:
            guided_decoding = GuidedDecodingParams(backend="outlines", regex=guided_decoding_regex)
        else:
            guided_decoding = None

        sampling_params = SamplingParams(
            n=1, # vLLM on each GPU generates only 1 in vllm_colocation mode
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            max_tokens=max_tokens,
            guided_decoding=guided_decoding,
        )
        
        all_outputs = self.llm.generate(
            prompts, sampling_params=sampling_params, use_tqdm=False
        )
        completion_ids = [output.token_ids for outputs in all_outputs for output in outputs.outputs]
        return completion_ids

    def reset_prefix_cache(self):
        """
        Resets the prefix cache for the model.
        """
        self.llm.reset_prefix_cache()

def get_vllm_client(args: GRPOConfig, model, accelerator: Accelerator) -> VLLMNoOpClient:
    """
    Returns the appropriate vLLM client based on the current configuration.

    This function acts as a proxy to initialize and return the correct vLLM client type:
    - If colocation is enabled, it returns `VLLMColocationClient`, which interacts directly with 
      the colocated vLLM process for faster integration.
    - If running in the main process (non-colocated mode), it returns `VLLMClient`, which communicates
      with an external vLLM server.
    - If not the main process and colocation is disabled, it returns a base client (`VLLMNoOpClient`)
      for compatibility in distributed settings.

    Args:
        args (`GRPOConfig`): Configuration object containing flags for colocation, server host, port, etc.
        model (`transformers.PreTrainedModel`): The model to use, passed only for the colocated client.
        accelerator (`Accelerator`): Hugging Face `Accelerator` object that helps with multi-GPU training.
    """
    if args.vllm_colocation:
        return VLLMColocationClient(args, model, accelerator.device)
    elif accelerator.is_main_process:
        vllm_client = VLLMClient(
            args.vllm_server_host, args.vllm_server_port, connection_timeout=args.vllm_server_timeout,
            distributed=accelerator.num_processes > 1,
        )
        vllm_client.init_communicator()
        return vllm_client
    return VLLMNoOpClient(accelerator.process_index)

# Example usage for VLLMCLient
if __name__ == "__main__":
    from vllm import SamplingParams

    client = VLLMClient()
    client.init_communicator()

    # Generate completions
    responses = client.generate(["Hello, AI!", "Tell me a joke"], n=4, max_tokens=32, sampling_params=SamplingParams())
    print("Responses:", responses)  # noqa

    # Update model weights
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B").to("cuda")
    client.update_model_params(model)

