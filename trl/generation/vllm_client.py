# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
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
import base64
import copy
import logging
import socket
import time
from io import BytesIO
from urllib.parse import urlparse

import torch
import torch.distributed.distributed_c10d as c10d
from requests.adapters import HTTPAdapter
from torch import nn
from transformers import is_torch_xpu_available
from urllib3.util.retry import Retry

from ..import_utils import is_requests_available, is_vllm_ascend_available, is_vllm_available


if is_requests_available():
    import requests
    from requests import ConnectionError


if is_vllm_available():
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup

    if is_vllm_ascend_available():
        from vllm_ascend.distributed.device_communicators.pyhccl import PyHcclCommunicator as PyNcclCommunicator


logger = logging.getLogger(__name__)


def pil_to_base64(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    img_bytes = buffer.getvalue()
    return base64.b64encode(img_bytes).decode("utf-8")


class VLLMClient:
    """
    A client class to interact with a vLLM server.

    This class provides methods to generate completions, initialize and manage weight update groups, and update model
    weights in a distributed setting. Before using it, start the vLLM server with `trl vllm-serve`.

    Args:
        base_url (`str`, *optional*):
            Base URL for the vLLM server (e.g., `"http://localhost:8000"`). If provided, `host` and `server_port` are
            ignored.
        host (`str`, *optional*, defaults to `"0.0.0.0"`):
            IP address of the vLLM server. Ignored if `base_url` is provided.
        server_port (`int`, *optional*, defaults to `8000`):
            Port number of the vLLM server. Ignored if `base_url` is provided.
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
        >>> from trl.generation.vllm_client import VLLMClient

        >>> client = VLLMClient()
        >>> client.generate(["Hello, AI!", "Tell me a joke"])
        {'prompt_ids': [[9707, 11, 15235, 0],
                        [40451, 752, 264, 21646]],
         'completion_ids': [[11479, 752, 5046, 279, 1465, 304, 419, 23670, 2038, 358, 2776, 4378, 369, 847, 15549, 6733],
                            [911, 19654, 382, 3838, 1558, 279, 16158, 1977, 979, 498, 2299, 4460, 311, 10542, 432, 518]],
         'logprobs': [[-5.193126201629639, -0.05592319369316101, -4.861808776855469, -1.673396110534668, -2.6316866874694824, -0.2861405313014984, -0.35006725788116455, -5.23351526260376, -0.1447441577911377, -5.21489953994751, -1.6022650003433228, -1.9649192094802856, -2.1338791847229004, -1.2775304317474365, -10.004860877990723, -4.171003818511963],
                      [-0.012896230444312096, -5.747106552124023, -1.5248860120773315, -1.9286258220672607, -2.8512537479400635, -2.8055880069732666, -3.019822835922241, -0.37132859230041504, -0.6311739087104797, -2.562908411026001, -3.1664533615112305, -2.685293436050415, -0.007259538397192955, -7.339841842651367, -1.188662052154541, -3.54781436920166]]}

        >>> from transformers import AutoModelForCausalLM

        >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B", device_map="cuda")
        >>> client.init_communicator(device="cuda")
        >>> client.update_model_params(model)
        ```

        There are several ways to initialize the client:

        ```python
        VLLMClient(base_url="http://localhost:8000")
        VLLMClient(base_url="http://192.168.1.100:8000")
        VLLMClient(host="localhost", server_port=8000)
        VLLMClient(host="192.168.1.100", server_port=8000)
        ```
    """

    def __init__(
        self,
        base_url: str | None = None,
        host: str = "0.0.0.0",
        server_port: int = 8000,
        group_port: int = 51216,
        connection_timeout: float = 0.0,
    ):
        if not is_requests_available():
            raise ImportError("requests is not installed. Please install it with `pip install requests`.")
        if not is_vllm_available():
            raise ImportError("vLLM is not installed. Please install it with `pip install trl[vllm]`.")

        self.session = requests.Session()

        # Configure retries for HTTP requests made through this session.
        # This is not strictly required for correctness, but it helps make training more robust to rare, transient
        # failures (network hiccups, temporary 5xx errors, overloaded servers). Without this, such failures could cause
        # an otherwise healthy training run to fail.
        retry_strategy = Retry(
            total=5,  # global cap on the total number of retries across all failure types
            connect=5,  # retry connection-level failures (DNS issues, refused connections, etc)
            read=5,  # retry failures while reading the response after the connection was successfully established
            status=3,  # retry a limited number of times when we receive certain HTTP error responses from the server
            status_forcelist=[500, 502, 503],  # only retry on server-side errors that are usually temporary
            backoff_factor=2,  # exponential backoff between retries (2s, 4s, 8s, ...)
            allowed_methods=["POST", "GET"],  # allow POST as well, even though we're not sure it's safe here
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        if base_url is not None:
            # Parse the base_url to extract host and port
            parsed_url = urlparse(base_url)
            self.host = socket.gethostbyname(parsed_url.hostname)
            scheme = parsed_url.scheme or "http"
            self.base_url = f"{scheme}://{parsed_url.netloc}{parsed_url.path}"
        else:
            self.host = host
            self.server_port = server_port
            self.base_url = f"http://{self.host}:{self.server_port}"
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
        url = f"{self.base_url}/health/"
        start_time = time.time()  # Record the start time

        while True:
            try:
                response = requests.get(url)
            except requests.exceptions.RequestException as exc:
                # Check if the total timeout duration has passed
                elapsed_time = time.time() - start_time
                if elapsed_time >= total_timeout:
                    raise ConnectionError(
                        f"The vLLM server can't be reached at {self.base_url} after {total_timeout} seconds. Make "
                        "sure the server is running by running `trl vllm-serve`."
                    ) from exc
            else:
                if response.status_code == 200:
                    if "X-Forwarded-For" in response.headers:
                        self.host = response.headers["X-Forwarded-For"]
                    logger.info("Server is up!")
                    return None

            # Retry logic: wait before trying again
            logger.info(f"Server is not up yet. Retrying in {retry_interval} seconds...")
            time.sleep(retry_interval)

    def generate(
        self,
        prompts: list[str],
        images: list | None = None,
        n: int = 1,
        repetition_penalty: float = 1.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
        max_tokens: int = 16,
        truncate_prompt_tokens: int | None = None,
        structured_outputs_regex: str | None = None,
        generation_kwargs: dict | None = None,
    ) -> dict[str, list[list[int]]]:
        """
        Generates model completions for the provided prompts.

        Args:
            prompts (`list[str]`):
                List of text prompts for which the model will generate completions.
            images (`list[PIL.Image]`, *optional*):
                List of PIL Images to send along with the prompts.
            n (`int`, *optional*, defaults to `1`):
                Number of completions to generate for each prompt.
            repetition_penalty (`float`, *optional*, defaults to `1.0`):
                Parameter for repetition penalty. 1.0 means no penalty.
            temperature (`float`, *optional*, defaults to `1.0`):
                Temperature parameter for sampling. Higher values increase diversity.
            top_p (`float`, *optional*, defaults to `1.0`):
                Top-p sampling parameter.`1.0` means no truncation.
            top_k (`int`, *optional*, defaults to `0`):
                Top-k sampling parameter. `0` means no truncation.
            min_p (`float`, *optional*, defaults to `0.0`):
                Minimum probability for sampling.
            max_tokens (`int`, *optional*, defaults to `16`):
                Maximum number of tokens to generate for each prompt.
            truncate_prompt_tokens (`int`, *optional*):
                If set to `-1`, will use the truncation size supported by the model. If set to an integer k, will use
                only the last k tokens from the prompt (i.e., left truncation). If set to `None`, truncation is
                disabled.
            structured_outputs_regex (`str`, *optional*):
                Regular expression to guide the decoding process.
            generation_kwargs (`dict`, *optional*):
                Additional generation parameters to pass to the vLLM `SamplingParams`. This can include parameters like
                `seed`, `frequency_penalty`, etc. If it contains keys that conflict with the other parameters, they
                will override them.

        Returns:
            `dict` with keys:
                - `prompt_ids` (`list[list[int]]`):
                    List of lists of token IDs representing the tokenized input prompts.
                - `completion_ids` (`list[list[int]]`):
                    List of lists of token IDs representing the model-generated completions for each prompt.
                - `logprobs` (`list[list[float]]`):
                    List of lists of log probabilities for each generated token.
        """
        url = f"{self.base_url}/generate/"

        # Convert PIL images to base64 strings
        images = [pil_to_base64(img) for img in images] if images else None

        response = self.session.post(
            url,
            json={
                "prompts": prompts,
                "images": images,
                "n": n,
                "repetition_penalty": repetition_penalty,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "min_p": min_p,
                "max_tokens": max_tokens,
                "truncate_prompt_tokens": truncate_prompt_tokens,
                "structured_outputs_regex": structured_outputs_regex,
                "generation_kwargs": generation_kwargs or {},
            },
        )
        if response.status_code == 200:
            json_response = response.json()
            return {
                "prompt_ids": json_response["prompt_ids"],
                "completion_ids": json_response["completion_ids"],
                "logprobs": json_response["logprobs"],
            }
        else:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

    def chat(
        self,
        messages: list[list[dict]],
        n: int = 1,
        repetition_penalty: float = 1.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
        max_tokens: int = 16,
        truncate_prompt_tokens: int | None = None,
        structured_outputs_regex: str | None = None,
        generation_kwargs: dict | None = None,
        chat_template_kwargs: dict | None = None,
        tools: list | None = None,
        chat_template: str | None = None,
    ) -> dict[str, list[list[int]]]:
        """
        Generates model completions for the provided chat messages.

        Args:
            messages (`list[list[dict]]`):
                List of message lists for which the model will generate completions. Each message is a dictionary with
                keys like "role" and "content".
            n (`int`, *optional*, defaults to `1`):
                Number of completions to generate for each message list.
            repetition_penalty (`float`, *optional*, defaults to `1.0`):
                Parameter for repetition penalty. 1.0 means no penalty.
            temperature (`float`, *optional*, defaults to `1.0`):
                Temperature parameter for sampling. Higher values increase diversity.
            top_p (`float`, *optional*, defaults to `1.0`):
                Top-p sampling parameter.`1.0` means no truncation.
            top_k (`int`, *optional*, defaults to `0`):
                Top-k sampling parameter. `0` means no truncation.
            min_p (`float`, *optional*, defaults to `0.0`):
                Minimum probability for sampling.
            max_tokens (`int`, *optional*, defaults to `16`):
                Maximum number of tokens to generate for each message list.
            truncate_prompt_tokens (`int`, *optional*):
                If set to `-1`, will use the truncation size supported by the model. If set to an integer k, will use
                only the last k tokens from the prompt (i.e., left truncation). If set to `None`, truncation is
                disabled.
            structured_outputs_regex (`str`, *optional*):
                Regular expression to guide the decoding process.
            generation_kwargs (`dict`, *optional*):
                Additional generation parameters to pass to the vLLM `SamplingParams`. This can include parameters like
                `seed`, `frequency_penalty`, etc. If it contains keys that conflict with the other parameters, they
                will override them.
            chat_template_kwargs (`dict`, *optional*):
                Additional keyword arguments to customize the chat template used by the model.
            tools (`list`, *optional*):
                List of tool functions available for tool calling during chat generation.
            chat_template (`str`, *optional*):
                Template to use for structuring the chat. If not provided, the model's default chat template will be
                used.

        Returns:
            `dict` with keys:
                - `prompt_ids` (`list[list[int]]`):
                    List of lists of token IDs representing the tokenized input messages.
                - `completion_ids` (`list[list[int]]`):
                    List of lists of token IDs representing the model-generated completions for each message list.
                - `logprobs` (`list[list[float]]`):
                    List of lists of log probabilities for each generated token.
        """
        if tools:
            raise NotImplementedError("Tool calling is not yet implemented in VLLMClient.chat().")
        if chat_template is not None:
            raise NotImplementedError("Custom chat templates are not yet implemented in VLLMClient.chat().")

        url = f"{self.base_url}/chat/"

        # Convert PIL images to base64 strings
        messages = copy.deepcopy(messages)  # avoid modifying the original messages
        for message_list in messages:
            for message in message_list:
                if isinstance(message["content"], list):
                    for part in message["content"]:
                        if part["type"] == "image_pil":
                            part["image_pil"] = pil_to_base64(part["image_pil"])

        response = self.session.post(
            url,
            json={
                "messages": messages,
                "n": n,
                "repetition_penalty": repetition_penalty,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "min_p": min_p,
                "max_tokens": max_tokens,
                "truncate_prompt_tokens": truncate_prompt_tokens,
                "structured_outputs_regex": structured_outputs_regex,
                "generation_kwargs": generation_kwargs or {},
                "chat_template_kwargs": chat_template_kwargs or {},
            },
        )
        if response.status_code == 200:
            json_response = response.json()
            return {
                "prompt_ids": json_response["prompt_ids"],
                "completion_ids": json_response["completion_ids"],
                "logprobs": json_response["logprobs"],
            }
        else:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

    def init_communicator(self, device: torch.device | str | int = 0):
        """
        Initializes the weight update group in a distributed setup for model synchronization.

        Args:
            device (`torch.device`, `str`, or `int`, *optional*, defaults to `0`):
                Device of trainer main process. It's the device that will be used for the weights synchronization. Can
                be a `torch.device` object, a string like `'cuda:0'`, or an integer device index.
        """
        # Get the world size from the server
        url = f"{self.base_url}/get_world_size/"
        response = requests.get(url)
        if response.status_code == 200:
            vllm_world_size = response.json()["world_size"]
        else:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

        world_size = vllm_world_size + 1  # add the client to the world
        self.rank = vllm_world_size  # the client's rank is the last process

        # Initialize weight update group
        url = f"{self.base_url}/init_communicator/"
        # Will simplify it after torch xpu 2.9 support get uuid.
        if is_torch_xpu_available():
            if hasattr(torch.xpu.get_device_properties(device), "uuid"):
                client_device_uuid = str(torch.xpu.get_device_properties(device).uuid)
            else:
                client_device_uuid = "42"
        else:
            client_device_uuid = str(torch.cuda.get_device_properties(device).uuid)

        # Set the weight update group's host to "0.0.0.0" so that
        # clients from different IPs can send updated weights
        response = self.session.post(
            url,
            json={
                "host": "0.0.0.0",
                "port": self.group_port,
                "world_size": world_size,
                "client_device_uuid": client_device_uuid,
            },
        )
        if response.status_code != 200:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

        # Brief delay to allow server initialization. While not strictly required (client socket will retry on
        # connection failure), this prevents log warnings like:
        # [W416 23:24:57.460001114 socket.cpp:204] [c10d] The hostname of the client socket cannot be retrieved. err=-3
        time.sleep(0.1)

        # Set up the communication group for weight broadcasting
        if is_torch_xpu_available():
            store = torch.distributed.TCPStore(
                host_name=self.host, port=self.group_port, world_size=world_size, is_master=(self.rank == 0)
            )
            prefixed_store = c10d.PrefixStore("client2server", store)
            xccl_options = c10d.ProcessGroupXCCL.Options()
            pg = c10d.ProcessGroupXCCL(
                store=prefixed_store,
                rank=self.rank,
                size=world_size,
                options=xccl_options,
            )
            self.communicator = pg
        else:
            pg = StatelessProcessGroup.create(
                host=self.host, port=self.group_port, rank=self.rank, world_size=world_size
            )
            self.communicator = PyNcclCommunicator(pg, device=device)

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
        url = f"{self.base_url}/update_named_param/"
        response = self.session.post(url, json={"name": name, "dtype": dtype, "shape": shape})
        if response.status_code != 200:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

        if is_torch_xpu_available():
            # Use XCCL to broadcast the updated weights from the client (src) to all workers.
            self.communicator.broadcast(weights, root=self.rank)
            self.communicator.barrier()
        else:
            # Use NCCL to broadcast the updated weights from the client (src) to all workers.
            self.communicator.broadcast(weights, src=self.rank)
            self.communicator.group.barrier()

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
        url = f"{self.base_url}/reset_prefix_cache/"
        response = self.session.post(url)
        if response.status_code != 200:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

    def chat_completions(
        self,
        messages: list[dict],
        model: str | None = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens: int | None = None,
        n: int = 1,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> dict:
        """
        OpenAI-compatible chat completions endpoint.

        Args:
            messages (`list[dict]`):
                List of messages in OpenAI format with "role" and "content" keys.
            model (`str`, *optional*):
                Model name to use.
            temperature (`float`, *optional*, defaults to `1.0`):
                Temperature for sampling.
            top_p (`float`, *optional*, defaults to `1.0`):
                Top-p sampling parameter.
            max_tokens (`int`, *optional*):
                Maximum number of tokens to generate.
            n (`int`, *optional*, defaults to `1`):
                Number of completions to generate.
            tools (`list[dict]`, *optional*):
                List of tool definitions for tool calling.
            **kwargs:
                Additional parameters to pass to the endpoint.

        Returns:
            `dict`:
                OpenAI-compatible response with "choices", "usage", etc.
        """
        url = f"{self.base_url}/v1/chat/completions"
        response = self.session.post(
            url,
            json={
                "messages": messages,
                "model": model,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "n": n,
                "tools": tools,
                **kwargs,
            },
        )
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

    def tokenize(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        """
        Tokenize messages to get token IDs.

        Args:
            messages (`list[dict]`):
                List of messages to tokenize.
            tools (`list[dict]`, *optional*):
                List of tool definitions.

        Returns:
            `dict`:
                Dictionary with "tokens" (list of token IDs) and "model" keys.
        """
        url = f"{self.base_url}/tokenize"
        response = self.session.post(url, json={"messages": messages, "tools": tools})
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

    def close_communicator(self):
        """
        Closes the weight update group and cleans up the communication group.
        """
        url = f"{self.base_url}/close_communicator/"

        try:
            response = self.session.post(url)
        except ConnectionError:
            # The server might be already down, so we don't need to close the communicator
            pass
        else:
            if response.status_code != 200:
                raise Exception(f"Request failed: {response.status_code}, {response.text}")

        if self.communicator is not None:
            self.communicator = None


# Example usage
if __name__ == "__main__":
    from vllm import SamplingParams

    device = "xpu" if is_torch_xpu_available() else "cuda"
    client = VLLMClient()
    client.init_communicator(device=device)

    # Generate completions
    responses = client.generate(["Hello, AI!", "Tell me a joke"], n=4, max_tokens=32, sampling_params=SamplingParams())
    print("Responses:", responses)  # noqa

    # Update model weights
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B").to(device)
    client.update_model_params(model)
