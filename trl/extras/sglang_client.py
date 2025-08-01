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
import base64
import logging
import socket
import time
from io import BytesIO
from typing import Optional, Union
from urllib.parse import urlparse

import torch
from torch import nn

from ..import_utils import is_requests_available, is_sglang_available


if is_requests_available():
    import requests
    from requests import ConnectionError


logger = logging.getLogger(__name__)


class SGLangClient:
    """
    A client class to interact with an SGLang server.

    This class provides methods to generate completions, initialize and manage weight update groups, and update model
    weights in a distributed setting. Before using it, start the SGLang server with `trl sglang-serve`.

    Args:
        base_url (`str` or `None`, *optional*, defaults to `None`):
            Base URL for the SGLang server (e.g., `"http://localhost:8001"`). If provided, `host` and `server_port` are
            ignored.
        host (`str`, *optional*, defaults to `"0.0.0.0"`):
            IP address of the SGLang server. Ignored if `base_url` is provided.
        server_port (`int`, *optional*, defaults to `8001`):
            Port number of the SGLang server. Ignored if `base_url` is provided.
        group_port (`int`, *optional*, defaults to `51217`):
            Port number for the weight update group.
        connection_timeout (`float`, *optional*, defaults to `0.0`):
            Total timeout duration in seconds to wait for the server to be up.

    Examples:
        Run the SGLang server with the model `Qwen/Qwen2.5-7B`:

        ```
        $ trl sglang-serve --model Qwen/Qwen2.5-7B
        ...
        INFO:     Application startup complete.
        INFO:     Uvicorn running on http://0.0.0.0:8001 (Press CTRL+C to quit)
        ```

        Use the client to generate completions and update model weights:

        ```python
        >>> from trl.extras.sglang_client import SGLangClient

        >>> client = SGLangClient()
        >>> client.generate(["Hello, AI!", "Tell me a joke"])
        [[2980, 498, 1492, 752, 448, 264, 13027, 8645, 30, 358, 2776, 4460, 311, 3270, 264, 2025],
         [911, 7988, 1251, 382, 3838, 653, 498, 1618, 4325, 879, 2581, 20027, 264, 21428, 30, 362]]

        >>> from transformers import AutoModelForCausalLM

        >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B", device_map="cuda")
        >>> client.init_communicator(device="cuda")
        >>> client.update_model_params(model)
        ```
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        host: str = "0.0.0.0",
        server_port: int = 8001,
        group_port: int = 51217,
        connection_timeout: float = 0.0,
    ):
        if not is_requests_available():
            raise ImportError("requests is not installed. Please install it with `pip install requests`.")
        if not is_sglang_available():
            raise ImportError("SGLang is not installed. Please install it with `pip install sglang`.")

        self.session = requests.Session()

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

        # Initialize communicator-related attributes
        self.pynccl_comm = None
        self.rank = None
        self.world_size = None

    def check_server(self, total_timeout: float = 0.0, retry_interval: float = 2.0):
        """
        Check server availability with retries on failure.

        Args:
            retry_interval (`float`, *optional*, defaults to `2.0`):
                Interval in seconds between retries.
            total_timeout (`float`, *optional*, defaults to `0.0`):
                Total timeout duration in seconds.
        """
        url = f"{self.base_url}/health/"
        start_time = time.time()

        while True:
            try:
                response = requests.get(url)
            except requests.exceptions.RequestException as exc:
                # Check if the total timeout duration has passed
                elapsed_time = time.time() - start_time
                if elapsed_time >= total_timeout:
                    raise ConnectionError(
                        f"The SGLang server can't be reached at {self.base_url} after {total_timeout} seconds. Make "
                        "sure the server is running by running `trl sglang-serve`."
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
        images: Optional[list] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        max_tokens: int = 16,
        sampling_params: Optional[dict] = None,
    ) -> list[list[int]]:
        """
        Generates model completions for the provided prompts.

        Args:
            prompts (`list[str]`):
                List of text prompts for which the model will generate completions.
            images (`list[PIL.Image]` or `None`, *optional*, defaults to `None`):
                List of PIL Images to send along with the prompts.
            temperature (`float`, *optional*, defaults to `1.0`):
                Temperature parameter for sampling.
            top_p (`float`, *optional*, defaults to `1.0`):
                Top-p sampling parameter.
            top_k (`int`, *optional*, defaults to `-1`):
                Top-k sampling parameter.
            max_tokens (`int`, *optional*, defaults to `16`):
                Maximum number of tokens to generate.
            sampling_params (`dict` or `None`, *optional*, defaults to `None`):
                Additional sampling parameters for SGLang.

        Returns:
            `list[list[int]]`:
                List of lists of token IDs representing the model-generated completions.
        """
        url = f"{self.base_url}/generate/"

        def pil_to_base64(image):
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            img_bytes = buffer.getvalue()
            return base64.b64encode(img_bytes).decode("utf-8")

        # Convert PIL images to base64 strings
        images = [pil_to_base64(img) for img in images] if images else None

        # Prepare sampling parameters
        params = sampling_params or {}
        params.update(
            {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "max_new_tokens": max_tokens,
            }
        )

        response = self.session.post(
            url,
            json={
                "prompts": prompts,
                "images": images,
                "sampling_params": params,
            },
        )
        if response.status_code == 200:
            return response.json()["completion_ids"]
        else:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

    def init_communicator(self, device: Union[torch.device, str, int] = 0):
        """
        Initializes the weight update group in a distributed setup for model synchronization.

        Args:
            device (`torch.device`, `str`, or `int`, *optional*, defaults to `0`):
                Device of trainer main process.
        """
        # Get the world size from the server
        url = f"{self.base_url}/get_world_size/"
        response = requests.get(url)
        if response.status_code == 200:
            sglang_world_size = response.json()["world_size"]
        else:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

        world_size = sglang_world_size + 1  # add the client to the world
        self.rank = sglang_world_size  # the client's rank is the last process
        self.world_size = world_size

        # Initialize weight update group
        url = f"{self.base_url}/init_communicator/"
        client_device_uuid = str(torch.cuda.get_device_properties(device).uuid)

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

        # Brief delay to allow server initialization
        time.sleep(0.1)

        # Set up the communication group for weight broadcasting
        import torch.distributed as dist

        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                init_method=f"tcp://{self.host}:{self.group_port}",
                rank=self.rank,
                world_size=world_size,
            )

        # When the client object is deleted, close the weight update group
        atexit.register(self.close_communicator)

    def update_named_param(self, name: str, weights: torch.Tensor):
        """
        Updates a specific named parameter in the model and broadcasts it to other processes.
        Uses SGLang's native weight update mechanism for efficiency.

        Args:
            name (`str`):
                Name of the layer whose weights are being updated.
            weights (`torch.Tensor`):
                Tensor containing the updated weights.
        """
        dtype_str = str(weights.dtype)
        shape = list(weights.shape)

        # Use SGLang's update_weights_from_distributed endpoint
        url = f"{self.base_url}/update_weights/"
        response = self.session.post(
            url,
            json={
                "names": [name],
                "dtypes": [dtype_str],
                "shapes": [shape],
                "group_name": "weight_sync",
                "flush_cache": True,
            },
        )
        if response.status_code != 200:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

        # Broadcast the weights to the other processes using NCCL
        import torch.distributed as dist

        dist.broadcast(weights, src=self.rank)
        dist.barrier()

    def update_model_params(self, model: nn.Module):
        """
        Updates all parameters of the given model.

        Args:
            model (`nn.Module`):
                Model whose parameters are to be updated.
        """
        # Batch all parameter updates
        names = []
        dtypes = []
        shapes = []
        weights_list = []

        for name, param in model.named_parameters():
            names.append(name)
            dtypes.append(str(param.data.dtype))
            shapes.append(list(param.data.shape))
            weights_list.append(param.data)

        # Send metadata to server using SGLang's batch update API
        url = f"{self.base_url}/update_weights/"
        response = self.session.post(
            url,
            json={
                "names": names,
                "dtypes": dtypes,
                "shapes": shapes,
                "group_name": "weight_sync",
                "flush_cache": True,
            },
        )
        if response.status_code != 200:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

        # Broadcast all weights
        import torch.distributed as dist

        for weight in weights_list:
            dist.broadcast(weight, src=self.rank)
        dist.barrier()

    def update_weights_bucketed(
        self,
        names: list[str],
        dtypes: list[str],
        shapes: list[list[int]],
        group_name: str = "weight_sync",
        flush_cache: bool = False,
    ):
        """
        Updates model weights using bucketed batch approach (slime-style).

        Args:
            names (`list[str]`):
                List of parameter names to update.
            dtypes (`list[str]`):
                List of parameter data types.
            shapes (`list[list[int]]`):
                List of parameter shapes.
            group_name (`str`, *optional*, defaults to `"weight_sync"`):
                Name of the distributed group for weight synchronization.
            flush_cache (`bool`, *optional*, defaults to `False`):
                Whether to flush the cache after this bucket update.
        """
        # Send metadata to server using SGLang's batch update API
        url = f"{self.base_url}/update_weights/"
        response = self.session.post(
            url,
            json={
                "names": names,
                "dtypes": dtypes,
                "shapes": shapes,
                "group_name": group_name,
                "flush_cache": flush_cache,
            },
        )
        if response.status_code != 200:
            raise Exception(f"SGLang bucketed weight update failed: {response.status_code}, {response.text}")

    def get_memory_info(self):
        """
        Get memory information from the SGLang server.

        Returns:
            dict: Memory information from the server.
        """
        url = f"{self.base_url}/get_memory_info/"
        try:
            response = self.session.get(url)
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Failed to get memory info: {response.status_code}")
                return {"error": "Unable to get memory info"}
        except Exception as e:
            logger.warning(f"Exception getting memory info: {e}")
            return {"error": str(e)}

    def pause_generation(self):
        """Pause generation on the SGLang server."""
        url = f"{self.base_url}/pause_generation/"
        response = self.session.post(url)
        if response.status_code != 200:
            raise Exception(f"Failed to pause generation: {response.status_code}, {response.text}")

    def continue_generation(self):
        """Continue generation on the SGLang server."""
        url = f"{self.base_url}/continue_generation/"
        response = self.session.post(url)
        if response.status_code != 200:
            raise Exception(f"Failed to continue generation: {response.status_code}, {response.text}")

    def flush_cache(self):
        """
        Flush the cache for the model.
        """
        url = f"{self.base_url}/flush_cache/"
        response = self.session.post(url)
        if response.status_code != 200:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

    def close_communicator(self):
        """
        Closes the weight update group and cleans up the communication group.
        """
        url = f"{self.base_url}/close_communicator/"

        try:
            response = self.session.post(url)
        except ConnectionError:
            # The server might be already down
            pass
        else:
            if response.status_code != 200:
                logger.warning(f"Failed to close communicator: {response.status_code}, {response.text}")


# Example usage
if __name__ == "__main__":
    client = SGLangClient()
    client.init_communicator(device="cuda")

    # Generate completions
    responses = client.generate(["Hello, AI!", "Tell me a joke"], max_tokens=32)
    # Example output would show responses here

    # Update model weights
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B").to("cuda")
    client.update_model_params(model)
