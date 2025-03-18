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

import atexit

import requests
import torch
from torch import nn
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.utils import StatelessProcessGroup


class VLLMClient:
    """
    A client class to interact with a vLLM server.

    This class provides methods to generate completions, initialize and manage weight update groups, and update model
    weights in a distributed setting. Before using it, make sure to start the vLLM server with `trl vllm-serve`.

    Args:
        server_address (`str`, *optional*, defaults to `"0.0.0.0:8000"`):
            Address of the VLLM server to connect to.

    Examples:
        Run the vLLM server with the model `Qwen/Qwen2.5-7B`:

        ```sh
        trl vllm-serve --model Qwen/Qwen2.5-7B
        ```

        Use the client to generate completions and update model weights:

        ```python
        >>> from trl.extras.vllm_client import VLLMClient
        >>> client = VLLMClient()
        >>> client.generate(["Hello, AI!", "Tell me a joke"])
        [[2980, 498, 1492, 752, 448, 264, 13027, 8645, 30, 358, 2776, 4460, 311, 3270, 264, 2025], [911, 7988, 1251, 382, 3838, 653, 498, 1618, 4325, 879, 2581, 20027, 264, 21428, 30, 362]]

        >>> from transformers import AutoModelForCausalLM
        >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B").to("cuda")
        >>> client.update_model_params(model)
        ```
    """

    def __init__(self, server_address: str = "0.0.0.0:8000"):
        self.session = requests.Session()
        self.server_address = server_address
        self.init_weight_update_group()

        # When the client object is deleted, close the weight update group
        atexit.register(self.close_weight_update_group)

    def generate(self, prompts: list[str], n: int = 1, max_tokens: int = 16) -> list[str]:
        """
        Generates model completions for the provided prompts.

        Args:
            prompts (`list[str]`):
                List of text prompts for which the model will generate completions.

        Returns:
            `list[list[int]]`:
                List of lists of token IDs representing the model-generated completions for each prompt.
        """
        url = f"http://{self.server_address}/generate/"
        response = self.session.post(url, json={"prompts": prompts, "n": n, "max_tokens": max_tokens})
        if response.status_code == 200:
            return response.json()["completion_ids"]
        else:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

    def init_weight_update_group(self, host: str = "0.0.0.0", port: int = 51216):
        """
        Initializes the weight update group in a distributed setup for model synchronization.

        Args:
            host (`str`, *optional*, defaults to `"0.0.0.0"`):
                Hostname or IP address to bind the weight update group.
            port (`int`, *optional*, defaults to `51217`):
                Port to bind for communication.
        """
        # Get the tensor parallel size from the server
        url = f"http://{self.server_address}/get_tensor_parallel_size/"
        response = requests.get(url)
        if response.status_code == 200:
            tensor_parallel_size = response.json()["tensor_parallel_size"]
        else:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

        world_size = tensor_parallel_size + 1
        self.rank = tensor_parallel_size  # The client's rank is the last process

        # Initialize weight update group
        url = f"http://{self.server_address}/init_weight_update_group/"
        response = self.session.post(url, json={"host": host, "port": port, "world_size": world_size})
        if response.status_code != 200:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

        # Set up the communication group for weight broadcasting
        pg = StatelessProcessGroup.create(host=host, port=port, rank=self.rank, world_size=world_size)
        self.model_update_group = PyNcclCommunicator(pg, device="cuda:0")

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
        url = f"http://{self.server_address}/update_named_param/"
        response = self.session.post(url, json={"name": name, "dtype": dtype, "shape": shape})
        if response.status_code != 200:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

        # Broadcast the weights to the other processes
        self.model_update_group.broadcast(weights, src=self.rank, stream=torch.cuda.current_stream())

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

    def close_weight_update_group(self):
        """
        Closes the weight update group and cleans up the communication group.
        """
        url = f"http://{self.server_address}/close_weight_update_group/"
        response = self.session.post(url)
        if response.status_code != 200:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")


# Example usage
if __name__ == "__main__":
    client = VLLMClient()

    # Generate completions
    responses = client.generate(["Hello, AI!", "Tell me a joke"], n=4, max_tokens=32)
    print("Responses:", responses)  # noqa

    # Update model weights
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B").to("cuda")
    client.update_model_params(model)
