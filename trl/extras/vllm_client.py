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
import logging
import time
from typing import Optional

import torch
from torch import nn

from ..import_utils import is_requests_available, is_vllm_available


if is_requests_available():
    import requests
    from requests import ConnectionError


if is_vllm_available():
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup


logger = logging.getLogger(__name__)


class VLLMClient:
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
        >>> client.update_model_params(model)
        ```
    """

    def __init__(
        self, host: str = "0.0.0.0", server_port: int = 8000, group_port: int = 51216, connection_timeout: float = 0.0
    ):
        if not is_requests_available():
            raise ImportError("requests is not installed. Please install it with `pip install requests`.")
        if not is_vllm_available():
            raise ImportError("vLLM is not installed. Please install it with `pip install vllm`.")

        self.session = requests.Session()
        self.host = host
        self.server_port = server_port
        self.group_port = group_port
        self.check_server(connection_timeout)  # check server and fail after timeout
        self.init_communicator()
        atexit.register(self.close_communicator)  # when the client object is deleted, close the weight update group

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
            return response.json()["completion_ids"]
        else:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

    def init_communicator(self):
        """
        Initializes the weight update group in a distributed setup for model synchronization.
        """
        # Get the tensor parallel size from the server
        url = f"http://{self.host}:{self.server_port}/get_tensor_parallel_size/"
        response = requests.get(url)
        if response.status_code == 200:
            tensor_parallel_size = response.json()["tensor_parallel_size"]
        else:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

        world_size = tensor_parallel_size + 1
        self.rank = tensor_parallel_size  # The client's rank is the last process

        # Initialize weight update group
        url = f"http://{self.host}:{self.server_port}/init_communicator/"
        # In the server side, the host is set to 0.0.0.0
        response = self.session.post(url, json={"host": "0.0.0.0", "port": self.group_port, "world_size": world_size})
        if response.status_code != 200:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

        # Set up the communication group for weight broadcasting
        pg = StatelessProcessGroup.create(host=self.host, port=self.group_port, rank=self.rank, world_size=world_size)
        self.pynccl_comm = PyNcclCommunicator(pg, device="cuda:0")

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
        self.pynccl_comm.broadcast(weights, src=self.rank, stream=torch.cuda.current_stream())
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
        response = self.session.post(url)
        if response.status_code != 200:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")


class VLLMDataParallelClient:
    """
    A client class to interact with multiple vLLM servers in a data-parallel setup.

    This class distributes requests across multiple vLLM server instances, enabling higher throughput
    and better resource utilization, especially for smaller models that don't require tensor parallelism.

    Args:
        hosts (`list[str]`, *optional*, defaults to `["localhost"]`):
            List of IP addresses or hostnames for the vLLM servers.
        ports (`list[int]` or `int`, *optional*, defaults to `8000`):
            Port numbers for the vLLM servers. If an integer is provided, ports will be assigned 
            incrementally (8000, 8001, etc.) for each host.
        connection_timeout (`float`, *optional*, defaults to `0.0`):
            Total timeout duration in seconds to wait for the server to be up.

    Examples:
        Run multiple vLLM servers:

        ```
        $ # On first GPU
        $ trl vllm-serve --model Qwen/Qwen2.5-7B --port 8000
        $ # On second GPU  
        $ trl vllm-serve --model Qwen/Qwen2.5-7B --port 8001
        ```

        Use the client to distribute completions across servers:

        ```python
        >>> from trl.extras.vllm_client import VLLMDataParallelClient
        >>> client = VLLMDataParallelClient(ports=[8000, 8001])
        >>> client.generate(["Hello, AI!", "Tell me a joke", "What is deep learning?", "Explain quantum computing"])
        ```
    """

    def __init__(
        self,
        hosts: list[str] = ["localhost"],
        ports: list[int] | int = 8000,
        connection_timeout: float = 0.0,
    ):
        if not is_requests_available():
            raise ImportError("requests is not installed. Please install it with `pip install requests`.")
        if not is_vllm_available():
            raise ImportError("vLLM is not installed. Please install it with `pip install vllm`.")

        # Normalize ports to a list if a single integer is given
        if isinstance(ports, int):
            ports = [ports + i for i in range(len(hosts))]
        elif len(ports) != len(hosts):
            raise ValueError(f"Number of ports ({len(ports)}) must match number of hosts ({len(hosts)})")

        # Initialize server connections
        self.servers = []
        for i, (host, port) in enumerate(zip(hosts, ports)):
            server = {
                "host": host,
                "port": port,
                "session": requests.Session(),
                "url": f"http://{host}:{port}",
                "rank": i,
                "health": False,
            }
            self.servers.append(server)

        # Check all servers
        for server in self.servers:
            self.check_server(server, connection_timeout)

        self.num_servers = len(self.servers)
        self.next_server = 0  # For round-robin request distribution

        # Track if servers are ready for weight updates
        self.communicator_initialized = False

    def check_server(self, server: dict, total_timeout: float = 0.0, retry_interval: float = 2.0):
        """
        Check server availability with retries on failure, within a total timeout duration.

        Args:
            server (`dict`):
                Server configuration dictionary.
            total_timeout (`float`, *optional*, defaults to `0.0`):
                Total timeout duration in seconds.
            retry_interval (`float`, *optional*, defaults to `2.0`):
                Interval in seconds between retries.
        """
        url = f"{server['url']}/health/"
        start_time = time.time()

        while True:
            try:
                response = requests.get(url)
            except requests.exceptions.RequestException as exc:
                # Check if the total timeout duration has passed
                elapsed_time = time.time() - start_time
                if elapsed_time >= total_timeout:
                    logger.warning(
                        f"vLLM server can't be reached at {server['host']}:{server['port']} after {total_timeout} "
                        "seconds. This server will be marked as unhealthy."
                    )
                    server["health"] = False
                    return
            else:
                if response.status_code == 200:
                    logger.info(f"Server at {server['host']}:{server['port']} is up!")
                    server["health"] = True
                    return

            # Retry logic: wait before trying again
            logger.info(f"Server at {server['host']}:{server['port']} is not up yet. Retrying in {retry_interval} seconds...")
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
    ) -> list[list[str]]:
        """
        Generates model completions for the provided prompts, distributing the load across servers.

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
        # Check if we have any healthy servers
        healthy_servers = [server for server in self.servers if server["health"]]
        if not healthy_servers:
            raise ConnectionError("No healthy vLLM servers available")

        # If single prompt, wrap in list
        if isinstance(prompts, str):
            prompts = [prompts]
            single_prompt = True
        else:
            single_prompt = False

        # Distribute prompts across servers using round-robin
        server_prompts = {i: [] for i in range(len(healthy_servers))}
        server_prompt_indices = {i: [] for i in range(len(healthy_servers))}
        
        for i, prompt in enumerate(prompts):
            server_idx = self.next_server % len(healthy_servers)
            server_prompts[server_idx].append(prompt)
            server_prompt_indices[server_idx].append(i)
            self.next_server += 1

        # Send requests to each server and collect results
        all_results = [None] * len(prompts)
        import threading
        
        def send_request(server_idx, prompt_list, indices):
            if not prompt_list:
                return
                
            server = healthy_servers[server_idx]
            url = f"{server['url']}/generate/"
            
            try:
                response = server["session"].post(
                    url,
                    json={
                        "prompts": prompt_list,
                        "n": n,
                        "repetition_penalty": repetition_penalty,
                        "temperature": temperature,
                        "top_p": top_p,
                        "top_k": top_k,
                        "min_p": min_p,
                        "max_tokens": max_tokens,
                        "guided_decoding_regex": guided_decoding_regex,
                    },
                    timeout=30,  # Add a reasonable timeout
                )
                
                if response.status_code == 200:
                    completion_ids = response.json()["completion_ids"]
                    for idx, completion in zip(indices, completion_ids):
                        all_results[idx] = completion
                else:
                    # Mark server as unhealthy if it fails
                    server["health"] = False
                    raise Exception(f"Request failed: {response.status_code}, {response.text}")
            except Exception as e:
                server["health"] = False
                logger.warning(f"Error with server {server['host']}:{server['port']}: {e}")
                # Fill in empty results for failed requests
                for idx in indices:
                    all_results[idx] = []

        # Send requests to all servers in parallel
        threads = []
        for i in range(len(healthy_servers)):
            if server_prompts[i]:
                thread = threading.Thread(
                    target=send_request,
                    args=(i, server_prompts[i], server_prompt_indices[i])
                )
                threads.append(thread)
                thread.start()
                
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
            
        # Check if any results are missing
        if None in all_results:
            missing_indices = [i for i, r in enumerate(all_results) if r is None]
            raise Exception(f"Missing results for prompts at indices: {missing_indices}")
            
        # Return single result if input was a single prompt
        if single_prompt:
            return all_results[0]
            
        return all_results
                
    def init_weight_update_group(self, world_size=None):
        """
        Initializes weight update communication group across all servers.
        This is useful when you want to sync model weights across all servers.
        
        Args:
            world_size (`int` or `None`, *optional*, defaults to `None`):
                Total world size including the client. If None, set to num_servers + 1.
        """
        if not self.servers:
            raise ValueError("No servers available to initialize weight update group")
            
        if world_size is None:
            world_size = len(self.servers) + 1  # +1 for the client
            
        # Initialize communicator for each server
        for server in self.servers:
            if not server["health"]:
                continue
                
            try:
                url = f"{server['url']}/init_communicator/"
                response = server["session"].post(
                    url, 
                    json={
                        "host": "0.0.0.0",  # Server side uses 0.0.0.0
                        "port": server["port"] + 1000,  # Use port+1000 for weight updates
                        "world_size": world_size
                    }
                )
                if response.status_code != 200:
                    server["health"] = False
                    logger.warning(f"Failed to initialize communicator for server {server['host']}:{server['port']}")
            except Exception as e:
                server["health"] = False
                logger.warning(f"Error initializing communicator for server {server['host']}:{server['port']}: {e}")
                
        self.communicator_initialized = True
        
    def update_model_params(self, model: nn.Module):
        """
        Updates model parameters on all servers.
        
        Args:
            model (`nn.Module`):
                Model with parameters to broadcast to all servers.
        """
        if not self.communicator_initialized:
            self.init_weight_update_group()
            
        # Update each parameter individually on each server
        for name, param in model.named_parameters():
            for server in self.servers:
                if not server["health"]:
                    continue
                    
                try:
                    # Notify server about incoming parameter update
                    dtype, shape = str(param.dtype), list(param.shape)
                    url = f"{server['url']}/update_named_param/"
                    response = server["session"].post(
                        url,
                        json={"name": name, "dtype": dtype, "shape": shape}
                    )
                    
                    if response.status_code != 200:
                        server["health"] = False
                        logger.warning(f"Failed to update parameter {name} on server {server['host']}:{server['port']}")
                except Exception as e:
                    server["health"] = False
                    logger.warning(f"Error updating parameter {name} on server {server['host']}:{server['port']}: {e}")
                    
    def reset_prefix_cache(self):
        """
        Resets the prefix cache for all servers.
        """
        for server in self.servers:
            if not server["health"]:
                continue
                
            try:
                url = f"{server['url']}/reset_prefix_cache/"
                response = server["session"].post(url)
                if response.status_code != 200:
                    server["health"] = False
                    logger.warning(f"Failed to reset prefix cache on server {server['host']}:{server['port']}")
            except Exception as e:
                server["health"] = False
                logger.warning(f"Error resetting prefix cache on server {server['host']}:{server['port']}: {e}")
                
    def close_communicator(self):
        """
        Closes the weight update group for all servers.
        """
        if not self.communicator_initialized:
            return
            
        for server in self.servers:
            if not server["health"]:
                continue
                
            try:
                url = f"{server['url']}/close_communicator/"
                response = server["session"].post(url)
                if response.status_code != 200:
                    server["health"] = False
                    logger.warning(f"Failed to close communicator on server {server['host']}:{server['port']}")
            except Exception as e:
                server["health"] = False
                logger.warning(f"Error closing communicator on server {server['host']}:{server['port']}: {e}")
                
        self.communicator_initialized = False


# Example usage
if __name__ == "__main__":
    from vllm import SamplingParams

    client = VLLMClient()

    # Generate completions
    responses = client.generate(["Hello, AI!", "Tell me a joke"], n=4, max_tokens=32, sampling_params=SamplingParams())
    print("Responses:", responses)  # noqa

    # Update model weights
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B").to("cuda")
    client.update_model_params(model)
