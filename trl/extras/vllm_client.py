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
import socket
import time
from typing import Dict, List, Optional, Union, Any, Tuple
from urllib.parse import urlparse

import torch
from torch import nn

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


class VLLMGenerationResponse:
    """Response object for vLLM generation requests with speed metrics."""
    
    def __init__(self, completion_ids: List[List[int]], completions: List[str], 
                 generation_time: float, tokens_per_second: float, total_tokens: int):
        self.completion_ids = completion_ids
        self.completions = completions
        self.generation_time = generation_time
        self.tokens_per_second = tokens_per_second
        self.total_tokens = total_tokens
    
    def __repr__(self):
        return (f"VLLMGenerationResponse(completions={len(self.completions)}, "
                f"total_tokens={self.total_tokens}, speed={self.tokens_per_second:.2f} tok/s)")


class VLLMClient:
    """
    A comprehensive client class to interact with a vLLM server with full DNA/protein/embedding support.

    This class provides methods to generate completions with multimodal support for DNA sequences,
    protein sequences, and direct prompt embeddings, initialize and manage weight update groups,
    and update model weights in a distributed setting. Before using it, start the vLLM server with `trl vllm-serve`.

    Args:
        base_url (`str` or `None`, *optional*, defaults to `None`):
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
        Run the vLLM server with DNA processing:

        ```bash
        $ trl vllm-serve --use_dna_llm --dna_model_name InstaDeepAI/nucleotide-transformer-v2-500m-multi-species --model YOUR_PATH_TO_MODEL
        
        ```

        Use the client for multimodal DNA+text generation:

        ```python
        >>> from trl.extras.vllm_client import VLLMClient

        >>> client = VLLMClient()
        >>> 
        >>> # Check server capabilities
        >>> health = client.health_check()
        >>> print(f"DNA processing: {health['dna_processing_enabled']}")
        >>> print(f"Protein processing: {health['protein_processing_enabled']}")
        >>> 
        >>> # Generate with DNA sequences
        >>> response = client.generate(
        ...     prompts=["Analyze this DNA sequence:"],
        ...     dna_sequences=[["ATCGATCGATCG"]],
        ...     temperature=0.7,
        ...     max_tokens=100
        ... )
        >>> print(f"Generated {len(response.completions)} completions")
        >>> print(f"Speed: {response.tokens_per_second:.2f} tokens/sec")
        >>> 
        >>> # Generate with protein sequences
        >>> response = client.generate(
        ...     prompts=["What does this protein do?"],
        ...     protein_sequences=[["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"]],
        ...     temperature=0.7,
        ...     max_tokens=100
        ... )
        >>> 
        >>> # Generate with direct prompt embeddings
        >>> embeddings = torch.randn(1, 10, 4096).tolist()  # (batch, seq_len, hidden_size)
        >>> response = client.generate_from_embeddings(
        ...     prompt_embeds=embeddings,
        ...     max_tokens=50
        ... )
        >>> 
        >>> # Update model weights
        >>> from transformers import AutoModelForCausalLM
        >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B", device_map="cuda")
        >>> client.init_communicator()
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
        base_url: Optional[str] = None,
        host: str = "0.0.0.0",
        server_port: int = 8000,
        group_port: int = 51216,
        connection_timeout: float = 0.0,
    ):
        if not is_requests_available():
            raise ImportError("requests is not installed. Please install it with `pip install requests`.")
        if not is_vllm_available():
            raise ImportError("vLLM is not installed. Please install it with `pip install vllm`.")

        self.session = requests.Session()

        if base_url is not None:
            # Parse the base_url to extract host and port
            parsed_url = urlparse(base_url)
            if parsed_url.hostname is None:
                raise ValueError(f"Invalid base_url: {base_url}")
            self.host = socket.gethostbyname(parsed_url.hostname)
            scheme = parsed_url.scheme or "http"
            self.base_url = f"{scheme}://{parsed_url.netloc}{parsed_url.path}"
        else:
            self.host = host
            self.server_port = server_port
            self.base_url = f"http://{self.host}:{self.server_port}"
        self.group_port = group_port
        self.check_server(connection_timeout)  # check server and fail after timeout
        
        # Cache server capabilities
        self._server_capabilities = None

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

    def health_check(self) -> Dict[str, Any]:
        """
        Check server health and get capabilities information.
        
        Returns:
            Dict containing server status and capabilities including DNA/protein processing support.
        """
        url = f"{self.base_url}/health/"
        response = self.session.get(url)
        if response.status_code == 200:
            capabilities = response.json()
            self._server_capabilities = capabilities
            return capabilities
        else:
            raise Exception(f"Health check failed: {response.status_code}, {response.text}")

    def get_server_capabilities(self) -> Dict[str, Any]:
        """
        Get cached server capabilities or fetch them if not cached.
        
        Returns:
            Dict containing server capabilities.
        """
        if self._server_capabilities is None:
            return self.health_check()
        return self._server_capabilities

    def generate(
        self,
        prompts: list[str],
        dna_sequences: list[list[str]] | None = None,
        protein_sequences: list[list[str]] | None = None,
        # batch_idx_map: list[int] = None,
        
        n: int = 1,
        repetition_penalty: float = 1.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        min_p: float = 0.1,
        max_tokens: int = 16,
        guided_decoding_regex: Optional[str] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> VLLMGenerationResponse:
        """
        Generates model completions for the provided prompts with optional DNA or protein sequences.

        Args:
            prompts (`List[str]`):
                List of text prompts for which the model will generate completions.
            dna_sequences (`List[List[str]]` or `None`, *optional*, defaults to `None`):
                List of DNA sequences for each prompt. Each inner list contains DNA sequences for that prompt.
                Length must match prompts length if provided.
            protein_sequences (`List[List[str]]` or `None`, *optional*, defaults to `None`):
                List of protein sequences for each prompt. Each inner list contains protein sequences for that prompt.
                Length must match prompts length if provided. Cannot be used with dna_sequences.
            n (`int`, *optional*, defaults to `1`):
                Number of completions to generate for each prompt.
            repetition_penalty (`float`, *optional*, defaults to `1.0`):
                Parameter for repetition penalty. 1.0 means no penalty.
            temperature (`float`, *optional*, defaults to `1.0`):
                Temperature parameter for sampling. Higher values increase diversity.
            top_p (`float`, *optional*, defaults to `1.0`):
                Top-p sampling parameter. 1.0 means no truncation.
            top_k (`int`, *optional*, defaults to `-1`):
                Top-k sampling parameter. -1 means no truncation.
            min_p (`float`, *optional*, defaults to `0.0`):
                Minimum probability for sampling.
            max_tokens (`int`, *optional*, defaults to `16`):
                Maximum number of tokens to generate for each prompt.
            guided_decoding_regex (`str` or `None`, *optional*, defaults to `None`):
                Regular expression to guide the decoding process.
            generation_kwargs (`Dict[str, Any]` or `None`, *optional*, defaults to `None`):
                Additional generation parameters to pass to the vLLM `SamplingParams`.

        Returns:
            `VLLMGenerationResponse`:
                Response object containing completion token IDs, completion texts, and speed metrics.
                
        Raises:
            ValueError: If both dna_sequences and protein_sequences are provided, or if sequence lengths don't match prompts.
            Exception: If the server request fails.
        """
        url = f"{self.base_url}/generate/"
        
        # Build request payload
        payload = {
            "prompts": prompts,
            "dna_sequences": dna_sequences,
            "protein_sequences": protein_sequences,
            "n": n,
            "repetition_penalty": repetition_penalty,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "max_tokens": max_tokens,
            "guided_decoding_regex": guided_decoding_regex,
            "generation_kwargs": generation_kwargs or {},
        }

        # print(f"prompts: {prompts}")
        
        # print(f"dna_sequences: {dna_sequences}")

        response = self.session.post(url, json=payload)
        
        if response.status_code == 200:
            data = response.json()
            return VLLMGenerationResponse(
                completion_ids=data["completion_ids"],
                completions=data["completions"],
                generation_time=data["generation_time"],
                tokens_per_second=data["tokens_per_second"],
                total_tokens=data["total_tokens"],
            )
        else:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

    def generate_from_embeddings(
        self,
        prompt_embeds: List[List[List[float]]],
        n: int = 1,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        min_p: float = 0.1,
        max_tokens: int = 16,
        repetition_penalty: float = 1.0,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> VLLMGenerationResponse:
        """
        Generate completions from pre-computed prompt embeddings.

        Args:
            prompt_embeds (`List[List[List[float]]]`):
                List of prompt embeddings. Each item is a 2D list of shape (seq_len, hidden_size).
            temperature (`float`, *optional*, defaults to `1.0`):
                Temperature parameter for sampling.
            top_p (`float`, *optional*, defaults to `1.0`):
                Top-p sampling parameter.
            top_k (`int`, *optional*, defaults to `-1`):
                Top-k sampling parameter.
            min_p (`float`, *optional*, defaults to `0.0`):
                Minimum probability for sampling.
            max_tokens (`int`, *optional*, defaults to `16`):
                Maximum number of tokens to generate.
            repetition_penalty (`float`, *optional*, defaults to `1.0`):
                Repetition penalty parameter.
            generation_kwargs (`Dict[str, Any]` or `None`, *optional*, defaults to `None`):
                Additional generation parameters.

        Returns:
            `VLLMGenerationResponse`:
                Response object containing completion token IDs, completion texts, and speed metrics.
        """
        url = f"{self.base_url}/generate_embeds/"
        
        payload = {
            "prompt_embeds": prompt_embeds,
            "n": n,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "max_tokens": max_tokens,
            "repetition_penalty": repetition_penalty,
            "generation_kwargs": generation_kwargs or {},
        }

        response = self.session.post(url, json=payload)
        
        if response.status_code == 200:
            data = response.json()
            return VLLMGenerationResponse(
                completion_ids=data["completion_ids"],
                completions=data["completions"],
                generation_time=data["generation_time"],
                tokens_per_second=data["tokens_per_second"],
                total_tokens=data["total_tokens"],
            )
        else:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

    def init_communicator(self):
        """
        Initializes the weight update group in a distributed setup for model synchronization.
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
        print(f"Updating parameter '{name}' with shape {weights.shape} and dtype {weights.dtype}")
        if not name.startswith("text_model.") and not name.startswith("base_model.") and not name.startswith("model."):
            return

        print(f"Updating parameter '{name}' with shape {weights.shape} and dtype {weights.dtype}")
        
        # print(f"Updating parameter '{name}' with shape {weights.shape} and dtype {weights.dtype}")
        name = name[len("text_model."):] if name.startswith("text_model.") else name
        # print(f"Updated parameter '{name}' with shape {weights.shape} and dtype {weights.dtype}")
        dtype, shape = str(weights.dtype), tuple(weights.shape)
        url = f"{self.base_url}/update_named_param/"
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
        url = f"{self.base_url}/reset_prefix_cache/"
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
            # The server might be already down, so we don't need to close the communicator
            pass
        else:
            if response.status_code != 200:
                raise Exception(f"Request failed: {response.status_code}, {response.text}")

    def get_world_size(self) -> int:
        """
        Get the world size of the vLLM server.
        
        Returns:
            int: The world size (number of processes) of the vLLM server.
        """
        url = f"{self.base_url}/get_world_size/"
        response = self.session.get(url)
        if response.status_code == 200:
            return response.json()["world_size"]
        else:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

    def is_dna_processing_enabled(self) -> bool:
        """
        Check if DNA processing is enabled on the server.
        
        Returns:
            bool: True if DNA processing is enabled, False otherwise.
        """
        capabilities = self.get_server_capabilities()
        return capabilities.get("dna_processing_enabled", False)

    def is_protein_processing_enabled(self) -> bool:
        """
        Check if protein processing is enabled on the server.
        
        Returns:
            bool: True if protein processing is enabled, False otherwise.
        """
        capabilities = self.get_server_capabilities()
        return capabilities.get("protein_processing_enabled", False)

    def get_supported_models(self) -> Dict[str, Optional[str]]:
        """
        Get information about the models loaded on the server.
        
        Returns:
            Dict containing model information:
                - text_model: The main text model
                - dna_model: The DNA model (if DNA processing is enabled)
                - protein_model: The protein model (if protein processing is enabled)
        """
        capabilities = self.get_server_capabilities()
        return {
            "text_model": capabilities.get("text_model"),
            "dna_model": capabilities.get("dna_model"),
            "protein_model": capabilities.get("protein_model"),
        }

    def __repr__(self):
        return f"VLLMClient(base_url='{self.base_url}')"


# Example usage
if __name__ == "__main__":
    # Create client
    client = VLLMClient()
    
    # Check server capabilities
    print("=== Server Capabilities ===")
    health = client.health_check()
    print(f"Server status: {health['status']}")
    print(f"DNA processing: {health.get('dna_processing_enabled', False)}")
    print(f"Protein processing: {health.get('protein_processing_enabled', False)}")
    print(f"Text model: {health.get('text_model', 'Unknown')}")
    print(f"DNA model: {health.get('dna_model', 'None')}")
    print(f"Protein model: {health.get('protein_model', 'None')}")
    
    # Example 1: Basic text generation
    print("\n=== Basic Text Generation ===")
    response = client.generate(
        prompts=["Hello, AI!", "Tell me a joke"],
        temperature=0.7,
        max_tokens=32
    )
    print(f"Generated {len(response.completions)} completions")
    print(f"Speed: {response.tokens_per_second:.2f} tokens/sec")
    for i, completion in enumerate(response.completions):
        print(f"  {i+1}: {completion}")
    
    # Example 2: DNA sequence generation (if enabled)
    if client.is_dna_processing_enabled():
        print("\n=== DNA Sequence Generation ===")
    
    # Example 3: Protein sequence generation (if enabled)
    if client.is_protein_processing_enabled():
        print("\n=== Protein Sequence Generation ===")
        
    
    # Example 4: Direct embedding generation
    print("\n=== Direct Embedding Generation ===")
    try:
        # Create dummy embeddings (in real usage, these would be computed from your model)
        import random
        dummy_embeddings = [[[random.random() for _ in range(4096)] for _ in range(10)]]
        
        response = client.generate_from_embeddings(
            prompt_embeds=dummy_embeddings,
            temperature=0.7,
            max_tokens=32
        )
        print(f"Embedding-based completion: {response.completions[0]}")
        print(f"Speed: {response.tokens_per_second:.2f} tokens/sec")
    except Exception as e:
        print(f"Embedding generation failed (expected for dummy data): {e}")
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
import socket
import time
from typing import Dict, List, Optional, Union, Any, Tuple
from urllib.parse import urlparse

import torch
from torch import nn

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


class VLLMGenerationResponse:
    """Response object for vLLM generation requests with speed metrics."""
    
    def __init__(self, completion_ids: List[List[int]], completions: List[str], 
                 generation_time: float, tokens_per_second: float, total_tokens: int):
        self.completion_ids = completion_ids
        self.completions = completions
        self.generation_time = generation_time
        self.tokens_per_second = tokens_per_second
        self.total_tokens = total_tokens
    
    def __repr__(self):
        return (f"VLLMGenerationResponse(completions={len(self.completions)}, "
                f"total_tokens={self.total_tokens}, speed={self.tokens_per_second:.2f} tok/s)")


class VLLMClient:
    """
    A comprehensive client class to interact with a vLLM server with full DNA/protein/embedding support.

    This class provides methods to generate completions with multimodal support for DNA sequences,
    protein sequences, and direct prompt embeddings, initialize and manage weight update groups,
    and update model weights in a distributed setting. Before using it, start the vLLM server with `trl vllm-serve`.

    Args:
        base_url (`str` or `None`, *optional*, defaults to `None`):
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
        Run the vLLM server with DNA processing:

        ```bash
        $ trl vllm-serve --use_dna_llm --dna_model_name InstaDeepAI/nucleotide-transformer-v2-500m-multi-species --model YOUR_PATH_TO_MODEL
        
        ```

        Use the client for multimodal DNA+text generation:

        ```python
        >>> from trl.extras.vllm_client import VLLMClient

        >>> client = VLLMClient()
        >>> 
        >>> # Check server capabilities
        >>> health = client.health_check()
        >>> print(f"DNA processing: {health['dna_processing_enabled']}")
        >>> print(f"Protein processing: {health['protein_processing_enabled']}")
        >>> 
        >>> # Generate with DNA sequences
        >>> response = client.generate(
        ...     prompts=["Analyze this DNA sequence:"],
        ...     dna_sequences=[["ATCGATCGATCG"]],
        ...     temperature=0.7,
        ...     max_tokens=100
        ... )
        >>> print(f"Generated {len(response.completions)} completions")
        >>> print(f"Speed: {response.tokens_per_second:.2f} tokens/sec")
        >>> 
        >>> # Generate with protein sequences
        >>> response = client.generate(
        ...     prompts=["What does this protein do?"],
        ...     protein_sequences=[["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"]],
        ...     temperature=0.7,
        ...     max_tokens=100
        ... )
        >>> 
        >>> # Generate with direct prompt embeddings
        >>> embeddings = torch.randn(1, 10, 4096).tolist()  # (batch, seq_len, hidden_size)
        >>> response = client.generate_from_embeddings(
        ...     prompt_embeds=embeddings,
        ...     max_tokens=50
        ... )
        >>> 
        >>> # Update model weights
        >>> from transformers import AutoModelForCausalLM
        >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B", device_map="cuda")
        >>> client.init_communicator()
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
        base_url: Optional[str] = None,
        host: str = "0.0.0.0",
        server_port: int = 8000,
        group_port: int = 51216,
        connection_timeout: float = 0.0,
    ):
        if not is_requests_available():
            raise ImportError("requests is not installed. Please install it with `pip install requests`.")
        if not is_vllm_available():
            raise ImportError("vLLM is not installed. Please install it with `pip install vllm`.")

        self.session = requests.Session()

        if base_url is not None:
            # Parse the base_url to extract host and port
            parsed_url = urlparse(base_url)
            if parsed_url.hostname is None:
                raise ValueError(f"Invalid base_url: {base_url}")
            self.host = socket.gethostbyname(parsed_url.hostname)
            scheme = parsed_url.scheme or "http"
            self.base_url = f"{scheme}://{parsed_url.netloc}{parsed_url.path}"
        else:
            self.host = host
            self.server_port = server_port
            self.base_url = f"http://{self.host}:{self.server_port}"
        self.group_port = group_port
        self.check_server(connection_timeout)  # check server and fail after timeout
        
        # Cache server capabilities
        self._server_capabilities = None

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

    def health_check(self) -> Dict[str, Any]:
        """
        Check server health and get capabilities information.
        
        Returns:
            Dict containing server status and capabilities including DNA/protein processing support.
        """
        url = f"{self.base_url}/health/"
        response = self.session.get(url)
        if response.status_code == 200:
            capabilities = response.json()
            self._server_capabilities = capabilities
            return capabilities
        else:
            raise Exception(f"Health check failed: {response.status_code}, {response.text}")

    def get_server_capabilities(self) -> Dict[str, Any]:
        """
        Get cached server capabilities or fetch them if not cached.
        
        Returns:
            Dict containing server capabilities.
        """
        if self._server_capabilities is None:
            return self.health_check()
        return self._server_capabilities

    def generate(
        self,
        prompts: list[str],
        dna_sequences: list[list[str]] | None = None,
        protein_sequences: list[list[str]] | None = None,
        # batch_idx_map: list[int] = None,
        
        n: int = 1,
        repetition_penalty: float = 1.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        min_p: float = 0.1,
        max_tokens: int = 16,
        guided_decoding_regex: Optional[str] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> VLLMGenerationResponse:
        """
        Generates model completions for the provided prompts with optional DNA or protein sequences.

        Args:
            prompts (`List[str]`):
                List of text prompts for which the model will generate completions.
            dna_sequences (`List[List[str]]` or `None`, *optional*, defaults to `None`):
                List of DNA sequences for each prompt. Each inner list contains DNA sequences for that prompt.
                Length must match prompts length if provided.
            protein_sequences (`List[List[str]]` or `None`, *optional*, defaults to `None`):
                List of protein sequences for each prompt. Each inner list contains protein sequences for that prompt.
                Length must match prompts length if provided. Cannot be used with dna_sequences.
            n (`int`, *optional*, defaults to `1`):
                Number of completions to generate for each prompt.
            repetition_penalty (`float`, *optional*, defaults to `1.0`):
                Parameter for repetition penalty. 1.0 means no penalty.
            temperature (`float`, *optional*, defaults to `1.0`):
                Temperature parameter for sampling. Higher values increase diversity.
            top_p (`float`, *optional*, defaults to `1.0`):
                Top-p sampling parameter. 1.0 means no truncation.
            top_k (`int`, *optional*, defaults to `-1`):
                Top-k sampling parameter. -1 means no truncation.
            min_p (`float`, *optional*, defaults to `0.0`):
                Minimum probability for sampling.
            max_tokens (`int`, *optional*, defaults to `16`):
                Maximum number of tokens to generate for each prompt.
            guided_decoding_regex (`str` or `None`, *optional*, defaults to `None`):
                Regular expression to guide the decoding process.
            generation_kwargs (`Dict[str, Any]` or `None`, *optional*, defaults to `None`):
                Additional generation parameters to pass to the vLLM `SamplingParams`.

        Returns:
            `VLLMGenerationResponse`:
                Response object containing completion token IDs, completion texts, and speed metrics.
                
        Raises:
            ValueError: If both dna_sequences and protein_sequences are provided, or if sequence lengths don't match prompts.
            Exception: If the server request fails.
        """
        url = f"{self.base_url}/generate/"
        
        # Build request payload
        payload = {
            "prompts": prompts,
            "dna_sequences": dna_sequences,
            "protein_sequences": protein_sequences,
            "n": n,
            "repetition_penalty": repetition_penalty,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "max_tokens": max_tokens,
            "guided_decoding_regex": guided_decoding_regex,
            "generation_kwargs": generation_kwargs or {},
        }

        # print(f"prompts: {prompts}")
        
        # print(f"dna_sequences: {dna_sequences}")

        response = self.session.post(url, json=payload)
        
        if response.status_code == 200:
            data = response.json()
            return VLLMGenerationResponse(
                completion_ids=data["completion_ids"],
                completions=data["completions"],
                generation_time=data["generation_time"],
                tokens_per_second=data["tokens_per_second"],
                total_tokens=data["total_tokens"],
            )
        else:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

    def generate_from_embeddings(
        self,
        prompt_embeds: List[List[List[float]]],
        n: int = 1,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        min_p: float = 0.1,
        max_tokens: int = 16,
        repetition_penalty: float = 1.0,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> VLLMGenerationResponse:
        """
        Generate completions from pre-computed prompt embeddings.

        Args:
            prompt_embeds (`List[List[List[float]]]`):
                List of prompt embeddings. Each item is a 2D list of shape (seq_len, hidden_size).
            temperature (`float`, *optional*, defaults to `1.0`):
                Temperature parameter for sampling.
            top_p (`float`, *optional*, defaults to `1.0`):
                Top-p sampling parameter.
            top_k (`int`, *optional*, defaults to `-1`):
                Top-k sampling parameter.
            min_p (`float`, *optional*, defaults to `0.0`):
                Minimum probability for sampling.
            max_tokens (`int`, *optional*, defaults to `16`):
                Maximum number of tokens to generate.
            repetition_penalty (`float`, *optional*, defaults to `1.0`):
                Repetition penalty parameter.
            generation_kwargs (`Dict[str, Any]` or `None`, *optional*, defaults to `None`):
                Additional generation parameters.

        Returns:
            `VLLMGenerationResponse`:
                Response object containing completion token IDs, completion texts, and speed metrics.
        """
        url = f"{self.base_url}/generate_embeds/"
        
        payload = {
            "prompt_embeds": prompt_embeds,
            "n": n,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "max_tokens": max_tokens,
            "repetition_penalty": repetition_penalty,
            "generation_kwargs": generation_kwargs or {},
        }

        response = self.session.post(url, json=payload)
        
        if response.status_code == 200:
            data = response.json()
            return VLLMGenerationResponse(
                completion_ids=data["completion_ids"],
                completions=data["completions"],
                generation_time=data["generation_time"],
                tokens_per_second=data["tokens_per_second"],
                total_tokens=data["total_tokens"],
            )
        else:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

    def init_communicator(self):
        """
        Initializes the weight update group in a distributed setup for model synchronization.
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
        # print(f"Updating parameter '{name}' with shape {weights.shape} and dtype {weights.dtype}")
        if not name.startswith("text_model.") and not name.startswith("base_model.") and not name.startswith("model."):
            return

        # print(f"Updating parameter '{name}' with shape {weights.shape} and dtype {weights.dtype}")
        
        # print(f"Updating parameter '{name}' with shape {weights.shape} and dtype {weights.dtype}")
        name = name[len("text_model."):] if name.startswith("text_model.") else name
        # print(f"Updated parameter '{name}' with shape {weights.shape} and dtype {weights.dtype}")
        dtype, shape = str(weights.dtype), tuple(weights.shape)
        url = f"{self.base_url}/update_named_param/"
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
        url = f"{self.base_url}/reset_prefix_cache/"
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
            # The server might be already down, so we don't need to close the communicator
            pass
        else:
            if response.status_code != 200:
                raise Exception(f"Request failed: {response.status_code}, {response.text}")

    def get_world_size(self) -> int:
        """
        Get the world size of the vLLM server.
        
        Returns:
            int: The world size (number of processes) of the vLLM server.
        """
        url = f"{self.base_url}/get_world_size/"
        response = self.session.get(url)
        if response.status_code == 200:
            return response.json()["world_size"]
        else:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

    def is_dna_processing_enabled(self) -> bool:
        """
        Check if DNA processing is enabled on the server.
        
        Returns:
            bool: True if DNA processing is enabled, False otherwise.
        """
        capabilities = self.get_server_capabilities()
        return capabilities.get("dna_processing_enabled", False)

    def is_protein_processing_enabled(self) -> bool:
        """
        Check if protein processing is enabled on the server.
        
        Returns:
            bool: True if protein processing is enabled, False otherwise.
        """
        capabilities = self.get_server_capabilities()
        return capabilities.get("protein_processing_enabled", False)

    def get_supported_models(self) -> Dict[str, Optional[str]]:
        """
        Get information about the models loaded on the server.
        
        Returns:
            Dict containing model information:
                - text_model: The main text model
                - dna_model: The DNA model (if DNA processing is enabled)
                - protein_model: The protein model (if protein processing is enabled)
        """
        capabilities = self.get_server_capabilities()
        return {
            "text_model": capabilities.get("text_model"),
            "dna_model": capabilities.get("dna_model"),
            "protein_model": capabilities.get("protein_model"),
        }

    def __repr__(self):
        return f"VLLMClient(base_url='{self.base_url}')"


# Example usage
if __name__ == "__main__":
    # Create client
    client = VLLMClient()
    
    # Check server capabilities
    print("=== Server Capabilities ===")
    health = client.health_check()
    print(f"Server status: {health['status']}")
    print(f"DNA processing: {health.get('dna_processing_enabled', False)}")
    print(f"Protein processing: {health.get('protein_processing_enabled', False)}")
    print(f"Text model: {health.get('text_model', 'Unknown')}")
    print(f"DNA model: {health.get('dna_model', 'None')}")
    print(f"Protein model: {health.get('protein_model', 'None')}")
    
    # Example 1: Basic text generation
    print("\n=== Basic Text Generation ===")
    response = client.generate(
        prompts=["Hello, AI!", "Tell me a joke"],
        temperature=0.7,
        max_tokens=32
    )
    print(f"Generated {len(response.completions)} completions")
    print(f"Speed: {response.tokens_per_second:.2f} tokens/sec")
    for i, completion in enumerate(response.completions):
        print(f"  {i+1}: {completion}")
    
    # Example 2: DNA sequence generation (if enabled)
    if client.is_dna_processing_enabled():
        print("\n=== DNA Sequence Generation ===")
    
    # Example 3: Protein sequence generation (if enabled)
    if client.is_protein_processing_enabled():
        print("\n=== Protein Sequence Generation ===")
        
    
    # Example 4: Direct embedding generation
    print("\n=== Direct Embedding Generation ===")
    try:
        # Create dummy embeddings (in real usage, these would be computed from your model)
        import random
        dummy_embeddings = [[[random.random() for _ in range(4096)] for _ in range(10)]]
        
        response = client.generate_from_embeddings(
            prompt_embeds=dummy_embeddings,
            temperature=0.7,
            max_tokens=32
        )
        print(f"Embedding-based completion: {response.completions[0]}")
        print(f"Speed: {response.tokens_per_second:.2f} tokens/sec")
    except Exception as e:
        print(f"Embedding generation failed (expected for dummy data): {e}")