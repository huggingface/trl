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

"""
SGLang Engine Adapter - Improved implementation based on slime patterns.
"""

import logging
import multiprocessing
import os
import time
from typing import Optional

import requests
from urllib3.exceptions import NewConnectionError

from ..import_utils import is_sglang_available


logger = logging.getLogger(__name__)


if is_sglang_available():
    from sglang.srt.entrypoints.http_server import launch_server
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.utils import kill_process_tree


def get_base_gpu_id(args, rank):
    """
    Calculate base GPU ID for SGLang engine based on slime's logic.

    Args:
        args: Configuration arguments
        rank: Rank of the current engine

    Returns:
        int: Base GPU ID to use
    """
    num_gpus = min(getattr(args, "sglang_num_gpus_per_node", 8), getattr(args, "sglang_num_gpus_per_engine", 1))

    if getattr(args, "colocate", True):
        start_index = (rank * num_gpus) % getattr(args, "sglang_num_gpus_per_node", 8)
    else:
        num_actor_gpus = getattr(args, "actor_num_gpus_per_node", 0) * getattr(args, "actor_num_nodes", 1)
        start_index = (num_actor_gpus + rank * num_gpus) % getattr(args, "sglang_num_gpus_per_node", 8)

    return start_index


def launch_server_process(server_args: ServerArgs) -> multiprocessing.Process:
    """
    Launch SGLang server in a separate process with proper health checking.
    Based on slime's implementation.
    """
    p = multiprocessing.Process(target=launch_server, args=(server_args,))
    p.start()

    if server_args.node_rank != 0:
        return p

    base_url = server_args.url()
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "Authorization": f"Bearer {server_args.api_key}" if server_args.api_key else "",
    }

    with requests.Session() as session:
        # Wait for server to be ready
        while True:
            try:
                response = session.get(f"{base_url}/health_generate", headers=headers)
                if response.status_code == 200:
                    break
            except requests.RequestException:
                pass

            if not p.is_alive():
                raise Exception("SGLang server process terminated unexpectedly.")

            time.sleep(2)

        # Ensure working queue is empty for offload support
        while True:
            try:
                response = session.get(f"{base_url}/flush_cache", headers=headers)
                if response.status_code == 200:
                    break
            except requests.RequestException:
                pass

            if not p.is_alive():
                raise Exception("SGLang server process terminated unexpectedly.")

            time.sleep(2)

    return p


class SGLangHttpServerEngineAdapter:
    """
    SGLang HTTP Server Engine Adapter based on slime's HttpServerEngineAdapter.

    This class provides a clean interface to launch and manage SGLang HTTP servers
    with proper weight synchronization, memory management, and distributed support.
    """

    def __init__(self, router_ip=None, router_port=None, **kwargs):
        self.router_ip = router_ip
        self.router_port = router_port
        self.server_args = ServerArgs(**kwargs)
        self.node_rank = self.server_args.node_rank

        logger.info(f"Launch SGLangHttpServerEngineAdapter at: {self.server_args.host}:{self.server_args.port}")

        # Launch server process
        self.process = launch_server_process(self.server_args)

        # Register with router if specified
        if self.node_rank == 0 and self.router_ip and self.router_port:
            try:
                requests.post(
                    f"http://{self.router_ip}:{self.router_port}/add_worker"
                    f"?url=http://{self.server_args.host}:{self.server_args.port}"
                )
            except requests.RequestException as e:
                logger.warning(f"Failed to register with router: {e}")

    def _make_request(self, endpoint: str, payload: Optional[dict] = None, method: str = "POST"):
        """
        Make a request to the SGLang server.

        Args:
            endpoint: The API endpoint to call
            payload: The JSON payload to send (default: empty dict)
            method: HTTP method (GET or POST)

        Returns:
            The JSON response from the server
        """
        if self.node_rank != 0:
            return

        url = f"http://{self.server_args.host}:{self.server_args.port}/{endpoint}"

        if method.upper() == "GET":
            response = requests.get(url)
        else:
            response = requests.post(url, json=payload or {})

        response.raise_for_status()
        return response.json()

    def update_weights_from_tensor(
        self,
        serialized_named_tensors: list[str],
        load_format: Optional[str] = None,
        flush_cache: bool = False,
    ):
        """
        Update model weights from tensor data using SGLang's native API.

        This method uses SGLang's built-in weight update mechanism for efficient
        GPU-to-GPU weight transfer without CPU intermediary.
        """
        return self._make_request(
            "update_weights_from_tensor",
            {
                "serialized_named_tensors": serialized_named_tensors,
                "load_format": load_format,
                "flush_cache": flush_cache,
            },
        )

    def update_weights_from_distributed(self, names, dtypes, shapes, group_name, flush_cache=False):
        """
        Update model weights from distributed training using NCCL broadcast.

        This is the preferred method for weight synchronization in distributed setups.
        """
        return self._make_request(
            "update_weights_from_distributed",
            {
                "names": names,
                "dtypes": [str(dtype).replace("torch.", "") for dtype in dtypes],
                "shapes": shapes,
                "group_name": group_name,
                "flush_cache": flush_cache,
            },
        )

    def init_weights_update_group(self, master_address, master_port, rank_offset, world_size, group_name, backend):
        """
        Initialize the distributed weight update group.
        """
        return self._make_request(
            "init_weights_update_group",
            {
                "master_address": master_address,
                "master_port": master_port,
                "rank_offset": rank_offset,
                "world_size": world_size,
                "group_name": group_name,
                "backend": backend,
            },
        )

    def flush_cache(self):
        """Flush the cache of the server."""
        if self.node_rank != 0:
            return

        # flush_cache will not return status_code 200 when there are pending requests
        while True:
            try:
                response = requests.get(f"http://{self.server_args.host}:{self.server_args.port}/flush_cache")
                if response.status_code == 200:
                    break
            except NewConnectionError as e:
                raise e
            except Exception as e:
                logger.error(f"Error flushing cache: {e}")
                continue

    def release_memory_occupation(self):
        """Release memory occupation for offloading support."""
        return self._make_request("release_memory_occupation")

    def resume_memory_occupation(self):
        """Resume memory occupation after offloading."""
        return self._make_request("resume_memory_occupation")

    def pause_generation(self):
        """Pause generation on the server."""
        return self._make_request("pause_generation")

    def continue_generation(self):
        """Continue generation on the server."""
        return self._make_request("continue_generation")

    def generate(self, prompts, sampling_params, images=None):
        """
        Generate completions using the SGLang server.

        Args:
            prompts: List of text prompts
            sampling_params: Dictionary of sampling parameters
            images: Optional list of images for multi-modal generation

        Returns:
            Generated completions
        """
        payload = {
            "text": prompts,
            "sampling_params": sampling_params,
        }

        if images:
            payload["images"] = images

        return self._make_request("generate", payload)

    def shutdown(self):
        """Shutdown the server and clean up resources."""
        # Deregister from router
        if self.router_ip and self.router_port:
            try:
                requests.post(
                    f"http://{self.router_ip}:{self.router_port}/remove_worker"
                    f"?url=http://{self.server_args.host}:{self.server_args.port}"
                )
            except requests.RequestException:
                pass  # Router might be down

        # Kill the server process
        kill_process_tree(self.process.pid)


class SGLangEngine:
    """
    SGLang Engine wrapper based on slime's SglangEngine.

    This class provides a higher-level interface for managing SGLang engines
    with proper resource management and distributed support.
    """

    def __init__(self, args, rank, dist_init_addr, port, nccl_port):
        self.args = args
        self.rank = rank

        # Remove CUDA_VISIBLE_DEVICES set by ray/accelerate and use base_gpu_id
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)

        # Calculate distributed configuration
        nnodes = max(
            1, getattr(args, "sglang_tensor_parallel_size", 1) // getattr(args, "sglang_num_gpus_per_node", 8)
        )
        node_rank = rank % nnodes

        # Prepare server configuration
        server_kwargs = {
            "model_path": args.model if hasattr(args, "model") else args.sglang_model_path,
            "trust_remote_code": getattr(args, "trust_remote_code", True),
            "random_seed": getattr(args, "seed", 42) + rank,
            # Memory configuration
            "enable_memory_saver": getattr(args, "offload", False),
            # Distributed configuration
            "host": getattr(args, "sglang_host", "0.0.0.0"),
            "port": port,
            "nccl_port": nccl_port,
            "nnodes": nnodes,
            "node_rank": node_rank,
            "dist_init_addr": dist_init_addr,
            "gpu_id_step": 1,
            "base_gpu_id": get_base_gpu_id(args, rank),
            # Parallelism configuration
            "tp_size": getattr(args, "sglang_tensor_parallel_size", 1),
            "dp_size": getattr(args, "sglang_data_parallel_size", 1),
            "pp_size": getattr(args, "sglang_pipeline_parallel_size", 1),
            "ep_size": getattr(args, "sglang_expert_parallel_size", 1),
            # Performance configuration
            "skip_server_warmup": True,  # Always skip warmup to prevent timeout
        }

        # Filter out None values and unsupported arguments
        server_kwargs = {k: v for k, v in server_kwargs.items() if v is not None}

        # Create the HTTP server engine adapter
        self.llm = SGLangHttpServerEngineAdapter(
            router_ip=getattr(args, "sglang_router_ip", None),
            router_port=getattr(args, "sglang_router_port", None),
            **server_kwargs,
        )

    def init_process_group(self, master_address, master_port, rank_offset, world_size, group_name, backend):
        """Initialize the distributed process group for weight updates."""
        return self.llm.init_weights_update_group(
            master_address, master_port, rank_offset, world_size, group_name, backend
        )

    def update_weights_from_distributed(self, names, dtypes, shapes, group_name):
        """Update weights from distributed training."""
        self.llm.update_weights_from_distributed(names, dtypes, shapes, group_name)
        return

    def update_weights_from_tensor(self, ipc_handles):
        """Update weights from tensor handles."""
        self.llm.update_weights_from_tensor(ipc_handles)
        return

    def reset_prefix_cache(self):
        """Reset the prefix cache."""
        self.llm.flush_cache()

    def sleep(self, level=1):
        """Release memory occupation for offloading."""
        self.llm.flush_cache()
        self.llm.release_memory_occupation()

    def wake_up(self):
        """Resume memory occupation after offloading."""
        self.llm.resume_memory_occupation()

    def pause_generation(self):
        """Pause generation."""
        self.llm.pause_generation()

    def continue_generation(self):
        """Continue generation."""
        self.llm.continue_generation()

    def generate(self, prompts, sampling_params, images=None):
        """Generate completions."""
        return self.llm.generate(prompts, sampling_params, images)

    def shutdown(self):
        """Shutdown the engine."""
        self.llm.shutdown()
