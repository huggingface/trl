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

import time

import requests
from accelerate.logging import get_logger


logger = get_logger(__name__)


class VLLMClient:
    """Synchronous HTTP client for the vLLM server used by async GRPO.

    The trainer and the weight-sync transports both talk to the same `vllm serve` instance, so this client is the
    single place that knows the server's HTTP API: readiness, model introspection, pause/resume, and the weight-update
    endpoints. It is stateless (only a URL and a timeout), so it can be pickled and reused across the spawned rollout
    process. The rollout worker's generation calls are the one exception — they run on an async `aiohttp` session in a
    child process and are not routed through here.

    Args:
        server_url (`str`):
            Base URL of the vLLM server, e.g. `"http://localhost:8000"`.
        server_timeout (`float`, *optional*, defaults to `240.0`):
            Seconds to wait for the server to become reachable in [`wait_for_server_ready`].
    """

    def __init__(self, server_url: str, server_timeout: float = 240.0):
        self.server_url = server_url.rstrip("/")
        self.server_timeout = server_timeout

    def wait_for_server_ready(self, poll_interval_s: float = 2.0) -> None:
        """Block until the server answers `/health`, or raise `TimeoutError` after `server_timeout` seconds."""
        logger.info(f"Waiting for vLLM server at {self.server_url} ...")
        start = time.time()
        while True:
            elapsed = time.time() - start
            try:
                response = requests.get(f"{self.server_url}/health", timeout=5)
                if response.status_code == 200:
                    logger.info(f"vLLM server ready after {elapsed:.1f}s")
                    return
            except (requests.ConnectionError, requests.Timeout, OSError):
                pass
            if elapsed >= self.server_timeout:
                raise TimeoutError(
                    f"Timed out after {self.server_timeout:.0f}s waiting for vLLM server at {self.server_url}. "
                    "Make sure the vLLM server is running and reachable. If the server needs more time to load "
                    "the model, increase `vllm_server_timeout` in your AsyncGRPOConfig."
                )
            if int(elapsed) % 10 < poll_interval_s:
                logger.info(f"Still waiting for vLLM server... ({elapsed:.0f}s)")
            time.sleep(poll_interval_s)

    def get_max_model_len(self) -> int:
        """Return the served model's `max_model_len` (the cap on prompt + completion tokens)."""
        response = requests.get(f"{self.server_url}/v1/models")
        response.raise_for_status()
        return response.json()["data"][0]["max_model_len"]

    def get_world_size(self, attempts: int = 30, poll_interval_s: float = 2.0) -> int:
        """Return the vLLM server's inference world size (tensor/pipeline parallel processes).

        `/health` can go green before the engine RPC answers, so poll instead of indexing the first response blindly.
        """
        for _ in range(attempts):
            try:
                world_size = requests.get(f"{self.server_url}/get_world_size", timeout=5).json().get("world_size")
                if world_size is not None:
                    return int(world_size)
            except (requests.RequestException, ValueError):
                pass
            time.sleep(poll_interval_s)
        raise RuntimeError(f"vLLM /get_world_size did not return a world_size after {attempts} attempts")

    def pause(self) -> None:
        """Pause generation while keeping the KV cache warm (`mode=keep`), so weights can be swapped in."""
        requests.post(f"{self.server_url}/pause", params={"mode": "keep"})

    def resume(self) -> None:
        """Resume generation after a weight update."""
        requests.post(f"{self.server_url}/resume")

    def init_weight_transfer_engine(self, init_info: dict, timeout: int) -> None:
        """Initialise the server side of the NCCL weight-transfer group."""
        requests.post(f"{self.server_url}/init_weight_transfer_engine", json={"init_info": init_info}, timeout=timeout)

    def start_weight_update(self, is_checkpoint_format: bool = True, timeout: int = 1800) -> None:
        """Prepare the workers for a weight reload; must complete before any weights are sent.

        Full policies are sent in HF checkpoint format; sparse deltas are sent in kernel format and applied in place,
        so they pass `is_checkpoint_format=False`.
        """
        requests.post(
            f"{self.server_url}/start_weight_update",
            json={"is_checkpoint_format": is_checkpoint_format},
            timeout=timeout,
        )

    def update_weights(self, update_info: dict, timeout: int = 1800, retries: int = 1) -> None:
        """Drive the weight reload on the workers.

        For NCCL this blocks in the workers' recv, so it is called on a thread concurrently with the trainer-side
        broadcast. For the bucket backend vLLM fetches the patch inside this call, which may transiently return 429
        under load — `retries` bounds the retry, and the timeout must cover a full anchor download.
        """
        for attempt in range(retries):
            try:
                response = requests.post(
                    f"{self.server_url}/update_weights", json={"update_info": update_info}, timeout=timeout
                )
                if response.status_code != 429:
                    response.raise_for_status()
                    return
                status = response.status_code
            except requests.RequestException as e:
                logger.warning(f"[weight_sync] /update_weights failed: {e}")
                status = "connection error"
            if attempt < retries - 1:
                wait = min(2**attempt, 30)
                logger.warning(
                    f"[weight_sync] /update_weights -> {status}, retry in {wait}s ({attempt + 1}/{retries})"
                )
                time.sleep(wait)
        raise RuntimeError(f"[weight_sync] /update_weights failed after {retries} attempt(s)")

    def finish_weight_update(self, timeout: int = 1800) -> None:
        """Finalise the weight update on the server."""
        requests.post(f"{self.server_url}/finish_weight_update", timeout=timeout)
