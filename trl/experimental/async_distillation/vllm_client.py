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

from accelerate.logging import get_logger

from ...import_utils import is_requests_available


if is_requests_available():
    import requests


logger = get_logger(__name__)


class VLLMClient:
    """Synchronous HTTP client for a vLLM server used by async distillation.

    Ported from [`~trl.experimental.async_grpo.vllm_client.VLLMClient`]. The student's server uses the full API
    (readiness, world size, pause/resume, weight-update endpoints) since [`WeightTransferClient`] streams updated
    weights to it; the teachers' servers don't use this client: they are static, and the rollout worker queries them
    directly over `/v1/completions` from its `aiohttp` session.

    Args:
        server_url (`str`):
            Base URL of the vLLM server, e.g. `"http://localhost:8000"`.
        timeout (`float`, *optional*, defaults to `1800.0`):
            Seconds allowed for every call: waiting for the server in [`wait_for_server_ready`] and each request after.
    """

    def __init__(self, server_url: str, timeout: float = 1800.0):
        if not is_requests_available():
            raise ImportError("requests is not installed. Please install it with `pip install requests`.")

        self.server_url = server_url.rstrip("/")
        self.timeout = timeout

    def wait_for_server_ready(self, poll_interval_s: float = 2.0) -> None:
        """Block until the server answers `/health`, or raise `TimeoutError` after `timeout` seconds."""
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
            if elapsed >= self.timeout:
                raise TimeoutError(
                    f"Timed out after {self.timeout:.0f}s waiting for vLLM server at {self.server_url}. "
                    "Make sure the vLLM server is running and reachable. If the server needs more time to load the "
                    "model, increase `vllm_server_timeout` in your AsyncDistillationConfig."
                )
            if int(elapsed) % 10 < poll_interval_s:
                logger.info(f"Still waiting for vLLM server... ({elapsed:.0f}s)")
            time.sleep(poll_interval_s)

    def get_max_model_len(self) -> int:
        """Return the served model's `max_model_len` (the cap on prompt + completion tokens)."""
        response = requests.get(f"{self.server_url}/v1/models", timeout=self.timeout)
        response.raise_for_status()
        return response.json()["data"][0]["max_model_len"]

    def get_server_info(self) -> dict:
        """Return the server's resolved `vllm_config`, as reported by `/server_info`."""
        response = requests.get(
            f"{self.server_url}/server_info", params={"config_format": "json"}, timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()["vllm_config"]

    def load_lora_adapter(self, lora_name: str, lora_path: str) -> None:
        """Register a PEFT adapter directory under `lora_name`, making it selectable as a model name.

        Args:
            lora_name (`str`):
                Name to register the adapter under. Generation requests select it through their `model` field.
            lora_path (`str`):
                Directory holding `adapter_config.json` and the adapter weights. The server process reads it, so it
                must sit on a filesystem the server can see: vLLM has no in-memory adapter API.
        """
        response = requests.post(
            f"{self.server_url}/v1/load_lora_adapter",
            # `load_inplace` stays `False`: it makes the server re-read the adapter from disk on every request
            # (https://github.com/vllm-project/vllm/pull/41482). New policies get a new name instead.
            json={"lora_name": lora_name, "lora_path": lora_path, "load_inplace": False},
            timeout=self.timeout,
        )
        if response.status_code == 404:
            raise RuntimeError(
                f"The vLLM server at {self.server_url} does not expose `/v1/load_lora_adapter`. Restart it with "
                "`VLLM_ALLOW_RUNTIME_LORA_UPDATING=1` in its environment."
            )
        if response.status_code != 200:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

    def unload_lora_adapter(self, lora_name: str) -> None:
        """Remove a previously registered adapter. Missing names are not an error."""
        requests.post(f"{self.server_url}/v1/unload_lora_adapter", json={"lora_name": lora_name}, timeout=self.timeout)

    def get_world_size(self) -> int:
        """Return the vLLM server's inference world size (tensor/pipeline parallel processes)."""
        response = requests.get(f"{self.server_url}/get_world_size", timeout=self.timeout)
        return response.json()["world_size"]

    def pause(self) -> None:
        """Freeze in-flight requests before replacing the model weights. The KV and prefix caches are kept: clearing
        them makes vLLM preempt every running request and recompute it from scratch after the update."""
        response = requests.post(
            f"{self.server_url}/pause", params={"mode": "keep", "clear_cache": "false"}, timeout=self.timeout
        )
        if response.status_code != 200:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

    def resume(self) -> None:
        """Resume generation after a weight update."""
        response = requests.post(f"{self.server_url}/resume", timeout=self.timeout)
        if response.status_code != 200:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

    def init_weight_transfer_engine(self, init_info: dict) -> None:
        """Initialise the server side of the NCCL weight-transfer group."""
        response = requests.post(
            f"{self.server_url}/init_weight_transfer_engine", json={"init_info": init_info}, timeout=self.timeout
        )
        if response.status_code != 200:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

    def start_weight_update(self) -> None:
        """Prepare the workers for a weight reload; must complete before any weights are sent."""
        response = requests.post(
            f"{self.server_url}/start_weight_update", json={"is_checkpoint_format": True}, timeout=self.timeout
        )
        if response.status_code != 200:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

    def update_weights(self, update_info: dict) -> None:
        """Drive the workers' blocking NCCL recv; returns once the server has received every chunk."""
        response = requests.post(
            f"{self.server_url}/update_weights", json={"update_info": update_info}, timeout=self.timeout
        )
        if response.status_code != 200:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

    def finish_weight_update(self) -> None:
        """Finalise the weight update on the server."""
        response = requests.post(f"{self.server_url}/finish_weight_update", timeout=self.timeout)
        if response.status_code != 200:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")
