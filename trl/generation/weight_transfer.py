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

import threading
import time

import requests
from accelerate.logging import get_logger

from ..import_utils import is_vllm_available


if is_vllm_available(min_version="0.22.0"):
    from vllm.distributed.weight_transfer.nccl_engine import NCCLTrainerSendWeightsArgs, NCCLWeightTransferEngine
    from vllm.utils.network_utils import get_ip, get_open_port


logger = get_logger(__name__)


class WeightTransferClient:
    def __init__(
        self,
        vllm_server_url: str,
        weight_update_info: dict,
        server_timeout: float = 240.0,
        init_weight_transfer_timeout: int = 1800,
    ):
        if not is_vllm_available(min_version="0.22.0"):
            raise ImportError(
                "vLLM >= 0.22.0 is required to use WeightTransferClient. Install it with: pip install 'vllm>=0.22.0'"
            )
        self.vllm_server_url = vllm_server_url.rstrip("/")
        self.server_timeout = server_timeout
        self.init_weight_transfer_timeout = init_weight_transfer_timeout
        self._weight_update_info = weight_update_info
        self.model_update_group = None

    def _wait_for_server_ready_sync(self, timeout_s: float | None = None, poll_interval_s: float = 2.0) -> None:
        timeout_s = timeout_s if timeout_s is not None else self.server_timeout
        logger.info(f"Waiting for vLLM server at {self.vllm_server_url} ...")
        start = time.time()
        while True:
            elapsed = time.time() - start
            try:
                response = requests.get(f"{self.vllm_server_url}/health", timeout=5)
                if response.status_code == 200:
                    logger.info(f"vLLM server ready after {elapsed:.1f}s")
                    return
            except (requests.ConnectionError, requests.Timeout, OSError):
                pass
            if elapsed >= timeout_s:
                raise TimeoutError(
                    f"Timed out after {timeout_s:.0f}s waiting for vLLM server at {self.vllm_server_url}. "
                    "Make sure the vLLM server is running and reachable. If the server needs more time to load "
                    "the model, increase `vllm_server_timeout` in your AsyncGRPOConfig."
                )
            if int(elapsed) % 10 < poll_interval_s:
                logger.info(f"Still waiting for vLLM server... ({elapsed:.0f}s)")
            time.sleep(poll_interval_s)

    def init_weight_transfer(self) -> None:
        self._wait_for_server_ready_sync()
        response = requests.get(f"{self.vllm_server_url}/get_world_size")
        inference_world_size = response.json()["world_size"]
        world_size = inference_world_size + 1
        master_address = get_ip()
        master_port = get_open_port()
        init_info = {
            "master_address": master_address,
            "master_port": master_port,
            "rank_offset": 1,
            "world_size": world_size,
        }
        t_init = threading.Thread(
            target=requests.post,
            args=(f"{self.vllm_server_url}/init_weight_transfer_engine",),
            kwargs={"json": {"init_info": init_info}, "timeout": self.init_weight_transfer_timeout},
        )
        t_init.start()
        self.model_update_group = NCCLWeightTransferEngine.trainer_init(
            {
                "master_address": master_address,
                "master_port": master_port,
                "world_size": world_size,
            }
        )
        t_init.join()
        logger.info("Initialised weight-transfer NCCL group with vLLM")

    def send_weights(self, iterator) -> None:
        if self.model_update_group is None:
            return
        t0 = time.time()
        # Prepare the workers for the reload; must complete before any weights are sent. The native transfer
        # engine always expects checkpoint-format weights, so the endpoint takes no body.
        requests.post(f"{self.vllm_server_url}/start_weight_update", timeout=1800)
        # The /update_weights POST drives the workers' blocking NCCL recv, so it runs on a thread
        # concurrently with the trainer-side broadcast.
        t_update = threading.Thread(
            target=requests.post,
            args=(f"{self.vllm_server_url}/update_weights",),
            kwargs={"json": {"update_info": self._weight_update_info}, "timeout": 1800},
        )
        t_update.start()
        NCCLWeightTransferEngine.trainer_send_weights(
            iterator=iterator,
            trainer_args=NCCLTrainerSendWeightsArgs(group=self.model_update_group, packed=True),
        )
        t_update.join()
        requests.post(f"{self.vllm_server_url}/finish_weight_update", timeout=1800)
        logger.debug(f"[weight_sync] send_weights took {time.time() - t0:.1f}s")

    def pause(self) -> None:
        t0 = time.time()
        requests.post(f"{self.vllm_server_url}/pause", params={"mode": "keep"})
        logger.debug(f"[weight_sync] pause HTTP took {time.time() - t0:.1f}s")

    def resume(self) -> None:
        t0 = time.time()
        requests.post(f"{self.vllm_server_url}/resume")
        logger.debug(f"[weight_sync] resume HTTP took {time.time() - t0:.1f}s")

    def destroy(self) -> None:
        if self.model_update_group is None:
            return
        self.model_update_group.group.store = None
        self.model_update_group.group.socket = None
        self.model_update_group = None
