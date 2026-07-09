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

"""Trainer-side client for direct LoRA NCCL weight transfer to vLLM.

Mirrors :class:`~trl.experimental.async_grpo.weight_transfer.WeightTransferClient` but transfers
only LoRA A/B tensors (~50-200 MB) instead of the full model (~multi-GB). The vLLM server must be
launched with :mod:`trl.experimental.async_grpo.lora.vllm_server` so the
:class:`~trl.experimental.async_grpo.lora.worker_extension.LoRADirectSyncExtension` is loaded.

Key advantages over full-weight transfer:
  - **Speed**: LoRA tensors are 10-100x smaller; NCCL broadcast takes <1s vs 10-30s.
  - **No pause/resume on trainer side**: the vLLM server pauses internally during
    ``receive_lora_update`` and resumes immediately after the in-place ``set_lora()``.
  - **No disk I/O**: weights flow directly over NCCL, no save-to-disk round-trip.
  - **Works with TP**: the NCCL group includes all vLLM workers.
"""

import json
import threading
import time
from collections.abc import Iterator

import requests
import torch
from accelerate.logging import get_logger

from ....import_utils import is_vllm_available


if is_vllm_available(min_version="0.17.1"):
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.utils.network_utils import get_ip, get_open_port

logger = get_logger(__name__)


class LoRAWeightTransferClient:
    """Trainer-side client for sending LoRA weights to vLLM via NCCL.

    Args:
        vllm_server_url: Base URL of the vLLM server (e.g. ``http://localhost:8000``).
        server_timeout: Seconds to wait for the server to become healthy.
    """

    def __init__(self, vllm_server_url: str, server_timeout: float = 240.0):
        if not is_vllm_available(min_version="0.17.1"):
            raise ImportError(
                "vLLM >= 0.17.1 is required for LoRA weight transfer. "
                "Install it with: pip install 'vllm>=0.17.1'"
            )
        self.vllm_server_url = vllm_server_url.rstrip("/")
        self.server_timeout = server_timeout
        self._nccl_group: PyNcclCommunicator | None = None

    def _wait_for_server_ready(self, timeout_s: float | None = None, poll_interval_s: float = 2.0) -> None:
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
                    "Make sure the vLLM LoRA server (trl.experimental.async_grpo.lora.vllm_server) is running."
                )
            if int(elapsed) % 10 < poll_interval_s:
                logger.info(f"Still waiting for vLLM server... ({elapsed:.0f}s)")
            time.sleep(poll_interval_s)

    def init_lora_sync_group(self) -> None:
        """Initialize the NCCL process group between trainer and vLLM for LoRA transfer."""
        self._wait_for_server_ready()

        response = requests.get(f"{self.vllm_server_url}/get_world_size")
        inference_world_size = response.json()["world_size"]
        world_size = inference_world_size + 1
        master_address = get_ip()
        master_port = get_open_port()

        init_info = {
            "init_info": {
                "master_address": master_address,
                "master_port": master_port,
                "rank_offset": 1,
                "world_size": world_size,
            }
        }
        t_init = threading.Thread(
            target=requests.post,
            args=(f"{self.vllm_server_url}/init_lora_sync_group",),
            kwargs={"json": init_info, "timeout": 120},
        )
        t_init.start()

        from .worker_extension import create_stateless_pg

        pg = create_stateless_pg(
            host=master_address, port=master_port, rank=0, world_size=world_size
        )
        self._nccl_group = PyNcclCommunicator(
            pg, device=torch.device(f"cuda:{torch.cuda.current_device()}")
        )
        t_init.join()
        logger.info("LoRA NCCL sync group initialized (world_size=%d)", world_size)

    def send_lora_weights(
        self,
        lora_param_iter: Iterator[tuple[str, torch.Tensor]],
        lora_alpha: float,
        lora_rank: int,
        lora_int_id: int,
    ) -> None:
        """Send LoRA A/B tensors to vLLM via NCCL broadcast.

        Args:
            lora_param_iter: Iterator of ``(peft_param_name, tensor)`` for LoRA params only.
            lora_alpha: The LoRA alpha scaling factor from the adapter config.
            lora_rank: The LoRA rank from the adapter config.
            lora_int_id: The integer ID of the LoRA adapter in vLLM's slot table.
        """
        if self._nccl_group is None:
            logger.warning("LoRA NCCL group not initialized, skipping send_lora_weights")
            return

        t0 = time.time()

        params = []
        tensors = []
        for name, tensor in lora_param_iter:
            params.append({
                "name": name,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype).split(".")[-1],
            })
            tensors.append(tensor)

        manifest = json.dumps({
            "lora_alpha": lora_alpha,
            "lora_rank": lora_rank,
            "lora_int_id": lora_int_id,
            "params": params,
        })

        stream = torch.cuda.current_stream()

        t_update = threading.Thread(
            target=requests.post,
            args=(f"{self.vllm_server_url}/update_lora",),
            kwargs={"json": {"manifest_json": manifest}, "timeout": 300},
        )
        t_update.start()
        for tensor in tensors:
            self._nccl_group.broadcast(tensor, src=0, stream=stream)
        t_update.join()

        logger.info(
            "[weight_sync] LoRA NCCL send took %.1fs (%d params)",
            time.time() - t0,
            len(tensors),
        )

    def destroy(self) -> None:
        if self._nccl_group is not None:
            del self._nccl_group
            self._nccl_group = None
