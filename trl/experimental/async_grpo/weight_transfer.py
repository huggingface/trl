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

from trl.import_utils import is_vllm_available


if is_vllm_available(min_version="0.17.1"):
    from vllm.distributed.weight_transfer.nccl_engine import NCCLTrainerSendWeightsArgs, NCCLWeightTransferEngine
    from vllm.utils.network_utils import get_ip, get_open_port


logger = get_logger(__name__)


class WeightTransferClient:
    """Parent-process handle for broadcasting trainer weights to a vLLM server.

    Holds the NCCL `StatelessProcessGroup` that the trainer's rank 0 shares with vLLM's worker(s), plus the HTTP
    control endpoints for pausing / resuming the engine during a sync. Does not produce rollouts and does not know
    about the rollout worker; lives alongside it in the trainer.

    Endpoints used:
      * GET /health
      * GET /get_world_size
      * POST /init_weight_transfer_engine
      * POST /update_weights
      * POST /pause
      * POST /resume

    Typical sequence inside the trainer:

        wt = WeightTransferClient(vllm_server_url=..., weight_update_info=...) # after accelerator.prepare():
        wt.init_weight_transfer() wt.pause() wt.send_weights(model_param_iterator) wt.resume() # at shutdown:
        wt.destroy()
    """

    def __init__(
        self,
        vllm_server_url: str,
        weight_update_info: dict,
        server_timeout: float = 240.0,
        init_weight_transfer_timeout: int = 1800,
    ):
        if not is_vllm_available(min_version="0.17.1"):
            raise ImportError(
                "vLLM >= 0.17.1 is required to use WeightTransferClient. Install it with: pip install 'vllm>=0.17.1'"
            )
        self.vllm_server_url = vllm_server_url.rstrip("/")
        self.server_timeout = server_timeout
        self.init_weight_transfer_timeout = init_weight_transfer_timeout
        # vLLM's `/update_weights` body keys on this dict — it tells the server which
        # tensors to expect, their dtypes, and their global shapes. The trainer (or a
        # subclass like `FusedMoEAsyncGRPOTrainer`) may mutate this dict between
        # construction and the first sync to handle EP-sharded or fused layouts.
        self._weight_update_info = weight_update_info
        self.model_update_group = None

    def _wait_for_server_ready_sync(self, timeout_s: float | None = None, poll_interval_s: float = 2.0) -> None:
        """Block until the vLLM server's `/health` returns 200."""
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
        """Wait for vLLM and create the NCCL weight-transfer process group.

        The trainer calls this from `on_train_begin` (after `accelerator.prepare()`). Binding the SPG's CUDA IPC pages
        before prepare would crash DeepSpeed-Z2's `Stage1And2ZeroOptimizer.__init__ → torch.cuda.empty_cache()` on the
        rank-0 allocator with `cudaErrorIllegalAddress`. Deferring is harmless under FSDP2 / single-node setups too.
        """
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
        """Broadcast model weights to vLLM. The iterator must yield (name, GPU tensor)
        tuples — the trainer's `_streaming_iter` provides this."""
        if self.model_update_group is None:
            return
        t0 = time.time()
        t_update = threading.Thread(
            target=requests.post,
            args=(f"{self.vllm_server_url}/update_weights",),
            kwargs={"json": {"update_info": self._weight_update_info}, "timeout": 1800},
        )
        t_update.start()
        logger.debug(f"[weight_sync] /update_weights POST sent ({time.time() - t0:.1f}s)")
        t_nccl = time.time()
        NCCLWeightTransferEngine.trainer_send_weights(
            iterator=iterator,
            trainer_args=NCCLTrainerSendWeightsArgs(group=self.model_update_group, packed=True),
        )
        logger.debug(f"[weight_sync] NCCL transfer took {time.time() - t_nccl:.1f}s")
        t_join = time.time()
        t_update.join()
        logger.debug(
            f"[weight_sync] /update_weights join took {time.time() - t_join:.1f}s "
            f"(total send_weights: {time.time() - t0:.1f}s)"
        )

    def pause(self) -> None:
        """Pause vLLM's engine via HTTP. In-flight `/v1/completions` requests will fail."""
        t0 = time.time()
        requests.post(f"{self.vllm_server_url}/pause", params={"mode": "keep"})
        logger.debug(f"[weight_sync] pause HTTP took {time.time() - t0:.1f}s")

    def resume(self) -> None:
        """Resume vLLM's engine via HTTP."""
        t0 = time.time()
        requests.post(f"{self.vllm_server_url}/resume")
        logger.debug(f"[weight_sync] resume HTTP took {time.time() - t0:.1f}s")

    def destroy(self) -> None:
        """Tear down the NCCL group. Called by the trainer at training end."""
        if self.model_update_group is None:
            return
        self.model_update_group.group.store = None
        self.model_update_group.group.socket = None
        self.model_update_group = None
