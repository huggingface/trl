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

import contextlib
import threading
import time
from collections import Counter

from accelerate.logging import get_logger

from ...import_utils import is_vllm_available
from .vllm_client import VLLMClient


if is_vllm_available(min_version="0.22.0"):
    from vllm.distributed.weight_transfer.nccl_engine import NCCLTrainerSendWeightsArgs, NCCLWeightTransferEngine
    from vllm.utils.network_utils import get_ip, get_open_port


logger = get_logger(__name__)


class WeightTransferClient:
    """Streams the trainer's weights into the vLLM server over NCCL.

    Each transfer runs its NCCL side on a daemon thread while the HTTP request that drives the server stays on the main
    thread: NCCL has no timeout and cannot be interrupted, so a failure surfaces as the HTTP error rather than a hang,
    and the abandoned thread dies with the process.

    Args:
        vllm_client ([`VLLMClient`]):
            Client for the vLLM server that receives the weights.
        weight_update_info (`dict`):
            Names, dtypes and shapes of the parameters to send. The server sizes its receive buffers from it, so it
            must describe exactly what [`send_weights`] streams.
        weight_sync_timeout (`int`, *optional*, defaults to `1800`):
            Seconds allowed for the steps that scale with model size: the NCCL handshake, the transfer itself, and the
            finalisation that follows it. Pause, resume and the reload setup are bounded by `_CONTROL_TIMEOUT` instead.
    """

    _CONTROL_TIMEOUT = 300

    def __init__(
        self,
        vllm_client: VLLMClient,
        weight_update_info: dict,
        weight_sync_timeout: int = 1800,
    ):
        if not is_vllm_available(min_version="0.22.0"):
            raise ImportError(
                "vLLM >= 0.22.0 is required to use WeightTransferClient. Install it with: pip install 'vllm>=0.22.0'"
            )
        self.vllm = vllm_client
        self.weight_sync_timeout = weight_sync_timeout
        self._weight_update_info = weight_update_info
        self.model_update_group = None

    def init_weight_transfer(self) -> None:
        self.vllm.wait_for_server_ready()
        # Trainer and server precisions should agree: the precision gap between the two biases the importance ratio
        # https://huggingface.co/papers/2510.26788
        # https://huggingface.co/spaces/aminediroHF/trainer-generator-bf16-mismatch
        train_dtype = Counter(self._weight_update_info["dtype_names"]).most_common(1)[0][0]
        vllm_dtype = self.vllm.get_dtype().removeprefix("torch.")
        if vllm_dtype != train_dtype:
            logger.warning(
                f"The vLLM server serves in {vllm_dtype} but the weights sent to it are {train_dtype}. Set `dtype` in "
                f"`AsyncGRPOConfig` to '{vllm_dtype}', or start the server with `--dtype {train_dtype}`."
            )
        inference_world_size = self.vllm.get_world_size()
        world_size = inference_world_size + 1
        master_address = get_ip()
        master_port = get_open_port()
        init_info = {
            "master_address": master_address,
            "master_port": master_port,
            "rank_offset": 1,
            "world_size": world_size,
        }

        error: list[BaseException] = []

        def trainer_init():
            try:
                self.model_update_group = NCCLWeightTransferEngine.trainer_init(
                    {
                        "master_address": master_address,
                        "master_port": master_port,
                        "world_size": world_size,
                    }
                )
            except BaseException as exc:  # noqa: BLE001
                error.append(exc)

        thread = threading.Thread(target=trainer_init, daemon=True)
        thread.start()
        try:
            self.vllm.init_weight_transfer_engine(init_info, timeout=self.weight_sync_timeout)
            thread.join()
            if error:
                raise error[0]
        except Exception as exc:
            raise RuntimeError(
                f"Failed to set up the NCCL weight-transfer group with the vLLM server at {self.vllm.server_url}. "
                "Check that the server was started with `VLLM_SERVER_DEV_MODE=1` and "
                '`--weight-transfer-config \'{"backend":"nccl"}\'`, and that the trainer can reach it on the '
                "port it advertises."
            ) from exc
        logger.info("Initialised weight-transfer NCCL group with vLLM")

    def send_weights(self, iterator) -> None:
        if self.model_update_group is None:
            return
        t0 = time.time()
        # Prepare the workers for the reload; must complete before any weights are sent.
        self.vllm.start_weight_update(timeout=self._CONTROL_TIMEOUT)

        error: list[BaseException] = []

        def trainer_send_weights():
            try:
                NCCLWeightTransferEngine.trainer_send_weights(
                    iterator=iterator,
                    trainer_args=NCCLTrainerSendWeightsArgs(group=self.model_update_group, packed=True),
                )
            except BaseException as exc:  # noqa: BLE001
                error.append(exc)

        thread = threading.Thread(target=trainer_send_weights, daemon=True)
        thread.start()
        try:
            self.vllm.update_weights(self._weight_update_info, timeout=self.weight_sync_timeout)
            thread.join()
            if error:
                raise error[0]
        except Exception as exc:
            # Best-effort: a failure to clean up must not mask the error below.
            with contextlib.suppress(Exception):
                self.vllm.finish_weight_update(timeout=self.weight_sync_timeout)
            raise RuntimeError(
                f"Weight sync to the vLLM server at {self.vllm.server_url} failed. The server keeps the partially "
                "updated weights but stays paused, and a new run re-sends every parameter at start-up. If the "
                "underlying error is a CUDA OOM on the server, restart it with a lower `--gpu-memory-utilization`: "
                "receiving a transfer needs a few GB of headroom on top of the weights and the KV cache."
            ) from exc
        self.vllm.finish_weight_update(timeout=self.weight_sync_timeout)
        logger.debug(f"[weight_sync] send_weights took {time.time() - t0:.1f}s")

    def pause(self) -> None:
        t0 = time.time()
        self.vllm.pause(timeout=self._CONTROL_TIMEOUT)
        logger.debug(f"[weight_sync] pause HTTP took {time.time() - t0:.1f}s")

    def resume(self) -> None:
        t0 = time.time()
        self.vllm.resume(timeout=self._CONTROL_TIMEOUT)
        logger.debug(f"[weight_sync] resume HTTP took {time.time() - t0:.1f}s")

    def destroy(self) -> None:
        if self.model_update_group is None:
            return
        self.model_update_group.group.store = None
        self.model_update_group.group.socket = None
        self.model_update_group = None
