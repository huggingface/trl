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

from accelerate.logging import get_logger

from ...import_utils import is_vllm_available
from .vllm_client import VLLMClient


if is_vllm_available(min_version="0.22.0"):
    from vllm.distributed.weight_transfer.nccl_engine import NCCLTrainerSendWeightsArgs, NCCLWeightTransferEngine
    from vllm.utils.network_utils import get_ip, get_open_port


logger = get_logger(__name__)


class WeightTransferClient:
    def __init__(
        self,
        vllm_client: VLLMClient,
        weight_update_info: dict,
        init_weight_transfer_timeout: int = 1800,
    ):
        if not is_vllm_available(min_version="0.22.0"):
            raise ImportError(
                "vLLM >= 0.22.0 is required to use WeightTransferClient. Install it with: pip install 'vllm>=0.22.0'"
            )
        self.vllm = vllm_client
        self.init_weight_transfer_timeout = init_weight_transfer_timeout
        self._weight_update_info = weight_update_info
        self.model_update_group = None

    def init_weight_transfer(self) -> None:
        self.vllm.wait_for_server_ready()
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
        t_init = threading.Thread(
            target=self.vllm.init_weight_transfer_engine,
            args=(init_info, self.init_weight_transfer_timeout),
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
        # Prepare the workers for the reload; must complete before any weights are sent.
        self.vllm.start_weight_update()
        # /update_weights drives the workers' blocking NCCL recv, so it runs on a thread
        # concurrently with the trainer-side broadcast.
        t_update = threading.Thread(target=self.vllm.update_weights, args=(self._weight_update_info,))
        t_update.start()
        NCCLWeightTransferEngine.trainer_send_weights(
            iterator=iterator,
            trainer_args=NCCLTrainerSendWeightsArgs(group=self.model_update_group, packed=True),
        )
        t_update.join()
        self.vllm.finish_weight_update()
        logger.debug(f"[weight_sync] send_weights took {time.time() - t0:.1f}s")

    def pause(self) -> None:
        t0 = time.time()
        self.vllm.pause()
        logger.debug(f"[weight_sync] pause HTTP took {time.time() - t0:.1f}s")

    def resume(self) -> None:
        t0 = time.time()
        self.vllm.resume()
        logger.debug(f"[weight_sync] resume HTTP took {time.time() - t0:.1f}s")

    def destroy(self) -> None:
        if self.model_update_group is None:
            return
        self.model_update_group.group.store = None
        self.model_update_group.group.socket = None
        self.model_update_group = None
