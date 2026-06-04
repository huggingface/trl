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
from huggingface_hub import create_bucket

from trl.import_utils import is_vllm_available


if is_vllm_available(min_version="0.17.1"):
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
        delta_sync_enabled: bool = False,
        delta_sync_repo_id: str | None = None,
        delta_sync_anchor_interval: int = 10,
        delta_sync_encoding: str = "gap_delta",
    ):
        if not is_vllm_available(min_version="0.17.1"):
            raise ImportError(
                "vLLM >= 0.17.1 is required to use WeightTransferClient. Install it with: pip install 'vllm>=0.17.1'"
            )
        self.vllm_server_url = vllm_server_url.rstrip("/")
        self.server_timeout = server_timeout
        self.init_weight_transfer_timeout = init_weight_transfer_timeout
        self._weight_update_info = weight_update_info
        self.model_update_group = None
        # Delta sync (Transport B): sparse patches over an HF Storage Bucket instead of NCCL.
        self.delta_sync_enabled = delta_sync_enabled
        self._delta_sync_repo_id = delta_sync_repo_id
        self._delta_sync_anchor_interval = delta_sync_anchor_interval
        self._delta_sync_encoding = delta_sync_encoding
        self._delta_model_version = 0
        self._delta_pending: dict | None = None

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
        if self.delta_sync_enabled:
            create_bucket(self._delta_sync_repo_id, exist_ok=True)
            requests.post(
                f"{self.vllm_server_url}/init_weight_transfer_engine",
                json={"init_info": {}},
                timeout=self.init_weight_transfer_timeout,
            )
            logger.info("Initialised delta weight transfer (bucket %s)", self._delta_sync_repo_id)
            return
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

    def upload_weights(self, iterator) -> None:
        """Delta phase 1 (inference still running): encode the changed params as a sparse patch,
        upload it to the bucket, and record where [`apply_weights`] should fetch it.

        Every Nth sync is a full anchor; the rest are gap-delta patches. An empty iterator (nothing changed) is a no-op
        and leaves ``_delta_pending`` cleared, so the apply is skipped. The phase is explicit — never inferred from
        emptiness — so a zero-change step can't trigger an apply.
        """
        from .delta_engine import DeltaWeightTransferEngine

        self._delta_model_version += 1
        is_anchor = self._delta_model_version == 1 or self._delta_model_version % self._delta_sync_anchor_interval == 0
        if is_anchor:
            iterator = ((name, tensor, None) for name, tensor, _mask in iterator)  # strip masks -> full tensors
        subdir = "anchors" if is_anchor else "deltas"
        filename = f"{subdir}/step_{self._delta_model_version:06d}.safetensors"
        meta = DeltaWeightTransferEngine.upload(
            iterator=iterator,
            bucket_id=self._delta_sync_repo_id,
            filename=filename,
            model_version=self._delta_model_version,
            encoding=self._delta_sync_encoding,
        )
        self._delta_pending = (
            None
            if meta is None
            else {
                "repo_id": self._delta_sync_repo_id,
                "filename": filename,
                "update_kind": "dense" if is_anchor else "sparse_flat",  # "dense" <=> anchor
            }
        )

    def apply_weights_delta(self) -> None:
        """Signal vLLM to fetch and apply the uploaded patch.

        No-op when nothing was uploaded this step; ``_delta_pending`` is cleared up front so a failed apply leaves no
        stale state.
        """
        if self._delta_pending is None:
            return
        info, self._delta_pending = self._delta_pending, None
        # Anchors are HF-checkpoint-format full tensors; deltas are sparse kernel-format.
        self._post_vllm("/start_weight_update", {"is_checkpoint_format": info["update_kind"] == "dense"})
        # vLLM fetches the patch from the bucket inside this call; a full anchor can take minutes, so
        # the timeout must cover the download — otherwise a read-timeout would retry into a re-download.
        self._post_vllm("/update_weights", {"update_info": info}, retries=5, timeout=1800)
        self._post_vllm("/finish_weight_update", {})

    def _post_vllm(self, path: str, json_body: dict, retries: int = 1, timeout: int = 300) -> None:
        """POST to a vLLM server endpoint with bounded retry on 429 / connection errors."""
        url = f"{self.vllm_server_url}{path}"
        for attempt in range(retries):
            try:
                resp = requests.post(url, json=json_body, timeout=timeout)
                if resp.status_code < 429:
                    resp.raise_for_status()
                    return
                status = resp.status_code
            except requests.RequestException as e:
                logger.warning(f"[weight_sync] POST {path} failed: {e}")
                status = "connection error"
            if attempt < retries - 1:
                wait = min(2**attempt, 30)
                logger.warning(f"[weight_sync] POST {path} -> {status}, retry in {wait}s ({attempt + 1}/{retries})")
                time.sleep(wait)
        raise RuntimeError(f"[weight_sync] POST {path} failed after {retries} attempt(s)")

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
