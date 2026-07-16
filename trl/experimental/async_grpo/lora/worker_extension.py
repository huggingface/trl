"""vLLM WorkerExtension for direct LoRA weight updates via NCCL.

Runs inside each vLLM worker process. Receives LoRA A/B tensors from the
trainer over NCCL and calls module.set_lora() to do in-place copy_ into
the pre-allocated stacked tensors -- no adapter lifecycle churn, no CUDA
graph invalidation.

Loaded via ``--worker-extension-cls``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import socket
from typing import Any, Protocol

import torch
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.utils import StatelessProcessGroup

logger = logging.getLogger("vllm.worker.lora_extension")

_DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}

_LAYER_TAIL_RE = re.compile(r"layers?[._](\d+)\b(.*)")


class _WorkerLike(Protocol):
    """Minimal typing for the vLLM Worker instance that ``self`` refers to."""

    rank: int
    device: torch.device
    model_runner: Any


def _peft_to_module_and_type(peft_name: str) -> tuple[str, str]:
    """Extract (bare_module_name, 'lora_a'|'lora_b') from a PEFT param name.

    Strips ``base_model.model.``, the adapter name (``default``), and the
    ``lora_A``/``lora_B`` suffix to yield only the module path.
    """
    name = peft_name
    if name.startswith("base_model.model."):
        name = name[len("base_model.model.") :]

    name = name.replace(".lora_A.default.", ".lora_A.")
    name = name.replace(".lora_B.default.", ".lora_B.")

    if ".lora_A." in name:
        return name.split(".lora_A.")[0], "lora_a"
    elif ".lora_B." in name:
        return name.split(".lora_B.")[0], "lora_b"
    raise ValueError(f"Not a LoRA param: {peft_name}")


def _extract_layer_tail(module_name: str) -> str | None:
    """Extract the 'layers.N.suffix' tail from a module name.

    Works regardless of the prefix hierarchy -- matches the first occurrence
    of ``layer(s).N`` and returns everything from there onward.

    Examples::
        language_model.model.cross_decoder.decoder_layers.0.mlp.down_proj
            -> layers.0.mlp.down_proj
        model.language_model.layers.0.mlp.down_proj
            -> layers.0.mlp.down_proj
    """
    m = _LAYER_TAIL_RE.search(module_name)
    if m:
        return f"layers.{m.group(1)}{m.group(2)}"
    return None


def build_packed_module_map(packed_modules_mapping: dict[str, list[str]]) -> dict[str, tuple[str, int]]:
    """Build a reverse map: sub_module_suffix -> (packed_module_suffix, slot_index).

    Given packed_modules_mapping like {"qkv_proj": ["q_proj", "k_proj", "v_proj"]},
    returns {"q_proj": ("qkv_proj", 0), "k_proj": ("qkv_proj", 1), ...}.
    """
    reverse: dict[str, tuple[str, int]] = {}
    for packed_name, sub_names in packed_modules_mapping.items():
        for idx, sub_name in enumerate(sub_names):
            reverse[sub_name] = (packed_name, idx)
    return reverse


def _build_tail_to_vllm_map(all_module_names: set[str]) -> dict[str, str]:
    """Map ``layers.N.suffix`` tails to full vLLM module names."""
    tail_map: dict[str, str] = {}
    for name in all_module_names:
        tail = _extract_layer_tail(name)
        if tail:
            tail_map[tail] = name
    return tail_map


def create_stateless_pg(host: str, port: int, rank: int, world_size: int) -> StatelessProcessGroup:
    """Create a StatelessProcessGroup with the listen socket bound to 0.0.0.0.

    The default StatelessProcessGroup.create() binds to `host`, which can
    fail inside vLLM's forked EngineCore subprocess. We pre-create the
    listen socket bound to all interfaces and pass it in.
    """
    listen_socket = None
    if rank == 0:
        listen_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listen_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listen_socket.bind(("0.0.0.0", port))
        listen_socket.listen()

    return StatelessProcessGroup.create(
        host=host,
        port=port,
        rank=rank,
        world_size=world_size,
        listen_socket=listen_socket,
    )


class LoRADirectSyncExtension:
    """WorkerExtension for direct LoRA NCCL updates."""

    def init_lora_sync_group(
        self: _WorkerLike,
        master_address: str,
        master_port: int,
        world_size: int,
        rank_offset: int,
    ) -> None:
        """Initialize NCCL process group for LoRA weight transfer from trainer."""
        pg_rank = rank_offset + self.rank

        for key in (
            "NCCL_LAUNCH_MODE",
            "NCCL_COLLNET_ENABLE",
            "NCCL_NVLS_ENABLE",
            "NCCL_P2P_NET_DISABLE",
            "NCCL_MIN_NCHANNELS",
            "NCCL_MAX_NCHANNELS",
            "NCCL_PROTO",
            "NCCL_ALGO",
            "NCCL_NTHREADS",
            "NCCL_SOCKET_NTHREADS",
        ):
            os.environ.pop(key, None)

        pg = create_stateless_pg(
            host=master_address, port=master_port, rank=pg_rank, world_size=world_size
        )
        self._lora_nccl = PyNcclCommunicator(pg, device=self.device)
        logger.info(
            "LoRA sync group initialized: rank=%d/%d (pg_rank=%d), device=%s",
            self.rank,
            world_size,
            pg_rank,
            self.device,
        )

    def receive_lora_update(self: _WorkerLike, manifest_json: str) -> None:
        """Receive LoRA A/B tensors via NCCL and copy into vLLM's stacked tensors.

        Uses suffix-based matching (``layers.N.module_type``) so PEFT param
        names are mapped to vLLM module names regardless of differing prefix
        hierarchies (e.g. multimodal wrappers). Works for any model.

        The manifest is a JSON object with:
          - lora_alpha: float
          - lora_rank: int
          - lora_int_id: int
          - params: list of {name: str, shape: list[int], dtype: str}
        """
        manifest = json.loads(manifest_json)
        lora_alpha = manifest["lora_alpha"]
        lora_rank = manifest["lora_rank"]
        lora_int_id = manifest["lora_int_id"]
        param_infos = manifest["params"]
        scaling = lora_alpha / lora_rank

        worker_lora_mgr = self.model_runner.lora_manager
        lora_mgr = worker_lora_mgr._adapter_manager

        slot_index = lora_mgr.lora_index_to_id.index(lora_int_id)

        packed_reverse = build_packed_module_map(lora_mgr.packed_modules_mapping)
        all_module_names = set(lora_mgr.modules.keys())
        tail_to_vllm = _build_tail_to_vllm_map(all_module_names)

        logger.info(
            "receive_lora_update: lora_int_id=%d, slot=%d, "
            "packed=%s, n_vllm_modules=%d, n_tail_entries=%d, n_params=%d",
            lora_int_id,
            slot_index,
            lora_mgr.packed_modules_mapping,
            len(all_module_names),
            len(tail_to_vllm),
            len(param_infos),
        )

        module_weights: dict[str, dict[str, Any]] = {}

        for info in param_infos:
            shape = tuple(info["shape"])
            dtype = _DTYPE_MAP[info["dtype"]]
            buf = torch.empty(shape, dtype=dtype, device=self.device)
            self._lora_nccl.broadcast(buf, src=0, stream=torch.cuda.current_stream())

            raw_module_name, weight_type = _peft_to_module_and_type(info["name"])

            peft_tail = _extract_layer_tail(raw_module_name)
            sub_suffix = raw_module_name.rsplit(".", 1)[-1] if "." in raw_module_name else raw_module_name

            if sub_suffix in packed_reverse:
                packed_suffix, packed_slot = packed_reverse[sub_suffix]
                packed_children = lora_mgr.packed_modules_mapping[packed_suffix]
                is_identity_packed = len(packed_children) == 1 and packed_children[0] == sub_suffix

                if peft_tail:
                    packed_tail = peft_tail.rsplit(".", 1)[0] + "." + packed_suffix
                    vllm_module_name = tail_to_vllm.get(packed_tail)
                else:
                    vllm_module_name = None

                if vllm_module_name is None:
                    module_prefix = raw_module_name.rsplit(".", 1)[0] + "." if "." in raw_module_name else ""
                    vllm_module_name = module_prefix + packed_suffix

                entry = module_weights.setdefault(vllm_module_name, {})

                if weight_type == "lora_b":
                    buf = buf * scaling

                if is_identity_packed:
                    entry[weight_type] = buf
                else:
                    for wt in ("lora_a", "lora_b"):
                        if wt not in entry:
                            entry[wt] = [None] * len(packed_children)
                    entry[weight_type][packed_slot] = buf
            else:
                if peft_tail:
                    vllm_module_name = tail_to_vllm.get(peft_tail, raw_module_name)
                else:
                    vllm_module_name = raw_module_name

                entry = module_weights.setdefault(vllm_module_name, {})

                if weight_type == "lora_b":
                    buf = buf * scaling

                entry[weight_type] = buf

        updated_count = 0
        skipped_names = []
        for vllm_module_name, weights in module_weights.items():
            if vllm_module_name not in all_module_names:
                skipped_names.append(vllm_module_name)
                continue

            module = lora_mgr.modules[vllm_module_name]
            lora_a = weights.get("lora_a")
            lora_b = weights.get("lora_b")

            if lora_a is None or lora_b is None:
                logger.warning(
                    "Incomplete weights for %s (a=%s, b=%s)",
                    vllm_module_name,
                    lora_a is not None,
                    lora_b is not None,
                )
                continue

            try:
                module.set_lora(slot_index, lora_a, lora_b)
                updated_count += 1
            except Exception as e:
                logger.error("LoRA set_lora failed for %s: %s", vllm_module_name, e)
                raise

        torch.cuda.synchronize(self.device)

        for sample_name in module_weights:
            if sample_name in all_module_names:
                m = lora_mgr.modules[sample_name]
                a_max = m.lora_a_stacked[0][slot_index].abs().max().item()
                b_max = m.lora_b_stacked[0][slot_index].abs().max().item()
                logger.info(
                    "Verify stacked tensor write: %s a_max=%.8f b_max=%.8f",
                    sample_name,
                    a_max,
                    b_max,
                )
                break

        logger.info(
            "LoRA update done: %d/%d modules updated, %d skipped, slot=%d, scaling=%.4f",
            updated_count,
            len(module_weights),
            len(skipped_names),
            slot_index,
            scaling,
        )
        if skipped_names:
            logger.info("Skipped modules (not in vLLM): %s", skipped_names[:20])

    def close_lora_sync_group(self: _WorkerLike) -> None:
        if hasattr(self, "_lora_nccl") and self._lora_nccl is not None:
            del self._lora_nccl
            self._lora_nccl = None
            logger.info("LoRA sync communicator closed")
