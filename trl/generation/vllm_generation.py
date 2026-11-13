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

"""vLLM-based generation backend for TRL trainers."""

import logging
import math
import os
import re
import shutil
from contextlib import nullcontext
from typing import TYPE_CHECKING

import torch
import transformers
from accelerate.utils import broadcast_object_list, gather_object, is_peft_model
from packaging.version import Version
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor
from transformers import PreTrainedModel, PreTrainedTokenizerBase, ProcessorMixin, is_bitsandbytes_available
from transformers.utils import (
    is_peft_available,
    is_torch_mlu_available,
    is_torch_mps_available,
    is_torch_npu_available,
    is_torch_xpu_available,
)

from ..distributed import DistributedBackend
from ..extras.profiling import ProfilingContext
from ..import_utils import is_vllm_available, is_vllm_bnb_plugin_available
from ..trainer.utils import ensure_master_addr_port
from .vllm_client import VLLMClient


if is_peft_available():
    from peft import LoraConfig
    from peft.tuners.lora import LoraLayer, ParamWrapper
    from peft.utils import ModulesToSaveWrapper

if is_vllm_available():
    from vllm import LLM, RequestOutput, SamplingParams
    from vllm.lora.request import LoRARequest
    from vllm.sampling_params import StructuredOutputsParams


logger = logging.getLogger(__name__)


def empty_cache() -> None:
    """Empties the cache of the available torch device.

    This function checks for the availability of different torch devices (CUDA, MLU, MPS, NPU, XPU) and empties the
    cache of the first available device it finds.

    If none of the specific devices are available, it defaults to emptying the CUDA cache.
    """
    if is_torch_mlu_available():
        torch.mlu.empty_cache()
    elif is_torch_mps_available():
        torch.mps.empty_cache()
    elif is_torch_npu_available():
        torch.npu.empty_cache()
    elif is_torch_xpu_available():
        torch.xpu.empty_cache()
    else:
        torch.cuda.empty_cache()


def extract_logprobs(all_outputs: list["RequestOutput"]):
    """
    Extract logprobs and token IDs from vLLM generation outputs.

    Returns logprobs and token IDs sorted by rank (most probable first). Each returned list has shape (num_sequences,
    seq_len, num_logprobs), where num_logprobs is determined by the `logprobs` parameter passed to vLLM (1 when
    `logprobs=0`, up to N+1 when `logprobs=N`). NaN logprob values are replaced with `None`.

    Args:
        all_outputs (list of `RequestOutput`):
            List of vLLM `RequestOutput` objects from generation.

    Returns:
        Tuple of (logprobs, logprob_token_ids), each of shape (num_sequences, seq_len, num_logprobs).
    """
    all_logprobs = []
    all_token_ids = []
    for outputs in all_outputs:
        for output in outputs.outputs:
            if output.logprobs is None:
                return None, None
            seq_logprobs = []
            seq_token_ids = []
            for lp in output.logprobs:
                sorted_items = sorted(lp.items(), key=lambda x: x[1].rank)
                seq_token_ids.append([token_id for token_id, _ in sorted_items])
                seq_logprobs.append([None if math.isnan(item.logprob) else item.logprob for _, item in sorted_items])
            all_logprobs.append(seq_logprobs)
            all_token_ids.append(seq_token_ids)
    return all_logprobs, all_token_ids


@torch.no_grad()
def _merge_lora(module: "LoraLayer", weight: torch.Tensor) -> torch.Tensor:
    """
    Merge the active adapters of a LoRA layer into the weight it adapts, out of place.

    The training model is left untouched: merging into it and unmerging after the sync leaves rounding errors in the
    frozen weight (#7423), and the unmerge can land before vLLM has read the merged weight (#8003).

    Args:
        module ([`~peft.tuners.lora.LoraLayer`]):
            LoRA layer whose active adapters are merged.
        weight (`torch.Tensor`):
            Weight adapted by `module`, dequantized if it is quantized.

    Returns:
        `torch.Tensor`:
            Merged weight, in the dtype of `weight`.
    """
    merged = weight
    for adapter in module.active_adapters:
        if adapter in module.lora_variant:  # e.g. DoRA
            merged = module.lora_variant[adapter].merge_safe(module, adapter, merged)
        elif adapter in module.lora_A or adapter in module.lora_embedding_A:
            merged = merged + module.get_delta_weight(adapter)
    return merged.to(weight.dtype)


if TYPE_CHECKING:
    from accelerate import Accelerator
    from peft import PeftModel


# Ranks vLLM builds stacked LoRA buffers for. Any other value crashes engine construction.
VLLM_LORA_RANKS = (1, 8, 16, 32, 64, 128, 256, 320, 512)


def round_lora_rank(adapter_rank: int) -> int:
    """
    Round a LoRA rank up to the smallest `max_lora_rank` vLLM accepts.

    `max_lora_rank` is a capacity bound rather than the served rank, so an `r=4` adapter is served correctly under `8`.

    Args:
        adapter_rank (`int`):
            Rank of the adapter to serve.

    Returns:
        `int`:
            Smallest value in `VLLM_LORA_RANKS` greater than or equal to `adapter_rank`.

    Examples:

    ```python
    >>> round_lora_rank(4)
    8
    ```
    """
    for rank in VLLM_LORA_RANKS:
        if rank >= adapter_rank:
            return rank
    raise ValueError(
        f"LoRA rank {adapter_rank} is larger than the largest rank vLLM can serve ({VLLM_LORA_RANKS[-1]}). Lower `r` "
        f"(and `rank_pattern`) to sync the adapter to vLLM, or sync merged weights instead."
    )


def validate_lora_for_vllm_sync(model: "PeftModel") -> "LoraConfig":
    """
    Return the active adapter's config, checking it can be served as a plain vLLM LoRA adapter.

    vLLM's LoRA manager only knows about `lora_A` / `lora_B` deltas on a fixed set of linear layers. Anything else a
    PEFT config can express — fully-trained modules, DoRA magnitudes, trained biases, fused-MoE parameter LoRA — has no
    home in the adapter checkpoint vLLM loads, and is silently ignored (or worse, misread) at serving time. Every such
    config is rejected here so the caller can fall back to syncing merged weights, which handles all of them.

    Args:
        model ([`~peft.PeftModel`]):
            PEFT model whose active adapter is checked.

    Returns:
        [`~peft.LoraConfig`]:
            Config of the single active adapter.
    """
    if len(model.active_adapters) != 1:
        raise ValueError(
            f"Adapter-only vLLM sync serves a single adapter, but {len(model.active_adapters)} are active "
            f"({model.active_adapters}). Activate one adapter, or sync merged weights instead."
        )
    peft_config = model.peft_config[model.active_adapters[0]]
    if not isinstance(peft_config, LoraConfig):
        raise ValueError(
            f"Adapter-only vLLM sync supports LoRA adapters only, got `{type(peft_config).__name__}`. Sync merged "
            f"weights instead."
        )
    if peft_config.modules_to_save:
        raise ValueError(
            f"`modules_to_save={peft_config.modules_to_save}` cannot ride in a LoRA adapter: those modules are "
            f"fully trained, not a low-rank delta, so vLLM would keep serving the base checkpoint's copy. Sync "
            f"merged weights instead."
        )
    if peft_config.use_dora:
        raise ValueError(
            "DoRA adapters are not servable by vLLM: the magnitude vector has no slot in the adapter format. Sync "
            "merged weights instead."
        )
    if peft_config.bias != "none":
        raise ValueError(
            f"`bias='{peft_config.bias}'` trains bias terms, which are not part of a LoRA adapter and would be "
            f"dropped on the way to vLLM. Use `bias='none'`, or sync merged weights instead."
        )
    if peft_config.target_parameters:
        raise ValueError(
            f"`target_parameters={peft_config.target_parameters}` (fused-MoE parameter LoRA) needs the adapter to be "
            f"declared with `is_3d_lora_weight=True` against a server started with "
            f"`--enable-mixed-moe-lora-format`. vLLM does not inspect the checkpoint, so a wrong declaration loads "
            f"the weights into the wrong stacked buffers and silently produces garbage. Sync merged weights instead."
        )
    target_modules = peft_config.target_modules or []
    if isinstance(target_modules, str):  # a regex matching module names, rather than a collection of them
        target_modules = [target_modules]
    if any("lm_head" in module or "embed_tokens" in module for module in target_modules):
        raise ValueError(
            f"`target_modules={peft_config.target_modules}` targets the head or the embeddings. vLLM restricts LoRA "
            f"on those layers and drops the head delta from `prompt_logprobs` entirely "
            f"(https://github.com/vllm-project/vllm/issues/51594), so the trainer and the server would disagree on "
            f"every logprob. Sync merged weights instead."
        )
    return peft_config


def save_lora_adapter(model: "PeftModel", accelerator: "Accelerator", adapter_name: str, dest_dir: str) -> None:
    """
    Save one PEFT adapter to `dest_dir`, gathering sharded parameters as needed.

    This is a collective: every process must call it, not only the main one, because materializing a sharded parameter
    all-gathers across ranks.

    `dest_dir` is named after the adapter and holds `adapter_config.json` at its root, the layout vLLM's filesystem
    LoRA resolver expects when it serves `<cache_dir>/<adapter_name>`. It is published with an atomic rename, so a
    server reading it over a shared filesystem never observes a half-flushed adapter.

    Args:
        model ([`~peft.PeftModel`]):
            PEFT model holding the adapter.
        accelerator ([`~accelerate.Accelerator`]):
            Accelerator whose device sharded parameters are gathered on.
        adapter_name (`str`):
            Name of the adapter to save.
        dest_dir (`str`):
            Directory the adapter is published to. Replaced if it already exists.
    """
    state_dict = {
        name: param for name, param in model.state_dict().items() if "lora_" in name and f".{adapter_name}." in name
    }
    materialized = {}
    for name, param in state_dict.items():
        # With `fsdp_offload_params`, the local shard lives on CPU; the all-gather needs it on device.
        if param.is_cpu:
            param = param.to(accelerator.device)
        if isinstance(param, DTensor):  # FSDP2
            param = param.full_tensor()
        materialized[name] = param.detach().cpu()

    if not accelerator.is_main_process:
        return
    # Staged under a dot-prefixed name: `dest_dir`'s parent is an adapter namespace, so a plain `<name>.tmp` sibling
    # would be a half-written adapter sitting in it, and a crash mid-save would leak one there permanently.
    tmp_dir = os.path.join(os.path.dirname(dest_dir), f".{os.path.basename(dest_dir)}.tmp")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)
    model.save_pretrained(tmp_dir, selected_adapters=[adapter_name], state_dict=materialized, safe_serialization=True)
    # PEFT nests every adapter but `"default"` in a subdirectory, while vLLM expects `adapter_config.json` at the root
    # of the path it is handed.
    saved_dir = tmp_dir if adapter_name == "default" else os.path.join(tmp_dir, adapter_name)
    shutil.rmtree(dest_dir, ignore_errors=True)
    os.rename(saved_dir, dest_dir)  # atomic within one filesystem
    shutil.rmtree(tmp_dir, ignore_errors=True)


if is_bitsandbytes_available():
    import bitsandbytes as bnb


class VLLMGeneration:
    """Handles vLLM-based generation for trainers.

    Extracts all vLLM-specific logic (initialization, generation, weight sync) from trainers into a separate, testable
    class.

    Args:
        model ([`~transformers.PreTrainedModel`] or [`~peft.PeftModel`]):
            Model to use for generation.
        accelerator ([`~accelerate.Accelerator`]):
            Accelerator for distributed training.
        processing_class ([`~transformers.PreTrainedTokenizerBase`] or [`~transformers.ProcessorMixin`]):
            Tokenizer or processor for the model.

        > Parameters for vLLM:

        mode (`str`, *optional*, defaults to `"colocate"`):
            vLLM mode. Must be one of `"colocate"` or `"server"`.

            - `"colocate"`: vLLM will run in the same process and share the training GPUs. This avoids the need for a
              separate server but may cause resource contention with training.
            - `"server"`: The trainer will send generation requests to a separate vLLM server. Make sure a vLLM server
              is running (start with `vllm serve`).

        structured_outputs_regex (`str`, *optional*):
            Regex for vLLM structured outputs. If `None` (default), structured outputs is disabled.
        lora_sync_output_dir (`str`, *optional*, defaults to `"."`):
            Directory under which the LoRA adapter is saved for vLLM to load, when only the adapter is synced. The vLLM
            server, or every node in `"colocate"` mode, must be able to read it.

        > Parameters for "server" vLLM mode:

        server_base_url (`str`, *optional*):
            Base URL for the vLLM server (e.g., `"http://localhost:8000"`). If provided, `server_host` and
            `server_port` are ignored.
        server_host (`str`, *optional*, defaults to `"0.0.0.0"`):
            Host of the vLLM server to connect to. Ignored if `server_base_url` is provided.
        server_port (`int`, *optional*, defaults to `8000`):
            Port of the vLLM server to connect to. Ignored if `server_base_url` is provided.
        server_timeout (`float`, *optional*, defaults to `240.0`):
            Total timeout duration in seconds to wait for the vLLM server to be up. If the server is not up after the
            timeout, a `ConnectionError` is raised.
        group_port (`int`, *optional*, defaults to `51216`):
            Port number for the weight update group. This is used to communicate with the vLLM server. Unless the port
            is occupied, there is no need to change it.

        > Parameters for "colocate" vLLM mode:

        tensor_parallel_size (`int`, *optional*, defaults to `1`):
            The number of GPUs to use for distributed execution with tensor parallelism. This setting only applies when
            `mode` is set to `"colocate"`. If you are using `mode="server"`, this parameter must be passed separately
            when launching the vLLM server via the `--vllm_tensor_parallel_size` flag.
        gpu_memory_utilization (`float`, *optional*, defaults to `0.9`):
            Ratio (between 0 and 1) of GPU memory to reserve for the model weights, activations, and KV cache. Higher
            values will increase the KV cache size and thus improve the model's throughput. However, if the value is
            too high, it may cause out-of- memory (OOM) errors. This setting only applies when `mode` is set to
            `"colocate"`. If you are using `mode="server"`, this parameter must be passed separately when launching the
            vLLM server via the `--vllm_gpu_memory_utilization` flag.
        max_model_length (`int`, *optional*):
            Model context length (prompt and completion). Set it to at least the maximum prompt length in the dataset
            plus `max_completion_length`; if omitted, it is inferred from the model config.
        max_num_seqs (`int`, *optional*):
            Maximum number of sequences to process in parallel, effectively capping the batch size.
        enable_sleep_mode (`bool`, *optional*, defaults to `False`):
            Whether to enable sleep mode for the engine to offload weights/cache during the optimizer step. Keeps GPU
            memory usage low, but waking the engine adds host–device transfer latency.
        model_impl (`str`, *optional*, defaults to `"auto"`):
            Model implementation to use for vLLM.
            - "auto" will try to use the vLLM implementation, if it exists, and fall back to the Transformers
              implementation if no vLLM implementation is available.
            - "vllm" will use the vLLM model implementation.
            - "transformers" will use the Transformers model implementation.
            - "terratorch" will use the TerraTorch model implementation.
        trust_remote_code (`bool`, *optional*, defaults to `False`):
            Trust remote code (e.g., from HuggingFace) when downloading the model and tokenizer.

        > Parameters for generation:

        repetition_penalty (`float`, *optional*, defaults to `1.0`):
            Parameter for repetition penalty. It penalizes new tokens based on whether they appear in the prompt and
            the generated text so far. Values > 1 encourage the model to use new tokens, while values < 1 encourage the
            model to repeat tokens. Default `1.0` means no penalty.
        temperature (`float`, *optional*, defaults to `1.0`):
            Sampling temperature. It controls the randomness of the sampling. Lower values make the model more
            deterministic, while higher values make the model more random and increase diversity.
        top_p (`float`, *optional*, defaults to `1.0`):
            Top-p sampling parameter. It controls the cumulative probability of the top tokens to consider. Defaults to
            `1.0` to consider all tokens.
        top_k (`int`, *optional*, defaults to `0`):
            Top-k sampling parameter. It controls the number of top tokens to consider. Defaults to `0` to consider all
            tokens.
        min_p (`float`, *optional*, defaults to `0.0`):
            Min-p sampling parameter. It represents the minimum probability for a token to be considered, relative to
            the probability of the most likely token. Default `0.0` means min-p is disabled.
        max_completion_length (`int`, *optional*, defaults to `16`):
            Maximum number of tokens to generate for each prompt.
        logprobs (`int` or `None`, *optional*, defaults to `0`):
            Number of top logprobs to return per token. When 0 (default), only the sampled token's logprob is returned
            (inner dimension = 1). When N>0, returns up to N+1 logprobs sorted by descending probability, because vLLM
            always includes the sampled token's logprob alongside the top-N (the sampled token may or may not already
            be in the top-N).
        generation_kwargs (`dict`, *optional*):
            Additional generation parameters to pass to the vLLM `SamplingParams`. This can include parameters like
            `seed`, `frequency_penalty`, etc. If it contains keys that conflict with the other parameters, they will
            override them.
        return_sampling_mask (`bool`, *optional*, defaults to `False`):
            Whether to also return, for every sampled token, the set of tokens it was sampled from after top-k, top-p
            and min-p filtering. Requires vLLM >= 0.28 and `top_k > 0`, and in server mode a server started with
            `--return-sampling-mask` (and `--enable-scale-out` from vLLM 0.30).

    """

    def __init__(
        self,
        model: "PreTrainedModel | PeftModel",
        accelerator: "Accelerator",
        processing_class: PreTrainedTokenizerBase | ProcessorMixin,
        # vLLM configuration
        mode: str = "colocate",
        structured_outputs_regex: str | None = None,
        lora_sync_output_dir: str = ".",
        # Server mode configuration
        server_base_url: str | None = None,
        server_host: str = "0.0.0.0",
        server_port: int = 8000,
        server_timeout: float = 240.0,
        group_port: int = 51216,
        # Colocate mode configuration
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_length: int | None = None,
        max_num_seqs: int | None = None,
        enable_sleep_mode: bool = False,
        model_impl: str = "auto",
        trust_remote_code: bool = False,
        # Generation configuration
        repetition_penalty: float = 1.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
        max_completion_length: int = 16,
        logprobs: int | None = 0,
        generation_kwargs: dict | None = None,
        return_sampling_mask: bool = False,
    ):
        self.model = model
        self.accelerator = accelerator
        self._dist = DistributedBackend(accelerator)
        self.processing_class = processing_class

        # vLLM configuration
        self.mode = mode
        self.structured_outputs_regex = structured_outputs_regex
        self.lora_sync_output_dir = lora_sync_output_dir

        # Server mode configuration
        self.server_base_url = server_base_url
        self.server_host = server_host
        self.server_port = server_port
        self.group_port = group_port
        self.server_timeout = server_timeout

        # Colocate mode configuration
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_length = max_model_length
        self.max_num_seqs = max_num_seqs
        self.enable_sleep_mode = enable_sleep_mode
        self.model_impl = model_impl
        self.trust_remote_code = trust_remote_code

        # Generation configuration
        self.repetition_penalty = repetition_penalty
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.max_completion_length = max_completion_length
        self.logprobs = logprobs
        self.generation_kwargs = {}
        if isinstance(processing_class, ProcessorMixin):
            # A sampled image token would mismatch the image features; -100 is the lowest bias vLLM accepts
            self.generation_kwargs["logit_bias"] = {model.config.image_token_id: -100}
            # transformers<5.3.0 crashes when a completion ends with <|vision_start|> (see #8163)
            if Version(transformers.__version__) < Version("5.3.0") and model.config.model_type in (
                "qwen2_vl",
                "qwen2_5_vl",
                "qwen3_vl",
                "qwen3_vl_moe",
                "qwen3_5",
                "qwen3_5_moe",
                "paddleocr_vl",
            ):
                self.generation_kwargs["logit_bias"][model.config.vision_start_token_id] = -100
        self.generation_kwargs.update(generation_kwargs or {})
        self.return_sampling_mask = return_sampling_mask

        # Tensor names, dtypes and shapes streamed to the server on each weight sync. Collected on the first sync, as
        # it requires gathering the parameters, and constant afterwards.
        self._weight_metadata = None

        # Whether only the LoRA adapter is synced, instead of the merged weights. Decided in `_init_vllm`.
        self.lora_sync = False
        # Each sync publishes the adapter under a new name, `trl-policy-v{N}`. `lora_name` is the one last published,
        # which server requests select as their model.
        self.lora_name = None
        self._lora_version = 0
        # The adapter colocated generation runs with, set by `sync_weights`
        self._lora_request = None

        if is_peft_model(model):
            # The weight sync sends vLLM the base weights with the LoRA deltas merged in, which other PEFT methods and
            # LoRA biases don't fit
            if any(not isinstance(config, LoraConfig) or config.lora_bias for config in model.peft_config.values()):
                raise ValueError(
                    "vLLM generation with a PEFT model supports LoRA adapters without `lora_bias` only: the weight "
                    "sync sends vLLM the base weights with the LoRA deltas merged in."
                )
            # vLLM holds tied word embeddings as one tensor, so PEFT must not train separate copies of them: #7955
            base_model = model.get_base_model()
            embeddings = [base_model.get_input_embeddings(), base_model.get_output_embeddings()]
            originals = [m.original_module if isinstance(m, ModulesToSaveWrapper) else m for m in embeddings]
            if originals[0].weight is originals[1].weight and embeddings[0].weight is not embeddings[1].weight:
                names = {module: name.rsplit(".", 1)[-1] for name, module in base_model.named_modules()}
                copies = " and ".join(f"`{names[m]}`" for m in embeddings if isinstance(m, ModulesToSaveWrapper))
                raise ValueError(
                    "The model ties its input embeddings and output layer, but PEFT's `modules_to_save` trains them "
                    "as separate copies, which vLLM cannot hold. Set `ensure_weight_tying=True` in the PEFT config "
                    f"(PEFT >= 0.18) to train a single copy, or remove {copies} from `modules_to_save`."
                )

        self._init_vllm()

    def _init_vllm(self):
        """Initialize vLLM in server or colocate mode."""
        model = self.model
        accelerator = self.accelerator

        if not is_vllm_available():
            raise ImportError(
                "vLLM is not available and `use_vllm` is set to True. Please install vLLM with "
                "`pip install trl[vllm]` to use it."
            )

        # vLLM can serve a plain LoRA adapter as is, so syncing the adapter is enough: no need to merge it and push the
        # full weights. `unservable` says why the adapter can't be, `None` if it can.
        unservable = None
        if is_peft_model(model):
            try:
                peft_config = validate_lora_for_vllm_sync(model)
                # `rank_pattern` can lift individual modules above the base `r`.
                adapter_rank = max([peft_config.r, *peft_config.rank_pattern.values()])
                max_lora_rank = round_lora_rank(adapter_rank)
            except ValueError as error:
                unservable = str(error)

        # Adapter sync never replaces vLLM's copy of the base weights, so a 4-bit base must be quantized in vLLM as in
        # the trainer, or the two disagree from the first step. vLLM quantizes it as NF4, whatever the trainer's 4-bit
        # type.
        quant_types = set()
        if is_bitsandbytes_available():
            quant_types = {
                module.weight.quant_type for module in model.modules() if isinstance(module, bnb.nn.Linear4bit)
            }

        if self.mode == "server":
            error = None
            lora_version = 0
            if accelerator.is_main_process:
                if self.server_base_url is not None:
                    base_url = self.server_base_url
                else:
                    base_url = f"http://{self.server_host}:{self.server_port}"
                self.vllm_client = VLLMClient(
                    base_url=base_url, group_port=self.group_port, connection_timeout=self.server_timeout
                )
                server_info = self.vllm_client.get_server_info()
                # `lora_config` is `None` unless the server was started with `--enable-lora`
                server_lora_config = server_info["lora_config"]
                server_quantization = server_info["model_config"]["quantization"]
                if server_lora_config is None:
                    lora_sync = False
                    if is_peft_model(model):
                        logger.warning(
                            "Training a PEFT model against a vLLM server started without `--enable-lora`: every "
                            "weight sync has to merge the adapter into the base model and push the full weights. "
                            "Restart the server with `--enable-lora --max-lora-rank <r>` and "
                            "`VLLM_ALLOW_RUNTIME_LORA_UPDATING=1` in its environment to push only the adapter instead."
                        )
                elif not is_peft_model(model):
                    lora_sync = False
                elif unservable is not None:
                    lora_sync = False
                    logger.warning(f"Falling back to merged-weight vLLM sync. {unservable}")
                elif server_lora_config["max_lora_rank"] < adapter_rank:
                    lora_sync = False
                    error = (
                        f"The adapter has rank {adapter_rank}, but the vLLM server was started with `--max-lora-rank "
                        f"{server_lora_config['max_lora_rank']}`. Restart it with `--max-lora-rank {max_lora_rank}`."
                    )
                # Merged weights can't be pushed into a server holding a 4-bit base either, so these can't sync at all
                elif server_quantization == "bitsandbytes" and quant_types != {"nf4"}:
                    lora_sync = False
                    error = (
                        "The vLLM server quantizes its base model as NF4 (`--quantization bitsandbytes`), but the "
                        f"trainer's base is {', '.join(sorted(quant_types)) or 'not quantized'}, so the two would "
                        "disagree on every logprob. Quantize the trainer's base as NF4, or restart the server without "
                        "`--quantization`."
                    )
                elif server_quantization is None and quant_types:
                    lora_sync = False
                    logger.warning(
                        "Falling back to merged-weight vLLM sync. The trainer's base is quantized in 4-bit but the vLLM "
                        "server's isn't, so the two would disagree on every logprob. Restart the server with "
                        "`--quantization bitsandbytes` and an NF4 base to sync only the adapter."
                    )
                else:
                    lora_sync = True
                    # A previous run against this server may have left its last adapter loaded, and loading another
                    # one under the same name fails. Numbering on from it avoids that, and the first sync unloads it.
                    served = self.vllm_client._get(f"{self.vllm_client.base_url}/v1/models")["data"]
                    versions = [re.fullmatch(r"trl-policy-v(\d+)", model["id"]) for model in served]
                    lora_version = max((int(match[1]) for match in versions if match), default=0)
            else:
                lora_sync = None
            # Every process takes part in syncing the weights, so they must all agree on how, and fail together
            obj_list = [lora_sync, error, lora_version]
            broadcast_object_list(obj_list, from_process=0)
            self.lora_sync, error, lora_version = obj_list
            if error is not None:
                raise ValueError(error)
            if lora_version:
                self._lora_version = lora_version
                self.lora_name = f"trl-policy-v{lora_version}"
            if accelerator.is_main_process and not self.lora_sync:
                self.vllm_client.init_communicator(device=accelerator.device)

        elif self.mode == "colocate":
            # Make sure tensor_parallel_size group size evenly divides the world size - each group should have
            # the same number of ranks
            if not accelerator.num_processes % self.tensor_parallel_size == 0:
                raise ValueError(
                    f"tensor_parallel_size ({self.tensor_parallel_size}) must divide world size "
                    f"({accelerator.num_processes}) evenly."
                )

            if self.tensor_parallel_size > 1:
                # Create subgroups of ranks for TP, each group with `tensor_parallel_size` ranks.
                # For example, if world_size=8 and tensor_parallel_size=2 → groups: [0,1], [2,3], [4,5], [6,7]
                self.tp_group, _ = torch.distributed.new_subgroups_by_enumeration(
                    [
                        list(range(i * self.tensor_parallel_size, (i + 1) * self.tensor_parallel_size))
                        for i in range(accelerator.num_processes // self.tensor_parallel_size)
                    ]
                )

            # vLLM requires the environment variables to be set for distributed training.
            os.environ["RANK"] = str(accelerator.process_index)
            os.environ["LOCAL_RANK"] = str(accelerator.local_process_index)
            os.environ["WORLD_SIZE"] = str(accelerator.num_processes)
            # Ensure distributed rendezvous variables are set without colliding across concurrent runs
            ensure_master_addr_port()

            if is_bitsandbytes_available():
                for _, module in model.named_modules():
                    if isinstance(module, bnb.nn.Linear8bitLt):
                        raise ValueError("vLLM does not support in-flight 8-bit quantization.")

            # The trainer owns the engine, so it enables LoRA on it whenever the adapter can be served as is
            self.lora_sync = is_peft_model(model) and unservable is None
            if is_peft_model(model) and unservable is not None:
                logger.warning(f"Falling back to merged-weight vLLM sync. {unservable}")

            quantization = None
            if self.lora_sync and quant_types and quant_types != {"nf4"}:
                self.lora_sync = False
                logger.warning(
                    f"Falling back to merged-weight vLLM sync. vLLM quantizes 4-bit bases as NF4, but this model is "
                    f"quantized as {', '.join(sorted(quant_types))}, so the two would disagree on every logprob."
                )
            elif self.lora_sync and quant_types:
                quantization = "bitsandbytes"
                # Without the plugin, vLLM fails with "Unknown quantization method: bitsandbytes"
                if is_vllm_available("0.28.0") and not is_vllm_bnb_plugin_available():
                    raise ImportError(
                        "vLLM 0.28.0 moved bitsandbytes support to a plugin, which syncing the LoRA adapter of a "
                        "quantized model requires. Install it with `pip install vllm-bnb-plugin`."
                    )

            # Build LLM initialization kwargs
            self.llm = LLM(
                model=model.name_or_path,
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_model_len=self.max_model_length,
                max_num_seqs=self.max_num_seqs,
                enable_sleep_mode=self.enable_sleep_mode,
                model_impl=self.model_impl,
                distributed_executor_backend="external_launcher",
                # Feed identical seed for tp groups to ensure sampling results are the same across workers
                seed=accelerator.process_index // self.tensor_parallel_size,
                # Latest vLLM v1 memory profiler is misled by the high default value (i.e., 32768) - thinking there's not enough memory
                max_num_batched_tokens=4096,
                # Important so temperature scaling/logit tweaking affects the TIS log probs
                logprobs_mode="processed_logprobs",
                quantization=quantization,
                trust_remote_code=self.trust_remote_code,
                # Not passed otherwise: vLLM < 0.28 doesn't accept it
                **({"return_sampling_mask": True} if self.return_sampling_mask else {}),
                **({"enable_lora": True, "max_lora_rank": max_lora_rank} if self.lora_sync else {}),
            )
            # Sleep level 2 discards the weights, which are pushed back at every sync. With adapter-only sync, the base
            # weights are never pushed, so level 1 offloads them to CPU instead, and waking up restores them.
            self._sleep_level = 1 if self.lora_sync else 2
            if self.enable_sleep_mode:
                self.llm.sleep(level=self._sleep_level)
            # Track whether the weights are off the device, so that generate() knows it must bring them back
            self._llm_weights_sleeping = self.enable_sleep_mode
        else:
            raise ValueError(f"vllm_mode must be either 'server' or 'colocate', got '{self.mode}'.")

        # When using vLLM, the main process is responsible for loading the model weights. This can cause process
        # desynchronization and seems to lead to DeepSpeed hanging during initialization. To prevent this, we
        # synchronize all processes after vLLM has been fully initialized.
        accelerator.wait_for_everyone()

    def _fix_param_name_to_vllm(self, name: str, extra_prefixes: list[str] | None = None) -> str:
        """Fix parameter name for vLLM compatibility."""
        extra_prefixes = extra_prefixes or []
        prefixes = ["_checkpoint_wrapped_module."] + extra_prefixes
        for prefix in prefixes:
            name = name.replace(prefix, "")
        return name

    def _iter_fsdp1_params(self, module: nn.Module, prefix: str = "", visited: set[str] | None = None):
        """Memory-efficient post-order traversal of FSDP modules to extract full parameters."""
        # For FSDP1, we need to recurse into children and also use summon_full_params
        if visited is None:
            visited = set()
        for child_name, child_module in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix else child_name
            yield from self._iter_fsdp1_params(
                child_module, prefix=child_prefix, visited=visited
            )  # recurse into the child

        if isinstance(module, FSDP):
            with FSDP.summon_full_params(module, recurse=False, writeback=False):
                for param_name, param in module.named_parameters():
                    full_name = f"{prefix}.{param_name}" if prefix else param_name
                    full_name = self._fix_param_name_to_vllm(full_name, extra_prefixes=["_fsdp_wrapped_module."])

                    if full_name in visited:
                        continue  # skip FSDP subtrees already traversed
                    visited.add(full_name)

                    yield full_name, param.data

    def _iter_fsdp2_params(self, module: nn.Module, lora_layers: dict):
        """FSDP2-specific parameter iteration."""
        # For FSDP2, module.named_parameters() covers all parameters and yields tied weights once, as vLLM requires
        for name, param in module.named_parameters():
            # When using PEFT, we need to recover the original parameter name
            name = name.removeprefix("base_model.model.").replace(".base_layer", "")
            # Skip PEFT layers: they don't exist in vLLM, and they are merged into the weights they adapt.
            if is_peft_model(module) and module.prefix in name:
                continue
            # When module to save, remove its prefix and discard the original module
            # and the other adapters' copies (e.g. a "ref" adapter)
            if "original_module" in name or ("modules_to_save." in name and "modules_to_save.default." not in name):
                continue
            name = self._fix_param_name_to_vllm(name, extra_prefixes=["modules_to_save.default."])

            if param in lora_layers:
                param = _merge_lora(lora_layers[param], param)
            if param.is_cpu:
                param = param.to(self.accelerator.device)
            param = param.full_tensor()

            yield name, param

    def _iter_fsdp_params(self, model: nn.Module, lora_layers: dict):
        """Dispatch FSDP parameter iteration to the version-appropriate method."""
        if self._dist.fsdp_version == 1:
            yield from self._iter_fsdp1_params(model)
        elif self._dist.fsdp_version == 2:
            yield from self._iter_fsdp2_params(model, lora_layers)

    def _iter_named_params(self):
        """Iterate over the model parameters, materialized one at a time under the name vLLM expects.

        Handles FSDP, DeepSpeed and PEFT. Gathering a parameter is a collective operation, so every process must
        iterate, even the ones that don't push the weights anywhere.
        """
        model = self.model

        if is_peft_model(model):
            # The weight each LoRA layer adapts, which vLLM receives with the adapters merged in
            lora_layers = {
                module.get_param() if isinstance(module, ParamWrapper) else module.get_base_layer().weight: module
                for module in model.modules()
                if isinstance(module, LoraLayer)
            }
            # With DeepSpeed ZeRO Stage 3, the full model is gathered at once, so that the adapters of each LoRA layer
            # are gathered along with the weight they adapt.
            with self._dist.gather_params(list(model.parameters())):
                # Read the vLLM weights while parameters are gathered
                if self._dist.is_fsdp:  # note if using FSDP, gather_params is a no-op
                    # For PEFT with FSDP we need to use the memory efficient post-order traversal
                    yield from self._iter_fsdp_params(model, lora_layers)
                else:
                    # DeepSpeed ZeRO-3 with PEFT
                    for name, param in model.named_parameters():
                        # When using PEFT, we need to recover the original parameter name
                        name = name.removeprefix("base_model.model.").replace(".base_layer", "")
                        # Skip PEFT layers: they don't exist in vLLM, and they are merged into the weights they adapt.
                        if model.prefix in name:
                            continue
                        # When module to save, remove its prefix and discard the original module
                        # and the other adapters' copies (e.g. a "ref" adapter)
                        if "original_module" in name or (
                            "modules_to_save." in name and "modules_to_save.default." not in name
                        ):
                            continue
                        name = self._fix_param_name_to_vllm(name, extra_prefixes=["modules_to_save.default."])

                        weight = param.data
                        # vLLM holds the model unquantized, so bitsandbytes 4-bit weights are sent dequantized
                        if is_bitsandbytes_available() and isinstance(param, bnb.nn.Params4bit):
                            weight = bnb.functional.dequantize_4bit(param.data, param.quant_state)
                        if param in lora_layers:
                            weight = _merge_lora(lora_layers[param], weight)
                        yield name, weight
                # Parameters will automatically be repartitioned when exiting the context
        else:
            # For non-PEFT models, simply gather (if needed) and read each parameter individually.
            if self._dist.is_fsdp:
                yield from self._iter_fsdp_params(model, {})
            else:
                for name, param in model.named_parameters():
                    name = self._fix_param_name_to_vllm(name)
                    with self._dist.gather_params([param]):
                        yield name, param.data

    def sync_weights(self):
        """Synchronize model weights to vLLM.

        Handles FSDP, DeepSpeed, PEFT weight synchronization.
        """
        # Wake up vLLM weights before loading to ensure device memory is mapped. Without this, load_weights() writes to
        # freed/unmapped memory when sleep mode is active, which crashes on backends with strict physical memory
        # management (e.g., Ascend NPU). See https://github.com/huggingface/trl/issues/5142
        if self.mode == "colocate" and self.enable_sleep_mode:
            empty_cache()  # required to avoid OOM in some cases
            self.llm.wake_up(tags=["weights"])
            self._llm_weights_sleeping = False

        accelerator = self.accelerator

        if self.lora_sync:
            model = self.model
            previous_name = self.lora_name
            self._lora_version += 1
            self.lora_name = f"trl-policy-v{self._lora_version}"
            # The leading underscore keeps it out of what `Trainer.push_to_hub` uploads from the output directory. Only
            # the main process writes it, so it must be on a filesystem every reader can see, as for checkpoints.
            lora_dir = os.path.join(self.lora_sync_output_dir, "_vllm_lora_sync")
            adapter_dir = os.path.join(lora_dir, self.lora_name)
            # Every process calls this, not just the main one: gathering a sharded parameter is a collective
            with self._dist.gather_params([param for param in model.parameters() if param.requires_grad]):
                save_lora_adapter(model, accelerator, model.active_adapters[0], adapter_dir)
            accelerator.wait_for_everyone()

            # Each sync publishes a new name instead of overwriting the previous adapter in place, because
            # `load_inplace=True` makes vLLM < 0.33 re-read the adapter from disk at every step
            # (https://github.com/vllm-project/vllm/pull/41482). A new name is also a fresh prefix cache namespace, so
            # the cache needs no reset.
            if self.mode == "server":
                if accelerator.is_main_process:
                    self.vllm_client.load_lora_adapter(self.lora_name, adapter_dir)
                    # The server applies the adapter to the requests that name it as their model
                    self.vllm_client.model = self.lora_name
                    if previous_name is not None:
                        self.vllm_client.unload_lora_adapter(previous_name)
            elif self.mode == "colocate":
                self._lora_request = LoRARequest(self.lora_name, self._lora_version, adapter_dir)
                self.llm.llm_engine.add_lora(self._lora_request)
                if self._lora_version > 1:
                    self.llm.llm_engine.remove_lora(self._lora_version - 1)
            accelerator.wait_for_everyone()
            # vLLM resolves the adapter files lazily, so they outlive the unload by one sync
            if accelerator.is_main_process and self._lora_version > 2:
                shutil.rmtree(os.path.join(lora_dir, f"trl-policy-v{self._lora_version - 2}"), ignore_errors=True)
            return

        if self.mode == "server":
            # The server must know every tensor it is about to receive before the first one is broadcast, so the
            # parameters are walked once to collect their metadata, and streamed on subsequent passes.
            if self._weight_metadata is None:
                self._weight_metadata = [
                    (name, str(param.dtype).removeprefix("torch."), list(param.shape))
                    for name, param in self._iter_named_params()
                ]
            if accelerator.is_main_process:
                self.vllm_client.update_named_params(self._weight_metadata, self._iter_named_params())
            else:
                for _ in self._iter_named_params():  # take part in the gather collectives
                    pass
        elif self.mode == "colocate":
            for name, param in self._iter_named_params():
                self.llm.llm_engine.model_executor.driver_worker.model_runner.model.load_weights([(name, param)])

        # Reset cache on vLLM
        if self.mode == "server" and accelerator.is_main_process:
            self.vllm_client.reset_prefix_cache()
        elif self.mode == "colocate":
            self.llm.reset_prefix_cache()

    def _place_features(self, features: dict | None, prompt_ids: list[int]) -> dict | None:
        """Point the image features at the image tokens of `prompt_ids`.

        The server reports where the images sit in the throwaway conversation it processed them in, which says nothing
        about the prompt being trained on, so their positions are recomputed from the runs of image tokens in the
        trainer's own token IDs.
        """
        if features is None:
            return None

        image_token_id = self.model.config.image_token_id
        placeholders = []
        offset = 0
        while offset < len(prompt_ids):
            if prompt_ids[offset] == image_token_id:
                length = 0
                while offset + length < len(prompt_ids) and prompt_ids[offset + length] == image_token_id:
                    length += 1
                placeholders.append({"offset": offset, "length": length})
                offset += length
            else:
                offset += 1

        expected = len(features["mm_placeholders"]["image"])
        if len(placeholders) != expected:
            raise ValueError(
                f"Found {len(placeholders)} runs of image tokens in the prompt but {expected} images were processed. "
                "The prompt must contain one run of image tokens per image."
            )
        return {**features, "mm_placeholders": {**features["mm_placeholders"], "image": placeholders}}

    def generate(
        self,
        prompts: list[list[int]],
        images: list[list | None] | None,
        num_generations: int,
        profiler: ProfilingContext | None = None,
    ) -> tuple:
        """Generate completions using vLLM.

        Args:
            prompts: List of token ID lists, one per prompt (already tokenized).
            images: Optional list of image lists for VLM support. Each element is a list of PIL images for the
                corresponding prompt, or `None` if no images for that prompt. `None` if no images at all.
            num_generations: Number of generations per prompt.
            profiler: Optional profiler for performance tracking.

        Returns:
            Tuple of (prompt_ids, completion_ids, logprobs, logprob_token_ids, sampling_mask).

            - `prompt_ids`: `list[list[int]]` of shape `(batch_size, prompt_len)`.
            - `completion_ids`: `list[list[int]]` of shape `(batch_size, completion_len)`.
            - `logprobs`: `list[list[list[float | None]]]` of shape `(batch_size, completion_len, num_logprobs)`.
            - `logprob_token_ids`: `list[list[list[int]]]` of shape `(batch_size, completion_len, num_logprobs)`.
            - `sampling_mask`: `list[list[list[int]]]` of shape `(batch_size, completion_len, set_size)`, the tokens
              each completion token was sampled from, or `None` unless `return_sampling_mask=True`.

            `num_logprobs` is 1 when `logprobs=0`, or up to N+1 when `logprobs=N` (the sampled token is always included
            and may fall outside the top-N).
        """
        profiler = profiler or nullcontext()
        accelerator = self.accelerator
        temperature = self.temperature
        top_p = self.top_p
        top_k = self.top_k
        min_p = self.min_p
        repetition_penalty = self.repetition_penalty
        max_completion_length = self.max_completion_length

        # Bring back the weights a previous generate() put to sleep. sync_weights() does it as well, but the trainer
        # only calls it when the training step advanced, so a second generate() within a step has to do it here.
        if self.mode == "colocate" and self.enable_sleep_mode and self._llm_weights_sleeping:
            if self._sleep_level == 2:
                # Sleep level 2 discards the weights, so waking up isn't enough: they must be re-pushed from the
                # training model. vLLM's `reload_weights` can't be used here, as it reloads the initial checkpoint
                # from disk rather than the current training weights.
                # See https://github.com/vllm-project/vllm/issues/29341
                self.sync_weights()
            else:
                # Sleep level 1 offloads the weights to CPU, and the adapter is still loaded, so waking up is enough.
                # Only done when asleep: waking up maps the memory again even if it is already mapped.
                self.llm.wake_up(tags=["weights"])
                self._llm_weights_sleeping = False

        # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
        if self.mode == "server":
            all_prompts = gather_object(prompts)
            # Always gather images (even when None) to avoid deadlock: images may be None on some ranks
            # and non-None on others in mixed datasets, and gather_object is a collective operation.
            all_images = gather_object(images if images is not None else [None] * len(prompts))
            if all(img is None for img in all_images):
                all_images = None

            if accelerator.is_main_process:
                # Since 'prompts' contains 'num_generations' duplicates, we first take unique prompts, and
                # generate num_generations outputs for each one. This is faster than generating outputs for each
                # duplicate prompt individually.
                ordered_set_of_prompt_ids = all_prompts[::num_generations]

                # The server generates from either token IDs or images, so images are processed on their own first
                # and the resulting features are paired with the token IDs.
                features = None
                if all_images is not None:
                    features = self.vllm_client.image_features(all_images[::num_generations])
                    features = [
                        self._place_features(prompt_features, prompt_ids)
                        for prompt_features, prompt_ids in zip(features, ordered_set_of_prompt_ids, strict=True)
                    ]

                sampling_params = {
                    "n": num_generations,
                    "repetition_penalty": repetition_penalty,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "min_p": 0.0 if min_p is None else min_p,
                    "max_tokens": max_completion_length,
                    "logprobs": self.logprobs,
                    "structured_outputs_regex": self.structured_outputs_regex,
                    "generation_kwargs": self.generation_kwargs,
                    "return_sampling_mask": self.return_sampling_mask,
                }
                with profiler:
                    output = self.vllm_client.generate(
                        prompts=ordered_set_of_prompt_ids, features=features, **sampling_params
                    )
                    payload = (
                        output["prompt_ids"],
                        output["completion_ids"],
                        output["logprobs"],
                        output.get("logprob_token_ids"),
                        output.get("sampling_mask"),
                    )
            else:
                payload = None

            # Broadcast the completions from the main process to all processes, ensuring each process receives its corresponding slice.
            obj_list = [payload]
            broadcast_object_list(obj_list, from_process=0)
            all_prompt_ids, all_completion_ids, all_logprobs, all_logprob_token_ids, all_sampling_mask = obj_list[0]

            # vllm_client.generate(n=num_generations) returns num_generations completions per prompt.
            # Duplicate prompt_ids to align with per-completion entries.
            all_prompt_ids = [ids for ids in all_prompt_ids for _ in range(num_generations)]

            process_slice = slice(
                accelerator.process_index * len(prompts),
                (accelerator.process_index + 1) * len(prompts),
            )
            prompt_ids = all_prompt_ids[process_slice]
            completion_ids = all_completion_ids[process_slice]
            logprobs = all_logprobs[process_slice] if all_logprobs is not None else None
            logprob_token_ids = all_logprob_token_ids[process_slice] if all_logprob_token_ids is not None else None
            sampling_mask = all_sampling_mask[process_slice] if all_sampling_mask is not None else None

        # Generate completions using colocated vLLM instances: each device holds vLLM copy and work on their own batch of prompts
        elif self.mode == "colocate":
            generation_kwargs = {
                "n": 1,  # vLLM on each GPU generates only 1 in colocate mode
                "repetition_penalty": repetition_penalty,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "min_p": 0.0 if min_p is None else min_p,
                "max_tokens": max_completion_length,
                "logprobs": self.logprobs,
            }
            generation_kwargs.update(self.generation_kwargs)

            if self.structured_outputs_regex is not None:
                if generation_kwargs.get("structured_outputs") is not None:
                    logger.warning(
                        "Both `structured_outputs_regex` and `generation_kwargs['structured_outputs']` are set; "
                        "`structured_outputs_regex` takes precedence."
                    )
                generation_kwargs["structured_outputs"] = StructuredOutputsParams(regex=self.structured_outputs_regex)
            elif isinstance(structured_outputs_kwargs := generation_kwargs.get("structured_outputs"), dict):
                generation_kwargs["structured_outputs"] = StructuredOutputsParams(**structured_outputs_kwargs)
            sampling_params = SamplingParams(**generation_kwargs)

            if self.tensor_parallel_size > 1:
                # Gather prompts from all ranks in the TP group and flatten.
                # Each rank starts with its own prompts; after gathering, all ranks see the full group set.
                orig_size = len(prompts)
                gathered_prompts = [None for _ in range(self.tensor_parallel_size)]
                torch.distributed.all_gather_object(gathered_prompts, prompts, group=self.tp_group)
                all_prompts = [p for sublist in gathered_prompts for p in sublist]
                # Always gather images (even when None) to avoid deadlock: images may be None on some
                # ranks and non-None on others in mixed datasets, and all_gather_object is collective.
                local_images = images if images is not None else [None] * len(prompts)
                gathered_images = [None for _ in range(self.tensor_parallel_size)]
                torch.distributed.all_gather_object(gathered_images, local_images, group=self.tp_group)
                all_images = [img for sublist in gathered_images for img in sublist]
                if all(img is None for img in all_images):
                    all_images = None
            else:
                all_prompts = prompts
                all_images = images

            if self.enable_sleep_mode:
                self.llm.wake_up(tags=["kv_cache"])

            # Build vLLM-compatible prompt inputs with token IDs and optional multi-modal data
            vllm_prompts = []
            if all_images is not None:
                for ids, img_list in zip(all_prompts, all_images, strict=True):
                    row = {"prompt_token_ids": ids}
                    if img_list is not None:
                        row["multi_modal_data"] = {"image": img_list if len(img_list) > 1 else img_list[0]}
                    vllm_prompts.append(row)
            else:
                vllm_prompts = [{"prompt_token_ids": ids} for ids in all_prompts]

            # When PEFT is used, DDP gradient all-reduce only covers the small LoRA parameters, so
            # NCCL operations complete very quickly. On non-NVLink hardware (e.g. A40/A100), vLLM's
            # TP NCCL collective can race with NCCL's internal P2P/SHM channel cleanup from that
            # all-reduce, causing llm.generate() to hang. A barrier on the default process group
            # forces NCCL to fully drain before vLLM's TP communication starts. We pass device_ids
            # so NCCL uses this rank's device rather than guessing, which itself risks a hang.
            # See https://github.com/huggingface/trl/issues/3671
            if is_peft_model(self.model) and self.tensor_parallel_size > 1:
                torch.distributed.barrier(device_ids=[accelerator.local_process_index])

            with profiler:
                all_outputs = self.llm.generate(
                    vllm_prompts, sampling_params=sampling_params, lora_request=self._lora_request, use_tqdm=False
                )

            all_prompt_ids = [output.prompt_token_ids for output in all_outputs]
            all_completion_ids = [output.token_ids for outputs in all_outputs for output in outputs.outputs]
            all_logprobs, all_logprob_token_ids = extract_logprobs(all_outputs)
            all_sampling_mask = None
            if self.return_sampling_mask:
                all_sampling_mask = [
                    output.sampling_mask.token_ids for outputs in all_outputs for output in outputs.outputs
                ]

            if self.tensor_parallel_size > 1:
                # Slice completions for this rank within its TP group.
                # Each rank generates all outputs — we keep only our share.
                local_rank_in_group = torch.distributed.get_rank(group=self.tp_group)
                tp_slice = slice(local_rank_in_group * orig_size, (local_rank_in_group + 1) * orig_size)
                prompt_ids = all_prompt_ids[tp_slice]
                completion_ids = all_completion_ids[tp_slice]
                logprobs = all_logprobs[tp_slice] if all_logprobs is not None else None
                logprob_token_ids = all_logprob_token_ids[tp_slice] if all_logprob_token_ids is not None else None
                sampling_mask = all_sampling_mask[tp_slice] if all_sampling_mask is not None else None
            else:
                prompt_ids = all_prompt_ids
                completion_ids = all_completion_ids
                logprobs = all_logprobs
                logprob_token_ids = all_logprob_token_ids
                sampling_mask = all_sampling_mask

            if self.enable_sleep_mode:
                self.llm.sleep(level=self._sleep_level)
                self._llm_weights_sleeping = True

        return prompt_ids, completion_ids, logprobs, logprob_token_ids, sampling_mask
