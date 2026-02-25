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

import json
import logging
import math
import os
from collections.abc import Callable
from contextlib import nullcontext
from typing import TYPE_CHECKING

import torch
from accelerate.utils import broadcast_object_list, gather_object, is_peft_model
from packaging.version import Version
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import PreTrainedModel, PreTrainedTokenizerBase, ProcessorMixin, is_bitsandbytes_available
from transformers.utils import is_torch_mlu_available, is_torch_npu_available, is_torch_xpu_available

from ..data_utils import apply_chat_template, is_conversational, prepare_multimodal_messages_vllm
from ..extras.profiling import ProfilingContext
from ..import_utils import is_vllm_available
from ..trainer.utils import ensure_master_addr_port
from .vllm_client import VLLMClient


def empty_cache() -> None:
    """Empties the cache of the available torch device.

    This function checks for the availability of different torch devices (XPU, MLU, NPU, CUDA) and empties the cache of
    the first available device it finds.

    If none of the specific devices are available, it defaults to emptying the CUDA cache.
    """
    if is_torch_xpu_available():
        torch.xpu.empty_cache()
    elif is_torch_mlu_available():
        torch.mlu.empty_cache()
    elif is_torch_npu_available():
        torch.npu.empty_cache()
    else:
        torch.cuda.empty_cache()


logger = logging.getLogger(__name__)


def sanitize_logprob(logprob):
    value = logprob.logprob
    if math.isnan(value):
        logger.warning(f"Generated NaN logprob, token logprob '{logprob}' will be ignored")
        return None

    return value


if TYPE_CHECKING:
    from accelerate import Accelerator
    from peft import PeftModel


if is_vllm_available():
    import vllm
    from vllm import LLM, SamplingParams

    if Version(vllm.__version__) <= Version("0.10.2"):
        from vllm.sampling_params import GuidedDecodingParams
    else:
        from vllm.sampling_params import StructuredOutputsParams

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
        is_fsdp_enabled (`bool`):
            Whether FSDP is enabled.
        processing_class ([`~transformers.PreTrainedTokenizerBase`] or [`~transformers.ProcessorMixin`]):
            Tokenizer or processor for the model.

        > Parameters for vLLM:

        mode (`str`, *optional*, defaults to `"server"`): vLLM mode. Must be one of `"server"` or
            `"colocate"`.

            - `"server"`: The trainer will send generation requests to a separate vLLM server. Make sure a TRL vLLM
              server is running (start with `trl vllm-serve`).
            - `"colocate"`: vLLM will run in the same process and share the training GPUs. This avoids the need for a
              separate server but may cause resource contention with training.
        structured_outputs_regex (`str`, *optional*):
            Regex for vLLM structured outputs. If `None` (default), structured outputs is disabled.

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

        > Parameters for generation:

        repetition_penalty (`float`, *optional*, defaults to `1.0`):
            Parameter for repetition penalty. It penalizes new tokens based on whether they appear in the prompt and
            the generated text so far. Values > 1 encourage the model to use new tokens, while values < 1 encourage the
            model to repeat tokens. Default `1.0` means no penalty.
        temperature(`float`, *optional*, defaults to `1.0`):
            Sampling temperature. It controls the randomness of the sampling. Lower values make the model more
            deterministic, while higher values make the model more random and increase diversity.
        top_p: (`float`, *optional*, defaults to `1.0`):
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
        generation_kwargs (`dict`, *optional*):
            Additional generation parameters to pass to the vLLM `SamplingParams`. This can include parameters like
            `seed`, `frequency_penalty`, etc. If it contains keys that conflict with the other parameters, they will
            override them.

        > Parameters for chat/tools:

        chat_template (`str`, *optional*):
            Template to use for structuring the chat. If not provided, the model's default chat template will be used.
        chat_template_kwargs (`dict`, *optional*):
            Additional keyword arguments to customize the chat template used by the model.
        tools (`list`, *optional*):
            Tools available for tool calling during chat generation.
        rollout_func (`Callable`, *optional*): Optional custom rollout function that accepts prompts and returns
            a dict with 'prompt_ids', 'completion_ids', 'logprobs', and optional extra fields. Should be a
            single-argument callable: rollout_func(prompts) -> dict. To pass additional context (e.g., trainer), use a
            closure or functools.partial:
                rollout_func = lambda prompts: my_custom_rollout(prompts, trainer)
            The closure will hold a reference to trainer and see its state updates.
    """

    def __init__(
        self,
        model: "PreTrainedModel | PeftModel",
        accelerator: "Accelerator",
        is_fsdp_enabled: bool,
        processing_class: PreTrainedTokenizerBase | ProcessorMixin,
        # vLLM configuration
        mode: str = "server",
        structured_outputs_regex: str | None = None,
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
        # Generation configuration
        repetition_penalty: float = 1.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
        max_completion_length: int = 16,
        generation_kwargs: dict | None = None,
        # Chat/tool configuration
        chat_template: str | None = None,
        chat_template_kwargs: dict | None = None,
        tools: list | None = None,
        rollout_func: Callable | None = None,
    ):
        self.model = model
        self.accelerator = accelerator
        self.is_fsdp_enabled = is_fsdp_enabled
        self.processing_class = processing_class

        # vLLM configuration
        self.mode = mode
        self.structured_outputs_regex = structured_outputs_regex

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

        # Generation configuration
        self.repetition_penalty = repetition_penalty
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.max_completion_length = max_completion_length
        self.generation_kwargs = generation_kwargs or {}

        # Chat/tool configuration
        self.chat_template = chat_template
        self.chat_template_kwargs = chat_template_kwargs or {}
        self.tools = tools
        self.rollout_func = rollout_func

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

        if self.mode == "server":
            if accelerator.is_main_process:
                if self.server_base_url is not None:
                    base_url = self.server_base_url
                else:
                    base_url = f"http://{self.server_host}:{self.server_port}"
                self.vllm_client = VLLMClient(
                    base_url=base_url, group_port=self.group_port, connection_timeout=self.server_timeout
                )
                self.vllm_client.init_communicator(device=torch.cuda.current_device())

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

            quantization = None
            if is_bitsandbytes_available():
                for _, module in model.named_modules():
                    if isinstance(module, bnb.nn.Linear4bit):
                        quantization = "bitsandbytes"
                        break
                    elif isinstance(module, bnb.nn.Linear8bitLt):
                        raise ValueError("vLLM does not support in-flight 8-bit quantization.")

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
            )
            if self.enable_sleep_mode:
                self.llm.sleep(level=2)
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

    def _sync_fsdp1_params_to_vllm(self, module: nn.Module, prefix: str = "", visited: set[str] | None = None):
        """Memory-efficient post-order traversal of FSDP modules to extract full parameters and sync with vLLM."""
        # For FSDP1, we need to recurse into children and also use summon_full_params
        accelerator = self.accelerator

        if visited is None:
            visited = set()
        for child_name, child_module in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix else child_name
            self._sync_fsdp1_params_to_vllm(
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

                    if self.mode == "server" and accelerator.is_main_process:
                        self.vllm_client.update_named_param(full_name, param.data)
                    elif self.mode == "colocate":
                        llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                        llm_model.load_weights([(full_name, param.data)])

    def _sync_fsdp2_params_to_vllm(self, module: nn.Module):
        """FSDP2-specific parameter synchronization."""
        accelerator = self.accelerator

        # For FSDP2, module.state_dict() already covers all parameters, so no need for recursion
        for name, param in module.state_dict().items():
            # When using PEFT, we need to recover the original parameter name
            name = name.removeprefix("base_model.model.").replace(".base_layer", "")
            # Skip PEFT layers: they don't exist in vLLM, and they are merged already.
            if is_peft_model(module) and module.prefix in name:
                continue
            # When module to save, remove its prefix and discard the original module
            if "original_module" in name:
                continue
            name = self._fix_param_name_to_vllm(name, extra_prefixes=["modules_to_save.default."])

            if param.is_cpu:
                param = param.to(torch.device("cuda"))
            param = param.full_tensor()

            if self.mode == "server" and accelerator.is_main_process:
                self.vllm_client.update_named_param(name, param)
            elif self.mode == "colocate":
                llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                llm_model.load_weights([(name, param)])

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

        model = self.model
        accelerator = self.accelerator
        is_fsdp_enabled = self.is_fsdp_enabled

        # For DeepSpeed ZeRO-3 and FSDP, we need to gather all parameters before operations
        deepspeed_plugin = accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
        if zero_stage_3:
            import deepspeed

            gather_if_zero3 = deepspeed.zero.GatheredParameters
        else:
            gather_if_zero3 = nullcontext

        if is_peft_model(model):
            # With PEFT and FSDP/DeepSpeed ZeRO Stage 3, we must gather the full model at once before merging, as
            # merging adapters in a sharded manner is not supported.
            # TODO: does this work with FSDP?
            with gather_if_zero3(list(model.parameters())):
                model.merge_adapter()

                # Update vLLM weights while parameters are gathered
                if is_fsdp_enabled:  # note if using FSDP, gather_if_zero3 is nullcontext
                    # Update vLLM weights while parameters are gathered
                    # For PEFT with FSDP we need to use the memory efficient post-order traversal
                    fsdp_plugin = getattr(accelerator.state, "fsdp_plugin", None)
                    fsdp_version = getattr(fsdp_plugin, "fsdp_version", 1) if fsdp_plugin else 1
                    if fsdp_version == 1:
                        self._sync_fsdp1_params_to_vllm(model)  # use memory-efficient post-order traversal for FSDP
                    elif fsdp_version == 2:
                        self._sync_fsdp2_params_to_vllm(model)
                else:
                    # DeepSpeed ZeRO-3 with PEFT
                    for name, param in model.named_parameters():
                        # When using PEFT, we need to recover the original parameter name
                        name = name.removeprefix("base_model.model.").replace(".base_layer", "")
                        # Skip PEFT layers: they don't exist in vLLM, and they are merged already.
                        if model.prefix in name:
                            continue
                        # When module to save, remove its prefix and discard the original module
                        if "original_module" in name:
                            continue
                        name = self._fix_param_name_to_vllm(name, extra_prefixes=["modules_to_save.default."])

                        if self.mode == "server" and accelerator.is_main_process:
                            self.vllm_client.update_named_param(name, param.data)
                        elif self.mode == "colocate":
                            llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                            llm_model.load_weights([(name, param.data)])
                # Unmerge adapters while parameters are still gathered
                model.unmerge_adapter()
                # Parameters will automatically be repartitioned when exiting the context
        else:
            # For non-PEFT models, simply gather (if needed) and update each parameter individually.
            if is_fsdp_enabled:
                fsdp_plugin = getattr(accelerator.state, "fsdp_plugin", None)
                fsdp_version = getattr(fsdp_plugin, "fsdp_version", 1) if fsdp_plugin else 1
                if fsdp_version == 1:
                    self._sync_fsdp1_params_to_vllm(model)  # use memory-efficient post-order traversal for FSDP
                elif fsdp_version == 2:
                    self._sync_fsdp2_params_to_vllm(model)
            else:
                for name, param in model.named_parameters():
                    name = self._fix_param_name_to_vllm(name)
                    with gather_if_zero3([param]):
                        if self.mode == "server" and accelerator.is_main_process:
                            self.vllm_client.update_named_param(name, param.data)
                        elif self.mode == "colocate":
                            llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                            llm_model.load_weights([(name, param.data)])

        # Reset cache on vLLM
        if self.mode == "server" and accelerator.is_main_process:
            self.vllm_client.reset_prefix_cache()
        elif self.mode == "colocate":
            self.llm.reset_prefix_cache()

    def generate(self, prompts: list, num_generations: int, profiler: ProfilingContext | None = None) -> tuple:
        """Generate completions using vLLM.

        Args:
            prompts: List of prompts (strings or chat conversations)
            num_generations: Number of generations per prompt
            profiler: Optional profiler for performance tracking

        Returns:
            Tuple of (prompt_ids, completion_ids, logprobs, extra_fields)
        """
        profiler = profiler or nullcontext()
        accelerator = self.accelerator
        rollout_func = self.rollout_func
        temperature = self.temperature
        top_p = self.top_p
        top_k = self.top_k
        min_p = self.min_p
        repetition_penalty = self.repetition_penalty
        max_completion_length = self.max_completion_length
        processing_class = self.processing_class
        chat_template_kwargs = self.chat_template_kwargs
        tools = self.tools
        chat_template = self.chat_template

        # Wake up colocated vLLM weights if needed (idempotent if already awake from sync_weights)
        if self.mode == "colocate" and self.enable_sleep_mode:
            empty_cache()  # required to avoid OOM in some cases
            self.llm.wake_up(tags=["weights"])
            # Work around for https://github.com/vllm-project/vllm/issues/29341
            try:
                self.llm.collective_rpc("reload_weights")
            except NotImplementedError:
                # Non-CUDA vLLM backends (e.g., vllm-ascend's NPUWorkerV1), don't implement reload_weights
                pass

        if is_conversational({"prompt": prompts[0]}):
            prompts = [prepare_multimodal_messages_vllm(prompt) for prompt in prompts]

        # In vLLM, tool call arguments must be JSON strings. See https://github.com/vllm-project/vllm/pull/28820
        for prompt in prompts:  # iterate over each conversation
            if is_conversational({"prompt": prompt}):
                for message in prompt:  # iterate over each message
                    if "tool_calls" in message:  # check if message has tool calls
                        for call in message["tool_calls"]:
                            args_value = call["function"]["arguments"]
                            if isinstance(args_value, dict):  # only convert dict → JSON string
                                call["function"]["arguments"] = json.dumps(args_value)

        # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
        if self.mode == "server":
            all_prompts = gather_object(prompts)

            if accelerator.is_main_process:
                # Since 'prompts' contains 'num_generations' duplicates, we first take unique prompts, and generate
                # num_generations outputs for each one. This is faster than generating outputs for each duplicate
                # prompt individually.
                ordered_set_of_prompts = all_prompts[::num_generations]

                sampling_params = {
                    "n": num_generations,
                    "repetition_penalty": repetition_penalty,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "min_p": 0.0 if min_p is None else min_p,
                    "max_tokens": max_completion_length,
                    "structured_outputs_regex": self.structured_outputs_regex,
                    "generation_kwargs": self.generation_kwargs,
                }
                with profiler:  # TODO: profiling_context(trainer, "vLLM.generate"):
                    if rollout_func is not None:
                        # Pass all prompts (with duplicates) to rollout_func for consistency with colocate mode
                        rollout_prompts = all_prompts
                        if rollout_prompts and is_conversational({"prompt": rollout_prompts[0]}):
                            rollout_prompts = [
                                apply_chat_template({"prompt": p}, processing_class, **chat_template_kwargs)["prompt"]
                                for p in rollout_prompts
                            ]
                        output = rollout_func(rollout_prompts)
                    else:
                        if is_conversational({"prompt": ordered_set_of_prompts[0]}):
                            output = self.vllm_client.chat(
                                messages=ordered_set_of_prompts,
                                **sampling_params,
                                chat_template_kwargs=chat_template_kwargs,
                                tools=tools,
                                chat_template=chat_template,
                            )
                        else:
                            output = self.vllm_client.generate(prompts=ordered_set_of_prompts, **sampling_params)
                    # Extract required fields and collect any extra fields for reward functions
                    required_keys = {"prompt_ids", "completion_ids", "logprobs"}
                    extra_fields = {k: v for k, v in output.items() if k not in required_keys}
                    payload = (output["prompt_ids"], output["completion_ids"], output["logprobs"], extra_fields)
            else:
                payload = None

            # Broadcast the completions from the main process to all processes, ensuring each process receives its corresponding slice.
            obj_list = [payload]
            broadcast_object_list(obj_list, from_process=0)
            all_prompt_ids, all_completion_ids, all_logprobs, all_extra_fields = obj_list[0]

            # When using rollout_func, it handles its own generation logic and returns one result per prompt.
            # When NOT using rollout_func, vllm_client.generate(n=num_generations) returns num_generations
            # completions per prompt, so we need to duplicate prompt_ids to match.
            if self.rollout_func is None:
                # At this point, we only get 1 copy of each prompt, so we need to repeat them num_generations times
                all_prompt_ids = [ids for ids in all_prompt_ids for _ in range(num_generations)]

            process_slice = slice(
                accelerator.process_index * len(prompts),
                (accelerator.process_index + 1) * len(prompts),
            )
            prompt_ids = all_prompt_ids[process_slice]
            completion_ids = all_completion_ids[process_slice]
            logprobs = all_logprobs[process_slice]

            # Slice extra fields dict-of-lists per process (extra fields are per-completion, like completion_ids)
            extra_fields = {}
            for key, values in all_extra_fields.items():
                if isinstance(values, list):
                    extra_fields[key] = values[process_slice]
                else:
                    extra_fields[key] = values

        # Generate completions using colocated vLLM instances: each device holds vLLM copy and work on their own batch of prompts
        elif self.mode == "colocate":
            if rollout_func is not None:
                rollout_prompts = prompts
                if rollout_prompts and is_conversational({"prompt": rollout_prompts[0]}):
                    rollout_prompts = [
                        apply_chat_template({"prompt": prompt}, processing_class, **chat_template_kwargs)["prompt"]
                        for prompt in rollout_prompts
                    ]
                output = rollout_func(rollout_prompts)
                required_keys = {"prompt_ids", "completion_ids", "logprobs"}
                extra_fields = {k: v for k, v in output.items() if k not in required_keys}
                prompt_ids = output["prompt_ids"]
                completion_ids = output["completion_ids"]
                logprobs = output["logprobs"]
            else:
                generation_kwargs = {
                    "n": 1,  # vLLM on each GPU generates only 1 in colocate mode
                    "repetition_penalty": repetition_penalty,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "min_p": 0.0 if min_p is None else min_p,
                    "max_tokens": max_completion_length,
                    "logprobs": 0,  # enable returning log probabilities; 0 means for the sampled tokens only
                }
                generation_kwargs.update(self.generation_kwargs)

                if Version(vllm.__version__) <= Version("0.10.2"):
                    structured_outputs_key = "guided_decoding"
                    if self.structured_outputs_regex is not None:
                        if generation_kwargs.get("guided_decoding") is not None:
                            logger.warning(
                                "Both `structured_outputs_regex` and `generation_kwargs['guided_decoding']` are set; "
                                "`structured_outputs_regex` takes precedence."
                            )
                        structured_outputs = GuidedDecodingParams(regex=self.structured_outputs_regex)
                    else:
                        structured_outputs = generation_kwargs.get("guided_decoding")
                else:
                    structured_outputs_key = "structured_outputs"
                    if self.structured_outputs_regex is not None:
                        if generation_kwargs.get("structured_outputs") is not None:
                            logger.warning(
                                "Both `structured_outputs_regex` and `generation_kwargs['structured_outputs']` are "
                                "set; `structured_outputs_regex` takes precedence."
                            )
                        structured_outputs = StructuredOutputsParams(regex=self.structured_outputs_regex)
                    elif isinstance(generation_kwargs.get("structured_outputs"), dict):
                        structured_outputs_dict = generation_kwargs.get("structured_outputs")
                        structured_outputs = StructuredOutputsParams(**structured_outputs_dict)
                    else:
                        structured_outputs = generation_kwargs.get("structured_outputs")

                generation_kwargs[structured_outputs_key] = structured_outputs
                sampling_params = SamplingParams(**generation_kwargs)

                if self.tensor_parallel_size > 1:
                    # Gather prompts from all ranks in the TP group and flatten.
                    # Each rank starts with its own prompts; after gathering, all ranks see the full group set.
                    orig_size = len(prompts)
                    gathered_prompts = [None for _ in range(self.tensor_parallel_size)]
                    torch.distributed.all_gather_object(gathered_prompts, prompts, group=self.tp_group)
                    all_prompts = [p for sublist in gathered_prompts for p in sublist]
                else:
                    all_prompts = prompts

                if self.enable_sleep_mode:
                    self.llm.wake_up(tags=["kv_cache"])

                with profiler:  # TODO: profiling_context(trainer, "vLLM.generate"):
                    if is_conversational({"prompt": prompts[0]}):
                        all_outputs = self.llm.chat(
                            all_prompts,
                            sampling_params=sampling_params,
                            use_tqdm=False,
                            chat_template_kwargs=chat_template_kwargs,
                            tools=tools,
                            chat_template=chat_template,
                        )
                    else:
                        all_outputs = self.llm.generate(all_prompts, sampling_params=sampling_params, use_tqdm=False)

                all_prompt_ids = [output.prompt_token_ids for output in all_outputs]
                all_completion_ids = [output.token_ids for outputs in all_outputs for output in outputs.outputs]
                all_logprobs = [
                    [sanitize_logprob(next(iter(lp.values()))) for lp in output.logprobs]
                    for outputs in all_outputs
                    for output in outputs.outputs
                ]

                if self.tensor_parallel_size > 1:
                    # Slice completions for this rank within its TP group.
                    # Each rank generates all outputs — we keep only our share.
                    local_rank_in_group = torch.distributed.get_rank(group=self.tp_group)
                    tp_slice = slice(local_rank_in_group * orig_size, (local_rank_in_group + 1) * orig_size)
                    prompt_ids = all_prompt_ids[tp_slice]
                    completion_ids = all_completion_ids[tp_slice]
                    logprobs = all_logprobs[tp_slice]
                else:
                    prompt_ids = all_prompt_ids
                    completion_ids = all_completion_ids
                    logprobs = all_logprobs

                extra_fields = {}  # No extra fields for colocate mode

                if self.enable_sleep_mode:
                    self.llm.sleep(level=2)

        return prompt_ids, completion_ids, logprobs, extra_fields
