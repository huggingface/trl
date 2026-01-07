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
import os
from contextlib import nullcontext

import torch
from accelerate.utils import broadcast_object_list, gather_object, is_peft_model
from packaging import version
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import is_bitsandbytes_available

from ..data_utils import apply_chat_template, is_conversational, prepare_multimodal_messages_vllm
from ..extras.profiling import profiling_decorator
from ..extras.vllm_client import VLLMClient
from ..import_utils import is_vllm_available
from ..trainer.utils import ensure_master_addr_port


if is_vllm_available():
    import vllm
    from vllm import LLM, SamplingParams

    if version.parse(vllm.__version__) <= version.parse("0.10.2"):
        from vllm.sampling_params import GuidedDecodingParams
    else:
        from vllm.sampling_params import StructuredOutputsParams

if is_bitsandbytes_available():
    import bitsandbytes as bnb


class VLLMGeneration:
    """Handles vLLM-based generation for trainers.

    Extracts all vLLM-specific logic (initialization, generation, weight sync)
    from trainers into a separate, testable class.
    """

    def __init__(
        self,
        model,
        accelerator,
        is_fsdp_enabled,
        processing_class,
        # vLLM configuration
        mode: str,
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
        enable_sleep_mode: bool,
        max_num_seqs: int | None = None,
        max_model_length: int | None = None,
        model_impl: str | None = None,
        structured_outputs_regex: str | None = None,
        # Server mode configuration
        server_base_url: str | None = None,
        server_host: str = "localhost",
        server_port: int = 8000,
        group_port: int | None = None,
        server_timeout: int = 600,
        # Generation configuration
        generation_kwargs: dict | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        min_p: float | None = None,
        repetition_penalty: float | None = None,
        max_completion_length: int | None = None,
        # Chat/tool configuration
        chat_template_kwargs: dict | None = None,
        tools: list | None = None,
        chat_template: str | None = None,
        rollout_func=None,
    ):
        """Initialize vLLM generation.

        Args:
            model: The model to use for generation
            accelerator: Accelerator instance for distributed training
            is_fsdp_enabled: Whether FSDP is enabled
            processing_class: Tokenizer or processor for the model
            mode: vLLM mode ('server' or 'colocate')
            tensor_parallel_size: Tensor parallel size for vLLM
            gpu_memory_utilization: GPU memory utilization (0.0-1.0)
            enable_sleep_mode: Whether to enable sleep mode
            max_num_seqs: Maximum number of sequences to process in parallel.
                If None, calculated as per_device_train_batch_size * tensor_parallel_size * steps_per_generation
            max_model_length: Maximum model length
            model_impl: Model implementation
            structured_outputs_regex: Regex for structured outputs
            server_base_url: Base URL for vLLM server (server mode)
            server_host: Host for vLLM server (server mode)
            server_port: Port for vLLM server (server mode)
            group_port: Group port for vLLM communicator (server mode)
            server_timeout: Connection timeout for vLLM server (server mode)
            generation_kwargs: Additional generation kwargs to pass to vLLM
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            min_p: Min-p sampling parameter
            repetition_penalty: Repetition penalty
            max_completion_length: Maximum completion length
            chat_template_kwargs: Chat template kwargs
            tools: Optional tools for function calling
            chat_template: Optional chat template
            rollout_func: Optional custom rollout function that accepts prompts and returns
                a dict with 'prompt_ids', 'completion_ids', 'logprobs', and optional extra fields.
                Should be a single-argument callable: rollout_func(prompts) -> dict.
                To pass additional context (e.g., trainer), use a closure or functools.partial:
                    rollout_func = lambda prompts: my_custom_rollout(prompts, trainer)
                The closure will hold a reference to trainer and see its state updates.
        """
        self.model = model
        self.accelerator = accelerator
        self.is_fsdp_enabled = is_fsdp_enabled
        self.processing_class = processing_class

        # vLLM configuration
        self.mode = mode
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.enable_sleep_mode = enable_sleep_mode
        self.max_num_seqs = max_num_seqs
        self.max_model_length = max_model_length
        self.model_impl = model_impl
        self.structured_outputs_regex = structured_outputs_regex

        # Server mode configuration
        self.server_base_url = server_base_url
        self.server_host = server_host
        self.server_port = server_port
        self.group_port = group_port
        self.server_timeout = server_timeout

        # Generation configuration
        self.generation_kwargs = generation_kwargs
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.repetition_penalty = repetition_penalty
        self.max_completion_length = max_completion_length

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
            # Make sure vllm_tensor_parallel_size group size evenly divides the world size - each group should have
            # the same number of ranks
            if not accelerator.num_processes % self.tensor_parallel_size == 0:
                raise ValueError(
                    f"vllm_tensor_parallel_size ({self.tensor_parallel_size}) must divide world size "
                    f"({accelerator.num_processes}) evenly."
                )

            if self.tensor_parallel_size > 1:
                # Create subgroups of ranks for TP, each group with `vllm_tensor_parallel_size` ranks.
                # For example, if world_size=8 and vllm_tensor_parallel_size=2 → groups: [0,1], [2,3], [4,5], [6,7]
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
            llm_kwargs = {
                "model": model.name_or_path,
                "tensor_parallel_size": self.tensor_parallel_size,
                "gpu_memory_utilization": self.gpu_memory_utilization,
                "max_num_seqs": self.max_num_seqs,
                "max_model_len": self.max_model_length,
                "distributed_executor_backend": "external_launcher",
                "seed": accelerator.process_index // self.tensor_parallel_size,
                "max_num_batched_tokens": 4096,
                "model_impl": self.model_impl,
                "enable_sleep_mode": self.enable_sleep_mode,
                # Important so temperature scaling/logit tweaking affects the TIS log probs
                "logprobs_mode": "processed_logprobs",
                "quantization": quantization,
            }
            self.llm = LLM(**llm_kwargs)
            if self.enable_sleep_mode:
                self.llm.sleep(level=2)
        else:
            raise ValueError(f"vllm_mode must be either 'server' or 'colocate', got '{self.mode}'.")

        # When using vLLM, the main process is responsible for loading the model weights. This can cause process
        # desynchronization and seems to lead to DeepSpeed hanging during initialization. To prevent this, we
        # synchronize all processes after vLLM has been fully initialized.
        accelerator.wait_for_everyone()

    def _fix_param_name_to_vllm(self, name, extra_prefixes=None):
        """Fix parameter name for vLLM compatibility (lines 991-996)."""
        extra_prefixes = extra_prefixes or []
        prefixes = ["_checkpoint_wrapped_module."] + extra_prefixes
        for prefix in prefixes:
            name = name.replace(prefix, "")
        return name

    def _sync_fsdp1_params_to_vllm(self, module, prefix="", visited=None):
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

    def _sync_fsdp2_params_to_vllm(self, module):
        """FSDP2-specific parameter synchronization (lines 1025-1046)."""
        accelerator = self.accelerator
        model = self.model

        # For FSDP2, module.state_dict() already covers all parameters, so no need for recursion
        for name, param in module.state_dict().items():
            # When using PEFT, we need to recover the original parameter name
            name = name.removeprefix("base_model.model.").replace(".base_layer", "")
            # Skip PEFT layers: they don't exist in vLLM, and they are merged already.
            if is_peft_model(model) and model.prefix in name:
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

    @profiling_decorator
    def sync_weights(self):
        """Synchronize model weights to vLLM.

        Handles FSDP, DeepSpeed, PEFT weight synchronization.
        """
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

    def generate(self, prompts, num_generations, profiler=None):
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

        # Wake up colocated vLLM instances if needed
        if self.mode == "colocate" and self.enable_sleep_mode:
            torch.cuda.empty_cache()  # required to avoid OOM in some cases
            self.llm.wake_up(tags=["weights"])
            # Work around for https://github.com/vllm-project/vllm/issues/29341
            self.llm.collective_rpc("reload_weights")

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
                        rollout_prompts = ordered_set_of_prompts
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
                if version.parse(vllm.__version__) <= version.parse("0.10.2"):
                    structured_outputs_key = "guided_decoding"
                    if self.structured_outputs_regex:
                        structured_outputs = GuidedDecodingParams(regex=self.structured_outputs_regex)
                    else:
                        structured_outputs = None
                else:
                    structured_outputs_key = "structured_outputs"
                    if self.structured_outputs_regex:
                        structured_outputs = StructuredOutputsParams(regex=self.structured_outputs_regex)
                    else:
                        structured_outputs = None

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
                generation_kwargs[structured_outputs_key] = structured_outputs
                if self.generation_kwargs is not None:
                    generation_kwargs.update(self.generation_kwargs)
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
                    [next(iter(lp.values())).logprob for lp in output.logprobs]
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
