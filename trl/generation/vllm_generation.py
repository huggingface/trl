# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
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
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import is_bitsandbytes_available

from ..data_utils import apply_chat_template, is_conversational, prepare_multimodal_messages_vllm
from ..extras.profiling import profiling_context, profiling_decorator
from ..extras.vllm_client import VLLMClient
from ..import_utils import is_vllm_available
from ..trainer.utils import ensure_master_addr_port


if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams

if is_bitsandbytes_available():
    import bitsandbytes as bnb


class VLLMGeneration:
    """Handles vLLM-based generation for trainers.

    Extracts all vLLM-specific logic (initialization, generation, weight sync)
    from trainers into a separate, testable class.
    """

    def __init__(self, trainer):
        """Initialize vLLM generation.

        Args:
            trainer: Reference to parent trainer for accessing config, model, accelerator, etc.
        """
        self.trainer = trainer
        self._init_vllm()

    def _init_vllm(self):
        """Initialize vLLM in server or colocate mode."""
        # Access trainer attributes for convenience
        args = self.trainer.args
        model = self.trainer.model
        accelerator = self.trainer.accelerator

        if not is_vllm_available():
            raise ImportError(
                "vLLM is not available and `use_vllm` is set to True. Please install vLLM with "
                "`pip install trl[vllm]` to use it."
            )

        if args.vllm_mode == "server":
            if accelerator.is_main_process:
                if args.vllm_server_base_url is not None:
                    base_url = args.vllm_server_base_url
                else:
                    base_url = f"http://{args.vllm_server_host}:{args.vllm_server_port}"
                self.vllm_client = VLLMClient(
                    base_url=base_url, group_port=args.vllm_group_port, connection_timeout=args.vllm_server_timeout
                )
                self.vllm_client.init_communicator(device=torch.cuda.current_device())

        elif args.vllm_mode == "colocate":
            # Make sure vllm_tensor_parallel_size group size evenly divides the world size - each group should have
            # the same number of ranks
            if not accelerator.num_processes % args.vllm_tensor_parallel_size == 0:
                raise ValueError(
                    f"vllm_tensor_parallel_size ({args.vllm_tensor_parallel_size}) must divide world size "
                    f"({accelerator.num_processes}) evenly."
                )

            if args.vllm_tensor_parallel_size > 1:
                # Create subgroups of ranks for TP, each group with `vllm_tensor_parallel_size` ranks.
                # For example, if world_size=8 and vllm_tensor_parallel_size=2 → groups: [0,1], [2,3], [4,5], [6,7]
                self.tp_group, _ = torch.distributed.new_subgroups_by_enumeration(
                    [
                        list(range(i * args.vllm_tensor_parallel_size, (i + 1) * args.vllm_tensor_parallel_size))
                        for i in range(accelerator.num_processes // args.vllm_tensor_parallel_size)
                    ]
                )

            # vLLM requires the environment variables to be set for distributed training.
            os.environ["RANK"] = str(accelerator.process_index)
            os.environ["LOCAL_RANK"] = str(accelerator.local_process_index)
            os.environ["WORLD_SIZE"] = str(accelerator.num_processes)
            # Ensure distributed rendezvous variables are set without colliding across concurrent runs
            ensure_master_addr_port()

            vllm_quantization = None
            if is_bitsandbytes_available():
                for _, module in model.named_modules():
                    if isinstance(module, bnb.nn.Linear4bit):
                        vllm_quantization = "bitsandbytes"
                        break
                    elif isinstance(module, bnb.nn.Linear8bitLt):
                        raise ValueError("vLLM does not support in-flight 8-bit quantization.")

            # Build LLM initialization kwargs
            llm_kwargs = {
                "model": model.name_or_path,
                "tensor_parallel_size": args.vllm_tensor_parallel_size,
                "gpu_memory_utilization": args.vllm_gpu_memory_utilization,
                "max_num_seqs": args.per_device_train_batch_size
                * args.vllm_tensor_parallel_size
                * args.steps_per_generation,
                "max_model_len": args.vllm_max_model_length,
                "distributed_executor_backend": "external_launcher",
                "seed": accelerator.process_index // args.vllm_tensor_parallel_size,
                "max_num_batched_tokens": 4096,
                "model_impl": args.vllm_model_impl,
                "enable_sleep_mode": args.vllm_enable_sleep_mode,
                # Important so temperature scaling/logit tweaking affects the TIS log probs
                "logprobs_mode": "processed_logprobs",
                "quantization": vllm_quantization,
            }
            self.llm = LLM(**llm_kwargs)
            if args.vllm_enable_sleep_mode:
                self.llm.sleep(level=2)
        else:
            raise ValueError(f"vllm_mode must be either 'server' or 'colocate', got '{args.vllm_mode}'.")

        # vLLM specific sampling arguments
        self.guided_decoding_regex = args.vllm_guided_decoding_regex

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
        args = self.trainer.args
        accelerator = self.trainer.accelerator

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

                    if args.vllm_mode == "server" and accelerator.is_main_process:
                        self.vllm_client.update_named_param(full_name, param.data)
                    elif args.vllm_mode == "colocate":
                        llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                        llm_model.load_weights([(full_name, param.data)])

    def _sync_fsdp2_params_to_vllm(self, module):
        """FSDP2-specific parameter synchronization (lines 1025-1046)."""
        args = self.trainer.args
        accelerator = self.trainer.accelerator
        model = self.trainer.model

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

            if args.vllm_mode == "server" and accelerator.is_main_process:
                self.vllm_client.update_named_param(name, param)
            elif args.vllm_mode == "colocate":
                llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                llm_model.load_weights([(name, param)])

    @profiling_decorator
    def sync_weights(self):
        """Synchronize model weights to vLLM.

        Handles FSDP, DeepSpeed, PEFT weight synchronization.
        """
        model = self.trainer.model
        args = self.trainer.args
        accelerator = self.trainer.accelerator
        is_fsdp_enabled = self.trainer.is_fsdp_enabled

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

                        if args.vllm_mode == "server" and accelerator.is_main_process:
                            self.vllm_client.update_named_param(name, param.data)
                        elif args.vllm_mode == "colocate":
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
                        if args.vllm_mode == "server" and accelerator.is_main_process:
                            self.vllm_client.update_named_param(name, param.data)
                        elif args.vllm_mode == "colocate":
                            llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                            llm_model.load_weights([(name, param.data)])

        # Reset cache on vLLM
        if args.vllm_mode == "server" and accelerator.is_main_process:
            self.vllm_client.reset_prefix_cache()
        elif args.vllm_mode == "colocate":
            self.llm.reset_prefix_cache()

    def generate(self, prompts, num_generations, mode="train"):
        """Generate completions using vLLM.

        Args:
            prompts: List of prompts (strings or chat conversations)
            num_generations: Number of generations per prompt
            mode: "train" or "eval"

        Returns:
            Tuple of (prompt_ids, completion_ids, logprobs, extra_fields)
        """
        # Access trainer attributes
        args = self.trainer.args
        accelerator = self.trainer.accelerator
        rollout_func = self.trainer.rollout_func
        temperature = self.trainer.temperature
        top_p = self.trainer.top_p
        top_k = self.trainer.top_k
        min_p = self.trainer.min_p
        repetition_penalty = self.trainer.repetition_penalty
        max_completion_length = self.trainer.max_completion_length
        processing_class = self.trainer.processing_class
        chat_template_kwargs = self.trainer.chat_template_kwargs
        tools = self.trainer.tools
        chat_template = self.trainer.chat_template

        # Wake up colocated vLLM instances if needed
        if args.vllm_mode == "colocate" and args.vllm_enable_sleep_mode:
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
        if args.vllm_mode == "server":
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
                    "guided_decoding_regex": self.guided_decoding_regex,
                    "generation_kwargs": args.generation_kwargs,
                }
                with profiling_context(self.trainer, "vLLM.generate"):
                    if rollout_func is not None:
                        rollout_prompts = ordered_set_of_prompts
                        if rollout_prompts and is_conversational({"prompt": rollout_prompts[0]}):
                            rollout_prompts = [
                                apply_chat_template({"prompt": p}, processing_class, **chat_template_kwargs)["prompt"]
                                for p in rollout_prompts
                            ]
                        output = rollout_func(rollout_prompts, self.trainer)
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
        elif args.vllm_mode == "colocate":
            if rollout_func is not None:
                rollout_prompts = prompts
                if rollout_prompts and is_conversational({"prompt": rollout_prompts[0]}):
                    rollout_prompts = [
                        apply_chat_template({"prompt": prompt}, processing_class, **chat_template_kwargs)["prompt"]
                        for prompt in rollout_prompts
                    ]
                output = rollout_func(rollout_prompts, self.trainer)
                required_keys = {"prompt_ids", "completion_ids", "logprobs"}
                extra_fields = {k: v for k, v in output.items() if k not in required_keys}
                prompt_ids = output["prompt_ids"]
                completion_ids = output["completion_ids"]
                logprobs = output["logprobs"]
            else:
                if self.guided_decoding_regex:
                    guided_decoding = GuidedDecodingParams(regex=self.guided_decoding_regex)
                else:
                    guided_decoding = None

                generation_kwargs = {
                    "n": 1,  # vLLM on each GPU generates only 1 in colocate mode
                    "repetition_penalty": repetition_penalty,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "min_p": 0.0 if min_p is None else min_p,
                    "max_tokens": max_completion_length,
                    "guided_decoding": guided_decoding,
                    "logprobs": 0,  # enable returning log probabilities; 0 means for the sampled tokens only
                }
                if args.generation_kwargs is not None:
                    generation_kwargs.update(args.generation_kwargs)
                sampling_params = SamplingParams(**generation_kwargs)

                if args.vllm_tensor_parallel_size > 1:
                    # Gather prompts from all ranks in the TP group and flatten.
                    # Each rank starts with its own prompts; after gathering, all ranks see the full group set.
                    orig_size = len(prompts)
                    gathered_prompts = [None for _ in range(args.vllm_tensor_parallel_size)]
                    torch.distributed.all_gather_object(gathered_prompts, prompts, group=self.tp_group)
                    all_prompts = [p for sublist in gathered_prompts for p in sublist]
                else:
                    all_prompts = prompts

                if args.vllm_enable_sleep_mode:
                    self.llm.wake_up(tags=["kv_cache"])

                with profiling_context(self.trainer, "vLLM.generate"):
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

                if args.vllm_tensor_parallel_size > 1:
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

                if args.vllm_enable_sleep_mode:
                    self.llm.sleep(level=2)

        return prompt_ids, completion_ids, logprobs, extra_fields
