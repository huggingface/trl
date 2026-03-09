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

from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Protocol

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import GenerationConfig

from ..data_utils import is_conversational
from ..extras.profiling import profiling_context
from ..models import unwrap_model_for_generation
from .vllm_generation import VLLMGeneration


@dataclass
class GenerationResult:
    prompt_ids: list[list[int]]
    completion_ids: list[list[int]]
    logprobs: list[list[float]] | None = None
    extra_fields: dict[str, Any] = field(default_factory=dict)


@dataclass
class RolloutCompletion:
    prompt_ids: list[int]
    completion_ids: list[int]
    logprobs: list[float]
    text: str


class GenerationBackend(Protocol):
    def generate(
        self,
        prompts: list,
        num_generations: int,
        processing_class: Any,
        generation_config: GenerationConfig | None = None,
    ) -> GenerationResult:
        ...

    def sync_weights(self) -> None:
        ...


class RolloutCompletionsBackend(Protocol):
    def generate_rollout_completions(
        self,
        prompts: list,
        processing_class: Any,
        generation_overrides: dict[str, Any] | None = None,
        as_chat: bool | None = None,
    ) -> list[RolloutCompletion]:
        ...


def _raise_rollout_capability_error(backend_name: str) -> None:
    raise RuntimeError(
        f"Backend '{backend_name}' does not support rollout completions. "
        "This capability is currently available only for vLLM backends."
    )


class VLLMBackendAdapter:
    def __init__(
        self,
        vllm_generation: VLLMGeneration,
        profiler_factory: Any | None = None,
        *,
        vllm_mode: str,
        processing_class: Any,
        temperature: float,
        top_k: int,
        min_p: float | None,
        max_completion_length: int,
        repetition_penalty: float | None,
        top_p: float | None,
        generation_kwargs: dict[str, Any] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
        tools: list[Any] | None = None,
        chat_template: str | None = None,
        vllm_tensor_parallel_size: int = 1,
        vllm_enable_sleep_mode: bool = False,
    ):
        self.vllm_generation = vllm_generation
        self.profiler_factory = profiler_factory or (lambda _label: nullcontext())
        self.vllm_mode = vllm_mode
        self.processing_class = processing_class
        self.temperature = temperature
        self.top_k = top_k
        self.min_p = min_p
        self.max_completion_length = max_completion_length
        self.repetition_penalty = repetition_penalty
        self.top_p = top_p
        self.generation_kwargs = generation_kwargs
        self.chat_template_kwargs = chat_template_kwargs or {}
        self.tools = tools
        self.chat_template = chat_template
        self.vllm_tensor_parallel_size = vllm_tensor_parallel_size
        self.vllm_enable_sleep_mode = vllm_enable_sleep_mode

    def generate(
        self,
        prompts: list,
        num_generations: int,
        processing_class: Any,
        generation_config: GenerationConfig | None = None,
    ) -> GenerationResult:
        del processing_class, generation_config
        prompt_ids, completion_ids, logprobs, _, extra_fields = self.vllm_generation.generate(
            prompts=prompts,
            num_generations=num_generations,
            profiler=self.profiler_factory("vLLM.generate"),
        )

        if logprobs is not None:
            logprobs = [[lp[0] for lp in seq] for seq in logprobs]

        return GenerationResult(
            prompt_ids=prompt_ids,
            completion_ids=completion_ids,
            logprobs=logprobs,
            extra_fields=extra_fields,
        )

    def sync_weights(self) -> None:
        self.vllm_generation.sync_weights()

    def _build_base_generation_kwargs(self, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        generation_kwargs: dict[str, Any] = {
            "n": 1,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "min_p": 0.0 if self.min_p is None else self.min_p,
            "max_tokens": self.max_completion_length,
        }
        if self.repetition_penalty is not None:
            generation_kwargs["repetition_penalty"] = self.repetition_penalty
        if self.top_p is not None:
            generation_kwargs["top_p"] = self.top_p

        if self.generation_kwargs is not None:
            generation_kwargs.update(self.generation_kwargs)

        if overrides is not None:
            generation_kwargs.update(overrides)

        generation_kwargs = {key: value for key, value in generation_kwargs.items() if value is not None}

        if generation_kwargs.get("n", 1) != 1:
            raise ValueError("generate_rollout_completions expects n=1.")

        return generation_kwargs

    def generate_rollout_completions(
        self,
        prompts: list,
        processing_class: Any,
        generation_overrides: dict[str, Any] | None = None,
        as_chat: bool | None = None,
    ) -> list[RolloutCompletion]:
        del processing_class

        if not prompts:
            return []

        if self.vllm_mode == "server":
            generation_kwargs = self._build_base_generation_kwargs(generation_overrides)

            if as_chat is None:
                as_chat = prompts and is_conversational({"prompt": prompts[0]})

            with self.profiler_factory("vLLM.generate_rollout_server"):
                if as_chat:
                    output = self.vllm_generation.vllm_client.chat(
                        messages=prompts,
                        **generation_kwargs,
                        chat_template_kwargs=self.chat_template_kwargs,
                        tools=self.tools or None,
                        chat_template=self.chat_template,
                    )
                else:
                    output = self.vllm_generation.vllm_client.generate(prompts=prompts, **generation_kwargs)

            return [
                RolloutCompletion(
                    prompt_ids=output["prompt_ids"][i],
                    completion_ids=list(output["completion_ids"][i]),
                    logprobs=list(output["logprobs"][i]),
                    text=self.processing_class.decode(output["completion_ids"][i], skip_special_tokens=True),
                )
                for i in range(len(prompts))
            ]

        if self.vllm_mode == "colocate":
            from vllm import SamplingParams
            from vllm.sampling_params import StructuredOutputsParams

            generation_kwargs = self._build_base_generation_kwargs(generation_overrides)
            if self.vllm_generation.structured_outputs_regex:
                generation_kwargs["structured_outputs"] = StructuredOutputsParams(
                    regex=self.vllm_generation.structured_outputs_regex
                )
            generation_kwargs["logprobs"] = 0
            sampling_params = SamplingParams(**generation_kwargs)

            prompts_for_generation = prompts
            original_size = len(prompts)

            if self.vllm_tensor_parallel_size > 1:
                gathered_prompts = [None for _ in range(self.vllm_tensor_parallel_size)]
                torch.distributed.all_gather_object(gathered_prompts, prompts, group=self.vllm_generation.tp_group)
                prompts_for_generation = [prompt for group_prompts in gathered_prompts for prompt in group_prompts]

            if as_chat is None:
                as_chat = prompts_for_generation and is_conversational({"prompt": prompts_for_generation[0]})

            if self.vllm_enable_sleep_mode:
                self.vllm_generation.llm.wake_up(tags=["kv_cache"])
                self.vllm_generation.llm.collective_rpc("reload_weights")

            with self.profiler_factory("vLLM.generate_rollout"):
                if as_chat:
                    vllm_outputs = self.vllm_generation.llm.chat(
                        prompts_for_generation,
                        sampling_params=sampling_params,
                        use_tqdm=False,
                    )
                else:
                    vllm_outputs = self.vllm_generation.llm.generate(
                        prompts_for_generation,
                        sampling_params=sampling_params,
                        use_tqdm=False,
                    )

            results: list[RolloutCompletion] = []
            for request in vllm_outputs:
                if not request.outputs:
                    results.append(
                        RolloutCompletion(
                            prompt_ids=request.prompt_token_ids,
                            completion_ids=[],
                            logprobs=[],
                            text="",
                        )
                    )
                    continue
                sequence = request.outputs[0]
                logprobs = [next(iter(token_logprob.values())).logprob for token_logprob in sequence.logprobs]
                results.append(
                    RolloutCompletion(
                        prompt_ids=request.prompt_token_ids,
                        completion_ids=sequence.token_ids,
                        logprobs=logprobs,
                        text=sequence.text,
                    )
                )

            if self.vllm_tensor_parallel_size > 1:
                local_rank_in_group = torch.distributed.get_rank(group=self.vllm_generation.tp_group)
                tp_slice = slice(local_rank_in_group * original_size, (local_rank_in_group + 1) * original_size)
                results = results[tp_slice]

            if self.vllm_enable_sleep_mode:
                self.vllm_generation.llm.sleep(level=2)

            return results

        raise ValueError(f"vllm_mode must be 'server' or 'colocate', got '{self.vllm_mode}'")


class TransformersPagedBackendAdapter:
    def __init__(
        self,
        model_wrapped: Any,
        accelerator: Any,
        is_fsdp_enabled: bool,
        ds3_gather_for_generation: bool,
        chat_template_kwargs: dict[str, Any] | None = None,
        bf16: bool = False,
        fp16: bool = False,
        cast_lm_head_to_fp32: bool = False,
        tools: list[Any] | None = None,
        chat_template: str | None = None,
        include_tools_in_chat_template: bool = False,
        profiler_factory: Any | None = None,
    ):
        self.model_wrapped = model_wrapped
        self.accelerator = accelerator
        self.is_fsdp_enabled = is_fsdp_enabled
        self.ds3_gather_for_generation = ds3_gather_for_generation
        self.chat_template_kwargs = chat_template_kwargs or {}
        self.bf16 = bf16
        self.fp16 = fp16
        self.cast_lm_head_to_fp32 = cast_lm_head_to_fp32
        self.tools = tools
        self.chat_template = chat_template
        self.include_tools_in_chat_template = include_tools_in_chat_template
        self.profiler_factory = profiler_factory or (lambda _label: nullcontext())

    def generate(
        self,
        prompts: list,
        num_generations: int,
        processing_class: Any,
        generation_config: GenerationConfig | None = None,
    ) -> GenerationResult:
        del num_generations

        if is_conversational({"prompt": prompts[0]}):
            chat_template_kwargs = {
                "conversation": prompts,
                "add_generation_prompt": True,
                "tokenize": True,
                "return_dict": True,
                **self.chat_template_kwargs,
            }
            if self.include_tools_in_chat_template:
                chat_template_kwargs["tools"] = self.tools
                chat_template_kwargs["chat_template"] = self.chat_template
            processor_outputs = processing_class.apply_chat_template(**chat_template_kwargs)
        else:
            processor_outputs = processing_class(text=prompts)

        with (
            self.profiler_factory("transformers.generate_batch"),
            unwrap_model_for_generation(
                self.model_wrapped,
                self.accelerator,
                gather_deepspeed3_params=self.ds3_gather_for_generation,
            ) as unwrapped_model,
            torch.no_grad(),
            FSDP.summon_full_params(self.model_wrapped, recurse=False) if self.is_fsdp_enabled else nullcontext(),
        ):
            if self.bf16:
                unwrapped_model.to(torch.bfloat16)
            elif self.fp16:
                unwrapped_model.to(torch.float16)
            if self.cast_lm_head_to_fp32:
                unwrapped_model.lm_head.to(torch.float32)
            with torch.inference_mode():
                all_outputs = unwrapped_model.generate_batch(
                    processor_outputs["input_ids"], generation_config=generation_config, progress_bar=False
                )
                unwrapped_model.train()

        completion_ids = [output.generated_tokens for output in all_outputs.values()]
        prompt_ids = processor_outputs["input_ids"]
        return GenerationResult(prompt_ids=prompt_ids, completion_ids=completion_ids, logprobs=None, extra_fields={})

    def sync_weights(self) -> None:
        pass

    def generate_rollout_completions(
        self,
        prompts: list,
        processing_class: Any,
        generation_overrides: dict[str, Any] | None = None,
        as_chat: bool | None = None,
    ) -> list[RolloutCompletion]:
        del prompts, processing_class, generation_overrides, as_chat
        _raise_rollout_capability_error("transformers_paged")


class TransformersBackendAdapter:
    def __init__(
        self,
        model_wrapped: Any,
        accelerator: Any,
        is_fsdp_enabled: bool,
        ds3_gather_for_generation: bool,
        generation_kwargs: dict[str, Any] | None,
        eos_token_id: int,
        chat_template_kwargs: dict[str, Any] | None = None,
        tools: list[Any] | None = None,
        chat_template: str | None = None,
        include_tools_in_chat_template: bool = False,
        prepare_inputs: Any | None = None,
        profiler_factory: Any | None = None,
    ):
        self.model_wrapped = model_wrapped
        self.accelerator = accelerator
        self.is_fsdp_enabled = is_fsdp_enabled
        self.ds3_gather_for_generation = ds3_gather_for_generation
        self.generation_kwargs = generation_kwargs
        self.eos_token_id = eos_token_id
        self.chat_template_kwargs = chat_template_kwargs or {}
        self.tools = tools
        self.chat_template = chat_template
        self.include_tools_in_chat_template = include_tools_in_chat_template
        self.prepare_inputs = prepare_inputs
        self.profiler_factory = profiler_factory or (lambda _label: nullcontext())

    def generate(
        self,
        prompts: list,
        num_generations: int,
        processing_class: Any,
        generation_config: GenerationConfig | None = None,
    ) -> GenerationResult:
        del num_generations

        if is_conversational({"prompt": prompts[0]}):
            chat_template_kwargs = {
                "conversation": prompts,
                "add_generation_prompt": True,
                "tokenize": True,
                "padding": True,
                "padding_side": "left",
                "return_tensors": "pt",
                "return_dict": True,
                **self.chat_template_kwargs,
            }
            if self.include_tools_in_chat_template:
                chat_template_kwargs["tools"] = self.tools
                chat_template_kwargs["chat_template"] = self.chat_template
            generate_inputs = processing_class.apply_chat_template(**chat_template_kwargs)
        else:
            generate_inputs = processing_class(text=prompts, padding=True, padding_side="left", return_tensors="pt")

        if self.prepare_inputs is not None:
            generate_inputs = self.prepare_inputs(generate_inputs)

        with (
            self.profiler_factory("transformers.generate"),
            unwrap_model_for_generation(
                self.model_wrapped,
                self.accelerator,
                gather_deepspeed3_params=self.ds3_gather_for_generation,
                generation_kwargs=self.generation_kwargs,
            ) as unwrapped_model,
            torch.no_grad(),
            FSDP.summon_full_params(self.model_wrapped, recurse=False) if self.is_fsdp_enabled else nullcontext(),
        ):
            prompt_completion_ids = unwrapped_model.generate(
                **generate_inputs,
                generation_config=generation_config,
                disable_compile=True,
            )

        device = self.accelerator.device
        prompt_ids, prompt_mask = generate_inputs["input_ids"], generate_inputs["attention_mask"]
        prompt_length = prompt_ids.size(1)
        completion_ids = prompt_completion_ids[:, prompt_length:]

        is_eos = completion_ids == self.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        prompt_ids = [p[m].tolist() for p, m in zip(prompt_ids, prompt_mask.bool(), strict=True)]
        completion_ids = [c[m].tolist() for c, m in zip(completion_ids, completion_mask.bool(), strict=True)]

        return GenerationResult(prompt_ids=prompt_ids, completion_ids=completion_ids, logprobs=None, extra_fields={})

    def sync_weights(self) -> None:
        pass

    def generate_rollout_completions(
        self,
        prompts: list,
        processing_class: Any,
        generation_overrides: dict[str, Any] | None = None,
        as_chat: bool | None = None,
    ) -> list[RolloutCompletion]:
        del prompts, processing_class, generation_overrides, as_chat
        _raise_rollout_capability_error("transformers")


def create_generation_backend(trainer: Any) -> GenerationBackend:
    def profiler_factory(label: str):
        return profiling_context(trainer, label)

    if trainer.use_vllm:
        return VLLMBackendAdapter(
            vllm_generation=trainer.vllm_generation,
            profiler_factory=profiler_factory,
            vllm_mode=trainer.vllm_mode,
            processing_class=trainer.processing_class,
            temperature=trainer.temperature,
            top_k=trainer.top_k,
            min_p=trainer.min_p,
            max_completion_length=trainer.max_completion_length,
            repetition_penalty=trainer.repetition_penalty,
            top_p=trainer.top_p,
            generation_kwargs=trainer.args.generation_kwargs,
            chat_template_kwargs=trainer.chat_template_kwargs,
            tools=getattr(trainer, "tools", None),
            chat_template=getattr(trainer, "chat_template", None),
            vllm_tensor_parallel_size=trainer.vllm_tensor_parallel_size,
            vllm_enable_sleep_mode=trainer.args.vllm_enable_sleep_mode,
        )

    include_tools_in_chat_template = hasattr(trainer, "tools") and hasattr(trainer, "chat_template")

    if trainer.use_transformers_paged:
        return TransformersPagedBackendAdapter(
            model_wrapped=trainer.model_wrapped,
            accelerator=trainer.accelerator,
            is_fsdp_enabled=trainer.is_fsdp_enabled,
            ds3_gather_for_generation=trainer.args.ds3_gather_for_generation,
            chat_template_kwargs=trainer.chat_template_kwargs,
            bf16=trainer.args.bf16,
            fp16=trainer.args.fp16,
            cast_lm_head_to_fp32=getattr(trainer.args, "cast_lm_head_to_fp32", False),
            tools=getattr(trainer, "tools", None),
            chat_template=getattr(trainer, "chat_template", None),
            include_tools_in_chat_template=include_tools_in_chat_template,
            profiler_factory=profiler_factory,
        )

    # Use the base Trainer input preparation path, not trainer-specific overrides
    # like GRPO/RLOO _prepare_inputs, to avoid recursive generation.
    base_prepare_inputs = super(type(trainer), trainer)._prepare_inputs

    return TransformersBackendAdapter(
        model_wrapped=trainer.model_wrapped,
        accelerator=trainer.accelerator,
        is_fsdp_enabled=trainer.is_fsdp_enabled,
        ds3_gather_for_generation=trainer.args.ds3_gather_for_generation,
        generation_kwargs=trainer.generation_kwargs,
        eos_token_id=trainer.eos_token_id,
        chat_template_kwargs=trainer.chat_template_kwargs,
        tools=getattr(trainer, "tools", None),
        chat_template=getattr(trainer, "chat_template", None),
        include_tools_in_chat_template=include_tools_in_chat_template,
        prepare_inputs=base_prepare_inputs,
        profiler_factory=profiler_factory,
    )
