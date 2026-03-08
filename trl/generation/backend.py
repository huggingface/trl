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


class VLLMBackendAdapter:
    def __init__(self, vllm_generation: VLLMGeneration, profiler_factory: Any | None = None):
        self.vllm_generation = vllm_generation
        self.profiler_factory = profiler_factory or (lambda _label: nullcontext())

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


def create_generation_backend(trainer: Any) -> GenerationBackend:
    def profiler_factory(label: str):
        return profiling_context(trainer, label)

    if trainer.use_vllm:
        return VLLMBackendAdapter(vllm_generation=trainer.vllm_generation, profiler_factory=profiler_factory)

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
