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
from dataclasses import dataclass
from typing import Any

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import GenerationConfig, PreTrainedTokenizerBase, ProcessorMixin

from ...models import unwrap_model_for_generation
from ...trainer.utils import pad


@dataclass
class GenerationOutput:
    """Token ids produced by a self-distillation generation backend."""

    prompt_ids: list[list[int]]
    completion_ids: list[list[int]]


class Generation:
    """Generate completion token ids from tokenized prompts for self-distillation trainers."""

    def __init__(
        self,
        *,
        model,
        model_wrapped,
        args,
        accelerator,
        processing_class: PreTrainedTokenizerBase | ProcessorMixin,
        tokenizer: PreTrainedTokenizerBase,
        is_fsdp_enabled: bool,
        max_completion_length: int,
        temperature: float,
    ):
        self.args = args
        self.accelerator = accelerator
        self.model_wrapped = model_wrapped
        self.processing_class = processing_class
        self._tokenizer = tokenizer
        self.is_fsdp_enabled = is_fsdp_enabled
        self.max_completion_length = max_completion_length
        self.temperature = temperature
        self.use_vllm = args.use_vllm

        self.generation_kwargs = {
            "max_new_tokens": self.max_completion_length,
            "do_sample": True,
            "pad_token_id": self._tokenizer.pad_token_id,
            "bos_token_id": self._tokenizer.bos_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "min_p": args.min_p,
            "repetition_penalty": args.repetition_penalty,
            "cache_implementation": args.cache_implementation,
        }
        if args.generation_kwargs is not None:
            self.generation_kwargs.update(args.generation_kwargs)
        self.generation_config = GenerationConfig(**self.generation_kwargs, disable_compile=True)

        if hasattr(model, "warnings_issued"):
            model.warnings_issued["estimate_tokens"] = True

        if self.use_vllm:
            from ...generation.vllm_generation import VLLMGeneration

            self.vllm_generation = VLLMGeneration(
                model=model,
                accelerator=self.accelerator,
                is_fsdp_enabled=self.is_fsdp_enabled,
                processing_class=self.processing_class,
                mode=args.vllm_mode,
                server_base_url=args.vllm_server_base_url,
                server_host=args.vllm_server_host,
                server_port=args.vllm_server_port,
                group_port=args.vllm_group_port,
                server_timeout=args.vllm_server_timeout,
                tensor_parallel_size=args.vllm_tensor_parallel_size,
                gpu_memory_utilization=args.vllm_gpu_memory_utilization,
                max_model_length=args.vllm_max_model_length,
                max_num_seqs=args.per_device_train_batch_size
                * args.vllm_tensor_parallel_size
                * args.steps_per_generation,
                enable_sleep_mode=args.vllm_enable_sleep_mode,
                model_impl=args.vllm_model_impl,
                repetition_penalty=args.repetition_penalty,
                temperature=self.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                min_p=args.min_p,
                max_completion_length=self.max_completion_length,
                logprobs=None,
                generation_kwargs=args.generation_kwargs,
            )

    def generate(
        self,
        prompt_ids: list[list[int]],
        *,
        num_generations: int,
    ) -> GenerationOutput:
        if self.use_vllm:
            return self._generate_vllm(prompt_ids, num_generations=num_generations)
        return self._generate_transformers(prompt_ids)

    def sync_weights(self) -> None:
        self.vllm_generation.sync_weights()

    def _generate_vllm(
        self,
        prompt_ids: list[list[int]],
        *,
        num_generations: int,
    ) -> GenerationOutput:
        prompt_ids_out, completion_ids, _, _ = self.vllm_generation.generate(
            prompts=prompt_ids,
            images=None,
            num_generations=num_generations,
        )
        return GenerationOutput(prompt_ids=prompt_ids_out, completion_ids=completion_ids)

    def _generate_transformers(self, prompt_ids: list[list[int]]) -> GenerationOutput:
        device = self.accelerator.device
        prompt_tensors = [torch.tensor(ids) for ids in prompt_ids]
        padded_ids = pad(prompt_tensors, padding_value=self._tokenizer.pad_token_id, padding_side="left").to(
            device=device
        )
        attention_mask = pad([torch.ones_like(t) for t in prompt_tensors], padding_value=0, padding_side="left").to(
            device=device
        )
        generate_inputs: dict[str, torch.Tensor | Any] = {"input_ids": padded_ids, "attention_mask": attention_mask}

        with (
            unwrap_model_for_generation(
                self.model_wrapped,
                self.accelerator,
                gather_deepspeed3_params=self.args.ds3_gather_for_generation,
                generation_kwargs=self.generation_kwargs,
            ) as unwrapped_model,
            torch.no_grad(),
            FSDP.summon_full_params(self.model_wrapped, recurse=False) if self.is_fsdp_enabled else nullcontext(),
        ):
            prompt_completion_ids = unwrapped_model.generate(
                **generate_inputs, generation_config=self.generation_config
            )

        prompt_length = generate_inputs["input_ids"].size(1)
        completion_ids = prompt_completion_ids[:, prompt_length:]
        is_eos = completion_ids == self._tokenizer.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        seq_idx = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (seq_idx <= eos_idx.unsqueeze(1)).int()
        completion_ids_list = [
            c[m].tolist() for c, m in zip(completion_ids.cpu(), completion_mask.bool().cpu(), strict=True)
        ]
        return GenerationOutput(prompt_ids=prompt_ids, completion_ids=completion_ids_list)
