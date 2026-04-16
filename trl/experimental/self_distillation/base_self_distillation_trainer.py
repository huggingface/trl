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

from __future__ import annotations

import copy
import inspect
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial
from typing import Any

import datasets
import torch
from accelerate.logging import get_logger
from accelerate.utils import is_peft_model
from datasets import Dataset, IterableDataset
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader, Sampler
from transformers import (
    AutoProcessor,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainerCallback,
)
from transformers.trainer_utils import seed_worker
from transformers.utils import is_datasets_available, is_peft_available

from ...data_utils import is_conversational
from ...models import prepare_deepspeed, prepare_fsdp, unwrap_model_for_generation
from ...trainer.base_trainer import _BaseTrainer
from ...trainer.utils import (
    RepeatSampler,
    create_model_from_path,
    disable_dropout_in_model,
    get_config_model_id,
    identity,
    pad,
    split_tensor_dict,
    use_adapter,
)
from ..utils import prepare_peft_model
from .loss_utils import (
    apply_importance_sampling_clipping,
    compute_full_logit_self_distillation_loss,
    compute_sampled_token_self_distillation_loss,
    compute_topk_self_distillation_loss,
    select_token_log_probs,
)
from .self_distillation_config import SelfDistillationConfig
from .teacher_sync import PEFTAdapterEMACallback, SyncTeacherModelCallback


if is_peft_available():
    from peft import PeftConfig


logger = get_logger(__name__)


@dataclass
class RolloutBatch:
    prompt_ids: torch.Tensor
    prompt_mask: torch.Tensor
    completion_ids: torch.Tensor
    completion_mask: torch.Tensor
    old_per_token_logps: torch.Tensor | None = None
    raw_completion_lengths: torch.Tensor | None = None

    def as_dict(self) -> dict[str, torch.Tensor | Any]:
        batch: dict[str, torch.Tensor | Any] = {
            "prompt_ids": self.prompt_ids,
            "prompt_mask": self.prompt_mask,
            "completion_ids": self.completion_ids,
            "completion_mask": self.completion_mask,
        }
        if self.old_per_token_logps is not None:
            batch["old_per_token_logps"] = self.old_per_token_logps
        if self.raw_completion_lengths is not None:
            batch["raw_completion_lengths"] = self.raw_completion_lengths
        return batch


TrainingBatch = dict[str, torch.Tensor | Any]


@dataclass
class DistillationLogits:
    """Aligned logits and masks used to compute a self-distillation objective."""

    completion_ids: torch.Tensor
    completion_mask: torch.Tensor
    response_mask: torch.Tensor
    student_logits: torch.Tensor
    teacher_logits: torch.Tensor


class BaseSelfDistillationTrainer(_BaseTrainer, ABC):
    """Base that centralizes shared self-distillation trainer lifecycle."""

    config_cls = SelfDistillationConfig
    _tag_names = ["trl", "self-distillation"]
    _name = "Self-Distillation"

    def __init__(
        self,
        model: str | PreTrainedModel | nn.Module,
        args: SelfDistillationConfig | None = None,
        train_dataset: Dataset | IterableDataset | None = None,
        eval_dataset: Dataset | IterableDataset | dict[str, Dataset | IterableDataset] | None = None,
        processing_class: PreTrainedTokenizerBase | ProcessorMixin | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None] = (None, None),
        peft_config: PeftConfig | None = None,
    ):
        if train_dataset is None:
            raise ValueError("`train_dataset` is required")
        self.use_vllm = args.use_vllm

        if isinstance(model, str):
            model_init_kwargs = args.model_init_kwargs or {}
            if args.distributed_state.distributed_type in ["MULTI_GPU", "DEEPSPEED"]:
                model_init_kwargs["device_map"] = None
            model = create_model_from_path(model, **model_init_kwargs)
        elif args.model_init_kwargs is not None:
            logger.warning(
                "You passed `model_init_kwargs` to the self-distillation config, but `model` is already "
                "instantiated. The `model_init_kwargs` will be ignored."
            )

        self.model_kwarg_keys = (
            inspect.signature(model.forward).parameters.keys()
            if not hasattr(model, "get_base_model")
            else inspect.signature(model.get_base_model().forward).parameters.keys()
        )

        if is_peft_model(model) and peft_config is not None:
            raise ValueError(
                "You passed a `PeftModel` instance together with a `peft_config`. Pass either a base "
                "model with `peft_config`, or a pre-wrapped PEFT model."
            )
        if peft_config is not None or getattr(model, "peft_config", None) is not None:
            model = prepare_peft_model(model, peft_config, args)

        if processing_class is None:
            processing_class = AutoProcessor.from_pretrained(
                get_config_model_id(model.config), truncation_side="left", padding_side="left"
            )

        if isinstance(processing_class, ProcessorMixin):
            tokenizer = processing_class.tokenizer
        elif isinstance(processing_class, PreTrainedTokenizerBase):
            tokenizer = processing_class
        else:
            raise TypeError("The `processing_class` must be either a `PreTrainedTokenizerBase` or a `ProcessorMixin`")

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length
        self.num_generations = args.num_generations
        self.num_generations_eval = args.num_generations_eval or args.num_generations
        self.num_iterations = args.num_iterations
        self.shuffle_dataset = args.shuffle_dataset
        self.loss_type = args.loss_type
        self.mask_truncated_completions = args.mask_truncated_completions
        self.temperature = args.temperature
        self.chat_template_kwargs = args.chat_template_kwargs or {}
        self._step = 0
        self._last_loaded_step = 0
        self._buffered_inputs = None
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._diagnostic_counters = {
            "train": defaultdict(int),
            "eval": defaultdict(int),
        }

        self.generation_kwargs = {
            "max_new_tokens": self.max_completion_length,
            "do_sample": True,
            "pad_token_id": tokenizer.pad_token_id,
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
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

        super().__init__(
            model=model,
            args=args,
            data_collator=identity,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
            compute_loss_func="non-None value to disable scaling",
        )

        if self.use_vllm:
            from ...generation.vllm_generation import VLLMGeneration

            self.vllm_generation = VLLMGeneration(
                model=self.model,
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
            self._last_loaded_step = -1

        if args.disable_dropout:
            disable_dropout_in_model(self.model)

        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        self._setup_teacher_model()
        self.model_accepts_loss_kwargs = False

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = ["prompt", "privileged_context"]

    def _dispatch_self_distillation_callback(self, event_name: str, **payload) -> None:
        for callback in self.callback_handler.callbacks:
            callback_fn = getattr(callback, event_name, None)
            if callback_fn is not None:
                callback_fn(
                    args=self.args,
                    state=self.state,
                    control=self.control,
                    model=self.model,
                    processing_class=self.processing_class,
                    **payload,
                )

    def _setup_teacher_model(self) -> None:
        """Prepare teacher state according to the semantic teacher choice."""

        teacher_model_kind = self.args.teacher_model_kind

        if teacher_model_kind == "live":
            self.teacher_model = self.model
            return

        if teacher_model_kind == "base" and is_peft_model(self.model):
            self.teacher_model = self.model
            return

        if self._use_peft_ema_teacher_adapter():
            self.add_callback(
                PEFTAdapterEMACallback(
                    model=self.model,
                    teacher_adapter_name="teacher",
                    update_rate=self.args.teacher_update_rate,
                    sync_steps=self.args.teacher_sync_steps,
                    accelerator=self.accelerator,
                )
            )
            self.teacher_model = self.model
            return

        # create teacher model from student copy
        student_model = self.accelerator.unwrap_model(self.model)
        self.teacher_model = copy.deepcopy(student_model)
        self.teacher_model.requires_grad_(False)
        self.teacher_model.eval()
        if self.is_deepspeed_enabled:
            self.teacher_model = prepare_deepspeed(self.teacher_model, self.accelerator)
        elif self.is_fsdp_enabled:
            self.teacher_model = prepare_fsdp(self.teacher_model, self.accelerator)
        else:
            self.teacher_model = self.accelerator.prepare_model(self.teacher_model, evaluation_mode=True)

        if teacher_model_kind == "ema":
            self.add_callback(SyncTeacherModelCallback(teacher_model=self.teacher_model, accelerator=self.accelerator))

    def _use_peft_ema_teacher_adapter(self) -> bool:
        return self.args.teacher_model_kind == "ema" and self._is_pure_lora_training()

    def _is_pure_lora_training(self) -> bool:
        if not is_peft_model(self.model):
            return False

        model = self.accelerator.unwrap_model(self.model)
        adapter_name = getattr(model, "active_adapter", None) or "default"
        adapter_config = model.peft_config.get(adapter_name)
        peft_type = getattr(adapter_config, "peft_type", None)
        if peft_type is None or str(peft_type).split(".")[-1] != "LORA":
            return False

        for name, param in model.named_parameters():
            if param.requires_grad and "lora_" not in name:
                return False
        return True

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size * self.args.steps_per_generation,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }
        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = partial(
                seed_worker, num_workers=self.args.dataloader_num_workers, rank=self.args.process_index
            )
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor
        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def _get_train_sampler(self, dataset=None) -> Sampler:
        if dataset is None:
            dataset = self.train_dataset
        return RepeatSampler(
            data_source=dataset,
            mini_repeat_count=self.num_generations,
            batch_size=self.args.generation_batch_size // self.num_generations,
            repeat_count=self.num_iterations * self.args.steps_per_generation,
            shuffle=self.shuffle_dataset,
            seed=self.args.seed,
        )

    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        return RepeatSampler(
            data_source=eval_dataset,
            mini_repeat_count=self.num_generations_eval,
            seed=self.args.seed,
        )

    def training_step(self, model, inputs, num_items_in_batch):
        output = super().training_step(model, inputs, num_items_in_batch)
        self._step += 1
        return output

    def _prepare_inputs(self, generation_batch):
        mode = "train" if self.model.training else "eval"
        if mode == "train":
            generate_every = self.args.steps_per_generation * self.num_iterations
            if self._step % generate_every == 0 or self._buffered_inputs is None:
                buffered_batch = self._prepare_training_batch(generation_batch)
                self._buffered_inputs = split_tensor_dict(buffered_batch, self.args.steps_per_generation)
                self._dispatch_self_distillation_callback(
                    "on_generation_batch_built",
                    generate_every=generate_every,
                    steps_per_generation=self.args.steps_per_generation,
                )
            return self._buffered_inputs[self._step % self.args.steps_per_generation]
        return self._prepare_training_batch(generation_batch)

    def _prepare_training_batch(self, inputs: list[dict[str, Any]]) -> TrainingBatch:
        rollout_batch = self.sample_rollouts(inputs)

        batch = self.finalize_batch(inputs, rollout_batch)
        self._validate_training_batch(batch)

        self._dispatch_self_distillation_callback(
            "on_self_distillation_batch_prepared",
            old_per_token_logps=batch.get("old_per_token_logps"),
            prompt_ids=batch["prompt_ids"],
            completion_ids=batch["completion_ids"],
            teacher_input_ids=batch["teacher_input_ids"],
            teacher_attention_mask=batch["teacher_attention_mask"],
            self_distillation_mask=batch.get("self_distillation_mask"),
        )
        return batch

    def _tokenize_prompts(self, prompts: list[Any]) -> list[list[int]]:
        if is_conversational({"prompt": prompts[0]}):
            tokenized = self.processing_class.apply_chat_template(
                conversation=prompts,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                **self.chat_template_kwargs,
            )
            prompt_ids = tokenized["input_ids"]
        else:
            prompt_ids = self.processing_class(text=prompts)["input_ids"]
        if self.max_prompt_length is not None:
            prompt_ids = [ids[-self.max_prompt_length :] for ids in prompt_ids]
        return prompt_ids

    def sample_rollouts(self, inputs: list[dict[str, Any]]) -> RolloutBatch:
        prompts, _ = self._split_prompt_and_privileged_context(inputs)
        prompt_ids = self._tokenize_prompts(prompts)
        self._dispatch_self_distillation_callback(
            "on_generation_prompts_selected",
            generation_prompts=prompts,
            generation_prompt_text=None,
        )

        prompt_ids_list, completion_ids_list = self._generate(prompt_ids)
        device = self.accelerator.device
        prompt_ids = [torch.tensor(ids) for ids in prompt_ids_list]
        prompt_mask = [torch.ones_like(ids, dtype=torch.long) for ids in prompt_ids]
        prompt_ids = pad(prompt_ids, padding_value=self.pad_token_id, padding_side="left").to(device=device)
        prompt_mask = pad(prompt_mask, padding_value=0, padding_side="left").to(device=device)
        completion_ids = [torch.tensor(ids) for ids in completion_ids_list]
        completion_mask = [torch.ones_like(ids, dtype=torch.long) for ids in completion_ids]
        completion_ids = pad(completion_ids, padding_value=self.pad_token_id, padding_side="right").to(device=device)
        completion_mask = pad(completion_mask, padding_value=0, padding_side="right").to(device=device)
        if self.mask_truncated_completions:
            eos_and_pad = [self.eos_token_id, self.pad_token_id]
            is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids_list], device=device)
            completion_mask = completion_mask * (~is_truncated).unsqueeze(1).int()
        old_per_token_logps = self._compute_rollout_logps(
            prompt_ids=prompt_ids,
            prompt_mask=prompt_mask,
            completion_ids=completion_ids,
            completion_mask=completion_mask,
        )

        return RolloutBatch(
            prompt_ids=prompt_ids,
            prompt_mask=prompt_mask,
            completion_ids=completion_ids,
            completion_mask=completion_mask,
            old_per_token_logps=old_per_token_logps,
            raw_completion_lengths=torch.tensor(
                [len(ids) for ids in completion_ids_list], device=device, dtype=torch.long
            ),
        )

    def _split_prompt_and_privileged_context(self, inputs: list[dict[str, Any]]) -> tuple[list[Any], list[Any]]:
        prompts = [example["prompt"] for example in inputs]
        privileged_contexts = [example.get("privileged_context") for example in inputs]
        return prompts, privileged_contexts

    def _generate(self, prompt_ids: list[list[int]]) -> tuple[list[list[int]], list[list[int]]]:
        if self.use_vllm:
            return self._generate_vllm(prompt_ids)
        return self._generate_transformers(prompt_ids)

    def _generate_vllm(self, prompt_ids: list[list[int]]) -> tuple[list[list[int]], list[list[int]]]:
        if self.state.global_step != self._last_loaded_step:
            self.vllm_generation.sync_weights()
            self._last_loaded_step = self.state.global_step

        mode = "train" if self.model.training else "eval"
        num_generations = self.num_generations if mode == "train" else self.num_generations_eval
        prompt_ids_out, completion_ids_list, _, _ = self.vllm_generation.generate(
            prompts=prompt_ids,
            images=None,
            num_generations=num_generations,
        )
        return prompt_ids_out, completion_ids_list

    def _generate_transformers(self, prompt_ids: list[list[int]]) -> tuple[list[list[int]], list[list[int]]]:
        device = self.accelerator.device
        prompt_tensors = [torch.tensor(ids) for ids in prompt_ids]
        padded_ids = pad(prompt_tensors, padding_value=self.pad_token_id, padding_side="left")
        attention_mask = pad([torch.ones_like(t) for t in prompt_tensors], padding_value=0, padding_side="left")
        generate_inputs = {"input_ids": padded_ids, "attention_mask": attention_mask}
        generate_inputs = _BaseTrainer._prepare_inputs(self, generate_inputs)

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
        is_eos = completion_ids == self.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        seq_idx = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (seq_idx <= eos_idx.unsqueeze(1)).int()
        completion_ids_list = [
            c[m].tolist() for c, m in zip(completion_ids.cpu(), completion_mask.bool().cpu(), strict=True)
        ]
        return prompt_ids, completion_ids_list

    def _compute_rollout_logps(
        self,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        completion_ids: torch.Tensor,
        completion_mask: torch.Tensor,
    ) -> torch.Tensor | None:
        generate_every = self.args.steps_per_generation * self.num_iterations
        old_per_token_logps = None

        if self.args.gradient_accumulation_steps % generate_every != 0:
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
            attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
            logits_to_keep = completion_ids.size(1)
            with torch.no_grad():
                logits = self._forward_logits(
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                )
                old_per_token_logps = select_token_log_probs(logits, completion_ids)

        return old_per_token_logps

    def _allow_topk_without_full_logit_distillation(self) -> bool:
        return True

    def _compute_self_distillation_loss(
        self,
        model,
        inputs: TrainingBatch,
        distillation_logits: DistillationLogits,
    ) -> torch.Tensor:
        if distillation_logits.response_mask.sum() == 0:
            mode = "train" if model.training else "eval"
            self._log_self_distillation_metric(mode, 0.0)
            return torch.tensor(0.0, device=distillation_logits.completion_ids.device, requires_grad=True)

        use_topk_distillation = self.args.distillation_topk is not None and (
            self.args.full_logit_distillation or self._allow_topk_without_full_logit_distillation()
        )
        if use_topk_distillation:
            per_token_loss = compute_topk_self_distillation_loss(
                distillation_logits.student_logits,
                distillation_logits.teacher_logits,
                distillation_topk=self.args.distillation_topk,
                distillation_alpha=self.args.distillation_alpha,
                distillation_add_tail=self.args.distillation_add_tail,
            )
        elif self.args.full_logit_distillation:
            per_token_loss = compute_full_logit_self_distillation_loss(
                distillation_logits.student_logits,
                distillation_logits.teacher_logits,
                distillation_alpha=self.args.distillation_alpha,
            )
        else:
            per_token_loss = compute_sampled_token_self_distillation_loss(
                distillation_logits.student_logits,
                distillation_logits.teacher_logits,
                distillation_logits.completion_ids,
                distillation_alpha=self.args.distillation_alpha,
            )

        old_per_token_logps = inputs.get("old_per_token_logps")
        if self.args.distillation_is_clip is not None and old_per_token_logps is not None:
            student_per_token_logps = select_token_log_probs(
                distillation_logits.student_logits,
                distillation_logits.completion_ids,
            )
            per_token_loss = apply_importance_sampling_clipping(
                per_token_loss,
                student_per_token_logps,
                old_per_token_logps,
                self.args.distillation_is_clip,
            )

        loss = self._aggregate_self_distillation_loss(per_token_loss, distillation_logits.response_mask)

        mode = "train" if model.training else "eval"
        mean_distill_loss = (
            per_token_loss * distillation_logits.response_mask
        ).sum() / distillation_logits.response_mask.sum().clamp(min=1.0)
        self._log_self_distillation_metric(
            mode,
            self.accelerator.gather(mean_distill_loss).mean().item(),
        )
        return loss

    def _compute_teacher_student_logits(
        self,
        model,
        teacher_model,
        inputs: TrainingBatch,
    ) -> DistillationLogits:
        prompt_ids = inputs["prompt_ids"]
        prompt_mask = inputs["prompt_mask"]
        completion_ids = inputs["completion_ids"]
        completion_mask = inputs["completion_mask"]
        logits_to_keep = completion_ids.size(1)

        response_mask = self._build_self_distillation_response_mask(
            completion_mask,
            inputs.get("self_distillation_mask"),
        )
        student_logits = self._compute_student_distillation_logits(
            model=model,
            prompt_ids=prompt_ids,
            prompt_mask=prompt_mask,
            completion_ids=completion_ids,
            completion_mask=completion_mask,
            logits_to_keep=logits_to_keep,
        )

        teacher_logits = self._compute_teacher_distillation_logits(
            teacher_model=teacher_model,
            teacher_input_ids=inputs["teacher_input_ids"],
            teacher_attention_mask=inputs["teacher_attention_mask"],
            logits_to_keep=logits_to_keep,
        )

        return DistillationLogits(
            completion_ids=completion_ids,
            completion_mask=completion_mask,
            response_mask=response_mask,
            student_logits=student_logits,
            teacher_logits=teacher_logits,
        )

    @staticmethod
    def _build_self_distillation_response_mask(
        completion_mask: torch.Tensor,
        self_distillation_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if self_distillation_mask is None:
            return completion_mask
        return completion_mask * self_distillation_mask.unsqueeze(1)

    def _compute_student_distillation_logits(
        self,
        model,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        completion_ids: torch.Tensor,
        completion_mask: torch.Tensor,
        logits_to_keep: int,
    ) -> torch.Tensor:
        student_input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        student_attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        return self._forward_logits(
            model=model,
            input_ids=student_input_ids,
            attention_mask=student_attention_mask,
            logits_to_keep=logits_to_keep,
        )

    def _compute_teacher_distillation_logits(
        self,
        teacher_model,
        teacher_input_ids: torch.Tensor,
        teacher_attention_mask: torch.Tensor,
        logits_to_keep: int,
    ) -> torch.Tensor:
        with torch.no_grad(), self._get_teacher_context_for_self_distillation():
            return self._forward_logits(
                model=teacher_model,
                input_ids=teacher_input_ids,
                attention_mask=teacher_attention_mask,
                logits_to_keep=logits_to_keep,
            )

    def _forward_logits(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        logits_to_keep: int,
    ) -> torch.Tensor:
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "use_cache": False,
        }
        if "logits_to_keep" in self.model_kwarg_keys:
            model_inputs["logits_to_keep"] = logits_to_keep + 1

        logits = model(**model_inputs).logits
        logits = logits[:, :-1, :]
        logits = logits[:, -logits_to_keep:, :]
        return logits / self.temperature

    def _validate_training_batch(self, batch: TrainingBatch) -> None:
        required_keys = {
            "prompt_ids",
            "prompt_mask",
            "completion_ids",
            "completion_mask",
            "teacher_input_ids",
            "teacher_attention_mask",
        }
        missing_keys = required_keys.difference(batch)
        if missing_keys:
            raise ValueError(f"`finalize_batch` must return all required batch keys. Missing: {sorted(missing_keys)}")

        batch_size = batch["prompt_ids"].size(0)
        if batch["prompt_mask"].size(0) != batch_size:
            raise ValueError("`prompt_mask` must have the same batch size as `prompt_ids`.")
        if batch["completion_ids"].size(0) != batch_size or batch["completion_mask"].size(0) != batch_size:
            raise ValueError("`completion_ids` and `completion_mask` must match the student batch size.")
        if batch["teacher_input_ids"].size(0) != batch_size or batch["teacher_attention_mask"].size(0) != batch_size:
            raise ValueError("`teacher_input_ids` and `teacher_attention_mask` must match the student batch size.")
        if batch["teacher_input_ids"].size(1) != batch["teacher_attention_mask"].size(1):
            raise ValueError("`teacher_input_ids` and `teacher_attention_mask` must have the same sequence length.")
        if "self_distillation_mask" in batch and batch["self_distillation_mask"] is not None:
            if batch["self_distillation_mask"].size(0) != batch_size:
                raise ValueError("`self_distillation_mask` must match the batch size when provided.")

    def finalize_batch(
        self,
        inputs: list[dict[str, Any]],
        rollout_batch: RolloutBatch,
    ) -> TrainingBatch:
        """Build the final training batch from a shared student rollout batch."""
        return rollout_batch.as_dict()

    def _get_teacher_context_for_self_distillation(self):
        teacher_model_kind = self.args.teacher_model_kind
        if not is_peft_model(self.model):
            return nullcontext()

        target_model = self.accelerator.unwrap_model(self.teacher_model)

        if teacher_model_kind == "base":
            return use_adapter(target_model, adapter_name=None)
        if teacher_model_kind == "ema" and self._use_peft_ema_teacher_adapter():
            return use_adapter(target_model, adapter_name="teacher")
        return nullcontext()

    def _log_self_distillation_metric(self, mode: str, value: float) -> None:
        metric_prefix = getattr(self, "_name", "self_distillation").lower().replace(" ", "_")
        self._metrics[mode]["self_distillation/distillation_loss"].append(value)
        self._metrics[mode][f"{metric_prefix}/distillation_loss"].append(value)

    def _aggregate_self_distillation_loss(
        self,
        per_token_loss: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> torch.Tensor:
        loss_type = self.loss_type
        if loss_type == "grpo":
            loss = (per_token_loss * response_mask).sum(-1) / response_mask.sum(-1).clamp(min=1.0)
            return loss.mean()
        if loss_type == "bnpo":
            return (per_token_loss * response_mask).sum() / response_mask.sum().clamp(min=1.0)
        if loss_type == "dr_grpo":
            return (per_token_loss * response_mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
        if loss_type in ["dapo", "luspo", "cispo", "sapo"]:
            return (per_token_loss * response_mask).sum() / response_mask.sum().clamp(min=1.0)
        raise ValueError(f"Unsupported loss_type for self-distillation: {loss_type}")

    @abstractmethod
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Subclasses own algorithm-specific loss composition on the final batch contract."""
