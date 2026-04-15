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
from dataclasses import dataclass, field
from functools import partial
from typing import Any

import datasets
import torch
from accelerate.logging import get_logger
from accelerate.utils import is_peft_model
from datasets import Dataset, IterableDataset
from torch import nn
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

from ...models import prepare_deepspeed, prepare_fsdp, unwrap_model_for_generation
from ...trainer.base_trainer import _BaseTrainer
from ...trainer.utils import (
    RepeatSampler,
    create_model_from_path,
    disable_dropout_in_model,
    entropy_from_logits,
    get_config_model_id,
    identity,
    pad,
    selective_log_softmax,
    split_tensor_dict,
    use_adapter,
)
from ..utils import prepare_peft_model
from .self_distillation_config import SelfDistillationConfig
from .self_distillation_mixin import SelfDistillationMixin
from .teacher_context import PromptTokenizer
from .teacher_sync import PEFTAdapterEMACallback, SyncTeacherModelCallback


if is_peft_available():
    from peft import PeftConfig
    from peft.peft_model import PeftModel


logger = get_logger(__name__)


@dataclass
class SelfDistillationRolloutBatch:
    """Common student rollout batch produced before algorithm-specific augmentation."""

    prompt_ids: torch.Tensor
    prompt_mask: torch.Tensor
    completion_ids: torch.Tensor
    completion_mask: torch.Tensor
    old_per_token_logps: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, torch.Tensor | Any]:
        output: dict[str, torch.Tensor | Any] = {
            "prompt_ids": self.prompt_ids,
            "prompt_mask": self.prompt_mask,
            "completion_ids": self.completion_ids,
            "completion_mask": self.completion_mask,
        }
        if self.old_per_token_logps is not None:
            output["old_per_token_logps"] = self.old_per_token_logps
        output.update(self.metadata)
        return output


@dataclass
class SelfDistillationBatch:
    """Final self-distillation batch contract consumed by `SelfDistillationMixin`."""

    prompt_ids: torch.Tensor
    prompt_mask: torch.Tensor
    completion_ids: torch.Tensor
    completion_mask: torch.Tensor
    teacher_input_ids: torch.Tensor
    teacher_attention_mask: torch.Tensor
    old_per_token_logps: torch.Tensor | None = None
    self_distillation_mask: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, torch.Tensor | Any]:
        output: dict[str, torch.Tensor | Any] = {
            "prompt_ids": self.prompt_ids,
            "prompt_mask": self.prompt_mask,
            "completion_ids": self.completion_ids,
            "completion_mask": self.completion_mask,
            "teacher_input_ids": self.teacher_input_ids,
            "teacher_attention_mask": self.teacher_attention_mask,
        }
        if self.old_per_token_logps is not None:
            output["old_per_token_logps"] = self.old_per_token_logps
        if self.self_distillation_mask is not None:
            output["self_distillation_mask"] = self.self_distillation_mask
        output.update(self.metadata)
        return output


class BaseSelfDistillationTrainer(SelfDistillationMixin, _BaseTrainer, ABC):
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

        if is_peft_available() and is_peft_model(model) and peft_config is not None:
            raise ValueError(
                "You passed a `PeftModel` instance together with a `peft_config`. Pass either a base "
                "model with `peft_config`, or a pre-wrapped PEFT model."
            )
        if peft_config is not None or (is_peft_available() and getattr(model, "peft_config", None) is not None):
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
        self.prompt_tokenizer = PromptTokenizer(self)

        generation_kwargs = {
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
            generation_kwargs.update(args.generation_kwargs)
        self.generation_config = GenerationConfig(**generation_kwargs, disable_compile=True)

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

    def _setup_teacher_model(self) -> None:
        """Prepare teacher state according to the shared teacher policy."""

        teacher_regularization = self.args.teacher_regularization
        peft_teacher_mode = self._resolve_peft_teacher_mode()
        self._validate_teacher_policy(teacher_regularization, peft_teacher_mode)

        if teacher_regularization == "none":
            return

        if is_peft_available() and is_peft_model(self.model) and peft_teacher_mode == "teacher_adapter":
            self.add_callback(
                PEFTAdapterEMACallback(
                    model=self.model,
                    teacher_adapter_name=self.args.teacher_adapter_name,
                    update_rate=self.args.teacher_update_rate,
                    sync_steps=self.args.teacher_sync_steps,
                    accelerator=self.accelerator,
                )
            )
            return

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

        self.add_callback(SyncTeacherModelCallback(teacher_model=self.teacher_model, accelerator=self.accelerator))

    def _resolve_peft_teacher_mode(self) -> str:
        peft_teacher_mode = self.args.peft_teacher_mode
        if not (is_peft_available() and is_peft_model(self.model)):
            if peft_teacher_mode in {"disable_adapter", "teacher_adapter"}:
                raise ValueError(f"PEFT teacher mode `{peft_teacher_mode}` requires a PEFT model.")
            return "inherit_adapter"

        if peft_teacher_mode == "auto":
            if self.args.teacher_regularization == "ema":
                return "teacher_adapter"
            return "disable_adapter"

        return peft_teacher_mode

    def _validate_teacher_policy(self, teacher_regularization: str, peft_teacher_mode: str) -> None:
        if teacher_regularization not in {"none", "ema"}:
            raise ValueError(f"Unsupported teacher regularization mode: {teacher_regularization}")
        if peft_teacher_mode not in {"inherit_adapter", "disable_adapter", "teacher_adapter"}:
            raise ValueError(f"Unsupported PEFT teacher mode: {peft_teacher_mode}")
        if peft_teacher_mode == "teacher_adapter" and teacher_regularization != "ema":
            raise ValueError("PEFT teacher mode `teacher_adapter` requires EMA teacher regularization.")

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
                buffered_batch = self._build_buffered_batch(generation_batch)
                self._buffered_inputs = split_tensor_dict(buffered_batch, self.args.steps_per_generation)
                self._dispatch_self_distillation_callback(
                    "on_generation_batch_built",
                    generate_every=generate_every,
                    steps_per_generation=self.args.steps_per_generation,
                )
            return self._buffered_inputs[self._step % self.args.steps_per_generation]
        return self._build_buffered_batch(generation_batch)

    def _build_buffered_batch(self, inputs: list[dict[str, Any]]) -> dict[str, torch.Tensor | Any]:
        return self.build_training_batch(inputs).to_dict()

    def build_training_batch(self, inputs: list[dict[str, Any]]) -> SelfDistillationBatch:
        rollout_batch = self.build_rollout_batch(inputs)

        batch = self.augment_training_batch(inputs, rollout_batch)
        self._validate_training_batch(batch)

        self._dispatch_self_distillation_callback(
            "on_self_distillation_batch_prepared",
            old_per_token_logps=batch.old_per_token_logps,
            prompt_ids=batch.prompt_ids,
            completion_ids=batch.completion_ids,
            teacher_input_ids=batch.teacher_input_ids,
            teacher_attention_mask=batch.teacher_attention_mask,
            self_distillation_mask=batch.self_distillation_mask,
        )
        return batch

    def build_rollout_batch(self, inputs: list[dict[str, Any]]) -> SelfDistillationRolloutBatch:
        prompts, _ = self._split_prompt_and_privileged_context(inputs)
        generation_prompts = prompts
        generation_prompt_text = self.prompt_tokenizer.apply_prompt_template(generation_prompts)
        self._dispatch_self_distillation_callback(
            "on_generation_prompts_selected",
            generation_prompts=generation_prompts,
            generation_prompt_text=generation_prompt_text,
        )

        prompt_ids_list, completion_ids_list = self._generate(generation_prompts)
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
        old_per_token_logps = self.compute_rollout_logps(
            prompt_ids=prompt_ids,
            prompt_mask=prompt_mask,
            completion_ids=completion_ids,
            completion_mask=completion_mask,
        )

        return SelfDistillationRolloutBatch(
            prompt_ids=prompt_ids,
            prompt_mask=prompt_mask,
            completion_ids=completion_ids,
            completion_mask=completion_mask,
            old_per_token_logps=old_per_token_logps,
            metadata={
                "raw_completion_lengths": torch.tensor(
                    [len(ids) for ids in completion_ids_list], device=device, dtype=torch.long
                )
            },
        )

    def _generate(self, prompts: list[Any]) -> tuple[list[list[int]], list[list[int]]]:
        if self.use_vllm:
            return self._generate_vllm(prompts)
        return self._generate_transformers(prompts)

    def _generate_vllm(self, prompts: list[Any]) -> tuple[list[list[int]], list[list[int]]]:
        if self.state.global_step != self._last_loaded_step:
            self.vllm_generation.sync_weights()
            self._last_loaded_step = self.state.global_step

        prompts_text = self.prompt_tokenizer.apply_prompt_template(prompts)
        tokenized = self.processing_class(
            text=prompts_text,
            return_tensors=None,
            padding=False,
            max_length=self.max_prompt_length,
            truncation=True,
            add_special_tokens=False,
        )
        prompt_ids = tokenized["input_ids"]
        mode = "train" if self.model.training else "eval"
        num_generations = self.num_generations if mode == "train" else self.num_generations_eval
        prompt_ids_out, completion_ids_list, _, _ = self.vllm_generation.generate(
            prompts=prompt_ids,
            images=None,
            num_generations=num_generations,
        )
        return prompt_ids_out, completion_ids_list

    def _generate_transformers(self, prompts: list[Any]) -> tuple[list[list[int]], list[list[int]]]:
        generate_inputs = self.processing_class(
            text=self.prompt_tokenizer.apply_prompt_template(prompts),
            return_tensors="pt",
            padding=True,
            padding_side="left",
            max_length=self.max_prompt_length,
            truncation=True,
            add_special_tokens=False,
        )
        generate_inputs = _BaseTrainer._prepare_inputs(self, generate_inputs)

        with (
            unwrap_model_for_generation(
                self.model_wrapped,
                self.accelerator,
                gather_deepspeed3_params=self.args.ds3_gather_for_generation,
            ) as unwrapped_model,
            torch.no_grad(),
        ):
            prompt_completion_ids = unwrapped_model.generate(
                **generate_inputs, generation_config=self.generation_config
            )

        prompt_ids = generate_inputs["input_ids"]
        prompt_mask = generate_inputs["attention_mask"]
        prompt_length = prompt_ids.size(1)
        completion_ids = prompt_completion_ids[:, prompt_length:]
        is_eos = completion_ids == self.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=completion_ids.device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        seq_idx = torch.arange(is_eos.size(1), device=completion_ids.device).expand(is_eos.size(0), -1)
        completion_mask = (seq_idx <= eos_idx.unsqueeze(1)).long()

        prompt_ids_list = [p[m].tolist() for p, m in zip(prompt_ids, prompt_mask.bool(), strict=False)]
        completion_ids_list = [c[m].tolist() for c, m in zip(completion_ids, completion_mask.bool(), strict=False)]
        return prompt_ids_list, completion_ids_list

    def compute_rollout_logps(
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
                old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    compute_entropy=False,
                )

        return old_per_token_logps

    def _get_per_token_logps_and_entropies(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        compute_entropy=False,
    ):
        model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask, "use_cache": False}
        if "logits_to_keep" in self.model_kwarg_keys:
            model_inputs["logits_to_keep"] = logits_to_keep + 1
        logits = model(**model_inputs).logits
        logits = logits[:, :-1, :]
        logits = logits[:, -logits_to_keep:, :]
        logits = logits / self.temperature
        completion_ids = input_ids[:, -logits_to_keep:]
        selected_logps = selective_log_softmax(logits, completion_ids)
        entropies = entropy_from_logits(logits) if compute_entropy else None
        return selected_logps, entropies

    def _validate_training_batch(self, batch: SelfDistillationBatch) -> None:
        batch_size = batch.prompt_ids.size(0)
        if batch.prompt_mask.size(0) != batch_size:
            raise ValueError("`prompt_mask` must have the same batch size as `prompt_ids`.")
        if batch.completion_ids.size(0) != batch_size or batch.completion_mask.size(0) != batch_size:
            raise ValueError("`completion_ids` and `completion_mask` must match the student batch size.")
        if batch.teacher_input_ids.size(0) != batch_size or batch.teacher_attention_mask.size(0) != batch_size:
            raise ValueError("`teacher_input_ids` and `teacher_attention_mask` must match the student batch size.")
        if batch.teacher_input_ids.size(1) != batch.teacher_attention_mask.size(1):
            raise ValueError("`teacher_input_ids` and `teacher_attention_mask` must have the same sequence length.")
        if batch.self_distillation_mask is not None and batch.self_distillation_mask.size(0) != batch_size:
            raise ValueError("`self_distillation_mask` must match the batch size when provided.")

    @abstractmethod
    def augment_training_batch(
        self,
        inputs: list[dict[str, Any]],
        rollout_batch: SelfDistillationRolloutBatch,
    ) -> SelfDistillationBatch:
        """Inject teacher-side inputs and algorithm-specific fields into a common student rollout batch."""

    def _get_teacher_context_for_self_distillation(self):
        peft_teacher_mode = self._resolve_peft_teacher_mode()
        if not (is_peft_available() and isinstance(self.model, PeftModel)) or peft_teacher_mode == "inherit_adapter":
            return nullcontext()

        target_model = self.teacher_model if self.teacher_model is not None else self.model
        target_model = self.accelerator.unwrap_model(target_model)

        if peft_teacher_mode == "disable_adapter":
            return use_adapter(target_model, adapter_name=None)
        if peft_teacher_mode == "teacher_adapter":
            teacher_adapter_name = self.args.teacher_adapter_name
            if teacher_adapter_name not in target_model.peft_config:
                raise RuntimeError(
                    f"Expected PEFT teacher adapter `{teacher_adapter_name}` to exist before teacher forward."
                )
            return use_adapter(target_model, adapter_name=teacher_adapter_name)

        raise ValueError(f"Unsupported PEFT teacher mode: {peft_teacher_mode}")

    @abstractmethod
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Subclasses own algorithm-specific loss composition on the final batch contract."""
