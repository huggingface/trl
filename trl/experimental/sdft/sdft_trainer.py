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

import inspect
from collections import defaultdict
from collections.abc import Callable
from functools import partial
from typing import Any

import datasets
import torch
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
from ...trainer.callbacks import SyncRefModelCallback
from ...trainer.utils import (
    RepeatSampler,
    create_model_from_path,
    disable_dropout_in_model,
    get_config_model_id,
    identity,
    pad,
    split_tensor_dict,
)
from ..utils import prepare_peft_model
from ..self_distillation.self_distillation_mixin import SelfDistillationMixin
from ..self_distillation.teacher_context import DemonstrationTeacherContextBuilder, PromptTokenizer
from .sdft_config import SDFTConfig


if is_peft_available():
    from peft import PeftConfig


class SDFTTrainer(SelfDistillationMixin, _BaseTrainer):
    """Trainer for SDFT-style on-policy self-distillation with explicit teacher prompts."""

    _tag_names = ["trl", "sdft"]
    _name = "SDFT"
    config_cls = SDFTConfig

    def __init__(
        self,
        model: str | PreTrainedModel | nn.Module,
        ref_model: str | PreTrainedModel | nn.Module,
        args: SDFTConfig | None = None,
        train_dataset: Dataset | IterableDataset | None = None,
        eval_dataset: Dataset | IterableDataset | dict[str, Dataset | IterableDataset] | None = None,
        processing_class: PreTrainedTokenizerBase | ProcessorMixin | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None] = (None, None),
        peft_config: "PeftConfig | None" = None,
    ):
        args = self._coerce_sdft_args(args)

        if train_dataset is None:
            raise ValueError("`train_dataset` is required")
        if isinstance(train_dataset, IterableDataset):
            raise NotImplementedError("Iterable datasets are not yet supported in SDFTTrainer.")
        if isinstance(eval_dataset, IterableDataset) or (
            isinstance(eval_dataset, dict) and any(isinstance(ds, IterableDataset) for ds in eval_dataset.values())
        ):
            raise NotImplementedError("Iterable eval datasets are not yet supported in SDFTTrainer.")
        if args.use_vllm:
            raise NotImplementedError("SDFTTrainer does not support `use_vllm=True` yet.")
        if ref_model is None:
            raise ValueError("`ref_model` is required for SDFTTrainer.")

        if isinstance(model, str):
            model_init_kwargs = args.model_init_kwargs or {}
            if args.distributed_state.distributed_type in ["MULTI_GPU", "DEEPSPEED"]:
                model_init_kwargs["device_map"] = None
            model = create_model_from_path(model, **model_init_kwargs)
        elif args.model_init_kwargs is not None:
            pass

        self.model_kwarg_keys = (
            inspect.signature(model.forward).parameters.keys()
            if not hasattr(model, "get_base_model")
            else inspect.signature(model.get_base_model().forward).parameters.keys()
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

        self.pad_token = tokenizer.pad_token
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length
        self.num_generations = args.num_generations
        self.num_iterations = args.num_iterations
        self.temperature = args.temperature
        self.loss_type = args.loss_type
        self.shuffle_dataset = args.shuffle_dataset
        self.generate_from_teacher = args.generate_from_teacher
        self.num_loss_tokens_to_skip = args.num_loss_tokens_to_skip
        self._step = 0
        self._buffered_inputs = None
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self.prompt_tokenizer = PromptTokenizer(self)
        self.teacher_context_builder = DemonstrationTeacherContextBuilder(self)

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
        self.generation_config = GenerationConfig(**generation_kwargs)

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

        if isinstance(ref_model, str):
            ref_model_init_kwargs = args.model_init_kwargs or {}
            if args.distributed_state.distributed_type in ["MULTI_GPU", "DEEPSPEED"]:
                ref_model_init_kwargs["device_map"] = None
            ref_model = create_model_from_path(ref_model, **ref_model_init_kwargs)

        self.ref_model = ref_model

        if args.disable_dropout:
            disable_dropout_in_model(self.model)
            disable_dropout_in_model(self.ref_model)

        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            elif self.is_fsdp_enabled:
                self.ref_model = prepare_fsdp(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
            self.teacher_model = self.ref_model

        if args.sync_ref_model:
            self.add_callback(SyncRefModelCallback(ref_model=self.ref_model, accelerator=self.accelerator))

        self.model_accepts_loss_kwargs = False

    @classmethod
    def _coerce_sdft_args(cls, args: Any | None):
        if isinstance(args, cls.config_cls):
            return args
        if args is None:
            return cls.config_cls(output_dir="sdft-output")
        if hasattr(args, "to_dict"):
            dict_args = args.to_dict()
            if hasattr(args, "hub_token"):
                dict_args["hub_token"] = args.hub_token
        else:
            dict_args = args.__dict__.copy()
        return cls.config_cls(**dict_args)

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

    def _get_train_sampler(self, dataset: Dataset | None = None) -> Sampler:
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
            mini_repeat_count=self.num_generations,
            seed=self.args.seed,
        )

    def _prepare_inputs(self, generation_batch: dict[str, torch.Tensor | Any]) -> dict[str, torch.Tensor | Any]:
        mode = "train" if self.model.training else "eval"
        if mode == "train":
            generate_every = self.args.steps_per_generation * self.num_iterations
            if self._step % generate_every == 0 or self._buffered_inputs is None:
                generation_batch = self._generate_and_prepare_batch(generation_batch)
                self._buffered_inputs = split_tensor_dict(generation_batch, self.args.steps_per_generation)
            inputs = self._buffered_inputs[self._step % self.args.steps_per_generation]
        else:
            inputs = self._generate_and_prepare_batch(generation_batch)
        return inputs

    def _generate_completion_ids(self, prompts: list[Any]) -> tuple[torch.Tensor, torch.Tensor]:
        generate_inputs = self.processing_class(
            text=self.prompt_tokenizer.apply_prompt_template(prompts),
            return_tensors="pt",
            padding=True,
            padding_side="left",
            max_length=self.max_prompt_length,
            truncation=True,
            add_special_tokens=False,
        )
        generate_inputs = super()._prepare_inputs(generate_inputs)

        with unwrap_model_for_generation(
            self.model_wrapped,
            self.accelerator,
            gather_deepspeed3_params=self.args.ds3_gather_for_generation,
        ) as unwrapped_model, torch.no_grad():
            prompt_completion_ids = unwrapped_model.generate(
                **generate_inputs,
                generation_config=self.generation_config,
                disable_compile=True,
            )

        prompt_length = generate_inputs["input_ids"].size(1)
        completion_ids = prompt_completion_ids[:, prompt_length:]
        is_eos = completion_ids == self.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=completion_ids.device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        seq_idx = torch.arange(is_eos.size(1), device=completion_ids.device).expand(is_eos.size(0), -1)
        completion_mask = (seq_idx <= eos_idx.unsqueeze(1)).long()

        completion_ids_list = [c[m].tolist() for c, m in zip(completion_ids, completion_mask.bool())]
        completion_ids = [torch.tensor(ids, device=self.accelerator.device) for ids in completion_ids_list]
        completion_mask = [torch.ones_like(ids, dtype=torch.long) for ids in completion_ids]
        return (
            pad(completion_ids, padding_value=self.pad_token_id, padding_side="right"),
            pad(completion_mask, padding_value=0, padding_side="right"),
        )

    def _generate_and_prepare_batch(self, inputs: list[dict[str, Any]]) -> dict[str, torch.Tensor | Any]:
        prompts, privileged_contexts = self._split_prompt_and_privileged_context(inputs)
        generation_prompts = self.teacher_context_builder.select_generation_prompts(prompts, privileged_contexts)
        generation_prompt_text = self.prompt_tokenizer.apply_prompt_template(generation_prompts)
        self._dispatch_self_distillation_callback(
            "on_generation_prompts_selected",
            generation_prompts=generation_prompts,
            generation_prompt_text=generation_prompt_text,
        )
        completion_ids, completion_mask = self._generate_completion_ids(generation_prompts)

        teacher_batch = self.teacher_context_builder.build(prompts, privileged_contexts, completion_ids, completion_mask)

        return {
            "prompt_ids": teacher_batch["prompt_ids"],
            "prompt_mask": teacher_batch["prompt_mask"],
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "teacher_input_ids": teacher_batch["teacher_input_ids"],
            "teacher_attention_mask": teacher_batch["teacher_attention_mask"],
        }

    def _log_self_distillation_metric(self, mode: str, metric_name: str, value: float) -> None:
        self._metrics[mode][f"self_distillation/{metric_name}"].append(value)
        self._metrics[mode][f"sdft/{metric_name}"].append(value)

    def training_step(self, model: nn.Module, inputs: dict[str, torch.Tensor | Any], num_items_in_batch: int | None = None):
        loss = super().training_step(model, inputs, num_items_in_batch)
        self._step += 1
        return loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The SDFTTrainer does not support returning outputs")

        if self.num_loss_tokens_to_skip > 0:
            inputs = dict(inputs)
            completion_mask = inputs["completion_mask"].clone()
            token_positions = torch.arange(completion_mask.size(1), device=completion_mask.device).unsqueeze(0)
            completion_mask = completion_mask * (token_positions >= self.num_loss_tokens_to_skip).long()
            inputs["completion_mask"] = completion_mask

        loss = self._compute_self_distillation_loss(model, inputs)
        return loss / self.current_gradient_accumulation_steps
