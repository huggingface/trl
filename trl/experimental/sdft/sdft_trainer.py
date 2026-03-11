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
from typing import Any

import torch
from accelerate.logging import get_logger
from accelerate.utils import is_peft_model
from datasets import Dataset, IterableDataset
from torch import nn
from transformers import (
    AutoProcessor,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainerCallback,
)
from transformers.utils import is_peft_available

from ...models import prepare_deepspeed, prepare_fsdp, unwrap_model_for_generation
from ...trainer.base_trainer import _BaseTrainer
from ...trainer.callbacks import SyncRefModelCallback
from ...trainer.utils import (
    create_model_from_path,
    disable_dropout_in_model,
    get_config_model_id,
    identity,
    pad,
    use_adapter,
)
from ..self_distillation.self_distillation_mixin import SelfDistillationMixin
from ..self_distillation.teacher_context import DemonstrationTeacherContextBuilder, PromptTokenizer
from ..utils import prepare_peft_model
from .sdft_config import SDFTConfig


if is_peft_available():
    from peft import PeftConfig
    from peft.peft_model import PeftModel


logger = get_logger(__name__)


class SDFTTrainer(SelfDistillationMixin, _BaseTrainer):
    """Trainer for SDFT-style on-policy self-distillation with explicit teacher prompts."""

    _tag_names = ["trl", "sdft"]
    _name = "SDFT"
    config_cls = SDFTConfig

    def __init__(
        self,
        model: str | PreTrainedModel | nn.Module,
        ref_model: str | PreTrainedModel | nn.Module | None,
        args: SDFTConfig | None = None,
        train_dataset: Dataset | IterableDataset | None = None,
        eval_dataset: Dataset | IterableDataset | dict[str, Dataset | IterableDataset] | None = None,
        processing_class: PreTrainedTokenizerBase | ProcessorMixin | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None] = (None, None),
        peft_config: PeftConfig | None = None,
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
        if isinstance(model, str):
            model_init_kwargs = args.model_init_kwargs or {}
            if args.distributed_state.distributed_type in ["MULTI_GPU", "DEEPSPEED"]:
                model_init_kwargs["device_map"] = None
            model = create_model_from_path(model, **model_init_kwargs)
        elif args.model_init_kwargs is not None:
            logger.warning(
                "You passed `model_init_kwargs` to `SDFTConfig`, but `model` is already instantiated. "
                "The `model_init_kwargs` will be ignored."
            )
        if ref_model is model:
            raise ValueError(
                "`model` and `ref_model` cannot be the same object. Pass a separate teacher model, or set "
                "`ref_model=None` and use the PEFT adapter-disabled teacher path."
            )

        self.model_kwarg_keys = (
            inspect.signature(model.forward).parameters.keys()
            if not hasattr(model, "get_base_model")
            else inspect.signature(model.get_base_model().forward).parameters.keys()
        )

        if is_peft_available() and is_peft_model(model) and peft_config is not None:
            raise ValueError(
                "You passed a `PeftModel` instance together with a `peft_config` to SDFTTrainer. Pass either a base "
                "model with `peft_config`, or a pre-wrapped PEFT model."
            )
        if peft_config is not None or (is_peft_available() and getattr(model, "peft_config", None) is not None):
            model = prepare_peft_model(model, peft_config, args)
        if ref_model is None and not (is_peft_available() and is_peft_model(model)):
            raise ValueError("`ref_model` is required for SDFTTrainer unless `model` is a PEFT model.")

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
            if self.ref_model is not None:
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
        elif is_peft_available() and is_peft_model(self.model):
            self.teacher_model = None

        if args.sync_ref_model:
            if self.ref_model is None:
                raise NotImplementedError(
                    "You passed `sync_ref_model=True` while using PEFT without an explicit `ref_model`. In this "
                    "setup, SDFT recovers teacher behavior by temporarily disabling the adapter, so there is no "
                    "standalone reference model to synchronize."
                )
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
        # This generation helper builds tokenized model inputs directly, so use the base Trainer tensor preparation
        # instead of re-entering the buffered outer training hook.
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

        completion_ids_list = [c[m].tolist() for c, m in zip(completion_ids, completion_mask.bool(), strict=True)]
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

        teacher_batch = self.teacher_context_builder.build(
            prompts, privileged_contexts, completion_ids, completion_mask
        )

        prompt_completion_ids = torch.cat([teacher_batch["prompt_ids"], completion_ids], dim=1)
        attention_mask = torch.cat([teacher_batch["prompt_mask"], completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        with torch.no_grad():
            generate_every = self.args.steps_per_generation * self.num_iterations
            if not self.generate_from_teacher and self.args.gradient_accumulation_steps % generate_every != 0:
                old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    compute_entropy=False,
                )
            else:
                old_per_token_logps = None

        self._dispatch_self_distillation_callback(
            "on_self_distillation_batch_prepared",
            old_per_token_logps=old_per_token_logps,
            prompt_ids=teacher_batch["prompt_ids"],
            completion_ids=completion_ids,
        )
        output = {
            "prompt_ids": teacher_batch["prompt_ids"],
            "prompt_mask": teacher_batch["prompt_mask"],
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "teacher_input_ids": teacher_batch["teacher_input_ids"],
            "teacher_attention_mask": teacher_batch["teacher_attention_mask"],
        }
        if old_per_token_logps is not None:
            output["old_per_token_logps"] = old_per_token_logps
        return output

    def _build_buffered_batch(self, generation_batch):
        return self._generate_and_prepare_batch(generation_batch)

    def _log_self_distillation_metric(self, mode: str, metric_name: str, value: float) -> None:
        self._metrics[mode][f"self_distillation/{metric_name}"].append(value)
        self._metrics[mode][f"sdft/{metric_name}"].append(value)

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

    def _get_teacher_context_for_self_distillation(self, model):
        if is_peft_available() and isinstance(self.model, PeftModel) and self.ref_model is None:
            model = self.accelerator.unwrap_model(self.model)
            return use_adapter(model, adapter_name="ref" if "ref" in model.peft_config else None)
        return super()._get_teacher_context_for_self_distillation(model)
