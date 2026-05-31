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
import textwrap
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
from torch.utils.data import DataLoader, Sampler
from transformers import (
    AutoProcessor,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainerCallback,
)
from transformers.trainer_utils import seed_worker
from transformers.utils import is_datasets_available, is_peft_available

from ...data_utils import is_conversational
from ...models import prepare_deepspeed, prepare_fsdp
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
from ..self_distillation.generation import Generation
from ..self_distillation.loss_utils import (
    aggregate_loss,
    apply_importance_sampling_clipping,
    compute_full_logit_self_distillation_loss,
    compute_sampled_token_self_distillation_loss,
    compute_topk_self_distillation_loss,
    select_token_log_probs,
)
from ..self_distillation.prompt_utils import extract_last_user_text
from ..self_distillation.teacher_sync import PEFTAdapterEMACallback, SyncTeacherModelCallback, is_pure_lora_training
from ..utils import prepare_peft_model
from .sdft_config import SDFTConfig


if is_peft_available():
    from peft import PeftConfig


logger = get_logger(__name__)


TrainingBatch = dict[str, torch.Tensor | Any]


@dataclass
class DistillationLogits:
    """Aligned logits and masks used to compute a self-distillation objective."""

    completion_ids: torch.Tensor
    completion_mask: torch.Tensor
    response_mask: torch.Tensor
    student_logits: torch.Tensor
    teacher_logits: torch.Tensor


class DemonstrationTeacherContextBuilder:
    """Builds student and teacher contexts from prompts plus privileged context"""

    def __init__(self, trainer):
        self.trainer = trainer

    def _stringify_privileged_context(self, privileged_context: Any) -> str:
        if privileged_context is None:
            raise ValueError(
                "`privileged_context` must not be None for self-distillation teacher prompt construction."
            )
        if isinstance(privileged_context, str):
            return privileged_context
        if isinstance(privileged_context, list) and privileged_context and isinstance(privileged_context[0], dict):
            chunks = []
            for message in privileged_context:
                content = message.get("content", "")
                if isinstance(content, list):
                    text = " ".join(part.get("text", "") for part in content if part.get("type") == "text")
                else:
                    text = str(content)
                if text:
                    chunks.append(text)
            return "\n".join(chunks)
        return str(privileged_context)

    def _compose_teacher_prompt(self, prompt: Any, privileged_context: Any) -> Any:
        privileged_text = self._stringify_privileged_context(privileged_context)
        if isinstance(prompt, list):
            system_messages = prompt[:-1]
            prompt_text = extract_last_user_text(prompt)
            teacher_text = self.trainer.args.teacher_prompt_template.format(
                prompt=prompt_text,
                privileged_context=privileged_text,
            )
            return system_messages + [{"role": "user", "content": teacher_text}]
        return self.trainer.args.teacher_prompt_template.format(prompt=prompt, privileged_context=privileged_text)

    def select_generation_prompts(self, prompts: list[Any], privileged_contexts: list[Any]) -> list[Any]:
        return [
            self._compose_teacher_prompt(prompt, privileged_context)
            for prompt, privileged_context in zip(prompts, privileged_contexts, strict=True)
        ]

    def build(
        self,
        prompts: list[Any],
        privileged_contexts: list[Any],
        completion_ids: torch.Tensor,
        completion_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        teacher_prompts = [
            self._compose_teacher_prompt(prompt, privileged_context)
            for prompt, privileged_context in zip(prompts, privileged_contexts, strict=True)
        ]
        teacher_prompt_ids_list = self.trainer._tokenize_prompts(teacher_prompts)
        device = completion_ids.device
        teacher_prompt_ids = [torch.tensor(ids) for ids in teacher_prompt_ids_list]
        teacher_prompt_mask = [torch.ones_like(ids, dtype=torch.long) for ids in teacher_prompt_ids]
        teacher_prompt_ids = pad(
            teacher_prompt_ids, padding_value=self.trainer._tokenizer.pad_token_id, padding_side="left"
        ).to(device=device)
        teacher_prompt_mask = pad(teacher_prompt_mask, padding_value=0, padding_side="left").to(device=device)
        teacher_input_ids = torch.cat([teacher_prompt_ids, completion_ids], dim=1)
        teacher_attention_mask = torch.cat([teacher_prompt_mask, completion_mask], dim=1)
        return {
            "teacher_input_ids": teacher_input_ids,
            "teacher_attention_mask": teacher_attention_mask,
        }


class SDFTTrainer(_BaseTrainer):
    """Trainer for SDFT-style on-policy self-distillation with explicit teacher prompts."""

    _tag_names = ["trl", "sdft"]
    _name = "SDFT"
    config_cls = SDFTConfig
    # docstyle-ignore
    _paper = {
        "title": "Self-Distillation Enables Continual Learning",
        "id": "2601.19897",
        "citation": textwrap.dedent("""\
            @article{shenfeld2026selfdistillation,
                title        = {{Self-Distillation Enables Continual Learning}},
                author       = {Idan Shenfeld and Mehul Damani and Jonas H\\"ubotter and Pulkit Agrawal},
                year         = 2026,
                eprint       = {arXiv:2601.19897}
            }"""),
    }

    def __init__(
        self,
        model: str | PreTrainedModel | nn.Module,
        args: SDFTConfig | None = None,
        train_dataset: Dataset | IterableDataset | None = None,
        eval_dataset: Dataset | IterableDataset | dict[str, Dataset | IterableDataset] | None = None,
        processing_class: PreTrainedTokenizerBase | ProcessorMixin | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None] = (None, None),
        peft_config: PeftConfig | None = None,
    ):
        if isinstance(train_dataset, IterableDataset):
            raise NotImplementedError("Iterable datasets are not yet supported in SDFTTrainer.")
        if isinstance(eval_dataset, IterableDataset) or (
            isinstance(eval_dataset, dict) and any(isinstance(ds, IterableDataset) for ds in eval_dataset.values())
        ):
            raise NotImplementedError("Iterable eval datasets are not yet supported in SDFTTrainer.")

        self.num_loss_tokens_to_skip = args.num_loss_tokens_to_skip
        self.teacher_context_builder = DemonstrationTeacherContextBuilder(self)

        if train_dataset is None:
            raise ValueError("`train_dataset` is required")

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

        if peft_config is None and getattr(model, "peft_config", None) is not None:
            logger.warning(
                "The provided self-distillation student model already contains a PEFT adapter. "
                "This setup is accepted but not directly supported. In particular, `teacher_model_kind='base'` "
                "may refer to the underlying base weights rather than the exact initially loaded student state "
                "including its adapter. For unambiguous teacher behavior, start from a merged/non-adapter model "
                "or manage separate adapters explicitly."
            )
        if peft_config is not None:
            if not is_peft_available():
                raise ImportError(
                    "You passed `peft_config` but the `peft` library is not installed. "
                    "Install it with `pip install trl[peft]`."
                )
            if not isinstance(peft_config, PeftConfig):
                raise TypeError(
                    f"`peft_config` must be a `peft.PeftConfig` instance (e.g. `peft.LoraConfig`), "
                    f"got {type(peft_config).__name__}."
                )
            if is_peft_model(model):
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
            self._tokenizer = processing_class.tokenizer
        elif isinstance(processing_class, PreTrainedTokenizerBase):
            self._tokenizer = processing_class
        else:
            raise TypeError("The `processing_class` must be either a `PreTrainedTokenizerBase` or a `ProcessorMixin`")

        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length
        self.num_generations = args.num_generations
        self.num_generations_eval = args.num_generations_eval or args.num_generations
        self.num_iterations = args.num_iterations
        self.shuffle_dataset = args.shuffle_dataset
        self.loss_type = args.loss_type
        self.temperature = args.temperature
        self.generate_from_teacher = args.generate_from_teacher
        self.chat_template_kwargs = args.chat_template_kwargs or {}
        self._step = 0
        self._buffered_inputs = None
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}

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

        self.generation_engine = Generation(
            model=self.model,
            model_wrapped=self.model_wrapped,
            args=args,
            accelerator=self.accelerator,
            processing_class=self.processing_class,
            tokenizer=self._tokenizer,
            is_fsdp_enabled=self.is_fsdp_enabled,
            max_completion_length=self.max_completion_length,
            temperature=self.temperature,
        )
        self._last_loaded_step = -1 if args.use_vllm else 0

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
        """Prepare teacher state according to the semantic teacher choice.

        Resolve `teacher_model_kind` × PEFT state into the effective teacher:

            - `"live"` (any model):
                Teacher is the student. No divergence, no callback.
            - `"base"` + PEFT model:
                Teacher reuses `self.model`; the base weights are recovered downstream by disabling the adapter via
                `use_adapter` during teacher forward.
            - `"base"` + non-PEFT model:
                Teacher is a frozen deepcopy of the initial student (falls through to the copy branch below).
            - `"ema"` + pure-LoRA training:
                Teacher reuses `self.model`; a dedicated `"teacher"` LoRA adapter is attached and updated by
                `PEFTAdapterEMACallback`. Teacher forward switches to that adapter downstream.
            - `"ema"` (otherwise):
                Teacher is a frozen deepcopy synchronized each step by `SyncTeacherModelCallback`.

        Must be called after `super().__init__` so that `self.callback_handler` is available.
        """

        teacher_model_kind = self.args.teacher_model_kind

        if teacher_model_kind == "live":
            self.teacher_model = self.model
            return

        if teacher_model_kind == "base" and is_peft_model(self.model):
            self.teacher_model = self.model
            return

        if self._use_peft_ema_teacher_adapter():
            # Must run after super().__init__ so self.callback_handler exists.
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
        return self.args.teacher_model_kind == "ema" and is_pure_lora_training(self.model, self.accelerator)

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

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        if not isinstance(inputs, dict):
            inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
        return loss.detach(), None, None

    def _prepare_inputs(self, generation_batch):
        """Return the per-step training batch, regenerating rollouts and buffering them for reuse in train mode.

        In train mode, rollouts are generated once every `steps_per_generation * num_iterations` steps and split into
        per-step slices reused until the next regeneration. In eval mode, every batch is freshly prepared.
        """
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
        """Sample student rollouts and construct teacher prompts"""
        batch = self.sample_rollouts(inputs)
        prompts = [example["prompt"] for example in inputs]
        privileged_contexts = [example.get("privileged_context") for example in inputs]
        teacher_batch = self.teacher_context_builder.build(
            prompts,
            privileged_contexts,
            batch["completion_ids"],
            batch["completion_mask"],
        )
        batch.update(
            {
                "teacher_input_ids": teacher_batch["teacher_input_ids"],
                "teacher_attention_mask": teacher_batch["teacher_attention_mask"],
            }
        )

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

    def sample_rollouts(self, inputs: list[dict[str, Any]]) -> TrainingBatch:
        """Generate completions for a batch of prompts and assemble the training batch."""
        prompts = [example["prompt"] for example in inputs]
        privileged_contexts = [example.get("privileged_context") for example in inputs]

        student_prompt_ids_list = self._tokenize_prompts(prompts)
        if self.generate_from_teacher:
            generation_prompts = self.teacher_context_builder.select_generation_prompts(prompts, privileged_contexts)
            generation_prompt_ids_list = self._tokenize_prompts(generation_prompts)
        else:
            generation_prompts = prompts
            generation_prompt_ids_list = student_prompt_ids_list

        self._dispatch_self_distillation_callback(
            "on_generation_prompts_selected",
            generation_prompts=generation_prompts,
            generation_prompt_text=None,
        )

        num_generations = self.num_generations if self.model.training else self.num_generations_eval
        if self.args.use_vllm and self.state.global_step != self._last_loaded_step:
            self.generation_engine.sync_weights()
            self._last_loaded_step = self.state.global_step
        generation_output = self.generation_engine.generate(
            generation_prompt_ids_list,
            num_generations=num_generations,
        )
        completion_ids_list = generation_output.completion_ids
        device = self.accelerator.device
        prompt_ids = [torch.tensor(ids) for ids in student_prompt_ids_list]
        prompt_mask = [torch.ones_like(ids, dtype=torch.long) for ids in prompt_ids]
        prompt_ids = pad(prompt_ids, padding_value=self._tokenizer.pad_token_id, padding_side="left").to(device=device)
        prompt_mask = pad(prompt_mask, padding_value=0, padding_side="left").to(device=device)

        completion_ids = [torch.tensor(ids) for ids in completion_ids_list]
        completion_mask = [torch.ones_like(ids, dtype=torch.long) for ids in completion_ids]
        completion_ids = pad(completion_ids, padding_value=self._tokenizer.pad_token_id, padding_side="right").to(
            device=device
        )
        completion_mask = pad(completion_mask, padding_value=0, padding_side="right").to(device=device)

        old_per_token_logps = None
        if not self.generate_from_teacher:
            old_per_token_logps = self._compute_rollout_logps(
                prompt_ids=prompt_ids,
                prompt_mask=prompt_mask,
                completion_ids=completion_ids,
                completion_mask=completion_mask,
            )
        batch: TrainingBatch = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "raw_completion_lengths": torch.tensor(
                [len(ids) for ids in completion_ids_list], device=device, dtype=torch.long
            ),
        }
        if old_per_token_logps is not None:
            batch["old_per_token_logps"] = old_per_token_logps
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

    def _compute_self_distillation_loss(
        self,
        model,
        inputs: TrainingBatch,
        distillation_logits: DistillationLogits,
    ) -> torch.Tensor:
        """Compute the per-token distillation loss and aggregate it according to `loss_type`.

        Dispatches between three objectives based on `distillation_mode`:

            - `"topk_logits"`: top-k approximation of the divergence, optionally with a tail bucket for the remaining
              probability mass (`distillation_add_tail`).
            - `"full_logits"`: full-vocab divergence.
            - `"sampled_token"`: token-level (reverse-KL) distillation on sampled `completion_ids`.

        When `distillation_is_clip` is set and `old_per_token_logps` are available, the loss is corrected by a clipped
        importance-sampling ratio between the current student and the student at rollout time.
        """
        if distillation_logits.response_mask.sum() == 0:
            mode = "train" if model.training else "eval"
            self._log_self_distillation_metric(mode, 0.0)
            # Keep the zero loss attached to the student graph so backward produces zero gradients instead of stopping.
            return distillation_logits.student_logits.sum() * 0.0

        if self.args.distillation_mode == "topk_logits":
            if self.args.distillation_topk is None:
                raise ValueError("`distillation_mode='topk_logits'` requires `distillation_topk` to be set.")
            per_token_loss = compute_topk_self_distillation_loss(
                distillation_logits.student_logits,
                distillation_logits.teacher_logits,
                distillation_topk=self.args.distillation_topk,
                distillation_alpha=self.args.distillation_alpha,
                distillation_add_tail=self.args.distillation_add_tail,
            )
        elif self.args.distillation_mode == "full_logits":
            per_token_loss = compute_full_logit_self_distillation_loss(
                distillation_logits.student_logits,
                distillation_logits.teacher_logits,
                distillation_alpha=self.args.distillation_alpha,
            )
        elif self.args.distillation_mode == "sampled_token":
            per_token_loss = compute_sampled_token_self_distillation_loss(
                distillation_logits.student_logits,
                distillation_logits.teacher_logits,
                distillation_logits.completion_ids,
                distillation_alpha=self.args.distillation_alpha,
            )
        else:
            raise ValueError(
                "distillation_mode must be one of: 'sampled_token', 'full_logits', 'topk_logits', "
                f"got {self.args.distillation_mode!r}"
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

        loss = aggregate_loss(
            per_token_loss,
            distillation_logits.response_mask,
            loss_type=self.loss_type,
            max_completion_length=self.max_completion_length,
        )

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
        """Compute the per-token logits of the student and teacher over the completion tokens.

        The student is forwarded on its own input (original prompt plus the sampled completion) while the teacher is
        forwarded on its input (prompt, privileged context, and the same completion). Both sets of logits are aligned
        to the completion tokens so they can be compared position-by-position in the distillation loss.

        The teacher forward runs under `torch.no_grad()` and the context resolved by
        `_get_teacher_context_for_self_distillation`, which routes it to the correct weights.
        """
        prompt_ids = inputs["prompt_ids"]
        prompt_mask = inputs["prompt_mask"]
        completion_ids = inputs["completion_ids"]
        completion_mask = inputs["completion_mask"]
        logits_to_keep = completion_ids.size(1)

        response_mask = self._build_self_distillation_response_mask(
            completion_mask,
            inputs.get("self_distillation_mask"),
        )
        student_input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        student_attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        student_logits = self._forward_logits(
            model=model,
            input_ids=student_input_ids,
            attention_mask=student_attention_mask,
            logits_to_keep=logits_to_keep,
        )

        with torch.no_grad(), self._get_teacher_context_for_self_distillation():
            teacher_logits = self._forward_logits(
                model=teacher_model,
                input_ids=inputs["teacher_input_ids"],
                attention_mask=inputs["teacher_attention_mask"],
                logits_to_keep=logits_to_keep,
            )

        return DistillationLogits(
            completion_ids=completion_ids,
            completion_mask=completion_mask,
            response_mask=response_mask,
            student_logits=student_logits,
            teacher_logits=teacher_logits,
        )

    def _forward_logits(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        logits_to_keep: int,
    ) -> torch.Tensor:
        """Forward the model and return temperature-scaled logits aligned to the completion tokens."""
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

    def _get_teacher_context_for_self_distillation(self):
        """Return the context manager that routes the teacher forward to the correct weights.

        For non-PEFT models this is a no-op. For PEFT models:

            - `teacher_model_kind == "base"`: disable the student adapter so the teacher forward uses the base weights.
            - `teacher_model_kind == "ema"` under pure-LoRA training: switch to the `"teacher"` LoRA adapter.
            - otherwise: no-op; the teacher is a separate deepcopy.
        """
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
        metric_prefix = self._name.lower().replace(" ", "_")
        self._metrics[mode]["self_distillation/distillation_loss"].append(value)
        self._metrics[mode][f"{metric_prefix}/distillation_loss"].append(value)

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        mode = "train" if self.model.training else "eval"
        metrics = {k: sum(v) / len(v) for k, v in self._metrics[mode].items() if v}
        if mode == "eval":
            metrics = {f"eval_{k}": v for k, v in metrics.items()}
        logs = {**logs, **metrics}
        super().log(logs, start_time)
        self._metrics[mode].clear()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The SDFTTrainer does not support returning outputs")

        distillation_logits = self._compute_teacher_student_logits(model, self.teacher_model, inputs)
        loss = self._compute_self_distillation_loss(model, inputs, distillation_logits)
        accumulation_scale = self.current_gradient_accumulation_steps if self.model.training else 1.0
        return loss / accumulation_scale

    def _build_self_distillation_response_mask(
        self,
        completion_mask: torch.Tensor,
        self_distillation_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if self_distillation_mask is None:
            response_mask = completion_mask
        else:
            response_mask = completion_mask * self_distillation_mask.unsqueeze(1)
        if self.num_loss_tokens_to_skip <= 0:
            return response_mask

        # SDFT skips the first few completion tokens only in the distillation loss to suppress teacher-prompt artifacts.
        token_positions = torch.arange(response_mask.size(1), device=response_mask.device).unsqueeze(0)
        skip_mask = (token_positions >= self.num_loss_tokens_to_skip).long()
        return response_mask * skip_mask
