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
from ...trainer.callbacks import SyncRefModelCallback
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
from ..self_distillation.self_distillation_mixin import SelfDistillationMixin
from ..self_distillation.teacher_context import PromptTokenizer, extract_last_user_text
from ..utils import prepare_peft_model
from .sdft_config import SDFTConfig


if is_peft_available():
    from peft import PeftConfig
    from peft.peft_model import PeftModel

    from ..self_distillation.peft_adapter_ema_callback import PEFTAdapterEMACallback


logger = get_logger(__name__)


class DemonstrationTeacherContextBuilder:
    """Builds student and teacher contexts from prompts plus privileged context, as in SDFT."""

    def __init__(self, trainer):
        self.trainer = trainer
        self.prompt_tokenizer = PromptTokenizer(trainer)

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
        if not self.trainer.generate_from_teacher:
            return prompts
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
        student_batch = self.prompt_tokenizer.tokenize_prompts(prompts)
        teacher_prompts = [
            self._compose_teacher_prompt(prompt, privileged_context)
            for prompt, privileged_context in zip(prompts, privileged_contexts, strict=True)
        ]
        teacher_batch = self.prompt_tokenizer.tokenize_prompts(teacher_prompts)
        teacher_input_ids = torch.cat([teacher_batch.prompt_ids, completion_ids], dim=1)
        teacher_attention_mask = torch.cat([teacher_batch.prompt_mask, completion_mask], dim=1)
        return {
            "prompt_ids": student_batch.prompt_ids,
            "prompt_mask": student_batch.prompt_mask,
            "teacher_input_ids": teacher_input_ids,
            "teacher_attention_mask": teacher_attention_mask,
        }


class SDFTTrainer(SelfDistillationMixin, _BaseTrainer):
    """Trainer for SDFT-style on-policy self-distillation with explicit teacher prompts."""

    _tag_names = ["trl", "sdft"]
    _name = "SDFT"
    config_cls = SDFTConfig
    # docstyle-ignore
    _paper = {
        "title": "Self-Training with On-Policy Self-Distillation for Language Model Alignment",
        "id": "2601.19897",
        "citation": textwrap.dedent("""\
            @article{hubotter2026selftraining,
                title        = {{Self-Training with On-Policy Self-Distillation for Language Model Alignment}},
                author       = {Jonas H\\"ubotter and Frederike L\\"ubeck and Lejs Behric and Anton Baumann and Marco Bagatella and Daniel Marta and Ido Hakimi and Idan Shenfeld and Thomas Kleine Buening and Carlos Guestrin and Andreas Krause},
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
        self.chat_template_kwargs = args.chat_template_kwargs or {}
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

        if args.disable_dropout:
            disable_dropout_in_model(self.model)

        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        # In self-distillation the teacher is always derived from the student:
        # - PEFT: base model with adapter disabled (or EMA teacher adapter when sync_ref_model=True)
        # - Non-PEFT: same model (or deep-copied EMA model when sync_ref_model=True)
        self.teacher_model = None

        if args.sync_ref_model:
            if is_peft_available() and is_peft_model(self.model):
                self.add_callback(
                    PEFTAdapterEMACallback(
                        model=self.model,
                        teacher_adapter_name="teacher",
                        update_rate=args.ref_model_mixup_alpha,
                        sync_steps=args.ref_model_sync_steps,
                        accelerator=self.accelerator,
                    )
                )
            else:
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
                self.add_callback(SyncRefModelCallback(ref_model=self.teacher_model, accelerator=self.accelerator))

        self.model_accepts_loss_kwargs = False

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
            mini_repeat_count=self.num_generations,
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
                generation_batch = self._build_buffered_batch(generation_batch)
                self._buffered_inputs = split_tensor_dict(generation_batch, self.args.steps_per_generation)
                self._dispatch_self_distillation_callback(
                    "on_generation_batch_built",
                    generate_every=generate_every,
                    steps_per_generation=self.args.steps_per_generation,
                )
            return self._buffered_inputs[self._step % self.args.steps_per_generation]
        return self._build_buffered_batch(generation_batch)

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
                **generate_inputs, generation_config=self.generation_config
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

    def _build_buffered_batch(self, inputs: list[dict[str, Any]]) -> dict[str, torch.Tensor | Any]:
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
        accumulation_scale = self.current_gradient_accumulation_steps if self.model.training else 1.0
        return loss / accumulation_scale

    def _get_teacher_context_for_self_distillation(self, model):
        if is_peft_available() and isinstance(self.model, PeftModel):
            model = self.accelerator.unwrap_model(self.model)
            if self.args.sync_ref_model and "teacher" in model.peft_config:
                return use_adapter(model, adapter_name="teacher")
            return use_adapter(model, adapter_name=None)
        return super()._get_teacher_context_for_self_distillation(model)
