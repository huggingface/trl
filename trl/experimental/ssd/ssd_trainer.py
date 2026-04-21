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

"""Trainer for Simple Self-Distillation (SSD).

Implements the method from *Embarrassingly Simple Self-Distillation Improves Code Generation* (Zhang et al., 2026):
sample completions from the frozen model at a training-time temperature and truncation configuration, then fine-tune on
those raw, unverified samples with standard cross-entropy loss. No reward model, verifier, teacher model, or
reinforcement learning is needed.
"""

from __future__ import annotations

import inspect
import math
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

from ...data_utils import maybe_apply_chat_template
from ...models import unwrap_model_for_generation
from ...trainer.base_trainer import _BaseTrainer
from ...trainer.utils import (
    RepeatSampler,
    create_model_from_path,
    disable_dropout_in_model,
    get_config_model_id,
    identity,
    pad,
    selective_log_softmax,
    split_tensor_dict,
)
from ..utils import prepare_peft_model
from .ssd_config import SSDConfig


if is_peft_available():
    from peft import PeftConfig


logger = get_logger(__name__)


class SSDTrainer(_BaseTrainer):
    """Trainer for SSD-style on-policy self-distillation with cross-entropy loss.

    SSD generates completions from the model at a specified training-time temperature and truncation configuration,
    then fine-tunes on those raw, unverified samples using standard cross-entropy loss. The dataset only requires a
    ``prompt`` column.
    """

    _tag_names = ["trl", "ssd"]
    _name = "SSD"
    config_cls = SSDConfig
    # docstyle-ignore
    _paper = {
        "title": "Embarrassingly Simple Self-Distillation Improves Code Generation",
        "id": "2604.01193",
        "citation": textwrap.dedent("""\
            @article{zhang2026ssd,
                title        = {{Embarrassingly Simple Self-Distillation Improves Code Generation}},
                author       = {Ruixiang Zhang and Richard He Bai and Huangjie Zheng and Navdeep Jaitly and Ronan Collobert and Yizhe Zhang},
                year         = 2026,
                eprint       = {arXiv:2604.01193}
            }"""),
    }

    def __init__(
        self,
        model: str | PreTrainedModel | nn.Module,
        args: SSDConfig | None = None,
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
            raise NotImplementedError("Iterable datasets are not yet supported in SSDTrainer.")
        if isinstance(eval_dataset, IterableDataset) or (
            isinstance(eval_dataset, dict) and any(isinstance(ds, IterableDataset) for ds in eval_dataset.values())
        ):
            raise NotImplementedError("Iterable eval datasets are not yet supported in SSDTrainer.")
        if args is None:
            model_name = model if isinstance(model, str) else get_config_model_id(model.config)
            model_name = model_name.split("/")[-1]
            args = SSDConfig(f"{model_name}-SSD")
        if isinstance(model, str):
            model_init_kwargs = args.model_init_kwargs or {}
            if args.distributed_state.distributed_type in ["MULTI_GPU", "DEEPSPEED"]:
                model_init_kwargs["device_map"] = None
            model = create_model_from_path(model, **model_init_kwargs)
        elif args.model_init_kwargs is not None:
            logger.warning(
                "You passed `model_init_kwargs` to `SSDConfig`, but `model` is already instantiated. "
                "The `model_init_kwargs` will be ignored."
            )

        self.model_kwarg_keys = (
            inspect.signature(model.forward).parameters.keys()
            if not hasattr(model, "get_base_model")
            else inspect.signature(model.get_base_model().forward).parameters.keys()
        )

        if is_peft_available() and is_peft_model(model) and peft_config is not None:
            raise ValueError(
                "You passed a `PeftModel` instance together with a `peft_config` to SSDTrainer. Pass either a base "
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
        # SSD always samples a single completion per prompt (N=1 in the paper).
        self.num_generations = 1
        self.num_iterations = args.num_iterations
        self.temperature = args.temperature
        self.shuffle_dataset = args.shuffle_dataset
        self.filter_empty = args.filter_empty
        self.use_vllm = args.use_vllm
        self.chat_template_kwargs = args.chat_template_kwargs or {}
        self._step = 0
        self._buffered_inputs = None
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}

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

        self.model.add_model_tags(self._tag_names)

        self.model_accepts_loss_kwargs = False

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

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    def _apply_prompt_template(self, prompts):
        return [
            maybe_apply_chat_template({"prompt": prompt}, self.processing_class, **self.chat_template_kwargs)["prompt"]
            for prompt in prompts
        ]

    # ------------------------------------------------------------------
    # DataLoader
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Training step management
    # ------------------------------------------------------------------

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
            return self._buffered_inputs[self._step % self.args.steps_per_generation]
        return self._build_buffered_batch(generation_batch)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _generate_completion_ids(self, prompts: list[Any]) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate completions from the model at the configured training-time temperature and truncation."""
        if self.use_vllm:
            return self._generate_completion_ids_vllm(prompts)
        return self._generate_completion_ids_transformers(prompts)

    def _generate_completion_ids_vllm(self, prompts: list[Any]) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate completions using vLLM."""
        if self.state.global_step != self._last_loaded_step:
            self.vllm_generation.sync_weights()
            self._last_loaded_step = self.state.global_step

        prompts_text = self._apply_prompt_template(prompts)
        tokenized = self.processing_class(
            text=prompts_text,
            return_tensors=None,
            padding=False,
            max_length=self.max_prompt_length,
            truncation=True,
            add_special_tokens=False,
        )
        prompt_ids = tokenized["input_ids"]

        _, completion_ids_list, _, _ = self.vllm_generation.generate(
            prompts=prompt_ids,
            images=None,
            num_generations=self.num_generations,
        )

        device = self.accelerator.device
        completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids_list]
        completion_mask = [torch.ones(len(ids), dtype=torch.long, device=device) for ids in completion_ids_list]
        return (
            pad(completion_ids, padding_value=self.pad_token_id, padding_side="right"),
            pad(completion_mask, padding_value=0, padding_side="right"),
        )

    def _generate_completion_ids_transformers(self, prompts: list[Any]) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate completions using the training model with transformers."""
        generate_inputs = self.processing_class(
            text=self._apply_prompt_template(prompts),
            return_tensors="pt",
            padding=True,
            padding_side="left",
            max_length=self.max_prompt_length,
            truncation=True,
            add_special_tokens=False,
        )
        # Use the base Trainer tensor preparation instead of re-entering the buffered outer training hook.
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

    # ------------------------------------------------------------------
    # Batch construction
    # ------------------------------------------------------------------

    def _build_buffered_batch(self, inputs: list[dict[str, Any]]) -> dict[str, torch.Tensor | Any]:
        """Generate completions and build the training batch."""
        prompts = [example["prompt"] for example in inputs]
        completion_ids, completion_mask = self._generate_completion_ids(prompts)

        # Optionally filter empty or single-line stub completions (the paper applies minimal syntactic filtering)
        if self.filter_empty:
            completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
            keep = torch.ones(len(completions_text), dtype=torch.bool, device=completion_ids.device)
            for i, text in enumerate(completions_text):
                stripped = text.strip()
                if len(stripped) == 0 or stripped.count("\n") == 0 and len(stripped) < 10:
                    keep[i] = False
            completion_mask = completion_mask * keep.unsqueeze(1).long()

        # Tokenize prompts for the training forward pass
        prompt_text = self._apply_prompt_template(prompts)
        prompt_inputs = self.processing_class(
            text=prompt_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            max_length=self.max_prompt_length,
            truncation=True,
            add_special_tokens=False,
        )
        prompt_inputs = _BaseTrainer._prepare_inputs(self, prompt_inputs)
        prompt_ids = prompt_inputs["input_ids"]
        prompt_mask = prompt_inputs["attention_mask"]

        # Log completion statistics
        mode = "train" if self.model.training else "eval"
        completion_lengths = completion_mask.sum(dim=1).float()
        agg_lengths = self.accelerator.gather(completion_lengths)
        self._metrics[mode]["completions/mean_length"].append(agg_lengths.mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_lengths.min().item())
        self._metrics[mode]["completions/max_length"].append(agg_lengths.max().item())
        active_ratio = (completion_mask.sum(dim=1) > 0).float().mean()
        self._metrics[mode]["ssd/active_sample_ratio"].append(self.accelerator.gather(active_ratio).mean().item())

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
        }

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The SSDTrainer does not support returning outputs")

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        # Forward pass
        model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask, "use_cache": False}
        if "logits_to_keep" in self.model_kwarg_keys:
            model_inputs["logits_to_keep"] = logits_to_keep + 1
        logits = model(**model_inputs).logits
        logits = logits[:, :-1, :]
        logits = logits[:, -logits_to_keep:, :]

        # Standard cross-entropy loss: -log p(y_t | x, y<t)
        per_token_logps = selective_log_softmax(logits, completion_ids)
        per_token_loss = -per_token_logps

        # Aggregate with mask
        loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)

        mode = "train" if self.model.training else "eval"
        self._metrics[mode]["ssd/cross_entropy_loss"].append(self.accelerator.gather(loss.detach()).mean().item())

        accumulation_scale = self.current_gradient_accumulation_steps if self.model.training else 1.0
        return loss / accumulation_scale

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        if not isinstance(inputs, dict):
            inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
        return loss.detach(), None, None

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        mode = "train" if self.model.training else "eval"
        metrics = {}
        for key, val in self._metrics[mode].items():
            valid = [v for v in val if not math.isnan(v)]
            metrics[key] = sum(valid) / len(valid) if valid else None

        # When called in evaluation, the keys in `logs` start with "eval_". We need to add the prefix "eval_" to the
        # keys in `metrics` to match the format.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        super().log(logs, start_time)
        self._metrics[mode].clear()
