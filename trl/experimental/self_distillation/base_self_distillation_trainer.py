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
from datasets import Dataset, IterableDataset
from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainerCallback,
)
from transformers.utils import is_peft_available

from ...models import prepare_deepspeed, prepare_fsdp
from ...trainer.base_trainer import _BaseTrainer
from ...trainer.utils import (
    create_model_from_path,
    disable_dropout_in_model,
    get_config_model_id,
    identity,
)
from ..utils import prepare_peft_model
from .online_rollout_mixin import OnlineRolloutMixin
from .self_distillation_config import SelfDistillationConfig
from .self_distillation_mixin import SelfDistillationMixin


if is_peft_available():
    from peft import PeftConfig


class BaseSelfDistillationTrainer(OnlineRolloutMixin, SelfDistillationMixin, _BaseTrainer):
    """Shared scaffold for experimental self-distillation trainers without GRPO inheritance."""

    config_cls = SelfDistillationConfig
    _tag_names = ["trl", "self-distillation"]
    _name = "SelfDistillation"

    def __init__(
        self,
        model: str | PreTrainedModel | nn.Module,
        reward_funcs: Any | list[Any] | None = None,
        args: SelfDistillationConfig | None = None,
        train_dataset: Dataset | IterableDataset | None = None,
        eval_dataset: Dataset | IterableDataset | dict[str, Dataset | IterableDataset] | None = None,
        processing_class: PreTrainedTokenizerBase | ProcessorMixin | None = None,
        reward_processing_classes: PreTrainedTokenizerBase | list[PreTrainedTokenizerBase] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None] = (None, None),
        peft_config: PeftConfig | None = None,
    ):
        args = self._coerce_self_distillation_args(args)
        if train_dataset is None:
            raise ValueError("`train_dataset` is required")
        if args.use_vllm:
            raise NotImplementedError("Self-distillation trainers do not support `use_vllm=True` yet.")

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
        self.temperature = args.temperature
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length
        self.num_generations = args.num_generations
        self.num_generations_eval = args.num_generations_eval or args.num_generations
        self.num_iterations = args.num_iterations
        self.shuffle_dataset = args.shuffle_dataset
        self.loss_type = args.loss_type
        self.importance_sampling_level = args.importance_sampling_level
        self.scale_rewards = args.scale_rewards
        self.epsilon_low = args.epsilon
        self.epsilon_high = args.epsilon_high
        self.beta = args.beta
        self.mask_truncated_completions = args.mask_truncated_completions
        self.chat_template_kwargs = args.chat_template_kwargs or {}
        self._step = 0
        self._buffered_inputs = None
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._diagnostic_counters = {
            "train": defaultdict(int),
            "eval": defaultdict(int),
        }

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

        if reward_funcs is None:
            reward_funcs = []
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        self.reward_func_names = []
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_model_init_kwargs = args.model_init_kwargs or {}
                if args.distributed_state.distributed_type in ["MULTI_GPU", "DEEPSPEED"]:
                    reward_model_init_kwargs["device_map"] = None
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func,
                    num_labels=1,
                    **reward_model_init_kwargs,
                )
            if isinstance(reward_funcs[i], nn.Module):
                self.reward_func_names.append(get_config_model_id(reward_funcs[i].config).split("/")[-1])
            else:
                self.reward_func_names.append(reward_funcs[i].__name__)
        self.reward_funcs = reward_funcs

        if args.reward_weights is not None:
            if len(args.reward_weights) != len(self.reward_funcs):
                raise ValueError("Number of reward weights must match number of reward functions")
            self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32)
        else:
            self.reward_weights = torch.ones(len(self.reward_funcs), dtype=torch.float32)

        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(self.reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        if len(reward_processing_classes) != len(self.reward_funcs):
            raise ValueError("Number of reward processing classes must match number of reward functions")

        for i, (reward_processing_class, reward_func) in enumerate(
            zip(reward_processing_classes, self.reward_funcs, strict=True)
        ):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(get_config_model_id(reward_func.config))
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        if args.disable_dropout:
            disable_dropout_in_model(self.model)

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, nn.Module):
                if self.is_deepspeed_enabled:
                    self.reward_funcs[i] = prepare_deepspeed(reward_func, self.accelerator)
                elif self.is_fsdp_enabled:
                    self.reward_funcs[i] = prepare_fsdp(reward_func, self.accelerator)
                else:
                    self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)
        self.model_accepts_loss_kwargs = False
        self.ref_model = None
        self.teacher_model = None
        if args.sync_ref_model:
            raise ValueError(
                "sync_ref_model is not supported on the shared online self-distillation base without `ref_model`."
            )

    def _prepare_auxiliary_model_for_eval(self, aux_model: nn.Module):
        if self.is_deepspeed_enabled:
            return prepare_deepspeed(aux_model, self.accelerator)
        if self.is_fsdp_enabled:
            return prepare_fsdp(aux_model, self.accelerator)
        return self.accelerator.prepare_model(aux_model, evaluation_mode=True)

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = ["prompt", "privileged_context"]
