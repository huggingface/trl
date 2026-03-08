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
from functools import partial
from typing import Any

import datasets
import torch
from datasets import Dataset, IterableDataset
from torch import nn
from torch.utils.data import DataLoader, Sampler
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
from transformers.trainer_utils import seed_worker
from transformers.utils import is_datasets_available, is_peft_available

from ...data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
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
)
from ..utils import prepare_peft_model
from .self_distillation_config import SelfDistillationConfig
from .self_distillation_mixin import SelfDistillationMixin


if is_peft_available():
    from peft import PeftConfig


class BaseSelfDistillationTrainer(SelfDistillationMixin, _BaseTrainer):
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
            data_source=eval_dataset, mini_repeat_count=self.num_generations_eval, seed=self.args.seed
        )

    def training_step(self, model, inputs, num_items_in_batch):
        output = super().training_step(model, inputs, num_items_in_batch)
        self._step += 1
        return output

    def _prepare_inputs(self, generation_batch: dict[str, torch.Tensor | Any]) -> dict[str, torch.Tensor | Any]:
        mode = "train" if self.model.training else "eval"
        if mode == "train":
            generate_every = self.args.steps_per_generation * self.num_iterations
            if self._step % generate_every == 0 or self._buffered_inputs is None:
                generation_batch = self._generate_and_score_completions(generation_batch)
                self._buffered_inputs = split_tensor_dict(generation_batch, self.args.steps_per_generation)
            return self._buffered_inputs[self._step % self.args.steps_per_generation]
        return self._generate_and_score_completions(generation_batch)

    def _apply_prompt_template(self, prompts: list[Any]) -> list[str]:
        return [
            maybe_apply_chat_template({"prompt": prompt}, self.processing_class, **self.chat_template_kwargs)["prompt"]
            for prompt in prompts
        ]

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

    def _generate(self, prompts: list[Any]):
        prompts_text = self._apply_prompt_template(prompts)
        generate_inputs = self.processing_class(
            text=prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            max_length=self.max_prompt_length,
            truncation=True,
            add_special_tokens=False,
        )
        generate_inputs = super()._prepare_inputs(generate_inputs)
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
        prompt_ids = generate_inputs["input_ids"]
        prompt_mask = generate_inputs["attention_mask"]
        prompt_length = prompt_ids.size(1)
        completion_ids = prompt_completion_ids[:, prompt_length:]
        is_eos = completion_ids == self.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=completion_ids.device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        seq_idx = torch.arange(is_eos.size(1), device=completion_ids.device).expand(is_eos.size(0), -1)
        completion_mask = (seq_idx <= eos_idx.unsqueeze(1)).int()
        prompt_ids_list = [p[m].tolist() for p, m in zip(prompt_ids, prompt_mask.bool(), strict=False)]
        completion_ids_list = [c[m].tolist() for c, m in zip(completion_ids, completion_mask.bool(), strict=False)]
        return prompt_ids_list, completion_ids_list

    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        device = self.accelerator.device
        if len(self.reward_funcs) == 0:
            return torch.zeros((len(prompts), 0), device=device)

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        keys = [key for key in inputs[0] if key not in ["prompt", "completion", "completion_ids"]]
        reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
        reward_kwargs["trainer_state"] = self.state

        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes, strict=True)
        ):
            if isinstance(reward_func, nn.Module):
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions, strict=True)]
                    texts = [
                        apply_chat_template(x, reward_processing_class, **self.chat_template_kwargs)["text"]
                        for x in messages
                    ]
                else:
                    texts = [p + c for p, c in zip(prompts, completions, strict=True)]
                reward_inputs = reward_processing_class(
                    text=texts,
                    return_tensors="pt",
                    padding=True,
                    padding_side="right",
                    add_special_tokens=False,
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]
            else:
                output_reward_func = reward_func(
                    prompts=prompts,
                    completions=completions,
                    completion_ids=completion_ids_list,
                    **reward_kwargs,
                )
                output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        return self.accelerator.gather(rewards_per_func)

    def _generate_and_score_completions(
        self, inputs: list[dict[str, torch.Tensor | Any]]
    ) -> dict[str, torch.Tensor | Any]:
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"
        prompts = [x["prompt"] for x in inputs]
        prompt_ids_list, completion_ids_list = self._generate(prompts)

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

        if is_conversational({"prompt": prompts[0]}):
            completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
            completions = [[{"role": "assistant", "content": content}] for content in completions_text]
        else:
            completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

        rewards_per_func = self._calculate_rewards(inputs, prompts, completions, completion_ids_list)
        if rewards_per_func.numel() == 0:
            rewards = torch.zeros(self.accelerator.num_processes * len(prompts), device=device)
        else:
            rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)
        num_generations = self.num_generations if mode == "train" else self.num_generations_eval
        mean_grouped_rewards = rewards.view(-1, num_generations).mean(dim=1).repeat_interleave(num_generations, dim=0)
        if self.scale_rewards == "batch":
            std_rewards = rewards.std().expand_as(rewards) if rewards.numel() > 1 else torch.zeros_like(rewards)
        elif self.scale_rewards == "none":
            std_rewards = torch.ones_like(rewards)
        else:
            std_rewards = rewards.view(-1, num_generations).std(dim=1).repeat_interleave(num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_rewards + 1e-4)

        local_batch_size = completion_ids.size(0)
        process_start = self.accelerator.process_index * local_batch_size
        process_slice = slice(process_start, process_start + local_batch_size)
        advantages = advantages[process_slice]

        agg_completion_lengths = self.accelerator.gather(
            torch.tensor([len(ids) for ids in completion_ids_list], device=device)
        )
        self._metrics[mode]["completions/mean_length"].append(agg_completion_lengths.float().mean().item())

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "num_items_in_batch": completion_mask.sum().detach(),
        }

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError(f"The {self.__class__.__name__} does not support returning outputs")
        return self._compute_loss(model, inputs)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        if not isinstance(inputs, dict):
            inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
        return loss.detach(), None, None

    def _compute_loss(self, model, inputs):
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        per_token_logps, _ = self._get_per_token_logps_and_entropies(
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            compute_entropy=False,
        )
        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps
        advantages = inputs["advantages"]
        if advantages.dim() == 1:
            advantages = advantages.unsqueeze(1)
        log_ratio = per_token_logps - old_per_token_logps
        if self.importance_sampling_level == "sequence":
            log_ratio = (log_ratio * completion_mask).sum(-1, keepdim=True) / completion_mask.sum(
                -1, keepdim=True
            ).clamp(min=1.0)
        coef_1 = torch.exp(log_ratio)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        per_token_loss = -torch.min(coef_1 * advantages, coef_2 * advantages)

        mode = "train" if self.model.training else "eval"
        if self.loss_type == "grpo":
            loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
            loss = loss / (self.current_gradient_accumulation_steps if mode == "train" else 1.0)
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
            loss = loss / (self.current_gradient_accumulation_steps if mode == "train" else 1.0)
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * completion_mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
            loss = loss / (self.current_gradient_accumulation_steps if mode == "train" else 1.0)
        else:
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
            loss = loss / (self.current_gradient_accumulation_steps if mode == "train" else 1.0)

        self._metrics[mode]["self_distillation/policy_loss"].append(
            self.accelerator.gather(loss.detach()).mean().item()
        )
        return loss
