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

import contextlib
from collections import defaultdict
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import accelerate
import torch
import torch.nn.functional as F
from accelerate.utils import recursively_apply
from datasets import Dataset, IterableDataset
from packaging.version import Version
from transformers import (
    AutoModelForDecision,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainerCallback,
    set_seed,
)
from transformers.utils import is_peft_available

from ...trainer.base_trainer import _BaseTrainer
from .decision_config import DecisionConfig


if is_peft_available():
    from peft import get_peft_model


def _get_decision_probabilities(outputs, inputs):
    logits = outputs.logits.float()
    mask = inputs.get("option_mask", torch.ones_like(logits, dtype=torch.bool))
    probabilities = logits.masked_fill(~mask, -torch.inf).softmax(-1)
    allow_null = inputs.get("allow_null")
    if allow_null is not None:
        coverage_logits = outputs.get("coverage_logits")
        if coverage_logits is None:
            if allow_null.any():
                raise ValueError("Nullable decisions require a model that returns coverage_logits")
            null_probability = torch.zeros_like(probabilities[:, :1])
        else:
            coverage_logits = coverage_logits.float()
            coverage = torch.where(allow_null, coverage_logits.sigmoid(), 1.0)
            null_probability = torch.where(allow_null, (-coverage_logits).sigmoid(), 0.0)[:, None]
            probabilities = coverage[:, None] * probabilities
        probabilities = torch.cat([null_probability, probabilities], -1)
        mask = F.pad(mask, (1, 0), value=True)
    return probabilities, mask


class DecisionTrainer(_BaseTrainer):
    """
    Trainer for models that return a supervised decision loss.

    Models return a mean `loss` and candidate `logits`. Labels may be class indices or target distributions. Coverage
    and MoE auxiliary outputs are optional. Optimization, distributed wrapping, checkpointing, and callbacks use
    [`~transformers.Trainer`].

    Args:
        model ([`~transformers.PreTrainedModel`] or `str`):
            Model to train, or identifier of a saved decision checkpoint.
        args ([`experimental.decision.DecisionConfig`], *optional*):
            Training configuration.
        data_collator (`Callable`, *optional*):
            Collator producing the model's forward inputs and `labels`.
        train_dataset (`Dataset` or `IterableDataset`, *optional*):
            Dataset of model inputs or raw requests prepared by the collator.
        eval_dataset (`Dataset` or `IterableDataset` or `dict`, *optional*):
            Evaluation dataset or named evaluation datasets.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*):
            Encoder tokenizer. Loaded from the checkpoint when omitted.
        compute_metrics (`Callable`, *optional*):
            Receives predicted probabilities and labels. Nullable decisions include null in the first column. Padding
            added during evaluation has value `-100`.
        callbacks (`list[TrainerCallback]`, *optional*):
            Callbacks passed to [`~transformers.Trainer`].
        optimizers (`tuple`, *optional*, defaults to `(None, None)`):
            Optimizer and scheduler overrides.
        optimizer_cls_and_kwargs (`tuple`, *optional*):
            Optimizer class and keyword arguments.
        preprocess_logits_for_metrics (`Callable`, *optional*):
            Prediction preprocessing passed to [`~transformers.Trainer`].
        peft_config (`PeftConfig`, *optional*):
            PEFT configuration used to wrap the model. Configure any trainable head through `modules_to_save`.
    """

    loss_is_scaled_for_ga = True
    _tag_names = ["trl", "decision"]
    _name = "Decision"

    def __init__(
        self,
        model: PreTrainedModel | str,
        args: DecisionConfig | None = None,
        data_collator: Callable | None = None,
        train_dataset: Dataset | IterableDataset | None = None,
        eval_dataset: Dataset | IterableDataset | dict[str, Dataset | IterableDataset] | None = None,
        processing_class: PreTrainedTokenizerBase | None = None,
        compute_metrics: Callable | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple = (None, None),
        optimizer_cls_and_kwargs: tuple | None = None,
        preprocess_logits_for_metrics: Callable | None = None,
        peft_config: Any | None = None,
    ):
        if args is None:
            args = DecisionConfig(output_dir="Decision")
        if train_dataset is None:
            raise ValueError("`train_dataset` is required")
        set_seed(args.seed)
        model_id = (
            model
            if isinstance(model, str)
            else model.config._name_or_path or model.config.get_text_config()._name_or_path
        )
        if isinstance(model, str):
            model = AutoModelForDecision.from_pretrained(model, **(args.model_init_kwargs or {}))
        if model.is_distributed_loading_by_transformers and "ep_dispatch_experts" in model.ep_plan.values():
            raise ValueError(
                "Token-dispatch expert parallelism is not supported by Trainer; use masked expert parallelism"
            )
        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(model_id)
        if processing_class.pad_token_id is None:
            processing_class.pad_token = processing_class.eos_token
        if args.n_gpu > 1:
            raise ValueError(
                "Use accelerate launch or torchrun for multi-GPU decision training instead of DataParallel"
            )
        if peft_config is not None:
            if not is_peft_available():
                raise ImportError("Install `trl[peft]` to use peft_config")
            model = get_peft_model(model, peft_config)
        text_config = model.config.get_text_config()
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            optimizer_cls_and_kwargs=optimizer_cls_and_kwargs,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        self.model_accepts_loss_kwargs = False
        self._cp_group = None
        parallelism_config = self.accelerator.parallelism_config
        if parallelism_config is not None and parallelism_config.cp_enabled:
            if parallelism_config.cp_backend != "torch":
                raise ValueError("Decision context parallelism requires the torch backend with FSDP2")
            if not model.base_model.supports_context_parallel:
                raise ValueError(
                    "Context parallelism requires a full causal attention encoder without recurrent layers"
                )
            if text_config._attn_implementation != "sdpa":
                raise ValueError("Context parallelism requires an encoder loaded with attn_implementation='sdpa'")
            self._cp_group = self.accelerator.torch_device_mesh["cp"].get_group()
        if (
            Version(accelerate.__version__) >= Version("1.12.0")
            and parallelism_config is not None
            and parallelism_config.sp_enabled
        ):
            raise ValueError(
                "DecisionTrainer does not support DeepSpeed sequence parallelism; use torch context parallelism"
            )

    def _get_num_items_in_batch(self, batch_samples, device):
        count = torch.tensor(sum(batch["labels"].shape[0] for batch in batch_samples), device=device)
        return self.accelerator.gather(count).sum()

    def _prepare_context_parallel_inputs(self, model, inputs):
        if self._cp_group is None:
            return contextlib.nullcontext, inputs
        cp_size = self.accelerator.parallelism_config.cp_size
        prefixes = ("", "option_") if "option_input_ids" in inputs else ("",)
        length = max(inputs[f"{prefix}input_ids"].shape[1] for prefix in prefixes)
        padded_length = length + (-length) % (2 * cp_size)
        buffers = []
        for prefix in prefixes:
            if "option_input_ids" in inputs:
                inputs[f"{prefix}sequence_lengths"] = inputs[f"{prefix}attention_mask"].sum(-1)
            length = inputs[f"{prefix}input_ids"].shape[1]
            pad_length = padded_length - length
            inputs[f"{prefix}input_ids"] = F.pad(
                inputs[f"{prefix}input_ids"], (0, pad_length), value=self.processing_class.pad_token_id
            )
            inputs[f"{prefix}attention_mask"] = F.pad(inputs[f"{prefix}attention_mask"], (0, pad_length))
            inputs[f"{prefix}position_ids"] = (
                torch.arange(padded_length, device=inputs[f"{prefix}input_ids"].device)[None, :]
                .expand(inputs[f"{prefix}input_ids"].shape[0], -1)
                .clone()
            )
            buffers.extend(inputs[f"{prefix}{key}"] for key in ("input_ids", "attention_mask", "position_ids"))
        return partial(
            self.accelerator.maybe_context_parallel, buffers=buffers, buffer_seq_dims=[1] * len(buffers)
        ), inputs

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        mode = "train" if self.model.training else "eval"
        model_inputs = {key: value for key, value in inputs.items() if key != "is_choice"}
        # Sharded encoders must execute the same number of option chunks on every rank.
        if self.accelerator.num_processes > 1 and inputs.get("option_chunk_size") is not None:
            count = inputs["option_input_ids"].new_tensor([inputs["option_input_ids"].shape[0]])
            padding = self.accelerator.gather(count).max().item() - count.item()
            for key in ("option_input_ids", "option_attention_mask", "option_position_ids", "option_sequence_lengths"):
                if key in model_inputs:
                    value = model_inputs[key]
                    model_inputs[key] = torch.cat([value, value[:1].expand(padding, *value.shape[1:])])
        if self._cp_group is not None and "option_input_ids" in inputs:
            model_inputs["cp_group"] = self._cp_group
        loss, outputs = super().compute_loss(model, model_inputs, return_outputs=True)
        loss = loss.float()
        if num_items_in_batch is not None:
            loss = loss * inputs["labels"].shape[0] / num_items_in_batch * self.accelerator.num_processes
        with torch.no_grad():
            probabilities, mask = _get_decision_probabilities(outputs, inputs)
            labels = inputs["labels"]
            targets = F.one_hot(labels, probabilities.shape[-1]).float() if labels.ndim == 1 else labels.float()
            has_null = "allow_null" in inputs
            candidates = targets[:, 1:] if has_null else targets
            option_mask = mask[:, 1:] if has_null else mask
            log_q = outputs.logits.float().masked_fill(~option_mask, -torch.inf).log_softmax(-1)
            candidate_loss = -(candidates * log_q.masked_fill((~option_mask) | (candidates == 0), 0.0)).sum(-1)
            predicted = probabilities.argmax(-1)
            target = targets.argmax(-1)
            is_choice = inputs.get("is_choice", torch.ones_like(target, dtype=torch.bool))
            hard_choice = is_choice & (targets.max(-1).values == 1)
            present = hard_choice & (target != 0) if has_null else hard_choice
            ones = torch.ones_like(candidate_loss)
            metrics = {
                "candidate_loss": (candidate_loss, ones),
                "choice_accuracy": ((predicted == target) & hard_choice, hard_choice),
                "candidate_accuracy": ((outputs.logits.argmax(-1) + int(has_null) == target) & present, present),
                "brier": ((probabilities - targets).square().sum(-1), ones),
                "entropy": (-(probabilities * probabilities.clamp_min(1e-30).log()).sum(-1), ones),
            }
            if has_null:
                predicted_null = (predicted == 0) & hard_choice
                target_null = (target == 0) & hard_choice
                true_null = predicted_null & target_null
                metrics.update(
                    {
                        "null_precision": (true_null, predicted_null),
                        "null_recall": (true_null, target_null),
                        "null_rate": ((predicted == 0) & inputs["allow_null"], inputs["allow_null"]),
                        "null_brier": ((probabilities[:, 0] - targets[:, 0]).square(), inputs["allow_null"]),
                    }
                )
                if outputs.get("coverage_logits") is not None:
                    logits = outputs.coverage_logits.float()
                    coverage = candidates.sum(-1)
                    coverage_loss = coverage * F.softplus(-logits) + (1 - coverage) * F.softplus(logits)
                    metrics["coverage_loss"] = (coverage_loss * inputs["allow_null"], ones)
            if outputs.get("aux_loss") is not None:
                metrics["aux_loss"] = (outputs.aux_loss.detach().expand_as(ones), ones)
            values = torch.stack(
                [
                    torch.stack([numerator.float(), denominator.float()], -1)
                    for numerator, denominator in metrics.values()
                ],
                1,
            )
            values = self._gather_for_metrics(values)
            for key, totals in zip(metrics, values.sum(0).tolist(), strict=True):
                self._metrics[mode][key].append(totals)
        return (loss, outputs) if return_outputs else loss

    def _gather_for_metrics(self, values):
        replicas = self.get_tp_size() * self.get_cp_size()
        if replicas == 1:
            if self.model.training:
                return self.accelerator.gather(values)
            return self.accelerator.gather_for_metrics(values)

        def remove_replicas(tensor):
            # Keep one copy per data-parallel rank before removing the final batch's padding.
            tensor = tensor.reshape(self.accelerator.num_processes, -1, *tensor.shape[1:])[::replicas].flatten(0, 1)
            if (
                not self.model.training
                and self.accelerator.gradient_state.end_of_dataloader
                and self.accelerator.gradient_state.remainder > 0
            ):
                tensor = tensor[: self.accelerator.gradient_state.remainder]
            return tensor

        return recursively_apply(remove_replicas, self.accelerator.gather(values))

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        if self.get_tp_size() * self.get_cp_size() > 1:
            self.gather_function = self._gather_for_metrics
        inputs = self._prepare_inputs(inputs)
        cp_context, inputs = self._prepare_context_parallel_inputs(model, inputs)
        with torch.no_grad(), self.compute_loss_context_manager(), cp_context():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
        if prediction_loss_only:
            return loss.detach(), None, None
        probabilities, mask = _get_decision_probabilities(outputs, inputs)
        labels = inputs["labels"]
        if labels.ndim == 2:
            labels = labels.masked_fill(~mask, -100)
        return loss.detach(), probabilities.masked_fill(~mask, -100).detach(), labels

    def log(self, logs, start_time=None):
        mode = "train" if self.model.training else "eval"
        metrics = {
            key: sum(numerator for numerator, _ in values) / max(sum(denominator for _, denominator in values), 1)
            for key, values in self._metrics[mode].items()
        }
        if mode == "eval":
            metrics = {f"eval_{key}": value for key, value in metrics.items()}
        logs.update(metrics)
        super().log(logs, start_time)
        self._metrics[mode].clear()

    def _save_checkpoint(self, model, trial):
        model_name = (
            self.args.hub_model_id.split("/")[-1] if self.args.hub_model_id else Path(self.args.output_dir).name
        )
        self.create_model_card(model_name=model_name)
        super()._save_checkpoint(model, trial)
