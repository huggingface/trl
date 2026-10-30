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
import transformers
from datasets import Dataset, IterableDataset
from packaging.version import Version
from transformers import AutoTokenizer, PreTrainedTokenizerBase, TrainerCallback, set_seed
from transformers.utils import is_peft_available

from ...trainer.base_trainer import _BaseTrainer
from .data import DataCollatorForJeva
from .jeva_config import JevaConfig
from .modeling_jeva import JevaModel


class JevaTrainer(_BaseTrainer):
    """
    Trainer for [Jeva: System One Models from First Principles](https://huggingface.co/papers/2610.18773).

    Trains candidate distributions and a separate coverage estimate from typed decisions. The shared encoder never
    constructs language-model vocabulary logits. Optimization, distributed wrapping, checkpointing, and callbacks use
    [`~transformers.Trainer`].

    Args:
        model ([`experimental.jeva.JevaModel`] or `str`):
            Model to train, or identifier of a saved JevaModel checkpoint.
        args ([`experimental.jeva.JevaConfig`], *optional*):
            Training configuration.
        data_collator (`Callable`, *optional*):
            Decision collator. Defaults to `DataCollatorForJeva`.
        train_dataset (`Dataset` or `IterableDataset`, *optional*):
            Dataset with `state`, `question`, `options`, and one of `target_index`, `target_probs`, `p_true`, or
            `score`. `decision_type` defaults to `"choice"`.
        eval_dataset (`Dataset` or `IterableDataset` or `dict`, *optional*):
            Evaluation dataset or named evaluation datasets.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*):
            Encoder tokenizer. Loaded from the checkpoint when omitted.
        compute_metrics (`Callable`, *optional*):
            Receives predicted probabilities and target distributions, each with null in the first column. Padding
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
            Adapter configuration applied to the shared encoder for `mode="lora"`.
    """

    loss_is_scaled_for_ga = True
    _tag_names = ["trl", "jeva"]
    _name = "Jeva"
    _paper = {"title": "Jeva: System One Models from First Principles", "id": "2610.18773"}

    def __init__(
        self,
        model: JevaModel | str,
        args: JevaConfig | None = None,
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
            args = JevaConfig(output_dir="Jeva")
        if train_dataset is None:
            raise ValueError("`train_dataset` is required")
        set_seed(args.seed)
        model_id = model if isinstance(model, str) else model.config.backbone_name_or_path
        if isinstance(model, str):
            model = JevaModel.from_pretrained(model, **(args.model_init_kwargs or {}))
        if Version(transformers.__version__) >= Version("5.21.0.dev0"):
            if model.is_distributed_loading_by_transformers and "ep_dispatch_experts" in model.ep_plan.values():
                raise ValueError(
                    "Token-dispatch expert parallelism is not supported by Trainer; use masked expert parallelism"
                )
        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(model_id)
        if processing_class.pad_token_id is None:
            processing_class.pad_token = processing_class.eos_token
        if args.n_gpu > 1:
            raise ValueError("Use accelerate launch or torchrun for multi-GPU Jeva training instead of DataParallel")
        if peft_config is not None:
            if not is_peft_available():
                raise ImportError("Install `trl[peft]` to use peft_config")
            if args.mode != "lora" or model.config.adapter_config is not None:
                raise ValueError("peft_config requires mode='lora' and an encoder without an existing adapter")
            from peft import get_peft_model

            model.backbone = get_peft_model(model.backbone, peft_config)
            model.config.adapter_config = peft_config.to_dict()
        model.backbone.requires_grad_(args.mode == "full")
        if args.mode == "lora":
            if model.config.adapter_config is None:
                raise ValueError("mode='lora' requires an existing adapter or peft_config")
            model.backbone.set_adapter("default")
            model.backbone.enable_input_require_grads()
        model.freeze_backbone = args.mode == "head"
        model.context_proj.requires_grad_(True)
        model.option_proj.requires_grad_(True)
        model.core.requires_grad_(True)

        text_config = model.config.get_text_config()
        text_config_dict = text_config.to_dict()
        coef = args.router_aux_loss_coef
        self.router_aux_loss_coef = text_config_dict.get("router_aux_loss_coef", 0.0) if coef is None else coef
        if self.router_aux_loss_coef and "output_router_logits" not in text_config_dict:
            raise ValueError("router_aux_loss_coef requires an MoE encoder that returns router logits")
        model.router_aux_loss_coef = 0.0 if model.freeze_backbone else self.router_aux_loss_coef
        self.aux_loss_enabled = model.router_aux_loss_coef != 0.0
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator
            or DataCollatorForJeva(processing_class, args.max_state_length, args.max_option_length, args.truncation),
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
            if Version(transformers.__version__) < Version("5.22.1"):
                raise ValueError("Jeva context parallelism requires transformers>=5.22.1")
            if parallelism_config.cp_backend != "torch":
                raise ValueError("Jeva context parallelism requires the torch backend with FSDP2")
            if not model.backbone.supports_context_parallel:
                raise ValueError(
                    "Context parallelism requires a full causal attention encoder without recurrent layers"
                )
            if text_config._attn_implementation != "sdpa":
                raise ValueError("Context parallelism requires an encoder loaded with attn_implementation='sdpa'")
            if compute_metrics is not None:
                raise ValueError("Context parallelism supports eval_loss only; omit compute_metrics")
            self._cp_group = self.accelerator.torch_device_mesh["cp"].get_group()
        if (
            Version(accelerate.__version__) >= Version("1.12.0")
            and parallelism_config is not None
            and parallelism_config.sp_enabled
        ):
            raise ValueError("Jeva does not support DeepSpeed sequence parallelism; use torch context parallelism")

    def _get_num_items_in_batch(self, batch_samples, device):
        count = torch.tensor(sum(batch["labels"].shape[0] for batch in batch_samples), device=device)
        return self.accelerator.gather(count).sum()

    def _prepare_context_parallel_inputs(self, model, inputs):
        if self._cp_group is None:
            return contextlib.nullcontext, inputs
        cp_size = self.accelerator.parallelism_config.cp_size
        length = max(inputs[name]["input_ids"].shape[1] for name in ("state_inputs", "option_inputs"))
        padded_length = length + (-length) % (2 * cp_size)
        buffers = []
        for name in ("state_inputs", "option_inputs"):
            values = dict(inputs[name])
            values["sequence_lengths"] = values["attention_mask"].sum(-1)
            length = values["input_ids"].shape[1]
            pad_length = padded_length - length
            values["input_ids"] = F.pad(values["input_ids"], (0, pad_length), value=self.processing_class.pad_token_id)
            values["attention_mask"] = F.pad(values["attention_mask"], (0, pad_length))
            values["position_ids"] = (
                torch.arange(length + pad_length, device=values["input_ids"].device)[None, :]
                .expand(values["input_ids"].shape[0], -1)
                .clone()
            )
            inputs[name] = values
            buffers.extend(values[key] for key in ("input_ids", "attention_mask", "position_ids"))
        return partial(
            self.accelerator.maybe_context_parallel, buffers=buffers, buffer_seq_dims=[1] * len(buffers)
        ), inputs

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        option_inputs = inputs["option_inputs"]
        # Sharded encoders must execute the same number of option chunks on every rank.
        if self.accelerator.num_processes > 1 and self.args.option_chunk_size is not None:
            count = option_inputs["input_ids"].new_tensor([option_inputs["input_ids"].shape[0]])
            padding = self.accelerator.gather(count).max().item() - count.item()
            option_inputs = {
                key: torch.cat([value, value[:1].expand(padding, *value.shape[1:])])
                for key, value in option_inputs.items()
            }
        outputs = model(
            state_inputs=inputs["state_inputs"],
            option_inputs=option_inputs,
            option_counts=inputs["option_counts"],
            option_mask=inputs["option_mask"],
            option_chunk_size=self.args.option_chunk_size,
            cp_group=self._cp_group,
        )
        candidates = inputs["labels"][:, 1:].float()
        log_q = outputs.log_q.float().masked_fill(~inputs["option_mask"], 0.0)
        candidate_loss = -(candidates * log_q).sum(-1)
        coverage_loss = F.binary_cross_entropy_with_logits(
            outputs.coverage_logit.float(), candidates.sum(-1), reduction="none"
        )
        per_example_loss = candidate_loss + coverage_loss * inputs["allow_null"]
        count = len(per_example_loss) if num_items_in_batch is None else num_items_in_batch
        loss = per_example_loss.sum() / count
        if num_items_in_batch is not None:
            loss = loss * self.accelerator.num_processes
        if self.aux_loss_enabled:
            steps = self.current_gradient_accumulation_steps if num_items_in_batch is not None else 1
            loss = loss + self.router_aux_loss_coef * outputs.aux_loss / steps
            mode = "train" if self.model.training else "eval"
            self._metrics[mode]["aux_loss"].append(self.accelerator.gather(outputs.aux_loss.detach()).mean().item())
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        if self._cp_group is not None and not prediction_loss_only:
            raise ValueError("Prediction gathering is not supported with context parallelism; use evaluate()")
        inputs = self._prepare_inputs(inputs)
        cp_context, inputs = self._prepare_context_parallel_inputs(model, inputs)
        with torch.no_grad(), self.compute_loss_context_manager(), cp_context():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
        if prediction_loss_only:
            return loss.detach(), None, None
        coverage = torch.where(inputs["allow_null"], outputs.coverage_logit.sigmoid(), 1.0)
        probabilities = torch.cat([(1 - coverage)[:, None], coverage[:, None] * outputs.log_q.exp()], dim=-1)
        mask = F.pad(inputs["option_mask"], (1, 0), value=True)
        return (
            loss.detach(),
            probabilities.masked_fill(~mask, -100).detach(),
            inputs["labels"].masked_fill(~mask, -100),
        )

    def log(self, logs, start_time=None):
        mode = "train" if self.model.training else "eval"
        metrics = {key: sum(value) / len(value) for key, value in self._metrics[mode].items()}
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
