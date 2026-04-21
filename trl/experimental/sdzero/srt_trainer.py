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

import textwrap
from collections.abc import Callable
from typing import Any

import torch
from datasets import Dataset, IterableDataset
from torch import nn
from transformers import (
    AutoProcessor,
    DataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainerCallback,
)
from transformers.trainer_utils import EvalPrediction
from transformers.utils import is_peft_available

from ...trainer.sft_trainer import SFTTrainer, get_dataset_column_names
from ...trainer.utils import get_config_model_id
from .srt_config import SRTConfig


if is_peft_available():
    from peft import PeftConfig


class SRTTrainer(SFTTrainer):
    """
    Trainer for Self-Revision Training (SRT) from [Self-Distillation Zero](https://huggingface.co/papers/2604.12002).

    SRT trains a model with a joint objective combining two complementary loss terms. Each dataset row is
    expanded into two training records that share the same token sequence but differ in which tokens are
    supervised:

    - **Revision record**: loss computed only on the revised answer, conditioned on the full context
      (problem, initial answer, control prompt) as input.
    - **Generation record**: loss computed on the entire assistant turn — initial answer, control prompt,
      and revised answer — conditioned on only the problem as input.

    The dataset must contain four string columns: `problem`, `y_init`, `control_prompt`, and `y_revised`.

    Args:
        model (`str` or [`~transformers.PreTrainedModel`] or [`~peft.PeftModel`]):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or a
              path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
              using `<ModelArchitecture>.from_pretrained` with the keyword arguments in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        args ([`SRTConfig`], *optional*):
            Configuration for this trainer. If `None`, a default configuration is used.
        data_collator ([`~transformers.DataCollator`], *optional*):
            Function to use to form a batch from a list of elements of the processed dataset.
        train_dataset ([`~datasets.Dataset`]):
            Dataset for training. Must contain columns `problem`, `y_init`, `control_prompt`, and `y_revised`.
        eval_dataset ([`~datasets.Dataset`] or `dict[str, Dataset]`, *optional*):
            Dataset for evaluation. Must meet the same column requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`] or [`~transformers.ProcessorMixin`], *optional*):
            Processing class used to tokenize the data. If `None`, loaded from the model name. A padding token
            must be set; if absent, `eos_token` is used.
        compute_loss_func (`Callable`, *optional*):
            Custom loss function. See [`SFTTrainer`] for details.
        compute_metrics (`Callable[[EvalPrediction], dict]`, *optional*):
            Function to compute metrics at evaluation.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*):
            List of callbacks to customize the training loop.
        optimizers (`tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None]`, *optional*, defaults to `(None, None)`):
            Tuple containing the optimizer and scheduler to use.
        optimizer_cls_and_kwargs (`tuple[Type[torch.optim.Optimizer], Dict[str, Any]]`, *optional*):
            Tuple containing the optimizer class and keyword arguments. Overrides `optim` and `optim_args` in `args`.
        preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`, *optional*):
            Function to preprocess logits before caching them at each evaluation step.
        peft_config ([`~peft.PeftConfig`], *optional*):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    _tag_names = ["trl", "sdzero", "srt"]
    _name = "SRT"
    config_cls = SRTConfig
    # docstyle-ignore
    _paper = {
        "title": "Self-Distillation Zero: Self-Revision Turns Binary Rewards into Dense Supervision",
        "id": "2604.12002",
        "citation": textwrap.dedent("""\
            @article{sdzero2026,
                title        = {{Self-Distillation Zero: Self-Revision Turns Binary Rewards into Dense Supervision}},
                year         = 2026,
                eprint       = {arXiv:2604.12002}
            }"""),
    }

    def __init__(
        self,
        model: str | PreTrainedModel | nn.Module,
        args: SRTConfig | None = None,
        data_collator: DataCollator | None = None,
        train_dataset: Dataset | IterableDataset | None = None,
        eval_dataset: Dataset | IterableDataset | dict[str, Dataset | IterableDataset] | None = None,
        processing_class: PreTrainedTokenizerBase | ProcessorMixin | None = None,
        compute_loss_func: Callable | None = None,
        compute_metrics: Callable[[EvalPrediction], dict] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None] = (None, None),
        optimizer_cls_and_kwargs: tuple[type[torch.optim.Optimizer], dict[str, Any]] | None = None,
        preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        peft_config: PeftConfig | None = None,
    ):
        if isinstance(train_dataset, IterableDataset) or isinstance(eval_dataset, IterableDataset):
            raise NotImplementedError("Iterable datasets are not supported by `SRTTrainer`.")
        if args is None:
            model_name = model if isinstance(model, str) else get_config_model_id(model.config)
            model_name = model_name.split("/")[-1]
            args = SRTConfig(f"{model_name}-SRT")

        if processing_class is None:
            model_id = model if isinstance(model, str) else get_config_model_id(model.config)
            processing_class = AutoProcessor.from_pretrained(model_id)
        if processing_class.pad_token is None:
            processing_class.pad_token = processing_class.eos_token

        if train_dataset is None:
            raise ValueError("`train_dataset` is required for `SRTTrainer`.")

        train_dataset = self._expand_srt_dataset(train_dataset, processing_class, args)
        if eval_dataset is not None:
            if isinstance(eval_dataset, dict):
                eval_dataset = {
                    name: self._expand_srt_dataset(ds, processing_class, args) for name, ds in eval_dataset.items()
                }
            else:
                eval_dataset = self._expand_srt_dataset(eval_dataset, processing_class, args)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_loss_func=compute_loss_func,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            optimizer_cls_and_kwargs=optimizer_cls_and_kwargs,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            peft_config=peft_config,
        )

    @staticmethod
    def _expand_srt_dataset(
        dataset: Dataset,
        processing_class: PreTrainedTokenizerBase | ProcessorMixin,
        args: SRTConfig,
    ) -> Dataset:
        """Expand each dataset row into tokenized revision and generation training records."""

        tokenizer = processing_class.tokenizer if isinstance(processing_class, ProcessorMixin) else processing_class

        columns = get_dataset_column_names(dataset)
        required = ["problem", "y_init", "control_prompt", "y_revised"]
        missing = [c for c in required if c not in columns]
        if missing:
            raise ValueError(f"SRT dataset is missing required columns: {missing}. Present columns: {columns}.")

        eos_token = tokenizer.eos_token
        if eos_token is None:
            raise ValueError("Tokenizer must have an `eos_token` to build the assistant-turn boundary.")

        def _row_to_records(example: dict[str, Any]) -> dict[str, list]:
            x = example["problem"]
            y_init = example["y_init"]
            p_r = example["control_prompt"]
            y_revised = example["y_revised"]

            prefix_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": x}],
                tokenize=False,
                add_generation_prompt=True,
            )
            pre_revised_text = y_init + args.separator + p_r + args.separator
            revised_text = y_revised

            prefix_ids = tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
            pre_revised_ids = tokenizer(pre_revised_text, add_special_tokens=False)["input_ids"]
            revised_ids = tokenizer(revised_text, add_special_tokens=False)["input_ids"]
            eos_ids = [tokenizer.eos_token_id]

            input_ids = prefix_ids + pre_revised_ids + revised_ids + eos_ids

            prefix_len = len(prefix_ids)
            pre_revised_len = len(pre_revised_ids)
            revised_len = len(revised_ids) + len(eos_ids)

            # 1 = token contributes to loss, 0 = ignored
            revision_mask = [0] * (prefix_len + pre_revised_len) + [1] * revised_len
            generation_mask = [0] * prefix_len + [1] * (pre_revised_len + revised_len)

            input_ids_list, completion_masks = [], []
            if args.include_revision_loss:
                input_ids_list.append(input_ids)
                completion_masks.append(revision_mask)
            if args.include_generation_loss:
                input_ids_list.append(input_ids)
                completion_masks.append(generation_mask)
            return {"input_ids": input_ids_list, "completion_mask": completion_masks}

        expanded = dataset.map(
            _row_to_records,
            batched=False,
            remove_columns=dataset.column_names,
        )
        return Dataset.from_dict(
            {
                "input_ids": [input_ids for row in expanded["input_ids"] for input_ids in row],
                "completion_mask": [mask for row in expanded["completion_mask"] for mask in row],
            }
        )
