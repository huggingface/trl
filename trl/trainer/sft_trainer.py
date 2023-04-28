# Copyright 2023 The HuggingFace Team. All rights reserved.
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
import os
import warnings
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from accelerate import Accelerator
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollator,
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction

from ..import_utils import is_peft_available
from .utils import ConstantLengthDataset


if is_peft_available():
    from peft import PeftConfig, PeftModel, get_peft_model, prepare_model_for_int8_training


class PeftSavingCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        kwargs["model"].save_pretrained(checkpoint_path)

        if "pytorch_model.bin" in os.listdir(checkpoint_path):
            os.remove(os.path.join(checkpoint_path, "pytorch_model.bin"))


class SFTTrainer(Trainer):
    r"""
    Class definition of the Supervised Finetuning Trainer (SFT Trainer).
    This class is a wrapper around the `transformers.Trainer` class and inherits all of its attributes and methods.
    The trainer takes care of properly initializing the PeftModel in case a user passes a `PeftConfig` object.

    Args:
        model (Union[`transformers.PreTrainedModel`, `nn.Module`, `str`]):
            The model to train, can be a `PreTrainedModel`, a `torch.nn.Module` or a string with the model name to
            load from cache or download. The model can be also converted to a `PeftModel` if a `PeftConfig` object is
            passed to the `peft_config` argument.
        args (Optional[`transformers.TrainingArguments`]):
            The arguments to tweak for training. Please refer to the official documentation of `transformers.TrainingArguments`
            for more information.
        data_collator (Optional[`transformers.DataCollator`]):
            The data collator to use for training.
        train_dataset (Optional[`datasets.Dataset`]):
            The dataset to use for training. We recommend users to use `trl.trainer.ConstantLengthDataset` to create their dataset.
        eval_dataset (Optional[Union[`datasets.Dataset`, Dict[`str`, `datasets.Dataset`]]]):
            The dataset to use for evaluation. We recommend users to use `trl.trainer.ConstantLengthDataset` to create their dataset.
        tokenizer (Optional[`transformers.PreTrainedTokenizer`]):
            The tokenizer to use for training. If not specified, the tokenizer associated to the model will be used.
        model_init (`Callable[[], transformers.PreTrainedModel]`):
                The model initializer to use for training. If None is specified, the default model initializer will be used.
        compute_metrics (`Callable[[transformers.EvalPrediction], Dict]`, *optional* defaults to `compute_accuracy`):
            The metrics to use for evaluation. If no metrics are specified, the default metric (`compute_accuracy`) will be used.
        callbacks (`List[transformers.TrainerCallback]`):
            The callbacks to use for training.
        optimizers (`Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`):
            The optimizer and scheduler to use for training.
        preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`):
            The function to use to preprocess the logits before computing the metrics.
        peft_config (`Optional[PeftConfig]`):
            The PeftConfig object to use to initialize the PeftModel.
        prepare_in_int8_kwargs (`Dict`):
            The arguments to pass to the `prepare_model_for_int8_training` function, in case a `PeftConfig` has been passed and in case
            users want to train their models in 8bit mode.
        dataset_text_field (`Optional[str]`):
            The name of the text field of the dataset, in case this is passed by a user, the trainer will automatically create a
            `ConstantLengthDataset` based on the `dataset_text_field` argument.
        formatting_func (`Optional[Callable]`):
            The formatting function to be used for creating the `ConstantLengthDataset`.
        max_seq_length (`Optional[int]`):
            The maximum sequence length to use for the `ConstantLengthDataset` and for automaticallty creating the Dataset. Defaults to `512`.
        infinite (`Optional[bool]`):
            Whether to use an infinite dataset or not. Defaults to `False`.
        num_of_sequences (`Optional[int]`):
            The number of sequences to use for the `ConstantLengthDataset`. Defaults to `1024`.
        chars_per_token (`Optional[float]`):
            The number of characters per token to use for the `ConstantLengthDataset`. Defaults to `3.6`.
        packing (`Optional[bool]`):
            Used only in case `dataset_text_field` is passed. This argument is used by the `ConstantLengthDataset` to pack the sequences
            of the dataset.
    """

    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module, str] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        peft_config: Optional[Dict] = None,
        dataset_text_field: Optional[str] = None,
        packing: Optional[bool] = True,
        formatting_func: Optional[Callable] = None,
        max_seq_length: Optional[int] = None,
        infinite: Optional[bool] = False,
        num_of_sequences: Optional[int] = 1024,
        chars_per_token: Optional[float] = 3.6,
        prepare_in_int8_kwargs: Optional[Dict] = {},
        **pretrained_kwargs,
    ):
        if isinstance(model, str):
            warnings.warn(
                "You passed a model_id and a `PeftConfig` to the SFTTrainer. This will automatically create an "
                "`AutoModelForCausalLM` or a `PeftModel` for you."
            )

        if is_peft_available() and peft_config is not None:
            if not isinstance(peft_config, PeftConfig):
                raise ValueError(
                    "If you want to use the PeftModel, you need to pass a PeftConfig object to the SFTTrainer."
                    f" and you passed a {type(peft_config)}."
                )

            if not isinstance(model, PeftModel):
                if pretrained_kwargs.get("load_in_8bit", False) and "device_map" not in pretrained_kwargs:
                    device_map = {"": Accelerator().process_index}
                    pretrained_kwargs["device_map"] = device_map

                model = AutoModelForCausalLM.from_pretrained(
                    model,
                    **pretrained_kwargs,
                )

                if getattr(model, "is_loaded_in_8bit", False):
                    model = prepare_model_for_int8_training(model, **prepare_in_int8_kwargs)

                model = get_peft_model(model, peft_config)

            if callbacks is None:
                callbacks = [PeftSavingCallback]
        elif not isinstance(model, PreTrainedModel):
            model = AutoModelForCausalLM.from_pretrained(model, **pretrained_kwargs)

        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)
            if getattr(tokenizer, "pad_token", None) is None:
                tokenizer.pad_token = tokenizer.eos_token

        if max_seq_length is None:
            max_seq_length = tokenizer.model_max_length

        # check if torch dataset / dataloader and do nothing
        if train_dataset is not None and (
            isinstance(
                train_dataset,
                (torch.utils.data.IterableDataset, torch.utils.data.Dataset),
            )
        ):
            is_already_dataset = True
        else:
            is_already_dataset = False

        if not packing:
            if dataset_text_field is None:
                raise ValueError(
                    "You passed `packing=False` to the SFTTrainer, but you didn't pass a `dataset_text_field` argument."
                )

            if data_collator is None:
                data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

            if train_dataset is not None:
                train_dataset = self._prepare_non_packed_dataloader(
                    tokenizer, train_dataset, dataset_text_field, data_collator, max_seq_length
                )
            if eval_dataset is not None:
                eval_dataset = self._prepare_non_packed_dataloader(
                    tokenizer, eval_dataset, dataset_text_field, data_collator, max_seq_length
                )

            is_already_dataset = True

        if not is_already_dataset and dataset_text_field is not None:
            if tokenizer is None:
                raise ValueError(
                    "You need to pass a tokenizer when using the SFT Trainer when passing a `dataset_text_field`."
                )

            if train_dataset is not None:
                train_dataset = ConstantLengthDataset(
                    tokenizer,
                    train_dataset[dataset_text_field],
                    formatting_func=formatting_func,
                    seq_length=max_seq_length,
                    infinite=infinite,
                    num_of_sequences=num_of_sequences,
                    chars_per_token=chars_per_token,
                    eos_token_id=tokenizer.eos_token_id,
                )
            if eval_dataset is not None:
                eval_dataset = ConstantLengthDataset(
                    tokenizer,
                    eval_dataset[dataset_text_field],
                    formatting_func=formatting_func,
                    seq_length=max_seq_length,
                    infinite=infinite,
                    num_of_sequences=num_of_sequences,
                    chars_per_token=chars_per_token,
                    eos_token_id=tokenizer.eos_token_id,
                )
        elif not is_already_dataset and dataset_text_field is None:
            raise ValueError(
                "You need to pass a `dataset_text_field` argument to the SFTTrainer if you want to use the `ConstantLengthDataset`."
            )

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

    def _prepare_non_packed_dataloader(self, tokenizer, dataset, dataset_text_field, data_collator, max_seq_len):
        # tokenize the dataset
        dataset = dataset.map(
            lambda x: data_collator(
                [tokenizer(x[dataset_text_field], padding="max_length", truncation=True, max_length=max_seq_len)]
            ),
            batched=False,
        )

        # convert to torch dataset
        dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

        # squueze unneeded args
        dataset = dataset.map(lambda x: {"input_ids": x["input_ids"].squeeze()})
        dataset = dataset.map(lambda x: {"attention_mask": x["attention_mask"].squeeze()})
        dataset = dataset.map(lambda x: {"labels": x["labels"].squeeze()})

        return dataset
