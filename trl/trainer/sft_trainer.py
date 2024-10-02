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

import datasets
import torch
import torch.nn as nn
from accelerate.state import PartialState
from datasets import Dataset, IterableDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollator,
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    is_wandb_available,
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction
from transformers.utils import is_peft_available

from trl import maybe_apply_chat_template
from trl.data_utils import pack_dataset

from ..import_utils import is_liger_kernel_available
from .sft_config import SFTConfig
from .utils import generate_model_card


if is_peft_available():
    from peft import PeftConfig

if is_liger_kernel_available():
    from liger_kernel.transformers import AutoLigerKernelForCausalLM

if is_wandb_available():
    import wandb


class SFTTrainer(Trainer):
    _tag_names = ["trl", "sft"]

    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        args: Optional[SFTConfig] = None,
        data_collator: Optional[DataCollator] = None,  # type: ignore
        train_dataset: Optional[Union[Dataset, IterableDataset, "datasets.Dataset"]] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset], "datasets.Dataset"]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        peft_config: Optional["PeftConfig"] = None,
        formatting_func: Optional[Callable] = None,
    ):
        # 1. Handle the model
        if args.model_init_kwargs is not None and not isinstance(model, str):
            warnings.warn(
                "You passed model_init_kwargs to the `SFTConfig`, but your model is already instantiated. "
                "The `model_init_kwargs` will be ignored."
            )

        if isinstance(model, str):
            # `model`` is a model id. We need to instantiate the model.
            model_init_kwargs = args.model_init_kwargs or {}

            # Get the torch dtype from the model_init_kwargs
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if torch_dtype is not None:
                # Convert to `torch.dtype` if an str is passed
                if isinstance(torch_dtype, str) and torch_dtype != "auto":
                    torch_dtype = getattr(torch, torch_dtype)
                if torch_dtype != "auto" and not isinstance(torch_dtype, torch.dtype):
                    raise ValueError(
                        "Invalid `torch_dtype` passed to the `SFTConfig`. Expected a string with either `torch.dtype` "
                        f"or 'auto', but got {torch_dtype}."
                    )
                model_init_kwargs["torch_dtype"] = torch_dtype

            # Create the model
            if args.use_liger:
                model = AutoLigerKernelForCausalLM.from_pretrained(model, **model_init_kwargs)
            else:
                model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)

        # 2. Convert the model to a PeftModel if a PeftConfig is passed
        if peft_config is not None:
            if not is_peft_available():
                raise ImportError("To use the PeftModel, you need to install the `peft` library.")

            # PEFT logif here

        # 3. Handle the tokenizer
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)
            tokenizer.pad_token = tokenizer.eos_token  # required for padding when collating data

        # 4. Handle the dataset
        with PartialState().local_main_process_first():
            # Apply the chat template if needed
            train_dataset = train_dataset.map(
                maybe_apply_chat_template,
                fn_kwargs={"tokenizer": tokenizer},
                num_proc=args.dataset_num_proc,
                desc="Applying chat template to train dataset",
            )
            if eval_dataset is not None:
                eval_dataset = eval_dataset.map(
                    maybe_apply_chat_template,
                    fn_kwargs={"tokenizer": tokenizer},
                    num_proc=args.dataset_num_proc,
                    desc="Applying chat template to eval dataset",
                )

            # Tokenize and prepare the training datasets
            max_length = args.max_seq_length or min(tokenizer.model_max_length, 1024)
            fn_kwargs = {
                "tokenizer": tokenizer,
                "dataset_text_field": args.dataset_text_field,
                "max_length": max_length if not args.packing else None,
            }
            train_dataset = train_dataset.map(
                self._tokenize, fn_kwargs=fn_kwargs, num_proc=args.dataset_num_proc, desc="Tokenizing train dataset"
            )
            if eval_dataset is not None:
                eval_dataset = eval_dataset.map(
                    self._tokenize, fn_kwargs=fn_kwargs, num_proc=args.dataset_num_proc, desc="Tokenizing eval dataset"
                )

            # Pack the datasets
            if args.packing:
                train_dataset = train_dataset.select_columns(["input_ids"])
                train_dataset = pack_dataset(
                    train_dataset,
                    max_length,
                    num_proc=args.dataset_num_proc,
                    desc="Packing train dataset",
                )
                if eval_dataset is not None:
                    eval_dataset = eval_dataset.select_columns(["input_ids"])
                    eval_dataset = pack_dataset(
                        eval_dataset,
                        max_length,
                        num_proc=args.dataset_num_proc,
                        desc="Packing eval dataset",
                    )

        # 5. Handle the data collator
        if data_collator is None:
            data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

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

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

    @staticmethod
    def _tokenize(
        examples: Dict[str, List[str]],
        tokenizer: PreTrainedTokenizerBase,
        dataset_text_field: str,
        max_length: Optional[int],
    ) -> Dict[str, List[int]]:
        return tokenizer(examples[dataset_text_field], max_length=max_length, truncation=max_length is not None)

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, List[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str`, *optional*, defaults to `None`):
                The name of the model.
            dataset_name (`str`, *optional*, defaults to `None`):
                The name of the dataset used for training.
            tags (`str`, `List[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            trainer_name="SFT",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))
