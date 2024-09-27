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
import inspect
import warnings
from collections import defaultdict
from dataclasses import FrozenInstanceError, replace
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
import torch.nn as nn
from accelerate import PartialState
from accelerate.utils import gather_object
from datasets import Dataset
from transformers import DataCollator, PreTrainedModel, PreTrainedTokenizerBase, Trainer, DataCollatorForTokenClassification
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_pt_utils import nested_detach
from transformers.trainer_utils import EvalPrediction
from transformers.utils import is_peft_available

from ..data_utils import maybe_apply_chat_template
from .prm_config import PRMConfig
from .utils import (
    compute_accuracy,
    trl_sanitze_kwargs_for_tagging,
)


if is_peft_available():
    from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training


def _tokenize(batch: Dict[str, List[Any]], tokenizer: "PreTrainedTokenizerBase", max_length: int) -> Dict[str, List[Any]]:
    """Tokenize a batch from a PRM modeling dataset."""
    new_examples = {
        "input_ids": [],
        "attention_mask": [],
        "labels": []
    }
    
    post_step_tokens = tokenizer.encode("\n", add_special_tokens=False)
    
    for prompt, steps, labels in zip(batch["prompt"], batch["completion_steps"], batch["labels"]):
        if len(steps)!=len(labels):
            raise NotImplementedError(
                "prm_trainer does not support training with unlabeled steps."
            )
        
        if getattr(tokenizer, 'add_bos_token', False) and tokenizer.bos_token_id is not None:
            input_ids.append(tokenizer.bos_token_id)
            labels.append(-100)
            
        input_ids = input_ids.append(tokenizer.encode(prompt, add_special_tokens=False))
        labels = labels.append([-100]*len(input_ids))
        
        for step, label in zip(steps, labels):
            tokenized_step = tokenizer.encode(step, add_special_tokens=False)
            step_labels = [-100] * len(tokenized_step)
            step_labels[-1] = label
            tokenized_step.extend(post_step_tokens)
            step_labels.extend([-100]* len(post_step_tokens))
            
            # Avoid adding steps if the maximum length is reached as in prm training only the token after the last token is labeled.
            if (len(input_ids)+len(tokenized_step))<(max_length-1):
                input_ids.extend(tokenized_step)
                labels.extend(step_labels)
            else:
                # exit if the maximum length is reached to avoid skipping steps in favor or shoter steps.
                break
            
        if getattr(tokenizer, 'add_eos_token', False) and tokenizer.eos_token_id is not None:
            input_ids.append(tokenizer.eos_token_id)
            labels.append(-100)
            
        new_examples["input_ids"].append(input_ids)
        new_examples["attention_mask"].append([1]*len(input_ids))
        new_examples["labels"].append(labels)
        
    return new_examples


class PRMTrainer(Trainer):
    _tag_names = ["trl", "prm-trainer"]

    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, nn.Module]] = None,
        args: Optional[PRMConfig] = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        peft_config: Optional[Dict] = None,
    ):
        """
        Initialize PRMTrainer.

        Args:
            model (`transformers.PreTrainedModel`):
                The model to train, preferably an `AutoModelForSequenceClassification`.
            args (`PRMConfig`):
                The arguments to use for training.
            data_collator (`transformers.DataCollator`):
                The data collator to use for training. If None is specified, the default data collator (`RewardDataCollatorWithPadding`) will be used
                which will pad the sequences to the maximum length of the sequences in the batch, given a dataset of paired sequences.
            train_dataset (`datasets.Dataset`):
                The dataset to use for training.
            eval_dataset (`datasets.Dataset`):
                The dataset to use for evaluation.
            tokenizer (`transformers.PreTrainedTokenizerBase`):
                The tokenizer to use for training. This argument is required if you want to use the default data collator.
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
            peft_config (`Dict`, defaults to `None`):
                The PEFT configuration to use for training. If you pass a PEFT configuration, the model will be wrapped in a PEFT model.
        """
        if not type(args) is PRMConfig:
            raise ValueError(
                f"args should be an instance of PRMConfig but got {type(args)}"
            )
        if not is_peft_available() and peft_config is not None:
            raise ValueError(
                "PEFT is not installed and you passed a `peft_config` in the trainer's kwargs, please install it to use the PEFT models"
            )
        elif is_peft_available() and peft_config is not None:
            if not isinstance(model, PeftModel):
                if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_quantized", False):
                    _supports_gc_kwargs = "gradient_checkpointing_kwargs" in list(
                        inspect.signature(prepare_model_for_kbit_training).parameters
                    )

                    prepare_model_kwargs = {"use_gradient_checkpointing": args.gradient_checkpointing}

                    if not _supports_gc_kwargs and args.gradient_checkpointing_kwargs is not None:
                        warnings.warn(
                            "You passed `gradient_checkpointing_kwargs` in the trainer's kwargs, but your peft version does not support it. "
                            "please update to the latest version of peft to use `gradient_checkpointing_kwargs`."
                        )
                    elif _supports_gc_kwargs and args.gradient_checkpointing_kwargs is not None:
                        prepare_model_kwargs["gradient_checkpointing_kwargs"] = args.gradient_checkpointing_kwargs

                    model = prepare_model_for_kbit_training(model, **prepare_model_kwargs)

                model = get_peft_model(model, peft_config)

        if compute_metrics is None:
            compute_metrics = compute_accuracy

        if data_collator is None:
            if tokenizer is None:
                raise ValueError(
                    "A tokenizer must be specified when using the default DataCollatorForTokenClassification"
                )
            data_collator = DataCollatorForTokenClassification(tokenizer, max_length=args.max_length)


        if "input_ids" not in train_dataset.column_names:
            with PartialState().local_main_process_first():
                fn_kwargs = {"tokenizer": tokenizer}
                eval_dataset = eval_dataset.map(maybe_apply_chat_template, fn_kwargs=fn_kwargs)
                train_dataset = train_dataset.map(
                    _tokenize,
                    batched=True,
                    fn_kwargs={"tokenizer": tokenizer, "max_length": args.max_length},
                    num_proc=args.dataset_num_proc,
                )
 
                if eval_dataset is not None:
                    eval_dataset = eval_dataset.map(maybe_apply_chat_template, fn_kwargs=fn_kwargs)
                    eval_dataset = eval_dataset.map(
                        _tokenize,
                        fn_kwargs={"tokenizer": tokenizer, "max_length": args.max_length},
                        batched=True,
                        num_proc=args.dataset_num_proc,
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

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

    @wraps(Trainer.push_to_hub)
    def push_to_hub(
        self,
        commit_message: Optional[str] = "End of training",
        blocking: bool = True,
        **kwargs,
    ) -> str:
        """
        Overwrite the `push_to_hub` method in order to force-add the tag "prm-trainer" when pushing the
        model on the Hub. Please refer to `~transformers.Trainer.push_to_hub` for more details.
        Unlike the parent class, we don't use the `token` argument to mitigate security risks.
        """
        kwargs = trl_sanitze_kwargs_for_tagging(model=self.model, tag_names=self._tag_names, kwargs=kwargs)
        return super().push_to_hub(commit_message=commit_message, blocking=blocking, **kwargs)
