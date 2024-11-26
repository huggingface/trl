# Copyright 2024 The HuggingFace Team. All rights reserved.
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
import os
import textwrap
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from accelerate import PartialState
from datasets import Dataset
from transformers import (
    BaseImageProcessor,
    DataCollator,
    DataCollatorForTokenClassification,
    FeatureExtractionMixin,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    Trainer,
    is_wandb_available,
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction
from transformers.utils import is_peft_available

from .stepwise_reward_config import StepwiseRewardConfig
from .utils import (
    compute_accuracy,
    generate_model_card,
)


if is_peft_available():
    from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training

if is_wandb_available():
    import wandb


def _tokenize_fn(
    batch: Dict[str, List[Any]],
    tokenizer,
    max_length: int,
    step_separator: str,
    train_on_last_step: bool,
) -> Dict[str, List[Any]]:
    """Tokenize a batch from a Stepwise supervision dataset."""
    new_examples = {"input_ids": [], "attention_mask": [], "labels": []}

    post_step_tokens = tokenizer.encode(step_separator, add_special_tokens=False)

    label_column_name = "label" if "label" in batch else "labels"

    for prompt, steps, labels in zip(batch["prompt"], batch["completions"], batch[label_column_name]):
        if isinstance(steps, str):
            steps = steps.strip().split(step_separator)

        if len(steps) != len(labels):
            raise ValueError("`labels` and `completions` should have the same length.")
        input_ids = []
        token_level_labels = []

        if tokenizer.bos_token_id is not None:
            if not prompt.startswith(tokenizer.bos_token):
                input_ids.append(tokenizer.bos_token_id)
                token_level_labels.append(-100)

        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        input_ids.extend(prompt_ids)
        token_level_labels.extend([-100] * len(prompt_ids))
        for i, (step, label) in enumerate(zip(steps, labels)):
            tokenized_step = tokenizer.encode(step, add_special_tokens=False)
            step_labels = [-100] * len(tokenized_step)
            if i == (len(steps) - 1) or not train_on_last_step:
                step_labels[-1] = int(label)

            if i < len(steps) - 1:
                tokenized_step.extend(post_step_tokens)
                step_labels.extend([-100] * len(post_step_tokens))

            # Avoid adding steps if the maximum length is reached as only the token after the last token of each step is labeled.
            if (len(input_ids) + len(tokenized_step)) < (max_length - 1):
                input_ids.extend(tokenized_step)
                token_level_labels.extend(step_labels)
            else:
                break

        new_examples["input_ids"].append(input_ids)
        new_examples["attention_mask"].append([1] * len(input_ids))
        new_examples["labels"].append(token_level_labels)

    return new_examples


class StepwiseRewardTrainer(Trainer):
    _tag_names = ["trl", "stepwise-reward-trainer"]

    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, nn.Module]] = None,
        args: Optional[StepwiseRewardConfig] = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        processing_class: Optional[
            Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin]
        ] = None,
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
        Initialize StepwiseRewardTrainer.

        Args:
            model (`transformers.PreTrainedModel`):
                The model to train, preferably an `AutoModelForTokenClassification`.
            args (`StepwiseRewardConfig`):
                The arguments to use for training.
            data_collator (`transformers.DataCollator`):
                The data collator to use for training. If None is specified, the default data collator (`DataCollatorForTokenClassification`) will be used
                which will pad the sequences to the maximum length of the sequences in the batch, given a dataset of paired sequences.
            train_dataset (`datasets.Dataset`):
                The dataset to use for training.
            eval_dataset (`datasets.Dataset`):
                The dataset to use for evaluation.
            processing_class (`PreTrainedTokenizerBase` or `BaseImageProcessor` or `FeatureExtractionMixin` or `ProcessorMixin`, *optional*):
                Processing class used to process the data. If provided, will be used to automatically process the inputs
                for the model, and it will be saved along the model to make it easier to rerun an interrupted training or
                reuse the fine-tuned model.
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
            if processing_class is None:
                raise ValueError(
                    "A processing_class must be specified when using the default DataCollatorForTokenClassification"
                )
            data_collator = DataCollatorForTokenClassification(processing_class, max_length=args.max_length)

        if "input_ids" not in train_dataset.column_names:
            if args.max_length is None:
                args.max_length = 512
                warnings.warn(
                    "When the dataset isn't pretokenized, you should set `max_length` in the `StepwiseRewardConfig`"
                    " we have set it for you, but you should do it yourself in the future."
                )
            with PartialState().local_main_process_first():
                tokenize_kwargs = {
                    "tokenizer": processing_class,
                    "max_length": args.max_length,
                    "step_separator": args.step_separator,
                    "train_on_last_step": args.train_on_last_step,
                }
                train_dataset = train_dataset.map(
                    _tokenize_fn,
                    batched=True,
                    fn_kwargs=tokenize_kwargs,
                    num_proc=args.dataset_num_proc,
                    remove_columns=train_dataset.features,
                )

                if eval_dataset is not None:
                    eval_dataset = eval_dataset.map(
                        _tokenize_fn,
                        batched=True,
                        fn_kwargs=tokenize_kwargs,
                        num_proc=args.dataset_num_proc,
                        remove_columns=eval_dataset.features,
                    )

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

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

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
        @article{uesato2022solving,
	title        = {Solving Math Word Problems With Process- and Outcome-Based Feedback},
	author       = {Uesato, Jonathan and Kushman, Nate and Kumar, Ramana and Song, Francis and Siegel, Noah and Wang, Lisa and Creswell, Antonia and Irving, Geoffrey and Higgins, Irina},
	year         = 2022,
	journal      = {arXiv preprint arXiv:2211.14275}
        }"""
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            trainer_name="Stepwise Reward",
            trainer_citation=citation,
            paper_title="Solving math word problems with process-and outcome-based feedback",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))
