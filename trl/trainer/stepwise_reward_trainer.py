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
from dataclasses import dataclass
from itertools import chain
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from accelerate import PartialState
from datasets import Dataset
from transformers import (
    BaseImageProcessor,
    DataCollator,
    FeatureExtractionMixin,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    Trainer,
    is_wandb_available,
)
from transformers.data.data_collator import DataCollatorMixin
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction
from transformers.utils import is_peft_available

from .stepwise_reward_config import StepwiseRewardConfig
from .utils import compute_accuracy, generate_model_card, pad


if is_peft_available():
    from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training

if is_wandb_available():
    import wandb


@dataclass
class PromptCompletionCollator(DataCollatorMixin):
    """
    Data collator used for prompt-completion data. Inputs are dynamically padded to the maximum length of a batch if
    they are not all of the same length.

    Args:
        pad_token_id (`int`):
            Token ID to use for padding.
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            Type of Tensor to return. Only `"pt"` is currently supported.

    Examples:
    ```python
    >>> from trl import PromptCompletionCollator
    >>> collator = PromptCompletionCollator(pad_token_id=0)
    >>> examples = [
    ...     {"prompt_input_ids": [1, 2, 3], "completion_input_ids": [4, 5], "label": [1, 0]},
    ...     {"prompt_input_ids": [7, 8], "completion_input_ids": [9], "label": [1]}
    ... ]
    >>> collator(examples)
    {'prompt_input_ids': tensor([[1, 2, 3],
                                 [0, 7, 8]]),
     'prompt_attention_mask': tensor([[1, 1, 1],
                                      [0, 1, 1]]),
     'completion_input_ids': tensor([[4, 5],
                                     [9, 0]]),
     'completion_attention_mask': tensor([[1, 1],
                                          [1, 0]]),
     'label': tensor([[1, 0],
                      [1, 0]])
    }
    ```
    """

    pad_token_id: int
    return_tensors: str = "pt"

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # Convert to tensor
        prompt_input_ids = [torch.tensor(example["prompt_input_ids"]) for example in examples]
        prompt_attention_mask = [torch.ones_like(input_ids) for input_ids in prompt_input_ids]
        completion_input_ids = [torch.tensor(example["completion_input_ids"]) for example in examples]
        completion_attention_mask = [torch.ones_like(input_ids) for input_ids in completion_input_ids]
        label = [torch.tensor(example["label"]) for example in examples]

        # Pad
        output = {}
        output["prompt_input_ids"] = pad(prompt_input_ids, padding_value=self.pad_token_id, padding_side="left")
        output["prompt_attention_mask"] = pad(prompt_attention_mask, padding_value=0, padding_side="left")
        output["completion_input_ids"] = pad(completion_input_ids, padding_value=self.pad_token_id)
        output["completion_attention_mask"] = pad(completion_attention_mask, padding_value=0)
        output["label"] = pad(label, padding_value=-100)

        return output


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

        if processing_class.pad_token_id is not None:
            self.padding_value = processing_class.pad_token_id
        else:
            raise ValueError(
                "Can't find `pad_token_id` in the `processing_class`. "
                "Explicitly set `tokenizer.pad_token` (e.g. `tokenizer.pad_token = tokenizer.eos_token`) "
                "before instantiating the trainer."
            )

        if data_collator is None:
            data_collator = PromptCompletionCollator(pad_token_id=self.padding_value)

        if "input_ids" not in train_dataset.column_names:
            with PartialState().local_main_process_first():
                fn_kwargs = {
                    "processing_class": processing_class,
                    "step_separator": args.step_separator,
                    "max_completion_length": args.max_completion_length,
                    "train_on_last_step_only": args.train_on_last_step_only,
                }
                train_dataset = train_dataset.map(
                    self.tokenize_row,
                    fn_kwargs=fn_kwargs,
                    num_proc=args.dataset_num_proc,
                    remove_columns=train_dataset.features,
                    desc="Tokenizing train dataset",
                )

                if eval_dataset is not None:
                    eval_dataset = eval_dataset.map(
                        self.tokenize_row,
                        fn_kwargs=fn_kwargs,
                        num_proc=args.dataset_num_proc,
                        remove_columns=eval_dataset.features,
                        desc="Tokenizing eval dataset",
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

    @staticmethod
    def tokenize_row(features, processing_class, step_separator, max_completion_length, train_on_last_step_only):
        """
        Tokenize a row of the dataset.

        Args:
            features (`Dict[str, str]`):
                Row of the dataset, should contain the keys `"prompt"`, `"completions"`, and `"labels"`.
            processing_class (`PreTrainedTokenizerBase`):
                Processing class used to process the data.
            step_separator (`str`):
                Separator between steps in the completion.
            max_completion_length (`int` or `None`):
                Maximum length of the completion sequences. If `None`, the completion sequences are not truncated.
            train_on_last_step_only (`bool`):
                Whether to train only on the last step. If `True`, the labels are `-100` for all tokens except the last
                token of the completion.

        Returns:
            `Dict[str, List[int]]`:
                Tokenized sequences with the keys `"prompt_input_ids"`, `"completion_input_ids"`, and `"label".

        Example:
        ```python
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
        >>> features = {"prompt": "Which number is larger, 9.8 or 9.11?",
        ...             "completions": ["11 is greater than 8.",
        ...                             "Hence, 9.11 > 9.8."],
        ...             "labels": [True, False]}
        >>> StepwiseRewardTrainer.tokenize_row(features, tokenizer, "\n", max_completion_length=None, train_on_last_step_only=False)
        {'prompt_input_ids': [23085, 1372, 374, 8131, 11, 220, 24, 13, 23, 476, 220, 24, 13, 16, 16, 30],
         'completion_input_ids': [16, 16, 374, 7046, 1091, 220, 23, 13, 198, 39, 763, 11, 220, 24, 13, 16, 16, 861, 220, 24, 13, 23, 13, 198],
         'label': [-100, -100, -100, -100, -100, -100, -100, -100, 1.0, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 0.0]}
        ```
        """
        tokenizer = processing_class  # the processing class is a tokenizer

        # Tokenize the prompt and completions
        prompt_ids = tokenizer(features["prompt"])["input_ids"]
        completions_ids = [tokenizer(completion)["input_ids"] for completion in features["completions"]]
        if train_on_last_step_only:
            labels = [-100] * (len(features["labels"]) - 1) + [int(features["labels"][-1])]
        else:
            labels = [int(label) for label in features["labels"]]

        # Get the ID of the separator token and add it to the completions
        separator_ids = tokenizer.encode(step_separator)
        completions_ids = [completion + separator_ids for completion in completions_ids]

        # Create the label
        labels = [[-100] * (len(completion) - 1) + [label] for completion, label in zip(completions_ids, labels)]

        # Join the completions and labels steps
        completion_ids = list(chain(*completions_ids))
        label = list(chain(*labels))

        if max_completion_length is not None:
            completion_ids = completion_ids[:max_completion_length]
            label = label[:max_completion_length]

        return {
            "prompt_input_ids": prompt_ids,
            "completion_input_ids": completion_ids,
            "label": label,
        }

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In DPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by `DPODataCollatorWithPadding`, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt_input_ids", "completion_input_ids", "label"]

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
