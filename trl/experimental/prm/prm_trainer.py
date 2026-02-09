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

import textwrap
from collections.abc import Callable
from itertools import chain
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import transformers
from accelerate import PartialState, logging
from datasets import Dataset, features
from packaging.version import Version
from transformers import (
    BaseImageProcessor,
    DataCollator,
    DataCollatorForTokenClassification,
    FeatureExtractionMixin,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainerCallback,
)
from transformers.trainer_utils import EvalPrediction
from transformers.utils import is_peft_available

from ...trainer.base_trainer import BaseTrainer
from ...trainer.utils import disable_dropout_in_model
from ..utils import prepare_peft_model
from .prm_config import PRMConfig


if is_peft_available():
    from peft import PeftModel

logger = logging.get_logger(__name__)


def compute_accuracy(eval_pred: EvalPrediction) -> dict[str, float]:
    predictions, labels = eval_pred
    if predictions.ndim == 3:
        # Token classification task. Shapes are (batch_size, seq_len, num_labels) and (batch_size, seq_len)
        # Used to compute the accuracy in the prm_trainer.
        predictions = np.argmax(predictions, axis=2)

        # Flatten the predictions and labels to remove the ignored tokens.
        predictions = np.array(
            [
                p
                for prediction, label in zip(predictions, labels, strict=True)
                for (p, lbl) in zip(prediction, label, strict=True)
                if lbl != -100
            ]
        )
        labels = np.array([lbl for label in labels for lbl in label if lbl != -100])

    else:
        # Here, predictions is rewards_chosen and rewards_rejected. Shapes are (batch_size, 2) and (batch_size,)
        # We want to see how much of the time rewards_chosen > rewards_rejected.
        equal_mask = predictions[:, 0] == predictions[:, 1]
        equal_predictions_count = int(equal_mask.sum())

        if equal_predictions_count > 0:
            # Before using the logger, the accelerate state must be initialized. It'susually the case when using this
            # function inside a Trainer, but it may not be the case otherwise, in particular when unit testing.
            PartialState()

            logger.warning(
                f"There are {equal_predictions_count} out of {len(predictions[:, 0])} instances where the predictions "
                "for both options are equal. These instances are ignored in the accuracy computation.",
            )

        # Filter out equal predictions
        predictions = predictions[~equal_mask]
        labels = labels[~equal_mask]

        # Use the remaining predictions for accuracy calculation
        predictions = np.argmax(predictions, axis=1)

    accuracy = np.array(predictions == labels, dtype=float).mean().item()
    return {"accuracy": accuracy}


class PRMTrainer(BaseTrainer):
    """
    Initialize PRMTrainer.

    Args:
        model ([`~transformers.PreTrainedModel`]):
            The model to train, preferably an `AutoModelForTokenClassification`.
        args ([`experimental.prm.PRMConfig`]):
            The arguments to use for training.
        data_collator ([`~transformers.DataCollator`]):
            The data collator to use for training. If None is specified, the default data collator
            ([`~transformers.DataCollatorForTokenClassification`]) will be used which will pad the sequences to the
            maximum length of the sequences in the batch, given a dataset of paired sequences.
        train_dataset ([`~datasets.Dataset`]):
            The dataset to use for training.
        eval_dataset ([`~datasets.Dataset`]):
            The dataset to use for evaluation.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], [`~transformers.BaseImageProcessor`], [`~transformers.FeatureExtractionMixin`] or [`~transformers.ProcessorMixin`], *optional*):
            Processing class used to process the data. If provided, will be used to automatically process the inputs
            for the model, and it will be saved along the model to make it easier to rerun an interrupted training or
            reuse the fine-tuned model.
        model_init (`Callable[[], transformers.PreTrainedModel]`):
            The model initializer to use for training. If None is specified, the default model initializer will be
            used.
        compute_metrics (`Callable[[transformers.EvalPrediction], dict]`, *optional* defaults to `compute_accuracy`):
            The metrics to use for evaluation. If no metrics are specified, the default metric (`compute_accuracy`)
            will be used.
        callbacks (`list[transformers.TrainerCallback]`):
            The callbacks to use for training.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`):
            The optimizer and scheduler to use for training.
        preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`):
            The function to use to preprocess the logits before computing the metrics.
        peft_config (`dict`, defaults to `None`):
            The PEFT configuration to use for training. If you pass a PEFT configuration, the model will be wrapped in
            a PEFT model.
    """

    _tag_names = ["trl", "prm"]
    _name = "PRM"
    _paper = {
        "title": "Solving math word problems with process-and outcome-based feedback",
        "id": "2211.14275",
        # docstyle-ignore
        "citation": textwrap.dedent("""\
            @article{uesato2022solving,
                title        = {{Solving Math Word Problems With Process- and Outcome-Based Feedback}},
                author       = {Uesato, Jonathan and Kushman, Nate and Kumar, Ramana and Song, Francis and Siegel, Noah and Wang, Lisa and Creswell, Antonia and Irving, Geoffrey and Higgins, Irina},
                year         = 2022,
                journal      = {arXiv preprint arXiv:2211.14275}
            }"""),
    }

    def __init__(
        self,
        model: PreTrainedModel | nn.Module | None = None,
        args: PRMConfig | None = None,
        data_collator: DataCollator | None = None,
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | dict[str, Dataset] | None = None,
        processing_class: PreTrainedTokenizerBase
        | BaseImageProcessor
        | FeatureExtractionMixin
        | ProcessorMixin
        | None = None,
        model_init: Callable[[], PreTrainedModel] | None = None,
        compute_metrics: Callable[[EvalPrediction], dict] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        peft_config: dict | None = None,
    ):
        if peft_config is not None or (is_peft_available() and isinstance(model, PeftModel)):
            model = prepare_peft_model(model, peft_config, args)

        # Disable dropout in the model
        if args.disable_dropout:
            disable_dropout_in_model(model)

        if compute_metrics is None:
            compute_metrics = compute_accuracy

        if data_collator is None:
            if processing_class is None:
                raise ValueError(
                    "A processing_class must be specified when using the default DataCollatorForTokenClassification"
                )
            data_collator = DataCollatorForTokenClassification(processing_class)

        if "input_ids" not in train_dataset.column_names:
            with PartialState().main_process_first():
                fn_kwargs = {
                    "tokenizer": processing_class,
                    "step_separator": args.step_separator,
                    "max_length": args.max_length,
                    "max_completion_length": args.max_completion_length,
                    "train_on_last_step_only": args.train_on_last_step_only,
                }
                train_fn_kwargs = {**fn_kwargs, "is_eval": False}
                train_dataset = train_dataset.map(
                    self.tokenize_row,
                    fn_kwargs=train_fn_kwargs,
                    num_proc=args.dataset_num_proc,
                    remove_columns=train_dataset.features,
                    desc="Tokenizing train dataset",
                    features=features.Features(  # needed to avoid map to cast labels to bool
                        {
                            "labels": features.Sequence(features.Value("int64")),
                            "input_ids": features.Sequence(features.Value("int64")),
                        }
                    ),
                )

                eval_fn_kwargs = {**fn_kwargs, "is_eval": True}
                if eval_dataset is not None:
                    eval_dataset = eval_dataset.map(
                        self.tokenize_row,
                        fn_kwargs=eval_fn_kwargs,
                        num_proc=args.dataset_num_proc,
                        remove_columns=eval_dataset.features,
                        desc="Tokenizing eval dataset",
                        features=features.Features(  # needed to avoid map to cast labels to bool
                            {
                                "labels": features.Sequence(features.Value("int64")),
                                "input_ids": features.Sequence(features.Value("int64")),
                            }
                        ),
                    )

        # Transformers explicitly set use_reentrant=True in the past to silence a PyTorch warning, but the default was
        # never updated once PyTorch switched to recommending use_reentrant=False. Until that change lands upstream
        # (see https://github.com/huggingface/transformers/pull/43203) and is released (most likely in 5.0.0), we
        # default to the recommended non-reentrant behavior here, while preserving any user-provided value.
        if args.gradient_checkpointing and Version(transformers.__version__) < Version("5.0.0"):
            args.gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs or {}
            args.gradient_checkpointing_kwargs.setdefault("use_reentrant", False)

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
    def tokenize_row(
        features,
        tokenizer,
        step_separator,
        max_length,
        max_completion_length,
        train_on_last_step_only,
        is_eval,
    ):
        r"""
        Tokenize a row of the dataset.

        Args:
            features (`dict[str, str]`):
                Row of the dataset, should contain the keys `"prompt"`, `"completions"`, and `"labels"`.
            tokenizer ([`~transformers.PreTrainedTokenizerBase`]):
                Tokenizer used to process the data.
            step_separator (`str`):
                Separator between steps in the completion.
            max_length (`int` or `None`):
               Maximum length of the sequences (prompt + completion). If `None`, the sequences are not truncated.
            max_completion_length (`int` or `None`):
                Maximum length of the completion sequences. If `None`, the completion sequences are not truncated.
            train_on_last_step_only (`bool`):
                Whether to train only on the last step. If `True`, the labels are `-100` for all tokens except the last
                token of the completion.
            is_eval (`bool`):
                Whether the function is used to tokenize samples from a training or an evaluation dataset. Used only if
                `train_on_last_step_only` is set to `True`.

        Returns:
            `dict[str, list[int]]`:
                Tokenized sequences with the keys `"input_ids"`, and `"labels".

        Example:
        ```python
        >>> from transformers import AutoTokenizer

        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
        >>> features = {
        ...     "prompt": "Which number is larger, 9.8 or 9.11?",
        ...     "completions": ["11 is greater than 8.", "Hence, 9.11 > 9.8."],
        ...     "labels": [True, False],
        ... }
        >>> PRMTrainer.tokenize_row(
        ...     features, tokenizer, "\n", max_completion_length=None, train_on_last_step_only=False, is_eval=False
        ... )
        {'input_ids': [23085, 1372, 374, 8131, 11, 220, 24, 13, 23, 476, 220, 24, 13, 16, 16, 30, 16, 16, 374, 7046, 1091, 220, 23, 13, 198, 39, 763, 11, 220, 24, 13, 16, 16, 861, 220, 24, 13, 23, 13, 198],
         'labels': [-100, -100, -100, -100, -100, -100, -100, -100, 1, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 0]}
        ```
        """
        # Tokenize the prompt and completions
        prompt_ids = tokenizer(features["prompt"], add_special_tokens=False)["input_ids"]
        completions_ids = [
            tokenizer(completion, add_special_tokens=False)["input_ids"] for completion in features["completions"]
        ]
        if train_on_last_step_only and not is_eval:
            labels = [-100] * (len(features["labels"]) - 1) + [int(features["labels"][-1])]
        else:
            labels = [int(label) for label in features["labels"]]

        # Get the ID of the separator token and add it to the completions
        separator_ids = tokenizer.encode(step_separator, add_special_tokens=False)
        completions_ids = [completion + separator_ids for completion in completions_ids]

        # Create the label
        labels = [
            [-100] * (len(completion) - 1) + [label] for completion, label in zip(completions_ids, labels, strict=True)
        ]

        # Join the completions and labels steps
        completion_ids = list(chain(*completions_ids))
        labels = list(chain(*labels))

        if tokenizer.bos_token_id is not None:
            prompt_ids = [tokenizer.bos_token_id] + prompt_ids

        # Truncate completion sequences
        if max_completion_length is not None:
            completion_ids = completion_ids[:max_completion_length]
            labels = labels[:max_completion_length]

        input_ids = prompt_ids + completion_ids
        labels = [-100] * len(prompt_ids) + labels

        if max_length is not None:
            input_ids = input_ids[:max_length]
            labels = labels[:max_length]

        return {"input_ids": input_ids, "labels": labels}

    # Ensure the model card is saved along with the checkpoint
    def _save_checkpoint(self, model, trial):
        if self.args.hub_model_id is None:
            model_name = Path(self.args.output_dir).name
        else:
            model_name = self.args.hub_model_id.split("/")[-1]
        self.create_model_card(model_name=model_name)
        super()._save_checkpoint(model, trial)
