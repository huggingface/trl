# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
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

import warnings
from collections.abc import Callable

import torch.nn as nn
from datasets import Dataset
from transformers import (
    BaseImageProcessor,
    DataCollator,
    FeatureExtractionMixin,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput

from ..experimental.cpo import CPOConfig, CPOTrainer as _CPOTrainer


class CPOTrainer(_CPOTrainer):
    r"""
    Initialize CPOTrainer.

    This class is now located in `trl.experimental.cpo`. Please update your imports to
    `from trl.experimental.cpo import CPOTrainer`. The current import path will be removed in TRL 0.29.
    For more information, see https://github.com/huggingface/trl/issues/4223.

    Args:
        model ([`~transformers.PreTrainedModel`]):
            The model to train, preferably an [`~transformers.AutoModelForSequenceClassification`].
        args ([`CPOConfig`]):
            The CPO config arguments to use for training.
        data_collator ([`~transformers.DataCollator`]):
            The data collator to use for training. If None is specified, the default data collator
            ([`DPODataCollatorWithPadding`]) will be used which will pad the sequences to the maximum length of the
            sequences in the batch, given a dataset of paired sequences.
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
        callbacks (`list[transformers.TrainerCallback]`):
            The callbacks to use for training.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`):
            The optimizer and scheduler to use for training.
        preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`):
            The function to use to preprocess the logits before computing the metrics.
        peft_config (`dict`, defaults to `None`):
            The PEFT configuration to use for training. If you pass a PEFT configuration, the model will be wrapped in
            a PEFT model.
        compute_metrics (`Callable[[EvalPrediction], dict]`, *optional*):
            The function to use to compute the metrics. Must take a `EvalPrediction` and return a dictionary string to
            metric values.
    """

    def __init__(
        self,
        model: PreTrainedModel | nn.Module | str | None = None,
        args: CPOConfig | None = None,
        data_collator: DataCollator | None = None,
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | dict[str, Dataset] | None = None,
        processing_class: PreTrainedTokenizerBase
        | BaseImageProcessor
        | FeatureExtractionMixin
        | ProcessorMixin
        | None = None,
        model_init: Callable[[], PreTrainedModel] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        peft_config: dict | None = None,
        compute_metrics: Callable[[EvalLoopOutput], dict] | None = None,
    ):
        warnings.warn(
            "The `CPOTrainer` is now located in `trl.experimental`. Please update your imports to "
            "`from trl.experimental.cpo import CPOTrainer`. The current import path will be removed and no longer "
            "supported in TRL 0.29. For more information, see https://github.com/huggingface/trl/issues/4223.",
            FutureWarning,
            stacklevel=2,
        )
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            model_init=model_init,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            peft_config=peft_config,
            compute_metrics=compute_metrics,
        )
