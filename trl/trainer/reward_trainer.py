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
import warnings
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset
from transformers import DataCollator, PreTrainedModel, PreTrainedTokenizerBase, Trainer, TrainingArguments
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction

from ..import_utils import is_peft_available
from .utils import RewardDataCollatorWithPadding


if is_peft_available():
    from peft import get_peft_model


def compute_accuracy(eval_pred):
    predictions, _ = eval_pred
    # Here, predictions is rewards_chosen and rewards_rejected.
    # We want to see how much of the time rewards_chosen > rewards_rejected.
    predictions = np.argmax(predictions, axis=0)
    labels = np.zeros(predictions.shape)

    accuracy = np.mean([p == l for p, l in zip(predictions, labels)])
    return {"accuracy": accuracy}


class RewardTrainer(Trainer):
    r"""
    The RewardTrainer can be used to train your custom Reward Model. It is a subclass of the
    `transformers.Trainer` class and inherits all of its attributes and methods. It is recommended to use
    an `AutoModelForSequenceClassification` as the reward model. The reward model should be trained on a dataset
    of paired examples, where each example is a tuple of two sequences. The reward model should be trained to
    predict which example in the pair is more relevant to the task at hand.

    The reward trainer expects a very specific format for the dataset. The dataset should contain two 4 entries at least
    if you don't use the default `RewardDataCollatorWithPadding` data collator. The entries should be named
    - `input_ids_chosen`
    - `attention_mask_chosen`
    - `input_ids_rejected`
    - `attention_mask_rejected`

    """

    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
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
        max_length: Optional[int] = None,
        peft_config: Optional[Dict] = None,
    ):
        """
        Initialize RewardTrainer.

        Args:
            model (`transformers.PreTrainedModel`):
                The model to train, preferably an `AutoModelForSequenceClassification`.
            args (`transformers.TrainingArguments`):
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
            max_length (`int`, defaults to `None`):
                The maximum length of the sequences in the batch. This argument is required if you want to use the default data collator.
            peft_config (`Dict`, defaults to `None`):
                The PEFT configuration to use for training. If you pass a PEFT configuration, the model will be wrapped in a PEFT model.
        """
        if not is_peft_available() and peft_config is not None:
            raise ValueError(
                "PEFT is not installed and you passed a `peft_config` in the trainer's kwargs, please install it to use the PEFT models"
            )
        elif is_peft_available() and peft_config is not None:
            model = get_peft_model(model, peft_config)

        if compute_metrics is None:
            compute_metrics = compute_accuracy

        if data_collator is None:
            if tokenizer is None:
                raise ValueError(
                    "max_length or a tokenizer must be specified when using the default RewardDataCollatorWithPadding"
                )
            if max_length is None:
                warnings.warn(
                    "When using RewardDataCollatorWithPadding, you should set `max_length` in the RewardTrainer's init"
                    " it will be set to `512` by default, but you should do it yourself in the future.",
                    UserWarning,
                )
                max_length = 512
            data_collator = RewardDataCollatorWithPadding(tokenizer, max_length=max_length)

            if args.remove_unused_columns:
                args.remove_unused_columns = False
                # warn users
                warnings.warn(
                    "When using RewardDataCollatorWithPadding, you should set `remove_unused_columns=False` in your TrainingArguments"
                    " we have set it for you, but you should do it yourself in the future.",
                    UserWarning,
                )

            self.use_reward_data_collator = True
        else:
            self.use_reward_data_collator = False
        super().__init__(
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            model_init,
            compute_metrics,
            callbacks,
            optimizers,
            preprocess_logits_for_metrics,
        )

    def compute_loss(self, model, inputs, return_outputs=False):
        if not self.use_reward_data_collator:
            raise NotImplementedError(
                "compute_loss is only implemented for RewardDataCollatorWithPadding, please implement your own compute_loss method if you are using a custom data collator"
            )
        rewards_chosen = model(input_ids=inputs["input_ids_chosen"], attention_mask=inputs["attention_mask_chosen"])[0]
        rewards_rejected = model(
            input_ids=inputs["input_ids_rejected"], attention_mask=inputs["attention_mask_rejected"]
        )[0]
        loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()
        if return_outputs:
            return loss, {"rewards_chosen": rewards_chosen, "rewards_rejected": rewards_rejected}
        return loss
