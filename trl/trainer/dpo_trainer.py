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
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from datasets import Dataset
from transformers import DataCollator, PreTrainedModel, PreTrainedTokenizerBase, Trainer, TrainingArguments
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction

from ..core import logprobs_from_logits, masked_mean
from ..import_utils import is_peft_available
from .utils import DPODataCollatorWithPadding, compute_accuracy


if is_peft_available():
    from peft import get_peft_model, prepare_model_for_int8_training


class DPOTrainer(Trainer):
    r"""
    Initialize DPOTrainer.

    Args:
        model (`transformers.PreTrainedModel`):
            The model to train, preferably an `AutoModelForSequenceClassification`.
        ref_model (`PreTrainedModelWrapper`):
            Hugging Face transformer model with a casual language modelling head. Used for implicit reward computation and loss.
        beta (`float`, defaults to 0.1):
            The beta factor in DPO loss. Higher beta means less divergence from the initial policy.
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

    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        ref_model: Union[PreTrainedModel, nn.Module] = None,
        beta: float = 0.1,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        label_pad_token_id: int = -100,
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
        max_length: Optional[int] = None,
        peft_config: Optional[Dict] = None,
    ):
        if not is_peft_available() and peft_config is not None:
            raise ValueError(
                "PEFT is not installed and you passed a `peft_config` in the trainer's kwargs, please install it to use the PEFT models"
            )
        elif is_peft_available() and peft_config is not None:
            if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False):
                model = prepare_model_for_int8_training(model)
            model = get_peft_model(model, peft_config)

        if compute_metrics is None:
            compute_metrics = compute_accuracy

        if data_collator is None:
            if tokenizer is None:
                raise ValueError(
                    "max_length or a tokenizer must be specified when using the default DPODataCollatorWithPadding"
                )
            if max_length is None:
                warnings.warn(
                    "When using DPODataCollatorWithPadding, you should set `max_length` in the DPOTrainer's init"
                    " it will be set to `512` by default, but you should do it yourself in the future.",
                    UserWarning,
                )
                max_length = 512
            data_collator = DPODataCollatorWithPadding(
                tokenizer, max_length=max_length, label_pad_token_id=label_pad_token_id
            )

            if args.remove_unused_columns:
                args.remove_unused_columns = False
                # warn users
                warnings.warn(
                    "When using DPODataCollatorWithPadding, you should set `remove_unused_columns=False` in your TrainingArguments"
                    " we have set it for you, but you should do it yourself in the future.",
                    UserWarning,
                )

            self.use_dpo_data_collator = True
        else:
            self.use_dpo_data_collator = False

        self.label_pad_token_id = label_pad_token_id

        self.beta = beta
        self.ref_model = ref_model

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

        # Since we inherit from trainer we always have access to an accelerator
        if hasattr(self, "accelerator"):
            self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
        else:
            raise AttributeError(
                "Your `Trainer` does not have an `accelerator` object. Consider upgrading `transformers`."
            )

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if not self.use_dpo_data_collator:
            raise NotImplementedError(
                "compute_loss is only implemented for DPODataCollatorWithPadding, please implement your own compute_loss method if you are using a custom data collator"
            )

        # concat the chosen and rejected inputs
        max_length = max(inputs["input_ids_chosen"].shape[1], inputs["input_ids_rejected"].shape[1])

        inputs["input_ids_chosen"] = torch.nn.functional.pad(
            inputs["input_ids_chosen"],
            (0, max_length - inputs["input_ids_chosen"].shape[1]),
            "constant",
            0,
        )
        inputs["input_ids_rejected"] = torch.nn.functional.pad(
            inputs["input_ids_rejected"],
            (0, max_length - inputs["input_ids_rejected"].shape[1]),
            "constant",
            0,
        )
        inputs["attention_mask_chosen"] = torch.nn.functional.pad(
            inputs["attention_mask_chosen"],
            (0, max_length - inputs["attention_mask_chosen"].shape[1]),
            "constant",
            0,
        )

        inputs["attention_mask_rejected"] = torch.nn.functional.pad(
            inputs["attention_mask_rejected"],
            (0, max_length - inputs["attention_mask_rejected"].shape[1]),
            "constant",
            0,
        )

        inputs["labels_chosen"] = torch.nn.functional.pad(
            inputs["labels_chosen"],
            (0, max_length - inputs["labels_chosen"].shape[1]),
            "constant",
            self.label_pad_token_id,
        )

        inputs["labels_rejected"] = torch.nn.functional.pad(
            inputs["labels_rejected"],
            (0, max_length - inputs["labels_rejected"].shape[1]),
            "constant",
            self.label_pad_token_id,
        )

        logits_model = model(
            input_ids=torch.cat((inputs["input_ids_chosen"], inputs["input_ids_rejected"]), dim=0),
            attention_mask=torch.cat(
                (inputs["attention_mask_chosen"], inputs["attention_mask_rejected"]),
                dim=0,
            ),
        )[0]
        logits_chosen_model, logits_rejected_model = logits_model.chunk(2, dim=0)

        with torch.no_grad():
            logits_ref = self.ref_model(
                input_ids=torch.cat((inputs["input_ids_chosen"], inputs["input_ids_rejected"]), dim=0),
                attention_mask=torch.cat(
                    (
                        inputs["attention_mask_chosen"],
                        inputs["attention_mask_rejected"],
                    ),
                    dim=0,
                ),
            )[0]
            logits_chosen_ref, logits_rejected_ref = logits_ref.chunk(2, dim=0)

        log_prob_chosen_model = logprobs_from_logits(logits_chosen_model, inputs["input_ids_chosen"])
        log_prob_rejected_model = logprobs_from_logits(logits_rejected_model, inputs["input_ids_rejected"])

        log_prob_chosen_ref = logprobs_from_logits(logits_chosen_ref, inputs["input_ids_chosen"])
        log_prob_rejected_ref = logprobs_from_logits(logits_rejected_ref, inputs["input_ids_rejected"])

        mask_chosen = (inputs["labels_chosen"] != self.label_pad_token_id).float()
        mask_rejected = (inputs["labels_rejected"] != self.label_pad_token_id).float()

        log_prob_chosen_model = masked_mean(log_prob_chosen_model, mask_chosen, 1)
        log_prob_rejected_model = masked_mean(log_prob_rejected_model, mask_rejected, 1)
        log_prob_chosen_ref = masked_mean(log_prob_chosen_ref, mask_chosen, 1)
        log_prob_rejected_ref = masked_mean(log_prob_rejected_ref, mask_rejected, 1)

        pi_logratios = log_prob_chosen_model - log_prob_rejected_model
        ref_logratios = log_prob_chosen_ref - log_prob_rejected_ref

        loss = -nn.functional.logsigmoid(self.beta * (pi_logratios - ref_logratios)).mean()
        rewards_chosen = self.beta * (log_prob_chosen_model - log_prob_chosen_ref).detach()
        rewards_rejected = self.beta * (log_prob_rejected_model - log_prob_rejected_ref).detach()
        if return_outputs:
            return loss, {
                "rewards_chosen": rewards_chosen,
                "rewards_rejected": rewards_rejected,
            }
        return loss
