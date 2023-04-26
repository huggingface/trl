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
    DataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction

from ..import_utils import is_peft_available


if is_peft_available():
    from peft import PeftConfig, get_peft_model, prepare_model_for_int8_training


class PeftSavingCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        peft_model_path = os.path.join(state.best_model_checkpoint, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)


class SFTTrainer(Trainer):
    r"""
    Class definition of the Supervised Finetuning Trainer (SFT Trainer).
    This class is a wrapper around the `transformers.Trainer` class and inherits all of its attributes and methods.
    The trainer takes care of properly initializing the PeftModel in case a user passes a `PeftConfig` object.
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
        prepare_in_int8_kwargs: Optional[Dict] = {},
        **kwargs,
    ):
        from_pretrained_kwargs = {}

        if isinstance(model, str):
            warnings.warn(
                "You passed a model_id and a `PeftConfig` to the SFTTrainer. This will automatically create an "
                "`AutoModelForCausalLM` or a `PeftModel` for you."
            )

            from_pretrained_signature = PreTrainedModel.from_pretrained.__code__.co_varnames
            for key in from_pretrained_signature:
                if key in kwargs:
                    from_pretrained_kwargs[key] = kwargs.pop(key)

        if is_peft_available() and peft_config is not None:
            if not isinstance(peft_config, PeftConfig):
                raise ValueError(
                    "If you want to use the PeftModel, you need to pass a PeftConfig object to the SFTTrainer."
                    f" and you passed a {type(peft_config)}."
                )

            if from_pretrained_kwargs.get("load_in_8bit", False) and "device_map" not in from_pretrained_kwargs:
                device_map = {"": Accelerator().process_index}
                from_pretrained_kwargs["device_map"] = device_map

            model = AutoModelForCausalLM.from_pretrained(
                model,
                **from_pretrained_kwargs,
            )

            if getattr(model, "is_loaded_in_8bit", False):
                model = prepare_model_for_int8_training(model, **prepare_in_int8_kwargs)

            model = get_peft_model(model, peft_config)

            if callbacks is None:
                callbacks = [PeftSavingCallback]

        else:
            model = AutoModelForCausalLM.from_pretrained(model, **from_pretrained_kwargs)

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
            **kwargs,
        )
