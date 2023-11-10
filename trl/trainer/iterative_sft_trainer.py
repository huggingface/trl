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

import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    DataCollator,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import EvalLoopOutput

from ..core import PPODecorators
from ..import_utils import is_peft_available


if is_peft_available():
    from peft import PeftModel


class IterativeSFTTrainer(Trainer):
    """
    The IterativeSFTTrainer can be used to finetune models with methods that requires some steps between optimization.

    Attributes:
        **model** (`PreTrainedModel`) -- Model to be optimized, either an 'AutoModelForCausalLM' or an 'AutoModelForSeq2SeqLM'.
            Check the documentation of `PreTrainedModel` for more details.
        **args** (`transformers.TrainingArguments`): -- The arguments to use for training.
        **tokenizer** (`PreTrainedTokenizerBase`) -- Tokenizer to be used for encoding the
            data. Check the documentation of `transformers.PreTrainedTokenizer` and
            `transformers.PreTrainedTokenizerFast` for more details.
        **optimizers** (`Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`): -- The optimizer and scheduler to use for training.
        **data_collator** (Union[DataCollatorForLanguageModeling, DataCollatorForSeq2Seq], *optional*) -- Data collator to be used for training and
            passed along the dataloader.
        **eval_dataset** (`datasets.Dataset`): The dataset to use for evaluation.
        **max_length** (`int`, defaults to `None`): -- The maximum length of the input.
        **truncation_mode** (`str`, defaults to `keep_end`): -- The truncation mode to use, either `keep_end` or `keep_start`.
        **preprocess_logits_for_metrics** (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`): -- The function to use to preprocess the logits before computing the metrics.
        **compute_metrics** (`Callable[[EvalPrediction], Dict]`, *optional*): -- The function to use to compute the metrics. Must take a `EvalPrediction` and return a dictionary string to metric values.
        **optimize_device_cache ** (`bool`, *optional*, defaults to `False`) -- Optimize CUDA cache for slightly more memory-efficient training.
    """

    def __init__(
        self,
        model: PreTrainedModel = None,
        args: TrainingArguments = None,
        tokenizer: PreTrainedTokenizerBase = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        data_collator: Optional[DataCollator] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        max_length: Optional[int] = None,
        truncation_mode: Optional[str] = "keep_end",
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        compute_metrics: Optional[Callable[[EvalLoopOutput], Dict]] = None,
        optimize_device_cache: Optional[bool] = False,
    ):
        # Step 0: check positional arguments validity
        if not isinstance(tokenizer, (PreTrainedTokenizerBase)):
            raise ValueError(
                f"tokenizer must be a PreTrainedTokenizerBase like a PreTrainedTokenizer or a PreTrainedTokenizerFast, got {type(tokenizer)}"
            )
        if not isinstance(model, PreTrainedModel):
            raise ValueError(f"model must be a PreTrainedModel, got {type(model)}")
        if not model.can_generate():
            warnings.warn(
                f"The current model class {type(model)} is not compatible with `.generate()`"
                "Please make sure that this is intended."
            )
        if optimizers[1] is None and args.max_steps == -1:
            raise ValueError(
                "When no scheduler is provided, you need to set the total number of training steps to perform `max_steps`"
            )

        self.is_encoder_decoder = getattr(model.config, "is_encoder_decoder", False)
        self.is_peft_model = is_peft_available() and isinstance(model, PeftModel)

        self.tokenizer = tokenizer

        if data_collator is None:
            if self.is_encoder_decoder:
                warnings.warn(
                    "No data collator is provided. Using 'DataCollatorForSeq2Seq' with"
                    "'labels_pad_token_id' set to '-100' and 'pad_to_multiple_of' set to 8."
                )
                self.data_collator = DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=-100, pad_to_multiple_of=8)
            else:
                warnings.warn("No data collator is provided. Using 'DataCollatorForLanguageModeling'")
                self.data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        else:
            self.data_collator = data_collator

        self.max_length = max_length
        self.truncation_mode = truncation_mode
        self.optimize_device_cache = optimize_device_cache

        super().__init__(
            model=model,
            args=args,
            data_collator=self.data_collator,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        self.create_optimizer_and_scheduler(self.args.max_steps)

        # prepare model, optimizer and lr_scheduler
        self.model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.lr_scheduler
        )

        self.tokenizer.truncation_side = "left" if self.truncation_mode == "keep_end" else "right"

        if not hasattr(self, "accelerator"):
            raise AttributeError(
                "Your `Trainer` does not have an `accelerator` object. Consider upgrading `transformers`."
            )

        PPODecorators.optimize_device_cache = self.optimize_device_cache

    def prepare_model_inputs(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor):
        if attention_mask is None:
            attention_mask = [torch.ones_like(ids) for ids in input_ids]

        if self.is_encoder_decoder:
            input_data = self.data_collator(
                [
                    {"input_ids": ids, "attention_mask": att, "labels": lab}
                    for ids, att, lab in zip(input_ids, attention_mask, labels)
                ]
            ).to(self.model.device)

            input_data.pop("decoder_input_ids", None)  # This is directly computed inside the model

            input_data["labels"][input_data["labels"] == self.tokenizer.pad_token_id] = -100

        else:
            input_data = self.data_collator(
                [{"input_ids": ids, "attention_mask": att} for ids, att in zip(input_ids, attention_mask)]
            ).to(self.model.device)

        # truncate in case the user has provided input_ids, attention_mask and labels
        if self.max_length is not None:
            if self.truncation_mode == "keep_start":
                input_data = {k: v[: self.max_length] for k, v in input_data.items()}
            elif self.truncation_mode == "keep_end":
                input_data = {k: v[-self.max_length :] for k, v in input_data.items()}
            else:
                raise ValueError(f"Unknown truncation mode: {self.truncation_mode}")

        return input_data

    @staticmethod
    def _step_safety_checker(
        input_ids: List[torch.LongTensor],
        attention_mask: List[torch.LongTensor],
        labels: List[torch.LongTensor],
        texts: List[str],
        texts_labels: List[str],
    ):
        """
        Check if the input data is valid for training.

        Args:
            input_ids (List[`torch.LongTensor`]):
                List of tensors containing the input_ids
            attention_mask (List[`torch.LongTensor`]):
                List of tensors containing the attention_mask
            labels (List[`torch.FloatTensor`]):
                List of tensors containing the labels
            texts (List[`str`]):
                List of string containing the text input.
            texts_labels (List[`str`]):
                List of string containing the text labels.
        Returns:
            `tuple`: The input data.
        """
        if texts is None:
            if attention_mask is None:
                for name, tensor_list in zip(["input_ids", "labels"], [input_ids, labels]):
                    if not isinstance(tensor_list, list):
                        raise ValueError(f"{name} must be a list of tensors - got {type(tensor_list)}")
                    if not isinstance(tensor_list[0], torch.Tensor):
                        raise ValueError(f"Elements in {name} must be tensors - got {type(tensor_list[0])}")
            else:
                for name, tensor_list in zip(
                    ["input_ids", "attention_mask", "labels"], [input_ids, attention_mask, labels]
                ):
                    if not isinstance(tensor_list, list):
                        raise ValueError(f"{name} must be a list of tensors - got {type(tensor_list)}")
                    if not isinstance(tensor_list[0], torch.Tensor):
                        raise ValueError(f"Elements in {name} must be tensors - got {type(tensor_list[0])}")
        else:
            if not isinstance(texts, list):
                raise ValueError(f"'text' must be a list of strings - got {type(texts)}")
            if not isinstance(texts[0], str):
                raise ValueError(f"Elements in 'text' must be strings - got {type(texts[0])}")
            if texts_labels is not None:
                if not isinstance(texts_labels, list):
                    raise ValueError(f"'text_labels' must be a list of strings - got {type(texts_labels)}")
                if not isinstance(texts_labels[0], str):
                    raise ValueError(f"Elements in 'text_labels' must be strings - got {type(texts_labels[0])}")

        return input_ids, attention_mask, labels, texts, texts_labels

    @PPODecorators.empty_device_cache()
    def step(
        self,
        input_ids: Optional[List[torch.LongTensor]] = None,
        attention_mask: Optional[List[torch.LongTensor]] = None,
        labels: Optional[List[torch.LongTensor]] = None,
        texts: Optional[List[str]] = None,
        texts_labels: Optional[List[str]] = None,
    ):
        """
        Run an optimisation step given a list of input_ids, attention_mask, and labels or a list of text and text_labels.
        Args:
            input_ids (List[`torch.LongTensor`]):
                List of tensors containing the input_ids (if not provided, text will be used)
            attention_mask (List[`torch.LongTensor`], , *optional*):
                List of tensors containing the attention_mask
            labels (List[`torch.FloatTensor`], *optional*):
                List of tensors containing the labels (if set to None, will default to input_ids)
            texts (List[`str`], *optional*):
                List of strings containing the text input (if not provided, input_ids will directly be used)
            texts_labels (List[`str`], *optional*):
                List of strings containing the text labels (if set to None, will default to text)
        Returns:
            `dict[str, Any]`: A summary of the training statistics
        """
        self.model.train()

        if self.state.global_step == 0:
            self.tr_loss = torch.tensor(0.0).to(self.args.device)
            self._globalstep_last_logged = self.state.global_step

        if input_ids is None and texts is None:
            raise ValueError("Step should include `input_ids` or `texts` as keyword arguments.")
        elif input_ids is not None and texts is not None:
            warnings.warn(
                "Both 'input_ids' and 'texts' are provided. 'input_ids' will be overwritten using inputs provided by the 'texts' keyword argument."
            )

        if labels is None and texts_labels is None and self.is_encoder_decoder:
            raise ValueError(
                "No 'labels' or 'text_labels' are provided. When using an encoder-decoder architecture, 'labels' or 'text_labels' must be passed."
            )

        input_ids, attention_mask, labels, texts, texts_labels = self._step_safety_checker(
            input_ids, attention_mask, labels, texts, texts_labels
        )

        if texts is not None:
            model_inputs = self.tokenizer(
                texts, max_length=self.max_length, truncation=True, padding=True, return_tensors="pt"
            )

            input_ids, attention_mask = model_inputs["input_ids"], model_inputs["attention_mask"]

        if texts_labels is not None:
            labels = self.tokenizer(
                texts, max_length=self.max_length, truncation=True, padding=True, return_tensors="pt"
            )["input_ids"]

        if labels is None:
            warnings.warn("No labels are provided. Setting labels to input_ids")
            labels = input_ids

        model_inputs = self.prepare_model_inputs(input_ids, attention_mask, labels)

        model_inputs_names = list(model_inputs.keys())

        batch_dict = {}
        batch_dict.update(model_inputs)

        def collator(data):
            return_dict = dict()
            for key in data[0]:
                if key in ["input_ids", "attention_mask", "labels"]:
                    return_dict[key] = torch.stack([d[key] for d in data]).to(self.model.device)
            return return_dict

        batch_data = Dataset.from_dict(batch_dict)
        batch_data.set_format("torch")

        step_dataloader = DataLoader(
            batch_data,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True,
            collate_fn=collator,
        )

        for _, batch in enumerate(step_dataloader):
            with self.accelerator.accumulate(self.model):
                model_inputs = {k: batch[k] for k in model_inputs_names}
                loss = self.compute_loss(self.model, model_inputs)

                if self.args.n_gpu > 1:
                    loss = loss.mean()

                tr_loss_step = loss.detach()

                self.accelerator.backward(loss)

                if self.accelerator.sync_gradients and self.args.max_grad_norm is not None:
                    self.accelerator.clip_grad_norm_(
                        self.model.parameters(),
                        self.args.max_grad_norm,
                    )

                self.optimizer.step()
                self.optimizer.zero_grad()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

                self.state.global_step += 1

                # update stats etc
                self.tr_loss += tr_loss_step

                self._maybe_log_save_evaluate()

    def _maybe_log_save_evaluate(self):
        # check if eval is required
        if self.args.eval_steps is not None:
            if self.state.global_step % self.args.eval_steps == 0 and self.state.global_step != 0:
                self.evaluate(self.eval_dataset)

        # check if logging is required
        if self.args.logging_steps is not None:
            if self.state.global_step % self.args.logging_steps == 0 and self.state.global_step != 0:
                logs: Dict[str, float] = {}

                tr_loss_scalar = self._nested_gather(self.tr_loss).mean().item()

                # reset tr_loss to zero
                self.tr_loss -= self.tr_loss

                logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
                logs["learning_rate"] = self._get_learning_rate()

                self._globalstep_last_logged = self.state.global_step

                self.log(logs)
