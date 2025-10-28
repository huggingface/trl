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

import contextlib
import logging
import os
import re
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Union

import torch
import torch.nn as nn
import transformers
from accelerate import PartialState
from accelerate.logging import get_logger
from datasets import Dataset, IterableDataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from transformers.data.data_collator import DataCollatorMixin
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction
from transformers.utils import is_peft_available

from ..data_utils import is_conversational
from ..models import clone_chat_template, get_act_offloading_ctx_manager, prepare_peft_model
from .base_trainer import BaseTrainer
from .reward_config import RewardConfig
from .utils import disable_dropout_in_model, get_config_model_id, pad, remove_none_values


if is_peft_available():
    from peft import PeftConfig, PeftModel


logger = get_logger(__name__)


# AutoModelForSequenceClassification adds a new classification head when loading a CausalLM. That head is randomly
# initialized and triggers a harmless warning about uninitialized weights. We suppress just that specific warning to
# avoid confusing users.
@contextmanager
def suppress_from_pretrained_warning(logger: logging.Logger):
    pattern = re.compile(
        r"^Some weights of \S+ were not initialized from the model checkpoint at \S+ and are newly initialized: "
        r"\[.*\]\nYou should probably TRAIN this model on a down-stream task to be able to use it for predictions and "
        r"inference\.$"
    )

    class _Filter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return not pattern.search(record.getMessage())

    f = _Filter()
    logger.addFilter(f)
    try:
        yield
    finally:
        logger.removeFilter(f)


@dataclass
class DataCollatorForPreference(DataCollatorMixin):
    """
    Data collator used for preference data. Inputs are dynamically padded to the maximum length of a batch.

    This collator expects each example in the input list to be a dictionary containing the `"chosen_input_ids"` and
    `"rejected_input_ids"` keys. The collator returns a dictionary containing the following keys:
    - `"input_ids"`: Tensor of input IDs, padded to the maximum length of the batch. The first half of the batch
        corresponds to the `"chosen_input_ids"` and the second half to the `"rejected_input_ids"`.
    - `"attention_mask"`: Tensor of attention mask, padded to the maximum length of the batch.

    Optionally, the examples can contain a `"margin"` key, in which case the returned dictionary will also contain a
    `"margin"` key with a tensor of margins.

    Args:
        pad_token_id (`int`):
            Token ID to use for padding.
        pad_to_multiple_of (`int`, *optional*):
            If set, the sequences will be padded to a multiple of this value.
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            Type of Tensor to return. Only `"pt"` is currently supported.

    Examples:
    ```python
    >>> from trl.trainer.reward_trainer import DataCollatorForPreference

    >>> collator = DataCollatorForPreference(pad_token_id=0)
    >>> examples = [
    ...     {"chosen_input_ids": [1, 2, 3], "rejected_input_ids": [4, 5]},
    ...     {"chosen_input_ids": [6, 7], "rejected_input_ids": [8]},
    ... ]
    >>> collator(examples)
    {'input_ids': tensor([[1, 2, 3],
                          [6, 7, 0],
                          [4, 5, 0],
                          [8, 0, 0]]),
     'attention_mask': tensor([[1, 1, 1],
                               [1, 1, 0],
                               [1, 1, 0],
                               [1, 0, 0]])}

    >>> examples = [
    ...     {"chosen_input_ids": [1, 2, 3], "rejected_input_ids": [4, 5], "margin": 0.5},
    ...     {"chosen_input_ids": [6, 7], "rejected_input_ids": [8], "margin": 0.0},
    ... ]
    >>> collator(examples)
    {'input_ids': tensor([[1, 2, 3],
                          [6, 7, 0],
                          [4, 5, 0],
                          [8, 0, 0]]),
     'attention_mask': tensor([[1, 1, 1],
                               [1, 1, 0],
                               [1, 1, 0],
                               [1, 0, 0]]),
     'margin': tensor([0.5, 0.0])}
    ```
    """

    pad_token_id: int
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def torch_call(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        # Convert to tensor
        chosen_input_ids = [torch.tensor(example["chosen_input_ids"]) for example in examples]
        rejected_input_ids = [torch.tensor(example["rejected_input_ids"]) for example in examples]
        if "margin" in examples[0]:
            margins = torch.tensor([example["margin"] for example in examples], dtype=torch.float)
        input_ids = chosen_input_ids + rejected_input_ids
        attention_mask = [torch.ones_like(ids) for ids in input_ids]

        output = {}

        # Pad
        output["input_ids"] = pad(
            input_ids,
            padding_value=self.pad_token_id,
            padding_side="right",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        output["attention_mask"] = pad(
            attention_mask,
            padding_value=0,
            padding_side="right",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        if "margin" in examples[0]:
            output["margin"] = margins
        return output


class RewardTrainer(BaseTrainer):
    """
    Trainer for Outcome-supervised Reward Models (ORM).

    This class is a wrapper around the [`~transformers.Trainer`] class and inherits all of its attributes and methods.

    Example:

    ```python
    from trl import RewardTrainer
    from datasets import load_dataset

    dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")

    trainer = RewardTrainer(model="Qwen/Qwen2.5-0.5B-Instruct", train_dataset=dataset)
    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or a
              path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
              using `AutoModelForSequenceClassification.from_pretrained` with the keyword arguments in
              `args.model_init_kwargs`.
            - A sequence classification [`~transformers.PreTrainedModel`] object.
        args ([`RewardConfig`], *optional*):
            Configuration for this trainer. If `None`, a default configuration is used.
        data_collator ([`~transformers.DataCollator`], *optional*):
            Function to use to form a batch from a list of elements of the processed `train_dataset` or `eval_dataset`.
            Will default to [`~trainer.reward_trainer.DataCollatorForPreference`].
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. This trainer supports [preference](#preference) type (both implicit and
            explicit prompt). The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).

            The trainer also supports processed datasets (tokenized) as long as they contain an `chosen_input_ids` and
            `rejected_input_ids` fields.
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*):
            Tokenizer used to process the data. If `None`, the tokenizer is loaded from the model's name with
            [`~transformers.AutoTokenizer.from_pretrained`]. A padding token, `processing_class.pad_token`, must be
            set. If the processing class has not set a padding token, `processing_class.eos_token` will be used as the
            default.
        compute_metrics (`Callable[[EvalPrediction], dict]`, *optional*):
            The function that will be used to compute metrics at evaluation. Must take a
            [`~transformers.EvalPrediction`] and return a dictionary string to metric values. When passing
            [`RewardConfig`] with `batch_eval_metrics` set to `True`, your `compute_metrics` function must take a
            boolean `compute_result` argument. This will be triggered after the last eval batch to signal that the
            function needs to calculate and return the global summary statistics rather than accumulating the
            batch-level statistics.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks detailed
            in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of `AdamW` on your
            model and a scheduler given by [`~transformers.get_linear_schedule_with_warmup`] controlled by `args`.
        optimizer_cls_and_kwargs (`tuple[Type[torch.optim.Optimizer], Dict[str, Any]]`, *optional*):
            A tuple containing the optimizer class and keyword arguments to use. Overrides `optim` and `optim_args` in
            `args`. Incompatible with the `optimizers` argument.

            Unlike `optimizers`, this argument avoids the need to place model parameters on the correct devices before
            initializing the Trainer.
        preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`, *optional*):
            A function that preprocess the logits right before caching them at each evaluation step. Must take two
            tensors, the logits and the labels, and return the logits once processed as desired. The modifications made
            by this function will be reflected in the predictions received by `compute_metrics`.

            Note that the labels (second parameter) will be `None` if the dataset does not have them.
        peft_config ([`~peft.PeftConfig`], *optional*):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped. Note that if the loaded
            model is a causal LM, it's highly recommended to set `modules_to_save=["score"]` in the PEFT configuration
            to ensure that the reward head is properly trained.
    """

    _tag_names = ["trl", "reward-trainer"]
    _name = "Reward"
    _template_file = "rm_model_card.md"

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        args: Optional[RewardConfig] = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], dict]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        optimizer_cls_and_kwargs: Optional[tuple[type[torch.optim.Optimizer], dict[str, Any]]] = None,
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        peft_config: Optional["PeftConfig"] = None,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else get_config_model_id(model.config)
            model_name = model_name.split("/")[-1]
            args = RewardConfig(f"{model_name}-Reward")

        # Model
        model_init_kwargs = args.model_init_kwargs or {}
        if isinstance(model, str):
            model_id = model
            dtype = model_init_kwargs.get("dtype")
            if isinstance(dtype, torch.dtype) or dtype == "auto" or dtype is None:
                pass  # dtype is already a torch.dtype or "auto" or None
            elif isinstance(dtype, str) and dtype in ["bfloat16", "float16", "float32"]:
                model_init_kwargs["dtype"] = getattr(torch, dtype)
            else:
                raise ValueError(
                    "Invalid `dtype` passed to `RewardConfig`. Expected either 'auto' or a string representing "
                    f"a valid `torch.dtype` (e.g., 'float32'), but got {dtype}."
                )
            with suppress_from_pretrained_warning(transformers.modeling_utils.logger):
                model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=1, **model_init_kwargs)
        else:
            model_id = get_config_model_id(model.config)
            if args.model_init_kwargs is not None:
                logger.warning(
                    "You passed `model_init_kwargs` to the `RewardConfig`, but your model is already instantiated. "
                    "The `model_init_kwargs` will be ignored."
                )

        # Processing class
        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(model_id)

        # Handle pad token for processors or tokenizers
        if args.eos_token is not None:
            eos_token = args.eos_token
            eos_token_id = processing_class.convert_tokens_to_ids(eos_token)
            if eos_token_id is None:
                raise ValueError(
                    f"The specified `eos_token` ('{eos_token}') is not found in the vocabulary of the given "
                    f"`processing_class` ({processing_class.__class__.__name__}). Ensure that the `eos_token` exists "
                    "in the vocabulary before using it as an EOS token."
                )
            processing_class.eos_token_id = eos_token_id

        if args.chat_template_path is not None:
            if os.path.isfile(args.chat_template_path) and args.chat_template_path.endswith((".jinja", ".j2")):
                with open(args.chat_template_path, encoding="utf-8") as chat_template_file:
                    processing_class.chat_template = chat_template_file.read()
                added_tokens = []
            else:
                model, processing_class, added_tokens = clone_chat_template(
                    model, processing_class, args.chat_template_path
                )
        else:
            added_tokens = []

        # PEFT configuration and model wrapping
        if peft_config is not None:
            if added_tokens:
                # Ensure that the added tokens are trainable
                if peft_config.trainable_token_indices is None:
                    peft_config.trainable_token_indices = {"embed_tokens": added_tokens}
                elif "embed_tokens" not in peft_config.trainable_token_indices:
                    peft_config.trainable_token_indices["embed_tokens"] = added_tokens
                else:
                    peft_config.trainable_token_indices["embed_tokens"].extend(added_tokens)

                # Ensure that the lm_head is trainable
                if peft_config.modules_to_save is None or "lm_head" not in peft_config.modules_to_save:
                    logger.warning(
                        "Cloning chat template added new tokens to the tokenizer, but 'lm_head' is not in PEFT's "
                        "`modules_to_save`. As a result, the model may not learn to generate outputs with these new "
                        "tokens, leading to degraded generation quality. To fix this, add "
                        "`modules_to_save=['lm_head']` to your PEFT configuration."
                    )

                    if peft_config.modules_to_save is None:
                        peft_config.modules_to_save = ["lm_head"]
                    else:
                        peft_config.modules_to_save.append("lm_head")

        if peft_config is not None or (is_peft_available() and isinstance(model, PeftModel)):
            model = prepare_peft_model(model, peft_config, args)

        # Disable dropout in the model
        if args.disable_dropout:
            disable_dropout_in_model(model)

        # Pad token (needed for SequenceClassification models)
        # If not provided, use the one from the processing class or the eos token if the processing class does not have
        # a pad token.
        pad_token = args.pad_token or processing_class.pad_token or processing_class.eos_token
        pad_token_id = processing_class.convert_tokens_to_ids(pad_token)
        if pad_token_id is None:
            raise ValueError(
                f"The specified `pad_token` ('{pad_token}') is not found in the vocabulary of the given "
                f"`processing_class` ({processing_class.__class__.__name__}). Ensure that the `pad_token` exists "
                "in the vocabulary before using it as a padding token."
            )
        model.config.pad_token_id = pad_token_id
        processing_class.pad_token_id = pad_token_id

        # Data collator
        if data_collator is None:
            data_collator = DataCollatorForPreference(
                pad_token_id=pad_token_id,
                pad_to_multiple_of=args.pad_to_multiple_of,
            )

        # Dataset
        train_dataset = self._prepare_dataset(train_dataset, processing_class, args, "train")
        if eval_dataset is not None:
            if isinstance(eval_dataset, dict):
                eval_dataset = {
                    key: self._prepare_dataset(dataset, processing_class, args, key)
                    for key, dataset in eval_dataset.items()
                }
            else:
                eval_dataset = self._prepare_dataset(eval_dataset, processing_class, args, "eval")

        # Initialize the metrics
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._total_train_tokens = 0

        # Initialize the Trainer. Parent class will handle:
        # - DeepSpeed configuration (through create_accelerator_and_postprocess)
        # - FSDP setup
        # - Distributed training setup
        # - Optimizer and scheduler creation

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            optimizer_cls_and_kwargs=optimizer_cls_and_kwargs,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        # During evaluation, Trainer calls compute_loss() only if can_return_loss is True and label_names is empty.
        self.can_return_loss = True
        self.label_names = []

        # Initialize activation offloading context
        if self.args.activation_offloading:
            self.maybe_activation_offload_context = get_act_offloading_ctx_manager(model=self.model)
        else:
            self.maybe_activation_offload_context = contextlib.nullcontext()

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        self.aux_loss_enabled = getattr(model.config, "output_router_logits", False)

    def _prepare_dataset(
        self,
        dataset: Union[Dataset, IterableDataset],
        processing_class: PreTrainedTokenizerBase,
        args: RewardConfig,
        dataset_name: str,
    ) -> Union[Dataset, IterableDataset]:
        # Tabular backends like Arrow/Parquet insert `None` for mismatched keys in nested structures. Clean them from
        # sampled data.
        if isinstance(dataset, Dataset):  # IterableDataset does not support `with_transform`
            dataset = dataset.with_transform(remove_none_values)

        # If the dataset is already preprocessed (tokenized), skip the processing steps.
        column_names = list(next(iter(dataset)).keys())
        is_processed = "chosen_input_ids" in column_names and "rejected_input_ids" in column_names

        # Build the kwargs for the `map` function
        map_kwargs = {}
        if isinstance(dataset, Dataset):  # IterableDataset does not support num_proc
            map_kwargs["num_proc"] = args.dataset_num_proc

        with PartialState().main_process_first():
            if not is_processed:
                # Add EOS token to the end of the sequences if needed
                first_example = next(iter(dataset))
                if not is_conversational(first_example):
                    if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                        map_kwargs["desc"] = f"Adding EOS to {dataset_name} dataset"

                    def add_eos(example, eos_token):
                        if not example["chosen"].endswith(eos_token):
                            example["chosen"] = example["chosen"] + eos_token
                        if "rejected" in example and not example["rejected"].endswith(eos_token):
                            example["rejected"] = example["rejected"] + eos_token
                        return example

                    dataset = dataset.map(
                        add_eos,
                        fn_kwargs={"eos_token": processing_class.eos_token},
                        **map_kwargs,
                    )

                # Tokenize the dataset
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Tokenizing {dataset_name} dataset"

                def tokenize_fn(example, processing_class):
                    if "prompt" in example:  # explicit prompt case
                        example["chosen"] = example["prompt"] + example["chosen"]
                        example["rejected"] = example["prompt"] + example["rejected"]

                    if is_conversational(example):
                        chosen_input_ids = processing_class.apply_chat_template(
                            example["chosen"],
                            tools=example.get("tools"),
                            **example.get("chat_template_kwargs", {}),
                        )
                        rejected_input_ids = processing_class.apply_chat_template(
                            example["rejected"],
                            tools=example.get("tools"),
                            **example.get("chat_template_kwargs", {}),
                        )
                        output = {"chosen_input_ids": chosen_input_ids, "rejected_input_ids": rejected_input_ids}
                    else:
                        output = {
                            "chosen_input_ids": processing_class(text=example["chosen"])["input_ids"],
                            "rejected_input_ids": processing_class(text=example["rejected"])["input_ids"],
                        }
                    return output

                dataset = dataset.map(tokenize_fn, fn_kwargs={"processing_class": processing_class}, **map_kwargs)

            # Filter samples that are longer than `max_length`
            if args.max_length is not None:
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Filtering {dataset_name} >{args.max_length} tokens"
                dataset = dataset.filter(
                    lambda example: len(example["chosen_input_ids"]) <= args.max_length
                    and len(example["rejected_input_ids"]) <= args.max_length,
                    **map_kwargs,
                )

        return dataset

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs (usually, "input_ids"
        # and "attention_mask").
        if self._signature_columns is None:
            self._signature_columns = ["chosen_input_ids", "rejected_input_ids", "margin"]

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[torch.Tensor] = None,
    ):
        """
        Compute training loss and additionally compute token accuracies
        """
        mode = "train" if self.model.training else "eval"

        # If not set, defaults from model config and may warn since cache isn't compatible with gradient checkpointing
        inputs["use_cache"] = False
        outputs = model(**inputs)

        # Split the rewards into chosen and rejected
        rewards_chosen, rewards_rejected = torch.chunk(outputs.logits.squeeze(-1), chunks=2)

        # Calculate loss, optionally modulate with margin
        if "margin" in inputs:
            loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected - inputs["margin"]).mean()
        else:
            loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()

        if self.args.center_rewards_coefficient is not None:
            loss += self.args.center_rewards_coefficient * torch.mean((rewards_chosen + rewards_rejected) ** 2)

        if mode == "train":
            num_tokens_in_batch = self.accelerator.gather_for_metrics(inputs["attention_mask"].sum()).sum().item()
            self._total_train_tokens += num_tokens_in_batch
        self._metrics[mode]["num_tokens"] = [self._total_train_tokens]

        # Compute min, mean, max, accuracy and margin
        with torch.no_grad():
            all_rewards = self.accelerator.gather(outputs.logits)
            self._metrics[mode]["min_reward"].append(all_rewards.min().item())
            self._metrics[mode]["mean_reward"].append(all_rewards.mean().item())
            self._metrics[mode]["max_reward"].append(all_rewards.max().item())

            mean_accuracy = (rewards_chosen > rewards_rejected).float().mean()
            mean_accuracy = self.accelerator.gather_for_metrics(mean_accuracy).mean().item()
            self._metrics[mode]["accuracy"].append(mean_accuracy)

            mean_margin = (rewards_chosen - rewards_rejected).mean()
            mean_margin = self.accelerator.gather_for_metrics(mean_margin).mean()
            self._metrics[mode]["margin"].append(mean_margin.item())

        return (loss, outputs) if return_outputs else loss

    # Override training step to add activation offloading context.
    def training_step(self, *args, **kwargs):
        with self.maybe_activation_offload_context:
            return super().training_step(*args, **kwargs)

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        mode = "train" if self.model.training else "eval"
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs.update(metrics)
        super().log(logs, start_time)
        self._metrics[mode].clear()

    # Ensure the model card is saved along with the checkpoint
    def _save_checkpoint(self, model, trial):
        if self.args.hub_model_id is None:
            model_name = Path(self.args.output_dir).name
        else:
            model_name = self.args.hub_model_id.split("/")[-1]
        self.create_model_card(model_name=model_name)
        super()._save_checkpoint(model, trial)
