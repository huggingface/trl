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
import dataclasses
import os
import warnings
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar, Union

import torch
import torch.nn as nn
from accelerate import PartialState
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BaseImageProcessor,
    DataCollator,
    FeatureExtractionMixin,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    Trainer,
    TrainingArguments,
    is_wandb_available,
)
from transformers.data.data_collator import DataCollatorMixin
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction
from transformers.utils import is_peft_available

from ..data_utils import (
    is_conversational,
    is_conversational_from_value,
    maybe_convert_to_chatml,
    pack_dataset,
    truncate_dataset,
)
from ..models import clone_chat_template, get_act_offloading_ctx_manager
from .sft_config import SFTConfig
from .utils import generate_model_card, get_comet_experiment_url, pad, peft_module_casting_to_bf16


if is_peft_available():
    import peft
    from peft import PeftConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

if is_wandb_available():
    import wandb


TListOrMapping = TypeVar("TListOrMapping", list, Mapping)


def remove_none_values(example: TListOrMapping) -> TListOrMapping:
    """
    Recursively removes entries with `None` values from a nested structure (list or dictionary).

    Args:
        example (`list` or `Mapping`):
            Input nested structure (list or dictionary) from which to remove `None`.

    Example:
    ```python
    >>> [{
    ...     "a": {"aa": None,
    ...           "ab": 1},
    ...     "b": "my_string",
    ... }]
    >>> remove_none_values(example)
    [{'a': {'ab': 1}, 'b': 'my_string'}]
    ```
    """
    if isinstance(example, list):
        return [remove_none_values(value) if isinstance(value, (dict, list)) else value for value in example]
    elif isinstance(example, Mapping):
        return {
            key: remove_none_values(value) if isinstance(value, (dict, list)) else value
            for key, value in example.items()
            if value is not None
        }
    else:
        raise TypeError("Input must be a list or a dictionary.")


@dataclass
class DataCollatorForLanguageModeling(DataCollatorMixin):
    """
    Data collator used for language modeling data. Inputs are dynamically padded to the maximum length of a batch.

    This collator expects each example in the input list to be a dictionary containing at least the `"input_ids"` key.
    If the input contains a `"completion_mask"`, it is used to set the labels to `-100` for tokens that are not in the
    completion. If `"assistant_masks"` are present, they are used to set the labels to `-100` for tokens that are not
    in the assistant part of the sequence. The collator returns a dictionary containing the following keys:
    - `"input_ids"`: Tensor of input IDs, padded to the maximum length of the batch.
    - `"attention_mask"`: Tensor of attention mask, padded to the maximum length of the batch.
    - `"position_ids"`: Tensor of position IDs, padded to the maximum length of the batch.
    - `"labels"`: Tensor of labels, padded to the maximum length of the batch. If `completion_only_loss` is set to
    `True`, tokens that are not in the completion are set to -100. If `assistant_masks` are present, tokens that are
    not in the assistant part of the sequence are set to -100.

    Args:
        pad_token_id (`int`):
            Token ID to use for padding.
        completion_only_loss (`bool`, *optional*, defaults to `True`):
            When the input contains a completion mask (`completion_mask`), the labels are set to -100 for the tokens
            that are no in the completion.
        padding_free (`bool`, *optional*, defaults to `False`):
            If set to `True`, the sequences will be flattened into a single sequence, and the position IDs will be
            generated accordingly. The attention mask will be set to 1 for all tokens.
        pad_to_multiple_of (`int` or `None`, *optional*, defaults to `None`):
            If set, the sequences will be padded to a multiple of this value.
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            Type of Tensor to return. Only `"pt"` is currently supported.

    Examples:
    ```python
    >>> from trl import DataCollatorForLanguageModeling

    >>> collator = DataCollatorForLanguageModeling(pad_token_id=0)
    >>> examples = [{"input_ids": [1, 2, 3]}, {"input_ids": [4, 5]}]
    >>> collator(examples)
    {'input_ids': tensor([[  1,  2,  3],
                          [  4,  5,  0]]),
     'attention_mask': tensor([[  1,  1,  1],
                               [  1,  1,  0]]),
     'position_ids': tensor([[0, 1, 2],
                             [0, 1, 0]]),
     'labels': tensor([[   1,    2,    3],
                       [   4,    5, -100]])}

    >>> # With completion mask
    >>> examples = [
    ...     {"input_ids": [1, 2, 3], "completion_mask": [0, 1, 1]},
    ...     {"input_ids": [4, 5], "completion_mask": [0, 1]},
    ... ]
    >>> collator(examples)
    {'input_ids': tensor([[  1,  2,  3],
                          [  4,  5,  0]]),
     'attention_mask': tensor([[  1,  1,  1],
                               [  1,  1,  0]]),
     'position_ids': tensor([[0, 1, 2],
                             [0, 1, 0]]),
     'labels': tensor([[-100,    2,    3],
                       [-100,    5, -100]])}

    >>> # With padding_free
    >>> collator = DataCollatorForLanguageModeling(pad_token_id=0, padding_free=True)
    >>> collator(examples)
    {'input_ids': tensor([[ 1, 2, 3, 4, 5]]),
     'attention_mask': tensor([[1, 1, 1, 1, 1]]),
     'position_ids': tensor([[0, 1, 2, 0, 1]]),
     'labels': tensor([[1, 2, 3, 4, 5]])}
    ```
    """

    pad_token_id: int
    completion_only_loss: bool = True
    padding_free: bool = False
    return_position_ids: bool = True
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def torch_call(self, examples: list[Union[list[int], Any, dict[str, Any]]]) -> dict[str, Any]:
        # Convert to tensor
        input_ids = [torch.tensor(example["input_ids"]) for example in examples]

        # Check if we have meaningful seq_lengths from packing (restarting sequences)
        has_packed_position_ids = self.return_position_ids and "seq_lengths" in examples[0] and self.padding_free

        # For packing with position_ids, we should NOT create attention_mask as it causes
        # flash attention to ignore position_ids and compute wrong cu_seq_lens from the all-1s mask
        if not has_packed_position_ids:
            attention_mask = [torch.ones_like(input_ids) for input_ids in input_ids]

        if self.return_position_ids:
            if "seq_lengths" in examples[0]:
                position_ids = self._convert_seq_lengths_to_position_ids(
                    [example["seq_lengths"] for example in examples]
                )
            else:
                position_ids = [torch.arange(len(ids)) for ids in input_ids]
        if "labels" in examples[0]:
            labels = [torch.tensor(example["labels"]) for example in examples]
        else:
            labels = [torch.tensor(example["input_ids"]) for example in examples]
        if self.completion_only_loss and "completion_mask" in examples[0]:
            completion_mask = [torch.tensor(example["completion_mask"]) for example in examples]
        if "assistant_masks" in examples[0]:
            assistant_masks = [torch.tensor(example["assistant_masks"]) for example in examples]

        # Pad
        output = {}
        if self.padding_free:
            output["input_ids"] = torch.cat(input_ids, dim=0).unsqueeze(0)
            if not has_packed_position_ids:
                output["attention_mask"] = torch.cat(attention_mask, dim=0).unsqueeze(0)
            if self.return_position_ids:
                output["position_ids"] = torch.cat(position_ids, dim=0).unsqueeze(0)
            output["labels"] = torch.cat(labels, dim=0).unsqueeze(0)
            if self.completion_only_loss and "completion_mask" in examples[0]:
                completion_mask = torch.cat(completion_mask, dim=0).unsqueeze(0)
                output["labels"][completion_mask == 0] = -100
            if "assistant_masks" in examples[0]:
                assistant_masks = torch.cat(assistant_masks, dim=0).unsqueeze(0)
                output["labels"][assistant_masks == 0] = -100
        else:
            output["input_ids"] = pad(
                input_ids,
                padding_value=self.pad_token_id,
                padding_side="right",
                pad_to_multiple_of=self.pad_to_multiple_of,
            )
            output["attention_mask"] = pad(
                attention_mask, padding_value=0, padding_side="right", pad_to_multiple_of=self.pad_to_multiple_of
            )
            if self.return_position_ids:
                output["position_ids"] = pad(
                    position_ids, padding_value=0, padding_side="right", pad_to_multiple_of=self.pad_to_multiple_of
                )
            output["labels"] = pad(
                labels, padding_value=-100, padding_side="right", pad_to_multiple_of=self.pad_to_multiple_of
            )
            if self.completion_only_loss and "completion_mask" in examples[0]:
                completion_mask = pad(
                    completion_mask, padding_value=0, padding_side="right", pad_to_multiple_of=self.pad_to_multiple_of
                )
                output["labels"][completion_mask == 0] = -100  # mask everything that is not in the completion
            if "assistant_masks" in examples[0]:
                assistant_masks = pad(
                    assistant_masks, padding_value=0, padding_side="right", pad_to_multiple_of=self.pad_to_multiple_of
                )
                output["labels"][assistant_masks == 0] = -100
        return output

    @staticmethod
    def _convert_seq_lengths_to_position_ids(batch_seq_lengths: list[list[int]]) -> list[torch.Tensor]:
        example_lengths = [sum(seq_lengths) for seq_lengths in batch_seq_lengths]
        batch_seq_lengths = torch.tensor(
            [seq_length for seq_lengths in batch_seq_lengths for seq_length in seq_lengths]
        )
        position_ids = torch.ones(sum(example_lengths), dtype=batch_seq_lengths.dtype)
        position_ids[0] = 0
        position_ids[batch_seq_lengths[:-1].cumsum(0)] = -(batch_seq_lengths[:-1] - 1)
        position_ids = position_ids.cumsum(0)
        return list(position_ids.split(example_lengths))


class SFTTrainer(Trainer):
    """
    Trainer for Supervised Fine-Tuning (SFT) method.

    This class is a wrapper around the [`transformers.Trainer`] class and inherits all of its attributes and methods.

    Example:

    ```python
    from datasets import load_dataset
    from trl import SFTTrainer

    dataset = load_dataset("roneneldan/TinyStories", split="train[:1%]")

    trainer = SFTTrainer(model="Qwen/Qwen2-0.5B-Instruct", train_dataset=dataset)
    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or a
              path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
              using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keyword arguments in
              `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        args ([`SFTConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        data_collator (`DataCollator`, *optional*):
            Function to use to form a batch from a list of elements of the processed `train_dataset` or `eval_dataset`.
            Will default to a custom [`DataCollatorForLanguageModeling`].
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. SFT supports both [language modeling](#language-modeling) type and
            [prompt-completion](#prompt-completion) type. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).

            The trainer also supports processed datasets (tokenized) as long as they contain an `input_ids` field.
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], [`~transformers.BaseImageProcessor`], [`~transformers.FeatureExtractionMixin`] or [`~transformers.ProcessorMixin`], *optional*, defaults to `None`):
            Processing class used to process the data. If `None`, the processing class is loaded from the model's name
            with [`~transformers.AutoTokenizer.from_pretrained`].
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks detailed
            in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        optimizer_cls_and_kwargs (`Tuple[Type[torch.optim.Optimizer], Dict[str, Any]]`, *optional*, defaults to `None`):
            A tuple containing the optimizer class and keyword arguments to use. Overrides `optim` and `optim_args` in
            `args`. Incompatible with the `optimizers` argument.

            Unlike `optimizers`, this argument avoids the need to place model parameters on the correct devices before
            initializing the Trainer.
        preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`, *optional*, defaults to `None`):
            A function that preprocess the logits right before caching them at each evaluation step. Must take two
            tensors, the logits and the labels, and return the logits once processed as desired. The modifications made
            by this function will be reflected in the predictions received by `compute_metrics`.

            Note that the labels (second parameter) will be `None` if the dataset does not have them.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
        formatting_func (`Optional[Callable]`):
            Formatting function applied to the dataset before tokenization. Applying the formatting function explicitly
            converts the dataset into a [language modeling](#language-modeling) type.
    """

    _tag_names = ["trl", "sft"]

    def __init__(
        self,
        model: Union[str, nn.Module, PreTrainedModel],
        args: Optional[Union[SFTConfig, TrainingArguments]] = None,
        data_collator: Optional[DataCollator] = None,  # type: ignore
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        processing_class: Optional[
            Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin]
        ] = None,
        compute_loss_func: Optional[Callable] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], dict]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        optimizer_cls_and_kwargs: Optional[tuple[type[torch.optim.Optimizer], dict[str, Any]]] = None,
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        peft_config: Optional["PeftConfig"] = None,
        formatting_func: Optional[Callable[[dict], str]] = None,
    ):
        # Args
        model_id = model if isinstance(model, str) else model.config._name_or_path
        if args is None:
            model_name = model_id.split("/")[-1]
            args = SFTConfig(f"{model_name}-SFT")
        elif isinstance(args, TrainingArguments) and not isinstance(args, SFTConfig):
            dict_args = args.to_dict()
            dict_args["hub_token"] = args.hub_token  # to_dict hides the hub_token
            dict_args.pop("push_to_hub_token")
            args = SFTConfig(**dict_args)

        # Handle the tokenizer
        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(model_id)

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

        # Model
        if args.model_init_kwargs is not None and not isinstance(model, str):
            warnings.warn(
                "You passed model_init_kwargs to the `SFTConfig`, but your model is already instantiated. "
                "The `model_init_kwargs` will be ignored."
            )
        if isinstance(model, str):
            model = self._create_model_from_path(model, args)

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
                    warnings.warn(
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
            model = self._prepare_peft_model(model, peft_config, args)

        # Data collator
        # BFD packing requires padding-free mode; otherwise, the collator outputs padded attention masks, causing
        # FlashAttention to ignore position_ids and recompute them incorrectly from the padded attention mask.
        self.padding_free = args.padding_free or (args.packing and args.packing_strategy == "bfd")
        if self.padding_free:
            if data_collator is not None:
                raise ValueError("Passing a custom data collator is not supported when using padding-free.")
            if args.packing and args.packing_strategy == "wrapped":
                warnings.warn(
                    "You are passing `padding_free=True` with the 'wrapped' packing strategy, which is not "
                    "recommended. Please refer to the documentation to understand why this is not recommended."
                )
            if model.config._attn_implementation != "flash_attention_2":
                warnings.warn(
                    "Padding-free training is enabled, but the attention implementation is not set to "
                    "'flash_attention_2'. Padding-free training flattens batches into a single sequence, and "
                    "'flash_attention_2' is the only known attention mechanism that reliably supports this. Using "
                    "other implementations may lead to unexpected behavior. To ensure compatibility, set "
                    "`attn_implementation='flash_attention_2'` in the model configuration, or verify that your "
                    "attention mechanism can handle flattened sequences."
                )
            if args.per_device_train_batch_size == 1 and not args.packing:
                warnings.warn(
                    "You are using a per_device_train_batch_size of 1 with padding-free training. Using a batch size "
                    "of 1 anihilate the benefits of padding-free training. Please consider increasing the batch size "
                    "to at least 2."
                )

        dataset_sample = next(iter(train_dataset))
        if args.completion_only_loss is None:
            self.completion_only_loss = "prompt" in dataset_sample
        else:
            self.completion_only_loss = args.completion_only_loss

        if data_collator is None:
            # Get the pad token: if not provided, use the one from the processing class or the eos token
            # if the processing class does not have a pad token.
            pad_token = args.pad_token or processing_class.pad_token or processing_class.eos_token
            pad_token_id = processing_class.convert_tokens_to_ids(pad_token)
            if pad_token_id is None:
                raise ValueError(
                    f"The specified `pad_token` ('{pad_token}') is not found in the vocabulary of the given "
                    f"`processing_class` ({processing_class.__class__.__name__}). Ensure that the `pad_token` exists "
                    "in the vocabulary before using it as a padding token."
                )
            data_collator = DataCollatorForLanguageModeling(
                pad_token_id=pad_token_id,
                completion_only_loss=self.completion_only_loss,
                padding_free=self.padding_free,
                # Using position_ids without flash_attn hurts the training
                return_position_ids=model.config._attn_implementation == "flash_attention_2",
                pad_to_multiple_of=args.pad_to_multiple_of,
            )

        if (
            args.packing
            and args.packing_strategy == "bfd"
            and model.config._attn_implementation != "flash_attention_2"
        ):
            warnings.warn(
                "You are using packing, but the attention implementation is not set to 'flash_attention_2'. Packing "
                "flattens batches into a single sequence, and 'flash_attention_2' is the only known attention "
                "mechanism that reliably supports this. Using other implementations may lead to cross-contamination "
                "between batches. To avoid this, either disable packing by setting `packing=False`, or set "
                "`attn_implementation='flash_attention_2'` in the model configuration."
            )
        if args.assistant_only_loss and not is_conversational(dataset_sample):
            raise ValueError(
                "You set `assistant_only_loss=True`, but the dataset is not conversational. This option is only "
                "supported for conversational datasets."
            )

        # Dataset
        preprocess_dataset = args.dataset_kwargs is None or not args.dataset_kwargs.get("skip_prepare_dataset", False)
        if preprocess_dataset:
            if self.completion_only_loss and formatting_func:
                raise ValueError(
                    "A formatting function was provided while `completion_only_loss=True`, which is incompatible. "
                    "Using a formatter converts the dataset to a language modeling type, conflicting with "
                    "completion-only loss. To resolve this, apply your formatting function before passing the "
                    "dataset, or disable `completion_only_loss` in `SFTConfig`."
                )
            train_dataset = self._prepare_dataset(
                train_dataset, processing_class, args, args.packing, formatting_func, "train"
            )
            if eval_dataset is not None:
                packing = args.packing if args.eval_packing is None else args.eval_packing
                if isinstance(eval_dataset, dict):
                    eval_dataset = {
                        key: self._prepare_dataset(dataset, processing_class, args, packing, formatting_func, key)
                        for key, dataset in eval_dataset.items()
                    }
                else:
                    eval_dataset = self._prepare_dataset(
                        eval_dataset, processing_class, args, packing, formatting_func, "eval"
                    )

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
            compute_loss_func=compute_loss_func,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            optimizer_cls_and_kwargs=optimizer_cls_and_kwargs,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        # Initialize activation offloading context
        if self.args.activation_offloading:
            self.maybe_activation_offload_context = get_act_offloading_ctx_manager(model=self.model)
        else:
            self.maybe_activation_offload_context = contextlib.nullcontext()

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

    def _create_model_from_path(self, model_path: str, args: SFTConfig) -> PreTrainedModel:
        """Creates a model from a path or model identifier."""
        model_init_kwargs = args.model_init_kwargs or {}
        # Handle torch dtype
        torch_dtype = model_init_kwargs.get("torch_dtype")
        if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
            pass  # torch_dtype is already a torch.dtype or "auto" or None
        elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
            torch_dtype = getattr(torch, torch_dtype)
            model_init_kwargs["torch_dtype"] = torch_dtype
        else:
            raise ValueError(
                "Invalid `torch_dtype` passed to `SFTConfig`. Expected either 'auto' or a string representing "
                f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
            )
        # Disable caching if gradient checkpointing is enabled (not supported)
        # if args.gradient_checkpointing:
        #     model_init_kwargs["use_cache"] = False

        # Create model
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_init_kwargs)
        return model

    def _prepare_peft_model(self, model: PreTrainedModel, peft_config: Any, args: SFTConfig) -> PreTrainedModel:
        """Prepares a model for PEFT training."""
        if not is_peft_available():
            raise ImportError("To use PeftModel, you need to install the `peft` library.")

        # Handle quantized models (QLoRA)
        is_qlora = getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False)

        is_sharded_qlora = False
        if getattr(model, "is_loaded_in_4bit", False):
            # Check if model is sharded (FSDP/DS-Zero3)
            for _, param in model.named_parameters():
                if param.__class__.__name__ == "Params4bit":
                    is_sharded_qlora = param.data.device.type in {"cpu", "meta"}
                    break

        # Prepare model for kbit training if needed
        if is_qlora and not is_sharded_qlora:
            model = self._prepare_model_for_kbit_training(model, args)
            # Disable gradient checkpointing as it's handled by prepare_model_for_kbit_training
            args = dataclasses.replace(args, gradient_checkpointing=False)
        elif args.gradient_checkpointing:
            model = self._enable_gradient_checkpointing(model, args)

        # Create PEFT model
        if peft_config is not None:
            if (
                version.parse(peft.__version__) >= version.parse("0.12")  # autocast_adapter_dtype introduced in 0.12
                and getattr(model, "is_loaded_in_4bit", False)
                and is_sharded_qlora
            ):
                model = get_peft_model(model, peft_config, autocast_adapter_dtype=False)
            else:
                model = get_peft_model(model, peft_config)

        # Handle bf16 casting for 4-bit models
        if args.bf16 and getattr(model, "is_loaded_in_4bit", False) and not is_sharded_qlora:
            peft_module_casting_to_bf16(model)

        return model

    def _prepare_model_for_kbit_training(self, model: PreTrainedModel, args: SFTConfig) -> PreTrainedModel:
        """Prepares a quantized model for kbit training."""
        prepare_model_kwargs = {
            "use_gradient_checkpointing": args.gradient_checkpointing,
            "gradient_checkpointing_kwargs": args.gradient_checkpointing_kwargs or {},
        }

        return prepare_model_for_kbit_training(model, **prepare_model_kwargs)

    def _enable_gradient_checkpointing(self, model: PreTrainedModel, args: SFTConfig) -> PreTrainedModel:
        """Enables gradient checkpointing for the model."""
        gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs or {}
        use_reentrant = (
            "use_reentrant" not in gradient_checkpointing_kwargs or gradient_checkpointing_kwargs["use_reentrant"]
        )

        if use_reentrant:
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:

                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        return model

    def _prepare_dataset(
        self,
        dataset: Union[Dataset, IterableDataset],
        processing_class: Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin],
        args: SFTConfig,
        packing: bool,
        formatting_func: Optional[Callable[[dict], str]],
        dataset_name: str,
    ) -> Union[Dataset, IterableDataset]:
        # Tabular backends like Arrow/Parquet insert `None` for mismatched keys in nested structures. Clean them from
        # sampled data.
        if isinstance(dataset, Dataset):  # IterableDataset does not support `with_transform`
            dataset = dataset.with_transform(remove_none_values)

        # If the dataset is already preprocessed (tokenized), skip the processing steps.
        column_names = list(next(iter(dataset)).keys())
        is_processed = "input_ids" in column_names

        # Build the kwargs for the `map` function
        map_kwargs = {}
        if isinstance(dataset, Dataset):  # IterableDataset does not support num_proc
            map_kwargs["num_proc"] = args.dataset_num_proc

        with PartialState().main_process_first():
            # Apply the formatting function if any
            if formatting_func is not None and is_processed:
                warnings.warn(
                    "You passed a dataset that is already processed (contains an `input_ids` field) together with a "
                    "formatting function. Therefore `formatting_func` will be ignored. Either remove the "
                    "`formatting_func` or pass a dataset that is not already processed.",
                    UserWarning,
                )

            if formatting_func is not None and not is_processed:
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Applying formatting function to {dataset_name} dataset"

                def _func(example):
                    return {"text": formatting_func(example)}

                try:
                    dataset = dataset.map(_func, batched=False, **map_kwargs)
                except Exception as e:
                    warnings.warn(
                        f"Failed to apply the formatting function due to the following error: {e}. This may be "
                        "because the function is designed for batched input. Please update it to process one example "
                        "at a time (i.e., accept and return a single example). For now, we will attempt to apply the "
                        "function in batched mode, but note that batched formatting is deprecated and will be removed "
                        "in version 0.21.",
                        DeprecationWarning,
                    )
                    dataset = dataset.map(_func, batched=True, **map_kwargs)

            if not is_processed:
                # Convert the dataset to ChatML if needed
                first_example = next(iter(dataset))
                if is_conversational_from_value(first_example):
                    if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                        map_kwargs["desc"] = f"Converting {dataset_name} dataset to ChatML"
                    column_names = next(iter(dataset)).keys()
                    dataset = dataset.map(
                        maybe_convert_to_chatml,
                        remove_columns="conversations" if "conversations" in column_names else None,
                        **map_kwargs,
                    )

                # Apply the chat template if needed
                first_example = next(iter(dataset))
                if not is_conversational(first_example):
                    if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                        map_kwargs["desc"] = f"Adding EOS to {dataset_name} dataset"

                    def add_eos(example, eos_token):
                        if "text" in example and not example["text"].endswith(eos_token):  # language modeling case
                            example["text"] = example["text"] + eos_token
                        elif "completion" in example and not example["completion"].endswith(eos_token):
                            example["completion"] = example["completion"] + eos_token
                        return example

                    dataset = dataset.map(
                        add_eos,
                        fn_kwargs={"eos_token": processing_class.eos_token},
                        remove_columns="messages" if "messages" in column_names else None,  # renamed to "text"
                        **map_kwargs,
                    )

                # Tokenize the dataset
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Tokenizing {dataset_name} dataset"

                def tokenize(example, processing_class, dataset_text_field, assistant_only_loss):
                    if "prompt" in example:  # prompt-completion case
                        output = {}
                        if is_conversational(example):
                            prompt_ids = processing_class.apply_chat_template(
                                example["prompt"],
                                tools=example.get("tools"),
                                **example.get("chat_template_kwargs", {}),
                            )
                            prompt_completion_processed = processing_class.apply_chat_template(
                                example["prompt"] + example["completion"],
                                return_dict=True,
                                return_assistant_tokens_mask=assistant_only_loss,
                                tools=example.get("tools"),
                                **example.get("chat_template_kwargs", {}),
                            )
                            prompt_completion_ids = prompt_completion_processed["input_ids"]
                            if "assistant_masks" in prompt_completion_processed:
                                output["assistant_masks"] = prompt_completion_processed["assistant_masks"]
                        else:
                            prompt_ids = processing_class(text=example["prompt"])["input_ids"]
                            prompt_completion_ids = processing_class(text=example["prompt"] + example["completion"])[
                                "input_ids"
                            ]

                        # Check if the tokenized prompt starts with the tokenized prompt+completion
                        if not prompt_completion_ids[: len(prompt_ids)] == prompt_ids:
                            warnings.warn(
                                "Mismatch between tokenized prompt and the start of tokenized prompt+completion. "
                                "This may be due to unexpected tokenizer behavior, whitespace issues, or special "
                                "token handling. Verify that the tokenizer is processing text consistently."
                            )

                        # Create a completion mask
                        completion_mask = [0] * len(prompt_ids) + [1] * (len(prompt_completion_ids) - len(prompt_ids))
                        output["input_ids"] = prompt_completion_ids
                        output["completion_mask"] = completion_mask

                    else:  # language modeling case
                        if is_conversational(example):
                            processed = processing_class.apply_chat_template(
                                example["messages"],
                                return_dict=True,
                                return_assistant_tokens_mask=assistant_only_loss,
                                tools=example.get("tools"),
                                **example.get("chat_template_kwargs", {}),
                            )
                            if "assistant_masks" in processed and 1 not in processed["assistant_masks"]:
                                raise RuntimeError(
                                    "You're using `assistant_only_loss=True`, but at least one example has no "
                                    "assistant tokens. This usually means the tokenizer's chat template doesn't "
                                    "generate assistant masks — it may be missing the `{% generation %}` keyword. Please "
                                    "check the template and ensure it's correctly configured to support assistant "
                                    "masking."
                                )
                            output = {k: processed[k] for k in ("input_ids", "assistant_masks") if k in processed}
                        else:
                            output = {"input_ids": processing_class(text=example[dataset_text_field])["input_ids"]}
                    return output

                dataset = dataset.map(
                    tokenize,
                    fn_kwargs={
                        "processing_class": processing_class,
                        "dataset_text_field": args.dataset_text_field,
                        "assistant_only_loss": args.assistant_only_loss,
                    },
                    **map_kwargs,
                )

            # Pack or truncate
            if packing:
                if args.max_length is None:
                    raise ValueError("When packing is enabled, `max_length` can't be `None`.")
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Packing {dataset_name} dataset"

                columns = ["input_ids"]
                if "completion_mask" in dataset.column_names:
                    columns.append("completion_mask")
                if "assistant_masks" in dataset.column_names:
                    columns.append("assistant_masks")

                dataset = dataset.select_columns(columns)

                # Packing adds new column "seq_lengths" needed for document aware flash attention
                dataset = pack_dataset(dataset, args.max_length, args.packing_strategy, map_kwargs)
            elif args.max_length is not None:
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Truncating {dataset_name} dataset"
                dataset = truncate_dataset(dataset, args.max_length, map_kwargs)
            # For Liger kernel, ensure only the essential columns
            if args.use_liger_kernel:
                dataset = dataset.select_columns(
                    {"input_ids", "seq_lengths", "completion_mask"}.intersection(dataset.column_names)
                )

        return dataset

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs (usually, "input_ids"
        # and "attention_mask"). When using `train_on_completion_only` we add a "completion_mask" column to the
        # dataset. So we need to override the default signature columns to include "completion_mask" as well.
        if self._signature_columns is None:
            self._signature_columns = [
                "input_ids",
                "labels",
                "seq_lengths",
                "completion_mask",
                "assistant_masks",
            ]

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute training loss and additionally compute token accuracies
        """
        mode = "train" if self.model.training else "eval"
        (loss, outputs) = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )
        if mode == "train":
            # When using padding-free, the attention_mask is not present in the inputs, instead we have cu_seq_lens_q,
            # cu_seq_lens_k, and max_length_k, max_length_q and position_ids.
            if "attention_mask" in inputs:
                num_tokens_in_batch = self.accelerator.gather_for_metrics(inputs["attention_mask"].sum()).sum().item()
            elif "position_ids" in inputs:
                local_num_tokens = torch.tensor(inputs["position_ids"].size(1), device=inputs["position_ids"].device)
                num_tokens_in_batch = self.accelerator.gather_for_metrics(local_num_tokens).sum().item()
            else:
                raise ValueError("Expected 'attention_mask' or 'position_ids' in inputs.")
            self._total_train_tokens += num_tokens_in_batch
        self._metrics[mode]["num_tokens"] = [self._total_train_tokens]

        # Compute token accuracy if we have labels and if the model is not using Liger (no logits)
        if "labels" in inputs and not self.args.use_liger_kernel:
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = inputs["labels"][..., 1:].contiguous()

            # Get predictions
            predictions = shift_logits.argmax(dim=-1)

            # Create mask for non-padding tokens (assuming ignore_index is -100)
            mask = shift_labels != -100

            # Calculate accuracy only on non-padding tokens
            correct_predictions = (predictions == shift_labels) & mask
            total_tokens = mask.sum()
            correct_tokens = correct_predictions.sum()

            # Gather the correct_tokens and total_tokens across all processes
            correct_tokens = self.accelerator.gather_for_metrics(correct_tokens)
            total_tokens = self.accelerator.gather_for_metrics(total_tokens)

            # Compute the mean token accuracy and log it
            total_sum = total_tokens.sum()
            accuracy = (correct_tokens.sum() / total_sum).item() if total_sum > 0 else 0.0
            self._metrics[mode]["mean_token_accuracy"].append(accuracy)

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

        logs = {**logs, **metrics}
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

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        # normalize `tags` to a mutable set
        if tags is None:
            tags = set()
        elif isinstance(tags, str):
            tags = {tags}
        else:
            tags = set(tags)

        if hasattr(self.model.config, "unsloth_version"):
            tags.add("unsloth")

        tags.update(self._tag_names)

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=list(tags),
            wandb_url=wandb.run.url if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="SFT",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))
