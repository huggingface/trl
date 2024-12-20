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
import random
import textwrap
import warnings
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from dataclasses import dataclass
from operator import itemgetter
from typing import Any, Callable, Literal, Optional, Union

import torch
import torch.amp as amp
import torch.nn as nn
import torch.nn.functional as F
import transformers
from accelerate import PartialState
from accelerate.utils import is_deepspeed_available
from datasets import Dataset, IterableDataset
from packaging import version
from torch.utils.data import DataLoader, SequentialSampler
from transformers import (
    AutoModelForCausalLM,
    BaseImageProcessor,
    DataCollator,
    FeatureExtractionMixin,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    is_wandb_available,
)
from transformers.data.data_collator import DataCollatorMixin
from transformers.trainer_utils import EvalLoopOutput, has_length
from transformers.utils import is_peft_available

from ..data_utils import maybe_apply_chat_template, maybe_extract_prompt, maybe_unpair_preference_dataset
from ..models import PreTrainedModelWrapper, create_reference_model
from .kto_config import KTOConfig
from .utils import (
    disable_dropout_in_model,
    generate_model_card,
    get_comet_experiment_url,
    pad,
    pad_to_length,
    peft_module_casting_to_bf16,
)


if is_peft_available():
    from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training


if is_wandb_available():
    import wandb

if is_deepspeed_available():
    import deepspeed


@dataclass
class DataCollatorForUnpairedPreference(DataCollatorMixin):
    """
    Data collator used for unpaired preference data. Inputs are dynamically padded to the maximum length of a batch if
    they are not all of the same length.

    Args:
        pad_token_id (`int`):
            Token ID to use for padding.
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            Type of Tensor to return. Only `"pt"` is currently supported.

    Examples:
    ```python
    >>> from trl import DataCollatorForUnpairedPreference
    >>> collator = DataCollatorForUnpairedPreference(pad_token_id=0)
    >>> examples = [
    ...     {"prompt_input_ids": [1, 2, 3], "completion_input_ids": [4, 5], "label": True},
    ...     {"prompt_input_ids": [7, 8], "completion_input_ids": [9, 10], "label": False}
    ... ]
    >>> collator(examples)
    {'prompt_input_ids': tensor([[1, 2, 3],
                                 [0, 7, 8]]),
     'prompt_attention_mask': tensor([[1, 1, 1],
                                      [0, 1, 1]]),
     'completion_input_ids': tensor([[ 4,  5],
                                     [ 9, 10]]),
     'completion_attention_mask': tensor([[1, 1],
                                          [1, 1]]),
     'labels': tensor([True, False])}
    }
    ```
    """

    pad_token_id: int
    return_tensors: str = "pt"

    def torch_call(self, examples: list[Union[list[int], Any, dict[str, Any]]]) -> dict[str, Any]:
        # Convert to tensor
        prompt_input_ids = [torch.tensor(example["prompt_input_ids"]) for example in examples]
        prompt_attention_mask = [torch.ones_like(input_ids) for input_ids in prompt_input_ids]
        completion_input_ids = [torch.tensor(example["completion_input_ids"]) for example in examples]
        completion_attention_mask = [torch.ones_like(input_ids) for input_ids in completion_input_ids]
        labels = torch.tensor([example["label"] for example in examples])
        if "ref_completion_logps" in examples[0]:
            ref_completion_logps = torch.tensor([example["ref_completion_logps"] for example in examples])

        # Pad
        output = {}
        output["prompt_input_ids"] = pad(prompt_input_ids, padding_value=self.pad_token_id, padding_side="left")
        output["prompt_attention_mask"] = pad(prompt_attention_mask, padding_value=0, padding_side="left")
        output["completion_input_ids"] = pad(completion_input_ids, padding_value=self.pad_token_id)
        output["completion_attention_mask"] = pad(completion_attention_mask, padding_value=0)
        output["labels"] = labels
        if "ref_completion_logps" in examples[0]:
            output["ref_completion_logps"] = ref_completion_logps

        return output


class KTOTrainer(Trainer):
    r"""
    Initialize KTOTrainer.

    Args:
        model (`transformers.PreTrainedModel`):
            The model to train, preferably an `AutoModelForSequenceClassification`.
        ref_model (`PreTrainedModelWrapper`):
            Hugging Face transformer model with a casual language modelling head. Used for implicit reward computation and loss. If no
            reference model is provided, the trainer will create a reference model with the same architecture as the model to be optimized.
        args (`KTOConfig`):
            The arguments to use for training.
        train_dataset (`datasets.Dataset`):
            The dataset to use for training.
        eval_dataset (`datasets.Dataset`):
            The dataset to use for evaluation.
        processing_class (`PreTrainedTokenizerBase` or `BaseImageProcessor` or `FeatureExtractionMixin` or `ProcessorMixin`, *optional*):
            Processing class used to process the data. If provided, will be used to automatically process the inputs
            for the model, and it will be saved along the model to make it easier to rerun an interrupted training or
            reuse the fine-tuned model.
        data_collator (`transformers.DataCollator`, *optional*, defaults to `None`):
            The data collator to use for training. If None is specified, the default data collator (`DPODataCollatorWithPadding`) will be used
            which will pad the sequences to the maximum length of the sequences in the batch, given a dataset of paired sequences.
        model_init (`Callable[[], transformers.PreTrainedModel]`):
            The model initializer to use for training. If None is specified, the default model initializer will be used.
        callbacks (`list[transformers.TrainerCallback]`):
            The callbacks to use for training.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`):
            The optimizer and scheduler to use for training.
        preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`):
            The function to use to preprocess the logits before computing the metrics.
        peft_config (`dict`, defaults to `None`):
            The PEFT configuration to use for training. If you pass a PEFT configuration, the model will be wrapped in a PEFT model.
        disable_dropout (`bool`, defaults to `True`):
            Whether or not to disable dropouts in `model` and `ref_model`.
        compute_metrics (`Callable[[EvalPrediction], dict]`, *optional*):
            The function to use to compute the metrics. Must take a `EvalPrediction` and return
            a dictionary string to metric values.
        model_adapter_name (`str`, defaults to `None`):
            Name of the train target PEFT adapter, when using LoRA with multiple adapters.
        ref_adapter_name (`str`, defaults to `None`):
            Name of the reference PEFT adapter, when using LoRA with multiple adapters.
    """

    _tag_names = ["trl", "kto"]

    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module, str] = None,
        ref_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        args: KTOConfig = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        processing_class: Optional[
            Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin]
        ] = None,
        data_collator: Optional[DataCollator] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        peft_config: Optional[dict] = None,
        compute_metrics: Optional[Callable[[EvalLoopOutput], dict]] = None,
        model_adapter_name: Optional[str] = None,
        ref_adapter_name: Optional[str] = None,
    ):
        if type(args) is TrainingArguments:
            raise ValueError("Please use `KTOConfig` instead TrainingArguments.")

        if not isinstance(model, str) and ref_model is model:
            raise ValueError(
                "`model` and `ref_model` cannot be the same object. If you want `ref_model` to be the "
                "same as `model`, you must mass a copy of it, or `None` if you use peft."
            )

        if args.model_init_kwargs is None:
            model_init_kwargs = {}
        elif not isinstance(model, str):
            raise ValueError("You passed model_kwargs to the KTOTrainer. But your model is already instantiated.")
        else:
            model_init_kwargs = args.model_init_kwargs
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if torch_dtype is not None:
                # Convert to `torch.dtype` if an str is passed
                if isinstance(torch_dtype, str) and torch_dtype != "auto":
                    torch_dtype = getattr(torch, torch_dtype)
                if torch_dtype != "auto" and not isinstance(torch_dtype, torch.dtype):
                    raise ValueError(
                        f"Invalid `torch_dtype` passed to the KTOConfig. Expected a string with either `torch.dtype` or 'auto', but got {torch_dtype}."
                    )
                model_init_kwargs["torch_dtype"] = torch_dtype

        if args.ref_model_init_kwargs is None:
            ref_model_init_kwargs = {}
        elif not isinstance(ref_model, str):
            raise ValueError(
                "You passed ref_model_kwargs to the KTOTrainer. But your ref_model is already instantiated."
            )
        else:
            ref_model_init_kwargs = args.ref_model_init_kwargs
            torch_dtype = ref_model_init_kwargs.get("torch_dtype")
            if torch_dtype is not None:
                # Convert to `torch.dtype` if an str is passed
                if isinstance(torch_dtype, str) and torch_dtype != "auto":
                    torch_dtype = getattr(torch, torch_dtype)
                if torch_dtype != "auto" and not isinstance(torch_dtype, torch.dtype):
                    raise ValueError(
                        f"Invalid `torch_dtype` passed to the KTOConfig. Expected a string with either `torch.dtype` or 'auto', but got {torch_dtype}."
                    )
                ref_model_init_kwargs["torch_dtype"] = torch_dtype

        if isinstance(model, str):
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)

        if isinstance(ref_model, str):
            ref_model = AutoModelForCausalLM.from_pretrained(ref_model, **ref_model_init_kwargs)

        # Initialize this variable to False. This helps tracking the case when `peft_module_casting_to_bf16`
        # has been called in order to properly call autocast if needed.
        self._peft_has_been_casted_to_bf16 = False

        if not is_peft_available() and peft_config is not None:
            raise ValueError(
                "PEFT is not installed and you passed a `peft_config` in the trainer's kwargs, please install it with `pip install peft` to use the PEFT models"
            )
        elif is_peft_available() and peft_config is not None:
            # if model is a peft model and we have a peft_config, we merge and unload it first
            if isinstance(model, PeftModel):
                model = model.merge_and_unload()

            if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False):
                _support_gc_kwargs = hasattr(
                    args, "gradient_checkpointing_kwargs"
                ) and "gradient_checkpointing_kwargs" in list(
                    inspect.signature(prepare_model_for_kbit_training).parameters
                )

                prepare_model_kwargs = {"use_gradient_checkpointing": args.gradient_checkpointing}

                if _support_gc_kwargs:
                    prepare_model_kwargs["gradient_checkpointing_kwargs"] = args.gradient_checkpointing_kwargs

                model = prepare_model_for_kbit_training(model, **prepare_model_kwargs)
            elif getattr(args, "gradient_checkpointing", False):
                # For backward compatibility with older versions of transformers
                if hasattr(model, "enable_input_require_grads"):
                    model.enable_input_require_grads()
                else:

                    def make_inputs_require_grad(module, input, output):
                        output.requires_grad_(True)

                    model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

            # get peft model with the given config
            model = get_peft_model(model, peft_config)
            if args.bf16 and getattr(model, "is_loaded_in_4bit", False):
                peft_module_casting_to_bf16(model)
                # If args.bf16 we need to explicitly call `generate` with torch amp autocast context manager
                self._peft_has_been_casted_to_bf16 = True

        # For models that use gradient_checkpointing, we need to attach a hook that enables input
        # to explicitly have `requires_grad=True`, otherwise training will either silently
        # fail or completely fail.
        elif getattr(args, "gradient_checkpointing", False):
            # For backward compatibility with older versions of transformers
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:

                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        if args.generate_during_eval and not is_wandb_available():
            raise ValueError(
                "`generate_during_eval=True` requires Weights and Biases to be installed."
                " Please install with `pip install wandb` to resolve."
            )

        self.is_peft_model = is_peft_available() and isinstance(model, PeftModel)
        self.model_adapter_name = model_adapter_name
        self.ref_adapter_name = ref_adapter_name

        if ref_model:
            self.ref_model = ref_model
        elif self.is_peft_model or False:
            # The `model` with adapters turned off will be used as the reference model
            self.ref_model = None
        else:
            self.ref_model = create_reference_model(model)

        if processing_class is None:
            raise ValueError(
                "max_length or a processing_class must be specified when using the default DPODataCollatorWithPadding"
            )
        if args.max_length is None:
            warnings.warn(
                "When using DPODataCollatorWithPadding, you should set `max_length` in the KTOTrainer's init"
                " it will be set to `512` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            max_length = 512
        if args.max_length is not None:
            max_length = args.max_length

        if args.max_prompt_length is None:
            warnings.warn(
                "When using DPODataCollatorWithPadding, you should set `max_prompt_length` in the KTOTrainer's init"
                " it will be set to `128` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            max_prompt_length = 128
        if args.max_prompt_length is not None:
            max_prompt_length = args.max_prompt_length

        max_completion_length = None

        if data_collator is None:
            data_collator = DataCollatorForUnpairedPreference(pad_token_id=processing_class.pad_token_id)

        if args.disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)

        self.max_length = max_length
        self.generate_during_eval = args.generate_during_eval
        self.label_pad_token_id = args.label_pad_token_id
        self.padding_value = args.padding_value if args.padding_value is not None else processing_class.pad_token_id
        self.max_prompt_length = max_prompt_length
        self.truncation_mode = args.truncation_mode
        self.max_completion_length = max_completion_length
        self.processing_class = processing_class

        # metric
        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        # KTO parameter
        self.beta = args.beta
        self.desirable_weight = args.desirable_weight
        self.undesirable_weight = args.undesirable_weight

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in KTO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys are "prompt_input_ids" and "completion_input_ids". As a result,
        # the trainer issues the warning: "Could not estimate the number of tokens of the input, floating-point
        # operations will not be computed." To suppress this warning, we set the "estimate_tokens" key in the model's
        # "warnings_issued" dictionary to True. This acts as a flag to indicate that the warning has already been
        # issued.
        model.warnings_issued["estimate_tokens"] = True

        # 4. Handle the dataset - UNCOMMENT WHEN _prepare_dataset READY
        train_dataset = self._prepare_dataset(train_dataset, processing_class, args, "train")
        if eval_dataset is not None:
            if isinstance(eval_dataset, dict):
                eval_dataset = {
                    key: self._prepare_dataset(dataset, processing_class, args, key)
                    for key, dataset in eval_dataset.items()
                }
            else:
                eval_dataset = self._prepare_dataset(eval_dataset, processing_class, args, "eval")

        # Calculate dataset desirability balance
        num_desirable = max(sum(train_dataset["label"]), 1)
        num_undesirable = max(len(train_dataset["label"]) - num_desirable, 1)  # "label" is binary

        if num_desirable != num_undesirable:
            # The lower and upper bounds come from Eq. (8) of https://huggingface.co/papers/2402.01306
            des_weight_lower_bound = num_undesirable * self.undesirable_weight / num_desirable
            des_weight_upper_bound = des_weight_lower_bound * 1.33

            und_weight_upper_bound = num_desirable * self.desirable_weight / num_undesirable
            und_weight_lower_bound = und_weight_upper_bound / 1.33

            # Check if weights are within bounds (use exact values for logic)
            des_weight_in_range = des_weight_lower_bound <= self.desirable_weight <= des_weight_upper_bound
            und_weight_in_range = und_weight_lower_bound <= self.undesirable_weight <= und_weight_upper_bound

            # Display warning with rounded bounds using f-strings
            if not (des_weight_in_range or und_weight_in_range):
                warnings.warn(
                    "You have different amounts of desirable/positive and undesirable/negative examples, but the "
                    "weights on the desirable and undesirable losses don't seem to be in an ideal range. Based on "
                    f"your data, we recommend EITHER desirable_weight in [{des_weight_lower_bound:.2f}, "
                    f"{des_weight_upper_bound:.2f}] OR undesirable_weight in [{und_weight_lower_bound:.2f}, "
                    f"{und_weight_upper_bound:.2f}] (but NOT BOTH). See the documentation on how to optimally set "
                    "these weights.",
                    UserWarning,
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

        if not hasattr(self, "accelerator"):
            raise AttributeError(
                "Your `Trainer` does not have an `accelerator` object. Consider upgrading `transformers`."
            )

        if self.ref_model is None:
            if not (self.is_peft_model or False):
                raise ValueError(
                    "No reference model and model is not a Peft model. Try setting `precompute_ref_log_probs=True`"
                )
        else:
            if self.is_deepspeed_enabled:
                self.ref_model = self._prepare_deepspeed(self.ref_model)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

    def _prepare_dataset(
        self,
        dataset: Union[Dataset, IterableDataset],
        processing_class: Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin],
        args: KTOConfig,
        dataset_name: str,
    ) -> Union[Dataset, IterableDataset]:
        # Build the kwargs for the `map` function
        map_kwargs = {}
        if isinstance(dataset, Dataset):  # IterableDataset does not support num_proc
            map_kwargs["num_proc"] = args.dataset_num_proc

        with PartialState().local_main_process_first():
            # Extract prompt if needed
            if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                map_kwargs["desc"] = f"Extracting prompt in {dataset_name} dataset"
            dataset = dataset.map(maybe_extract_prompt, **map_kwargs)

            # Unpair the dataset if needed
            if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                map_kwargs["desc"] = f"Unpairing {dataset_name} dataset"
            dataset = maybe_unpair_preference_dataset(dataset, **map_kwargs)

            # Apply the chat template if needed
            if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                map_kwargs["desc"] = f"Applying chat template to {dataset_name} dataset"
            dataset = dataset.map(maybe_apply_chat_template, fn_kwargs={"tokenizer": processing_class}, **map_kwargs)

            # Tokenize the dataset
            if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                map_kwargs["desc"] = f"Tokenizing {dataset_name} dataset"

            dataset = dataset.map(
                self.tokenize_row,
                remove_columns=["prompt", "completion"],
                fn_kwargs={
                    "processing_class": processing_class,
                    "max_prompt_length": args.max_prompt_length,
                    "max_completion_length": args.max_completion_length,
                    # for enc-dec, we add the special tokens ([bos_token] + prompt + [eos_token]; completion + [eos_token])
                    "add_special_tokens": False,
                },
                **map_kwargs,
            )

        return dataset

    @staticmethod
    def tokenize_row(features, processing_class, max_prompt_length, max_completion_length, add_special_tokens):
        """
        Tokenize a row of the dataset.

        Args:
            features (`dict[str, str]`):
                Row of the dataset, should contain the keys `"prompt"`, `"completion"` and `"label"`.
            processing_class (`PreTrainedTokenizerBase`):
                Processing class used to process the data.
            max_prompt_length (`int` or `None`):
                Maximum length of the prompt sequence. If `None`, the prompt sequence is not truncated.
            max_completion_length (`int` or `None`):
                Maximum length of the completion sequences. If `None`, the completion sequences are not truncated.
            add_special_tokens (`bool`):
                Whether to add special tokens to the sequences. Typically used for encoder-decoder models. If `True`,
                the prompt sequence will have a bos token prepended and an eos token appended. In any case, the
                completion sequences will have an eos token appended.

        Returns:
            `dict[str, list[int]]`:
                Tokenized sequences with the keys `"prompt_input_ids"`, `"completion_input_ids"`.

        Example:
        ```python
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
        >>> features = {"prompt": "The sky is", "completion": " blue.", "label": 1}
        >>> KTOTrainer.tokenize_row(features, tokenizer, max_prompt_length=3, max_completion_length=3, add_special_tokens=False)
        {'prompt_input_ids': [785, 12884, 374], 'completion_input_ids': [6303, 13, 151643]}
        ```
        """
        tokenizer = processing_class  # the processing class is a tokenizer
        prompt_input_ids = tokenizer(features["prompt"], add_special_tokens=False)["input_ids"]
        completion_input_ids = tokenizer(features["completion"], add_special_tokens=False)["input_ids"]

        # Add special tokens (typically for encoder-decoder models)
        if add_special_tokens:
            if tokenizer.bos_token_id is not None:
                prompt_input_ids = [tokenizer.bos_token_id] + prompt_input_ids
            if tokenizer.eos_token_id is not None:
                prompt_input_ids = prompt_input_ids + [tokenizer.eos_token_id]
        completion_input_ids = completion_input_ids + [tokenizer.eos_token_id]

        # Truncate prompt and completion sequences
        if max_prompt_length is not None:
            prompt_input_ids = prompt_input_ids[-max_prompt_length:]
        if max_completion_length is not None:
            completion_input_ids = completion_input_ids[:max_completion_length]

        return {
            "prompt_input_ids": prompt_input_ids,
            "completion_input_ids": completion_input_ids,
        }

    def _prepare_deepspeed(self, model: PreTrainedModelWrapper):
        # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)

        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                    # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                    # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                        }
                    )

        # If ZeRO-3 is used, we shard both the active and reference model.
        # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        return model

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In KTOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by `DataCollatorForUnpairedPreference`, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt_input_ids", "completion_input_ids", "ref_completion_logps", "label"]

    @contextmanager
    def null_ref_context(self):
        """Context manager for handling null reference model (that is, peft adapter manipulation)."""
        with self.accelerator.unwrap_model(
            self.model
        ).disable_adapter() if self.is_peft_model and not self.ref_adapter_name else nullcontext():
            if self.ref_adapter_name:
                self.model.set_adapter(self.ref_adapter_name)
            yield
            if self.ref_adapter_name:
                self.model.set_adapter(self.model_adapter_name or "default")

    def forward(
        self, model: nn.Module, batch: dict[str, Union[list, torch.LongTensor]]
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        kl_logps = None

        with torch.no_grad():
            kl_logits = model(
                input_ids=batch["kl_completion_input_ids"], attention_mask=batch["kl_completion_attention_mask"]
            ).logits

        labels = batch["kl_completion_labels"][:, 1:].clone()
        logits = kl_logits[:, :-1, :]
        loss_mask = labels != self.label_pad_token_id
        labels[labels == self.label_pad_token_id] = 0
        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
        kl_logps = (per_token_logps * loss_mask).sum(-1)

        outputs = model(batch["completion_input_ids"], attention_mask=batch["completion_attention_mask"])
        completion_logits = outputs.logits

        labels = batch["completion_labels"][:, 1:].clone()
        logits = completion_logits[:, :-1, :]
        loss_mask = labels != self.label_pad_token_id
        labels[labels == self.label_pad_token_id] = 0
        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
        completion_logps = (per_token_logps * loss_mask).sum(-1)

        chosen_idx = torch.nonzero(batch["labels"], as_tuple=True)[0]
        rejected_idx = torch.nonzero(~batch["labels"], as_tuple=True)[0]

        chosen_logps = completion_logps[chosen_idx]
        rejected_logps = completion_logps[rejected_idx]

        chosen_logits = completion_logits[chosen_idx]
        rejected_logits = completion_logits[rejected_idx]

        return {
            "chosen_logps": chosen_logps,
            "rejected_logps": rejected_logps,
            "chosen_logits": chosen_logits,
            "rejected_logits": rejected_logits,
            "kl_logps": kl_logps,
        }

    def kto_loss(
        self,
        chosen_logps: torch.FloatTensor,
        rejected_logps: torch.FloatTensor,
        kl_logps: torch.FloatTensor,
        ref_chosen_logps: torch.FloatTensor,
        ref_rejected_logps: torch.FloatTensor,
        ref_kl_logps: torch.FloatTensor,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the KTO loss for a batch of policy and reference model log probabilities.

        Args:
            chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (num(chosen) in batch_size,)
            rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (num(rejected) in batch_size,)
            kl_logps: Log probabilities of the policy model for the kl responses. Shape: (batch_size,)
            ref_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (num(chosen) in batch_size,)
            ref_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (num(rejected) in batch_size,)
            ref_kl_logps: Log probabilities of the reference model for the kl responses. Shape: (batch_size,)

        Returns:
            A tuple of four tensors: (losses, chosen_rewards, rejected_rewards, kl).
            The losses tensor contains the KTO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
            The kl tensor contains the detached kl divergence estimate between the policy and reference models.
        """
        kl = (kl_logps - ref_kl_logps).mean().detach()
        kl = self.accelerator.gather(kl).mean().clamp(min=0)

        # Chosen losses
        if chosen_logps.shape[0] != 0 or ref_chosen_logps.shape[0] != 0:
            chosen_logratios = chosen_logps - ref_chosen_logps
            # Eqn (7) of the KTO paper (https://huggingface.co/papers/2402.01306)
            chosen_losses = 1 - F.sigmoid(self.beta * (chosen_logratios - kl))
            chosen_rewards = self.beta * chosen_logratios.detach()

        else:
            # lists can't be empty -- if they are, then accelerate.gather will hang
            chosen_losses = torch.Tensor([]).to(self.accelerator.device)
            chosen_rewards = torch.Tensor([]).to(self.accelerator.device)

        # Rejected losses
        if rejected_logps.shape[0] != 0 or ref_rejected_logps.shape[0] != 0:
            rejected_logratios = rejected_logps - ref_rejected_logps
            rejected_losses = 1 - F.sigmoid(self.beta * (kl - rejected_logratios))

            rejected_rewards = self.beta * rejected_logratios.detach()
        else:
            # lists can't be empty -- if they are, then accelerate.gather will hang
            rejected_losses = torch.Tensor([]).to(self.accelerator.device)
            rejected_rewards = torch.Tensor([]).to(self.accelerator.device)

        losses = torch.cat(
            (self.desirable_weight * chosen_losses, self.undesirable_weight * rejected_losses),
            0,
        )

        return losses, chosen_rewards, rejected_rewards, kl

    def get_batch_loss_metrics(
        self,
        model,
        batch: dict[str, Union[list, torch.LongTensor]],
    ):
        """Compute the KTO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}
        batch = {k: v.to(self.accelerator.device) for k, v in batch.items()}

        _batch = {}
        # Concat the prompt and completion input ids, and flush left

        input_ids = torch.cat((batch["prompt_input_ids"], batch["completion_input_ids"]), dim=1)
        attention_mask = torch.cat((batch["prompt_attention_mask"], batch["completion_attention_mask"]), dim=1)
        prompt_labels = torch.ones_like(batch["prompt_input_ids"]) * self.label_pad_token_id
        completions_labels = batch["completion_input_ids"].clone()
        # replace the label pad token with -100
        completions_labels[~batch["completion_attention_mask"].bool()] = -100
        labels = torch.cat((prompt_labels, completions_labels), dim=1)

        # Flush left to reduce the memory usage
        # [[0, 0, x, x, x, x],  ->  [[x, x, x, x],
        #  [0, x, x, x, 0, 0]]       [x, x, x, 0]]
        for i in range(attention_mask.size(0)):
            first_one_idx = torch.nonzero(attention_mask[i])[0].item()
            input_ids[i] = torch.roll(input_ids[i], shifts=-first_one_idx)
            attention_mask[i] = torch.roll(attention_mask[i], shifts=-first_one_idx)
            labels[i] = torch.roll(labels[i], shifts=-first_one_idx)

        _batch["completion_input_ids"] = input_ids
        _batch["completion_attention_mask"] = attention_mask
        _batch["completion_labels"] = labels

        # Roll completion to create mismatch pairs
        completion_input_ids = torch.roll(batch["completion_input_ids"], shifts=1, dims=0)
        completion_attention_mask = torch.roll(batch["completion_attention_mask"], shifts=1, dims=0)

        input_ids = torch.cat((batch["prompt_input_ids"], completion_input_ids), dim=1)
        attention_mask = torch.cat((batch["prompt_attention_mask"], completion_attention_mask), dim=1)
        # prompt_labels = torch.ones_like(batch["prompt_input_ids"]) * self.label_pad_token_id
        completions_labels = completion_input_ids.clone()
        # replace the label pad token with -100
        completions_labels[~batch["completion_attention_mask"].bool()] = -100
        labels = torch.cat((prompt_labels, completions_labels), dim=1)

        # Flush left to reduce the memory usage
        # [[0, 0, x, x, x, x],  ->  [[x, x, x, x],
        #  [0, x, x, x, 0, 0]]       [x, x, x, 0]]
        for i in range(attention_mask.size(0)):
            first_one_idx = torch.nonzero(attention_mask[i])[0].item()
            input_ids[i] = torch.roll(input_ids[i], shifts=-first_one_idx)
            attention_mask[i] = torch.roll(attention_mask[i], shifts=-first_one_idx)
            labels[i] = torch.roll(labels[i], shifts=-first_one_idx)

        _batch["kl_completion_input_ids"] = input_ids
        _batch["kl_completion_attention_mask"] = attention_mask
        _batch["kl_completion_labels"] = labels
        _batch["labels"] = batch["labels"]

        batch = _batch

        model_output = self.forward(model, batch)
        with torch.no_grad():
            ref_model_output = self.forward(self.ref_model, batch)

        losses, chosen_rewards, rejected_rewards, kl = self.kto_loss(
            model_output["chosen_logps"],
            model_output["rejected_logps"],
            model_output["kl_logps"],
            ref_model_output["chosen_logps"],
            ref_model_output["rejected_logps"],
            ref_model_output["kl_logps"],
        )
        metrics["kl"] = kl.item()

        num_chosen = torch.Tensor([len(chosen_rewards)]).to(self.accelerator.device)
        num_rejected = torch.Tensor([len(rejected_rewards)]).to(self.accelerator.device)

        all_num_chosen = self.accelerator.gather(num_chosen).sum().item()
        all_num_rejected = self.accelerator.gather(num_rejected).sum().item()

        if all_num_chosen > 0:
            metrics["rewards/chosen_sum"] = self.accelerator.gather(chosen_rewards.nansum()).nansum().item()
            metrics["logps/chosen_sum"] = (
                self.accelerator.gather(model_output["chosen_logps"].nansum()).nansum().item()
            )
            metrics["logits/chosen_sum"] = (
                self.accelerator.gather(model_output["chosen_logits"].nansum()).nansum().item()
            )
            metrics["count/chosen"] = all_num_chosen

        if all_num_rejected > 0:
            metrics["rewards/rejected_sum"] = self.accelerator.gather(rejected_rewards.nansum()).nansum().item()
            metrics["logps/rejected_sum"] = (
                self.accelerator.gather(model_output["rejected_logps"].nansum()).nansum().item()
            )
            metrics["logits/rejected_sum"] = (
                self.accelerator.gather(model_output["rejected_logits"].nansum()).nansum().item()
            )
            metrics["count/rejected"] = all_num_rejected

        loss = losses.nanmean()

        return loss, metrics

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
        num_items_in_batch=None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, dict[str, torch.Tensor]]]:
        compute_loss_context_manager = amp.autocast("cuda") if self._peft_has_been_casted_to_bf16 else nullcontext()

        with compute_loss_context_manager:
            loss, metrics = self.get_batch_loss_metrics(model, inputs)

        # Make sure to move the loss to the device the original accumulating loss is at back in the `Trainer` class:
        loss = loss.to(self.args.device)
        # force log the metrics
        if self.accelerator.is_main_process:
            self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return (loss, metrics)
        return loss

    def store_metrics(self, metrics: dict[str, float], train_eval: Literal["train", "eval"] = "train") -> None:
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None
        return SequentialSampler(self.train_dataset)

    def generate_from_model_and_ref(self, model, batch: dict[str, torch.LongTensor]) -> tuple[str, str]:
        """Generate samples from the model and reference model for the given batch of inputs."""

        # If one uses `generate_during_eval` with peft + bf16, we need to explicitly call generate with
        # the torch cuda amp context manager as some hidden states are silently casted to full precision.
        generate_context_manager = amp.autocast("cuda") if self._peft_has_been_casted_to_bf16 else nullcontext()

        with generate_context_manager:
            policy_output = model.generate(
                input_ids=batch["prompt_input_ids"],
                attention_mask=batch["prompt_attention_mask"],
                max_length=self.max_length,
                do_sample=True,
                pad_token_id=self.processing_class.pad_token_id,
            )

            # if ref_output in batch use that otherwise use the reference model
            if "ref_output" in batch:
                ref_output = batch["ref_output"]
            else:
                if self.ref_model is None:
                    with self.null_ref_context():
                        ref_output = self.model.generate(
                            input_ids=batch["prompt_input_ids"],
                            attention_mask=batch["prompt_attention_mask"],
                            max_length=self.max_length,
                            do_sample=True,
                            pad_token_id=self.processing_class.pad_token_id,
                        )
                else:
                    ref_output = self.ref_model.generate(
                        input_ids=batch["prompt_input_ids"],
                        attention_mask=batch["prompt_attention_mask"],
                        max_length=self.max_length,
                        do_sample=True,
                        pad_token_id=self.processing_class.pad_token_id,
                    )

        policy_output = pad_to_length(policy_output, self.max_length, self.processing_class.pad_token_id)
        policy_output_decoded = self.processing_class.batch_decode(policy_output, skip_special_tokens=True)

        ref_output = pad_to_length(ref_output, self.max_length, self.processing_class.pad_token_id)
        ref_output_decoded = self.processing_class.batch_decode(ref_output, skip_special_tokens=True)

        return policy_output_decoded, ref_output_decoded

    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
    ):
        if ignore_keys is None:
            if hasattr(model, "config"):
                ignore_keys = getattr(model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        prediction_context_manager = amp.autocast("cuda") if self._peft_has_been_casted_to_bf16 else nullcontext()
        with torch.no_grad(), prediction_context_manager:
            loss, metrics = self.get_batch_loss_metrics(model, inputs)

        # force log the metrics
        if self.accelerator.is_main_process:
            self.store_metrics(metrics, train_eval="eval")

        if prediction_loss_only:
            return (loss.detach(), None, None)

        # logits for the chosen and rejected samples from model
        logits_dict = {
            "eval_logits/chosen": metrics["logits/chosen"],
            "eval_logits/rejected": metrics["logits/rejected"],
        }
        logits = torch.tensor(
            [v for k, v in logits_dict.items() if k not in ignore_keys], device=self.accelerator.device
        )
        labels = torch.zeros(logits.shape[0], device=self.accelerator.device)

        return (loss.detach(), logits, labels)

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[list[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Overriding built-in evaluation loop to store metrics for each batch.
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """

        # Sample and save to game log if requested (for one batch to save time)
        if self.generate_during_eval:
            # Generate random indices within the range of the total number of samples
            num_samples = len(dataloader.dataset)
            random_indices = random.sample(range(num_samples), k=self.args.eval_batch_size)

            # Use dataloader.dataset.select to get the random batch without iterating over the DataLoader
            random_batch_dataset = dataloader.dataset.select(random_indices)
            random_batch = self.data_collator(random_batch_dataset)
            random_batch = self._prepare_inputs(random_batch)

            target_indicies = [i for i in range(len(random_batch["label"])) if random_batch["label"][i] is False]
            target_batch = {
                "prompt_input_ids": random_batch["prompt_input_ids"][target_indicies],
                "prompt_attention_mask": random_batch["prompt_attention_mask"][target_indicies],
                "prompt": itemgetter(*target_indicies)(random_batch["prompt"]),
            }
            policy_output_decoded, ref_output_decoded = self.generate_from_model_and_ref(self.model, target_batch)

            self.log(
                {
                    "game_log": wandb.Table(
                        columns=["Prompt", "Policy", "Ref Model"],
                        rows=[
                            [prompt, pol[len(prompt) :], ref[len(prompt) :]]
                            for prompt, pol, ref in zip(
                                target_batch["prompt"], policy_output_decoded, ref_output_decoded
                            )
                        ],
                    )
                }
            )
            self.state.log_history.pop()

        # Base evaluation
        initial_output = super().evaluation_loop(
            dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix
        )

        return initial_output

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        """
        Log `logs` on the various objects watching training, including stored metrics.

        Args:
            logs (`dict[str, float]`):
                The values to log.
            start_time (`float` or `None`, *optional*, defaults to `None`):
                Start time of the training.
        """
        # logs either has 'loss' or 'eval_loss'
        train_eval = "train" if "loss" in logs else "eval"
        # train metrics should have no prefix, eval should have 'eval_'
        prefix = "eval_" if train_eval == "eval" else ""
        # accumulate average metrics from sums and lengths
        for split in ["chosen", "rejected"]:
            if f"count/{split}" in self._stored_metrics[train_eval]:
                count_sum = torch.Tensor(self._stored_metrics[train_eval][f"count/{split}"]).sum().item()
                for metric in ["rewards", "logps", "logits"]:
                    logs[f"{prefix}{metric}/{split}"] = (
                        torch.Tensor(self._stored_metrics[train_eval][f"{metric}/{split}_sum"]).sum().item()
                        / count_sum
                    )
                    # delete obsolete metric
                    del self._stored_metrics[train_eval][f"{metric}/{split}_sum"]
                del self._stored_metrics[train_eval][f"count/{split}"]
        # calculate reward margin
        if f"{prefix}rewards/chosen" in logs and f"{prefix}rewards/rejected" in logs:
            logs[f"{prefix}rewards/margins"] = logs[f"{prefix}rewards/chosen"] - logs[f"{prefix}rewards/rejected"]
        # Add averaged stored metrics to logs
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[f"{prefix}{key}"] = torch.Tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]

        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            return super().log(logs, start_time)
        else:  # transformers<=4.46
            return super().log(logs)

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str`, *optional*, defaults to `None`):
                The name of the model.
            dataset_name (`str`, *optional*, defaults to `None`):
                The name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
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

        citation = textwrap.dedent("""\
        @article{ethayarajh2024kto,
            title        = {{KTO: Model Alignment as Prospect Theoretic Optimization}},
            author       = {Kawin Ethayarajh and Winnie Xu and Niklas Muennighoff and Dan Jurafsky and Douwe Kiela},
            year         = 2024,
            eprint       = {arXiv:2402.01306},
        }""")

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="KTO",
            trainer_citation=citation,
            paper_title="KTO: Model Alignment as Prospect Theoretic Optimization",
            paper_id="2402.01306",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))
