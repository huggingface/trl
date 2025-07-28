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

import inspect
import os
import random
import textwrap
import warnings
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from operator import itemgetter
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import PartialState
from accelerate.logging import get_logger
from accelerate.utils import tqdm
from datasets import Dataset
from torch import autocast
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
    TrainingArguments,
    is_comet_available,
    is_sklearn_available,
    is_wandb_available,
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput, has_length
from transformers.utils import is_peft_available

from ..data_utils import maybe_apply_chat_template
from ..import_utils import is_joblib_available
from ..models import create_reference_model, prepare_deepspeed
from .bco_config import BCOConfig
from .utils import (
    DPODataCollatorWithPadding,
    RunningMoments,
    disable_dropout_in_model,
    generate_model_card,
    get_comet_experiment_url,
    log_table_to_comet_experiment,
    pad_to_length,
    peft_module_casting_to_bf16,
    selective_log_softmax,
)


if is_peft_available():
    from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training

if is_wandb_available():
    import wandb

if is_sklearn_available():
    from sklearn.linear_model import LogisticRegression

if is_joblib_available():
    import joblib

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer

logger = get_logger(__name__)

RUNNING_NAME = "running.json"
CLF_NAME = "clf.pkl"


def _tokenize(
    batch: dict[str, list[Any]],
    tokenizer: "PreTrainedTokenizer",
    embedding_tokenizer: Optional["PreTrainedTokenizer"] = None,
) -> dict[str, list[Any]]:
    """Tokenize a batch from a BCO specific dataset."""
    prompt_tokenized = tokenizer(batch["prompt"], add_special_tokens=False)
    prompt_input_ids = prompt_tokenized["input_ids"]
    prompt_attention_mask = prompt_tokenized["attention_mask"]
    prompt_and_completion = [prompt + completion for prompt, completion in zip(batch["prompt"], batch["completion"])]
    full_tokenized = tokenizer(prompt_and_completion, add_special_tokens=False)
    full_input_ids = full_tokenized["input_ids"]
    full_attention_mask = full_tokenized["attention_mask"]

    answer_input_ids = [f[len(p) :] for f, p in zip(full_input_ids, prompt_input_ids)]
    answer_attention_mask = [f[len(p) :] for f, p in zip(full_attention_mask, prompt_attention_mask)]

    # Concat tokens to form `enc(a) + enc(a + b)[len(enc(a)):]`
    full_concat_input_ids = [np.concatenate([p, a]) for p, a in zip(prompt_input_ids, answer_input_ids)]
    # Prepare input tokens for token by token comparison
    full_input_ids = [np.array(f) for f in full_input_ids]
    for full, concat in zip(full_input_ids, full_concat_input_ids):
        if len(full) != len(concat):
            raise ValueError(
                "The elements in 'full_input_ids' and 'full_concat_input_ids' must have the same pairwise length."
            )

    # On some tokenizers, like Llama-2 tokenizer, there are occasions where tokens
    # can be merged together when tokenizing prompt+answer. This could result
    # on the last token from the prompt being different when tokenized on its own
    # vs when done as prompt+answer.
    response_token_ids_start_idx = [len(p) for p in prompt_input_ids]

    # If tokenized prompt is different than both prompt+answer, then it means the
    # last token has changed due to merging.
    for idx, (p, f, r) in enumerate(zip(prompt_input_ids, full_input_ids, response_token_ids_start_idx)):
        if not np.array_equal(p, f[:r]):
            response_token_ids_start_idx[idx] -= 1

    prompt_input_ids = [f[:r] for f, r in zip(full_input_ids, response_token_ids_start_idx)]
    prompt_attention_mask = [f[:r] for f, r in zip(full_attention_mask, response_token_ids_start_idx)]

    for p, m in zip(prompt_input_ids, prompt_attention_mask):
        if len(p) != len(m):
            raise ValueError("Prompt input ids and attention mask should have the same length.")

    answer_input_ids = [f[r:] for f, r in zip(full_input_ids, response_token_ids_start_idx)]
    answer_attention_mask = [f[r:] for f, r in zip(full_attention_mask, response_token_ids_start_idx)]

    output = dict(
        prompt_input_ids=prompt_input_ids,
        prompt_attention_mask=prompt_attention_mask,
        answer_input_ids=answer_input_ids,
        answer_attention_mask=answer_attention_mask,
    )

    if embedding_tokenizer is not None:
        embedding_tokenized = embedding_tokenizer(batch["prompt"], truncation=True, add_special_tokens=False)

        output.update(
            {
                "embedding_input_ids": embedding_tokenized["input_ids"],
                "embedding_attention_mask": embedding_tokenized["attention_mask"],
            }
        )

    return output


def _process_tokens(example: dict[str, Any], model: "PreTrainedModel" = None, **kwargs) -> dict:
    """Process tokens of a BCO specific dataset.

    At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation in case the prompt +
    completion responses is/are too long. First we truncate the prompt; if we're still too long, we truncate the
    completion.

    We also create the labels for the completion responses, which are of length equal to the sum of the length of the
    prompt and the completion response, with label_pad_token_id for the prompt tokens.
    """
    prompt = example["prompt"]
    completion = example["completion"]

    batch = {
        f"{kwargs['prefix']}prompt": prompt,
        f"{kwargs['prefix']}completion": completion,
        f"{kwargs['prefix']}label": example["label"],
    }

    if not kwargs["is_encoder_decoder"]:
        # Check issues below for more details
        #  1. https://github.com/huggingface/trl/issues/907
        #  2. https://github.com/EleutherAI/lm-evaluation-harness/pull/531#issuecomment-1595586257
        #  3. https://github.com/LianjiaTech/BELLE/issues/337

        if not isinstance(prompt, str):
            raise ValueError(f"prompt should be an str but got {type(prompt)}")

        if not isinstance(completion, str):
            raise ValueError(f"completion should be an str but got {type(completion)}")

        # keys of format prompt_* refers to just the prompt and answer_* refers to just the answer
        all_tokens = {
            "prompt_input_ids": example["prompt_input_ids"],
            "prompt_attention_mask": example["prompt_attention_mask"],
            "answer_input_ids": example["answer_input_ids"],
            "answer_attention_mask": example["answer_attention_mask"],
        }

        # calculate max length by checking if BOS/EOS is already there
        max_length = kwargs["max_length"]
        bos_token_id = kwargs["tokenizer"].bos_token_id
        eos_token_id = kwargs["tokenizer"].eos_token_id
        if bos_token_id != all_tokens["prompt_input_ids"][0]:
            max_length -= 1
        if eos_token_id != all_tokens["answer_input_ids"][-1]:
            max_length -= 1

        # if combined sequence is too long (> max_length - 1 for BOS token - 1 for EOS), truncate the prompt
        if len(all_tokens["prompt_input_ids"]) + len(all_tokens["answer_input_ids"]) > max_length:
            for k in ["prompt_input_ids", "prompt_attention_mask"]:
                if kwargs["truncation_mode"] == "keep_start":
                    all_tokens[k] = all_tokens[k][: kwargs["max_prompt_length"]]
                elif kwargs["truncation_mode"] == "keep_end":
                    all_tokens[k] = all_tokens[k][-kwargs["max_prompt_length"] :]
                else:
                    raise ValueError(f"Unknown truncation mode: {kwargs['truncation_mode']}")

        # if that's still too long, truncate the response
        if len(all_tokens["prompt_input_ids"]) + len(all_tokens["answer_input_ids"]) > max_length:
            for k in ["answer_input_ids", "answer_attention_mask"]:
                all_tokens[k] = all_tokens[k][: max_length - kwargs["max_prompt_length"]]

        # all input_ids and attention mask as is. We then check if we need to add BOS/EOS tokens
        batch[f"{kwargs['prefix']}prompt_input_ids"] = all_tokens["prompt_input_ids"]
        batch[f"{kwargs['prefix']}prompt_attention_mask"] = all_tokens["prompt_attention_mask"]
        batch[f"{kwargs['prefix']}completion_input_ids"] = (
            all_tokens["prompt_input_ids"] + all_tokens["answer_input_ids"]
        )
        batch[f"{kwargs['prefix']}completion_attention_mask"] = (
            all_tokens["prompt_attention_mask"] + all_tokens["answer_attention_mask"]
        )

        # add BOS, which affects both prompt and the full completion
        if bos_token_id is not None:
            if len(all_tokens["prompt_input_ids"]) == 0 or bos_token_id != all_tokens["prompt_input_ids"][0]:
                batch[f"{kwargs['prefix']}prompt_input_ids"] = [bos_token_id] + batch[
                    f"{kwargs['prefix']}prompt_input_ids"
                ]
                batch[f"{kwargs['prefix']}prompt_attention_mask"] = [1] + batch[
                    f"{kwargs['prefix']}prompt_attention_mask"
                ]
                batch[f"{kwargs['prefix']}completion_input_ids"] = [bos_token_id] + batch[
                    f"{kwargs['prefix']}completion_input_ids"
                ]
                batch[f"{kwargs['prefix']}completion_attention_mask"] = [1] + batch[
                    f"{kwargs['prefix']}completion_attention_mask"
                ]
        # add EOS, which affects only the full completion
        if len(all_tokens["answer_input_ids"]) == 0 or eos_token_id != all_tokens["answer_input_ids"][-1]:
            batch[f"{kwargs['prefix']}completion_input_ids"] = batch[f"{kwargs['prefix']}completion_input_ids"] + [
                eos_token_id
            ]
            batch[f"{kwargs['prefix']}completion_attention_mask"] = batch[
                f"{kwargs['prefix']}completion_attention_mask"
            ] + [1]

        batch[f"{kwargs['prefix']}completion_labels"] = batch[f"{kwargs['prefix']}completion_input_ids"][:]
        batch[f"{kwargs['prefix']}completion_labels"][: len(batch[f"{kwargs['prefix']}prompt_input_ids"])] = [
            kwargs["label_pad_token_id"]
        ] * len(batch[f"{kwargs['prefix']}prompt_input_ids"])
    else:
        completion_tokens = kwargs["tokenizer"](
            completion, truncation=True, max_length=kwargs["max_completion_length"], add_special_tokens=True
        )
        prompt_tokens = kwargs["tokenizer"](
            prompt, truncation=True, max_length=kwargs["max_prompt_length"], add_special_tokens=True
        )

        batch[f"{kwargs['prefix']}prompt_input_ids"] = prompt_tokens["input_ids"]
        batch[f"{kwargs['prefix']}prompt_attention_mask"] = prompt_tokens["attention_mask"]

        batch[f"{kwargs['prefix']}completion_labels"] = completion_tokens["input_ids"]
        batch[f"{kwargs['prefix']}completion_attention_mask"] = completion_tokens["attention_mask"]
        if model is not None and hasattr(model, "prepare_decoder_input_ids_from_labels"):
            batch[f"{kwargs['prefix']}completion_decoder_input_ids"] = model.prepare_decoder_input_ids_from_labels(
                labels=torch.tensor(batch["completion_labels"])
            )

    return batch


class BCOTrainer(Trainer):
    r"""
    Initialize BCOTrainer from [BCO](https://huggingface.co/papers/2404.04656) paper.

    Args:
        model (`transformers.PreTrainedModel`):
            The model to train, preferably an `AutoModelForSequenceClassification`.
        ref_model (`PreTrainedModelWrapper`):
            Hugging Face transformer model with a casual language modelling head. Used for implicit reward computation
            and loss. If no reference model is provided, the trainer will create a reference model with the same
            architecture as the model to be optimized.
        args (`BCOConfig`):
            The arguments to use for training.
        train_dataset (`datasets.Dataset`):
            The dataset to use for training.
        eval_dataset (`datasets.Dataset`):
            The dataset to use for evaluation.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], [`~transformers.BaseImageProcessor`], [`~transformers.FeatureExtractionMixin`] or [`~transformers.ProcessorMixin`], *optional*, defaults to `None`):
            Processing class used to process the data. If provided, will be used to automatically process the inputs
            for the model, and it will be saved along the model to make it easier to rerun an interrupted training or
            reuse the fine-tuned model.
        data_collator (`transformers.DataCollator`, *optional*, defaults to `None`):
            The data collator to use for training. If None is specified, the default data collator
            (`DPODataCollatorWithPadding`) will be used which will pad the sequences to the maximum length of the
            sequences in the batch, given a dataset of paired sequences.
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
        model_adapter_name (`str`, defaults to `None`):
            Name of the train target PEFT adapter, when using LoRA with multiple adapters.
        ref_adapter_name (`str`, defaults to `None`):
            Name of the reference PEFT adapter, when using LoRA with multiple adapters.
    """

    _tag_names = ["trl", "bco"]

    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module, str] = None,
        ref_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        args: BCOConfig = None,
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
        embedding_func: Optional[Callable] = None,
        embedding_tokenizer: Optional[PreTrainedTokenizerBase] = None,
    ):
        if embedding_func is not None and not (is_sklearn_available() and is_joblib_available()):
            raise ImportError(
                "BCOTrainer with UDM requires the scikit-learn and joblib libraries. Please install it with `pip install scikit-learn joblib`."
            )

        if type(args) is TrainingArguments:
            raise ValueError("Please use `BCOConfig` instead `TrainingArguments`.")

        if not isinstance(model, str) and model is not None and ref_model is model:
            raise ValueError(
                "`model` and `ref_model` cannot be the same object. If you want `ref_model` to be the "
                "same as `model`, you must mass a copy of it, or `None` if you use peft."
            )

        if args.model_init_kwargs is None:
            model_init_kwargs = {}
        elif not isinstance(model, str):
            raise ValueError("You passed model_kwargs to the BCOTrainer. But your model is already instantiated.")
        else:
            model_init_kwargs = args.model_init_kwargs
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if torch_dtype is not None:
                # Convert to `torch.dtype` if an str is passed
                if isinstance(torch_dtype, str) and torch_dtype != "auto":
                    torch_dtype = getattr(torch, torch_dtype)
                if torch_dtype != "auto" and not isinstance(torch_dtype, torch.dtype):
                    raise ValueError(
                        f"Invalid `torch_dtype` passed to the BCOConfig. Expected a string with either `torch.dtype` or 'auto', but got {torch_dtype}."
                    )
                model_init_kwargs["torch_dtype"] = torch_dtype

        if args.ref_model_init_kwargs is None:
            ref_model_init_kwargs = {}
        elif not isinstance(ref_model, str):
            raise ValueError(
                "You passed ref_model_kwargs to the BCOTrainer. But your ref_model is already instantiated."
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
                        f"Invalid `torch_dtype` passed to the BCOConfig. Expected a string with either `torch.dtype` or 'auto', but got {torch_dtype}."
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
            elif args.gradient_checkpointing:
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
        elif args.gradient_checkpointing:
            # For backward compatibility with older versions of transformers
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:

                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        if args.generate_during_eval and not (is_wandb_available() or is_comet_available()):
            raise ValueError(
                "`generate_during_eval=True` requires Weights and Biases or Comet to be installed."
                " Please install `wandb` or `comet-ml` to resolve."
            )

        if model is not None:
            self.is_encoder_decoder = model.config.is_encoder_decoder
        elif args.is_encoder_decoder is None:
            raise ValueError("When no model is provided, you need to pass the parameter is_encoder_decoder.")
        else:
            self.is_encoder_decoder = args.is_encoder_decoder

        self.is_peft_model = is_peft_available() and isinstance(model, PeftModel)
        self.model_adapter_name = model_adapter_name
        self.ref_adapter_name = ref_adapter_name

        if ref_model:
            self.ref_model = ref_model
        elif self.is_peft_model or args.precompute_ref_log_probs:
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
                "When using DPODataCollatorWithPadding, you should set `max_length` in the `BCOConfig`. "
                "It will be set to `512` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            max_length = 512
        if args.max_length is not None:
            max_length = args.max_length

        if args.max_prompt_length is None:
            warnings.warn(
                "When using DPODataCollatorWithPadding, you should set `max_prompt_length` in the `BCOConfig`. "
                "It will be set to `128` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            max_prompt_length = 128
        if args.max_prompt_length is not None:
            max_prompt_length = args.max_prompt_length

        max_completion_length = None
        if args.max_completion_length is None and self.is_encoder_decoder:
            warnings.warn(
                "When using DPODataCollatorWithPadding with an encoder decoder architecture, you should set `max_completion_length` in the BCOTrainer's init"
                " it will be set to `128` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            max_completion_length = 128
        if args.max_completion_length is not None and self.is_encoder_decoder:
            max_completion_length = args.max_completion_length

        if data_collator is None:
            data_collator = DPODataCollatorWithPadding(
                pad_token_id=processing_class.pad_token_id,
                label_pad_token_id=args.label_pad_token_id,
                is_encoder_decoder=self.is_encoder_decoder,
            )

            if args.remove_unused_columns:
                args.remove_unused_columns = False
                # warn users
                warnings.warn(
                    "When using DPODataCollatorWithPadding, you should set `remove_unused_columns=False` in your BCOConfig"
                    " we have set it for you, but you should do it yourself in the future.",
                    UserWarning,
                )

            self.use_dpo_data_collator = True
        else:
            self.use_dpo_data_collator = False

        # Disable dropout in the model and reference model
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
        self.precompute_ref_log_probs = args.precompute_ref_log_probs

        # Since ref_logs are precomputed on the first call to get_train/eval_dataloader
        # keep track of first called to avoid computation of future calls
        self._precomputed_train_ref_log_probs = False
        self._precomputed_eval_ref_log_probs = False

        # metric
        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        # BCO parameter
        self.beta = args.beta
        self.aux_loss_enabled = getattr(model.config, "output_router_logits", False)
        self.aux_loss_coef = getattr(model.config, "router_aux_loss_coef", 0.0)
        if self.aux_loss_enabled and self.aux_loss_coef == 0.0:
            warnings.warn(
                "You set `output_router_logits` to `True` in the model config, but `router_aux_loss_coef` is set to "
                "`0.0`, meaning the auxiliary loss will not be used. Either set `router_aux_loss_coef` to a value "
                "greater than `0.0`, or set `output_router_logits` to `False` if you don't want to use the auxiliary "
                "loss.",
                UserWarning,
            )

        # Underlying Distribution Matching argument
        self.embedding_func = embedding_func
        self.embedding_tokenizer = embedding_tokenizer

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in BCO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys are "prompt_input_ids" and "completion_input_ids". As a result,
        # the trainer issues the warning: "Could not estimate the number of tokens of the input, floating-point
        # operations will not be computed." To suppress this warning, we set the "estimate_tokens" key in the model's
        # "warnings_issued" dictionary to True. This acts as a flag to indicate that the warning has already been
        # issued.
        model.warnings_issued["estimate_tokens"] = True

        with PartialState().main_process_first():
            # Apply the chat template if needed
            train_dataset = train_dataset.map(
                maybe_apply_chat_template, fn_kwargs={"tokenizer": processing_class}, num_proc=args.dataset_num_proc
            )
            if eval_dataset is not None:
                eval_dataset = eval_dataset.map(
                    maybe_apply_chat_template,
                    fn_kwargs={"tokenizer": processing_class},
                    num_proc=args.dataset_num_proc,
                )

            # Tokenize and prepare the training datasets
            train_dataset = train_dataset.map(
                _tokenize,
                batched=True,
                fn_kwargs={"tokenizer": processing_class, "embedding_tokenizer": self.embedding_tokenizer},
                num_proc=args.dataset_num_proc,
                desc="Tokenizing train dataset",
            )

            # Prepare the datasets
            fn_kwargs = {
                "prefix": "",
                "is_encoder_decoder": self.is_encoder_decoder,
                "tokenizer": processing_class,
                "max_length": self.max_length,
                "truncation_mode": self.truncation_mode,
                "label_pad_token_id": self.label_pad_token_id,
                "max_prompt_length": self.max_prompt_length,
                "max_completion_length": self.max_completion_length,
            }
            train_dataset = train_dataset.map(
                _process_tokens,
                fn_kwargs=fn_kwargs,
                num_proc=args.dataset_num_proc,
                desc="Processing tokenized train dataset",
            )

            if eval_dataset is not None:
                # Tokenize
                eval_dataset = eval_dataset.map(
                    _tokenize,
                    fn_kwargs={"tokenizer": processing_class, "embedding_tokenizer": self.embedding_tokenizer},
                    batched=True,
                    num_proc=args.dataset_num_proc,
                    desc="Tokenizing eval dataset",
                )

                # Process
                fn_kwargs = {
                    "prefix": "",
                    "is_encoder_decoder": self.is_encoder_decoder,
                    "tokenizer": processing_class,
                    "max_length": self.max_length,
                    "truncation_mode": self.truncation_mode,
                    "label_pad_token_id": self.label_pad_token_id,
                    "max_prompt_length": self.max_prompt_length,
                    "max_completion_length": self.max_completion_length,
                }
                eval_dataset = eval_dataset.map(
                    _process_tokens,
                    fn_kwargs=fn_kwargs,
                    num_proc=args.dataset_num_proc,
                    desc="Processing tokenized eval dataset",
                )

            desirable = train_dataset.filter(
                lambda x: x["label"], num_proc=args.dataset_num_proc, desc="Filtering desirable examples"
            )
            undesirable = train_dataset.filter(
                lambda x: not x["label"], num_proc=args.dataset_num_proc, desc="Filtering undesirable examples"
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

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        if not hasattr(self, "accelerator"):
            raise AttributeError(
                "Your `Trainer` does not have an `accelerator` object. Consider upgrading `transformers`."
            )

        # Deepspeed Zero-3 does not support precompute_ref_log_probs
        if self.is_deepspeed_enabled:
            if self.accelerator.state.deepspeed_plugin.zero_stage == 3 and self.precompute_ref_log_probs:
                raise ValueError(
                    "You cannot use `precompute_ref_log_probs=True` with Deepspeed ZeRO-3. Please set `precompute_ref_log_probs=False`."
                )

        if self.ref_model is None:
            if not (self.is_peft_model or self.precompute_ref_log_probs):
                raise ValueError(
                    "No reference model and model is not a Peft model. Try setting `precompute_ref_log_probs=True`"
                )
        else:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        self.running = RunningMoments(accelerator=self.accelerator)

        if self.embedding_func is None or args.resume_from_checkpoint:
            return

        chosen_embeddings = self._get_sample_prompt_embeddings(desirable, sample_size=self.args.prompt_sample_size)
        rejected_embeddings = self._get_sample_prompt_embeddings(undesirable, sample_size=self.args.prompt_sample_size)

        embeddings = torch.cat((chosen_embeddings, rejected_embeddings), dim=0)
        labels = torch.cat(
            (torch.ones_like(chosen_embeddings[:, 0]), torch.zeros_like(rejected_embeddings[:, 0])), dim=0
        )

        self.clf = LogisticRegression(class_weight="balanced").fit(
            embeddings.cpu().float().numpy(), labels.cpu().numpy()
        )
        chosen_mean = self.clf.score(
            chosen_embeddings.cpu().float().numpy(), torch.ones_like(chosen_embeddings[:, 0]).cpu().numpy()
        )
        rejected_mean = self.clf.score(
            rejected_embeddings.cpu().float().numpy(), torch.zeros_like(rejected_embeddings[:, 0]).cpu().numpy()
        )
        logger.info(f"UDM classifier training scores: chosen: {chosen_mean}, rejected: {rejected_mean}")

    @property
    def match_underlying_distribution(self):
        return self.embedding_func is not None and self.embedding_tokenizer is not None

    def _get_chosen_prob(self, prompt_embeddings: torch.FloatTensor) -> torch.FloatTensor:
        """
        Calculates the probability if the given prompt embedding is from desirable dataset. This function calculates
        the probability in the process and ensemble across processes.
        """
        dtype = prompt_embeddings.dtype
        device = prompt_embeddings.device
        rank = self.accelerator.process_index

        padded_prompt_embeddings = self.accelerator.pad_across_processes(
            prompt_embeddings, pad_index=self.embedding_tokenizer.pad_token_id
        )
        sample_size = padded_prompt_embeddings.shape[0]
        nonzero = padded_prompt_embeddings.mean(dim=1) != self.embedding_tokenizer.pad_token_id
        prompt_embeddings = self.accelerator.gather(padded_prompt_embeddings)

        # cannot predict for all empty values
        if prompt_embeddings.shape[0] == 0:
            return torch.tensor([], device=device, dtype=dtype)

        prob = self.clf.predict_proba(prompt_embeddings.cpu().float().numpy())[:, 1]
        prob = torch.as_tensor(prob, dtype=dtype, device=device)
        prob = self.accelerator.reduce(prob, reduction="mean")

        prob = prob[sample_size * rank : sample_size * (rank + 1)]
        prob = prob[nonzero]

        return prob

    def _vectorize_prompt(self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor) -> torch.FloatTensor:
        """
        Replaces processing_class.pad_token_id to embedding_tokenizer.pad_token_id and applies self.embedding_func
        """
        input_ids = torch.where(
            input_ids == self.processing_class.pad_token_id,
            self.embedding_tokenizer.pad_token_id,
            input_ids,
        )

        with torch.no_grad():
            embeddings = self.embedding_func(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        return embeddings

    def _get_prompt_embeddings(
        self, batch: dict[str, Union[list, torch.LongTensor]]
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        """Extract embeddings from frozen embedding model"""

        if not self.match_underlying_distribution:
            return None, None

        embeddings = self._vectorize_prompt(
            input_ids=batch["embedding_input_ids"],
            attention_mask=batch["embedding_attention_mask"],
        )

        chosen_idx = [i for i in range(len(batch["label"])) if batch["label"][i] is True]
        rejected_idx = [i for i in range(len(batch["label"])) if batch["label"][i] is False]

        chosen_embeddings = embeddings[chosen_idx, ...]
        rejected_embeddings = embeddings[rejected_idx, ...]

        return (chosen_embeddings, rejected_embeddings)

    def _get_sample_prompt_embeddings(self, dataset: Dataset, sample_size: int = 512) -> torch.FloatTensor:
        """
        Sample instances from dataset and get prompt embeddings. Used for density ratio classifier training.
        """
        n_samples = min(len(dataset), sample_size)
        rand_indices = np.random.choice(len(dataset), size=(n_samples,))

        embedding_dataset = dataset.select(rand_indices)

        dataloader_params = {
            "batch_size": self.args.per_device_train_batch_size,
            "collate_fn": self.data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "shuffle": False,
        }

        # prepare dataloader
        data_loader = self.accelerator.prepare(DataLoader(embedding_dataset, **dataloader_params))

        with torch.no_grad():
            all_embeddings = torch.empty(0)
            for padded_batch in tqdm(iterable=data_loader, desc="Building sample prompt embeddings"):
                embeddings = self._vectorize_prompt(
                    input_ids=padded_batch["embedding_input_ids"],
                    attention_mask=padded_batch["embedding_attention_mask"],
                )
                embeddings = self.accelerator.gather_for_metrics(embeddings)
                all_embeddings = torch.cat((all_embeddings, embeddings.cpu()))

        return all_embeddings

    def _save_optimizer_and_scheduler(self, output_dir):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        super()._save_optimizer_and_scheduler(output_dir)

        if self.accelerator.is_main_process:
            # When saving optimizer and scheduler to checkpoint, save also the running delta object.
            self.running.save_to_json(os.path.join(output_dir, RUNNING_NAME))

            if self.match_underlying_distribution:
                joblib.dump(self.clf, os.path.join(output_dir, CLF_NAME), compress=True)

    def _load_optimizer_and_scheduler(self, checkpoint):
        if checkpoint is None:
            logger.warning_once(f"Missing Checkpoint {checkpoint}")
            return

        super()._load_optimizer_and_scheduler(checkpoint)

        # when loading optimizer and scheduler from checkpoint, also load the running delta object.
        running_file = os.path.join(checkpoint, RUNNING_NAME)
        if os.path.isfile(running_file):
            self.running = RunningMoments.load_from_json(self.accelerator, running_file)

        if self.match_underlying_distribution:
            clf_file = os.path.join(checkpoint, CLF_NAME)
            if os.path.isfile(clf_file):
                self.clf = joblib.load(clf_file)

    @contextmanager
    def null_ref_context(self):
        """Context manager for handling null reference model (that is, peft adapter manipulation)."""
        with (
            self.accelerator.unwrap_model(self.model).disable_adapter()
            if self.is_peft_model and not self.ref_adapter_name
            else nullcontext()
        ):
            if self.ref_adapter_name:
                self.model.set_adapter(self.ref_adapter_name)
            yield
            if self.ref_adapter_name:
                self.model.set_adapter(self.model_adapter_name or "default")

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Subclass of transformers.src.transformers.trainer.get_train_dataloader to precompute `ref_log_probs`.
        """

        if self.precompute_ref_log_probs and not self._precomputed_train_ref_log_probs:
            dataloader_params = {
                "batch_size": self.args.per_device_train_batch_size,
                "collate_fn": self.data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "shuffle": False,
            }

            # prepare dataloader
            data_loader = self.accelerator.prepare(DataLoader(self.train_dataset, **dataloader_params))
            reference_completion_logps = []

            for padded_batch in tqdm(iterable=data_loader, desc="Train dataset reference log probs"):
                reference_completion_logp = self.compute_reference_log_probs(padded_batch)

                reference_completion_logp = self.accelerator.gather_for_metrics(reference_completion_logp)
                reference_completion_logps.append(reference_completion_logp.cpu())

            self.train_dataset = self.train_dataset.add_column(
                name="reference_logps", column=torch.cat(reference_completion_logps).float().numpy()
            )

            self._precomputed_train_ref_log_probs = True

        return super().get_train_dataloader()

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass of transformers.src.transformers.trainer.get_eval_dataloader to precompute `ref_log_probs`.

        Args:
            eval_dataset (`torch.utils.data.Dataset`, *optional*):
                If provided, will override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns not accepted
                by the `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        if self.precompute_ref_log_probs and not self._precomputed_eval_ref_log_probs:
            dataloader_params = {
                "batch_size": self.args.per_device_eval_batch_size,
                "collate_fn": self.data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "shuffle": False,
            }

            # prepare dataloader
            data_loader = self.accelerator.prepare(DataLoader(eval_dataset, **dataloader_params))

            reference_completion_logps = []

            for padded_batch in tqdm(iterable=data_loader, desc="Eval dataset reference log probs"):
                reference_completion_logp = self.compute_reference_log_probs(padded_batch)

                reference_completion_logp = self.accelerator.gather_for_metrics(reference_completion_logp)
                reference_completion_logps.append(reference_completion_logp.cpu())

            eval_dataset = eval_dataset.add_column(
                name="reference_logps", column=torch.cat(reference_completion_logps).float().numpy()
            )

            # Save calculated reference_chosen_logps and reference_rejected_logps to the eval_dataset for subsequent runs
            if self.eval_dataset is not None:
                self.eval_dataset = eval_dataset
            self._precomputed_eval_ref_log_probs = True

        return super().get_eval_dataloader(eval_dataset=eval_dataset)

    def compute_reference_log_probs(self, padded_batch: dict) -> dict:
        """Computes log probabilities of the reference model for a single padded batch of a BCO specific dataset."""
        with torch.no_grad():
            if self.ref_model is None:
                with self.null_ref_context():
                    if self.is_encoder_decoder:
                        completion_logits = self.model(
                            padded_batch["prompt_input_ids"],
                            attention_mask=padded_batch["prompt_attention_mask"],
                            decoder_input_ids=padded_batch.get("completion_decoder_input_ids"),
                            labels=padded_batch["completion_labels"],
                        ).logits

                    else:
                        completion_logits = self.model(
                            padded_batch["completion_input_ids"],
                            attention_mask=padded_batch["completion_attention_mask"],
                        ).logits

            else:
                if self.is_encoder_decoder:
                    completion_logits = self.ref_model(
                        padded_batch["prompt_input_ids"],
                        attention_mask=padded_batch["prompt_attention_mask"],
                        decoder_input_ids=padded_batch.get("completion_decoder_input_ids"),
                        labels=padded_batch["completion_labels"],
                    ).logits

                else:
                    completion_logits = self.ref_model(
                        padded_batch["completion_input_ids"], attention_mask=padded_batch["completion_attention_mask"]
                    ).logits

        completion_logps = self.get_batch_logps(
            completion_logits,
            padded_batch["completion_labels"],
            average_log_prob=False,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )

        return completion_logps

    @staticmethod
    def get_batch_logps(
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = False,
        label_pad_token_id: int = -100,
        is_encoder_decoder: bool = False,
    ) -> torch.FloatTensor:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels:
                Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are
                ignored. Shape: (batch_size, sequence_length)
            average_log_prob:
                If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the
                log probabilities of the (non-masked) tokens.

        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the
            given logits.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        if not is_encoder_decoder:
            labels = labels[:, 1:].clone()
            logits = logits[:, :-1, :]
        else:
            # Fixes end-dec RuntimeError
            labels = labels.clone()

        loss_mask = labels != label_pad_token_id

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == label_pad_token_id] = 0

        per_token_logps = selective_log_softmax(logits, labels)

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return (per_token_logps * loss_mask).sum(-1)

    def forward(
        self, model: nn.Module, batch: dict[str, Union[list, torch.LongTensor]]
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        model_kwargs = (
            {
                "labels": batch["completion_labels"],
                "decoder_input_ids": batch.get("completion_decoder_input_ids"),
            }
            if self.is_encoder_decoder
            else {}
        )
        if self.aux_loss_enabled:
            model_kwargs["output_router_logits"] = True

        outputs = model(
            batch["completion_input_ids"],
            attention_mask=batch["completion_attention_mask"],
            **model_kwargs,
        )
        completion_logits = outputs.logits

        completion_logps = self.get_batch_logps(
            completion_logits,
            batch["completion_labels"],
            average_log_prob=False,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )

        if completion_logps.shape[0] != len(batch["label"]):
            raise ValueError(
                "There is a mismatch between the number of examples in this batch and the number of "
                "examples for which an output sequence was predicted."
            )

        chosen_idx = [i for i in range(completion_logps.shape[0]) if batch["label"][i] is True]
        rejected_idx = [i for i in range(completion_logps.shape[0]) if batch["label"][i] is False]

        chosen_logps = completion_logps[chosen_idx, ...]
        rejected_logps = completion_logps[rejected_idx, ...]

        chosen_logits = completion_logits[chosen_idx, ...]
        rejected_logits = completion_logits[rejected_idx, ...]

        if self.aux_loss_enabled:
            return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, outputs.aux_loss)
        else:
            return (chosen_logps, rejected_logps, chosen_logits, rejected_logits)

    def _get_udm_weight(self, rejected_embeddings: torch.FloatTensor) -> torch.FloatTensor:
        prob_desirable = self._get_chosen_prob(rejected_embeddings)
        min_ratio = self.args.min_density_ratio
        max_ratio = self.args.max_density_ratio

        weight = (prob_desirable / (1 - prob_desirable + 1e-8)).clamp(min=min_ratio, max=max_ratio)

        return weight

    def bco_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        chosen_embeddings: Optional[torch.FloatTensor],
        rejected_embeddings: Optional[torch.FloatTensor],
        do_train: bool = True,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the BCO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps:
                Log probabilities of the policy model for the chosen responses. Shape: (num(chosen) in batch_size,)
            policy_rejected_logps:
                Log probabilities of the policy model for the rejected responses. Shape: (num(rejected) in batch_size,)
            reference_chosen_logps:
                Log probabilities of the reference model for the chosen responses. Shape: (num(chosen) in batch_size,)
            reference_rejected_logps:
                Log probabilities of the reference model for the rejected responses. Shape: (num(rejected) in
                batch_size,)
            chosen_embeddings: embeddings of desirable prompts
            rejected_embeddings: embeddings of undesirable prompts

        Returns:
            A tuple of four tensors: (losses, chosen_rewards, rejected_rewards, delta). The losses tensor contains the
            BCO loss for each example in the batch. The chosen_rewards and rejected_rewards tensors contain the rewards
            for the chosen and rejected responses, respectively. The delta value contains the moving average of all
            implicit rewards.
        """

        chosen_logratios = policy_chosen_logps - reference_chosen_logps
        chosen_rewards = self.beta * chosen_logratios

        rejected_logratios = policy_rejected_logps - reference_rejected_logps
        rejected_rewards = self.beta * rejected_logratios

        if do_train:
            self.running.update(torch.cat((chosen_rewards, rejected_rewards), 0).detach())
        delta = torch.as_tensor(self.running.mean, device=chosen_rewards.device)

        chosen_losses = -F.logsigmoid(chosen_rewards - delta)
        rejected_losses = -F.logsigmoid(-(rejected_rewards - delta))

        if self.match_underlying_distribution:
            chosen_weight = torch.ones_like(chosen_losses)
            rejected_weight = self._get_udm_weight(rejected_embeddings)

            losses = torch.cat((chosen_weight * chosen_losses, rejected_weight * rejected_losses), dim=0)
        else:
            losses = torch.cat((chosen_losses, rejected_losses), dim=0)

        return losses, chosen_rewards, rejected_rewards, delta

    def get_batch_loss_metrics(
        self,
        model,
        batch: dict[str, Union[list, torch.LongTensor]],
        do_train: bool = True,
    ):
        """Compute the BCO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}
        batch = {k: (v.to(self.accelerator.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

        forward_output = self.forward(model, batch)
        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
        ) = forward_output[:4]
        if self.aux_loss_enabled:
            aux_loss = forward_output[4]

        # if reference_logps in batch use them, otherwise use the reference model
        if "reference_logps" in batch:
            chosen_idx = [i for i in range(batch["reference_logps"].shape[0]) if batch["label"][i] is True]
            rejected_idx = [i for i in range(batch["reference_logps"].shape[0]) if batch["label"][i] is False]

            reference_chosen_logps = batch["reference_logps"][chosen_idx, ...]
            reference_rejected_logps = batch["reference_logps"][rejected_idx, ...]
        else:
            with torch.no_grad():
                if self.ref_model is None:
                    with self.null_ref_context():
                        (
                            reference_chosen_logps,
                            reference_rejected_logps,
                            _,
                            _,
                        ) = self.forward(self.model, batch)[:4]
                else:
                    (
                        reference_chosen_logps,
                        reference_rejected_logps,
                        _,
                        _,
                    ) = self.forward(self.ref_model, batch)[:4]

        chosen_embeddings, rejected_embeddings = self._get_prompt_embeddings(batch)

        losses, chosen_rewards, rejected_rewards, delta = self.bco_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            chosen_embeddings,
            rejected_embeddings,
            do_train=do_train,
        )
        metrics["delta"] = self.accelerator.gather_for_metrics(delta).mean().item()

        num_chosen = torch.Tensor([len(chosen_rewards)]).to(self.accelerator.device)
        num_rejected = torch.Tensor([len(rejected_rewards)]).to(self.accelerator.device)

        all_num_chosen = self.accelerator.gather_for_metrics(num_chosen).sum().item()
        all_num_rejected = self.accelerator.gather_for_metrics(num_rejected).sum().item()

        if all_num_chosen > 0:
            metrics["rewards/chosen_sum"] = (
                self.accelerator.gather_for_metrics(chosen_rewards.nansum()).nansum().item()
            )
            metrics["logps/chosen_sum"] = (
                self.accelerator.gather_for_metrics(policy_chosen_logps.nansum()).nansum().item()
            )
            metrics["logits/chosen_sum"] = (
                self.accelerator.gather_for_metrics(policy_chosen_logits.nansum()).nansum().item()
            )
            metrics["count/chosen"] = all_num_chosen

        if all_num_rejected > 0:
            metrics["rewards/rejected_sum"] = (
                self.accelerator.gather_for_metrics(rejected_rewards.nansum()).nansum().item()
            )
            metrics["logps/rejected_sum"] = (
                self.accelerator.gather_for_metrics(policy_rejected_logps.nansum()).nansum().item()
            )
            metrics["logits/rejected_sum"] = (
                self.accelerator.gather_for_metrics(policy_rejected_logits.nansum()).nansum().item()
            )
            metrics["count/rejected"] = all_num_rejected

        loss = losses.nanmean()
        if self.aux_loss_enabled:
            loss += self.aux_loss_coef * aux_loss

        return loss, metrics

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
        num_items_in_batch=None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, dict[str, torch.Tensor]]]:
        compute_loss_context_manager = (
            autocast(self.accelerator.device.type) if self._peft_has_been_casted_to_bf16 else nullcontext()
        )

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

    def _get_train_sampler(self, dataset: Optional[Dataset] = None) -> Optional[torch.utils.data.Sampler]:
        if dataset is None:
            dataset = self.train_dataset
        if dataset is None or not has_length(dataset):
            return None
        return SequentialSampler(dataset)

    def generate_from_model_and_ref(self, model, batch: dict[str, torch.LongTensor]) -> tuple[str, str]:
        """Generate samples from the model and reference model for the given batch of inputs."""

        # If one uses `generate_during_eval` with peft + bf16, we need to explicitly call generate with
        # the torch amp context manager as some hidden states are silently casted to full precision.
        generate_context_manager = (
            autocast(self.accelerator.device.type) if self._peft_has_been_casted_to_bf16 else nullcontext()
        )
        with generate_context_manager:
            policy_output = model.generate(
                input_ids=batch["prompt_input_ids"],
                attention_mask=batch["prompt_attention_mask"],
                max_length=self.max_length,
                do_sample=True,
                pad_token_id=self.processing_class.pad_token_id,
            )

            # if reference_output in batch use that otherwise use the reference model
            if "reference_output" in batch:
                reference_output = batch["reference_output"]
            else:
                if self.ref_model is None:
                    with self.null_ref_context():
                        reference_output = self.model.generate(
                            input_ids=batch["prompt_input_ids"],
                            attention_mask=batch["prompt_attention_mask"],
                            max_length=self.max_length,
                            do_sample=True,
                            pad_token_id=self.processing_class.pad_token_id,
                        )
                else:
                    reference_output = self.ref_model.generate(
                        input_ids=batch["prompt_input_ids"],
                        attention_mask=batch["prompt_attention_mask"],
                        max_length=self.max_length,
                        do_sample=True,
                        pad_token_id=self.processing_class.pad_token_id,
                    )

        policy_output = pad_to_length(policy_output, self.max_length, self.processing_class.pad_token_id)
        policy_output_decoded = self.processing_class.batch_decode(policy_output, skip_special_tokens=True)

        reference_output = pad_to_length(reference_output, self.max_length, self.processing_class.pad_token_id)
        reference_output_decoded = self.processing_class.batch_decode(reference_output, skip_special_tokens=True)

        return policy_output_decoded, reference_output_decoded

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

        prediction_context_manager = (
            autocast(self.accelerator.device.type) if self._peft_has_been_casted_to_bf16 else nullcontext()
        )
        with torch.no_grad(), prediction_context_manager:
            loss, metrics = self.get_batch_loss_metrics(model, inputs, do_train=False)

        # force log the metrics
        if self.accelerator.is_main_process:
            self.store_metrics(metrics, train_eval="eval")

        if prediction_loss_only:
            return (loss.detach(), None, None)

        # logits for the chosen and rejected samples from model
        logits_dict = {}
        if "logits/chosen_sum" in metrics:
            logits_dict["eval_logits/chosen"] = metrics["logits/chosen_sum"]
        if "logits/rejected_sum" in metrics:
            logits_dict["eval_logits/rejected"] = metrics["logits/rejected_sum"]
        logits = [v for k, v in logits_dict.items() if k not in ignore_keys]
        logits = torch.tensor(logits, device=self.accelerator.device)
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
        Overriding built-in evaluation loop to store metrics for each batch. Prediction/evaluation loop, shared by
        `Trainer.evaluate()` and `Trainer.predict()`.

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

            table = pd.DataFrame(
                columns=["Prompt", "Policy", "Ref Model"],
                data=[
                    [prompt, pol[len(prompt) :], ref[len(prompt) :]]
                    for prompt, pol, ref in zip(target_batch["prompt"], policy_output_decoded, ref_output_decoded)
                ],
            )
            if "wandb" in self.args.report_to:
                wandb.log({"game_log": wandb.Table(data=table)})

            if "comet_ml" in self.args.report_to:
                log_table_to_comet_experiment(
                    name="game_log.csv",
                    table=table,
                )

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
        return super().log(logs, start_time)

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

        citation = textwrap.dedent("""\
        @article{jung2024binary,
            title        = {{Binary Classifier Optimization for Large Language Model Alignment}},
            author       = {Seungjae Jung and Gunsoo Han and Daniel Wontae Nam and Kyoung{-}Woon On},
            year         = 2024,
            eprint       = {arXiv:2404.04656}
        }""")

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.url if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="BCO",
            trainer_citation=citation,
            paper_title="Binary Classifier Optimization for Large Language Model Alignment",
            paper_id="2404.04656",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))
