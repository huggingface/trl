# KTO Authors: Kawin Ethayarajh, Winnie Xu, Niklas Muennighoff, Dan Jurafsky, and Douwe Kiela
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
import random
import warnings
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from functools import wraps
from operator import itemgetter
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import PartialState
from accelerate.utils import is_deepspeed_available, tqdm
from datasets import Dataset, concatenate_datasets, interleave_datasets
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, SequentialSampler
from transformers import (
    AutoModelForCausalLM,
    DataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput, has_length

from ..import_utils import is_peft_available, is_sklearn_available, is_wandb_available
from ..models import PreTrainedModelWrapper, create_reference_model
from .kto_config import KTOConfig
from .utils import (
    DPODataCollatorWithPadding,
    RunningMoments,
    disable_dropout_in_model,
    pad_to_length,
    peft_module_casting_to_bf16,
    trl_sanitze_kwargs_for_tagging,
)


if is_peft_available():
    from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training


if is_wandb_available():
    import wandb

if is_sklearn_available():
    from sklearn.linear_model import LogisticRegression

if is_deepspeed_available():
    import deepspeed

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer


def _get_kl_dataset(batch: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
    """Creates mismatched pairs of prompts and completions for the KL dataset by reversing the order of completions."""
    batch["answer_input_ids"] = batch["answer_input_ids"][::-1]
    batch["answer_attention_mask"] = batch["answer_attention_mask"][::-1]
    return batch


def _tokenize(
    batch: Dict[str, List[Any]],
    tokenizer: "PreTrainedTokenizer",
    embedding_tokenizer: Optional["PreTrainedTokenizer"] = None,
) -> Dict[str, List[Any]]:
    """Tokenize a batch from a KTO/BCO specific dataset."""
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
            raise ValueError("Prompt input ids and answer input ids should have the same length.")

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


def _process_tokens(example: Dict[str, Any], model: "PreTrainedModel" = None, **kwargs) -> Dict:
    """Process tokens of a KTO specific dataset.

    At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
    in case the prompt + completion responses is/are too long. First
        we truncate the prompt; if we're still too long, we truncate the completion.

    We also create the labels for the completion responses, which are of length equal to
        the sum of the length of the prompt and the completion response, with
        label_pad_token_id  for the prompt tokens.
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
        if len(all_tokens["prompt_input_ids"]) == 0 or bos_token_id != all_tokens["prompt_input_ids"][0]:
            batch[f"{kwargs['prefix']}prompt_input_ids"] = [bos_token_id] + batch[
                f"{kwargs['prefix']}prompt_input_ids"
            ]
            batch[f"{kwargs['prefix']}prompt_attention_mask"] = [1] + batch[f"{kwargs['prefix']}prompt_attention_mask"]
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
        tokenizer (`transformers.PreTrainedTokenizerBase`):
            The tokenizer to use for training. This argument is required if you want to use the default data collator.
        data_collator (`transformers.DataCollator`, *optional*, defaults to `None`):
            The data collator to use for training. If None is specified, the default data collator (`DPODataCollatorWithPadding`) will be used
            which will pad the sequences to the maximum length of the sequences in the batch, given a dataset of paired sequences.
        model_init (`Callable[[], transformers.PreTrainedModel]`):
            The model initializer to use for training. If None is specified, the default model initializer will be used.
        callbacks (`List[transformers.TrainerCallback]`):
            The callbacks to use for training.
        optimizers (`Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`):
            The optimizer and scheduler to use for training.
        preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`):
            The function to use to preprocess the logits before computing the metrics.
        peft_config (`Dict`, defaults to `None`):
            The PEFT configuration to use for training. If you pass a PEFT configuration, the model will be wrapped in a PEFT model.
        disable_dropout (`bool`, defaults to `True`):
            Whether or not to disable dropouts in `model` and `ref_model`.
        compute_metrics (`Callable[[EvalPrediction], Dict]`, *optional*):
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
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        data_collator: Optional[DataCollator] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        peft_config: Optional[Dict] = None,
        compute_metrics: Optional[Callable[[EvalLoopOutput], Dict]] = None,
        model_adapter_name: Optional[str] = None,
        ref_adapter_name: Optional[str] = None,
        embedding_func: Optional[Callable] = None,
        embedding_tokenizer: Optional[PreTrainedTokenizerBase] = None,
    ):
        if type(args) == TrainingArguments:
            raise ValueError("Please use `KTOConfig` instead TrainingArguments.")

        if args.model_init_kwargs is None:
            model_init_kwargs = {}
        elif not isinstance(model, str):
            raise ValueError("You passed model_kwargs to the KTOTrainer. But your model is already instantiated.")
        else:
            model_init_kwargs = args.model_init_kwargs
            model_init_kwargs["torch_dtype"] = (
                model_init_kwargs["torch_dtype"]
                if model_init_kwargs["torch_dtype"] in ["auto", None]
                else getattr(torch, model_init_kwargs["torch_dtype"])
            )

        if args.ref_model_init_kwargs is None:
            ref_model_init_kwargs = {}
        elif not isinstance(ref_model, str):
            raise ValueError(
                "You passed ref_model_kwargs to the KTOTrainer. But your ref_model is already instantiated."
            )
        else:
            ref_model_init_kwargs = args.ref_model_init_kwargs
            ref_model_init_kwargs["torch_dtype"] = (
                ref_model_init_kwargs["torch_dtype"]
                if ref_model_init_kwargs["torch_dtype"] in ["auto", None]
                else getattr(torch, ref_model_init_kwargs["torch_dtype"])
            )

        if isinstance(model, str):
            warnings.warn(
                "You passed a model_id to the KTOTrainer. This will automatically create an "
                "`AutoModelForCausalLM` or a `PeftModel` (if you passed a `peft_config`) for you."
            )
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)

        if isinstance(ref_model, str):
            warnings.warn(
                "You passed a ref model_id to the KTOTrainer. This will automatically create an "
                "`AutoModelForCausalLM`"
            )
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

        if tokenizer is None:
            raise ValueError(
                "max_length or a tokenizer must be specified when using the default DPODataCollatorWithPadding"
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
        if args.max_completion_length is None and self.is_encoder_decoder:
            warnings.warn(
                "When using DPODataCollatorWithPadding with an encoder decoder architecture, you should set `max_completion_length` in the KTOTrainer's init"
                " it will be set to `128` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            max_completion_length = 128
        if args.max_completion_length is not None and self.is_encoder_decoder:
            max_completion_length = args.max_completion_length

        if data_collator is None:
            data_collator = DPODataCollatorWithPadding(
                pad_token_id=tokenizer.pad_token_id,
                label_pad_token_id=args.label_pad_token_id,
                is_encoder_decoder=self.is_encoder_decoder,
            )

            if args.remove_unused_columns:
                args.remove_unused_columns = False
                # warn users
                warnings.warn(
                    "When using DPODataCollatorWithPadding, you should set `remove_unused_columns=False` in your KTOConfig"
                    " we have set it for you, but you should do it yourself in the future.",
                    UserWarning,
                )

            self.use_dpo_data_collator = True
        else:
            self.use_dpo_data_collator = False

        # disable dropout in the model and reference model
        disable_dropout_in_model(model)
        if self.ref_model is not None:
            disable_dropout_in_model(self.ref_model)

        self.max_length = max_length
        self.generate_during_eval = args.generate_during_eval
        self.label_pad_token_id = args.label_pad_token_id
        self.padding_value = args.padding_value if args.padding_value is not None else tokenizer.pad_token_id
        self.max_prompt_length = max_prompt_length
        self.truncation_mode = args.truncation_mode
        self.max_completion_length = max_completion_length
        self.tokenizer = tokenizer
        self.precompute_ref_log_probs = args.precompute_ref_log_probs

        # Since ref_logs are precomputed on the first call to get_train/eval_dataloader
        # keep track of first called to avoid computation of future calls
        self._precomputed_train_ref_log_probs = False
        self._precomputed_eval_ref_log_probs = False

        # metric
        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        # KTO parameter
        self.beta = args.beta
        self.desirable_weight = args.desirable_weight
        self.undesirable_weight = args.undesirable_weight
        self.loss_type = args.loss_type
        self.aux_loss_enabled = getattr(model.config, "output_router_logits", False)

        # Underlying Distribution Matching argument
        self.embedding_func = embedding_func
        self.embedding_tokenizer = embedding_tokenizer

        with PartialState().local_main_process_first():
            # Shuffle the datasets
            train_dataset = train_dataset.shuffle(seed=args.data_seed)
            if eval_dataset is not None:
                eval_dataset = eval_dataset.shuffle(seed=args.data_seed)
            # Tokenize and prepare the training datasets
            train_dataset = train_dataset.map(
                _tokenize,
                fn_kwargs={"tokenizer": self.tokenizer, "embedding_tokenizer": self.embedding_tokenizer},
                batched=True,
                desc="Tokenizing train dataset",
            )
            # Get KL datasets
            total_batch_size = (
                max(torch.cuda.device_count(), 1) * args.per_device_train_batch_size * args.gradient_accumulation_steps
            )
            if total_batch_size <= 1:
                raise ValueError(
                    "Batch size is 1 (too small). KTO will not work properly because the KL term will be equivalent to the implied reward."
                )
            # create pairs for estimating the KL term by flipping the matched pairs in each batch of size total_batch_size
            # i.e., (x_1, y_1), ..., (x_n, y_n) --> (x_1, y_n), ..., (x_n, y_1) = (x'_1, y'_1), ..., (x'_n, y'_n)
            train_kl_dataset = train_dataset.map(
                _get_kl_dataset, batched=True, batch_size=total_batch_size, desc="Extracting KL train dataset"
            )
            # Prepare the datasets
            fn_kwargs = {
                "prefix": "",
                "is_encoder_decoder": self.is_encoder_decoder,
                "tokenizer": self.tokenizer,
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
            fn_kwargs["prefix"] = "KL_"
            train_kl_dataset = train_kl_dataset.map(
                _process_tokens,
                fn_kwargs=fn_kwargs,
                num_proc=args.dataset_num_proc,
                remove_columns=[c for c in train_kl_dataset.column_names if c in train_dataset.column_names],
                desc="Processing tokenized train KL dataset",
            )

            # merge the datasets
            train_dataset = concatenate_datasets([train_dataset, train_kl_dataset], axis=1)

            if eval_dataset is not None:
                # Tokenize
                eval_dataset = eval_dataset.map(
                    _tokenize,
                    fn_kwargs={"tokenizer": self.tokenizer, "embedding_tokenizer": self.embedding_tokenizer},
                    batched=True,
                    desc="Tokenizing eval dataset",
                )
                # Get KL dataset
                eval_kl_dataset = eval_dataset.map(
                    _get_kl_dataset, batched=True, batch_size=total_batch_size, desc="Extracting eval KL dataset"
                )
                # Process
                fn_kwargs = {
                    "prefix": "",
                    "is_encoder_decoder": self.is_encoder_decoder,
                    "tokenizer": self.tokenizer,
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
                fn_kwargs["prefix"] = "KL_"
                eval_kl_dataset = eval_kl_dataset.map(
                    _process_tokens,
                    fn_kwargs=fn_kwargs,
                    num_proc=args.dataset_num_proc,
                    remove_columns=[c for c in eval_kl_dataset.column_names if c in eval_dataset.column_names],
                    desc="Processing tokenized eval KL dataset",
                )

                # merge the datasets
                eval_dataset = concatenate_datasets([eval_dataset, eval_kl_dataset], axis=1)

            num_desirable = max(sum(train_dataset["label"]), 1)
            num_undesirable = max(len(train_dataset["label"]) - num_desirable, 1)  # "label" is binary

            if num_desirable != num_undesirable:
                # The lower and upper bounds come from Eq. (8) of https://arxiv.org/abs/2402.01306
                des_weight_lower_bound = round((num_undesirable * self.undesirable_weight / num_desirable) * 1, 2)
                des_weight_upper_bound = round((num_undesirable * self.undesirable_weight / num_desirable) * 1.33, 2)
                und_weight_lower_bound = round((num_desirable * self.desirable_weight / num_undesirable) / 1.33, 2)
                und_weight_upper_bound = round((num_desirable * self.desirable_weight / num_undesirable) / 1, 2)

                des_weight_in_range = des_weight_lower_bound <= self.desirable_weight <= des_weight_upper_bound
                und_weight_in_range = und_weight_lower_bound <= self.undesirable_weight <= und_weight_upper_bound

                if not (des_weight_in_range or und_weight_in_range):
                    warnings.warn(
                        f"""
                        You have different amounts of desirable/positive and undesirable/negative examples but the
                        weights on the desirable and undesirable losses don't seem to be in an ideal range. Based
                        on your data, we recommend EITHER desirable_weight in [{des_weight_lower_bound}, {des_weight_upper_bound}]
                        or undesirable_weight in [{und_weight_lower_bound}, {und_weight_upper_bound}] (but NOT BOTH).
                        See the documentation on how to optimally set these weights.""",
                        UserWarning,
                    )

            if self.loss_type == "bco":
                desirable = train_dataset.filter(
                    lambda x: x["label"], num_proc=args.dataset_num_proc, desc="Filtering desirable examples"
                )
                undesirable = train_dataset.filter(
                    lambda x: not x["label"], num_proc=args.dataset_num_proc, desc="Filtering undesirable examples"
                )

                desirable = desirable.shuffle(seed=args.data_seed)
                undesirable = undesirable.shuffle(seed=args.data_seed)

                # split the dataset and interleave them together with equal probability of choosing chosen or rejected
                interleaved_train_dataset = interleave_datasets(
                    [desirable, undesirable],
                    stopping_strategy="all_exhausted",
                )
                train_dataset = interleaved_train_dataset

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
        )

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
                self.ref_model = self._prepare_deepspeed(self.ref_model)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        if self.loss_type == "bco":
            self.running = RunningMoments(self.accelerator)

            if self.embedding_func is None:
                return

            chosen_embeddings = self._get_sample_prompt_embeddings(desirable, sample_size=self.args.prompt_sample_size)
            rejected_embeddings = self._get_sample_prompt_embeddings(
                undesirable, sample_size=self.args.prompt_sample_size
            )

            embeddings = torch.cat((chosen_embeddings, rejected_embeddings), dim=0)
            labels = torch.cat(
                (torch.ones_like(chosen_embeddings[:, 0]), torch.zeros_like(rejected_embeddings[:, 0])), dim=0
            )

            self.clf = LogisticRegression(class_weight="balanced").fit(embeddings.cpu().numpy(), labels.cpu().numpy())

    @property
    def match_underlying_distribution(self):
        return self.embedding_func is not None and self.embedding_tokenizer is not None

    def _get_chosen_prob(self, prompt_embeddings: torch.FloatTensor) -> torch.FloatTensor:
        """
        Calculates the probability if the given prompt embedding is from desirable dataset.
        This function calculates the probability in the process and ensemble across processes.
        """
        dtype = prompt_embeddings.dtype
        device = prompt_embeddings.device
        sample_size = prompt_embeddings.shape[0]

        padded_prompt_embeddings = self.accelerator.pad_across_processes(prompt_embeddings)
        nonzero = padded_prompt_embeddings.sum(dim=1) != 0
        prompt_embeddings = self.accelerator.gather(padded_prompt_embeddings)

        prob = self.clf.predict_proba(prompt_embeddings.cpu().numpy())[:, 1]
        prob = torch.as_tensor(prob, dtype=dtype, device=device)
        prob = self.accelerator.reduce(prob, reduction="mean")

        rank = self.accelerator.process_index
        prob = prob[sample_size * rank : sample_size * (rank + 1)]
        prob = prob[nonzero]

        return prob

    def _vectorize_prompt(self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor) -> torch.FloatTensor:
        """
        Replaces tokenizer.pad_token_id to embedding_tokenizer.pad_token_id
        and applies self.embedding_func
        """
        input_ids = torch.where(
            input_ids == self.tokenizer.pad_token_id,
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
        self, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
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
        Sample instances from dataset and get prompt embeddings.
        Used for density ratio classifier training.
        """
        n_samples = min(len(dataset), sample_size)
        rand_indices = np.random.choice(len(dataset), size=(n_samples,))

        batch = dataset.select(rand_indices)
        input_ids = pad_sequence(
            [torch.as_tensor(ids) for ids in batch["embedding_input_ids"]],
            batch_first=True,
            padding_value=self.embedding_tokenizer.pad_token_id,
        ).to(self.accelerator.device)
        attention_mask = pad_sequence(
            [torch.as_tensor(ids) for ids in batch["embedding_attention_mask"]],
            batch_first=True,
            padding_value=0,
        ).to(self.accelerator.device)

        with torch.no_grad():
            embeddings = self.embedding_func(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        return embeddings

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
            reference_KL_logps = []

            for padded_batch in tqdm(iterable=data_loader, desc="Train dataset reference log probs"):
                reference_completion_logp, reference_KL_logp = self.compute_reference_log_probs(padded_batch)

                reference_completion_logp = self.accelerator.gather_for_metrics(reference_completion_logp)
                reference_completion_logps.append(reference_completion_logp.cpu())

                reference_KL_logp = self.accelerator.gather_for_metrics(reference_KL_logp)
                reference_KL_logps.append(reference_KL_logp.cpu())

            self.train_dataset = self.train_dataset.add_column(
                name="reference_logps", column=torch.cat(reference_completion_logps).float().numpy()
            )
            self.train_dataset = self.train_dataset.add_column(
                name="reference_KL_logps", column=torch.cat(reference_KL_logps).float().numpy()
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
            reference_KL_logps = []

            for padded_batch in tqdm(iterable=data_loader, desc="Eval dataset reference log probs"):
                reference_completion_logp, reference_KL_logp = self.compute_reference_log_probs(padded_batch)

                reference_completion_logp = self.accelerator.gather_for_metrics(reference_completion_logp)
                reference_completion_logps.append(reference_completion_logp.cpu())

                reference_KL_logp = self.accelerator.gather_for_metrics(reference_KL_logp)
                reference_KL_logps.append(reference_KL_logp.cpu())

            eval_dataset = eval_dataset.add_column(
                name="reference_logps", column=torch.cat(reference_completion_logps).float().numpy()
            )
            eval_dataset = eval_dataset.add_column(
                name="reference_KL_logps", column=torch.cat(reference_KL_logps).float().numpy()
            )

            # Save calculated reference_chosen_logps and reference_rejected_logps to the eval_dataset for subsequent runs
            if self.eval_dataset is not None:
                self.eval_dataset = eval_dataset
            self._precomputed_eval_ref_log_probs = True

        return super().get_eval_dataloader(eval_dataset=eval_dataset)

    def compute_reference_log_probs(self, padded_batch: Dict) -> Dict:
        """Computes log probabilities of the reference model for a single padded batch of a KTO specific dataset."""
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

                        KL_logits = self.model(
                            padded_batch["KL_prompt_input_ids"],
                            attention_mask=padded_batch["KL_prompt_attention_mask"],
                            decoder_input_ids=padded_batch.get("KL_completion_decoder_input_ids"),
                            labels=padded_batch["KL_completion_labels"],
                        ).logits
                    else:
                        completion_logits = self.model(
                            padded_batch["completion_input_ids"],
                            attention_mask=padded_batch["completion_attention_mask"],
                        ).logits

                        KL_logits = self.model(
                            padded_batch["KL_completion_input_ids"],
                            attention_mask=padded_batch["KL_completion_attention_mask"],
                        ).logits
            else:
                if self.is_encoder_decoder:
                    completion_logits = self.ref_model(
                        padded_batch["prompt_input_ids"],
                        attention_mask=padded_batch["prompt_attention_mask"],
                        decoder_input_ids=padded_batch.get("completion_decoder_input_ids"),
                        labels=padded_batch["completion_labels"],
                    ).logits

                    KL_logits = self.ref_model(
                        padded_batch["KL_prompt_input_ids"],
                        attention_mask=padded_batch["KL_prompt_attention_mask"],
                        decoder_input_ids=padded_batch.get("KL_completion_decoder_input_ids"),
                        labels=padded_batch["KL_completion_labels"],
                    ).logits
                else:
                    completion_logits = self.ref_model(
                        padded_batch["completion_input_ids"], attention_mask=padded_batch["completion_attention_mask"]
                    ).logits

                    KL_logits = self.ref_model(
                        padded_batch["KL_completion_input_ids"],
                        attention_mask=padded_batch["KL_completion_attention_mask"],
                    ).logits

        completion_logps = self.get_batch_logps(
            completion_logits,
            padded_batch["completion_labels"],
            average_log_prob=False,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )

        KL_logps = self.get_batch_logps(
            KL_logits,
            padded_batch["KL_completion_labels"],
            average_log_prob=False,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )

        return completion_logps, KL_logps

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
            labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
            average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
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

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return (per_token_logps * loss_mask).sum(-1)

    def forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        KL_model_kwargs = (
            {
                "input_ids": batch["KL_prompt_input_ids"],
                "attention_mask": batch["KL_prompt_attention_mask"],
                "labels": batch["KL_completion_labels"],
                "decoder_input_ids": batch.get("KL_completion_decoder_input_ids"),
            }
            if self.is_encoder_decoder
            else {
                "input_ids": batch["KL_completion_input_ids"],
                "attention_mask": batch["KL_completion_attention_mask"],
            }
        )
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

        with torch.no_grad():
            KL_logits = model(
                **KL_model_kwargs,
            ).logits

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

        KL_logps = self.get_batch_logps(
            KL_logits,
            batch["KL_completion_labels"],
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
            return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, KL_logps, outputs.aux_loss)
        else:
            return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, KL_logps)

    def kto_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        policy_KL_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        reference_KL_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the KTO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (num(chosen) in batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (num(rejected) in batch_size,)
            policy_KL_logps: Log probabilities of the policy model for the KL responses. Shape: (batch_size,)
            reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (num(chosen) in batch_size,)
            reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (num(rejected) in batch_size,)
            reference_KL_logps: Log probabilities of the reference model for the KL responses. Shape: (batch_size,)

        Returns:
            A tuple of four tensors: (losses, chosen_rewards, rejected_rewards, KL).
            The losses tensor contains the KTO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
            The KL tensor contains the detached KL divergence estimate between the policy and reference models.
        """
        kl = (policy_KL_logps - reference_KL_logps).mean().detach()
        kl = self.accelerator.gather(kl).mean().clamp(min=0)

        if policy_chosen_logps.shape[0] != 0 or reference_chosen_logps.shape[0] != 0:
            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            chosen_losses = 1 - F.sigmoid(self.beta * (chosen_logratios - kl))
            chosen_rewards = self.beta * chosen_logratios.detach()
        else:
            # lists can't be empty -- if they are, then accelerate.gather will hang
            chosen_losses = torch.Tensor([]).to(self.accelerator.device)
            chosen_rewards = torch.Tensor([]).to(self.accelerator.device)

        if policy_rejected_logps.shape[0] != 0 or reference_rejected_logps.shape[0] != 0:
            rejected_logratios = policy_rejected_logps - reference_rejected_logps
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
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the BCO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (num(chosen) in batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (num(rejected) in batch_size,)
            reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (num(chosen) in batch_size,)
            reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (num(rejected) in batch_size,)
            chosen_embeddings: embeddings of desirable prompts
            rejected_embeddings: embeddings of undesirable prompts

        Returns:
            A tuple of four tensors: (losses, chosen_rewards, rejected_rewards, KL).
            The losses tensor contains the KTO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
            The delta value contains the moving average of all implicit rewards.
        """
        assert policy_chosen_logps.shape[0] > 0, f"no chosen data at {self.accelerator.local_process_index}"
        assert policy_rejected_logps.shape[0] > 0, f"no rejected data at {self.accelerator.local_process_index}"
        chosen_logratios = policy_chosen_logps - reference_chosen_logps
        rejected_logratios = policy_rejected_logps - reference_rejected_logps

        chosen_rewards = self.beta * chosen_logratios
        rejected_rewards = self.beta * rejected_logratios

        rewards = torch.cat((chosen_rewards, rejected_rewards), 0).mean().detach()
        self.running.update(rewards)
        rewards_mean = self.running.mean

        chosen_losses = -F.logsigmoid(chosen_rewards - rewards_mean)
        rejected_losses = -F.logsigmoid(-(rejected_rewards - rewards_mean))

        if self.match_underlying_distribution:
            chosen_weight = torch.ones_like(chosen_losses)
            rejected_weight = self._get_udm_weight(rejected_embeddings)

            losses = torch.cat((chosen_weight * chosen_losses, rejected_weight * rejected_losses), dim=0)
        else:
            losses = torch.cat((chosen_losses, rejected_losses), dim=0)

        return losses, chosen_rewards, rejected_rewards, torch.as_tensor(rewards_mean)

    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
    ):
        """Compute the KTO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}
        batch = {k: (v.to(self.accelerator.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

        forward_output = self.forward(model, batch)
        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            policy_KL_logps,
        ) = forward_output[:5]
        if self.aux_loss_enabled:
            aux_loss = forward_output[5]

        # if reference_logps in batch use them, otherwise use the reference model
        if "reference_logps" in batch:
            chosen_idx = [i for i in range(batch["reference_logps"].shape[0]) if batch["label"][i] is True]
            rejected_idx = [i for i in range(batch["reference_logps"].shape[0]) if batch["label"][i] is False]

            reference_chosen_logps = batch["reference_logps"][chosen_idx, ...]
            reference_rejected_logps = batch["reference_logps"][rejected_idx, ...]
            reference_KL_logps = batch["reference_KL_logps"]
        else:
            with torch.no_grad():
                if self.ref_model is None:
                    with self.null_ref_context():
                        (
                            reference_chosen_logps,
                            reference_rejected_logps,
                            _,
                            _,
                            reference_KL_logps,
                        ) = self.forward(self.model, batch)[:5]
                else:
                    (
                        reference_chosen_logps,
                        reference_rejected_logps,
                        _,
                        _,
                        reference_KL_logps,
                    ) = self.forward(self.ref_model, batch)[:5]

        if self.loss_type == "kto":
            losses, chosen_rewards, rejected_rewards, kl = self.kto_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                policy_KL_logps,
                reference_chosen_logps,
                reference_rejected_logps,
                reference_KL_logps,
            )
        elif self.loss_type == "bco":
            chosen_embeddings, rejected_embeddings = self._get_prompt_embeddings(batch)

            losses, chosen_rewards, rejected_rewards, kl = self.bco_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                reference_chosen_logps,
                reference_rejected_logps,
                chosen_embeddings,
                rejected_embeddings,
            )

        num_chosen = torch.Tensor([len(chosen_rewards)]).to(self.accelerator.device)
        num_rejected = torch.Tensor([len(rejected_rewards)]).to(self.accelerator.device)

        all_num_chosen = self.accelerator.gather(num_chosen).sum().item()
        all_num_rejected = self.accelerator.gather(num_rejected).sum().item()

        if all_num_chosen > 0:
            metrics["rewards/chosen_sum"] = self.accelerator.gather(chosen_rewards.nansum()).nansum().item()
            metrics["logps/chosen_sum"] = self.accelerator.gather(policy_chosen_logps.nansum()).nansum().item()
            metrics["count/chosen"] = all_num_chosen

        if all_num_rejected > 0:
            metrics["rewards/rejected_sum"] = self.accelerator.gather(rejected_rewards.nansum()).nansum().item()
            metrics["logps/rejected_sum"] = self.accelerator.gather(policy_rejected_logps.nansum()).nansum().item()
            metrics["count/rejected"] = all_num_rejected

        metrics["kl"] = kl.item()

        loss = losses.nanmean()
        if self.aux_loss_enabled:
            loss += getattr(model.config, "router_aux_loss_coef", 0.0) * aux_loss

        return loss, metrics

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if not self.use_dpo_data_collator:
            warnings.warn(
                "compute_loss is only implemented for DPODataCollatorWithPadding, and you passed a datacollator that is different than "
                "DPODataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )
        compute_loss_context_manager = torch.cuda.amp.autocast if self._peft_has_been_casted_to_bf16 else nullcontext

        with compute_loss_context_manager():
            loss, metrics = self.get_batch_loss_metrics(model, inputs)

        # Make sure to move the loss to the device the original accumulating loss is at back in the `Trainer` class:
        loss = loss.to(self.args.device)
        # force log the metrics
        if self.accelerator.is_main_process:
            self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return (loss, metrics)
        return loss

    def store_metrics(self, metrics: Dict[str, float], train_eval: Literal["train", "eval"] = "train") -> None:
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None
        return SequentialSampler(self.train_dataset)

    def get_batch_samples(self, model, batch: Dict[str, torch.LongTensor]) -> Tuple[str, str]:
        """Generate samples from the model and reference model for the given batch of inputs."""

        # If one uses `generate_during_eval` with peft + bf16, we need to explicitly call generate with
        # the torch cuda amp context manager as some hidden states are silently casted to full precision.
        generate_context_manager = nullcontext if not self._peft_has_been_casted_to_bf16 else torch.cuda.amp.autocast

        with generate_context_manager():
            policy_output = model.generate(
                input_ids=batch["prompt_input_ids"],
                attention_mask=batch["prompt_attention_mask"],
                max_length=self.max_length,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
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
                            pad_token_id=self.tokenizer.pad_token_id,
                        )
                else:
                    reference_output = self.ref_model.generate(
                        input_ids=batch["prompt_input_ids"],
                        attention_mask=batch["prompt_attention_mask"],
                        max_length=self.max_length,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )

        policy_output = pad_to_length(policy_output, self.max_length, self.tokenizer.pad_token_id)
        policy_output_decoded = self.tokenizer.batch_decode(policy_output, skip_special_tokens=True)

        reference_output = pad_to_length(reference_output, self.max_length, self.tokenizer.pad_token_id)
        reference_output_decoded = self.tokenizer.batch_decode(reference_output, skip_special_tokens=True)

        return policy_output_decoded, reference_output_decoded

    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
        if not self.use_dpo_data_collator:
            warnings.warn(
                "prediction_step is only implemented for DPODataCollatorWithPadding, and you passed a datacollator that is different than "
                "DPODataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )
        if ignore_keys is None:
            if hasattr(model, "config"):
                ignore_keys = getattr(model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        prediction_context_manager = torch.cuda.amp.autocast if self._peft_has_been_casted_to_bf16 else nullcontext
        with torch.no_grad(), prediction_context_manager():
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
        logits = tuple(v.unsqueeze(dim=0) for k, v in logits_dict.items() if k not in ignore_keys)
        logits = torch.stack(logits).mean(axis=1).to(self.accelerator.device)
        labels = torch.zeros(logits.shape[0], device=self.accelerator.device)

        return (loss.detach(), logits, labels)

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
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

            target_indicies = [i for i in range(len(random_batch["kl"])) if random_batch["kl"][i] is False]
            target_batch = {
                "prompt_input_ids": itemgetter(*target_indicies)(random_batch["prompt_input_ids"]),
                "prompt_attention_mask": itemgetter(*target_indicies)(random_batch["prompt_attention_mask"]),
                "prompt": itemgetter(*target_indicies)(random_batch["prompt"]),
            }
            policy_output_decoded, ref_output_decoded = self.get_batch_samples(self.model, target_batch)

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

    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training, including stored metrics.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        # logs either has 'loss' or 'eval_loss'
        train_eval = "train" if "loss" in logs else "eval"
        # train metrics should have no prefix, eval should have 'eval_'
        prefix = "eval_" if train_eval == "eval" else ""
        # accumulate average metrics from sums and lengths
        for split in ["chosen", "rejected"]:
            if f"count/{split}" in self._stored_metrics[train_eval]:
                count_sum = torch.Tensor(self._stored_metrics[train_eval][f"count/{split}"]).sum().item()
                logs[f"{prefix}rewards/{split}"] = (
                    torch.Tensor(self._stored_metrics[train_eval][f"rewards/{split}_sum"]).sum().item() / count_sum
                )
                logs[f"{prefix}logps/{split}"] = (
                    torch.Tensor(self._stored_metrics[train_eval][f"logps/{split}_sum"]).sum().item() / count_sum
                )
                for key in [f"count/{split}", f"rewards/{split}_sum", f"logps/{split}_sum"]:
                    del self._stored_metrics[train_eval][key]
        # calculate reward margin
        if f"{prefix}rewards/chosen" in logs and f"{prefix}rewards/rejected" in logs:
            logs[f"{prefix}rewards/margins"] = logs[f"{prefix}rewards/chosen"] - logs[f"{prefix}rewards/rejected"]
        # Add averaged stored metrics to logs
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[f"{prefix}{key}"] = torch.Tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]
        return super().log(logs)

    @wraps(Trainer.push_to_hub)
    def push_to_hub(self, commit_message: Optional[str] = "End of training", blocking: bool = True, **kwargs) -> str:
        """
        Overwrite the `push_to_hub` method in order to force-add the tag "kto" when pushing the
        model on the Hub. Please refer to `~transformers.Trainer.push_to_hub` for more details.
        """
        kwargs = trl_sanitze_kwargs_for_tagging(model=self.model, tag_names=self._tag_names, kwargs=kwargs)

        return super().push_to_hub(commit_message=commit_message, blocking=blocking, **kwargs)
