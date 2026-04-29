# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
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

import textwrap
from collections import defaultdict
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from accelerate import PartialState, logging
from accelerate.utils import is_peft_model, tqdm
from datasets import Dataset, IterableDataset, IterableDatasetDict, concatenate_datasets
from packaging.version import Version
from torch.utils.data import DataLoader, SequentialSampler
from transformers import (
    AutoProcessor,
    DataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainerCallback,
)
from transformers.data.data_collator import DataCollatorMixin
from transformers.trainer_utils import EvalLoopOutput, has_length
from transformers.utils import is_peft_available

from ...data_utils import (
    extract_prompt,
    is_conversational,
    unpair_preference_dataset,
)
from ...import_utils import is_liger_kernel_available
from ...models.utils import prepare_deepspeed, prepare_fsdp
from ...trainer.base_trainer import _BaseTrainer
from ...trainer.utils import (
    create_model_from_path,
    disable_dropout_in_model,
    get_config_model_id,
    pad,
    selective_log_softmax,
    use_adapter,
)
from .kto_config import KTOConfig


if is_liger_kernel_available():
    from liger_kernel.chunked_loss import LigerFusedLinearKTOLoss

if is_peft_available():
    from peft import PeftConfig, PeftModel, get_peft_model


if TYPE_CHECKING:
    from transformers import PreTrainedModel


logger = logging.get_logger(__name__)

RUNNING_NAME = "running.pt"


def get_dataset_column_names(dataset: Dataset | IterableDataset) -> list[str]:
    return list(next(iter(dataset)).keys()) if dataset.column_names is None else dataset.column_names


def _get_kl_dataset(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
    """
    Creates mismatched pairs of prompts and completions for the KL dataset by adding a +1 offset to the order of
    completions. For best results, the mismatched outputs y' used to estimate the KL term for a batch should be the
    same set as the matched outputs y used to estimate the rewards in that batch, just paired with different x.
    """
    batch["completion_ids"] = [batch["completion_ids"][-1]] + batch["completion_ids"][:-1]
    return batch


@dataclass
class DataCollatorForUnpairedPreference(DataCollatorMixin):
    """
    Data collator for unpaired preference data. Assembles completions from raw token IDs and pads sequences to the
    maximum length of the batch.

    Args:
        pad_token_id (`int`):
            Token ID to use for padding `input_ids` sequences.
        max_length (`int`, *optional*):
            Maximum sequence length after assembly. Sequences longer than `max_length` are truncated from the end.
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            The tensor type to return. Currently, only `"pt"` (PyTorch tensors) is supported.
    """

    pad_token_id: int
    max_length: int | None = None
    return_tensors: str = "pt"

    def torch_call(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        batch = {}
        for prefix, ids_key in [("completion", "completion_ids"), ("KL_completion", "KL_completion_ids")]:
            if ids_key not in examples[0]:
                continue

            full_ids_list = []
            labels_list = []
            for ex in examples:
                prompt_ids = ex["prompt_ids"]
                answer_ids = ex[ids_key]
                full_ids = prompt_ids + answer_ids
                labels = [-100] * len(prompt_ids) + answer_ids
                if self.max_length is not None:
                    full_ids = full_ids[: self.max_length]
                    labels = labels[: self.max_length]
                full_ids_list.append(full_ids)
                labels_list.append(labels)

            batch[f"{prefix}_input_ids"] = pad(
                [torch.tensor(ids, dtype=torch.int64) for ids in full_ids_list],
                padding_value=self.pad_token_id,
                padding_side="right",
            )
            batch[f"{prefix}_attention_mask"] = pad(
                [torch.ones(len(ids), dtype=torch.int64) for ids in full_ids_list],
                padding_value=0,
                padding_side="right",
            )
            batch[f"{prefix}_labels"] = pad(
                [torch.tensor(lbl, dtype=torch.int64) for lbl in labels_list],
                padding_value=-100,
                padding_side="right",
            )

        if "reference_logps" in examples[0]:
            batch["reference_logps"] = torch.tensor([ex["reference_logps"] for ex in examples])
        if "reference_KL_logps" in examples[0]:
            batch["reference_KL_logps"] = torch.tensor([ex["reference_KL_logps"] for ex in examples])
        batch["label"] = [ex["label"] for ex in examples]
        return batch


class KTOTrainer(_BaseTrainer):
    """
    Initialize KTOTrainer.

    Args:
        model (`str` or [`~transformers.PreTrainedModel`] or [`~peft.PeftModel`]):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or a
              path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
              using `<ModelArchitecture>.from_pretrained` (where `<ModelArchitecture>` is derived from the model
              config) with the keyword arguments in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
            - A [`~peft.PeftModel`] object. Only causal language models are supported.
        ref_model ([`~transformers.PreTrainedModel`], *optional*):
            Reference model used to compute the reference log probabilities.

            - If provided, this model is used directly as the reference policy.
            - If `None`, the trainer will automatically use the initial policy corresponding to `model`, i.e. the model
              state before KTO training starts.
        args ([`experimental.kto.KTOConfig`], *optional*):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            The dataset to use for training.
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Dataset | IterableDataset]`):
            The dataset to use for evaluation.
        processing_class ([`~transformers.PreTrainedTokenizerBase`] or [`~transformers.ProcessorMixin`], *optional*):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoProcessor.from_pretrained`]. A
            padding token, `tokenizer.pad_token`, must be set. If the processing class has not set a padding token,
            `tokenizer.eos_token` will be used as the default.
        data_collator ([`~transformers.DataCollator`], *optional*):
            The data collator to use for training. If None is specified, the default data collator
            ([`~experimental.kto.kto_trainer.DataCollatorForUnpairedPreference`]) will be used which will pad the
            sequences to the maximum length of the sequences in the batch.
        callbacks (`list[transformers.TrainerCallback]`):
            The callbacks to use for training.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`):
            The optimizer and scheduler to use for training.
        peft_config ([`~peft.PeftConfig`], *optional*):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
        compute_metrics (`Callable[[EvalPrediction], dict]`, *optional*):
            The function to use to compute the metrics. Must take a `EvalPrediction` and return a dictionary string to
            metric values.
    """

    _tag_names = ["trl", "kto"]
    _name = "KTO"
    _paper = {
        "title": "KTO: Model Alignment as Prospect Theoretic Optimization",
        "id": "2402.01306",
        # docstyle-ignore
        "citation": textwrap.dedent("""\
            @article{ethayarajh2024kto,
                title        = {{KTO: Model Alignment as Prospect Theoretic Optimization}},
                author       = {Kawin Ethayarajh and Winnie Xu and Niklas Muennighoff and Dan Jurafsky and Douwe Kiela},
                year         = 2024,
                eprint       = {arXiv:2402.01306},
            }"""),
    }

    def __init__(
        self,
        model: "str | PreTrainedModel | PeftModel",
        ref_model: PreTrainedModel | None = None,
        args: KTOConfig | None = None,
        train_dataset: Dataset | IterableDataset | None = None,
        eval_dataset: Dataset | IterableDataset | dict[str, Dataset | IterableDataset] | None = None,
        processing_class: PreTrainedTokenizerBase | ProcessorMixin | None = None,
        data_collator: DataCollator | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        peft_config: "PeftConfig | None" = None,
        compute_metrics: Callable[[EvalLoopOutput], dict] | None = None,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else get_config_model_id(model.config)
            model_name = model_name.split("/")[-1]
            args = KTOConfig(f"{model_name}-KTO")

        if train_dataset is None:
            raise ValueError("`train_dataset` is required")
        elif isinstance(train_dataset, IterableDataset):
            # IterableDataset requires dispatch_batches=False because Accelerate's dispatch mode may try to concatenate
            # batches from multiple processes, leading to mismatch errors.
            if args.accelerator_config.dispatch_batches is True:
                logger.warning(
                    "You are using an `IterableDataset` for training with `dispatch_batches=True`. `dispatch_batches` "
                    "is forced to `False` when using an `IterableDataset`. To remove this warning, unset "
                    "`dispatch_batches` in `KTOConfig` or set it to `False`."
                )
            args.accelerator_config.dispatch_batches = False

        # Model
        if isinstance(model, str):
            model_init_kwargs = args.model_init_kwargs or {}
            # Distributed training requires device_map=None ("auto" fails)
            if args.distributed_state.distributed_type in ["MULTI_GPU", "DEEPSPEED"]:
                model_init_kwargs["device_map"] = None
            model = create_model_from_path(model, **model_init_kwargs)
        else:
            if args.model_init_kwargs is not None:
                logger.warning(
                    "You passed `model_init_kwargs` to the KTOConfig, but your model is already instantiated. "
                    "The `model_init_kwargs` will be ignored."
                )
        if ref_model is model:
            raise ValueError(
                "`model` and `ref_model` cannot be the same object. In most cases you should omit `ref_model` and "
                "we'll initialize it to a copy of `model` for you."
            )

        # Processing class
        if processing_class is None:
            processing_class = AutoProcessor.from_pretrained(get_config_model_id(model.config))
        if isinstance(processing_class, ProcessorMixin):
            self._tokenizer = processing_class.tokenizer
        elif isinstance(processing_class, PreTrainedTokenizerBase):
            self._tokenizer = processing_class
        else:
            raise TypeError("The `processing_class` must be either a `PreTrainedTokenizerBase` or a `ProcessorMixin`")
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # PEFT
        if peft_config is not None:
            if not is_peft_available():
                raise ImportError(
                    "You passed `peft_config` but the `peft` library is not installed. "
                    "Install it with `pip install trl[peft]`."
                )
            if not isinstance(peft_config, PeftConfig):
                raise TypeError(
                    f"`peft_config` must be a `peft.PeftConfig` instance (e.g. `peft.LoraConfig`), "
                    f"got {type(peft_config).__name__}."
                )
            if is_peft_model(model):
                raise ValueError(
                    "You passed a `PeftModel` instance together with a `peft_config` to the trainer. Please first merge "
                    "and unload the existing adapter, save the resulting base model, and then pass that base model along "
                    "with the new `peft_config` to the trainer."
                )
            # Create PEFT model
            model = get_peft_model(model, peft_config)

        elif is_peft_model(model) and ref_model is None:
            # If the model is a PEFT model with a pretrained adapter, we need to create a "ref" adapter that is a copy
            # of the "default" adapter, so that we can use it as the reference model during KTO training.
            model.add_adapter("ref", model.peft_config["default"])
            for name, param in model.named_parameters():
                if ".default." in name:
                    ref_name = name.replace(".default.", ".ref.")
                    ref_param = model.get_parameter(ref_name)
                    ref_param.data.copy_(param.data)

        # When using gradient checkpointing with PEFT, we need to enable input gradients. transformers.Trainer normally
        # handles this, but a bug currently prevents it; see https://github.com/huggingface/transformers/issues/42489
        if is_peft_model(model) and args.gradient_checkpointing:
            model.enable_input_require_grads()

        # When using QLoRA, the PEFT adapter weights are converted to bf16 to follow the recommendations from the
        # original paper (see https://huggingface.co/papers/2305.14314, paragraph 3). Normally, this can be done by
        # passing `autocast_adapter_dtype=False` to `get_peft_model`, but this option is not yet supported for
        # quantized models. See: https://github.com/huggingface/peft/issues/2889
        # Non-quantized models do not have the `is_loaded_in_{8,4}bit` attributes, whereas quantized models do
        if getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False):
            for param in model.parameters():
                if param.requires_grad:
                    param.data = param.data.to(torch.bfloat16)

        # KTO only supports causal language models, not encoder-decoder models
        if model is not None and hasattr(model.config, "is_encoder_decoder") and model.config.is_encoder_decoder:
            raise ValueError(
                "KTO only supports causal language models. Encoder-decoder models are not supported. "
                "Please use a causal LM (e.g., GPT, Llama, Mistral) instead of an encoder-decoder model (e.g., T5, BART)."
            )

        if args.max_length is None:
            logger.warning(
                "When using DataCollatorForUnpairedPreference, you should set `max_length` in the KTOTrainer's init"
                " it will be set to `512` by default, but you should do it yourself in the future.",
            )
            max_length = 512
        if args.max_length is not None:
            max_length = args.max_length

        if data_collator is None:
            data_collator = DataCollatorForUnpairedPreference(
                pad_token_id=self._tokenizer.pad_token_id,
                max_length=max_length,
            )

            if args.remove_unused_columns:
                args.remove_unused_columns = False
                # warn users
                logger.warning(
                    "When using DataCollatorForUnpairedPreference, you should set `remove_unused_columns=False` in your KTOConfig"
                    " we have set it for you, but you should do it yourself in the future.",
                )

            self.use_dpo_data_collator = True
        else:
            self.use_dpo_data_collator = False

        self.loss_type = args.loss_type
        self.max_length = max_length
        self.precompute_ref_log_probs = args.precompute_ref_log_probs

        # Not all losses require a KL calculation
        self.calculate_KL = True
        if self.loss_type in ["apo_zero_unpaired"]:
            self.calculate_KL = False
        if self.calculate_KL and args.per_device_train_batch_size <= 1:
            raise ValueError(
                "Actual (not effective) batch size must be > 1. KTO will not work properly because the KL term will be equivalent to the implied reward."
            )

        # metric
        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        # KTO parameter
        self.beta = args.beta
        self.desirable_weight = args.desirable_weight
        self.undesirable_weight = args.undesirable_weight
        self.aux_loss_enabled = getattr(model.config, "output_router_logits", False)
        self.aux_loss_coef = getattr(model.config, "router_aux_loss_coef", 0.0)
        if self.aux_loss_enabled and self.aux_loss_coef == 0.0:
            logger.warning(
                "You set `output_router_logits` to `True` in the model config, but `router_aux_loss_coef` is set to "
                "`0.0`, meaning the auxiliary loss will not be used. Either set `router_aux_loss_coef` to a value "
                "greater than `0.0`, or set `output_router_logits` to `False` if you don't want to use the auxiliary "
                "loss.",
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

        # Transformers explicitly set use_reentrant=True in the past to silence a PyTorch warning, but the default was
        # never updated once PyTorch switched to recommending use_reentrant=False. Until that change lands upstream
        # (see https://github.com/huggingface/transformers/pull/43203) and is released (most likely in 5.0.0), we
        # default to the recommended non-reentrant behavior here, while preserving any user-provided value.
        if args.gradient_checkpointing and Version(transformers.__version__) < Version("5.0.0"):
            args.gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs or {}
            args.gradient_checkpointing_kwargs.setdefault("use_reentrant", False)

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
        )

        # Reference model
        if ref_model is None:
            if is_peft_model(self.model) or args.precompute_ref_log_probs:
                # If PEFT is used, the reference model is not needed since the adapter can be disabled to revert to the
                # initial model. If precompute_ref_log_probs is True, the reference model does not need to be kept in
                # memory during training.
                self.ref_model = None
            else:
                ref_model_init_kwargs = args.model_init_kwargs or {}
                # Distributed training requires device_map=None ("auto" fails)
                if self.args.distributed_state.distributed_type in ["MULTI_GPU", "DEEPSPEED"]:
                    ref_model_init_kwargs["device_map"] = None
                ref_model_path = get_config_model_id(self.model.config)
                self.ref_model = create_model_from_path(ref_model_path, **ref_model_init_kwargs)
        else:
            self.ref_model = ref_model

        # Disable dropout in the model and reference model
        if args.disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        # Add tags to the model
        self.model.add_model_tags(self._tag_names)

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            elif self.is_fsdp_enabled:
                self.ref_model = prepare_fsdp(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        # Import Liger kernel if enabled
        if self.args.use_liger_kernel:
            if not is_liger_kernel_available():
                raise ImportError(
                    "You set `use_liger_kernel=True` but the liger kernel is not available. "
                    "Please install liger-kernel first: `pip install liger-kernel`"
                )
            if self.loss_type in ["apo_zero_unpaired"]:
                raise ValueError(
                    "You cannot set `loss_type='apo_zero_unpaired'` with liger-kernel."
                    "Only KTO loss is supported with liger-kernel."
                )
            if self.precompute_ref_log_probs:
                raise ValueError(
                    "You cannot use `precompute_ref_log_probs=True` with liger kernel. Please set "
                    "`precompute_ref_log_probs=False`."
                )
            if is_peft_model(self.model):
                raise ValueError(
                    "You cannot use `use_liger_kernel=True` with Peft models. Please set `use_liger_kernel=False`."
                )
            self.kto_loss_fn = LigerFusedLinearKTOLoss(beta=self.beta, use_ref_model=(self.ref_model is not None))

        if self.precompute_ref_log_probs:
            if isinstance(self.train_dataset, IterableDataset) or isinstance(
                self.eval_dataset, (IterableDataset, IterableDatasetDict)
            ):
                raise ValueError(
                    "`precompute_ref_log_probs=True` is not supported with IterableDataset. Please use a map-style "
                    "Dataset or set `precompute_ref_log_probs=False`."
                )
            self.train_dataset = self._precompute_ref_logps(
                self.train_dataset,
                "train",
                self.args.precompute_ref_batch_size or self.args.per_device_train_batch_size,
            )
            if self.eval_dataset is not None:
                if isinstance(self.eval_dataset, dict):
                    self.eval_dataset = {
                        name: self._precompute_ref_logps(
                            dataset, name, self.args.precompute_ref_batch_size or self.args.per_device_eval_batch_size
                        )
                        for name, dataset in self.eval_dataset.items()
                    }
                else:
                    self.eval_dataset = self._precompute_ref_logps(
                        self.eval_dataset,
                        "eval",
                        self.args.precompute_ref_batch_size or self.args.per_device_eval_batch_size,
                    )

    def _tokenize(
        self,
        processing_class: PreTrainedTokenizerBase | ProcessorMixin,
        input: str | list,
        **kwargs,
    ) -> dict[str, list]:
        """Tokenize a single example for dataset preprocessing.

        Dispatches to `apply_chat_template` for conversational input (list of message dicts) and to `__call__` for
        non-conversational input (str).

        Args:
            processing_class ([`~transformers.PreTrainedTokenizerBase`] or [`~transformers.ProcessorMixin`]):
                The tokenizer or processor to use.
            input (`str` or `list`):
                A string for non-conversational input, or a list of message dicts for conversational input.
            **kwargs:
                Forwarded to `apply_chat_template` (e.g. `add_generation_prompt`, `return_assistant_tokens_mask`).

        Returns:
            `dict` with at least an `"input_ids"` key mapping to a flat `list[int]`.
        """
        if isinstance(input, list):  # conversational: list of message dicts
            result = processing_class.apply_chat_template(input, tokenize=True, return_dict=True, **kwargs)
        else:  # non-conversational: plain text string
            result = processing_class(text=input)
        return result

    def _prepare_dataset(
        self,
        dataset: Dataset | IterableDataset,
        processing_class: PreTrainedTokenizerBase | ProcessorMixin,
        args: KTOConfig | None,
        dataset_name: str,
    ) -> Dataset | IterableDataset:
        # Build the kwargs for the `map` function
        map_kwargs = {}
        if isinstance(dataset, Dataset):  # IterableDataset does not support num_proc
            map_kwargs["num_proc"] = args.dataset_num_proc

        # Compute that only on the main process for faster data processing.
        # see: https://github.com/huggingface/trl/pull/1255
        with PartialState().main_process_first():
            # Extract the prompt if needed
            first_example = next(iter(dataset))
            if "prompt" not in first_example:
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Extracting prompt from {dataset_name} dataset"
                dataset = dataset.map(extract_prompt, **map_kwargs)

            # Unpair the dataset if needed
            first_example = next(iter(dataset))
            if "chosen" in first_example and "rejected" in first_example:
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Unpairing {dataset_name} dataset"
                dataset = unpair_preference_dataset(dataset, **map_kwargs)

            # Add EOS token if needed: non-conversational only
            first_example = next(iter(dataset))
            if not is_conversational(first_example):
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Adding EOS to {dataset_name} dataset"

                def add_eos(example, eos_token):
                    if not example["completion"].endswith(eos_token):
                        example["completion"] = example["completion"] + eos_token
                    return example

                dataset = dataset.map(add_eos, fn_kwargs={"eos_token": self._tokenizer.eos_token}, **map_kwargs)

            # Tokenize dataset
            if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                map_kwargs["desc"] = f"Tokenizing {dataset_name} dataset"

            def tokenize_fn(example, processing_class):
                if is_conversational(example):
                    chat_template_kwargs = example.get("chat_template_kwargs", {})
                    prompt_ids = self._tokenize(
                        processing_class,
                        example["prompt"],
                        add_generation_prompt=True,
                        **chat_template_kwargs,
                    )["input_ids"]
                    prompt_completion_ids = self._tokenize(
                        processing_class,
                        example["prompt"] + example["completion"],
                        **chat_template_kwargs,
                    )["input_ids"]
                else:
                    prompt_ids = self._tokenize(processing_class, example["prompt"])["input_ids"]
                    prompt_completion_ids = self._tokenize(
                        processing_class, example["prompt"] + example["completion"]
                    )["input_ids"]

                if not prompt_completion_ids[: len(prompt_ids)] == prompt_ids:
                    logger.warning(
                        "Mismatch between tokenized prompt and the start of tokenized prompt+completion. "
                        "This may be due to unexpected tokenizer behavior, whitespace issues, or special "
                        "token handling. Verify that the tokenizer is processing text consistently."
                    )

                return {
                    "prompt_ids": prompt_ids,
                    "completion_ids": prompt_completion_ids[len(prompt_ids) :],
                }

            dataset = dataset.map(tokenize_fn, fn_kwargs={"processing_class": processing_class}, **map_kwargs)

            # Get KL datasets if needed
            if self.calculate_KL:

                def rename_kl_fn(example):
                    return {"KL_completion_ids": example["completion_ids"]}

                # create pairs for estimating the KL term by flipping the matched pairs in each batch of size total_batch_size
                # i.e., (x_1, y_1), ..., (x_n, y_n) --> (x_1, y_n), ..., (x_n, y_1) = (x'_1, y'_1), ..., (x'_n, y'_n)
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Extracting KL {dataset_name} dataset"
                kl_dataset = dataset.map(
                    _get_kl_dataset, batched=True, batch_size=args.per_device_train_batch_size, **map_kwargs
                )

                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Assembling KL {dataset_name} dataset"
                column_names = get_dataset_column_names(dataset)
                kl_dataset = kl_dataset.map(
                    rename_kl_fn,
                    remove_columns=[c for c in get_dataset_column_names(kl_dataset) if c in column_names],
                    **map_kwargs,
                )

                # merge the datasets
                dataset = concatenate_datasets([dataset, kl_dataset], axis=1)

            # Calculate dataset desirability balance
            if dataset_name == "train" and isinstance(dataset, Dataset):  # IterableDataset does not support len
                num_desirable = max(sum(dataset["label"]), 1)
                num_undesirable = max(len(dataset["label"]) - num_desirable, 1)  # "label" is binary

                if num_desirable != num_undesirable:
                    # The lower and upper bounds come from Eq. (8) of https://huggingface.co/papers/2402.01306
                    des_weight_lower_bound = round((num_undesirable * self.undesirable_weight / num_desirable) * 1, 2)
                    des_weight_upper_bound = round(
                        (num_undesirable * self.undesirable_weight / num_desirable) * 1.33, 2
                    )
                    und_weight_lower_bound = round((num_desirable * self.desirable_weight / num_undesirable) / 1.33, 2)
                    und_weight_upper_bound = round((num_desirable * self.desirable_weight / num_undesirable) / 1, 2)

                    des_weight_in_range = des_weight_lower_bound <= self.desirable_weight <= des_weight_upper_bound
                    und_weight_in_range = und_weight_lower_bound <= self.undesirable_weight <= und_weight_upper_bound

                    if not (des_weight_in_range or und_weight_in_range):
                        logger.warning(
                            "You have different amounts of desirable/positive and undesirable/negative examples but the "
                            "weights on the desirable and undesirable losses don't seem to be in an ideal range. Based "
                            f"on your data, we recommend EITHER "
                            f"desirable_weight in [{des_weight_lower_bound}, {des_weight_upper_bound}] or "
                            f"undesirable_weight in [{und_weight_lower_bound}, {und_weight_upper_bound}] (but NOT BOTH). "
                            "See the documentation on how to optimally set these weights.",
                        )
        return dataset

    @contextmanager
    def null_ref_context(self):
        """Context manager for handling null reference model (that is, peft adapter manipulation)."""
        if is_peft_model(self.model):
            model = self.accelerator.unwrap_model(self.model)
            with use_adapter(model, adapter_name="ref" if "ref" in model.peft_config else None):
                yield
        else:
            yield

    def _precompute_ref_logps(self, dataset: Dataset, name: str, batch_size: int) -> Dataset:
        dataloader_params = {
            "batch_size": batch_size,
            "collate_fn": self.data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "shuffle": False,
        }
        data_loader = self.accelerator.prepare(DataLoader(dataset, **dataloader_params))
        reference_completion_logps = []
        reference_KL_logps = []
        for padded_batch in tqdm(iterable=data_loader, desc=f"Computing reference log probs for {name} dataset"):
            reference_completion_logp, reference_KL_logp = self.compute_reference_log_probs(padded_batch)
            reference_completion_logp = self.accelerator.gather_for_metrics(reference_completion_logp)
            reference_completion_logps.append(reference_completion_logp.cpu())
            if self.calculate_KL:
                reference_KL_logp = self.accelerator.gather_for_metrics(reference_KL_logp)
                reference_KL_logps.append(reference_KL_logp.cpu())
        dataset = dataset.add_column(
            name="reference_logps", column=torch.cat(reference_completion_logps).float().numpy()
        )
        if self.calculate_KL:
            dataset = dataset.add_column(
                name="reference_KL_logps", column=torch.cat(reference_KL_logps).float().numpy()
            )
        return dataset

    def compute_reference_log_probs(self, padded_batch: dict) -> dict:
        """Computes log probabilities of the reference model for a single padded batch of a KTO specific dataset."""
        with torch.no_grad():
            if self.ref_model is None:
                with self.null_ref_context():
                    completion_logits = self.model(
                        padded_batch["completion_input_ids"],
                        attention_mask=padded_batch["completion_attention_mask"],
                    ).logits

                    if self.calculate_KL:
                        KL_logits = self.model(
                            padded_batch["KL_completion_input_ids"],
                            attention_mask=padded_batch["KL_completion_attention_mask"],
                        ).logits
            else:
                completion_logits = self.ref_model(
                    padded_batch["completion_input_ids"], attention_mask=padded_batch["completion_attention_mask"]
                ).logits

                if self.calculate_KL:
                    KL_logits = self.ref_model(
                        padded_batch["KL_completion_input_ids"],
                        attention_mask=padded_batch["KL_completion_attention_mask"],
                    ).logits

        completion_logps = self.get_batch_logps(
            completion_logits,
            padded_batch["completion_labels"],
            average_log_prob=False,
        )

        if self.calculate_KL:
            KL_logps = self.get_batch_logps(
                KL_logits,
                padded_batch["KL_completion_labels"],
                average_log_prob=False,
            )
        else:
            KL_logps = None

        return completion_logps, KL_logps

    @staticmethod
    def get_batch_logps(
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = False,
    ) -> torch.FloatTensor:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits:
                Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels:
                Labels for which to compute the log probabilities. Label tokens with a value of `-100` are ignored.
                Shape: (batch_size, sequence_length)
            average_log_prob:
                If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the
                log probabilities of the (non-masked) tokens.

        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the
            given logits.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        # For causal LM, shift labels and logits by one position
        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]

        loss_mask = labels != -100

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == -100] = 0

        per_token_logps = selective_log_softmax(logits, labels)

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return (per_token_logps * loss_mask).sum(-1)

    def forward(
        self, model: nn.Module, batch: dict[str, list | torch.LongTensor]
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        KL_logps = self._compute_kl_logps(model, batch)

        model_kwargs = {}
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
        )

        if completion_logps.shape[0] != len(batch["label"]):
            raise ValueError(
                "There is a mismatch between the number of examples in this batch and the number of "
                "examples for which an output sequence was predicted."
            )

        # Use torch.nonzero for efficient tensor index selection
        device = completion_logits.device
        labels = torch.as_tensor(batch["label"], dtype=torch.bool, device=device)
        chosen_idx = torch.nonzero(labels, as_tuple=False).view(-1)
        rejected_idx = torch.nonzero(~labels, as_tuple=False).view(-1)

        # Use index_select for efficient CUDA operations
        chosen_logps = completion_logps.index_select(0, chosen_idx)
        rejected_logps = completion_logps.index_select(0, rejected_idx)

        chosen_logits = completion_logits.index_select(0, chosen_idx)
        rejected_logits = completion_logits.index_select(0, rejected_idx)

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
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the KTO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps:
                Log probabilities of the policy model for the chosen responses. Shape: (num(chosen) in batch_size,)
            policy_rejected_logps:
                Log probabilities of the policy model for the rejected responses. Shape: (num(rejected) in batch_size,)
            policy_KL_logps: Log probabilities of the policy model for the KL responses. Shape: (batch_size,)
            reference_chosen_logps:
                Log probabilities of the reference model for the chosen responses. Shape: (num(chosen) in batch_size,)
            reference_rejected_logps:
                Log probabilities of the reference model for the rejected responses. Shape: (num(rejected) in
                batch_size,)
            reference_KL_logps: Log probabilities of the reference model for the KL responses. Shape: (batch_size,)

        Returns:
            A tuple of four tensors: (losses, chosen_rewards, rejected_rewards, KL). The losses tensor contains the KTO
            loss for each example in the batch. The chosen_rewards and rejected_rewards tensors contain the rewards for
            the chosen and rejected responses, respectively. The KL tensor contains the detached KL divergence estimate
            between the policy and reference models.
        """
        if self.calculate_KL:
            kl = (policy_KL_logps - reference_KL_logps).mean().detach()
            kl = self.accelerator.gather_for_metrics(kl).mean().clamp(min=0)
        else:
            kl = torch.zeros(1).to(policy_chosen_logps.device)

        # Chosen losses
        if policy_chosen_logps.shape[0] != 0 or reference_chosen_logps.shape[0] != 0:
            chosen_logratios = policy_chosen_logps - reference_chosen_logps

            if self.loss_type == "kto":
                # Eqn (7) of the KTO paper (https://huggingface.co/papers/2402.01306)
                chosen_losses = 1 - F.sigmoid(self.beta * (chosen_logratios - kl))
            elif self.loss_type == "apo_zero_unpaired":
                # Unpaired variant of Eqn (7) of the APO paper (https://huggingface.co/papers/2408.06266)
                # Use this loss when you believe the chosen outputs are better than your model's default output
                chosen_losses = 1 - F.sigmoid(self.beta * chosen_logratios)

            chosen_rewards = self.beta * chosen_logratios.detach()

        else:
            # lists can't be empty -- if they are, then accelerate.gather will hang
            chosen_losses = torch.Tensor([]).to(self.accelerator.device)
            chosen_rewards = torch.Tensor([]).to(self.accelerator.device)

        # Rejected losses
        if policy_rejected_logps.shape[0] != 0 or reference_rejected_logps.shape[0] != 0:
            rejected_logratios = policy_rejected_logps - reference_rejected_logps

            if self.loss_type == "kto":
                rejected_losses = 1 - F.sigmoid(self.beta * (kl - rejected_logratios))
            elif self.loss_type == "apo_zero_unpaired":
                rejected_losses = F.sigmoid(self.beta * rejected_logratios)

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

    def _compute_kl_logps(self, model, batch):
        """Compute KL log probabilities for a given batch."""
        KL_logps = None
        if self.calculate_KL:
            KL_model_kwargs = {
                "input_ids": batch["KL_completion_input_ids"],
                "attention_mask": batch["KL_completion_attention_mask"],
            }

            with torch.no_grad():
                KL_logits = model(**KL_model_kwargs).logits

            KL_logps = self.get_batch_logps(
                KL_logits,
                batch["KL_completion_labels"],
                average_log_prob=False,
            )
        return KL_logps

    def _compute_loss_liger(self, model, batch):
        """
        Compute the KTO loss using the Liger-Kernel's LigerFusedLinearKTOLoss.

        Args:
            model:
                The policy model used for generating log probabilities and outputs. It could be an encoder-decoder
                model or a regular language model.
            batch: A dictionary containing the input data and labels for the batch.

        Returns:
            A dictionary containing the following keys:
                - "loss": The computed KTO loss for the batch.
                - "chosen_logits_sum": Sum of the logits for the chosen responses from the policy model.
                - "rejected_logits_sum": Sum of the logits for the rejected responses from the policy model.
                - "chosen_logps": Log probabilities of the chosen responses from the policy model.
                - "rejected_logps": Log probabilities of the rejected responses from the policy model.
                - "chosen_rewards": Rewards for the chosen responses.
                - "rejected_rewards": Rewards for the rejected responses.
                - "kl": The KL divergence between the policy and reference models (detached).

            If auxiliary loss is enabled, the dictionary will also include:
                - "aux_loss": The auxiliary loss from the model outputs.
        """
        policy_KL_logps = self._compute_kl_logps(model, batch)
        reference_KL_logps = self._compute_kl_logps(self.ref_model, batch)
        if self.calculate_KL:
            kl = (policy_KL_logps - reference_KL_logps).mean().detach()
            kl = self.accelerator.gather_for_metrics(kl).mean().clamp(min=0)
        else:
            kl = torch.zeros(1).to(self.accelerator.device)

        model_kwargs = {}
        if self.aux_loss_enabled:
            model_kwargs["output_router_logits"] = True

        # skip the lm head and get the last hidden state
        base_model = model.get_decoder()
        outputs = base_model(
            batch["completion_input_ids"],
            attention_mask=batch["completion_attention_mask"],
            use_cache=False,
            **model_kwargs,
        )

        # reference model
        ref_base_model = self.ref_model.get_decoder()
        ref_outputs = ref_base_model(
            batch["completion_input_ids"],
            attention_mask=batch["completion_attention_mask"],
            use_cache=False,
            **model_kwargs,
        )
        lm_head = model.get_output_embeddings()
        ref_lm_head = self.ref_model.get_output_embeddings()

        (
            loss,
            (
                chosen_logps_sum,
                rejected_logps_sum,
                chosen_logits_sum,
                rejected_logits_sum,
                chosen_rewards_sum,
                rejected_rewards_sum,
            ),
        ) = self.kto_loss_fn(
            _input=outputs.last_hidden_state[:, :-1],
            lin_weight=lm_head.weight,
            target=batch["completion_labels"][:, 1:],
            bias=lm_head.bias if hasattr(lm_head, "bias") else None,
            preference_labels=torch.tensor(batch["label"], dtype=torch.bool).to(self.accelerator.device),
            ref_input=ref_outputs.last_hidden_state[:, :-1],
            ref_weight=ref_lm_head.weight,
            ref_bias=ref_lm_head.bias if hasattr(lm_head, "bias") else None,
            kl=kl,
        )

        output = {
            "loss": loss,
            "chosen_logits_sum": chosen_logits_sum,
            "rejected_logits_sum": rejected_logits_sum,
            "chosen_logps_sum": chosen_logps_sum,
            "rejected_logps_sum": rejected_logps_sum,
            "chosen_rewards_sum": chosen_rewards_sum,
            "rejected_rewards_sum": rejected_rewards_sum,
            "kl": kl,
        }
        if self.aux_loss_enabled:
            output["aux_loss"] = outputs.aux_loss

        return output

    def get_batch_loss_metrics(
        self,
        model,
        batch: dict[str, list | torch.LongTensor],
    ):
        """Compute the KTO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}
        batch = {k: (v.to(self.accelerator.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

        labels = torch.tensor(batch["label"])
        num_chosen = labels.sum().to(self.accelerator.device)
        num_rejected = (len(labels) - num_chosen).to(self.accelerator.device)

        if self.args.use_liger_kernel:
            model_output = self._compute_loss_liger(model, batch)
            losses = model_output["loss"]
            policy_chosen_logits = model_output["chosen_logits_sum"]
            policy_rejected_logits = model_output["rejected_logits_sum"]
            policy_chosen_logps = model_output["chosen_logps_sum"]
            policy_rejected_logps = model_output["rejected_logps_sum"]
            chosen_rewards = model_output["chosen_rewards_sum"]
            rejected_rewards = model_output["rejected_rewards_sum"]
            kl = model_output["kl"]
            if self.aux_loss_enabled:
                aux_loss = model_output["aux_loss"]
        else:
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
                # Convert Python lists to tensor indices for efficient CUDA operations
                device = batch["reference_logps"].device
                labels = torch.as_tensor(batch["label"], dtype=torch.bool, device=device)
                chosen_idx = torch.nonzero(labels, as_tuple=False).view(-1)
                rejected_idx = torch.nonzero(~labels, as_tuple=False).view(-1)

                # Use index_select for efficient CUDA operations
                reference_chosen_logps = batch["reference_logps"].index_select(0, chosen_idx)
                reference_rejected_logps = batch["reference_logps"].index_select(0, rejected_idx)
                if self.calculate_KL:
                    reference_KL_logps = batch["reference_KL_logps"]
                else:
                    reference_KL_logps = None
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

            losses, chosen_rewards, rejected_rewards, kl = self.kto_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                policy_KL_logps,
                reference_chosen_logps,
                reference_rejected_logps,
                reference_KL_logps,
            )

        metrics["kl"] = kl.item()

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
        model: PreTrainedModel | nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs=False,
        num_items_in_batch=None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
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

    def _get_train_sampler(self, dataset: Dataset | None = None) -> torch.utils.data.Sampler | None:
        if dataset is None:
            dataset = self.train_dataset
        if dataset is None or not has_length(dataset):
            return None
        return SequentialSampler(dataset)

    def prediction_step(
        self,
        model: PreTrainedModel | nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ):
        if ignore_keys is None:
            if hasattr(model, "config"):
                ignore_keys = getattr(model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        with torch.no_grad():
            loss, metrics = self.get_batch_loss_metrics(model, inputs)

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

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        """
        Log `logs` on the various objects watching training, including stored metrics.

        Args:
            logs (`dict[str, float]`):
                The values to log.
            start_time (`float`, *optional*):
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
