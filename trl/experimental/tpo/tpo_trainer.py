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

import json
import textwrap
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import transformers
from accelerate import PartialState
from accelerate.logging import get_logger
from accelerate.utils import is_peft_model
from datasets import Dataset, IterableDataset
from packaging.version import Version
from transformers import (
    AutoProcessor,
    DataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from transformers.data.data_collator import DataCollatorMixin
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction
from transformers.utils import is_peft_available

from ...data_utils import extract_prompt, is_conversational
from ...trainer.base_trainer import _BaseTrainer
from ...trainer.utils import (
    create_model_from_path,
    disable_dropout_in_model,
    entropy_from_logits,
    get_config_model_id,
    pad,
    selective_log_softmax,
)
from .tpo_config import TPOConfig


if is_peft_available():
    from peft import PeftConfig, PeftModel, get_peft_model


logger = get_logger(__name__)


def _extract_triple_prompt(example: dict) -> dict:
    """Extract the shared prompt from `chosen`/`rejected` and also strip it from `reference`.

    Wraps [`~trl.data_utils.extract_prompt`] — which only rewrites `chosen` and `rejected` — and additionally strips
    the extracted prompt prefix from the `reference` (gold) completion. This is specific to TPO and assumes that the
    `reference` completion shares the same implicit prompt prefix as `chosen` and `rejected`. If it does not, a
    `ValueError` is raised asking the caller to provide an explicit `prompt` column.
    """
    extracted = extract_prompt(example)
    prompt = extracted["prompt"]
    reference = example["reference"]
    if reference[: len(prompt)] != prompt:
        raise ValueError(
            "The `reference` completion does not start with the implicit prompt extracted from `chosen`/`rejected`. "
            "Either provide an explicit `prompt` column, or make sure the `reference` completion shares the same "
            "prompt prefix as the `chosen` and `rejected` completions."
        )
    extracted["reference"] = reference[len(prompt) :]
    return extracted


@dataclass
class DataCollatorForTriplePreference(DataCollatorMixin):
    """
    Data collator used for triple-preference data. Inputs are dynamically padded to the maximum length of a batch.

    This collator expects each example in the input list to be a dictionary containing the keys `"prompt_ids"`,
    `"chosen_ids"` and `"rejected_ids"`. When `include_reference=True` (the default) each example must additionally
    contain `"reference_ids"`. The collator returns a dictionary containing the following keys:
    - `"input_ids"`: Tensor of input IDs, padded to the maximum length of the batch. When
        `include_reference=True`, the first third of the batch corresponds to the `"chosen_ids"`, the second third to
        the `"rejected_ids"` and the last third to the `"reference_ids"`. When `include_reference=False`, the first
        half corresponds to the `"chosen_ids"` and the second half to the `"rejected_ids"` (matching the layout of
        [`~trl.trainer.dpo_trainer.DataCollatorForPreference`]).
    - `"attention_mask"`: Tensor of attention mask, padded to the maximum length of the batch.
    - `"completion_mask"`: Tensor indicating the positions of the completion tokens, padded to the maximum length of
        the batch.

    Args:
        pad_token_id (`int`):
            Token ID to use for padding.
        max_length (`int`, *optional*):
            Maximum length of the sequences after concatenation. Sequences longer than `max_length` are truncated
            before padding, which avoids allocating oversized tensors for batches containing very long sequences.
        truncation_mode (`str`, *optional*, defaults to `"keep_start"`):
            Truncation mode when a concatenated sequence exceeds `max_length`. Possible values are `"keep_end"` and
            `"keep_start"`.
        pad_to_multiple_of (`int`, *optional*):
            If set, the sequences will be padded to a multiple of this value.
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            Type of Tensor to return. Only `"pt"` is currently supported.
        include_reference (`bool`, *optional*, defaults to `True`):
            Whether to include the `"reference_ids"` branch in the collated batch. When `False`, the collator emits
            only the chosen/rejected halves and skips the gold-response sequences entirely, which matches the behavior
            expected when `tpo_alpha=0.0` (no NLL term).

    Examples:
    ```python
    >>> from trl.experimental.tpo.tpo_trainer import DataCollatorForTriplePreference

    >>> collator = DataCollatorForTriplePreference(pad_token_id=0)
    >>> examples = [
    ...     {"prompt_ids": [1, 2, 3], "chosen_ids": [4, 5], "rejected_ids": [6], "reference_ids": [7, 8]},
    ...     {"prompt_ids": [9, 10], "chosen_ids": [11], "rejected_ids": [12, 13], "reference_ids": [14]},
    ... ]
    >>> collator(examples)
    {'input_ids': tensor([[ 1,  2,  3,  4,  5],
                          [ 9, 10, 11,  0,  0],
                          [ 1,  2,  3,  6,  0],
                          [ 9, 10, 12, 13,  0],
                          [ 1,  2,  3,  7,  8],
                          [ 9, 10, 14,  0,  0]]),
     'attention_mask': tensor([[1, 1, 1, 1, 1],
                               [1, 1, 1, 0, 0],
                               [1, 1, 1, 1, 0],
                               [1, 1, 1, 1, 0],
                               [1, 1, 1, 1, 1],
                               [1, 1, 1, 0, 0]]),
     'completion_mask': tensor([[0, 0, 0, 1, 1],
                                [0, 0, 1, 0, 0],
                                [0, 0, 0, 1, 0],
                                [0, 0, 1, 1, 0],
                                [0, 0, 0, 1, 1],
                                [0, 0, 1, 0, 0]])}
    ```
    """

    pad_token_id: int
    max_length: int | None = None
    truncation_mode: str = "keep_start"
    pad_to_multiple_of: int | None = None
    return_tensors: str = "pt"
    include_reference: bool = True

    def torch_call(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        prompt_chosen_ids = [example["prompt_ids"] + example["chosen_ids"] for example in examples]
        prompt_rejected_ids = [example["prompt_ids"] + example["rejected_ids"] for example in examples]
        chosen_mask = [[0] * len(example["prompt_ids"]) + [1] * len(example["chosen_ids"]) for example in examples]
        rejected_mask = [[0] * len(example["prompt_ids"]) + [1] * len(example["rejected_ids"]) for example in examples]
        if self.include_reference:
            prompt_reference_ids = [example["prompt_ids"] + example["reference_ids"] for example in examples]
            reference_mask = [
                [0] * len(example["prompt_ids"]) + [1] * len(example["reference_ids"]) for example in examples
            ]

        if self.max_length is not None:
            if self.truncation_mode == "keep_start":
                sl = slice(None, self.max_length)
            elif self.truncation_mode == "keep_end":
                sl = slice(-self.max_length, None)
            else:
                raise ValueError(
                    f"Unsupported truncation mode: {self.truncation_mode}, expected 'keep_start' or 'keep_end'"
                )
            prompt_chosen_ids = [ids[sl] for ids in prompt_chosen_ids]
            prompt_rejected_ids = [ids[sl] for ids in prompt_rejected_ids]
            chosen_mask = [m[sl] for m in chosen_mask]
            rejected_mask = [m[sl] for m in rejected_mask]
            if self.include_reference:
                prompt_reference_ids = [ids[sl] for ids in prompt_reference_ids]
                reference_mask = [m[sl] for m in reference_mask]

        chosen_attention_mask = [[1] * len(ids) for ids in prompt_chosen_ids]
        rejected_attention_mask = [[1] * len(ids) for ids in prompt_rejected_ids]
        input_ids = prompt_chosen_ids + prompt_rejected_ids
        attention_mask = chosen_attention_mask + rejected_attention_mask
        completion_mask = chosen_mask + rejected_mask
        if self.include_reference:
            reference_attention_mask = [[1] * len(ids) for ids in prompt_reference_ids]
            input_ids = input_ids + prompt_reference_ids
            attention_mask = attention_mask + reference_attention_mask
            completion_mask = completion_mask + reference_mask

        # Convert to tensor
        input_ids = [torch.tensor(ids) for ids in input_ids]
        attention_mask = [torch.tensor(m, dtype=torch.long) for m in attention_mask]
        completion_mask = [torch.tensor(m, dtype=torch.long) for m in completion_mask]

        # Pad
        output = {}
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
        output["completion_mask"] = pad(
            completion_mask,
            padding_value=0,
            padding_side="right",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        return output


class TPOTrainer(_BaseTrainer):
    """
    Trainer for Triple Preference Optimization (TPO) method. This algorithm was initially proposed in the paper [Triple
    Preference Optimization: Achieving Better Alignment using a Single Step
    Optimization](https://huggingface.co/papers/2405.16681). This class is a wrapper around the
    [`~transformers.Trainer`] class and inherits all of its attributes and methods.

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
        args ([`experimental.tpo.TPOConfig`], *optional*):
            Configuration for this trainer. If `None`, a default configuration is used.
        data_collator ([`~transformers.DataCollator`], *optional*):
            Function to use to form a batch from a list of elements of the processed `train_dataset` or `eval_dataset`.
            Will default to [`~trl.experimental.tpo.tpo_trainer.DataCollatorForTriplePreference`]. Custom collators
            must truncate sequences before padding; the trainer does not apply post-collation truncation.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. TPO requires a *triple-preference* dataset: each sample must contain a
            `"chosen"`, a `"rejected"` and a `"reference"` (gold) completion. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Dataset | IterableDataset]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*):
            Processing class used to process the data. If `None`, the processing class is loaded from the model's name
            with [`~transformers.AutoProcessor.from_pretrained`]. A padding token, `tokenizer.pad_token`, must be set.
            If the processing class has not set a padding token, `tokenizer.eos_token` will be used as the default.
        compute_metrics (`Callable[[EvalPrediction], dict]`, *optional*):
            The function that will be used to compute metrics at evaluation. Must take a
            [`~transformers.EvalPrediction`] and return a dictionary string to metric values.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks detailed
            in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of `AdamW` on your
            model and a scheduler given by [`~transformers.get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    _tag_names = ["trl", "tpo"]
    _name = "TPO"
    _paper = {
        "title": "Triple Preference Optimization: Achieving Better Alignment using a Single Step Optimization",
        "id": "2405.16681",
        # docstyle-ignore
        "citation": textwrap.dedent("""\
            @misc{saeidi2025triplepreferenceoptimizationachieving,
                title        = {{Triple Preference Optimization: Achieving Better Alignment using a Single Step Optimization}},
                author       = {Amir Saeidi and Shivanshu Verma and Aswin RRV and Kashif Rasul and Chitta Baral},
                year         = 2025,
                eprint       = {2405.16681},
                archivePrefix= {arXiv},
                primaryClass = {cs.CL},
                url          = {https://arxiv.org/abs/2405.16681},
            }"""),
    }

    def __init__(
        self,
        model: "str | PreTrainedModel | PeftModel",
        args: TPOConfig | None = None,
        data_collator: DataCollator | None = None,
        train_dataset: Dataset | IterableDataset | None = None,
        eval_dataset: Dataset | IterableDataset | dict[str, Dataset | IterableDataset] | None = None,
        processing_class: PreTrainedTokenizerBase | None = None,
        compute_metrics: Callable[[EvalPrediction], dict] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None] = (None, None),
        peft_config: "PeftConfig | None" = None,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else get_config_model_id(model.config)
            model_name = model_name.split("/")[-1]
            args = TPOConfig(f"{model_name}-TPO")

        if train_dataset is None:
            raise ValueError("`train_dataset` is required")
        elif isinstance(train_dataset, IterableDataset):
            # IterableDataset requires dispatch_batches=False because Accelerate's dispatch mode may try to concatenate
            # batches from multiple processes, leading to mismatch errors.
            if args.accelerator_config.dispatch_batches is True:
                logger.warning(
                    "You are using an `IterableDataset` for training with `dispatch_batches=True`. `dispatch_batches` "
                    "is forced to `False` when using an `IterableDataset`. To remove this warning, unset "
                    "`dispatch_batches` in `TPOConfig` or set it to `False`."
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
                    "You passed `model_init_kwargs` to the `TPOConfig`, but your model is already instantiated. "
                    "The `model_init_kwargs` will be ignored."
                )

        # Processing class
        if processing_class is None:
            processing_class = AutoProcessor.from_pretrained(get_config_model_id(model.config))
        if not isinstance(processing_class, PreTrainedTokenizerBase):
            raise TypeError(
                "The `processing_class` must be a `PreTrainedTokenizerBase`. `TPOTrainer` does not currently "
                "support vision-language models."
            )
        tokenizer = processing_class

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if is_peft_available() and is_peft_model(model) and peft_config is not None:
            raise ValueError(
                "You passed a `PeftModel` instance together with a `peft_config` to the trainer. Please first merge "
                "and unload the existing adapter, save the resulting base model, and then pass that base model along "
                "with the new `peft_config` to the trainer."
            )

        # Create PEFT model
        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        # When using gradient checkpointing with PEFT, we need to enable input gradients. transformers.Trainer normally
        # handles this, but a bug currently prevents it; see https://github.com/huggingface/transformers/issues/42489
        if is_peft_available() and isinstance(model, PeftModel) and args.gradient_checkpointing:
            model.enable_input_require_grads()

        # Data collator. When `tpo_alpha=0.0`, the NLL term on the gold response is disabled, so we can drop the
        # reference branch from the batch entirely — this spares the model from computing logits for a third of
        # each step.
        if data_collator is None:
            data_collator = DataCollatorForTriplePreference(
                pad_token_id=tokenizer.pad_token_id,
                max_length=args.max_length,
                truncation_mode=args.truncation_mode,
                pad_to_multiple_of=args.pad_to_multiple_of,
                include_reference=args.tpo_alpha != 0.0,
            )

        # Training arguments
        self.beta = args.beta
        self.loss_type = args.loss_type
        self.label_smoothing = args.label_smoothing
        self.tpo_alpha = args.tpo_alpha
        self.tpo_l_gamma = args.tpo_l_gamma
        if self.loss_type in ["hinge", "ipo"] and self.label_smoothing > 0:
            logger.warning(
                f"You are using the {self.loss_type} loss type that does not support label smoothing. The "
                "`label_smoothing` parameter will be ignored. Set `label_smoothing` to `0.0` to remove this warning."
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

        # Disable dropout in the model
        if args.disable_dropout:
            disable_dropout_in_model(model)

        # Initialize the metrics
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._total_train_tokens = 0

        # Add tags to the model
        self.model.add_model_tags(self._tag_names)

    def _tokenize(
        self,
        processing_class: PreTrainedTokenizerBase,
        input: str | list,
        **kwargs,
    ) -> dict[str, list]:
        """Tokenize a single example for dataset preprocessing.

        Dispatches to `apply_chat_template` for conversational input (list of message dicts) and to `__call__` for
        non-conversational input (str).

        Args:
            processing_class ([`~transformers.PreTrainedTokenizerBase`]):
                The tokenizer to use.
            input (`str` or `list`):
                A string for non-conversational input, or a list of message dicts for conversational input.
            **kwargs:
                Forwarded to `apply_chat_template` (e.g. `add_generation_prompt`).

        Returns:
            `dict` with at least an `"input_ids"` key mapping to a flat `list[int]`.
        """
        if isinstance(input, list):  # conversational: list of message dicts
            return processing_class.apply_chat_template(input, tokenize=True, return_dict=True, **kwargs)
        # non-conversational: plain text string
        return processing_class(text=input)

    def _prepare_dataset(
        self,
        dataset: Dataset | IterableDataset,
        processing_class: PreTrainedTokenizerBase,
        args: TPOConfig,
        dataset_name: str,
    ) -> Dataset | IterableDataset:
        # Validate that the triple-preference columns are present
        first_example = next(iter(dataset))
        if "chosen" not in first_example or "rejected" not in first_example:
            raise ValueError(
                "TPO requires a triple-preference dataset with `chosen`, `rejected` and `reference` columns, but the "
                f"dataset is missing `chosen` or `rejected`. Got columns: {list(first_example.keys())}."
            )
        if "reference" not in first_example:
            raise ValueError(
                "TPO requires a triple-preference dataset with `chosen`, `rejected` and `reference` columns, but the "
                f"dataset is missing the `reference` (gold) column. Got columns: {list(first_example.keys())}."
            )

        # Build the kwargs for the `map` function
        map_kwargs = {}
        if isinstance(dataset, Dataset):  # IterableDataset does not support num_proc
            map_kwargs["num_proc"] = args.dataset_num_proc

        with PartialState().main_process_first():
            # Extract the prompt if needed. Unlike DPO, we must also strip the extracted prompt from the reference
            # column (see `_extract_triple_prompt`), which assumes the reference shares the same implicit prompt.
            first_example = next(iter(dataset))
            if "prompt" not in first_example:
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Extracting prompt from {dataset_name} dataset"
                dataset = dataset.map(_extract_triple_prompt, **map_kwargs)

            # Add EOS to completions for non-conversational data
            first_example = next(iter(dataset))
            if not is_conversational(first_example):
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Adding EOS to {dataset_name} dataset"

                def add_eos(example, eos_token):
                    if not example["chosen"].endswith(eos_token):
                        example["chosen"] = example["chosen"] + eos_token
                    if not example["rejected"].endswith(eos_token):
                        example["rejected"] = example["rejected"] + eos_token
                    if not example["reference"].endswith(eos_token):
                        example["reference"] = example["reference"] + eos_token
                    return example

                dataset = dataset.map(add_eos, fn_kwargs={"eos_token": processing_class.eos_token}, **map_kwargs)

            # Tokenize the dataset
            if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                map_kwargs["desc"] = f"Tokenizing {dataset_name} dataset"

            def tokenize_fn(example, processing_class):
                tools = example.get("tools")
                tools = json.loads(tools) if isinstance(tools, str) else tools
                output = {}
                if is_conversational(example):
                    prompt_ids = self._tokenize(
                        processing_class,
                        example["prompt"],
                        tools=tools,
                        add_generation_prompt=True,
                        **example.get("chat_template_kwargs", {}),
                    )["input_ids"]
                    prompt_chosen_ids = self._tokenize(
                        processing_class,
                        example["prompt"] + example["chosen"],
                        tools=tools,
                        **example.get("chat_template_kwargs", {}),
                    )["input_ids"]
                    prompt_rejected_ids = self._tokenize(
                        processing_class,
                        example["prompt"] + example["rejected"],
                        tools=tools,
                        **example.get("chat_template_kwargs", {}),
                    )["input_ids"]
                    prompt_reference_ids = self._tokenize(
                        processing_class,
                        example["prompt"] + example["reference"],
                        tools=tools,
                        **example.get("chat_template_kwargs", {}),
                    )["input_ids"]
                else:
                    prompt_ids = self._tokenize(processing_class, example["prompt"])["input_ids"]
                    prompt_chosen_ids = self._tokenize(processing_class, example["prompt"] + example["chosen"])[
                        "input_ids"
                    ]
                    prompt_rejected_ids = self._tokenize(processing_class, example["prompt"] + example["rejected"])[
                        "input_ids"
                    ]
                    prompt_reference_ids = self._tokenize(processing_class, example["prompt"] + example["reference"])[
                        "input_ids"
                    ]

                # Check if the tokenized prompt starts with the tokenized prompt+completion
                if not prompt_chosen_ids[: len(prompt_ids)] == prompt_ids:
                    logger.warning(
                        "Mismatch between tokenized prompt and the start of tokenized prompt+chosen. "
                        "This may be due to unexpected tokenizer behavior, whitespace issues, or special "
                        "token handling. Verify that the tokenizer is processing text consistently."
                    )
                if not prompt_rejected_ids[: len(prompt_ids)] == prompt_ids:
                    logger.warning(
                        "Mismatch between tokenized prompt and the start of tokenized prompt+rejected. "
                        "This may be due to unexpected tokenizer behavior, whitespace issues, or special "
                        "token handling. Verify that the tokenizer is processing text consistently."
                    )
                if not prompt_reference_ids[: len(prompt_ids)] == prompt_ids:
                    logger.warning(
                        "Mismatch between tokenized prompt and the start of tokenized prompt+reference. "
                        "This may be due to unexpected tokenizer behavior, whitespace issues, or special "
                        "token handling. Verify that the tokenizer is processing text consistently."
                    )

                output["prompt_ids"] = prompt_ids
                output["chosen_ids"] = prompt_chosen_ids[len(prompt_ids) :]
                output["rejected_ids"] = prompt_rejected_ids[len(prompt_ids) :]
                output["reference_ids"] = prompt_reference_ids[len(prompt_ids) :]
                return output

            dataset = dataset.map(tokenize_fn, fn_kwargs={"processing_class": processing_class}, **map_kwargs)

        return dataset

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs (usually, "input_ids"
        # and "attention_mask").
        if self._signature_columns is None:
            self._signature_columns = ["prompt_ids", "chosen_ids", "rejected_ids", "reference_ids"]

    def _compute_loss(self, model, inputs, return_outputs):
        mode = "train" if self.model.training else "eval"

        # When `tpo_alpha=0.0` the NLL term is disabled and the collator drops the reference branch, so the batch
        # is laid out as `[chosen, rejected]` (n_branches=2). Otherwise it is `[chosen, rejected, reference]`
        # (n_branches=3).
        n_branches = 3 if self.tpo_alpha != 0.0 else 2

        _non_model_keys = {"completion_mask"}
        model_kwargs = {k: v for k, v in inputs.items() if k not in _non_model_keys}
        model_kwargs["use_cache"] = False
        outputs = model(**model_kwargs)

        input_ids = inputs["input_ids"]
        completion_mask = inputs["completion_mask"]
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        shift_completion_mask = completion_mask[..., 1:].contiguous()
        per_token_logps = selective_log_softmax(shift_logits, shift_labels)
        per_token_logps[shift_completion_mask == 0] = 0.0  # mask out non-completion tokens

        # Length-normalized for IPO and TPO-L (matches the SimPO-style implicit reward used by the TPO paper);
        # summed otherwise.
        if self.loss_type in ("ipo", "tpo-l"):
            completion_lengths = shift_completion_mask.sum(dim=1).clamp(min=1)
            logps = per_token_logps.sum(dim=1) / completion_lengths
        else:
            logps = per_token_logps.sum(dim=1)
        logps_chunks = logps.chunk(n_branches, dim=0)
        chosen_logps, rejected_logps = logps_chunks[0], logps_chunks[1]

        # Contrastive loss between chosen and rejected. Unlike DPO, TPO does not subtract reference-model log-probs:
        # the "reference" in TPO is a gold response used in the NLL term below, not a separate reference policy.
        delta_score = chosen_logps - rejected_logps

        if self.loss_type == "sigmoid":
            per_sequence_loss = (
                -F.logsigmoid(self.beta * delta_score) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * delta_score) * self.label_smoothing
            )

        elif self.loss_type == "hinge":
            per_sequence_loss = torch.relu(1 - self.beta * delta_score)

        elif self.loss_type == "ipo":
            # (Eq. 17) of the IPO paper where beta is the regularization parameter for the IPO loss, denoted by τ.
            per_sequence_loss = (delta_score - 1 / (2 * self.beta)) ** 2

        elif self.loss_type == "tpo-l":
            # Length-normalized TPO-L variant: subtract a target reward margin γ/β before the sigmoid.
            gamma_logratios = self.tpo_l_gamma / self.beta
            shifted_delta = delta_score - gamma_logratios
            per_sequence_loss = (
                -F.logsigmoid(self.beta * shifted_delta) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * shifted_delta) * self.label_smoothing
            )

        else:
            raise ValueError(
                f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge', 'ipo', 'tpo-l']"
            )

        loss = per_sequence_loss.mean()

        # NLL loss on the gold (`reference`) response. Mirrors the `"sft"` loss branch of `DPOTrainer._compute_loss`:
        # we restrict the cross-entropy to the completion tokens of the reference sequence and let `F.cross_entropy`
        # average over them. The NLL contribution is folded into the main `loss` (matching DPO/SFT convention: the
        # individual NLL term is not logged separately).
        if n_branches == 3:
            _, _, ref_logits = shift_logits.chunk(3, dim=0)
            _, _, ref_labels = shift_labels.chunk(3, dim=0)
            _, _, ref_mask = shift_completion_mask.chunk(3, dim=0)
            ref_mask = ref_mask.bool()
            nll_loss = F.cross_entropy(ref_logits[ref_mask], ref_labels[ref_mask])
            loss = loss + self.tpo_alpha * nll_loss

        # Log the metrics
        # Entropy
        per_token_entropy = entropy_from_logits(shift_logits.detach())
        entropy = per_token_entropy[shift_completion_mask.bool()].mean()
        entropy = self.accelerator.gather_for_metrics(entropy).mean().item()
        self._metrics[mode]["entropy"].append(entropy)

        # Number of tokens
        if mode == "train":
            num_tokens_in_batch = self.accelerator.gather_for_metrics(inputs["attention_mask"].sum()).sum().item()
            self._total_train_tokens += num_tokens_in_batch
        self._metrics[mode]["num_tokens"] = [self._total_train_tokens]

        # Average logits for chosen and rejected completions
        logits_chunks = shift_logits.detach().chunk(n_branches, dim=0)
        mask_chunks = shift_completion_mask.chunk(n_branches, dim=0)
        labels_chunks = shift_labels.chunk(n_branches, dim=0)
        chosen_logits, rejected_logits = logits_chunks[0], logits_chunks[1]
        chosen_mask, rejected_mask = mask_chunks[0], mask_chunks[1]
        chosen_labels = labels_chunks[0]
        total_chosen_logits = chosen_logits[chosen_mask.bool()].mean(-1).sum()
        total_chosen_tokens = chosen_mask.sum()
        total_rejected_logits = rejected_logits[rejected_mask.bool()].mean(-1).sum()
        total_rejected_tokens = rejected_mask.sum()
        total_chosen_logits = self.accelerator.gather_for_metrics(total_chosen_logits).sum().item()
        total_chosen_tokens = self.accelerator.gather_for_metrics(total_chosen_tokens).sum().item()
        total_rejected_logits = self.accelerator.gather_for_metrics(total_rejected_logits).sum().item()
        total_rejected_tokens = self.accelerator.gather_for_metrics(total_rejected_tokens).sum().item()
        avg_chosen_logits = total_chosen_logits / total_chosen_tokens if total_chosen_tokens > 0 else 0.0
        avg_rejected_logits = total_rejected_logits / total_rejected_tokens if total_rejected_tokens > 0 else 0.0
        self._metrics[mode]["logits/chosen"].append(avg_chosen_logits)
        self._metrics[mode]["logits/rejected"].append(avg_rejected_logits)

        # Token accuracy for the chosen completions
        predictions = chosen_logits.argmax(dim=-1)
        chosen_bool_mask = chosen_mask.bool()
        correct_predictions = (predictions == chosen_labels) & chosen_bool_mask
        total_tokens = chosen_bool_mask.sum()
        correct_tokens = correct_predictions.sum()
        correct_tokens = self.accelerator.gather_for_metrics(correct_tokens)
        total_tokens = self.accelerator.gather_for_metrics(total_tokens)
        total_sum = total_tokens.sum()
        accuracy = (correct_tokens.sum() / total_sum).item() if total_sum > 0 else 0.0
        self._metrics[mode]["mean_token_accuracy"].append(accuracy)

        # Rewards for chosen and rejected completions (β · log π_θ as in the SimPO/TPO implicit reward)
        chosen_rewards = self.beta * chosen_logps.detach()
        rejected_rewards = self.beta * rejected_logps.detach()
        agg_chosen_rewards = self.accelerator.gather(chosen_rewards)
        agg_rejected_rewards = self.accelerator.gather(rejected_rewards)
        self._metrics[mode]["rewards/chosen"].append(agg_chosen_rewards.mean().item())
        self._metrics[mode]["rewards/rejected"].append(agg_rejected_rewards.mean().item())

        # Reward accuracy
        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        agg_reward_accuracies = self.accelerator.gather(reward_accuracies)
        self._metrics[mode]["rewards/accuracies"].append(agg_reward_accuracies.mean().item())

        # Reward margins
        margins = chosen_rewards - rejected_rewards
        agg_margins = self.accelerator.gather(margins)
        self._metrics[mode]["rewards/margins"].append(agg_margins.mean().item())

        # Average log probabilities for chosen and rejected completions
        self._metrics[mode]["logps/chosen"].append(self.accelerator.gather(chosen_logps).mean().item())
        self._metrics[mode]["logps/rejected"].append(self.accelerator.gather(rejected_logps).mean().item())

        return (loss, outputs) if return_outputs else loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        return self._compute_loss(model, inputs, return_outputs)

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        mode = "train" if self.model.training else "eval"
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        super().log(logs, start_time)
        self._metrics[mode].clear()

    # During eval, Trainer calls prediction_step. If no labels are present in the inputs, it only runs forward and
    # returns logits. We override prediction_step to force compute_loss, because this trainer doesn't involve labels.
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys: list[str] | None = None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad(), self.compute_loss_context_manager():
            if prediction_loss_only:
                loss = self.compute_loss(model, inputs, return_outputs=False)
                logits, labels = None, None
            else:
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                logits, labels = outputs.logits, inputs["input_ids"]
        return loss, logits, labels

    # Ensure the model card is saved along with the checkpoint
    def _save_checkpoint(self, model, trial):
        if self.args.hub_model_id is None:
            model_name = Path(self.args.output_dir).name
        else:
            model_name = self.args.hub_model_id.split("/")[-1]
        self.create_model_card(model_name=model_name)
        super()._save_checkpoint(model, trial)
