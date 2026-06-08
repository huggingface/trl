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

import contextlib
import os
import textwrap
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import transformers
from accelerate import PartialState, logging
from accelerate.utils import is_peft_model, tqdm
from datasets import Dataset, IterableDataset, IterableDatasetDict, concatenate_datasets
from datasets.fingerprint import Hasher
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
    apply_chat_template,
    extract_prompt,
    is_conversational,
    prepare_multimodal_messages,
    unpair_preference_dataset,
)
from ...import_utils import is_liger_kernel_available
from ...models import get_act_offloading_ctx_manager
from ...models.utils import disable_gradient_checkpointing, prepare_deepspeed, prepare_fsdp
from ...trainer.base_trainer import _BaseTrainer
from ...trainer.utils import (
    create_model_from_path,
    disable_dropout_in_model,
    flush_left,
    get_config_model_id,
    hash_module,
    pad,
    selective_log_softmax,
    use_adapter,
)
from .kto_config import KTOConfig


if is_liger_kernel_available():
    from liger_kernel.chunked_loss import LigerFusedLinearKTOLoss

if is_peft_available():
    from peft import PeftConfig, PeftModel, get_peft_model


logger = logging.get_logger(__name__)

RUNNING_NAME = "running.pt"


def get_dataset_column_names(dataset: Dataset | IterableDataset) -> list[str]:
    return list(next(iter(dataset)).keys()) if dataset.column_names is None else dataset.column_names


def _get_kl_completion_ids(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
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

    Each example is expected to contain `"prompt_ids"`, `"completion_ids"` (and optionally `"KL_completion_ids"`) keys.
    The collator returns a dictionary with the following keys for each prefix (`"completion"` and, if present,
    `"KL_completion"`):
    - `"{prefix}_input_ids"`: full prompt + completion token IDs, padded to the batch maximum length.
    - `"{prefix}_attention_mask"`: attention mask, padded with 0s.
    - `"{prefix}_mask"`: binary mask where 1 marks completion tokens and 0 marks prompt or padding tokens.

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
            completion_mask_list = []
            for ex in examples:
                prompt_ids = ex["prompt_ids"]
                answer_ids = ex[ids_key]
                full_ids = prompt_ids + answer_ids
                completion_mask = [0] * len(prompt_ids) + [1] * len(answer_ids)
                if self.max_length is not None:
                    full_ids = full_ids[: self.max_length]
                    completion_mask = completion_mask[: self.max_length]
                full_ids_list.append(full_ids)
                completion_mask_list.append(completion_mask)

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
            batch[f"{prefix}_mask"] = pad(
                [torch.tensor(m, dtype=torch.int64) for m in completion_mask_list],
                padding_value=0,
                padding_side="right",
            )

        if "ref_logps" in examples[0]:
            batch["ref_logps"] = torch.tensor([ex["ref_logps"] for ex in examples])
        if "ref_KL_logps" in examples[0]:
            batch["ref_KL_logps"] = torch.tensor([ex["ref_KL_logps"] for ex in examples])
        batch["label"] = [ex["label"] for ex in examples]
        return batch


@dataclass
class DataCollatorForVisionUnpairedPreference(DataCollatorMixin):
    """
    Data collator for vision unpaired preference data. Performs tokenization and image processing on-the-fly.

    Unlike the text-only [`DataCollatorForUnpairedPreference`], this collator does not expect pre-tokenized inputs.
    Instead, it takes raw examples with `"prompt"`, `"completion"`, and `"images"` (or `"image"`) keys and processes
    them at collation time. When `calculate_kl` is `True`, the collator also produces KL sequences by cycling
    completions within the batch — the same mismatching strategy as the text-only path, but done here rather than as a
    dataset pre-processing step.

    Each input example should contain at least:
    - A `"prompt"` key with either a plain text string or a list of message dicts.
    - A `"completion"` key with a plain text string or a list of message dicts.
    - An `"images"` key holding a list of images, or an `"image"` key holding a single image.
    - A `"label"` key (`bool`) indicating whether the completion is desirable.

    The collator outputs:
    - `"completion_input_ids"`, `"completion_attention_mask"`, `"completion_mask"`: full prompt+completion sequence.
    - `"pixel_values"` and any additional processor outputs (e.g., `"image_grid_thw"`).
    - `"label"`: list of booleans.
    - When `calculate_kl=True`: `"KL_completion_input_ids"`, `"KL_completion_attention_mask"`, `"KL_completion_mask"`
      for the cycled KL sequences.

    Args:
        processor ([`~transformers.ProcessorMixin`]):
            The processor used to tokenize text and process images.
        max_length (`int`, *optional*):
            Maximum sequence length. Sequences longer than `max_length` are truncated.
        calculate_kl (`bool`, *optional*, defaults to `True`):
            Whether to produce KL sequences by cycling completions within the batch.
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            The tensor type to return. Only `"pt"` is supported.
    """

    processor: ProcessorMixin
    max_length: int | None = None
    calculate_kl: bool = True
    return_tensors: str = "pt"

    def torch_call(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        # Normalize "image" → "images"
        if "image" in examples[0]:
            for example in examples:
                example["images"] = [example.pop("image")]

        images = [example["images"] for example in examples]
        if all(img_list == [] for img_list in images):
            images = None

        if is_conversational(examples[0]):
            for example in examples:
                example["prompt"] = prepare_multimodal_messages(example["prompt"], images=example["images"])
                example["completion"] = prepare_multimodal_messages(example["completion"])
            examples = [apply_chat_template(example, self.processor) for example in examples]

        prompts = [example["prompt"] for example in examples]
        completions = [example["completion"] for example in examples]

        processed_prompts = self.processor(
            images=images,
            text=prompts,
            padding=True,
            padding_side="left",
            return_tensors=self.return_tensors,
            add_special_tokens=False,  # to avoid adding the BOS twice, see https://huggingface.co/blog/qgallouedec/gotchas-in-tokenizer-behavior#7-chat-template-and-tokenization-dont-compose-due-to-special-tokens
        )
        processed_completions = self.processor(
            text=completions,
            padding=True,
            padding_side="right",
            return_tensors=self.return_tensors,
            add_special_tokens=False,  # to avoid adding the BOS twice, see https://huggingface.co/blog/qgallouedec/gotchas-in-tokenizer-behavior#7-chat-template-and-tokenization-dont-compose-due-to-special-tokens
        )

        # Concatenate prompts and completions
        prompt_ids, prompt_mask = processed_prompts["input_ids"], processed_prompts["attention_mask"]
        completion_ids, completion_mask = processed_completions["input_ids"], processed_completions["attention_mask"]
        input_ids = torch.cat((prompt_ids, completion_ids), dim=1)
        attention_mask = torch.cat((prompt_mask, completion_mask), dim=1)
        completion_mask = torch.cat((torch.zeros_like(prompt_mask), completion_mask), dim=1)

        has_tti = "token_type_ids" in processed_prompts
        has_mm_tti = "mm_token_type_ids" in processed_prompts

        if has_tti:  # special case for Gemma
            prompt_token_type_ids = processed_prompts["token_type_ids"]
            completion_token_type_ids = processed_completions["token_type_ids"]
            token_type_ids = torch.cat((prompt_token_type_ids, completion_token_type_ids), dim=1)
        if has_mm_tti:  # special case for Qwen2.5-VL
            prompt_mm_token_type_ids = processed_prompts["mm_token_type_ids"]
            completion_mm_token_type_ids = processed_completions.get(
                "mm_token_type_ids", torch.zeros_like(completion_ids)
            )
            mm_token_type_ids = torch.cat((prompt_mm_token_type_ids, completion_mm_token_type_ids), dim=1)

        # Flush left to reduce padding
        if has_tti and has_mm_tti:
            attention_mask, input_ids, completion_mask, token_type_ids, mm_token_type_ids = flush_left(
                attention_mask, input_ids, completion_mask, token_type_ids, mm_token_type_ids
            )
        elif has_tti:
            attention_mask, input_ids, completion_mask, token_type_ids = flush_left(
                attention_mask, input_ids, completion_mask, token_type_ids
            )
        elif has_mm_tti:
            attention_mask, input_ids, completion_mask, mm_token_type_ids = flush_left(
                attention_mask, input_ids, completion_mask, mm_token_type_ids
            )
        else:
            attention_mask, input_ids, completion_mask = flush_left(attention_mask, input_ids, completion_mask)

        # Truncate if necessary
        if self.max_length is not None:
            input_ids = input_ids[:, : self.max_length]
            attention_mask = attention_mask[:, : self.max_length]
            completion_mask = completion_mask[:, : self.max_length]
            if has_tti:
                token_type_ids = token_type_ids[:, : self.max_length]
            if has_mm_tti:
                mm_token_type_ids = mm_token_type_ids[:, : self.max_length]

        # Build the output dictionary
        output = processed_prompts
        output.pop("input_ids", None)
        output.pop("attention_mask", None)
        output["completion_input_ids"] = input_ids
        output["completion_attention_mask"] = attention_mask
        output["completion_mask"] = completion_mask
        if has_tti:
            output["token_type_ids"] = token_type_ids
        if has_mm_tti:
            output["mm_token_type_ids"] = mm_token_type_ids

        if self.calculate_kl:
            # Cycle completions by +1 within the batch to create mismatched KL pairs — same strategy as
            # _get_kl_completion_ids in the text-only path, but done here to keep the VLM dataset fully raw.
            kl_completions = completions[-1:] + completions[:-1]
            processed_kl = self.processor(
                text=kl_completions,
                padding=True,
                padding_side="right",
                return_tensors=self.return_tensors,
                add_special_tokens=False,  # to avoid adding the BOS twice, see https://huggingface.co/blog/qgallouedec/gotchas-in-tokenizer-behavior#7-chat-template-and-tokenization-dont-compose-due-to-special-tokens
            )
            kl_ids = processed_kl["input_ids"]
            kl_mask = processed_kl["attention_mask"]

            kl_input_ids = torch.cat((prompt_ids, kl_ids), dim=1)
            kl_attention_mask = torch.cat((prompt_mask, kl_mask), dim=1)
            kl_completion_mask = torch.cat((torch.zeros_like(prompt_mask), kl_mask), dim=1)

            # Build KL token-type tensors using the original (pre-flush) prompt tensors
            if has_tti:
                kl_completion_token_type_ids = processed_kl.get("token_type_ids", torch.zeros_like(kl_ids))
                kl_token_type_ids = torch.cat((prompt_token_type_ids, kl_completion_token_type_ids), dim=1)
            if has_mm_tti:
                kl_completion_mm_token_type_ids = processed_kl.get("mm_token_type_ids", torch.zeros_like(kl_ids))
                kl_mm_token_type_ids = torch.cat((prompt_mm_token_type_ids, kl_completion_mm_token_type_ids), dim=1)

            if has_tti and has_mm_tti:
                kl_attention_mask, kl_input_ids, kl_completion_mask, kl_token_type_ids, kl_mm_token_type_ids = (
                    flush_left(
                        kl_attention_mask, kl_input_ids, kl_completion_mask, kl_token_type_ids, kl_mm_token_type_ids
                    )
                )
            elif has_tti:
                kl_attention_mask, kl_input_ids, kl_completion_mask, kl_token_type_ids = flush_left(
                    kl_attention_mask, kl_input_ids, kl_completion_mask, kl_token_type_ids
                )
            elif has_mm_tti:
                kl_attention_mask, kl_input_ids, kl_completion_mask, kl_mm_token_type_ids = flush_left(
                    kl_attention_mask, kl_input_ids, kl_completion_mask, kl_mm_token_type_ids
                )
            else:
                kl_attention_mask, kl_input_ids, kl_completion_mask = flush_left(
                    kl_attention_mask, kl_input_ids, kl_completion_mask
                )

            # Truncate if necessary
            if self.max_length is not None:
                kl_input_ids = kl_input_ids[:, : self.max_length]
                kl_attention_mask = kl_attention_mask[:, : self.max_length]
                kl_completion_mask = kl_completion_mask[:, : self.max_length]
                if has_tti:
                    kl_token_type_ids = kl_token_type_ids[:, : self.max_length]
                if has_mm_tti:
                    kl_mm_token_type_ids = kl_mm_token_type_ids[:, : self.max_length]

            output["KL_completion_input_ids"] = kl_input_ids
            output["KL_completion_attention_mask"] = kl_attention_mask
            output["KL_completion_mask"] = kl_completion_mask
            if has_tti:
                output["KL_completion_token_type_ids"] = kl_token_type_ids
            if has_mm_tti:
                output["KL_completion_mm_token_type_ids"] = kl_mm_token_type_ids

        output["label"] = [ex["label"] for ex in examples]
        return output


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
            self._is_vlm = True
        elif isinstance(processing_class, PreTrainedTokenizerBase):
            self._tokenizer = processing_class
            self._is_vlm = False
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

        # Vision dataset detection
        dataset_sample = next(iter(train_dataset))
        self._is_vision_dataset = "image" in dataset_sample or "images" in dataset_sample
        if self._is_vision_dataset and not self._is_vlm:
            raise ValueError(
                "The dataset appears to be vision-related (contains 'image' or 'images' keys), but the provided "
                "model does not seem to be a vision-language model. Please check your model and dataset."
            )
        if self._is_vision_dataset and args.precompute_ref_log_probs:
            raise ValueError(
                "`precompute_ref_log_probs=True` is not supported for vision datasets. For vision-language "
                "models, all data processing is performed on the fly rather than upfront. "
                "Set `precompute_ref_log_probs=False`."
            )
        if self._is_vision_dataset and ("chosen" in dataset_sample or "rejected" in dataset_sample):
            raise ValueError(
                "Vision datasets must be in unpaired format with `completion` and `label` columns. "
                "Paired format (`chosen`/`rejected`) is not supported for vision datasets because "
                "iterating over the full dataset to unpair it would be too expensive for large image "
                "collections. Unpair your dataset first: `dataset = unpair_preference_dataset(dataset)`."
            )

        # Data collator
        calculate_kl = args.loss_type not in ["apo_zero_unpaired"]
        if data_collator is None and not self._is_vision_dataset:
            data_collator = DataCollatorForUnpairedPreference(
                pad_token_id=self._tokenizer.pad_token_id,
                max_length=args.max_length,
            )
            self.use_dpo_data_collator = True
        elif data_collator is None and self._is_vision_dataset:
            data_collator = DataCollatorForVisionUnpairedPreference(
                processor=processing_class,
                max_length=args.max_length,
                calculate_kl=calculate_kl,
            )
            self.use_dpo_data_collator = True
        else:
            self.use_dpo_data_collator = False

        # Training arguments
        self.beta = args.beta
        self.precompute_ref_logps = args.precompute_ref_log_probs
        self.loss_type = args.loss_type
        self.desirable_weight = args.desirable_weight
        self.undesirable_weight = args.undesirable_weight
        self.aux_loss_enabled = getattr(model.config, "output_router_logits", False)
        self.aux_loss_coef = getattr(model.config, "router_aux_loss_coef", 0.0)
        self.calculate_KL = calculate_kl
        if self.calculate_KL and args.train_sampling_strategy != "sequential":
            raise ValueError(
                f"Loss type `'{args.loss_type}'` estimates the KL divergence term and requires "
                f"`train_sampling_strategy='sequential'` because the KL completion for each example is precomputed "
                f"against its neighbors in a fixed-order batch; any other strategy breaks that pairing. "
                f"Got `train_sampling_strategy='{args.train_sampling_strategy}'`."
            )
        if self.calculate_KL and args.per_device_train_batch_size <= 1:
            raise ValueError(
                "Actual (not effective) batch size must be > 1. KTO will not work properly because the KL term will be equivalent to the implied reward."
            )
        if self.aux_loss_enabled and self.aux_loss_coef == 0.0:
            logger.warning(
                "You set `output_router_logits` to `True` in the model config, but `router_aux_loss_coef` is set to "
                "`0.0`, meaning the auxiliary loss will not be used. Either set `router_aux_loss_coef` to a value "
                "greater than `0.0`, or set `output_router_logits` to `False` if you don't want to use the auxiliary "
                "loss.",
            )

        # Dataset
        # Skip dataset preparation for VLMs: tokenization and image processing happen on-the-fly in the collator.
        if not self._is_vision_dataset:
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

        # Initialize activation offloading context
        if self.args.activation_offloading:
            self.maybe_activation_offload_context = get_act_offloading_ctx_manager(model=self.model)
        else:
            self.maybe_activation_offload_context = contextlib.nullcontext()

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

        # Initialize the metrics
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}

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

        self.use_liger_kernel = args.use_liger_kernel
        # Import Liger kernel if enabled
        if self.use_liger_kernel:
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
            if self.precompute_ref_logps:
                raise ValueError(
                    "You cannot use `precompute_ref_log_probs=True` with liger kernel. Please set "
                    "`precompute_ref_log_probs=False`."
                )
            if is_peft_model(self.model):
                raise ValueError(
                    "You cannot use `use_liger_kernel=True` with Peft models. Please set `use_liger_kernel=False`."
                )
            self.kto_loss_fn = LigerFusedLinearKTOLoss(beta=self.beta, use_ref_model=(self.ref_model is not None))

        if self.precompute_ref_logps:
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

    def _get_kl_dataset(
        self,
        dataset: Dataset | IterableDataset,
        dataset_name: str,
        args: KTOConfig,
    ) -> Dataset | IterableDataset:
        """
        Creates the KL dataset by creating mismatched (prompt, completion) pairs for KL divergence estimation.

        Args:
            dataset (`Dataset` or `IterableDataset`):
                Tokenized dataset with `prompt_ids` and `completion_ids` columns.
            dataset_name (`str`):
                Name used in progress bar descriptions.
            args ([`KTOConfig`]):
                Training arguments providing `per_device_train_batch_size` and `dataset_num_proc`.

        Returns:
            `Dataset` or `IterableDataset` with a single `KL_completion_ids` column.
        """
        map_kwargs = {}
        if isinstance(dataset, Dataset):  # IterableDataset does not support num_proc or desc
            map_kwargs["num_proc"] = args.dataset_num_proc
            map_kwargs["desc"] = f"Extracting KL {dataset_name} dataset"
        kl_dataset = dataset.map(
            _get_kl_completion_ids, batched=True, batch_size=args.per_device_train_batch_size, **map_kwargs
        )

        def rename_kl_fn(example):
            return {"KL_completion_ids": example["completion_ids"]}

        if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
            map_kwargs["desc"] = f"Assembling KL {dataset_name} dataset"
        column_names = get_dataset_column_names(dataset)
        kl_dataset = kl_dataset.map(
            rename_kl_fn,
            remove_columns=[c for c in get_dataset_column_names(kl_dataset) if c in column_names],
            **map_kwargs,
        )
        return kl_dataset

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
                # create pairs for estimating the KL term by flipping the matched pairs in each batch of size total_batch_size
                # i.e., (x_1, y_1), ..., (x_n, y_n) --> (x_1, y_n), ..., (x_n, y_1) = (x'_1, y'_1), ..., (x'_n, y'_n)
                kl_dataset = self._get_kl_dataset(dataset, dataset_name, args)
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

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs (usually, "input_ids"
        # and "attention_mask").
        if self._signature_columns is None:
            if self._is_vision_dataset:
                self._signature_columns = [
                    "prompt",
                    "completion",
                    "image",
                    "images",
                    "label",
                    "chat_template_kwargs",
                ]
            else:
                self._signature_columns = [
                    "prompt_ids",
                    "completion_ids",
                    "KL_completion_ids",
                    "label",
                    "ref_logps",
                    "ref_KL_logps",
                ]

    def _precompute_ref_logps(self, dataset: Dataset, name: str, batch_size: int) -> Dataset:
        model_hash = hash_module(self.ref_model or self.model)
        fingerprint = Hasher.hash((dataset._fingerprint, model_hash, self.calculate_KL))
        cache_file = dataset._get_cache_file_path(fingerprint)
        if os.path.exists(cache_file):
            return concatenate_datasets([dataset, Dataset.from_file(cache_file)], axis=1)

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            shuffle=False,
        )
        data_loader = self.accelerator.prepare(dataloader)
        ref_logps = []
        ref_KL_logps = []
        for padded_batch in tqdm(iterable=data_loader, desc=f"Computing reference log probs for {name} dataset"):
            ref_logp, ref_KL_logp = self.compute_ref_log_probs(padded_batch)
            if self.calculate_KL:
                ref_logp, ref_KL_logp = self.accelerator.gather_for_metrics((ref_logp, ref_KL_logp))
                ref_KL_logps.append(ref_KL_logp.cpu())
            else:
                ref_logp = self.accelerator.gather_for_metrics(ref_logp)
            ref_logps.append(ref_logp.cpu())

        ref_logps = torch.cat(ref_logps)
        if self.calculate_KL:
            ref_KL_logps = torch.cat(ref_KL_logps)

        if self.accelerator.is_main_process:

            def add_ref_logps(batch, indices):
                result = {"ref_logps": ref_logps[indices]}
                if self.calculate_KL:
                    result.update({"ref_KL_logps": ref_KL_logps[indices]})
                return result

            dataset.map(
                add_ref_logps,
                with_indices=True,
                batched=True,
                remove_columns=dataset.column_names,
                new_fingerprint=fingerprint,
                desc=f"Caching reference log probs for {name} dataset",
            )
        self.accelerator.wait_for_everyone()

        return concatenate_datasets([dataset, Dataset.from_file(cache_file)], axis=1)

    def compute_ref_log_probs(self, inputs):
        """Computes reference log probabilities for a single padded batch."""
        with torch.no_grad(), disable_gradient_checkpointing(self.model, self.args.gradient_checkpointing_kwargs):
            if self.ref_model is None:
                if is_peft_model(self.model):
                    model = self.accelerator.unwrap_model(self.model)
                    with use_adapter(model, adapter_name="ref" if "ref" in model.peft_config else None):
                        completion_logits = self.model(
                            inputs["completion_input_ids"],
                            attention_mask=inputs["completion_attention_mask"],
                        ).logits

                        if self.calculate_KL:
                            KL_logits = self.model(
                                inputs["KL_completion_input_ids"],
                                attention_mask=inputs["KL_completion_attention_mask"],
                            ).logits
                else:
                    completion_logits = self.model(
                        inputs["completion_input_ids"],
                        attention_mask=inputs["completion_attention_mask"],
                    ).logits

                    if self.calculate_KL:
                        KL_logits = self.model(
                            inputs["KL_completion_input_ids"],
                            attention_mask=inputs["KL_completion_attention_mask"],
                        ).logits
            else:
                completion_logits = self.ref_model(
                    inputs["completion_input_ids"], attention_mask=inputs["completion_attention_mask"]
                ).logits

                if self.calculate_KL:
                    KL_logits = self.ref_model(
                        inputs["KL_completion_input_ids"],
                        attention_mask=inputs["KL_completion_attention_mask"],
                    ).logits

        shift_logits = completion_logits[:, :-1, :].contiguous()
        per_token_logps = selective_log_softmax(shift_logits, inputs["completion_input_ids"][:, 1:].contiguous())
        per_token_logps[inputs["completion_mask"][:, 1:] == 0] = 0.0
        completion_logps = per_token_logps.sum(-1)

        if self.calculate_KL:
            shift_KL_logits = KL_logits[:, :-1, :].contiguous()
            KL_per_token_logps = selective_log_softmax(
                shift_KL_logits, inputs["KL_completion_input_ids"][:, 1:].contiguous()
            )
            KL_per_token_logps[inputs["KL_completion_mask"][:, 1:] == 0] = 0.0
            KL_logps = KL_per_token_logps.sum(-1)
        else:
            KL_logps = None

        return completion_logps, KL_logps

    def kto_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        policy_KL_logps: torch.FloatTensor,
        ref_chosen_logps: torch.FloatTensor,
        ref_rejected_logps: torch.FloatTensor,
        ref_KL_logps: torch.FloatTensor,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the KTO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps:
                Log probabilities of the policy model for the chosen responses. Shape: (num(chosen) in batch_size,)
            policy_rejected_logps:
                Log probabilities of the policy model for the rejected responses. Shape: (num(rejected) in batch_size,)
            policy_KL_logps: Log probabilities of the policy model for the KL responses. Shape: (batch_size,)
            ref_chosen_logps:
                Log probabilities of the reference model for the chosen responses. Shape: (num(chosen) in batch_size,)
            ref_rejected_logps:
                Log probabilities of the reference model for the rejected responses. Shape: (num(rejected) in
                batch_size,)
            ref_KL_logps: Log probabilities of the reference model for the KL responses. Shape: (batch_size,)

        Returns:
            A tuple of four tensors: (losses, chosen_rewards, rejected_rewards, KL). The losses tensor contains the KTO
            loss for each example in the batch. The chosen_rewards and rejected_rewards tensors contain the rewards for
            the chosen and rejected responses, respectively. The KL tensor contains the detached KL divergence estimate
            between the policy and reference models.
        """
        if self.calculate_KL:
            kl = (policy_KL_logps - ref_KL_logps).mean().detach()
            kl = self.accelerator.gather_for_metrics(kl).mean().clamp(min=0)
        else:
            kl = torch.zeros(1).to(policy_chosen_logps.device)

        # Chosen losses
        if policy_chosen_logps.shape[0] != 0 or ref_chosen_logps.shape[0] != 0:
            chosen_logratios = policy_chosen_logps - ref_chosen_logps

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
        if policy_rejected_logps.shape[0] != 0 or ref_rejected_logps.shape[0] != 0:
            rejected_logratios = policy_rejected_logps - ref_rejected_logps

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
            _non_model_keys = {
                "completion_input_ids",
                "completion_attention_mask",
                "completion_mask",
                "KL_completion_mask",
                "KL_completion_token_type_ids",
                "KL_completion_mm_token_type_ids",
                "label",
                "ref_logps",
                "ref_KL_logps",
            }
            KL_model_kwargs = {k: v for k, v in batch.items() if k not in _non_model_keys}
            KL_model_kwargs["input_ids"] = KL_model_kwargs.pop("KL_completion_input_ids")
            KL_model_kwargs["attention_mask"] = KL_model_kwargs.pop("KL_completion_attention_mask")
            # KL sequences have different widths from the main completion after flush_left; override token-type
            # tensors with the KL-specific ones the collator built for exactly this purpose.
            if "KL_completion_token_type_ids" in batch:
                KL_model_kwargs["token_type_ids"] = batch["KL_completion_token_type_ids"]
            if "KL_completion_mm_token_type_ids" in batch:
                KL_model_kwargs["mm_token_type_ids"] = batch["KL_completion_mm_token_type_ids"]

            with torch.no_grad():
                KL_logits = model(**KL_model_kwargs).logits

            shift_KL_logits = KL_logits[:, :-1, :].contiguous()
            KL_per_token_logps = selective_log_softmax(
                shift_KL_logits, batch["KL_completion_input_ids"][:, 1:].contiguous()
            )
            KL_per_token_logps[batch["KL_completion_mask"][:, 1:] == 0] = 0.0
            KL_logps = KL_per_token_logps.sum(-1)
        return KL_logps

    def _compute_loss_liger(self, model, inputs, return_outputs):
        if return_outputs:
            raise RuntimeError(
                "return_outputs=True is not supported with the Liger KTO loss. The Liger loss computes the loss "
                "without materializing logits, so outputs cannot be returned."
            )
        mode = "train" if self.model.training else "eval"
        batch = {k: (v.to(self.accelerator.device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

        labels = torch.tensor(batch["label"])
        num_chosen = labels.sum().to(self.accelerator.device)
        num_rejected = (len(labels) - num_chosen).to(self.accelerator.device)

        policy_KL_logps = self._compute_kl_logps(model, batch)
        ref_KL_logps = self._compute_kl_logps(self.ref_model, batch)
        if self.calculate_KL:
            kl = (policy_KL_logps - ref_KL_logps).mean().detach()
            kl = self.accelerator.gather_for_metrics(kl).mean().clamp(min=0)
        else:
            kl = torch.zeros(1).to(self.accelerator.device)

        _non_model_keys = {
            "completion_mask",
            "KL_completion_input_ids",
            "KL_completion_attention_mask",
            "KL_completion_mask",
            "KL_completion_token_type_ids",
            "KL_completion_mm_token_type_ids",
            "label",
            "ref_logps",
            "ref_KL_logps",
        }
        model_kwargs = {k: v for k, v in batch.items() if k not in _non_model_keys}
        model_kwargs["input_ids"] = model_kwargs.pop("completion_input_ids")
        model_kwargs["attention_mask"] = model_kwargs.pop("completion_attention_mask")
        model_kwargs["use_cache"] = False
        if self.aux_loss_enabled:
            model_kwargs["output_router_logits"] = True

        # `base_model` gives the inner module (skipping `lm_head`) — text decoder for LMs, multimodal wrapper for
        # VLMs (so vision-token injection runs before the text decoder). `get_decoder()` won't do: on VLMs it
        # returns just the text stack and feeds image-placeholder IDs through it.
        # Pre-5.0 transformers VLMs set `base_model_prefix = ""` so `base_model is self` (re-runs `lm_head`).
        # Fall back to `.model` there.
        if self._is_vlm and Version(transformers.__version__) < Version("5.0.0"):
            backbone, ref_backbone = model.model, self.ref_model.model
        else:
            backbone, ref_backbone = model.base_model, self.ref_model.base_model

        outputs = backbone(**model_kwargs)

        # reference model
        with torch.no_grad(), disable_gradient_checkpointing(self.model, self.args.gradient_checkpointing_kwargs):
            ref_outputs = ref_backbone(**{k: v for k, v in model_kwargs.items() if k != "output_router_logits"})
        lm_head = model.get_output_embeddings()
        ref_lm_head = self.ref_model.get_output_embeddings()

        shift_completion_mask = batch["completion_mask"][:, 1:].contiguous()
        target = batch["completion_input_ids"][:, 1:].clone()
        target[shift_completion_mask == 0] = -100

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
            target=target,
            bias=lm_head.bias if hasattr(lm_head, "bias") else None,
            preference_labels=torch.tensor(batch["label"], dtype=torch.bool).to(self.accelerator.device),
            ref_input=ref_outputs.last_hidden_state[:, :-1],
            ref_weight=ref_lm_head.weight,
            ref_bias=ref_lm_head.bias if hasattr(lm_head, "bias") else None,
            kl=kl,
        )
        if self.aux_loss_enabled:
            loss += self.aux_loss_coef * outputs.aux_loss

        self._metrics[mode]["kl"].append(kl.item())

        all_num_chosen = self.accelerator.gather_for_metrics(num_chosen).sum().item()
        all_num_rejected = self.accelerator.gather_for_metrics(num_rejected).sum().item()

        if all_num_chosen > 0:
            self._metrics[mode]["rewards/chosen"].append(
                self.accelerator.gather_for_metrics(chosen_rewards_sum.nansum()).nansum().item() / all_num_chosen
            )
            self._metrics[mode]["logps/chosen"].append(
                self.accelerator.gather_for_metrics(chosen_logps_sum.nansum()).nansum().item() / all_num_chosen
            )
            self._metrics[mode]["logits/chosen"].append(
                self.accelerator.gather_for_metrics(chosen_logits_sum.nansum()).nansum().item() / all_num_chosen
            )

        if all_num_rejected > 0:
            self._metrics[mode]["rewards/rejected"].append(
                self.accelerator.gather_for_metrics(rejected_rewards_sum.nansum()).nansum().item() / all_num_rejected
            )
            self._metrics[mode]["logps/rejected"].append(
                self.accelerator.gather_for_metrics(rejected_logps_sum.nansum()).nansum().item() / all_num_rejected
            )
            self._metrics[mode]["logits/rejected"].append(
                self.accelerator.gather_for_metrics(rejected_logits_sum.nansum()).nansum().item() / all_num_rejected
            )

        return loss

    def _compute_loss(self, model, inputs, return_outputs):
        """Compute the KTO loss and other metrics for the given batch of inputs for train or test."""
        mode = "train" if self.model.training else "eval"
        batch = {k: (v.to(self.accelerator.device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

        labels = torch.tensor(batch["label"])
        num_chosen = labels.sum().to(self.accelerator.device)
        num_rejected = (len(labels) - num_chosen).to(self.accelerator.device)

        policy_KL_logps = self._compute_kl_logps(model, batch)

        _non_model_keys = {
            "completion_mask",
            "KL_completion_input_ids",
            "KL_completion_attention_mask",
            "KL_completion_mask",
            "KL_completion_token_type_ids",
            "KL_completion_mm_token_type_ids",
            "label",
            "ref_logps",
            "ref_KL_logps",
        }
        model_kwargs = {k: v for k, v in batch.items() if k not in _non_model_keys}
        model_kwargs["input_ids"] = model_kwargs.pop("completion_input_ids")
        model_kwargs["attention_mask"] = model_kwargs.pop("completion_attention_mask")
        if self.aux_loss_enabled:
            model_kwargs["output_router_logits"] = True

        outputs = model(**model_kwargs)
        if self.aux_loss_enabled:
            aux_loss = outputs.aux_loss

        shift_logits = outputs.logits[:, :-1, :].contiguous()
        per_token_logps = selective_log_softmax(shift_logits, batch["completion_input_ids"][:, 1:].contiguous())
        per_token_logps[batch["completion_mask"][:, 1:] == 0] = 0.0
        completion_logps = per_token_logps.sum(-1)

        if completion_logps.shape[0] != len(batch["label"]):
            raise ValueError(
                "There is a mismatch between the number of examples in this batch and the number of "
                "examples for which an output sequence was predicted."
            )

        device = outputs.logits.device
        bool_labels = torch.as_tensor(batch["label"], dtype=torch.bool, device=device)
        chosen_idx = torch.nonzero(bool_labels, as_tuple=False).view(-1)
        rejected_idx = torch.nonzero(~bool_labels, as_tuple=False).view(-1)

        policy_chosen_logps = completion_logps.index_select(0, chosen_idx)
        policy_rejected_logps = completion_logps.index_select(0, rejected_idx)
        policy_chosen_logits = outputs.logits.index_select(0, chosen_idx)
        policy_rejected_logits = outputs.logits.index_select(0, rejected_idx)

        if self.precompute_ref_logps:
            ref_chosen_logps = batch["ref_logps"].index_select(0, chosen_idx)
            ref_rejected_logps = batch["ref_logps"].index_select(0, rejected_idx)
            if self.calculate_KL:
                ref_KL_logps = batch["ref_KL_logps"]
            else:
                ref_KL_logps = None
        else:
            ref_model_kwargs = {k: v for k, v in model_kwargs.items() if k != "output_router_logits"}
            with torch.no_grad(), disable_gradient_checkpointing(self.model, self.args.gradient_checkpointing_kwargs):
                if is_peft_model(self.model) and self.ref_model is None:
                    ref_model_unwrapped = self.accelerator.unwrap_model(self.model)
                    with use_adapter(
                        ref_model_unwrapped, adapter_name="ref" if "ref" in ref_model_unwrapped.peft_config else None
                    ):
                        ref_KL_logps = self._compute_kl_logps(self.model, batch)
                        ref_outputs = self.model(**ref_model_kwargs)
                else:
                    ref_KL_logps = self._compute_kl_logps(self.ref_model, batch)
                    ref_outputs = self.ref_model(**ref_model_kwargs)
            ref_shift_logits = ref_outputs.logits[:, :-1, :].contiguous()
            ref_per_token_logps = selective_log_softmax(
                ref_shift_logits, batch["completion_input_ids"][:, 1:].contiguous()
            )
            ref_per_token_logps[batch["completion_mask"][:, 1:] == 0] = 0.0
            ref_completion_logps = ref_per_token_logps.sum(-1)
            ref_chosen_logps = ref_completion_logps.index_select(0, chosen_idx)
            ref_rejected_logps = ref_completion_logps.index_select(0, rejected_idx)

        losses, chosen_rewards, rejected_rewards, kl = self.kto_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            policy_KL_logps,
            ref_chosen_logps,
            ref_rejected_logps,
            ref_KL_logps,
        )

        self._metrics[mode]["kl"].append(kl.item())

        all_num_chosen = self.accelerator.gather_for_metrics(num_chosen).sum().item()
        all_num_rejected = self.accelerator.gather_for_metrics(num_rejected).sum().item()

        if all_num_chosen > 0:
            self._metrics[mode]["rewards/chosen"].append(
                self.accelerator.gather_for_metrics(chosen_rewards.nansum()).nansum().item() / all_num_chosen
            )
            self._metrics[mode]["logps/chosen"].append(
                self.accelerator.gather_for_metrics(policy_chosen_logps.nansum()).nansum().item() / all_num_chosen
            )
            self._metrics[mode]["logits/chosen"].append(
                self.accelerator.gather_for_metrics(policy_chosen_logits.nansum()).nansum().item() / all_num_chosen
            )

        if all_num_rejected > 0:
            self._metrics[mode]["rewards/rejected"].append(
                self.accelerator.gather_for_metrics(rejected_rewards.nansum()).nansum().item() / all_num_rejected
            )
            self._metrics[mode]["logps/rejected"].append(
                self.accelerator.gather_for_metrics(policy_rejected_logps.nansum()).nansum().item() / all_num_rejected
            )
            self._metrics[mode]["logits/rejected"].append(
                self.accelerator.gather_for_metrics(policy_rejected_logits.nansum()).nansum().item() / all_num_rejected
            )

        loss = losses.nanmean()
        if self.aux_loss_enabled:
            loss += self.aux_loss_coef * aux_loss

        return (loss, outputs) if return_outputs else loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if self.use_liger_kernel:
            return self._compute_loss_liger(model, inputs, return_outputs)
        return self._compute_loss(model, inputs, return_outputs)

    def _get_train_sampler(self, train_dataset: Dataset | None = None) -> torch.utils.data.Sampler | None:
        if self.calculate_KL and Version(transformers.__version__) < Version("5.2.0"):
            if train_dataset is None:
                train_dataset = self.train_dataset
            if train_dataset is None or not has_length(train_dataset):
                return None
            return SequentialSampler(train_dataset)
        return super()._get_train_sampler(train_dataset)

    # During eval, Trainer calls prediction_step. If no labels are present in the inputs, it only runs forward and
    # returns logits. We override prediction_step to force compute_loss, because this trainer doesn't involve labels.
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad(), self.compute_loss_context_manager():
            if prediction_loss_only:
                loss = self.compute_loss(model, inputs, return_outputs=False)  # logits aren't materialized with liger
                logits, labels = None, None
            else:
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                logits, labels = outputs.logits, inputs["completion_input_ids"]
        return loss, logits, labels

    # Override training step to add activation offloading context.
    def training_step(self, *args, **kwargs):
        with self.maybe_activation_offload_context:
            return super().training_step(*args, **kwargs)

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        mode = "train" if self.model.training else "eval"
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}  # average the metrics
        if "rewards/chosen" in metrics and "rewards/rejected" in metrics:
            metrics["rewards/margins"] = metrics["rewards/chosen"] - metrics["rewards/rejected"]
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
