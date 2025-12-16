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
import textwrap
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from accelerate import PartialState
from accelerate.logging import get_logger
from accelerate.utils import is_peft_model
from datasets import Dataset, IterableDataset
from datasets.fingerprint import Hasher
from transformers import (
    AutoProcessor,
    DataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
)
from transformers.data.data_collator import DataCollatorMixin
from transformers.trainer_callback import TrainerCallback
from transformers.utils import is_peft_available

from ...data_utils import extract_prompt, is_conversational, prepare_multimodal_messages
from ...models import get_act_offloading_ctx_manager, prepare_deepspeed, prepare_fsdp
from ...models.utils import disable_gradient_checkpointing
from ...trainer.base_trainer import BaseTrainer
from ...trainer.callbacks import SyncRefModelCallback
from ...trainer.utils import (
    create_model_from_path,
    disable_dropout_in_model,
    entropy_from_logits,
    flush_left,
    flush_right,
    get_config_model_id,
    hash_module,
    pad,
    remove_none_values,
    selective_log_softmax,
)
from .dpo_config import DPOConfig


if is_peft_available():
    from peft import PeftConfig, PeftModel, get_peft_model


logger = get_logger(__name__)


FLASH_ATTENTION_VARIANTS = {
    "flash_attention_2",
    "flash_attention_3",
    "kernels-community/flash-attn2",
    "kernels-community/flash-attn3",
    "kernels-community/vllm-flash-attn3",
}


def get_dataset_column_names(dataset: Dataset | IterableDataset) -> list[str]:
    return list(next(iter(dataset)).keys()) if dataset.column_names is None else dataset.column_names


@dataclass
class DataCollatorForPreference(DataCollatorMixin):
    """
    Data collator used for preference data. Inputs are dynamically padded to the maximum length of a batch.

    This collator expects each example in the input list to be a dictionary containing the keys `"prompt_ids"`,
    `"chosen_ids"` and `"rejected_input_ids"`. The collator returns a dictionary containing the following keys:
    - `"input_ids"`: Tensor of input IDs, padded to the maximum length of the batch. The first half of the batch
        corresponds to the `"chosen_input_ids"` and the second half to the `"rejected_input_ids"`.
    - `"attention_mask"`: Tensor of attention mask, padded to the maximum length of the batch.
    - `"completion_mask"`: Tensor indicating the positions of the completion tokens, padded to the maximum length of
        the batch.

    Optionally, the examples can contain a `"ref_chosen_logps"` and `"ref_rejected_logps"` keys, in which case the
    returned dictionary will also contain these keys with the corresponding tensors.

    Args:
        pad_token_id (`int`):
            Token ID to use for padding.
        pad_to_multiple_of (`int`, *optional*):
            If set, the sequences will be padded to a multiple of this value.
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            Type of Tensor to return. Only `"pt"` is currently supported.

    Examples:
    ```python
    >>> from trl.trainer.dpo_trainer import DataCollatorForPreference

    >>> collator = DataCollatorForPreference(pad_token_id=0)
    >>> examples = [
    ...     {"prompt_ids": [1, 2, 3], "chosen_ids": [4, 5], "rejected_ids": [6]},
    ...     {"prompt_ids": [7, 8], "chosen_ids": [9], "rejected_ids": [10, 11]},
    ... ]
    >>> collator(examples)
    {'input_ids': tensor([[ 1,  2,  3,  4,  5],
                          [ 7,  8,  9,  0,  0],
                          [ 1,  2,  3,  6,  0],
                          [ 7,  8, 10, 11,  0]]),
     'attention_mask': tensor([[1, 1, 1, 1, 1],
                               [1, 1, 1, 0, 0],
                               [1, 1, 1, 1, 0],
                               [1, 1, 1, 1, 0]]),
     'completion_mask': tensor([[0, 0, 0, 1, 1],
                                [0, 0, 1, 0, 0],
                                [0, 0, 0, 1, 0],
                                [0, 0, 1, 1, 0]])}
    ```
    """

    pad_token_id: int
    pad_to_multiple_of: int | None = None
    return_tensors: str = "pt"

    def torch_call(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        prompt_chosen_ids = [example["prompt_ids"] + example["chosen_ids"] for example in examples]
        prompt_rejected_ids = [example["prompt_ids"] + example["rejected_ids"] for example in examples]
        chosen_attention_mask = [[1] * len(example["prompt_ids"] + example["chosen_ids"]) for example in examples]
        rejected_attention_mask = [[1] * len(example["prompt_ids"] + example["rejected_ids"]) for example in examples]
        chosen_mask = [[0] * len(example["prompt_ids"]) + [1] * len(example["chosen_ids"]) for example in examples]
        rejected_mask = [[0] * len(example["prompt_ids"]) + [1] * len(example["rejected_ids"]) for example in examples]
        input_ids = prompt_chosen_ids + prompt_rejected_ids
        attention_mask = chosen_attention_mask + rejected_attention_mask
        completion_mask = chosen_mask + rejected_mask

        # Convert to tensor
        input_ids = [torch.tensor(ids) for ids in input_ids]
        attention_mask = [torch.tensor(m, dtype=torch.long) for m in attention_mask]
        completion_mask = [torch.tensor(m, dtype=torch.long) for m in completion_mask]
        if "ref_chosen_logps" in examples[0]:
            ref_chosen_logps = torch.tensor([example["ref_chosen_logps"] for example in examples])
        if "ref_rejected_logps" in examples[0]:
            ref_rejected_logps = torch.tensor([example["ref_rejected_logps"] for example in examples])

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
        if "ref_chosen_logps" in examples[0]:
            output["ref_chosen_logps"] = ref_chosen_logps
        if "ref_rejected_logps" in examples[0]:
            output["ref_rejected_logps"] = ref_rejected_logps
        return output


class DPOTrainer(BaseTrainer):
    """
    Trainer for Direct Preference Optimization (DPO) method. This algorithm was initially proposed in the paper [Direct
    Preference Optimization: Your Language Model is Secretly a Reward Model](https://huggingface.co/papers/2305.18290).
    This class is a wrapper around the [`~transformers.Trainer`] class and inherits all of its attributes and methods.

    Example:

    ```python
    from trl import DPOTrainer
    from datasets import load_dataset

    dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")

    trainer = DPOTrainer(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        train_dataset=dataset,
    )
    trainer.train()
    ```

    Args:
        model (`str | PreTrainedModel`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or a
              path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
              using `<ModelArchitecture>.from_pretrained` (where `<ModelArchitecture>` is derived from the model
              config) with the keyword arguments in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        args ([`DPOConfig`], *optional*):
            Configuration for this trainer. If `None`, a default configuration is used.
        data_collator ([`~transformers.DataCollator`], *optional*):
            Function to use to form a batch from a list of elements of the processed `train_dataset` or `eval_dataset`.
            Will default to [`~trainer.dpo_trainer.DataCollatorForPreference`] if the model is a language model and
            [`~trainer.dpo_trainer.DataCollatorForVisionLanguageModeling`] if the model is a vision-language model.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. This trainer supports both [language modeling](#language-modeling) type and
            [prompt-completion](#prompt-completion) type. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Dataset | IterableDataset]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], [`~transformers.ProcessorMixin`], *optional*):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoProcessor.from_pretrained`]. A
            padding token, `tokenizer.pad_token`, must be set. If the processing class has not set a padding token,
            `tokenizer.eos_token` will be used as the default.
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

    _tag_names = ["trl", "dpo"]
    _name = "DPO"
    _paper = {
        "title": "Direct Preference Optimization: Your Language Model is Secretly a Reward Model",
        "id": "2305.18290",
        # docstyle-ignore
        "citation": textwrap.dedent("""\
            @inproceedings{rafailov2023direct,
                title        = {{Direct Preference Optimization: Your Language Model is Secretly a Reward Model}},
                author       = {Rafael Rafailov and Archit Sharma and Eric Mitchell and Christopher D. Manning and Stefano Ermon and Chelsea Finn},
                year         = 2023,
                booktitle    = {Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023},
                url          = {http://papers.nips.cc/paper_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract-Conference.html},
                editor       = {Alice Oh and Tristan Naumann and Amir Globerson and Kate Saenko and Moritz Hardt and Sergey Levine},
            }"""),
    }

    def __init__(
        self,
        model: str | PreTrainedModel,
        args: DPOConfig | None = None,
        data_collator: DataCollator | None = None,
        train_dataset: Dataset | IterableDataset | None = None,
        eval_dataset: Dataset | IterableDataset | dict[str, Dataset | IterableDataset] | None = None,
        processing_class: PreTrainedTokenizerBase | ProcessorMixin | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None] = (None, None),
        peft_config: "PeftConfig | None" = None,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else get_config_model_id(model.config)
            model_name = model_name.split("/")[-1]
            args = DPOConfig(f"{model_name}-DPO")

        # Model
        if isinstance(model, str):
            model_init_kwargs = args.model_init_kwargs or {}
            # Special case for DeepSpeed: requires device_map=None ("auto" fails)
            if args.distributed_state.distributed_type == "DEEPSPEED":
                model_init_kwargs["device_map"] = None
            model = create_model_from_path(model, **model_init_kwargs)
        else:
            if args.model_init_kwargs is not None:
                logger.warning(
                    "You passed `model_init_kwargs` to the `DPOConfig`, but your model is already instantiated. "
                    "The `model_init_kwargs` will be ignored."
                )

        # Processing class
        if processing_class is None:
            processing_class = AutoProcessor.from_pretrained(get_config_model_id(model.config))

        # Handle pad token for processors or tokenizers
        if isinstance(processing_class, ProcessorMixin):
            tokenizer = processing_class.tokenizer
            self._is_vlm = True
        elif isinstance(processing_class, PreTrainedTokenizerBase):
            tokenizer = processing_class
            self._is_vlm = False
        else:
            raise TypeError("The `processing_class` must be either a `PreTrainedTokenizerBase` or a `ProcessorMixin`")

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        self.pad_token = tokenizer.pad_token
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id

        if is_peft_available() and isinstance(model, PeftModel) and peft_config is not None:
            # If the model is already a PeftModel, we need to merge and unload it.
            # Further information: https://huggingface.co/docs/trl/dpo_trainer#reference-model-considerations-with-peft
            model = model.merge_and_unload()

        # Create PEFT model
        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        # When using gradient checkpointing with PEFT, we need to enable input gradients. transformers.Trainer normally
        # handles this, but a bug currently prevents it; see https://github.com/huggingface/transformers/issues/42489
        if is_peft_available() and isinstance(model, PeftModel) and args.gradient_checkpointing:
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

        # Data collator
        dataset_sample = next(iter(train_dataset))
        self._is_vision_dataset = "image" in dataset_sample or "images" in dataset_sample
        if self._is_vision_dataset and not self._is_vlm:
            raise ValueError(
                "The dataset appears to be vision-related (contains 'image' or 'images' keys), but the provided "
                "model does not seem to be a vision-language model. Please check your model and dataset."
            )
        if data_collator is None and not self._is_vision_dataset:
            # Get the pad token: if not provided, use the one from the processing class or the eos token
            # if the processing class does not have a pad token.
            pad_token = args.pad_token or tokenizer.pad_token or tokenizer.eos_token
            pad_token_id = tokenizer.convert_tokens_to_ids(pad_token)
            if pad_token_id is None:
                raise ValueError(
                    f"The specified `pad_token` ('{pad_token}') is not found in the vocabulary of the given "
                    f"`processing_class` ({processing_class.__class__.__name__}). Ensure that the `pad_token` exists "
                    "in the vocabulary before using it as a padding token."
                )
            data_collator = DataCollatorForPreference(
                pad_token_id=pad_token_id,
                pad_to_multiple_of=args.pad_to_multiple_of,
            )
        elif data_collator is None and self._is_vision_dataset:
            raise NotImplementedError("VLM training is not yet implemented.")

        # Training arguments
        self.precompute_ref_logps = args.precompute_ref_log_probs
        self.loss_types = args.loss_type  # args.loss_type is already a list
        self.label_smoothing = args.label_smoothing
        if "robust" in self.loss_types and not (0.0 <= self.label_smoothing < 0.5):
            logger.warning(
                "The `label_smoothing` parameter should lie in [0.0, 0.5) for the 'robust' loss. You provided "
                f"{self.label_smoothing}."
            )
        if "exo_pair" in self.loss_types and self.label_smoothing == 0.0:
            raise ValueError(
                "Label smoothing must be greater than 0.0 when using 'exo_pair' loss. The EXO paper recommends a "
                "value of 1e-3."
            )

        # Dataset
        # Skip dataset preparation if it's a VLM, where preprocessing (e.g., image-to-pixel conversion) is too costly
        # and done on the fly instead.
        skip_prepare_dataset = self._is_vision_dataset
        if not skip_prepare_dataset:
            train_dataset = self._prepare_dataset(train_dataset, processing_class, args, "train")
            if eval_dataset is not None:
                if isinstance(eval_dataset, dict):
                    eval_dataset = {
                        key: self._prepare_dataset(dataset, processing_class, args, key)
                        for key, dataset in eval_dataset.items()
                    }
                else:
                    eval_dataset = self._prepare_dataset(eval_dataset, processing_class, args, "eval")

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in DPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt", "chosen" and "rejected". As a result, the trainer
        # issues the warning: "Could not estimate the number of tokens of the input, floating-point operations will not
        # be computed." To suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued"
        # dictionary to True. This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Initialize activation offloading context
        if self.args.activation_offloading:
            self.maybe_activation_offload_context = get_act_offloading_ctx_manager(model=self.model)
        else:
            self.maybe_activation_offload_context = contextlib.nullcontext()

        # Reference model
        self.beta = args.beta
        if is_peft_model(model):
            # If PEFT is used, the reference model is not needed since the adapter can be disabled to revert to the
            # initial model.
            self.ref_model = None
        else:
            # For deepspeed, fsdp or non-distributed models, create a reference model from scratch
            model_init_kwargs = args.model_init_kwargs or {}
            # Special case for DeepSpeed: requires device_map=None ("auto" fails)
            if self.args.distributed_state.distributed_type == "DEEPSPEED":
                model_init_kwargs["device_map"] = None
            self.ref_model = create_model_from_path(get_config_model_id(self.model.config), **model_init_kwargs)

        # Disable dropout in the models
        if args.disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)

        # Initialize the metrics
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._total_train_tokens = 0

        # Add tags to the model
        self.model.add_model_tags(self._tag_names)

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            elif self.is_fsdp_enabled:
                self.ref_model = prepare_fsdp(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        if args.sync_ref_model:
            self.add_callback(SyncRefModelCallback(ref_model=self.ref_model, accelerator=self.accelerator))

        #

        # if self.args.precompute_ref_log_probs:

        #     self.train_dataset = self._precompute_ref_logps(
        #         ref_model, self.train_dataset, self.args.per_device_train_batch_size, "train"
        #     )
        #     if self.eval_dataset is not None:
        #         if isinstance(self.eval_dataset, dict):
        #             self.eval_dataset = {
        #                 key: self._precompute_ref_logps(ref_model, dataset, self.args.per_device_eval_batch_size, key)
        #                 for key, dataset in self.eval_dataset.items()
        #             }
        #         else:
        #             self.eval_dataset = self._precompute_ref_logps(
        #                 self.eval_dataset, self.args.per_device_eval_batch_size, "eval"
        #             )

    def _prepare_dataset(
        self,
        dataset: Dataset | IterableDataset,
        processing_class: PreTrainedTokenizerBase | ProcessorMixin,
        args: DPOConfig,
        dataset_name: str,
    ) -> Dataset | IterableDataset:
        # Tabular backends like Arrow/Parquet insert `None` for mismatched keys in nested structures. Clean them from
        # sampled data.
        if isinstance(dataset, Dataset):  # IterableDataset does not support `with_transform`
            dataset = dataset.with_transform(remove_none_values)

        # Build the kwargs for the `map` function
        map_kwargs = {}
        if isinstance(dataset, Dataset):  # IterableDataset does not support num_proc
            map_kwargs["num_proc"] = args.dataset_num_proc

        with PartialState().main_process_first():
            # Extract the prompt if needed
            first_example = next(iter(dataset))
            if "prompt" not in first_example:
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Extracting prompt from {dataset_name} dataset"
                dataset = dataset.map(extract_prompt, **map_kwargs)

            # Apply the chat template if needed
            first_example = next(iter(dataset))
            if not is_conversational(first_example):
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Adding EOS to {dataset_name} dataset"

                def add_eos(example, eos_token):
                    if not example["chosen"].endswith(eos_token):
                        example["chosen"] = example["chosen"] + eos_token
                    if not example["rejected"].endswith(eos_token):
                        example["rejected"] = example["rejected"] + eos_token
                    return example

                dataset = dataset.map(add_eos, fn_kwargs={"eos_token": processing_class.eos_token}, **map_kwargs)

            # Tokenize the dataset
            if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                map_kwargs["desc"] = f"Tokenizing {dataset_name} dataset"

            def tokenize_fn(example, processing_class):
                output = {}
                if is_conversational(example):
                    if self._is_vlm:
                        prepare_multimodal_messages(example["prompt"], num_images=0)
                        prepare_multimodal_messages(example["completion"], num_images=0)
                    prompt_ids = processing_class.apply_chat_template(
                        example["prompt"],
                        tokenize=True,
                        add_generation_prompt=True,
                        tools=example.get("tools"),
                        **example.get("chat_template_kwargs", {}),
                    )
                    prompt_chosen_processed = processing_class.apply_chat_template(
                        example["prompt"] + example["chosen"],
                        return_dict=True,
                        tokenize=True,
                        tools=example.get("tools"),
                        **example.get("chat_template_kwargs", {}),
                    )
                    prompt_rejected_processed = processing_class.apply_chat_template(
                        example["prompt"] + example["rejected"],
                        return_dict=True,
                        tokenize=True,
                        tools=example.get("tools"),
                        **example.get("chat_template_kwargs", {}),
                    )
                    # Fix transformers inconsistency: for VLMs, apply_chat_template returns lists of lists
                    # even for single examples, while for LLMs it returns lists of ints.
                    prompt_ids = prompt_ids[0] if isinstance(prompt_ids[0], list) else prompt_ids
                    prompt_chosen_processed = {
                        k: v[0] if isinstance(v[0], list) else v for k, v in prompt_chosen_processed.items()
                    }
                    prompt_rejected_processed = {
                        k: v[0] if isinstance(v[0], list) else v for k, v in prompt_rejected_processed.items()
                    }
                    prompt_chosen_ids = prompt_chosen_processed["input_ids"]
                    prompt_rejected_ids = prompt_rejected_processed["input_ids"]
                else:
                    prompt_ids = processing_class(text=example["prompt"])["input_ids"]
                    prompt_chosen_ids = processing_class(text=example["prompt"] + example["chosen"])["input_ids"]
                    prompt_rejected_ids = processing_class(text=example["prompt"] + example["rejected"])["input_ids"]

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

                output["prompt_ids"] = prompt_ids
                output["chosen_ids"] = prompt_chosen_ids[len(prompt_ids) :]
                output["rejected_ids"] = prompt_rejected_ids[len(prompt_ids) :]
                return output

            dataset = dataset.map(tokenize_fn, fn_kwargs={"processing_class": processing_class}, **map_kwargs)

            # For Liger kernel, ensure only the essential columns
            if args.use_liger_kernel:
                collator_expected_keys = {"input_ids", "completion_mask"}
                column_names = get_dataset_column_names(dataset)
                dataset = dataset.select_columns(collator_expected_keys.intersection(column_names))

        return dataset

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs (usually, "input_ids"
        # and "attention_mask").
        if self._signature_columns is None:
            if self._is_vision_dataset:
                self._signature_columns = ["prompt", "chosen", "rejected"]
            else:
                self._signature_columns = [
                    "prompt_ids",
                    "chosen_ids",
                    "rejected_ids",
                    "ref_chosen_logps",
                    "ref_rejected_logps",
                ]

    def _precompute_ref_logps(
        self, model, dataset: Dataset | IterableDataset, batch_size: int, dataset_name: str
    ) -> None:
        def compute_ref_logps(examples, collator, max_length, truncation_mode):
            examples = [
                dict(zip(examples.keys(), v, strict=False)) for v in zip(*examples.values(), strict=False)
            ]  # dict[list] to list[dict]
            inputs = collator(examples)
            input_ids = inputs["input_ids"].to(model.device)
            attention_mask = inputs["attention_mask"].to(model.device)
            completion_mask = inputs["completion_mask"].to(model.device)

            # Truncate inputs
            if max_length is not None:
                if truncation_mode == "keep_start":
                    input_ids = input_ids[:, :max_length]
                    attention_mask = attention_mask[:, :max_length]
                    completion_mask = completion_mask[:, :max_length]
                elif truncation_mode == "keep_end":
                    attention_mask, input_ids, completion_mask = flush_right(
                        attention_mask, input_ids, completion_mask
                    )
                    input_ids = input_ids[:, -max_length:]
                    attention_mask = attention_mask[:, -max_length:]
                    completion_mask = completion_mask[:, -max_length:]
                    attention_mask, input_ids, completion_mask = flush_left(attention_mask, input_ids, completion_mask)
                else:
                    raise ValueError(
                        f"Unsupported truncation mode: {truncation_mode}, expected 'keep_start' or 'keep_end'"
                    )

            outputs = model(input_ids, attention_mak=attention_mask, use_cache=False)
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            shift_completion_mask = completion_mask[..., 1:].contiguous()
            per_token_logps = selective_log_softmax(shift_logits, shift_labels)
            per_token_logps[shift_completion_mask == 0] = 0.0  # mask out non-completion tokens
            logps = per_token_logps.sum(dim=1)  # sum over sequence length
            chosen_logps, rejected_logps = logps.chunk(2, dim=0)  # batch is [chosen, rejected]
            return {"ref_chosen_logps": chosen_logps.tolist(), "ref_rejected_logps": rejected_logps.tolist()}

        # Normally, `map` creates a fingerprint based on the transform function and its arguments. However, the model’s
        # produces a different fingerprint on each run, which prevents the cache from being used. To fix this, we
        # manually compute a stable fingerprint for the model instead.
        fn_kwargs = {
            "collator": self.data_collator,
            "max_length": self.args.max_length,
            "truncation_mode": self.args.truncation_mode,
        }
        model_hash = hash_module(model)
        dataset = dataset.map(
            compute_ref_logps,
            batched=True,
            batch_size=batch_size,
            fn_kwargs=fn_kwargs,
            desc=f"Computing reference logps for {dataset_name} dataset",
            new_fingerprint=Hasher.hash((dataset._fingerprint, fn_kwargs, model_hash)),
        )
        return dataset

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        mode = "train" if self.model.training else "eval"
        device = self.accelerator.device

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        completion_mask = inputs["completion_mask"]

        # Truncate inputs
        if self.args.max_length is not None:
            if self.args.truncation_mode == "keep_start":
                input_ids = input_ids[:, : self.args.max_length]
                attention_mask = attention_mask[:, : self.args.max_length]
                completion_mask = completion_mask[:, : self.args.max_length]
            elif self.args.truncation_mode == "keep_end":
                attention_mask, input_ids, completion_mask = flush_right(attention_mask, input_ids, completion_mask)
                input_ids = input_ids[:, -self.args.max_length :]
                attention_mask = attention_mask[:, -self.args.max_length :]
                completion_mask = completion_mask[:, -self.args.max_length :]
                attention_mask, input_ids, completion_mask = flush_left(attention_mask, input_ids, completion_mask)
            else:
                raise ValueError(
                    f"Unsupported truncation mode: {self.args.truncation_mode}, expected 'keep_start' or 'keep_end'"
                )

        outputs = model(input_ids, attention_mak=attention_mask, use_cache=False)
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        shift_completion_mask = completion_mask[..., 1:].contiguous()
        per_token_logps = selective_log_softmax(shift_logits, shift_labels)
        per_token_logps[shift_completion_mask == 0] = 0.0  # mask out non-completion tokens
        logps = per_token_logps.sum(dim=1)  # sum over sequence length
        chosen_logps, rejected_logps = logps.chunk(2, dim=0)  # batch is [chosen, rejected]

        if self.precompute_ref_logps:
            ref_chosen_logps, ref_rejected_logps = inputs["ref_chosen_logps"], inputs["ref_rejected_logps"]
        else:
            # When gradient checkpointing is enabled with use_reentrant=True (default), calling the model inside a
            # torch.no_grad() block triggers a harmless PyTorch warning ("None of the inputs have requires_grad=True").
            # Temporarily disable checkpointing to avoid this warning during inference.
            with torch.no_grad(), disable_gradient_checkpointing(self.model, self.args.gradient_checkpointing_kwargs):
                if is_peft_model(model):
                    # Disable PEFT adapters to get the reference model behavior
                    with model.disable_adapter():
                        ref_outputs = model(input_ids, attention_mak=attention_mask, use_cache=False)
                else:
                    ref_outputs = self.ref_model(input_ids, attention_mak=attention_mask, use_cache=False)
            ref_shift_logits = ref_outputs.logits[..., :-1, :].contiguous()
            ref_per_token_logps = selective_log_softmax(ref_shift_logits, shift_labels)
            ref_per_token_logps[shift_completion_mask == 0] = 0.0  # mask out non-completion tokens
            ref_logps = ref_per_token_logps.sum(dim=1)  # sum over sequence length
            ref_chosen_logps, ref_rejected_logps = ref_logps.chunk(2, dim=0)  # batch is [chosen, rejected]

        # Get the log ratios for the chosen and rejected responses
        chosen_logratios = chosen_logps - ref_chosen_logps
        rejected_logratios = rejected_logps - ref_rejected_logps
        delta_log_odds = chosen_logratios - rejected_logratios

        loss = 0.0
        for loss_type in self.loss_types:
            if loss_type == "sigmoid":
                per_sequence_loss = -F.logsigmoid(self.beta * delta_log_odds)

            elif loss_type == "hinge":
                per_sequence_loss = torch.relu(1 - self.beta * delta_log_odds)

            elif loss_type == "ipo":
                # (Eq. 17) of the paper where beta is the regularization parameter for the IPO loss, denoted by τ.
                per_sequence_loss = (delta_log_odds - 1 / (2 * self.beta)) ** 2

            elif loss_type == "exo_pair":
                # Implements EXO-pref from the paper https://huggingface.co/papers/2402.00856, (Eq. 16)
                # Minimize KL(p_fθ || p_rh) for K=2; p_fθ = softmax(βπ * (log πθ − log π_ref)) over {chosen, rejected}
                # p_rh = [(1−ε), ε]; expanded KL gives the weighted logsigmoid form below
                epsilon = torch.tensor(self.label_smoothing, device=device)
                qw = F.sigmoid(self.beta * delta_log_odds)
                log_qw = F.logsigmoid(self.beta * delta_log_odds)
                log_pw = torch.log1p(-epsilon)
                ql = F.sigmoid(-self.beta * delta_log_odds)
                log_ql = F.logsigmoid(-self.beta * delta_log_odds)
                log_pl = torch.log(epsilon)
                per_sequence_loss = qw * (log_qw - log_pw) + ql * (log_ql - log_pl)

            elif loss_type == "nca_pair":
                chosen_rewards = self.beta * chosen_logratios
                rejected_rewards = self.beta * rejected_logratios
                per_sequence_loss = (
                    -F.logsigmoid(chosen_rewards)
                    - 0.5 * F.logsigmoid(-chosen_rewards)
                    - 0.5 * F.logsigmoid(-rejected_rewards)
                )

            elif loss_type == "robust":
                per_sequence_loss = -F.logsigmoid(self.beta * delta_log_odds)
                per_sequence_loss = (
                    (1 - self.label_smoothing) * per_sequence_loss - self.label_smoothing * per_sequence_loss
                ) / (1 - 2 * self.label_smoothing)

            elif loss_type == "bco_pair":
                chosen_rewards = self.beta * chosen_logratios
                rejected_rewards = self.beta * rejected_logratios
                per_sequence_loss = -F.logsigmoid(chosen_rewards) - F.logsigmoid(-rejected_rewards)

            elif loss_type == "sppo_hard":
                # In the paper (https://huggingface.co/papers/2405.00675), SPPO employs a soft probability approach,
                # estimated using the PairRM score. The probability calculation is conducted outside of the trainer
                # class. The version described here is the hard probability version, where P in Equation (4.7) of
                # Algorithm 1 is set to 1 for the winner and 0 for the loser.
                winner_margin_error = (chosen_logratios - 0.5 / self.beta) ** 2
                loser_margin_error = (rejected_logratios + 0.5 / self.beta) ** 2
                per_sequence_loss = winner_margin_error + loser_margin_error

            elif loss_type == "aot":
                logratios = chosen_logps - rejected_logps
                ref_logratios = ref_chosen_logps - ref_rejected_logps
                logratios_sorted, _ = torch.sort(logratios, dim=0)
                ref_logratios_sorted, _ = torch.sort(ref_logratios, dim=0)
                delta = logratios_sorted - ref_logratios_sorted
                per_sequence_loss = (
                    -F.logsigmoid(self.beta * delta) * (1 - self.label_smoothing)
                    - F.logsigmoid(-self.beta * delta) * self.label_smoothing
                )

            elif loss_type == "aot_unpaired":
                chosen_logratios_sorted, _ = torch.sort(chosen_logratios, dim=0)
                rejected_logratios_sorted, _ = torch.sort(rejected_logratios, dim=0)
                delta = chosen_logratios_sorted - rejected_logratios_sorted
                per_sequence_loss = (
                    -F.logsigmoid(self.beta * delta) * (1 - self.label_smoothing)
                    - F.logsigmoid(-self.beta * delta) * self.label_smoothing
                )

            elif loss_type == "apo_zero":
                # Eqn (7) of the APO paper (https://huggingface.co/papers/2408.06266)
                # Use this loss when you believe the chosen outputs are better than your model's default output
                # Increase chosen likelihood and decrease rejected likelihood
                losses_chosen = 1 - F.sigmoid(self.beta * chosen_logratios)
                losses_rejected = F.sigmoid(self.beta * rejected_logratios)
                per_sequence_loss = losses_chosen + losses_rejected

            elif loss_type == "apo_down":
                # Eqn (8) of the APO paper (https://huggingface.co/papers/2408.06266)
                # Use this loss when you believe the chosen outputs are worse than your model's default output.
                # Decrease chosen likelihood and decrease rejected likelihood more
                losses_chosen = F.sigmoid(self.beta * chosen_logratios)
                losses_rejected = 1 - F.sigmoid(self.beta * (chosen_logratios - rejected_logratios))
                per_sequence_loss = losses_chosen + losses_rejected

            elif loss_type == "discopop":
                # Eqn (5) of the DiscoPOP paper (https://huggingface.co/papers/2406.08414)
                logits = delta_log_odds * self.beta
                # Modulate the mixing coefficient based on the log ratio magnitudes
                log_ratio_modulation = torch.sigmoid(logits / self.args.discopop_tau)
                logistic_component = -F.logsigmoid(logits)
                exp_component = torch.exp(-logits)
                # Blend between logistic and exponential component based on log ratio modulation
                per_sequence_loss = (
                    logistic_component * (1 - log_ratio_modulation) + exp_component * log_ratio_modulation
                )

            elif loss_type == "sft":
                chosen_logits, _ = shift_logits.chunk(2, dim=0)
                chosen_labels, _ = shift_labels.chunk(2, dim=0)
                chosen_mask, _ = shift_completion_mask.chunk(2, dim=0)
                batch_loss = F.cross_entropy(chosen_logits[chosen_mask.bool()], chosen_labels[chosen_mask.bool()])
                # Implementation convenience: expand the scalar SFT loss to a per-sequence tensor so it matches the
                # shape of other losses; only the mean is used, so this is a no-op numerically.
                per_sequence_loss = batch_loss.expand(chosen_logits.size(0))

            else:
                raise ValueError(
                    f"Unknown loss type: {loss_type}. Should be one of ['sigmoid', 'hinge', 'ipo', 'exo_pair', "
                    "'nca_pair', 'robust', 'bco_pair', 'sppo_hard', 'aot', 'aot_unpaired', 'apo_zero', 'apo_down', "
                    "'discopop', 'sft']"
                )

            loss += per_sequence_loss.mean()

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
        chosen_logits, rejected_logits = shift_logits.detach().chunk(2, dim=0)
        chosen_mask, rejected_mask = shift_completion_mask.chunk(2, dim=0)
        total_chosen_logits = chosen_logits[chosen_mask.bool()].mean(-1)
        total_chosen_tokens = chosen_mask.sum()
        total_rejected_logits = rejected_logits[rejected_mask.bool()].mean(-1)
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
        chosen_mask = shift_completion_mask[: len(shift_completion_mask) // 2].bool()
        chosen_labels = shift_labels[: len(shift_labels) // 2]
        correct_predictions = (predictions == chosen_labels) & chosen_mask
        total_tokens = chosen_mask.sum()
        correct_tokens = correct_predictions.sum()
        correct_tokens = self.accelerator.gather_for_metrics(correct_tokens)
        total_tokens = self.accelerator.gather_for_metrics(total_tokens)
        total_sum = total_tokens.sum()
        accuracy = (correct_tokens.sum() / total_sum).item() if total_sum > 0 else 0.0
        self._metrics[mode]["mean_token_accuracy"].append(accuracy)

        # Rewards for chosen and rejected completions
        chosen_rewards = self.beta * chosen_logratios.detach()
        rejected_rewards = self.beta * rejected_logratios.detach()
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

    # Override training step to add activation offloading context.
    def training_step(self, *args, **kwargs):
        with self.maybe_activation_offload_context:
            return super().training_step(*args, **kwargs)

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

    # Ensure the model card is saved along with the checkpoint
    def _save_checkpoint(self, model, trial):
        if self.args.hub_model_id is None:
            model_name = Path(self.args.output_dir).name
        else:
            model_name = self.args.hub_model_id.split("/")[-1]
        self.create_model_card(model_name=model_name)
        super()._save_checkpoint(model, trial)
