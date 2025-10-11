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
from typing import Any, Callable, Optional, Union
from ...trainer.utils import     disable_dropout_in_model
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import PartialState, logging
from datasets import Dataset, IterableDataset
from transformers import (
    AutoProcessor,
    BaseImageProcessor,
    DataCollator,
    FeatureExtractionMixin,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
)
from transformers.data.data_collator import DataCollatorMixin
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction
from transformers.utils import is_peft_available

from ..data_utils import (
    apply_chat_template,
    extract_prompt,
    is_conversational,
    prepare_multimodal_messages,
    truncate_dataset,
)
from ..models import get_act_offloading_ctx_manager, prepare_peft_model
from .base_trainer import BaseTrainer
from .dpo_config import DPOConfig
from .utils import (
    create_model_from_path,
    entropy_from_logits,
    flush_left,
    flush_right,
    pad,
    remove_none_values,
    selective_log_softmax,
)


if is_peft_available():
    from peft import PeftConfig, PeftModel


logger = logging.get_logger(__name__)


FLASH_ATTENTION_VARIANTS = {
    "flash_attention_2",
    "flash_attention_3",
    "kernels-community/flash-attn",
    "kernels-community/vllm-flash-attn3",
    "kernels-community/flash-attn3",
}


def get_dataset_column_names(dataset: Union[Dataset, IterableDataset]) -> list[str]:
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
    >>> examples = [{"prompt_ids": [1, 2, 3], {"chosen_ids": [4, 5], "rejected_ids": [6]}]
    >>> collator(examples)
    {'input_ids': tensor([[ 1,  2,  3,  4,  5],
                          [ 1,  2,  3,  6,  0]]),
     'attention_mask': tensor([[1, 1, 1, 1, 1],
                               [1, 1, 1, 1, 0]]),
     'completion_mask': tensor([[0, 0, 0, 1, 1],
                                [0, 0, 0, 1, 0]])}
    ```
    """

    pad_token_id: int
    pad_to_multiple_of: Optional[int] = None
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


@dataclass
class DataCollatorForVisionLanguageModeling(DataCollatorMixin):
    """
    Data collator for vision-language modeling tasks.

    Unlike text-only datasets—where the collator typically receives pre-tokenized inputs ready for batching,
    vision-language data processing involves converting images into pixel values. This conversion is disk-intensive,
    making upfront preprocessing of the entire dataset impractical. Therefore, this collator performs tokenization and
    image processing on-the-fly to efficiently prepare batches.

    Each input example should be a dictionary containing at least:
    - An `"images"` key holding the image data.
    - [language modeling](#language-modeling) type: either a `"messages"` key for conversational inputs or a `"text"`
      key for standard text inputs.
    - [prompt-completion](#prompt-completion) type: keys `"prompt"` and `"completion"` for the prompt and completion.

    The collator outputs a dictionary including:
    - `"input_ids"`: Tensor of token IDs.
    - `"attention_mask"`: Tensor indicating attention mask.
    - `"pixel_values"`: Tensor representing image pixel values.
    - `"labels"`: Tensor for training labels.

    Additional keys may be present depending on the processor, such as `"image_grid_thw"`.

    Args:
        processor (`ProcessorMixin`):
            The processor used to tokenize text and process images. It must be a subclass of `ProcessorMixin` and
            include a `tokenizer` with a defined `pad_token_id`.
        max_length (`int` or `None`, optional, defaults to `None`):
            Maximum sequence length for input tokens. If `None`, no truncation is applied.
        pad_to_multiple_of (`int` or `None`, optional, defaults to `None`):
            If set, the sequences will be padded to a multiple of this value.
        return_tensors (`str`, optional, defaults to `"pt"`):
            The tensor type to return. Currently, only `"pt"` (PyTorch tensors) is supported.

    Example:
    ```python
    >>> from trl.trainer.dpo_trainer import DataCollatorForVisionLanguageModeling
    >>> from transformers import AutoProcessor

    >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    >>> collator = DataCollatorForVisionLanguageModeling(processor)
    >>> examples = [
    ...     {"images": [Image.open("image_0.png")], "messages": [{"role": "user", "content": "What is this?"}]},
    ...     {"images": [Image.open("image_1.png")], "messages": [{"role": "user", "content": "Describe this image."}]},
    ... ]
    >>> collator(examples)
    {'input_ids': tensor([[151644,   8948,    198,   2610,    525,    264,  10950,  17847,     13,  151645,    198,
                           151644,    872,    198, 151652, 151655, 151655, 151655,  151655, 151653,   3838,    374,
                              419,     30, 151645,    198],
                          [151644,   8948,    198,   2610,    525,    264,  10950,  17847,     13,  151645,    198,
                           151644,    872,    198, 151652, 151655, 151655, 151655,  151655, 151653,  74785,    419,
                             2168,     13, 151645,    198]]),
     'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                               [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]),
     'pixel_values': tensor([[-0.9893,  0.1785,  1.5362,  ..., -0.0582,  0.8661, -0.2431],
                             [-0.2302,  0.9522, -1.1061,  ...,  0.0555,  1.3354, -0.6412],
                             [ 1.2150,  0.9084,  0.7041,  ...,  0.2404, -0.8403, -0.5133],
                             ...,
                             [ 0.6895,  0.2807,  0.2515,  ..., -0.2004, -1.2100,  0.0555],
                             [ 0.8209, -0.9748,  1.5654,  ...,  1.6055, -0.4706,  0.5817],
                             [-1.0915,  0.4559,  0.9230,  ...,  0.5106,  0.0982, -0.1720]]),
     'image_grid_thw': tensor([[1, 4, 4],
                               [1, 4, 4]]),
     'labels': tensor([[151644,   8948,    198,   2610,    525,    264,  10950,  17847,     13,  151645,    198,
                        151644,    872,    198, 151652, 151655, 151655, 151655,  151655, 151653,   3838,    374,
                           419,     30, 151645,    198],
                        [151644,   8948,    198,   2610,    525,    264,  10950,  17847,     13,  151645,    198,
                         151644,    872,    198, 151652, 151655, 151655, 151655,  151655, 151653,  74785,    419,
                           2168,     13, 151645,    198]])}
    ```
    """

    processor: ProcessorMixin
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def torch_call(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        if "messages" in examples[0] or self.dataset_text_field in examples[0]:
            return self._collate_language_modeling(examples)
        elif "prompt" in examples[0] and "completion" in examples[0]:
            return self._collate_prompt_completion(examples)
        else:
            raise KeyError(f"Unexpected input keys in examples: {list(examples[0].keys())}.")

    def _collate_language_modeling(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        images = [example["images"] for example in examples]
        # Transformers requires at least one image in the batch, otherwise it throws an error
        if all(img_list == [] for img_list in images):
            images = None

        if "messages" in examples[0]:  # conversational case
            for example in examples:
                prepare_multimodal_messages(example["messages"], len(example["images"]))
            messages = [example["messages"] for example in examples]
            texts = self.processor.apply_chat_template(messages)
        elif self.dataset_text_field in examples[0]:  # standard case
            texts = [example[self.dataset_text_field] for example in examples]
        else:
            raise KeyError(
                "The input examples must contain either 'messages' for conversational data or 'text' for standard "
                "data."
            )

        output = self.processor(
            images=images,
            text=texts,
            padding=True,
            padding_side="right",
            pad_to_multiple_of=self.pad_to_multiple_of,
            truncation=self.max_length is not None,
            max_length=self.max_length,
            return_tensors=self.return_tensors,
            add_special_tokens=False,  # to avoid adding the BOS, twice see https://huggingface.co/blog/qgallouedec/gotchas-in-tokenizer-behavior#7-chat-template-and-tokenization-dont-compose-due-to-special-tokens
        )
        labels = output["input_ids"].clone()
        labels[output["attention_mask"] == 0] = -100
        # We mask only padding tokens (-100) in the labels. Vision tokens are left unchanged because their handling in
        # loss computation has to be done by the model, and masking them here would be infeasible in practice as vision
        # token definitions vary across architectures.
        output["labels"] = labels
        return output

    def _collate_prompt_completion(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        if self.pad_to_multiple_of is not None:
            raise NotImplementedError(
                "Padding to a multiple of a value is not yet implemented for vision-language modeling and "
                "prompt-completion data yet."
            )
        images = [example["images"] for example in examples]
        # Transformers requires at least one image in the batch, otherwise it throws an error
        if all(img_list == [] for img_list in images):
            images = None
        if is_conversational(examples[0]):  # conversational case
            for example in examples:
                prepare_multimodal_messages(example["prompt"] + example["completion"], len(example["images"]))
            examples = [apply_chat_template(example, self.processor) for example in examples]

        prompts = [example["prompt"] for example in examples]
        completions = [example["completion"] for example in examples]

        processed_prompts = self.processor(
            images=images,
            text=prompts,
            padding=True,
            padding_side="left",
            return_tensors=self.return_tensors,
            add_special_tokens=False,  # to avoid adding the BOS, twice see https://huggingface.co/blog/qgallouedec/gotchas-in-tokenizer-behavior#7-chat-template-and-tokenization-dont-compose-due-to-special-tokens
        )
        processed_completions = self.processor(
            text=completions,
            padding=True,
            padding_side="right",
            return_tensors=self.return_tensors,
            add_special_tokens=False,  # to avoid adding the BOS, twice see https://huggingface.co/blog/qgallouedec/gotchas-in-tokenizer-behavior#7-chat-template-and-tokenization-dont-compose-due-to-special-tokens
        )

        # Concatenate prompts and completions
        prompt_ids, completion_ids = processed_prompts["input_ids"], processed_completions["input_ids"]
        prompt_mask, completion_mask = processed_prompts["attention_mask"], processed_completions["attention_mask"]
        input_ids = torch.cat((prompt_ids, completion_ids), dim=1)
        attention_mask = torch.cat((prompt_mask, completion_mask), dim=1)
        completion_mask = torch.cat((torch.zeros_like(prompt_mask), completion_mask), dim=1)

        # Flush left to reduce padding
        attention_mask, input_ids, completion_mask = flush_left(attention_mask, input_ids, completion_mask)

        # Truncate if necessary
        if self.max_length is not None:
            input_ids = input_ids[:, : self.max_length]
            attention_mask = attention_mask[:, : self.max_length]
            completion_mask = completion_mask[:, : self.max_length]

        # Create labels and mask padding tokens
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        # Build the output dictionary
        output = processed_prompts  # we take processed_prompts because it contains the images
        output["input_ids"] = input_ids
        output["attention_mask"] = attention_mask
        output["labels"] = labels
        return output


class DPOTrainer(BaseTrainer):
    """
    Trainer for Direct Preference Optimization (DPO) method.

    This class is a wrapper around the [`~transformers.Trainer`] class and inherits all of its attributes and methods.

    Example:

    ```python
    from datasets import load_dataset
    from trl import DPOTrainer

    dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")

    trainer = DPOTrainer(model="Qwen/Qwen2-0.5B-Instruct", train_dataset=dataset)
    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or a
              path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
              using `<ModelArchitecture>.from_pretrained` (where `<ModelArchitecture>` is derived from the model
              config) with the keyword arguments in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object.
            If you're training a model with an MoE architecture and want to include the load balancing/auxilliary loss
            as a part of the final loss, remember to set the `output_router_logits` config of the model to `True`.
        args ([`DPOConfig`], *optional*):
            Configuration for this trainer. If `None`, a default configuration is used.
        data_collator ([`~transformers.DataCollator`], *optional*):
            Function to use to form a batch from a list of elements of the processed `train_dataset` or `eval_dataset`.
            Will default to [`~trainer.dpo_trainer.DataCollatorForPreference`] if the model is a language model and
            [`~trainer.dpo_trainer.DataCollatorForVisionLanguageModeling`] if the model is a vision-language model.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. DPO supports both [language modeling](#language-modeling) type and
            [prompt-completion](#prompt-completion) type. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], [`~transformers.ProcessorMixin`], *optional*):
            Processing class used to process the data. If `None`, the processing class is loaded from the model's name
            with [`~transformers.AutoProcessor.from_pretrained`]. A padding token, `tokenizer.pad_token`, must be set.
            If the processing class has not set a padding token, `tokenizer.eos_token` will be used as the default.
        compute_loss_func (`Callable`, *optional*):
            A function that accepts the raw model outputs, labels, and the number of items in the entire accumulated
            batch (batch_size * gradient_accumulation_steps) and returns the loss. For example, see the default [loss
            function](https://github.com/huggingface/transformers/blob/052e652d6d53c2b26ffde87e039b723949a53493/src/transformers/trainer.py#L3618)
            used by [`Trainer`].
        compute_metrics (`Callable[[EvalPrediction], dict]`, *optional*):
            The function that will be used to compute metrics at evaluation. Must take a
            [`~transformers.EvalPrediction`] and return a dictionary string to metric values. When passing
            [`DPOConfig`] with `batch_eval_metrics` set to `True`, your `compute_metrics` function must take a boolean
            `compute_result` argument. This will be triggered after the last eval batch to signal that the function
            needs to calculate and return the global summary statistics rather than accumulating the batch-level
            statistics.
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
        model: Union[str, PreTrainedModel],
        args: Optional[DPOConfig] = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        processing_class: Optional[Union[PreTrainedTokenizerBase, ProcessorMixin]] = None,
        compute_loss_func: Optional[Callable] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], dict]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        optimizer_cls_and_kwargs: Optional[tuple[type[torch.optim.Optimizer], dict[str, Any]]] = None,
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        peft_config: Optional["PeftConfig"] = None,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = DPOConfig(f"{model_name}-DPO")

        # Model
        if isinstance(model, str):
            model = create_model_from_path(model, **args.model_init_kwargs or {})
        else:
            if args.model_init_kwargs is not None:
                logger.warning(
                    "You passed `model_init_kwargs` to the `DPOConfig`, but your model is already instantiated. "
                    "The `model_init_kwargs` will be ignored."
                )
        model_id = model.config._name_or_path

        disable_dropout_in_model(model)

        # Processing class
        if processing_class is None:
            processing_class = AutoProcessor.from_pretrained(model_id)

        # Handle pad token for processors or tokenizers
        if isinstance(processing_class, ProcessorMixin):
            tokenizer = processing_class.tokenizer
            self._is_vlm = True
        elif isinstance(processing_class, PreTrainedTokenizerBase):
            tokenizer = processing_class
            self._is_vlm = False
        else:
            raise TypeError("The `processing_class` must be either a `PreTrainedTokenizerBase` or a `ProcessorMixin`")

        if self._is_vlm and args.padding_free:
            raise ValueError(
                "Padding-free training is yet not supported for vision-language models. Please set "
                "`padding_free=False` in the `DPOConfig`."
            )

        if peft_config is not None or (is_peft_available() and isinstance(model, PeftModel)):
            model = prepare_peft_model(model, peft_config, args)

        # Data collator
        self.padding_free = args.padding_free
        use_flash_attention = model.config._attn_implementation in FLASH_ATTENTION_VARIANTS
        if self.padding_free:
            if data_collator is not None:
                raise ValueError("Passing a custom data collator is not supported when using padding-free.")
            if not use_flash_attention:
                logger.warning(
                    "Padding-free training is enabled, but the attention implementation is not set to a supported "
                    "flash attention variant. Padding-free training flattens batches into a single sequence, and only "
                    "the following implementations are known to reliably support this: "
                    f"{', '.join(sorted(FLASH_ATTENTION_VARIANTS))}. Using other implementations may lead to "
                    "unexpected behavior. To ensure compatibility, set `attn_implementation` in the model "
                    "configuration to one of these supported options or verify that your attention mechanism can "
                    "handle flattened sequences."
                )

            if args.per_device_train_batch_size == 1:
                logger.warning(
                    "You are using a per_device_train_batch_size of 1 with padding-free training. Using a batch size "
                    "of 1 anihilate the benefits of padding-free training. Please consider increasing the batch size "
                    "to at least 2."
                )

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
            data_collator = DataCollatorForVisionLanguageModeling(
                processor=processing_class,
                max_length=args.max_length,
                pad_to_multiple_of=args.pad_to_multiple_of,
            )

        # Training arguments
        self.loss_type = args.loss_type if isinstance(args.loss_type, list) else [args.loss_type]
        self.beta = args.beta

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

    def _prepare_dataset(
        self,
        dataset: Union[Dataset, IterableDataset],
        processing_class: Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin],
        args: DPOConfig,
        dataset_name: str,
    ) -> Union[Dataset, IterableDataset]:
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

                dataset = dataset.map(
                    add_eos,
                    fn_kwargs={"eos_token": processing_class.eos_token},
                    **map_kwargs,
                )

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
                    # Fix transformers inconsistency: for VLMs, apply_chat_template returns lists of lists
                    # even for single examples, while for LLMs it returns lists of ints.
                    prompt_ids = prompt_ids[0] if isinstance(prompt_ids[0], list) else prompt_ids
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

            # Truncate
            if args.max_prompt_length is not None:
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Truncating prompt in {dataset_name} dataset"
                # FIXME: this ignores truncation side
                dataset = truncate_dataset(
                    dataset, args.max_prompt_length, columns=["prompt_ids"], map_kwargs=map_kwargs
                )
            if args.max_completion_length is not None:
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Truncating completions in {dataset_name} dataset"
                dataset = truncate_dataset(
                    dataset, args.max_completion_length, columns=["chosen_ids", "rejected_ids"], map_kwargs=map_kwargs
                )
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
                self._signature_columns = ["prompt", "chosen", "rejectedimages"]
            else:
                self._signature_columns = [
                    "prompt_ids",
                    "chosen_ids",
                    "rejected_ids",
                    "ref_chosen_logps",
                    "ref_rejected_logps",
                ]

    def train(self, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None, **kwargs):
        if self.args.precompute_ref_log_probs:
            self.train_dataset = self._precompute_ref_logps(
                self.train_dataset, self.args.per_device_train_batch_size, "train"
            )
            if self.eval_dataset is not None:
                if isinstance(self.eval_dataset, dict):
                    self.eval_dataset = {
                        key: self._precompute_ref_logps(dataset, self.args.per_device_eval_batch_size, key)
                        for key, dataset in self.eval_dataset.items()
                    }
                else:
                    self.eval_dataset = self._precompute_ref_logps(
                        self.eval_dataset, self.args.per_device_eval_batch_size, "eval"
                    )
        return super().train()

    def _precompute_ref_logps(
        self, dataset: Union[Dataset, IterableDataset], batch_size: int, dataset_name: str
    ) -> None:
        def compute_ref_logps(examples, collator, model):
            examples = [dict(zip(examples.keys(), v)) for v in zip(*examples.values())]  # dict[list] to list[dict]
            inputs = collator(examples)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            # Truncate inputs
            if self.args.max_length is not None:
                if self.args.truncation_mode == "keep_start":
                    input_ids = inputs["input_ids"][:, : self.args.max_length]
                    attention_mask = inputs["attention_mask"][:, : self.args.max_length]
                    completion_mask = inputs["completion_mask"][:, : self.args.max_length]
                elif self.args.truncation_mode == "keep_end":
                    attention_mask, input_ids, completion_mask = flush_right(
                        inputs["attention_mask"], inputs["input_ids"], inputs["completion_mask"]
                    )
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

            return {
                "ref_chosen_logps": chosen_logps.tolist(),
                "ref_rejected_logps": rejected_logps.tolist(),
            }

        dataset = dataset.map(
            compute_ref_logps,
            batched=True,
            batch_size=batch_size,
            fn_kwargs={"collator": self.data_collator, "model": self.model},
            desc=f"Computing reference logps for {dataset_name} dataset",
        )
        return dataset

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

        # Truncate inputs
        if self.args.max_length is not None:
            if self.args.truncation_mode == "keep_start":
                input_ids = inputs["input_ids"][:, : self.args.max_length]
                attention_mask = inputs["attention_mask"][:, : self.args.max_length]
                completion_mask = inputs["completion_mask"][:, : self.args.max_length]
            elif self.args.truncation_mode == "keep_end":
                attention_mask, input_ids, completion_mask = flush_right(
                    inputs["attention_mask"], inputs["input_ids"], inputs["completion_mask"]
                )
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
        ref_chosen_logps, ref_rejected_logps = inputs["ref_chosen_logps"], inputs["ref_rejected_logps"]

        # Get the log ratios for the chosen and rejected responses
        chosen_logratios = chosen_logps - ref_chosen_logps
        rejected_logratios = rejected_logps - ref_rejected_logps

        loss = 0

        for loss_type in self.loss_type:
            if loss_type == "sigmoid":
                per_sequence_loss = -F.logsigmoid(self.beta * chosen_logratios - self.beta * rejected_logratios)

            elif loss_type == "hinge":
                per_sequence_loss = torch.relu(1 - (self.beta * chosen_logratios - self.beta * rejected_logratios))

            loss += per_sequence_loss.mean()

        # Compute entropy
        if not self.args.use_liger_kernel:  # liger doesn't return logits
            with torch.no_grad():
                per_token_entropy = entropy_from_logits(shift_logits)
                entropy = per_token_entropy[shift_completion_mask == 1].mean()
                entropy = self.accelerator.gather_for_metrics(entropy).mean().item()
            self._metrics[mode]["entropy"].append(entropy)

        if mode == "train":
            num_tokens_in_batch = self.accelerator.gather_for_metrics(inputs["attention_mask"].sum()).sum().item()
            self._total_train_tokens += num_tokens_in_batch
        self._metrics[mode]["num_tokens"] = [self._total_train_tokens]

        # Compute token accuracy if we have labels and if the model is not using Liger (no logits)
        if not self.args.use_liger_kernel:
            with torch.no_grad():
                # Get predictions (first half of the logits corresponding to the chosen responses)
                predictions = shift_logits[: len(shift_logits) // 2].argmax(dim=-1)
                chosen_mask = shift_completion_mask[: len(shift_completion_mask) // 2].bool()
                chosen_labels = shift_labels[: len(shift_labels) // 2]

                # Calculate accuracy only on non-padding tokens
                correct_predictions = (predictions == chosen_labels) & chosen_mask
                total_tokens = chosen_mask.sum()
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
