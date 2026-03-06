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
import json
import os
import textwrap
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from accelerate import PartialState
from accelerate.logging import get_logger
from accelerate.utils import is_peft_model, tqdm
from datasets import Dataset, IterableDataset, IterableDatasetDict
from datasets.fingerprint import Hasher
from packaging.version import Version
from torch.utils.data import DataLoader
from transformers import (
    AutoProcessor,
    DataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
)
from transformers.data.data_collator import DataCollatorMixin
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction
from transformers.utils import is_liger_kernel_available, is_peft_available

from ..data_utils import apply_chat_template, extract_prompt, is_conversational, prepare_multimodal_messages
from ..models import get_act_offloading_ctx_manager, prepare_deepspeed, prepare_fsdp
from ..models.utils import disable_gradient_checkpointing
from .base_trainer import _BaseTrainer
from .callbacks import SyncRefModelCallback
from .dpo_config import DPOConfig
from .utils import (
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
    use_adapter,
)


if is_peft_available():
    from peft import PeftConfig, PeftModel, get_peft_model


if is_liger_kernel_available():
    from liger_kernel.chunked_loss import LigerFusedLinearDPOLoss


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
    `"chosen_ids"` and `"rejected_ids"`. The collator returns a dictionary containing the following keys:
    - `"input_ids"`: Tensor of input IDs, padded to the maximum length of the batch. The first half of the batch
        corresponds to the `"chosen_ids"` and the second half to the `"rejected_ids"`.
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


@dataclass
class DataCollatorForVisionPreference(DataCollatorMixin):
    """
    Data collator for vision-preference tasks.

    Unlike text-only datasets, where the collator typically receives pre-tokenized inputs ready for batching,
    vision-language data processing involves converting images into pixel values. This conversion is disk-intensive,
    making upfront preprocessing of the entire dataset impractical. Therefore, this collator performs tokenization and
    image processing on-the-fly to efficiently prepare batches.

    Each input example should be a dictionary containing at least:
    - An `"images"` key holding a list of images, or an `"image"` key holding a single image.
    - Keys `"prompt"` `"chosen"` and `"rejected"` for the prompt and preference responses.

    The collator outputs a dictionary including:
    - `"input_ids"`: Tensor of token IDs.
    - `"attention_mask"`: Tensor indicating attention mask.
    - `"completion_mask"`: Tensor indicating which tokens correspond to completions.
    - `"pixel_values"`: Tensor representing image pixel values.

    Additional keys may be present depending on the processor, such as `"image_grid_thw"`.

    Args:
        processor ([`~transformers.ProcessorMixin`]):
            The processor used to tokenize text and process images. It must be a subclass of
            [`~transformers.ProcessorMixin`] and include a `tokenizer` with a defined `pad_token_id`.
        pad_to_multiple_of (`int` or `None`, optional, defaults to `None`):
            If set, the sequences will be padded to a multiple of this value.
        return_tensors (`str`, optional, defaults to `"pt"`):
            The tensor type to return. Currently, only `"pt"` (PyTorch tensors) is supported.

    Example:
    ```python
    >>> from trl.trainer.dpo_trainer import DataCollatorForVisionPreference
    >>> from transformers import AutoProcessor

    >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    >>> collator = DataCollatorForVisionPreference(processor)
    >>> examples = [
    ...     {
    ...         "images": [Image.open("image_0.png")],
    ...         "prompt": [{"role": "user", "content": "What is this?"}],
    ...         "chosen": [{"role": "assistant", "content": "This is a cat."}],
    ...         "rejected": [{"role": "assistant", "content": "This is a dog."}],
    ...     },
    ...     {
    ...         "images": [Image.open("image_1.png")],
    ...         "prompt": [{"role": "user", "content": "Describe this image."}],
    ...         "chosen": [{"role": "assistant", "content": "A beautiful landscape."}],
    ...         "rejected": [{"role": "assistant", "content": "An urban cityscape."}],
    ...     },
    ... ]
    >>> collator(examples)
    {'input_ids': tensor([[151644,   8948,    198,   2610,    525,    264,  10950,  17847,     13, 151645,    198, 151644,    872,    198, 151652, 151655, 151655, 151655, 151655, 151653,   3838,    374,    419,     30, 151645,    198, 151644,  77091,    198,   1986,    374,    264,   8251,     13, 151645,    198],
                          [151644,   8948,    198,   2610,    525,    264,  10950,  17847,     13, 151645,    198, 151644,    872,    198, 151652, 151655, 151655, 151655, 151655, 151653,  74785,    419,   2168,     13, 151645,    198, 151644,  77091,    198,     32,   6233,  18414,     13, 151645,    198, 151643],
                          [151644,   8948,    198,   2610,    525,    264,  10950,  17847,     13, 151645,    198, 151644,    872,    198, 151652, 151655, 151655, 151655, 151655, 151653,   3838,    374,    419,     30, 151645,    198, 151644,  77091,    198,   1986,    374,    264,   5562,     13, 151645,    198],
                          [151644,   8948,    198,   2610,    525,    264,  10950,  17847,     13, 151645,    198, 151644,    872,    198, 151652, 151655, 151655, 151655, 151655, 151653,  74785,    419,   2168,     13, 151645,    198, 151644,  77091,    198,   2082,  15662,   3283,  57518,     13, 151645,    198]]),
     'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                               [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                               [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                               [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]),
     'pixel_values': tensor([[-1.3251,  0.1347, -0.4784,  ...,  0.4537, -0.0156,  1.2358],
                             [ 0.5727,  0.4997, -0.9164,  ..., -0.5701,  0.7950, -0.7123],
                             [-0.0550, -0.8288,  1.0690,  ..., -0.1293, -0.1151,  1.6055],
                             ...,
                             [ 0.2953,  0.5581,  0.1785,  ..., -0.7123, -0.7977,  0.1693],
                             [-0.7558,  1.0398,  1.3464,  ..., -0.5417, -0.5417,  0.4395],
                             [ 0.8063,  0.6895,  0.4267,  ..., -0.4422,  1.3354,  0.1266]]),
     'image_grid_thw': tensor([[1, 4, 4],
                               [1, 4, 4],
                               [1, 4, 4],
                               [1, 4, 4]]),
     'completion_mask': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]])}
    ```
    """

    processor: ProcessorMixin
    pad_to_multiple_of: int | None = None
    return_tensors: str = "pt"

    def torch_call(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        if self.pad_to_multiple_of is not None:
            raise NotImplementedError(
                "Padding to a multiple of a value is not yet implemented for vision-language modeling and "
                "prompt-completion data."
            )
        if "image" in examples[0]:
            for example in examples:
                example["images"] = [example.pop("image")]
        images = [example["images"] for example in examples] * 2  # repeat for chosen and rejected
        # Transformers requires at least one image in the batch, otherwise it throws an error
        if all(img_list == [] for img_list in images):
            images = None
        if is_conversational(examples[0]):  # conversational case
            for example in examples:
                example["prompt"] = prepare_multimodal_messages(example["prompt"], images=example["images"])
                example["chosen"] = prepare_multimodal_messages(example["chosen"], images=[])
                example["rejected"] = prepare_multimodal_messages(example["rejected"], images=[])
            examples = [apply_chat_template(example, self.processor) for example in examples]

        prompts = [example["prompt"] for example in examples] * 2  # repeat for chosen and rejected
        chosens = [example["chosen"] for example in examples]
        rejecteds = [example["rejected"] for example in examples]

        processed_prompts = self.processor(
            images=images,
            text=prompts,
            padding=True,
            padding_side="left",
            return_tensors=self.return_tensors,
            add_special_tokens=False,  # to avoid adding the BOS, twice see https://huggingface.co/blog/qgallouedec/gotchas-in-tokenizer-behavior#7-chat-template-and-tokenization-dont-compose-due-to-special-tokens
        )
        processed_chosens = self.processor(
            text=chosens,
            padding=True,
            padding_side="right",
            return_tensors=self.return_tensors,
            add_special_tokens=False,  # to avoid adding the BOS, twice see https://huggingface.co/blog/qgallouedec/gotchas-in-tokenizer-behavior#7-chat-template-and-tokenization-dont-compose-due-to-special-tokens
        )
        processed_rejecteds = self.processor(
            text=rejecteds,
            padding=True,
            padding_side="right",
            return_tensors=self.return_tensors,
            add_special_tokens=False,  # to avoid adding the BOS, twice see https://huggingface.co/blog/qgallouedec/gotchas-in-tokenizer-behavior#7-chat-template-and-tokenization-dont-compose-due-to-special-tokens
        )

        # Concatenate prompts and completions
        prompt_ids, prompt_mask = processed_prompts["input_ids"], processed_prompts["attention_mask"]
        chosen_ids, chosen_mask = processed_chosens["input_ids"], processed_chosens["attention_mask"]
        rejected_ids, rejected_mask = processed_rejecteds["input_ids"], processed_rejecteds["attention_mask"]
        pad_token_id = self.processor.tokenizer.pad_token_id or self.processor.tokenizer.eos_token_id
        completion_ids = torch.cat(tuple(pad([chosen_ids, rejected_ids], padding_value=pad_token_id)))
        completion_mask = torch.cat(tuple(pad([chosen_mask, rejected_mask], padding_value=0)))
        input_ids = torch.cat((prompt_ids, completion_ids), dim=1)
        attention_mask = torch.cat((prompt_mask, completion_mask), dim=1)
        completion_mask = torch.cat((torch.zeros_like(prompt_mask), completion_mask), dim=1)
        if "token_type_ids" in processed_prompts:  # special case for Gemma
            prompt_token_type_ids = processed_prompts["token_type_ids"]
            chosen_type_ids = processed_chosens["token_type_ids"]
            rejected_type_ids = processed_rejecteds["token_type_ids"]
            completion_token_type_ids = torch.cat(tuple(pad([chosen_type_ids, rejected_type_ids], padding_value=0)))
            token_type_ids = torch.cat((prompt_token_type_ids, completion_token_type_ids), dim=1)

        # Flush left to reduce padding
        if "token_type_ids" in processed_prompts:
            attention_mask, input_ids, completion_mask, token_type_ids = flush_left(
                attention_mask, input_ids, completion_mask, token_type_ids
            )
        else:
            attention_mask, input_ids, completion_mask = flush_left(attention_mask, input_ids, completion_mask)

        # Build the output dictionary
        output = processed_prompts  # we take processed_prompts because it contains the images
        output["input_ids"] = input_ids
        output["attention_mask"] = attention_mask
        output["completion_mask"] = completion_mask
        if "token_type_ids" in processed_prompts:
            output["token_type_ids"] = token_type_ids
        return output


class DPOTrainer(_BaseTrainer):
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
        model (`str` or [`~transformers.PreTrainedModel`] or [`~peft.PeftModel`]):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or a
              path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
              using `<ModelArchitecture>.from_pretrained` (where `<ModelArchitecture>` is derived from the model
              config) with the keyword arguments in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
            - A [`~peft.PeftModel`] object. Only causal language models are supported.
        ref_model (`PreTrainedModel`, *optional*):
            Reference model used to compute the reference log probabilities.

            - If provided, this model is used directly as the reference policy.
            - If `None`, the trainer will automatically use the initial policy corresponding to `model`, i.e. the model
              state before DPO training starts.
        args ([`DPOConfig`], *optional*):
            Configuration for this trainer. If `None`, a default configuration is used.
        data_collator ([`~transformers.DataCollator`], *optional*):
            Function to use to form a batch from a list of elements of the processed `train_dataset` or `eval_dataset`.
            Will default to [`~trainer.dpo_trainer.DataCollatorForPreference`] if the model is a language model and
            [`~trainer.dpo_trainer.DataCollatorForVisionPreference`] if the model is a vision-language model.
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
        compute_metrics (`Callable[[EvalPrediction], dict]`, *optional*):
            The function that will be used to compute metrics at evaluation. Must take a
            [`~transformers.EvalPrediction`] and return a dictionary string to metric values. When passing
            [`SFTConfig`] with `batch_eval_metrics` set to `True`, your `compute_metrics` function must take a boolean
            `compute_result` argument. This will be triggered after the last eval batch to signal that the function
            needs to calculate and return the global summary statistics rather than accumulating the batch-level
            statistics.
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
        model: "str | PreTrainedModel | PeftModel",
        ref_model: PreTrainedModel | None = None,
        args: DPOConfig | None = None,
        data_collator: DataCollator | None = None,
        train_dataset: Dataset | IterableDataset | None = None,
        eval_dataset: Dataset | IterableDataset | dict[str, Dataset | IterableDataset] | None = None,
        processing_class: PreTrainedTokenizerBase | ProcessorMixin | None = None,
        compute_metrics: Callable[[EvalPrediction], dict] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None] = (None, None),
        peft_config: "PeftConfig | None" = None,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else get_config_model_id(model.config)
            model_name = model_name.split("/")[-1]
            args = DPOConfig(f"{model_name}-DPO")

        if train_dataset is None:
            raise ValueError("`train_dataset` is required")
        elif isinstance(train_dataset, IterableDataset):
            # IterableDataset requires dispatch_batches=False because Accelerate's dispatch mode may try to concatenate
            # batches from multiple processes, leading to mismatch errors.
            if args.accelerator_config.dispatch_batches is True:
                logger.warning(
                    "You are using an `IterableDataset` for training with `dispatch_batches=True`. `dispatch_batches` "
                    "is forced to `False` when using an `IterableDataset`. To remove this warning, unset "
                    "`dispatch_batches` in `DPOConfig` or set it to `False`."
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
                    "You passed `model_init_kwargs` to the `DPOConfig`, but your model is already instantiated. "
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

        if is_peft_available() and is_peft_model(model) and peft_config is not None:
            raise ValueError(
                "You passed a `PeftModel` instance together with a `peft_config` to the trainer. Please first merge "
                "and unload the existing adapter, save the resulting base model, and then pass that base model along "
                "with the new `peft_config` to the trainer."
            )
        if is_peft_available() and is_peft_model(model) and ref_model is None:
            # If the model is a PEFT model with a pretrained adapter, we need to create a "ref" adapter that is a copy
            # of the "default" adapter, so that we can use it as the reference model during DPO training.
            model.add_adapter("ref", model.peft_config["default"])
            for name, param in model.named_parameters():
                if ".default." in name:
                    ref_name = name.replace(".default.", ".ref.")
                    ref_param = model.get_parameter(ref_name)
                    ref_param.data.copy_(param.data)

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
        self.padding_free = args.padding_free
        if self.padding_free:
            logger.warning(
                "`padding_free=True` is temporarily unavailable after a refactor and is currently disabled. Falling "
                "back to standard padding (`padding_free=False`). This feature is planned to return in a future "
                "update; for now, please set `padding_free=False` explicitly."
            )
            self.padding_free = False
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
            data_collator = DataCollatorForVisionPreference(
                processor=processing_class,
                pad_to_multiple_of=args.pad_to_multiple_of,
            )

        # Training arguments
        self.beta = args.beta
        self.precompute_ref_logps = args.precompute_ref_log_probs
        self.loss_types = args.loss_type  # args.loss_type is already a list
        self.loss_weights = args.loss_weights or [1.0] * len(self.loss_types)
        self.ld_alpha = args.ld_alpha
        self.f_divergence_type = args.f_divergence_type
        self.f_alpha_divergence_coef = args.f_alpha_divergence_coef
        self.label_smoothing = args.label_smoothing
        self.use_weighting = args.use_weighting
        if self.use_weighting and any(loss_type in {"aot", "aot_unpaired"} for loss_type in self.loss_types):
            raise NotImplementedError(
                "WPO-style weighting is not implemented for 'aot' or 'aot_unpaired' because those losses sort "
                "samples, which would misalign per-pair weights."
            )
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
        self.use_liger_kernel = args.use_liger_kernel
        if args.use_liger_kernel:
            if not is_liger_kernel_available():
                raise ImportError(
                    "You set `use_liger_kernel=True` but the liger kernel is not available. "
                    "Please install liger-kernel first: `pip install liger-kernel`"
                )
            if len(self.loss_types) != 1:
                raise NotImplementedError(
                    "Multiple loss types are not yet supported when using Liger kernel. If you need this feature, "
                    "please open a feature request at https://github.com/huggingface/trl/issues."
                )
            self.liger_loss_fn = LigerFusedLinearDPOLoss(beta=args.beta, loss_type=self.loss_types[0])
            if compute_metrics is not None:
                raise ValueError(
                    "compute_metrics is not supported with the Liger kernel. compute_metrics requires to be able to "
                    "recover the logits from the forward pass, but Liger kernel does not materialize logits."
                )
            if self.precompute_ref_logps:
                raise ValueError(
                    "Liger DPO loss does not support precomputing reference log probabilities. Either disable "
                    "`precompute_ref_log_probs` or set `use_liger_kernel` to False."
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
            if is_peft_model(self.model):
                # If PEFT is used, the reference model is not needed since the adapter can be disabled to revert to the
                # initial model.
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
            if self.ref_model is None:
                raise NotImplementedError(
                    "You passed `sync_ref_model=True` while using a PEFT model, which is currently not supported. "
                    "With PEFT, DPOTrainer does not keep a separate reference model in memory; instead, it recovers "
                    "reference behavior by temporarily disabling the adapter. As a result, there is no standalone "
                    "`ref_model` instance to synchronize. Use `sync_ref_model=False`, or opt for full fine-tuning if "
                    "you need a synced reference model. If you need `sync_ref_model` to work with PEFT, please open a "
                    "feature request at https://github.com/huggingface/trl/issues."
                )
            if args.precompute_ref_log_probs:
                raise ValueError(
                    "You cannot use `sync_ref_model=True` together with `precompute_ref_log_probs=True`. "
                    "`precompute_ref_log_probs=True` assumes a fixed reference model, but with `sync_ref_model=True` "
                    "the reference model is periodically updated during training, making any precomputed reference "
                    "log-probs stale. Set `precompute_ref_log_probs=False` or disable `sync_ref_model`."
                )
            self.add_callback(SyncRefModelCallback(ref_model=self.ref_model, accelerator=self.accelerator))

        if args.precompute_ref_log_probs:
            if isinstance(self.train_dataset, IterableDataset) or isinstance(
                self.eval_dataset, (IterableDataset, IterableDatasetDict)
            ):
                raise ValueError(
                    "`precompute_ref_log_probs=True` is not supported with IterableDataset. Please use a map-style "
                    "Dataset or set `precompute_ref_log_probs=False`."
                )

            batch_size = self.args.precompute_ref_batch_size or self.args.per_device_train_batch_size
            self.train_dataset = self._precompute_ref_logps(self.train_dataset, "train", batch_size)
            if self.eval_dataset is not None:
                batch_size = self.args.precompute_ref_batch_size or self.args.per_device_eval_batch_size
                if isinstance(self.eval_dataset, dict):
                    self.eval_dataset = {
                        name: self._precompute_ref_logps(dataset, name, batch_size)
                        for name, dataset in self.eval_dataset.items()
                    }
                else:
                    self.eval_dataset = self._precompute_ref_logps(self.eval_dataset, "eval", batch_size)

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

                eos_token = processing_class.tokenizer.eos_token if self._is_vlm else processing_class.eos_token
                dataset = dataset.map(add_eos, fn_kwargs={"eos_token": eos_token}, **map_kwargs)

            # Tokenize the dataset
            if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                map_kwargs["desc"] = f"Tokenizing {dataset_name} dataset"

            def tokenize_fn(example, processing_class):
                tools = example.get("tools")
                tools = json.loads(tools) if isinstance(tools, str) else tools
                output = {}
                if is_conversational(example):
                    if self._is_vlm:
                        prompt = prepare_multimodal_messages(example["prompt"], images=[])
                        chosen = prepare_multimodal_messages(example["chosen"], images=[])
                        rejected = prepare_multimodal_messages(example["rejected"], images=[])
                    else:
                        prompt = example["prompt"]
                        chosen = example["chosen"]
                        rejected = example["rejected"]
                    prompt_ids = processing_class.apply_chat_template(
                        prompt,
                        tools=tools,
                        add_generation_prompt=True,
                        tokenize=True,
                        return_dict=False,
                        **example.get("chat_template_kwargs", {}),
                    )
                    prompt_chosen_processed = processing_class.apply_chat_template(
                        prompt + chosen,
                        tools=tools,
                        tokenize=True,
                        return_dict=True,
                        **example.get("chat_template_kwargs", {}),
                    )
                    prompt_rejected_processed = processing_class.apply_chat_template(
                        prompt + rejected,
                        tools=tools,
                        tokenize=True,
                        return_dict=True,
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
                    # Fix transformers inconsistency: for VLMs, processing_class returns lists of lists
                    # even for single examples, while for LLMs it returns lists of ints.
                    prompt_ids = prompt_ids[0] if isinstance(prompt_ids[0], list) else prompt_ids
                    prompt_chosen_ids = (
                        prompt_chosen_ids[0] if isinstance(prompt_chosen_ids[0], list) else prompt_chosen_ids
                    )
                    prompt_rejected_ids = (
                        prompt_rejected_ids[0] if isinstance(prompt_rejected_ids[0], list) else prompt_rejected_ids
                    )

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

        return dataset

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs (usually, "input_ids"
        # and "attention_mask").
        if self._signature_columns is None:
            if self._is_vision_dataset:
                self._signature_columns = [
                    "prompt",
                    "chosen",
                    "rejected",
                    "image",
                    "images",
                    "tools",
                    "chat_template_kwargs",
                ]
            else:
                self._signature_columns = [
                    "prompt_ids",
                    "chosen_ids",
                    "rejected_ids",
                    "ref_chosen_logps",
                    "ref_rejected_logps",
                ]

    def _precompute_ref_logps(self, dataset: Dataset, name: str, batch_size: int) -> Dataset:
        model_hash = hash_module(self.ref_model or self.model)
        fingerprint = Hasher.hash((dataset._fingerprint, model_hash))
        cache_file = dataset._get_cache_file_path(fingerprint).removesuffix(".arrow") + ".npz"
        if os.path.exists(cache_file):
            loaded = np.load(cache_file)
            ref_chosen_logps = loaded["ref_chosen_logps"]
            ref_rejected_logps = loaded["ref_rejected_logps"]
        else:
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
                shuffle=False,
            )
            data_loader = self.accelerator.prepare(dataloader)
            ref_chosen_logps = []
            ref_rejected_logps = []
            for padded_batch in tqdm(iterable=data_loader, desc=f"Computing reference log probs for {name} dataset"):
                ref_chosen_logp, ref_rejected_logp = self.compute_ref_log_probs(padded_batch)
                ref_chosen_logp, ref_rejected_logp = self.accelerator.gather_for_metrics(
                    (ref_chosen_logp, ref_rejected_logp)
                )
                ref_chosen_logps.append(ref_chosen_logp.cpu())
                ref_rejected_logps.append(ref_rejected_logp.cpu())

            # Save the reference log probabilities to cache. We need .float() because bf16 is not supported by numpy
            ref_chosen_logps = torch.cat(ref_chosen_logps).float().numpy()
            ref_rejected_logps = torch.cat(ref_rejected_logps).float().numpy()
            if self.accelerator.is_main_process:
                np.savez_compressed(
                    cache_file, ref_chosen_logps=ref_chosen_logps, ref_rejected_logps=ref_rejected_logps
                )
            self.accelerator.wait_for_everyone()

        dataset = dataset.add_column(name="ref_chosen_logps", column=ref_chosen_logps)
        dataset = dataset.add_column(name="ref_rejected_logps", column=ref_rejected_logps, new_fingerprint=fingerprint)

        return dataset

    def _truncate_inputs(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, completion_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.args.max_length is None:
            return input_ids, attention_mask, completion_mask

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

        return input_ids, attention_mask, completion_mask

    def compute_ref_log_probs(self, inputs):
        """Computes reference log probabilities for a single padded batch."""
        device = self.accelerator.device

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        completion_mask = inputs["completion_mask"]
        input_ids, attention_mask, completion_mask = self._truncate_inputs(input_ids, attention_mask, completion_mask)

        shift_labels = input_ids[..., 1:].contiguous()
        shift_completion_mask = completion_mask[..., 1:].contiguous()

        model_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask, "use_cache": False}
        for key in ("pixel_values", "pixel_attention_mask", "image_grid_thw", "image_sizes", "token_type_ids"):
            if key in inputs:
                model_kwargs[key] = inputs[key]

        with torch.no_grad(), disable_gradient_checkpointing(self.model, self.args.gradient_checkpointing_kwargs):
            if is_peft_model(self.model) and self.ref_model is None:
                model = self.accelerator.unwrap_model(self.model)
                with use_adapter(model, adapter_name="ref" if "ref" in model.peft_config else None):
                    ref_outputs = self.model(**model_kwargs)
            else:
                ref_outputs = self.ref_model(**model_kwargs)

        ref_shift_logits = ref_outputs.logits[..., :-1, :].contiguous()
        ref_per_token_logps = selective_log_softmax(ref_shift_logits, shift_labels)
        ref_per_token_logps[shift_completion_mask == 0] = 0.0

        if self.ld_alpha is None:
            ref_logps = ref_per_token_logps.sum(dim=1)
        else:
            comp_pos = shift_completion_mask.cumsum(dim=1)
            comp_lens = shift_completion_mask.sum(dim=1).long()
            chosen_lens, rejected_lens = comp_lens.chunk(2, dim=0)
            shared_lens = torch.minimum(chosen_lens, rejected_lens)
            shared_lens = torch.cat([shared_lens, shared_lens], dim=0).to(device)
            shared_mask = (comp_pos > 0) & (comp_pos <= shared_lens.unsqueeze(1))
            tail_mask = comp_pos > shared_lens.unsqueeze(1)
            shared_logps = (ref_per_token_logps * shared_mask).sum(dim=1)
            tail_logps = (ref_per_token_logps * tail_mask).sum(dim=1)
            ref_logps = shared_logps + self.ld_alpha * tail_logps

        ref_chosen_logps, ref_rejected_logps = ref_logps.chunk(2, dim=0)
        return ref_chosen_logps, ref_rejected_logps

    def _compute_loss_liger(self, model, inputs, return_outputs):
        if return_outputs:
            raise RuntimeError(
                "return_outputs=True is not supported with the Liger DPO loss. The Liger loss computes the loss "
                "without materializing logits, so outputs cannot be returned."
            )

        mode = "train" if self.model.training else "eval"

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        completion_mask = inputs["completion_mask"]
        input_ids, attention_mask, completion_mask = self._truncate_inputs(input_ids, attention_mask, completion_mask)

        decoder = model.get_decoder()
        outputs = decoder(input_ids, attention_mask=attention_mask, use_cache=False)
        hidden_states = outputs.last_hidden_state[:, :-1].contiguous()
        lm_head = model.get_output_embeddings()
        weight = lm_head.weight
        bias = lm_head.bias

        if is_peft_model(model):
            raise NotImplementedError("Liger DPO loss is not implemented for PEFT models.")
        else:
            with torch.no_grad(), disable_gradient_checkpointing(self.model, self.args.gradient_checkpointing_kwargs):
                ref_decoder = self.ref_model.get_decoder()
                ref_outputs = ref_decoder(input_ids, attention_mask=attention_mask, use_cache=False)
                ref_lm_head = self.ref_model.get_output_embeddings()
                ref_hidden_states = ref_outputs.last_hidden_state[:, :-1].contiguous()
                ref_weight = ref_lm_head.weight
                ref_bias = ref_lm_head.bias

        shift_completion_mask = completion_mask[:, 1:].contiguous()
        labels = input_ids[:, 1:].clone()
        labels[shift_completion_mask == 0] = -100

        loss, metrics = self.liger_loss_fn(
            weight, hidden_states, labels, bias, ref_hidden_states, ref_weight, ref_bias
        )

        (
            chosen_logps,
            rejected_logps,
            chosen_logits_mean,
            rejected_logits_mean,
            nll_loss,
            chosen_rewards,
            rejected_rewards,
        ) = metrics

        if mode == "train":
            num_tokens_in_batch = self.accelerator.gather_for_metrics(inputs["attention_mask"].sum()).sum().item()
            self._total_train_tokens += num_tokens_in_batch
        self._metrics[mode]["num_tokens"] = [self._total_train_tokens]

        avg_chosen_logits = self.accelerator.gather_for_metrics(chosen_logits_mean).mean().item()
        avg_rejected_logits = self.accelerator.gather_for_metrics(rejected_logits_mean).mean().item()
        self._metrics[mode]["logits/chosen"].append(avg_chosen_logits)
        self._metrics[mode]["logits/rejected"].append(avg_rejected_logits)

        agg_chosen_rewards = self.accelerator.gather(chosen_rewards)
        agg_rejected_rewards = self.accelerator.gather(rejected_rewards)
        self._metrics[mode]["rewards/chosen"].append(agg_chosen_rewards.mean().item())
        self._metrics[mode]["rewards/rejected"].append(agg_rejected_rewards.mean().item())

        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        agg_reward_accuracies = self.accelerator.gather(reward_accuracies)
        self._metrics[mode]["rewards/accuracies"].append(agg_reward_accuracies.mean().item())

        margins = chosen_rewards - rejected_rewards
        agg_margins = self.accelerator.gather(margins)
        self._metrics[mode]["rewards/margins"].append(agg_margins.mean().item())

        self._metrics[mode]["logps/chosen"].append(self.accelerator.gather(chosen_logps).mean().item())
        self._metrics[mode]["logps/rejected"].append(self.accelerator.gather(rejected_logps).mean().item())

        return loss

    def _compute_loss(self, model, inputs, return_outputs):
        mode = "train" if self.model.training else "eval"
        device = self.accelerator.device

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        completion_mask = inputs["completion_mask"]
        input_ids, attention_mask, completion_mask = self._truncate_inputs(input_ids, attention_mask, completion_mask)

        model_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask, "use_cache": False}
        for key in ("pixel_values", "pixel_attention_mask", "image_grid_thw", "image_sizes", "token_type_ids"):
            if key in inputs:
                model_kwargs[key] = inputs[key]

        outputs = model(**model_kwargs)
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        shift_completion_mask = completion_mask[..., 1:].contiguous()
        per_token_logps = selective_log_softmax(shift_logits, shift_labels)
        per_token_logps[shift_completion_mask == 0] = 0.0  # mask out non-completion tokens
        if self.ld_alpha is None:
            logps = per_token_logps.sum(dim=1)  # sum over sequence length
        else:
            comp_pos = shift_completion_mask.cumsum(dim=1)
            comp_lens = shift_completion_mask.sum(dim=1).long()
            chosen_lens, rejected_lens = comp_lens.chunk(2, dim=0)
            shared_lens = torch.minimum(chosen_lens, rejected_lens)
            shared_lens = torch.cat([shared_lens, shared_lens], dim=0).to(device)
            shared_mask = (comp_pos > 0) & (comp_pos <= shared_lens.unsqueeze(1))  # shared: 1 <= pos <= shared_len
            tail_mask = comp_pos > shared_lens.unsqueeze(1)  # tail: pos > shared_len
            shared_logps = (per_token_logps * shared_mask).sum(dim=1)
            tail_logps = (per_token_logps * tail_mask).sum(dim=1)
            logps = shared_logps + self.ld_alpha * tail_logps
        chosen_logps, rejected_logps = logps.chunk(2, dim=0)  # batch is [chosen, rejected]

        if self.precompute_ref_logps:
            ref_chosen_logps, ref_rejected_logps = inputs["ref_chosen_logps"], inputs["ref_rejected_logps"]
        else:
            # When gradient checkpointing is enabled with use_reentrant=True (default), calling the model inside a
            # torch.no_grad() block triggers a harmless PyTorch warning ("None of the inputs have requires_grad=True").
            # Temporarily disable checkpointing to avoid this warning during inference.
            with torch.no_grad(), disable_gradient_checkpointing(self.model, self.args.gradient_checkpointing_kwargs):
                if is_peft_model(model) and self.ref_model is None:
                    # When training a PEFT adapter, how we obtain the reference depends on the setup:
                    # - New adapter: disabling adapters yields the base model.
                    # - Re-training an existing adapter: an initial copy is loaded under the name "ref".
                    model = self.accelerator.unwrap_model(model)
                    with use_adapter(model, adapter_name="ref" if "ref" in model.peft_config else None):
                        ref_outputs = self.model(**model_kwargs)
                else:
                    ref_outputs = self.ref_model(**model_kwargs)

            ref_shift_logits = ref_outputs.logits[..., :-1, :].contiguous()
            ref_per_token_logps = selective_log_softmax(ref_shift_logits, shift_labels)
            ref_per_token_logps[shift_completion_mask == 0] = 0.0  # mask out non-completion tokens
            if self.ld_alpha is None:
                ref_logps = ref_per_token_logps.sum(dim=1)  # sum over sequence length
            else:
                # reuse comp_pos/shared_mask/tail_mask computed above (they depend only on completion_mask)
                ref_shared_logps = (ref_per_token_logps * shared_mask).sum(dim=1)
                ref_tail_logps = (ref_per_token_logps * tail_mask).sum(dim=1)
                ref_logps = ref_shared_logps + self.ld_alpha * ref_tail_logps
            ref_chosen_logps, ref_rejected_logps = ref_logps.chunk(2, dim=0)  # batch is [chosen, rejected]

        # Get the log ratios for the chosen and rejected responses
        chosen_logratios = chosen_logps - ref_chosen_logps
        rejected_logratios = rejected_logps - ref_rejected_logps

        if self.f_divergence_type == "reverse_kl":  # standard DPO
            chosen_scores = chosen_logratios
            rejected_scores = rejected_logratios
        elif self.f_divergence_type == "forward_kl":
            # f'(t) = 1 - 1/t  -> drop constant -> -exp(-logratio)
            chosen_scores = -torch.exp(-chosen_logratios)
            rejected_scores = -torch.exp(-rejected_logratios)
        elif self.f_divergence_type == "js_divergence":
            # f'(t) = log(2t/(t+1)) -> drop log 2
            chosen_scores = F.logsigmoid(chosen_logratios)
            rejected_scores = F.logsigmoid(rejected_logratios)
        elif self.f_divergence_type == "alpha_divergence":
            # alpha-divergence: f'(t) = (t^(α-1) - 1)/(α-1)
            if abs(self.f_alpha_divergence_coef - 1.0) < 1e-6:  # limit case f'(t) -> log(t), fall back to reverse_kl
                chosen_scores = chosen_logratios
                rejected_scores = rejected_logratios
            else:
                coef = 1.0 / (self.f_alpha_divergence_coef - 1.0)
                t_chosen = (self.f_alpha_divergence_coef - 1.0) * chosen_logratios
                t_rejected = (self.f_alpha_divergence_coef - 1.0) * rejected_logratios
                dtype = t_chosen.dtype
                # Clamp max so exp(.) stays representable after casting back
                clamp_max = {torch.float16: 11.0, torch.bfloat16: 80.0, torch.float32: 80.0}[dtype]
                t_chosen_float = torch.clamp(t_chosen.float(), max=clamp_max)
                t_rejected_float = torch.clamp(t_rejected.float(), max=clamp_max)
                chosen_scores = torch.exp(t_chosen_float).to(dtype) * coef
                rejected_scores = torch.exp(t_rejected_float).to(dtype) * coef
        else:
            raise ValueError(f"Unknown f_divergence_type: {self.f_divergence_type}")

        delta_score = chosen_scores - rejected_scores

        loss = 0.0
        for loss_type, loss_weight in zip(self.loss_types, self.loss_weights, strict=True):
            if loss_type == "sigmoid":
                per_sequence_loss = -F.logsigmoid(self.beta * delta_score)

            elif loss_type == "hinge":
                per_sequence_loss = torch.relu(1 - self.beta * delta_score)

            elif loss_type == "ipo":
                # IPO uses sequence-level log-prob differences; in code these are token-summed over the completion,
                # which makes the squared loss scale with completion length. We therefore normalize by the number of
                # completion tokens (average per token) to make β/loss comparable across variable lengths. This length
                # normalization is not explicitly discussed in the IPO paper; we confirmed this choice with the IPO
                # authors, and the results reported in the paper correspond to this normalized form.
                chosen_mask, rejected_mask = completion_mask.chunk(2, dim=0)
                chosen_avg_score = chosen_scores / chosen_mask.sum(dim=1).clamp(min=1.0)
                rejected_avg_score = rejected_scores / rejected_mask.sum(dim=1).clamp(min=1.0)
                ipo_delta = chosen_avg_score - rejected_avg_score
                # (Eq. 17) of the paper where beta is the regularization parameter for the IPO loss, denoted by τ.
                per_sequence_loss = (ipo_delta - 1 / (2 * self.beta)) ** 2

            elif loss_type == "exo_pair":
                # Implements EXO-pref from the paper https://huggingface.co/papers/2402.00856, (Eq. 16)
                # Minimize KL(p_fθ || p_rh) for K=2; p_fθ = softmax(βπ * (log πθ − log π_ref)) over {chosen, rejected}
                # p_rh = [(1−ε), ε]; expanded KL gives the weighted logsigmoid form below
                epsilon = torch.tensor(self.label_smoothing, device=device)
                qw = torch.sigmoid(self.beta * delta_score)
                log_qw = F.logsigmoid(self.beta * delta_score)
                log_pw = torch.log1p(-epsilon)
                ql = torch.sigmoid(-self.beta * delta_score)
                log_ql = F.logsigmoid(-self.beta * delta_score)
                log_pl = torch.log(epsilon)
                per_sequence_loss = qw * (log_qw - log_pw) + ql * (log_ql - log_pl)

            elif loss_type == "nca_pair":
                chosen_rewards = self.beta * chosen_scores
                rejected_rewards = self.beta * rejected_scores
                per_sequence_loss = (
                    -F.logsigmoid(chosen_rewards)
                    - 0.5 * F.logsigmoid(-chosen_rewards)
                    - 0.5 * F.logsigmoid(-rejected_rewards)
                )

            elif loss_type == "robust":
                clean_loss_term = -(1 - self.label_smoothing) * F.logsigmoid(self.beta * delta_score)
                flipped_loss_term = -self.label_smoothing * F.logsigmoid(-self.beta * delta_score)
                per_sequence_loss = (clean_loss_term - flipped_loss_term) / (1 - 2 * self.label_smoothing)

            elif loss_type == "bco_pair":
                chosen_rewards = self.beta * chosen_scores
                rejected_rewards = self.beta * rejected_scores
                per_sequence_loss = -F.logsigmoid(chosen_rewards) - F.logsigmoid(-rejected_rewards)

            elif loss_type == "sppo_hard":
                # In the paper (https://huggingface.co/papers/2405.00675), SPPO employs a soft probability approach,
                # estimated using the PairRM score. The probability calculation is conducted outside of the trainer
                # class. The version described here is the hard probability version, where P in Equation (4.7) of
                # Algorithm 1 is set to 1 for the winner and 0 for the loser.
                winner_margin_error = (chosen_scores - 0.5 / self.beta) ** 2
                loser_margin_error = (rejected_scores + 0.5 / self.beta) ** 2
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
                losses_chosen = 1 - torch.sigmoid(self.beta * chosen_logratios)
                losses_rejected = torch.sigmoid(self.beta * rejected_logratios)
                per_sequence_loss = losses_chosen + losses_rejected

            elif loss_type == "apo_down":
                # Eqn (8) of the APO paper (https://huggingface.co/papers/2408.06266)
                # Use this loss when you believe the chosen outputs are worse than your model's default output.
                # Decrease chosen likelihood and decrease rejected likelihood more
                losses_chosen = torch.sigmoid(self.beta * chosen_logratios)
                losses_rejected = 1 - torch.sigmoid(self.beta * delta_score)
                per_sequence_loss = losses_chosen + losses_rejected

            elif loss_type == "discopop":
                # Eqn (5) of the DiscoPOP paper (https://huggingface.co/papers/2406.08414)
                logits = delta_score * self.beta
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

            if self.use_weighting:
                # Eq (2) of the WPO paper: https://huggingface.co/papers/2406.11827
                completion_lengths = shift_completion_mask.sum(dim=1).clamp_min(1)
                with torch.no_grad():
                    lse1 = torch.logsumexp(shift_logits, dim=-1)
                    lse2 = torch.logsumexp(2.0 * shift_logits, dim=-1)
                    log_denom = lse2 - 2.0 * lse1
                    aligned_logps = (per_token_logps - log_denom) * shift_completion_mask
                mean_logps = aligned_logps.sum(dim=1) / completion_lengths
                weights = torch.exp(mean_logps)
                chosen_weights, rejected_weights = weights.chunk(2, dim=0)
                per_sequence_loss *= chosen_weights * rejected_weights

            loss += per_sequence_loss.mean() * loss_weight

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

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if self.use_liger_kernel:
            return self._compute_loss_liger(model, inputs, return_outputs)
        else:
            return self._compute_loss(model, inputs, return_outputs)

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

    # During eval, Trainer calls prediction_step. If no labels are present in the inputs, it only runs forward and
    # returns logits. We override prediction_step to force compute_loss, because this trainer doesn't involve labels.
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys: list[str] | None = None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad(), self.compute_loss_context_manager():
            if prediction_loss_only:
                loss = self.compute_loss(model, inputs, return_outputs=False)  # logits aren't materialized with liger
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
