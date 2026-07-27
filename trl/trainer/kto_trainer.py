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

import torch
import torch.nn.functional as F
import transformers
from accelerate import PartialState
from accelerate.logging import get_logger
from accelerate.utils import broadcast_object_list, is_peft_model, tqdm
from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, concatenate_datasets
from datasets.fingerprint import Hasher
from packaging.version import Version
from torch.utils.data import DataLoader, Sampler, SequentialSampler
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    DataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainerCallback,
)
from transformers.data.data_collator import DataCollatorMixin
from transformers.trainer_utils import EvalPrediction, has_length
from transformers.utils import is_peft_available

from ..data_utils import (
    _tokenize,
    apply_chat_template,
    extract_prompt,
    is_conversational,
    prepare_multimodal_messages,
    unpair_preference_dataset,
)
from ..import_utils import is_liger_kernel_available
from ..models import get_act_offloading_ctx_manager, prepare_deepspeed, prepare_fsdp
from ..models.utils import _ForwardRedirection, disable_gradient_checkpointing
from .base_trainer import _BaseTrainer
from .callbacks import SyncRefModelCallback
from .kto_config import KTOConfig
from .utils import (
    create_model_from_path,
    disable_dropout_in_model,
    entropy_from_logits,
    flush_left,
    get_config_model_id,
    hash_module,
    maybe_gather_lm_head_ctx,
    pad,
    selective_log_softmax,
    use_adapter,
)


if is_liger_kernel_available():
    from liger_kernel.chunked_loss import LigerFusedLinearKTOLoss


if is_peft_available():
    import peft
    from peft import LoraConfig, PeftConfig, PeftModel, PromptLearningConfig, get_peft_model
    from peft.tuners.tuners_utils import BaseTunerLayer


logger = get_logger(__name__)


@dataclass
class DataCollatorForUnpairedPreference(DataCollatorMixin):
    """
    Data collator used for unpaired preference data. Inputs are dynamically padded to the maximum length of a batch.

    This collator expects each example in the input list to be a dictionary containing the `"prompt_ids"` and
    `"completion_ids"` keys (and optionally a `"KL_completion_ids"` key). The collator returns a dictionary containing
    the following keys:
    - `"input_ids"`: Tensor of input IDs, padded to the maximum length of the batch.
    - `"attention_mask"`: Tensor of attention mask, padded to the maximum length of the batch.
    - `"completion_mask"`: Tensor indicating the positions of the completion tokens, padded to the maximum length of
        the batch.

    When `"KL_completion_ids"` is present, the same three tensors are returned for the (mismatched) KL sequence under
    the `"KL_input_ids"`, `"KL_attention_mask"` and `"KL_completion_mask"` keys.

    The returned dictionary also contains a `"label"` key with a tensor of labels. Optionally, the examples can contain
    `"ref_logps"` and `"ref_KL_logps"` keys, in which case the returned dictionary will also contain these keys with
    the corresponding tensors.

    Args:
        pad_token_id (`int`):
            Token ID to use for padding.
        max_length (`int`, *optional*):
            Maximum length of the sequences in the batch. Sequences longer than `max_length` are truncated before
            padding, which avoids allocating oversized tensors for batches containing very long sequences.
        pad_to_multiple_of (`int`, *optional*):
            If set, the sequences will be padded to a multiple of this value.
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            Type of Tensor to return. Only `"pt"` is currently supported.

    Examples:
    ```python
    >>> from trl.trainer.kto_trainer import DataCollatorForUnpairedPreference

    >>> collator = DataCollatorForUnpairedPreference(pad_token_id=0)
    >>> examples = [
    ...     {"prompt_ids": [1, 2, 3], "completion_ids": [4, 5], "label": True},
    ...     {"prompt_ids": [7, 8], "completion_ids": [9], "label": False},
    ... ]
    >>> collator(examples)
    {'input_ids': tensor([[1, 2, 3, 4, 5],
                          [7, 8, 9, 0, 0]]),
     'attention_mask': tensor([[1, 1, 1, 1, 1],
                               [1, 1, 1, 0, 0]]),
     'completion_mask': tensor([[0, 0, 0, 1, 1],
                                [0, 0, 1, 0, 0]]),
     'label': tensor([ True, False])}

    >>> # With KL completions
    >>> examples = [
    ...     {"prompt_ids": [1, 2, 3], "completion_ids": [4, 5], "KL_completion_ids": [6], "label": True},
    ...     {"prompt_ids": [7, 8], "completion_ids": [9], "KL_completion_ids": [10, 11], "label": False},
    ... ]
    >>> collator(examples)
    {'input_ids': tensor([[1, 2, 3, 4, 5],
                          [7, 8, 9, 0, 0]]),
     'attention_mask': tensor([[1, 1, 1, 1, 1],
                               [1, 1, 1, 0, 0]]),
     'completion_mask': tensor([[0, 0, 0, 1, 1],
                                [0, 0, 1, 0, 0]]),
     'KL_input_ids': tensor([[ 1,  2,  3,  6],
                             [ 7,  8, 10, 11]]),
     'KL_attention_mask': tensor([[1, 1, 1, 1],
                                  [1, 1, 1, 1]]),
     'KL_completion_mask': tensor([[0, 0, 0, 1],
                                   [0, 0, 1, 1]]),
     'label': tensor([ True, False])}
    ```
    """

    pad_token_id: int
    max_length: int | None = None
    pad_to_multiple_of: int | None = None
    return_tensors: str = "pt"

    def torch_call(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        output = {}
        for prefix, ids_key in [("", "completion_ids"), ("KL_", "KL_completion_ids")]:
            if ids_key not in examples[0]:
                continue

            full_ids_list = []
            completion_mask_list = []
            for example in examples:
                prompt_ids = example["prompt_ids"]
                answer_ids = example[ids_key]
                full_ids = prompt_ids + answer_ids
                completion_mask = [0] * len(prompt_ids) + [1] * len(answer_ids)
                # Truncate per sequence if necessary
                if self.max_length is not None:
                    full_ids = full_ids[: self.max_length]
                    completion_mask = completion_mask[: self.max_length]
                full_ids_list.append(full_ids)
                completion_mask_list.append(completion_mask)

            output[f"{prefix}input_ids"] = pad(
                [torch.tensor(ids, dtype=torch.int64) for ids in full_ids_list],
                padding_value=self.pad_token_id,
                padding_side="right",
                pad_to_multiple_of=self.pad_to_multiple_of,
            )
            output[f"{prefix}attention_mask"] = pad(
                [torch.ones(len(ids), dtype=torch.int64) for ids in full_ids_list],
                padding_value=0,
                padding_side="right",
                pad_to_multiple_of=self.pad_to_multiple_of,
            )
            output[f"{prefix}completion_mask"] = pad(
                [torch.tensor(m, dtype=torch.int64) for m in completion_mask_list],
                padding_value=0,
                padding_side="right",
                pad_to_multiple_of=self.pad_to_multiple_of,
            )

        if "ref_logps" in examples[0]:
            output["ref_logps"] = torch.tensor([example["ref_logps"] for example in examples])
        if "ref_KL_logps" in examples[0]:
            output["ref_KL_logps"] = torch.tensor([example["ref_KL_logps"] for example in examples])
        # Must be a tensor: Accelerate cannot concatenate non-tensor fields across processes (accelerate#4111)
        output["label"] = torch.tensor([example["label"] for example in examples])
        return output


@dataclass
class DataCollatorForVisionUnpairedPreference(DataCollatorMixin):
    """
    Data collator for vision unpaired-preference tasks.

    Unlike text-only datasets, where the collator typically receives pre-tokenized inputs ready for batching,
    vision-language data processing involves converting images into pixel values. This conversion is disk-intensive,
    making upfront preprocessing of the entire dataset impractical. Therefore, this collator performs tokenization and
    image processing on-the-fly to efficiently prepare batches.

    When `calculate_kl` is `True`, the collator also produces KL sequences by cycling completions within the batch.

    Each input example should be a dictionary containing at least:
    - An `"images"` key holding a list of images, or an `"image"` key holding a single image.
    - Keys `"prompt"` and `"completion"` for the prompt and completion.
    - A `"label"` key (`bool`) indicating whether the completion is desirable.

    The collator outputs a dictionary including:
    - `"input_ids"`: Tensor of token IDs.
    - `"attention_mask"`: Tensor indicating attention mask.
    - `"pixel_values"`: Tensor representing image pixel values.
    - `"completion_mask"`: Tensor indicating which tokens correspond to completions.
    - `"label"`: Tensor of booleans indicating whether each completion is desirable.
    - When `calculate_kl` is `True`: `"KL_input_ids"`, `"KL_attention_mask"` and `"KL_completion_mask"` for the cycled
      KL sequences.

    Additional keys may be present depending on the processor, such as `"image_grid_thw"` or `"image_position_ids"`.

    Args:
        processor ([`~transformers.ProcessorMixin`]):
            The processor used to tokenize text and process images. It must be a subclass of
            [`~transformers.ProcessorMixin`] and include a `tokenizer` with a defined `pad_token_id`.
        max_length (`int`, *optional*):
            Maximum sequence length. Sequences longer than `max_length` are truncated to `max_length`. If `None`, no
            truncation is applied.
        calculate_kl (`bool`, *optional*, defaults to `True`):
            Whether to produce KL sequences by cycling completions within the batch.
        pad_to_multiple_of (`int`, *optional*):
            If set, the sequences will be padded to a multiple of this value.
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            Type of Tensor to return. Only `"pt"` is currently supported.

    Example:
    ```python
    >>> from trl.trainer.kto_trainer import DataCollatorForVisionUnpairedPreference
    >>> from transformers import AutoProcessor

    >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    >>> collator = DataCollatorForVisionUnpairedPreference(processor)
    >>> examples = [
    ...     {
    ...         "images": [Image.open("image_0.png")],
    ...         "prompt": [{"role": "user", "content": "What is this?"}],
    ...         "completion": [{"role": "assistant", "content": "This is a cat."}],
    ...         "label": True,
    ...     },
    ...     {
    ...         "images": [Image.open("image_1.png")],
    ...         "prompt": [{"role": "user", "content": "Describe this image."}],
    ...         "completion": [{"role": "assistant", "content": "A beautiful landscape."}],
    ...         "label": False,
    ...     },
    ... ]
    >>> collator(examples)
    {'input_ids': tensor([[151644,   8948,    198,   2610,    525,    264,  10950,  17847,     13, 151645,    198, 151644,    872,    198, 151652, 151655, 151655, 151655, 151655, 151653,   3838,    374,    419,     30, 151645,    198, 151644,  77091,    198,   1986,    374,    264,   8251,     13, 151645,    198],
                          [151644,   8948,    198,   2610,    525,    264,  10950,  17847,     13, 151645,    198, 151644,    872,    198, 151652, 151655, 151655, 151655, 151655, 151653,  74785,    419,   2168,     13, 151645,    198, 151644,  77091,    198,     32,   6233,  18414,     13, 151645,    198, 151643]]),
     'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                               [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]]),
     'pixel_values': tensor([[-0.0405, -0.0405, -0.0405,  ...,  1.3638,  1.3638,  1.3638],
                             [-0.0405, -0.0405, -0.0405,  ...,  1.3638,  1.3638,  1.3638],
                             [-0.0405, -0.0405, -0.0405,  ...,  1.3638,  1.3638,  1.3638],
                             ...,
                             [-1.3543, -1.3543, -1.3543,  ..., -0.2004, -0.2004, -0.2004],
                             [-1.3543, -1.3543, -1.3543,  ..., -0.2004, -0.2004, -0.2004],
                             [-1.3543, -1.3543, -1.3543,  ..., -0.2004, -0.2004, -0.2004]]),
     'image_grid_thw': tensor([[1, 4, 4],
                               [1, 4, 4]]),
     'completion_mask': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0]]),
     'KL_input_ids': tensor([[151644,   8948,    198,   2610,    525,    264,  10950,  17847,     13, 151645,    198, 151644,    872,    198, 151652, 151655, 151655, 151655, 151655, 151653,   3838,    374,    419,     30, 151645,    198, 151644,  77091,    198,     32,   6233,  18414,     13, 151645,    198, 151643],
                             [151644,   8948,    198,   2610,    525,    264,  10950,  17847,     13, 151645,    198, 151644,    872,    198, 151652, 151655, 151655, 151655, 151655, 151653,  74785,    419,   2168,     13, 151645,    198, 151644,  77091,    198,   1986,    374,    264,   8251,     13, 151645,    198]]),
     'KL_attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]),
     'KL_completion_mask': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]]),
     'label': tensor([ True, False])}
    ```
    """

    processor: ProcessorMixin
    max_length: int | None = None
    calculate_kl: bool = True
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
        images = [example["images"] for example in examples]
        # Transformers requires at least one image in the batch, otherwise it throws an error
        if all(img_list == [] for img_list in images):
            images = None
        if is_conversational(examples[0]):  # conversational case
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
        if "token_type_ids" in processed_prompts:  # special case for Gemma
            prompt_token_type_ids = processed_prompts["token_type_ids"]
            completion_token_type_ids = processed_completions["token_type_ids"]
            token_type_ids = torch.cat((prompt_token_type_ids, completion_token_type_ids), dim=1)
        if "mm_token_type_ids" in processed_prompts:  # special case for Qwen2.5-VL
            prompt_mm_token_type_ids = processed_prompts["mm_token_type_ids"]
            mm_token_type_ids = torch.cat((prompt_mm_token_type_ids, torch.zeros_like(completion_ids)), dim=1)

        # Flush left to reduce padding
        if "token_type_ids" in processed_prompts and "mm_token_type_ids" in processed_prompts:
            attention_mask, input_ids, completion_mask, token_type_ids, mm_token_type_ids = flush_left(
                attention_mask, input_ids, completion_mask, token_type_ids, mm_token_type_ids
            )
        elif "token_type_ids" in processed_prompts:
            attention_mask, input_ids, completion_mask, token_type_ids = flush_left(
                attention_mask, input_ids, completion_mask, token_type_ids
            )
        elif "mm_token_type_ids" in processed_prompts:
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
            if "token_type_ids" in processed_prompts:
                token_type_ids = token_type_ids[:, : self.max_length]
            if "mm_token_type_ids" in processed_prompts:
                mm_token_type_ids = mm_token_type_ids[:, : self.max_length]

        # Build the output dictionary
        output = processed_prompts  # we take processed_prompts because it contains the images
        output["input_ids"] = input_ids
        output["attention_mask"] = attention_mask
        output["completion_mask"] = completion_mask
        if "token_type_ids" in processed_prompts:
            output["token_type_ids"] = token_type_ids
        if "mm_token_type_ids" in processed_prompts:
            output["mm_token_type_ids"] = mm_token_type_ids

        if self.calculate_kl:
            # Cycle completions by +1 within the batch to create mismatched KL pairs — same strategy as
            # get_kl_completion_ids in the text-only path, but done here to keep the VLM dataset fully raw.
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
            if "token_type_ids" in processed_prompts:
                kl_completion_token_type_ids = processed_kl["token_type_ids"]
                kl_token_type_ids = torch.cat((prompt_token_type_ids, kl_completion_token_type_ids), dim=1)
            if "mm_token_type_ids" in processed_prompts:
                kl_mm_token_type_ids = torch.cat((prompt_mm_token_type_ids, torch.zeros_like(kl_ids)), dim=1)

            if "token_type_ids" in processed_prompts and "mm_token_type_ids" in processed_prompts:
                kl_attention_mask, kl_input_ids, kl_completion_mask, kl_token_type_ids, kl_mm_token_type_ids = (
                    flush_left(
                        kl_attention_mask, kl_input_ids, kl_completion_mask, kl_token_type_ids, kl_mm_token_type_ids
                    )
                )
            elif "token_type_ids" in processed_prompts:
                kl_attention_mask, kl_input_ids, kl_completion_mask, kl_token_type_ids = flush_left(
                    kl_attention_mask, kl_input_ids, kl_completion_mask, kl_token_type_ids
                )
            elif "mm_token_type_ids" in processed_prompts:
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
                if "token_type_ids" in processed_prompts:
                    kl_token_type_ids = kl_token_type_ids[:, : self.max_length]
                if "mm_token_type_ids" in processed_prompts:
                    kl_mm_token_type_ids = kl_mm_token_type_ids[:, : self.max_length]

            output["KL_input_ids"] = kl_input_ids
            output["KL_attention_mask"] = kl_attention_mask
            output["KL_completion_mask"] = kl_completion_mask
            if "token_type_ids" in processed_prompts:
                output["KL_token_type_ids"] = kl_token_type_ids
            if "mm_token_type_ids" in processed_prompts:
                output["KL_mm_token_type_ids"] = kl_mm_token_type_ids

        # Must be a tensor: Accelerate cannot concatenate non-tensor fields across processes (accelerate#4111)
        output["label"] = torch.tensor([example["label"] for example in examples])
        return output


class KTOTrainer(_BaseTrainer):
    """
    Trainer for Kahneman-Tversky Optimization (KTO) method. This algorithm was initially proposed in the paper [KTO:
    Model Alignment as Prospect Theoretic Optimization](https://huggingface.co/papers/2402.01306). This class is a
    wrapper around the [`~transformers.Trainer`] class and inherits all of its attributes and methods.

    Example:

    ```python
    >>> from trl import KTOTrainer
    >>> from datasets import load_dataset

    >>> dataset = load_dataset("trl-lib/kto-mix-14k", split="train")

    >>> trainer = KTOTrainer(
    ...     model="Qwen/Qwen2.5-0.5B-Instruct",
    ...     train_dataset=dataset,
    ... )
    >>> trainer.train()
    ```

    Args:
        model (`str` or [`~transformers.PreTrainedModel`] or [`~peft.PeftModel`]):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or a
              path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
              using `<ModelArchitecture>.from_pretrained` (where `<ModelArchitecture>` is derived from the model
              config) with the keyword arguments in `args.model_init_kwargs`. If `dtype` is not specified in
              `args.model_init_kwargs`, it defaults to `float32`. This differs from
              [`~transformers.PreTrainedModel.from_pretrained`], where (since Transformers v5) the dtype is inferred
              from the model config.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
            - A [`~peft.PeftModel`] object. Only causal language models are supported.
        ref_model ([`~transformers.PreTrainedModel`], *optional*):
            Reference model used to compute the reference log probabilities.

            - If provided, this model is used directly as the reference policy.
            - If `None`, the trainer will automatically use the initial policy corresponding to `model`, i.e. the model
              state before KTO training starts.
        args ([`KTOConfig`], *optional*):
            Configuration for this trainer. If `None`, a default configuration is used.
        data_collator ([`~transformers.DataCollator`], *optional*):
            Function to use to form a batch from a list of elements of the processed `train_dataset` or `eval_dataset`.
            Will default to [`~trainer.kto_trainer.DataCollatorForUnpairedPreference`] if the model is a language model
            and [`~trainer.kto_trainer.DataCollatorForVisionUnpairedPreference`] if the model is a vision-language
            model. Custom collators must truncate sequences before padding; the trainer does not apply post-collation
            truncation.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. This trainer supports [unpaired preference](#unpaired-preference) type. The
            format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).

            When `train_dataset` is an [`~datasets.IterableDataset`] (e.g. a streaming dataset), `max_steps` must be
            set in the training arguments, since its length cannot be inferred and the total number of training steps
            is required to bound the training loop and configure the learning rate scheduler.
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`], [`~datasets.DatasetDict`], [`~datasets.IterableDatasetDict`] or `dict[str, Dataset | IterableDataset]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`] or [`~transformers.ProcessorMixin`], *optional*):
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
        quantization_config ([`~transformers.BitsAndBytesConfig`], *optional*):
            Quantization configuration used when loading the model from a model identifier. Combine with `peft_config`
            for QLoRA training. Ignored if the model is already instantiated.
        peft_config ([`~peft.PeftConfig`], *optional*):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
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
        data_collator: DataCollator | None = None,
        train_dataset: Dataset | IterableDataset | None = None,
        eval_dataset: Dataset
        | IterableDataset
        | DatasetDict
        | IterableDatasetDict
        | dict[str, Dataset | IterableDataset]
        | None = None,
        processing_class: PreTrainedTokenizerBase | ProcessorMixin | None = None,
        compute_metrics: Callable[[EvalPrediction], dict] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None] = (None, None),
        quantization_config: "BitsAndBytesConfig | None" = None,
        peft_config: "PeftConfig | None" = None,
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
        elif not isinstance(train_dataset, Dataset):
            raise TypeError(
                f"`train_dataset` must be a `Dataset` or `IterableDataset`, got `{type(train_dataset).__name__}`."
            )

        # Model
        if isinstance(model, str):
            model_init_kwargs = dict(args.model_init_kwargs or {})  # copy to avoid mutating model_init_kwargs
            if quantization_config is not None:
                if "quantization_config" in model_init_kwargs:
                    raise ValueError(
                        "You set `quantization_config` both as a trainer argument and in `args.model_init_kwargs`. "
                        "Please set it in only one place, preferably as a trainer argument."
                    )
                model_init_kwargs["quantization_config"] = quantization_config
            # Distributed training requires device_map=None ("auto" fails)
            if args.distributed_state.distributed_type in ["MULTI_GPU", "DEEPSPEED"]:
                model_init_kwargs["device_map"] = None
            model_init_kwargs.setdefault("trust_remote_code", args.trust_remote_code)
            model = create_model_from_path(model, **model_init_kwargs)
        else:
            if args.model_init_kwargs is not None:
                logger.warning(
                    "You passed `model_init_kwargs` to the `KTOConfig`, but your model is already instantiated. "
                    "The `model_init_kwargs` will be ignored."
                )
            if quantization_config is not None:
                raise ValueError(
                    "You passed `quantization_config` to the trainer, but your model is already instantiated. "
                    "Quantization can only be applied when the model is loaded from a model identifier (`str`). "
                    "Either pass the model as a model identifier, or omit `quantization_config`."
                )
        # Non-quantized models do not have the `is_loaded_in_{8,4}bit` attributes, whereas quantized models do
        _is_quantized_model = getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False)
        if ref_model is model:
            raise ValueError(
                "`model` and `ref_model` cannot be the same object. In most cases you should omit `ref_model` and "
                "we'll initialize it to a copy of `model` for you."
            )

        # Processing class
        if processing_class is None:
            processing_class = AutoProcessor.from_pretrained(
                get_config_model_id(model.config), trust_remote_code=args.trust_remote_code
            )
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
            # ZeRO-3 + PEFT for non-quantized models:
            # - PEFT's default autocast_adapter_dtype=True upcasts LoRA adapter params to fp32 even when the base model is bf16.
            # - ZeRO-3's _allgather_params_coalesced allocates output buffers using the dtype of the first persistent parameter,
            #   so mixed-dtype persistent_parameters (bf16 base + fp32 LoRA) cause a TypeError on the first optimizer step.
            # - Passing autocast_adapter_dtype=False keeps adapter params in the base model dtype (bf16), fixing the mismatch.
            # - This is safe: the fp32 upcast is a QLoRA-specific concern (low-bit quantized base models), not needed for
            #   non-quantized bf16 training.
            # - See:
            #   - TRL issue: https://github.com/huggingface/trl/issues/6089
            #   - Upstream issue: https://github.com/deepspeedai/DeepSpeed/issues/8072
            # - autocast_adapter_dtype was introduced in PEFT 0.12.0; before, no upcast existed: no need to pass the kwarg
            get_peft_model_kwargs = {}
            if (
                args.deepspeed_plugin is not None
                and args.deepspeed_plugin.zero_stage == 3
                and not _is_quantized_model
                and Version(peft.__version__) >= Version("0.12.0")
            ):
                get_peft_model_kwargs["autocast_adapter_dtype"] = False
            model = get_peft_model(model, peft_config, **get_peft_model_kwargs)

        elif is_peft_model(model) and ref_model is None:
            # If the model is a PEFT model with a pretrained adapter, we need to create a "ref" adapter that is a copy
            # of the "default" adapter, so that we can use it as the reference model during KTO training. PEFT only
            # supports one adapter per model when the LoRA config uses `target_parameters` (see peft#3340), so in that
            # case we skip the "ref" adapter and compute the reference log probs with adapters disabled, i.e. with the
            # base model.
            default_config = model.peft_config["default"]
            if isinstance(default_config, LoraConfig) and default_config.target_parameters:
                logger.warning(
                    "PEFT can't add a frozen reference adapter alongside one that uses `target_parameters` "
                    "(peft#3340), so the reference log probs are computed from the base model (adapters disabled). "
                    "If you wrapped the model only to apply LoRA, pass a `peft_config` to the trainer instead; if you "
                    "wrapped it deliberately (pretrained adapter or custom init), note that the base model matches "
                    "your adapter only when it's freshly zero-initialized. If it is, this warning is safe to ignore."
                )
            else:
                model.add_adapter("ref", default_config)
                for name, param in model.named_parameters():
                    if ".default." in name:
                        ref_name = name.replace(".default.", ".ref.")
                        ref_param = model.get_parameter(ref_name)
                        ref_param.data.copy_(param.data)

        # PEFT + DeepSpeed ZeRO-3 requires reentrant checkpointing. For more details, see
        # https://github.com/huggingface/trl/issues/2514#issuecomment-2692152703.
        # Can be removed once https://github.com/deepspeedai/DeepSpeed/pull/8130 is merged and released.
        if (
            is_peft_model(model)
            and args.deepspeed_plugin is not None
            and args.deepspeed_plugin.zero_stage == 3
            and args.gradient_checkpointing
        ):
            args.gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs or {}
            use_reentrant = args.gradient_checkpointing_kwargs.get("use_reentrant")
            if use_reentrant is False:
                logger.warning(
                    "You are using PEFT with DeepSpeed ZeRO-3 and gradient checkpointing with `use_reentrant=False`. "
                    "`use_reentrant` is forced to `True` in this configuration to ensure correct training. To remove "
                    "this warning, unset `use_reentrant` in `gradient_checkpointing_kwargs` or set it to `True`."
                )
            args.gradient_checkpointing_kwargs["use_reentrant"] = True

        # When using gradient checkpointing with PEFT, we need to enable input gradients. transformers.Trainer normally
        # handles this, but a bug currently prevents it; see https://github.com/huggingface/transformers/issues/42489
        if is_peft_model(model) and args.gradient_checkpointing:
            model.enable_input_require_grads()

        # When using QLoRA, the PEFT adapter weights are converted to bf16 to follow the recommendations from the
        # original paper (see https://huggingface.co/papers/2305.14314, paragraph 3). Normally, this can be done by
        # passing `autocast_adapter_dtype=False` to `get_peft_model`, but this option is not yet supported for
        # quantized models. See: https://github.com/huggingface/peft/issues/2889
        if _is_quantized_model:
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
                pad_to_multiple_of=args.pad_to_multiple_of,
            )
        elif data_collator is None and self._is_vision_dataset:
            data_collator = DataCollatorForVisionUnpairedPreference(
                processor=processing_class,
                max_length=args.max_length,
                calculate_kl=calculate_kl,
                pad_to_multiple_of=args.pad_to_multiple_of,
            )

        # Training arguments
        self.beta = args.beta
        self.precompute_ref_logps = args.precompute_ref_log_probs
        self.loss_type = args.loss_type
        self.desirable_weight = args.desirable_weight
        self.undesirable_weight = args.undesirable_weight
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
        # Liger loss
        self.use_liger_kernel = args.use_liger_kernel
        if self.use_liger_kernel:
            if not is_liger_kernel_available():
                raise ImportError(
                    "You set `use_liger_kernel=True` but the liger kernel is not available. "
                    "Please install liger-kernel first: `pip install liger-kernel`"
                )
            if self.loss_type in ["apo_zero_unpaired"]:
                raise ValueError(
                    "You cannot set `loss_type='apo_zero_unpaired'` with liger-kernel. "
                    "Only KTO loss is supported with liger-kernel."
                )
            if compute_metrics is not None:
                raise ValueError(
                    "compute_metrics is not supported with the Liger kernel. compute_metrics requires to be able to "
                    "recover the logits from the forward pass, but Liger kernel does not materialize logits."
                )
            if self.precompute_ref_logps:
                raise ValueError(
                    "Liger KTO loss does not support precomputing reference log probabilities. Either disable "
                    "`precompute_ref_log_probs` or set `use_liger_kernel` to False."
                )
            if is_peft_model(model):
                # The Liger fused KTO loss multiplies the hidden states by `lm_head.weight` directly. When the LM head
                # is targeted by a PEFT adapter (`"lm_head"` in `target_modules`), `lm_head.weight` is the frozen base
                # weight and the trainable adapter parameters live in separate submodules that Liger never sees. The
                # head adapter would silently receive no gradient, so the model trains as if `lm_head` were frozen.
                # Fail loudly rather than train a silently-frozen head.
                output_embeddings = model.get_output_embeddings()
                if isinstance(output_embeddings, BaseTunerLayer):
                    raise ValueError(
                        "`use_liger_kernel=True` is incompatible with applying a PEFT adapter to `lm_head`. The Liger "
                        "fused KTO loss reads `lm_head.weight` directly, so the adapter on the head is ignored and "
                        "never trained. Either remove `'lm_head'` from your `target_modules`, or set "
                        "`use_liger_kernel=False`."
                    )
                # Prompt-learning methods (PromptTuning, PrefixTuning, P-Tuning) inject virtual tokens via
                # `PeftModel.forward()`. The Liger KTO loss bypasses `PeftModel.forward()` by calling the backbone
                # directly, so virtual tokens are never prepended and the loss is computed on the wrong sequence.
                # Fail loudly rather than train on a silently corrupted input.
                if any(isinstance(cfg, PromptLearningConfig) for cfg in model.peft_config.values()):
                    raise ValueError(
                        "`use_liger_kernel=True` is incompatible with prompt-learning PEFT methods (PromptTuning, "
                        "PrefixTuning, P-Tuning). The Liger KTO loss bypasses `PeftModel.forward()` by calling the "
                        "backbone directly, so virtual tokens are never prepended and the loss is computed on the "
                        "wrong sequence. Use a weight-based adapter such as LoRA instead, or set "
                        "`use_liger_kernel=False`."
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

        # MoE load-balancing auxiliary loss, applied to Mixture-of-Experts models (no effect otherwise)
        text_config = model.config.get_text_config()
        is_moe = getattr(text_config, "output_router_logits", None) is not None
        self.aux_loss_enabled = is_moe and args.router_aux_loss_coef != 0.0
        self.router_aux_loss_coef = args.router_aux_loss_coef
        if self.aux_loss_enabled and self.use_liger_kernel:
            raise ValueError(
                "Liger KTO loss does not support the Mixture-of-Experts load-balancing auxiliary loss, because it "
                "fuses the loss without materializing the router logits. Either set `router_aux_loss_coef` to `0.0` "
                "to disable the auxiliary loss, or set `use_liger_kernel` to False."
            )

        # Reference model
        if ref_model is None:
            if is_peft_model(self.model) or args.precompute_ref_log_probs:
                # If PEFT is used, the reference model is not needed since the adapter can be disabled to revert to the
                # initial model. If precompute_ref_log_probs is True, the reference model does not need to be kept in
                # memory during training.
                self.ref_model = None
            else:
                ref_model_init_kwargs = dict(args.model_init_kwargs or {})  # copy to avoid mutating model_init_kwargs
                if quantization_config is not None:
                    ref_model_init_kwargs["quantization_config"] = quantization_config
                # Distributed training requires device_map=None ("auto" fails)
                if self.args.distributed_state.distributed_type in ["MULTI_GPU", "DEEPSPEED"]:
                    ref_model_init_kwargs["device_map"] = None
                ref_model_init_kwargs.setdefault("trust_remote_code", args.trust_remote_code)
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
        self._total_train_tokens = 0

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

        if args.sync_ref_model:
            if is_peft_model(self.model):
                raise NotImplementedError(
                    "You passed `sync_ref_model=True` while using a PEFT model, which is currently not supported. "
                    "With PEFT, KTOTrainer does not keep a separate reference model in memory; instead, it recovers "
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

        # The Liger loss is built here, because it needs `self.ref_model`
        if self.use_liger_kernel:
            self.liger_loss = LigerFusedLinearKTOLoss(beta=self.beta, use_ref_model=(self.ref_model is not None))
            # Redirect the model.module forward to the model forward to ensure pre-forward hooks are called, so that
            # under ZeRO-3 the parameter coordinator gathers/reduces `lm_head.weight` around the fused loss.
            self._forward_redirection = _ForwardRedirection()

        # Reference forwards during precompute reuse a single DeepSpeed inference engine (see `_precompute_ref_logps`).
        self._precompute_engine = None
        if self.precompute_ref_logps:
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

    def _prepare_dataset(
        self,
        dataset: Dataset | IterableDataset,
        processing_class: PreTrainedTokenizerBase | ProcessorMixin,
        args: KTOConfig,
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
                tools = example.get("tools")
                tools = json.loads(tools) if isinstance(tools, str) else tools
                apply_chat_template_kwargs = {"tools": tools, **example.get("chat_template_kwargs", {})}
                if is_conversational(example):
                    prompt_ids = _tokenize(
                        processing_class,
                        example["prompt"],
                        add_generation_prompt=True,
                        **apply_chat_template_kwargs,
                    )["input_ids"]
                    prompt_completion_ids = _tokenize(
                        processing_class,
                        example["prompt"] + example["completion"],
                        **apply_chat_template_kwargs,
                    )["input_ids"]
                else:
                    prompt_ids = _tokenize(processing_class, example["prompt"])["input_ids"]
                    prompt_completion_ids = _tokenize(processing_class, example["prompt"] + example["completion"])[
                        "input_ids"
                    ]

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

            # Drop examples whose prompt alone fills `max_length`: with `keep_start` truncation the collator would
            # remove every completion token, leaving no learning signal. `keep_end` keeps the completion end, so
            # nothing is dropped there.
            if args.max_length is not None:
                if isinstance(dataset, Dataset):  # `IterableDataset.filter` does not support `desc`
                    map_kwargs["desc"] = f"Dropping fully truncated examples from {dataset_name} dataset"
                dataset = dataset.filter(lambda example: len(example["prompt_ids"]) < args.max_length, **map_kwargs)

            # Add KL completions if needed. The KL term is estimated from mismatched (prompt, completion) pairs, built
            # by rotating the completions by +1 within each batch of size `per_device_train_batch_size`:
            # (x_1, y_1), ..., (x_n, y_n) --> (x_1, y_n), (x_2, y_1), ..., (x_n, y_{n-1}). For best results, the
            # mismatched outputs y' used to estimate the KL term for a batch should be the same set as the matched
            # outputs y used to estimate the rewards in that batch, just paired with different x.
            if self.calculate_KL:
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Extracting KL {dataset_name} dataset"

                def add_kl_completion_ids(examples):
                    examples["KL_completion_ids"] = [examples["completion_ids"][-1]] + examples["completion_ids"][:-1]
                    return examples

                dataset = dataset.map(
                    add_kl_completion_ids, batched=True, batch_size=args.per_device_train_batch_size, **map_kwargs
                )

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
                    "tools",
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

    def _get_train_sampler(self, train_dataset: Dataset | None = None) -> Sampler | None:
        if self.calculate_KL and Version(transformers.__version__) < Version("5.2.0"):
            if train_dataset is None:
                train_dataset = self.train_dataset
            if train_dataset is None or not has_length(train_dataset):
                return None
            return SequentialSampler(train_dataset)
        return super()._get_train_sampler(train_dataset)

    def _precompute_ref_logps(self, dataset: Dataset, name: str, batch_size: int) -> Dataset:
        if isinstance(dataset, IterableDataset):
            raise ValueError(
                "`precompute_ref_log_probs=True` is not supported with IterableDataset. Please use a map-style "
                "Dataset or set `precompute_ref_log_probs=False`."
            )
        model_hash = hash_module(self.ref_model or self.model)
        # Both inputs are rank-dependent under distributed training (ZeRO-3 shards the model), so broadcast rank 0's
        # value so all ranks share one cache file.
        fingerprint = [Hasher.hash((dataset._fingerprint, model_hash, self.calculate_KL))]
        broadcast_object_list(fingerprint, from_process=0)
        fingerprint = fingerprint[0]
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

        # This runs before the parent class prepares the model in `train`, so with DeepSpeed the parameters are still
        # on CPU (ZeRO-1/2) and sharded (ZeRO-3). Wrap the model in an inference engine to place and gather them. Build
        # it once and reuse it across precompute passes (train, eval, and later `evaluate` calls)
        if self.ref_model is None and self.is_deepspeed_enabled:
            if self._precompute_engine is None:
                self._precompute_engine = prepare_deepspeed(self.model, self.accelerator)
            model = self._precompute_engine
        else:
            model = self.ref_model or self.model

        ref_logps = []
        ref_KL_logps = []
        for padded_batch in tqdm(iterable=data_loader, desc=f"Computing reference log probs for {name} dataset"):
            ref_logp, ref_KL_logp = self.compute_ref_log_probs(model, padded_batch)
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
                cache_file_name=cache_file,
                desc=f"Caching reference log probs for {name} dataset",
            )
        self.accelerator.wait_for_everyone()

        return concatenate_datasets([dataset, Dataset.from_file(cache_file)], axis=1)

    def compute_ref_log_probs(self, model, inputs):
        """Computes reference log probabilities for a single padded batch."""
        with torch.no_grad(), disable_gradient_checkpointing(self.model, self.args.gradient_checkpointing_kwargs):
            if self.ref_model is None and is_peft_model(self.model):
                unwrapped_model = self.accelerator.unwrap_model(self.model)
                with use_adapter(
                    unwrapped_model, adapter_name="ref" if "ref" in unwrapped_model.peft_config else None
                ):
                    completion_logits = model(inputs["input_ids"], attention_mask=inputs["attention_mask"]).logits

                    if self.calculate_KL:
                        KL_logits = model(inputs["KL_input_ids"], attention_mask=inputs["KL_attention_mask"]).logits
            else:
                completion_logits = model(inputs["input_ids"], attention_mask=inputs["attention_mask"]).logits

                if self.calculate_KL:
                    KL_logits = model(inputs["KL_input_ids"], attention_mask=inputs["KL_attention_mask"]).logits

        shift_logits = completion_logits[:, :-1, :]
        per_token_logps = selective_log_softmax(shift_logits, inputs["input_ids"][:, 1:])
        per_token_logps[inputs["completion_mask"][:, 1:] == 0] = 0.0
        completion_logps = per_token_logps.sum(-1)

        if self.calculate_KL:
            shift_KL_logits = KL_logits[:, :-1, :]
            KL_per_token_logps = selective_log_softmax(shift_KL_logits, inputs["KL_input_ids"][:, 1:])
            KL_per_token_logps[inputs["KL_completion_mask"][:, 1:] == 0] = 0.0
            KL_logps = KL_per_token_logps.sum(-1)
        else:
            KL_logps = None

        return completion_logps, KL_logps

    def _compute_kl_logps(self, model, batch):
        """Compute KL log probabilities for a given batch."""
        KL_logps = None
        if self.calculate_KL:
            _non_model_keys = {
                "input_ids",
                "attention_mask",
                "completion_mask",
                "KL_completion_mask",
                "KL_token_type_ids",
                "KL_mm_token_type_ids",
                "label",
                "ref_logps",
                "ref_KL_logps",
            }
            KL_model_kwargs = {k: v for k, v in batch.items() if k not in _non_model_keys}
            KL_model_kwargs["input_ids"] = KL_model_kwargs.pop("KL_input_ids")
            KL_model_kwargs["attention_mask"] = KL_model_kwargs.pop("KL_attention_mask")
            # KL sequences have different widths from the main completion after flush_left; override token-type
            # tensors with the KL-specific ones the collator built for exactly this purpose.
            if "KL_token_type_ids" in batch:
                KL_model_kwargs["token_type_ids"] = batch["KL_token_type_ids"]
            if "KL_mm_token_type_ids" in batch:
                KL_model_kwargs["mm_token_type_ids"] = batch["KL_mm_token_type_ids"]

            with torch.no_grad():
                if self.use_liger_kernel:
                    # Running the full model would call `lm_head` as a module. Under ZeRO-3 that registers
                    # `lm_head.weight` in the parameter coordinator's forward trace, which then conflicts with the
                    # dense weight gradient the fused loss produces for the same parameter (the fused backward fails
                    # with a shape mismatch). Mirror the fused path instead: run the backbone and matmul with the
                    # gathered `lm_head` weight so `lm_head` is only ever touched directly, never as a module.
                    inner = model.base_model.model if is_peft_model(model) else model
                    if self._is_vlm and Version(transformers.__version__) < Version("5.0.0"):
                        backbone = inner.model
                    else:
                        backbone = inner.base_model
                    lm_head = inner.get_output_embeddings()
                    KL_hidden_states = backbone(**KL_model_kwargs).last_hidden_state
                    with maybe_gather_lm_head_ctx(lm_head.weight, lm_head.bias):
                        KL_logits = KL_hidden_states @ lm_head.weight.t()
                        if lm_head.bias is not None:
                            KL_logits = KL_logits + lm_head.bias
                else:
                    KL_logits = model(**KL_model_kwargs).logits

            shift_KL_logits = KL_logits[:, :-1, :]
            KL_per_token_logps = selective_log_softmax(shift_KL_logits, batch["KL_input_ids"][:, 1:])
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

        labels = batch["label"]
        num_chosen = labels.sum().to(self.accelerator.device)
        num_rejected = (len(labels) - num_chosen).to(self.accelerator.device)

        KL_logps = self._compute_kl_logps(model, batch)

        _non_model_keys = {
            "completion_mask",
            "KL_input_ids",
            "KL_attention_mask",
            "KL_completion_mask",
            "KL_token_type_ids",
            "KL_mm_token_type_ids",
            "label",
            "ref_logps",
            "ref_KL_logps",
        }
        model_kwargs = {k: v for k, v in batch.items() if k not in _non_model_keys}
        model_kwargs["use_cache"] = False

        if is_peft_model(model):
            model = model.base_model.model

        # `base_model` gives the inner module (skipping `lm_head`) — text decoder for LMs, multimodal wrapper for
        # VLMs (so vision-token injection runs before the text decoder). `get_decoder()` won't do: on VLMs it
        # returns just the text stack and feeds image-placeholder IDs through it.
        # Pre-5.0 transformers VLMs set `base_model_prefix = ""` so `base_model is self` (re-runs `lm_head`).
        # Fall back to `.model` there.
        if self._is_vlm and Version(transformers.__version__) < Version("5.0.0"):
            backbone = model.model
        else:
            backbone = model.base_model

        outputs = backbone(**model_kwargs)
        lm_head = model.get_output_embeddings()

        # reference model
        with torch.no_grad(), disable_gradient_checkpointing(self.model, self.args.gradient_checkpointing_kwargs):
            if self.ref_model is None:
                # PEFT model with no explicit reference model: recover reference behaviour by disabling / switching to
                # the frozen "ref" adapter, exactly as _compute_loss does for logit-based reference computation.
                model_unwrapped = self.accelerator.unwrap_model(self.model)
                with use_adapter(
                    model_unwrapped, adapter_name="ref" if "ref" in model_unwrapped.peft_config else None
                ):
                    ref_KL_logps = self._compute_kl_logps(self.model, batch)
                    ref_model_inner = model_unwrapped.base_model.model
                    if self._is_vlm and Version(transformers.__version__) < Version("5.0.0"):
                        ref_backbone = ref_model_inner.model
                    else:
                        ref_backbone = ref_model_inner.base_model
                    ref_outputs = ref_backbone(**model_kwargs)
                    ref_lm_head = model_unwrapped.get_output_embeddings()
            else:
                ref_KL_logps = self._compute_kl_logps(self.ref_model, batch)
                ref_model_inner = self.ref_model.base_model.model if is_peft_model(self.ref_model) else self.ref_model
                if self._is_vlm and Version(transformers.__version__) < Version("5.0.0"):
                    ref_backbone = ref_model_inner.model
                else:
                    ref_backbone = ref_model_inner.base_model
                ref_outputs = ref_backbone(**model_kwargs)
                ref_lm_head = self.ref_model.get_output_embeddings()

        if self.calculate_KL:
            kl = (KL_logps - ref_KL_logps).mean().detach()
            kl = self.accelerator.gather_for_metrics(kl).mean().clamp(min=0)
        else:
            kl = torch.zeros(1).to(self.accelerator.device)

        shift_completion_mask = batch["completion_mask"][:, 1:]
        target = batch["input_ids"][:, 1:].clone()
        target[shift_completion_mask == 0] = -100

        with maybe_gather_lm_head_ctx(lm_head.weight, lm_head.bias, ref_lm_head.weight, ref_lm_head.bias):
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
            ) = self.liger_loss(
                _input=outputs.last_hidden_state[:, :-1],
                lin_weight=lm_head.weight,
                target=target,
                bias=lm_head.bias,
                preference_labels=batch["label"],
                ref_input=ref_outputs.last_hidden_state[:, :-1],
                ref_weight=ref_lm_head.weight,
                ref_bias=ref_lm_head.bias,
                kl=kl,
            )

        self._metrics[mode]["kl"].append(kl.item())

        # Number of tokens
        if mode == "train":
            num_tokens_in_batch = self.accelerator.gather_for_metrics(batch["attention_mask"].sum()).sum().item()
            self._total_train_tokens += num_tokens_in_batch
        self._metrics[mode]["num_tokens"] = [self._total_train_tokens]

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

        if all_num_chosen > 0 and all_num_rejected > 0:
            self._metrics[mode]["rewards/margins"].append(
                self._metrics[mode]["rewards/chosen"][-1] - self._metrics[mode]["rewards/rejected"][-1]
            )

        return loss

    def _compute_loss(self, model, inputs, return_outputs):
        """Compute the KTO loss and other metrics for the given batch of inputs for train or test."""
        mode = "train" if self.model.training else "eval"
        batch = {k: (v.to(self.accelerator.device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

        labels = batch["label"]
        num_chosen = labels.sum().to(self.accelerator.device)
        num_rejected = (len(labels) - num_chosen).to(self.accelerator.device)

        KL_logps = self._compute_kl_logps(model, batch)

        _non_model_keys = {
            "completion_mask",
            "KL_input_ids",
            "KL_attention_mask",
            "KL_completion_mask",
            "KL_token_type_ids",
            "KL_mm_token_type_ids",
            "label",
            "ref_logps",
            "ref_KL_logps",
        }
        model_kwargs = {k: v for k, v in batch.items() if k not in _non_model_keys}
        if self.aux_loss_enabled:
            model_kwargs["output_router_logits"] = True

        outputs = model(**model_kwargs)

        shift_logits = outputs.logits[:, :-1, :]
        per_token_logps = selective_log_softmax(shift_logits, batch["input_ids"][:, 1:])
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

        chosen_logps = completion_logps.index_select(0, chosen_idx)
        rejected_logps = completion_logps.index_select(0, rejected_idx)

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
            ref_shift_logits = ref_outputs.logits[:, :-1, :]
            ref_per_token_logps = selective_log_softmax(ref_shift_logits, batch["input_ids"][:, 1:])
            ref_per_token_logps[batch["completion_mask"][:, 1:] == 0] = 0.0
            ref_completion_logps = ref_per_token_logps.sum(-1)
            ref_chosen_logps = ref_completion_logps.index_select(0, chosen_idx)
            ref_rejected_logps = ref_completion_logps.index_select(0, rejected_idx)

        if self.calculate_KL:
            kl = (KL_logps - ref_KL_logps).mean().detach()
            kl = self.accelerator.gather_for_metrics(kl).mean().clamp(min=0)
        else:
            kl = torch.zeros(1).to(chosen_logps.device)
        # Chosen losses
        if chosen_logps.shape[0] != 0 or ref_chosen_logps.shape[0] != 0:
            chosen_logratios = chosen_logps - ref_chosen_logps

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
        if rejected_logps.shape[0] != 0 or ref_rejected_logps.shape[0] != 0:
            rejected_logratios = rejected_logps - ref_rejected_logps

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

        self._metrics[mode]["kl"].append(kl.item())

        # Entropy
        per_token_entropy = entropy_from_logits(shift_logits.detach())
        mask = batch["completion_mask"][:, 1:]
        entropy_sum = (per_token_entropy * mask).sum()
        total_tokens = mask.sum()

        # Gather counts across ranks and weight-average
        entropy_sum = self.accelerator.gather_for_metrics(entropy_sum).sum()
        total_tokens = self.accelerator.gather_for_metrics(total_tokens).sum()
        entropy = (entropy_sum / total_tokens).item() if total_tokens > 0 else 0.0
        self._metrics[mode]["entropy"].append(entropy)

        # Number of tokens
        if mode == "train":
            num_tokens_in_batch = self.accelerator.gather_for_metrics(batch["attention_mask"].sum()).sum().item()
            self._total_train_tokens += num_tokens_in_batch
        self._metrics[mode]["num_tokens"] = [self._total_train_tokens]

        # Average logits for chosen and rejected completions
        shift_completion_mask = batch["completion_mask"][:, 1:]
        chosen_logits = shift_logits.detach().index_select(0, chosen_idx)
        rejected_logits = shift_logits.detach().index_select(0, rejected_idx)
        chosen_mask = shift_completion_mask.index_select(0, chosen_idx)
        rejected_mask = shift_completion_mask.index_select(0, rejected_idx)
        total_chosen_logits = chosen_logits[chosen_mask.bool()].mean(-1).sum()
        total_chosen_tokens = chosen_mask.sum()
        total_rejected_logits = rejected_logits[rejected_mask.bool()].mean(-1).sum()
        total_rejected_tokens = rejected_mask.sum()
        total_chosen_logits = self.accelerator.gather_for_metrics(total_chosen_logits).sum().item()
        total_chosen_tokens = self.accelerator.gather_for_metrics(total_chosen_tokens).sum().item()
        total_rejected_logits = self.accelerator.gather_for_metrics(total_rejected_logits).sum().item()
        total_rejected_tokens = self.accelerator.gather_for_metrics(total_rejected_tokens).sum().item()
        if total_chosen_tokens > 0:
            self._metrics[mode]["logits/chosen"].append(total_chosen_logits / total_chosen_tokens)
        if total_rejected_tokens > 0:
            self._metrics[mode]["logits/rejected"].append(total_rejected_logits / total_rejected_tokens)

        all_num_chosen = self.accelerator.gather_for_metrics(num_chosen).sum().item()
        all_num_rejected = self.accelerator.gather_for_metrics(num_rejected).sum().item()

        if all_num_chosen > 0:
            self._metrics[mode]["rewards/chosen"].append(
                self.accelerator.gather_for_metrics(chosen_rewards.nansum()).nansum().item() / all_num_chosen
            )
            self._metrics[mode]["logps/chosen"].append(
                self.accelerator.gather_for_metrics(chosen_logps.nansum()).nansum().item() / all_num_chosen
            )

        if all_num_rejected > 0:
            self._metrics[mode]["rewards/rejected"].append(
                self.accelerator.gather_for_metrics(rejected_rewards.nansum()).nansum().item() / all_num_rejected
            )
            self._metrics[mode]["logps/rejected"].append(
                self.accelerator.gather_for_metrics(rejected_logps.nansum()).nansum().item() / all_num_rejected
            )

        if all_num_chosen > 0 and all_num_rejected > 0:
            self._metrics[mode]["rewards/margins"].append(
                self._metrics[mode]["rewards/chosen"][-1] - self._metrics[mode]["rewards/rejected"][-1]
            )

        loss = losses.nanmean()
        if self.aux_loss_enabled:
            aux_loss = outputs.aux_loss
            loss = loss + self.router_aux_loss_coef * aux_loss
            self._metrics[mode]["aux_loss"].append(self.accelerator.gather_for_metrics(aux_loss).mean().item())

        return (loss, outputs) if return_outputs else loss

    def evaluate(
        self,
        eval_dataset: Dataset
        | IterableDataset
        | DatasetDict
        | IterableDatasetDict
        | dict[str, Dataset | IterableDataset]
        | None = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        # When a dataset is passed directly to `evaluate` (e.g. a held-out test set), preprocess it the same way
        # `__init__` does, so that `evaluate` accepts the same dataset types as the trainer. `_prepare_dataset` is
        # idempotent: it skips datasets that are already tokenized. A `str` selects a dataset that was already prepared
        # at init time, so it's left untouched.
        if not self._is_vision_dataset and eval_dataset is not None and not isinstance(eval_dataset, str):
            # Full fine-tuning with no `ref_model` uses `self.model` as the reference, which is only valid before
            # training. After a step (`global_step > 0`) it's the trained policy, so we can't precompute a correct
            # reference here. (PEFT is exempt: the reference is recovered by disabling the adapter.) Checked before
            # tokenizing so we fail fast.
            if (
                self.precompute_ref_logps
                and self.ref_model is None
                and not is_peft_model(self.model)
                and self.state.global_step > 0
            ):
                raise ValueError(
                    "Cannot compute reference log-probs for a dataset passed to `evaluate()` after training has "
                    "started, because `precompute_ref_log_probs=True` and no `ref_model` was provided (full "
                    "fine-tuning). In this setup the reference model is not kept in memory, so it is only available "
                    "before training. Provide this dataset as `eval_dataset` at initialization, pass an explicit "
                    "`ref_model`, or set `precompute_ref_log_probs=False`."
                )
            if isinstance(eval_dataset, dict):
                eval_dataset = {
                    key: self._prepare_dataset(dataset, self.processing_class, self.args, key)
                    for key, dataset in eval_dataset.items()
                }
            else:
                eval_dataset = self._prepare_dataset(eval_dataset, self.processing_class, self.args, "eval")
            # With `precompute_ref_log_probs`, `_compute_loss` reads the reference log-probs from the batch, so they
            # must be precomputed here as well, mirroring `__init__`.
            if self.precompute_ref_logps:
                batch_size = self.args.precompute_ref_batch_size or self.args.per_device_eval_batch_size
                if isinstance(eval_dataset, dict):
                    eval_dataset = {
                        name: self._precompute_ref_logps(dataset, name, batch_size)
                        for name, dataset in eval_dataset.items()
                    }
                else:
                    eval_dataset = self._precompute_ref_logps(eval_dataset, "eval", batch_size)
            # Call `super().evaluate()` once per split ourselves instead of handing the whole dict to it: `Trainer.evaluate`
            # would otherwise recurse into `self.evaluate` per split, re-entering this override on an already-prepared dataset.
            if isinstance(eval_dataset, dict):
                metrics = {}
                for name, dataset in eval_dataset.items():
                    metrics.update(
                        super().evaluate(
                            eval_dataset=dataset,
                            ignore_keys=ignore_keys,
                            metric_key_prefix=f"{metric_key_prefix}_{name}",
                        )
                    )
                return metrics
        return super().evaluate(
            eval_dataset=eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        try:
            if self.use_liger_kernel:
                # Under ZeRO-3, `lm_head.weight` is sharded and the fused loss reads it directly (bypassing the
                # module), so run the loss inside the engine's forward via `_forward_redirection` to arm the parameter
                # coordinator's gather/reduce hooks.
                deepspeed_plugin = self.accelerator.state.deepspeed_plugin
                is_zero3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
                unwrapped_model = self.accelerator.unwrap_model(model)
                if is_zero3 or self.is_fsdp_enabled:
                    return self._forward_redirection(
                        model, unwrapped_model, self._compute_loss_liger, unwrapped_model, inputs, return_outputs
                    )
                return self._compute_loss_liger(unwrapped_model, inputs, return_outputs)
            return self._compute_loss(model, inputs, return_outputs)
        except ValueError as e:
            if "Image features and image tokens do not match" in str(e) and self.args.max_length is not None:
                raise ValueError(
                    f"The current `max_length` ({self.args.max_length}) is too short and causes image placeholder "
                    f"tokens in `input_ids` to be truncated, while the corresponding image features remain intact. "
                    f"Please increase `max_length` or set it to `None` to disable truncation."
                ) from e
            raise

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
        logs.update(metrics)
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
