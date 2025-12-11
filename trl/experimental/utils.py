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

# This file contains utility classes and functions that are used across more than one experimental trainer or feature.

from dataclasses import dataclass
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizerBase

from ..trainer.utils import pad


@dataclass
class DPODataCollatorWithPadding:
    r"""
    DPO DataCollator class that pads the tokenized inputs to the maximum length of the batch.

    Args:
        pad_token_id (`int` defaults to 0):
            The tokenizer's pad_token_id.
        label_pad_token_id (`int`, defaults to -100):
            The label used for masking.
        is_encoder_decoder (`bool` or `None`, `optional`, defaults to `None`):
            Whether you model has an encoder_decoder architecture.
    """

    pad_token_id: int = 0
    label_pad_token_id: int = -100
    is_encoder_decoder: bool | None = False

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        # first, pad everything to the same length
        padded_batch = {}
        for k in features[0].keys():
            if k.endswith(("_input_ids", "_attention_mask", "_labels", "_pixel_values")):
                if self.is_encoder_decoder:
                    to_pad = [torch.LongTensor(ex[k]) for ex in features]

                    if (k.startswith("prompt")) and (k.endswith("input_ids")):
                        if self.pad_token_id is None:
                            raise ValueError(
                                "Padding is enabled, but the tokenizer is not configured with a padding token."
                                " Explicitly set `tokenizer.pad_token` (e.g. `tokenizer.pad_token = tokenizer.eos_token`)"
                                " before calling the trainer."
                            )
                        padding_value = self.pad_token_id
                    elif k.endswith("_attention_mask"):
                        padding_value = 0
                    elif k.startswith(("chosen", "rejected", "completion")) or ("decoder" in k):
                        padding_value = self.label_pad_token_id
                    else:
                        raise ValueError(f"Unexpected key in batch '{k}'")
                    padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                else:
                    # Set padding value based on the key
                    if k.endswith("_input_ids"):
                        if self.pad_token_id is None:
                            raise ValueError(
                                "Padding is enabled, but the tokenizer is not configured with a padding token."
                                " Explicitly set `tokenizer.pad_token` (e.g. `tokenizer.pad_token = tokenizer.eos_token`)"
                                " before calling the trainer."
                            )
                        padding_value = self.pad_token_id
                    elif k.endswith("_labels"):
                        padding_value = self.label_pad_token_id
                    elif k.endswith("_attention_mask"):
                        padding_value = 0
                    elif k.endswith("_pixel_values"):
                        padding_value = 0  # TODO: check if this is correct
                    else:
                        raise ValueError(f"Unexpected key in batch '{k}'")

                    # Set padding side based on the key
                    if k in ["prompt_input_ids", "prompt_attention_mask"]:
                        padding_side = "left"
                    else:
                        padding_side = "right"

                    # Set the dtype
                    if k.endswith("_pixel_values"):
                        dtype = torch.float32  # will be downcasted if necessary by the Trainer
                    else:
                        dtype = torch.int64

                    # Convert to tensor and pad
                    to_pad = [torch.tensor(ex[k], dtype=dtype) for ex in features]
                    padded_batch[k] = pad(to_pad, padding_value=padding_value, padding_side=padding_side)
            elif k.endswith("_logps"):
                # the cached reference model logprobs
                padded_batch[k] = torch.tensor([ex[k] for ex in features])
            else:
                padded_batch[k] = [ex[k] for ex in features]

        return padded_batch


@dataclass
class DataCollatorForChatML:
    """
    Data collator for ChatML format datasets.
    """

    tokenizer: PreTrainedTokenizerBase
    ignore_index: int = -100
    max_length: int = None
    prompt_key: str = "prompt"
    messages_key: str = "messages"

    def __post_init__(self):
        if self.tokenizer.pad_token_id is None:
            raise ValueError("The tokenizer does not have a pad token. Please set `pad_token_id` in the tokenizer.")
        if self.max_length is None:
            # set a sensible default
            self.max_length = min(self.tokenizer.model_max_length, 1024)

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_ids = []
        attention_mask = []
        prompts_input_ids = []
        prompt_attention_mask = []
        labels = []

        for example in examples:
            formatted_prompt = example.get(self.prompt_key, None)
            if formatted_prompt is None:
                prompt = example[self.messages_key][:-1]
                formatted_prompt = self.tokenizer.apply_chat_template(
                    prompt, tokenize=False, add_generation_prompt=True
                )

            if "input_ids" not in example:
                message = example[self.messages_key]
                formatted_message = self.tokenizer.apply_chat_template(
                    message, tokenize=False, add_generation_prompt=False
                )

                tokenized_message = self.tokenizer(
                    formatted_message,
                    truncation=False,
                    padding=False,
                    return_tensors=None,
                    add_special_tokens=False,
                    return_offsets_mapping=True,
                )
                message_input_ids_full = tokenized_message["input_ids"]
                offsets = tokenized_message.get("offset_mapping")

                if offsets is not None:
                    prompt_char_len = len(formatted_prompt)
                    completion_start_idx_full = next(
                        (idx for idx, (start, _) in enumerate(offsets) if start >= prompt_char_len),
                        len(message_input_ids_full),
                    )
                else:
                    tokenized_prompt_full = self.tokenizer(
                        formatted_prompt,
                        truncation=False,
                        padding=False,
                        return_tensors=None,
                        add_special_tokens=False,
                    )
                    completion_start_idx_full = len(tokenized_prompt_full["input_ids"])

                prompt_tokens_full = message_input_ids_full[:completion_start_idx_full]
                completion_input_ids_full = message_input_ids_full[completion_start_idx_full:]

                if self.max_length is not None and len(message_input_ids_full) > self.max_length:
                    completion_ids = completion_input_ids_full
                    if len(completion_ids) >= self.max_length:
                        completion_ids = completion_ids[-self.max_length :]
                        prompt_ids = []
                    else:
                        max_prompt_tokens = self.max_length - len(completion_ids)
                        prompt_ids = prompt_tokens_full[-max_prompt_tokens:] if max_prompt_tokens > 0 else []
                    message_input_ids = prompt_ids + completion_ids
                else:
                    message_input_ids = message_input_ids_full
                    prompt_ids = prompt_tokens_full

                input_ids.append(message_input_ids)
                attention_mask.append([1] * len(message_input_ids))
                current_prompt_ids = prompt_ids
            else:
                message_input_ids = example["input_ids"]
                input_ids.append(message_input_ids)
                if "attention_mask" in example:
                    attention_mask.append(example["attention_mask"])
                else:
                    attention_mask.append([1] * len(message_input_ids))

                tokenized_prompt = self.tokenizer(
                    formatted_prompt,
                    truncation=True,
                    max_length=len(message_input_ids),
                    padding=False,
                    return_tensors=None,
                    add_special_tokens=False,
                )
                current_prompt_ids = tokenized_prompt["input_ids"]

            prompts_input_ids.append(current_prompt_ids)
            prompt_attention_mask.append([1] * len(current_prompt_ids))

            label = [self.ignore_index] * len(input_ids[-1])
            completion_start_idx = len(current_prompt_ids)
            label[completion_start_idx:] = input_ids[-1][completion_start_idx:]
            labels.append(label)

        # convert to list of tensors and pad
        input_ids = [torch.tensor(ids, dtype=torch.long) for ids in input_ids]
        attention_mask = [torch.tensor(mask, dtype=torch.long) for mask in attention_mask]
        labels = [torch.tensor(label, dtype=torch.long) for label in labels]
        input_ids = pad(input_ids, padding_side="left", padding_value=self.tokenizer.pad_token_id)
        attention_mask = pad(attention_mask, padding_side="left", padding_value=0)
        labels = pad(labels, padding_side="left", padding_value=self.ignore_index)

        prompts_input_ids = [torch.tensor(ids, dtype=torch.long) for ids in prompts_input_ids]
        prompt_attention_mask = [torch.tensor(mask, dtype=torch.long) for mask in prompt_attention_mask]
        prompts_input_ids = pad(prompts_input_ids, padding_side="left", padding_value=self.tokenizer.pad_token_id)
        prompt_attention_mask = pad(prompt_attention_mask, padding_side="left", padding_value=0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "prompts": prompts_input_ids,
            "prompt_attention_mask": prompt_attention_mask,
        }


def add_bos_token_if_needed(
    bos_token_id: int | None,
    prompt_len_input_ids: int,
    prompt_tokens: dict[str, list[int]],
    chosen_prompt_len_input_ids: int,
    chosen_tokens: dict[str, list[int]],
    rejected_prompt_len_input_ids: int,
    rejected_tokens: dict[str, list[int]],
):
    if bos_token_id is not None:
        if prompt_len_input_ids == 0 or bos_token_id != prompt_tokens["prompt_input_ids"][0]:
            prompt_tokens["prompt_input_ids"] = [bos_token_id] + prompt_tokens["prompt_input_ids"]
            prompt_tokens["prompt_attention_mask"] = [1] + prompt_tokens["prompt_attention_mask"]
        if chosen_prompt_len_input_ids == 0 or bos_token_id != chosen_tokens["prompt_input_ids"][0]:
            chosen_tokens["prompt_input_ids"] = [bos_token_id] + chosen_tokens["prompt_input_ids"]
            chosen_tokens["prompt_attention_mask"] = [1] + chosen_tokens["prompt_attention_mask"]
        if rejected_prompt_len_input_ids == 0 or bos_token_id != rejected_tokens["prompt_input_ids"][0]:
            rejected_tokens["prompt_input_ids"] = [bos_token_id] + rejected_tokens["prompt_input_ids"]
            rejected_tokens["prompt_attention_mask"] = [1] + rejected_tokens["prompt_attention_mask"]
    return prompt_tokens, chosen_tokens, rejected_tokens


def add_eos_token_if_needed(
    eos_token_id: int, chosen_tokens: dict[str, list[int]], rejected_tokens: dict[str, list[int]]
):
    if len(chosen_tokens["input_ids"]) == 0 or eos_token_id != chosen_tokens["input_ids"][-1]:
        chosen_tokens["input_ids"].append(eos_token_id)
        chosen_tokens["attention_mask"].append(1)
    if len(rejected_tokens["input_ids"]) == 0 or eos_token_id != rejected_tokens["input_ids"][-1]:
        rejected_tokens["input_ids"].append(eos_token_id)
        rejected_tokens["attention_mask"].append(1)
    return chosen_tokens, rejected_tokens
