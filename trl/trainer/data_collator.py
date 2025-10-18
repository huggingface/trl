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

from dataclasses import dataclass
from typing import Any, Optional, Union

import torch
from transformers.data.data_collator import DataCollatorMixin

from .utils import pad


@dataclass
class DataCollatorForLanguageModeling(DataCollatorMixin):
    """
    Data collator used for language modeling data. Inputs are dynamically padded to the maximum length of a batch.

    This collator expects each example in the input list to be a dictionary containing at least the `"input_ids"` key.
    If the input contains a `"completion_mask"`, it is used to set the labels to `-100` for tokens that are not in the
    completion. If `"assistant_masks"` are present, they are used to set the labels to `-100` for tokens that are not
    in the assistant part of the sequence. The collator returns a dictionary containing the following keys:
    - `"input_ids"`: Tensor of input IDs, padded to the maximum length of the batch.
    - `"attention_mask"`: Tensor of attention mask, padded to the maximum length of the batch.
    - `"position_ids"`: Tensor of position IDs, padded to the maximum length of the batch.
    - `"labels"`: Tensor of labels, padded to the maximum length of the batch. If `completion_only_loss` is set to
    `True`, tokens that are not in the completion are set to -100. If `assistant_masks` are present, tokens that are
    not in the assistant part of the sequence are set to -100.

    Args:
        pad_token_id (`int`):
            Token ID to use for padding.
        completion_only_loss (`bool`, *optional*, defaults to `True`):
            When the input contains a completion mask (`completion_mask`), the labels are set to -100 for the tokens
            that are no in the completion.
        padding_free (`bool`, *optional*, defaults to `False`):
            If set to `True`, the sequences will be flattened into a single sequence, and the position IDs will be
            generated accordingly. The attention mask will be set to 1 for all tokens.
        pad_to_multiple_of (`int` or `None`, *optional*, defaults to `None`):
            If set, the sequences will be padded to a multiple of this value.
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            Type of Tensor to return. Only `"pt"` is currently supported.

    Examples:
    ```python
    >>> from trl import DataCollatorForLanguageModeling

    >>> collator = DataCollatorForLanguageModeling(pad_token_id=0)
    >>> examples = [{"input_ids": [1, 2, 3]}, {"input_ids": [4, 5]}]
    >>> collator(examples)
    {'input_ids': tensor([[  1,  2,  3],
                          [  4,  5,  0]]),
     'attention_mask': tensor([[  1,  1,  1],
                               [  1,  1,  0]]),
     'position_ids': tensor([[0, 1, 2],
                             [0, 1, 0]]),
     'labels': tensor([[   1,    2,    3],
                       [   4,    5, -100]])}

    >>> # With completion mask
    >>> examples = [
    ...     {"input_ids": [1, 2, 3], "completion_mask": [0, 1, 1]},
    ...     {"input_ids": [4, 5], "completion_mask": [0, 1]},
    ... ]
    >>> collator(examples)
    {'input_ids': tensor([[  1,  2,  3],
                          [  4,  5,  0]]),
     'attention_mask': tensor([[  1,  1,  1],
                               [  1,  1,  0]]),
     'position_ids': tensor([[0, 1, 2],
                             [0, 1, 0]]),
     'labels': tensor([[-100,    2,    3],
                       [-100,    5, -100]])}

    >>> # With padding_free
    >>> collator = DataCollatorForLanguageModeling(pad_token_id=0, padding_free=True)
    >>> collator(examples)
    {'input_ids': tensor([[ 1, 2, 3, 4, 5]]),
     'attention_mask': tensor([[1, 1, 1, 1, 1]]),
     'position_ids': tensor([[0, 1, 2, 0, 1]]),
     'labels': tensor([[1, 2, 3, 4, 5]])}
    ```
    """

    pad_token_id: int
    completion_only_loss: bool = True
    padding_free: bool = False
    return_position_ids: bool = True
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def torch_call(self, examples: list[Union[list[int], Any, dict[str, Any]]]) -> dict[str, Any]:
        # Convert to tensor
        input_ids = [torch.tensor(example["input_ids"]) for example in examples]

        # Check if we have meaningful seq_lengths from packing (restarting sequences)
        has_packed_position_ids = self.return_position_ids and "seq_lengths" in examples[0] and self.padding_free

        # For packing with position_ids, we should NOT create attention_mask as it causes
        # FlashAttention to ignore position_ids and compute wrong cu_seq_lens from the all-1s mask
        if not has_packed_position_ids:
            attention_mask = [torch.ones_like(input_ids) for input_ids in input_ids]

        if self.return_position_ids:
            if "seq_lengths" in examples[0]:
                position_ids = self.get_position_ids_from_packed_seq_lengths(
                    [example["seq_lengths"] for example in examples]
                )
            else:
                position_ids = [torch.arange(len(ids)) for ids in input_ids]
        if "labels" in examples[0]:
            labels = [torch.tensor(example["labels"]) for example in examples]
        else:
            labels = [torch.tensor(example["input_ids"]) for example in examples]
        if self.completion_only_loss and "completion_mask" in examples[0]:
            completion_mask = [torch.tensor(example["completion_mask"]) for example in examples]
        if "assistant_masks" in examples[0]:
            assistant_masks = [torch.tensor(example["assistant_masks"]) for example in examples]

        # Pad
        output = {}
        if self.padding_free:
            output["input_ids"] = torch.cat(input_ids, dim=0).unsqueeze(0)
            if not has_packed_position_ids:
                output["attention_mask"] = torch.cat(attention_mask, dim=0).unsqueeze(0)
            if self.return_position_ids:
                output["position_ids"] = torch.cat(position_ids, dim=0).unsqueeze(0)
            output["labels"] = torch.cat(labels, dim=0).unsqueeze(0)
            if self.completion_only_loss and "completion_mask" in examples[0]:
                completion_mask = torch.cat(completion_mask, dim=0).unsqueeze(0)
                output["labels"][completion_mask == 0] = -100
            if "assistant_masks" in examples[0]:
                assistant_masks = torch.cat(assistant_masks, dim=0).unsqueeze(0)
                output["labels"][assistant_masks == 0] = -100
        else:
            output["input_ids"] = pad(
                input_ids,
                padding_value=self.pad_token_id,
                padding_side="right",
                pad_to_multiple_of=self.pad_to_multiple_of,
            )
            output["attention_mask"] = pad(
                attention_mask, padding_value=0, padding_side="right", pad_to_multiple_of=self.pad_to_multiple_of
            )
            if self.return_position_ids:
                output["position_ids"] = pad(
                    position_ids, padding_value=0, padding_side="right", pad_to_multiple_of=self.pad_to_multiple_of
                )
            output["labels"] = pad(
                labels, padding_value=-100, padding_side="right", pad_to_multiple_of=self.pad_to_multiple_of
            )
            if self.completion_only_loss and "completion_mask" in examples[0]:
                completion_mask = pad(
                    completion_mask, padding_value=0, padding_side="right", pad_to_multiple_of=self.pad_to_multiple_of
                )
                output["labels"][completion_mask == 0] = -100  # mask everything that is not in the completion
            if "assistant_masks" in examples[0]:
                assistant_masks = pad(
                    assistant_masks, padding_value=0, padding_side="right", pad_to_multiple_of=self.pad_to_multiple_of
                )
                output["labels"][assistant_masks == 0] = -100
        return output

    @staticmethod
    def get_position_ids_from_packed_seq_lengths(batch_seq_lengths: list[list[int]]) -> list[torch.Tensor]:
        """
        Get position IDs for packed sequences.

        Args:
            batch_seq_lengths (`list[list[int]]`):
                A list of lists containing the lengths of each individual document in the packed batch.

        Return:
            `list[torch.Tensor]`:
                A list of tensors containing the position IDs for each packed sequence.
        """
        # Get lengths per row
        example_lengths = [sum(seq_lengths) for seq_lengths in batch_seq_lengths]
        # Flat list of lengths
        batch_seq_lengths = torch.tensor(
            [seq_length for seq_lengths in batch_seq_lengths for seq_length in seq_lengths]
        )
        position_ids = torch.ones(sum(example_lengths), dtype=batch_seq_lengths.dtype)
        position_ids[0] = 0
        # Reset position ids to 0 at the start of each sequence
        position_ids[batch_seq_lengths[:-1].cumsum(0)] = -(batch_seq_lengths[:-1] - 1)
        position_ids = position_ids.cumsum(0)
        # Split back into one tensor per example
        return list(position_ids.split(example_lengths))
