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

import functools
from collections.abc import Sequence
from typing import Any, Callable, Optional, TypeVar, Union

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.types
from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizerBase


DatasetType = TypeVar("DatasetType", Dataset, DatasetDict)


def is_conversational(example: dict[str, Any]) -> bool:
    r"""
    Check if the example is in a conversational format.

    Args:
        example (`dict[str, Any]`):
            A single data entry of a dataset. The example can have different keys depending on the
            dataset type.

    Returns:
        `bool`:
            `True` if the data is in a conversational format, `False` otherwise.

    Examples:

    ```python
    >>> example = {"prompt": [{"role": "user", "content": "What color is the sky?"}]}
    >>> is_conversational(example)
    True
    >>> example = {"prompt": "The sky is"})
    >>> is_conversational(example)
    False
    ```
    """
    supported_keys = ["prompt", "chosen", "rejected", "completion", "messages"]
    example_keys = {key for key in example.keys() if key in supported_keys}

    # It must have one of the supported keys
    if example_keys:
        key = example_keys.pop()  # take the first supported key
        maybe_messages = example[key]
        # It must be a list of messages,
        if isinstance(maybe_messages, list):
            maybe_message = maybe_messages[0]
            # Each message must a list of dictionaries with keys "role" and "content"
            if isinstance(maybe_message, dict) and "role" in maybe_message and "content" in maybe_message:
                return True

    return False


def apply_chat_template(
    example: dict[str, list[dict[str, str]]],
    tokenizer: PreTrainedTokenizerBase,
    tools: Optional[list[Union[dict, Callable]]] = None,
) -> dict[str, str]:
    r"""
    Apply a chat template to a conversational example along with the schema for a list of functions in `tools`.

    For more details, see [`maybe_apply_chat_template`].
    """
    # Check that the example has the correct keys
    supported_keys = ["prompt", "chosen", "rejected", "completion", "messages", "label"]
    example_keys = {key for key in example.keys() if key in supported_keys}
    if example_keys not in [
        {"messages"},  # language modeling
        {"prompt"},  # prompt-only
        {"prompt", "completion"},  # prompt-completion
        {"prompt", "chosen", "rejected"},  # preference
        {"chosen", "rejected"},  # preference with implicit prompt
        {"prompt", "completion", "label"},  # unpaired preference
    ]:
        raise KeyError(f"Invalid keys in the example: {example_keys}")

    # Apply the chat template to the whole conversation
    if "messages" in example:
        messages = tokenizer.apply_chat_template(example["messages"], tools=tools, tokenize=False)

    # Apply the chat template to the prompt, adding the generation prompt
    if "prompt" in example:
        last_role = example["prompt"][-1]["role"]
        if last_role == "user":
            add_generation_prompt = True
            continue_final_message = False
        elif last_role == "assistant":
            add_generation_prompt = False
            continue_final_message = True
        else:
            raise ValueError(f"Invalid role in the last message: {last_role}")
        prompt = tokenizer.apply_chat_template(
            example["prompt"],
            tools=tools,
            continue_final_message=continue_final_message,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    # Apply the chat template to the entire prompt + completion
    if "prompt" in example:  # explicit prompt and prompt-completion case
        if "chosen" in example:
            prompt_chosen = tokenizer.apply_chat_template(
                example["prompt"] + example["chosen"], tools=tools, tokenize=False
            )
            chosen = prompt_chosen[len(prompt) :]
        if "rejected" in example and "prompt" in example:  # explicit prompt
            prompt_rejected = tokenizer.apply_chat_template(
                example["prompt"] + example["rejected"], tools=tools, tokenize=False
            )
            rejected = prompt_rejected[len(prompt) :]
        if "completion" in example:
            prompt_completion = tokenizer.apply_chat_template(
                example["prompt"] + example["completion"], tools=tools, tokenize=False
            )
            completion = prompt_completion[len(prompt) :]
    else:  # implicit prompt case
        if "chosen" in example:
            chosen = tokenizer.apply_chat_template(example["chosen"], tools=tools, tokenize=False)
        if "rejected" in example:
            rejected = tokenizer.apply_chat_template(example["rejected"], tools=tools, tokenize=False)

    # Ensure that the prompt is the initial part of the prompt-completion string
    if "prompt" in example:
        error_message = (
            "The chat template applied to the prompt + completion does not start with the chat template applied to "
            "the prompt alone. This can indicate that the chat template is not supported by TRL."
            "\n**Prompt**:\n{}\n\n**Prompt + Completion**:\n{}"
        )
        if "chosen" in example and not prompt_chosen.startswith(prompt):
            raise ValueError(error_message.format(prompt, prompt_chosen))
        if "rejected" in example and not prompt_rejected.startswith(prompt):
            raise ValueError(error_message.format(prompt, prompt_rejected))
        if "completion" in example and not prompt_completion.startswith(prompt):
            raise ValueError(error_message.format(prompt, prompt_completion))

    # Extract the completion by removing the prompt part from the prompt-completion string
    output = {}
    if "messages" in example:
        output["text"] = messages
    if "prompt" in example:
        output["prompt"] = prompt
    if "chosen" in example:
        output["chosen"] = chosen
    if "rejected" in example:
        output["rejected"] = rejected
    if "completion" in example:
        output["completion"] = completion
    if "label" in example:
        output["label"] = example["label"]

    return output


def maybe_apply_chat_template(
    example: dict[str, list[dict[str, str]]],
    tokenizer: PreTrainedTokenizerBase,
    tools: Optional[list[Union[dict, Callable]]] = None,
) -> dict[str, str]:
    r"""
    If the example is in a conversational format, apply a chat template to it.

    Args:
        example (`dict[str, list[dict[str, str]]`):
            Dictionary representing a single data entry of a conversational dataset. Each data entry can have different
            keys depending on the dataset type. The supported dataset types are:

                - Language modeling dataset: `"messages"`.
                - Prompt-only dataset: `"prompt"`.
                - Prompt-completion dataset: `"prompt"` and `"completion"`.
                - Preference dataset: `"prompt"`, `"chosen"`, and `"rejected"`.
                - Preference dataset with implicit prompt: `"chosen"` and `"rejected"`.
                - Unpaired preference dataset: `"prompt"`, `"completion"`, and `"label"`.

            For keys `"messages"`, `"prompt"`, `"chosen"`, `"rejected"`, and `"completion"`, the values are lists of
            messages, where each message is a dictionary with keys `"role"` and `"content"`.
        tokenizer (`PreTrainedTokenizerBase`):
            Tokenizer to apply the chat template with.
        tools (`list[Union[dict, Callable]]` or `None`, *optional*, defaults to `None`):
            A list of tools (callable functions) that will be accessible to the model.
            If the template does not support function calling, this argument will have no effect

    Returns:
        `dict[str, str]`:
            Formatted example with the chat template applied.

    Notes:
        - This function does not alter the keys, except for Language modeling dataset, where `"messages"` is replaced
        by `"text"`.

        - In case of prompt-only data, if the last role is `"user"`, the generation prompt is added to the prompt.
        Else, if the last role is `"assistant"`, the final message is continued.

    Example:

    ```python
    >>> from transformers import AutoTokenizer
    >>> tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")
    >>> example = {
    ...     "prompt": [{"role": "user", "content": "What color is the sky?"}],
    ...     "completion": [{"role": "assistant", "content": "It is blue."}]
    ... }
    >>> apply_chat_template(example, tokenizer)
    {'prompt': '<|user|>\nWhat color is the sky?<|end|>\n<|assistant|>\n', 'completion': 'It is blue.<|end|>\n<|endoftext|>'}
    ```
    """
    if is_conversational(example):
        return apply_chat_template(example, tokenizer, tools)
    else:
        return example


def _unpair_row(examples: list[dict[str, list[dict[str, str]]]]) -> list[dict[str, list[dict[str, str]]]]:
    batch_size = len(examples["chosen"])
    new_rows = {
        "completion": examples["chosen"] + examples["rejected"],
        "label": [True] * batch_size + [False] * batch_size,
    }
    if "prompt" in examples:
        new_rows["prompt"] = examples["prompt"] + examples["prompt"]
    return new_rows


def unpair_preference_dataset(
    dataset: DatasetType, num_proc: Optional[int] = None, desc: Optional[str] = None
) -> DatasetType:
    r"""
    Unpair a preference dataset.

    Args:
        dataset (`Dataset` or `DatasetDict`):
            Preference dataset to unpair. The dataset must have columns `"chosen"`, `"rejected"` and optionally
            `"prompt"`.
        num_proc (`int` or `None`, *optional*, defaults to `None`):
            Number of processes to use for processing the dataset.
        desc (`str` or `None`, *optional*, defaults to `None`):
            Meaningful description to be displayed alongside with the progress bar while mapping examples.

    Returns:
        `Dataset`: The unpaired preference dataset.

    Example:

    ```python
    >>> from datasets import Dataset
    >>> dataset_dict = {
    ...     "prompt": ["The sky is", "The sun is"]
    ...     "chosen": [" blue.", "in the sky."],
    ...     "rejected": [" green.", " in the sea."]
    ... }
    >>> dataset = Dataset.from_dict(dataset_dict)
    >>> dataset = unpair_preference_dataset(dataset)
    >>> dataset
    Dataset({
        features: ['prompt', 'completion', 'label'],
        num_rows: 4
    })
    >>> dataset[0]
    {'prompt': 'The sky is', 'completion': ' blue.', 'label': True}
    ```
    """
    return dataset.map(_unpair_row, batched=True, remove_columns=["chosen", "rejected"], num_proc=num_proc, desc=desc)


def maybe_unpair_preference_dataset(
    dataset: DatasetType, num_proc: Optional[int] = None, desc: Optional[str] = None
) -> DatasetType:
    r"""
    Unpair a preference dataset if it is paired.

    Args:
        dataset (`Dataset` or `DatasetDict`):
            Preference dataset to unpair. The dataset must have columns `"chosen"`, `"rejected"` and optionally
            `"prompt"`.
        num_proc (`int` or `None`, *optional*, defaults to `None`):
            Number of processes to use for processing the dataset.
        desc (`str` or `None`, *optional*, defaults to `None`):
            Meaningful description to be displayed alongside with the progress bar while mapping examples.

    Returns:
        `Dataset` or `DatasetDict`: The unpaired preference dataset if it was paired, otherwise the original dataset.

    Example:

    ```python
    >>> from datasets import Dataset
    >>> dataset_dict = {
    ...     "prompt": ["The sky is", "The sun is"]
    ...     "chosen": [" blue.", "in the sky."],
    ...     "rejected": [" green.", " in the sea."]
    ... }
    >>> dataset = Dataset.from_dict(dataset_dict)
    >>> dataset = unpair_preference_dataset(dataset)
    >>> dataset
    Dataset({
        features: ['prompt', 'completion', 'label'],
        num_rows: 4
    })
    >>> dataset[0]
    {'prompt': 'The sky is', 'completion': ' blue.', 'label': True}
    ```
    """
    if isinstance(dataset, DatasetDict):
        column_names = dataset[list(dataset.keys())[0]].column_names
    else:
        column_names = dataset.column_names
    if "chosen" in column_names and "rejected" in column_names:
        return unpair_preference_dataset(dataset, num_proc=num_proc, desc=desc)
    else:
        return dataset


def extract_prompt(example: dict[str, Sequence]) -> dict[str, Sequence]:
    r"""
    Extracts the shared prompt from a preference data example, where the prompt is implicit within both
    the chosen and rejected completions.

    For more details, see [`maybe_extract_prompt`].
    """
    for idx in range(min(len(example["chosen"]), len(example["rejected"]))):
        if example["chosen"][idx] != example["rejected"][idx]:
            if example["chosen"][idx - 1] == " ":  # remove space before the prompt
                idx -= 1
            break
    return {
        "prompt": example["chosen"][:idx],
        "chosen": example["chosen"][idx:],
        "rejected": example["rejected"][idx:],
    }


def maybe_extract_prompt(example: dict[str, list]) -> dict[str, list]:
    r"""
    Extracts the shared prompt from a preference data example, where the prompt is implicit within both
    the chosen and rejected completions.

    If the example already contains a `"prompt"` key, the function returns the example as is. Else, the function
    identifies the longest common sequence (prefix) of conversation turns between the "chosen" and "rejected"
    completions and extracts this as the prompt. It then removes this prompt from the respective "chosen" and
    "rejected" completions.

    Args:
        example (`dict[str, list]`):
            A dictionary representing a single data entry in the preference dataset. It must contain the keys
            `"chosen"` and `"rejected"`, where each value is either conversational or standard (`str`).

    Returns:
        `dict[str, list]`: A dictionary containing:
            - `"prompt"`: The longest common prefix between the "chosen" and "rejected" completions.
            - `"chosen"`: The remainder of the "chosen" completion, with the prompt removed.
            - `"rejected"`: The remainder of the "rejected" completion, with the prompt removed.

    Examples:

    ```python
    >>> example = {
    ...     "chosen": [
    ...         {"role": "user", "content": "What color is the sky?"},
    ...         {"role": "assistant", "content": "It is blue."}
    ...     ],
    ...     "rejected": [
    ...         {"role": "user", "content": "What color is the sky?"},
    ...         {"role": "assistant", "content": "It is green."}
    ...     ]
    ... }
    >>> extract_prompt(example)
    {'prompt': [{'role': 'user', 'content': 'What color is the sky?'}],
     'chosen': [{'role': 'assistant', 'content': 'It is blue.'}],
     'rejected': [{'role': 'assistant', 'content': 'It is green.'}]}
    ```

    Or, with the `map` method of `datasets.Dataset`:

    ```python
    >>> from trl import extract_prompt
    >>> from datasets import Dataset
    >>> dataset_dict = {
    ...     "chosen": [
    ...         [
    ...             {"role": "user", "content": "What color is the sky?"},
    ...             {"role": "assistant", "content": "It is blue."},
    ...         ],
    ...         [
    ...             {"role": "user", "content": "Where is the sun?"},
    ...             {"role": "assistant", "content": "In the sky."},
    ...         ],
    ...     ],
    ...     "rejected": [
    ...         [
    ...             {"role": "user", "content": "What color is the sky?"},
    ...             {"role": "assistant", "content": "It is green."},
    ...         ],
    ...         [
    ...             {"role": "user", "content": "Where is the sun?"},
    ...             {"role": "assistant", "content": "In the sea."},
    ...         ],
    ...     ],
    ... }
    >>> dataset = Dataset.from_dict(dataset_dict)
    >>> dataset = dataset.map(extract_prompt)
    >>> dataset[0]
    {'prompt': [{'role': 'user', 'content': 'What color is the sky?'}],
     'chosen': [{'role': 'assistant', 'content': 'It is blue.'}],
     'rejected': [{'role': 'assistant', 'content': 'It is green.'}]}
    ```
    """
    # Some dataset add a `"prompt"` column, even though the prompt is implicit and included in the "chosen" and
    # "rejected" completions. E.g.:
    # {"prompt": "What color is the sky?",
    #  "chosen": [{"role": "user", "content": "What color is the sky?"}, {"role": "assistant", "content": "It is blue."}],
    #  "rejected": [{"role": "user", "content": "What color is the sky?"}, {"role": "assistant", "content": "It is green."}]}
    # That's why we check if the prompt is also conversational before deciding not to extract it.
    if "chosen" not in example or "rejected" not in example:  # not a preference example
        return example
    if "prompt" in example:
        # Both conversational or both non-conversational
        chosen_conv = is_conversational({"chosen": example["chosen"]})
        prompt_conv = is_conversational({"prompt": example["prompt"]})
        if (chosen_conv and prompt_conv) or (not chosen_conv and not prompt_conv):
            return example
    return extract_prompt({"chosen": example["chosen"], "rejected": example["rejected"]})


def pack_examples(examples: dict[str, list[list]], seq_length: int) -> dict[str, list[list]]:
    """
    Pack examples into chunks of size `seq_length`.

    Args:
        examples (`dict[str, list[list]]`):
            Dictionary of examples with keys as strings and values as lists of lists.
        seq_length (`int`):
            Maximum sequence length.

    Returns:
        `dict[str, list[list]]`: Dictionary of examples with keys as strings and values as lists of lists.

    Example:

    ```python
    >>> from trl import pack_examples
    >>> examples = {
    ...     "input_ids": [[1, 2, 3], [4, 5, 6, 7], [8]],
    ...     "attention_mask": [[0, 1, 1], [0, 0, 1, 1], [1]],
    ... }
    >>> pack_examples(examples, seq_length=5)
    {'input_ids': [[1, 2, 3, 4, 5], [6, 7, 8]], 'attention_mask': [[0, 1, 1, 0, 0], [1, 1, 1]]}
    >>> pack_examples(examples, seq_length=2)
    {'input_ids': [[1, 2], [3, 4], [5, 6], [7, 8]], 'attention_mask': [[0, 1], [1, 0], [0, 1], [1, 1]]}
    ```
    """
    # Join  all the values into a single list
    examples = {k: sum(v, []) for k, v in examples.items()}
    # Split the values into chunks of size seq_length
    examples = {k: [v[i : i + seq_length] for i in range(0, len(v), seq_length)] for k, v in examples.items()}
    return examples


def pack_dataset(dataset: DatasetType, seq_length: int, map_kwargs: Optional[dict[str, Any]] = None) -> DatasetType:
    r"""
    Pack sequences in a dataset into chunks of size `seq_length`.

    Args:
        dataset (`Dataset` or `DatasetDict`):
            Dataset to pack
        seq_length (`int`):
            Target sequence length to pack to.
        map_kwargs (`dict` or `None`, *optional*, defaults to `None`):
            Additional keyword arguments to pass to the dataset's map method when packing examples.

    Returns:
        `Dataset` or `DatasetDict`: The dataset with packed sequences. The number of examples may
        decrease as sequences are combined.

    Example:
    ```python
    >>> from datasets import Dataset
    >>> examples = {
    ...     "input_ids": [[1, 2], [3, 4], [5, 6], [7]],
    ...     "attention_mask": [[1, 1], [0, 1], [1, 1], [1]],
    ... }
    >>> dataset = Dataset.from_dict(examples)
    >>> packed_dataset = pack_dataset(dataset, seq_length=4)
    >>> packed_dataset[:]
    {'input_ids': [[1, 2, 3, 4], [5, 6, 7]],
     'attention_mask': [[1, 1, 0, 1], [1, 1, 1]]}
    ```
    """
    if map_kwargs is None:
        map_kwargs = {}
    if isinstance(dataset, Dataset):
        # Fast packing with pyarrow
        def pack(examples):
            packed_columns = []
            for column in examples.columns:
                if pyarrow.types.is_list(column.type) or pyarrow.types.is_large_list(column.type):
                    if isinstance(column, pa.ChunkedArray):
                        column = column.combine_chunks()
                    offsets, values = column.offsets, column.values
                    values = values[offsets[0].as_py() : offsets[-1].as_py()]
                    num_elements = len(values)
                    dtype = offsets.type.to_pandas_dtype()  # np.int32 or np.int64
                    offsets = np.arange(0, num_elements, seq_length, dtype=dtype)
                    offsets = np.concatenate((offsets, [num_elements]))
                    column = type(column).from_arrays(offsets, values)
                packed_columns.append(column)
            return pa.Table.from_arrays(packed_columns, names=examples.column_names)

        dataset = dataset.with_format("arrow")
        dataset = dataset.map(pack, batched=True, **map_kwargs)
        dataset = dataset.with_format(None)
    else:
        dataset = dataset.map(
            functools.partial(pack_examples, seq_length=seq_length),
            batched=True,
            **map_kwargs,
        )
    return dataset


def truncate_dataset(
    dataset: DatasetType, max_length: int, map_kwargs: Optional[dict[str, Any]] = None
) -> DatasetType:
    r"""
    Truncate sequences in a dataset to a specifed `max_length`.

    Args:
        dataset (`Dataset` or `DatasetDict`):
            Dataset to truncate.
        seq_length (`int`):
            Maximum sequence length to truncate to.
        map_kwargs (`dict` or `None`, *optional*, defaults to `None`):
            Additional keyword arguments to pass to the dataset's map method when truncating examples.

    Returns:
        `Dataset` or `DatasetDict`: The dataset with truncated sequences.

    Example:
    ```python
    >>> from datasets import Dataset
    >>> examples = {
    ...     "input_ids": [[1, 2, 3], [4, 5, 6, 7], [8]],
    ...     "attention_mask": [[0, 1, 1], [0, 0, 1, 1], [1]],
    ... }
    >>> dataset = Dataset.from_dict(examples)
    >>> truncated_dataset = truncate_dataset(dataset, max_length=2)
    >>> truncated_dataset[:]
    {'input_ids': [[1, 2], [4, 5], [8]],
     'attention_mask': [[0, 1], [0, 0], [1]]}
    ```
    """
    if map_kwargs is None:
        map_kwargs = {}
    if isinstance(dataset, Dataset):
        # Fast truncation with pyarrow
        def truncate(examples):
            truncated_columns = []
            for column in examples.columns:
                if pyarrow.types.is_list(column.type) or pyarrow.types.is_large_list(column.type):
                    column = pc.list_slice(column, 0, max_length)
                truncated_columns.append(column)
            return pa.Table.from_arrays(truncated_columns, names=examples.column_names)

        dataset = dataset.with_format("arrow")
        dataset = dataset.map(truncate, batched=True, **map_kwargs)
        dataset = dataset.with_format(None)
    else:

        def truncate(examples):
            truncated_examples = {}
            for key, column in examples.items():
                if column and isinstance(column[0], list):
                    column = [val[:max_length] for val in column]
                truncated_examples[key] = column
            return truncated_examples

        dataset = dataset.map(
            truncate,
            batched=True,
            **map_kwargs,
        )
    return dataset


def maybe_convert_to_chatml(example: dict[str, list]) -> dict[str, list]:
    """
    Convert a conversational dataset with fields `from` and `value` to ChatML format.

    This function modifies conversational data to align with OpenAI's ChatML format:
    - Replaces the key `"from"` with `"role"` in message dictionaries.
    - Replaces the key `"value"` with `"content"` in message dictionaries.
    - Renames `"conversations"` to `"messages"` for consistency with ChatML.

    Args:
        example (`dict[str, list]`):
            A single data entry containing a list of messages.

    Returns:
        `dict[str, list]`:
            Example reformatted to ChatML style.

    Example:
    ```python
    >>> from trl import maybe_convert_to_chatml
    >>> example = {
    ...     "conversations": [
    ...         {"from": "user", "value": "What color is the sky?"},
    ...         {"from": "assistant", "value": "It is blue."}
    ...     ]
    ... }
    >>> maybe_convert_to_chatml(example)
    {'messages': [{'role': 'user', 'content': 'What color is the sky?'},
                  {'role': 'assistant', 'content': 'It is blue.'}]}
    ```
    """
    # List of possible keys containing message lists
    for key in ["prompt", "completion", "chosen", "rejected", "messages", "conversations"]:
        if key in example and isinstance(example[key], list):
            messages = example[key]
            for message in messages:
                if isinstance(message, dict):
                    if "from" in message:
                        message["role"] = message.pop("from")
                    if "value" in message:
                        message["content"] = message.pop("value")

    # Rename "conversations" to "messages"
    if "conversations" in example:
        example["messages"] = example.pop("conversations")

    return example
