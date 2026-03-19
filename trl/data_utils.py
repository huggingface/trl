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

import copy
from collections import defaultdict, deque
from collections.abc import Callable, Sequence
from itertools import takewhile
from typing import Any, Literal, TypeVar

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.types
from datasets import Dataset, DatasetDict, IterableDatasetDict
from transformers import PreTrainedTokenizerBase, ProcessorMixin


DatasetType = TypeVar("DatasetType", Dataset, DatasetDict)


def prepare_multimodal_messages(messages: list[dict[str, Any]], images: list) -> list[dict[str, Any]]:
    # docstyle-ignore  # because <Image> is not parsable in the code block
    """
    Convert messages into a structured multimodal format and inject the provided images into the message contents.

    Args:
        messages (`list[dict[str, Any]]`):
            Messages with `"role"`, `"content"` (or `"tool_calls"`). Content may be a raw string before transformation.
            List of messages with a `"role"` key (`"system"`, `"user"`, `"assistant"`, or `"tool"`) and a `"content"` key containing
            either a string or a list of structured blocks if already prepared. Optionally, the `"content"` might
            be `None` or not provided in favour of `"tool_calls"` in the `"assistant"` turns if applicable.
        images (`list`):
            List of image objects to insert. Can be empty if no images are included in the messages.

    Returns:
        `list[dict[str, Any]]`: A deep-copied list of messages where every `"content"` value is a list of structured
        content blocks, and all `"image"` placeholders are populated with the corresponding image objects. If the
        assistant turns contains `"tool_calls"`, then the `"content"` might be empty.

    Notes:
        - When the input `messages` isn't already in the structured format, (i.e., all `"content"` values are strings),
          the function transforms them into the structured format by wrapping text in `{"type": "text", "text": ...}`
          and inserting `{"type": "image"}` placeholders for the images *before* the first user message.
          If the number of placeholders does not match the number of provided images, an error is raised.
        - When the input `messages` contains either `"tool_calls"` in the `"assistant"` turns, or `"tool"` roles with
          `"content"` and `"name"` those are left as-is, since those don't require any specific handling for multimodal data.

    Example:
    ```python
    # Input
    [
        {"role": "user", "content": "What's in this image?"},
        {"role": "assistant", "content": "It looks like a cat."},
    ]

    # Output, one image provided
    [
        {"role": "user", "content": [{"type": "image", "image": <PIL.Image.Image>}, {"type": "text", "text": "What's in this image?"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "It looks like a cat."}]},
    ]
    ```
    """

    messages = copy.deepcopy(messages)  # avoid modifying the original messages

    # First, convert all messages to the structured format if needed, and insert image placeholders if needed
    images_included = False
    for message in messages:
        if message["role"] == "system":
            if isinstance(message["content"], str):  # if already prepared, the content will be a list
                message["content"] = [{"type": "text", "text": message["content"]}]
        elif message["role"] == "user":
            if isinstance(message["content"], str) and not images_included:
                image_entries = [{"type": "image"} for _ in range(len(images))]
                message["content"] = [*image_entries, {"type": "text", "text": message["content"]}]
                images_included = True
            elif isinstance(message["content"], str) and images_included:
                message["content"] = [{"type": "text", "text": message["content"]}]
        elif message["role"] == "assistant":
            if message.get("content") and isinstance(message["content"], str):
                message["content"] = [{"type": "text", "text": message["content"]}]
        elif message["role"] == "tool":
            # NOTE: `tool` contains `name` (name of the tool used) and `content` (output of the tool call as a string)
            # but there's no need to prepare it for multimodal specifically but rather leave it as-is
            continue
        else:
            raise ValueError(
                f"Invalid role in message: {message['role']}. Expected 'system', 'user', 'assistant', or 'tool'."
            )

    # Then, check that the number of image placeholders matches the number of images provided
    num_placeholders = sum(
        sum(1 for part in message["content"] if part["type"] == "image")
        for message in messages
        if message.get("content") and message["role"] != "tool"
    )
    if num_placeholders != len(images):
        raise ValueError(
            f"Number of images provided ({len(images)}) does not match number of image placeholders ({num_placeholders})."
        )

    # Then, fill in the actual images in the placeholders
    img_idx = 0
    for message in messages:
        if not message.get("content") or message["role"] == "tool":
            continue
        for part in message["content"]:
            if part["type"] == "image":
                part["image"] = images[img_idx]
                img_idx += 1

    return messages


def prepare_multimodal_messages_vllm(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # docstyle-ignore  # because <Image> is not parsable in the code block
    """
    Convert structured multimodal messages into a format compatible with vLLM. Replaces `"type": "image"` blocks with
    `"type": "image_pil"` blocks, and `"image": Image` with `"image_pil": Image`.

    Args:
        messages (`list[dict[str, Any]]`):
            Messages with `"role"` and `"content"`. Content is expected to be a list of structured blocks.

    Returns:
        `list[dict[str, Any]]`:
            A deep-copied list of messages compatible with vLLM's expected input format.

    Example:
    ```python
    # Input
    [{"role": "user", "content": [{"type": "image", "image": <PIL.Image.Image>}, {"type": "text", "text": "What's in this image?"}]}]

    # Output
    [{"role": "user", "content": [{"type": "image_pil", "image_pil": <PIL.Image.Image>}, {"type": "text", "text": "What's in this image?"}]}]
    ```
    """
    messages = copy.deepcopy(messages)  # avoid modifying the original messages
    for message in messages:
        if isinstance(message["content"], list):
            for part in message["content"]:
                if part["type"] == "image":
                    part["type"] = "image_pil"  # vLLM expects 'image_pil' key for images
                    part["image_pil"] = part.pop("image")
    return messages


def is_conversational(example: dict[str, Any]) -> bool:
    r"""
    Check if the example is in a conversational format.

    Args:
        example (`dict[str, Any]`):
            A single data entry of a dataset. The example can have different keys depending on the dataset type.

    Returns:
        `bool`:
            `True` if the data is in a conversational format, `False` otherwise.

    Examples:

    ```python
    >>> example = {"prompt": [{"role": "user", "content": "What color is the sky?"}]}
    >>> is_conversational(example)
    True

    >>> example = {"prompt": "The sky is"}
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
        # It must be a list of messages
        if isinstance(maybe_messages, list):
            maybe_message = maybe_messages[0]
            # Each message must a list of dictionaries with keys "role" and "content"
            if isinstance(maybe_message, dict) and "role" in maybe_message:
                return True

    return False


def apply_chat_template(
    example: dict[str, list[dict[str, str]]],
    tokenizer: PreTrainedTokenizerBase | ProcessorMixin,
    tools: list[dict | Callable] | None = None,
    **template_kwargs,
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
        messages = tokenizer.apply_chat_template(
            example["messages"],
            tools=tools,
            tokenize=False,
            **example.get("chat_template_kwargs", {}),
            **template_kwargs,
        )

    # Apply the chat template to the prompt, adding the generation prompt
    if "prompt" in example:
        last_role = example["prompt"][-1]["role"]
        if last_role in ["user", "tool"]:
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
            **example.get("chat_template_kwargs", {}),
            **template_kwargs,
        )

    # Apply the chat template to the entire prompt + completion
    if "prompt" in example:  # explicit prompt and prompt-completion case
        if "chosen" in example:
            prompt_chosen = tokenizer.apply_chat_template(
                example["prompt"] + example["chosen"],
                tools=tools,
                tokenize=False,
                **example.get("chat_template_kwargs", {}),
                **template_kwargs,
            )
            # DeepSeek-R1 inserts a <tool_call> token when using `add_generation_prompt`, which can cause discrepancies
            # between the prompt alone and the combined prompt+completion. To ensure consistency, we extract the
            # common prefix between the two. In most cases, this is a no-op.
            prompt = "".join(x for x, _ in takewhile(lambda x: x[0] == x[1], zip(prompt, prompt_chosen, strict=False)))

            chosen = prompt_chosen[len(prompt) :]
        if "rejected" in example and "prompt" in example:  # explicit prompt
            prompt_rejected = tokenizer.apply_chat_template(
                example["prompt"] + example["rejected"],
                tools=tools,
                tokenize=False,
                **example.get("chat_template_kwargs", {}),
                **template_kwargs,
            )
            # Handle DeepSeek-R1 <tool_call> token, see the above comment for details
            prompt = "".join(
                x for x, _ in takewhile(lambda x: x[0] == x[1], zip(prompt, prompt_rejected, strict=False))
            )
            rejected = prompt_rejected[len(prompt) :]
        if "completion" in example:
            prompt_completion = tokenizer.apply_chat_template(
                example["prompt"] + example["completion"],
                tools=tools,
                tokenize=False,
                **example.get("chat_template_kwargs", {}),
                **template_kwargs,
            )
            # Handle DeepSeek-R1 <tool_call> token, see the above comment for details
            prompt = "".join(
                x for x, _ in takewhile(lambda x: x[0] == x[1], zip(prompt, prompt_completion, strict=False))
            )
            completion = prompt_completion[len(prompt) :]
    else:  # implicit prompt case
        if "chosen" in example:
            chosen = tokenizer.apply_chat_template(
                example["chosen"],
                tools=tools,
                tokenize=False,
                **example.get("chat_template_kwargs", {}),
                **template_kwargs,
            )
        if "rejected" in example:
            rejected = tokenizer.apply_chat_template(
                example["rejected"],
                tools=tools,
                tokenize=False,
                **example.get("chat_template_kwargs", {}),
                **template_kwargs,
            )

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
    tools: list[dict | Callable] | None = None,
    **template_kwargs: Any,
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
            messages, where each message is a dictionary with keys `"role"` and `"content"`. Additionally, the example
            may contain a `"chat_template_kwargs"` key, which is a dictionary of additional keyword arguments to pass
            to the chat template renderer.
        tokenizer ([`~transformers.PreTrainedTokenizerBase`]):
            Tokenizer to apply the chat template with.
        tools (`list[dict | Callable]`, *optional*):
            A list of tools (callable functions) that will be accessible to the model. If the template does not support
            function calling, this argument will have no effect.
        **template_kwargs (`Any`, *optional*):
            Additional kwargs to pass to the template renderer. Will be accessible by the chat template.

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
    ...     "completion": [{"role": "assistant", "content": "It is blue."}],
    ... }
    >>> apply_chat_template(example, tokenizer)
    {'prompt': '<|user|>\nWhat color is the sky?<|end|>\n<|assistant|>\n', 'completion': 'It is blue.<|end|>\n'}
    ```
    """
    if is_conversational(example):
        return apply_chat_template(example, tokenizer, tools, **template_kwargs)
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
    dataset: DatasetType, num_proc: int | None = None, desc: str | None = None
) -> DatasetType:
    r"""
    Unpair a preference dataset.

    Args:
        dataset ([`~datasets.Dataset`] or [`~datasets.DatasetDict`]):
            Preference dataset to unpair. The dataset must have columns `"chosen"`, `"rejected"` and optionally
            `"prompt"`.
        num_proc (`int`, *optional*):
            Number of processes to use for processing the dataset.
        desc (`str`, *optional*):
            Meaningful description to be displayed alongside with the progress bar while mapping examples.

    Returns:
        [`~datasets.Dataset`]: The unpaired preference dataset.

    Example:

    ```python
    >>> from datasets import Dataset

    >>> dataset_dict = {
    ...     "prompt": ["The sky is", "The sun is"],
    ...     "chosen": [" blue.", "in the sky."],
    ...     "rejected": [" green.", " in the sea."],
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
    dataset: DatasetType, num_proc: int | None = None, desc: str | None = None
) -> DatasetType:
    r"""
    Unpair a preference dataset if it is paired.

    Args:
        dataset ([`~datasets.Dataset`] or [`~datasets.DatasetDict`]):
            Preference dataset to unpair. The dataset must have columns `"chosen"`, `"rejected"` and optionally
            `"prompt"`.
        num_proc (`int`, *optional*):
            Number of processes to use for processing the dataset.
        desc (`str`, *optional*):
            Meaningful description to be displayed alongside with the progress bar while mapping examples.

    Returns:
        [`~datasets.Dataset`] or [`~datasets.DatasetDict`]: The unpaired preference dataset if it was paired, otherwise
        the original dataset.

    Example:

    ```python
    >>> from datasets import Dataset

    >>> dataset_dict = {
    ...     "prompt": ["The sky is", "The sun is"],
    ...     "chosen": [" blue.", "in the sky."],
    ...     "rejected": [" green.", " in the sea."],
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
    Extracts the shared prompt from a preference data example, where the prompt is implicit within both the chosen and
    rejected completions.

    The function identifies the longest common sequence (prefix) of conversation turns between the "chosen" and
    "rejected" completions and extracts this as the prompt. It then removes this prompt from the respective "chosen"
    and "rejected" completions.

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
    ...         {"role": "assistant", "content": "It is blue."},
    ...     ],
    ...     "rejected": [
    ...         {"role": "user", "content": "What color is the sky?"},
    ...         {"role": "assistant", "content": "It is green."},
    ...     ],
    ... }
    >>> extract_prompt(example)
    {'prompt': [{'role': 'user', 'content': 'What color is the sky?'}],
     'chosen': [{'role': 'assistant', 'content': 'It is blue.'}],
     'rejected': [{'role': 'assistant', 'content': 'It is green.'}]}
    ```

    Or, with the `map` method of [`~datasets.Dataset`]:

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
    Extracts the shared prompt from a preference data example, where the prompt is implicit within both the chosen and
    rejected completions.

    If the example already contains a `"prompt"` key, the function returns the example as is. For more details, see
    [`extract_prompt`].
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


def _get_dataset_format(dataset: DatasetType) -> dict[str, Any]:
    if isinstance(dataset, (DatasetDict, IterableDatasetDict)):
        dataset = dataset[next(iter(dataset))]
    if isinstance(dataset, Dataset):
        format = dataset.format
    else:
        format_type = dataset._formatting.format_type if dataset._formatting is not None else None
        format = {"type": format_type}
    format.update(format.pop("format_kwargs", {}))
    return format


def _check_if_columns_can_be_packed(columns: list[pa.Array]):
    first_column_offsets = None
    for idx, column in enumerate(columns):
        if not (pyarrow.types.is_list(column.type) or pyarrow.types.is_large_list(column.type)):
            raise TypeError("Packing requires all columns to be lists of lists.")

        if idx == 0:
            first_column_offsets = column.offsets
        elif not first_column_offsets.equals(column.offsets):
            raise ValueError("All columns must have values of the same length.")


class _SegmentTree:
    """
    A segment tree data structure that, when initialized as `_SegmentTree(maxval)`, efficiently finds the next larger
    value for a given input within the range [1, maxval].

    See [Fewer Truncations Improve Language Modeling](https://huggingface.co/papers/2404.10830) for more details.
    """

    def __init__(self, maxval: int):
        self.maxval = maxval
        # For non-power-of-2 values, we need to round up to the next power of 2 for the tree size
        self.tree_size = 1 << (maxval - 1).bit_length()
        self.tree = [0] * (2 * self.tree_size)

    def add(self, val):
        assert 0 < val <= self.maxval
        i = self.tree_size + val - 1
        self.tree[i] = val
        while i > 1:
            i >>= 1
            left, right = self.tree[i << 1], self.tree[(i << 1) + 1]
            # Compare the values using if-else otherwise repeated calls to `builtins.max` become the bottleneck
            self.tree[i] = left if left >= right else right

    def remove(self, val):
        assert 0 < val <= self.maxval
        i = self.tree_size + val - 1
        self.tree[i] = 0
        while i > 1:
            i >>= 1
            left, right = self.tree[i << 1], self.tree[(i << 1) + 1]
            # Compare the values using if-else otherwise repeated calls to `builtins.max` become the bottleneck
            self.tree[i] = left if left >= right else right

    def search(self, val):
        assert 0 < val <= self.maxval
        i = 1
        while i < self.tree_size:
            if self.tree[i << 1] >= val:
                i = i << 1
            else:
                i = (i << 1) + 1
        return self.tree[i]


def _pack_bfd(
    examples: pa.Table, seq_length: int, on_seq_length_overflow: Literal["truncate", "split"] = "truncate"
) -> pa.Table:
    """Pack sequences in a pyarrow Table using Best Fit Decreasing strategy."""
    columns = [column.chunks[0] for column in examples.combine_chunks().columns]
    _check_if_columns_can_be_packed(columns)
    assert len(columns) > 0

    lengths = pc.list_value_length(columns[0])

    # Filter out empty sequences
    non_empty_mask = pc.greater(lengths, 0)
    columns = [pc.filter(column, non_empty_mask) for column in columns]
    lengths = pc.filter(lengths, non_empty_mask)

    if on_seq_length_overflow == "truncate":
        columns = [pc.list_slice(column, 0, seq_length) for column in columns]
    elif on_seq_length_overflow == "split":
        lengths = lengths.to_numpy()
        # Split the sequences longer than `seq_length` into chunks (of length `seq_length` or less) while respecting sequence boundaries
        num_fragments = np.ceil(lengths / seq_length).astype(int)
        offsets = np.arange(np.sum(num_fragments) + 1, dtype=columns[0].offsets.type.to_pandas_dtype()) * seq_length
        # "Left-shift" the offsets to account for the last fragment of each original sequence possibly being shorter than `seq_length`
        diff = np.zeros_like(offsets)
        diff[np.cumsum(num_fragments)] = -lengths % seq_length
        diff = np.cumsum(diff)
        offsets -= diff
        columns = [
            type(column).from_arrays(offsets.astype(column.offsets.type.to_pandas_dtype()), column.values)
            for column in columns
        ]
    else:
        raise ValueError(f"Invalid `on_seq_length_overflow`: {on_seq_length_overflow}. Use 'truncate' or 'split'.")

    examples = pa.Table.from_arrays(columns, names=examples.column_names)
    lengths = pc.list_value_length(columns[0])
    examples = examples.append_column("seq_lengths", lengths)  # Allows us to later construct `position_ids`
    ids = np.arange(len(examples))
    lengths = pc.make_struct(lengths, ids)
    lengths = lengths.sort("descending", by=0)

    # Greedy BFD binning using a segment tree to quickly find best-fit remaining space.
    segment_tree = _SegmentTree(seq_length)
    segment_tree.add(seq_length)  # the max, `seq_length` bin is always available
    space_to_bin = defaultdict(deque)

    # Bin is represented as a dict (of example ids and sum of their lengths) to allow in-place updates
    bins: list[dict] = []
    for length, idx in zip(lengths.field(0).to_numpy(), lengths.field(1).to_numpy(), strict=True):
        space = segment_tree.search(length)

        if space < seq_length:
            # Use existing bin with exactly this amount of space
            bin = space_to_bin[space].popleft()
        else:
            # Create a new bin
            bin = {"ids": [], "length": 0}
            bins.append(bin)

        bin["ids"].append(idx)
        bin["length"] += length
        if space < seq_length and not space_to_bin[space]:
            segment_tree.remove(space)

        space = space - length
        space_to_bin[space].append(bin)
        if space > 0:
            segment_tree.add(space)

    examples = pc.take(examples, [id_ for bin in bins for id_ in bin["ids"]])
    offsets = np.cumsum([0] + [bin["length"] for bin in bins])

    assert all(
        column.num_chunks == 1 for column in examples.columns
    )  # `pc.take` returns a ChunkedArray with a single chunk

    lengths = examples["seq_lengths"].chunks[0]
    examples = examples.drop_columns("seq_lengths")
    lengths = pa.ListArray.from_arrays(np.cumsum([0] + [len(bin["ids"]) for bin in bins], dtype=np.int32), lengths)

    columns = []
    for column in examples.columns:
        column = column.chunks[0]
        assert pa.types.is_list(column.type) or pa.types.is_large_list(column.type)
        dtype = column.offsets.type.to_pandas_dtype()
        column = type(column).from_arrays(offsets.astype(dtype), column.values)
        columns.append(column)
    return pa.Table.from_arrays(columns + [lengths], names=examples.column_names + ["seq_lengths"])


def _pack_wrapped(examples: pa.Table, seq_length: int) -> pa.Table:
    """Pack sequences in a pyarrow Table using a wrapped strategy."""
    columns = [column.chunks[0] for column in examples.combine_chunks().columns]
    _check_if_columns_can_be_packed(columns)
    offsets, values = columns[0].offsets, columns[0].values
    values = values[offsets[0].as_py() : offsets[-1].as_py()]
    num_elements = len(values)
    offsets = np.arange(0, num_elements, seq_length, dtype=columns[0].offsets.type.to_pandas_dtype())
    offsets = np.concatenate((offsets, [num_elements]))
    columns = [
        type(column).from_arrays(offsets.astype(column.offsets.type.to_pandas_dtype()), column.values)
        for column in columns
    ]
    return pa.Table.from_arrays(columns, names=examples.column_names)


def pack_dataset(
    dataset: DatasetType,
    seq_length: int,
    strategy: str = "bfd",
    map_kwargs: dict[str, Any] | None = None,
) -> DatasetType:
    r"""
    Pack sequences in a dataset into chunks of size `seq_length`.

    Args:
        dataset ([`~datasets.Dataset`] or [`~datasets.DatasetDict`]):
            Dataset to pack
        seq_length (`int`):
            Target sequence length to pack to.
        strategy (`str`, *optional*, defaults to `"bfd"`):
            Packing strategy to use. Can be either:

            - `"bfd"` (Best Fit Decreasing): Preserves sequence boundaries and truncates sequences that exceed
                `seq_length`, discarding overflow tokens. Ideal for SFT and conversational datasets where maintaining
                conversation structure is important.
            - `"bfd_split"`: Similar to `"bfd"` but splits overflow sequences for packing into other examples. Prevents
                token loss for pre-training or long documents, but may break conversation structure in SFT datasets.
            - `"wrapped"`: Faster but more aggressive. Ignores sequence boundaries and will cut sequences in the middle
                to completely fill each packed sequence with data.
        map_kwargs (`dict`, *optional*):
            Additional keyword arguments to pass to the dataset's map method when packing examples.

    Returns:
        [`~datasets.Dataset`] or [`~datasets.DatasetDict`]: The dataset with packed sequences. The number of examples
        may decrease as sequences are combined.

    Example:
    ```python
    >>> from datasets import Dataset
    >>> from trl import pack_dataset

    >>> examples = {
    ...     "input_ids": [[1, 2, 3, 4, 5], [6, 7], [8, 9, 10], [11]],
    ...     "attention_mask": [[1, 1, 1, 0, 0], [1, 0], [1, 1, 0], [1]],
    ... }
    >>> dataset = Dataset.from_dict(examples)
    >>> # Default "bfd" strategy (SFT-friendly): truncates long sequences
    >>> packed_dataset = pack_dataset(dataset, seq_length=4, strategy="bfd")
    >>> packed_dataset[:]
    {'input_ids': [[1, 2, 3, 4], [8, 9, 10, 11], [6, 7]],
     'attention_mask': [[1, 1, 1, 0], [1, 1, 0, 1], [1, 0]],
     'seq_lengths': [[4], [3, 1], [2]]}

    >>> # "bfd_split" strategy: preserves all tokens
    >>> packed_dataset = pack_dataset(dataset, seq_length=4, strategy="bfd_split")
    >>> packed_dataset[:]
    {'input_ids': [[1, 2, 3, 4], [8, 9, 10, 5], [6, 7, 11]],
     'attention_mask': [[1, 1, 1, 0], [1, 1, 0, 0], [1, 0, 1]],
     'seq_lengths': [[4], [3, 1], [2, 1]]}
    ```
    """
    if map_kwargs is None:
        map_kwargs = {}

    valid_strategies = ("bfd", "bfd_split", "wrapped")
    if strategy not in valid_strategies:
        raise ValueError(f"Invalid packing strategy '{strategy}', must be one of {valid_strategies}.")
    format = _get_dataset_format(dataset)
    dataset = dataset.with_format("arrow")
    if strategy == "bfd":
        dataset = dataset.map(
            _pack_bfd,
            batched=True,
            fn_kwargs={"seq_length": seq_length, "on_seq_length_overflow": "truncate"},
            **map_kwargs,
        )
    elif strategy == "bfd_split":
        dataset = dataset.map(
            _pack_bfd,
            batched=True,
            fn_kwargs={"seq_length": seq_length, "on_seq_length_overflow": "split"},
            **map_kwargs,
        )
    elif strategy == "wrapped":
        dataset = dataset.map(_pack_wrapped, batched=True, fn_kwargs={"seq_length": seq_length}, **map_kwargs)
    else:
        raise ValueError(f"Invalid packing strategy: '{strategy}', must be one of {valid_strategies}.")

    if strategy in {"bfd", "bfd_split"} and "columns" in format:
        format["columns"] = format["columns"] + ["seq_lengths"]

    dataset = dataset.with_format(**format)
    return dataset


def truncate_dataset(
    dataset: DatasetType,
    max_length: int,
    truncation_mode: str = "keep_start",
    map_kwargs: dict[str, Any] | None = None,
) -> DatasetType:
    r"""
    Truncate sequences in a dataset to a specified `max_length`.

    Args:
        dataset ([`~datasets.Dataset`] or [`~datasets.DatasetDict`]):
            Dataset to truncate.
        max_length (`int`):
            Maximum sequence length to truncate to.
        truncation_mode (`str`, *optional*, defaults to `"keep_start"`):
            Whether to keep the start (`"keep_start"`) or the end (`"keep_end"`) of the sequence when truncating.
        map_kwargs (`dict`, *optional*):
            Additional keyword arguments to pass to the dataset's map method when truncating examples.

    Returns:
        [`~datasets.Dataset`] or [`~datasets.DatasetDict`]: The dataset with truncated sequences.

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
    if truncation_mode not in {"keep_start", "keep_end"}:
        raise ValueError(f"Invalid truncation mode '{truncation_mode}'.")
    if map_kwargs is None:
        map_kwargs = {}

    def truncate(examples):
        truncated_columns = []
        for column in examples.columns:
            if pyarrow.types.is_list(column.type) or pyarrow.types.is_large_list(column.type):
                if truncation_mode == "keep_start":
                    column = pc.list_slice(column, 0, max_length)
                else:  # keep_end
                    column = (
                        pa.array([[] for _ in range(len(column))], type=column.type)
                        if max_length == 0
                        else pa.array([values[-max_length:] for values in column.to_pylist()], type=column.type)
                    )
            truncated_columns.append(column)
        return pa.Table.from_arrays(truncated_columns, names=examples.column_names)

    format = _get_dataset_format(dataset)
    dataset = dataset.with_format("arrow")
    dataset = dataset.map(truncate, batched=True, **map_kwargs)
    dataset = dataset.with_format(**format)
    return dataset


def is_conversational_from_value(example: dict[str, Any]) -> bool:
    r"""
    Check if the example is in a conversational format (from/value). Note that this format isn't recommended. Prefer
    the ChatML format (role/content)

    Args:
        example (`dict[str, Any]`):
            A single data entry of a dataset. The example can have different keys depending on the dataset type.

    Returns:
        `bool`:
            `True` if the data is in a conversational Chatformat, `False` otherwise.

    Examples:

    ```python
    >>> example = {"conversations": [{"from": "user", "value": "What color is the sky?"}]}
    >>> is_conversational_from_value(example)
    True

    >>> example = {"conversations": [{"role": "user", "content": "What color is the sky?"}]}
    >>> is_conversational_from_value(example)
    False

    >>> example = {"conversations": "The sky is"}
    >>> is_conversational_from_value(example)
    False
    ```
    """
    maybe_messages = example.get("conversations")
    # It must be a list of messages
    if isinstance(maybe_messages, list):
        maybe_message = maybe_messages[0]
        # Each message must a list of dictionaries with keys "from" and "value"
        if isinstance(maybe_message, dict) and "from" in maybe_message and "value" in maybe_message:
            return True

    return False


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
    ...         {"from": "assistant", "value": "It is blue."},
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
