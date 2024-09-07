# Copyright 2022 The HuggingFace Team. All rights reserved.
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
from typing import Dict, List, Optional, TypeVar, Union

from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict
from transformers import PreTrainedTokenizer


DatasetType = TypeVar("DatasetType", Dataset, DatasetDict, IterableDataset, IterableDatasetDict)


def apply_chat_template(example: Dict[str, List[Dict[str, str]]], tokenizer: PreTrainedTokenizer) -> Dict[str, str]:
    r"""
    Apply a chat template to a conversational example.

    Args:
        example (`Dict[str, List[Dict[str, str]]`):
            Dictionary representing a single data entry of a conversational dataset. Each data entry can have different
            keys depending on the dataset format. The supported dataset formats are:

                - Name to find dataset: `"messages"`.
                - Prompt-only dataset: `"prompt"`.
                - Prompt-completion dataset: `"prompt"` and `"completion"`.
                - Preference dataset: `"prompt"`, `"chosen"`, and `"rejected"`.
                - Preference dataset with implicit prompt: `"chosen"` and `"rejected"`.
                - Unpaired preference dataset: `"prompt"`, `"completion"`, and `"label"`.

            For keys `"messages"`, `"prompt"`, `"chosen"`, `"rejected"`, and `"completion"`, the values are lists of
            messages, where each message is a dictionary with keys `"role"` and `"content"`.

        tokenizer (`PreTrainedTokenizer`):
            The tokenizer to apply the chat template with.

    Returns:
        `Dict[str, str]`: The formatted example with the chat template applied.

    Note:
        This function does not alter the keys, except for name to find dataset, where `"messages"` is replaced by
        `"text"`.

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
    # Check that the example has the correct keys
    supported_keys = ["prompt", "chosen", "rejected", "completion", "messages", "label"]
    example_keys = {key for key in example.keys() if key in supported_keys}
    if example_keys not in [
        {"messages"},  # name to find
        {"prompt"},  # prompt-only
        {"prompt", "completion"},  # prompt-completion
        {"prompt", "chosen", "rejected"},  # preference
        {"chosen", "rejected"},  # preference with implicit prompt
        {"prompt", "completion", "label"},  # unpaired preference
    ]:
        raise KeyError(f"Invalid keys in the example: {example_keys}")

    # Apply the chat template to the whole conversation
    if "messages" in example:
        messages = tokenizer.apply_chat_template(example["messages"], tokenize=False)

    # Apply the chat template to the prompt, adding the generation prompt
    if "prompt" in example:
        prompt = tokenizer.apply_chat_template(example["prompt"], tokenize=False, add_generation_prompt=True)

    # Apply the chat template to the entire prompt + completion
    if "prompt" in example:  # explicit prompt and prompt-completion case
        if "chosen" in example:
            prompt_chosen = tokenizer.apply_chat_template(example["prompt"] + example["chosen"], tokenize=False)
            chosen = prompt_chosen[len(prompt) :]
        if "rejected" in example and "prompt" in example:  # explicit prompt
            prompt_rejected = tokenizer.apply_chat_template(example["prompt"] + example["rejected"], tokenize=False)
            rejected = prompt_rejected[len(prompt) :]
        if "completion" in example:
            prompt_completion = tokenizer.apply_chat_template(
                example["prompt"] + example["completion"], tokenize=False
            )
            completion = prompt_completion[len(prompt) :]
    else:  # implicit prompt case
        if "chosen" in example:
            chosen = tokenizer.apply_chat_template(example["chosen"], tokenize=False)
        if "rejected" in example:
            rejected = tokenizer.apply_chat_template(example["rejected"], tokenize=False)

    # Ensure that the prompt is the initial part of the prompt-completion string
    if "prompt" in example:
        error_message = (
            "The chat template applied to the prompt + completion does not start with the chat template applied to "
            "the prompt alone. This can indicate that the chat template is not supported by TRL."
        )
        if "chosen" in example and not prompt_chosen.startswith(prompt):
            raise ValueError(error_message)
        if "rejected" in example and not prompt_rejected.startswith(prompt):
            raise ValueError(error_message)
        if "completion" in example and not prompt_completion.startswith(prompt):
            raise ValueError(error_message)

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


def is_conversational(dataset: Union[Dataset, DatasetDict]) -> bool:
    r"""
    Check if the dataset is in a conversational format.

    Args:
        dataset (`Dataset` or `DatasetDict`):
            The dataset to check.

    Returns:
        `bool`: `True` if the dataset is in a conversational format, `False` otherwise.

    Examples:

    ```python
    >>> from datasets import Dataset
    >>> dataset = Dataset.from_dict({"prompt": [[{"role": "user", "content": "What color is the sky?"}]]})
    >>> is_conversational(dataset)
    True
    >>> dataset = Dataset.from_dict({"prompt": ["The sky is"]})
    >>> is_conversational(dataset)
    False
    ```

    Note:
        When `dataset` is a `DatasetDict`, this function checks only the first split. Consequently, we don't
        consider the case where different splits have different formats.
    """
    if isinstance(dataset, DatasetDict):
        dataset = dataset[list(dataset.keys())[0]]  # take the first split

    if "prompt" in dataset.features:
        messages = dataset["prompt"][0]
    elif "messages" in dataset.features:
        messages = dataset["messages"][0]
    else:
        return False

    # It should be a list of messages, where each message is a list of dictionaries with keys "role" and "content"
    if isinstance(messages, list) and isinstance(messages[0], dict) and "role" in messages[0]:
        return True


def _unpair_row(examples: List[Dict[str, List[Dict[str, str]]]]) -> List[Dict[str, List[Dict[str, str]]]]:
    batch_size = len(examples["chosen"])
    new_rows = {
        "prompt": examples["prompt"] + examples["prompt"],
        "completion": examples["chosen"] + examples["rejected"],
        "label": [True] * batch_size + [False] * batch_size,
    }
    return new_rows


def unpair_preference_dataset(dataset: DatasetType, num_proc: Optional[int] = None) -> DatasetType:
    r"""
    Unpair a preference dataset.

    Args:
        dataset (`Dataset`):
            Preference dataset to unpair. The dataset must have columns `"prompt"`, `"chosen"`, and `"rejected"`.
        num_proc (`Optional[int]`, *optional*, defaults to `None`):
            Number of processes to use for processing the dataset.

    Returns:
        `Dataset`: The unpaired preference dataset.

    Example:

    ```python
    >>> from datasets import Dataset
    >>> dataset_dict = {
    ...     "prompt": [
    ...         [{"role": "user", "content": "What color is the sky?"}],
    ...         [{"role": "user", "content": "Where is the sun?"}],
    ...     ],
    ...     "chosen": [
    ...         [{"role": "assistant", "content": "It is blue."}],
    ...         [{"role": "assistant", "content": "In the sky."}],
    ...     ],
    ...     "rejected": [
    ...         [{"role": "assistant", "content": "It is green."}],
    ...         [{"role": "assistant", "content": "In the sea."}],
    ...     ],
    ... }
    >>> dataset = Dataset.from_dict(dataset_dict)
    >>> dataset = unpair_preference_dataset(dataset)
    >>> dataset
    Dataset({
        features: ['prompt', 'completion', 'label'],
        num_rows: 4
    })
    >>> dataset[0]
    {'prompt': [{'content': 'What color is the sky?', 'role': 'user'}], 'completion': [{'content': 'It is blue.', 'role': 'assistant'}], 'label': True}
    ```
    """
    return dataset.map(_unpair_row, batched=True, remove_columns=["chosen", "rejected"], num_proc=num_proc)


def maybe_unpair_preference_dataset(dataset: DatasetType, num_proc: Optional[int] = None) -> DatasetType:
    raise NotImplementedError("This function is not implemented yet.")
