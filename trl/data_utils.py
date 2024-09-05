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
from copy import deepcopy
from typing import Dict, List, Union

from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizer


def _reformat_row_dpo_to_kto(row: dict):
    """Turn a DPO-formatted dataset row into two KTO-formatted rows."""

    chosen_row = {"prompt": row["prompt"], "completion": row["chosen"], "label": [True] * len(row["chosen"])}
    rejected_row = {
        "prompt": row["prompt"],
        "completion": row["rejected"],
        "label": [False] * len(row["chosen"]),
    }
    new_rows = {k: chosen_row[k] + rejected_row[k] for k in chosen_row.keys()}
    return new_rows


def maybe_reformat_dpo_to_kto(dataset: DatasetDict, num_proc: int = None):
    """
    Reformat a dataset from the DPO format to the KTO format if necessary.

    This function checks whether the input dataset is already in the KTO format (containing "prompt", "completion", and "label" fields).
    If the dataset is in DPO format (with "prompt", "chosen", and "rejected" fields), it converts it to KTO format by:
    - Removing any unnecessary columns.
    - Reformatting each row to create a unified format suitable for KTO training.

    Args:
        dataset (DatasetDict): The dataset to potentially reformat.
        num_proc (int, optional): The number of processes to use for multiprocessing during dataset transformation. Defaults to None.

    Returns:
        DatasetDict: The reformatted dataset, if conversion was needed; otherwise, the original dataset.

    Raises:
        ValueError: If the dataset format is not compatible with KTO or DPO.
    """
    keys = list(dataset["train"].features.keys())

    # check if the dataset is in the KTO format or needs to be reformatted
    if "prompt" in keys and "completion" in keys and "label" in keys:
        return dataset
    elif "prompt" in keys and "rejected" in keys and "chosen" in keys:
        # remove unnecessary fields
        keys_to_remove = deepcopy(keys)
        keys_to_remove.remove("prompt")
        keys_to_remove.remove("chosen")
        keys_to_remove.remove("rejected")
        dataset = dataset.remove_columns(keys_to_remove)

        # turn each DPO-formatted row into two KTO-formatted rows.
        dataset = dataset.map(
            _reformat_row_dpo_to_kto,
            num_proc=num_proc,
            batched=True,
            remove_columns=["chosen", "rejected"],
            desc="Reformatting Dataset from DPO format to KTO format.",
        )
        return dataset
    else:
        raise ValueError("Dataset format not compatible with KTO.")


def apply_chat_template(example: Dict[str, List[Dict[str, str]]], tokenizer: PreTrainedTokenizer) -> Dict[str, str]:
    r"""
    Apply a chat template to a conversational example.

    Args:
        example (`Dict[str, List[Dict[str, str]]`):
            Dictionary representing a single data entry. Each data entry can have different keys depending on the
            dataset format. The supported dataset formats are:

                - Non-preference conversational dataset: `"prompt"` and `"completion"`.
                - Prompt-only conversational dataset: `"prompt"`.
                - Preference conversational dataset: `"prompt"`, `"chosen"`, and `"rejected"`.
                - Unpaired preference conversational dataset: `"prompt"`, `"completion"`, and `"label"`.

            For keys `"prompt"`, `"chosen"`, `"rejected"`, and `"completion"`, the values are lists of messages, where
            each message is a dictionary with keys `"role"` and `"content"`.

        tokenizer (`PreTrainedTokenizer`):
            The tokenizer to apply the chat template with.

    Returns:
        `Dict[str, str]`: The formatted example with the chat template applied.

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

    # Apply the chat template to the prompt only and adding the generation prompt
    prompt = tokenizer.apply_chat_template(example["prompt"], tokenize=False, add_generation_prompt=True)

    # Apply the chat template to the entire prompt + completion
    if "chosen" in example:
        prompt_chosen = tokenizer.apply_chat_template(example["prompt"] + example["chosen"], tokenize=False)
    if "rejected" in example:
        prompt_rejected = tokenizer.apply_chat_template(example["prompt"] + example["rejected"], tokenize=False)
    if "completion" in example:
        prompt_completion = tokenizer.apply_chat_template(example["prompt"] + example["completion"], tokenize=False)

    # Ensure that the prompt is the initial part of the prompt-completion string
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
    output = {"prompt": prompt}
    if "chosen" in example:
        output["chosen"] = prompt_chosen[len(prompt) :]
    if "rejected" in example:
        output["rejected"] = prompt_rejected[len(prompt) :]
    if "completion" in example:
        output["completion"] = prompt_completion[len(prompt) :]
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
        prompt = dataset["prompt"][0]
        # it should be a list of messages, where each message is a list of dictionaries with keys "role" and "content"
        if isinstance(prompt, list) and isinstance(prompt[0], dict) and "role" in prompt[0]:
            return True
    return False
