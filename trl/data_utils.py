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


def apply_chat_template_non_preference(example, tokenizer):
    r"""
    Apply a chat template to a non-preference example.

    Args:
        example (`Dict[str, List[Dict[str, str]]`):
            Dictionary representing a single data entry, which contains:

                - `"prompt"` (`List[Dict[str, str]]`): List of messages representing the prompt conversation.
                - `"completion"` (`List[Dict[str, str]]`): List of messages representing the completion conversation.

        tokenizer (`PreTrainedTokenizer`):
            The tokenizer to apply the chat template with.

    Returns:
        `Dict[str, str]`: The formatted example with the chat template applied.

    Raises:
        `ValueError`:
            If the chat template applied to the prompt + completion does not start with the chat template applied to
            the prompt alone. It can indicate that the chat template is not supported by TRL.

    Example:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")
        >>> example = {
        ...     "prompt": [{"role": "user", "content": "What color is the sky?"}],
        ...     "completion": [{"role": "assistant", "content": "It is blue."}]
        ... }
        >>> apply_chat_template_non_preference(example, tokenizer)
        {'prompt': '<|user|>\nWhat color is the sky?<|end|>\n<|assistant|>\n', 'completion': 'It is blue.<|end|>\n<|endoftext|>'}
    """
    # Apply the chat template to the prompt only and adding the generation prompt
    prompt = tokenizer.apply_chat_template(example["prompt"], tokenize=False, add_generation_prompt=True)

    # Apply the chat template to the entire prompt + completion
    prompt_completion = tokenizer.apply_chat_template(example["prompt"] + example["completion"], tokenize=False)

    # Ensure that the prompt is the initial part of the prompt-completion string
    if not prompt_completion.startswith(prompt):
        raise ValueError(
            "The chat template applied to the prompt + completion does not start with the chat template applied to "
            "the prompt only. This can indicate that the chat template is not supported by TRL."
        )

    # Extract the completion by removing the prompt part from the prompt-completion string
    completion = prompt_completion[len(prompt) :]

    # Return the formatted prompt and extracted completion
    return {"prompt": prompt, "completion": completion}


def apply_chat_template_prompt_only(
    example: Dict[str, List[Dict[str, str]]], tokenizer: PreTrainedTokenizer
) -> Dict[str, str]:
    r"""
    Apply a chat template to a prompt-only example.

    Args:
        example (`Dict[str, List[Dict[str, str]]`):
            Dictionary representing a single data entry, which contains:

                - `"prompt"` (`List[Dict[str, str]]`): List of messages representing the prompt conversation.

        tokenizer (`PreTrainedTokenizer`):
            The tokenizer to apply the chat template with.

    Returns:
        `Dict[str, str]`: The formatted example with the chat template applied.

    Example:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")
        >>> example = {
        ...     "prompt": [{"role": "user", "content": "What color is the sky?"}]
        ... }
        >>> apply_chat_template_prompt_only(example, tokenizer)
        {'prompt': '<|user|>\nWhat color is the sky?<|end|>\n<|assistant|>\n'}
    """
    prompt = tokenizer.apply_chat_template(example["prompt"], tokenize=False, add_generation_prompt=True)
    return {"prompt": prompt}


def apply_chat_template_preference(
    example: Dict[str, List[Dict[str, str]]], tokenizer: PreTrainedTokenizer
) -> Dict[str, str]:
    r"""
    Apply a chat template to a preference example.

    Args:
        example (`Dict[str, List[Dict[str, str]]`):
            Dictionary representing a single data entry, which contains:

                - `"prompt"` (`List[Dict[str, str]]`): List of messages representing the prompt conversation.
                - `"chosen"` (`List[Dict[str, str]]`): List of messages representing the chosen conversation.
                - `"rejected"` (`List[Dict[str, str]]`): List of message representing the rejected conversation.

        tokenizer (`PreTrainedTokenizer`):
            The tokenizer to apply the chat template with.

    Returns:
        `Dict[str, str]`: The formatted example with the chat template applied.

    Raises:
        `ValueError`:
            If the chat template applied to the prompt + chosen (or rejected) does not start with the chat template
            applied to the prompt alone. It can indicate that the chat template is not supported by TRL.

    Example:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")
        >>> example = {
        ...     "prompt": [{"role": "user", "content": "What color is the sky?"}],
        ...     "chosen": [{"role": "assistant", "content": "It is blue."}],
        ...     "rejected": [{"role": "assistant", "content": "It is green."}]
        ... }
        >>> apply_chat_template_preference(example, tokenizer)
        {'prompt': '<|user|>\nWhat color is the sky?<|end|>\n<|assistant|>\n', 'chosen': 'It is blue.<|end|>\n<|endoftext|>', 'rejected': 'It is green.<|end|>\n<|endoftext|>'}
    """
    # Apply the chat template to the prompt only and adding the generation prompt
    prompt = tokenizer.apply_chat_template(example["prompt"], tokenize=False, add_generation_prompt=True)

    # Apply the chat template to the entire prompt + chosen and prompt + rejected
    prompt_chosen = tokenizer.apply_chat_template(example["prompt"] + example["chosen"], tokenize=False)
    prompt_rejected = tokenizer.apply_chat_template(example["prompt"] + example["rejected"], tokenize=False)

    # Ensure that the prompt is the initial part of the prompt-chosen and prompt-rejected strings
    if not prompt_chosen.startswith(prompt) or not prompt_rejected.startswith(prompt):
        raise ValueError(
            "The chat template applied to the prompt + chosen (or rejected) does not start with the chat template "
            "applied to the prompt alone. This can indicate that the chat template is not supported by TRL."
        )

    # Extract the chosen and rejected completions by removing the prompt part from the prompt-chosen and prompt-rejected strings
    chosen = prompt_chosen[len(prompt) :]
    rejected = prompt_rejected[len(prompt) :]

    # Return the formatted prompt, extracted chosen and rejected completions
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


def apply_chat_template_unpaired_preference(
    example: Dict[str, List[Dict[str, str]]], tokenizer: PreTrainedTokenizer
) -> Dict[str, str]:
    r"""
    Apply a chat template to an unpaired preference example.

    Args:
        example (`Dict[str, List[Dict[str, str]]`):
            Dictionary representing a single data entry, which contains:

                - `"prompt"` (`List[Dict[str, str]]`): List of messages representing the prompt conversation.
                - `"completion"` (`List[Dict[str, str]]`): List of messages representing the completion conversation.
                - `"label"` (`bool`): The preference label

        tokenizer (`PreTrainedTokenizer`):
            The tokenizer to apply the chat template with.

    Returns:
        `Dict[str, str]`: The formatted example with the chat template applied.

    Raises:
        `ValueError`:
            If the chat template applied to the prompt + completion does not start with the chat template applied to
            the prompt alone. It can indicate that the chat template is not supported by TRL.

    Example:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")
        >>> example = {
        ...     "prompt": [{"role": "user", "content": "What color is the sky?"}],
        ...     "completion": [{"role": "assistant", "content": "It is blue."}],
        ...     "label": True
        ... }
        >>> apply_chat_template_unpaired_preference(example, tokenizer)
        {'prompt': '<|user|>\nWhat color is the sky?<|end|>\n<|assistant|>\n', 'completion': 'It is blue.<|end|>\n<|endoftext|>', 'label': True}
    """
    # Apply the chat template to the prompt only and adding the generation prompt
    prompt = tokenizer.apply_chat_template(example["prompt"], tokenize=False, add_generation_prompt=True)

    # Apply the chat template to the entire prompt + completion
    prompt_completion = tokenizer.apply_chat_template(example["prompt"] + example["completion"], tokenize=False)

    # Ensure that the prompt is the initial part of the prompt-completion string
    if not prompt_completion.startswith(prompt):
        raise ValueError(
            "The chat template applied to the prompt + completion does not start with the chat template applied to "
            "the prompt only. This can indicate that the chat template is not suported by TRL."
        )

    # Extract the completion by removing the prompt part from the prompt-completion string
    completion = prompt_completion[len(prompt) :]

    # Return the formatted prompt, extracted completion, and original label
    return {"prompt": prompt, "completion": completion, "label": example["label"]}


def convert_conversational_to_standard(example, tokenizer):
    if set(example.keys()) == {"prompt", "completion"}:
        return apply_chat_template_non_preference(example, tokenizer)
    elif set(example.keys()) == {"prompt"}:
        return apply_chat_template_prompt_only(example, tokenizer)
    elif set(example.keys()) == {"prompt", "chosen", "rejected"}:
        return apply_chat_template_preference(example, tokenizer)
    elif set(example.keys()) == {"prompt", "completion", "label"}:
        return apply_chat_template_unpaired_preference(example, tokenizer)
    else:
        raise RuntimeError("Unknown dataset format")


def is_conversational(dataset: Union[Dataset, DatasetDict]) -> bool:
    r"""
    Check if the dataset is in a conversational format.

    Args:
        dataset (`Dataset` or `DatasetDict`):
            The dataset to check.

    Returns:
        `bool`: `True` if the dataset is in a conversational format, `False` otherwise.

    Examples:

        >>> from datasets import Dataset
        >>> dataset = Dataset.from_dict({"prompt": [[{"role": "user", "content": "What color is the sky?"}]]})
        >>> is_conversational(dataset)
        True
        >>> dataset = Dataset.from_dict({"prompt": ["The sky is"]})
        >>> is_conversational(dataset)
        False

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
