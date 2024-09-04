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

from datasets import DatasetDict


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
