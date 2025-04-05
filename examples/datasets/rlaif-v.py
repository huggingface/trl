# Copyright 2025 The HuggingFace Team. All rights reserved.
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

from dataclasses import dataclass, field
from typing import Optional

from datasets import features, load_dataset
from huggingface_hub import ModelCard
from transformers import HfArgumentParser


@dataclass
class ScriptArguments:
    r"""
    Arguments for the script.

    Args:
        push_to_hub (`bool`, *optional*, defaults to `False`):
            Whether to push the dataset to the Hugging Face Hub.
        repo_id (`str`, *optional*, defaults to `"trl-lib/rlaif-v"`):
            Hugging Face repository ID to push the dataset to.
        dataset_num_proc (`int` or `None`, *optional*, defaults to `None`):
            Number of workers to use for dataset processing.
    """

    push_to_hub: bool = field(
        default=False,
        metadata={"help": "Whether to push the dataset to the Hugging Face Hub."},
    )
    repo_id: str = field(
        default="trl-lib/rlaif-v",
        metadata={"help": "Hugging Face repository ID to push the dataset to."},
    )
    dataset_num_proc: Optional[int] = field(
        default=None,
        metadata={"help": "Number of workers to use for dataset processing."},
    )


def to_conversational(example):
    """
    Convert prompt from "xxx" to [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "xxx"}]}]
    and chosen and rejected from "xxx" to [{"role": "assistant", "content": [{"type": "text", "text": "xxx"}]}].
    Images are wrapped into a list.
    """
    prompt = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": example["question"]}]}]
    chosen = [{"role": "assistant", "content": [{"type": "text", "text": example["chosen"]}]}]
    rejected = [{"role": "assistant", "content": [{"type": "text", "text": example["rejected"]}]}]
    return {"prompt": prompt, "images": [example["image"]], "chosen": chosen, "rejected": rejected}


model_card = ModelCard("""
---
tags: [trl]
---

# RLAIF-V Dataset

## Summary

The RLAIF-V dataset is a processed version of the [openbmb/RLAIF-V-Dataset](https://huggingface.co/datasets/openbmb/RLAIF-V-Dataset#dataset-card-for-rlaif-v-dataset), specifically curated to train vision-language models using the [TRL library](https://github.com/huggingface/trl) for preference learning tasks. It contains 83,132 high-quality comparison pairs, each comprising an image and two textual descriptions: one preferred and one rejected. This dataset enables models to learn human preferences in visual contexts, enhancing their ability to generate and evaluate image captions.

## Data Structure

- **Format**: [Conversational](https://huggingface.co/docs/trl/main/dataset_formats#conversational)
- **Type**: [Preference](https://huggingface.co/docs/trl/main/dataset_formats#preference)

Columns:
- `"prompt"`: The task related to the image.
- `"images"`: The image.
- `"chosen"`: The preferred answer.
- `"rejected"`: An alternative answer that was not preferred.

This structure allows models to learn to prefer the _chosen_ response over the _rejected_ one, thereby aligning with human preferences in visual tasks.

## Generation script

The script used to generate this dataset can be found [here](https://github.com/huggingface/trl/blob/main/examples/datasets/rlaif-v.py).
""")

if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    dataset = load_dataset("openbmb/RLAIF-V-Dataset", split="train")
    dataset = dataset.map(
        to_conversational,
        num_proc=script_args.dataset_num_proc,
        remove_columns=dataset.column_names,
        writer_batch_size=128,
    )

    # Cast the images to Sequence[Image] to avoid bytes format
    f = dataset.features
    f["images"] = features.Sequence(features.Image(decode=True))
    dataset = dataset.cast(f)

    dataset = dataset.train_test_split(test_size=0.01, writer_batch_size=128)

    if script_args.push_to_hub:
        dataset.push_to_hub(script_args.repo_id)
        model_card.push_to_hub(script_args.repo_id, repo_type="dataset")
