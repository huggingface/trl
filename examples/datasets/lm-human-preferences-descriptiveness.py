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

from datasets import load_dataset
from huggingface_hub import ModelCard
from transformers import AutoTokenizer, HfArgumentParser


@dataclass
class ScriptArguments:
    r"""
    Arguments for the script.

    Args:
        push_to_hub (`bool`, *optional*, defaults to `False`):
            Whether to push the dataset to the Hugging Face Hub.
        repo_id (`str`, *optional*, defaults to `"trl-lib/lm-human-preferences-descriptiveness"`):
            Hugging Face repository ID to push the dataset to.
        dataset_num_proc (`int` or `None`, *optional*, defaults to `None`):
            Number of workers to use for dataset processing.
    """

    push_to_hub: bool = field(
        default=False,
        metadata={"help": "Whether to push the dataset to the Hugging Face Hub."},
    )
    repo_id: str = field(
        default="trl-lib/lm-human-preferences-descriptiveness",
        metadata={"help": "Hugging Face repository ID to push the dataset to."},
    )
    dataset_num_proc: Optional[int] = field(
        default=None,
        metadata={"help": "Number of workers to use for dataset processing."},
    )


# Edge cases handling: remove the cases where all samples are the same
def samples_not_all_same(example):
    return not all(example["sample0"] == example[f"sample{j}"] for j in range(1, 4))


def to_prompt_completion(example, tokenizer):
    prompt = tokenizer.decode(example["query"]).strip()
    best_idx = example["best"]
    chosen = tokenizer.decode(example[f"sample{best_idx}"])
    for rejected_idx in range(4):  # take the first rejected sample that is different from the chosen one
        rejected = tokenizer.decode(example[f"sample{rejected_idx}"])
        if chosen != rejected:
            break
    assert chosen != rejected
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


model_card = ModelCard("""
---
tags: [trl]
---

# LM-Human-Preferences-Descriptiveness Dataset

## Summary

The LM-Human-Preferences-Descriptiveness dataset is a processed subset of [OpenAI's LM-Human-Preferences](https://github.com/openai/lm-human-preferences), focusing specifically on enhancing the descriptiveness of generated text. It contains pairs of text samples, each labeled as either "chosen" or "rejected," based on human preferences regarding the level of detail and vividness in the descriptions. This dataset enables models to learn human preferences in descriptive language, improving their ability to generate rich and engaging narratives.

## Data Structure

- **Format**: [Standard](https://huggingface.co/docs/trl/main/dataset_formats#standard)
- **Type**: [Preference](https://huggingface.co/docs/trl/main/dataset_formats#preference)

Columns:
- `"prompt"`: The text sample.
- `"chosen"`: A version of the text with enhanced descriptiveness.
- `"rejected"`: A version of the text with less descriptiveness.

This structure allows models to learn to prefer the _chosen_ response over the _rejected_ one, thereby aligning with human preferences in descriptive language.

## Generation script

The script used to generate this dataset can be found [here](https://github.com/huggingface/trl/blob/main/examples/datasets/lm-human-preferences-descriptiveness.py).
""")

if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    dataset = load_dataset(
        "json",
        data_files="https://openaipublic.blob.core.windows.net/lm-human-preferences/labels/descriptiveness/offline_5k.json",
        split="train",
    )

    dataset = dataset.filter(samples_not_all_same, num_proc=script_args.dataset_num_proc)

    dataset = dataset.map(
        to_prompt_completion,
        num_proc=script_args.dataset_num_proc,
        remove_columns=["query", "sample0", "sample1", "sample2", "sample3", "best"],
        fn_kwargs={"tokenizer": AutoTokenizer.from_pretrained("gpt2")},
    )

    # train_size taken from https://github.com/openai/lm-human-preferences/blob/cbfd210bb8b08f6bc5c26878c10984b90f516c66/launch.py#L79)
    dataset = dataset.train_test_split(train_size=4992)

    if script_args.push_to_hub:
        dataset.push_to_hub(script_args.repo_id)
        model_card.push_to_hub(script_args.repo_id, repo_type="dataset")
