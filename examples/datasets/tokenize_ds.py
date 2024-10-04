# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
from transformers import AutoTokenizer, HfArgumentParser

from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE


"""
python -i examples/datasets/tokenize_ds.py --model HuggingFaceH4/zephyr-7b-beta
python -i examples/datasets/tokenize_ds.py --model gpt2
"""


@dataclass
class ScriptArguments:
    dataset_name: str = field(
        default="trl-internal-testing/hh-rlhf-helpful-base-trl-style", metadata={"help": "The dataset to load"}
    )
    model: str = field(default="gpt2", metadata={"help": "The model to use for tokenization"})
    dataset_num_proc: Optional[int] = field(
        default=None, metadata={"help": "The number of workers to use to tokenize the data"}
    )


if __name__ == "__main__":
    script_args = HfArgumentParser(ScriptArguments).parse_args_into_dataclasses()[0]
    dataset = load_dataset(script_args.dataset_name)
    tokenizer = AutoTokenizer.from_pretrained(script_args.model)
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE

    def process(row):
        row["chosen"] = tokenizer.apply_chat_template(row["chosen"], tokenize=False)
        row["rejected"] = tokenizer.apply_chat_template(row["rejected"], tokenize=False)
        return row

    dataset = dataset.map(process, num_proc=script_args.dataset_num_proc)
    print(dataset["train"][0]["chosen"])
