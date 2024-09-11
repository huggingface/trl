from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset
from transformers import AutoTokenizer, HfArgumentParser


"""
python -i examples/datasets/tokenize_ds.py --model HuggingFaceH4/zephyr-7b-beta
python -i examples/datasets/tokenize_ds.py --model gpt2
"""


@dataclass
class ScriptArguments:
    dataset: str = field(
        default="trl-internal-testing/hh-rlhf-helpful-base-trl-style", metadata={"help": "The dataset to load"}
    )
    model: str = field(default="gpt2", metadata={"help": "The model to use for tokenization"})
    dataset_num_proc: Optional[int] = field(
        default=None, metadata={"help": "The number of workers to use to tokenize the data"}
    )


if __name__ == "__main__":
    args = HfArgumentParser(ScriptArguments).parse_args_into_dataclasses()[0]
    ds = load_dataset(args.dataset)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.chat_template is None:
        tokenizer.chat_template = "{% for message in messages %}{{message['role'] + ': ' + message['content'] + '\n\n'}}{% endfor %}{{ eos_token }}"

    def process(row):
        row["chosen"] = tokenizer.apply_chat_template(row["chosen"], tokenize=False)
        row["rejected"] = tokenizer.apply_chat_template(row["rejected"], tokenize=False)
        return row

    ds = ds.map(process, num_proc=args.dataset_num_proc)
    print(ds["train"][0]["chosen"])
