import multiprocessing
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset
from transformers import AutoTokenizer, HfArgumentParser


"""
python -i examples/datasets/tokenize_ds.py --debug --model HuggingFaceH4/zephyr-7b-beta
python -i examples/datasets/tokenize_ds.py --debug --model gpt2
"""


@dataclass
class ScriptArguments:
    debug: Optional[bool] = field(default=False, metadata={"help": "Enable debug mode"})
    dataset: str = field(
        default="trl-internal-testing/hh-rlhf-helpful-base-trl-style", metadata={"help": "The dataset to load"}
    )
    model: str = field(default="gpt2", metadata={"help": "The model to use for tokenization"})


if __name__ == "__main__":
    args = HfArgumentParser(ScriptArguments).parse_args_into_dataclasses()[0]
    ds = load_dataset(args.dataset)
    if args.debug:
        for key in ds:
            ds[key] = ds[key].select(range(50))
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.chat_template is None:
        tokenizer.chat_template = "{% for message in messages %}{{message['role'] + ': ' + message['content'] + '\n\n'}}{% endfor %}{{ eos_token }}"

    def process(row):
        row["chosen"] = tokenizer.apply_chat_template(row["chosen"], tokenize=False)
        row["rejected"] = tokenizer.apply_chat_template(row["rejected"], tokenize=False)
        return row

    ds = ds.map(
        process,
        num_proc=1 if args.debug else multiprocessing.cpu_count(),
        load_from_cache_file=False,
    )
    print(ds["train"][0]["chosen"])
