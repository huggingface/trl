import multiprocessing
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.repocard import RepoCard
from transformers import AutoTokenizer, HfArgumentParser


"""
# debug
python -i examples/datasets/descriptiveness_sentiment.py --debug --push_to_hub
# actual push
python examples/datasets/descriptiveness_sentiment.py --push_to_hub --hf_entity trl-internal-testing
"""


api = HfApi()


@dataclass
class ScriptArguments:
    debug: Optional[bool] = field(default=False, metadata={"help": "Enable debug mode"})
    hf_entity: Optional[str] = field(default=None, metadata={"help": "The Hugging Face entity to use"})
    hf_repo_id: Optional[str] = field(
        default="descriptiveness-sentiment-trl-style", metadata={"help": "The Hugging Face repository ID"}
    )
    revision: Optional[str] = field(default="0.1.0", metadata={"help": "The revision of the repository"})
    update_main_revision: Optional[bool] = field(
        default=True, metadata={"help": "Update the main revision of the repository"}
    )
    push_to_hub: Optional[bool] = field(default=False, metadata={"help": "Push the dataset to the Hugging Face Hub"})


def deduplicate_query(ds):
    query = set()
    ranges = []
    for i in range(len(ds)):
        query_str = str(ds[i]["query"])
        if query_str not in query:
            query.add(query_str)
            ranges.append(i)
    return ds.select(ranges)


if __name__ == "__main__":
    args = HfArgumentParser(ScriptArguments).parse_args_into_dataclasses()[0]
    if args.hf_entity is None:
        args.hf_entity = api.whoami()["name"]
    full_repo_id = f"{args.hf_entity}/{args.hf_repo_id}"

    model_name = "gpt2"
    dataset_tokenizer = AutoTokenizer.from_pretrained("gpt2")  # of the dataset

    ################
    # Dataset
    ################
    descriptiveness = hf_hub_download(
        repo_id="vwxyzjn/lm-human-preferences",
        repo_type="dataset",
        filename="descriptiveness/offline_5k.json",  # or "sentiment/offline_5k.json"
    )
    descriptiveness_ds = Dataset.from_json(descriptiveness)
    descriptiveness_ds = deduplicate_query(descriptiveness_ds)
    sentiment = hf_hub_download(
        repo_id="vwxyzjn/lm-human-preferences",
        repo_type="dataset",
        filename="sentiment/offline_5k.json",  # or "sentiment/offline_5k.json"
    )
    sentiment_ds = Dataset.from_json(sentiment)
    sentiment_ds = deduplicate_query(sentiment_ds)
    ds = DatasetDict({"descriptiveness": descriptiveness_ds, "sentiment": sentiment_ds})

    if args.debug:
        for key in ds:
            ds[key] = ds[key].select(range(50))
    cnndm_batches = ["batch0_cnndm", "cnndm0", "cnndm2"]

    # columns are `['sample2', 'sample3', 'sample0', 'query', 'sample1', 'best']`
    NUM_SAMPLES = 4

    def process(row):
        for j in range(NUM_SAMPLES):
            row[f"sample{j}"] = dataset_tokenizer.batch_decode(row[f"sample{j}"])
        row["prompt"] = dataset_tokenizer.batch_decode(row["query"])
        row["prompt"] = [item.strip() for item in row["prompt"]]
        row["chosen"] = []
        row["rejected"] = []
        for i in range(len(row["best"])):
            best_idx = row["best"][i]
            row["chosen"].append(
                [
                    {"role": "user", "content": row["prompt"][i].strip()},
                    {"role": "assistant", "content": row[f"sample{best_idx}"][i].strip()},
                ]
            )
            rejected_ids = [k for k in [0, 1, 2, 3] if k != best_idx]
            rejected_idx = np.argmin(rejected_ids)  # select the first rejected sample for reproducibility
            row["rejected"].append(
                [
                    {"role": "user", "content": row["prompt"][i].strip()},
                    {"role": "assistant", "content": row[f"sample{rejected_idx}"][i].strip()},
                ]
            )
        return row

    ds = ds.map(
        process,
        batched=True,
        num_proc=1 if args.debug else multiprocessing.cpu_count(),
        load_from_cache_file=False,
    )
    for key in ds:  # reorder columns
        ds[key] = ds[key].select_columns(["prompt", "chosen", "rejected"])
    if args.push_to_hub:
        revisions = ["main"] if args.update_main_revision else []
        revisions.append(args.revision)

        # get the commnad used to run the script
        run_command = " ".join(["python"] + sys.argv)

        for revision in revisions:
            ds.push_to_hub(full_repo_id, revision=revision)
            repo_full_url = f"https://huggingface.co/datasets/{full_repo_id}/tree/{revision}"

            # get the name of the current file
            file_name = __file__.split("/")[-1]
            api.upload_file(
                path_or_fileobj=__file__,
                path_in_repo=file_name,
                revision=revision,
                repo_id=full_repo_id,
                repo_type="dataset",
            )

        sft_card = RepoCard.load(
            full_repo_id,
            repo_type="dataset",
        )
        sft_card.text = f"""\
# TRL's Sentiment and Descriptiveness Preference Dataset

The dataset comes from https://arxiv.org/abs/1909.08593, one of the earliest RLHF work from OpenAI.

We preprocess the dataset using our standard `prompt, chosen, rejected` format.

## Reproduce this dataset

1. Download the `{file_name}` from the {repo_full_url}.
2. Run `{run_command}`
"""
        sft_card.push_to_hub(
            full_repo_id,
            repo_type="dataset",
        )
