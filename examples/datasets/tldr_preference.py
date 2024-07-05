import multiprocessing
import sys
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset
from huggingface_hub import HfApi
from huggingface_hub.repocard import RepoCard
from transformers import HfArgumentParser


"""
# debug
python -i examples/datasets/tldr_preference.py --debug --push_to_hub
# actual push
python examples/datasets/tldr_preference.py --push_to_hub --hf_entity trl-internal-testing
"""


api = HfApi()


@dataclass
class ScriptArguments:
    debug: Optional[bool] = field(default=False, metadata={"help": "Enable debug mode"})
    hf_entity: Optional[str] = field(default=None, metadata={"help": "The Hugging Face entity to use"})
    hf_repo_id: Optional[str] = field(
        default="tldr-preference-trl-style", metadata={"help": "The Hugging Face repository ID"}
    )
    sft_hf_repo_id: Optional[str] = field(
        default="tldr-preference-sft-trl-style", metadata={"help": "The Hugging Face repository ID"}
    )
    revision: Optional[str] = field(default="0.1.0", metadata={"help": "The revision of the repository"})
    update_main_revision: Optional[bool] = field(
        default=True, metadata={"help": "Update the main revision of the repository"}
    )
    push_to_hub: Optional[bool] = field(default=False, metadata={"help": "Push the dataset to the Hugging Face Hub"})


if __name__ == "__main__":
    args = HfArgumentParser(ScriptArguments).parse_args_into_dataclasses()[0]
    if args.hf_entity is None:
        args.hf_entity = api.whoami()["name"]
    full_repo_id = f"{args.hf_entity}/{args.hf_repo_id}"
    full_sft_repo_id = f"{args.hf_entity}/{args.sft_hf_repo_id}"

    ################
    # Preference dataset
    ################
    ds = load_dataset("openai/summarize_from_feedback", "comparisons")
    if args.debug:
        for key in ds:
            ds[key] = ds[key].select(range(50))
    cnndm_batches = ["batch0_cnndm", "cnndm0", "cnndm2"]
    if not args.debug:
        ds["validation_cnndm"] = ds["validation"].filter(lambda x: x["batch"] in cnndm_batches)
    ds["validation"] = ds["validation"].filter(lambda x: x["batch"] not in cnndm_batches)

    tldr_format_str = "SUBREDDIT: r/{subreddit}\n\nTITLE: {title}\n\nPOST: {post}\n\nTL;DR:"
    cnndm_format_str = "Article:\n{article}\n\nTL;DR:"

    def process(row):
        format_str = cnndm_format_str if row["batch"] in cnndm_batches else tldr_format_str
        row["prompt"] = format_str.format(**row["info"])
        choice = row["choice"]
        # need to remove the leading space
        chosen = row["summaries"][choice]["text"].strip()
        rejected = row["summaries"][1 - choice]["text"].strip()
        row["chosen"] = [{"role": "user", "content": row["prompt"]}, {"role": "assistant", "content": chosen}]
        row["rejected"] = [{"role": "user", "content": row["prompt"]}, {"role": "assistant", "content": rejected}]
        return row

    ds = ds.map(
        process,
        num_proc=1 if args.debug else multiprocessing.cpu_count(),
        load_from_cache_file=False,
    )
    for key in ds:  # reorder columns
        ds[key] = ds[key].select_columns(
            ["prompt", "chosen", "rejected", "info", "summaries", "choice", "worker", "batch", "split", "extra"]
        )
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

        preference_card = RepoCard.load(
            full_repo_id,
            repo_type="dataset",
        )
        preference_card.text = f"""\
# TRL's TL;DR Preference Dataset

We preprocess the dataset using our standard `prompt, chosen, rejected` format.

## Source of the dataset

We take the dataset from https://huggingface.co/datasets/openai/summarize_from_feedback.

## Reproduce this dataset

1. Download the `{file_name}` from the {repo_full_url}.
2. Run `{run_command}`
"""
        preference_card.push_to_hub(
            full_repo_id,
            repo_type="dataset",
        )

    ################
    # SFT dataset
    ################
    sft_ds = load_dataset("vwxyzjn/summarize_from_feedback_tldr_3_filtered")
    if args.debug:
        for key in sft_ds:
            sft_ds[key] = sft_ds[key].select(range(50))

    def sft_process(row):
        row["prompt"] = tldr_format_str.format(**row)
        row["messages"] = [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["summary"]},
        ]
        return row

    sft_ds = sft_ds.map(
        sft_process,
        num_proc=1 if args.debug else multiprocessing.cpu_count(),
        load_from_cache_file=False,
    )
    for key in sft_ds:  # reorder columns
        sft_ds[key] = sft_ds[key].select_columns(["prompt", "messages", "id", "subreddit", "title", "post", "summary"])
    if args.push_to_hub:
        revisions = ["main"] if args.update_main_revision else []
        revisions.append(args.revision)

        # get the commnad used to run the script
        run_command = " ".join(["python"] + sys.argv)

        for revision in revisions:
            sft_ds.push_to_hub(full_sft_repo_id, revision=revision)
            repo_full_url = f"https://huggingface.co/datasets/{full_sft_repo_id}/tree/{revision}"

            # get the name of the current file
            file_name = __file__.split("/")[-1]
            api.upload_file(
                path_or_fileobj=__file__,
                path_in_repo=file_name,
                revision=revision,
                repo_id=full_sft_repo_id,
                repo_type="dataset",
            )

        sft_card = RepoCard.load(
            full_sft_repo_id,
            repo_type="dataset",
        )
        sft_card.text = f"""\
# TRL's TL;DR SFT Dataset

We preprocess the dataset using our standard `prompt, messages` format.

## Source of the dataset

We take the dataset from https://huggingface.co/datasets/vwxyzjn/summarize_from_feedback_tldr_3_filtered.

## Reproduce this dataset

1. Download the `{file_name}` from the {repo_full_url}.
2. Run `{run_command}`
"""
