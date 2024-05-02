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
python -i examples/datasets/anthropic_hh.py --debug --push_to_hub
# actual push
python examples/datasets/anthropic_hh.py --push_to_hub --hf_entity trl-internal-testing
"""


api = HfApi()


@dataclass
class ScriptArguments:
    debug: Optional[bool] = field(default=False, metadata={"help": "Enable debug mode"})
    hf_entity: Optional[str] = field(default=None, metadata={"help": "The Hugging Face entity to use"})
    hf_repo_id: Optional[str] = field(
        default="hh-rlhf-helpful-base-trl-style", metadata={"help": "The Hugging Face repository ID"}
    )
    revision: Optional[str] = field(default="0.1.0", metadata={"help": "The revision of the repository"})
    update_main_revision: Optional[bool] = field(
        default=True, metadata={"help": "Update the main revision of the repository"}
    )
    push_to_hub: Optional[bool] = field(default=False, metadata={"help": "Push the dataset to the Hugging Face Hub"})


# GPT-4 generated 😄 Define a function to process the input and extract the dialogue into structured format
def extract_dialogue(input_text):
    # Split the input by lines and initialize variables
    lines = input_text.strip().split("\n\n")
    dialogue_list = []

    # Iterate through each line and extract the dialogue
    for line in lines:
        # Check if the line starts with "Human" or "Assistant" and split accordingly
        if line.startswith("Human:"):
            role = "user"
            content = line.replace("Human: ", "").strip()
        elif line.startswith("Assistant:"):
            role = "assistant"
            content = line.replace("Assistant: ", "").strip()
        else:
            # If the line doesn't start with "Human" or "Assistant", it's part of the previous message's content
            # Append it to the last message's content
            dialogue_list[-1]["content"] += "\n\n" + line.strip()
            continue

        # Append the extracted dialogue piece to the list
        dialogue_list.append({"role": role, "content": content})

    return dialogue_list


if __name__ == "__main__":
    args = HfArgumentParser(ScriptArguments).parse_args_into_dataclasses()[0]
    if args.hf_entity is None:
        args.hf_entity = api.whoami()["name"]
    full_repo_id = f"{args.hf_entity}/{args.hf_repo_id}"
    ds = load_dataset("Anthropic/hh-rlhf", data_dir="helpful-base")
    if args.debug:
        for key in ds:
            ds[key] = ds[key].select(range(50))

    def process(row):
        row["chosen"] = extract_dialogue(row["chosen"])
        row["rejected"] = extract_dialogue(row["rejected"])
        row["prompt"] = row["chosen"][0]["content"]
        return row

    ds = ds.map(
        process,
        num_proc=1 if args.debug else multiprocessing.cpu_count(),
        load_from_cache_file=False,
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

        sft_card = RepoCard.load(
            full_repo_id,
            repo_type="dataset",
        )
        sft_card.text = f"""\
# TRL's Anthropic HH Dataset

We preprocess the dataset using our standard `prompt, chosen, rejected` format.


## Reproduce this dataset

1. Download the `{file_name}` from the {repo_full_url}.
2. Run `{run_command}`
"""
        sft_card.push_to_hub(
            full_repo_id,
            repo_type="dataset",
        )
