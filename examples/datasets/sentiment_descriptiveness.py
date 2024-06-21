import sys
from dataclasses import dataclass, field
from typing import Optional

from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.repocard import RepoCard
from transformers import AutoTokenizer, HfArgumentParser


"""
# debug
python -i examples/datasets/sentiment_descriptiveness.py  --push_to_hub
# actual push
python examples/datasets/sentiment_descriptiveness.py \
    --hf_repo_id sentiment-trl-style \
    --task sentiment \
    --push_to_hub \
    --hf_entity trl-internal-testing
python examples/datasets/sentiment_descriptiveness.py \
    --hf_repo_id descriptiveness-trl-style \
    --task descriptiveness \
    --push_to_hub \
    --hf_entity trl-internal-testing
"""


api = HfApi()


@dataclass
class ScriptArguments:
    debug: Optional[bool] = field(default=False, metadata={"help": "Enable debug mode"})
    hf_entity: Optional[str] = field(default=None, metadata={"help": "The Hugging Face entity to use"})
    hf_repo_id: Optional[str] = field(
        default="sentiment-trl-style", metadata={"help": "The Hugging Face repository ID"}
    )
    revision: Optional[str] = field(default="0.1.0", metadata={"help": "The revision of the repository"})
    update_main_revision: Optional[bool] = field(
        default=True, metadata={"help": "Update the main revision of the repository"}
    )
    push_to_hub: Optional[bool] = field(default=False, metadata={"help": "Push the dataset to the Hugging Face Hub"})
    task: str = field(default="sentiment", metadata={"help": "The task of the dataset"})


task_to_filename = {
    "sentiment": "sentiment/offline_5k.json",
    "descriptiveness": "descriptiveness/offline_5k.json",
}


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
    json = hf_hub_download(
        repo_id="vwxyzjn/lm-human-preferences",
        repo_type="dataset",
        filename=task_to_filename[args.task],
    )

    MAGIC_TRAIN_NUMBER = 4992  # taken from https://github.com/openai/lm-human-preferences/blob/cbfd210bb8b08f6bc5c26878c10984b90f516c66/launch.py#L70
    individual_ds = Dataset.from_json(json)
    individual_ds = deduplicate_query(individual_ds)
    ds = DatasetDict(
        {
            "train": individual_ds.select(range(MAGIC_TRAIN_NUMBER)),
            "test": individual_ds.select(range(MAGIC_TRAIN_NUMBER, len(individual_ds))),
        }
    )

    MAX_DEBUG_SAMPLES = 50
    if args.debug:
        for key in ds:
            ds[key] = ds[key].select(range(min(MAX_DEBUG_SAMPLES, len(ds[key]))))

    # columns are `['sample2', 'sample3', 'sample0', 'query', 'sample1', 'best']`
    NUM_SAMPLES = 4

    # edge cases handling: remove the cases where all samples are the same
    def filter(row):
        best_idx = row["best"]
        chosen_sample = row[f"sample{best_idx}"]
        if all(chosen_sample == row[f"sample{j}"] for j in range(NUM_SAMPLES)):
            return False
        else:
            return True

    print("=== Before filtering ===", ds)
    ds = ds.filter(filter, load_from_cache_file=False)
    print("=== After filtering ===", ds)

    # here we simply take the preferred sample as the chosen one and the first non-preferred sample as the rejected one
    def process(row):
        for j in range(NUM_SAMPLES):
            row[f"sample{j}"] = dataset_tokenizer.batch_decode(row[f"sample{j}"])
        row["prompt"] = dataset_tokenizer.batch_decode(row["query"])
        row["prompt"] = [item.strip() for item in row["prompt"]]
        row["chosen"] = []
        row["rejected"] = []
        for i in range(len(row["best"])):
            best_idx = row["best"][i]
            chosen_sample = row[f"sample{best_idx}"][i].strip()
            row["chosen"].append(
                [
                    {"role": "user", "content": row["prompt"][i].strip()},
                    {"role": "assistant", "content": chosen_sample},
                ]
            )
            # find the first rejected sample which is different from the chosen one
            rejected_idx = -1
            for k in range(4):
                if k != best_idx and row[f"sample{k}"][i].strip() != chosen_sample:
                    rejected_idx = k
                    break
            rejected_sample = row[f"sample{rejected_idx}"][i].strip()
            assert rejected_idx != -1, "No rejected sample found! This should not happen!"
            row["rejected"].append(
                [
                    {"role": "user", "content": row["prompt"][i].strip()},
                    {"role": "assistant", "content": rejected_sample},
                ]
            )
            assert chosen_sample != rejected_sample
        return row

    ds = ds.map(
        process,
        batched=True,
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
# TRL's Preference Dataset: {args.task}
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
