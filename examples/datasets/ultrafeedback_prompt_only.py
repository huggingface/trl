from dataclasses import dataclass
from typing import Optional

from datasets import load_dataset
from transformers import HfArgumentParser


@dataclass
class ScriptArguments:
    r"""
    Arguments for the script.

    Args:
        push_to_hub (`bool`, *optional*, defaults to `False`):
            Whether to push the dataset to the Hugging Face Hub.
        repo_id (`str`, *optional*, defaults to `"trl-lib/ultrafeedback-prompt"`):
            Hugging Face repository ID to push the dataset to.
        dataset_num_proc (`Optional[int]`, *optional*, defaults to `None`):
            Number of workers to use for dataset processing.
    """

    push_to_hub: bool = False
    repo_id: str = "trl-lib/ultrafeedback-prompt"
    dataset_num_proc: Optional[int] = None


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]

    dataset = load_dataset("argilla/ultrafeedback-binarized-preferences-cleaned", split="train")
    dataset = dataset.map(
        lambda x: {"prompt": [{"role": "user", "content": x["prompt"]}]},
        remove_columns=[
            "source",
            "chosen",
            "chosen-rating",
            "chosen-model",
            "rejected",
            "rejected-rating",
            "rejected-model",
        ],
        num_proc=args.dataset_num_proc,
    )
    dataset = dataset.train_test_split(test_size=0.05, seed=42)

    if args.push_to_hub:
        dataset.push_to_hub(args.repo_id)
