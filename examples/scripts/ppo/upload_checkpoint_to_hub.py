#!/usr/bin/env python3
"""
Upload a PPO checkpoint (or any saved model folder) to the Hugging Face Hub.

Update only root files (no checkpoint-* subdirs) when the repo already exists:
    python examples/scripts/ppo/upload_checkpoint_to_hub.py \
        --local_dir models/minimal/ppo_tldr \
        --repo_id YOUR_USERNAME/ppo_tldr \
        --root_files_only

Upload a single checkpoint folder:
    python examples/scripts/ppo/upload_checkpoint_to_hub.py \
        --local_dir models/minimal/ppo_tldr/checkpoint-1954 \
        --repo_id YOUR_USERNAME/ppo_tldr

Requires: pip install huggingface_hub; and run `huggingface-cli login`.
"""

# With --root_files_only, only upload these (no .pth, .pt, or other training artifacts).
ROOT_FILES_ALLOWED = (
    ".gitattributes",
    "README.md",
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "training_args.bin",
)


import argparse
import os

from huggingface_hub import HfApi, create_repo


def main():
    parser = argparse.ArgumentParser(description="Upload a model checkpoint to the Hugging Face Hub.")
    parser.add_argument(
        "--local_dir",
        type=str,
        required=True,
        help="Path to the folder to upload (e.g. models/minimal/ppo_tldr or .../checkpoint-1954)",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="Repo id on the Hub (e.g. your_username/ppo_tldr)",
    )
    parser.add_argument(
        "--root_files_only",
        action="store_true",
        help="Upload only files in the directory (no subdirs like checkpoint-*). Use to update an existing repo with the final model.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the repo as private (only if creating).",
    )
    args = parser.parse_args()

    api = HfApi()
    create_repo(args.repo_id, exist_ok=True, private=args.private)

    if args.root_files_only:
        for name in os.listdir(args.local_dir):
            if name not in ROOT_FILES_ALLOWED:
                continue
            path = os.path.join(args.local_dir, name)
            if os.path.isfile(path):
                api.upload_file(
                    path_or_fileobj=path,
                    path_in_repo=name,
                    repo_id=args.repo_id,
                    repo_type="model",
                )
                print(f"Uploaded {name}")
        print(f"Updated repo https://huggingface.co/{args.repo_id} with root files from {args.local_dir}")
    else:
        api.upload_folder(
            folder_path=args.local_dir,
            repo_id=args.repo_id,
            repo_type="model",
        )
        print(f"Uploaded {args.local_dir} to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
