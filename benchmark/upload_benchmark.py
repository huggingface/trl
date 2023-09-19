from dataclasses import dataclass

import tyro
from huggingface_hub import HfApi


@dataclass
class Args:
    folder_path: str = "benchmark/trl"
    path_in_repo: str = "images/benchmark"
    repo_id: str = "trl-internal-testing/example-images"
    repo_type: str = "dataset"


args = tyro.cli(Args)
api = HfApi()

api.upload_folder(
    folder_path=args.folder_path,
    path_in_repo=args.path_in_repo,
    repo_id=args.repo_id,
    repo_type=args.repo_type,
)
