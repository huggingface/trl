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
