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

import json
import os

from ghapi.all import GhApi


FOLDER_STRING = os.environ.get("FOLDER_STRING", "")
folder = f"benchmark/trl/{FOLDER_STRING}"
host_url = f"https://huggingface.co/datasets/trl-internal-testing/example-images/resolve/main/images/benchmark/{FOLDER_STRING}"

# Create a GitHub API instance
github_context = json.loads(os.environ["GITHUB_CONTEXT"])
token = os.environ["PERSONAL_ACCESS_TOKEN_GITHUB"]  # this needs to refreshed every 12 months
status_message = "**[COSTA BENCHMARK BOT]**: Here are the results"
body = status_message
repo = github_context["repository"]
owner, repo = repo.split("/")
api = GhApi(owner=owner, repo=repo, token=token)

# for each `.png` file in the folder, add it to the comment
for file in os.listdir(folder):
    if file.endswith(".png"):
        body += f"\n![{file}]({host_url}/{file})"

# Create a comment on the issue
api.issues.create_comment(issue_number=github_context["event"]["issue"]["number"], body=body)
