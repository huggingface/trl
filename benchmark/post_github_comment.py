import argparse
import math
import os
import shlex
import subprocess
import time
import uuid
from distutils.util import strtobool

from ghapi.all import GhApi
import json

FOLDER_STRING = os.environ.get("FOLDER_STRING", "")
folder = f"benchmark/trl/{FOLDER_STRING}"
host_url = f"https://huggingface.co/datasets/trl-internal-testing/example-images/resolve/main/images/benchmark/{FOLDER_STRING}"

# Create a GitHub API instance
github_context = json.loads(os.environ['GITHUB_CONTEXT'])
token = os.environ["PERSONAL_ACCESS_TOKEN_GITHUB"] # this needs to refreshed every 12 months
status_message = f"**[COSTA BENCHMARK BOT]**: Here are the results"
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