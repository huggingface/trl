"""Build (or rebuild) the TermGrade base image as a Docker Space and wait for the build.

    python build.py [--space aminediroHF/termgrade-base] [--private]

HF Sandboxes pull it as `hf.co/spaces/<space>`. Run on a hopper-cpu node (needs HF_TOKEN or a cached login).
"""

import argparse
import os
import time

from huggingface_hub import HfApi


p = argparse.ArgumentParser()
p.add_argument("--space", default="aminediroHF/termgrade-base")
p.add_argument("--private", action="store_true")
args = p.parse_args()
api = HfApi()
here = os.path.dirname(os.path.abspath(__file__))
api.create_repo(args.space, repo_type="space", space_sdk="docker", private=args.private, exist_ok=True)
readme = (
    "---\ntitle: termgrade-base\nsdk: docker\napp_port: 7860\n---\n\n"
    "TermGrade base image (ai-and/termgrade-environments `base_image/`, plus tmux and asciinema for terminus-2), "
    "for running TermGrade tasks on HF Sandboxes as `hf.co/spaces/" + args.space + "`.\n"
)
api.upload_file(path_or_fileobj=readme.encode(), path_in_repo="README.md", repo_id=args.space, repo_type="space")
info = api.upload_file(
    path_or_fileobj=os.path.join(here, "Dockerfile"), path_in_repo="Dockerfile", repo_id=args.space, repo_type="space"
)
print("uploaded", info)
t0 = time.time()
while True:
    rt = api.get_space_runtime(args.space)
    print(f"{time.time() - t0:6.0f}s stage={rt.stage} hardware={rt.hardware}", flush=True)
    if rt.stage in ("RUNNING", "BUILD_ERROR", "RUNTIME_ERROR", "CONFIG_ERROR", "NO_APP_FILE"):
        break
    time.sleep(20)
print("final", rt.stage, rt.raw.get("errorMessage") if rt.raw else None)
