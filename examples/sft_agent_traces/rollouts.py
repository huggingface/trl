# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
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

# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "openenv[harbor] @ git+https://github.com/openenv/OpenEnv.git@e5a9268b373478c6e8a6f1cbe43fe1a7845fb78a",
#     "websockets>=15,<16",
#     "huggingface_hub",
# ]
# ///

"""Run a coding agent on Harbor tasks and keep every rollout: to generate SFT traces with a teacher, or to evaluate a
model.

Each task runs as one `openenv harbor rollout` call: the harness (OpenCode by default) works in its own Hugging Face
Sandbox, OpenEnv sits between the harness and the model and records every LLM call (messages, tools, reply), and the
task's verifier grades the final answer. A result JSON per task is uploaded to `--push-to` as soon as it exists, so a
stopped job loses nothing.

The model is either an OpenAI-compatible endpoint (`--llm-url`, Inference Providers by default), which is how the
teacher runs, or served by the script itself with vLLM (`--serve`, needs a GPU), which is how a student is evaluated.

Generate traces with a teacher (no GPU, the teacher runs on Inference Providers):

```sh
hf jobs uv run --flavor cpu-upgrade --timeout 4h --secrets HF_TOKEN \
    https://raw.githubusercontent.com/huggingface/trl/main/examples/sft_agent_traces/rollouts.py \
    --model qwen/qwen3.5-397b-a17b --llm-url https://router.huggingface.co/novita/v3/openai \
    --dataset FineEnvs/SmolDataEnvs-harbor-train --exclude-datasets-of FineEnvs/SmolDataEnvs-harbor-eval \
    --num-tasks 640 --workers 12 --push-to <you>/smoldataenvs-rollouts --private --run-name teacher-397b
```

Evaluate a model (vLLM on the job's GPU):

```sh
hf jobs uv run --flavor a10g-large --timeout 3h --secrets HF_TOKEN \
    https://raw.githubusercontent.com/huggingface/trl/main/examples/sft_agent_traces/rollouts.py \
    --model Qwen/Qwen3.5-2B --serve --reward-key correctness,reward \
    --dataset FineEnvs/SmolDataEnvs-harbor-eval --num-tasks 144 --workers 8 \
    --push-to <you>/smoldataenvs-rollouts --private --run-name eval-qwen3.5-2b
```

Locally, replace `hf jobs uv run ... rollouts.py` with `uv run examples/sft_agent_traces/rollouts.py`. Sandboxes are
Hugging Face Jobs either way, billed to the account of `HF_TOKEN`.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import queue
import re
import subprocess
import sys
import tempfile
import time
import tomllib
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.hf_api import RepoFolder


# Image of the SmolDataEnvs train split. The eval split ships the same Dockerfile but doesn't pin an image in
# `task.toml`, and HF Sandboxes can't build one, so tasks without an image get this one.
SMOLDATAENVS_IMAGE = (
    "docker.io/savatar101/env-data-agent-train@sha256:7b3f7e22780b431ce0d43717dcef6875cde84baeee824cc87e968374d62d3470"
)

# huggingface_hub names a Job after its image, and for a digest-pinned image that name goes past the Hub's 100
# characters, so every sandbox creation fails. Fixed in huggingface/huggingface_hub#5306: until a release ships it,
# this patch, loaded by every `openenv` process the script starts, truncates the name.
_JOB_NAME_PATCH = """
try:
    import huggingface_hub.hf_api as _hf_api
    _orig = _hf_api._default_job_name_from_image
    def _short_name(image, command, **kwargs):
        name = _orig(image, command, **kwargs)
        return name if len(name) <= 100 else name[:91] + name[-9:]
    _hf_api._default_job_name_from_image = _short_name
except Exception:
    pass
"""


def task_reward(result: dict) -> float | None:
    """The task's reward, whatever the grader called it.

    SmolDataEnvs graders write either a single `reward` or several keys (`correctness`, `submission`,
    `tool_efficiency`), even within one split, and OpenEnv leaves `reward` empty when the key it expects is missing.
    `correctness` is the answer check in both.
    """
    rewards = result.get("rewards") or {}
    for value in (rewards.get("correctness"), result.get("reward"), rewards.get("reward")):
        if value is not None:
            return value
    return None


def kaggle_datasets(dataset: str, workdir: pathlib.Path) -> dict[str, str]:
    """Map every task of a SmolDataEnvs split to the Kaggle dataset its files come from (only `task.toml`s are read)."""
    root = pathlib.Path(
        snapshot_download(dataset, repo_type="dataset", local_dir=workdir, allow_patterns="tasks/*/task.toml")
    )
    return {
        f"tasks/{toml.parent.name}": tomllib.loads(toml.read_text())["metadata"]["kaggle_dataset_name"]
        for toml in root.glob("tasks/*/task.toml")
    }


def sandbox_env(config: dict) -> list:
    """Every value of an `env` table in a `task.toml`, at any depth: what the task passes into its sandbox."""
    values = []
    for key, value in config.items():
        if isinstance(value, dict):
            values += list(value.values()) if key == "env" else sandbox_env(value)
    return values


def fetch_tasks(
    dataset: str, indices: list[int], workdir: pathlib.Path, docker_image: str, exclude_of: str | None = None
) -> list[str]:
    """Download only the picked tasks (a split can hold thousands) and pin an image on those without one.

    With `exclude_of`, tasks built on a Kaggle dataset that also backs a task of that split are skipped first, and
    `indices` count the remaining tasks.
    """
    tree = HfApi().list_repo_tree(dataset, repo_type="dataset", path_in_repo="tasks")
    names = sorted(f.path for f in tree if isinstance(f, RepoFolder))  # one folder per task, as OpenEnv indexes them
    if exclude_of:
        held_out = set(kaggle_datasets(exclude_of, workdir / "exclude").values())
        own = kaggle_datasets(dataset, workdir / "metadata")
        kept = [n for n in names if own[n] not in held_out]
        print(f"{len(names) - len(kept)} of {len(names)} tasks share a Kaggle dataset with {exclude_of}, skipped")
        names = kept
    if indices and max(indices) >= len(names):
        print(f"only {len(names)} tasks, indices from {len(names)} on are skipped", flush=True)
    picked = [names[i] for i in indices if i < len(names)]
    snapshot_download(dataset, repo_type="dataset", local_dir=workdir, allow_patterns=[f"{p}/**" for p in picked])
    for task in picked:
        toml = workdir / task / "task.toml"
        text = toml.read_text()
        if "docker_image" not in text:
            text = text.replace("[environment]\n", f'[environment]\ndocker_image = "{docker_image}"\n', 1)
        # The task's files come from a public bucket, so the sandbox doesn't need the caller's token.
        text = re.sub(r'(?m)^HF_TOKEN\s*=\s*"\$\{HF_TOKEN\}"\s*\n', "", text)
        if any("HF_TOKEN" in str(value) for value in sandbox_env(tomllib.loads(text))):
            sys.exit(f"{task}/task.toml passes HF_TOKEN to the sandbox in a form this script doesn't remove")
        toml.write_text(text)
    return [p.split("/")[-1] for p in picked]


def serve(model: str, max_model_len: int, log_path: pathlib.Path) -> tuple[str, subprocess.Popen]:
    """Serve `model` with vLLM on this machine and return its OpenAI-compatible URL."""
    cmd = [
        "uvx", "--from", "vllm==0.25.1", "--with", "transformers>=5.22", "vllm", "serve", model,
        "--host", "127.0.0.1", "--port", "8000", "--served-model-name", model,
        "--dtype", "bfloat16", "--max-model-len", str(max_model_len), "--gpu-memory-utilization", "0.85",
        "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_xml",  # Qwen3.5's tool-call format
        "--default-chat-template-kwargs", '{"enable_thinking": false}',
        "--generation-config", "vllm",
        "--logprobs-mode", "processed_logprobs", "--return-tokens-as-token-ids",
        "--limit-mm-per-prompt", '{"image": 0, "video": 0}',
    ]  # fmt: skip
    proc = subprocess.Popen(cmd, stdout=log_path.open("w"), stderr=subprocess.STDOUT)
    url = "http://127.0.0.1:8000/v1"
    for _ in range(180):
        if proc.poll() is not None:
            sys.exit(f"vLLM exited with code {proc.returncode}, see {log_path}")
        try:
            urllib.request.urlopen(f"{url}/models", timeout=5)
            print(f"vLLM is serving {model}", flush=True)
            return url, proc
        except OSError:
            time.sleep(10)
    sys.exit(f"vLLM did not come up in time, see {log_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True, help="model the harness calls, as the endpoint names it")
    p.add_argument("--llm-url", default="https://router.huggingface.co/v1", help="OpenAI-compatible endpoint")
    p.add_argument("--serve", action="store_true", help="serve --model with vLLM here instead of calling --llm-url")
    p.add_argument("--max-model-len", type=int, default=65536, help="vLLM context length, with --serve")
    p.add_argument("--dataset", default="FineEnvs/SmolDataEnvs-harbor-train", help="Hub dataset of Harbor tasks")
    p.add_argument("--start", type=int, default=0, help="first task, in the sorted order of the split")
    p.add_argument("--num-tasks", type=int, default=10, help="number of tasks from --start")
    p.add_argument(
        "--task-indices",
        default=os.environ.get("TASK_INDICES"),
        help="comma-separated task indices, instead of a range (or the TASK_INDICES env var, for long lists)",
    )
    p.add_argument("--workers", type=int, default=8, help="tasks in flight at once, one sandbox each")
    p.add_argument("--harness", default="opencode")
    p.add_argument("--reward-key", default=None, help="reward the verifier reports as the training signal")
    p.add_argument("--docker-image", default=SMOLDATAENVS_IMAGE, help="image for tasks that don't pin one")
    p.add_argument(
        "--exclude-datasets-of",
        default=None,
        help="skip tasks whose Kaggle dataset backs a task of this split, e.g. the eval split when generating traces",
    )
    p.add_argument("--task-timeout", type=int, default=2400, help="seconds per rollout before it is killed")
    p.add_argument("--push-to", required=True, help="Hub dataset the result JSONs are uploaded to")
    p.add_argument("--run-name", default=None, help="folder of this run in --push-to")
    p.add_argument("--private", action="store_true", help="create --push-to as a private dataset")
    args = p.parse_args()

    run_name = args.run_name or time.strftime("run-%Y%m%d-%H%M%S")
    workdir = pathlib.Path(tempfile.mkdtemp(prefix="rollouts-"))
    out = workdir / "results"
    out.mkdir()
    api = HfApi()
    api.create_repo(args.push_to, repo_type="dataset", private=args.private, exist_ok=True)

    def upload(path: pathlib.Path):
        try:
            api.upload_file(
                path_or_fileobj=path, path_in_repo=f"{run_name}/{path.name}", repo_id=args.push_to,
                repo_type="dataset", commit_message=f"{run_name}: {path.name}",
            )  # fmt: skip
        except Exception as e:  # one failed upload shouldn't stop the run, the file is still on disk
            print(f"upload failed for {path.name}: {e}", flush=True)

    if args.task_indices:
        # Sorted, since the local copy lists tasks in the split's order and the harness picks them by position.
        indices = sorted({int(i) for i in args.task_indices.split(",")})
    else:
        indices = list(range(args.start, args.start + args.num_tasks))
    names = fetch_tasks(args.dataset, indices, workdir / "tasks_root", args.docker_image, args.exclude_datasets_of)
    print(f"{len(names)} tasks from {args.dataset}", flush=True)

    patch_dir = workdir / "patch"
    patch_dir.mkdir()
    (patch_dir / "sitecustomize.py").write_text(_JOB_NAME_PATCH)
    env = {
        **os.environ,
        "PYTHONPATH": f"{patch_dir}{os.pathsep}{os.environ.get('PYTHONPATH', '')}",
        "OPENENV_LLM_API_KEY": os.environ.get("HF_TOKEN", ""),
        "PYTHONUNBUFFERED": "1",
    }

    server = None
    if args.serve:
        llm_url, server = serve(args.model, args.max_model_len, out / "vllm.log")
    else:
        llm_url = args.llm_url

    # One capture port per worker. The first rollout of each worker downloads gradio's tunnel binary to the same
    # path, and concurrent downloads fail with "Text file busy", so the first wave starts 20 s apart.
    ports = queue.Queue()
    for w in range(args.workers):
        ports.put(8100 + w)

    def rollout(position: int, name: str):
        if position < args.workers:
            time.sleep(20 * position)
        port = ports.get()
        result = out / f"result_{indices[position]:05d}.json"
        cmd = [
            "openenv", "harbor", "rollout", "--llm-url", llm_url, "--model", args.model,
            "--dataset", str(workdir / "tasks_root"), "--task-index", str(position), "-n", "1",
            "--harness", args.harness, "--sandbox", "hf-sandbox", "--port", str(port),
            "--trials-dir", str(workdir / "trials"), "--out", str(result),
        ]  # fmt: skip
        if args.reward_key:
            cmd += ["--reward-key", args.reward_key]
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            output, _ = proc.communicate(timeout=args.task_timeout)
        except subprocess.TimeoutExpired:
            # SIGTERM first, so `openenv` closes its tunnel and sandbox before the port is reused
            proc.terminate()
            try:
                proc.communicate(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()
            print(f"[{name}] stopped after {args.task_timeout} s", flush=True)
            return
        finally:
            ports.put(port)
        if not result.exists():
            fatal = [line.strip() for line in output.splitlines() if "FATAL" in line or "Error" in line]
            print(f"[{name}] no result (exit {proc.returncode}): {' | '.join(fatal[-3:])}", flush=True)
            return
        upload(result)
        # OpenEnv labels most OpenCode rollouts FAILED, since OpenCode exits with code 1 even when it finishes
        # normally, so report what the result says.
        data = json.loads(result.read_text())
        data = data[0] if isinstance(data, list) else data
        print(
            f"[{name}] reward={task_reward(data)} turns={data.get('n_turns')} {data.get('wall_s') or 0:.0f}s",
            flush=True,
        )

    # A task's index in the local copy is its position in `names`.
    with ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(rollout, position, name): name for position, name in enumerate(names)}
    for future, name in futures.items():
        if future.exception() is not None:  # one task's error shouldn't hide, or stop, the others
            print(f"[{name}] error: {future.exception()!r}", flush=True)

    results = []
    for path in sorted(out.glob("result_*.json")):
        data = json.loads(path.read_text())
        results.extend(data if isinstance(data, list) else [data])
    rewards = [task_reward(r) for r in results]
    summary = {
        "model": args.model,
        "dataset": args.dataset,
        "tasks": len(names),
        "results": len(results),
        "solved": sum(r == 1.0 for r in rewards),
        "ungraded": sum(r is None for r in rewards),
    }
    summary["solve_rate"] = summary["solved"] / max(len(names), 1)  # a task with no result counts as unsolved
    # named after the first task, so Jobs splitting one run into slices don't overwrite each other's
    summary_path = out / f"summary_{indices[0]:05d}.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    upload(summary_path)
    if args.serve:
        upload(out / "vllm.log")
    print(json.dumps(summary, indent=2), flush=True)
    if server is not None:
        server.terminate()


if __name__ == "__main__":
    main()
