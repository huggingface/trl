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
# dependencies = [
#     "trl[vllm]",
#     "trackio",
#     "openenv>=0.15.0",
#     "openenv-data-analysis-env @ git+https://github.com/openenv/OpenEnv.git@v0.15.0#subdirectory=envs/data_analysis_env",
#     "flash-linear-attention>=0.6.0",
# ]
# ///


"""
On-policy distillation of a data-analysis agent with OpenEnv's data_analysis_env: the student (Qwen3.5-2B) answers
SmolDataEnvs questions by exploring the task's files with a `run_python` tool over up to 10 turns, and the teacher
(Qwen3.5-9B) scores every token the student wrote. With `--first-turn`, only the first turn is distilled. See
README.md for details and results.

The script starts the environment server itself, so it must run as root on Linux (a container, or Hugging Face
Jobs):

```sh
hf jobs uv run --flavor a100-large --timeout 2h \
    examples/distillation_data_analysis_env/distillation_data_analysis_env.py
```
"""

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from data_analysis_env import CallToolAction, DataAnalysisEnv
from data_analysis_env.server.tasks import (
    DATASET_ID,
    build_prompt,
    list_bucket_prefix,
    runnable,
    safe_task_id,
)
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi

from trl import DistillationConfig, DistillationTrainer


# Hugging Face Jobs images have no nvcc (FlashInfer's sampler) and no DeepGEMM build
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


DATASET_REVISION = "v1.1"
MAX_TURNS = 10
MAX_TOOL_OUTPUT = 2000  # characters of each tool result the model sees
MAX_FILE_MB = 10  # tasks with a bigger data file are skipped
PER_DATASET = 6  # training tasks per Kaggle dataset, so a few big datasets don't dominate
LAST_TURN_REMINDER = "You have one turn left. Reply with your final answer now."
TASKS_ROOT = "/tmp/data-analysis-tasks"  # where the server downloads the task files, as the prompts say
SERVER_RUNTIME = "/opt/data-analysis-env"
SERVER_REQUIREMENTS = [
    "openenv>=0.15.0",
    "openenv-data-analysis-env @ git+https://github.com/openenv/OpenEnv.git@v0.15.0#subdirectory=envs/data_analysis_env",
]
SECRET_NAME = re.compile(r"TOKEN|KEY|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTH|COOKIE|SESSION", re.IGNORECASE)
SECRET_VALUE = re.compile(
    rb"hf_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9_-]{20,}|ghp_[A-Za-z0-9]{30,}|AKIA[A-Z0-9]{16}|-----BEGIN [A-Z ]*PRIVATE KEY"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="On-policy distillation of a data-analysis agent with OpenEnv.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-2B", help="Student model.")
    parser.add_argument(
        "--teacher", type=str, default="Qwen/Qwen3.5-9B", help="Teacher model, with the student's tokenizer."
    )
    parser.add_argument(
        "--first-turn", action="store_true", help="Distill the first turn only, with no tool run (the control)."
    )
    parser.add_argument("--max-steps", type=int, default=28, help="Number of training steps.")
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed. The tasks and their order stay the same.")
    parser.add_argument("--output-dir", type=str, default=None, help="Where to store outputs.")
    parser.add_argument("--report-to", type=str, default="trackio", help="Where to log metrics, e.g. `none`.")
    parser.add_argument("--push-to-hub", action="store_true", help="Push the trained model to the Hub.")
    return parser.parse_args()


def start_env_server(port: int = 8001, max_sessions: int = 96) -> str:
    # The server runs model-written code. It runs as root with an environment built from constants (no token), so
    # that each session's code runs as its own unprivileged user and can't read this process's secrets.
    if not sys.platform.startswith("linux") or os.geteuid() != 0:
        raise SystemExit("The environment server must run as root on Linux, so that sessions get their own users.")
    # The session users import modules from the server's Python, so it lives outside /root, readable by them
    python = f"{SERVER_RUNTIME}/venv/bin/python"
    if not os.path.exists(python):
        env = {k: v for k, v in os.environ.items() if not SECRET_NAME.search(k)}
        env.update(
            UV_PYTHON_INSTALL_DIR=f"{SERVER_RUNTIME}/python", UV_PYTHON_PREFERENCE="only-managed", UV_LINK_MODE="copy"
        )
        version = f"{sys.version_info.major}.{sys.version_info.minor}"
        subprocess.run(["uv", "venv", "--quiet", "--python", version, f"{SERVER_RUNTIME}/venv"], env=env, check=True)
        subprocess.run(
            ["uv", "pip", "install", "--quiet", "--python", python, *SERVER_REQUIREMENTS], env=env, check=True
        )
        subprocess.run(["chmod", "-R", "a+rX", SERVER_RUNTIME], check=True)
    server_env = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HOME": "/nonexistent",
        "LANG": "C.UTF-8",
        "PYTHONUNBUFFERED": "1",
        "DATA_ANALYSIS_TASKS_ROOT": TASKS_ROOT,
        "DATA_ANALYSIS_MAX_FILE_MB": str(MAX_FILE_MB),
        "MAX_CONCURRENT_ENVS": str(max_sessions),
    }
    log = open("data_analysis_env_server.log", "w")
    cmd = [python, "-m", "uvicorn", "data_analysis_env.server.app:app", "--host", "127.0.0.1", "--port", str(port)]
    subprocess.Popen(cmd, env=server_env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, cwd="/")
    url = f"http://127.0.0.1:{port}"
    for _ in range(300):
        try:
            urllib.request.urlopen(f"{url}/health", timeout=2)
            break
        except OSError:
            time.sleep(1)
    else:
        raise RuntimeError("The data-analysis environment server did not start.")

    # Check from a session that model code runs as another user and sees no secret-looking variable
    with open_session(url) as client:
        client.reset(split="eval", seed=0)
        output = client.call_tool("run_python", code="import os\nprint(os.getuid(), sorted(os.environ))")
    uid, names = output.split(" ", 1)
    if int(uid) == os.getuid() or SECRET_NAME.search(names):
        raise SystemExit(f"The sandbox is not isolated: sessions run as uid {uid} and see {names}")
    sandbox = [line for line in Path("data_analysis_env_server.log").read_text().splitlines() if "sandbox:" in line]
    print(f"Sessions run as uid {uid} and see {names.strip()}. {sandbox[-1] if sandbox else ''}", flush=True)
    return url


def open_session(url: str):
    # No client-side keepalive pings: generation can starve the client's event loop, and healthy connections close
    return DataAnalysisEnv(base_url=url, message_timeout_s=300, websocket_ping_interval_s=None).sync()


def task_prompt(split: str, row: dict) -> str:
    # The environment's own prompt for the task: the question and the paths where the server puts its files
    paths = [os.path.join(TASKS_ROOT, split, safe_task_id(row["task_id"]), name) for name in row["files"]]
    return build_prompt(row, paths)


def load_train_tasks() -> list[dict]:
    """
    SmolDataEnvs `train`, without the Kaggle datasets of the `eval` split, with only CSV, TSV and JSON files under
    `MAX_FILE_MB`, at most `PER_DATASET` tasks per Kaggle dataset (round-robin over hard, medium and easy), shuffled
    once with a fixed seed.
    """
    train = load_dataset(DATASET_ID, split="train", revision=DATASET_REVISION).to_list()
    evals = load_dataset(DATASET_ID, split="eval", revision=DATASET_REVISION).to_list()
    eval_datasets = {row["kaggle_dataset"] for row in evals}
    eval_prefixes = {row["bucket_prefix"] for row in evals}
    rows = [r for r in train if r["kaggle_dataset"] not in eval_datasets and r["bucket_prefix"] not in eval_prefixes]
    rows = sorted([r for r in rows if runnable(r)], key=lambda r: r["task_id"])
    buckets = sorted({(r["hf_bucket"], r["bucket_prefix"]) for r in rows})
    with ThreadPoolExecutor(8) as pool:
        listings = dict(zip(buckets, pool.map(lambda b: list_bucket_prefix(*b), buckets), strict=True))
    by_dataset = defaultdict(lambda: defaultdict(list))
    for row in rows:
        listing = listings[(row["hf_bucket"], row["bucket_prefix"])]
        if all(name in listing and listing[name] <= MAX_FILE_MB * 2**20 for name in row["files"]):
            by_dataset[row["kaggle_dataset"]][row["difficulty_tier"]].append(row)
    tasks = []
    for dataset in sorted(by_dataset):
        pools, picked = by_dataset[dataset], []
        while len(picked) < PER_DATASET and any(pools.values()):
            for tier in ("hard", "medium", "easy"):
                if pools[tier] and len(picked) < PER_DATASET:
                    picked.append(pools[tier].pop(0))
        tasks.extend(picked)
    random.Random(0).shuffle(tasks)
    return tasks


def start_tasks(url: str, rows: list[dict]) -> list[dict]:
    # Starting a task makes the server download its files. Keep the tasks that start with the expected prompt.
    def starts(row: dict) -> bool:
        try:
            with open_session(url) as client:
                metadata = client.reset(split="train", task_id=row["task_id"]).observation.metadata
            return metadata["prompt"] == task_prompt("train", row)
        except Exception:
            return False

    with ThreadPoolExecutor(8) as pool:
        return [row for row, ok in zip(rows, pool.map(starts, rows), strict=True) if ok]


def main() -> None:
    args = parse_args()

    env_url = start_env_server()

    class DataAnalysisSession:
        def __init__(self):
            self.client = None

        def reset(self, task_id: str, **kwargs) -> None:
            if self.client is not None:
                self.client.close()
            self.calls, self.reward, self.answered = 0, 0.0, False
            try:
                self.client = open_session(env_url)
                self.client.reset(split="train", task_id=task_id)
            except Exception:  # a task that fails to start gives tool errors and a reward of 0
                self.client = None

        def run_python(self, code: str):
            """
            Run Python code in the live session. Variables persist across calls. pandas and numpy are available.

            Args:
                code: The Python code to execute.
            """
            self.calls += 1
            if self.answered:
                output = "(tool error: an answer was already submitted, the task is over)"
            elif self.client is None:
                output = "(tool error: the session is gone)"
            else:
                try:
                    output = str(self.client.call_tool("run_python", code=str(code)))[:MAX_TOOL_OUTPUT]
                except Exception as e:  # a lost connection loses the session, the episode scores 0
                    self.client, output = None, f"(tool error: {type(e).__name__})"
            # Tell the model when its next turn is the last one
            if self.calls == MAX_TURNS - 1:
                output += f"\n\n{LAST_TURN_REMINDER}"
            return output

        def submit_answer(self, answer: str):
            """
            Submit the final answer to the question and end the task. Call it once, when you are sure.

            Args:
                answer: A single clean value.
            """
            self.calls += 1
            if self.answered:
                return "(tool error: an answer was already submitted)"
            self.answered = True
            if self.client is not None:
                try:
                    action = CallToolAction(tool_name="submit_answer", arguments={"answer": str(answer)})
                    self.reward = float(self.client.step(action).reward or 0.0)
                except Exception:
                    pass
            return "Answer received."

        def get_reward(self) -> float:
            # Logged only, not used in the loss: 1.0 for a correct answer sent with `submit_answer`, else 0.0. With
            # --first-turn no tool runs, so it stays at 0.
            if self.client is not None:
                self.client.close()
                self.client = None
            return self.reward

    # 28 steps of 32 tasks go about once over the 885 training tasks
    tasks = load_train_tasks()
    wanted = args.max_steps * 32
    tasks = start_tasks(env_url, tasks[:wanted])
    print(f"{len(tasks)} training tasks ready", flush=True)
    rows = [tasks[i % len(tasks)] for i in range(wanted)]
    dataset = Dataset.from_list(
        [
            {"prompt": [{"role": "user", "content": task_prompt("train", row)}], "task_id": row["task_id"]}
            for row in rows
        ]
    )
    max_completion_length = 2048 if args.first_turn else 8192
    output_dir = (
        args.output_dir
        or f"{args.model.split('/')[-1]}-Distillation-Data-{'First-Turn' if args.first_turn else 'Episodes'}"
    )

    trainer = DistillationTrainer(
        model=args.model,
        teacher_model=args.teacher,
        train_dataset=dataset,
        args=DistillationConfig(
            output_dir=output_dir,
            model_init_kwargs={"dtype": "bfloat16"},
            teacher_model_init_kwargs={"dtype": "bfloat16"},
            use_vllm=True,
            vllm_mode="colocate",
            vllm_gpu_memory_utilization=0.25,
            vllm_max_model_length=max_completion_length + 2048,
            chat_template_kwargs={"enable_thinking": False},
            # Whole episodes: 9 rounds of tool calls, then a last model turn. First turn: no tool runs.
            max_tool_calling_iterations=0 if args.first_turn else MAX_TURNS - 1,
            max_completion_length=max_completion_length,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=8,
            learning_rate=args.learning_rate,
            max_steps=args.max_steps,
            seed=args.seed,
            shuffle_dataset=False,  # the tasks are already shuffled, every run sees them in the same order
            logging_steps=1,
            report_to=args.report_to,
        ),
        environment_factory=DataAnalysisSession,
    )
    trainer.train()
    trainer.save_model(output_dir)
    # vLLM reads `rope_theta` at the top level of the config
    config_path = os.path.join(output_dir, "config.json")
    config = json.load(open(config_path))
    rope_parameters = config.get("rope_parameters") or config.get("text_config", {}).get("rope_parameters", {})
    if "rope_theta" in rope_parameters:
        config["rope_theta"] = rope_parameters["rope_theta"]
        json.dump(config, open(config_path, "w"), indent=2)

    if args.push_to_hub:
        # Everything saved here passed through a process that ran next to model code: scan it before it leaves
        token = os.environ.get("HF_TOKEN", "").encode()
        for path in Path(output_dir).rglob("*"):
            data = path.read_bytes() if path.is_file() else b""
            if SECRET_VALUE.search(data) or (token and token in data):
                raise RuntimeError(f"Possible secret in {path}, not pushing.")
        api = HfApi()
        repo_id = api.create_repo(Path(output_dir).name, exist_ok=True).repo_id
        api.upload_folder(folder_path=output_dir, repo_id=repo_id, ignore_patterns=["checkpoint-*"])
        print(f"Pushed to https://huggingface.co/{repo_id}", flush=True)


if __name__ == "__main__":
    main()
