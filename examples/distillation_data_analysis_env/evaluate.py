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
#     "openenv>=0.15.0",
#     "openenv-data-analysis-env @ git+https://github.com/openenv/OpenEnv.git@v0.15.0#subdirectory=envs/data_analysis_env",
# ]
# ///

"""
Evaluate data-analysis agents on SmolDataEnvs tasks, with the prompt, tools, turn limit and tool-result cap of
`distillation_data_analysis_env.py`, whose environment server is copied here. The base model and the given models
run the same tasks in the same job, each `--num-samples` times, and are compared task by task: mean reward (the
share of correct answers) with a 95% bootstrap interval over tasks, and the paired difference between every two
models.

```sh
hf jobs uv run --flavor a100-large --timeout 2h examples/distillation_data_analysis_env/evaluate.py \
    --models distilled=<your-username>/Qwen3.5-2B-Distillation-Data-Episodes
```
"""

import argparse
import json
import os
import random
import re
import statistics
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from pathlib import Path

from data_analysis_env import CallToolAction, DataAnalysisEnv, extract_answer
from data_analysis_env.server.tasks import DATASET_ID, build_prompt, list_bucket_prefix, runnable, safe_task_id
from datasets import load_dataset
from openai import OpenAI


DATASET_REVISION = "v1.1"
MAX_TOOL_OUTPUT = 2000  # characters of each tool result the model sees
MAX_FILE_MB = 10  # tasks with a bigger data file are skipped
LAST_TURN_REMINDER = "You have one turn left. Reply with your final answer now."
TASKS_ROOT = "/tmp/data-analysis-tasks"  # where the server downloads the task files, as the prompts say
SERVER_RUNTIME = "/opt/data-analysis-env"
SERVER_REQUIREMENTS = [
    "openenv>=0.15.0",
    "openenv-data-analysis-env @ git+https://github.com/openenv/OpenEnv.git@v0.15.0#subdirectory=envs/data_analysis_env",
]
SECRET_NAME = re.compile(r"TOKEN|KEY|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTH|COOKIE|SESSION", re.IGNORECASE)

# The tools of `DataAnalysisSession` in `distillation_data_analysis_env.py`
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": "Run Python code in the live session. Variables persist across calls. pandas and numpy are available.",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string", "description": "The Python code to execute."}},
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_answer",
            "description": "Submit the final answer to the question and end the task. Call it once, when you are sure.",
            "parameters": {
                "type": "object",
                "properties": {"answer": {"type": "string", "description": "A single clean value."}},
                "required": ["answer"],
            },
        },
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate data-analysis agents on SmolDataEnvs tasks.")
    parser.add_argument(
        "--base-model", type=str, default="Qwen/Qwen3.5-2B", help="Evaluated first. Also the tokenizer."
    )
    parser.add_argument("--models", nargs="*", default=[], help="Models to compare with the base, as NAME=ID or ID.")
    parser.add_argument(
        "--split",
        choices=["eval_verified", "test"],
        default="eval_verified",
        help="`eval_verified` (61 checked tasks) or `test` (194 harder tasks with files the environment serves).",
    )
    parser.add_argument("--max-turns", type=int, default=10, help="Model turns per run.")
    parser.add_argument("--num-samples", type=int, default=8, help="Runs per task.")
    parser.add_argument("--num-tasks", type=int, default=None, help="Only the first tasks, for a quick check.")
    parser.add_argument("--concurrency", type=int, default=64, help="Runs at once.")
    return parser.parse_args()


def start_env_server(port: int = 8001, max_sessions: int = 128) -> str:
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


def load_tasks(split: str) -> list[dict]:
    # The tasks whose files are all CSV, TSV or JSON under MAX_FILE_MB, sorted by id. `eval_verified` tasks are served
    # as tasks of `eval`, which holds the same rows.
    rows = sorted(load_dataset(DATASET_ID, split=split, revision=DATASET_REVISION), key=lambda r: r["task_id"])
    tasks = []
    for row in rows:
        listing = list_bucket_prefix(row["hf_bucket"], row["bucket_prefix"]) if runnable(row) else {}
        if row["files"] and all(name in listing and listing[name] <= MAX_FILE_MB * 2**20 for name in row["files"]):
            server_split = "eval" if split == "eval_verified" else split
            tasks.append({"task_id": row["task_id"], "split": server_split, "prompt": task_prompt(server_split, row)})
    return tasks


def run_episode(client: OpenAI, model: str, task: dict, env_url: str, max_turns: int) -> float:
    """
    One run of a task, graded by the environment. The run ends when the model calls `submit_answer`, replies without
    a tool call (the answer is then read from that reply), or runs out of turns.
    """
    with open_session(env_url) as env:
        env.reset(split=task["split"], task_id=task["task_id"])
        messages = [{"role": "user", "content": task["prompt"]}]
        answer = None
        for turn in range(1, max_turns + 1):
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOLS,
                temperature=1.0,
                max_tokens=2048,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            message = response.choices[0].message
            messages.append(message.model_dump(exclude_none=True))
            if not message.tool_calls:
                answer = extract_answer(message.content or "")
                break
            for call in message.tool_calls:
                try:
                    arguments = json.loads(call.function.arguments)
                    if call.function.name == "run_python":
                        output = str(env.call_tool("run_python", code=str(arguments["code"])))
                    elif call.function.name == "submit_answer":
                        answer = str(arguments["answer"]) if answer is None else answer
                        output = "Answer received."
                    else:
                        output = f"(tool error: unknown tool {call.function.name!r})"
                except Exception as e:
                    output = f"(tool error: {type(e).__name__})"
                messages.append({"role": "tool", "tool_call_id": call.id, "content": output[:MAX_TOOL_OUTPUT]})
            if answer is not None:
                break
            # Tell the model when its next turn is the last one
            if turn == max_turns - 1:
                messages[-1]["content"] += f"\n\n{LAST_TURN_REMINDER}"
        if answer is None:
            return 0.0
        action = CallToolAction(tool_name="submit_answer", arguments={"answer": answer})
        return float(env.step(action).reward or 0.0)


def evaluate(model: str, tokenizer: str, tasks: list[dict], env_url: str, args: argparse.Namespace) -> list[float]:
    """The mean reward of each task over `num_samples` runs."""
    cmd = ["vllm", "serve", model, "--port", "8000", "--tokenizer", tokenizer, "--max-model-len", "32768"]
    cmd += ["--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder"]
    env = {**os.environ, "VLLM_USE_FLASHINFER_SAMPLER": "0", "VLLM_USE_DEEP_GEMM": "0"}
    server = subprocess.Popen(cmd, env=env, stdout=open("vllm_server.log", "w"), stderr=subprocess.STDOUT)
    client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="none")
    for _ in range(120):
        if server.poll() is not None:
            raise RuntimeError(f"vLLM failed to serve {model}, see vllm_server.log")
        try:
            client.models.list()
            break
        except Exception:
            time.sleep(10)
    else:
        raise RuntimeError(f"vLLM did not serve {model} in 20 minutes, see vllm_server.log")

    def run(task: dict) -> float:
        for _ in range(3):  # a dropped connection to the environment loses the session, start again
            try:
                return run_episode(client, model, task, env_url, args.max_turns)
            except Exception as e:
                print(f"Run of {task['task_id']} failed: {type(e).__name__}", flush=True)
        return 0.0

    try:
        with ThreadPoolExecutor(args.concurrency) as pool:
            rewards = list(pool.map(run, tasks * args.num_samples))
    finally:
        server.terminate()
        server.wait()
        time.sleep(10)  # let the GPU memory go before the next model
    return [statistics.mean(rewards[i :: len(tasks)]) for i in range(len(tasks))]


def bootstrap(values: list[float], rng: random.Random, draws: int = 5000) -> tuple[float, float]:
    # 95% interval of the mean, resampling tasks
    means = sorted(statistics.fmean(rng.choices(values, k=len(values))) for _ in range(draws))
    return means[int(0.025 * draws)], means[int(0.975 * draws) - 1]


def main() -> None:
    args = parse_args()
    models = {"base": args.base_model}
    for model in args.models:
        name, _, path = model.rpartition("=")
        models[name or path] = path

    env_url = start_env_server()
    tasks = load_tasks(args.split)[: args.num_tasks]
    print(f"{len(tasks)} tasks of {args.split}, {args.num_samples} runs each, {args.max_turns} turns", flush=True)
    scores = {name: evaluate(path, args.base_model, tasks, env_url, args) for name, path in models.items()}

    rng = random.Random(0)
    for name, per_task in scores.items():
        low, high = bootstrap(per_task, rng)
        print(f"[result] {name}: {statistics.mean(per_task):.3f} [{low:.3f}, {high:.3f}]", flush=True)
    for a, b in combinations(scores, 2):
        diffs = [y - x for x, y in zip(scores[a], scores[b], strict=True)]
        low, high = bootstrap(diffs, rng)
        print(f"[compare] {b} - {a}: {statistics.mean(diffs):+.3f} [{low:+.3f}, {high:+.3f}]", flush=True)


if __name__ == "__main__":
    main()
