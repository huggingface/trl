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
#     "openenv>=0.13.0",
#     "openenv-coding_env @ git+https://github.com/openenv/OpenEnv.git@v0.13.0#subdirectory=envs/coding_env",
# ]
# ///

"""
Evaluate a model as a coding agent on the 500 problems of the MBPP test split, with the prompt, tool, turn limit and
episode budget of `distillation_coding_env.py`, whose server and prompt are copied here. Reports the fraction of the MBPP tests passed and the fraction of
problems fully solved.

```sh
python examples/distillation_coding_env/evaluate.py --model Qwen/Qwen3-1.7B
```

On Hugging Face Jobs:

```sh
hf jobs uv run --flavor a100-large --timeout 90m examples/distillation_coding_env/evaluate.py --model Qwen/Qwen3-1.7B
```
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from coding_env import CodeAction, CodingEnv
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams, TokensPrompt


# One Python session per connection, up to 64 at once
SERVER = """
import sys

import uvicorn
from coding_env.models import CodeAction, CodeObservation
from coding_env.server.python_codeact_env import PythonCodeActEnv
from openenv.core.env_server import create_app


class PythonSession(PythonCodeActEnv):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        super().__init__(additional_imports=["bisect", "cmath", "heapq", "operator"])


app = create_app(PythonSession, CodeAction, CodeObservation, env_name="coding_env", max_concurrent_envs=64)
uvicorn.run(app, host="127.0.0.1", port=int(sys.argv[1]), log_level="warning")
"""


def start_env_server(port: int = 8001) -> str:
    # The server runs model-written code, so it gets no tokens or keys
    env = {k: v for k, v in os.environ.items() if not re.search("TOKEN|KEY|SECRET|PASSWORD", k, re.IGNORECASE)}
    subprocess.Popen([sys.executable, "-c", SERVER, str(port)], env=env)
    url = f"http://127.0.0.1:{port}"
    for _ in range(120):
        try:
            urllib.request.urlopen(f"{url}/health", timeout=2)
            return url
        except OSError:
            time.sleep(1)
    raise RuntimeError("The coding environment server did not start.")


PROMPT = """Write a Python function for the task below. Use the `run_python` tool to define the function and to test \
it: the session keeps what you define between calls. When the function works, reply with a one-line summary.

Task: {text}
Your function must pass this test:
{test}"""


MAX_COMPLETION_LENGTH = 4096
MAX_TOOL_CALLING_ITERATIONS = 6
NUM_SESSIONS = 64


def run_python(code: str) -> str:
    """
    Run Python code in the session and return its output.

    Args:
        code: The Python code to run.

    Returns:
        The standard output and standard error of the code.
    """


def parse_tool_calls(text: str) -> list[dict] | None:
    calls = []
    for block in re.findall(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL):
        try:
            call = json.loads(block)
            calls.append({"name": call["name"], "arguments": call["arguments"]})
        except (json.JSONDecodeError, KeyError, TypeError):
            return None
    return calls or None


def call_tools(client: CodingEnv, calls: list[dict]) -> list[str]:
    results = []
    for call in calls:
        try:
            if call["name"] != "run_python":
                raise ValueError(f"Tool {call['name']} not found.")
            observation = client.step(CodeAction(code=call["arguments"]["code"])).observation
            results.append(
                (observation.stdout + observation.stderr) or f"(no output, exit code {observation.exit_code})"
            )
        except Exception as e:
            results.append(str({"error": str(e)}))
    return results


def score(client: CodingEnv, example: dict) -> float | None:
    try:
        client.step(CodeAction(code=example["test_setup_code"]))
        passed = [client.step(CodeAction(code=test)).observation.exit_code == 0 for test in example["test_list"]]
    except Exception:  # the session was lost during the episode
        return None
    return sum(passed) / len(passed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a coding agent on the MBPP test split.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=2, help="Episodes per problem, each with its own seed.")
    parser.add_argument(
        "--env-url", type=str, default=None, help="Coding environment server. Default: start one here."
    )
    args = parser.parse_args()

    port, restart = 8000, True
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    template_kwargs = {"tools": [run_python], "enable_thinking": False}
    dummy = [
        {"role": "user", "content": "dummy"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"type": "function", "function": {"name": "run_python", "arguments": {}}}],
        },
    ]
    dummy_ids = tokenizer.apply_chat_template(dummy, tokenize=True, return_dict=False, **template_kwargs)

    def tool_suffix_ids(results: list[str]) -> list[int]:
        messages = dummy + [{"role": "tool", "name": "run_python", "content": r} for r in results]
        ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=False, **template_kwargs
        )
        return ids[len(dummy_ids) :]

    llm = LLM(model=args.model, max_model_len=8192, gpu_memory_utilization=0.85)
    dataset = load_dataset("google-research-datasets/mbpp", "full", split="test")
    episodes = [(seed, example) for seed in range(args.num_samples) for example in dataset]
    results = []

    for start in range(0, len(episodes), NUM_SESSIONS):
        chunk = episodes[start : start + NUM_SESSIONS]
        # A lost session stays open on the server, so start a fresh server after one is lost
        if restart:
            port += 1
            env_url = args.env_url or start_env_server(port)
            clients = [
                CodingEnv(base_url=env_url, message_timeout_s=300, websocket_ping_interval_s=None)
                for _ in range(NUM_SESSIONS)
            ]
        with ThreadPoolExecutor(len(chunk)) as pool:
            list(pool.map(lambda c: c.reset(), clients[: len(chunk)]))
        prompt_ids, completion_ids, num_calls = [], [], [0] * len(chunk)
        for _, example in chunk:
            prompt = [{"role": "user", "content": PROMPT.format(text=example["text"], test=example["test_list"][0])}]
            prompt_ids.append(
                tokenizer.apply_chat_template(
                    prompt, add_generation_prompt=True, tokenize=True, return_dict=False, **template_kwargs
                )
            )
            completion_ids.append([])
        active = list(range(len(chunk)))
        for iteration in range(MAX_TOOL_CALLING_ITERATIONS + 1):
            outputs = llm.generate(
                [TokensPrompt(prompt_token_ids=prompt_ids[i] + completion_ids[i]) for i in active],
                [
                    SamplingParams(
                        temperature=0.7,
                        top_p=0.8,
                        top_k=20,
                        max_tokens=MAX_COMPLETION_LENGTH - len(completion_ids[i]),
                        seed=start + i,
                    )
                    for i in active
                ],
                use_tqdm=False,
            )
            calls = {}
            for i, output in zip(active, outputs, strict=True):
                completion_ids[i] += list(output.outputs[0].token_ids)
                if iteration < MAX_TOOL_CALLING_ITERATIONS:
                    tool_calls = parse_tool_calls(output.outputs[0].text)
                    if tool_calls:
                        calls[i] = tool_calls
            if not calls:
                break
            with ThreadPoolExecutor(len(calls)) as pool:
                tool_results = dict(
                    zip(calls, pool.map(call_tools, [clients[i] for i in calls], calls.values()), strict=True)
                )
            active = []
            for i, tool_result in tool_results.items():
                num_calls[i] += len(tool_result)
                suffix = tool_suffix_ids(tool_result)
                # Like the trainer, drop tool results that would push the episode past its budget, and stop there
                if len(completion_ids[i]) + len(suffix) < MAX_COMPLETION_LENGTH:
                    completion_ids[i] += suffix
                    active.append(i)
            if not active:
                break
        with ThreadPoolExecutor(len(chunk)) as pool:
            scores = list(pool.map(score, clients, [example for _, example in chunk]))
        restart = args.env_url is None and None in scores
        for i, (seed, example) in enumerate(chunk):
            results.append(
                {
                    "seed": seed,
                    "task_id": example["task_id"],
                    "score": scores[i] or 0.0,  # a lost session counts as failed
                    "lost": scores[i] is None,
                    "tool_calls": num_calls[i],
                }
            )
        lost = [example["task_id"] for (_, example), r in zip(chunk, scores, strict=True) if r is None]
        print(f"{len(results)}/{len(episodes)} episodes, lost sessions: {lost}", flush=True)

    summary = {"model": args.model, "samples": []}
    for seed in range(args.num_samples):
        rows = [r for r in results if r["seed"] == seed]
        summary["samples"].append(
            {
                "seed": seed,
                "tests_passed": sum(r["score"] for r in rows) / len(rows),
                "solved": sum(r["score"] == 1.0 for r in rows) / len(rows),
                "tool_calls": sum(r["tool_calls"] for r in rows) / len(rows),
                "lost_sessions": sum(r["lost"] for r in rows),
            }
        )
    print("RESULTS " + json.dumps(summary))
    with open("results.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in results)


if __name__ == "__main__":
    main()
