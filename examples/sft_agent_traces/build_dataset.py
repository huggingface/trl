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
#     "datasets>=5",
#     "huggingface_hub",
# ]
# ///

"""Turn the rollouts of `rollouts.py` into an SFT dataset: one conversational row per solved task.

A row is the agent's last request to the model (`request_messages`, which hold the whole conversation so far: system
prompt, task, every earlier assistant turn and tool output) plus the reply to it, with the tools the harness offered.
Calls the harness makes on the side (session titles, summaries) and unsolved rollouts are dropped. Tool-call arguments
are parsed into dicts, since chat templates such as Qwen3.5's iterate over them.

```sh
uv run examples/sft_agent_traces/build_dataset.py \
    --rollouts <you>/smoldataenvs-rollouts --runs teacher-397b \
    --push-to <you>/smoldataenvs-agent-traces
```

`--max-rows` caps the dataset, to train on a subset.
"""

from __future__ import annotations

import argparse
import json
import pathlib

from datasets import Dataset, Features, Json, List, Value
from huggingface_hub import snapshot_download


def task_reward(result: dict) -> float | None:
    """The task's reward, whatever the grader called it (same rule as in `rollouts.py`)."""
    rewards = result.get("rewards") or {}
    for value in (rewards.get("correctness"), result.get("reward"), rewards.get("reward")):
        if value is not None:
            return value
    return None


def last_agent_turn(result: dict) -> dict | None:
    turns = [t for t in result.get("turns", []) if t.get("role") == "agent" and not t.get("discarded")]
    return turns[-1] if turns else None


def cut_short(result: dict) -> bool:
    """The agent stopped right after a tool call, with no final answer: usually a failed LLM call, worth a retry."""
    turn = last_agent_turn(result)
    return turn is not None and turn.get("finish_reason") == "tool_calls"


def normalize_tool_calls(calls: list[dict]) -> list[dict] | None:
    """Tool calls with their arguments as a dict, or None if one call's arguments aren't a JSON object."""
    out = []
    for call in calls:
        fn = call.get("function", call)  # turns carry flat {name, arguments}, request messages nest them
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except json.JSONDecodeError:
                return None
        if not isinstance(args, dict):  # the chat template iterates over the arguments as a dict
            return None
        out.append({"type": "function", "function": {"name": fn["name"], "arguments": args}})
    return out


def to_row(result: dict) -> dict | None:
    if task_reward(result) != 1.0:
        return None
    turn = last_agent_turn(result)
    if turn is None or not turn.get("request_messages"):
        return None
    reply = {"role": "assistant", "content": turn.get("text") or ""}
    if turn.get("tool_calls"):
        reply["tool_calls"] = turn["tool_calls"]
    messages = [*turn["request_messages"], reply]
    for msg in messages:
        if msg["role"] == "assistant" and msg.get("tool_calls"):
            msg["tool_calls"] = normalize_tool_calls(msg["tool_calls"])
            if msg["tool_calls"] is None:  # a malformed call would teach the student a malformed call
                print(f"{result.get('task_name')}: a tool call's arguments aren't a JSON object, skipped")
                return None
    return {
        "messages": messages,
        "tools": turn.get("request_tools") or [],
        "task_name": result.get("task_name"),
        "n_turns": result.get("n_turns"),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--rollouts", required=True, help="Hub dataset `rollouts.py` pushed to, or a local folder")
    p.add_argument("--runs", nargs="+", required=True, help="run folders to read")
    p.add_argument("--max-rows", type=int, default=None, help="keep at most this many rows")
    p.add_argument("--push-to", required=True, help="Hub dataset to push the SFT rows to")
    p.add_argument("--private", action="store_true")
    args = p.parse_args()

    root = pathlib.Path(args.rollouts)
    if not root.is_dir():
        root = pathlib.Path(
            snapshot_download(
                args.rollouts, repo_type="dataset", allow_patterns=[f"{r}/result_*.json" for r in args.runs]
            )
        )
    results = []  # (task index, result): the index is the number in the file name, as `rollouts.py` wrote it
    for run in args.runs:
        for path in sorted((root / run).glob("result_*.json")):
            data = json.loads(path.read_text())
            index = int(path.stem.split("_")[-1])
            results.extend((index, r) for r in (data if isinstance(data, list) else [data]))

    rows, seen = [], set()
    for index, result in results:
        row = to_row(result)
        key = row and (row["task_name"] or f"task {index}")
        if row and key not in seen:  # a task retried in a later run counts once
            seen.add(key)
            rows.append(row)
    rows = rows[: args.max_rows]
    print(f"{len(rows)} rows from {len(results)} rollouts")
    retry = sorted({i for i, r in results if cut_short(r) and r.get("task_name") not in seen})
    if retry:
        print(
            f"{len(retry)} tasks ended on a tool call and were never solved. Rerun them with the same --dataset and "
            f"--exclude-datasets-of, and --task-indices {','.join(map(str, retry))}"
        )

    # `Json` keeps each message as it is. Inferred from the rows, every tool call would get one struct with the argument
    # names of every tool, filled with None where a call doesn't use them, and the chat template would render those.
    features = Features(
        {"messages": List(Json()), "tools": List(Json()), "task_name": Value("string"), "n_turns": Value("int64")}
    )
    Dataset.from_list(rows, features=features).push_to_hub(args.push_to, private=args.private)
    print(f"pushed to https://huggingface.co/datasets/{args.push_to}")


if __name__ == "__main__":
    main()
