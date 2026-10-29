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

"""
System One questions (`{"type", "instructions", "criteria"}`, the format of the `/v1/systemone` endpoint) for training
and evaluation. Option order follows the endpoint: `true`, `false` for `noul`, the criteria keys for `choice` and the
levels for `score`.
"""

import json
from collections import defaultdict

from datasets import Dataset, Features, Json, Sequence, Value, concatenate_datasets, load_dataset


def load_train(num_samples: int) -> Dataset:
    """
    `num_samples` questions from [tasksource/tasksource-jev-typed-decisions](https://huggingface.co/datasets/tasksource/tasksource-jev-typed-decisions),
    from its sources licensed for commercial use: half `choice`, a quarter `noul` and a quarter `score`. Each row is a
    `state`, a `question` and a `target` probability per option.
    """
    dataset = load_dataset("tasksource/tasksource-jev-typed-decisions", split="train")
    # The endpoint labels options A to Z, and a choice question's options are the keys of a dict
    dataset = dataset.filter(
        lambda row: row["license_use"] == "commercial"
        and len(row["options"]) <= 26
        and len(set(row["options"])) == len(row["options"])
    )
    parts = []
    for kind, share in [("choice", 0.5), ("noul", 0.25), ("score", 0.25)]:
        part = dataset.filter(lambda row, kind: row["kind"] == kind, fn_kwargs={"kind": kind}).shuffle(seed=0)
        parts.append(part.select(range(min(int(num_samples * share), len(part)))))

    def to_question(row):
        if row["kind"] == "noul":
            question, target = {"type": "noul"}, [row["target"][0], 1 - row["target"][0]]
        elif row["kind"] == "choice":
            question, target = {"type": "choice", "criteria": dict.fromkeys(row["options"])}, row["target"]
        else:
            question, target = {"type": "score", "criteria": row["options"]}, row["target"]
        return {"state": row["state"], "question": {**question, "instructions": row["question"]}, "target": target}

    # The types of questions have different criteria, which Arrow can only store as JSON
    features = Features({"state": Value("string"), "question": Json(), "target": Sequence(Value("float32"))})
    dataset = concatenate_datasets(parts)
    return dataset.map(to_question, remove_columns=dataset.column_names, features=features).shuffle(seed=0)


def load_eval(name: str) -> list[dict]:
    """
    Requests (`state` and `questions` keyed by name) with the probability of each option in `gold`, from the test split
    of [LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) (`"typed-decisions"`,
    400 states with 5 questions, soft gold from 3 annotators) or from
    [JevBench/jevbench](https://huggingface.co/datasets/JevBench/jevbench) (`"jevbench"`, 240 states with 8 to 12
    questions, of which the 1,938 whose gold is one of their options).
    """
    if name == "typed-decisions":
        requests = []
        for row in load_dataset("LocalLLaMA/typed-decisions", "all", split="test"):
            gold = {key: answer["probabilities"] for key, answer in json.loads(row["gold"]).items()}
            requests.append({"state": row["state"], "questions": json.loads(row["questions"]), "gold": gold})
        return requests

    cases = defaultdict(lambda: {"questions": {}, "gold": {}})
    for row in load_dataset("JevBench/jevbench", split="test"):
        if row["type"] == "noul":
            criteria, options = None, ["true", "false"]
        elif row["type"] == "choice":
            criteria = {option["id"]: option["description"] for option in row["options"]}
            options = list(criteria)
        else:
            criteria, options = row["levels"], [str(level) for level in range(len(row["levels"]))]
        # 302 questions have the gold "unknown", which is none of their options
        if row["gold"] not in options:
            continue
        case = cases[row["case_id"]]
        case["state"] = row["state"]
        case["questions"][row["question_key"]] = {
            "type": row["type"],
            "instructions": row["instructions"],
            **({"criteria": criteria} if criteria is not None else {}),
        }
        case["gold"][row["question_key"]] = {option: float(option == row["gold"]) for option in options}
    return list(cases.values())
