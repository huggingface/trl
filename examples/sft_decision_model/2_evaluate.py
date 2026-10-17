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
#     "datasets>=5.0.0",  # `Json` feature
# ]
# ///

# docstyle-ignore
"""
Measures a decision model through the `/v1/systemone` endpoint of a running `transformers serve`, on two held-out sets
of System One requests (see `decisions.load_eval`): accuracy (the most probable option is the most probable in the
gold) and Brier score (squared distance between the probabilities of the options and the gold ones, 0 is perfect, 2
the worst).

    transformers serve
    python examples/sft_decision_model/2_evaluate.py --model Qwen/Qwen3.5-2B --output base.jsonl
    python examples/sft_decision_model/2_evaluate.py --model Qwen3.5-2B-Decision --output trained.jsonl
"""

import argparse
import json
import urllib.request
from collections import defaultdict

from decisions import load_eval


parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--url", default="http://localhost:8000/v1/systemone")
parser.add_argument("--output", help="JSON lines file with the probabilities and the gold of every question.")
args = parser.parse_args()

records = []
print("| | accuracy | Brier |\n| --- | --- | --- |")
for name in ["typed-decisions", "jevbench"]:
    correct, brier, total = defaultdict(int), defaultdict(float), defaultdict(int)
    for request in load_eval(name):
        body = {"model": args.model, "state": request["state"], "questions": request["questions"]}
        http_request = urllib.request.Request(
            args.url, json.dumps(body).encode(), {"Content-Type": "application/json"}
        )
        answers = json.load(urllib.request.urlopen(http_request))["answers"]
        for key, gold in request["gold"].items():
            answer = answers[key]
            if answer["type"] == "noul":
                probabilities = {"true": answer["noul"], "false": 1 - answer["noul"]}
            else:
                probabilities = answer["probabilities"]
            records.append({"set": name, "question": key, "probabilities": probabilities, "gold": gold})
            total[answer["type"]] += 1
            correct[answer["type"]] += max(probabilities, key=probabilities.get) == max(gold, key=gold.get)
            brier[answer["type"]] += sum((probabilities[option] - p) ** 2 for option, p in gold.items())
    for kind in ["noul", "choice", "score"]:
        print(f"| {name} {kind} | {correct[kind] / total[kind]:.1%} | {brier[kind] / total[kind]:.3f} |")
    n = sum(total.values())
    print(f"| **{name}** | **{sum(correct.values()) / n:.1%}** | **{sum(brier.values()) / n:.3f}** |")

if args.output:
    with open(args.output, "w") as f:
        f.writelines(json.dumps(record) + "\n" for record in records)
