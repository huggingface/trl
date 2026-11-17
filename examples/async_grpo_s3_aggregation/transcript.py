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

"""Decode one agent's token context (what the model saw and wrote) from `evaluate.py --save-tokens` output.

python examples/async_grpo_s3_aggregation/transcript.py runs/base.jsonl PROBLEM AGENT_NAME [--model Qwen/Qwen3-0.6B]
"""

import argparse
import json

from transformers import AutoTokenizer


ap = argparse.ArgumentParser()
ap.add_argument("jsonl")
ap.add_argument("problem", type=int)
ap.add_argument("agent")
ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
a = ap.parse_args()
ep = next(r for r in map(json.loads, open(a.jsonl)) if r["problem"] == a.problem)
ag = next(x for x in ep["agents"] if x["name"] == a.agent)
toks, mask = ag["tokens"], ag["mask"]
tok = AutoTokenizer.from_pretrained(a.model)
i = 0
while i < len(toks):  # runs of model (mask 1) and environment (mask 0) tokens, decoded separately
    j = i
    while j < len(toks) and mask[j] == mask[i]:
        j += 1
    print(("MODEL" if mask[i] else "ENV") + f" [{j - i} tok]:\n{tok.decode(toks[i:j])}\n")
    i = j
