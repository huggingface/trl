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

"""Per-episode summary of `evaluate.py` output: success, per-agent tokens, messages, reads, answers, and the paper's
Table 11 failure causes (first matching).

    python examples/async_grpo_s3_aggregation/analyze_episodes.py runs/base.jsonl [--show PROBLEM] [--quiet]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from harness.task import number_words


def sent_own(agent: dict) -> bool:
    words, digits = number_words(agent["number"]), str(agent["number"])
    for e in agent["events"]:
        if e["kind"] == "message" and e.get("ok"):
            t = e["text"].lower()
            if words in t or re.search(rf"(?<!\d){digits}(?!\d)", t):
                return True
    return False


def cause(ep: dict) -> str | None:
    if ep["success"]:
        return None
    ags = ep["agents"]
    if any(not any(e["kind"] == "message" and e.get("ok") for e in a["events"]) for a in ags):
        return "a number withheld: an agent never messaged"
    if not all(sent_own(a) for a in ags):
        return "a number withheld: asked only, or misstated"
    if not any(e["kind"] == "read" for a in ags for e in a["events"]):
        return "sent but never read"
    return "had or could have had all numbers, answer wrong"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", type=Path, nargs="+")
    ap.add_argument("--show", type=int, help="print the agents of this problem id")
    ap.add_argument("--quiet", action="store_true", help="totals only")
    a = ap.parse_args()
    for path in a.jsonl:
        rows = [json.loads(line) for line in path.open()]
        for ep in rows:
            if a.quiet or (a.show is not None and ep["problem"] != a.show):
                continue
            print(
                f"problem {ep['problem']}: success={ep['success']} reward={ep['reward']:.3f} tokens={ep['tokens']} "
                f"target={ep['target']} cause={cause(ep)}"
            )
            for ag in ep["agents"]:
                ev = ag["events"]
                msgs = [e for e in ev if e["kind"] == "message"]
                end = next((e["reason"] for e in ev if e["kind"] == "work_end"), None)
                print(
                    f"   {ag['name']} ({ag['number']}): work={ag['work']} final={ag['final']} msgs={len(msgs)} "
                    f"(all={sum(e['to'].lower() == 'all' for e in msgs)}) reads={sum(e['kind'] == 'read' for e in ev)} "
                    f"sent_own={sent_own(ag)} end={end} answer={ag['answer']}"
                )
        k = len(rows)
        if not k:
            continue
        msgs = sum(e["kind"] == "message" for r in rows for ag in r["agents"] for e in ag["events"])
        print(
            f"\n{path}: {k} episodes, success {sum(r['success'] for r in rows) / k:.1%}, "
            f"mean tokens/team {sum(r['tokens'] for r in rows) / k:.0f}, messages/team {msgs / k:.2f}"
        )
        causes: dict[str, int] = {}
        for r in rows:
            c = cause(r)
            if c:
                causes[c] = causes.get(c, 0) + 1
        fails = sum(causes.values())
        for c, v in sorted(causes.items(), key=lambda x: -x[1]):
            print(f"  {c}: {v / fails:.0%} of failures")


if __name__ == "__main__":
    main()
