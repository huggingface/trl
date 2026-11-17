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

"""The harness end to end with a scripted fake model (characters as tokens), and the board's idle/wake/stall rules."""

from __future__ import annotations

import asyncio
import json
import re

from harness import Board, Problem, Settings, parse_answer, run_team, team_reward
from harness.task import after_thinking


END_ID = 1
READ = r'"from": "agent_\d+", "text": "number (\d+)"'  # a read result, not our own sent text


class FakeSampler:
    """Each agent: message its number to all, read when notified, answer once it holds every number, else idle.
    `silent` agents never message or read. Responses are cut into `chunk`-sized requests like the real server's."""

    def __init__(self, silent: frozenset[str] = frozenset()):
        self.silent = silent
        self.pending: dict[str, str] = {}

    async def chat_prompt(self, messages, tools, enable_thinking):
        return [ord(c) for c in messages[0]["content"] + "\n" + messages[1]["content"] + "\n"]

    async def encode(self, text):
        return [END_ID] if text == "<|im_end|>" else [ord(c) for c in text]

    def _plan(self, ctx: str, name: str, n: int) -> str:
        if "Submit the sorted list" in ctx[ctx.rfind("<|im_start|>user") :]:
            got = [int(x) for x in re.findall(READ, ctx)]
            return "<think>final</think>#### " + json.dumps(sorted(got + [self.own[name]]))
        if name in self.silent:
            return "<think>stay quiet</think>ok"
        if '"status": "sent"' not in ctx:
            return (
                '<think>share</think><tool_call>\n{"name": "message", "arguments": {"to": "all", "text": '
                f'"number {self.own[name]}"}}}}\n</tool_call>'
            )
        got = re.findall(READ, ctx)
        if len(got) == n - 1:
            return "<think>done</think>#### " + json.dumps(sorted([int(x) for x in got] + [self.own[name]]))
        if ctx.rfind('"unread_by_sender"') > ctx.rfind('"messages"'):
            return '<think>notified</think><tool_call>\n{"name": "read_messages", "arguments": {}}\n</tool_call>'
        return "<think>nothing to do; wait</think>ok"

    async def generate(self, prompt, max_tokens, temperature):
        ctx = "".join(chr(t) if t > 1 else "<|im_end|>" for t in prompt)
        name = re.search(r"Your name is (agent_\d+)", ctx).group(1)
        n = len(re.search(r"Your teammates: (.*)\.", ctx).group(1).split(", ")) + 1
        await asyncio.sleep(0)  # let teammates interleave
        at, rest = self.pending.pop(name, (None, ""))
        # continue a cut response only if nothing was appended since; after a notification, re-plan like a model would
        text = rest if at == len(prompt) and rest else self._plan(ctx, name, n)
        out, rest = text[:max_tokens], text[max_tokens:]
        if rest:
            self.pending[name] = (len(prompt) + len(out), rest)
        return [ord(c) for c in out], [-0.5] * len(out), out, "length" if rest else "stop"


def team(silent=frozenset(), chunk=40, n=4):
    p = Problem.make(problem=3, seed=11, n=n)
    fake = FakeSampler(silent=frozenset(p.names[i] for i in silent))
    fake.own = dict(zip(p.names, p.numbers, strict=True))
    s = Settings(chunk=chunk, work_tokens=4000, final_tokens=200)
    return p, asyncio.run(run_team(fake, 3, 11, n, s)), s


def test_scripted_team_shares_reads_and_answers_through_chunks_and_notifications():
    p, ep, s = team()
    assert ep.success and all(a.answer == p.target for a in ep.agents)
    for a in ep.agents:
        assert len(a.tokens) == len(a.mask) == len(a.logprobs)
        assert a.work_generated + a.final_generated == sum(a.mask)
        assert all(
            (lp != 0.0) == bool(m) for lp, m in zip(a.logprobs, a.mask, strict=True)
        )  # logprobs only on sampled tokens
        kinds = [e["kind"] for e in a.events]
        assert "message" in kinds and "read" in kinds and kinds.count("submit") == 1
    # 40-token requests cut every response, so notifications also arrive at request boundaries
    assert any(e["kind"] == "notified" for a in ep.agents for e in a.events)
    assert ep.tokens == sum(sum(a.mask) for a in ep.agents)
    assert ep.reward == team_reward(True, ep.tokens, 4, s.allowance) and 0.75 <= ep.reward < 1


def test_a_silent_agent_stalls_the_team_into_the_final_request_and_fails():
    p, ep, _ = team(silent={2})
    assert not ep.success
    for a in ep.agents:
        reasons = [e["reason"] for e in a.events if e["kind"] == "work_end"]
        assert reasons == ["idle"], (a.name, reasons)  # nobody could finish: everyone idles out
        assert a.final_generated > 0 and a.answer is not None and len(a.answer) in (1, 3)  # asked once


def test_board_post_wakes_an_idle_agent_and_the_last_one_out_releases_everyone():
    async def go():
        b = Board(["a", "b", "c"])
        idle_a = asyncio.create_task(b.idle("a", set()))
        await asyncio.sleep(0)
        assert b.active == 2
        b.post("b", "a", "hi")  # wake: counted active again at once
        assert await idle_a is False and b.active == 3
        assert [m.sender for m in b.read("a")] == ["b"] and b.unread("a") == []
        idle_a = asyncio.create_task(b.idle("a", {0}))
        idle_c = asyncio.create_task(b.idle("c", set()))
        await asyncio.sleep(0)
        b.finish("b")  # nobody left to post: the work phase is over
        assert await idle_a is True and await idle_c is True and b.stalled

    asyncio.run(go())


def test_final_answer_is_read_after_the_thinking_only():
    text = "<think>maybe #### [9, 9, 9, 9]</think>#### [3, 1] then #### [1, 2, 3, 4]"
    assert parse_answer(after_thinking(text)) == [1, 2, 3, 4]
    assert parse_answer(after_thinking("<think>#### [1, 2, 3, 4]</think>[1, 2, 3, 4]")) is None


def test_team_batcher_puts_one_whole_team_on_each_row():
    import numpy as np
    from async_grpo_s3_aggregation import TeamBatcher

    def row(team, i):
        ids = np.array([7, 7, 7, team, i], dtype=np.int64)  # a shared prefix, then the agent's own tokens
        return {
            "input_ids": ids,
            "completion_mask": np.array([0, 0, 0, 1, 1]),
            "group_id": 0,
            "metrics": {"team": f"team {team}", "reward_trained": 1.0},
        }

    stream = [row(t, i) for t in range(5) for i in range(4)]
    batches = list(TeamBatcher(stream, num_processes=2, token_budget=0, metrics={}))
    assert len(batches) == 2  # teams 0-3 in pairs; team 4 waits for a teammate's successor
    for b in batches:
        for r in b:
            assert len(r.samples) == 4 and len({int(s["input_ids"][3]) for s in r.samples}) == 1
            assert all("team" not in s["metrics"] for s in r.samples)  # never logged
            assert len(r.tokens) == 3 + 4 * 2  # the shared prefix is forwarded once
