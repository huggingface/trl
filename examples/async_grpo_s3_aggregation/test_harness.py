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

import pytest
from harness import Board, Chat, Problem, Settings, parse_answer, run_team, team_reward
from harness.task import TOOLS, after_thinking


SPECIAL = {"<|im_end|>": 1, "<|im_start|>": 2}
READ = r'"from": "agent_\d+", "text": "number (\d+)"'  # a read result, not our own sent text


class CharTokenizer:
    """Characters as tokens, two special tokens, and a ChatML template that renders tool outputs like Qwen3's."""

    def apply_chat_template(self, messages, tools=None, chat_template=None, add_generation_prompt=False, **kwargs):
        out = ""
        for i, m in enumerate(messages):
            if m["role"] != "tool":
                out += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
                continue
            if i == 0 or messages[i - 1]["role"] != "tool":
                out += "<|im_start|>user"
            out += f"\n<tool_response>\n{m['content']}\n</tool_response>"
            if i == len(messages) - 1 or messages[i + 1]["role"] != "tool":
                out += "<|im_end|>\n"
        return out + ("<|im_start|>assistant\n" if add_generation_prompt else "")

    def encode(self, text, add_special_tokens=False):
        parts = re.split(r"(<\|im_end\|>|<\|im_start\|>)", text)
        return [i for p in parts for i in ([SPECIAL[p]] if p in SPECIAL else [ord(c) for c in p])]

    def decode(self, ids, skip_special_tokens=False):
        names = {v: k for k, v in SPECIAL.items()}
        return "".join(names.get(t, chr(t)) for t in ids)


class FakeSampler:
    """Each agent: message its number to all, read when notified, answer once it holds every number, else idle.
    `silent` agents never message or read. Responses are cut into `chunk`-sized requests like the real server's."""

    def __init__(self, silent: frozenset[str] = frozenset()):
        self.silent = silent
        self.pending: dict[str, str] = {}

    async def generate(self, prompt, max_tokens):
        tok = CharTokenizer()
        ctx = tok.decode(prompt)
        name = re.search(r"Your name is (agent_\d+)", ctx).group(1)
        n = len(re.search(r"Your teammates: (.*)\.", ctx).group(1).split(", ")) + 1
        await asyncio.sleep(0)  # let teammates interleave
        at, rest = self.pending.pop(name, (None, ""))
        got = [int(x) for x in re.findall(READ, ctx)]
        # continue a cut response only if nothing was appended since; after a notification, re-plan like a model would
        if at == len(prompt) and rest:
            text = rest
        elif "Submit the sorted list" in ctx[ctx.rfind("<|im_start|>user") :]:
            text = "<think>final</think>#### " + json.dumps(sorted(got + [self.own[name]]))
        elif name in self.silent:
            text = "<think>stay quiet</think>ok"
        elif '"status": "sent"' not in ctx:
            text = (
                '<think>share</think><tool_call>\n{"name": "message", "arguments": {"to": "all", "text": '
                f'"number {self.own[name]}"}}}}\n</tool_call>'
            )
        elif len(got) == n - 1:
            text = "<think>done</think>#### " + json.dumps(sorted(got + [self.own[name]]))
        elif ctx.rfind('"unread_by_sender"') > ctx.rfind('"messages"'):
            text = '<think>notified</think><tool_call>\n{"name": "read_messages", "arguments": {}}\n</tool_call>'
        else:
            text = "<think>nothing to do; wait</think>ok"
        ids = tok.encode(text + "<|im_end|>")
        out, rest = ids[:max_tokens], ids[max_tokens:]
        if rest:
            self.pending[name] = (len(prompt) + len(out), tok.decode(rest)[: -len("<|im_end|>")])
        return out, [-0.5] * len(out), "length" if rest else "stop"


def team(silent=frozenset(), chunk=40, n=4):
    p = Problem.make(problem=3, seed=11, n=n)
    fake = FakeSampler(silent=frozenset(p.names[i] for i in silent))
    fake.own = dict(zip(p.names, p.numbers, strict=True))
    s = Settings(chunk=chunk, work_tokens=4000, final_tokens=200)
    return p, asyncio.run(run_team(fake, Chat(CharTokenizer()), 3, 11, n, s)), s


def test_chat_appends_what_the_template_renders_after_an_assistant_turn():
    chat = Chat(CharTokenizer())
    assert chat.decode(chat.close) == "<|im_end|>\n"
    two = chat.then([{"role": "tool", "content": "a"}, {"role": "tool", "content": "b"}])
    assert chat.decode(two) == (
        "<|im_end|>\n<|im_start|>user\n<tool_response>\na\n</tool_response>\n<tool_response>\nb\n"
        "</tool_response><|im_end|>\n<|im_start|>assistant\n"
    )


def test_chat_matches_the_qwen3_chat_template():
    """With Qwen3's template, `Chat` appends exactly the ChatML text that the harness used to write by hand."""
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    except OSError:
        pytest.skip("Qwen/Qwen3-0.6B tokenizer not available")
    chat = Chat(tok, enable_thinking=True)
    enc = lambda text: tok.encode(text, add_special_tokens=False)  # noqa: E731
    assert chat.close == enc("<|im_end|>") + enc("\n")
    results = ['{"status": "sent", "to": "all"}', '{"unread_by_sender": {"agent_3": 1}}']
    by_hand = "\n".join(f"<tool_response>\n{r}\n</tool_response>" for r in results)
    assert chat.then([{"role": "tool", "content": r} for r in results]) == chat.close + enc(
        f"<|im_start|>user\n{by_hand}<|im_end|>\n<|im_start|>assistant\n"
    )
    final = "Submit the sorted list now."
    assert chat.then([{"role": "user", "content": final}]) == chat.close + enc(
        f"<|im_start|>user\n{final}<|im_end|>\n<|im_start|>assistant\n"
    )
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
    rendered = tok.apply_chat_template(messages, tools=TOOLS, add_generation_prompt=True, enable_thinking=True)
    assert chat.prompt(messages) == list(rendered if isinstance(rendered, list) else rendered["input_ids"])


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
