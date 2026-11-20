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

"""The s3 harness (§3.1, Appendix F): N asynchronous token-level agents, an in-memory board, one team episode.

Each agent works on token ids, so the exact sampled tokens and their sampling logprobs are kept for training:
- `Chat` renders text with the tokenizer's chat template. The first prompt is the system prompt and first user turn
  with the paper's tools. From then on the context only grows: the template's rendering of each new message (tool
  results, notifications, the final request) is appended as tokens, and earlier turns are never re-rendered.
- A sampler generates at most `chunk` (1,024) new tokens per request. At a request boundary a new message arrival is
  delivered as a notification, appended as a tool output (App. F: "the sampled prefix is kept unchanged, a
  notification is appended as a tool output, and the agent continues"); with the chat template that closes the
  assistant message in progress.
- A complete response with tool calls runs them on the agent's `AgentTools` and appends their results (plus a
  notification if messages arrived). A response with neither a tool call nor an answer makes the agent idle until a
  message it has not been told about arrives. When every agent is idle or finished, the work phase ends.
- An agent that has not answered when its work phase ends (idle with nobody left to wake it, `work_tokens` spent,
  or the context full) is asked once with the final-answer request and gets `final_tokens` more.

Known deviation: an agent whose own allowance runs out is asked for its final answer at once, not when the whole
team's work phase ends. It cannot read anything in between, so its context at the request is the same.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Protocol

import httpx

from .task import (
    FINAL_REQUEST,
    TOOL_CALL,
    TOOLS,
    Problem,
    after_thinking,
    count_words,
    notice,
    opening,
    parse_answer,
    team_reward,
)


READ_LIMIT = 16


@dataclass
class Settings:
    work_tokens: int = 8192
    final_tokens: int = 1024
    chunk: int = 1024
    context: int = 40960
    temperature: float = 1.0
    token_cost: bool = True  # say so in the system prompt (Eq. 1's cost is applied either way by the caller)
    enable_thinking: bool = True

    @property
    def allowance(self) -> int:  # L in Eq. 1
        return self.work_tokens + self.final_tokens


class Sampler(Protocol):
    async def generate(self, prompt_ids: list[int], max_tokens: int) -> tuple[list[int], list[float], str]:
        """Sample at most `max_tokens` after `prompt_ids`: the token ids, their sampling logprobs, and the finish
        reason, `"length"` when cut at `max_tokens` and `"stop"` otherwise."""


class VLLMSampler:
    """`/v1/completions` on token ids, for evaluating a served model outside the trainer (the trainer samples through
    its rollout loop instead)."""

    def __init__(self, base_url: str, model: str, temperature: float = 1.0, timeout_s: float = 900.0):
        self.base_url, self.model, self.temperature = base_url.rstrip("/").removesuffix("/v1"), model, temperature
        # uvicorn closes idle keep-alive connections after 5 s; idle agents leave pooled ones longer than that, so
        # expire them first (a reused dead connection fails with ReadError)
        self.client = httpx.AsyncClient(
            timeout=timeout_s,
            limits=httpx.Limits(max_connections=None, max_keepalive_connections=512, keepalive_expiry=2.0),
        )

    async def generate(self, prompt_ids, max_tokens):
        body = {
            "model": self.model,
            "prompt": prompt_ids,
            "max_tokens": max_tokens,
            "temperature": self.temperature,
            "top_p": 1.0,
            "logprobs": 0,
            "return_token_ids": True,
        }
        for k in range(5):  # transport errors are retried: nothing is appended to a context until a response arrives
            try:
                r = await self.client.post(f"{self.base_url}/v1/completions", json=body)
                r.raise_for_status()
                break
            except (httpx.TransportError, httpx.HTTPStatusError) as e:
                if k == 4 or (isinstance(e, httpx.HTTPStatusError) and e.response.status_code < 500):
                    raise
                await asyncio.sleep(0.5 * 2**k)
        c = r.json()["choices"][0]
        lps = [float(x) if x is not None else 0.0 for x in c["logprobs"]["token_logprobs"]]
        return list(c["token_ids"]), lps, c.get("finish_reason") or "stop"

    async def aclose(self):
        await self.client.aclose()


class Chat:
    """The chat template as token ids that only grow.

    `then(messages)` is what the template renders for `messages` after an assistant turn: the turn's close, the
    messages and the generation prompt, read off a rendering behind a placeholder assistant turn. `close` is the
    close alone; when the model has already sampled the close's first token (its end-of-turn token), the rest of the
    close is all that is still missing."""

    _MARK = "\x00assistant turn\x00"

    def __init__(self, tokenizer, tools=TOOLS, chat_template=None, **chat_template_kwargs):
        self.tok, self.tools, self.template, self.kwargs = tokenizer, tools, chat_template, chat_template_kwargs
        self._close_text = self._after([], add_generation_prompt=False)
        self.close = self.tok.encode(self._close_text, add_special_tokens=False)

    def _render(self, messages, add_generation_prompt):
        return self.tok.apply_chat_template(
            messages,
            tools=self.tools,
            chat_template=self.template,
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
            **self.kwargs,
        )

    def _after(self, messages, add_generation_prompt=True):
        turn = [{"role": "user", "content": "."}, {"role": "assistant", "content": self._MARK}]
        return self._render(turn + messages, add_generation_prompt).split(self._MARK, 1)[1]

    def prompt(self, messages) -> list[int]:
        return self.tok.encode(self._render(messages, True), add_special_tokens=False)

    def then(self, messages) -> list[int]:
        text = self._after(messages)
        if not text.startswith(self._close_text):
            raise ValueError("the chat template does not render messages after an assistant turn as an append")
        return self.close + self.tok.encode(text[len(self._close_text) :], add_special_tokens=False)

    def decode(self, ids) -> str:
        return self.tok.decode(ids, skip_special_tokens=False)


@dataclass
class Msg:
    id: int
    sender: str
    text: str
    read: bool = False


class Board:
    """Direct messages (to a teammate or all), unread tracking, and the team's idle/wake/end-of-work bookkeeping.

    `active` counts agents that are working (not idle, not finished). A post to an idle agent wakes it (counted
    active again at once, so the count never reaches 0 while a wake-up is pending); when it reaches 0 nobody can post
    again, so every idle agent is released as stalled and the work phase is over."""

    def __init__(self, names: list[str]):
        self.names = list(names)
        self.inbox: dict[str, list[Msg]] = {n: [] for n in names}
        self.active = len(names)
        self.stalled = False
        self._idle: dict[str, asyncio.Future] = {}
        self._next = 0

    def post(self, sender: str, to: str, text: str) -> list[str]:
        to_ = [n for n in self.names if n != sender] if to == "all" else [to]
        for r in to_:
            self.inbox[r].append(Msg(self._next, sender, text))
            self._next += 1
            fut = self._idle.pop(r, None)
            if fut is not None and not fut.done():
                self.active += 1
                fut.set_result(False)
        return to_

    def unread(self, name: str) -> list[Msg]:
        return [m for m in self.inbox[name] if not m.read]

    def read(self, name: str, limit: int = READ_LIMIT) -> list[Msg]:
        out = self.unread(name)[:limit]
        for m in out:
            m.read = True
        return out

    async def idle(self, name: str, known: set[int]) -> bool:
        """Park until an unread message not in `known` arrives (False), or the work phase ends (True)."""
        if any(m.id not in known for m in self.unread(name)):
            return False
        fut = asyncio.get_running_loop().create_future()
        self._idle[name] = fut
        self._leave()
        return await fut

    def finish(self, name: str) -> None:
        self._leave()

    def _leave(self) -> None:
        self.active -= 1
        if self.active == 0:
            self.stalled = True
            for fut in self._idle.values():
                if not fut.done():
                    fut.set_result(True)
            self._idle.clear()


class AgentTools:
    """One agent's tools (Table 2): its public methods are the tools the model can call, on the team's board.

    Their schemas in the prompt are `task.TOOLS`, the paper's Table 2 text."""

    def __init__(self, board: Board, name: str, teammates: list[str], events: list[dict]):
        self.board, self.name, self.teammates, self.events = board, name, teammates, events
        self.at = 0  # the agent's context length, for the event log

    def message(self, to: str, text: str) -> str:
        if not isinstance(to, str) or not isinstance(text, str):
            return json.dumps({"error": "message needs `to` (a teammate's name or all) and `text`"})
        to = to.strip()
        target = "all" if to.lower() == "all" else to
        ok = target == "all" or target in self.teammates
        self.events.append({"kind": "message", "at": self.at, "to": to, "text": text, "ok": ok})
        if not ok:
            return json.dumps({"error": f"unknown recipient {to!r}; use one of {self.teammates} or all"})
        self.board.post(self.name, target, text)
        return json.dumps({"status": "sent", "to": to})

    def read_messages(self) -> str:
        msgs = [{"from": m.sender, "text": m.text} for m in self.board.read(self.name)]
        self.events.append({"kind": "read", "at": self.at, "n": len(msgs), "from": [m["from"] for m in msgs]})
        return json.dumps({"messages": msgs})

    def check_notifications(self) -> str:
        counts: dict[str, int] = {}
        for m in self.board.unread(self.name):
            counts[m.sender] = counts.get(m.sender, 0) + 1
        return json.dumps({"unread_by_sender": counts})

    def _run(self, call: dict) -> str:
        """Run one parsed `<tool_call>` body: `{"name": ..., "arguments": {...}}`."""
        name, args = call.get("name"), call.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                return json.dumps({"error": "arguments are not valid JSON"})
        if not isinstance(args, dict):
            return json.dumps({"error": "arguments must be an object"})
        if name == "message":
            return self.message(args.get("to"), args.get("text"))
        if name in ("read_messages", "check_notifications"):
            return getattr(self, name)()  # no arguments: any given are ignored
        return json.dumps({"error": f"unknown tool {name!r}"})


@dataclass
class AgentState:
    name: str
    number: int
    tokens: list[int] = field(default_factory=list)
    mask: list[int] = field(default_factory=list)  # 1 = sampled by the model
    logprobs: list[float] = field(default_factory=list)  # sampling logprob, 0.0 where mask is 0
    prompt_len: int = 0
    work_generated: int = 0
    final_generated: int = 0
    answer: list[int] | None = None
    events: list[dict] = field(default_factory=list)


class Agent:
    def __init__(self, i: int, problem: Problem, board: Board, sampler: Sampler, chat: Chat, s: Settings):
        self.p, self.board, self.sampler, self.chat, self.s = problem, board, sampler, chat, s
        self.n = len(problem.names)
        self.st = AgentState(problem.names[i], problem.numbers[i])
        self.teammates = problem.teammates[i]
        self.tools = AgentTools(board, self.st.name, self.teammates, self.st.events)
        self.notified: set[int] = set()
        self._finish = "stop"
        self._last_then: tuple[int, list[dict]] | None = None

    # ---- tokens -------------------------------------------------------------------------------------------------
    async def _gen(self, max_tokens: int) -> str:
        st = self.st
        ids, lps, self._finish = await self.sampler.generate(st.tokens, max_tokens)
        st.tokens += ids
        st.mask += [1] * len(ids)
        st.logprobs += lps
        self.tools.at = len(st.tokens)
        return self.chat.decode(ids)

    def _then(self, messages: list[dict]) -> None:
        """Close the assistant turn in progress and append `messages` with the generation prompt."""
        st, ids = self.st, self.chat.then(messages)
        self._last_then = (len(st.tokens), messages)
        if self._finish == "stop" and st.mask[-1] == 1 and st.tokens[-1] == ids[0]:
            ids = ids[1:]  # the model ended its turn itself
        st.tokens += ids
        st.mask += [0] * len(ids)
        st.logprobs += [0.0] * len(ids)
        self.tools.at = len(st.tokens)

    def _tool_results(self, results: list[str]) -> None:
        self._then([{"role": "tool", "content": r} for r in results])

    # ---- board --------------------------------------------------------------------------------------------------
    def _notice(self) -> str | None:
        unread = self.board.unread(self.st.name)
        if not any(m.id not in self.notified for m in unread):
            return None
        self.notified |= {m.id for m in unread}
        counts: dict[str, int] = {}
        for m in unread:
            counts[m.sender] = counts.get(m.sender, 0) + 1
        self.st.events.append({"kind": "notified", "at": len(self.st.tokens), "unread": counts})
        return notice(counts)

    # ---- the episode --------------------------------------------------------------------------------------------
    def _room(self) -> int:
        """Work tokens left: the agent's allowance, and its context minus room for the final answer."""
        s, st = self.s, self.st
        return min(s.work_tokens - st.work_generated, s.context - len(st.tokens) - s.final_tokens - 256)

    async def run(self) -> AgentState:
        st, i = self.st, self.p.names.index(self.st.name)
        st.tokens = self.chat.prompt(opening(st.name, self.teammates, self.n, self.p.numbers[i], self.s.token_cost))
        st.mask, st.logprobs, st.prompt_len = [0] * len(st.tokens), [0.0] * len(st.tokens), len(st.tokens)
        s, segment = self.s, ""
        try:
            while True:
                room = self._room()
                if room <= 0:
                    self.board.finish(st.name)
                    reason = "allowance" if st.work_generated >= s.work_tokens else "context"
                    break
                before = len(st.tokens)
                segment += await self._gen(min(s.chunk, room))
                st.work_generated += len(st.tokens) - before
                if self._finish == "length":  # a request boundary: deliver a notification if one is due
                    # unless the work budget is spent: the final request comes next, with no turn to act on it
                    note = self._notice() if self._room() > 0 else None
                    if note is not None:
                        self._tool_results([note])
                        segment = ""
                    continue
                visible = after_thinking(segment)
                segment = ""
                calls = []
                for raw in TOOL_CALL.findall(visible):
                    try:
                        c = json.loads(raw)
                        calls.append(c if isinstance(c, dict) else {"_bad": raw})
                    except json.JSONDecodeError:
                        calls.append({"_bad": raw})
                if calls:
                    results = [
                        json.dumps({"error": "tool call is not valid JSON"}) if "_bad" in c else self.tools._run(c)
                        for c in calls
                    ]
                    note = self._notice()
                    if note is not None:
                        results.append(note)
                    self._tool_results(results)
                    continue
                answer = parse_answer(visible)
                if answer is not None:
                    st.answer = answer
                    st.events.append({"kind": "submit", "at": len(st.tokens), "phase": "work", "answer": answer})
                    self.board.finish(st.name)
                    reason = "answered"
                    break
                stalled = await self.board.idle(st.name, self.notified)
                if stalled:
                    reason = "idle"
                    break
                st.events.append({"kind": "woken", "at": len(st.tokens)})
                self._tool_results([self._notice()])
        except BaseException:
            if not self.board.stalled:
                self.board.finish(st.name)  # never leave teammates parked on an agent that died
            raise
        pending = []
        if st.answer is None and st.mask[-1] == 0 and self._last_then is not None:
            # the work phase ended right after tool results, whose generation prompt is still open: take them back,
            # to render the final request after them instead of closing an empty assistant turn
            at, pending = self._last_then
            del st.tokens[at:], st.mask[at:], st.logprobs[at:]
        st.events.append({"kind": "work_end", "at": len(st.tokens), "reason": reason})
        if st.answer is None:
            self._then([*pending, {"role": "user", "content": FINAL_REQUEST.format(n=count_words(self.n))}])
            before = len(st.tokens)
            text = await self._gen(self.s.final_tokens)
            st.final_generated = len(st.tokens) - before
            st.answer = parse_answer(after_thinking(text)) or []
            st.events.append({"kind": "submit", "at": len(st.tokens), "phase": "final", "answer": st.answer})
        return st


@dataclass
class Episode:
    problem: Problem
    agents: list[AgentState]
    success: bool
    tokens: int  # sum_i |tau_i|: every generated token, work and final
    reward: float

    def summary(self, with_tokens: bool = False) -> dict:
        target = self.problem.target
        out = {
            "problem": self.problem.problem,
            "seed": self.problem.seed,
            "n": len(self.agents),
            "target": target,
            "success": self.success,
            "tokens": self.tokens,
            "reward": self.reward,
            "agents": [],
        }
        for a in self.agents:
            d = {
                "name": a.name,
                "number": a.number,
                "work": a.work_generated,
                "final": a.final_generated,
                "answer": a.answer,
                "correct": a.answer == target,
                "events": a.events,
            }
            if with_tokens:
                d |= {"tokens": a.tokens, "mask": a.mask}
            out["agents"].append(d)
        return out


async def run_team(
    sampler: Sampler, chat: Chat, problem: int, seed: int, n: int = 4, s: Settings | None = None
) -> Episode:
    s = s or Settings()
    p = Problem.make(problem, seed, n)
    board = Board(p.names)
    agents = [Agent(i, p, board, sampler, chat, s) for i in range(n)]
    async with asyncio.TaskGroup() as tg:  # one agent failing cancels the rest (they release the board)
        tasks = [tg.create_task(a.run()) for a in agents]
    states = [t.result() for t in tasks]
    success = any(a.answer == p.target for a in states)
    tokens = sum(a.work_generated + a.final_generated for a in states)
    return Episode(p, list(states), success, tokens, team_reward(success, tokens, n, s.allowance))
