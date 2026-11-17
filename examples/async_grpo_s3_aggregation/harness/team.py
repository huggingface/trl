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

Each agent works against a vLLM server token by token, so the exact sampled tokens and their sampling logprobs are
kept for training:
- The first prompt is the system prompt and first user turn, rendered with the chat template and the paper's tools
  by the server's `/tokenize`. From then on the context only grows: tool results, notifications and the final
  request are appended as tokens, never re-rendered.
- Generation runs in requests of at most `chunk` (1,024) new tokens on the token ids. At a request boundary a new
  message arrival is delivered as a notification, appended as a tool output (App. F: "the sampled prefix is kept
  unchanged, a notification is appended as a tool output, and the agent continues"); with the chat template that
  closes the assistant message in progress.
- A complete response with tool calls runs them and appends their results (plus a notification if messages arrived).
  A response with neither a tool call nor an answer makes the agent idle until a message it has not been told about
  arrives. When every agent is idle or finished, the work phase ends.
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
    first_user_turn,
    notice,
    parse_answer,
    system_prompt,
    team_reward,
)


READ_LIMIT = 16
END = "<|im_end|>"


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
    async def chat_prompt(self, messages: list[dict], tools: list[dict], enable_thinking: bool) -> list[int]: ...
    async def encode(self, text: str) -> list[int]: ...
    async def generate(
        self, prompt: list[int], max_tokens: int, temperature: float
    ) -> tuple[list[int], list[float], str, str]: ...


class VLLMSampler:
    """`/tokenize` for the chat template, `/v1/completions` on token ids with the sampled ids and logprobs."""

    def __init__(self, base_url: str, model: str, client: httpx.AsyncClient | None = None, timeout_s: float = 900.0):
        self.base_url, self.model = base_url.rstrip("/").removesuffix("/v1"), model
        # uvicorn closes idle keep-alive connections after 5 s; idle agents leave pooled ones longer than that, so
        # expire them first (a reused dead connection fails with ReadError)
        self.client = client or httpx.AsyncClient(
            timeout=timeout_s,
            limits=httpx.Limits(max_connections=None, max_keepalive_connections=512, keepalive_expiry=2.0),
        )
        self._cache: dict[str, list[int]] = {}

    async def _post(self, path: str, body: dict, attempts: int = 5) -> dict:
        """Transport errors are retried: nothing is appended to an agent's context until a response arrives."""
        for k in range(attempts):
            try:
                r = await self.client.post(f"{self.base_url}{path}", json=body)
                r.raise_for_status()
                return r.json()
            except (httpx.TransportError, httpx.HTTPStatusError) as e:
                if k == attempts - 1 or (isinstance(e, httpx.HTTPStatusError) and e.response.status_code < 500):
                    raise
                await asyncio.sleep(0.5 * 2**k)
        raise AssertionError("unreachable")

    async def chat_prompt(self, messages, tools, enable_thinking):
        out = await self._post(
            "/tokenize",
            {
                "model": self.model,
                "messages": messages,
                "tools": tools,
                "add_generation_prompt": True,
                "chat_template_kwargs": {"enable_thinking": enable_thinking},
            },
        )
        return list(out["tokens"])

    async def encode(self, text):
        if text not in self._cache:
            out = await self._post("/tokenize", {"model": self.model, "prompt": text, "add_special_tokens": False})
            ids = list(out["tokens"])
            if len(text) > 64:
                return ids
            self._cache[text] = ids
        return list(self._cache[text])

    async def generate(self, prompt, max_tokens, temperature):
        out = await self._post(
            "/v1/completions",
            {
                "model": self.model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": 1.0,
                "logprobs": 0,
                "return_tokens_as_token_ids": True,
                "skip_special_tokens": False,
            },
        )
        c = out["choices"][0]
        lp = c.get("logprobs") or {}
        ids = [int(t.split(":", 1)[1]) for t in lp.get("tokens") or []]
        lps = [float(x) if x is not None else 0.0 for x in lp.get("token_logprobs") or []]
        return ids, lps, c.get("text") or "", c.get("finish_reason") or "stop"

    async def aclose(self):
        await self.client.aclose()


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
    def __init__(self, i: int, problem: Problem, board: Board, sampler: Sampler, s: Settings):
        self.p, self.board, self.sampler, self.s = problem, board, sampler, s
        self.n = len(problem.names)
        self.st = AgentState(problem.names[i], problem.numbers[i])
        self.teammates = problem.teammates[i]
        self.notified: set[int] = set()
        self._end: set[int] = set()

    # ---- tokens -------------------------------------------------------------------------------------------------
    async def _env(self, text: str) -> None:
        ids = await self.sampler.encode(text)
        self.st.tokens += ids
        self.st.mask += [0] * len(ids)
        self.st.logprobs += [0.0] * len(ids)

    async def _gen(self, max_tokens: int) -> tuple[str, str]:
        ids, lps, text, finish = await self.sampler.generate(self.st.tokens, max_tokens, self.s.temperature)
        self.st.tokens += ids
        self.st.mask += [1] * len(ids)
        self.st.logprobs += lps
        return text, finish

    async def _close_assistant(self, finish: str) -> None:
        st = self.st
        if not (finish == "stop" and st.mask and st.mask[-1] == 1 and st.tokens[-1] in self._end):
            await self._env(END)
        await self._env("\n")

    async def _user(self, body: str) -> None:
        await self._env(f"<|im_start|>user\n{body}{END}\n<|im_start|>assistant\n")

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

    def _tool(self, call: dict) -> str:
        name, args = call.get("name"), call.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                return json.dumps({"error": "arguments are not valid JSON"})
        if not isinstance(args, dict):
            return json.dumps({"error": "arguments must be an object"})
        if name == "message":
            to, text = args.get("to"), args.get("text")
            if not isinstance(to, str) or not isinstance(text, str):
                return json.dumps({"error": "message needs `to` (a teammate's name or all) and `text`"})
            to = to.strip()
            target = "all" if to.lower() == "all" else to
            ok = target == "all" or target in self.teammates
            self.st.events.append({"kind": "message", "at": len(self.st.tokens), "to": to, "text": text, "ok": ok})
            if not ok:
                return json.dumps({"error": f"unknown recipient {to!r}; use one of {self.teammates} or all"})
            self.board.post(self.st.name, target, text)
            return json.dumps({"status": "sent", "to": to})
        if name == "read_messages":
            msgs = [{"from": m.sender, "text": m.text} for m in self.board.read(self.st.name)]
            self.st.events.append(
                {"kind": "read", "at": len(self.st.tokens), "n": len(msgs), "from": [m["from"] for m in msgs]}
            )
            return json.dumps({"messages": msgs})
        if name == "check_notifications":
            counts: dict[str, int] = {}
            for m in self.board.unread(self.st.name):
                counts[m.sender] = counts.get(m.sender, 0) + 1
            return json.dumps({"unread_by_sender": counts})
        return json.dumps({"error": f"unknown tool {name!r}"})

    # ---- the episode --------------------------------------------------------------------------------------------
    async def run(self) -> AgentState:
        st, i = self.st, self.p.names.index(self.st.name)
        st.tokens = await self.sampler.chat_prompt(
            [
                {"role": "system", "content": system_prompt(st.name, self.teammates, self.n, self.s.token_cost)},
                {"role": "user", "content": first_user_turn(self.n, self.p.numbers[i])},
            ],
            TOOLS,
            self.s.enable_thinking,
        )
        st.mask, st.logprobs, st.prompt_len = [0] * len(st.tokens), [0.0] * len(st.tokens), len(st.tokens)
        self._end = set(await self.sampler.encode(END))
        try:
            reason = await self._work()
        except BaseException:
            if not self.board.stalled:
                self.board.finish(st.name)  # never leave teammates parked on an agent that died
            raise
        st.events.append({"kind": "work_end", "at": len(st.tokens), "reason": reason})
        if st.answer is None:
            await self._user(FINAL_REQUEST.format(n=count_words(self.n)))
            before = len(st.tokens)
            text, _ = await self._gen(self.s.final_tokens)
            st.final_generated = len(st.tokens) - before
            st.answer = parse_answer(after_thinking(text)) or []
            st.events.append({"kind": "submit", "at": len(st.tokens), "phase": "final", "answer": st.answer})
        return st

    async def _work(self) -> str:
        st, s, segment = self.st, self.s, ""
        while True:
            room = min(s.work_tokens - st.work_generated, s.context - len(st.tokens) - s.final_tokens - 256)
            if room <= 0:
                self.board.finish(st.name)
                return "allowance" if st.work_generated >= s.work_tokens else "context"
            before = len(st.tokens)
            text, finish = await self._gen(min(s.chunk, room))
            st.work_generated += len(st.tokens) - before
            segment += text
            if finish == "length":  # a request boundary: deliver a notification if one is due
                note = self._notice()
                if note is not None:
                    await self._close_assistant(finish)
                    await self._user(f"<tool_response>\n{note}\n</tool_response>")
                    segment = ""
                continue
            await self._close_assistant(finish)
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
                    json.dumps({"error": "tool call is not valid JSON"}) if "_bad" in c else self._tool(c)
                    for c in calls
                ]
                note = self._notice()
                if note is not None:
                    results.append(note)
                await self._user("\n".join(f"<tool_response>\n{r}\n</tool_response>" for r in results))
                continue
            answer = parse_answer(visible)
            if answer is not None:
                st.answer = answer
                st.events.append({"kind": "submit", "at": len(st.tokens), "phase": "work", "answer": answer})
                self.board.finish(st.name)
                return "answered"
            stalled = await self.board.idle(st.name, self.notified)
            if stalled:
                return "idle"
            st.events.append({"kind": "woken", "at": len(st.tokens)})
            await self._user(f"<tool_response>\n{self._notice()}\n</tool_response>")


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


async def run_team(sampler: Sampler, problem: int, seed: int, n: int = 4, s: Settings | None = None) -> Episode:
    s = s or Settings()
    p = Problem.make(problem, seed, n)
    board = Board(p.names)
    agents = [Agent(i, p, board, sampler, s) for i in range(n)]
    async with asyncio.TaskGroup() as tg:  # one agent failing cancels the rest (they release the board)
        tasks = [tg.create_task(a.run()) for a in agents]
    states = [t.result() for t in tasks]
    success = any(a.answer == p.target for a in states)
    tokens = sum(a.work_generated + a.final_generated for a in states)
    return Episode(p, list(states), success, tokens, team_reward(success, tokens, n, s.allowance))
