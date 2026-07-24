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

from __future__ import annotations

import asyncio
import functools
import json
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Protocol, TypedDict, cast

from accelerate.logging import get_logger
from openenv.core.harness import HarnessAdapter, HarnessRunLimits, ModelStepResult, ResourceSessionFactory
from openenv.core.llm_client import LLMResponse, ToolCall

from ...chat_template_utils import parse_response
from .async_rollout_worker import (
    AsyncRolloutWorker,
    TurnRecord,
    _AsyncRolloutLoop,
    _chain_to_sequences,
)


logger = get_logger(__name__)

Message = dict[str, Any]


# TODO(@openenv): this is OpenEnv's proxy-trace record shape; it should be defined and exported by OpenEnv, not here.
class TraceEntry(TypedDict, total=False):
    request: dict[str, Any]  # forwarded chat body, e.g. {"messages": [...], "tools": [...] | None}
    response: dict[str, Any]  # upstream reply, e.g. {"choices": [{"message": {"content", "tool_calls"}}]}
    completion_token_ids: list[int]  # generated token ids for this turn
    completion_tokens: list[str]  # fallback token strings ("token_id:{id}") when ids are absent
    per_token_logps: list[float]  # generator logprobs for the generated tokens


# TODO(@openenv): this probably should live in OpenEnv to extend the base session for loop-owning harnesses.
class LoopOwningSession(Protocol):
    """The session contract the loop-owning path needs BEYOND OpenEnv's base `ResourceSession`. The agent runs its own
    loop, so we block until it finishes and read its captured proxy trace. `wait_for_completion`/`fetch_proxy_trace`
    are not on the base `ResourceSession` (they are loop-owning extensions, e.g. `OpenCodeSession`), so a factory used
    in loop-owning mode must return sessions satisfying this protocol."""

    def wait_for_completion(self, timeout_s: float | None = ...) -> int: ...
    def fetch_proxy_trace(self) -> list[TraceEntry]: ...


@dataclass
class HarnessRolloutOutcome:
    env_reward: float | None
    completion: list[Message]
    trace: list[TraceEntry]
    tool_call_count: int
    tool_failure_count: int
    tool_calls_by_name: dict[str, int]
    timed_out: bool


@dataclass
class HarnessTurn:
    """One agent turn from the trace, passed to `train_turn_fn` to decide whether it is trained."""

    messages: list[Message]  # the conversation sent to the model this turn (the prompt)
    tools: list[dict] | None  # tools available to the model this turn
    content: str  # the assistant's text content this turn
    tool_calls: list[dict]  # the tool calls the assistant emitted (empty for a pure-text turn)


def _tools_to_schema(tools: list) -> list[dict] | None:
    """Convert OpenEnv `Tool`s (MCP spec) into the OpenAI function schema `apply_chat_template` expects."""
    if not tools:
        return None
    return [
        {
            "type": "function",
            "function": {"name": t.name, "description": t.description, "parameters": t.input_schema},
        }
        for t in tools
    ]


def _msg_to_llm_response(msg: Message) -> LLMResponse:
    """Convert a `parse_response` assistant message into the `LLMResponse` the harness adapter expects."""
    tool_calls = []
    for i, tc in enumerate(msg.get("tool_calls") or []):
        fn = tc.get("function", tc)
        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        tool_calls.append(ToolCall(id=tc.get("id") or f"call_{i}", name=fn["name"], args=args))
    return LLMResponse(content=msg.get("content") or "", tool_calls=tool_calls)


class _HarnessRolloutLoop(_AsyncRolloutLoop):
    """`_AsyncRolloutLoop` whose `_generate_one` drives an OpenEnv session instead of TRL's own turn loop."""

    _provides_rollout_reward = True

    # An unscorable rollout: no training rows, reward None -> `_score_group` NaNs it out of the group baseline.
    _EMPTY_ROLLOUT = ([], [], [], 0, 0, None)

    def __init__(
        self,
        *,
        harness_session_factory: ResourceSessionFactory,
        harness_adapter: HarnessAdapter | None = None,
        rollout_reward_fn: Callable[[HarnessRolloutOutcome], float | None] | None = None,
        train_turn_fn: Callable[[HarnessTurn], bool] | None = None,
        agent_turn_fn: Callable[[list[TraceEntry]], list[TraceEntry]] | None = None,
        **loop_kwargs,
    ):
        super().__init__(**loop_kwargs)
        self._factory = harness_session_factory
        # An adapter (e.g. MCPHarnessAdapter) selects white-box (TRL samples each turn); `None` selects loop-owning
        # (the agent runs its own loop and we read its proxy trace).
        self._adapter = harness_adapter
        self._limits = HarnessRunLimits(
            max_turns=self.max_tool_calling_iterations if self.max_tool_calling_iterations is not None else 8,
            sampling={"temperature": self.temperature, "max_tokens": self.max_tokens},
        )
        self._rollout_reward_fn = rollout_reward_fn
        self._train_turn_fn = train_turn_fn
        self._agent_turn_fn = agent_turn_fn or _default_agent_entries
        self.reward_func_names.append("harness_reward")

        # Sized to max_inflight so sessions run concurrently without queuing or per-rollout thread churn.
        self._session_pool = ThreadPoolExecutor(
            max_workers=max(1, self.max_inflight_tasks), thread_name_prefix="harness-session"
        )
        # In-flight sessions, so `_run_loops` can close them on stop (see there). set ops are atomic under the GIL.
        self._live_sessions: set = set()

    async def _generate_one(self, prompt, tool_dict, tools, group_id=0):
        # TODO(@openenv): provide an async version for performance
        #  OpenEnv's harness layer is synchronous, so run the whole session on the pool.
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._session_pool, self._run_session, prompt, group_id)

    async def _run_loops(self, stop_event) -> None:
        async def _close_live_sessions_on_stop() -> None:
            await stop_event.wait()
            for session in list(self._live_sessions):
                try:
                    session.close()
                except Exception:
                    logger.warning("closing in-flight harness session on stop failed", exc_info=True)

        try:
            await asyncio.gather(super()._run_loops(stop_event), _close_live_sessions_on_stop())
        finally:
            self._session_pool.shutdown(wait=True)

    def _run_session(self, prompt, group_id=0):
        """Drive one OpenEnv session to completion and return the `_generate_one` tuple + the verify reward.

        The agent and its proxy are external, flaky processes; a single rollout that fails to launch or whose trace is
        malformed must NOT crash the worker (it would kill the whole run). Such rollouts are returned as unscorable
        (reward None, no rows) so training continues and the group baseline ignores them."""
        rollout_id = uuid.uuid4().hex
        # Stable per-group seed so seed-driven factories hand every generation of a group the same task. Keyed on
        # `group_id` (not the prompt) so it works with or without a dataset: when there is no dataset the prompt is
        # empty and the factory selects the task from this seed.
        seed = group_id
        # TODO(@openenv): session creation spins up external processes (sandbox + proxy) and is the flakiest step;
        # it should retry with backoff (ideally inside OpenEnv's factory). For now a failed create drops the rollout as
        # unscorable, which shrinks the effective group size - a high create-failure rate silently weakens the signal.
        try:
            session = self._factory.create(prompt, seed=seed, episode_id=rollout_id)
        except Exception:
            logger.warning("harness session create failed; scoring rollout as unscorable", exc_info=True)
            return self._EMPTY_ROLLOUT
        self._live_sessions.add(session)  # tracked so a stop can close it (unblocks wait_for_completion below)
        timed_out = False
        trace: list[TraceEntry] = []
        tool_calls_by_name: dict[str, int] = {}
        try:
            if self._adapter is not None:
                # white-box: the adapter runs the tool loop, calling `_sample_turn` each turn.
                turns: list[TurnRecord] = []
                result = self._adapter.run_white_box(
                    functools.partial(self._sample_turn, turns), session, self._limits
                )
                completion = result.messages
                tool_call_count = int(result.metrics.get("tool_calls", len(result.tool_trace)))
                tool_failure_count = sum(1 for entry in result.tool_trace if entry.result.error is not None)
            else:
                # TODO(@openenv): ResourceSessionFactory should probably be generic over the session type it creates.
                # so we can type hint that we need a LoopOwningSession and not a simple RessouceSession
                loop_session = cast(LoopOwningSession, session)
                try:
                    loop_session.wait_for_completion()
                except TimeoutError:
                    logger.warning("harness agent timed out; training captured turns, timed_out flagged")
                    timed_out = True
                trace = loop_session.fetch_proxy_trace()
                # Resolve the real agent turns ONCE (dropping any framework aux calls), then derive everything from them.
                entries = self._agent_turn_fn(trace)
                turns = _turns_from_trace(entries, self.tokenizer, self._train_turn_fn)
                completion = _messages_from_trace(entries)
                tool_calls_by_name = _tool_call_counts_by_name(entries)
                tool_call_count = sum(tool_calls_by_name.values())
                tool_failure_count = _tool_failure_count(entries)
            verify = session.verify(completion)
            env_reward = float(verify.env_reward) if verify.env_reward is not None else None
            outcome = HarnessRolloutOutcome(
                env_reward=env_reward,
                completion=completion,
                trace=trace,
                tool_call_count=tool_call_count,
                tool_failure_count=tool_failure_count,
                tool_calls_by_name=tool_calls_by_name,
                timed_out=timed_out,
            )
            reward = self._rollout_reward_fn(outcome) if self._rollout_reward_fn else env_reward
            sequences = _chain_to_sequences(turns, rollout_id, self._fork_threshold_tokens)
            completion_ids = [tid for turn in turns for tid in turn.output_ids]
            return completion, completion_ids, sequences, tool_call_count, tool_failure_count, reward
        except Exception:
            logger.warning("harness rollout failed; scoring as unscorable", exc_info=True)
            return self._EMPTY_ROLLOUT
        finally:
            self._live_sessions.discard(session)
            try:
                session.close()
            except Exception:
                logger.warning("harness session close failed", exc_info=True)

    def _sample_turn(self, turns: list[TurnRecord], messages, tools, sampling) -> ModelStepResult:
        """OpenEnv `ModelStep`: sample one assistant turn against vLLM and record a `TurnRecord` into `turns`."""
        prompt_ids = self.tokenizer.apply_chat_template(
            messages,
            tools=_tools_to_schema(tools),
            add_generation_prompt=True,
            tokenize=True,
            return_dict=False,
            chat_template=self.chat_template,
            **self.chat_template_kwargs,
        )
        # ModelStep is sync on a pool thread; bridge the async vLLM POST onto the loop's event loop.
        turn_ids, logprobs = asyncio.run_coroutine_threadsafe(self._generate_one_turn(prompt_ids), self._loop).result()
        turns.append(TurnRecord(prompt_ids, turn_ids, logprobs))
        message = parse_response(self.tokenizer, turn_ids, prefix=prompt_ids)
        return ModelStepResult(
            response=_msg_to_llm_response(message), prompt_ids=prompt_ids, completion_ids=turn_ids, logprobs=logprobs
        )


def _trace_output_ids(entry: TraceEntry) -> list[int]:
    """Generated token ids for one proxy-trace turn.

    Prefer `completion_token_ids`; if empty, recover them from `completion_tokens`, which vLLM renders as
    `"token_id:{id}"` when launched with `--return-tokens-as-token-ids`, avoiding a re-encode of the decoded text.
    """
    ids = entry.get("completion_token_ids") or []
    if ids:
        return list(ids)
    return [
        int(t[len("token_id:") :]) for t in (entry.get("completion_tokens") or []) if str(t).startswith("token_id:")
    ]


# TODO(@openenv): identifying which proxy-trace entries are real agent turns is really OpenEnv's job - the proxy KNOWS which
# call is the agent loop vs an auxiliary one (title generator, context summarizer). If OpenEnv tagged each trace
# record with its purpose, this filtering (and the `agent_turn_fn` hook below) would be unnecessary: the library would
# just read the tag. Until then we default to a framework-agnostic filter and let the caller override.
def _default_agent_entries(trace: list[TraceEntry]) -> list[TraceEntry]:
    """Default `agent_turn_fn`: every captured turn that has request messages and a response. Framework-agnostic - it
    only drops malformed/empty captures, NOT a framework's auxiliary calls (a title generator, a context summarizer,
    etc.). An agent framework that fires such calls must pass an `agent_turn_fn` that recognizes and drops them, so
    they are never trained with the rollout's reward, scored, or logged as the transcript (see the opencode example)."""
    return [entry for entry in trace if (entry.get("request") or {}).get("messages") and entry.get("response")]


def has_tool_call(turn: HarnessTurn) -> bool:
    """A ready-made `train_turn_fn`: keep only turns where the model took an ACTION (emitted a tool call)."""
    return bool(turn.tool_calls)


def _entry_to_turn(entry: TraceEntry) -> HarnessTurn:
    """View one proxy-trace entry as the `HarnessTurn` handed to `train_turn_fn`."""
    request = entry.get("request") or {}
    message = (entry.get("response", {}).get("choices") or [{}])[0].get("message") or {}
    return HarnessTurn(
        messages=request.get("messages") or [],
        tools=request.get("tools"),
        content=message.get("content") or "",
        tool_calls=message.get("tool_calls") or [],
    )


def _turns_from_trace(
    entries: list[TraceEntry], tokenizer, train_turn_fn: Callable[[HarnessTurn], bool] | None = None
) -> list[TurnRecord]:
    """Loop-owning path: rebuild per-turn `TurnRecord`s from the real agent turns (`entries`, already selected by the
    loop's `agent_turn_fn`). Re-tokenize each request's messages (passing its `tools` so the prompt matches what the
    upstream rendered); ids + logprobs come from the capture.

    By default every agent turn is trained. Which turns to reinforce beyond that is the CALLER's policy: pass
    `train_turn_fn(turn: HarnessTurn) -> bool` to narrow it, e.g. `has_tool_call` to train only turns that took an
    ACTION. That suits tool-heavy agents, where advantage is stamped per-rollout and training pure-text turns would
    reinforce prose over tool use; it is wrong for agents whose answer IS text (QA, math), so the library never imposes
    it."""
    if train_turn_fn is not None:
        entries = [entry for entry in entries if train_turn_fn(_entry_to_turn(entry))]
    turns = []
    for entry in entries:
        request = entry["request"]
        prompt_ids = tokenizer.apply_chat_template(
            request["messages"],
            tools=request.get("tools"),
            add_generation_prompt=True,
            tokenize=True,
            return_dict=False,
        )
        turns.append(TurnRecord(prompt_ids, _trace_output_ids(entry), entry.get("per_token_logps") or []))
    return turns


def _tool_call_counts_by_name(entries: list[TraceEntry]) -> dict[str, int]:
    """Per-tool-name call counts across the real agent turns (`entries`) - the single source of tool-call counting
    (the total is just `sum(...values())`). Generic (no tool is special-cased); a reward fn can read whichever it cares
    about, e.g. `counts.get("bash", 0)`."""
    counts: dict[str, int] = {}
    for entry in entries:
        for tc in (entry.get("response", {}).get("choices") or [{}])[0].get("message", {}).get("tool_calls") or []:
            name = (tc.get("function") or tc).get("name")
            if name:
                counts[name] = counts.get(name, 0) + 1
    return counts


def _tool_failure_count(entries: list[TraceEntry]) -> int:
    """Best-effort tool-failure count across the real agent turns (`entries`). The agent (not TRL) executed the tools,
    so we only see their free-text RESULTS, which come back as `role="tool"` messages in a later request; a failure can
    only be inferred from error-looking result text. The last agent request holds the fullest message list; dedupe by
    (name, content) so a result echoed across requests is not counted twice."""
    seen, failures = set(), 0
    for msg in (entries[-1]["request"] if entries else {}).get("messages") or []:
        if msg.get("role") != "tool":
            continue
        key = (msg.get("name"), str(msg.get("content"))[:200])
        if key in seen:
            continue
        seen.add(key)
        text = str(msg.get("content") or "").lower()
        if any(w in text for w in ("error", "failed", "traceback", "exception")):
            failures += 1
    return failures


def _messages_from_trace(entries: list[TraceEntry]) -> list[Message]:
    """Loop-owning path: the transcript is the last agent request's messages plus the final assistant reply. Uses the
    last REAL agent turn (`entries[-1]`, already selected by `agent_turn_fn`), not raw `trace[-1]` which can be an
    auxiliary title/summary call the framework fired last. Some proxy entries capture an empty `choices` list (upstream
    returned no completion), so fall back to empty content."""
    if not entries:
        return []
    last = entries[-1]
    choices = (last.get("response") or {}).get("choices") or []
    content = choices[0]["message"].get("content", "") if choices else ""
    return list(last["request"].get("messages") or []) + [{"role": "assistant", "content": content}]


class HarnessRolloutWorker(AsyncRolloutWorker):
    """AsyncGRPO rollout worker that drives an OpenEnv `ResourceSessionFactory`.

    Construct it with the usual `AsyncRolloutWorker` kwargs plus `harness_session_factory` (and optionally
    `harness_adapter`), then inject it via `AsyncGRPOTrainer(rollout_worker=...)`. Only the spawned child's loop class
    differs.
    """

    _loop_cls = _HarnessRolloutLoop
