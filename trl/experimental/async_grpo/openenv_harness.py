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
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, cast

from accelerate.logging import get_logger
from openenv.core.harness import (
    HarnessAdapter,
    HarnessRunLimits,
    LoopOwningSession,
    ModelStepResult,
    ResourceSessionFactory,
    TraceEntry,
)
from openenv.core.harness.capture.upstream import training_sampling
from openenv.core.harness.capture.validate import validate_training_turn
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


class CaptureContractError(ValueError):
    """Captured tokens, masks or sampling metadata do not satisfy the training contract."""


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

    # Unscorable rollouts are excluded from the group baseline.
    _EMPTY_ROLLOUT = ([], [], [], 0, 0, None)

    def __init__(
        self,
        *,
        harness_session_factory: ResourceSessionFactory,
        harness_adapter: HarnessAdapter | None = None,
        rollout_reward_fn: Callable[[HarnessRolloutOutcome], float | None] | None = None,
        train_turn_fn: Callable[[HarnessTurn], bool] | None = None,
        agent_turn_fn: Callable[[list[TraceEntry]], list[TraceEntry]] | None = None,
        lossless_capture: bool = True,
        **loop_kwargs,
    ):
        # Fork rewritten captures so reconciliation preserves eligible sampled tokens.
        if harness_adapter is None and lossless_capture:
            loop_kwargs["fork_threshold_tokens"] = 0
        super().__init__(**loop_kwargs)
        self._factory = harness_session_factory
        # An adapter lets TRL sample each turn; otherwise the agent owns the loop.
        self._adapter = harness_adapter
        self._sampling_policy = None
        if harness_adapter is None:
            self._sampling_policy = training_sampling(
                {
                    "temperature": self.temperature,
                    **{
                        key: value
                        for key, value in {
                            "top_p": self.top_p,
                            "top_k": self.top_k,
                            "min_p": self.min_p,
                            "repetition_penalty": self.repetition_penalty,
                        }.items()
                        if value is not None
                    },
                }
            )
        self._limits = HarnessRunLimits(
            max_turns=self.max_tool_calling_iterations if self.max_tool_calling_iterations is not None else 8,
            sampling={"temperature": self.temperature, "max_tokens": self.max_tokens},
        )
        self._rollout_reward_fn = rollout_reward_fn
        self._train_turn_fn = train_turn_fn
        self._agent_turn_fn = agent_turn_fn or _default_agent_entries
        self.reward_func_names.append("harness_reward")

        self._session_pool = ThreadPoolExecutor(
            max_workers=max(1, self.max_inflight_tasks), thread_name_prefix="harness-session"
        )
        self._live_sessions: set = set()

    async def _generate_one(self, prompt, tool_dict, tools, group_id=0):
        # OpenEnv sessions are synchronous; metrics stay on the event loop.
        loop = asyncio.get_running_loop()
        result, metrics = await loop.run_in_executor(self._session_pool, self._run_session, prompt, group_id)
        if metrics is not None:
            correctness = metrics.pop("correctness", None)
            if correctness is not None:
                self._rates["rollout/correctness_mean"][0] += correctness
                self._rates["rollout/correctness_mean"][1] += 1.0
            self._push_rollout_metrics(**metrics)
        return result

    async def _run_loops(self, stop_event) -> None:
        async def _close_live_sessions_on_stop() -> None:
            await stop_event.wait()

            def _close(session):
                try:
                    session.close()
                except Exception:
                    logger.warning("closing in-flight harness session on stop failed", exc_info=True)

            await asyncio.gather(*(asyncio.to_thread(_close, session) for session in list(self._live_sessions)))

        close_task = asyncio.create_task(_close_live_sessions_on_stop())
        try:
            await super()._run_loops(stop_event)
        finally:
            # Unblock other sessions before joining their threads, including on capture errors.
            stop_event.set()
            await close_task
            await asyncio.to_thread(self._session_pool.shutdown, wait=True)

    def _run_session(self, prompt, group_id=0):
        """Return a rollout and its metrics from a pool thread.

        Session failures are unscorable. Invalid training captures stop the worker.
        """
        t_dispatch = time.monotonic()
        rollout_id = uuid.uuid4().hex
        # Every generation in a group must select the same task.
        seed = group_id
        try:
            session = self._factory.create(prompt, seed=seed, episode_id=rollout_id)
        except Exception:
            logger.warning("harness session create failed; scoring rollout as unscorable", exc_info=True)
            return self._EMPTY_ROLLOUT, None
        self._live_sessions.add(session)
        timed_out = False
        trace: list[TraceEntry] = []
        tool_calls_by_name: dict[str, int] = {}
        try:
            if self._stop_event.is_set():
                return self._EMPTY_ROLLOUT, None
            if self._adapter is not None:
                turns: list[TurnRecord] = []
                result = self._adapter.run_white_box(
                    functools.partial(self._sample_turn, turns), session, self._limits
                )
                completion = result.messages
                tool_call_count = int(result.metrics.get("tool_calls", len(result.tool_trace)))
                tool_failure_count = sum(1 for entry in result.tool_trace if entry.result.error is not None)
            else:
                loop_session = cast(LoopOwningSession, session)
                try:
                    loop_session.wait_for_completion()
                except TimeoutError:
                    logger.warning("harness agent timed out; training captured turns, timed_out flagged")
                    timed_out = True
                try:
                    trace = loop_session.fetch_proxy_trace()
                    entries = self._agent_turn_fn(trace)
                    turns = _turns_from_trace(entries, self._train_turn_fn, sampling=self._sampling_policy)
                except (ValueError, TypeError, KeyError) as exc:
                    raise CaptureContractError(str(exc)) from exc
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
            try:
                sequences, tally = _chain_to_sequences(turns, rollout_id, self._fork_threshold_tokens)
            except (ValueError, TypeError, KeyError) as exc:
                raise CaptureContractError(str(exc)) from exc
            completion_ids = [tid for turn in turns for tid in turn.output_ids]
            metrics = dict(
                turns=len(turns),
                sequences=len(sequences),
                completion_ids=completion_ids,
                tally=tally,
                loop_exhausted=timed_out,
                duration_s=time.monotonic() - t_dispatch,
            )
            if self._rollout_reward_fn is not None and env_reward is not None:
                metrics["correctness"] = env_reward
            return (completion, completion_ids, sequences, tool_call_count, tool_failure_count, reward), metrics
        except CaptureContractError:
            raise
        except Exception:
            logger.warning("harness rollout failed; scoring as unscorable", exc_info=True)
            return self._EMPTY_ROLLOUT, None
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


def _default_agent_entries(trace: list[TraceEntry]) -> list[TraceEntry]:
    """Select captures with messages and a response. Producers or `agent_turn_fn` must exclude auxiliary calls."""
    return [entry for entry in trace if (entry.get("request") or {}).get("messages") and entry.get("response")]


def has_tool_call(turn: HarnessTurn) -> bool:
    """Select turns that emitted a tool call."""
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
    entries: list[TraceEntry],
    train_turn_fn: Callable[[HarnessTurn], bool] | None = None,
    *,
    sampling: dict[str, float | int] | None = None,
) -> list[TurnRecord]:
    """Convert captured engine tokens into TRL turns without re-tokenizing.

    OpenEnv's `loss_mask` covers prompt plus completion; `TurnRecord.output_mask` covers only completion. Partial
    completion masks are preserved. `train_turn_fn` optionally excludes whole turns and cannot replace token masks.
    """
    if train_turn_fn is not None:
        entries = [entry for entry in entries if train_turn_fn(_entry_to_turn(entry))]
    turns = []
    for entry in entries:
        if sampling is not None:
            captured = (entry.get("metadata") or {}).get("sampling_params") or {}
            if any(captured.get(key) != value for key, value in sampling.items()):
                raise ValueError(
                    "capture sampling does not match the trainer policy; pass the trainer's temperature "
                    "as sampling to HarborSessionFactory and use full-vocabulary sampling"
                )
        prompt_ids = entry.get("prompt_token_ids")
        if not prompt_ids:
            raise ValueError(
                "a captured turn carried no `prompt_token_ids`. Serve the engine with "
                "`--return-tokens-as-token-ids --logprobs-mode processed_logprobs`."
            )
        output_ids = _trace_output_ids(entry)
        # Convert the producer's full-sequence mask at the OpenEnv boundary.
        mask = entry.get("loss_mask")
        if mask is None:
            mask = [0] * len(prompt_ids) + [1] * len(output_ids)
        validate_training_turn(prompt_ids, output_ids, entry.get("per_token_logps") or [], mask)
        output_mask = list(mask[len(prompt_ids) :])
        turns.append(TurnRecord(list(prompt_ids), output_ids, entry.get("per_token_logps") or [], output_mask))
    return turns


def _tool_call_counts_by_name(entries: list[TraceEntry]) -> dict[str, int]:
    """Count tool calls by name across the selected agent turns."""
    counts: dict[str, int] = {}
    for entry in entries:
        for tc in (entry.get("response", {}).get("choices") or [{}])[0].get("message", {}).get("tool_calls") or []:
            name = (tc.get("function") or tc).get("name")
            if name:
                counts[name] = counts.get(name, 0) + 1
    return counts


def _tool_failure_count(entries: list[TraceEntry]) -> int:
    """Estimate failures from the last request's tool-result text, deduplicated by name and content."""
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
    """Return the last selected request's messages followed by its assistant reply."""
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
