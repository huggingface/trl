# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
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
import hashlib
import json
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from openenv.core.harness import HarnessRunLimits, ModelStepResult
from openenv.core.llm_client import LLMResponse, ToolCall

from ...chat_template_utils import parse_response
from .async_rollout_worker import (
    AsyncRolloutWorker,
    TurnRecord,
    _AsyncRolloutLoop,
    _chain_to_sequences,
    _scrub_child_env,
    _spawn_stop_watcher,
)


Message = dict[str, Any]


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

    def __init__(self, *, harness_session_factory, harness_adapter=None, **loop_kwargs):
        super().__init__(**loop_kwargs)  # ALL base setup, unchanged
        self._factory = harness_session_factory  # an OpenEnv ResourceSessionFactory
        # An OpenEnv HarnessAdapter (e.g. MCPHarnessAdapter) selects white-box: TRL samples each turn and the
        # adapter runs the tool loop. `None` selects loop-owning: the agent runs its own loop and we read its
        # proxy trace. No default coercion, so `None` genuinely reaches the loop-owning branch in `_run_session`.
        self._adapter = harness_adapter
        self._limits = HarnessRunLimits(
            max_turns=self.max_tool_calling_iterations if self.max_tool_calling_iterations is not None else 8,
            sampling={"temperature": self.temperature, "max_tokens": self.max_tokens},
        )
        # One bounded pool, sized to max_inflight (the base loop never has more than that many `_generate_one`
        # tasks outstanding), so sessions run concurrently with no queuing and no per-rollout thread churn.
        self._session_pool = ThreadPoolExecutor(
            max_workers=max(1, self.max_inflight_tasks), thread_name_prefix="harness-session"
        )
        self.reward_func_names.append("harness_reward")  # keep per-func metrics aligned with the reward column

    async def _generate_one(self, prompt, tool_dict, tools):
        # OpenEnv's harness layer is synchronous (SyncEnvClient facade), so run the whole session on the pool.
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._session_pool, self._run_session, prompt)

    def _run_session(self, prompt):
        """Drive one OpenEnv session to completion and return the base `_generate_one` tuple + the verify reward."""
        rollout_id = uuid.uuid4().hex
        # Stable per-group seed: all `num_generations` rollouts of one prompt share the same `prompt` object, so
        # factories that generate the task from a seed (e.g. reasoning_gym `env.reset(seed=...)`) hand the whole
        # group the SAME question -> a valid group-relative advantage. Factories that ignore `seed` (e.g. the calc
        # harness, whose task is fully in the prompt) are unaffected.
        seed = int(hashlib.sha256(repr(prompt).encode()).hexdigest(), 16) % (2**31)
        session = self._factory.create(prompt, seed=seed, episode_id=rollout_id)
        try:
            if self._adapter is not None:
                # white-box: OpenEnv's adapter runs the tool loop, calling `_sample_turn` (a ModelStep) each turn.
                # `turns` is bound per session (sessions run concurrently on the pool, so it can't live on self).
                turns: list[TurnRecord] = []
                result = self._adapter.run_white_box(
                    functools.partial(self._sample_turn, turns), session, self._limits
                )
                completion = result.messages
                tool_call_count = int(result.metrics.get("tool_calls", len(result.tool_trace)))
                # A tool call "failed" when its ToolResult carries an error (e.g. the calc tool rejecting a
                # malformed expression). run_white_box keeps each call's result in tool_trace.
                tool_failure_count = sum(1 for entry in result.tool_trace if entry.result.error is not None)
            else:
                # loop-owning: the agent hits vLLM through its own in-sandbox proxy; read the per-turn trace.
                # Tools run inside the agent's sandbox, not visible to TRL, so we don't count them here.
                session.wait_for_completion()
                trace = session.fetch_proxy_trace()
                turns = _turns_from_trace(trace, self.tokenizer)
                completion = _messages_from_trace(trace)
                # TODO(@aminediro): find a way to extract more info from the trace
                tool_call_count = tool_failure_count = 0
            verify = session.verify(completion)
            reward = float(verify.env_reward) if verify.env_reward is not None else None
            sequences = _chain_to_sequences(turns, rollout_id, self._fork_threshold_tokens)
            completion_ids = [tid for turn in turns for tid in turn.output_ids]
            return completion, completion_ids, sequences, tool_call_count, tool_failure_count, reward
        finally:
            session.close()

    def _sample_turn(self, turns: list[TurnRecord], messages, tools, sampling) -> ModelStepResult:
        """OpenEnv `ModelStep`: sample one assistant turn against the trainer's vLLM and record a `TurnRecord`
        into `turns` (the calling session's capture list, bound via `functools.partial`)."""
        prompt_ids = self.tokenizer.apply_chat_template(
            messages,
            tools=_tools_to_schema(tools),
            add_generation_prompt=True,
            tokenize=True,
            return_dict=False,
            chat_template=self.chat_template,
            **self.chat_template_kwargs,
        )
        # ModelStep is sync and runs on the pool thread; bridge the async vLLM POST onto the loop's event loop.
        turn_ids, logprobs = asyncio.run_coroutine_threadsafe(self._generate_one_turn(prompt_ids), self._loop).result()
        turns.append(TurnRecord(prompt_ids, turn_ids, logprobs))
        message = parse_response(self.tokenizer, turn_ids, prefix=prompt_ids)
        return ModelStepResult(
            response=_msg_to_llm_response(message), prompt_ids=prompt_ids, completion_ids=turn_ids, logprobs=logprobs
        )


def _trace_output_ids(entry: dict) -> list[int]:
    """The generated token ids for one proxy-trace turn.

    Prefer the proxy's `completion_token_ids` (populated when the upstream returns a `token_id` per logprob). If it is
    empty, recover them from `completion_tokens`, which vLLM renders as `"token_id:{id}"` strings when the server is
    launched with `--return-tokens-as-token-ids`. This keeps the exact generated ids (aligned to `per_token_logps`)
    without re-encoding the decoded text (which could split differently)."""
    ids = entry.get("completion_token_ids") or []
    if ids:
        return list(ids)
    return [
        int(t[len("token_id:") :]) for t in (entry.get("completion_tokens") or []) if str(t).startswith("token_id:")
    ]


def _turns_from_trace(trace: list[dict], tokenizer) -> list[TurnRecord]:
    """Loop-owning path: rebuild per-turn `TurnRecord`s from an in-sandbox proxy trace. Recover `prompt_ids` by
    re-tokenizing each captured request's messages (same as message-mode), passing the request's `tools` through so the
    reconstructed prompt matches what the upstream actually rendered (agents like opencode put the tool schema in the
    request, not the messages); the generated token ids + per-token logprobs come from the proxy capture (see
    `_trace_output_ids`)."""
    turns = []
    for entry in trace:
        request = entry["request"]
        prompt_ids = tokenizer.apply_chat_template(
            request["messages"],
            tools=request.get("tools"),
            add_generation_prompt=True,
            tokenize=True,
            return_dict=False,
        )
        turns.append(TurnRecord(prompt_ids, _trace_output_ids(entry), entry["per_token_logps"]))
    return turns


def _messages_from_trace(trace: list[dict]) -> list[Message]:
    """Loop-owning path: the transcript is the last request's messages plus the final assistant reply."""
    if not trace:
        return []
    last = trace[-1]
    content = last["response"]["choices"][0]["message"].get("content", "")
    return list(last["request"]["messages"]) + [{"role": "assistant", "content": content}]


def _harness_child_main(
    loop_kwargs,
    samples_queue,
    model_version_value,
    stop_event,
    child_ready_event,
    heartbeat_value,
    failed_event,
    exception_info_queue,
):
    """Child entrypoint: build a `_HarnessRolloutLoop` and run it. Near-verbatim copy of `_child_main` with the
    loop class swapped, so `async_rollout_worker.py` stays free of any OpenEnv reference."""
    _scrub_child_env()
    from accelerate.state import PartialState

    PartialState()

    rollout_loop = _HarnessRolloutLoop(
        **loop_kwargs,
        rollout_buffer=samples_queue,
        model_version_value=model_version_value,
        heartbeat_value=heartbeat_value,
        failed_event=failed_event,
        exception_info_queue=exception_info_queue,
    )
    child_ready_event.set()
    _spawn_stop_watcher(rollout_loop, stop_event)
    try:
        rollout_loop.run()
    except Exception:
        traceback.print_exc()
        raise


class HarnessRolloutWorker(AsyncRolloutWorker):
    """AsyncGRPO rollout worker that drives an OpenEnv `ResourceSessionFactory`.

    Construct it with the usual `AsyncRolloutWorker` kwargs plus `harness_session_factory` (and optionally
    `harness_adapter`), then inject it via `AsyncGRPOTrainer(rollout_worker=...)`. Inherits all lifecycle machinery;
    only the spawned child's loop class differs.
    """

    def _child_entry(self):
        return _harness_child_main
