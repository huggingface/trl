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
import itertools
import json
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from accelerate.logging import get_logger
from datasets import Dataset, IterableDataset
from openenv.core.harness import (
    HarnessAdapter,
    HarnessRunLimits,
    ModelStepResult,
    ResourceSessionFactory,
    TraceEntry,
    TrainableSession,
    TrainingTrace,
)
from openenv.core.harness.capture.upstream import training_sampling
from openenv.core.llm_client import LLMResponse, ToolCall
from transformers import PreTrainedModel, PreTrainedTokenizerBase, TrainerCallback

from ...chat_template_utils import parse_response
from .async_grpo_config import AsyncGRPOConfig
from .async_grpo_trainer import AsyncGRPOTrainer
from .async_rollout_worker import AsyncRolloutLoop, Rollout, TurnRecord


if TYPE_CHECKING:
    from peft import PeftConfig


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


class _HarnessRolloutLoop(AsyncRolloutLoop):
    """`AsyncRolloutLoop` whose `generate_one` drives an OpenEnv session instead of TRL's own turn loop."""

    # Unscorable rollouts are excluded from the group baseline.
    _EMPTY_ROLLOUT = Rollout([], [], [])

    def __init__(
        self,
        *,
        harness_session_factory: Callable[..., ResourceSessionFactory],
        harness_adapter: HarnessAdapter | None = None,
        rollout_reward_fn: Callable[[HarnessRolloutOutcome], float | None] | None = None,
        max_session_failures_per_slot: int = 10,
        max_session_backoff_s: float = 16,
        **loop_kwargs,
    ):
        super().__init__(**loop_kwargs)
        # An adapter lets TRL sample each turn; otherwise the agent owns the loop.
        self._adapter = harness_adapter
        if harness_adapter is None and (self.config.top_p < 1.0 or self.config.top_k > 0):
            raise ValueError(
                "`top_p < 1.0` or `top_k > 0` needs the candidate set of every sampled token, which an agent-owned "
                "loop does not record. Pass a `harness_adapter` so TRL samples each turn, or sample the full vocabulary."
            )
        if harness_adapter is None:
            sampling_kwargs = {
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "top_k": self.config.top_k,
                "min_p": self.config.min_p,
                # Explicit: vLLM fills an omitted penalty from the model's generation config (1.05 for
                # Qwen2.5-Instruct), and the trainer's log-probs carry none
                "repetition_penalty": 1.0,
            }
            sampling_kwargs = {key: value for key, value in sampling_kwargs.items() if value is not None}
            self._factory = harness_session_factory(sampling=training_sampling(sampling_kwargs))
        else:
            self._factory = harness_session_factory()
        self._limits = HarnessRunLimits(
            max_turns=self.config.max_tool_calling_iterations
            if self.config.max_tool_calling_iterations is not None
            else 8,
            sampling={"temperature": self.config.temperature, "max_tokens": self.config.max_completion_length},
        )
        self._rollout_reward_fn = rollout_reward_fn

        self._session_pool = ThreadPoolExecutor(
            max_workers=max(1, self.config.max_concurrent_rollouts), thread_name_prefix="harness-session"
        )
        self._live_sessions: set = set()
        self._max_session_failures_per_slot = max_session_failures_per_slot
        self._max_session_backoff_s = max_session_backoff_s
        self._consecutive_session_failures = 0

    async def generate_one(self, prompt, tools, group_id=0):
        # OpenEnv sessions are synchronous; metrics stay on the event loop.
        loop = asyncio.get_running_loop()
        result, metrics = await loop.run_in_executor(self._session_pool, self._run_session, prompt, group_id)
        if metrics is not None:
            correctness = metrics.pop("correctness", None)
            if correctness is not None:
                self._rates["rollout/correctness_mean"][0] += correctness
                self._rates["rollout/correctness_mean"][1] += 1.0
            for name, count in metrics.pop("tool_calls_by_name").items():
                self._counters[f"tools/{name}/call_total"] += count
            for name, count in metrics.pop("tool_failures_by_name").items():
                self._counters[f"tools/{name}/failure_total"] += count
            for name, elapsed in metrics.pop("tool_latency"):
                self._rates["tools/latency_s"][0] += elapsed
                self._rates["tools/latency_s"][1] += 1
                self._rates[f"tools/{name}/latency_s"][0] += elapsed
                self._rates[f"tools/{name}/latency_s"][1] += 1
        if result is not self._EMPTY_ROLLOUT:
            self._consecutive_session_failures = 0
        elif not self._stop_event.is_set():
            # A persistent failure (harness server down, broken sandbox) would otherwise spin without producing samples
            self._consecutive_session_failures += 1
            if (
                self._consecutive_session_failures
                >= self._max_session_failures_per_slot * self.config.max_concurrent_rollouts
            ):
                raise RuntimeError(
                    f"{self._consecutive_session_failures} harness sessions in a row were unscorable; see the "
                    "warnings above for their errors."
                )
            backoff = min(2 ** (self._consecutive_session_failures - 1), self._max_session_backoff_s)
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=backoff)
            except TimeoutError:
                pass
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

        Session failures and rejected captures of failed agents are unscorable. Other invalid training captures stop
        the worker.
        """
        rollout_id = uuid.uuid4().hex
        # Every generation in a group must select the same task.
        seed = group_id
        if self.lora_name is not None:
            # The agent sends the factory's model name on every call, and in vLLM's API an adapter is a model name:
            # point each new session at the policy version published now.
            self._factory.model = self._request_model
        try:
            session = self._factory.create(prompt, seed=seed, episode_id=rollout_id)
        except Exception:
            logger.warning("harness session create failed; scoring rollout as unscorable", exc_info=True)
            return self._EMPTY_ROLLOUT, None
        self._live_sessions.add(session)
        timed_out = False
        trace: list[TraceEntry] = []
        tool_calls_by_name: dict[str, int] = {}
        tool_failures_by_name: dict[str, int] = {}
        tool_latency: list[tuple[str, float]] = []
        try:
            if self._stop_event.is_set():
                return self._EMPTY_ROLLOUT, None
            if self._adapter is not None:
                turns: list[TurnRecord] = []
                turn_times: list[tuple[float, float, list[str]]] = []
                result = self._adapter.run_white_box(
                    functools.partial(self._sample_turn, turns, turn_times), session, self._limits
                )
                completion = result.messages
                tool_call_count = int(result.metrics.get("tool_calls", len(result.tool_trace)))
                for entry in result.tool_trace:
                    tool_calls_by_name[entry.tool_name] = tool_calls_by_name.get(entry.tool_name, 0) + 1
                    if entry.result.error is not None:
                        tool_failures_by_name[entry.tool_name] = tool_failures_by_name.get(entry.tool_name, 0) + 1
                # A turn's tool calls run between its reply and the next request, so that gap (harness overhead
                # included) is their latency. The trailing turn has no next request and is not timed.
                tool_latency = [
                    (name, (t_request - t_reply) / len(names))
                    for (_, t_reply, names), (t_request, _, _) in itertools.pairwise(turn_times)
                    for name in names
                ]
            else:
                if not isinstance(session, TrainableSession):
                    raise CaptureContractError("loop-owning training requires OpenEnv fetch_training_trace()")
                loop_session = cast(TrainableSession, session)
                exit_code = 0
                try:
                    exit_code = loop_session.wait_for_completion()
                except TimeoutError:
                    logger.warning("harness agent timed out; training captured turns, timed_out flagged")
                    timed_out = True
                try:
                    captured = loop_session.fetch_training_trace()
                    turns = _turns_from_training_trace(captured)
                    entries = captured.to_trace_entries()
                    trace = entries
                except (ValueError, TypeError, KeyError) as exc:
                    # A session that exited non-zero already reported a failed rollout, whose capture may be
                    # untrainable (e.g. fatal validation findings). Drop it like a transport failure; a malformed
                    # capture from a successful rollout still stops the worker.
                    if exit_code != 0:
                        logger.warning("capture of failed rollout (exit code %s) rejected: %s", exit_code, exc)
                        return self._EMPTY_ROLLOUT, None
                    raise CaptureContractError(str(exc)) from exc
                completion = _messages_from_trace(entries)
                tool_calls_by_name = _tool_call_counts_by_name(entries)
                tool_call_count = sum(tool_calls_by_name.values())
                tool_failures_by_name = _tool_failures_by_name(entries)
            tool_failure_count = sum(tool_failures_by_name.values())
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
            completion_ids = [tid for turn in turns for tid in turn.output_ids]
            metrics = dict(
                tool_calls_by_name=tool_calls_by_name,
                tool_failures_by_name=tool_failures_by_name,
                tool_latency=tool_latency,
            )
            if self._rollout_reward_fn is not None and env_reward is not None:
                metrics["correctness"] = env_reward
            return (
                Rollout(completion, completion_ids, turns, tool_call_count, tool_failure_count, reward, rollout_id),
                metrics,
            )
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

    def _sample_turn(
        self, turns: list[TurnRecord], turn_times: list[tuple[float, float, list[str]]], messages, tools, sampling
    ) -> ModelStepResult:
        """OpenEnv `ModelStep`: sample one assistant turn against vLLM and record a `TurnRecord` into `turns`.

        `turn_times` gets `(request time, reply time, requested tool names)` per turn, for the tool latency metrics.
        """
        t_request = time.monotonic()
        tools = _tools_to_schema(tools)
        prompt_ids = self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=False,
            chat_template=self.chat_template,
            **self.chat_template_kwargs,
        )
        # ModelStep is sync on a pool thread; bridge the async vLLM POST onto the loop's event loop.
        turn = asyncio.run_coroutine_threadsafe(self.generate_one_turn(prompt_ids), self._loop).result()
        turns.append(turn)
        response = _msg_to_llm_response(
            parse_response(self.tokenizer, turn.output_ids, prefix=prompt_ids, tools=tools)
        )
        turn_times.append((t_request, time.monotonic(), [tc.name for tc in response.tool_calls]))
        return ModelStepResult(
            response=response,
            prompt_ids=prompt_ids,
            completion_ids=turn.output_ids,
            logprobs=turn.output_log_probs,
        )


def _turns_from_training_trace(trace: TrainingTrace) -> list[TurnRecord]:
    """Map OpenEnv's validated full masks to TRL's completion masks."""
    return [
        TurnRecord(
            list(turn.prompt_token_ids),
            list(turn.completion_token_ids),
            list(turn.per_token_logps),
            list(turn.loss_mask[len(turn.prompt_token_ids) :]),
        )
        for turn in trace.turns
    ]


def _tool_call_counts_by_name(entries: list[TraceEntry]) -> dict[str, int]:
    """Count tool calls by name across the selected agent turns."""
    counts: dict[str, int] = {}
    for entry in entries:
        for tc in (entry.get("response", {}).get("choices") or [{}])[0].get("message", {}).get("tool_calls") or []:
            name = (tc.get("function") or tc).get("name")
            if name:
                counts[name] = counts.get(name, 0) + 1
    return counts


def _tool_failures_by_name(entries: list[TraceEntry]) -> dict[str, int]:
    """Estimate failures per tool from the last request's tool-result text, deduplicated by name and content."""
    messages = (entries[-1]["request"] if entries else {}).get("messages") or []
    # A tool result names its call id, not its tool; the assistant message that issued the call has the name.
    names = {
        tc.get("id"): (tc.get("function") or tc).get("name") for m in messages for tc in m.get("tool_calls") or []
    }
    seen, failures = set(), {}
    for msg in messages:
        if msg.get("role") != "tool":
            continue
        name = msg.get("name") or names.get(msg.get("tool_call_id"))
        key = (name, str(msg.get("content"))[:200])
        if key in seen:
            continue
        seen.add(key)
        text = str(msg.get("content") or "").lower()
        if any(w in text for w in ("error", "failed", "traceback", "exception")):
            failures[name] = failures.get(name, 0) + 1
    return failures


def _messages_from_trace(entries: list[TraceEntry]) -> list[Message]:
    """Return the last selected request's messages followed by its assistant reply."""
    if not entries:
        return []
    last = entries[-1]
    choices = (last.get("response") or {}).get("choices") or []
    content = choices[0]["message"].get("content", "") if choices else ""
    return list(last["request"].get("messages") or []) + [{"role": "assistant", "content": content}]


class HarnessAsyncGRPOTrainer(AsyncGRPOTrainer):
    """
    [`experimental.async_grpo.AsyncGRPOTrainer`] whose rollouts are OpenEnv harness sessions, for example an agent
    served by Harbor that runs its own loop.

    Each rollout creates a session from the factory and lets it run to completion. Without `harness_adapter`, the agent
    owns its loop and TRL trains on the `TrainingTrace` the session captured; with one, TRL samples each turn itself.
    The reward is the session's verifier reward, or `rollout_reward_fn` applied to the outcome. A failed session is
    unscorable, and the next one waits first, doubling the wait up to 16 seconds. After 10 × `max_concurrent_rollouts`
    unscorable sessions in a row, across all rollout slots, training stops with an error.

    Args:
        model (`str` or [`~transformers.PreTrainedModel`]):
            Model to be trained, as in [`experimental.async_grpo.AsyncGRPOTrainer`].
        harness_session_factory (`Callable[..., ResourceSessionFactory]`):
            Callable returning an OpenEnv `ResourceSessionFactory`, for example
            `functools.partial(HarborSessionFactory, server_url, ...)`. Without `harness_adapter`, it is called with
            `sampling=` built from the config's sampling parameters, so the agent samples with the training policy.
            With `harness_adapter`, it is called without arguments. It is called in the rollout process, so it must be
            picklable: a module-level function, a `functools.partial` or a callable class instance, not a lambda.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset of prompts. Each prompt is passed to the factory's `create` to start a session on its task.
        args ([`experimental.async_grpo.AsyncGRPOConfig`], *optional*):
            Configuration for this trainer. If `None`, a default configuration is used.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*):
            Processing class used to process the data, as in [`experimental.async_grpo.AsyncGRPOTrainer`].
        harness_adapter (`HarnessAdapter`, *optional*):
            Adapter that lets TRL sample each turn of the session. If `None`, the agent owns its loop and TRL trains on
            what the session captured, which requires sampling the full vocabulary (`top_p=1.0`, `top_k=0`). Used in
            the rollout process, so it must be picklable.
        rollout_reward_fn (`Callable[[HarnessRolloutOutcome], float | None]`, *optional*):
            Turns a finished session into a reward. Return `None` to mark the rollout unscorable. If `None`, the
            session's verifier reward is used. Called in the rollout process, so it must be picklable, like
            `harness_session_factory`.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*):
            List of callbacks to customize the training loop.
        optimizers (`tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use.
        peft_config ([`~peft.PeftConfig`], *optional*):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    loop_cls = _HarnessRolloutLoop

    def __init__(
        self,
        model: str | PreTrainedModel,
        harness_session_factory: Callable[..., ResourceSessionFactory],
        train_dataset: Dataset | IterableDataset,
        args: AsyncGRPOConfig | None = None,
        processing_class: PreTrainedTokenizerBase | None = None,
        *,
        harness_adapter: HarnessAdapter | None = None,
        rollout_reward_fn: Callable[[HarnessRolloutOutcome], float | None] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple = (None, None),
        peft_config: PeftConfig | None = None,
    ):
        self._rollout_loop_kwargs = {
            "harness_session_factory": harness_session_factory,
            "harness_adapter": harness_adapter,
            "rollout_reward_fn": rollout_reward_fn,
        }
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
        )
