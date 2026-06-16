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

import asyncio
import inspect
import multiprocessing as mp
import os
import pickle
import queue
import threading
import time
import traceback
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass
from multiprocessing.queues import Queue as MPQueue
from multiprocessing.sharedctypes import Synchronized as MPValue
from multiprocessing.synchronize import Event as MPEvent
from typing import Any, TypeAlias

import aiohttp
import numpy as np
from accelerate.logging import get_logger
from datasets import Dataset
from transformers import PreTrainedTokenizerBase

from trl.chat_template_utils import (
    add_response_schema,
    get_training_chat_template,
    is_chat_template_prefix_preserving,
    parse_response,
)
from trl.import_utils import is_vllm_available
from trl.trainer.utils import print_prompt_completions_sample


logger = get_logger(__name__)

Messages: TypeAlias = list[dict[str, str]]

_RETRYABLE_HTTP_ERRORS = (aiohttp.ClientError, asyncio.TimeoutError, TimeoutError, ConnectionResetError)


async def _retry_on_http_error(coro_factory: Callable[[], Awaitable], *, label: str, max_attempts: int = 1):
    """Retry an aiohttp coroutine on transport errors with bounded exponential backoff."""
    for attempt in range(max_attempts):
        try:
            return await coro_factory()
        except _RETRYABLE_HTTP_ERRORS as e:
            if attempt >= max_attempts - 1:
                raise
            sleep = min(2 ** min(attempt, 4), 16)
            logger.warning(f"{label} failed ({type(e).__name__}: {e}); retry {attempt + 1}/{max_attempts} in {sleep}s")
            await asyncio.sleep(sleep)


@dataclass(slots=True)
class RolloutGroup:
    prompt: Messages
    prompt_ids: list[int]
    reward_kwargs: dict[str, list[Any]]
    completions: list[Messages]
    completions_ids: list[list[int]]
    completions_logprobs: list[list[float]]
    tool_mask: list[list[int]]
    tool_call_counts: list[int]
    tool_failure_counts: list[int]
    model_version: int
    queued_at: float = 0.0


@dataclass(slots=True)
class RolloutSample:
    prompt: Messages
    completion: Messages
    input_ids: list[int]
    completion_mask: list[int]
    old_log_probs: list[float]
    advantage: float
    model_version: int
    metrics: dict[str, float]


# Env vars the child must drop so accelerate's `PartialState()` initialises in
# single-process mode instead of trying to join the parent's process group.
_CHILD_ENV_TO_STRIP = (
    "RANK",
    "WORLD_SIZE",
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "GROUP_RANK",
    "ROLE_RANK",
    "ROLE_WORLD_SIZE",
    "TORCHELASTIC_RUN_ID",
    "TORCHELASTIC_RESTART_COUNT",
    "TORCHELASTIC_MAX_RESTARTS",
    "TORCH_FR_DUMP_TEMP_FILE",
    "NCCL_DEBUG_FILE",
)


def _scrub_child_env() -> None:
    # The child has no business touching CUDA; any library that imports torch
    # and lazily probes devices would race the parent's allocator.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for k in _CHILD_ENV_TO_STRIP:
        os.environ.pop(k, None)


def _spawn_stop_watcher(rollout_loop: "_AsyncRolloutLoop", stop_event: MPEvent) -> None:
    # Daemon thread that translates the parent's mp.Event into the child's
    # asyncio.Event so _run_loops breaks out of its gather.
    def _watch():
        stop_event.wait()
        try:
            rollout_loop._loop.call_soon_threadsafe(rollout_loop._stop_event.set)
        except RuntimeError:
            # Loop already closed (run() returned before stop fired). Nothing to do.
            pass

    threading.Thread(target=_watch, daemon=True, name="grpo-mp-stop-watcher").start()


def _child_main(
    loop_kwargs: dict[str, Any],
    samples_queue: MPQueue,
    model_version_value: MPValue,
    stop_event: MPEvent,
    child_ready_event: MPEvent,
    heartbeat_value: MPValue,
    failed_event: MPEvent,
    exception_info_queue: MPQueue,
) -> None:
    _scrub_child_env()
    # `accelerate.logging.get_logger` requires `PartialState()` to have been called.
    from accelerate.state import PartialState

    PartialState()

    rollout_loop = _AsyncRolloutLoop(
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


class _AsyncRolloutLoop:
    """Asyncio generate and score loops. Lives entirely inside the spawned child process.

    Owns the tokenizer, dataset iterator, reward funcs, environments, and the asyncio event loop. Talks to vLLM via
    `/v1/completions`. Pushes scored `RolloutSample`s into the shared `mp.Queue` (`rollout_buffer`); reads the bumped
    policy version from the shared `mp.Value` (`model_version_value`).
    """

    def __init__(
        self,
        *,
        model_name: str,
        dataset: Dataset,
        reward_funcs: list[Callable[..., list[float]]],
        processing_class: PreTrainedTokenizerBase,
        rollout_buffer: MPQueue,
        model_version_value: MPValue,
        heartbeat_value: MPValue,
        failed_event: MPEvent,
        exception_info_queue: MPQueue,
        tools: list[Callable] | None = None,
        environment_factory: Callable[[], object] | None = None,
        num_generations: int = 8,
        max_inflight_tasks: int = 128,
        queue_maxsize: int = 0,
        vllm_server_url: str = "http://localhost:8000",
        max_tokens: int = 32,
        temperature: float = 1.0,
        request_timeout: int = 120,
        chat_template_kwargs: dict[str, Any] | None = None,
        max_tool_calling_iterations: int | None = None,
        log_completions: bool = False,
        num_completions_to_print: int = 3,
    ):
        self.model_name = model_name
        self.dataset = dataset
        self._dataset_iter = iter(dataset)
        self.reward_funcs = reward_funcs
        self.reward_func_names = [f.__name__ for f in reward_funcs]
        self.tokenizer = add_response_schema(processing_class)
        self.rollout_buffer = rollout_buffer  # shared mp.Queue
        self._model_version_value = model_version_value  # shared mp.Value
        self._heartbeat_value = heartbeat_value  # shared mp.Value('d'); wall-clock seconds
        self._failed_event = failed_event  # shared mp.Event
        self._exception_info_queue = exception_info_queue  # shared mp.Queue(maxsize=1)

        self.num_generations = num_generations
        self.max_inflight_tasks = max_inflight_tasks
        self.queue_maxsize = queue_maxsize
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.request_timeout = request_timeout
        self.chat_template_kwargs = chat_template_kwargs or {}
        self.max_tool_calling_iterations = max_tool_calling_iterations
        self.log_completions = log_completions
        self.num_completions_to_print = num_completions_to_print
        self.vllm_server_url = vllm_server_url.rstrip("/")

        self.environments = None
        environment_methods = [[] for _ in range(max_inflight_tasks)]
        if environment_factory is not None:
            self.environments = [environment_factory() for _ in range(max_inflight_tasks)]
            for i, environment in enumerate(self.environments):
                has_reset = False
                for name, member in inspect.getmembers(environment, predicate=inspect.ismethod):
                    if name == "reset":
                        has_reset = True
                    elif not name.startswith("_"):
                        environment_methods[i].append(member)
                if not has_reset:
                    raise ValueError(
                        "Each environment instance returned by `environment_factory` must define `reset`."
                    )

        base_tools = tools or []
        self._sync_tool_dicts = [{} for _ in range(max_inflight_tasks)]
        for i in range(max_inflight_tasks):
            for tool in base_tools + (environment_methods[i] if self.environments is not None else []):
                if inspect.iscoroutinefunction(tool):
                    raise ValueError("Asynchronous tools are not supported yet.")
                self._sync_tool_dicts[i][tool.__name__] = tool
        self.tools = base_tools + (environment_methods[0] if self.environments is not None else [])

        # The chat template must be prefix-preserving in multi-turn training; if the tokenizer's
        # template isn't, swap in a training-safe one.
        if self.tools and not is_chat_template_prefix_preserving(self.tokenizer):
            self.chat_template = get_training_chat_template(self.tokenizer)
        else:
            self.chat_template = None

        self._groups_to_score: asyncio.Queue[RolloutGroup | None] = asyncio.Queue(maxsize=16)
        self._total_completion_tokens = 0
        self._total_groups_scored = 0
        self._generation_start_time: float | None = None
        self.session: aiohttp.ClientSession | None = None

        self._loop = asyncio.new_event_loop()
        self._stop_event = asyncio.Event()

    @property
    def model_version(self) -> int:
        return int(self._model_version_value.value)

    def run(self) -> None:
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._run_loops(stop_event=self._stop_event))
        except BaseException as e:
            # Push pickle-safe exception info to the parent before setting _failed_event, so a
            # reader that sees the event is guaranteed to also see the info on the queue.
            info = (type(e).__name__, str(e), traceback.format_exc())
            try:
                self._exception_info_queue.put_nowait(info)
            except Exception:
                pass  # queue full (parent hasn't drained a prior failure), best-effort put
            self._failed_event.set()
            logger.exception(f"Worker process failed: {e}")
            raise
        finally:
            self._loop.close()

    async def _run_loops(self, stop_event: asyncio.Event) -> None:
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=self.max_inflight_tasks)) as session:
            self.session = session
            logger.info(
                f"vllm worker started: num_generations={self.num_generations}, "
                f"max_inflight_tasks={self.max_inflight_tasks}"
            )
            await asyncio.gather(
                asyncio.create_task(self._generate_loop(stop_event=stop_event)),
                asyncio.create_task(self._score_loop(stop_event=stop_event)),
            )

    async def _generate_loop(self, stop_event: asyncio.Event) -> None:
        pending_groups: dict[int, RolloutGroup] = {}
        pending_completed: dict[int, int] = {}
        inflight_tasks: dict[asyncio.Task, tuple[int, int]] = {}
        free_slots = set(range(self.max_inflight_tasks))
        work_iter = self._repeat_iterator()

        self._generation_start_time = time.monotonic()
        try:
            while True:
                # Wall-clock for cross-process comparison; parent uses time.time() in check_health.
                self._heartbeat_value.value = time.time()
                while free_slots and not stop_event.is_set():
                    group_id, row = next(work_iter)
                    if group_id not in pending_groups:
                        prompt = row["prompt"]
                        prompt_ids = self.tokenizer.apply_chat_template(
                            prompt,
                            return_dict=False,
                            add_generation_prompt=True,
                            tools=self.tools or None,  # `or None`: Llama bug: renders tool boilerplate for tools=[]
                            chat_template=self.chat_template,
                            **self.chat_template_kwargs,
                        )
                        reward_kwargs = {
                            key: [row[key]] * self.num_generations
                            for key in row
                            if key not in {"prompt", "completion", "completion_ids"}
                        }
                        pending_groups[group_id] = RolloutGroup(
                            prompt=prompt,
                            prompt_ids=prompt_ids,
                            reward_kwargs=reward_kwargs,
                            completions=[],
                            completions_ids=[],
                            completions_logprobs=[],
                            tool_mask=[],
                            tool_call_counts=[],
                            tool_failure_counts=[],
                            model_version=self.model_version,
                        )
                        pending_completed[group_id] = 0

                    slot = free_slots.pop()
                    if self.environments is not None:
                        self.environments[slot].reset(**row)

                    task = asyncio.create_task(
                        self._generate_one(pending_groups[group_id].prompt, tool_dict=self._sync_tool_dicts[slot])
                    )
                    inflight_tasks[task] = (group_id, slot)

                if not inflight_tasks:
                    if stop_event.is_set():
                        return
                    await asyncio.sleep(0.01)
                    continue

                done, _ = await asyncio.wait(inflight_tasks, return_when=asyncio.FIRST_COMPLETED, timeout=0.1)
                if not done:
                    continue

                for task in done:
                    group_id, slot = inflight_tasks.pop(task)
                    free_slots.add(slot)
                    if task.exception() is not None:
                        raise task.exception()

                    (
                        completion,
                        completion_ids,
                        completion_logprobs,
                        tool_mask,
                        tool_call_count,
                        tool_failure_count,
                    ) = task.result()
                    group = pending_groups[group_id]
                    group.completions.append(completion)
                    group.completions_ids.append(completion_ids)
                    group.completions_logprobs.append(completion_logprobs)
                    group.tool_mask.append(tool_mask)
                    group.tool_call_counts.append(tool_call_count)
                    group.tool_failure_counts.append(tool_failure_count)
                    self._total_completion_tokens += sum(tool_mask)
                    pending_completed[group_id] += 1

                    if pending_completed[group_id] == self.num_generations:
                        group.queued_at = time.monotonic()
                        while True:
                            try:
                                self._groups_to_score.put_nowait(group)
                                break
                            except asyncio.QueueFull:
                                if stop_event.is_set():
                                    return
                                await asyncio.sleep(0.1)
                        del pending_groups[group_id]
                        del pending_completed[group_id]
        finally:
            for task in inflight_tasks:
                task.cancel()
            if inflight_tasks:
                await asyncio.gather(*inflight_tasks, return_exceptions=True)
            try:
                self._groups_to_score.put_nowait(None)
            except asyncio.QueueFull:
                pass

    async def _score_loop(self, stop_event: asyncio.Event) -> None:
        while not stop_event.is_set():
            self._heartbeat_value.value = time.time()
            t_wait = time.monotonic()
            try:
                group = await asyncio.wait_for(self._groups_to_score.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            if group is None:
                return
            score_queue_wait = time.monotonic() - t_wait
            wait_scoring = time.monotonic() - group.queued_at

            if score_queue_wait > 0.5:
                logger.info(f"[score] waited {score_queue_wait:.1f}s for a group to score")

            t0 = time.monotonic()
            samples = await self._score_group(group)
            scoring_time = time.monotonic() - t0
            logger.info(
                f"[score] scored {len(samples)} samples in {scoring_time:.2f}s, "
                f"buffer_qsize={self.rollout_buffer.qsize()}"
            )

            self._compute_rollout_metrics(samples, scoring_time, wait_scoring)

            if self.log_completions and samples:
                print_prompt_completions_sample(
                    prompts=[s.prompt for s in samples],
                    completions=[s.completion for s in samples],
                    rewards={"reward": [s.metrics["reward"] for s in samples]},
                    advantages=[s.advantage for s in samples],
                    step=self._total_groups_scored,
                    num_samples=self.num_completions_to_print,
                )
            self._total_groups_scored += 1

            for sample in samples:
                while True:
                    try:
                        self.rollout_buffer.put_nowait(sample)
                        break
                    except queue.Full:
                        if stop_event.is_set():
                            return
                        logger.info(
                            f"[score] rollout buffer full (maxsize={self.queue_maxsize}), "
                            "waiting for trainer to consume..."
                        )
                        await asyncio.sleep(0.1)

    def _compute_rollout_metrics(self, samples: list[RolloutSample], scoring_time: float, wait_scoring: float) -> None:
        assert self._generation_start_time is not None
        elapsed = time.monotonic() - self._generation_start_time
        generation_tok_per_sec = self._total_completion_tokens / elapsed if elapsed > 0 else 0.0
        for sample in samples:
            sample.metrics["generation_tok_per_s"] = generation_tok_per_sec
            sample.metrics["scoring_time_ms"] = scoring_time * 1000
            sample.metrics["wait_scoring_ms"] = wait_scoring * 1000
            sample.metrics["buffer_qsize"] = self.rollout_buffer.qsize()

    def _repeat_iterator(self) -> Iterator[tuple[int, dict[str, Any]]]:
        group_id = 0
        while True:
            try:
                row = next(self._dataset_iter)
            except StopIteration:
                self._dataset_iter = iter(self.dataset)
                row = next(self._dataset_iter)
            for _ in range(self.num_generations):
                yield group_id, row
            group_id += 1

    async def _generate_one(
        self, prompt: Messages, tool_dict: dict[str, Callable]
    ) -> tuple[list[dict[str, str]], list[int], list[float], list[int], int, int]:
        completion, completion_ids, completion_logprobs, tool_mask = [], [], [], []
        tool_call_count = 0
        tool_failure_count = 0
        iteration_num = 0
        max_iterations = self.max_tool_calling_iterations
        prompt_ids = self.tokenizer.apply_chat_template(
            prompt,
            return_dict=False,
            add_generation_prompt=True,
            tools=self.tools or None,
            chat_template=self.chat_template,
            **self.chat_template_kwargs,
        )
        while True:
            turn_ids, turn_logprobs = await self._generate_one_turn(prompt_ids)
            assistant_message = parse_response(self.tokenizer, turn_ids)
            completion.append(assistant_message)
            completion_ids.extend(turn_ids)
            completion_logprobs.extend(turn_logprobs)
            tool_mask.extend([1] * len(turn_ids))
            tool_calls = assistant_message.get("tool_calls")
            if tool_calls is None or (max_iterations is not None and iteration_num >= max_iterations):
                return completion, completion_ids, completion_logprobs, tool_mask, tool_call_count, tool_failure_count

            tool_messages, n_calls, n_failures = self._execute_tool_calls(tool_calls, tool_dict)
            tool_call_count += n_calls
            tool_failure_count += n_failures
            completion.extend(tool_messages)
            suffix_ids = self._get_tool_suffix_ids(tool_messages)
            completion_ids.extend(suffix_ids)
            completion_logprobs.extend([0.0] * len(suffix_ids))
            tool_mask.extend([0] * len(suffix_ids))
            prompt_ids = prompt_ids + turn_ids + suffix_ids
            iteration_num += 1

    def _get_tool_suffix_ids(self, tool_messages: list[dict[str, Any]]) -> list[int]:
        # Use the real tool name: some templates (e.g. GPT-OSS) derive the tool response header from
        # the assistant's tool call name.
        dummy_tool_calls = [{"type": "function", "function": {"name": tool_messages[0]["name"], "arguments": {}}}]
        dummy_messages = [
            {"role": "user", "content": "dummy"},
            # `content: ""` is required: VLM processors crash on tokenize=True without it
            # (KeyError in processing_utils.py, see huggingface/transformers#45290).
            {"role": "assistant", "content": "", "tool_calls": dummy_tool_calls},
        ]
        prefix_ids = self.tokenizer.apply_chat_template(
            dummy_messages,
            add_generation_prompt=False,
            tokenize=True,
            chat_template=self.chat_template,
            return_dict=False,
            **self.chat_template_kwargs,
        )
        full_ids = self.tokenizer.apply_chat_template(
            dummy_messages + tool_messages,
            add_generation_prompt=True,
            tokenize=True,
            chat_template=self.chat_template,
            return_dict=False,
            **self.chat_template_kwargs,
        )
        # Some chat templates (Qwen3/Qwen3.5) render "...<|im_end|>\n" after assistant/tool blocks.
        # Align the slicing boundary to EOS, not EOS + newline.
        eos_positions = [i for i, tok_id in enumerate(prefix_ids) if tok_id == self.tokenizer.eos_token_id]
        if eos_positions:
            prefix_ids = prefix_ids[: eos_positions[-1] + 1]
        if full_ids[: len(prefix_ids)] != prefix_ids:
            raise ValueError("Unexpected tokenization: the EOS-trimmed prefix IDs are not a prefix of the full IDs.")
        return full_ids[len(prefix_ids) :]

    def _execute_tool_calls(
        self, tool_calls: list[dict[str, Any]], tool_dict: dict[str, Callable]
    ) -> tuple[list[dict[str, str]], int, int]:
        tool_messages = []
        n_calls = 0
        n_failures = 0
        for tool_call in tool_calls:
            n_calls += 1
            function = tool_call["function"]
            name = function["name"]
            try:
                arguments = function.get("arguments", {})
                result = tool_dict[name](**arguments)
            except Exception as error:
                n_failures += 1
                result = {"error": str(error)}
            tool_messages.append({"role": "tool", "name": name, "content": str(result)})
        return tool_messages, n_calls, n_failures

    async def _generate_one_turn(self, prompt_ids: list[int]) -> tuple[list[int], list[float]]:
        payload = {
            "model": self.model_name,
            "prompt": prompt_ids,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "n": 1,
            "return_token_ids": True,
            "logprobs": 0,
        }
        output = await _retry_on_http_error(
            lambda: self._post("/v1/completions", payload, self.request_timeout),
            max_attempts=30,
            label="vllm /v1/completions",
        )
        choice = output["choices"][0]
        return choice["token_ids"], choice["logprobs"]["token_logprobs"]

    async def _score_group(self, group: RolloutGroup) -> list[RolloutSample]:
        kwargs = dict(
            completions=group.completions,
            prompt=group.prompt,
            prompts=[group.prompt] * len(group.completions),
            completion_ids=group.completions_ids,
            **group.reward_kwargs,
        )
        all_rewards = await asyncio.gather(
            *[
                reward_func(**kwargs)
                if inspect.iscoroutinefunction(reward_func)
                else asyncio.to_thread(reward_func, **kwargs)
                for reward_func in self.reward_funcs
            ]
        )

        # Reward funcs may return None per-sample (unparseable gold). Convert to NaN. A completion
        # for which every func returned None is unscorable: nansum would give 0 and the row would
        # pull the policy away from actually-correct answers.
        # Mark such rows NaN, then compute advantage on the scorable subset only.
        all_rewards = [[r if r is not None else float("nan") for r in row] for row in all_rewards]
        arr = np.array(all_rewards, dtype=float)
        all_nan_mask = np.all(np.isnan(arr), axis=0)
        rewards = np.nansum(arr, axis=0)
        rewards[all_nan_mask] = np.nan

        scored_mask = ~np.isnan(rewards)
        # NOTE: for NaN reward we set advantage to 0 !
        advantages = np.zeros_like(rewards)
        if scored_mask.any():
            scored = rewards[scored_mask]
            advantages[scored_mask] = (scored - scored.mean()) / (scored.std() + 1e-8)
            reward_mean = float(scored.mean())
            reward_std = float(scored.std())
        else:
            reward_mean = reward_std = float("nan")
        logger.info(f"Rollout metrics: reward_mean={reward_mean:.4f}, reward_std={reward_std:.4f}")

        total_calls = sum(group.tool_call_counts)
        tool_metrics = (
            [
                {
                    "tools/call_frequency": float(n_calls),
                    "tools/failure_frequency": (n_failures / n_calls) if n_calls > 0 else 0.0,
                }
                for n_calls, n_failures in zip(group.tool_call_counts, group.tool_failure_counts, strict=True)
            ]
            if total_calls > 0
            else [{}] * len(group.completions)
        )

        per_func_rewards = np.array(all_rewards, dtype=float)
        return [
            RolloutSample(
                prompt=group.prompt,
                completion=completion,
                input_ids=group.prompt_ids + completion_ids,
                completion_mask=[0] * len(group.prompt_ids) + tool_mask,
                old_log_probs=[0.0] * len(group.prompt_ids) + logprobs,
                advantage=advantage,
                model_version=group.model_version,
                metrics={
                    "reward": float(reward),
                    "reward_std": reward_std,
                    **{
                        f"rewards/{name}": float(func_reward)
                        for name, func_reward in zip(self.reward_func_names, per_func_rewards[:, i], strict=True)
                    },
                    **tm,
                },
            )
            for i, (completion, completion_ids, logprobs, tool_mask, advantage, reward, tm) in enumerate(
                zip(
                    group.completions,
                    group.completions_ids,
                    group.completions_logprobs,
                    group.tool_mask,
                    advantages,
                    rewards,
                    tool_metrics,
                    strict=True,
                )
            )
        ]

    async def _post(self, path: str, payload: dict, timeout: float, max_retries: int = 3) -> dict:
        client_timeout = aiohttp.ClientTimeout(total=timeout)

        async def _do_post():
            async with self.session.post(
                f"{self.vllm_server_url}{path}", json=payload, timeout=client_timeout
            ) as response:
                response.raise_for_status()
                content = await response.json()
                return content if content else {}

        return await _retry_on_http_error(_do_post, label=f"POST {path}", max_attempts=max_retries)


class AsyncRolloutWorker:
    """Parent-side controller: spawns a child process running `_AsyncRolloutLoop`.

    The trainer holds this object on rank 0. The child does the actual rollout work; this class only manages lifecycle
    (start/stop) and exposes the shared `mp.Queue` (`rollout_buffer`) and `mp.Value` (`model_version`) the trainer
    reads/writes.

    Constructor kwargs are forwarded as-is to `_AsyncRolloutLoop` when the child spawns; only `queue_maxsize` and
    `child_ready_timeout` are consumed here. Because the child is spawned, every forwarded kwarg is pickled:
    `reward_funcs`, `tools`, and `environment_factory` (and anything they close over) must be picklable — module-level
    functions, `functools.partial`, or callable instances, never lambdas or closures. `start()` validates this up front
    and raises a `TypeError` otherwise. The child also runs with `CUDA_VISIBLE_DEVICES=""`, so GPU reward models
    execute on CPU.
    """

    def __init__(
        self,
        *,
        queue_maxsize: int = 0,
        child_ready_timeout: int = 300,
        **loop_kwargs: Any,
    ):
        if not is_vllm_available(min_version="0.17.1"):
            raise ImportError(
                "vLLM >= 0.17.1 is required to use AsyncRolloutWorker. Install it with: pip install 'vllm>=0.17.1'"
            )
        ctx = mp.get_context("spawn")
        self._mp_ctx = ctx
        self.rollout_buffer = ctx.Queue(maxsize=queue_maxsize)
        self._model_version_value = ctx.Value("i", 0)
        self._stop_event_mp = ctx.Event()
        self._child_ready_event = ctx.Event()
        # Liveness state shared with the child. Wall-clock seconds because monotonic() is per-process.
        self._heartbeat_value = ctx.Value("d", 0.0)
        self._failed_event = ctx.Event()
        self._exception_info_queue = ctx.Queue(maxsize=1)
        # Forwarded verbatim to _AsyncRolloutLoop in the child. queue_maxsize is also
        # forwarded — the child reads it for "rollout buffer full" log lines.
        loop_kwargs["queue_maxsize"] = queue_maxsize
        self._loop_kwargs = loop_kwargs
        self._child_ready_timeout = child_ready_timeout
        self._process: mp.Process | None = None

    @property
    def model_version(self) -> int:
        return int(self._model_version_value.value)

    @model_version.setter
    def model_version(self, value: int) -> None:
        # NOTE(@aminediro) Read/write ops like += are not atomic with mp.Value
        with self._model_version_value.get_lock():
            self._model_version_value.value = int(value)

    def update_model_version(self, model_version: int) -> None:
        self.model_version = model_version

    def start(self) -> None:
        if self._process is not None:
            logger.warning("AsyncRolloutWorker.start() called but child process is already running; ignoring.")
            return
        # Reset so spawn-import latency (~tens of seconds) doesn't immediately trip check_health.
        self._heartbeat_value.value = time.time()
        try:
            pickle.dumps(self._loop_kwargs)
        except (pickle.PicklingError, AttributeError, TypeError) as e:
            # fails fast with an actionable message instead of an opaque traceback
            raise TypeError(
                "AsyncRolloutWorker forwards reward_funcs / tools / environment_factory to a spawned "
                "child process, so they must be picklable. Lambdas and closures are not: use a "
                "module-level function, functools.partial, or a callable class instance instead."
            ) from e
        self._process = self._mp_ctx.Process(
            target=_child_main,
            args=(
                self._loop_kwargs,
                self.rollout_buffer,
                self._model_version_value,
                self._stop_event_mp,
                self._child_ready_event,
                self._heartbeat_value,
                self._failed_event,
                self._exception_info_queue,
            ),
            name="grpo-rollout-worker-child",
            daemon=True,
        )
        self._process.start()
        logger.info(
            f"AsyncRolloutWorker spawned child pid={self._process.pid}; "
            f"waiting up to {self._child_ready_timeout}s for the ready signal"
        )
        # spawn re-imports torch+transformers+trl+vllm in the child — slow on cold launch. Poll
        # liveness so an early crash surfaces immediately instead of after the full timeout.
        deadline = time.monotonic() + self._child_ready_timeout
        while not self._child_ready_event.wait(timeout=1.0):
            if not self._process.is_alive():
                exit_code = self._process.exitcode
                self._process = None
                raise RuntimeError(
                    f"AsyncRolloutWorker child exited during init (exitcode={exit_code}). "
                    "Check the child's stderr for the traceback."
                )
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"AsyncRolloutWorker child did not signal ready within {self._child_ready_timeout}s."
                )
        logger.info("AsyncRolloutWorker child is ready")

    def check_health(self, stale_after_s: float) -> None:
        """Raise if the child crashed or hasn't ticked the heartbeat within `stale_after_s`."""
        if self._failed_event.is_set():
            try:
                type_name, msg, tb = self._exception_info_queue.get_nowait()
                cause = RuntimeError(f"{type_name}: {msg}\n{tb}")
            except queue.Empty:
                cause = None
            raise RuntimeError("Rollout worker child has failed; see chained exception.") from cause
        age = time.time() - self._heartbeat_value.value
        if age > stale_after_s:
            raise RuntimeError(f"Rollout worker heartbeat stale: {age:.0f}s > {stale_after_s:.0f}s; child is hung.")

    def stop(self) -> None:
        if self._process is None:
            return
        logger.info("Stopping AsyncRolloutWorker child process...")
        self._stop_event_mp.set()
        # If start() raised before Process.start() returned (e.g. pickle failure during spawn),
        # _popen is None and .join() would assert — skip cleanly.
        if self._process._popen is not None:
            self._process.join(timeout=15)
            if self._process.is_alive():
                logger.warning("Child did not exit within 15s; terminating.")
                self._process.terminate()
                self._process.join(timeout=5)
                if self._process.is_alive():
                    self._process.kill()
        self._process = None
