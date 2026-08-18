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
import math
import multiprocessing as mp
import os
import pickle
import queue
import threading
import time
import traceback
from collections import defaultdict
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass
from multiprocessing.queues import Queue as MPQueue
from multiprocessing.sharedctypes import Synchronized as MPValue
from multiprocessing.synchronize import Event as MPEvent
from typing import Any, TypeAlias

import aiohttp
from accelerate.logging import get_logger
from datasets import Dataset
from transformers import PreTrainedTokenizerBase

from ...import_utils import is_vllm_available
from ...trainer.utils import print_prompt_completions_sample


logger = get_logger(__name__)

Messages: TypeAlias = list[dict[str, str]]

_RETRYABLE_HTTP_ERRORS = (aiohttp.ClientError, asyncio.TimeoutError, TimeoutError, ConnectionResetError)


@dataclass(slots=True)
class RolloutSample:
    """One training row: a student on-policy completion plus the teacher's sparse per-position scoring.

    Unlike [`~trl.experimental.async_grpo.async_rollout_worker.RolloutSample`], there is no `advantage` (no
    group-relative baseline: distillation has no policy-gradient term), no reward metrics, and no student-side
    generation-time logprobs (nothing in `compute_loss` reads them today; the student side of the divergence is
    computed fresh from the live model). `teacher_topk_ids` / `teacher_topk_logprobs` carry, per completion position,
    the teacher's top-`teacher_top_k` candidate tokens (plus the realized token if it fell outside that set), see
    `_parse_teacher_logprobs_at_position`. Rows are padded to a fixed width with id `-1` / logprob `-inf`, masked out
    in `compute_loss`.
    """

    prompt: Messages
    completion: Messages
    input_ids: list[int]
    completion_mask: list[int]
    teacher_topk_ids: list[list[int]]  # per completion position; empty list where mask is 0
    teacher_topk_logprobs: list[list[float]]  # per completion position; empty list where mask is 0
    model_version: int
    metrics: dict[str, float]
    teacher_id: str  # which teacher_server_urls entry scored this sample (see _resolve_teacher_server_url)
    enqueued_at: float | None = None


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

    threading.Thread(target=_watch, daemon=True, name="distillation-mp-stop-watcher").start()


def _child_main(
    loop_kwargs: dict[str, Any],
    samples_queue: MPQueue,
    model_version_value: MPValue,
    stop_event: MPEvent,
    child_ready_event: MPEvent,
    heartbeat_value: MPValue,
    failed_event: MPEvent,
    exception_info_queue: MPQueue,
    metrics_queue: MPQueue,
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
        metrics_queue=metrics_queue,
    )
    child_ready_event.set()
    _spawn_stop_watcher(rollout_loop, stop_event)
    try:
        rollout_loop.run()
    except Exception:
        traceback.print_exc()
        raise


def _parse_teacher_logprobs_at_position(position_logprobs: dict[str, dict] | None) -> tuple[list[int], list[float]]:
    """Parse one completion position's entry of a vLLM `prompt_logprobs` response.

    vLLM reports `{token_id: {"logprob": ..., "rank": ...}}` per position, holding the top-`teacher_top_k` candidates
    plus the realized/actual token when that one fell outside the top-k, in unspecified order. Sorting by `rank` puts
    the teacher's own top-1 candidate first, which is the ordering `_narrow_top1_actual_support` indexes into. Keeping
    the whole mapping (rather than truncating back to `teacher_top_k`, the way
    [`~trl.generation.vllm_client.VLLMClient.get_sequence_logprobs`] does for the sync trainers) is what guarantees the
    realized token is always present, at any rank. A NaN logprob is dropped rather than coerced to a sentinel, since it
    reflects a genuinely invalid score, not a padding slot; a position vLLM scored no candidate for at all (`None`,
    e.g. the sequence's very first token) yields empty lists, masked out in `compute_loss`.
    """
    if not position_logprobs:
        return [], []
    items = sorted(position_logprobs.items(), key=lambda item: item[1]["rank"])
    parsed = [(int(tid), entry["logprob"]) for tid, entry in items if not math.isnan(entry["logprob"])]
    if not parsed:
        return [], []
    ids, lps = zip(*parsed, strict=True)
    return list(ids), list(lps)


class _AsyncRolloutLoop:
    """Asyncio generate-and-score loop. Lives entirely inside the spawned child process.

    Structurally this is [`~trl.experimental.async_grpo.async_rollout_worker._AsyncRolloutLoop`] with the
    group-relative-advantage machinery removed (distillation has no group baseline: each prompt yields exactly one
    on-policy sample) and the reward scoring step replaced by a teacher-scoring HTTP call. Because there is no
    grouping, generation and scoring happen in a single task per sample (`_generate_and_score_one`) rather than two
    loops connected by an internal queue, concurrency across in-flight samples still comes from running up to
    `max_inflight_tasks` such tasks at once, exactly as in the GRPO worker.

    Every server it talks to is a plain `vllm serve`, reached through the same OpenAI-compatible `/v1/completions`
    endpoint, in two different modes: the student's server samples a real completion (`_generate_one_turn`), and one or
    more teachers' servers teacher-force that completion and report their own per-position distribution over it
    (`_score_with_teacher`, no new tokens generated). With multiple teachers configured (`teacher_server_urls`),
    `_resolve_teacher_server_url` routes each row to its `teacher_id`'s server (MOPD: multi-teacher on-policy
    distillation). Pushes scored [`RolloutSample`]s into the shared `mp.Queue` (`rollout_buffer`); reads the bumped
    policy version from the shared `mp.Value` (`model_version_value`).
    """

    def __init__(
        self,
        *,
        model_name: str,
        dataset: Dataset,
        processing_class: PreTrainedTokenizerBase,
        rollout_buffer: MPQueue,
        model_version_value: MPValue,
        heartbeat_value: MPValue,
        failed_event: MPEvent,
        exception_info_queue: MPQueue,
        metrics_queue: MPQueue,
        max_inflight_tasks: int = 128,
        queue_maxsize: int = 0,
        vllm_server_url: str = "http://localhost:8000",
        teacher_server_urls: dict[str, str] | None = None,
        teacher_top_k: int = 8,
        teacher_temperature: float = 1.0,
        max_tokens: int = 32,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float | None = None,
        repetition_penalty: float = 1.0,
        request_timeout: int = 120,
        chat_template_kwargs: dict[str, Any] | None = None,
        log_completions: bool = False,
        log_completions_steps: int = 100,
        num_completions_to_print: int | None = None,
    ):
        self.model_name = model_name
        self.dataset = dataset
        self._dataset_iter = iter(dataset)
        self.tokenizer = processing_class
        self.rollout_buffer = rollout_buffer  # shared mp.Queue
        self._model_version_value = model_version_value  # shared mp.Value
        self._heartbeat_value = heartbeat_value  # shared mp.Value('d'); wall-clock seconds
        self._failed_event = failed_event  # shared mp.Event
        self._exception_info_queue = exception_info_queue  # shared mp.Queue(maxsize=1)
        self._metrics_queue = metrics_queue  # shared mp.Queue; drained by the trainer in `log()`
        # Metric accumulators
        self._counters: dict[str, float] = defaultdict(float)
        self._rates: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
        self._pushed_completion_tokens = 0
        self._pushed_at = time.monotonic()

        self.max_inflight_tasks = max_inflight_tasks
        self.queue_maxsize = queue_maxsize
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.repetition_penalty = repetition_penalty
        self.request_timeout = request_timeout
        self.chat_template_kwargs = chat_template_kwargs or {}
        self.log_completions = log_completions
        self.log_completions_steps = log_completions_steps
        self.num_completions_to_print = num_completions_to_print
        self.vllm_server_url = vllm_server_url.rstrip("/")
        teacher_server_urls = teacher_server_urls or {"default": "http://localhost:8001"}
        self.teacher_server_urls = {k: v.rstrip("/") for k, v in teacher_server_urls.items()}
        self.teacher_top_k = teacher_top_k
        self.teacher_temperature = teacher_temperature

        # Served model id per teacher, resolved from each teacher server's `/v1/models` in `_run_loops`, since
        # `/v1/completions` names the model it is addressing and a teacher serves a different one than the student.
        self.teacher_model_names: dict[str, str] = {}

        self._total_completion_tokens = 0
        self._total_samples_scored = 0
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
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=2 * self.max_inflight_tasks)) as session:
            self.session = session
            logger.info(
                f"vllm worker started: max_inflight_tasks={self.max_inflight_tasks}, temperature={self.temperature}, "
                f"top_p={self.top_p}, top_k={self.top_k}, min_p={self.min_p}, "
                f"repetition_penalty={self.repetition_penalty}"
            )
            await self._resolve_teacher_model_names()
            await self._generate_loop(stop_event=stop_event)

    async def _resolve_teacher_model_names(self) -> None:
        """Ask every teacher server which model it serves, so `_score_with_teacher` can name it in its request.

        Doubles as the readiness wait for the teachers: the retry budget is the same generous one the generation and
        scoring calls use, so a teacher still loading its weights when the worker starts is waited out rather than
        failing the run.
        """
        for teacher_id, url in self.teacher_server_urls.items():
            models = await self._retry(
                lambda url=url: self._get(url, "/v1/models", self.request_timeout),
                max_attempts=30,
                label=f"teacher {teacher_id} /v1/models",
            )
            self.teacher_model_names[teacher_id] = models["data"][0]["id"]
            logger.info(f"teacher {teacher_id!r} at {url} serves {self.teacher_model_names[teacher_id]}")

    async def _generate_loop(self, stop_event: asyncio.Event) -> None:
        inflight_tasks: dict[asyncio.Task, int] = {}
        free_slots = set(range(self.max_inflight_tasks))
        work_iter = self._repeat_iterator()

        self._generation_start_time = time.monotonic()
        try:
            while True:
                self._heartbeat_value.value = time.time()
                while free_slots and not stop_event.is_set():
                    row = next(work_iter)
                    slot = free_slots.pop()
                    task = asyncio.create_task(self._generate_and_score_one(row))
                    inflight_tasks[task] = slot

                if not inflight_tasks:
                    if stop_event.is_set():
                        return
                    await asyncio.sleep(0.01)
                    continue

                done, _ = await asyncio.wait(inflight_tasks, return_when=asyncio.FIRST_COMPLETED, timeout=0.1)
                if not done:
                    continue

                for task in done:
                    slot = inflight_tasks.pop(task)
                    free_slots.add(slot)
                    if task.exception() is not None:
                        raise task.exception()
                    sample = task.result()
                    self._total_completion_tokens += sum(sample.completion_mask)
                    self._total_samples_scored += 1
                    now = time.monotonic()
                    self._push_metrics(
                        {
                            "rollout/inflight": float(len(inflight_tasks)),
                            # Windowed, not cumulative. A cumulative average cannot show a generation stall.
                            "rollout/generated_tok_s": (
                                float(self._total_completion_tokens - self._pushed_completion_tokens),
                                now - self._pushed_at,
                            ),
                        }
                    )
                    self._pushed_completion_tokens = self._total_completion_tokens
                    self._pushed_at = now

                    # Each scored sample is its own single-element batch (no group-relative multi-sample scoring the
                    # way GRPO has), so unlike GRPO's per-group call, `num_completions_to_print` has nothing to
                    # subsample here without this gate — printing on every sample would flood logs over a long run.
                    if self.log_completions and self._total_samples_scored % self.log_completions_steps == 0:
                        print_prompt_completions_sample(
                            prompts=[sample.prompt],
                            completions=[sample.completion],
                            rewards={},
                            advantages=None,  # no group-relative advantage in distillation
                            step=self._total_samples_scored,
                            num_samples=self.num_completions_to_print,
                        )

                    t_blocked = None
                    while True:
                        try:
                            sample.enqueued_at = time.time()
                            self.rollout_buffer.put_nowait(sample)
                            break
                        except queue.Full:
                            if stop_event.is_set():
                                return
                            if t_blocked is None:
                                t_blocked = time.monotonic()
                                logger.info(
                                    f"rollout buffer full (maxsize={self.queue_maxsize}), "
                                    "waiting for trainer to consume..."
                                )
                            await asyncio.sleep(0.1)
                    if t_blocked is not None:
                        self._push_metrics({"rollout/backpressure_s": time.monotonic() - t_blocked})
        finally:
            for task in inflight_tasks:
                task.cancel()
            if inflight_tasks:
                await asyncio.gather(*inflight_tasks, return_exceptions=True)

    async def _retry(self, coro_factory: Callable[[], Awaitable], *, label: str, max_attempts: int = 1):
        """Retry an aiohttp coroutine on transport errors with bounded exponential backoff, counting the retries.

        Ported from [`~trl.experimental.async_grpo.async_rollout_worker._AsyncRolloutLoop._retry`]. Every retried call
        in this loop targets a vLLM server — the student's or a teacher's — so the counter is named for the dependency
        rather than for the transport.
        """
        for attempt in range(max_attempts):
            try:
                return await coro_factory()
            except _RETRYABLE_HTTP_ERRORS as e:
                if attempt >= max_attempts - 1:
                    raise
                self._counters["rollout/vllm_retry_total"] += 1
                sleep = min(2 ** min(attempt, 4), 16)
                logger.warning(
                    f"{label} failed ({type(e).__name__}: {e}); retry {attempt + 1}/{max_attempts} in {sleep}s"
                )
                await asyncio.sleep(sleep)

    def _push_metrics(self, values: dict[str, float | tuple[float, float]]) -> None:
        """Send metrics to the trainer, along with the counters and rates accumulated since the last push."""
        payload = dict(values)
        payload.update(self._counters)
        payload.update({key: (num, den) for key, (num, den) in self._rates.items()})
        self._counters.clear()
        self._rates.clear()
        try:
            self._metrics_queue.put_nowait(payload)
        except queue.Full:
            pass  # never block generation

    def _push_rollout_metrics(
        self, *, completion_ids: list[int], teacher_id: str, score_s: float, duration_s: float
    ) -> None:
        """One rollout: what the student generated, and the teacher call that scored it.

        A rollout yields exactly one training sample here — one turn, no group baseline, no re-tokenization forking —
        so unlike [`~trl.experimental.async_grpo.async_rollout_worker._AsyncRolloutLoop._push_rollout_metrics`] there
        is no `rollout/samples_per_rollout`, `rollout/turns_*`, drift tally or tool tally to report.
        """
        completion_tokens = len(completion_ids)
        self._rates["completions/mean_length"][0] += completion_tokens
        self._rates["completions/mean_length"][1] += 1
        self._rates["rollout/score_s"][0] += score_s
        self._rates["rollout/score_s"][1] += 1
        if len(self.teacher_server_urls) > 1:
            # MOPD: one slow teacher throttles only the rollouts routed to it, which the blended mean above hides.
            # Keyed the way the trainer names `teacher_jsd/{teacher_id}`, so the per-teacher series group together.
            self._rates[f"teacher_score_s/{teacher_id}"][0] += score_s
            self._rates[f"teacher_score_s/{teacher_id}"][1] += 1
        if completion_ids:
            # NOTE(@aminediro):
            # Truncation is read off the same way [`GRPOTrainer`] and [`RLOOTrainer`] define
            # `completions/clipped_ratio`: a completion that does not end on EOS (or pad) was cut off by `max_tokens`
            # rather than finishing. Deliberately NOT vLLM's `finish_reason`, so the metric means the same thing here
            eos_and_pad = (self.tokenizer.eos_token_id, self.tokenizer.pad_token_id)
            self._rates["completions/clipped_ratio"][0] += completion_ids[-1] not in eos_and_pad
            self._rates["completions/clipped_ratio"][1] += 1
        self._push_metrics(
            {
                "rollout/duration_s": duration_s,
                "completions/max_length": float(completion_tokens),
                "completions/min_length": float(completion_tokens),
            }
        )

    def _repeat_iterator(self) -> Iterator[dict[str, Any]]:
        while True:
            try:
                row = next(self._dataset_iter)
            except StopIteration:
                self._dataset_iter = iter(self.dataset)
                row = next(self._dataset_iter)
            yield row

    def _resolve_teacher_server_url(self, row: dict[str, Any]) -> tuple[str, str]:
        """Pick which teacher server scores `row` (MOPD: multi-teacher on-policy distillation).

        A single configured teacher scores every row regardless of any `teacher_id` column (plain on-policy
        distillation). With multiple teachers, `row["teacher_id"]` selects among them; a missing or unmapped
        `teacher_id` is a configuration error (raises), not a silent misroute to the wrong teacher. Returns
        `(teacher_id, server_url)`; the resolved id is carried on the `RolloutSample` so `compute_loss` can break down
        `jsd`/`entropy`/`teacher_entropy` per teacher.
        """
        if len(self.teacher_server_urls) == 1:
            ((teacher_id, url),) = self.teacher_server_urls.items()
            return teacher_id, url
        teacher_id = row.get("teacher_id")
        if teacher_id not in self.teacher_server_urls:
            raise ValueError(
                f"Example has `teacher_id={teacher_id!r}`, which is not among the teachers passed to "
                f"`teacher_server_urls`. Expected one of: {list(self.teacher_server_urls)}."
            )
        return teacher_id, self.teacher_server_urls[teacher_id]

    async def _generate_and_score_one(self, row: dict[str, Any]) -> RolloutSample:
        model_version = self.model_version
        t_dispatch = time.monotonic()
        prompt = row["prompt"]
        prompt_ids = self.tokenizer.apply_chat_template(
            prompt, return_dict=False, add_generation_prompt=True, **self.chat_template_kwargs
        )
        completion_ids = await self._generate_one_turn(prompt_ids)
        completion_text = self.tokenizer.decode(completion_ids, skip_special_tokens=False)
        completion = [{"role": "assistant", "content": completion_text}]

        teacher_id, teacher_server_url = self._resolve_teacher_server_url(row)
        t_score = time.monotonic()
        teacher_topk_ids, teacher_topk_logprobs = await self._score_with_teacher(
            prompt_ids, completion_ids, teacher_server_url, self.teacher_model_names[teacher_id]
        )
        score_s = time.monotonic() - t_score

        input_ids = prompt_ids + completion_ids
        completion_mask = [0] * len(prompt_ids) + [1] * len(completion_ids)
        full_teacher_ids = [[] for _ in prompt_ids] + teacher_topk_ids
        full_teacher_logprobs = [[] for _ in prompt_ids] + teacher_topk_logprobs

        self._push_rollout_metrics(
            completion_ids=completion_ids,
            teacher_id=teacher_id,
            score_s=score_s,
            duration_s=time.monotonic() - t_dispatch,
        )
        return RolloutSample(
            prompt=prompt,
            completion=completion,
            input_ids=input_ids,
            completion_mask=completion_mask,
            teacher_topk_ids=full_teacher_ids,
            teacher_topk_logprobs=full_teacher_logprobs,
            model_version=model_version,
            teacher_id=teacher_id,
            metrics={},
        )

    async def _generate_one_turn(self, prompt_ids: list[int]) -> list[int]:
        payload = {
            "model": self.model_name,
            "prompt": prompt_ids,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "n": 1,
            "return_token_ids": True,
        }
        if self.min_p is not None:
            payload["min_p"] = self.min_p
        output = await self._retry(
            lambda: self._post(self.vllm_server_url, "/v1/completions", payload, self.request_timeout),
            max_attempts=30,
            label="student vllm /v1/completions",
        )
        return output["choices"][0]["token_ids"]

    async def _score_with_teacher(
        self, prompt_ids: list[int], completion_ids: list[int], teacher_server_url: str, teacher_model_name: str
    ) -> tuple[list[list[int]], list[list[float]]]:
        """Teacher-force the student's completion through a teacher and read back its per-position distribution.

        `max_tokens=1` generates a single throwaway token (the teacher is never sampled from), and
        `prompt_logprobs=teacher_top_k` makes the server score every position of the sequence it was handed, reporting
        the top-`teacher_top_k` candidates plus the realized token when that one fell outside the top-k. This is the
        request [`~trl.generation.vllm_client.VLLMClient.get_sequence_logprobs`] issues for the synchronous
        server-teacher trainers; the response is parsed here rather than through that client because it truncates each
        position back to exactly `top_logprobs` entries and reports the realized token in separate fields, whereas the
        loss here wants one rank-sorted list per position (see `_parse_teacher_logprobs_at_position`).

        `temperature` reaches the reported logprobs only on a server started with `--logprobs-mode processed_logprobs`,
        and a `teacher_top_k` above vLLM's default cap of 20 additionally needs `--max-logprobs -1`; see
        `AsyncDistillationConfig.teacher_server_urls` for the full `vllm serve` command.
        """
        payload = {
            "model": teacher_model_name,
            "prompt": prompt_ids + completion_ids,
            "max_tokens": 1,
            "temperature": self.teacher_temperature,
            "prompt_logprobs": self.teacher_top_k,
        }
        output = await self._retry(
            lambda: self._post(teacher_server_url, "/v1/completions", payload, self.request_timeout),
            max_attempts=30,
            label="teacher /v1/completions",
        )
        # `prompt_logprobs[i]` scores `sequence[i]` conditioned on `sequence[:i]`, so the row spans the whole sequence
        # and the completion region starts at `len(prompt_ids)`.
        prompt_logprobs = output["choices"][0]["prompt_logprobs"]
        topk_ids, topk_logprobs = [], []
        for position in range(len(prompt_ids), len(prompt_logprobs)):
            ids, lps = _parse_teacher_logprobs_at_position(prompt_logprobs[position])
            topk_ids.append(ids)
            topk_logprobs.append(lps)
        return topk_ids, topk_logprobs

    async def _get(self, base_url: str, path: str, timeout: float, max_retries: int = 3) -> dict:
        client_timeout = aiohttp.ClientTimeout(total=timeout)

        async def _do_get():
            async with self.session.get(f"{base_url}{path}", timeout=client_timeout) as response:
                response.raise_for_status()
                return await response.json()

        return await self._retry(_do_get, label=f"GET {base_url}{path}", max_attempts=max_retries)

    async def _post(self, base_url: str, path: str, payload: dict, timeout: float, max_retries: int = 3) -> dict:
        client_timeout = aiohttp.ClientTimeout(total=timeout)

        async def _do_post():
            async with self.session.post(f"{base_url}{path}", json=payload, timeout=client_timeout) as response:
                response.raise_for_status()
                content = await response.json()
                return content if content else {}

        return await self._retry(_do_post, label=f"POST {base_url}{path}", max_attempts=max_retries)


class AsyncRolloutWorker:
    """Parent-side controller: spawns a child process running `_AsyncRolloutLoop`.

    Ported from [`~trl.experimental.async_grpo.async_rollout_worker.AsyncRolloutWorker`], same lifecycle contract
    (start/stop/check_health/update_model_version), same picklability validation, same `CUDA_VISIBLE_DEVICES=""` child
    isolation. No `reward_funcs`/`tools`/`environment_factory` to forward: distillation's only external calls are the
    vLLM servers already carried by `loop_kwargs` (`vllm_server_url`, `teacher_server_urls`).
    """

    def __init__(
        self,
        *,
        queue_maxsize: int = 0,
        child_ready_timeout: int = 300,
        **loop_kwargs: Any,
    ):
        if not is_vllm_available(min_version="0.22.0"):
            raise ImportError(
                "vLLM >= 0.22.0 is required to use AsyncRolloutWorker. Install it with: pip install 'vllm>=0.22.0'"
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
        # Metrics the child measures, drained by the trainer in `log()`. Bounded and dropped-on-full in the child, so a
        # trainer that stops draining can never block generation.
        self.metrics_queue = ctx.Queue(maxsize=4096)
        # Forwarded verbatim to _AsyncRolloutLoop in the child. queue_maxsize is also
        # forwarded, the child reads it for "rollout buffer full" log lines.
        loop_kwargs["queue_maxsize"] = queue_maxsize
        self._loop_kwargs = loop_kwargs
        self._child_ready_timeout = child_ready_timeout
        self._process: mp.Process | None = None

    @property
    def model_version(self) -> int:
        return int(self._model_version_value.value)

    @model_version.setter
    def model_version(self, value: int) -> None:
        # NOTE Read/write ops like += are not atomic with mp.Value
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
            raise TypeError(
                "AsyncRolloutWorker forwards its constructor kwargs to a spawned child process, so they must be "
                "picklable. Lambdas and closures are not: use a module-level function, functools.partial, or a "
                "callable class instance instead."
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
                self.metrics_queue,
            ),
            name="distillation-rollout-worker-child",
            daemon=True,
        )
        self._process.start()
        logger.info(
            f"AsyncRolloutWorker spawned child pid={self._process.pid}; "
            f"waiting up to {self._child_ready_timeout}s for the ready signal"
        )
        # spawn re-imports torch+transformers+trl+vllm in the child, slow on cold launch. Poll
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
        # _popen is None and .join() would assert, skip cleanly.
        if self._process._popen is not None:
            self._process.join(timeout=15)
            if self._process.is_alive():
                logger.warning("Child did not exit within 15s; terminating.")
                self._process.terminate()
                self._process.join(timeout=5)
                if self._process.is_alive():
                    self._process.kill()
        self._process = None
