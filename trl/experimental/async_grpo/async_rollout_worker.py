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
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, TypeAlias

import aiohttp
import numpy as np
import requests
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


if is_vllm_available(min_version="0.17.1"):
    from vllm.distributed.weight_transfer.nccl_engine import NCCLTrainerSendWeightsArgs, NCCLWeightTransferEngine
    from vllm.utils.network_utils import get_ip, get_open_port


logger = get_logger(__name__)

Messages: TypeAlias = list[dict[str, str]]


@dataclass(slots=True)
class RolloutGroup:
    """Single GRPO group for one prompt with multiple completions."""

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
    metrics: dict[str, float]  # logging metadata only, not used in loss computation


# ---------------------------------------------------------------------------
# Queue shims
#
# The trainer's `RolloutQueueDataset.__iter__` reads from `rollout_worker.rollout_buffer`
# as if it were a `queue.Queue` (calls `.get(timeout=...)` and `.qsize()`). The
# child process's `_score_loop` writes to `self.rollout_buffer` with `.put_nowait()`
# and reads `.maxsize`. We adapt the shared `mp.Queue` to both interfaces with
# tiny shims — one for each end.
# ---------------------------------------------------------------------------


class _MPQueueParentShim:
    """Parent-side adapter so the trainer reads the shared `mp.Queue` via
    the `queue.Queue` interface (`.get()`, `.qsize()`, `.empty()`)."""

    def __init__(self, mp_q):
        self._q = mp_q

    def get(self, block: bool = True, timeout: float | None = None):
        return self._q.get(block=block, timeout=timeout)

    def qsize(self) -> int:
        try:
            return self._q.qsize()
        except (NotImplementedError, OSError):
            # macOS doesn't support qsize() on mp.Queue.
            return 0

    def empty(self) -> bool:
        try:
            return self._q.empty()
        except (NotImplementedError, OSError):
            return False


class _MPQueueChildShim:
    """Child-side adapter so `_score_loop` can `.put_nowait()` into the shared
    `mp.Queue` and read `.maxsize` the same way it would on a stdlib `queue.Queue`."""

    def __init__(self, mp_q):
        self._q = mp_q
        # `_score_loop` reads `rollout_buffer.maxsize` for log lines.
        # `mp.Queue` stores its capacity in the private `_maxsize`.
        self.maxsize = getattr(mp_q, "_maxsize", 0)

    def put(self, item, block: bool = True, timeout: float | None = None):
        self._q.put(item, block=block, timeout=timeout)

    def put_nowait(self, item):
        # `mp.Queue.put_nowait` raises `queue.Full` on overflow, matching stdlib.
        self._q.put_nowait(item)

    def qsize(self) -> int:
        try:
            return self._q.qsize()
        except (NotImplementedError, OSError):
            return 0

    def empty(self) -> bool:
        try:
            return self._q.empty()
        except (NotImplementedError, OSError):
            return False

    def full(self) -> bool:
        try:
            return self._q.full()
        except (NotImplementedError, OSError):
            return False


def _child_main(
    init_kwargs_pkl: bytes,
    samples_queue: "mp.Queue",
    model_version_value: "mp.Value",  # noqa: F821 (mp.Value is a function, not a type)
    stop_event: "mp.Event",
    pause_event: "mp.Event",
    child_ready_event: "mp.Event",
) -> None:
    """Entry point of the spawned rollout-worker child process.

    The child runs the asyncio generate + score loops in isolation from the trainer's GIL. It never touches CUDA — only
    HTTP to vLLM and pure Python work (chat parsing, reward computation).
    """
    # Hide GPUs from the child: it has no business touching CUDA, and any
    # library that imports `torch` and lazily probes devices would race the
    # parent's allocator state. spawn already gave us a fresh process; this
    # makes it definitively CPU-only.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Strip distributed env so accelerate's `PartialState()` initialises in
    # single-process mode rather than trying to join the parent's process group.
    for _k in (
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
    ):
        os.environ.pop(_k, None)
    # Avoid sharing parent's flight-recorder dump path / NCCL debug log.
    os.environ.pop("TORCH_FR_DUMP_TEMP_FILE", None)
    os.environ.pop("NCCL_DEBUG_FILE", None)

    # `accelerate.logging.get_logger` requires `PartialState()` to have been
    # called at least once. Do it after env scrubbing so it picks up
    # single-process mode.
    from accelerate.state import PartialState

    PartialState()

    init_kwargs = pickle.loads(init_kwargs_pkl)
    worker = AsyncRolloutWorker(_run_in_child=True, **init_kwargs)
    # Inject the parent-owned IPC primitives. The mp.Value and mp.Queue can't
    # be pickled inside `init_kwargs_pkl` reliably — they're shared by being
    # passed as positional args to the Process target, which the multiprocessing
    # runtime sets up correctly across the spawn boundary.
    worker._model_version_value = model_version_value
    worker._stop_event_mp = stop_event
    worker._pause_event_mp = pause_event
    worker.rollout_buffer = _MPQueueChildShim(samples_queue)

    child_ready_event.set()

    # Translate the parent's mp.Event into the worker's asyncio.Event so a
    # call to `worker.stop()` in the parent breaks out of `_run_loops` here.
    def _stop_watcher():
        stop_event.wait()
        if worker._loop is not None and worker._stop_event is not None:
            try:
                worker._loop.call_soon_threadsafe(worker._stop_event.set)
            except Exception:
                pass

    threading.Thread(target=_stop_watcher, daemon=True, name="grpo-mp-stop-watcher").start()

    try:
        worker._run()
    except Exception:
        # Print the traceback so it shows up in the child's stderr. The parent
        # already monitors `_process.is_alive()` and reports exit codes from
        # `start()` / `stop()`, but the traceback is what's actually useful.
        traceback.print_exc()
        raise


class AsyncRolloutWorker:
    """Asynchronous rollout worker for AsyncGRPO.

    The worker generates samples in an OS-level child process so its Python GIL is isolated from the trainer's autograd
    engine. At Qwen3-30B-A3B scale with 16 k completion length, the trainer's backward pass holds the GIL in 1–5 s
    bursts; running rollout in the same process would starve rank 0's CUDA progress and time out NCCL collectives on
    other ranks.

    Architectural split:

      Parent (trainer process):
        - `init_weight_transfer()` — binds CUDA IPC pages to vLLM
        - `send_weights(iterator)` — broadcasts `model.named_parameters()` via NCCL; iterator must yield (name, GPU
          tensor) tuples
        - `pause()` / `resume()` — HTTP to vLLM
        - reads scored samples from `rollout_buffer` (an mp.Queue shim)

      Child (spawned process):
        - `_generate_loop` — async HTTP to vLLM
        - `_score_loop` — chat-template parsing + reward functions
                             (the GIL holders)
        - pushes scored `RolloutSample`s into the shared `mp.Queue`

    Call sequence the trainer must follow:

        worker = AsyncRolloutWorker(...) # cheap; no CUDA, no spawn ... trainer.accelerator.prepare(...)
        worker.init_weight_transfer() # NCCL setup with vLLM trainer._sync_weight() # cold sync to vLLM worker.start()
        # spawn the child process ... training loop ... worker.stop() # signal + join the child

    Deferring `init_weight_transfer()` until after `accelerator.prepare()` is required under DeepSpeed-Z2:
    `Stage1And2ZeroOptimizer.__init__` calls `torch.cuda.empty_cache()` which traverses the rank-0 allocator and would
    trip on any IPC-bound pages set up before prepare. Doing it after prepare is harmless under FSDP2 / single-node
    setups too, so the trainer always defers regardless of backend.
    """

    def __init__(
        self,
        model_name: str,
        dataset: Dataset,
        reward_funcs: list[Callable[..., list[float]]],
        processing_class: PreTrainedTokenizerBase,
        tools: list[Callable] | None = None,
        environment_factory: Callable[[], object] | None = None,
        num_generations: int = 8,
        max_inflight_tasks: int = 128,
        queue_maxsize: int = 0,
        vllm_server_url: str = "http://localhost:8000",
        max_tokens: int = 32,
        temperature: float = 1.0,
        request_timeout: int = 120,
        server_timeout: float = 240.0,
        chat_template_kwargs: dict[str, Any] | None = None,
        max_tool_calling_iterations: int | None = None,
        log_completions: bool = False,
        num_completions_to_print: int = 3,
        weight_names: list[str] | None = None,
        weight_dtype_names: list[str] | None = None,
        weight_shapes: list[list[int]] | None = None,
        init_weight_transfer_timeout: int = 1800,
        child_ready_timeout: int = 300,
        # Internal flag: True when the worker is being constructed inside the
        # spawned child process (`_child_main`). Skips parent-only IPC setup.
        _run_in_child: bool = False,
    ):
        if not is_vllm_available(min_version="0.17.1"):
            raise ImportError(
                "vLLM >= 0.17.1 is required to use AsyncRolloutWorker. Install it with: pip install 'vllm>=0.17.1'"
            )

        # ---- state shared by parent and child ----
        self.model_name = model_name
        self.max_tool_calling_iterations = max_tool_calling_iterations
        self.dataset = dataset
        self._dataset_iter = iter(dataset)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop_event: asyncio.Event | None = None
        self._weight_update_info = {
            "names": weight_names,
            "dtype_names": weight_dtype_names,
            "shapes": weight_shapes,
            "packed": True,
            "is_checkpoint_format": True,
        }

        self.reward_funcs = reward_funcs
        self.reward_func_names = [f.__name__ for f in reward_funcs]
        self.num_generations = num_generations
        self.max_inflight_tasks = max_inflight_tasks
        self.environments = None
        environment_methods = [[] for _ in range(self.max_inflight_tasks)]
        if environment_factory is not None:
            self.environments = [environment_factory() for _ in range(self.max_inflight_tasks)]
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
        self._sync_tool_dicts = [{} for _ in range(self.max_inflight_tasks)]
        for i in range(self.max_inflight_tasks):
            for tool in base_tools + (environment_methods[i] if self.environments is not None else []):
                if inspect.iscoroutinefunction(tool):
                    raise ValueError("Asynchronous tools are not supported in AsyncRolloutWorker yet.")
                self._sync_tool_dicts[i][tool.__name__] = tool
        self.tools = base_tools + (environment_methods[0] if self.environments is not None else [])

        self.vllm_server_url = vllm_server_url.rstrip("/")
        self.model_update_group = None
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.request_timeout = request_timeout
        self.server_timeout = server_timeout
        self.chat_template_kwargs = chat_template_kwargs or {}
        self.log_completions = log_completions
        self.num_completions_to_print = num_completions_to_print
        self.tokenizer = processing_class
        self.tokenizer = add_response_schema(self.tokenizer)
        # In multi-turn training, the chat template *must* be prefix-preserving. If the tokenizer's original template
        # isn't, we replace it at initialization with a training-safe, prefix-preserving template.
        if self.tools and not is_chat_template_prefix_preserving(self.tokenizer):
            self.chat_template = get_training_chat_template(self.tokenizer)
        else:
            self.chat_template = None

        self._groups_to_score: asyncio.Queue[RolloutGroup | None] = asyncio.Queue(maxsize=16)
        self._total_completion_tokens = 0
        self._total_groups_scored = 0
        self._generation_start_time: float | None = None
        self.session = None

        self.init_weight_transfer_timeout = init_weight_transfer_timeout
        self.child_ready_timeout = child_ready_timeout
        self._is_child = _run_in_child

        if _run_in_child:
            # Child: IPC primitives are patched by `_child_main` after construction.
            # `rollout_buffer` is overridden there with the shared mp.Queue shim.
            self._mp_ctx = None
            self._samples_queue = None
            self._model_version_value = None
            self._stop_event_mp = None
            self._pause_event_mp = None
            self._child_ready_event = None
            self.rollout_buffer = None
            self._child_init_kwargs = None
            self._process: mp.Process | None = None
            return

        # Parent: build IPC primitives that will be shared with the child.
        ctx = mp.get_context("spawn")
        self._mp_ctx = ctx
        self._samples_queue = ctx.Queue(maxsize=queue_maxsize)
        self._model_version_value = ctx.Value("i", 0)
        self._stop_event_mp = ctx.Event()
        self._pause_event_mp = ctx.Event()
        self._child_ready_event = ctx.Event()
        # The trainer's `RolloutQueueDataset` reads samples through this attribute.
        self.rollout_buffer = _MPQueueParentShim(self._samples_queue)
        # Kwargs that re-construct the worker inside the child. Pickled once at
        # `start()` time so all changes between __init__ and start() are captured.
        self._child_init_kwargs = dict(
            model_name=model_name,
            dataset=dataset,
            reward_funcs=reward_funcs,
            processing_class=processing_class,
            tools=tools,
            environment_factory=environment_factory,
            num_generations=num_generations,
            max_inflight_tasks=max_inflight_tasks,
            queue_maxsize=queue_maxsize,
            vllm_server_url=vllm_server_url,
            max_tokens=max_tokens,
            temperature=temperature,
            request_timeout=request_timeout,
            server_timeout=server_timeout,
            chat_template_kwargs=chat_template_kwargs,
            max_tool_calling_iterations=max_tool_calling_iterations,
            log_completions=log_completions,
            num_completions_to_print=num_completions_to_print,
            weight_names=weight_names,
            weight_dtype_names=weight_dtype_names,
            weight_shapes=weight_shapes,
            init_weight_transfer_timeout=init_weight_transfer_timeout,
            child_ready_timeout=child_ready_timeout,
        )
        self._process: mp.Process | None = None

    # ---------------- model_version (parent writes; child reads via mp.Value) ----------------

    @property
    def model_version(self) -> int:
        if self._model_version_value is None:
            return 0
        return int(self._model_version_value.value)

    @model_version.setter
    def model_version(self, value: int) -> None:
        if self._model_version_value is None:
            return
        with self._model_version_value.get_lock():
            self._model_version_value.value = int(value)

    def update_model_version(self, model_version: int) -> None:
        self.model_version = model_version

    # ---------------- parent-side weight transfer ----------------

    def _wait_for_server_ready_sync(self, timeout_s: float = 240.0, poll_interval_s: float = 2.0) -> None:
        """Block until the vLLM server is healthy."""
        logger.info(f"Waiting for vLLM server at {self.vllm_server_url} ...")
        start = time.time()
        while True:
            elapsed = time.time() - start
            try:
                response = requests.get(f"{self.vllm_server_url}/health", timeout=5)
                if response.status_code == 200:
                    logger.info(f"vLLM server ready after {elapsed:.1f}s")
                    return
            except (requests.ConnectionError, requests.Timeout, OSError):
                pass
            if elapsed >= timeout_s:
                raise TimeoutError(
                    f"Timed out after {timeout_s:.0f}s waiting for vLLM server at {self.vllm_server_url}. "
                    "Make sure the vLLM server is running and reachable. If the server needs more time to load "
                    "the model, increase `vllm_server_timeout` in your AsyncGRPOConfig."
                )
            if int(elapsed) % 10 < poll_interval_s:
                logger.info(f"Still waiting for vLLM server... ({elapsed:.0f}s)")
            time.sleep(poll_interval_s)

    def init_weight_transfer(self) -> None:
        """Wait for vLLM and set up the NCCL weight-transfer process group.

        Called by the trainer from the `on_train_begin` callback, AFTER `accelerator.prepare()`. Binding the SPG's CUDA
        IPC pages before prepare would crash DeepSpeed-Z2's `Stage1And2ZeroOptimizer.__init__ →
        torch.cuda.empty_cache()` with `cudaErrorIllegalAddress`.

        No-op when called from the child process — parent owns the group.
        """
        if self._is_child:
            return
        self._wait_for_server_ready_sync(timeout_s=self.server_timeout)
        response = requests.get(f"{self.vllm_server_url}/get_world_size")
        inference_world_size = response.json()["world_size"]
        world_size = inference_world_size + 1
        master_address = get_ip()
        master_port = get_open_port()
        init_info = {
            "master_address": master_address,
            "master_port": master_port,
            "rank_offset": 1,
            "world_size": world_size,
        }
        t_init = threading.Thread(
            target=requests.post,
            args=(f"{self.vllm_server_url}/init_weight_transfer_engine",),
            kwargs={"json": {"init_info": init_info}, "timeout": self.init_weight_transfer_timeout},
        )
        t_init.start()
        self.model_update_group = NCCLWeightTransferEngine.trainer_init(
            {
                "master_address": master_address,
                "master_port": master_port,
                "world_size": world_size,
            }
        )
        t_init.join()
        logger.info("Initialised weight-transfer NCCL group with vLLM")

    def send_weights(self, iterator) -> None:
        """Broadcast model weights to vLLM. Parent-only — uses parent's NCCL group
        and the trainer's GPU tensors via the supplied iterator."""
        if self.model_update_group is None:
            return
        t0 = time.time()
        t_update = threading.Thread(
            target=requests.post,
            args=(f"{self.vllm_server_url}/update_weights",),
            kwargs={"json": {"update_info": self._weight_update_info}, "timeout": 1800},
        )
        t_update.start()
        logger.debug(f"[weight_sync] /update_weights POST sent ({time.time() - t0:.1f}s)")
        t_nccl = time.time()
        NCCLWeightTransferEngine.trainer_send_weights(
            iterator=iterator,
            trainer_args=NCCLTrainerSendWeightsArgs(group=self.model_update_group, packed=True),
        )
        logger.debug(f"[weight_sync] NCCL transfer took {time.time() - t_nccl:.1f}s")
        t_join = time.time()
        t_update.join()
        logger.debug(
            f"[weight_sync] /update_weights join took {time.time() - t_join:.1f}s "
            f"(total send_weights: {time.time() - t0:.1f}s)"
        )

    def _destroy_model_update_group(self) -> None:
        # Critical at shutdown: leaking the SPG store / socket on exit produces
        # noisy stack traces from NCCL teardown.
        if self.model_update_group is None:
            return
        self.model_update_group.group.store = None
        self.model_update_group.group.socket = None
        self.model_update_group = None

    # ---------------- lifecycle ----------------

    def start(self) -> None:
        """Spawn the rollout-worker child process. Idempotent."""
        if self._is_child:
            # Child reaches its loop via `_child_main → worker._run()` directly.
            return
        if self._process is not None:
            logger.warning("AsyncRolloutWorker.start() called but child process is already running; ignoring.")
            return
        init_kwargs_pkl = pickle.dumps(self._child_init_kwargs)
        self._process = self._mp_ctx.Process(
            target=_child_main,
            args=(
                init_kwargs_pkl,
                self._samples_queue,
                self._model_version_value,
                self._stop_event_mp,
                self._pause_event_mp,
                self._child_ready_event,
            ),
            name="grpo-rollout-worker-child",
            daemon=True,
        )
        self._process.start()
        logger.info(
            f"AsyncRolloutWorker spawned child pid={self._process.pid}; "
            f"waiting up to {self.child_ready_timeout}s for the ready signal"
        )
        # spawn re-imports the parent script in the child (torch, transformers,
        # accelerate, trl, vllm — 60–90 s at Qwen3-30B-A3B scale). Poll the
        # child's liveness so an early crash surfaces immediately rather than
        # after the full timeout.
        deadline = time.monotonic() + self.child_ready_timeout
        while not self._child_ready_event.wait(timeout=1.0):
            if not self._process.is_alive():
                exit_code = self._process.exitcode
                self._process = None  # so a retry can re-spawn
                raise RuntimeError(
                    f"AsyncRolloutWorker child exited during init (exitcode={exit_code}). "
                    "Check the child's stderr for the traceback."
                )
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"AsyncRolloutWorker child did not signal ready within {self.child_ready_timeout}s. "
                    "spawn re-imports torch+transformers+trl+vllm in the child — slow at first launch; "
                    "increase `child_ready_timeout` if needed."
                )
        logger.info("AsyncRolloutWorker child is ready")

    def stop(self) -> None:
        """Signal the child to stop, join the process, and destroy the NCCL group."""
        if self._is_child:
            return
        if self._process is None:
            self._destroy_model_update_group()
            return
        logger.info("Stopping AsyncRolloutWorker child process...")
        self._stop_event_mp.set()
        # Guard: if start() raised before `Process.start()` returned (e.g. pickle
        # failure during spawn), `_process` is set but `_popen` is None. Calling
        # `.join()` then asserts and masks the real exception.
        if self._process._popen is not None:
            self._process.join(timeout=15)
            if self._process.is_alive():
                logger.warning("Child did not exit within 15s; terminating.")
                self._process.terminate()
                self._process.join(timeout=5)
                if self._process.is_alive():
                    self._process.kill()
        self._process = None
        self._destroy_model_update_group()

    def pause(self) -> None:
        """Pause vLLM via HTTP. Called from the parent before each weight sync."""
        if not self._is_child:
            self._pause_event_mp.set()
        t0 = time.time()
        requests.post(f"{self.vllm_server_url}/pause", params={"mode": "keep"})
        logger.debug(f"[weight_sync] pause HTTP took {time.time() - t0:.1f}s")

    def resume(self) -> None:
        if not self._is_child:
            self._pause_event_mp.clear()
        t0 = time.time()
        requests.post(f"{self.vllm_server_url}/resume")
        logger.debug(f"[weight_sync] resume HTTP took {time.time() - t0:.1f}s")

    # ---------------- child-side asyncio loops ----------------

    async def _run_loops(self, stop_event: asyncio.Event) -> None:
        async with aiohttp.ClientSession() as session:
            self.session = session
            logger.info(
                f"vllm worker started: num_generations={self.num_generations}, "
                f"max_inflight_tasks={self.max_inflight_tasks}"
            )
            await asyncio.gather(
                asyncio.create_task(self._generate_loop(stop_event=stop_event)),
                asyncio.create_task(self._score_loop(stop_event=stop_event)),
            )

    def _run(self) -> None:
        """Run the asyncio event loop until `stop_event` fires. Child-only."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._stop_event = asyncio.Event()
        try:
            loop.run_until_complete(self._run_loops(stop_event=self._stop_event))
        except Exception as e:
            logger.exception(f"Worker process failed: {e}")
            raise
        finally:
            loop.close()

    async def _generate_loop(self, stop_event: asyncio.Event) -> None:
        pending_groups: dict[int, RolloutGroup] = {}
        pending_completed: dict[int, int] = {}
        inflight_tasks: dict[asyncio.Task, tuple[int, int]] = {}
        free_slots = set(range(self.max_inflight_tasks))
        work_iter = self._repeat_iterator()

        self._generation_start_time = time.monotonic()
        try:
            while True:
                self._last_heartbeat = time.monotonic()
                while free_slots and not stop_event.is_set():
                    group_id, row = next(work_iter)
                    if group_id not in pending_groups:
                        prompt = row["prompt"]
                        prompt_ids = self.tokenizer.apply_chat_template(
                            prompt,
                            return_dict=False,
                            add_generation_prompt=True,
                            tools=self.tools or None,  # `or None`: Llama bug: it renders tool boilerplate for tools=[]
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
                        logger.debug(f"Started group {group_id}; pending_groups={len(pending_groups)}")

                    slot = free_slots.pop()
                    if self.environments is not None:
                        # Current assumption: reset side effects matter, return value is ignored.
                        self.environments[slot].reset(**row)

                    logger.info(f"[slot] assigned slot={slot} group={group_id} free_after={len(free_slots)}")
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
                    if not free_slots:
                        logger.debug(
                            f"[generate] all {self.max_inflight_tasks} slots busy, "
                            f"pending_groups={len(pending_groups)}, waiting for completions..."
                        )
                    continue

                for task in done:
                    group_id, slot = inflight_tasks.pop(task)
                    free_slots.add(slot)
                    logger.debug(f"[slot] freed   slot={slot} group={group_id} free_after={len(free_slots)}")
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
                    # TODO: move this in generation task, shouldn't matter but is correct
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
                        logger.debug(f"Group {group_id} complete; queued_for_scoring={self._groups_to_score.qsize()}")
                        del pending_groups[group_id]
                        del pending_completed[group_id]
        finally:
            for task in inflight_tasks:
                task.cancel()
            if inflight_tasks:
                await asyncio.gather(*inflight_tasks, return_exceptions=True)
            # Use put_nowait: if the queue is full at shutdown, skip the sentinel —
            # _score_loop will exit via stop_event check in its outer loop.
            try:
                self._groups_to_score.put_nowait(None)
            except asyncio.QueueFull:
                pass

    def _compute_rollout_metrics(self, samples: list[RolloutSample], scoring_time: float, wait_scoring: float) -> None:
        assert self._generation_start_time is not None, "generation_start_time init in run()"
        elapsed = time.monotonic() - self._generation_start_time
        generation_tok_per_sec = self._total_completion_tokens / elapsed if elapsed > 0 else 0.0

        scoring_time_ms = scoring_time * 1000
        wait_scoring_ms = wait_scoring * 1000

        for sample in samples:
            sample.metrics["generation_tok_per_s"] = generation_tok_per_sec
            sample.metrics["scoring_time_ms"] = scoring_time_ms
            sample.metrics["wait_scoring_ms"] = wait_scoring_ms
            sample.metrics["buffer_qsize"] = self.rollout_buffer.qsize()

        logger.info(
            f"[inference] total_completion_tokens={self._total_completion_tokens}, "
            f"generation_tok/s={generation_tok_per_sec:.1f}, scoring_time={scoring_time_ms:.1f}ms, "
            f"wait_scoring={wait_scoring_ms:.1f}ms"
        )

    async def _score_loop(self, stop_event: asyncio.Event) -> None:
        while not stop_event.is_set():
            self._last_heartbeat = time.monotonic()
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
                        # Wait for trainer to consume loop
                        logger.info(
                            f"[score] rollout buffer full (maxsize={self.rollout_buffer.maxsize}), "
                            "waiting for trainer to consume..."
                        )
                        await asyncio.sleep(0.1)

            logger.debug(
                f"Scored group with {len(samples)} samples; rollout_buffer_qsize={self.rollout_buffer.qsize()}"
            )

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
            tools=self.tools or None,  # `or None`: Llama bug: it renders tool boilerplate for tools=[]
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
        """Get token IDs for tool result formatting by using a minimal dummy conversation."""
        # Use the real tool name instead of a dummy: some templates (e.g. GPT-OSS) derive the tool response
        # header from the assistant's tool call name.
        dummy_tool_calls = [{"type": "function", "function": {"name": tool_messages[0]["name"], "arguments": {}}}]
        dummy_messages = [
            {"role": "user", "content": "dummy"},
            {
                "role": "assistant",
                # "content" is required here because VLM processors crash on tokenize=True without it
                # (KeyError in processing_utils.py). See huggingface/transformers#45290.
                "content": "",
                "tool_calls": dummy_tool_calls,
            },
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

        # Some chat templates (notably Qwen3/Qwen3.5) render "...<|im_end|>\n" after an assistant/tool block.
        # When we compute `suffix_ids` by slicing `full_ids`, we must align the slicing boundary to
        # EOS (not EOS + newline). Templates that don't use EOS as end-of-turn (e.g. Gemma uses
        # <turn|>) skip this trimming.
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
        while True:
            try:
                output = await self._post("/v1/completions", payload, self.request_timeout)
                break
            except (aiohttp.ServerDisconnectedError, aiohttp.ClientConnectionError, aiohttp.ClientResponseError):
                # vLLM drops connections or returns 503 during weight sync (/pause). Wait briefly and retry.
                logger.debug("Server unavailable (likely weight sync pause), retrying...")
                await asyncio.sleep(1.0)
        choice = output["choices"][0]
        completion_ids = choice["token_ids"]
        completion_logprobs = choice["logprobs"]["token_logprobs"]
        return completion_ids, completion_logprobs

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

        # Sum rewards across all reward functions. Reward functions may return None for individual
        # samples (e.g. accuracy_reward when the gold solution is unparseable). Convert None → nan
        # and use nansum so that a None from one function doesn't affect the others, matching TRL.
        all_rewards = [[r if r is not None else float("nan") for r in row] for row in all_rewards]
        rewards = np.nansum(np.array(all_rewards, dtype=float), axis=0)
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        reward_mean = float(rewards.mean())
        reward_std = float(rewards.std())
        logger.info(f"Rollout metrics: reward_mean={reward_mean:.4f}, reward_std={reward_std:.4f}")

        # tools/call_frequency: mean calls per completion (matches TRL's total_calls / num_completions)
        # tools/failure_frequency: per-completion failure rate; averaged across samples in compute_loss
        #   (TRL uses total_failures / total_calls, ours weights equally per completion — close enough)
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

        per_func_rewards = np.array(all_rewards, dtype=float)  # shape (num_funcs, num_completions)

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
        for attempt in range(max_retries):
            try:
                async with self.session.post(
                    f"{self.vllm_server_url}{path}", json=payload, timeout=client_timeout
                ) as response:
                    response.raise_for_status()
                    content = await response.json()
                    return content if content else {}
            except (TimeoutError, asyncio.TimeoutError):
                if attempt < max_retries - 1:
                    logger.warning(f"POST {path} timed out (attempt {attempt + 1}/{max_retries}), retrying...")
                    await asyncio.sleep(1)
                else:
                    raise
