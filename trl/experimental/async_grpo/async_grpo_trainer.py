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


import contextvars
import math
import queue
import textwrap
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from multiprocessing.queues import Queue as MPQueue
from typing import Any, Protocol

import torch
from accelerate.logging import get_logger
from datasets import Dataset, IterableDataset
from torch.distributed._tensor import DTensor
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase, TrainerCallback
from transformers.data.data_collator import DataCollatorMixin

from ...trainer.base_trainer import _BaseTrainer
from ...trainer.utils import (
    compute_flops_per_token,
    compute_mfu,
    get_config_model_id,
    is_trackio_available,
    nanmax,
    nanmin,
    pad,
    patch_chunked_lm_head,
)
from .async_grpo_config import AsyncGRPOConfig
from .async_rollout_worker import AsyncRolloutWorker, RolloutSample
from .vllm_client import VLLMClient
from .weight_transfer import WeightTransferClient


logger = get_logger(__name__)

if is_trackio_available():
    import trackio

# A reward function is a callable that returns a list of floats (the rewards). The callable receives prompts,
# completions, and additional arguments from the trainer (refer to the trainer's source for details). To ensure forward
# compatibility, it should accept **kwargs.
RewardFunc = Callable[..., list[float]]

# One logged value is either a float or a `(numerator, denominator)` pair; see `_reduce_metric`.
MetricValue = float | tuple[float, float]


def _reduce_metric(key: str, values: list[MetricValue]) -> float:
    """Reduce one logging window into the number that gets logged.

    The value's shape and the key's suffix declare the reduction, so nothing has to be registered or configured:

    - `(numerator, denominator)` pairs are a rate, reduced as Σnum / Σden. A rate is never stored *as* a rate, which is
      what makes a mean-of-ratios impossible — the defect that made `training_tok/s` unusable.
    - a `total` word in the name is a counter, summed over the window (producers push deltas).
    - a `max` or `min` word is an extremum. Matching whole words rather than a suffix keeps the upstream spellings
      (`completions/max_length`, `clip_ratio/high_max`) working alongside this trainer's own (`row_tokens_max`).
    - anything else is a gauge, and its window mean is meaningful.

    A key must always be logged with the same shape, since the reduction is picked from the first value seen.
    """
    if isinstance(values[0], tuple):
        numerator, denominator = (sum(v) for v in zip(*values, strict=True))
        return numerator / denominator if denominator else float("nan")
    words = key.split("/")[-1].split("_")
    if "total" in words:
        return sum(values)
    if "max" in words:
        return max(values)
    if "min" in words:
        return min(values)
    return sum(values) / len(values)


class _SupportsReset(Protocol):
    def reset(self, **kwargs) -> str | None: ...


EnvironmentFactory = Callable[[], _SupportsReset]


class RolloutWorkerProtocol(Protocol):
    """Interface a rollout worker must implement to be passed as `rollout_worker` to [`AsyncGRPOTrainer`].

    The default [`AsyncRolloutWorker`] spawns a CUDA-free child process and scores completions with the trainer's
    `reward_funcs`. Implement this protocol to plug in a custom rollout/scoring backend instead — for example, one that
    runs reward models on their own GPUs.

    Attributes:
        rollout_buffer (`queue.Queue` or `multiprocessing.queues.Queue`):
            Queue the trainer drains; the worker pushes scored `RolloutSample`s onto it. The two queue types are
            structurally identical (`get` / `put_nowait` / `qsize`) but nominally unrelated, so both are allowed: the
            default [`AsyncRolloutWorker`] runs its loop in a spawned process and uses `multiprocessing.Queue`, while
            an in-process worker uses `queue.Queue`.
        metrics_queue (`queue.Queue` or `multiprocessing.queues.Queue`):
            Queue the trainer drains in `log()` for metrics the worker measured itself. Each item is one dict shaped
            like the trainer's metric sink — `{key: float}` for a gauge or a counter, `{key: (numerator, denominator)}`
            for a rate — so draining it is an append. A worker that measures nothing exposes an empty queue.
    """

    rollout_buffer: queue.Queue | MPQueue
    metrics_queue: queue.Queue | MPQueue

    def start(self) -> None:
        """Begin producing rollouts. Called once on train begin, after the initial weight sync."""
        ...

    def stop(self) -> None:
        """Stop the worker and release its resources. Called on train end."""
        ...

    def update_model_version(self, model_version: int) -> None:
        """Tell the worker which policy version is now live, so it can tag or discard stale samples."""
        ...

    def check_health(self, stale_after_s: float) -> None:
        """Raise if the worker has crashed or stopped producing within `stale_after_s` seconds."""
        ...


class WeightTransferProtocol(Protocol):
    """Interface a weight-sync backend must implement to be passed as `weight_transfer` to [`AsyncGRPOTrainer`].

    The default [`WeightTransferClient`] streams the trainer's weights into the vLLM server over NCCL. Implement this
    protocol to plug in a different sync mechanism, or pass a no-op implementation to disable trainer-side weight sync
    (e.g. when a custom `rollout_worker` updates the policy itself).
    """

    def init_weight_transfer(self) -> None:
        """Set up the transfer (e.g. the NCCL group). Called once on train begin, before the first sync."""
        ...

    def pause(self) -> None:
        """Pause the inference server before weights are swapped in."""
        ...

    def send_weights(self, iterator: Iterator[tuple[str, torch.Tensor]]) -> None:
        """Stream `(name, tensor)` pairs from `iterator` to the inference server."""
        ...

    def resume(self) -> None:
        """Resume the inference server after the weights are updated."""
        ...

    def destroy(self) -> None:
        """Release transfer resources. Called on train end."""
        ...


class StepIntervalCallback(TrainerCallback):
    """
    A callback that calls a function every N optimization steps.
    """

    def __init__(self, fn, every_n_steps: int):
        self.fn = fn
        self.every_n_steps = every_n_steps

    def on_step_end(self, _args, state, _control, **_kwargs):
        if state.global_step % self.every_n_steps == 0:
            self.fn()


class _OptimizerTimeCallback(TrainerCallback):
    """Times the optimizer step, which `training_step` cannot see.

    Without it, clip + step + zero_grad land in the same unmeasured gap as the rollout-queue wait and get read as
    starvation. Covers `optimizer.step()` only, not the gradient clipping that precedes it.
    """

    def __init__(self, trainer: "AsyncGRPOTrainer"):
        self._trainer = trainer
        self._t0 = None

    def on_pre_optimizer_step(self, _args, _state, _control, **_kwargs):
        self._t0 = time.perf_counter()

    def on_optimizer_step(self, _args, _state, _control, **_kwargs):
        self._trainer._step_optimizer_s += time.perf_counter() - self._t0


class _TrainBeginCallback(TrainerCallback):
    """Idempotent train-begin setup: NCCL group setup + cold weight sync to vLLM, then start the rollout worker.
    The weight sync must complete before the worker starts, which the ordering here guarantees.
    """

    def __init__(self, trainer: "AsyncGRPOTrainer"):
        self._trainer = trainer
        self._fired = False

    def on_train_begin(self, _args, _state, _control, **_kwargs):
        if self._fired:
            return
        self._fired = True
        if self._trainer.accelerator.is_main_process and self._trainer.weight_transfer is not None:
            self._trainer.weight_transfer.init_weight_transfer()
        self._trainer._sync_weight()
        if self._trainer.accelerator.is_main_process and self._trainer.rollout_worker is not None:
            self._trainer.rollout_worker.start()


class _EpochStopCallback(TrainerCallback):
    """Stop after `num_train_epochs` full passes over the prompt dataset.

    An epoch is counted in distinct prompt-groups actually trained (accumulated in the collator, which runs on the main
    process just before the model forward). This is fork-independent: all generations of a prompt and all forked rows
    of a conversation share one `group_id`, so a conversation forking into many rows still counts once. Only the main
    process collates (`dispatch_batches=True`), so the stop decision is reduced across ranks to keep data-parallel
    workers in lockstep.
    """

    def __init__(self, trainer: "AsyncGRPOTrainer", target_groups: int):
        self._trainer = trainer
        self._target = target_groups

    def on_step_end(self, _args, _state, control, **_kwargs):
        acc = self._trainer.accelerator
        reached = torch.tensor(int(len(self._trainer._trained_groups) >= self._target), device=acc.device)
        if int(acc.reduce(reached, reduction="sum").item()) >= 1:
            control.should_training_stop = True


def log_rollout_traces(samples: list[RolloutSample], step: int, report_to: list[str], max_traces: int = 16) -> None:
    """Log rollout samples to trackio as inspectable traces (prompt + completion + reward/advantage per sample).

    Call from rank 0 during training, where the HF trackio callback has already initialised the run; the traces then
    show up under the run's Traces tab so rollouts can be read directly instead of grepping logs. No-op unless trackio
    is the active logging backend (installed and listed in `report_to`). Best-effort: a trackio hiccup must never break
    training.

    Args:
        samples (`list[RolloutSample]`):
            Consumed rollout samples to log; the first `max_traces` are recorded.
        step (`int`):
            Step value the traces are logged under (e.g. the policy version, so the UI groups by policy).
        report_to (`list[str]`):
            The training args' `report_to`; logging happens only when it contains `"trackio"`.
        max_traces (`int`, *optional*, defaults to `16`):
            Maximum number of traces to log per call.
    """
    if not samples or "trackio" not in report_to or not is_trackio_available():
        return
    try:
        traces = [
            trackio.Trace(
                messages=list(sample.prompt) + list(sample.completion),
                metadata={
                    **sample.metrics,  # reward, reward_std, rewards/<func>, tools/call_frequency, tools/failure_frequency
                    "advantage": float(sample.advantage),
                    "group_id": int(sample.group_id),
                    "model_version": int(sample.model_version),
                    "prompt_tokens": len(sample.input_ids) - int(sum(sample.completion_mask)),
                    "completion_tokens": int(sum(sample.completion_mask)),
                },
            )
            for sample in samples[:max_traces]
        ]
        trackio.log({"rollouts": traces}, step=step)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"rollout trace logging skipped: {type(e).__name__}: {e}")


class RolloutQueueDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        rollout_queue,
        model_version_fn,
        check_health_fn,
        stale_after_s,
        metrics,
        max_staleness=3,
        poll_interval_s=5.0,
        report_to=None,
    ):
        self.queue = rollout_queue
        self.model_version_fn = model_version_fn
        self.check_health_fn = check_health_fn
        self.stale_after_s = stale_after_s
        # The trainer's metric sink, shared by reference. This dataset lives in the main process (`num_workers=0`,
        # `dispatch_batches=True`), which is also the only process where the queue wait is real, so its metrics need no
        # communication at all — they are appended straight into the sink the trainer reduces in `log()`.
        self.metrics = metrics
        # Blocking-get time, accumulated here and flushed by `training_step` at the optimizer-step boundary — the only
        # place that knows when a step's worth of waiting is done.
        self.wait_s = 0.0
        self.max_staleness = max_staleness
        self.poll_interval_s = poll_interval_s
        self.report_to = report_to or []
        self._trace_buf: list = []
        # Traces exist to be read, not to cover the run: a handful per sampled policy version is enough to eyeball
        # what the generator produced. Both numbers are therefore per *policy version* — `_traces_per_log` samples,
        # once every `_trace_log_interval` versions. Flushing every `_trace_log_interval` *samples* instead fired
        # ~60x per optimizer step at 484 samples/step, and since trackio writes a `metrics` row for every `log()`
        # call — empty, because the payload is all traces — the real per-step metrics ended up buried under ~60x
        # their own number of `{}` rows, which is what made the dashboard slow to query.
        self._traces_per_log = 8
        self._trace_log_interval = 8
        self._last_traced_version = -1
        # Log traces off the training path: __iter__ enqueues, a daemon thread drains. Bounded + drop-on-full so a
        # slow trackio backend never blocks sample delivery.
        self._trace_queue: queue.Queue = queue.Queue(maxsize=4)
        threading.Thread(target=self._drain_traces, name="rollout-trace-logger", daemon=True).start()

    def _drain_traces(self):
        while True:
            samples, step, ctx = self._trace_queue.get()
            # Replay the main-thread context captured at enqueue: trackio's run lives in a ContextVar this daemon
            # thread wouldn't otherwise inherit, so trackio.log() would raise "call init() first".
            ctx.run(log_rollout_traces, samples, step=step, report_to=self.report_to)

    def __iter__(self):
        while True:
            t0 = time.time()
            while True:
                try:
                    sample = self.queue.get(timeout=self.poll_interval_s)
                    break
                except queue.Empty:
                    # Returning here would broadcast None through accelerate's dispatch loop.
                    self.check_health_fn(self.stale_after_s)
            now = time.time()
            wait_s = now - t0
            self.wait_s += wait_s
            if wait_s > 1.0:
                logger.info(f"waited {wait_s:.1f}s for sample (qsize={self.queue.qsize()})")

            version = self.model_version_fn()
            staleness = version - sample.model_version
            if staleness > self.max_staleness:
                logger.info(f"dropping stale sample (staleness={staleness}, max={self.max_staleness})")
                self.metrics["sample/dropped_stale_total"].append(1.0)
                continue  # drop stale, pull next

            # Three different views of the same queue, and they must not be confused with each other:
            #   `sample/rollout_queue_size`  how many scored samples are in it, counted where the TRAINER consumes
            #                                them rather than where the worker filled it
            #   `sample/time_in_queue_s`     how long THIS sample sat in it — its own idle time, and the seconds
            #                                half of its off-policyness
            #   `perf/rollout_wait_s`        how long the TRAINER sat blocked because the queue was empty
            # An empty queue with the trainer waiting is generation-bound; a full queue with no wait is trainer-bound
            # (and then `rollout/backpressure_s` is what generation lost to it).
            self.metrics["sample/rollout_queue_size"].append(float(self.queue.qsize()))
            if sample.enqueued_at is not None:
                self.metrics["sample/time_in_queue_s"].append(now - sample.enqueued_at)
            # Freshness
            self.metrics["sample/staleness_mean"].append(float(staleness))
            self.metrics["sample/staleness_max"].append(float(staleness))

            if version % self._trace_log_interval == 0 and version != self._last_traced_version:
                self._trace_buf.append(sample)
                if len(self._trace_buf) >= self._traces_per_log:
                    try:
                        # Capture the main-thread context (holds trackio's run) so the drain thread can replay it.
                        self._trace_queue.put_nowait((self._trace_buf, version, contextvars.copy_context()))
                    except queue.Full:
                        pass
                    self._trace_buf = []
                    self._last_traced_version = version

            yield {
                "input_ids": sample.input_ids,
                "completion_mask": sample.completion_mask,
                "old_log_probs": sample.old_log_probs,
                "advantage": sample.advantage,
                "group_id": sample.group_id,
                "metrics": sample.metrics,  # per-sample rewards; aggregated by the collator, never sent to the model
            }


def _balance_by_squared_length(examples: list[dict[str, Any]], num_groups: int) -> list[list[dict[str, Any]]]:
    """Greedily partition `examples` into `num_groups` rows (one per DP rank), balancing each row's Σ Lᵢ².

    Attention is O(L²) while the FFN is O(L), so equal token counts wouldn't equalize wall-time; balancing Σ Lᵢ² keeps
    the per-micro-batch all-reduce free of stragglers. Samples are placed longest-first into the row with the smallest
    running Σ Lᵢ² (LPT scheduling). With at least `num_groups` samples every row ends up non-empty.
    """
    groups = [[] for _ in range(num_groups)]
    squared_loads = [0] * num_groups
    for example in sorted(examples, key=lambda e: len(e["input_ids"]), reverse=True):
        n = len(example["input_ids"])
        i = min(range(num_groups), key=lambda j: squared_loads[j])
        groups[i].append(example)
        squared_loads[i] += n * n
    return groups


class FixedCountBatcher(torch.utils.data.IterableDataset):
    """Fixed-count batcher (the planner) wrapping [`RolloutQueueDataset`].

    Buffers `microbatch_size` (= `per_device_train_batch_size × num_processes`) samples, then partitions them across
    the `num_processes` rows (one per DP rank) balanced by Σ Lᵢ² (attention cost) so no rank straggles at the
    per-micro-batch all-reduce. The sample count is fixed, so this does not bound peak memory — use
    [`TokenBudgetBatcher`] for that. With `microbatch_size >= num_processes` every row is non-empty.

    Args:
        dataset ([`RolloutQueueDataset`]):
            Source yielding single rollout-sample dicts.
        num_processes (`int`):
            Number of DP ranks; the number of rows (one per rank) in each micro-batch.
        microbatch_size (`int`):
            Number of samples buffered into each micro-batch before it is partitioned and emitted.
    """

    def __init__(self, dataset: "RolloutQueueDataset", num_processes: int, microbatch_size: int):
        self.dataset = dataset
        self.num_processes = num_processes
        self.microbatch_size = microbatch_size

    def __iter__(self):
        batch = []
        for sample in self.dataset:
            batch.append(sample)
            if len(batch) == self.microbatch_size:
                yield _balance_by_squared_length(batch, self.num_processes)
                batch = []


class TokenBudgetBatcher(torch.utils.data.IterableDataset):
    """Token-budgeted dynamic batcher (the planner) wrapping [`RolloutQueueDataset`].

    Keeps `num_processes` open rows (one per DP rank) and pulls single samples from the source one at a time, dropping
    each into the row with the smallest running Σ Lᵢ² (attention cost) that still fits within `token_budget` tokens.
    When the next sample fits in no row, the current micro-batch is emitted — a list of `num_processes` groups, already
    partitioned per rank — and a fresh one is started with that sample. The number of samples per row is therefore
    dynamic: short samples pack many per row, long ones pack few, while every row stays within `token_budget` tokens.
    This bounds peak memory independently of `per_device_train_batch_size` and keeps the rows Σ Lᵢ²-balanced so no rank
    straggles at the per-micro-batch all-reduce.

    Every emitted micro-batch has all `num_processes` rows non-empty (a rank forwarding zero tokens would desync
    FSDP/EP collectives): a micro-batch is only closed once every row holds at least one sample. A sample longer than
    `token_budget` fits in no row, so it is dropped with a warning; set `token_budget` ≥ the vLLM server's
    `max_model_len` (the cap on prompt + completion) to avoid dropping samples.

    Args:
        dataset ([`RolloutQueueDataset`]):
            Source yielding single rollout-sample dicts.
        num_processes (`int`):
            Number of DP ranks; the number of rows (one per rank) in each micro-batch.
        token_budget (`int`):
            Maximum real tokens packed into a single row (one rank's forward).
        metrics (`dict`):
            The trainer's metric sink, appended to when a sample is dropped for exceeding the budget.
    """

    def __init__(self, dataset: "RolloutQueueDataset", num_processes: int, token_budget: int, metrics: dict):
        self.dataset = dataset
        self.num_processes = num_processes
        self.token_budget = token_budget
        self.metrics = metrics  # the trainer's sink, for the drop counter below

    def __iter__(self):
        rows = [[] for _ in range(self.num_processes)]
        squared_loads = [0] * self.num_processes  # Σ Lᵢ² per row, drives the balancing
        token_counts = [0] * self.num_processes  # tokens per row, drives the budget
        for sample in self.dataset:
            n = len(sample["input_ids"])
            if n > self.token_budget:
                # Longer than the whole budget: fits in no row, so drop it (placing it would overshoot the budget
                # or force an empty row that desyncs FSDP/EP collectives).
                logger.warning(
                    f"Dropping a rollout sample of {n} tokens that exceeds token_budget={self.token_budget}. "
                    "Raise token_budget to avoid dropping samples."
                )
                self.metrics["batch/dropped_oversize_total"].append(1.0)
                continue
            fits = [i for i in range(self.num_processes) if token_counts[i] + n <= self.token_budget]
            if not fits:
                # No row has room (all are non-empty, since this sample fits an empty one): close and reset.
                yield rows
                rows = [[] for _ in range(self.num_processes)]
                squared_loads = [0] * self.num_processes
                token_counts = [0] * self.num_processes
                fits = list(range(self.num_processes))
            i = min(fits, key=lambda j: squared_loads[j])
            rows[i].append(sample)
            squared_loads[i] += n * n
            token_counts[i] += n


class _EmptyIterableDataset(torch.utils.data.IterableDataset):
    """Placeholder for non-rank-0 processes. Never actually iterated."""

    def __iter__(self):
        return iter([])


@dataclass
class DataCollatorForRollout(DataCollatorMixin):
    """
    Padding-free collator (the packer) for rollout samples. Packs a micro-batch into `num_processes` rows (one per DP
    rank): each row concatenates its samples into a single sequence, with `position_ids` resetting per sequence and
    advantages expanded per-token. Rows are padded only to the longest row, so the batch stays rectangular for
    `DataLoaderDispatcher` to scatter row `i` -> rank `i`; this inter-rank padding is stripped per-rank in
    `compute_loss`.

    The micro-batch arrives already partitioned into `num_processes` rows by the upstream planner
    ([`FixedCountBatcher`] or [`TokenBudgetBatcher`]) — which balances each row's Σ Lᵢ² (attention cost) to avoid
    stragglers at the gradient all-reduce — so the collator only tensorizes the given rows.

    Args:
        pad_token_id (`int`):
            Token id used to pad `input_ids`.
        num_processes (`int`, *optional*, defaults to `1`):
            Number of DP ranks; the micro-batch is packed into this many rows.
        metrics (`dict[str, list]`, *optional*):
            The trainer's metric sink, appended to with this micro-batch's sample and packing metrics.
        token_budget (`int`, *optional*, defaults to `0`):
            Per-row token cap of the planner, or `0` when batching by fixed sample count.
    """

    pad_token_id: int
    num_processes: int = 1
    return_tensors: str = "pt"
    # Distinct prompt-group ids, it counts exactly the prompt-groups that get trained
    groups_trained: set[int] = field(default_factory=set)
    # The trainer's metric sink, shared by reference (like `groups_trained`).
    metrics: dict[str, list] = field(default_factory=lambda: defaultdict(list))
    # Per-row token cap of the planner,
    token_budget: int = 0

    def torch_call(self, examples: list[Any]) -> dict[str, Any]:
        # The dataloader uses batch_size=1 over a planner that pre-partitions each micro-batch into `num_processes`
        # rows, so `examples` is a length-1 list holding that single micro-batch (one group per rank).
        (groups,) = examples

        input_ids, attention_mask, completion_mask, old_log_probs, position_ids, advantages = [], [], [], [], [], []
        for group in groups:
            seq_lengths = [len(example["input_ids"]) for example in group]
            ids = [token for example in group for token in example["input_ids"]]
            input_ids.append(torch.tensor(ids, dtype=torch.long))
            attention_mask.append(torch.ones(len(ids), dtype=torch.long))
            completion_mask.append(
                torch.tensor([m for example in group for m in example["completion_mask"]], dtype=torch.long)
            )
            old_log_probs.append(
                torch.tensor([lp for example in group for lp in example["old_log_probs"]], dtype=torch.float32)
            )
            position_ids.append(torch.cat([torch.arange(n) for n in seq_lengths]))
            advantages.append(
                torch.cat(
                    [torch.full((n,), example["advantage"]) for example, n in zip(group, seq_lengths, strict=False)]
                )
            )

        input_ids = pad(input_ids, padding_value=self.pad_token_id)
        attention_mask = pad(attention_mask, padding_value=0)
        completion_mask = pad(completion_mask, padding_value=0)
        old_log_probs = pad(old_log_probs, padding_value=0.0)
        position_ids = pad(position_ids, padding_value=0)
        advantages = pad(advantages, padding_value=0.0)

        all_examples = [example for group in groups for example in group]
        self.groups_trained.update(example["group_id"] for example in all_examples)

        # Total valid completion tokens across all samples in the full batch.
        # Repeated per rank so that DataLoaderDispatcher (dispatch_batches=True) slices correctly on dim=0
        n_trained_tokens = sum(sum(example["completion_mask"]) for example in all_examples)
        global_n_tokens = torch.full((self.num_processes,), float(n_trained_tokens), dtype=torch.float32)

        sample_tokens = [len(example["input_ids"]) for example in all_examples]
        n_forward_tokens = sum(sample_tokens)
        mean_seq_len = n_forward_tokens / len(all_examples)
        global_n_forward_tokens = torch.full((self.num_processes,), float(n_forward_tokens), dtype=torch.float32)
        mean_seq_len_t = torch.full((self.num_processes,), float(mean_seq_len), dtype=torch.float32)

        self._log_metrics(groups=groups, all_examples=all_examples, padded=attention_mask)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "completion_mask": completion_mask,
            "old_log_probs": old_log_probs,
            "position_ids": position_ids,
            "advantages": advantages,
            "global_n_tokens": global_n_tokens,
            "global_n_forward_tokens": global_n_forward_tokens,
            "mean_seq_len": mean_seq_len_t,
        }

    def _log_metrics(
        self, groups: list[list[dict[str, Any]]], all_examples: list[dict[str, Any]], padded: torch.Tensor
    ) -> None:
        """Append this micro-batch's sample and packing metrics straight into the trainer's sink.

        Rank 0 holds the whole micro-batch, so the per-sample rewards can be aggregated here instead of being packed
        into NaN-padded tensors, broadcast to every rank and reduced back — which is what this trainer used to do to
        compute a number rank 0 already had.
        """
        # A training row's tokens split in two: those the loss is taken over (`completion_mask == 1`) and those merely
        # forwarded — the prompt, tool results, and any tail a realign demoted to context. FLOPs are spent on both, so
        # both are worth a number, and their ratio is what says how much of the forward earns no gradient.
        forwarded = [len(example["input_ids"]) for example in all_examples]
        trained = [sum(example["completion_mask"]) for example in all_examples]
        self.metrics["sample/forwarded_tokens_mean"].append(sum(forwarded) / len(forwarded))
        self.metrics["sample/forwarded_tokens_max"].append(float(max(forwarded)))
        self.metrics["sample/trained_tokens_mean"].append(sum(trained) / len(trained))
        self.metrics["batch/masked_token_frac"].append((sum(forwarded) - sum(trained), sum(forwarded)))

        # Per-sample rewards, nan-aware: a reward func may return None for an unscorable sample, and a sample for which
        # every func returned None carries NaN rather than a misleading 0.
        # Union of keys, not the first sample's: `tools/*` is stamped per group, so a micro-batch mixes samples that
        # have those keys with samples that do not. Each key averages over the samples that carry it.
        keys = dict.fromkeys(key for example in all_examples for key in example["metrics"])
        for key in keys:
            values = [example["metrics"][key] for example in all_examples if key in example["metrics"]]
            valid = [v for v in values if not math.isnan(v)]
            self.metrics[key].append(sum(valid) / len(valid) if valid else float("nan"))

        # Packing quality. `row_imbalance` is what the Σ Lᵢ²-balancing planner exists to keep near 1.0: attention is
        # O(L²), so it is Σ Lᵢ² and not the token count that predicts which rank stalls the gradient all-reduce.
        row_tokens = [sum(len(example["input_ids"]) for example in group) for group in groups]
        squared_loads = [sum(len(example["input_ids"]) ** 2 for example in group) for group in groups]
        self.metrics["batch/samples_per_row"].append(len(all_examples) / len(groups))
        self.metrics["batch/row_tokens_mean"].append(sum(row_tokens) / len(row_tokens))
        self.metrics["batch/row_tokens_max"].append(float(max(row_tokens)))
        self.metrics["batch/row_imbalance"].append(max(squared_loads) / (sum(squared_loads) / len(squared_loads)))
        if self.token_budget:
            self.metrics["batch/row_fill_frac"].append((sum(row_tokens), self.token_budget * len(row_tokens)))
        # Inter-rank padding, added so the batch is rectangular for the dispatcher. It costs broadcast bytes only:
        # `compute_loss` strips it before the forward, so no FLOPs are spent on it.
        self.metrics["batch/pad_frac"].append((padded.numel() - int(padded.sum()), padded.numel()))


class AsyncGRPOTrainer(_BaseTrainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language
    Models](https://huggingface.co/papers/2402.03300). This trainer is the asynchronous version of GRPO, where
    generation is offloaded to an external vLLM server that runs asynchronously alongside training, decoupling rollout
    from the gradient update loop.

    Example:

    ```python
    >>> from trl.experimental.async_grpo import AsyncGRPOTrainer
    >>> from trl.rewards import accuracy_reward
    >>> from datasets import load_dataset

    >>> dataset = load_dataset("trl-lib/DeepMath-103K", split="train")

    >>> trainer = AsyncGRPOTrainer(
    ...     model="Qwen/Qwen2.5-0.5B-Instruct",
    ...     reward_funcs=accuracy_reward,
    ...     train_dataset=dataset,
    ... )
    >>> trainer.train()
    ```

    Args:
        model (`str`):
            Model to be trained. Must be a string, being the *model id* of a pretrained model hosted inside a model
            repo on huggingface.co, or a path to a *directory* containing model weights saved using
            [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
            using [`~transformers.AutoModelForCausalLM.from_pretrained`]. The model name is also used to identify the
            model on the vLLM server used for generation.
        reward_funcs (`RewardFunc | list[RewardFunc]`, *optional*):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. May be omitted when the reward is supplied
            by the environment through `environment_factory` (see below). Can be either:

            - A single reward function: The function is provided with the prompts and the generated completions, plus
              any additional columns in the dataset. It should return a list of rewards. Reward functions can be either
              synchronous or asynchronous and can also return `None` when the reward is not applicable to those
              samples. This is useful for multi-task training where different reward functions apply to different types
              of samples. When a reward function returns `None` for a sample, that reward function is excluded from the
              reward calculation for that sample. For more details, see [Using a custom reward
              function](#using-a-custom-reward-function).
            - A list of reward functions, where each item is a reward function as described above. Rewards from all
              functions are summed.

            Unlike [`GRPOTrainer`], rewards are computed in a spawned child process, so each reward function (along
            with `tools` and `environment_factory`) must be picklable: use a module-level function,
            `functools.partial`, or a callable class instance — lambdas and closures will fail at startup. The child
            process also runs with `CUDA_VISIBLE_DEVICES=""`, so a GPU-backed reward model runs on CPU (slow), not the
            trainer's GPU.
        args ([`AsyncGRPOConfig`], *optional*):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`], *optional*):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset are
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).

            May be omitted only when an `environment_factory` is provided and the environment owns (or procedurally
            generates) the data, returning the prompt from its `reset()` method. In that case, `max_steps` must be set
            to define the training length.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`]. A
            padding token, `tokenizer.pad_token`, must be set. If the processing class has not set a padding token,
            `tokenizer.eos_token` will be used as the default.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks detailed
            in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of `AdamW` on your
            model and a scheduler given by [`~transformers.get_linear_schedule_with_warmup`] controlled by `args`.
        tools (list of `Callable`, *optional*):
            A list of callable tool functions (sync or async) that the model can invoke during generation. Each tool
            should be a standard Python function with properly type-hinted arguments and return values, and a
            Google-style docstring describing its purpose, arguments, and return value. For more details, see:
            https://huggingface.co/docs/transformers/en/chat_extras#passing-tools. The model uses the function's name,
            type hints, and docstring to determine how to call it. Ensure that the model's chat template supports tool
            use and that it has been fine-tuned for tool calling.
        environment_factory (`EnvironmentFactory` or `dict[str, EnvironmentFactory]`, *optional*):
            A callable that creates and returns an environment instance, or a dictionary mapping environment names to
            such callables. The environment class should define methods that can be invoked as tools during generation.
            Each method should comply with the same requirements as the `tools` described above. The environment must
            also implement a callable `reset` method that can be used to reset state between generations. The `reset`
            method should return either `None` or a string: when it returns a string, that string is appended to the
            last user message before generation. The environment may also define a `get_reward` method taking no
            argument and returning a `float`: when present, the environment owns the reward, and `get_reward` is called
            once per completed rollout to score it from the environment's internal state. It acts as an additional
            reward source (with weight 1, logged under the environment's class name) alongside `reward_funcs`, which
            then becomes optional.

            With a single callable, every example uses the same environment, with one instance per rollout so their
            interactions stay isolated. With a dictionary, each example must carry an `environment` field selecting its
            environment by name, and only that environment's tools are exposed in its prompt — letting a single run mix
            tasks (e.g. a coding environment and a game). This feature is experimental and may change or be removed at
            any time without prior notice.
        rollout_worker (`RolloutWorkerProtocol`, *optional*):
            Custom rollout worker implementing [`RolloutWorkerProtocol`]. If `None`, a default [`AsyncRolloutWorker`]
            is created, which spawns a CUDA-free child process and scores completions with the trainer's
            `reward_funcs`. Pass a custom worker to plug in a different rollout/scoring backend instead — for example,
            one that runs reward models on their own GPUs.
        weight_transfer (`WeightTransferProtocol`, *optional*):
            Custom weight-sync backend implementing [`WeightTransferProtocol`]. If `None`, a default
            [`WeightTransferClient`] is created that streams the trainer's weights into the config's vLLM server over
            NCCL. This is independent of `rollout_worker`: a custom rollout worker still gets weight sync. Pass a no-op
            implementation to disable trainer-side weight sync.
    """

    _tag_names = ["trl", "async-grpo"]
    _name = "AsyncGRPO"
    _paper = {
        "title": "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
        "id": "2402.03300",
        # docstyle-ignore
        "citation": textwrap.dedent("""\
            @article{shao2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            }"""),
    }

    def __init__(
        self,
        model: str,
        reward_funcs: RewardFunc | list[RewardFunc] | None = None,
        args: AsyncGRPOConfig | None = None,
        train_dataset: Dataset | IterableDataset | None = None,
        processing_class: PreTrainedTokenizerBase | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None] = (None, None),
        tools: list[Callable] | None = None,
        environment_factory: EnvironmentFactory | dict[str, EnvironmentFactory] | None = None,
        rollout_worker: RolloutWorkerProtocol | None = None,
        weight_transfer: WeightTransferProtocol | None = None,
    ):
        # Args
        if args is None:
            model_name = model.split("/")[-1]
            args = AsyncGRPOConfig(f"{model_name}-AsyncGRPO")

        # Training arguments
        self.epsilon_low = args.epsilon
        self.epsilon_high = args.epsilon_high if args.epsilon_high is not None else args.epsilon
        self.temperature = args.temperature

        # Model
        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs.setdefault("trust_remote_code", args.trust_remote_code)
        model_init_kwargs.setdefault("dtype", args.dtype)
        # FlashAttention is required: training runs in padding-free mode, where sequences are concatenated into a
        # single row and `cu_seq_lens` are derived from `position_ids` resets. SDPA/eager can't handle this.
        model = AutoModelForCausalLM.from_pretrained(
            model,
            device_map=None,
            attn_implementation="kernels-community/flash-attn3",
            **model_init_kwargs,
        )

        if args.use_liger_kernel:
            raise NotImplementedError("`use_liger_kernel` is not supported yet.")

        # MoE load-balancing auxiliary loss, applied to Mixture-of-Experts models (no effect otherwise)
        text_config = model.config.get_text_config()
        is_moe = getattr(text_config, "output_router_logits", None) is not None
        self.aux_loss_enabled = is_moe and args.router_aux_loss_coef != 0.0
        self.router_aux_loss_coef = args.router_aux_loss_coef

        patch_chunked_lm_head(
            model, chunk_size=8192, temperature=self.temperature, output_router_logits=self.aux_loss_enabled
        )

        # Processing class
        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(
                get_config_model_id(model.config), trust_remote_code=args.trust_remote_code
            )
        if processing_class.pad_token is None:
            processing_class.pad_token = processing_class.eos_token

        # Reward functions
        if reward_funcs is None:
            reward_funcs = []
        elif not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]

        if train_dataset is None:
            # A dataset is optional when an environment owns the data and returns the prompt from `reset()`; then
            # `max_steps` sets the length. Build a placeholder dataset of empty prompts to feed the rollout worker,
            # which cycles it indefinitely (so its length only needs to be non-degenerate).
            if environment_factory is None:
                raise ValueError("`train_dataset` is required unless an `environment_factory` is provided.")
            if isinstance(environment_factory, dict):
                raise ValueError(
                    "A `dict` `environment_factory` (multiple environments) requires a `train_dataset` with an "
                    "`environment` column to route each example to its environment. Provide a dataset, or pass a "
                    "single environment factory."
                )
            if self.args.max_steps <= 0:
                raise ValueError(
                    "When training without a `train_dataset` (the environment owns the data and returns the prompt "
                    "from `reset()`), `max_steps` must be set to a positive value to define the training length. Set "
                    "it via `AsyncGRPOConfig(max_steps=...)`."
                )
            num_placeholder_rows = self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps
            train_dataset = Dataset.from_dict({"prompt": [[{"role": "user", "content": ""}]] * num_placeholder_rows})

        # Initialize the Trainer
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
            compute_loss_func="non-None value to disable scaling",
        )
        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        # Epoch handling: stop after num_train_epochs full passes over the PROMPT dataset, counted as distinct
        # prompt-groups trained (fork-independent).
        self._trained_groups: set[int] = set()
        self._epoch_stop_groups: int | None = None
        samples_per_step = (
            self.args.per_device_train_batch_size
            * self.args.gradient_accumulation_steps
            * self.accelerator.num_processes
        )
        if self.args.max_steps <= 0 and train_dataset is not None and hasattr(train_dataset, "__len__"):
            # Fork-independent stop: num_train_epochs full passes over the prompts.
            self._epoch_stop_groups = math.ceil(self.args.num_train_epochs * len(train_dataset))
            # max_steps is a generous safety ceiling; with the default constant LR its exact value doesn't matter.
            max_rows_per_conv = (
                self.args.max_tool_calling_iterations + 1 if self.args.max_tool_calling_iterations is not None else 32
            )
            samples_per_epoch = len(train_dataset) * self.args.num_generations * max_rows_per_conv
            self.args.max_steps = math.ceil(self.args.num_train_epochs * samples_per_epoch / samples_per_step)
            logger.info(
                f"Epoch-driven stop: {self._epoch_stop_groups} prompt-groups "
                f"({self.args.num_train_epochs} epochs x {len(train_dataset)} prompts); "
                f"max_steps={self.args.max_steps} is a safety ceiling."
            )

        # Infer max_inflight_tasks when not explicitly set. Generating more samples than the trainer can consume
        # before they become stale is wasteful. The useful upper bound is max_staleness * samples_per_step, floored
        # at samples_per_step so max_staleness=0 (a valid, strict discard policy) can't also zero out the rollout
        # loop's own scheduling capacity and hang the trainer forever on an empty queue.
        if self.args.max_inflight_tasks < 0:
            self.args.max_inflight_tasks = max(self.args.max_staleness, 1) * samples_per_step
            logger.info(
                f"max_inflight_tasks set to {self.args.max_inflight_tasks} "
                f"(max_staleness={self.args.max_staleness} × samples_per_step={samples_per_step})"
            )

        # The metric sink. Values are floats, or `(numerator, denominator)` pairs for rates
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._current_train_step_time = 0.0
        self._last_step_end_time = None
        self._rollout_dataset = None
        # Accumulated across one optimizer step's micro-batches, flushed by `_log_step_metrics`.
        self._step_forward_s = 0.0
        self._step_optimizer_s = 0.0
        self._step_microbatches = 0
        self._step_forward_tokens = 0.0
        self._step_trained_tokens = 0.0
        self._step_seq_len_weighted = 0.0
        self._step_samples = 0.0
        self._last_groups_trained = 0
        self.model_version = 0
        # Create worker and queue on rank 0
        if self.accelerator.is_main_process:
            # Weight sync and the token-budget query target the vLLM server from the config; the client is the single
            # place that talks to it, independent of how rollouts are produced.
            self.vllm_client = VLLMClient(self.args.vllm_server_base_url, self.args.vllm_server_timeout)

            if weight_transfer is not None:
                # Injected backend (e.g. a no-op stub in tests, or a custom sync mechanism).
                self.weight_transfer = weight_transfer
            else:
                # Collect weight metadata once — names/dtypes/shapes are fixed for the lifetime of training.
                # DTensor.shape returns the global shape without triggering any all-gather.
                weight_names, weight_dtype_names, weight_shapes = [], [], []
                for name, param in model.named_parameters():
                    # DDP/FSDP1 wrapping, avoids vllm module not exist error
                    name = name.removeprefix("module.")
                    weight_names.append(name)
                    weight_dtype_names.append(str(param.dtype).split(".")[-1])
                    weight_shapes.append(list(param.shape))
                self.weight_transfer = WeightTransferClient(
                    vllm_client=self.vllm_client,
                    weight_update_info={
                        "names": weight_names,
                        "dtype_names": weight_dtype_names,
                        "shapes": weight_shapes,
                        "packed": True,
                    },
                )

            if rollout_worker is not None:
                # Use the injected worker (e.g. a stub in tests). The queue is owned by the worker.
                self.rollout_worker = rollout_worker
            else:
                self.rollout_worker = AsyncRolloutWorker(
                    model_name=get_config_model_id(model.config),
                    dataset=train_dataset,
                    reward_funcs=reward_funcs,
                    processing_class=processing_class,
                    tools=tools,
                    environment_factory=environment_factory,
                    num_generations=self.args.num_generations,
                    max_inflight_tasks=self.args.max_inflight_tasks,
                    queue_maxsize=self.args.queue_maxsize,
                    vllm_server_url=self.args.vllm_server_base_url,
                    max_tokens=self.args.max_completion_length,
                    temperature=self.args.temperature,
                    top_p=self.args.top_p,
                    top_k=self.args.top_k,
                    min_p=self.args.min_p,
                    repetition_penalty=self.args.repetition_penalty,
                    request_timeout=self.args.request_timeout,
                    chat_template_kwargs=self.args.chat_template_kwargs,
                    max_tool_calling_iterations=self.args.max_tool_calling_iterations,
                    log_completions=self.args.log_completions,
                    num_completions_to_print=self.args.num_completions_to_print,
                    fork_threshold_tokens=self.args.fork_threshold_tokens,
                )
            # TODO(@aminediro): decide if this is returned by the worker or common API that is passed to the worker later.
            self.rollout_queue = self.rollout_worker.rollout_buffer
        else:
            self.rollout_queue = None
            self.rollout_worker = None
            self.vllm_client = None
            self.weight_transfer = None

        # Add callbacks. Cold weight sync + worker start on train begin, then periodic weight syncs.
        self.add_callback(_OptimizerTimeCallback(self))
        self.add_callback(_TrainBeginCallback(self))
        self.add_callback(StepIntervalCallback(self._sync_weight, self.args.weight_sync_steps))
        self.add_callback(StepIntervalCallback(self._log_step_metrics, 1))
        if self._epoch_stop_groups is not None:
            self.add_callback(_EpochStopCallback(self, self._epoch_stop_groups))

    def get_train_dataloader(self) -> DataLoader:
        num_processes = self.accelerator.num_processes
        if self.accelerator.is_main_process:
            dataset = RolloutQueueDataset(
                rollout_queue=self.rollout_queue,
                model_version_fn=lambda: self.model_version,
                check_health_fn=self.rollout_worker.check_health,
                stale_after_s=self.args.heartbeat_stale_after_s,
                metrics=self._metrics["train"],
                max_staleness=self.args.max_staleness,
                report_to=self.args.report_to,
            )
            # Kept so `_log_step_metrics` can flush the queue wait at the optimizer-step boundary, which is the only
            # place that knows a step's worth of waiting is over.
            self._rollout_dataset = dataset
            # Default the token budget to the vLLM server's max_model_len (the cap on prompt + completion), so no
            # rollout sample can exceed it. Wait for the server like weight sync does, so a still-loading vLLM doesn't
            # fail training here.
            if self.args.token_budget is None:
                self.vllm_client.wait_for_server_ready()
                self.args.token_budget = self.vllm_client.get_max_model_len()
                logger.info(f"token_budget unset; defaulting to vLLM max_model_len={self.args.token_budget}")
            # The planner partitions the rollout stream into Σ Lᵢ²-balanced micro-batches of `num_processes` rows.
            # TokenBudgetBatcher caps each row at `token_budget` tokens (dynamic count, bounds peak memory);
            # FixedCountBatcher uses a fixed `per_device_train_batch_size × num_processes` samples per micro-batch.
            if self.args.token_budget > 0:
                dataset = TokenBudgetBatcher(dataset, num_processes, self.args.token_budget, self._metrics["train"])
            else:
                dataset = FixedCountBatcher(
                    dataset, num_processes, self.args.per_device_train_batch_size * num_processes
                )
        else:
            dataset = _EmptyIterableDataset()

        # Each planner item is one complete micro-batch (`num_processes` pre-packed rows), so the dataloader pulls them
        # one at a time (batch_size=1) and the collator tensorizes each into a rectangular `(num_processes, T_max)`
        # batch that DataLoaderDispatcher scatters row `i` -> rank `i`.
        return self.accelerator.prepare(
            DataLoader(
                dataset,
                batch_size=1,
                collate_fn=DataCollatorForRollout(
                    self.processing_class.pad_token_id,
                    num_processes,
                    groups_trained=self._trained_groups,
                    metrics=self._metrics["train"],
                    # `or 0` because only rank 0 fills an unset budget from the vLLM server above; the other ranks
                    # construct the collator (and never use it) while `token_budget` is still `None`.
                    token_budget=max(self.args.token_budget or 0, 0),
                ),
                num_workers=0,
                # NOTE(@aminediro):
                # dispatch_batches = True for DataLoader whose underlying dataset is an IterableDataset
                # dataloader prepared by the Accelerator is only iterated through on the main process a
            )
        )

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs (usually, "input_ids"
        # and "attention_mask"). In AsyncGRPOTrainer, we need additional columns ("completion_mask", "old_log_probs",
        # "advantages", "global_n_tokens") to compute the loss, hence the override.
        if self._signature_columns is None:
            self._signature_columns = [
                "input_ids",
                "attention_mask",
                "completion_mask",
                "old_log_probs",
                "position_ids",
                "advantages",
                "global_n_tokens",
                "global_n_forward_tokens",
                "mean_seq_len",
            ]

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Padding-free: the collator already packed this rank's samples into a single row (real tokens concatenated,
        # `position_ids` resetting per sequence, advantages expanded per-token), then padded the row to the longest
        # rank's length so DataLoaderDispatcher could scatter rectangular rows. Strip that trailing inter-rank padding
        # here.
        mask_bool = inputs["attention_mask"].bool()
        input_ids = inputs["input_ids"][mask_bool].unsqueeze(0)
        completion_mask = inputs["completion_mask"][mask_bool].unsqueeze(0)
        old_log_probs = inputs["old_log_probs"][mask_bool].unsqueeze(0)
        position_ids = inputs["position_ids"][mask_bool].unsqueeze(0)
        advantages = inputs["advantages"][mask_bool].unsqueeze(0)

        forward_start = time.time()
        outputs = model(
            input_ids=input_ids,
            position_ids=position_ids,
            labels=input_ids,
            completion_mask=completion_mask,
            use_cache=False,
        )
        log_probs, entropy = outputs["log_probs"], outputs["entropy"]
        self._last_forward_time_s = time.time() - forward_start

        completion_mask = completion_mask[:, 1:]
        old_log_probs = old_log_probs[:, 1:]
        advantages = advantages[:, 1:]
        log_ratio = log_probs - old_log_probs
        coef_1 = torch.exp(log_ratio)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        per_token_loss1 = coef_1 * advantages
        per_token_loss2 = coef_2 * advantages
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        # DDP/FSDP averages gradients across ranks (world_size).
        # To get correct per-token normalization we scale by 1/tokens_per_rank
        # = world_size / global_n_tokens, so after DDP averaging the effective
        loss = (per_token_loss * completion_mask).sum()
        global_n_tokens = inputs["global_n_tokens"][0]
        world_size = self.accelerator.num_processes
        tokens_per_rank = (global_n_tokens / world_size).clamp(min=1.0)
        loss = loss / tokens_per_rank.to(torch.float32)
        # For DAPO, we would scale like this instead:
        # loss = loss / max(per_token_loss.size(0), 1)
        loss = loss / self.current_gradient_accumulation_steps

        # The policy loss above is scaled for gradient accumulation (HF auto-scaling is off here), so scale aux too
        if self.aux_loss_enabled:
            aux_loss = outputs["aux_loss"]
            loss = loss + self.router_aux_loss_coef * aux_loss / self.current_gradient_accumulation_steps

        with torch.no_grad():
            valid_mask = completion_mask > 0
            local_count = valid_mask.sum().float()

            # Empty masked selections sum to a 0 scalar on the right device, so no valid_mask.any() guard is needed.
            local_ratio_sum = coef_1[valid_mask].sum()
            # Approx KL: http://joschu.net/blog/kl-approx.html
            local_kl_sum = ((coef_1[valid_mask] - 1) - log_ratio[valid_mask]).sum()
            local_entropy_sum = entropy[valid_mask].sum()

            # Compute the clipped probability ratios
            is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages < 0)
            is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages > 0)
            is_region_clipped = is_low_clipped | is_high_clipped
            local_low_clip_sum = is_low_clipped[valid_mask].float().sum()
            local_high_clip_sum = is_high_clipped[valid_mask].float().sum()
            local_region_clip_sum = is_region_clipped[valid_mask].float().sum()

            # Batch all-reduce: [ratio_sum, kl_sum, entropy_sum, low_clip_sum, high_clip_sum, region_clip_sum, count]
            stats = torch.stack(
                [
                    local_ratio_sum,
                    local_kl_sum,
                    local_entropy_sum,
                    local_low_clip_sum,
                    local_high_clip_sum,
                    local_region_clip_sum,
                    local_count,
                ]
            )
            stats = self.accelerator.reduce(stats, reduction="sum")
            (
                global_ratio_sum,
                global_kl_sum,
                global_entropy_sum,
                global_low_clip_sum,
                global_high_clip_sum,
                global_region_clip_sum,
                global_count,
            ) = stats.unbind(0)
            self._metrics["train"]["ratio"].append((global_ratio_sum / global_count).item())
            self._metrics["train"]["kl"].append((global_kl_sum / global_count).item())
            self._metrics["train"]["entropy"].append((global_entropy_sum / global_count).item())
            self._metrics["train"]["clip_ratio/low_mean"].append((global_low_clip_sum / global_count).item())
            self._metrics["train"]["clip_ratio/high_mean"].append((global_high_clip_sum / global_count).item())
            self._metrics["train"]["clip_ratio/region_mean"].append((global_region_clip_sum / global_count).item())

            seq_ids = (position_ids[0] == 0).cumsum(0)[1:] - 1  # (T-1,) completion index per (shifted) token
            n_completions = (position_ids == 0).sum()  # number of packed completions in this rank's row
            num_seq = int(n_completions)
            comp_mask = completion_mask[0].float()  # (T-1,) valid completion-token mask

            def seg_sum(vals):  # per-completion segment sum over the packed row
                return torch.zeros(num_seq, device=comp_mask.device).index_add_(0, seq_ids, vals)

            seq_tokens = seg_sum(comp_mask)
            seq_low = seg_sum(is_low_clipped[0].float() * comp_mask)
            seq_high = seg_sum(is_high_clipped[0].float() * comp_mask)
            per_seq_low = seq_low / seq_tokens  # NaN for a completion with no valid tokens; ignored by nan-aware min
            per_seq_high = seq_high / seq_tokens
            gathered_low_min = self.accelerator.gather(nanmin(per_seq_low))
            gathered_high_max = self.accelerator.gather(nanmax(per_seq_high))
            self._metrics["train"]["clip_ratio/low_min"].append(nanmin(gathered_low_min).item())
            self._metrics["train"]["clip_ratio/high_max"].append(nanmax(gathered_high_max).item())

            if self.aux_loss_enabled:
                gathered_aux = self.accelerator.reduce(aux_loss.detach().to(torch.float32), reduction="sum")
                self._metrics["train"]["aux_loss"].append((gathered_aux / world_size).item())

        # Per-step accounting, accumulated across the micro-batches of one optimizer step and flushed in
        # `training_step`. The counts are batch-wide (the collator broadcasts one value per rank), so they are read off
        # rank-local inputs without a collective. Sample rewards and packing metrics are NOT gathered here — rank 0
        # already logged them in the collator.
        n_forward_tokens = float(inputs["global_n_forward_tokens"][0])
        mean_seq_len = float(inputs["mean_seq_len"][0])
        self._step_forward_tokens += n_forward_tokens
        self._step_trained_tokens += float(global_n_tokens)
        self._step_seq_len_weighted += mean_seq_len * n_forward_tokens
        self._step_samples += n_forward_tokens / mean_seq_len
        self._step_forward_s += self._last_forward_time_s
        return loss

    def training_step(self, model, inputs, num_items_in_batch):
        time_before = time.perf_counter()
        output = super().training_step(model, inputs, num_items_in_batch)
        self._step_microbatches += 1
        self._current_train_step_time += time.perf_counter() - time_before
        return output

    def _log_step_metrics(self) -> None:
        """Flush one optimizer step's worth of accounting: the time budget, what the batch held, and throughput.

        Called from `on_step_end`, i.e. after `optimizer.step()`, so `perf/optimizer_s` covers this step.
        """
        time_after = time.perf_counter()
        metrics = self._metrics["train"]
        fwd_bwd_s = self._current_train_step_time
        # The first step has no predecessor, so there is nothing to measure between yet.
        step_s = time_after - self._last_step_end_time if self._last_step_end_time is not None else None

        metrics["perf/fwd_bwd_s"].append(fwd_bwd_s)
        metrics["perf/fwd_s"].append(self._step_forward_s)
        metrics["perf/optimizer_s"].append(self._step_optimizer_s)
        metrics["batch/microbatches_per_step"].append(float(self._step_microbatches))
        metrics["batch/forwarded_tokens_per_step"].append(self._step_forward_tokens)
        metrics["batch/trained_tokens_per_step"].append(self._step_trained_tokens)
        metrics["batch/samples_per_step"].append(self._step_samples)
        if step_s is not None:
            metrics["perf/step_s"].append(step_s)

        if self.accelerator.is_main_process:
            metrics["perf/rollout_wait_s"].append(self._rollout_dataset.wait_s)
            self._rollout_dataset.wait_s = 0.0
            metrics["batch/groups_per_step"].append(float(len(self._trained_groups) - self._last_groups_trained))
            self._last_groups_trained = len(self._trained_groups)

        # Throughput and MFU are reported on TWO bases, because for async RL one number cannot answer both questions.
        ## `_fwd_bwd` divides by `perf/fwd_bwd_s`: the compute alone.
        ## `_wall_clock` divides by `perf/step_s`: the whole step, rollout waits included  and says what fraction of the allocation actually became training.
        if self._step_forward_tokens > 0:
            mean_seq_len = self._step_seq_len_weighted / self._step_forward_tokens
            flops_per_token = compute_flops_per_token(self.model.config, int(mean_seq_len))
            world_size = self.accelerator.num_processes
            metrics["perf/forwarded_tok_s_fwd_bwd"].append((self._step_forward_tokens, fwd_bwd_s))
            metrics["perf/mfu_fwd_bwd"].append(
                compute_mfu(flops_per_token, self._step_forward_tokens / fwd_bwd_s, world_size)
            )
            if step_s is not None:
                metrics["perf/forwarded_tok_s_wall_clock"].append((self._step_forward_tokens, step_s))
                metrics["perf/trained_tok_s_wall_clock"].append((self._step_trained_tokens, step_s))
                metrics["perf/mfu_wall_clock"].append(
                    compute_mfu(flops_per_token, self._step_forward_tokens / step_s, world_size)
                )

        self._last_step_end_time = time_after
        self._current_train_step_time = 0.0
        self._step_forward_s = 0.0
        self._step_optimizer_s = 0.0
        self._step_microbatches = 0
        self._step_forward_tokens = 0.0
        self._step_trained_tokens = 0.0
        self._step_seq_len_weighted = 0.0
        self._step_samples = 0.0

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        mode = "train" if self.model.training else "eval"
        if self.accelerator.is_main_process and self.rollout_worker:
            while True:
                try:
                    for key, value in self.rollout_worker.metrics_queue.get_nowait().items():
                        # NOTE(@aminediro): we might be filling train metrics dict even in eval mode
                        self._metrics["train"][key].append(value)
                except queue.Empty:
                    break

        metrics = {}
        for key, val in self._metrics[mode].items():
            valid = [v for v in val if isinstance(v, tuple) or not math.isnan(v)]
            metrics[key] = _reduce_metric(key, valid) if valid else None

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs.update(metrics)
        super().log(logs, start_time)
        self._metrics[mode].clear()

    def _streaming_iter(self):
        # Iterate parameters one at a time. For FSDP2 (DTensor), full_tensor() all-gathers just this parameter across
        # FSDP ranks, then frees it once the generator advances — avoiding materializing the full model in memory.
        device = self.accelerator.device
        for name, param in self.model.named_parameters():
            name = name.removeprefix("module.")  # DDP/FSDP1 wrapping
            full = param.full_tensor() if isinstance(param, DTensor) else param.detach()
            if full.device != device:
                full = full.to(device)
            yield name, full

    def _sync_weight(self):
        t0 = time.time()
        logger.info("Weight sync: pausing vLLM...")
        if self.accelerator.is_main_process and self.weight_transfer:
            self.weight_transfer.pause()
        t_pause = time.time()
        logger.info(f"Weight sync: pause took {t_pause - t0:.1f}s, waiting for all ranks...")

        self.accelerator.wait_for_everyone()
        t_barrier = time.time()

        logger.info(f"Weight sync: transferring weights... (barrier took {t_barrier - t_pause:.1f}s)")
        if self.accelerator.is_main_process and self.weight_transfer:
            self.weight_transfer.send_weights(self._streaming_iter())
        else:
            # Non-rank-0 processes must still participate in full_tensor() collectives for FSDP2.
            for _ in self._streaming_iter():
                pass
        t_transfer = time.time()

        self.accelerator.wait_for_everyone()

        logger.info(f"Weight sync: resuming vLLM... (transfer took {t_transfer - t_barrier:.1f}s)")
        if self.accelerator.is_main_process:
            if self.weight_transfer:
                self.weight_transfer.resume()
            self.model_version += 1
            if self.rollout_worker:
                self.rollout_worker.update_model_version(self.model_version)
        weight_sync_s = time.time() - t0
        # log the three phases  of weight sync
        self._metrics["train"]["perf/weight_sync_s"].append(weight_sync_s)
        self._metrics["train"]["perf/weight_sync_pause_s"].append(t_pause - t0)
        self._metrics["train"]["perf/weight_sync_barrier_s"].append(t_barrier - t_pause)
        self._metrics["train"]["perf/weight_sync_transfer_s"].append(t_transfer - t_barrier)
        logger.info(f"Weight sync: done. Total {weight_sync_s:.1f}s")

    def _inner_training_loop(self, *args, **kwargs):
        try:
            return super()._inner_training_loop(*args, **kwargs)
        finally:
            if self.accelerator.is_main_process:
                if self.rollout_worker:
                    self.rollout_worker.stop()
                if self.weight_transfer:
                    self.weight_transfer.destroy()
