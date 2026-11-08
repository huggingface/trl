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
import contextvars
import functools
import inspect
import math
import os
import queue
import textwrap
import threading
import time
import uuid
import warnings
from collections import defaultdict, deque
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from itertools import chain, repeat
from typing import Any, Protocol, TypeAlias

import numpy as np
import torch
from accelerate.logging import get_logger
from accelerate.parallelism_config import ParallelismConfig
from datasets import Dataset, IterableDataset
from torch import nn
from torch.distributed.tensor import DTensor
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainerCallback,
)
from transformers.distributed.configuration_utils import DistributedConfig
from transformers.generation import ContinuousBatchingConfig
from transformers.generation.continuous_batching import ContinuousBatchingManager
from transformers.utils import is_flash_attn_2_available, is_flash_attn_3_available

from ...chat_template_utils import (
    _SUPPORTS_RESPONSE_TEMPLATE,
    add_response_schema,
    get_training_chat_template,
    is_chat_template_prefix_preserving,
    parse_response,
)
from ...trainer.base_trainer import _BaseTrainer
from ...trainer.utils import (
    add_fused_lm_head,
    compute_flops_per_token,
    compute_mfu,
    create_model_from_path,
    get_callable_name,
    get_config_model_id,
    get_peak_flops_per_device,
    identity,
    is_async_callable,
    is_trackio_available,
    nanmax,
    nanmin,
    print_prompt_completions_sample,
)
from .zero_sync_grpo_config import ZeroSyncGRPOConfig


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


Messages: TypeAlias = list[dict[str, str]]
RolloutId: TypeAlias = str


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

    def __init__(self, trainer: "ZeroSyncGRPOTrainer"):
        self._trainer = trainer
        self._t0 = None

    def on_pre_optimizer_step(self, _args, _state, _control, **_kwargs):
        self._t0 = time.perf_counter()

    def on_optimizer_step(self, _args, _state, _control, **_kwargs):
        self._trainer._step_optimizer_s += time.perf_counter() - self._t0


@dataclass(frozen=True)
class TurnRecord:
    """
    One sampling request: the prompt sent to the sampler and the tokens it generated.

    Args:
        prompt_ids (`list[int]`):
            Token ids of the full prompt sent for this turn, including every earlier turn of the conversation.
        output_ids (`list[int]`):
            Token ids the sampler generated.
        output_log_probs (`list[float]`, *optional*, defaults to `[]`):
            Sampler log-probability of each token in `output_ids`. Required when any of them is trained on.
        output_mask (`list[int]`, *optional*):
            One value per token in `output_ids`: `1` to train on the token, `0` to keep it as context only. If `None`,
            every output token is trained on.
        output_sampling_mask (`list[list[int]]`, *optional*):
            One list per token in `output_ids`: the token ids the sampler could draw from under `top_k` / `top_p`
            truncation. If `None`, the full vocabulary.
        output_sampler_logprobs (`list[dict[int, float]]`, *optional*):
            One dict per token in `output_ids`: the sampler's log-probabilities of its most likely tokens, keyed by
            token id. Used by score centering (`num_sampler_logprobs`).
    """

    prompt_ids: list[int]
    output_ids: list[int]
    output_log_probs: list[float] = field(default_factory=list)
    output_mask: list[int] | None = None
    output_sampling_mask: list[list[int]] | None = None
    output_sampler_logprobs: list[dict[int, float]] | None = None


@dataclass
class TrainingSequence:
    """One training row. Maps 1:1 onto the array fields of `RolloutSample`."""

    input_ids: list[int]  # full tokens (prompt included)
    completion_mask: list[int]  # 1 = train this token, 0 = context
    old_log_probs: list[float]  # sampled logprobs; prompt-only context is zero-filled
    sampling_mask: list[list[int]]  # per token, the sampler's candidate set; empty for context and without truncation
    sampler_logprobs: list[dict[int, float]]  # per token, the sampler's top log-probs by id; empty when not asked
    rollout_id: RolloutId  # which conversation this row came from


def _common_prefix_len(a: list[int], b: list[int], chunk: int = 4096) -> int:
    """How many tokens at the start are identical in `a` and `b`."""
    limit = min(len(a), len(b))
    matched = 0
    while matched < limit:
        end = min(matched + chunk, limit)
        if a[matched:end] == b[matched:end]:
            matched = end
        else:
            while matched < end and a[matched] == b[matched]:
                matched += 1
            return matched
    return matched


class _SampleBuilder:
    def __init__(self, rollout_id: RolloutId):
        self.rollout_id = rollout_id
        self.tokens: list[int] = []
        self.loss_mask: list[int] = []
        self.logprobs: list[float] = []
        self.sampling_mask: list[list[int]] = []
        self.sampler_logprobs: list[dict[int, float]] = []

    def append_turn(self, turn: TurnRecord) -> None:
        self._append(turn.prompt_ids[len(self.tokens) :], mask=0)
        self._append(
            turn.output_ids,
            mask=turn.output_mask if turn.output_mask is not None else 1,
            logprobs=turn.output_log_probs,
            sampling_mask=turn.output_sampling_mask,
            sampler_logprobs=turn.output_sampler_logprobs,
        )

    def _append(
        self,
        ids: list[int],
        *,
        mask: int | list[int],
        logprobs: list[float] | None = None,
        sampling_mask: list[list[int]] | None = None,
        sampler_logprobs: list[dict[int, float]] | None = None,
    ) -> None:
        """Append tokens with a uniform or per-token supervision mask."""
        masks = [mask] * len(ids) if isinstance(mask, int) else mask
        if len(masks) != len(ids):
            raise ValueError("loss mask must contain one entry per token")
        if any(masks) and logprobs is None:
            raise ValueError("trainable tokens require one sampled logprob per token")
        if logprobs is not None and len(logprobs) != len(ids):
            raise ValueError("logprobs must contain one entry per token")
        if sampling_mask is not None and len(sampling_mask) != len(ids):
            raise ValueError("sampling mask must contain one entry per token")
        self.tokens.extend(ids)
        self.loss_mask.extend(masks)
        self.logprobs.extend(logprobs if logprobs is not None else [0.0] * len(ids))
        self.sampling_mask.extend(sampling_mask if sampling_mask is not None else [[] for _ in ids])
        self.sampler_logprobs.extend(sampler_logprobs if sampler_logprobs is not None else [{} for _ in ids])

    def has_trained_token(self) -> bool:
        return any(self.loss_mask)

    def to_training_sequence(self) -> TrainingSequence:
        return TrainingSequence(
            list(self.tokens),
            list(self.loss_mask),
            list(self.logprobs),
            list(self.sampling_mask),
            list(self.sampler_logprobs),
            self.rollout_id,
        )


def _chain_to_sequences(rollout: "Rollout") -> tuple[list[TrainingSequence], list[int]]:
    """Reconcile a rollout's turns (in order) into training rows, and report how many held tokens each fork
    re-rendered.

    A turn whose prompt starts with the tokens held so far extends the current row. Otherwise the chat template
    rendered an earlier turn differently (dropped reasoning, summarized history, a re-tokenization wobble), and the
    turn starts a new row, so every generated token keeps its training signal.
    """
    builders = [_SampleBuilder(rollout.rollout_id)]
    drifts = []
    for turn in rollout.turns:
        held = builders[-1].tokens
        if turn.prompt_ids[: len(held)] != held:
            drifts.append(len(held) - _common_prefix_len(held, turn.prompt_ids))
            builders.append(_SampleBuilder(rollout.rollout_id))
        builders[-1].append_turn(turn)
    return [b.to_training_sequence() for b in builders if b.has_trained_token()], drifts


@dataclass(frozen=True)
class Rollout:
    """
    One finished rollout, as returned by [`~experimental.zero_sync_grpo.ZeroSyncRolloutLoop.generate_one`].

    Args:
        completion (`list[dict[str, Any]]`):
            Messages generated after the prompt, in conversational format. Passed to the reward functions as
            `completions`.
        completion_ids (`list[int]`):
            Token ids of the generated completion. Passed to the reward functions as `completion_ids`.
        turns (`list[TurnRecord]`):
            The sampling requests that produced the rollout, in order. The training sequences are built from them: the
            sampled tokens are trained on, the rest of each prompt is context.
        tool_call_count (`int`, *optional*, defaults to `0`):
            Number of tool calls made during the rollout.
        tool_failure_count (`int`, *optional*, defaults to `0`):
            Number of those tool calls that failed.
        reward (`float`, *optional*):
            Reward computed while generating, for example by an external grader. It is added to the rewards of the
            reward functions and environments. If `None`, this rollout gets no reward from this source.
        rollout_id (`str`, *optional*):
            Identifies the rollout in its training rows, for debugging. Defaults to a new random hex id.
    """

    completion: Messages
    completion_ids: list[int]
    turns: list[TurnRecord]
    tool_call_count: int = 0
    tool_failure_count: int = 0
    reward: float | None = None
    rollout_id: RolloutId = field(default_factory=lambda: uuid.uuid4().hex)


@dataclass(slots=True)
class RolloutGroup:
    """
    The `num_generations` rollouts of one prompt, scored together. Each list has one entry per rollout.

    Args:
        prompts (`list[list[dict[str, Any]]]`):
            Prompt of each rollout.
        reward_kwargs (`dict[str, list[Any]]`):
            The other columns of the prompt's dataset row, passed to the reward functions as keyword arguments.
        completions (`list[list[dict[str, Any]]]`):
            Completion messages of each rollout.
        completions_ids (`list[list[int]]`):
            Completion token ids of each rollout.
        completions_sequences (`list[list[TrainingSequence]]`):
            Training sequences built from each rollout. A rollout whose re-tokenized history changed between turns
            gives more than one.
        tool_call_counts (`list[int]`):
            Number of tool calls in each rollout.
        tool_failure_counts (`list[int]`):
            Number of failed tool calls in each rollout.
        model_version (`int`):
            Policy version when the group's first rollout started.
        group_id (`int`):
            Identifier of the group, counted from `0` by the rollout loop.
        row_index (`int`):
            Position of the prompt in the epoch, in the order this process draws them.
        env_rewards (`list[tuple[type, float] or None]`):
            For a rollout whose environment defines `get_reward`, the environment class and its reward; else `None`.
        rollout_rewards (`list[float or None]`):
            [`~experimental.zero_sync_grpo.Rollout`]`.reward` of each rollout.
        queued_at (`float`, *optional*, defaults to `0.0`):
            `time.monotonic()` when the group was queued for scoring.
        rewards (`np.ndarray`, *optional*):
            Total reward of each rollout, summed over every reward source, NaN when no source scored it. Set before
            [`~experimental.zero_sync_grpo.ZeroSyncRolloutLoop.compute_advantages`] is called.
    """

    prompts: list[Messages]
    reward_kwargs: dict[str, list[Any]]
    completions: list[Messages]
    completions_ids: list[list[int]]
    completions_sequences: list[list[TrainingSequence]]
    tool_call_counts: list[int]
    tool_failure_counts: list[int]
    model_version: int
    group_id: int
    row_index: int
    env_rewards: list[tuple[type, float] | None]
    rollout_rewards: list[float | None]
    queued_at: float = 0.0
    rewards: np.ndarray | None = None


@dataclass(slots=True)
class RolloutSample:
    prompt: Messages
    completion: Messages
    input_ids: np.ndarray
    completion_mask: np.ndarray
    old_log_probs: np.ndarray
    sampling_mask: list[list[int]]
    sampler_ids: np.ndarray  # (len(input_ids), K), the sampler's most likely next tokens by position, padded with -1
    sampler_log_probs: np.ndarray  # (len(input_ids), K), their log-probs, padded with -inf
    advantage: float
    model_version: int
    group_id: int
    metrics: dict[str, float]
    enqueued_at: float | None = None


class ZeroSyncRolloutLoop:
    """
    Generates and scores rollouts for [`experimental.zero_sync_grpo.ZeroSyncGRPOTrainer`], on the trainer's continuous
    batching engine, in the training process.

    The default implements GRPO. To customize a run, subclass it, override only the methods that change, and set the
    subclass as the trainer's `loop_cls`. Each prompt goes through, in order:

    - [`~experimental.zero_sync_grpo.ZeroSyncRolloutLoop.filter_prompt`]: whether to generate for the prompt at all.
    - [`~experimental.zero_sync_grpo.ZeroSyncRolloutLoop.generate_one`]: one full rollout, which calls
      [`~experimental.zero_sync_grpo.ZeroSyncRolloutLoop.generate_one_turn`] for each sampling request.
    - The group of `num_generations` rollouts is scored: its rewards are the sum of the trainer's `reward_funcs`, of
      environments that define `get_reward`, and of [`~experimental.zero_sync_grpo.Rollout`]`.reward`. Then
      [`~experimental.zero_sync_grpo.ZeroSyncRolloutLoop.compute_advantages`] runs once per group.

    [`~experimental.zero_sync_grpo.ZeroSyncRolloutLoop.setup`] and
    [`~experimental.zero_sync_grpo.ZeroSyncRolloutLoop.close`] create and release resources on the loop's event loop,
    like an HTTP client. A hook can read `self.config` (a subclass of
    [`experimental.zero_sync_grpo.ZeroSyncGRPOConfig`] can carry its own fields) and `self.tokenizer`.
    """

    def __init__(
        self,
        config: ZeroSyncGRPOConfig,
        *,
        manager: ContinuousBatchingManager,
        reward_funcs: list[Callable[..., list[float]]],
        processing_class: PreTrainedTokenizerBase,
        eos_token_ids: list[int],
        metrics: dict[str, list],
        model_version_fn: Callable[[], int],
        tools: list[Callable] | None = None,
        environment_factory: Callable[[], object] | dict[str, Callable[[], object]] | None = None,
        score_queue_maxsize: int = 16,
    ):
        self.config = config
        self.prompts: Iterator[list[dict[str, Any]]] | None = None  # the epoch's prompt batches, set by the trainer
        self._epoch, self._row_index = 0, 0
        self.reward_funcs = reward_funcs
        self.reward_func_names = [get_callable_name(f) for f in reward_funcs]
        # `add_response_schema` sets the response template (transformers >= 5.13) or legacy schema for known chat
        # templates, so tool calls can be parsed. Skip if one is already set; warn if it's a migratable legacy schema.
        has_template = getattr(processing_class, "response_template", None) is not None
        has_schema = getattr(processing_class, "response_schema", None) is not None
        if not has_template and not has_schema:
            processing_class = add_response_schema(processing_class)
        elif has_schema and not has_template and _SUPPORTS_RESPONSE_TEMPLATE:
            warnings.warn(
                "The tokenizer has a legacy `response_schema` set but no `response_template`. The installed "
                "transformers supports the new-style `response_template`; consider migrating, as `response_schema` "
                "support will eventually be removed. See the Transformers response-parsing docs.",
                FutureWarning,
            )
        self.tokenizer = processing_class
        self.eos_token_ids = eos_token_ids
        self.rollout_buffer: deque[RolloutSample] = deque()  # scored samples, taken by the trainer
        self.manager = manager  # the trainer's continuous batching manager
        self._model_version_fn = model_version_fn  # the trainer's optimizer step
        self.metrics = metrics  # the trainer's metric sink, reduced in its `log()`
        # Metric accumulators
        self._counters: dict[str, float] = defaultdict(float)
        self._rates: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
        self._concurrency = 0  # set by the generate loop, reported by the score loop
        self._pushed_completion_tokens = 0
        self._pushed_at = time.monotonic()

        self.chat_template_kwargs = config.chat_template_kwargs or {}

        tools = tools or []
        self._standalone_tools = tools  # tools that are not bound to an environment
        self._env_reward_types = []  # env classes that own their reward via `get_reward`; one reward column each

        # Normalize `environment_factory` to a `{name: factory}` mapping. A single callable is the special case of one
        # unnamed (`None`) environment shared by every example; a dict maps the `environment` field of each example to
        # its factory. `_multi_environment` records which case we are in: only then is the `environment` field read.
        self._multi_environment = isinstance(environment_factory, dict)
        if environment_factory is None:
            self.environment_factories = None
        elif self._multi_environment:
            self.environment_factories = environment_factory
        else:
            self.environment_factories = {None: environment_factory}

        if self.environment_factories is not None:
            # Probe one instance of each environment to validate its `reset` method and find its reward. Instances
            # are pooled and reused (reset) across rollouts; the probe seeds the pool so it is not wasted. The pool
            # grows only when more concurrent instances of an environment are needed than have been created so far,
            # preserving the "construct once, reset often" contract even when rollouts mix environments.
            self._environment_pool = {}  # {environment name: list of reusable instances}
            # `self.tools` is the union of every environment's tools, accumulated below as each environment is probed.
            # Used only to decide whether a training chat template is needed.
            self.tools = list(tools)
            for name, factory in self.environment_factories.items():
                instance = factory()
                has_reset = False
                methods = []
                # List on the class: getmembers on the instance evaluates properties
                for member_name, _ in inspect.getmembers(type(instance), predicate=inspect.isfunction):
                    if member_name == "reset":
                        has_reset = True
                    elif member_name == "get_reward":
                        if type(instance) not in self._env_reward_types:
                            self._env_reward_types.append(type(instance))
                    elif not member_name.startswith("_"):
                        methods.append(getattr(instance, member_name))
                if not has_reset:
                    raise ValueError(
                        "Each environment instance returned by `environment_factory` must define a callable `reset`."
                    )
                self._environment_pool[name] = [instance]
                self.tools += [method for method in methods if method not in self.tools]
            # Each environment that owns its reward via `get_reward` becomes an extra reward source, named after the env
            # class (its per-rollout value is scored inline in `_generate_loop` and summed in `_score_group`). One
            # column per such class: mixing an env that owns its reward with one that does not is safe.
            for env_type in self._env_reward_types:
                self.reward_func_names.append(env_type.__name__)
        else:
            self.tools = tools

        # The reward a rollout carries in `Rollout.reward` is the last reward column, set by a `generate_one` that
        # scores its own rollouts. The built-in `generate_one` leaves it at `None`.
        self.reward_func_names.append("rollout")
        if (
            not self.reward_funcs
            and not self._env_reward_types
            and type(self).generate_one is ZeroSyncRolloutLoop.generate_one
        ):
            raise ValueError(
                "No reward source provided. Pass `reward_funcs`, an `environment_factory` whose environment defines a "
                "`get_reward` method, or a `loop_cls` whose `generate_one` sets `Rollout.reward`."
            )

        # Sync tools run here so a slow one never blocks the event loop; sized so every rollout can be in a tool call.
        self._tool_pool = ThreadPoolExecutor(
            max_workers=max(1, config.max_concurrent_rollouts), thread_name_prefix="grpo-tool"
        )

        self._concurrency_cap = config.max_concurrent_rollouts
        self._total_rollouts = 0

        # The chat template must be prefix-preserving in multi-turn training; if the tokenizer's
        # template isn't, swap in a training-safe one.
        if self.tools and not is_chat_template_prefix_preserving(self.tokenizer):
            self.chat_template = get_training_chat_template(self.tokenizer)
        else:
            self.chat_template = None

        self._groups_to_score: asyncio.Queue[RolloutGroup | None] = asyncio.Queue(maxsize=score_queue_maxsize)
        self._total_completion_tokens = 0
        self._total_groups_scored = 0
        # Each engine request in flight, mapped to the result its `generate_one_turn` awaits
        self._turn_results: dict[str, asyncio.Future] = {}
        # What `drained` reads: the epoch's prompts once every one is started, and the groups queued but not scored
        self._exhausted_prompts = None
        self._groups_unscored = 0

        self._loop = asyncio.new_event_loop()
        self._stop_event = asyncio.Event()
        self._run_task: asyncio.Task | None = None

    @property
    def model_version(self) -> int:
        return self._model_version_fn()

    def run_until(self, done: Callable[[], bool]) -> None:
        """
        Run the loops until `done()` holds. They are suspended when this returns, with every rollout in flight, and
        carry on at the next call: the trainer pauses the engine and runs its step in between.
        """
        if self._run_task is None:
            self._run_task = self._loop.create_task(self._run_loops(stop_event=self._stop_event))

        async def wait() -> None:
            while True:
                await asyncio.sleep(0.001)  # at least one turn of the loops, whatever `done()` says
                if self._run_task.done():
                    self._run_task.result()  # a loop failed: raise its error here
                if done():
                    return

        self._loop.run_until_complete(wait())

    def drained(self) -> bool:
        """Whether every prompt of the epoch has been generated and scored, and its samples put in the buffer."""
        return self._exhausted_prompts is self.prompts and self._concurrency == 0 and self._groups_unscored == 0

    def stop(self) -> None:
        """Stop the loops, dropping the rollouts in flight, and release their resources."""
        self._stop_event.set()
        if self._run_task is not None:
            self._run_task.cancel()
            self._loop.run_until_complete(asyncio.gather(self._run_task, return_exceptions=True))
        self._tool_pool.shutdown(wait=True, cancel_futures=True)
        self._loop.close()

    async def setup(self) -> None:
        """
        Create the resources the other hooks need, such as an HTTP client. Runs on the loop's event loop before the
        first prompt. Does nothing by default.
        """

    async def close(self) -> None:
        """
        Release what [`~experimental.zero_sync_grpo.ZeroSyncRolloutLoop.setup`] created. Runs when the loops stop,
        including on failure. Does nothing by default.
        """

    async def _run_loops(self, stop_event: asyncio.Event) -> None:
        logger.info(
            f"rollout loop started: num_generations={self.config.num_generations}, "
            f"max_concurrent_rollouts={self.config.max_concurrent_rollouts}, concurrency_cap={self._concurrency_cap}, "
            f"temperature={self.config.temperature}, top_p={self.config.top_p}, top_k={self.config.top_k}"
        )
        await self.setup()
        try:
            await asyncio.gather(
                asyncio.create_task(self._generate_loop(stop_event=stop_event)),
                asyncio.create_task(self._score_loop(stop_event=stop_event)),
                asyncio.create_task(self._result_loop(stop_event=stop_event)),
            )
        finally:
            await self.close()

    async def _result_loop(self, stop_event: asyncio.Event) -> None:
        """Hand each finished engine request to the `generate_one_turn` awaiting it.

        A dead background thread never raises from `get_result` (it returns None forever, transformers#48334), so poll
        `fatal_error` to fail fast instead of spinning.
        """
        while not stop_event.is_set():
            if not self._turn_results:  # no request in flight
                await asyncio.sleep(0.001)
                continue
            output = self.manager.get_result(timeout=0)
            if output is None:
                fatal_error = self.manager.background_thread_status.fatal_error
                if fatal_error is not None:
                    raise RuntimeError("The continuous batching background thread died.") from fatal_error
                await asyncio.sleep(0.001)
                continue
            self._turn_results.pop(output.request_id).set_result(output)

    async def _generate_loop(self, stop_event: asyncio.Event) -> None:
        pending_groups: dict[int, RolloutGroup] = {}
        pending_completed: dict[int, int] = {}
        rollout_tasks: dict[asyncio.Task, tuple[int, Any, object, Messages, float]] = {}
        work_iter = self._repeat_iterator()
        loop = asyncio.get_running_loop()

        try:
            while True:
                while len(rollout_tasks) < self._concurrency_cap and not stop_event.is_set():
                    work = next(work_iter)
                    if work is None:  # the epoch's prompts are all started
                        break
                    group_id, row_index, row = work
                    if row is None:  # rejected by filter_prompt
                        continue
                    # The environment is selected per example via its `environment` field (multi-env); only its tools
                    # are exposed in the example's prompt. When there are no environments, every example shares the
                    # standalone tools.
                    name = row.get("environment") if self._multi_environment else None
                    if self._multi_environment and name not in self.environment_factories:
                        raise ValueError(
                            f"Example has `environment={name!r}`, which is not among the environments passed to "
                            f"`environment_factory`. Expected one of: {list(self.environment_factories)}."
                        )
                    # Draw a reusable environment instance for this rollout (creating one only if the pool is
                    # exhausted); it is returned to the pool when the task completes. Reset it BEFORE building the
                    # prompt and capture its initial observation (e.g. a task instruction). The reset() return was
                    # previously discarded, so an environment_factory whose task lives in the observation (not the
                    # dataset prompt) was generated against the bare prompt. Mirror GRPOTrainer: fold the observation
                    # into the last prompt message. reset() may be stochastic, so this is done per generation (each
                    # generation gets its own observation -> its own prompt and prompt_ids). `environment` is a control
                    # field in multi-environment mode, so it is not forwarded to `reset`.
                    environment = None
                    observation = None
                    if self.environment_factories is not None:
                        pool = self._environment_pool[name]
                        environment = pool.pop() if pool else self.environment_factories[name]()
                        reset_kwargs = (
                            {k: v for k, v in row.items() if k != "environment"} if self._multi_environment else row
                        )
                        t0 = time.monotonic()
                        observation = await loop.run_in_executor(
                            self._tool_pool, functools.partial(environment.reset, **reset_kwargs)
                        )
                        self._rates["env/reset_s"][0] += time.monotonic() - t0
                        self._rates["env/reset_s"][1] += 1
                    # `prompt` is optional only when an environment owns the data; `reset()` then supplies it. Without
                    # an environment, a missing `prompt` is a malformed dataset and must still fail fast (KeyError).
                    if "prompt" not in row and self.environment_factories is not None:
                        prompt = [{"role": "user", "content": ""}]
                    else:
                        prompt = row["prompt"]
                    if observation is not None:
                        # Rebuild the last message instead of mutating in place (as GRPOTrainer does): the
                        # same row is reused across the group's generations and across epochs. Normalize str vs
                        # list (multimodal) content the same way GRPOTrainer does before concatenating.
                        last = prompt[-1]
                        content = last["content"]
                        if isinstance(observation, list) and isinstance(content, str):
                            content = [{"type": "text", "text": content}]
                        if isinstance(observation, str) and isinstance(content, list):
                            observation = [{"type": "text", "text": observation}]
                        prompt = prompt[:-1] + [{**last, "content": content + observation}]

                    # This rollout's tools: the standalone tools plus the methods of its own environment instance.
                    methods = []
                    if environment is not None:
                        methods = [
                            getattr(environment, member_name)
                            for member_name, _ in inspect.getmembers(type(environment), predicate=inspect.isfunction)
                            if member_name not in ("reset", "get_reward") and not member_name.startswith("_")
                        ]
                    tools = self._standalone_tools + methods

                    if group_id not in pending_groups:
                        reward_kwargs = {
                            key: [row[key]] * self.config.num_generations
                            for key in row
                            if key not in {"prompt", "completion", "completion_ids"}
                        }
                        pending_groups[group_id] = RolloutGroup(
                            prompts=[],
                            reward_kwargs=reward_kwargs,
                            completions=[],
                            completions_ids=[],
                            completions_sequences=[],
                            tool_call_counts=[],
                            tool_failure_counts=[],
                            model_version=self.model_version,
                            group_id=group_id,
                            row_index=row_index,
                            env_rewards=[],
                            rollout_rewards=[],
                        )
                        pending_completed[group_id] = 0

                    task = asyncio.create_task(self.generate_one(prompt, tools, group_id=group_id))
                    rollout_tasks[task] = (group_id, name, environment, prompt, time.monotonic())

                self._concurrency = len(rollout_tasks)  # gauge, reported by the score loop's push
                if not rollout_tasks:
                    if stop_event.is_set():
                        return
                    await asyncio.sleep(0.01)
                    continue

                done, _ = await asyncio.wait(rollout_tasks, return_when=asyncio.FIRST_COMPLETED, timeout=0.1)
                if not done:
                    continue

                for task in done:
                    group_id, name, environment, prompt, t_dispatch = rollout_tasks.pop(task)
                    if task.exception() is not None:
                        raise task.exception()
                    self._total_rollouts += 1

                    rollout = task.result()
                    sequences, drifts = _chain_to_sequences(rollout)
                    if rollout.turns:
                        self._push_rollout_metrics(
                            turns=len(rollout.turns),
                            sequences=len(sequences),
                            completion_ids=rollout.completion_ids,
                            drifts=drifts,
                            duration_s=time.monotonic() - t_dispatch,
                        )
                    group = pending_groups[group_id]
                    group.prompts.append(prompt)
                    group.completions.append(rollout.completion)
                    group.completions_ids.append(rollout.completion_ids)
                    group.completions_sequences.append(sequences)
                    group.tool_call_counts.append(rollout.tool_call_count)
                    group.tool_failure_counts.append(rollout.tool_failure_count)
                    group.rollout_rewards.append(rollout.reward)
                    # The environment owns the reward: score it now, while this rollout's environment still holds its
                    # final state and before returning it to the pool. `get_reward` may be async awaiting yields to
                    # in-flight requests instead of halting them. The env is returned to the pool only after scoring, so
                    # a concurrent rollout can't draw and reset it during the await. Record `(env class, reward)` so
                    # `_score_group` can place it in the matching env's reward column; rollouts whose env owns no reward
                    # record `None` (turned into NaN and ignored) to stay aligned with the group's other per-rollout lists.
                    if self._env_reward_types:
                        env_type = type(environment)
                        if env_type in self._env_reward_types:
                            get_reward = environment.get_reward
                            t0 = time.monotonic()
                            if inspect.iscoroutinefunction(get_reward):
                                reward = await get_reward()
                            else:
                                reward = await loop.run_in_executor(self._tool_pool, get_reward)
                            self._rates["env/reward_s"][0] += time.monotonic() - t0
                            self._rates["env/reward_s"][1] += 1
                            group.env_rewards.append((env_type, reward))
                        else:
                            group.env_rewards.append(None)
                    if environment is not None:
                        self._environment_pool[name].append(environment)
                    self._total_completion_tokens += len(rollout.completion_ids)
                    pending_completed[group_id] += 1

                    if pending_completed[group_id] == self.config.num_generations:
                        self._groups_unscored += 1
                        group.queued_at = time.monotonic()
                        t_blocked = None
                        while True:
                            try:
                                self._groups_to_score.put_nowait(group)
                                break
                            except asyncio.QueueFull:
                                if stop_event.is_set():
                                    return
                                if t_blocked is None:
                                    t_blocked = time.monotonic()
                                await asyncio.sleep(0.1)
                        if t_blocked is not None:
                            # Generation held back by scoring
                            self._push_metrics({"rollout/score_block_s": time.monotonic() - t_blocked})
                        del pending_groups[group_id]
                        del pending_completed[group_id]
        finally:
            for task in rollout_tasks:
                task.cancel()
            if rollout_tasks:
                await asyncio.gather(*rollout_tasks, return_exceptions=True)
            try:
                self._groups_to_score.put_nowait(None)
            except asyncio.QueueFull:
                pass

    async def _score_loop(self, stop_event: asyncio.Event) -> None:
        t_idle_start = time.monotonic()
        while not stop_event.is_set():
            try:
                group = await asyncio.wait_for(self._groups_to_score.get(), timeout=0.5)
            except TimeoutError:
                continue
            if group is None:
                return
            score_idle = time.monotonic() - t_idle_start
            wait_scoring = time.monotonic() - group.queued_at

            if score_idle > 0.5:
                logger.info(f"[score] waited {score_idle:.1f}s for a group to score")

            t0 = time.monotonic()
            samples = await self._score_group(group)
            scoring_time = time.monotonic() - t0
            logger.info(
                f"[score] scored {len(samples)} samples in {scoring_time:.2f}s, buffer_size={len(self.rollout_buffer)}"
            )

            now = time.monotonic()
            self._push_metrics(
                {
                    "rollout/score_s": scoring_time,
                    "rollout/score_wait_s": wait_scoring,
                    "rollout/score_queue_size": float(self._groups_to_score.qsize()),
                    "rollout/concurrency": float(self._concurrency),
                    # Windowed, not cumulative.  A cumulative average cannot show a generation stall.
                    "rollout/generated_tok_s": (
                        float(self._total_completion_tokens - self._pushed_completion_tokens),
                        now - self._pushed_at,
                    ),
                }
            )
            self._pushed_completion_tokens = self._total_completion_tokens
            self._pushed_at = now

            if self.config.log_completions and samples:
                print_prompt_completions_sample(
                    prompts=[s.prompt for s in samples],
                    completions=[s.completion for s in samples],
                    rewards={"reward": [s.metrics["reward_kept"] for s in samples]},
                    advantages=[s.advantage for s in samples],
                    step=self._total_groups_scored,
                    num_samples=self.config.num_completions_to_print,
                )
            self._total_groups_scored += 1

            for item in samples:
                if not await self._put_in_buffer(item, stop_event):
                    return
            self._groups_unscored -= 1

            t_idle_start = time.monotonic()

    async def _put_in_buffer(self, item: RolloutSample, stop_event: asyncio.Event) -> bool:
        """Put `item` in the rollout buffer. Returns `False` if `stop_event` is set first."""
        if stop_event.is_set():
            return False
        item.enqueued_at = time.time()
        self.rollout_buffer.append(item)
        return True

    def _push_metrics(self, values: dict[str, float | tuple[float, float]]) -> None:
        """Send metrics to the trainer, along with the counters and rates accumulated since the last push."""
        payload = dict(values)
        payload.update(self._counters)
        payload.update({key: (num, den) for key, (num, den) in self._rates.items()})
        self._counters.clear()
        self._rates.clear()
        for key, value in payload.items():
            self.metrics[key].append(value)

    def _push_rollout_metrics(
        self,
        *,
        turns: int,
        sequences: int,
        completion_ids: list[int],
        drifts: list[int],
        duration_s: float,
    ) -> None:
        """One conversation's structure and its completion, aggregated per rollout."""
        completion_tokens = len(completion_ids)
        self._rates["completions/mean_length"][0] += completion_tokens
        self._rates["completions/mean_length"][1] += 1
        self._rates["rollout/samples_per_rollout"][0] += sequences
        self._rates["rollout/samples_per_rollout"][1] += 1
        self._rates["rollout/turns_mean"][0] += turns
        self._rates["rollout/turns_mean"][1] += 1
        if completion_ids:
            # NOTE(@aminediro):
            # Truncation is read off the same way [`GRPOTrainer`] and [`RLOOTrainer`] define
            # `completions/clipped_ratio`: a completion that does not end on EOS (or pad) was cut off by
            # `max_new_tokens` rather than finishing. Deliberately NOT the engine's own status, so the metric means the
            # same thing here
            eos_and_pad = (*self.eos_token_ids, self.tokenizer.pad_token_id)
            self._rates["completions/clipped_ratio"][0] += completion_ids[-1] not in eos_and_pad
            self._rates["completions/clipped_ratio"][1] += 1
        if turns > 1:
            self._rates["rollout/fork_frac"][0] += len(drifts)
            self._rates["rollout/fork_frac"][1] += turns - 1
            self._rates["rollout/drift_tokens_mean"][0] += sum(drifts)
            self._rates["rollout/drift_tokens_mean"][1] += turns - 1
        self._push_metrics(
            {
                "rollout/duration_s": duration_s,
                "rollout/turns_max": float(turns),
                "rollout/drift_tokens_max": float(max(drifts, default=0)),
                "completions/max_length": float(completion_tokens),
                "completions/min_length": float(completion_tokens),
            }
        )

    def filter_prompt(self, row_index: int, row: dict[str, Any], epoch: int) -> bool:
        """
        Whether to generate for a prompt. A rejected prompt is skipped for this epoch. Admits every prompt by default.
        Rejecting every prompt of a full epoch raises, since no batch can reach the trainer after that.

        Args:
            row_index (`int`):
                Position of the prompt in the epoch, in the order this process draws them.
            row (`dict[str, Any]`):
                The dataset row.
            epoch (`int`):
                Number of full passes over the dataset before this one.

        Returns:
            `bool`: `True` to generate `num_generations` rollouts for the prompt, `False` to skip it.
        """
        return True

    def _repeat_iterator(self) -> Iterator[tuple[int, int, dict[str, Any] | None] | None]:
        """Yield `(group_id, row_index, row)` once per generation of each prompt, or `(group_id, row_index, None)` once
        for a prompt that `filter_prompt` rejects. Yields `None` once the epoch's prompts are all started, until the
        trainer sets the next epoch's."""
        group_id = 0
        prompts = rows = None
        accepted_in_epoch = False
        while True:
            if self.prompts is not prompts:
                if prompts is not None:
                    self._row_index, self._epoch = 0, self._epoch + 1
                prompts, rows, accepted_in_epoch = self.prompts, chain.from_iterable(self.prompts), False
            row = next(rows, None)
            if row is None:
                self._exhausted_prompts = prompts
                if not accepted_in_epoch:
                    raise RuntimeError(
                        f"`filter_prompt` rejected every prompt of epoch {self._epoch}, so no batch can reach the "
                        "trainer."
                    )
                yield None
                continue
            if self.filter_prompt(self._row_index, row, self._epoch):
                accepted_in_epoch = True
                for _ in range(self.config.num_generations):
                    yield group_id, self._row_index, row
            else:
                yield group_id, self._row_index, None
            group_id += 1
            self._row_index += 1

    async def generate_one(self, prompt: Messages, tools: list[Callable], group_id: int = 0) -> Rollout:
        """
        Generate one rollout for a prompt.

        By default, a multi-turn tool loop: each turn renders the whole conversation with the chat template, samples a
        reply with [`~experimental.zero_sync_grpo.ZeroSyncRolloutLoop.generate_one_turn`] and runs the tool calls it
        contains, until the model stops calling tools or `max_tool_calling_iterations` is reached. Override it to
        generate differently, for example with an agent framework. Return every sampling request in `Rollout.turns`,
        since the training sequences are built from them, and set `Rollout.reward` to score the rollout here.

        Args:
            prompt (`list[dict[str, Any]]`):
                Prompt messages.
            tools (`list[Callable]`):
                Tools available to this rollout: the trainer's `tools`, plus the public methods of the environment when
                using `environment_factory`.
            group_id (`int`, *optional*, defaults to `0`):
                Identifier of the group the rollout belongs to.

        Returns:
            [`~experimental.zero_sync_grpo.Rollout`]: The finished rollout.
        """
        messages = list(prompt)  # a MESSAGE list, not a token list
        tool_dict = {tool.__name__: tool for tool in tools}
        turns: list[TurnRecord] = []
        completion, completion_ids = [], []
        tool_call_count = 0
        tool_failure_count = 0
        iteration_num = 0
        loop_exhausted = False
        max_iterations = self.config.max_tool_calling_iterations
        while True:
            prompt_ids = self.tokenizer.apply_chat_template(  # re-tokenize the WHOLE conversation
                messages,
                return_dict=False,
                add_generation_prompt=True,
                tools=tools or None,
                chat_template=self.chat_template,
                **self.chat_template_kwargs,
            )
            turn = await self.generate_one_turn(prompt_ids)
            assistant_message = parse_response(self.tokenizer, turn.output_ids, prefix=prompt_ids, tools=tools)
            completion.append(assistant_message)
            completion_ids.extend(turn.output_ids)
            messages.append(assistant_message)
            turns.append(turn)
            tool_calls = assistant_message.get("tool_calls")
            if tool_calls is None:
                break
            if max_iterations is not None and iteration_num >= max_iterations:
                # Cut off mid-loop: the model asked for another tool and we refused. The conversation is trained as if
                # it had finished, so this is a silent truncation — hence the metric.
                loop_exhausted = True
                break
            tool_messages, n_calls, n_failures = await self._execute_tool_calls(tool_calls, tool_dict)
            tool_call_count += n_calls
            tool_failure_count += n_failures
            completion.extend(tool_messages)
            messages.extend(tool_messages)  # tool result goes back as a MESSAGE, re-tokenized next turn
            iteration_num += 1
        if self.tools:
            # Only meaningful when the model has tools to call: with none, the loop always ends on the first turn and
            # the metric would be a constant 0 cluttering every run that does not use tools.
            self._rates["tools/loop_exhausted_frac"][0] += loop_exhausted
            self._rates["tools/loop_exhausted_frac"][1] += 1
        return Rollout(completion, completion_ids, turns, tool_call_count, tool_failure_count)

    async def _execute_tool_calls(
        self, tool_calls: list[dict[str, Any]], tool_dict: dict[str, Callable]
    ) -> tuple[list[dict[str, str]], int, int]:
        loop = asyncio.get_running_loop()
        tool_messages = []
        n_calls = 0
        n_failures = 0
        for tool_call in tool_calls:
            n_calls += 1
            function = tool_call["function"]
            name = function["name"]
            self._counters[f"tools/{name}/call_total"] += 1
            tool = tool_dict.get(name)
            if tool is None:
                # A hallucinated tool name is a policy error that should decay with training, unlike a tool that ran
                # and raised — which is an environment problem. Counted apart so the two are not one number.
                n_failures += 1
                self._counters["tools/unknown_name_total"] += 1
                self._counters[f"tools/{name}/failure_total"] += 1
                tool_messages.append({"role": "tool", "name": name, "content": str({"error": f"unknown tool {name}"})})
                continue
            t0 = time.monotonic()
            try:
                arguments = function.get("arguments", {})
                if inspect.iscoroutinefunction(tool):
                    result = await tool(**arguments)
                else:
                    result = await loop.run_in_executor(self._tool_pool, functools.partial(tool, **arguments))
            except Exception as error:
                n_failures += 1
                self._counters[f"tools/{name}/failure_total"] += 1
                result = {"error": str(error)}
            elapsed = time.monotonic() - t0
            self._rates["tools/latency_s"][0] += elapsed
            self._rates["tools/latency_s"][1] += 1
            self._rates[f"tools/{name}/latency_s"][0] += elapsed
            self._rates[f"tools/{name}/latency_s"][1] += 1
            tool_messages.append({"role": "tool", "name": name, "content": str(result)})
        self._rates["tools/parallel_calls_mean"][0] += len(tool_calls)
        self._rates["tools/parallel_calls_mean"][1] += 1
        return tool_messages, n_calls, n_failures

    async def generate_one_turn(self, prompt_ids: list[int]) -> TurnRecord:
        """
        Sample one reply from the trainer's continuous batching engine, with the config's sampling settings
        (`temperature`, `top_p`, `top_k` and `max_completion_length`).

        Args:
            prompt_ids (`list[int]`):
                Token ids of the full prompt.

        Returns:
            [`TurnRecord`]: The prompt and the sampled tokens with their log-probabilities.
        """
        request_id = self.manager.add_request(prompt_ids, max_new_tokens=self.config.max_completion_length)
        self._turn_results[request_id] = self._loop.create_future()
        output = await self._turn_results[request_id]
        if output.error is not None:
            raise RuntimeError(f"The continuous batching engine failed the request: {output.error}")
        output_ids = list(output.generated_tokens)
        return TurnRecord(prompt_ids, output_ids, list(output.logprobs[: len(output_ids)]))

    async def _calculate_rewards(self, group: RolloutGroup) -> np.ndarray:
        """Score every rollout of `group` with each reward source.

        Returns:
            `np.ndarray` of shape `(len(reward_func_names), len(group.completions))`, NaN where a reward function
            returned `None`.
        """
        kwargs = dict(
            completions=group.completions,
            prompts=group.prompts,
            completion_ids=group.completions_ids,
            **group.reward_kwargs,
        )
        all_rewards = await asyncio.gather(
            *[
                reward_func(**kwargs) if is_async_callable(reward_func) else asyncio.to_thread(reward_func, **kwargs)
                for reward_func in self.reward_funcs
            ]
        )
        # Each environment that owns its reward contributes one reward column (named after its env class, in
        # `_env_reward_types` order to match `reward_func_names`), summed in with weight 1 like any reward func. Scores
        # were captured per rollout at generation time in `group.env_rewards` as `(env class, reward)`; a rollout is
        # placed in a column only when its env is that class, so mixing reward-owning and plain envs is safe.
        for env_type in self._env_reward_types:
            column = [r[1] if (r is not None and r[0] is env_type) else None for r in group.env_rewards]
            all_rewards = [*all_rewards, column]

        all_rewards = [*all_rewards, list(group.rollout_rewards)]

        # Reward funcs may return None per-sample (unparseable gold). Convert to NaN.
        all_rewards = [[r if r is not None else float("nan") for r in row] for row in all_rewards]
        return np.array(all_rewards, dtype=float)

    def compute_advantages(self, group: RolloutGroup) -> np.ndarray:
        """
        Compute one advantage per rollout of a group. By default, each reward standardized within the group; a rollout
        with a NaN reward gets advantage `0` and is left out of the mean and std.

        It runs for every group. A group whose scored rewards are all equal is then dropped, whatever advantages this
        returns.

        Args:
            group ([`~experimental.zero_sync_grpo.zero_sync_grpo_trainer.RolloutGroup`]):
                The `num_generations` rollouts of one prompt, with their total rewards in `group.rewards`.

        Returns:
            `np.ndarray`: One advantage per rollout, in the order of `group.rewards`.
        """
        rewards = group.rewards
        scored_mask = ~np.isnan(rewards)
        scored = rewards[scored_mask]
        advantages = np.zeros_like(rewards)
        if scored_mask.any():
            advantages[scored_mask] = (scored - scored.mean()) / (scored.std() + 1e-8)
        return advantages

    async def _score_group(self, group: RolloutGroup) -> list[RolloutSample]:
        per_func_rewards = await self._calculate_rewards(group)
        # A rollout for which every func returned None is unscorable: nansum would give 0 and the row would pull
        # the policy away from actually-correct answers.
        rewards = np.nansum(per_func_rewards, axis=0)
        rewards[np.all(np.isnan(per_func_rewards), axis=0)] = np.nan
        group.rewards = rewards
        advantages = self.compute_advantages(group)

        scored = rewards[~np.isnan(rewards)]
        if len(scored):
            reward_mean = float(scored.mean())
            reward_std = float(scored.std())
        else:
            reward_mean = reward_std = float("nan")
        logger.info(f"Rollout metrics: reward_mean={reward_mean:.4f}, reward_std={reward_std:.4f}")

        self._rates["reward"][0] += float(scored.sum())
        self._rates["reward"][1] += len(scored)
        zero_std = bool(np.all(scored == scored[:1]))
        self._rates["frac_reward_zero_std"][0] += zero_std
        self._rates["frac_reward_zero_std"][1] += 1
        if zero_std:
            return []

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
        # A reward source that scored no rollout of the group (e.g. the `rollout` column when `generate_one` sets no
        # reward) is not logged, rather than logged as NaN.
        used_columns = ~np.all(np.isnan(per_func_rewards), axis=1)

        per_func_std = [float(np.nanstd(r)) if not np.isnan(r).all() else float("nan") for r in per_func_rewards]
        # Option 3 advantage: one advantage per conversation, stamped on every training row it produced.
        # Under TRL's global token-mean loss a fork is invisible to the loss (see slime_research.md B.5),
        # so we do not split the reward or the advantage across a conversation's rows.
        samples = []
        for i, (prompt, completion, sequences, advantage, reward, tm) in enumerate(
            zip(
                group.prompts,
                group.completions,
                group.completions_sequences,
                advantages,
                rewards,
                tool_metrics,
                strict=True,
            )
        ):
            metrics = {
                "reward_kept": float(reward),
                "reward_group_std": reward_std,
                **{
                    f"rewards/{name}/mean": float(func_reward)
                    for name, func_reward, used in zip(
                        self.reward_func_names, per_func_rewards[:, i], used_columns, strict=True
                    )
                    if used
                },
                **{
                    f"rewards/{name}/group_std": std
                    for name, std, used in zip(self.reward_func_names, per_func_std, used_columns, strict=True)
                    if used
                },
                **tm,
            }
            for seq in sequences:
                width = max(map(len, seq.sampler_logprobs))
                shape = (len(seq.input_ids), width)
                tops = seq.sampler_logprobs
                sampler_ids = np.fromiter(
                    chain.from_iterable(chain(top, repeat(-1, width - len(top))) for top in tops),
                    dtype=np.int64,
                    count=shape[0] * width,
                ).reshape(shape)
                sampler_log_probs = np.fromiter(
                    chain.from_iterable(chain(top.values(), repeat(-math.inf, width - len(top))) for top in tops),
                    dtype=np.float32,
                    count=shape[0] * width,
                ).reshape(shape)
                samples.append(
                    RolloutSample(
                        prompt=prompt,
                        completion=completion,
                        input_ids=np.asarray(seq.input_ids, dtype=np.int64),
                        completion_mask=np.asarray(seq.completion_mask, dtype=np.int64),
                        old_log_probs=np.asarray(seq.old_log_probs, dtype=np.float32),
                        sampling_mask=seq.sampling_mask,
                        sampler_ids=sampler_ids,
                        sampler_log_probs=sampler_log_probs,
                        advantage=float(advantage),
                        model_version=group.model_version,
                        group_id=group.group_id,
                        metrics=dict(metrics),  # own copy per row
                    )
                )
        return samples


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
                    **sample.metrics,  # reward_kept, reward_group_std, rewards/<func>/{mean,group_std}, tools/*_frequency
                    "advantage": float(sample.advantage),
                    "group_id": int(sample.group_id),
                    "model_version": int(sample.model_version),
                    "prompt_tokens": len(sample.input_ids) - int(sample.completion_mask.sum()),
                    "completion_tokens": int(sample.completion_mask.sum()),
                },
            )
            for sample in samples[:max_traces]
        ]
        trackio.log({"rollouts": traces}, step=step)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"rollout trace logging skipped: {type(e).__name__}: {e}")


class _RolloutStream(torch.utils.data.IterableDataset):
    """Prompts in, finished rollouts out: what the training loop iterates.

    One pass over the prompt dataset yields `num_generations` rollouts per prompt, so the stream is that many items
    long. That length is all the `Trainer` needs to size its epochs, its schedule and its progress, which is why
    feeding the engine and waiting on it can live here instead of in the training step.

    Each rank generates its own rollouts, so the stream is already rank-local and must not be sharded again.
    """

    def __init__(self, trainer, prompt_loader):
        self._trainer = trainer
        self._prompt_loader = prompt_loader

    def __len__(self):
        return len(self._prompt_loader.dataset) * self._trainer.args.num_generations

    def __iter__(self):
        prompts = iter(self._prompt_loader)
        while True:
            # A whole optimizer step at a time. The `Trainer` collects every micro-batch of a step before it trains on
            # the first, so one pause per step is all training needs. The wait, the pause and the pool exchange are
            # also collectives: every rank runs them once per step, so the cadence is the same on all of them and none
            # can arrive at a collective its peers have already left.
            samples = self._trainer._next_rollout_batch(prompts)
            if not samples:
                return
            yield from samples


class ZeroSyncGRPOTrainer(_BaseTrainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language
    Models](https://huggingface.co/papers/2402.03300). This trainer is the zero-sync version of GRPO, where generation
    and training share one copy of the weights: a transformers continuous batching engine generates from the parameter
    tensors the optimizer updates in place, so there is no second engine and no weight synchronization. The engine is
    paused for each training step and resumes with its rollouts in flight.

    Example:

    ```python
    >>> from trl.experimental.zero_sync_grpo import ZeroSyncGRPOTrainer
    >>> from trl.rewards import accuracy_reward
    >>> from datasets import load_dataset

    >>> dataset = load_dataset("trl-lib/DeepMath-103K", split="train")

    >>> trainer = ZeroSyncGRPOTrainer(
    ...     model="Qwen/Qwen2.5-0.5B-Instruct",
    ...     reward_funcs=accuracy_reward,
    ...     train_dataset=dataset,
    ... )
    >>> trainer.train()
    ```

    Args:
        model (`str` or [`~transformers.PreTrainedModel`]):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or a
              path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
              with the architecture declared in its config: [`~transformers.AutoModelForImageTextToText`] for a
              vision-language checkpoint (whose vision tower is loaded but frozen),
              [`~transformers.AutoModelForCausalLM`] otherwise.
            - A [`~transformers.PreTrainedModel`] object.
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

        args ([`experimental.zero_sync_grpo.ZeroSyncGRPOConfig`], *optional*):
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

    To customize which prompts are sampled, how a rollout is generated or how advantages are computed, subclass
    [`experimental.zero_sync_grpo.ZeroSyncRolloutLoop`] and set it as the trainer's `loop_cls`. TRL builds the loop
    when generation starts:

    ```python
    >>> class CurriculumTrainer(ZeroSyncGRPOTrainer):
    ...     loop_cls = CurriculumRolloutLoop
    ```
    """

    loop_cls: type[ZeroSyncRolloutLoop] = ZeroSyncRolloutLoop
    _rollout_loop_kwargs: dict[str, Any] = {}
    loss_is_scaled_for_ga = True

    _tag_names = ["trl", "zero-sync-grpo"]
    _name = "Zero-Sync GRPO"
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
        model: "str | PreTrainedModel",
        reward_funcs: RewardFunc | list[RewardFunc] | None = None,
        args: ZeroSyncGRPOConfig | None = None,
        train_dataset: Dataset | IterableDataset | None = None,
        processing_class: PreTrainedTokenizerBase | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None] = (None, None),
        tools: list[Callable] | None = None,
        environment_factory: EnvironmentFactory | dict[str, EnvironmentFactory] | None = None,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else get_config_model_id(model.config)
            model_name = model_name.split("/")[-1]
            args = ZeroSyncGRPOConfig(f"{model_name}-ZeroSyncGRPO")

        # Training arguments
        self.epsilon_low = args.epsilon
        self.epsilon_high = args.epsilon_high if args.epsilon_high is not None else args.epsilon
        self.temperature = args.temperature
        self.tp_size = args.tp_size

        # Model
        if isinstance(model, str):
            model_init_kwargs = dict(args.model_init_kwargs or {})  # copy to avoid mutating model_init_kwargs
            model_init_kwargs.setdefault("trust_remote_code", args.trust_remote_code)
            # The same weights both generate and train, so default to the checkpoint dtype rather than fp32
            model_init_kwargs.setdefault("dtype", "auto")
            # Distributed training requires device_map=None ("auto" fails)
            if args.distributed_state.distributed_type in ["MULTI_GPU", "DEEPSPEED"]:
                model_init_kwargs["device_map"] = None
            if args.tp_size > 1:
                # The model is split across the processes rather than placed on one device, and the two ways of
                # deciding where parameters go are mutually exclusive.
                model_init_kwargs["distributed_config"] = DistributedConfig(tp_size=args.tp_size)
                model_init_kwargs["device_map"] = None
            model_revision = model_init_kwargs.get("revision")
            model = create_model_from_path(model, **model_init_kwargs)
            if "attn_implementation" not in model_init_kwargs:
                # Training packs the samples of a micro-batch in one flat row whose boundaries travel as the flash
                # varlen kwargs; sdpa reads them from the position ids instead. Linear attention layers (hybrid
                # models like Qwen3.5) scan the row with varlen kernels and have no sdpa path.
                if is_flash_attn_3_available(kernels_fallback_ok=True):
                    model.set_attn_implementation("flash_attention_3")
                elif is_flash_attn_2_available(kernels_fallback_ok=True):
                    model.set_attn_implementation("flash_attention_2")
                elif "linear_attention" in (getattr(model.config.get_text_config(), "layer_types", None) or []):
                    raise ValueError("Packed training on a hybrid model requires flash attention")
                else:
                    model.set_attn_implementation("sdpa")
        else:
            model_revision = None
            if args.model_init_kwargs is not None:
                logger.warning(
                    "You passed `model_init_kwargs` to the `ZeroSyncGRPOConfig`, but your model is already instantiated. "
                    "The `model_init_kwargs` will be ignored."
                )

        if args.use_liger_kernel:
            raise NotImplementedError("`use_liger_kernel` is not supported yet.")

        text_config = model.config.get_text_config()

        # MoE load-balancing auxiliary loss. `output_router_logits` in the config means the model returns its router
        # logits; `router_aux_loss_coef` is the architecture's own coefficient, 0.0 when the config declares none.
        coef = args.router_aux_loss_coef
        self.router_aux_loss_coef = getattr(text_config, "router_aux_loss_coef", 0.0) if coef is None else coef
        if coef and not hasattr(text_config, "output_router_logits"):
            raise ValueError(
                f"`router_aux_loss_coef` is set to {coef} but {type(model).__name__} is not a Mixture-of-Experts model "
                f"that returns its router logits, so there is no auxiliary loss to weight."
            )
        self.aux_loss_enabled = hasattr(text_config, "output_router_logits") and self.router_aux_loss_coef != 0.0

        self._is_vlm = text_config is not model.config
        if self._is_vlm:
            # Train the text model only. It is located through the text config, since module names differ across
            # architectures (`model.language_model` for Qwen-VL and Gemma 3, `model.text_model` for SmolVLM).
            text_model = next(
                module
                for module in model.modules()
                if isinstance(module, PreTrainedModel) and module is not model and module.config is text_config
            )
            model.requires_grad_(False)
            text_model.requires_grad_(True)
            model.get_output_embeddings().requires_grad_(True)

        # The ranks a model is not split across hold replicas of it, which train on their own prompts and have to
        # agree on the update. Their gradients are summed over a group holding one rank per replica: the ranks that
        # carry the same tensor parallel shard. Parameters stay whole, which is what lets generation read them in
        # place, so this is ZeRO-0 over the replicas; the optimizer states are what a ZeRO-2 step would split next.
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size % args.tp_size != 0:
            raise ValueError(
                f"The world ({world_size} processes) must divide into replicas of `tp_size` ({args.tp_size}) "
                "processes each."
            )
        self.dp_size = world_size // args.tp_size
        if self.dp_size > 1 and args.parallelism_config is None:
            # The split of the world into replicas follows from `tp_size`, so there is nothing for the caller to
            # decide. Accelerate needs to be told it, or it wraps the model for data parallelism itself, which it
            # cannot do to a tensor-parallel one.
            args.parallelism_config = ParallelismConfig(dp_replicate_size=self.dp_size, tp_size=args.tp_size)
        self._replica_group = None
        self._replica_cpu_group = None
        self._parameter_owner: dict[str, int] = {}
        rank = int(os.environ.get("RANK", "0"))
        if self.dp_size > 1:
            # Every rank builds every group, in the same order, and keeps the one it belongs to. The gradients and
            # the updated parameters travel over NCCL inside the pause; the sample counts the replicas exchange while
            # the engine is decoding travel over gloo, so that no NCCL work of the trainer's is ever in flight next
            # to the engine's.
            ranks = [[r for r in range(world_size) if r % args.tp_size == shard] for shard in range(args.tp_size)]
            groups = [torch.distributed.new_group(group_ranks) for group_ranks in ranks]
            cpu_groups = [torch.distributed.new_group(group_ranks, backend="gloo") for group_ranks in ranks]
            self._replica_group = groups[rank % args.tp_size]
            self._replica_cpu_group = cpu_groups[rank % args.tp_size]
        # The ranks of a tensor parallel group generate together and must train on the same batch, so one of them runs
        # the rollouts: the first rank alone submits the requests, reads the completions, runs the tools and the reward
        # functions, computes the advantages and picks the batch, which it sends to the others over gloo.
        self._tp_driver = rank - rank % args.tp_size
        self._is_tp_driver = rank == self._tp_driver
        self._tp_cpu_group = None
        if args.tp_size > 1:
            tp_ranks = [list(range(start, start + args.tp_size)) for start in range(0, world_size, args.tp_size)]
            tp_cpu_groups = [torch.distributed.new_group(group_ranks, backend="gloo") for group_ranks in tp_ranks]
            self._tp_cpu_group = tp_cpu_groups[rank // args.tp_size]
        # The engine decodes through the training model itself; the trainer pauses it for the duration of its own
        # step, see `_pause_generation`.
        self._pause = None
        self._loan = None  # (sublet loan of the KV cache, mode) during a step, when lending the KV cache
        self._step_rollout_wait_s = 0.0

        # Under tensor parallelism the vocabulary projection is replicated rather than split. The chunked lm_head
        # below multiplies by the weight directly, which a sharded weight cannot serve, and the projection is the one
        # place where splitting buys least: it costs one copy of that matrix per process and removes a gather from
        # every forward. Everything else stays split.
        if args.tp_size > 1:
            output_embeddings = model.get_output_embeddings()
            input_embeddings = model.get_input_embeddings()
            tied = input_embeddings.weight is output_embeddings.weight
            # A plain full tensor: the chunked lm_head multiplies by the weight directly, and the gradient norm
            # spans sharded and plain parameters on its own, so there is nothing to gain from a DTensor here.
            weight = output_embeddings.weight
            # Some models keep the projection out of their tensor parallel plan, so it is already a full tensor
            local_weight = weight.full_tensor().contiguous() if isinstance(weight, DTensor) else weight.data
            replicated = nn.Parameter(local_weight)
            output_embeddings.weight = replicated
            # The transform installed for the split weight would now mix a plain weight with a DTensor input.
            output_embeddings.__dict__.pop("forward", None)
            if tied:
                input_embeddings.weight = replicated
                # The transform installed for the split weight would now mix a plain weight with a DTensor input.
                input_embeddings.__dict__.pop("forward", None)
            # An untied input embedding loads outside the plan, as a plain tensor, and the sweep below replicates it

        # Compute per-token logprobs without ever materializing the [batch, seq, vocab] logits: the lm_head projects
        # in tiles, only at the scored positions. Long completions make this the difference between training and an
        # OOM. The forward is only fused when asked for with `fused_lm_head=True`, so the generation engine, which
        # decodes through this same object, keeps the standard one.
        add_fused_lm_head(model, temperature=self.temperature, outputs=("log_probs", "entropy"))

        # Processing class
        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(
                get_config_model_id(model.config), revision=model_revision, trust_remote_code=args.trust_remote_code
            )
        if processing_class.pad_token is None:
            processing_class.pad_token = processing_class.eos_token
        # The model must agree with the tokenizer on the pad token from construction, so mirror it onto the model
        # configs.
        model.config.get_text_config().pad_token_id = processing_class.pad_token_id
        model.generation_config.pad_token_id = processing_class.pad_token_id

        # Every eos id the model declares (e.g. an end-of-turn token) ends a completion, not only the tokenizer's
        eos_token_ids = model.generation_config.eos_token_id
        if eos_token_ids is None:
            eos_token_ids = []
        elif isinstance(eos_token_ids, int):
            eos_token_ids = [eos_token_ids]
        if processing_class.eos_token_id not in eos_token_ids:
            eos_token_ids = [processing_class.eos_token_id, *eos_token_ids]
        self.eos_token_ids = eos_token_ids

        # Reward functions
        if reward_funcs is None:
            reward_funcs = []
        elif not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]

        if train_dataset is None:
            # A dataset is optional when an environment owns the data and returns the prompt from `reset()`; then
            # `max_steps` sets the length. Build a placeholder dataset of empty prompts to feed the rollout loop,
            # which goes over it once per epoch (so its length only needs to be non-degenerate).
            if environment_factory is None:
                raise ValueError("`train_dataset` is required unless an `environment_factory` is provided.")
            if isinstance(environment_factory, dict):
                raise ValueError(
                    "A `dict` `environment_factory` (multiple environments) requires a `train_dataset` with an "
                    "`environment` column to route each example to its environment. Provide a dataset, or pass a "
                    "single environment factory."
                )
            if args.max_steps <= 0:
                raise ValueError(
                    "When training without a `train_dataset` (the environment owns the data and returns the prompt "
                    "from `reset()`), `max_steps` must be set to a positive value to define the training length. Set "
                    "it via `ZeroSyncGRPOConfig(max_steps=...)`."
                )
            num_placeholder_rows = args.per_device_train_batch_size * args.gradient_accumulation_steps
            train_dataset = Dataset.from_dict({"prompt": [[{"role": "user", "content": ""}]] * num_placeholder_rows})

        # The rollout loop gets them, with the tools and the environments, when the engine starts
        self.reward_funcs = reward_funcs
        self.tools = tools
        self.environment_factory = environment_factory

        super().__init__(
            model=model,
            args=args,
            data_collator=identity,  # No data collation is needed in GRPO
            train_dataset=train_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        precision = self.accelerator.mixed_precision
        self.compute_dtype = {
            "bf16": "bfloat16",
            "fp16": "float16",
            "no": str(self.model.dtype).removeprefix("torch."),
        }.get(precision, precision)
        self._peak_flops_per_device = get_peak_flops_per_device(self.accelerator, self.compute_dtype)

        # The base Trainer's data-skip replay loop doesn't apply to ZeroSyncGRPO's live rollout stream;
        # force it off regardless of what the user passed.
        if not self.args.ignore_data_skip:
            logger.warning(
                "`ignore_data_skip` is forced to `True` for ZeroSyncGRPO because the base Trainer's skip-and-replay "
                "loop does not apply to a live rollout stream."
            )
        self.args.ignore_data_skip = True

        # The metric sink. Values are floats, or `(numerator, denominator)` pairs for rates
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._current_train_step_time = 0.0
        self._last_step_end_time = None
        # Accumulated across one optimizer step's micro-batches, flushed by `_log_step_metrics`.
        self._step_forward_s = 0.0
        self._step_optimizer_s = 0.0
        self._step_microbatches = 0
        self._step_forward_tokens = 0.0
        self._step_trained_tokens = 0.0
        self._step_seq_len_weighted = 0.0
        self._step_samples = 0.0
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

        # The continuous batching manager and the rollout loop that feeds it are created on the first generation (the
        # model must be on its final device first), the loop on the ranks that run the rollouts
        self._manager = None
        self.rollout_loop = None

        # Add callbacks
        self.add_callback(_OptimizerTimeCallback(self))
        self.add_callback(StepIntervalCallback(self._log_step_metrics, 1))

    def get_train_dataloader(self):
        return self._build_train_dataloader()

    def _build_train_dataloader(self):
        # The prompts keep the sampler and the per-replica sharding; every prompt produces `num_generations`
        # rollouts, so this feeds `per_device_train_batch_size / num_generations` prompts for each batch of samples.
        prompt_loader = self._get_dataloader(
            dataset=self.train_dataset,
            description="Training",
            batch_size=self._train_batch_size // self.args.num_generations,
            sampler_fn=self._get_train_sampler,
            is_training=True,
        )
        # What the loop iterates is the rollouts, not the prompts. This loader is deliberately not prepared by
        # accelerate: every rank generates its own rollouts, so there is nothing to shard, and the dispatcher
        # accelerate wraps an iterable dataset in would have rank zero alone walk the stream and broadcast the
        # batches, leaving its peers out of the engine's collectives.
        return DataLoader(
            _RolloutStream(self, prompt_loader),
            batch_size=self._train_batch_size,
            collate_fn=self.data_collator,
            num_workers=0,  # the engine lives in this process
        )

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs (usually, "input_ids"
        # and "attention_mask"). In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't
        # work. Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    def _init_manager(self):
        # The manager is attached to the unwrapped training model: decoding reads the same parameter tensors the
        # optimizer updates in place.
        model = self.accelerator.unwrap_model(self.model)
        generation_kwargs = {
            "max_new_tokens": self.args.max_completion_length,
            "do_sample": True,
            "pad_token_id": self.processing_class.pad_token_id,
            "bos_token_id": self.processing_class.bos_token_id,
            "eos_token_id": self.eos_token_ids,
            "temperature": self.temperature,
            "top_p": self.args.top_p,
            "top_k": self.args.top_k,
        }
        generation_config = GenerationConfig(**generation_kwargs)
        cb_kwargs = dict(self.args.continuous_batching_config or {})
        cb_kwargs["return_logprobs"] = True  # the engine's logprobs are the behavior-policy logprobs
        # Generation and training share the device, so the KV cache must leave room for gradients, optimizer states
        # and activations; the engine's own default claims most of the free memory.
        cb_kwargs.setdefault("max_memory_percent", 0.2)
        if self.tp_size > 1:
            # The engine turns NCCL's graph-mixing support off, which fits a pure-decode server but slows the
            # training collectives here (measured 3.62 against 3.71 s/step at batch 128). Generation is paused while
            # the training collectives run, so captured and eager collectives are never in flight together and the
            # NCCL default, which is also the safe setting, can stay.
            cb_kwargs.setdefault("disable_nccl_graph_mixing", False)
        # The engine decodes in its own thread, from the very tensors the optimizer updates, and the trainer pauses
        # it for its step (`_pause_generation`): the engine waits for its in-flight step on the device before it
        # reports itself paused, so the forward and backward have the device to themselves. Under tensor parallelism
        # this is also what keeps the two NCCL communicators from racing: NCCL requires every rank to issue the
        # operations on its communicators in the same host-side order.
        continuous_batching_config = ContinuousBatchingConfig(**cb_kwargs)
        if continuous_batching_config.max_blocks_per_request is None:
            # The engine decodes a batch on its fast path only if every sequence in it fits the block table, which
            # otherwise holds 8192 tokens: past that, a single long completion moves every decode step to the slow
            # path. Size it for the longest sequence the model takes, rounded up to an even number of blocks as the
            # engine does.
            max_length = model.config.get_text_config().max_position_embeddings
            num_blocks = math.ceil(max_length / continuous_batching_config.page_size) + 1
            continuous_batching_config.max_blocks_per_request = num_blocks + num_blocks % 2
        self._manager = model.init_continuous_batching(
            generation_config=generation_config, continuous_batching_config=continuous_batching_config
        )
        if self.args.kv_cache_during_step == "lend":
            import sublet

            # The engine creates its KV cache as it starts, unless it has one already. It is created here, as the
            # engine's `warmup` does, and moved onto a tensor whose memory sublet can lend before the engine uses it:
            # the cache allocators rebuild their views of the new tensor.
            self._manager.batch_processor = self._manager._create_batch_processor()
            cache = self._manager.batch_processor.cache
            nbytes = cache.cache_tensor.numel()
            cache.cache_tensor.untyped_storage().resize_(0)  # both would not fit on the device
            torch.cuda.empty_cache()
            cache.cache_tensor = sublet.empty(nbytes, dtype=torch.uint8, device=cache.device).zero_()
            non_trash_bytes = nbytes - 2 * cache.bytes_per_sector  # the cache starts with two trash sectors
            for allocator in cache.cache_allocators.values():
                allocator.register_cache_tensor(
                    cache.bytes_per_sector, non_trash_bytes, cache.cache_tensor, cache.pool
                )
        self._manager.start()
        if self._is_tp_driver:
            self.rollout_loop = self.loop_cls(
                self.args,
                manager=self._manager,
                reward_funcs=self.reward_funcs,
                processing_class=self.processing_class,
                eos_token_ids=self.eos_token_ids,
                metrics=self._metrics["train"],
                model_version_fn=lambda: self.state.global_step,
                tools=self.tools,
                environment_factory=self.environment_factory,
                **self._rollout_loop_kwargs,
            )

    def _pause_generation(self, mode: str) -> None:
        """Park the engine for the training step that follows, until the next `_prepare_inputs` resumes it.

        Generation and training take turns: the forward, backward and optimizer step get the device to themselves, and
        under tensor parallelism the trainer's collectives never run next to the engine's. The engine grants the pause
        at the same step on every rank of a tensor parallel group, and resumes only once every rank of the group has
        released it: the group's first rank asks, and the others join once it has sent them the batch.
        """
        if self._pause is not None:
            return
        # The pause is granted at the engine's next step boundary, so this is the length of the step it had in flight
        pause_start = time.perf_counter()
        self._pause = self._manager.pause()
        self._pause.__enter__()
        if self.args.kv_cache_during_step == "release":
            self._manager.release_memory()
        self._metrics[mode]["generation/pause_s"].append(time.perf_counter() - pause_start)
        if self.args.kv_cache_during_step == "lend":
            # The step allocates from the cache's memory until the loan ends in `_resume_generation`, which gives back
            # all of it but the free memory: the rollouts in flight, and the zeros padding tokens read
            import sublet

            lend_start = time.perf_counter()
            cache = self._manager.batch_processor.cache
            cache.evict_cached_blocks()  # blocks kept only to share prefixes: the step may write over them
            free = [(cache.bytes_per_sector, 2 * cache.bytes_per_sector)]  # the trash sector padding tokens write to
            free += [(s * cache.bytes_per_sector, (s + 1) * cache.bytes_per_sector) for s in cache.pool.free_sectors]
            for allocator in cache.cache_allocators.values():
                size = allocator.bytes_per_block
                for leaves in cache.pool._free_blocks[allocator.index].trees.values():
                    free += [(leaf * size, (leaf + 1) * size) for leaf in leaves]
            keep, start = [], 0
            for free_start, free_end in sorted(free):
                if free_start > start:
                    keep.append(cache.cache_tensor[start:free_start])
                start = free_end
            keep.append(cache.cache_tensor[start:])
            self._loan = (sublet.lend(cache.cache_tensor, keep=keep), mode)
            self._metrics[mode]["generation/lend_s"].append(time.perf_counter() - lend_start)

    def _resume_generation(self) -> None:
        try:
            if self._loan is not None:
                (loan, mode), self._loan = self._loan, None
                give_back_start = time.perf_counter()
                loan.end()  # the rollouts' blocks the step overwrote come back from the host
                self._metrics[mode]["generation/give_back_s"].append(time.perf_counter() - give_back_start)
                self._metrics[mode]["generation/lent_copied_gib"].append(loan.copied_bytes / 2**30)
                self._metrics[mode]["generation/lent_beyond_gib"].append(loan.extra_bytes / 2**30)
        finally:  # the loan gives the cache back intact even when it raises
            if self._pause is not None:
                if self.args.kv_cache_during_step == "release":
                    self._manager.restore_memory()
                self._pause.__exit__(None, None, None)
                self._pause = None

    def _drain_traces(self):
        while True:
            samples, step, ctx = self._trace_queue.get()
            # Replay the main-thread context captured at enqueue: trackio's run lives in a ContextVar this daemon
            # thread wouldn't otherwise inherit, so trackio.log() would raise "call init() first".
            ctx.run(log_rollout_traces, samples, step=step, report_to=self.args.report_to)

    def _take_from_pool(self, num_samples: int) -> list[RolloutSample]:
        """Hand this replica its share of the samples every replica has ready.

        A sample is data: nothing ties it to the replica that generated it. Pooling them lets the batches be built for
        balance instead of for locality, so the replicas reach their forward together and carry the same load. Every
        replica's first rank runs the same assignment over the same pool, so no one has to be told the result.
        """
        buffer = self.rollout_loop.rollout_buffer
        pool: list[Any] = [None] * self.dp_size
        torch.distributed.all_gather_object(pool, list(buffer), group=self._replica_group)
        replica = self.accelerator.process_index // self.tp_size

        # Which samples to train on is decided by age, not by length. Preferring the long ones leaves the short
        # ones sitting in the ready lists, and since a rollout is either in flight or waiting there, a growing
        # backlog is fewer sequences decoding at once, which is exactly what makes the engine efficient. Each
        # replica gives up its oldest, and whoever has a surplus covers a replica that is short.
        taking = [min(num_samples, len(samples)) for samples in pool]
        while sum(taking) < num_samples * self.dp_size:
            spare = [r for r in range(self.dp_size) if taking[r] < len(pool[r])]
            taking[min(spare, key=lambda r: taking[r])] += 1
        chosen = [
            (owner, index, pool[owner][index]) for owner in range(self.dp_size) for index in range(taking[owner])
        ]

        # Longest first into the emptiest replica, by the sum of squared lengths: attention is quadratic, so that
        # is what predicts both the step time and the activation peak, and it is the quantity worth equalizing.
        loads = [0] * self.dp_size
        assigned: list[list[Any]] = [[] for _ in range(self.dp_size)]
        for owner, index, sample in sorted(chosen, key=lambda item: -len(item[2].input_ids)):
            open_replicas = [r for r in range(self.dp_size) if len(assigned[r]) < num_samples]
            target = min(open_replicas, key=lambda r: loads[r])
            assigned[target].append((owner, index, sample))
            loads[target] += len(sample.input_ids) ** 2
        taken = {index for group in assigned for owner, index, _ in group if owner == replica}
        kept = [sample for index, sample in enumerate(buffer) if index not in taken]
        buffer.clear()
        buffer.extend(kept)
        self._metrics["train"]["batch/row_imbalance"].append(max(loads) / (sum(loads) / len(loads)))
        self._metrics["train"]["batch/from_other_replicas"].append(
            sum(1 for owner, _, _ in assigned[replica] if owner != replica) / max(len(assigned[replica]), 1)
        )
        # Arrival order within the replica's share: the step is split into micro-batches in this order, and the length
        # order used above would put all the longest samples into the first one.
        return [sample for _, _, sample in sorted(assigned[replica], key=lambda item: (item[0], item[1]))]

    def _next_rollout_batch(self, prompts) -> list[RolloutSample]:
        """Keep the engine fed and return one optimizer step's worth of finished rollouts, or nothing once the prompts
        run out.

        Called once per optimizer step by `_RolloutStream`, on every rank, which is what keeps the collectives below
        aligned.
        """
        mode = "train" if self.model.training else "eval"
        num_samples = self._train_batch_size * self.args.gradient_accumulation_steps

        if self._manager is None:
            self._init_manager()
        self._resume_generation()

        if not self._is_tp_driver:
            # The group's first rank runs the rollouts and sends the batch once it has paused the engine. This rank
            # then joins the pause: the engine leaves it only once every rank of the group has released it.
            wait_start = time.perf_counter()
            batch = [None]
            torch.distributed.broadcast_object_list(batch, src=self._tp_driver, group=self._tp_cpu_group)
            self._step_rollout_wait_s += time.perf_counter() - wait_start
            samples = batch[0]
            if samples:
                self._pause_generation(mode)
                # The engine delivers every completion on every rank of the group, and only the first rank reads them
                while self._manager.get_result(timeout=0) is not None:
                    pass
            return samples

        # The loop draws the epoch's prompts as its slots free up
        self.rollout_loop.prompts = prompts
        buffer = self.rollout_loop.rollout_buffer

        # Time spent waiting for the engine. Near zero means the samples were ready before the trainer asked; a large
        # value means the engine is the bottleneck, so raise `max_concurrent_rollouts` (more requests generating at
        # once) or give it a bigger KV pool.
        wait_start = time.perf_counter()
        if self._replica_group is None:
            self.rollout_loop.run_until(lambda: len(buffer) >= num_samples or self.rollout_loop.drained())
            enough = len(buffer) >= num_samples
        else:
            # The replicas draw one pool between them. A replica whose completions come back short would otherwise
            # fill its batch early, train, and then sit at the gradient sum waiting for a replica still decoding,
            # and it would carry a lighter forward while that one carries the heavy tail alone. Waiting for the
            # count to cover every replica, then handing the samples out, is what keeps the steps aligned. The sum of
            # the ready counts travels while each replica keeps generating and scoring, so one busy with a slow reward
            # never holds up the other, and both see the same sums in the same order, so they stop on the same one.
            while True:
                counts = torch.tensor([len(buffer), int(self.rollout_loop.drained())])
                summed = torch.distributed.all_reduce(counts, group=self._replica_cpu_group, async_op=True)
                self.rollout_loop.run_until(summed.is_completed)
                summed.wait()
                enough = int(counts[0]) >= num_samples * self.dp_size
                if enough or int(counts[1]) == self.dp_size:
                    break
        # Logged once per optimizer step, in `training_step`
        self._step_rollout_wait_s += time.perf_counter() - wait_start
        if not enough:  # every prompt of the epoch is generated and scored, short of a step's samples
            if self._tp_cpu_group is not None:
                torch.distributed.broadcast_object_list([[]], src=self._tp_driver, group=self._tp_cpu_group)
            return []
        # Allocator retries are the allocator freeing its cache and asking the driver again: a sign that the KV pool
        # and the training step are fighting over memory, which slows both.
        self._metrics[mode]["memory/alloc_retries"].append(torch.cuda.memory_stats().get("num_alloc_retries", 0))
        self._pause_generation(mode)
        if self._replica_group is not None:
            pool_start = time.perf_counter()
            samples = self._take_from_pool(num_samples)
            self._metrics[mode]["batch/pool_exchange_s"].append(time.perf_counter() - pool_start)
        else:
            # Packing already removes the padding, and picking like-length samples would concentrate the long
            # ones into a single batch, whose activations then blow past memory (a batch of 256 samples all near
            # the length cap is over twice the tokens of a mixed one). Arrival order keeps the mix.
            samples = [buffer.popleft() for _ in range(num_samples)]
        now = time.time()
        for sample in samples:
            if sample.enqueued_at is not None:
                self._metrics[mode]["sample/time_in_queue_s"].append(now - sample.enqueued_at)
            # Freshness
            version = self.state.global_step
            staleness = version - sample.model_version
            self._metrics[mode]["sample/staleness_mean"].append(float(staleness))
            self._metrics[mode]["sample/staleness_max"].append(float(staleness))

            # Traces are logged by the run's process, see `log_rollout_traces`
            if not self.accelerator.is_main_process:
                continue
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
        if self._tp_cpu_group is not None:
            torch.distributed.broadcast_object_list([samples], src=self._tp_driver, group=self._tp_cpu_group)
        return samples

    def _prepare_inputs(self, samples: list[RolloutSample]) -> dict[str, torch.Tensor | Any]:
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        # Metrics of the samples this micro-batch trains on. They are per replica: the replica's first rank forms and
        # scores the groups and sends the samples, and `Trainer.log` reports process zero.
        # A training row's tokens split in two: those the loss is taken over (`completion_mask == 1`) and those merely
        # forwarded — the prompt, tool results, masked tokens, and the earlier turns a fork re-sends as context. FLOPs
        # are spent on both, so both are worth a number, and their ratio is what says how much of the forward earns no
        # gradient.
        forwarded = [len(sample.input_ids) for sample in samples]
        trained = [int(sample.completion_mask.sum()) for sample in samples]
        self._metrics[mode]["sample/forwarded_tokens_mean"].append(sum(forwarded) / len(forwarded))
        self._metrics[mode]["sample/forwarded_tokens_max"].append(float(max(forwarded)))
        self._metrics[mode]["sample/trained_tokens_mean"].append(sum(trained) / len(trained))
        self._metrics[mode]["batch/masked_token_frac"].append((sum(forwarded) - sum(trained), sum(forwarded)))

        # Per-sample rewards, nan-aware: a reward func may return None for an unscorable sample, and a sample for which
        # every func returned None carries NaN rather than a misleading 0.
        # Union of keys, not the first sample's: `tools/*` is stamped per group, so a micro-batch mixes samples that
        # have those keys with samples that do not. Each key averages over the samples that carry it.
        keys = dict.fromkeys(key for sample in samples for key in sample.metrics)
        for key in keys:
            values = [sample.metrics[key] for sample in samples if key in sample.metrics]
            valid = [v for v in values if not math.isnan(v)]
            self._metrics[mode][key].append(sum(valid) / len(valid) if valid else float("nan"))

        # Build the training tensors. Each row already carries its full token sequence, the mask over the trained
        # (model-generated) tokens and the engine's behavior-policy logprobs aligned to them, 0.0 on context.
        # One flat row, no padding: samples go back to back and the boundaries travel as cu_seq_lens_q, which the
        # varlen attention kernels and the gated delta rule of linear attention layers consume directly, and as
        # seq_idx, which their causal convolution needs; sdpa builds its block-diagonal mask from the position ids,
        # which restart at every sample. Flash attention reads packing from the position
        # ids only when the batch has a single row, so several rows would attend across their samples. The global
        # shift in the loss stays sound at the seams: the first token of a sample is always context (mask 0), so no
        # term ever crosses two samples. Advantages and behavior logprobs are per token, since the row holds several
        # samples.
        lengths = torch.tensor([len(s.input_ids) for s in samples])
        cu_seq_lens = torch.cat([torch.zeros(1, dtype=torch.int32), lengths.cumsum(0).to(torch.int32)])
        return {
            "input_ids": torch.from_numpy(np.concatenate([s.input_ids for s in samples])).unsqueeze(0).to(device),
            "position_ids": torch.cat([torch.arange(len(s.input_ids)) for s in samples]).unsqueeze(0).to(device),
            "completion_mask": torch.from_numpy(np.concatenate([s.completion_mask for s in samples]))
            .unsqueeze(0)
            .to(device),
            "old_per_token_logps": torch.from_numpy(np.concatenate([s.old_log_probs for s in samples]))
            .unsqueeze(0)
            .to(device),
            "advantages": torch.cat([torch.full((len(s.input_ids),), s.advantage) for s in samples])
            .to(torch.float32)
            .unsqueeze(0)
            .to(device),
            "cu_seq_lens_q": cu_seq_lens.to(device),
            "cu_seq_lens_k": cu_seq_lens.to(device),
            "max_length_q": int(lengths.max()),
            "max_length_k": int(lengths.max()),
            "seq_idx": torch.repeat_interleave(torch.arange(len(samples), dtype=torch.int32), lengths)
            .unsqueeze(0)
            .to(device),
        }

    def create_optimizer(self, model=None):
        """Step fp32 copies of the parameters kept in host memory, and give each replica the state of only its share.

        Generation reads the model's own parameters, so they stay in the checkpoint dtype, and bf16 cannot absorb small
        updates: a weight of 1.0 moved by 1e-6 rounds back to 1.0. Stepping them directly, one AdamW step at
        `learning_rate=1e-6` lands 5% of its size on Qwen3.5-2B. The optimizer therefore updates fp32 copies and writes
        them back to the parameters after each step. The copies and the optimizer state live in pinned host memory and
        come onto the GPU one parameter at a time during the step, so the GPU holds only the parameters and their
        gradients, 4 bytes per parameter in bf16, next to generation's KV cache and training's activations.

        Adam carries two moments per parameter, so its state is what usually decides whether a model fits. The replicas
        split the parameters between them, each updates its own share, and passes the result back to the others. The
        parameters themselves stay whole everywhere, which is what generation reads.

        Measured on a 14B at tp=4 with two replicas: 9.2 GiB less per GPU at the peak, and the step itself takes half
        as long, since each replica now updates half the parameters.
        """
        if self.optimizer is not None:
            return self.optimizer

        opt_model = self.model if model is None else model
        if self._replica_group is None:
            super().create_optimizer(model)
        else:
            trainable = [(name, param) for name, param in opt_model.named_parameters() if param.requires_grad]
            # Largest first into the emptiest replica, so the shares come out even whatever the parameter sizes are.
            self._parameter_owner = {}
            held = [0] * self.dp_size
            for name, param in sorted(trainable, key=lambda item: -item[1].numel()):
                owner = min(range(self.dp_size), key=lambda replica: held[replica])
                self._parameter_owner[name] = owner
                held[owner] += param.numel()

            replica = self.accelerator.process_index // self.tp_size
            decay = set(self.get_decay_parameter_names(opt_model))
            mine = [(name, param) for name, param in trainable if self._parameter_owner[name] == replica]
            groups = [
                {"params": [p for n, p in mine if n in decay], "weight_decay": self.args.weight_decay},
                {"params": [p for n, p in mine if n not in decay], "weight_decay": 0.0},
            ]
            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, opt_model)
            self.optimizer = optimizer_cls(groups, **optimizer_kwargs)

        def local_tensors(master):
            # The copy and its optimizer state as the plain tensors that hold this rank's shard (under no_grad,
            # to_local() returns the DTensor's own). Moving them means swapping their storage, so the optimizer's
            # references stay valid. Scalars such as Adam's step count stay where the optimizer put them.
            tensors = [master, *(value for value in self.optimizer.state[master].values() if value.dim() > 0)]
            return [tensor.to_local() if isinstance(tensor, DTensor) else tensor for tensor in tensors]

        masters = []  # (parameter, its fp32 copy)
        with torch.no_grad():
            for group in self.optimizer.param_groups:
                for i, param in enumerate(group["params"]):
                    group["params"][i] = param.detach().to(torch.float32, copy=True)
                    for local in local_tensors(group["params"][i]):
                        local.data = local.data.to("cpu", non_blocking=True)
                    masters.append((param, group["params"][i]))

        step = self.optimizer.step
        # Copies come in and go out on two side streams, so the transfers in both directions overlap the updates
        copy_in, copy_out = torch.Stream(device=self.accelerator.device), torch.Stream(device=self.accelerator.device)

        @torch.no_grad()
        def step_and_share(*args, **kwargs):
            compute = torch.accelerator.current_stream()
            work = [(param, master) for param, master in masters if param.grad is not None]

            def fetch(param, master):
                with copy_in:
                    for local in local_tensors(master):
                        local.data = local.data.to(param.device, non_blocking=True)
                        local.data.record_stream(compute)
                return copy_in.record_event()

            out = None
            ready = fetch(*work[0]) if work else None
            for k, (param, master) in enumerate(work):
                compute.wait_event(ready)
                if k + 1 < len(work):
                    ready = fetch(*work[k + 1])
                # The step updates only the copies that hold a gradient, here this one
                master.grad = param.grad.float()
                param.grad = None
                out = step(*args, **kwargs)
                master.grad = None
                param.copy_(master)
                copy_out.wait_stream(compute)
                with copy_out:
                    for local in local_tensors(master):
                        local.data.record_stream(copy_out)
                        local.data = local.data.to("cpu", non_blocking=True)
            # The copies to host memory must land before anything reads them there, such as a checkpoint
            torch.accelerator.synchronize()
            if self._replica_group is not None:
                self._share_updated_parameters()
                # The optimizer only knows this replica's share, so its zero_grad would leave the rest holding
                # gradients that the next step would accumulate onto.
                opt_model.zero_grad(set_to_none=True)
            return out

        self.optimizer.step = step_and_share
        return self.optimizer

    def _share_updated_parameters(self) -> None:
        """Send each parameter from the replica that updated it to the others."""
        model = self.accelerator.unwrap_model(self.model)
        shard = self.accelerator.process_index % self.tp_size
        with torch.no_grad():
            for name, param in model.named_parameters():
                owner = self._parameter_owner.get(name)
                if owner is None:
                    continue
                # Every rank in the group holds the same slice of the model, so the local views line up
                local = param.data.to_local() if isinstance(param.data, DTensor) else param.data
                torch.distributed.broadcast(local, src=owner * self.tp_size + shard, group=self._replica_group)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        completion_mask = inputs["completion_mask"]

        # Per-token logprobs of the sampled tokens under the current policy, computed by the chunked lm_head, so
        # only completion positions are scored and the full-vocab logits never materialize.
        boundary_kwargs = {
            key: inputs[key]
            for key in ("cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k", "seq_idx")
            if key in inputs
        }
        # The fused head scores the positions whose next-token label is not -100, which is the completion tokens.
        labels = inputs["input_ids"].masked_fill(completion_mask == 0, -100)
        forward_start = time.time()
        # MoE models: request router logits so the forward returns the load-balancing loss
        router_kwargs = {"output_router_logits": True} if self.aux_loss_enabled else {}
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            position_ids=inputs.get("position_ids"),
            labels=labels,
            fused_lm_head=True,
            **router_kwargs,
            **boundary_kwargs,
        )
        # The loss terms: the completion tokens of the packed row, each predicted by the position before it
        selected = completion_mask[:, 1:].bool()
        log_probs = outputs.log_probs[selected]
        entropy = outputs.entropy[selected]
        old_log_probs = inputs["old_per_token_logps"][:, 1:][selected]
        advantages = inputs["advantages"][:, 1:][selected]
        segment_id = inputs["seq_idx"][:, 1:][selected].long()
        self._last_forward_time_s = time.time() - forward_start

        # The trained tokens, forwarded tokens and samples of the micro-batch, summed over the ranks. Each rank of a
        # tensor parallel group holds its replica's micro-batch, so the sum counts each replica `tp_size` times.
        local_counts = [
            float(log_probs.numel()),
            float(inputs["input_ids"].numel()),
            float(len(inputs["cu_seq_lens_q"]) - 1),
        ]
        global_counts = self.accelerator.reduce(torch.tensor(local_counts, device=log_probs.device), "sum")

        log_ratio = log_probs - old_log_probs
        coef_1 = torch.exp(log_ratio)
        low, high = self.args.importance_sampling_clip_min, self.args.importance_sampling_clip_max
        is_masked = (coef_1 < low) | (coef_1 > high)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        per_token_loss1 = coef_1 * advantages
        per_token_loss2 = coef_2 * advantages
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2) * ~is_masked

        # DDP/FSDP averages gradients across ranks (world_size).
        # To get correct per-token normalization we scale by 1/tokens_per_rank
        # = world_size / global_n_tokens, so after DDP averaging the effective
        loss = per_token_loss.sum()
        global_n_tokens = global_counts[0]
        world_size = self.accelerator.num_processes
        tokens_per_rank = (global_n_tokens / world_size).clamp(min=1.0)
        loss = loss / tokens_per_rank.to(torch.float32)
        # For DAPO, we would scale like this instead:
        # loss = loss / max(per_token_loss.size(0), 1)
        loss = loss / self.current_gradient_accumulation_steps

        # The policy loss above is scaled for gradient accumulation (HF auto-scaling is off here), so scale aux too
        if self.aux_loss_enabled:
            aux_loss = outputs.aux_loss
            loss = loss + self.router_aux_loss_coef * aux_loss / self.current_gradient_accumulation_steps

        with torch.no_grad():
            local_ratio_sum = coef_1.sum()
            local_count = torch.full_like(local_ratio_sum, float(log_probs.numel()))
            local_logp_diff_sum = log_ratio.abs().sum()
            local_entropy_sum = entropy.sum()

            # Compute the clipped probability ratios
            is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages < 0)
            is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages > 0)
            is_region_clipped = is_low_clipped | is_high_clipped
            local_low_clip_sum = is_low_clipped.float().sum()
            local_high_clip_sum = is_high_clipped.float().sum()
            local_region_clip_sum = is_region_clipped.float().sum()
            local_masked_sum = is_masked.float().sum()

            # Batch all-reduce: [ratio_sum, logp_diff_sum, entropy_sum, low_clip_sum, high_clip_sum, region_clip_sum,
            # masked_sum, count]
            stats = torch.stack(
                [
                    local_ratio_sum,
                    local_logp_diff_sum,
                    local_entropy_sum,
                    local_low_clip_sum,
                    local_high_clip_sum,
                    local_region_clip_sum,
                    local_masked_sum,
                    local_count,
                ]
            )
            stats = self.accelerator.reduce(stats, reduction="sum")
            (
                global_ratio_sum,
                global_logp_diff_sum,
                global_entropy_sum,
                global_low_clip_sum,
                global_high_clip_sum,
                global_region_clip_sum,
                global_masked_sum,
                global_count,
            ) = stats.unbind(0)
            ratio_min, ratio_max, logp_diff_max = (
                self.accelerator.gather(torch.stack([coef_1.min(), coef_1.max(), log_ratio.abs().max()]))
                .view(-1, 3)
                .unbind(1)
            )
            self._metrics["train"]["sampling/importance_sampling_ratio/min"].append(nanmin(ratio_min).item())
            self._metrics["train"]["sampling/importance_sampling_ratio/mean"].append(
                (global_ratio_sum / global_count).item()
            )
            self._metrics["train"]["sampling/importance_sampling_ratio/max"].append(nanmax(ratio_max).item())
            self._metrics["train"]["sampling/sampling_logp_difference/mean"].append(
                (global_logp_diff_sum / global_count).item()
            )
            self._metrics["train"]["sampling/sampling_logp_difference/max"].append(nanmax(logp_diff_max).item())
            self._metrics["train"]["entropy"].append((global_entropy_sum / global_count).item())
            self._metrics["train"]["clip_ratio/low_mean"].append((global_low_clip_sum / global_count).item())
            self._metrics["train"]["clip_ratio/high_mean"].append((global_high_clip_sum / global_count).item())
            self._metrics["train"]["clip_ratio/region_mean"].append((global_region_clip_sum / global_count).item())
            self._metrics["train"]["sampling/masked_mean"].append((global_masked_sum / global_count).item())

            num_seq = int(segment_id.max()) + 1  # completions in this rank's row

            def seg_sum(vals):  # per-completion segment sum over the flat selection
                return torch.zeros(num_seq, device=vals.device).index_add_(0, segment_id, vals)

            seq_tokens = seg_sum(torch.ones_like(coef_1, dtype=torch.float32))
            seq_low = seg_sum(is_low_clipped.float())
            seq_high = seg_sum(is_high_clipped.float())
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
        # `training_step`. The counts are batch-wide, each replica counted once.
        n_forward_tokens = float(global_counts[1]) / self.tp_size
        mean_seq_len = float(global_counts[1] / global_counts[2])
        self._step_forward_tokens += n_forward_tokens
        self._step_trained_tokens += float(global_n_tokens) / self.tp_size
        self._step_seq_len_weighted += mean_seq_len * n_forward_tokens
        self._step_samples += n_forward_tokens / mean_seq_len
        self._step_forward_s += self._last_forward_time_s
        return loss

    def training_step(self, model, inputs, num_items_in_batch):
        time_before = time.perf_counter()
        output = super().training_step(model, inputs, num_items_in_batch)
        # `sync_gradients` is true only on the last micro-step, so the replicas exchange one summed gradient per
        # optimizer step rather than one per accumulation micro-step.
        # Without tensor parallelism the model is a plain DDP one and accelerate already reduced the gradients;
        # DDP cannot wrap a tensor-parallel model, which is why the reduction is done here instead.
        if self.tp_size > 1 and self._replica_group is not None and self.accelerator.sync_gradients:
            for param in self.accelerator.unwrap_model(self.model).parameters():
                if param.grad is None:
                    continue
                # A sharded gradient is summed shard by shard: every rank in the group holds the same shard of the
                # model, so summing their local views is the same as summing the whole gradients.
                grad = param.grad.to_local() if isinstance(param.grad, DTensor) else param.grad
                torch.distributed.all_reduce(grad, group=self._replica_group)
                grad /= self.dp_size
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
        self.state.num_input_tokens_seen += int(self._step_forward_tokens)
        metrics["num_tokens"] = [self.state.num_input_tokens_seen]

        metrics["perf/rollout_wait_s"].append(self._step_rollout_wait_s)
        self._step_rollout_wait_s = 0.0

        # Throughput and MFU are reported on TWO bases, because for async RL one number cannot answer both questions.
        ## `_fwd_bwd` divides by `perf/fwd_bwd_s`: the compute alone.
        ## `_wall_clock` divides by `perf/step_s`: the whole step, rollout waits included  and says what fraction of the allocation actually became training.
        if self._step_forward_tokens > 0:
            mean_seq_len = self._step_seq_len_weighted / self._step_forward_tokens
            flops_per_token = compute_flops_per_token(self.model.config.get_text_config(), int(mean_seq_len))
            world_size = self.accelerator.num_processes
            metrics["perf/forwarded_tok_s_fwd_bwd"].append((self._step_forward_tokens, fwd_bwd_s))
            if self._peak_flops_per_device is not None:
                metrics["perf/mfu_fwd_bwd"].append(
                    compute_mfu(
                        flops_per_token, self._step_forward_tokens / fwd_bwd_s, world_size, self._peak_flops_per_device
                    )
                )
            if step_s is not None:
                metrics["perf/forwarded_tok_s_wall_clock"].append((self._step_forward_tokens, step_s))
                metrics["perf/trained_tok_s_wall_clock"].append((self._step_trained_tokens, step_s))
                if self._peak_flops_per_device is not None:
                    metrics["perf/mfu_wall_clock"].append(
                        compute_mfu(
                            flops_per_token,
                            self._step_forward_tokens / step_s,
                            world_size,
                            self._peak_flops_per_device,
                        )
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

    def _inner_training_loop(self, *args, **kwargs):
        try:
            return super()._inner_training_loop(*args, **kwargs)
        finally:
            if self._manager is not None:
                # The generation thread is not a daemon, and Python joins those before it runs any atexit hook, so
                # the process would never exit. A flush would wait for every rollout still in flight, and there are
                # always some, since the trainer keeps `max_concurrent_rollouts` rollouts generating; none of them will
                # be trained on now. The engine cannot stop while it is paused, so generation resumes first.
                try:
                    self._resume_generation()
                finally:
                    if self.rollout_loop is not None:
                        self.rollout_loop.stop()
                        self.rollout_loop = None
                    self._manager.stop(block=True, timeout=30, hard_stop=True)
                    self._manager = None
