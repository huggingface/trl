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
from collections.abc import Iterator
from dataclasses import dataclass, field
from multiprocessing.queues import Queue as MPQueue
from typing import Any, Protocol

import torch
import torch.nn.functional as F
from accelerate.logging import get_logger
from datasets import Dataset, IterableDataset
from torch.distributed._tensor import DTensor
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase, TrainerCallback
from transformers.data.data_collator import DataCollatorMixin

from ...models.utils import _ForwardRedirection
from ...trainer.base_trainer import _BaseTrainer
from ...trainer.utils import (
    compute_flops_per_token,
    compute_mfu,
    get_config_model_id,
    is_trackio_available,
    pad,
)
from .async_distillation_config import AsyncDistillationConfig
from .async_rollout_worker import AsyncRolloutWorker, RolloutSample
from .vllm_client import VLLMClient
from .weight_transfer import WeightTransferClient


logger = get_logger(__name__)

if is_trackio_available():
    import trackio

# One logged value is either a float or a `(numerator, denominator)` pair; see `_reduce_metric`.
MetricValue = float | tuple[float, float]


def _reduce_metric(key: str, values: list[MetricValue]) -> float:
    """Reduce one logging window into the number that gets logged.

    Ported verbatim from [`~trl.experimental.async_grpo.async_grpo_trainer._reduce_metric`]. The value's shape and the
    key's suffix declare the reduction, so nothing has to be registered or configured:

    - `(numerator, denominator)` pairs are a rate, reduced as Σnum / Σden. A rate is never stored *as* a rate, which is
      what makes a mean-of-ratios impossible — the defect that made `training_tok/s` unusable.
    - a `total` word in the name is a counter, summed over the window (producers push deltas).
    - a `max` or `min` word is an extremum. Matching whole words rather than a suffix keeps the upstream spellings
      (`completions/max_length`) working alongside this trainer's own (`row_tokens_max`).
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


def _add_tail_bucket(log_probs, valid_mask):
    """Append a (K+1)-th tail element: log(1 - sum(exp(top_k_logps))).

    Ported verbatim from [`~trl.experimental.server_distillation.server_distillation_trainer._add_tail_bucket`].
    Creates a proper probability distribution over K+1 elements, preventing trivial zero loss when the candidate set is
    small.
    """
    log_sum = torch.logsumexp(log_probs, dim=-1, keepdim=True)
    log_sum = torch.clamp(log_sum, max=-1e-7)  # ensure sum < 1
    tail = torch.log(-torch.expm1(log_sum))  # log(1 - exp(log_sum))
    tail_mask = torch.ones_like(valid_mask[..., :1], dtype=torch.bool)
    return torch.cat([log_probs, tail], dim=-1), torch.cat([valid_mask, tail_mask], dim=-1)


def _jsd_divergence(student_log_probs, teacher_log_probs, beta, support_mask):
    """Compute per-element JSD (or forward/reverse KL) from log-probability tensors over a fixed support.

    Ported verbatim from [`~trl.experimental.server_distillation.server_distillation_trainer._jsd_divergence`]'s masked
    branch: the async teacher is always scored over a sparse, fixed support (`teacher_top_k` candidates), so the dense
    `F.kl_div` branch that function also has (used when the sync trainer has full local access to both distributions)
    does not apply here.
    """
    safe_student = torch.where(support_mask, student_log_probs, torch.zeros_like(student_log_probs))
    safe_teacher = torch.where(support_mask, teacher_log_probs, torch.zeros_like(teacher_log_probs))
    student_probs = torch.where(support_mask, student_log_probs.exp(), torch.zeros_like(student_log_probs))
    teacher_probs = torch.where(support_mask, teacher_log_probs.exp(), torch.zeros_like(teacher_log_probs))

    if beta == 0:
        return torch.nan_to_num(teacher_probs * (safe_teacher - safe_student), nan=0.0)
    elif beta == 1:
        return torch.nan_to_num(student_probs * (safe_student - safe_teacher), nan=0.0)
    else:
        beta_t = torch.tensor(beta, dtype=student_log_probs.dtype, device=student_log_probs.device)
        tiny = torch.finfo(student_probs.dtype).tiny
        mixture_probs = (1 - beta_t) * student_probs + beta_t * teacher_probs
        safe_mixture = torch.where(
            support_mask, torch.log(mixture_probs.clamp_min(tiny)), torch.zeros_like(student_log_probs)
        )
        kl_teacher = torch.nan_to_num(teacher_probs * (safe_teacher - safe_mixture), nan=0.0)
        kl_student = torch.nan_to_num(student_probs * (safe_student - safe_mixture), nan=0.0)
        return beta_t * kl_teacher + (1 - beta_t) * kl_student


def _narrow_top1_actual_support(
    teacher_topk_ids, teacher_topk_logprobs, actual_token_ids, valid_candidate_mask_wide, beta
):
    """Narrow the teacher's reported top-k support to the candidates a `beta != 0.0` divergence needs.

    Mirrors [`~trl.experimental.server_distillation.ServerDistillationTrainer`]'s `beta > 0` path exactly, including
    its `beta == 1.0` special case: pure reverse KL is a purely student-weighted expectation, so the teacher's own
    top-1 token contributes nothing to it and is dropped, leaving only the completion's actual/realized token as
    support (width 1). For any other `beta != 0.0` (generalized JSD mixture), both the teacher's own top-1 token and
    the actual token are kept (width 2, deduplicated if they're the same token) since the mixture's forward-KL
    component does need the teacher's own top-1. These are the only token identities the wire protocol guarantees a
    teacher logprob for without transmitting a wider (or full) vocabulary. `teacher_topk_ids` is already rank-sorted
    (see `_parse_teacher_logprobs_at_position`), so index 0 is always the top-1 token; the actual token's logprob is
    somewhere within `teacher_topk_ids` because vLLM's `prompt_logprobs` always includes it, regardless of rank.

    Returns `(narrow_token_ids, teacher_support_logps, valid_candidate_mask)`, each shaped `[..., 1]` (beta == 1.0) or
    `[..., 2]` (otherwise), ready to feed into `_add_tail_bucket`/`_jsd_divergence` the same way the `beta=0.0` path's
    wide support does.
    """
    neg_inf = torch.full((), float("-inf"), dtype=teacher_topk_logprobs.dtype, device=teacher_topk_logprobs.device)
    actual_match_mask = (teacher_topk_ids == actual_token_ids.unsqueeze(-1)) & valid_candidate_mask_wide
    has_actual_signal = actual_match_mask.any(dim=-1)
    actual_logprobs = torch.where(actual_match_mask, teacher_topk_logprobs, neg_inf).max(dim=-1).values

    if beta == 1.0:
        narrow_token_ids = actual_token_ids.unsqueeze(-1)
        teacher_support_logps = actual_logprobs.unsqueeze(-1)
        valid_candidate_mask = has_actual_signal.unsqueeze(-1)
        return narrow_token_ids, teacher_support_logps, valid_candidate_mask

    top1_token_ids = teacher_topk_ids[..., 0]
    top1_logprobs = teacher_topk_logprobs[..., 0]
    has_top1_signal = valid_candidate_mask_wide[..., 0]

    narrow_token_ids = torch.stack([top1_token_ids, actual_token_ids], dim=-1)
    teacher_support_logps = torch.stack([top1_logprobs, actual_logprobs], dim=-1)
    valid_candidate_mask = torch.stack([has_top1_signal, has_actual_signal], dim=-1)
    # Deduplicate: if the teacher's top-1 token is the same as the actual token, count it once, not twice.
    is_duplicate = narrow_token_ids[..., 1] == narrow_token_ids[..., 0]
    valid_candidate_mask = valid_candidate_mask & torch.stack([torch.ones_like(is_duplicate), ~is_duplicate], dim=-1)

    return narrow_token_ids, teacher_support_logps, valid_candidate_mask


# Number of valid completion positions projected through the `lm_head` per chunk (mirrors
# `DistillationTrainer`'s `_CHUNKED_LM_HEAD_CHUNK_SIZE`).
_CHUNKED_LM_HEAD_CHUNK_SIZE = 256


def _jsd_loss_chunk(
    hidden_chunk: torch.Tensor,
    lm_head_weight: torch.Tensor,
    lm_head_bias: torch.Tensor | None,
    logit_scale: float,
    final_logit_softcapping: float | None,
    target_ids_chunk: torch.Tensor,
    teacher_logps_chunk: torch.Tensor,
    candidate_mask_chunk: torch.Tensor,
    beta: float,
    temperature: float,
    add_tail_bucket: bool,
    teacher_id_idx_chunk: torch.Tensor | None,
    num_teachers: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project one chunk of student hidden states through `lm_head` and accumulate this chunk's loss/metric sums.

    Mirrors `DistillationTrainer`'s `_chunk`/`_chunked_divergence_loss`: the full `(chunk_size, vocab_size)` logits are
    the only per-chunk tensor that scales with `vocab_size`, and `torch.utils.checkpoint.checkpoint` (see the caller)
    discards it after this chunk's forward, recomputing it on demand during backward — so peak logits memory is
    `chunk_size * vocab_size`, not `total_valid_tokens * vocab_size`. Unlike that sync-trainer function, there is only
    one model here (the teacher's sparse candidates are already precomputed off the wire, no second `lm_head` to
    chunk), and the target ids are a sparse candidate set (`teacher_top_k` wide, or narrowed to top-1+actual for `beta
    != 0.0`), not the full vocabulary.

    Args:
        hidden_chunk: `(chunk_size, H)`, this chunk's student backbone output (before `lm_head`).
        lm_head_weight: `(V, H)`, the student's `lm_head` weight (already `full_tensor()`-ed if it was a DTensor).
        lm_head_bias: `(V,)` or `None`.
        logit_scale: multiplier applied to the logits before the softmax (Cohere-style `logit_scale`; `1.0` if the
            model has none).
        final_logit_softcapping: if set, applies `softcap * tanh(logits / softcap)` (Gemma-style), after the scale.
        target_ids_chunk: `(chunk_size, K)`, the teacher's candidate token ids to gather the student's log-probs at.
        teacher_logps_chunk: `(chunk_size, K)`, the teacher's log-probs at `target_ids_chunk`.
        candidate_mask_chunk: `(chunk_size, K)` bool, `True` where that candidate slot is a real (not padding)
            candidate.
        beta, temperature, add_tail_bucket: see `AsyncDistillationConfig`.
        teacher_id_idx_chunk: `(chunk_size,)` long, or `None` with a single teacher (see MOPD in
            `AsyncDistillationConfig.teacher_server_urls`).
        num_teachers: `len(self._teacher_ids)`, fixed for the training run; sizes the (possibly all-zero)
            per-teacher stats tensor so every chunk/rank returns the same shape regardless of `teacher_id_idx_chunk`.

    Returns:
        `(chunk_loss, chunk_jsd_sum, chunk_entropy_sum, chunk_teacher_entropy_sum, chunk_per_teacher_stats)`: the first
        is connected to the autograd graph (student-parameter gradients only, the teacher data carries none); the rest
        are `torch.no_grad()` sums for this chunk only — callers accumulate them across chunks and reduce across ranks,
        exactly as the non-chunked path already did.
    """
    logits = hidden_chunk.float() @ lm_head_weight.float().t()
    if lm_head_bias is not None:
        logits = logits + lm_head_bias.float()
    if logit_scale != 1.0:
        logits = logits * logit_scale
    if final_logit_softcapping is not None:
        logits = final_logit_softcapping * torch.tanh(logits / final_logit_softcapping)
    logits = logits / temperature

    student_log_probs_full = F.log_softmax(logits, dim=-1)
    neg_inf = torch.full((), float("-inf"), dtype=student_log_probs_full.dtype, device=student_log_probs_full.device)
    student_support_logps = student_log_probs_full.gather(-1, target_ids_chunk.clamp_min(0))
    student_support_logps = torch.where(candidate_mask_chunk, student_support_logps, neg_inf)
    teacher_support_logps = torch.where(candidate_mask_chunk, teacher_logps_chunk, neg_inf)

    if add_tail_bucket:
        student_log_probs, support_mask = _add_tail_bucket(student_support_logps, candidate_mask_chunk)
        teacher_log_probs, _ = _add_tail_bucket(teacher_support_logps, candidate_mask_chunk)
    else:
        student_log_probs = student_support_logps - torch.logsumexp(student_support_logps, dim=-1, keepdim=True)
        teacher_log_probs = teacher_support_logps - torch.logsumexp(teacher_support_logps, dim=-1, keepdim=True)
        support_mask = candidate_mask_chunk

    jsd = _jsd_divergence(student_log_probs, teacher_log_probs, beta=beta, support_mask=support_mask)
    chunk_loss = jsd.sum()

    with torch.no_grad():
        per_token_jsd = jsd.sum(dim=-1)
        chunk_jsd_sum = per_token_jsd.sum()

        student_probs_full = student_log_probs_full.exp()
        per_token_entropy = -(student_probs_full * student_log_probs_full).sum(dim=-1)
        chunk_entropy_sum = per_token_entropy.sum()

        safe_teacher_log_probs = torch.where(support_mask, teacher_log_probs, torch.zeros_like(teacher_log_probs))
        teacher_probs = torch.where(support_mask, teacher_log_probs.exp(), torch.zeros_like(teacher_log_probs))
        per_token_teacher_entropy = torch.nan_to_num(-(teacher_probs * safe_teacher_log_probs).sum(dim=-1), nan=0.0)
        chunk_teacher_entropy_sum = per_token_teacher_entropy.sum()

        chunk_per_teacher_stats = jsd.new_zeros(3 * num_teachers)
        if teacher_id_idx_chunk is not None:
            stats = []
            for idx in range(num_teachers):
                teacher_mask = teacher_id_idx_chunk == idx
                has_any = teacher_mask.any()
                stats.append(teacher_mask.sum().float())
                stats.append(per_token_jsd[teacher_mask].sum() if has_any else jsd.new_zeros(()))
                stats.append(per_token_teacher_entropy[teacher_mask].sum() if has_any else jsd.new_zeros(()))
            chunk_per_teacher_stats = torch.stack(stats)

    return chunk_loss, chunk_jsd_sum, chunk_entropy_sum, chunk_teacher_entropy_sum, chunk_per_teacher_stats


def _chunked_jsd_loss(
    hidden_valid: torch.Tensor,
    lm_head_weight: torch.Tensor,
    lm_head_bias: torch.Tensor | None,
    logit_scale: float,
    final_logit_softcapping: float | None,
    target_ids: torch.Tensor,
    teacher_target_logps: torch.Tensor,
    valid_candidate_mask: torch.Tensor,
    beta: float,
    temperature: float,
    add_tail_bucket: bool,
    teacher_id_idx: torch.Tensor | None,
    num_teachers: int,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run `_jsd_loss_chunk` over `hidden_valid` in `chunk_size`-sized chunks and sum the results.

    Every position here is already known to carry usable teacher signal (the caller packs the valid ones to the front),
    so there is no masked tail to skip, unlike `DistillationTrainer._chunked_divergence_loss`'s static-shape padding.
    Each chunk runs under `torch.utils.checkpoint.checkpoint`, so its `(chunk_size, vocab_size)` logits are freed after
    the chunk's forward and recomputed during backward.

    Args:
        hidden_valid: `(n_valid, H)`, student backbone output at the valid completion positions.
        target_ids, teacher_target_logps, valid_candidate_mask: `(n_valid, K)`, the teacher-side candidates, already
            restricted to the same valid positions.
        teacher_id_idx: `(n_valid,)` long, or `None` with a single teacher.
        chunk_size: number of positions projected through `lm_head` at a time.

        The remaining arguments are passed through unchanged; see `_jsd_loss_chunk`.

    Returns:
        `(loss, jsd_sum, entropy_sum, teacher_entropy_sum, per_teacher_stats)`, each summed over all chunks.
    """
    loss = hidden_valid.new_zeros(())
    jsd_sum = hidden_valid.new_zeros(())
    entropy_sum = hidden_valid.new_zeros(())
    teacher_entropy_sum = hidden_valid.new_zeros(())
    per_teacher_stats = hidden_valid.new_zeros(3 * num_teachers)
    for start in range(0, hidden_valid.size(0), chunk_size):
        end = start + chunk_size
        chunk_loss, chunk_jsd_sum, chunk_entropy_sum, chunk_teacher_entropy_sum, chunk_per_teacher_stats = (
            torch.utils.checkpoint.checkpoint(
                _jsd_loss_chunk,
                hidden_valid[start:end],
                lm_head_weight,
                lm_head_bias,
                logit_scale,
                final_logit_softcapping,
                target_ids[start:end],
                teacher_target_logps[start:end],
                valid_candidate_mask[start:end],
                beta,
                temperature,
                add_tail_bucket,
                teacher_id_idx[start:end] if teacher_id_idx is not None else None,
                num_teachers,
                use_reentrant=False,
            )
        )
        loss = loss + chunk_loss
        jsd_sum = jsd_sum + chunk_jsd_sum
        entropy_sum = entropy_sum + chunk_entropy_sum
        teacher_entropy_sum = teacher_entropy_sum + chunk_teacher_entropy_sum
        per_teacher_stats = per_teacher_stats + chunk_per_teacher_stats
    return loss, jsd_sum, entropy_sum, teacher_entropy_sum, per_teacher_stats


def _balance_by_squared_length(examples: list[dict[str, Any]], num_groups: int) -> list[list[dict[str, Any]]]:
    """Greedily partition `examples` into `num_groups` rows (one per DP rank), balancing each row's Σ Lᵢ².

    Ported verbatim from [`~trl.experimental.async_grpo.async_grpo_trainer._balance_by_squared_length`], packing and
    load-balancing are loss-agnostic, so this needs no adaptation for distillation.
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

    Ported verbatim from [`~trl.experimental.async_grpo.async_grpo_trainer.FixedCountBatcher`].
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

    Ported verbatim from [`~trl.experimental.async_grpo.async_grpo_trainer.TokenBudgetBatcher`].
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
                logger.warning(
                    f"Dropping a rollout sample of {n} tokens that exceeds token_budget={self.token_budget}. "
                    "Raise token_budget to avoid dropping samples."
                )
                self.metrics["batch/dropped_oversize_total"].append(1.0)
                continue
            fits = [i for i in range(self.num_processes) if token_counts[i] + n <= self.token_budget]
            if not fits:
                yield rows
                rows = [[] for _ in range(self.num_processes)]
                squared_loads = [0] * self.num_processes
                token_counts = [0] * self.num_processes
                fits = list(range(self.num_processes))
            i = min(fits, key=lambda j: squared_loads[j])
            rows[i].append(sample)
            squared_loads[i] += n * n
            token_counts[i] += n


class RolloutWorkerProtocol(Protocol):
    """Interface a rollout worker must implement to be passed as `rollout_worker` to [`AsyncDistillationTrainer`].

    Same contract as [`~trl.experimental.async_grpo.async_grpo_trainer.RolloutWorkerProtocol`].

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

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def update_model_version(self, model_version: int) -> None: ...
    def check_health(self, stale_after_s: float) -> None: ...


class WeightTransferProtocol(Protocol):
    """Interface a weight-sync backend must implement to be passed as `weight_transfer` to
    [`AsyncDistillationTrainer`].

    Same contract as [`~trl.experimental.async_grpo.async_grpo_trainer.WeightTransferProtocol`]. The default
    [`WeightTransferClient`] streams the student's weights into its vLLM server over NCCL; pass a no-op implementation
    (as tests do) to exercise the trainer without a real vLLM server.
    """

    def init_weight_transfer(self) -> None: ...
    def pause(self) -> None: ...
    def send_weights(self, iterator: Iterator[tuple[str, torch.Tensor]]) -> None: ...
    def resume(self) -> None: ...
    def destroy(self) -> None: ...


class StepIntervalCallback(TrainerCallback):
    """A callback that calls a function every N optimization steps."""

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

    def __init__(self, trainer: "AsyncDistillationTrainer"):
        self._trainer = trainer
        self._t0 = None

    def on_pre_optimizer_step(self, _args, _state, _control, **_kwargs):
        self._t0 = time.perf_counter()

    def on_optimizer_step(self, _args, _state, _control, **_kwargs):
        self._trainer._step_optimizer_s += time.perf_counter() - self._t0


class _InitialWeightSyncCallback(TrainerCallback):
    """Idempotent: NCCL group setup + cold weight sync to vLLM on train begin."""

    def __init__(self, trainer: "AsyncDistillationTrainer"):
        self._trainer = trainer
        self._fired = False

    def on_train_begin(self, _args, _state, _control, **_kwargs):
        if self._fired:
            return
        self._fired = True
        if self._trainer.accelerator.is_main_process and self._trainer.weight_transfer is not None:
            self._trainer.weight_transfer.init_weight_transfer()
        self._trainer._sync_weight()


class _StartRolloutWorkerCallback(TrainerCallback):
    """Idempotent: starts the rollout worker. Must be registered AFTER `_InitialWeightSyncCallback`."""

    def __init__(self, trainer: "AsyncDistillationTrainer"):
        self._trainer = trainer
        self._fired = False

    def on_train_begin(self, _args, _state, _control, **_kwargs):
        if self._fired:
            return
        self._fired = True
        if self._trainer.accelerator.is_main_process and self._trainer.rollout_worker is not None:
            self._trainer.rollout_worker.start()


def log_rollout_traces(samples: list[RolloutSample], step: int, report_to: list[str], max_traces: int = 16) -> None:
    """Log rollout samples to trackio as inspectable traces (prompt + completion + teacher/student metrics per sample).

    Call from rank 0 during training, where the HF trackio callback has already initialised the run; the traces then
    show up under the run's Traces tab so rollouts can be read directly instead of grepping logs. No-op unless trackio
    is the active logging backend (installed and listed in `report_to`). Best-effort: a trackio hiccup must never break
    training. Ported from [`~trl.experimental.async_grpo.async_grpo_trainer.log_rollout_traces`]; unlike GRPO's
    version, there is no `advantage`/`group_id` to log (distillation has no group-relative baseline, see
    `RolloutSample`).

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
                    **sample.metrics,
                    "model_version": int(sample.model_version),
                    "teacher_id": sample.teacher_id,  # which teacher scored it; the one thing MOPD traces need
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
        # Blocking-get time, accumulated here and flushed by `_log_step_metrics` at the optimizer-step boundary — the
        # only place that knows when a step's worth of waiting is done.
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
                    # Returning here would broadcast None through accelerate's dispatch loop. A single `t0` is kept
                    # across repeated empty polls (rather than restarting the outer loop) so `perf/rollout_wait_s`
                    # reflects the true time blocked on an empty queue, not just the last poll interval (see
                    # huggingface/trl#6489, the same bug in `AsyncGRPOTrainer.RolloutQueueDataset`).
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
                "teacher_topk_ids": sample.teacher_topk_ids,
                "teacher_topk_logprobs": sample.teacher_topk_logprobs,
                "teacher_id": sample.teacher_id,
                "metrics": sample.metrics,  # per-sample scalars a custom worker stamped; aggregated by the collator
            }


class _EmptyIterableDataset(torch.utils.data.IterableDataset):
    """Placeholder for non-rank-0 processes. Never actually iterated."""

    def __iter__(self):
        return iter([])


@dataclass
class DataCollatorForRollout(DataCollatorMixin):
    """
    Padding-free collator (the packer) for rollout samples. Packs a micro-batch (already partitioned into
    `num_processes` rows by the upstream planner, [`FixedCountBatcher`] or [`TokenBudgetBatcher`]) into one packed row
    per DP rank, with `position_ids` resetting per sequence. Rows are padded only to the longest row, so the batch
    stays rectangular for `DataLoaderDispatcher` to scatter row `i` -> rank `i`; this inter-rank padding is stripped
    per-rank in `compute_loss`.

    Structurally the same as [`~trl.experimental.async_grpo.async_grpo_trainer.DataCollatorForRollout`], with
    `advantages` replaced by the ragged `teacher_topk_ids` / `teacher_topk_logprobs` (padded per-position to
    `teacher_top_k + 1` candidates with id `-1` / logprob `-inf`, then packed along the sequence dimension like every
    other per-token field).

    Args:
        pad_token_id (`int`):
            Token id used to pad `input_ids`.
        teacher_top_k (`int`):
            Number of teacher candidates per position (before the trainer's own tail-bucket handling); every position's
            candidate list is padded/truncated to this width so it tensorizes.
        num_processes (`int`, *optional*, defaults to `1`):
            Number of DP ranks; the micro-batch is packed into this many rows.
        teacher_id_to_idx (`dict[str, int]`, *optional*):
            Maps each configured `teacher_server_urls` key to a stable integer index, packed per-position (like every
            other per-token field) as `teacher_id_idx` so `compute_loss` can break `jsd`/`entropy`/ `teacher_entropy`
            down per teacher (MOPD). `None` when there's only one teacher (no breakdown needed).
        metrics (`dict[str, list]`, *optional*):
            The trainer's metric sink, appended to with this micro-batch's sample and packing metrics.
        token_budget (`int`, *optional*, defaults to `0`):
            Per-row token cap of the planner, or `0` when batching by fixed sample count.
    """

    pad_token_id: int
    teacher_top_k: int
    num_processes: int = 1
    teacher_id_to_idx: dict[str, int] | None = None
    return_tensors: str = "pt"
    # The trainer's metric sink, shared by reference.
    metrics: dict[str, list] = field(default_factory=lambda: defaultdict(list))
    # Per-row token cap of the planner,
    token_budget: int = 0

    def torch_call(self, examples: list[Any]) -> dict[str, Any]:
        # The dataloader uses batch_size=1 over a planner that pre-partitions each micro-batch into `num_processes`
        # rows, so `examples` is a length-1 list holding that single micro-batch (one group per rank).
        (groups,) = examples
        width = self.teacher_top_k + 1

        input_ids, attention_mask, completion_mask, position_ids = [], [], [], []
        teacher_topk_ids, teacher_topk_logprobs = [], []
        teacher_id_idx = [] if self.teacher_id_to_idx else None
        for group in groups:
            seq_lengths = [len(example["input_ids"]) for example in group]
            ids = [token for example in group for token in example["input_ids"]]
            input_ids.append(torch.tensor(ids, dtype=torch.long))
            attention_mask.append(torch.ones(len(ids), dtype=torch.long))
            completion_mask.append(
                torch.tensor([m for example in group for m in example["completion_mask"]], dtype=torch.long)
            )
            position_ids.append(torch.cat([torch.arange(n) for n in seq_lengths]))

            row_teacher_ids, row_teacher_logprobs = [], []
            for example in group:
                for cand_ids, cand_lps in zip(
                    example["teacher_topk_ids"], example["teacher_topk_logprobs"], strict=False
                ):
                    pad_n = width - len(cand_ids)
                    row_teacher_ids.append(list(cand_ids) + [-1] * pad_n)
                    row_teacher_logprobs.append(list(cand_lps) + [float("-inf")] * pad_n)
            teacher_topk_ids.append(torch.tensor(row_teacher_ids, dtype=torch.long))
            teacher_topk_logprobs.append(torch.tensor(row_teacher_logprobs, dtype=torch.float32))

            if teacher_id_idx is not None:
                # One teacher_id per example, broadcast across its own tokens (constant within a sample).
                idx_per_token = [
                    self.teacher_id_to_idx[example["teacher_id"]]
                    for example, n in zip(group, seq_lengths, strict=True)
                    for _ in range(n)
                ]
                teacher_id_idx.append(torch.tensor(idx_per_token, dtype=torch.long))

        input_ids = pad(input_ids, padding_value=self.pad_token_id)
        attention_mask = pad(attention_mask, padding_value=0)
        completion_mask = pad(completion_mask, padding_value=0)
        position_ids = pad(position_ids, padding_value=0)
        teacher_topk_ids = pad(teacher_topk_ids, padding_value=-1)
        if teacher_id_idx is not None:
            teacher_id_idx = pad(teacher_id_idx, padding_value=-1)
        teacher_topk_logprobs = pad(teacher_topk_logprobs, padding_value=float("-inf"))

        all_examples = [example for group in groups for example in group]

        # Total valid completion tokens across the micro-batch. Repeated per rank so that DataLoaderDispatcher
        # (dispatch_batches=True) slices correctly on dim=0.
        n_trained_tokens = sum(sum(example["completion_mask"]) for example in all_examples)
        global_n_tokens = torch.full((self.num_processes,), float(n_trained_tokens), dtype=torch.float32)

        sample_tokens = [len(example["input_ids"]) for example in all_examples]
        n_forward_tokens = sum(sample_tokens)
        mean_seq_len = n_forward_tokens / len(all_examples)
        global_n_forward_tokens = torch.full((self.num_processes,), float(n_forward_tokens), dtype=torch.float32)
        mean_seq_len_t = torch.full((self.num_processes,), float(mean_seq_len), dtype=torch.float32)

        self._log_metrics(groups=groups, all_examples=all_examples, padded=attention_mask)

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "completion_mask": completion_mask,
            "position_ids": position_ids,
            "teacher_topk_ids": teacher_topk_ids,
            "teacher_topk_logprobs": teacher_topk_logprobs,
            "global_n_tokens": global_n_tokens,
            "global_n_forward_tokens": global_n_forward_tokens,
            "mean_seq_len": mean_seq_len_t,
        }
        if teacher_id_idx is not None:
            batch["teacher_id_idx"] = teacher_id_idx
        return batch

    def _log_metrics(
        self, groups: list[list[dict[str, Any]]], all_examples: list[dict[str, Any]], padded: torch.Tensor
    ) -> None:
        """Append this micro-batch's sample and packing metrics straight into the trainer's sink.

        Rank 0 holds the whole micro-batch, so anything that is a property of the batch rather than of the model can be
        aggregated here instead of being packed into NaN-padded tensors, broadcast to every rank and reduced back —
        which is what this trainer used to do to compute a number rank 0 already had.
        """
        # A training row's tokens split in two: those the loss is taken over (`completion_mask == 1`) and those merely
        # forwarded — the prompt, and any completion position the teacher scored no candidate for. FLOPs are spent on
        # both, so both are worth a number, and their ratio is what says how much of the forward earns no gradient.
        forwarded = [len(example["input_ids"]) for example in all_examples]
        trained = [sum(example["completion_mask"]) for example in all_examples]
        self.metrics["sample/forwarded_tokens_mean"].append(sum(forwarded) / len(forwarded))
        self.metrics["sample/forwarded_tokens_max"].append(float(max(forwarded)))
        self.metrics["sample/trained_tokens_mean"].append(sum(trained) / len(trained))
        self.metrics["batch/masked_token_frac"].append((sum(forwarded) - sum(trained), sum(forwarded)))

        # Per-sample scalars, nan-aware. Empty for the built-in worker — distillation has no reward function, so it
        # measures nothing per sample — but a custom worker can stamp its own, and each key averages over the samples
        # that carry it rather than over the whole micro-batch.
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


class AsyncDistillationTrainer(_BaseTrainer):
    """
    Async counterpart to [`~trl.experimental.distillation.DistillationTrainer`], architected exactly like
    [`~trl.experimental.async_grpo.AsyncGRPOTrainer`]: a background rollout worker generates the student's on-policy
    completions and scores them against a teacher server while training proceeds concurrently, decoupling rollout from
    the gradient-update loop. Where GRPO's clipped policy-gradient loss reads a group-relative advantage, this
    trainer's `compute_loss` reads a sparse per-position teacher distribution and minimizes a generalized JSD against
    it (see `AsyncDistillationConfig.beta`), always on-policy: the student generates every completion it trains on.

    Unlike the synchronous `DistillationTrainer`, this trainer only supports scoring against a teacher served over HTTP
    (an external vLLM server). A local (in-process, GPU) teacher forward pass cannot run inside the rollout worker's
    CUDA-disabled spawned child process the way GRPO's reward functions can, and would instead need to run back in the
    main training process; that path is not implemented yet.

    Example:

    ```python
    >>> from trl.experimental.async_distillation import AsyncDistillationTrainer
    >>> from datasets import load_dataset

    >>> dataset = load_dataset("trl-lib/DeepMath-103K", split="train")

    >>> trainer = AsyncDistillationTrainer(
    ...     model="Qwen/Qwen2.5-0.5B-Instruct",
    ...     train_dataset=dataset,
    ... )
    >>> trainer.train()
    ```

    Args:
        model (`str`):
            Student model to be trained. Must be a string, being the *model id* of a pretrained model hosted inside a
            model repo on huggingface.co, or a path to a *directory* containing model weights saved using
            [`~transformers.PreTrainedModel.save_pretrained`]. Loaded with
            [`~transformers.AutoModelForCausalLM.from_pretrained`]. The model name is also used to identify the student
            model on its vLLM server.
        args ([`AsyncDistillationConfig`], *optional*):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. Must include a `"prompt"` column
            ([conversational](dataset_formats#conversational) format). Any additional columns are ignored.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*):
            Processing class used to process the data. If `None`, it is loaded from the model's name with
            [`~transformers.AutoTokenizer.from_pretrained`]. If it has no padding token, `tokenizer.eos_token` is used.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*):
            List of callbacks to customize the training loop, added to the default callbacks (see
            [here](https://huggingface.co/docs/transformers/main_classes/callback)).
        optimizers (`tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Defaults to `AdamW` and a linear schedule
            controlled by `args`.
        rollout_worker (`RolloutWorkerProtocol`, *optional*):
            Custom rollout worker implementing [`RolloutWorkerProtocol`]. If `None`, a default [`AsyncRolloutWorker`]
            is created, spawning a CUDA-free child process that generates from the student's vLLM server and scores
            against the teacher's.
        weight_transfer (`WeightTransferProtocol`, *optional*):
            Custom weight-sync backend implementing [`WeightTransferProtocol`]. If `None`, a default
            [`WeightTransferClient`] is created that streams the student's weights into its vLLM server over NCCL. Pass
            a no-op implementation to disable trainer-side weight sync (e.g. in tests, or when a custom
            `rollout_worker` updates the policy itself).
    """

    _tag_names = ["trl", "async-distillation"]
    _name = "AsyncDistillation"
    _paper = {
        "title": "On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes",
        "id": "2306.13649",
        # docstyle-ignore
        "citation": textwrap.dedent("""\
            @inproceedings{agarwal2024onpolicy,
                title        = {On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes},
                author       = {Rishabh Agarwal and Nino Vieillard and Yongchao Zhou and Piotr Stanczyk and Sabela Ramos Garea and Matthieu Geist and Olivier Bachem},
                booktitle    = {The Twelfth International Conference on Learning Representations},
                year         = 2024,
                url          = {https://openreview.net/forum?id=3zKtaqxLhW},
            }"""),
    }

    def __init__(
        self,
        model: str,
        args: AsyncDistillationConfig | None = None,
        train_dataset: Dataset | IterableDataset | None = None,
        processing_class: PreTrainedTokenizerBase | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None] = (None, None),
        rollout_worker: RolloutWorkerProtocol | None = None,
        weight_transfer: WeightTransferProtocol | None = None,
    ):
        if args is None:
            args = AsyncDistillationConfig(f"{model.split('/')[-1]}-AsyncDistillation")

        # Model
        model_name = model
        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs.setdefault("trust_remote_code", args.trust_remote_code)
        # FlashAttention is required: training runs in padding-free mode, where sequences are concatenated into a
        # single row and attention is derived from `position_ids` resets. SDPA/eager can't handle this. Unlike
        # AsyncGRPOTrainer, the student's own lm_head is NOT patched (via `patch_chunked_lm_head`) to a chunked
        # realized-token-only head at load time: the divergence loss needs the student's log-probs at several
        # candidate token ids per position (the teacher's top-k + tail), not just the realized token's logprob, which
        # is all that patch supports. `compute_loss` chunks the lm_head projection itself instead (`_jsd_loss_chunk`),
        # a different mechanism that does support multiple candidate ids per position.
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=None,
            dtype=torch.float32,
            attn_implementation="kernels-community/flash-attn3",
            **model_init_kwargs,
        )

        if args.use_liger_kernel:
            raise NotImplementedError("`use_liger_kernel` is not supported yet.")

        # Processing class
        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(model_name, trust_remote_code=args.trust_remote_code)
        if processing_class.pad_token is None:
            processing_class.pad_token = processing_class.eos_token

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
        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether
        # the model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        # Infer max_steps from dataset size when not explicitly set. This must happen after super().__init__() so
        # that self.accelerator.num_processes is available for the correct calculation. The training dataloader is
        # driven by the async rollout queue (an IterableDataset with no __len__), so max_steps must be set explicitly
        # for transformers.Trainer's step-counting to work, unlike AsyncGRPOTrainer, there is no num_generations
        # multiplier here: each dataset row yields exactly one training sample, not a group of them.
        samples_per_step = (
            self.args.per_device_train_batch_size
            * self.args.gradient_accumulation_steps
            * self.accelerator.num_processes
        )
        # Known limitation, shared verbatim with AsyncGRPOTrainer: this assumes `samples_per_step` samples per step,
        # which only holds under `FixedCountBatcher`. Under `TokenBudgetBatcher` (the default once `token_budget` is
        # set, see below), the number of samples per micro-batch is dynamic — driven by sequence lengths, not
        # `per_device_train_batch_size` — so an inferred `max_steps`/`num_train_epochs` no longer corresponds exactly
        # to the requested number of dataset passes. Fixing this requires touching AsyncGRPOTrainer's identical
        # derivation too, out of scope for this PR.
        if self.args.max_steps <= 0 and train_dataset is not None and hasattr(train_dataset, "__len__"):
            samples_per_epoch = len(train_dataset)
            self.args.max_steps = math.ceil(self.args.num_train_epochs * samples_per_epoch / samples_per_step)

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

        # `compute_loss` bypasses `lm_head` by calling `unwrapped_model.base_model(...)` directly, which skips the
        # wrapper's forward; route it through `_forward_redirection` so DDP.forward() fires `prepare_for_backward()`
        # and FSDP keeps the student's sharded parameters materialized for the projection. Mirrors
        # `DistillationTrainer`'s identical need for its own chunked JSD path.
        self._forward_redirection = _ForwardRedirection()

        # The metric sink. Values are floats, or `(numerator, denominator)` pairs for rates
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        # MOPD (multi-teacher on-policy distillation): a stable teacher_id -> index mapping so compute_loss can
        # break jsd/entropy/teacher_entropy down per teacher. Skipped (no per-teacher breakdown) with one teacher.
        self._teacher_ids = list(self.args.teacher_server_urls)
        self._teacher_id_to_idx = {tid: i for i, tid in enumerate(self._teacher_ids)}
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
        self.model_version = 0
        # Create worker and queue on rank 0
        if self.accelerator.is_main_process:
            if self.train_dataset is None:
                raise ValueError("train_dataset is required for AsyncDistillationTrainer")

            # Weight sync and the token-budget query target the STUDENT's vLLM server. The teacher's server is never
            # written to (no weight transfer), so it has no analogous client here.
            self.vllm_client = VLLMClient(self.args.vllm_server_base_url, self.args.vllm_server_timeout)

            if weight_transfer is not None:
                # Injected backend (e.g. a no-op stub in tests, or a custom sync mechanism).
                self.weight_transfer = weight_transfer
            else:
                # Collect weight metadata once, names/dtypes/shapes are fixed for the lifetime of training.
                # DTensor.shape returns the global shape without triggering any all-gather.
                weight_names, weight_dtype_names, weight_shapes = [], [], []
                for name, param in model.named_parameters():
                    name = name.removeprefix("module.")  # DDP/FSDP1 wrapping
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
                    weight_sync_timeout=self.args.weight_sync_timeout,
                )

            if rollout_worker is not None:
                # Use the injected worker (e.g. a stub in tests). The queue is owned by the worker.
                self.rollout_worker = rollout_worker
            else:
                self.rollout_worker = AsyncRolloutWorker(
                    model_name=get_config_model_id(model.config),
                    dataset=train_dataset,
                    processing_class=processing_class,
                    max_inflight_tasks=self.args.max_inflight_tasks,
                    queue_maxsize=self.args.queue_maxsize,
                    vllm_server_url=self.args.vllm_server_base_url,
                    teacher_server_urls=self.args.teacher_server_urls,
                    teacher_top_k=self.args.teacher_top_k,
                    teacher_temperature=self.args.teacher_temperature,
                    max_tokens=self.args.max_completion_length,
                    temperature=self.args.temperature,
                    top_p=self.args.top_p,
                    top_k=self.args.top_k,
                    min_p=self.args.min_p,
                    repetition_penalty=self.args.repetition_penalty,
                    request_timeout=self.args.request_timeout,
                    chat_template_kwargs=self.args.chat_template_kwargs,
                    log_completions=self.args.log_completions,
                    log_completions_steps=self.args.log_completions_steps,
                    num_completions_to_print=self.args.num_completions_to_print,
                )
            self.rollout_queue = self.rollout_worker.rollout_buffer
        else:
            self.rollout_queue = None
            self.rollout_worker = None
            self.vllm_client = None
            self.weight_transfer = None

        # Add callbacks. Registration order matters: weight sync first, then worker start.
        self.add_callback(_OptimizerTimeCallback(self))
        self.add_callback(_InitialWeightSyncCallback(self))
        self.add_callback(_StartRolloutWorkerCallback(self))
        self.add_callback(StepIntervalCallback(self._sync_weight, self.args.weight_sync_steps))
        self.add_callback(StepIntervalCallback(self._log_step_metrics, 1))

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
            # Default the token budget to the student vLLM server's max_model_len, so no rollout sample can exceed
            # it. Wait for the server like weight sync does, so a still-loading vLLM doesn't fail training here.
            if self.args.token_budget is None:
                self.vllm_client.wait_for_server_ready()
                self.args.token_budget = self.vllm_client.get_max_model_len()
                logger.info(f"token_budget unset; defaulting to vLLM max_model_len={self.args.token_budget}")
            if self.args.token_budget > 0:
                dataset = TokenBudgetBatcher(dataset, num_processes, self.args.token_budget, self._metrics["train"])
            else:
                dataset = FixedCountBatcher(
                    dataset, num_processes, self.args.per_device_train_batch_size * num_processes
                )
        else:
            dataset = _EmptyIterableDataset()

        return self.accelerator.prepare(
            DataLoader(
                dataset,
                batch_size=1,
                collate_fn=DataCollatorForRollout(
                    self.processing_class.pad_token_id,
                    self.args.teacher_top_k,
                    num_processes,
                    self._teacher_id_to_idx if len(self._teacher_ids) > 1 else None,
                    metrics=self._metrics["train"],
                    # `or 0` because only rank 0 fills an unset budget from the vLLM server above; the other ranks
                    # construct the collator (and never use it) while `token_budget` is still `None`.
                    token_budget=max(self.args.token_budget or 0, 0),
                ),
                num_workers=0,
            )
        )

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed. In
        # AsyncDistillationTrainer, we need additional columns to compute the loss, hence the override.
        if self._signature_columns is None:
            self._signature_columns = [
                "input_ids",
                "attention_mask",
                "completion_mask",
                "position_ids",
                "teacher_topk_ids",
                "teacher_topk_logprobs",
                "teacher_id",
                "teacher_id_idx",
                "global_n_tokens",
                "global_n_forward_tokens",
                "mean_seq_len",
            ]

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Route the whole loss (backbone + `lm_head` projection + JSD) through the DDP/FSDP wrapper via
        # `_forward_redirection`, so DDP.forward() fires `prepare_for_backward()` and FSDP keeps the student's
        # sharded parameters (including the `lm_head`) materialized for the projection. Mirrors
        # `DistillationTrainer.compute_loss`.
        unwrapped_model = self.accelerator.unwrap_model(model)
        loss = self._forward_redirection(model, unwrapped_model, self._compute_loss, unwrapped_model, inputs)
        return (loss, None) if return_outputs else loss

    def _compute_loss(self, unwrapped_model, inputs):
        # Padding-free: the collator already packed this rank's samples into a single row (real tokens concatenated,
        # `position_ids` resetting per sequence), then padded the row to the longest rank's length so
        # DataLoaderDispatcher could scatter rectangular rows. Strip that trailing inter-rank padding here.
        mask_bool = inputs["attention_mask"].bool()
        input_ids = inputs["input_ids"][mask_bool].unsqueeze(0)
        completion_mask = inputs["completion_mask"][mask_bool].unsqueeze(0)
        position_ids = inputs["position_ids"][mask_bool].unsqueeze(0)
        teacher_topk_ids = inputs["teacher_topk_ids"][mask_bool].unsqueeze(0)
        teacher_topk_logprobs = inputs["teacher_topk_logprobs"][mask_bool].unsqueeze(0)
        # MOPD (multi-teacher on-policy distillation): present only when multiple teachers are configured (see
        # AsyncDistillationConfig.teacher_server_urls), used below to break jsd/entropy/teacher_entropy down per
        # teacher; absent (None) with a single teacher, where every sample shares one teacher anyway.
        teacher_id_idx = inputs["teacher_id_idx"][mask_bool].unsqueeze(0) if "teacher_id_idx" in inputs else None

        # Next-token shift: position j predicts input_ids[:, j+1], so it is compared against the teacher candidates
        # recorded AT position j+1; teacher_topk_ids[i] holds the candidates for predicting input_ids[i] (it comes
        # from the teacher's `prompt_logprobs`, whose entry i conditions on the tokens before i).
        completion_mask = completion_mask[:, 1:]
        teacher_topk_ids = teacher_topk_ids[:, 1:, :]
        teacher_topk_logprobs = teacher_topk_logprobs[:, 1:, :]
        if teacher_id_idx is not None:
            teacher_id_idx = teacher_id_idx[:, 1:]

        # Which candidate ids the student needs a log-prob at, and the teacher's log-prob at those same ids — pure
        # teacher-side/input_ids bookkeeping, no student logits needed yet, so this happens before the forward pass.
        valid_candidate_mask_wide = teacher_topk_ids != -1
        if self.args.beta == 0.0:
            # Forward KL: the teacher's own top-k support (its weighting is exactly what this sum needs).
            valid_candidate_mask = valid_candidate_mask_wide
            target_ids = teacher_topk_ids
            teacher_target_logps = teacher_topk_logprobs
        else:
            # beta != 0.0: narrow the support to just the candidates a mixture/reverse-KL divergence needs,
            # mirroring `ServerDistillationTrainer`'s beta>0 path (including its beta==1.0 special case). A wider
            # support would only probabilistically (not guaranteed) cover what the student itself considers likely.
            actual_token_ids = input_ids[:, 1:]
            target_ids, teacher_target_logps, valid_candidate_mask = _narrow_top1_actual_support(
                teacher_topk_ids, teacher_topk_logprobs, actual_token_ids, valid_candidate_mask_wide, self.args.beta
            )

        # A completion position can carry no usable teacher signal at all (the teacher returned no `prompt_logprobs`
        # entry for it, or every candidate was NaN) while still being marked as a completion token. Without this, such a
        # position would degenerate through `_add_tail_bucket` into a fabricated "100% tail mass" distribution on
        # both sides, i.e. a synthetic near-zero divergence trained on, instead of being excluded like padding.
        has_teacher_signal = valid_candidate_mask.any(dim=-1)
        # DDP/FSDP averages gradients across ranks (world_size). To get correct per-token normalization we scale by
        # 1/tokens_per_rank = world_size / global_n_tokens, so after DDP averaging the effective scale is 1/global_n_tokens.
        # Mirrors AsyncGRPOTrainer.compute_loss's scaling: this is infra-level (padding-free packing, DDP averaging,
        # gradient accumulation), not policy-gradient-specific, so it needs no adaptation here. `global_n_tokens` is
        # computed by the collator from `completion_mask` alone, so it does not shrink when `has_teacher_signal`
        # excludes a position above; a run with such gaps is very slightly under-scaled rather than exactly
        # renormalized, which is an acceptable trade-off for what should be a rare, teacher-side data gap.
        token_mask_1d = (completion_mask.bool() & has_teacher_signal)[0]  # (seq_len - 1,), batch dim always 1

        # Memory-efficient path: get the student's backbone hidden states (skip `lm_head`, so this alone never
        # materializes a vocab-sized tensor), then project only the *valid* positions' hidden states through
        # `lm_head` in `_CHUNKED_LM_HEAD_CHUNK_SIZE`-sized chunks (see `_chunked_jsd_loss`), never materializing more
        # than one chunk's `(chunk_size, vocab_size)` logits at a time — mirrors
        # `DistillationTrainer._get_last_hidden_state` + `_chunked_divergence_loss`, adapted to one (student-only)
        # model and a sparse (already off-the-wire) teacher target instead of two locally-projected dense ones.
        # Calling `base_model` directly (instead of `model`) bypasses the mixed-precision autocast that Accelerate
        # attaches to the outer model's `forward` — `_forward_redirection` above preserves DDP/FSDP's own hooks, but
        # not this one, since it also reaches `base_model` around the patched `forward`. Re-applied explicitly here
        # (a no-op if bf16/fp16 aren't enabled) — otherwise FlashAttention (which requires fp16/bf16 inputs) sees
        # fp32 hidden states and errors.
        forward_start = time.time()
        with self.accelerator.autocast():
            hidden_states = unwrapped_model.base_model(
                input_ids=input_ids, position_ids=position_ids, use_cache=False
            ).last_hidden_state
        self._last_forward_time_s = time.time() - forward_start
        hidden_states = hidden_states[:, :-1, :][0]  # (seq_len - 1, H), drop the unused last-position prediction

        lm_head = unwrapped_model.get_output_embeddings()
        lm_head_weight = lm_head.weight
        lm_head_bias = lm_head.bias
        # Under FSDP2, lm_head.weight is a DTensor (Shard(0) or Replicate). Gathering it once here (instead of once
        # per chunk) means only one all-gather happens, in `full_tensor()`'s own backward, not one per chunk.
        if isinstance(lm_head_weight, DTensor):
            lm_head_weight = lm_head_weight.full_tensor()
            if lm_head_bias is not None:
                lm_head_bias = lm_head_bias.full_tensor()
        # NOTE(@aminediro): supporting Cohere2 models (mirrors `patch_chunked_lm_head`'s own handling).
        # Read through `get_text_config()` like [`~trl.trainer.DistillationTrainer`] does: on VLMs the logit
        # post-processing lives on `text_config`. Muse Glimmer applies the same pre-softcap multiplier under the name
        # `output_multiplier`, and a real `logit_scale` of 0.0 is kept as-is and applied faithfully.
        config = unwrapped_model.config.get_text_config()
        logit_scale = getattr(config, "logit_scale", None)
        if logit_scale is None:
            logit_scale = getattr(config, "output_multiplier", None)
        logit_scale = 1.0 if logit_scale is None else logit_scale
        final_logit_softcapping = getattr(config, "final_logit_softcapping", None)

        target_ids = target_ids[0][token_mask_1d]
        teacher_target_logps = teacher_target_logps[0][token_mask_1d]
        valid_candidate_mask = valid_candidate_mask[0][token_mask_1d]
        teacher_id_idx_valid = teacher_id_idx[0][token_mask_1d] if teacher_id_idx is not None else None
        hidden_valid = hidden_states[token_mask_1d]
        n_valid = hidden_valid.size(0)
        num_teachers = len(self._teacher_ids)

        if n_valid == 0:
            # Whole micro-batch has no usable teacher signal. Keep the loss connected to the autograd graph through
            # every trainable parameter (backbone via `hidden_states`, `lm_head` via `lm_head_weight`) so
            # `.backward()` succeeds and DDP/FSDP gradient sync doesn't hang on a "missing" parameter's grad.
            loss = (hidden_states.sum() + lm_head_weight.sum()) * 0.0
            if lm_head_bias is not None:
                loss = loss + lm_head_bias.sum() * 0.0
            local_jsd_sum = hidden_valid.new_zeros(())
            local_entropy_sum = hidden_valid.new_zeros(())
            local_teacher_entropy_sum = hidden_valid.new_zeros(())
            local_per_teacher_stats = hidden_valid.new_zeros(3 * num_teachers)
        else:
            loss, local_jsd_sum, local_entropy_sum, local_teacher_entropy_sum, local_per_teacher_stats = (
                _chunked_jsd_loss(
                    hidden_valid,
                    lm_head_weight,
                    lm_head_bias,
                    logit_scale,
                    final_logit_softcapping,
                    target_ids,
                    teacher_target_logps,
                    valid_candidate_mask,
                    self.args.beta,
                    self.args.teacher_temperature,
                    self.args.add_tail_bucket,
                    teacher_id_idx_valid,
                    num_teachers,
                    _CHUNKED_LM_HEAD_CHUNK_SIZE,
                )
            )

        global_n_tokens = inputs["global_n_tokens"][0]
        world_size = self.accelerator.num_processes
        tokens_per_rank = (global_n_tokens / world_size).clamp(min=1.0)
        loss = loss / tokens_per_rank.to(torch.float32)
        loss = loss / self.current_gradient_accumulation_steps

        with torch.no_grad():
            local_count = token_mask_1d.sum().float()
            stats = torch.stack([local_jsd_sum, local_entropy_sum, local_teacher_entropy_sum, local_count])
            stats = self.accelerator.reduce(stats, reduction="sum")
            global_jsd_sum, global_entropy_sum, global_teacher_entropy_sum, global_count = stats.unbind(0)
            # A micro-batch can have completion tokens but zero usable teacher signal across every rank (see
            # `has_teacher_signal` above), making `global_count` 0; NaN then marks the window as having no measurement
            # rather than reporting a divergence of zero, and `log` drops it before reducing.
            if global_count > 0:
                self._metrics["train"]["jsd"].append((global_jsd_sum / global_count).item())
                self._metrics["train"]["entropy"].append((global_entropy_sum / global_count).item())
                self._metrics["train"]["teacher_entropy"].append((global_teacher_entropy_sum / global_count).item())
            else:
                self._metrics["train"]["jsd"].append(float("nan"))
                self._metrics["train"]["entropy"].append(float("nan"))
                self._metrics["train"]["teacher_entropy"].append(float("nan"))

            # MOPD (multi-teacher on-policy distillation): break jsd/teacher_entropy down per teacher, since the
            # blended versions above conflate teachers that can behave very differently (e.g. a "math" and a "code"
            # teacher). Student entropy isn't broken down here: it's a property of the student's own policy, not of
            # which teacher scored the sample, so the blended `entropy` metric already covers it. Named
            # `teacher_jsd/{teacher_id}` (not `jsd/{teacher_id}`) so it doesn't collide with the flat `jsd` metric
            # above in dashboards that treat `/` as a grouping separator (e.g. wandb/trackio).
            # `teacher_id_idx` is either present on every rank or absent on every rank (it depends only on
            # `self.args.teacher_server_urls`, identical config on all ranks), so every rank takes this branch
            # together or skips it together — required for `accelerator.reduce` below to stay in sync across ranks.
            if teacher_id_idx is not None:
                per_teacher_stats = self.accelerator.reduce(local_per_teacher_stats, reduction="sum")
                for i, teacher_id in enumerate(self._teacher_ids):
                    t_count, t_jsd_sum, t_teacher_entropy_sum = per_teacher_stats[3 * i : 3 * i + 3]
                    if t_count > 0:
                        self._metrics["train"][f"teacher_jsd/{teacher_id}"].append((t_jsd_sum / t_count).item())
                        self._metrics["train"][f"teacher_entropy/{teacher_id}"].append(
                            (t_teacher_entropy_sum / t_count).item()
                        )
                    else:
                        self._metrics["train"][f"teacher_jsd/{teacher_id}"].append(float("nan"))
                        self._metrics["train"][f"teacher_entropy/{teacher_id}"].append(float("nan"))

        # Per-step accounting, accumulated across the micro-batches of one optimizer step and flushed in
        # `_log_step_metrics`. The counts are batch-wide (the collator broadcasts one value per rank), so they are read
        # off rank-local inputs without a collective. Sample and packing metrics are NOT gathered here — rank 0 already
        # logged them in the collator.
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

        # Throughput and MFU are reported on TWO bases, because for async distillation one number cannot answer both
        # questions.
        ## `_fwd_bwd` divides by `perf/fwd_bwd_s`: the compute alone.
        ## `_wall_clock` divides by `perf/step_s`: the whole step, rollout waits included  and says what fraction of the allocation actually became training.
        if self._step_forward_tokens > 0:
            mean_seq_len = self._step_seq_len_weighted / self._step_forward_tokens
            flops_per_token = compute_flops_per_token(self.model.config.get_text_config(), int(mean_seq_len))
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

        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs.update(metrics)
        super().log(logs, start_time)
        self._metrics[mode].clear()

    def _streaming_iter(self):
        # Iterate parameters one at a time. For FSDP2 (DTensor), full_tensor() all-gathers just this parameter across
        # FSDP ranks, then frees it once the generator advances, avoiding materializing the full model in memory.
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
