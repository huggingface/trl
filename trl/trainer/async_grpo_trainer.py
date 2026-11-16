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
import copy
import functools
import inspect
import itertools
import json
import math
import multiprocessing as mp
import os
import pickle
import queue
import random
import re
import shutil
import sys
import textwrap
import threading
import time
import traceback
import uuid
import warnings
from collections import defaultdict
from collections.abc import Awaitable, Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import cache
from itertools import chain, repeat
from multiprocessing.queues import Queue as MPQueue
from multiprocessing.sharedctypes import Synchronized as MPValue
from multiprocessing.synchronize import Event as MPEvent
from typing import Any, Protocol, TypeAlias

import numpy as np
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import broadcast_object_list, is_peft_model, set_seed
from datasets import Dataset, IterableDataset
from packaging.version import Version
from torch.distributed._tensor import DTensor
from torch.distributed.tensor.experimental import implicit_replication
from torch.utils.data import DataLoader
from transformers import AttentionInterface, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase, TrainerCallback
from transformers.data.data_collator import DataCollatorMixin
from transformers.integrations.flash_attention import get_target_dtype
from transformers.masking_utils import AttentionMaskInterface
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, SaveStrategy
from transformers.utils import is_peft_available

from ..chat_template_utils import (
    _SUPPORTS_RESPONSE_TEMPLATE,
    add_response_schema,
    get_training_chat_template,
    is_chat_template_prefix_preserving,
    parse_response,
    supports_tool_calling,
)
from ..generation.async_vllm_client import AsyncVLLMClient
from ..generation.weight_transfer import WeightTransferClient
from ..import_utils import is_aiohttp_available, is_jmespath_available
from .async_grpo_config import AsyncGRPOConfig
from .base_trainer import _BaseTrainer
from .utils import (
    add_fused_lm_head,
    compute_flops_per_token,
    compute_mfu,
    create_model_from_path,
    get_callable_name,
    get_config_model_id,
    get_peak_flops_per_device,
    is_async_callable,
    is_trackio_available,
    nanmax,
    nanmin,
    pad,
    print_prompt_completions_sample,
)


if is_aiohttp_available():
    import aiohttp

    _RETRYABLE_HTTP_ERRORS = (aiohttp.ClientError, asyncio.TimeoutError, TimeoutError, ConnectionResetError)

if is_peft_available():
    from peft import LoraConfig, PeftConfig, PeftModel, PromptLearningConfig, get_peft_model

if is_trackio_available():
    import trackio


class TreeRow:
    """
    One DP rank's forward: its samples folded into a prefix trie, so a token shared by several samples is forwarded
    once.

    Multi-turn rollouts overlap heavily — a shared prompt, and after a fork a shared conversation — so the decoder's
    per-token work drops by the packing ratio (raw tokens ÷ unique tokens). Attention still shows each sample exactly
    what it saw: each segment of the trie attends to the K/V of its root-to-segment path (see [`tree_attention`]). With
    nothing shared, the trie is one chain per sample, which is plain padding-free packing.

    Any common prefix is shared: the rollouts of a prompt, and across prompts a common system prompt. The planner
    checks a sample against the row ([`~TreeRow.fits`]) before placing it, so the token budget bounds unique tokens,
    not raw ones, and it sends a sample to a row already holding its `group_id` ([`~TreeRow.groups`]), where its prompt
    is.

    A node is keyed on its token *and* the next one, so samples share a token only when they also agree on what follows
    it. The trie therefore forks one token before the samples diverge, and every node has a single next-token label: a
    branch costs one extra token per extra branch, and the loss can go through plain per-position labels. A sample's
    last token is keyed on `None`, which no other sample extends, so every sample ends on a leaf.
    """

    def __init__(self):
        self.samples: list[dict[str, Any]] = []
        self.sample_nodes: list[list[int]] = []  # sample -> trie node of each of its tokens
        self.tokens: list[int] = []  # node -> its token
        self.children: list[dict[tuple[int, int | None], int]] = []  # node -> {(token, next token): child node}
        self.roots: dict[tuple[int, int | None], int] = {}  # {(token, next token): root node}
        self.groups: set[Any] = set()
        self.trained = 0
        self.load = 0

    def fits(self, sample: dict[str, Any], budget: int) -> bool:
        """Whether the row stays within `budget` unique tokens and `budget` trained tokens with `sample` added."""
        input_ids = sample["input_ids"].tolist()
        children, shared = self.roots, 0
        for key in zip(input_ids, [*input_ids[1:], None], strict=True):
            if key not in children:
                break
            children = self.children[children[key]]
            shared += 1
        novel = len(input_ids) - shared
        return len(self.tokens) + novel <= budget and self.trained + int(sample["completion_mask"].sum()) <= budget

    def cost(self, sample: dict[str, Any]) -> int:
        """Load `add(sample)` would add to the row: `depth + 1` for each of its tokens not already in the trie."""
        input_ids = sample["input_ids"].tolist()
        children, shared = self.roots, 0
        for key in zip(input_ids, [*input_ids[1:], None], strict=True):
            if key not in children:
                break
            children = self.children[children[key]]
            shared += 1
        return sum(depth + 1 for depth in range(shared, len(input_ids)))

    def add(self, sample: dict[str, Any]) -> None:
        input_ids = sample["input_ids"].tolist()
        children = self.roots
        nodes = []
        for depth, key in enumerate(zip(input_ids, [*input_ids[1:], None], strict=True)):
            if key not in children:
                children[key] = len(self.tokens)
                self.tokens.append(key[0])
                self.children.append({})
                self.load += depth + 1
            nodes.append(children[key])
            children = self.children[nodes[-1]]
        self.samples.append(sample)
        self.sample_nodes.append(nodes)
        self.groups.add(sample["group_id"])
        self.trained += int(sample["completion_mask"].sum())

    def pack(self) -> dict[str, torch.Tensor]:
        """
        Lay the trie out in DFS pre-order, and select the loss terms taken over it.

        Unary chains collapse into contiguous segments. A segment ends at a branch or at a sample end, so every sample
        is exactly the concatenation of whole segments along one root-to-node path. A trained token is scored off its
        predecessor's hidden state, so each loss term points at the packed position of that predecessor's node.

        Returns:
            `dict[str, torch.Tensor]`, of `N` forwarded tokens, `S` segments and `M` trained tokens:
                - `input_ids` (`N`): the unique tokens.
                - `position_ids` (`N`): each token's position in its samples, which is its depth in the trie.
                - `shift_labels` (`N`): next token each position predicts, `-100` where it is not trained.
                - `sampling_mask` (`T`), `sampling_mask_counts` (`N`): candidate sets the sampler drew the next tokens
                  from, laid end to end in position order, and the size of each position's set; `T` is `0` without
                  top-k/top-p truncation. Ragged, because the sets average a few tokens but can reach the top-k cap.
                - `sampler_ids` (`N`, `K`), `sampler_log_probs` (`N`, `K`): the sampler's most likely next tokens at
                  the position and their log-probs, padded with `-1` and `-inf`; `K` is `0` unless requested.
                - `tree_kv_index` (`K`): packed position of every K/V row [`tree_attention`] gathers, the
                  root-to-segment paths of all segments laid end to end.
                - `tree_cu_seqlens_q` (`S + 1`), `tree_cu_seqlens_k` (`S + 1`): segment boundaries in the row, path
                  boundaries in `tree_kv_index`.
                - `tree_segment_parents` (`S`), `tree_segment_depths` (`S`): the segment tree, for the recurrent layers
                  patched by [`patch_tree_gdn`].
                - `tree_sizes` (`4 + depth`): longest segment, longest path, `K`, 64-token chunks, then the segments
                  per depth. Every length the kernels launch with, so a rank reads them in one go (see [`unpad_tree`]).
                - `token_index` (`M`): position whose prediction is each trained token. A shared position serves every
                  sample that shares it, so there are more loss terms than labelled positions.
                - `old_log_probs` (`M`), `advantages` (`M`): generator log-probability and advantage of each target.
                - `segment_id` (`M`): index of the sample each target came from.
        """
        input_ids, position_ids, offsets, paths, kv_index, cu_seqlens_k = [], [], [0], [], [], [0]
        parents, depths = [], []
        packed = [0] * len(self.tokens)
        stack = [(node, None, 0) for node in reversed(self.roots.values())]
        while stack:
            node, parent, depth = stack.pop()
            segment = len(paths)
            start = len(input_ids)
            parents.append(-1 if parent is None else parent)
            depths.append(0 if parent is None else depths[parent] + 1)
            while True:
                packed[node] = len(input_ids)
                input_ids.append(self.tokens[node])
                position_ids.append(depth)
                depth += 1
                if len(self.children[node]) != 1:
                    break
                node = next(iter(self.children[node].values()))
            offsets.append(len(input_ids))
            paths.append((paths[parent] if parent is not None else []) + list(range(start, len(input_ids))))
            kv_index.extend(paths[-1])
            cu_seqlens_k.append(len(kv_index))
            stack.extend((child, segment, depth) for child in reversed(self.children[node].values()))

        lengths = [end - start for start, end in zip(offsets[:-1], offsets[1:], strict=True)]
        shift_labels = [-100] * len(input_ids)
        sampling_mask = [[] for _ in input_ids]
        num_logprobs = max(sample["sampler_ids"].shape[1] for sample in self.samples)
        sampler_ids = np.full((len(input_ids), num_logprobs), -1, dtype=np.int64)
        sampler_log_probs = np.full((len(input_ids), num_logprobs), -math.inf, dtype=np.float32)
        token_index, old_log_probs, advantages, segment_id = [], [], [], []
        for i, (sample, nodes) in enumerate(zip(self.samples, self.sample_nodes, strict=True)):
            ids, mask, logps = (sample[key].tolist() for key in ("input_ids", "completion_mask", "old_log_probs"))
            first = len(token_index)
            for p in range(1, len(nodes)):
                if mask[p]:
                    shift_labels[packed[nodes[p - 1]]] = ids[p]
                    sampling_mask[packed[nodes[p - 1]]] = sample["sampling_mask"][p]
                    token_index.append(packed[nodes[p - 1]])
                    old_log_probs.append(logps[p])
                    advantages.append(sample["advantage"])
                    segment_id.append(i)
            trained = np.flatnonzero(sample["completion_mask"][1:]) + 1
            sampler_ids[token_index[first:], : sample["sampler_ids"].shape[1]] = sample["sampler_ids"][trained]
            sampler_log_probs[token_index[first:], : sample["sampler_ids"].shape[1]] = sample["sampler_log_probs"][
                trained
            ]
        return {
            "input_ids": torch.tensor(input_ids),
            "position_ids": torch.tensor(position_ids),
            "shift_labels": torch.tensor(shift_labels),
            "sampling_mask": torch.tensor(list(chain.from_iterable(sampling_mask)), dtype=torch.long),
            "sampling_mask_counts": torch.tensor(list(map(len, sampling_mask))),
            "sampler_ids": torch.from_numpy(sampler_ids),
            "sampler_log_probs": torch.from_numpy(sampler_log_probs),
            "tree_kv_index": torch.tensor(kv_index, dtype=torch.long),
            "tree_cu_seqlens_q": torch.tensor(offsets, dtype=torch.int32),
            "tree_cu_seqlens_k": torch.tensor(cu_seqlens_k, dtype=torch.int32),
            "tree_segment_parents": torch.tensor(parents, dtype=torch.int32),
            "tree_segment_depths": torch.tensor(depths, dtype=torch.int32),
            "tree_sizes": torch.tensor(
                [max(lengths), max(map(len, paths)), len(kv_index), sum(-(-length // 64) for length in lengths)]
                + [depths.count(depth) for depth in range(max(depths) + 1)]
            ),
            "token_index": torch.tensor(token_index, dtype=torch.long),
            "old_log_probs": torch.tensor(old_log_probs, dtype=torch.float32),
            "advantages": torch.tensor(advantages, dtype=torch.float32),
            "segment_id": torch.tensor(segment_id, dtype=torch.long),
        }


def unpad_tree(inputs: dict[str, torch.Tensor]) -> dict[str, Any]:
    """
    This rank's tree from [`TreeRow.pack`], as the model's forward kwargs: the collator's inter-rank padding sliced
    off, and the sizes the kernels launch with read to the host in a single transfer.
    """
    row = {key: value[0] for key, value in inputs.items() if key.startswith("tree_")}
    max_seqlen_q, max_seqlen_k, n_kv, n_chunks, *level_sizes = [n for n in row["tree_sizes"].tolist() if n > 0]
    n_segments = sum(level_sizes)
    return {
        "tree_kv_index": row["tree_kv_index"][:n_kv],
        "tree_cu_seqlens_q": row["tree_cu_seqlens_q"][: n_segments + 1],
        "tree_cu_seqlens_k": row["tree_cu_seqlens_k"][: n_segments + 1],
        "tree_max_seqlen_q": max_seqlen_q,
        "tree_max_seqlen_k": max_seqlen_k,
        "tree_segment_parents": row["tree_segment_parents"][:n_segments],
        "tree_segment_depths": row["tree_segment_depths"][:n_segments],
        "tree_level_sizes": level_sizes,
        "tree_n_chunks": n_chunks,
    }


TREE_ATTENTION = "tree_fa3"


@cache
def _flash_attn():
    from kernels import get_kernel

    if torch.version.hip:
        return get_kernel("kernels-community/aiter-flash-attn", version=2)
    # FlashAttention-3 has no kernels for Blackwell GPUs (compute capability 10 and above), FlashAttention-4 does
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10:
        return get_kernel("kernels-community/flash-attn4", version=1)
    return get_kernel("kernels-community/flash-attn3", version=1)


def tree_attention(
    module,
    query,
    key,
    value,
    attention_mask,
    tree_kv_index,
    tree_cu_seqlens_q,
    tree_cu_seqlens_k,
    tree_max_seqlen_q,
    tree_max_seqlen_k,
    scaling=None,
    sliding_window=None,
    softcap=None,
    **kwargs,
):
    """
    Attention over a packed [`TreeRow`]: each segment's queries attend to the K/V of its root-to-segment path, gathered
    with `tree_kv_index`, in one varlen FlashAttention call.

    A segment has fewer queries than keys: its ancestors' tokens, then its own. FlashAttention aligns the causal mask
    bottom-right, so each query sees every ancestor and its own causal past. For the segment `[p3 a1 a2]` under the
    segment `[p1 p2]`:

    ```
            p1  p2  p3  a1  a2
        p3   ■   ■   ■   ·   ·
        a1   ■   ■   ■   ■   ·
        a2   ■   ■   ■   ■   ■
    ```
    """
    dtype = get_target_dtype(query, module) or query.dtype
    query, key, value = (x.transpose(1, 2)[0].to(dtype) for x in (query, key, value))
    out = _flash_attn().flash_attn_varlen_func(
        query,
        key.index_select(0, tree_kv_index),
        value.index_select(0, tree_kv_index),
        cu_seqlens_q=tree_cu_seqlens_q,
        cu_seqlens_k=tree_cu_seqlens_k,
        max_seqlen_q=tree_max_seqlen_q,
        max_seqlen_k=tree_max_seqlen_k,
        softmax_scale=scaling,
        causal=True,
        window_size=(sliding_window - 1, 0) if sliding_window else (-1, -1),
        **({"softcap": softcap} if softcap else {}),
    )
    out = out[0] if isinstance(out, tuple) else out
    return out.unsqueeze(0), None


def register_tree_attention() -> None:
    AttentionInterface.register(TREE_ATTENTION, tree_attention)
    AttentionMaskInterface.register(TREE_ATTENTION, lambda **kwargs: None)


_PATCHED = set()


def patch_tree_gdn(model) -> bool:
    """
    Make the model's gated DeltaNet layers follow the tree of a tree-packed forward.

    Transformers' hybrids (Qwen3.5, Qwen3-Next, OLMo-Hybrid, ...) run these layers through two module-level functions,
    `causal_conv1d_fn` and `torch_chunk_gated_delta_rule`, and pass the forward's keyword arguments down to both. Both
    are wrapped: given the tree of [`~packing.TreeRow.pack`] they follow it, otherwise they run unchanged.

    Returns:
        `bool`: whether the model's modeling module has those functions.
    """
    modeling = sys.modules[type(model).forward.__module__]
    conv, chunk = vars(modeling).get("causal_conv1d_fn"), vars(modeling).get("torch_chunk_gated_delta_rule")
    if conv is None or chunk is None:
        return False
    if modeling in _PATCHED:
        return True
    from ..kernels.tree_gated_delta_rule import tree_causal_conv1d, tree_chunk_gated_delta_rule

    def causal_conv1d_fn(hidden_states, weight, bias=None, activation=None, **kwargs):
        if "tree_segment_parents" not in kwargs:
            return conv(hidden_states, weight, bias, activation=activation, **kwargs)
        return tree_causal_conv1d(hidden_states, weight, bias, activation, kwargs, conv)

    def torch_chunk_gated_delta_rule(query, key, value, g, beta, use_qk_l2norm_in_kernel=False, **kwargs):
        if "tree_segment_parents" not in kwargs:
            return chunk(query, key, value, g=g, beta=beta, use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel, **kwargs)
        return tree_chunk_gated_delta_rule(query, key, value, g, beta, kwargs, use_qk_l2norm_in_kernel), None

    modeling.causal_conv1d_fn, modeling.torch_chunk_gated_delta_rule = causal_conv1d_fn, torch_chunk_gated_delta_rule
    _PATCHED.add(modeling)
    return True


logger = get_logger(__name__)


Messages: TypeAlias = list[dict[str, str]]


RolloutId: TypeAlias = str


def _by_token_id(top_logprobs: list[dict[str, float]]) -> list[dict[int, float]]:
    """Key each token's top log-probs by token id, which vLLM spells `token_id:<id>` when asked for ids."""
    return [{int(token.removeprefix("token_id:")): logprob for token, logprob in top.items()} for top in top_logprobs]


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
    One finished rollout, as returned by [`~trainer.async_grpo_trainer.AsyncRolloutLoop.generate_one`].

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
            Identifier of the group, counted from `0` in the rollout process.
        row_index (`int`):
            Index of the prompt's row in the training dataset.
        env_rewards (`list[tuple[type, float] or None]`):
            For a rollout whose environment defines `get_reward`, the environment class and its reward; else `None`.
        rollout_rewards (`list[float or None]`):
            [`~trainer.async_grpo_trainer.Rollout`]`.reward` of each rollout.
        queued_at (`float`, *optional*, defaults to `0.0`):
            `time.monotonic()` when the group was queued for scoring.
        rewards (`np.ndarray`, *optional*):
            Total reward of each rollout, summed over every reward source, NaN when no source scored it. Set before
            [`~trainer.async_grpo_trainer.AsyncRolloutLoop.compute_advantages`] is called.
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
    # On ROCm, an empty CUDA_VISIBLE_DEVICES is ignored when HIP_VISIBLE_DEVICES or ROCR_VISIBLE_DEVICES is set.
    os.environ["HIP_VISIBLE_DEVICES"] = ""
    for k in _CHILD_ENV_TO_STRIP:
        os.environ.pop(k, None)


def _spawn_stop_watcher(rollout_loop: "AsyncRolloutLoop", stop_event: MPEvent) -> threading.Thread:
    # Daemon thread that translates the parent's mp.Event into the child's
    # asyncio.Event so _run_loops breaks out of its gather.
    def _watch():
        # Poll instead of `stop_event.wait()`: a thread still blocked in `wait()` when this process dies is never
        # woken, and the parent's `stop_event.set()` in `stop()` then waits for it forever
        while not stop_event.is_set():
            time.sleep(0.1)
        try:
            rollout_loop._loop.call_soon_threadsafe(rollout_loop._stop_event.set)
        except RuntimeError:
            # Loop already closed (run() returned before stop fired). Nothing to do.
            pass

    watcher = threading.Thread(target=_watch, daemon=True, name="grpo-mp-stop-watcher")
    watcher.start()
    return watcher


def _child_main(
    loop_cls: "type[AsyncRolloutLoop]",
    loop_kwargs: dict[str, Any],
    samples_queue: MPQueue,
    model_version_value: MPValue,
    token_allowance: MPValue,
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

    rollout_loop = loop_cls(
        **loop_kwargs,
        rollout_buffer=samples_queue,
        model_version_value=model_version_value,
        token_allowance=token_allowance,
        heartbeat_value=heartbeat_value,
        failed_event=failed_event,
        exception_info_queue=exception_info_queue,
        metrics_queue=metrics_queue,
    )
    child_ready_event.set()
    watcher = _spawn_stop_watcher(rollout_loop, stop_event)
    try:
        rollout_loop._run()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        # The worker is stopping either way: release the watcher so it does not outlive `run()`
        stop_event.set()
        watcher.join(timeout=1.0)
        if watcher.is_alive():
            logger.warning("Stop watcher did not exit within 1s.")


class AsyncRolloutLoop:
    """
    Generates and scores rollouts for [`AsyncGRPOTrainer`], in a spawned rollout process.

    The default implements GRPO. To customize a run, subclass it, override only the methods that change, and set the
    subclass as the trainer's `loop_cls`. Each prompt goes through, in order:

    - [`~trainer.async_grpo_trainer.AsyncRolloutLoop.filter_prompt`]: whether to generate for the prompt at all.
    - [`~trainer.async_grpo_trainer.AsyncRolloutLoop.generate_one`]: one full rollout, which calls
      [`~trainer.async_grpo_trainer.AsyncRolloutLoop.generate_one_turn`] for each sampling request.
    - The group of `num_generations` rollouts is scored: its rewards are the sum of the trainer's `reward_funcs`, of
      environments that define `get_reward`, and of [`~trainer.async_grpo_trainer.Rollout`]`.reward`. Then
      [`~trainer.async_grpo_trainer.AsyncRolloutLoop.compute_advantages`] runs once per group.

    [`~trainer.async_grpo_trainer.AsyncRolloutLoop.setup`] and [`~trainer.async_grpo_trainer.AsyncRolloutLoop.close`]
    create and release resources on the loop's event loop, like an HTTP client. A hook can read `self.config` (a
    subclass of [`AsyncGRPOConfig`] can carry its own fields) and `self.tokenizer`. Since the loop is built in a
    spawned process, the subclass must be importable (defined at module level).
    """

    def __init__(
        self,
        config: AsyncGRPOConfig,
        model_name: str,
        dataset: Dataset,
        reward_funcs: list[Callable[..., list[float]]],
        processing_class: PreTrainedTokenizerBase,
        eos_token_ids: list[int],
        rollout_buffer: MPQueue,
        model_version_value: MPValue,
        token_allowance: MPValue,
        heartbeat_value: MPValue,
        failed_event: MPEvent,
        exception_info_queue: MPQueue,
        metrics_queue: MPQueue,
        tools: list[Callable] | None = None,
        environment_factory: Callable[[], object] | dict[str, Callable[[], object]] | None = None,
        score_queue_maxsize: int = 16,
        lora_name: str | None = None,
        dataset_start_index: int = 0,
    ):
        self.config = config
        self.model_name = model_name
        self.lora_name = lora_name
        self.dataset = dataset
        self._epoch, self._row_index = 0, 0
        self._dataset_iter = iter(dataset)
        if dataset_start_index > 0:
            self._epoch, self._row_index = divmod(dataset_start_index, len(dataset))
            self._dataset_iter = iter(dataset.select(range(self._row_index, len(dataset))))
        self.reward_funcs = reward_funcs
        self.reward_func_names = [get_callable_name(f) for f in reward_funcs]
        self.tokenizer = processing_class
        self.eos_token_ids = eos_token_ids
        self.rollout_buffer = rollout_buffer  # shared mp.Queue
        self._model_version_value = model_version_value  # shared mp.Value
        self._token_allowance = token_allowance
        self._produced_tokens = 0
        self._rollouts_unqueued = 0
        self._tokens_per_rollout = 0.0
        self._heartbeat_value = heartbeat_value  # shared mp.Value('d'); wall-clock seconds
        self._failed_event = failed_event  # shared mp.Event
        self._exception_info_queue = exception_info_queue  # shared mp.Queue(maxsize=1)
        self._metrics_queue = metrics_queue  # shared mp.Queue; drained by the trainer in `log()`
        # Metric accumulators
        self._counters: dict[str, float] = defaultdict(float)
        self._rates: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
        self._concurrency = 0  # set by the generate loop, reported by the score loop
        self._pushed_completion_tokens = 0
        self._pushed_at = time.monotonic()

        self.sampling_replay = config.top_p < 1.0 or config.top_k > 0
        self.num_sampler_logprobs = config.num_sampler_logprobs if config.score_centering else 0
        self.chat_template_kwargs = config.chat_template_kwargs or {}
        self.vllm_server_url = config.vllm_server_base_url.rstrip("/")

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

        # `add_response_schema` sets the response template (transformers >= 5.13) or legacy schema for known chat
        # templates, so tool calls can be parsed. Skip if one is already set; warn if it's a migratable legacy schema.
        has_template = getattr(processing_class, "response_template", None) is not None
        has_schema = getattr(processing_class, "response_schema", None) is not None
        if not has_template and not has_schema:
            try:
                processing_class = add_response_schema(processing_class)
            except ValueError:
                # Only tool calls need a parser: without tools, turns of an unknown template are decoded as plain text
                if self.tools:
                    raise
        elif has_schema and not has_template and _SUPPORTS_RESPONSE_TEMPLATE:
            warnings.warn(
                "The tokenizer has a legacy `response_schema` set but no `response_template`. The installed "
                "transformers supports the new-style `response_template`; consider migrating, as `response_schema` "
                "support will eventually be removed. See the Transformers response-parsing docs.",
                FutureWarning,
            )

        # The reward a rollout carries in `Rollout.reward` is the last reward column, set by a `generate_one` that
        # scores its own rollouts. The built-in `generate_one` leaves it at `None`.
        self.reward_func_names.append("rollout")
        if (
            not self.reward_funcs
            and not self._env_reward_types
            and type(self).generate_one is AsyncRolloutLoop.generate_one
        ):
            raise ValueError(
                "No reward source provided. Pass `reward_funcs`, an `environment_factory` whose environment defines a "
                "`get_reward` method, or a `loop_cls` whose `generate_one` sets `Rollout.reward`."
            )

        # Sync tools run here so a slow one never blocks the event loop; sized so every rollout can be in a tool call.
        self._tool_pool = ThreadPoolExecutor(
            max_workers=max(1, config.max_concurrent_rollouts), thread_name_prefix="grpo-tool"
        )

        self._kv_cap = config.max_concurrent_rollouts
        self._total_rollouts = 0
        self._metrics_poll_s = 5.0

        # The chat template must be prefix-preserving in multi-turn training; if the tokenizer's
        # template isn't, swap in a training-safe one.
        if self.tools and not is_chat_template_prefix_preserving(self.tokenizer):
            self.chat_template = get_training_chat_template(self.tokenizer)
        else:
            self.chat_template = None

        self._groups_to_score: asyncio.Queue[RolloutGroup | None] = asyncio.Queue(maxsize=score_queue_maxsize)
        self._total_completion_tokens = 0
        self._total_groups_scored = 0
        self.session: aiohttp.ClientSession | None = None

        self._loop = asyncio.new_event_loop()
        self._stop_event = asyncio.Event()

    @property
    def model_version(self) -> int:
        return int(self._model_version_value.value)

    @property
    def _request_model(self) -> str:
        # In vLLM's API an adapter *is* a model name: naming the base model while an adapter is loaded silently
        # serves the base model, so the published version has to be part of the request. `model_version` reads the
        # shared `mp.Value`, so a sync is picked up without extra IPC.
        if self.lora_name is None:
            return self.model_name
        return f"{self.lora_name}-v{self.model_version}"

    def _run(self) -> None:
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
            self._tool_pool.shutdown(wait=True, cancel_futures=True)
            self._loop.close()

    async def _check_sampling_replay(self) -> None:
        """With top-k/top-p, make sure the server returns the candidate set of every sampled token.

        That takes vLLM >= 0.28 started with `--return-sampling-mask` (recorded in its config) and `--enable-scale-out`
        (the endpoint that returns the set). Checked once up front: without them the first rollout would fail on a
        route that does not exist, after retrying it for minutes. An older vLLM cannot return the set at all, so it
        keeps the full-vocabulary normalization, with a warning about the bias.
        """
        if not self.sampling_replay:
            return
        async with self.session.get(f"{self.vllm_server_url}/version") as r:
            r.raise_for_status()
            version = (await r.json())["version"]
        if Version(version) < Version("0.28.0"):
            logger.warning(
                f"vLLM {version} cannot return the candidate set a token was sampled from (vLLM >= 0.28), so with "
                "top_k/top_p the log-probabilities are normalized over the whole vocabulary and the importance ratio "
                "is biased."
            )
            self.sampling_replay = False
            return
        async with self.session.get(f"{self.vllm_server_url}/server_info", params={"config_format": "json"}) as r:
            r.raise_for_status()
            model_config = (await r.json())["vllm_config"]["model_config"]
        async with self.session.get(f"{self.vllm_server_url}/openapi.json") as r:
            r.raise_for_status()
            paths = (await r.json())["paths"]
        if not model_config.get("return_sampling_mask") or "/inference/v1/generate" not in paths:
            raise RuntimeError(
                "top_k/top_p sampling needs the candidate set of every sampled token, which the trainer renormalizes "
                "its log-probabilities over. Start the vLLM server (>= 0.28) with "
                "`--return-sampling-mask --enable-scale-out`."
            )

    async def setup(self) -> None:
        """
        Create the resources the other hooks need, such as an HTTP client. Runs on the loop's event loop before the
        first prompt. Does nothing by default.
        """

    async def close(self) -> None:
        """
        Release what [`~trainer.async_grpo_trainer.AsyncRolloutLoop.setup`] created. Runs when the loops stop,
        including on failure. Does nothing by default.
        """

    async def _run_loops(self, stop_event: asyncio.Event) -> None:
        connector = aiohttp.TCPConnector(limit=self.config.max_concurrent_rollouts + 1)
        timeout = aiohttp.ClientTimeout(total=self.config.vllm_server_timeout)
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            self.session = session
            await self._check_sampling_replay()
            await self._init_kv_cap()
            logger.info(
                f"vllm worker started: num_generations={self.config.num_generations}, "
                f"max_concurrent_rollouts={self.config.max_concurrent_rollouts}, kv_cap={self._kv_cap}, "
                f"temperature={self.config.temperature}, top_p={self.config.top_p}, top_k={self.config.top_k}, min_p={self.config.min_p}"
            )
            await self.setup()
            try:
                await asyncio.gather(
                    asyncio.create_task(self._generate_loop(stop_event=stop_event)),
                    asyncio.create_task(self._score_loop(stop_event=stop_event)),
                    asyncio.create_task(self._control_loop(stop_event=stop_event)),
                )
            finally:
                await self.close()

    def _fits_token_allowance(self) -> bool:
        """Whether a new group of rollouts fits in the token allowance the trainer sets.

        Counts the tokens already queued, plus the rollouts in flight at the mean tokens per rollout. Until a group has
        been scored there is no mean, so groups start one at a time.
        """
        if self._rollouts_unqueued and not self._tokens_per_rollout:
            return False
        expected_tokens = self._produced_tokens + self._rollouts_unqueued * self._tokens_per_rollout
        return expected_tokens < self._token_allowance.value

    async def _init_kv_cap(self) -> None:
        """Start the cap at `KV cache tokens / max_model_len`: as many requests as fit at full length."""
        try:
            async with self.session.get(f"{self.vllm_server_url}/metrics") as response:
                response.raise_for_status()
                text = await response.text()
            async with self.session.get(f"{self.vllm_server_url}/v1/models") as response:
                response.raise_for_status()
                models = await response.json()
        except Exception as e:
            logger.warning(f"Could not read the vLLM server's KV capacity, starting at max_concurrent_rollouts: {e!r}")
            return
        kv_matches = re.findall(r'^vllm:cache_config_info\{.*kv_cache_size_tokens="(\d+)"', text, re.M)
        if not kv_matches:
            logger.warning("`/metrics` has no `kv_cache_size_tokens`, starting at max_concurrent_rollouts")
            return
        kv_tokens = sum(map(int, kv_matches))
        max_model_len = max(model["max_model_len"] for model in models["data"] if model["max_model_len"])
        self._kv_cap = max(1, min(self.config.max_concurrent_rollouts, kv_tokens // max_model_len))

    async def _control_loop(self, stop_event: asyncio.Event) -> None:
        """Size the KV cap from the vLLM server's load, polled from `/metrics`.

        Cut the cap by 30% when the server's KV cache usage passes 0.8, once the pool has drained below the previous
        cut. Nothing is cancelled; admissions stop until enough rollouts finish. Grow it by 25% while usage stays under
        0.6 with nothing queued, once per turnover of the pool. Preemptions are reported, not acted on: a weight sync
        resets the server's caches, which preempts every running request.
        """
        rollouts_at_change = 0
        preemptions_seen = None
        while not stop_event.is_set():
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=self._metrics_poll_s)
                return
            except TimeoutError:
                pass
            try:
                timeout = aiohttp.ClientTimeout(total=30)
                async with self.session.get(f"{self.vllm_server_url}/metrics", timeout=timeout) as response:
                    response.raise_for_status()
                    text = await response.text()
            except Exception as e:
                logger.warning(f"vLLM /metrics poll failed, KV cap unchanged: {e!r}")
                continue
            usage_matches = re.findall(r"^vllm:kv_cache_usage_perc\{.*\} (\S+)", text, re.M)
            queued_matches = re.findall(r"^vllm:num_requests_waiting\{.*\} (\S+)", text, re.M)
            if not usage_matches or not queued_matches:
                # An empty match looks identical to a genuinely idle server (usage=0.0, queued=0), which is exactly
                # the condition that grows the cap -- so a metrics rename/reformat must not fall through to that
                # arithmetic silently.
                logger.warning("vLLM /metrics missing KV usage or queue depth, KV cap unchanged")
                continue
            usage = max(map(float, usage_matches))
            running = sum(map(float, re.findall(r"^vllm:num_requests_running\{.*\} (\S+)", text, re.M)))
            queued = sum(map(float, queued_matches))
            preemptions = sum(map(float, re.findall(r"^vllm:num_preemptions_total\{.*\} (\S+)", text, re.M)))
            if preemptions_seen is None:
                preemptions_seen = preemptions
            if usage > 0.8 and self._concurrency <= self._kv_cap:
                self._kv_cap = max(1, int(self._kv_cap * 0.7))
                rollouts_at_change = self._total_rollouts
                logger.info(f"[control] cut KV cap to {self._kv_cap} (kv cache usage {usage:.2f})")
            elif usage < 0.6 and queued == 0 and self._total_rollouts - rollouts_at_change >= self._kv_cap:
                self._kv_cap = min(math.ceil(self._kv_cap * 1.25), self.config.max_concurrent_rollouts)
                rollouts_at_change = self._total_rollouts
            self._push_metrics(
                {
                    "rollout/concurrency_cap": float(self._kv_cap),
                    "rollout/vllm_kv_cache_usage_frac": usage,
                    "rollout/vllm_running_requests": running,
                    "rollout/vllm_queued_requests": queued,
                    "rollout/vllm_preemption_count": preemptions - preemptions_seen,
                }
            )
            preemptions_seen = preemptions

    async def _generate_loop(self, stop_event: asyncio.Event) -> None:
        pending_groups: dict[int, RolloutGroup] = {}
        pending_completed: dict[int, int] = {}
        rollout_tasks: dict[asyncio.Task, tuple[int, Any, object, Messages, float]] = {}
        work_iter = self._repeat_iterator()
        loop = asyncio.get_running_loop()
        t_gated = None
        dispatched = 0

        try:
            while True:
                # Wall-clock for cross-process comparison; parent uses time.time() in check_health.
                self._heartbeat_value.value = time.time()
                while len(rollout_tasks) < self._kv_cap and not stop_event.is_set():
                    starts_group = dispatched % self.config.num_generations == 0
                    if starts_group and not self._fits_token_allowance():
                        break
                    group_id, row_index, row = next(work_iter)
                    if row is None:  # rejected by filter_prompt: the trainer counts it as consumed
                        self._put_in_buffer(group_id)
                        await asyncio.sleep(0)
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
                    # prompt and capture its initial observation (e.g. a task instruction), since an environment can
                    # hold the task in the observation rather than in the dataset prompt. Mirror GRPOTrainer: fold the
                    # observation into the last prompt message. reset() may be stochastic, so this is done per generation (each
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
                    self._rollouts_unqueued += 1
                    dispatched += 1
                    if t_gated is not None:
                        self._push_metrics({"rollout/backpressure_s": time.monotonic() - t_gated})
                        t_gated = None

                if (
                    t_gated is None
                    and self._tokens_per_rollout
                    and len(rollout_tasks) < self._kv_cap
                    and not stop_event.is_set()
                ):
                    t_gated = time.monotonic()
                    logger.info("generation is max_staleness steps ahead, waiting for the trainer to consume...")
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
            self._heartbeat_value.value = time.time()
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
                f"[score] scored {len(samples)} samples in {scoring_time:.2f}s, "
                f"buffer_qsize={self.rollout_buffer.qsize()}"
            )

            now = time.monotonic()
            self._push_metrics(
                {
                    "rollout/score_s": scoring_time,
                    "rollout/score_wait_s": wait_scoring,
                    "rollout/score_queue_size": float(self._groups_to_score.qsize()),
                    "rollout/concurrency": float(self._concurrency),
                    # Windowed, not cumulative.  A cumulative average cannot show a generation stall.
                    "rollout/generated_tokens_per_s": (
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
                    rewards={"reward": [s.metrics["reward_trained"] for s in samples]},
                    advantages=[s.advantage for s in samples],
                    step=self._total_groups_scored,
                    num_samples=self.config.num_completions_to_print,
                )
            self._total_groups_scored += 1

            for item in samples or [group.group_id]:
                self._put_in_buffer(item)
            self._rollouts_unqueued -= self.config.num_generations
            tokens = sum(len(seq.input_ids) for seqs in group.completions_sequences for seq in seqs)
            tokens /= self.config.num_generations
            self._tokens_per_rollout = (
                tokens if not self._tokens_per_rollout else 0.9 * self._tokens_per_rollout + 0.1 * tokens
            )

            t_idle_start = time.monotonic()

    def _put_in_buffer(self, item: RolloutSample | int) -> None:
        """Put `item` in the rollout buffer and count its tokens toward the token allowance."""
        if isinstance(item, RolloutSample):
            item.enqueued_at = time.time()
            self._produced_tokens += len(item.input_ids)
        self.rollout_buffer.put_nowait(item)

    async def _retry(self, coro_factory: Callable[[], Awaitable], label: str, max_attempts: int = 1):
        """Retry an aiohttp coroutine on transport errors and 5xx responses with bounded exponential backoff, counting the
        retries.

        A 4xx is raised at once, since sending the same request again gets the same answer.

        Every retried call in this loop targets vLLM, so the counter is named for the dependency rather than for the
        transport.
        """
        for attempt in range(max_attempts):
            try:
                return await coro_factory()
            except _RETRYABLE_HTTP_ERRORS as e:
                if attempt >= max_attempts - 1 or (
                    isinstance(e, aiohttp.ClientResponseError) and 400 <= e.status < 500
                ):
                    raise
                self._counters["rollout/vllm_retry_count"] += 1
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
        self,
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
        terminated_lengths = {}
        if completion_ids:
            # Truncation is read off the same way [`GRPOTrainer`] and [`RLOOTrainer`] define
            # `completions/clipped_ratio`: a completion that does not end on EOS (or pad) was cut off by `max_tokens`
            # rather than finishing. Deliberately NOT vLLM's `finish_reason`, so the metric means the same thing here.
            eos_and_pad = (*self.eos_token_ids, self.tokenizer.pad_token_id)
            is_truncated = completion_ids[-1] not in eos_and_pad
            self._rates["completions/clipped_ratio"][0] += is_truncated
            self._rates["completions/clipped_ratio"][1] += 1
            # Identify sequences that terminated with EOS and log their lengths
            if not is_truncated:
                self._rates["completions/mean_terminated_length"][0] += completion_tokens
                self._rates["completions/mean_terminated_length"][1] += 1
                terminated_lengths = {
                    "completions/max_terminated_length": float(completion_tokens),
                    "completions/min_terminated_length": float(completion_tokens),
                }
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
                **terminated_lengths,
            }
        )

    def filter_prompt(self, row_index: int, row: dict[str, Any], epoch: int) -> bool:
        """
        Whether to generate for a prompt. A rejected prompt is skipped for this epoch and counts as consumed, like a
        dropped group. Admits every prompt by default. Rejecting every prompt of a full epoch raises, since no batch
        can reach the trainer after that.

        Args:
            row_index (`int`):
                Index of the row in the training dataset.
            row (`dict[str, Any]`):
                The dataset row.
            epoch (`int`):
                Number of full passes over the dataset before this one.

        Returns:
            `bool`: `True` to generate `num_generations` rollouts for the prompt, `False` to skip it.
        """
        return True

    def _repeat_iterator(self) -> Iterator[tuple[int, int, dict[str, Any] | None]]:
        """Yield `(group_id, row_index, row)` once per generation of each prompt, or `(group_id, row_index, None)` once
        for a prompt that `filter_prompt` rejects."""
        group_id = 0
        accepted_in_epoch = self._row_index > 0
        while True:
            try:
                row = next(self._dataset_iter)
            except StopIteration:
                if not accepted_in_epoch:
                    raise RuntimeError(
                        f"`filter_prompt` rejected every prompt of epoch {self._epoch}, so no batch can reach the "
                        "trainer."
                    ) from None
                self._dataset_iter = iter(self.dataset)
                self._row_index, self._epoch = 0, self._epoch + 1
                accepted_in_epoch = False
                row = next(self._dataset_iter)
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
        reply with [`~trainer.async_grpo_trainer.AsyncRolloutLoop.generate_one_turn`] and runs the tool calls it
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
            [`~trainer.async_grpo_trainer.Rollout`]: The finished rollout.
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
            # Decode the turn. It's important to use `parse_response` when possible, because it handles tool calls.
            if (
                getattr(self.tokenizer, "response_template", None) is not None  # new-style
                or getattr(self.tokenizer, "response_schema", None) is not None  # old-style
            ):
                assistant_message = parse_response(self.tokenizer, turn.output_ids, prefix=prompt_ids, tools=tools)
            else:
                content = self.tokenizer.decode(turn.output_ids, skip_special_tokens=True)
                assistant_message = {"role": "assistant", "content": content}
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
            self._counters[f"tools/{name}/call_count"] += 1
            tool = tool_dict.get(name)
            if tool is None:
                # A hallucinated tool name is a policy error that should decay with training, unlike a tool that ran
                # and raised — which is an environment problem. Counted apart so the two are not one number.
                n_failures += 1
                self._counters["tools/unknown_name_count"] += 1
                self._counters[f"tools/{name}/failure_count"] += 1
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
                self._counters[f"tools/{name}/failure_count"] += 1
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
        Sample one reply from the vLLM server, with the config's sampling settings (`temperature`, `top_p`, `top_k`,
        `min_p`, `repetition_penalty` and `max_completion_length`).

        Args:
            prompt_ids (`list[int]`):
                Token ids of the full prompt.

        Returns:
            [`TurnRecord`]: The prompt and the sampled tokens with their log-probabilities. With `top_k` or `top_p`, it
            also holds the candidate set of each sampled token, and with `num_sampler_logprobs`, the sampler's top
            log-probabilities.
        """
        sampling_params = {
            "max_tokens": self.config.max_completion_length,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
            # Explicit: vLLM fills an omitted penalty from the model's generation config (1.05 for Qwen2.5-Instruct),
            # and the trainer's log-probs carry none
            "repetition_penalty": 1.0,
            "n": 1,
            "logprobs": self.num_sampler_logprobs,
            # Unseeded requests draw their sampling seed from an RNG that every data-parallel vLLM engine seeds
            # identically, so samples of the same prompt spread across engines can decode the same text. A unique seed
            # per request keeps them independent.
            "seed": random.getrandbits(63),
        }
        if self.config.min_p is not None:
            sampling_params["min_p"] = self.config.min_p
        if not self.sampling_replay:
            payload = {
                "model": self._request_model,
                "prompt": prompt_ids,
                "return_token_ids": True,
                "return_tokens_as_token_ids": True,
                "cache_salt": f"v{self.model_version}",
                **sampling_params,
            }
            output = await self._retry(
                lambda: self._post("/v1/completions", payload, self.config.vllm_server_timeout),
                max_attempts=30,
                label="vllm /v1/completions",
            )
            choice = output["choices"][0]
            sampler_logprobs = _by_token_id(choice["logprobs"]["top_logprobs"]) if self.num_sampler_logprobs else None
            return TurnRecord(
                prompt_ids,
                choice["token_ids"],
                choice["logprobs"]["token_logprobs"],
                output_sampler_logprobs=sampler_logprobs,
            )
        payload = {
            "model": self._request_model,
            "token_ids": prompt_ids,
            "sampling_params": sampling_params,
            "cache_salt": f"v{self.model_version}",
        }
        output = await self._retry(
            lambda: self._post("/inference/v1/generate", payload, self.config.vllm_server_timeout),
            max_attempts=30,
            label="vllm /inference/v1/generate",
        )
        choice = output["choices"][0]
        content = choice["logprobs"]["content"]
        logprobs = [token["logprob"] for token in content]
        sampler_logprobs = None
        if self.num_sampler_logprobs:
            sampler_logprobs = _by_token_id(
                [{t["token"]: t["logprob"] for t in token["top_logprobs"]} for token in content]
            )
        return TurnRecord(
            prompt_ids,
            choice["token_ids"],
            logprobs,
            output_sampling_mask=choice["sampling_mask"],
            output_sampler_logprobs=sampler_logprobs,
        )

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
        for reward_func_name, rewards in zip(self.reward_func_names, all_rewards, strict=False):
            if len(rewards) != len(group.completions):
                raise ValueError(
                    f"The reward function '{reward_func_name}' returned {len(rewards)} rewards, but "
                    f"{len(group.completions)} were expected (one per prompt-completion pair). Make sure the reward "
                    f"function returns exactly one reward per completion."
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
        all_rewards = np.array(
            [[r if r is not None else float("nan") for r in row] for row in all_rewards], dtype=float
        )

        # If all reward functions return None for a given row, issue a detailed warning
        if np.isnan(all_rewards).all(axis=0).any():
            nan_row_idx = int(np.isnan(all_rewards).all(axis=0).nonzero()[0][0])
            row_reward_kwargs = {key: value[nan_row_idx] for key, value in group.reward_kwargs.items()}
            row_reward_kwargs["prompt"] = group.prompts[nan_row_idx]
            row_reward_kwargs["completion"] = group.completions[nan_row_idx]
            logger.warning(
                f"All reward functions returned None for the following kwargs:\n{row_reward_kwargs}\n"
                "Please ensure that at least one reward function returns a valid reward."
            )
        return all_rewards

    def compute_advantages(self, group: RolloutGroup) -> np.ndarray:
        """
        Compute one advantage per rollout of a group. By default, each reward standardized within the group; a rollout
        with a NaN reward gets advantage `0` and is left out of the mean and std.

        It runs for every group. A group whose scored rewards are all equal is then dropped, whatever advantages this
        returns.

        Args:
            group ([`~trainer.async_grpo_trainer.RolloutGroup`]):
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
        # Completions that repeat an earlier one of the group token for token; about 0 when sampling is random
        num_completions = len(group.completions_ids)
        num_distinct = len({tuple(ids) for ids in group.completions_ids})
        self._rates["completions/duplicate_frac"][0] += num_completions - num_distinct
        self._rates["completions/duplicate_frac"][1] += num_completions
        logger.info(f"Rollout metrics: reward_mean={reward_mean:.4f}, reward_std={reward_std:.4f}")

        self._rates["reward"][0] += float(scored.sum())
        self._rates["reward"][1] += len(scored)
        # Each reward function's mean over the rollouts it scored, dropped groups included, like `reward`. A function
        # that scored no rollout of the group (e.g. the `rollout` column when `generate_one` sets no reward) adds nothing.
        for name, func_rewards in zip(self.reward_func_names, per_func_rewards, strict=True):
            func_scored = func_rewards[~np.isnan(func_rewards)]
            if len(func_scored):
                self._rates[f"rewards/{name}/mean"][0] += float(func_scored.sum())
                self._rates[f"rewards/{name}/mean"][1] += len(func_scored)
        if self.tools or any(group.tool_call_counts):
            self._rates["tools/call_frequency"][0] += sum(group.tool_call_counts)
            self._rates["tools/call_frequency"][1] += len(group.tool_call_counts)
            self._rates["tools/failure_frequency"][0] += sum(group.tool_failure_counts)
            self._rates["tools/failure_frequency"][1] += sum(group.tool_call_counts)
        zero_std = bool(np.all(scored == scored[:1]))
        self._rates["frac_zero_std_groups"][0] += zero_std
        self._rates["frac_zero_std_groups"][1] += 1
        if zero_std:
            if len(scored):
                # The reward a dropped group is stuck at: near the top of its range, the prompts are too easy; near the
                # bottom, too hard. With 0/1 rewards, this is the share of dropped groups whose completions all pass.
                self._rates["reward_zero_std_groups"][0] += reward_mean
                self._rates["reward_zero_std_groups"][1] += 1
            return []

        # A reward source that scored no rollout of the group (e.g. the `rollout` column when `generate_one` sets no
        # reward) is not logged, rather than logged as NaN.
        used_columns = ~np.all(np.isnan(per_func_rewards), axis=1)

        per_func_std = [float(np.nanstd(r)) if not np.isnan(r).all() else float("nan") for r in per_func_rewards]
        # Option 3 advantage: one advantage per conversation, stamped on every training row it produced.
        # Under TRL's global token-mean loss a fork is invisible to the loss (see slime_research.md B.5),
        # so we do not split the reward or the advantage across a conversation's rows.
        samples = []
        for prompt, completion, sequences, advantage, reward in zip(
            group.prompts, group.completions, group.completions_sequences, advantages, rewards, strict=True
        ):
            metrics = {
                "reward_trained": float(reward),
                "reward_group_std": reward_std,
                **{
                    f"rewards/{name}/group_std": std
                    for name, std, used in zip(self.reward_func_names, per_func_std, used_columns, strict=True)
                    if used
                },
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

    async def _post(self, path: str, payload: dict, timeout: float, max_retries: int = 3) -> dict:
        client_timeout = aiohttp.ClientTimeout(total=timeout)

        async def _do_post():
            async with self.session.post(
                f"{self.vllm_server_url}{path}", json=payload, timeout=client_timeout
            ) as response:
                if response.status >= 400:
                    raise aiohttp.ClientResponseError(
                        response.request_info, response.history, status=response.status, message=await response.text()
                    )
                content = await response.json()
                return content if content else {}

        return await self._retry(_do_post, label=f"POST {path}", max_attempts=max_retries)


class AsyncRolloutWorker:
    """Parent-side controller: spawns a child process running `AsyncRolloutLoop`.

    The trainer holds this object on rank 0. The child does the actual rollout work; this class only manages lifecycle
    (start/stop) and exposes the shared `mp.Queue` (`rollout_buffer`) and `mp.Value` (`model_version`) the trainer
    reads/writes.

    Constructor kwargs are forwarded as-is to `loop_cls` (an [`~trainer.async_grpo_trainer.AsyncRolloutLoop`] subclass)
    when the child spawns; only `loop_cls` and `child_ready_timeout` are consumed here. Because the child is spawned,
    every forwarded kwarg is pickled: `reward_funcs`, `tools`, and `environment_factory` (and anything they close over)
    must be picklable — module-level functions, `functools.partial`, or callable instances, never lambdas or closures.
    `start()` validates this up front and raises a `TypeError` otherwise. The child also runs with
    `CUDA_VISIBLE_DEVICES=""`, so GPU reward models execute on CPU.
    """

    def __init__(
        self,
        config: AsyncGRPOConfig,
        loop_cls: type[AsyncRolloutLoop] = AsyncRolloutLoop,
        child_ready_timeout: int = 300,
        **loop_kwargs: Any,
    ):
        if not is_aiohttp_available():
            raise ImportError("aiohttp is not installed. Please install it with `pip install aiohttp`.")
        ctx = mp.get_context("spawn")
        self._mp_ctx = ctx
        self.rollout_buffer = ctx.Queue()
        self._model_version_value = ctx.Value("i", 0)
        self.token_allowance = ctx.Value("q", 2**62)
        self._stop_event_mp = ctx.Event()
        self._child_ready_event = ctx.Event()
        # Liveness state shared with the child. Wall-clock seconds because monotonic() is per-process.
        self._heartbeat_value = ctx.Value("d", 0.0)
        self._failed_event = ctx.Event()
        self._exception_info_queue = ctx.Queue(maxsize=1)
        # Metrics the child measures, drained by the trainer in `log()`. Bounded and dropped-on-full in the child, so a
        # trainer that stops draining can never block generation.
        self.metrics_queue = ctx.Queue(maxsize=4096)
        # Forwarded verbatim to AsyncRolloutLoop in the child.
        loop_kwargs["config"] = config
        loop_kwargs.setdefault("dataset_start_index", 0)
        self._loop_kwargs = loop_kwargs
        self._loop_cls = loop_cls
        self._child_ready_timeout = child_ready_timeout
        self._process: mp.Process | None = None

    @property
    def model_version(self) -> int:
        return int(self._model_version_value.value)

    @model_version.setter
    def model_version(self, value: int) -> None:
        # mp.Value reads and writes aren't atomic, so hold its lock
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
                "AsyncRolloutWorker forwards the rollout loop's arguments (reward_funcs, tools, "
                "environment_factory, harness callables) to a spawned child process, so they must be picklable. "
                "Lambdas and closures are not: use a module-level function, functools.partial, or a callable class "
                "instance instead (tools are registered under their __name__, so they must be module-level functions)."
            ) from e
        self._process = self._mp_ctx.Process(
            target=_child_main,
            args=(
                self._loop_cls,
                self._loop_kwargs,
                self.rollout_buffer,
                self._model_version_value,
                self.token_allowance,
                self._stop_event_mp,
                self._child_ready_event,
                self._heartbeat_value,
                self._failed_event,
                self._exception_info_queue,
                self.metrics_queue,
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
    - a `count` word in the name is a count of events in the window, summed (producers push deltas).
    - a `max` or `min` word is an extremum. Matching whole words rather than a suffix keeps the upstream spellings
      (`completions/max_length`, `clip_ratio/high_max`) working alongside this trainer's own (`row_tokens_max`).
    - anything else is a gauge, and its window mean is meaningful.

    A key must always be logged with the same shape, since the reduction is picked from the first value seen.
    """
    if isinstance(values[0], tuple):
        numerator, denominator = (sum(v) for v in zip(*values, strict=True))
        return numerator / denominator if denominator else float("nan")
    words = key.split("/")[-1].split("_")
    if "count" in words:
        return sum(values)
    if "max" in words:
        return max(values)
    if "min" in words:
        return min(values)
    return sum(values) / len(values)


class _SupportsReset(Protocol):
    def reset(self, **kwargs) -> str | None: ...


EnvironmentFactory = Callable[[], _SupportsReset]


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

    An epoch is counted in distinct prompt-groups consumed: trained (accumulated in the collator, which runs on the
    main process just before the model forward), dropped because all its scored rewards were equal, or rejected by
    `filter_prompt`. This is fork-independent: all generations of a prompt and all forked rows of a conversation share
    one `group_id`, so a conversation forking into many rows still counts once. Only the main process collates
    (`dispatch_batches=True`), so the stop decision is reduced across ranks to keep data-parallel workers in lockstep.
    """

    def __init__(self, trainer: "AsyncGRPOTrainer", target_groups: int):
        self._trainer = trainer
        self._target = target_groups

    def on_step_end(self, args, _state, control, **_kwargs):
        acc = self._trainer.accelerator
        trainer = self._trainer
        consumed = trainer._groups_before_resume + len(trainer._trained_groups | trainer._dropped_groups)
        reached = torch.tensor(int(consumed >= self._target), device=acc.device)
        if int(acc.reduce(reached, reduction="sum").item()) >= 1:
            control.should_training_stop = True
            # `max_steps` is only a safety ceiling here, so `DefaultFlowCallback` never reaches its end-of-training save.
            # Save the final step the same way, or the steps since the last multiple of `save_steps` are lost.
            if args.save_strategy == SaveStrategy.STEPS:
                control.should_save = True


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
                    **sample.metrics,  # reward_trained, reward_group_std, rewards/<func>/group_std
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


class RolloutQueueDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        rollout_queue,
        model_version_fn,
        check_health_fn,
        stale_after_s,
        metrics,
        dropped_groups,
        token_allowance,
        max_staleness=3,
        poll_interval_s=5.0,
        report_to=None,
    ):
        self.queue = rollout_queue
        self.token_allowance = token_allowance
        self.model_version_fn = model_version_fn
        self.check_health_fn = check_health_fn
        self.stale_after_s = stale_after_s
        # The trainer's metric sink, shared by reference. This dataset lives in the main process (`num_workers=0`,
        # `dispatch_batches=True`), which is also the only process where the queue wait is real, so its metrics need no
        # communication at all — they are appended straight into the sink the trainer reduces in `log()`.
        self.metrics = metrics
        self.dropped_groups = dropped_groups
        # Blocking-get time, accumulated here and flushed by `training_step` at the optimizer-step boundary — the only
        # place that knows when a step's worth of waiting is done.
        self.wait_s = 0.0
        self.consumed_tokens = 0
        self.step_tokens = 0
        self.lookahead_tokens = 0
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

            if isinstance(sample, int):
                self.dropped_groups.add(sample)
                continue
            self.consumed_tokens += len(sample.input_ids)
            self.step_tokens += len(sample.input_ids)
            self.token_allowance.value = self.consumed_tokens + self.lookahead_tokens

            version = self.model_version_fn()
            staleness = version - sample.model_version
            if staleness > self.max_staleness:
                logger.info(f"dropping stale sample (staleness={staleness}, max={self.max_staleness})")
                self.metrics["sample/dropped_stale_count"].append(1.0)
                continue  # drop stale, pull next

            # Three different views of the same queue, and they must not be confused with each other:
            #   `rollout/queue_size`         how many scored samples are in it, counted where the TRAINER consumes
            #                                them rather than where the worker filled it
            #   `sample/time_in_queue_s`     how long THIS sample sat in it — its own idle time, and the seconds
            #                                half of its off-policyness
            #   `perf/rollout_wait_s`        how long the TRAINER sat blocked because the queue was empty
            # An empty queue with the trainer waiting is generation-bound; a queue holding `max_staleness` steps with no
            # wait is trainer-bound (and then `rollout/backpressure_s` is what generation lost to it).
            self.metrics["rollout/queue_size"].append(float(self.queue.qsize()))
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
                "sampling_mask": sample.sampling_mask,
                "sampler_ids": sample.sampler_ids,
                "sampler_log_probs": sample.sampler_log_probs,
                "advantage": sample.advantage,
                "group_id": sample.group_id,
                "metrics": sample.metrics,  # per-sample rewards; aggregated by the collator, never sent to the model
            }


class TokenBudgetBatcher(torch.utils.data.IterableDataset):
    """Token-budgeted dynamic batcher (the planner) wrapping [`RolloutQueueDataset`].

    Keeps `num_processes` open rows (one per DP rank) and pulls single samples from the source one at a time, dropping
    each into the row with the smallest running attention cost that still fits within `token_budget` forwarded tokens.
    When the next sample fits in no row, the current micro-batch is emitted — a list of `num_processes` groups, already
    partitioned per rank — and a fresh one is started with that sample. The number of samples per row is therefore
    dynamic: short samples pack many per row, long ones pack few, while every row stays within `token_budget` tokens.
    This bounds peak memory independently of `per_device_train_batch_size` and keeps the rows cost-balanced so no rank
    straggles at the per-micro-batch all-reduce.

    The budget counts *unique* tokens, so it fits more samples than their raw length suggests: a sample costs only what
    it does not already share with its row. It also bounds the row's loss terms, which packing does not shrink —
    otherwise rows that share everything would add no tokens and the row would never close. A sample's standalone
    length upper-bounds both, so the drop rule below stays correct.

    Every emitted micro-batch has all `num_processes` rows non-empty (a rank forwarding zero tokens would desync
    FSDP/EP collectives): a micro-batch is only closed once every row holds at least one sample. A sample longer than
    `token_budget` fits in no row, so it is dropped with a warning; set `token_budget` >= the vLLM server's
    `max_model_len` (the cap on prompt + completion) to avoid dropping samples.

    Args:
        dataset ([`RolloutQueueDataset`]):
            Source yielding single rollout-sample dicts.
        num_processes (`int`):
            Number of DP ranks; the number of rows (one per rank) in each micro-batch.
        token_budget (`int`):
            Maximum tokens forwarded in a single row (one rank's forward).
        metrics (`dict`):
            The trainer's metric sink, appended to when a sample is dropped for exceeding the budget.
    """

    def __init__(self, dataset: "RolloutQueueDataset", num_processes: int, token_budget: int, metrics: dict):
        self.dataset = dataset
        self.num_processes = num_processes
        self.token_budget = token_budget
        self.metrics = metrics  # the trainer's sink, for the drop counter below

    def __iter__(self):
        rows = [TreeRow() for _ in range(self.num_processes)]
        for sample in self.dataset:
            n = len(sample["input_ids"])
            if n > self.token_budget:
                # Longer than the whole budget: fits in no row, so drop it (placing it would overshoot the budget
                # or force an empty row that desyncs FSDP/EP collectives).
                logger.warning(
                    f"Dropping a rollout sample of {n} tokens that exceeds token_budget={self.token_budget}. "
                    "Raise token_budget to avoid dropping samples."
                )
                self.metrics["batch/dropped_oversize_count"].append(1.0)
                continue
            while True:
                placed_row = self._place(rows, sample)
                if placed_row is not None:
                    placed_row.add(sample)
                    break
                # No more place in rows
                yield rows
                rows = [TreeRow() for _ in range(self.num_processes)]

    def _place(self, rows: list[TreeRow], sample: dict[str, Any]) -> TreeRow | None:
        """Pick the row `sample` belongs in, or `None` when the micro-batch has to close first.

        A sample is cheapest in a row that already holds its `group_id`, so those rows get first refusal — all of them,
        since an earlier split can leave one group on several rows. When none of them has room, the sample goes to the
        least loaded row it fits in. Once every rank is busy, it only does so if that does not raise the most loaded
        row: the micro-batch takes as long as that row, so the split is free, whereas closing would add a micro-batch.
        """

        def least_loaded(candidates: list[TreeRow]) -> TreeRow | None:
            return min(
                (row for row in candidates if row.fits(sample, self.token_budget)),
                key=lambda row: row.load,
                default=None,
            )

        home = [row for row in rows if sample["group_id"] in row.groups]
        placed = least_loaded(home)
        if placed is not None:
            return placed
        placed = least_loaded(rows)
        if placed is not None and home and all(row.samples for row in rows):
            if placed.load + placed.cost(sample) > max(row.load for row in rows):
                return None
        return placed


class _EmptyIterableDataset(torch.utils.data.IterableDataset):
    """Placeholder for non-rank-0 processes. Never actually iterated."""

    def __iter__(self):
        return iter([])


@dataclass
class DataCollatorForRollout(DataCollatorMixin):
    """
    Padding-free collator (the packer) for rollout samples. Packs a micro-batch into `num_processes` rows (one per DP
    rank) and, for each row, the flat list of loss terms taken over it: which packed position predicts which target,
    with what old log-probability, advantage and originating completion. Rows are padded only to the longest row, so
    the batch stays rectangular for `DataLoaderDispatcher` to scatter row `i` -> rank `i`; this inter-rank padding is
    stripped per-rank in `compute_loss`.

    Each row is a [`~packing.TreeRow`], so a token shared by several of its samples is forwarded once.

    The micro-batch arrives already partitioned into `num_processes` rows by the upstream planner
    ([`TokenBudgetBatcher`]) — which balances each row's attention cost to avoid stragglers at the gradient all-reduce
    — so the collator only packs and tensorizes the given rows.

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
    # Distinct prompt-group ids: exactly the prompt-groups that get trained
    groups_trained: set[int] = field(default_factory=set)
    # The trainer's metric sink, shared by reference (like `groups_trained`).
    metrics: dict[str, list] = field(default_factory=lambda: defaultdict(list))
    token_budget: int = 0

    def torch_call(self, examples: list[Any]) -> dict[str, Any]:
        # The dataloader uses batch_size=1 over a planner that pre-partitions each micro-batch into `num_processes`
        # rows, so `examples` is a length-1 list holding that single micro-batch (one row per rank).
        (rows,) = examples

        packed = [row.pack() for row in rows]
        padding = {
            "input_ids": self.pad_token_id,
            "shift_labels": -100,
            "token_index": -1,
            "sampling_mask": -1,
            "sampler_ids": -1,
            "sampler_log_probs": -math.inf,
        }
        batch = {key: pad([p[key] for p in packed], padding_value=padding.get(key, 0)) for key in packed[0]}
        batch["attention_mask"] = pad([torch.ones_like(p["input_ids"]) for p in packed], padding_value=0)

        all_examples = [example for row in rows for example in row.samples]
        self.groups_trained.update(example["group_id"] for example in all_examples)

        # Repeated per rank so that DataLoaderDispatcher (dispatch_batches=True) slices correctly on dim=0
        n_trained_tokens = sum(p["token_index"].numel() for p in packed)
        n_forward_tokens = sum(p["input_ids"].numel() for p in packed)
        batch["global_n_tokens"] = torch.full((self.num_processes,), float(n_trained_tokens))
        batch["global_n_forward_tokens"] = torch.full((self.num_processes,), float(n_forward_tokens))
        batch["mean_seq_len"] = torch.full((self.num_processes,), n_forward_tokens / len(all_examples))

        self._log_metrics(rows=rows, all_examples=all_examples, padded=batch["attention_mask"])
        return batch

    def _log_metrics(self, rows: list[TreeRow], all_examples: list[dict[str, Any]], padded: torch.Tensor) -> None:
        """Append this micro-batch's sample and packing metrics straight into the trainer's sink.

        Rank 0 holds the whole micro-batch, so the per-sample rewards can be aggregated here instead of being packed
        into NaN-padded tensors, broadcast to every rank and reduced back — which is what this trainer used to do to
        compute a number rank 0 already had.
        """
        # A training row's tokens split in two: those the loss is taken over (`completion_mask == 1`) and those merely
        # forwarded — the prompt, tool results, masked tokens, and the earlier turns a fork re-sends as context. FLOPs
        # are spent on both, so both are worth a number, and their ratio is what says how much of the forward earns no
        # gradient.
        forwarded = [len(example["input_ids"]) for example in all_examples]
        trained = [int(example["completion_mask"].sum()) for example in all_examples]
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

        row_tokens = [len(row.tokens) for row in rows]
        loads = [row.load for row in rows]
        self.metrics["batch/samples_per_row"].append(len(all_examples) / len(rows))
        self.metrics["batch/row_tokens_mean"].append(sum(row_tokens) / len(row_tokens))
        self.metrics["batch/row_tokens_max"].append(float(max(row_tokens)))
        self.metrics["batch/row_imbalance"].append(max(loads) / (sum(loads) / len(loads)))
        self.metrics["batch/packing_ratio"].append((sum(forwarded), sum(row_tokens)))
        if self.token_budget:
            self.metrics["batch/row_fill_frac"].append((sum(row_tokens), self.token_budget * len(row_tokens)))
        # Inter-rank padding, added so the batch is rectangular for the dispatcher. It costs broadcast bytes only:
        # `compute_loss` strips it before the forward, so no FLOPs are spent on it.
        self.metrics["batch/pad_frac"].append((padded.numel() - int(padded.sum()), padded.numel()))


def _iter_vllm_named_params(model: torch.nn.Module) -> Iterator[tuple[str, torch.nn.Parameter]]:
    """Yield `(name, param)` for every parameter the vLLM server receives, under its base-checkpoint name.

    The weight manifest built once in `__init__` and the stream sent by `_streaming_iter` must agree
    element-for-element — the server sizes its receive buffers from the manifest — so both walk this generator rather
    than each re-deriving the name cleanup. Parameters are yielded as-is: `DTensor.shape` is the global shape, so
    building the manifest costs no all-gather.
    """
    is_peft = is_peft_model(model)
    for name, param in model.named_parameters():
        # Frozen parameters (a VLM's vision tower) never change, so they are never sent. Under PEFT the filter is
        # inverted: the trainable parameters are the adapter tensors vLLM never sees, and it is the frozen base
        # weights that move once the adapter has been merged into them.
        if not is_peft and not param.requires_grad:
            continue
        # Strip the prefixes DDP/FSDP1 and gradient checkpointing add, so the names match vLLM's
        name = name.removeprefix("module.").replace("_checkpoint_wrapped_module.", "")
        # When using PEFT, we need to recover the original parameter name
        name = name.removeprefix("base_model.model.").replace(".base_layer", "")
        # Skip PEFT layers: they don't exist in vLLM, and they are merged already.
        if is_peft and model.prefix in name:
            continue
        # When module to save, remove its prefix and discard the original module
        # and the other adapters' copies (e.g. a "ref" adapter)
        if "original_module" in name or ("modules_to_save." in name and "modules_to_save.default." not in name):
            continue
        name = name.replace("modules_to_save.default.", "")
        yield name, param


# Ranks vLLM builds stacked LoRA buffers for. Any other value crashes engine construction.
VLLM_LORA_RANKS = (1, 8, 16, 32, 64, 128, 256, 320, 512)


def round_lora_rank(adapter_rank: int) -> int:
    """
    Round a LoRA rank up to the smallest `max_lora_rank` vLLM accepts.

    `max_lora_rank` is a capacity bound rather than the served rank, so an `r=4` adapter is served correctly under `8`.

    Args:
        adapter_rank (`int`):
            Rank of the adapter to serve.

    Returns:
        `int`:
            Smallest value in `VLLM_LORA_RANKS` greater than or equal to `adapter_rank`.

    Examples:

    ```python
    >>> round_lora_rank(4)
    8
    ```
    """
    for rank in VLLM_LORA_RANKS:
        if rank >= adapter_rank:
            return rank
    raise ValueError(
        f"LoRA rank {adapter_rank} is larger than the largest rank vLLM can serve ({VLLM_LORA_RANKS[-1]}). Lower `r` "
        f"(and `rank_pattern`) to sync the adapter to vLLM, or sync merged weights instead."
    )


def validate_lora_for_vllm_sync(model: "PeftModel") -> "LoraConfig":
    """
    Return the active adapter's config, checking it can be served as a plain vLLM LoRA adapter.

    vLLM's LoRA manager only knows about `lora_A` / `lora_B` deltas on a fixed set of linear layers. Anything else a
    PEFT config can express — fully-trained modules, DoRA magnitudes, trained biases, fused-MoE parameter LoRA — has no
    home in the adapter checkpoint vLLM loads, and is silently ignored (or worse, misread) at serving time. Every such
    config is rejected here so the caller can fall back to syncing merged weights, which handles all of them.

    Args:
        model ([`~peft.PeftModel`]):
            PEFT model whose active adapter is checked.

    Returns:
        [`~peft.LoraConfig`]:
            Config of the single active adapter.
    """
    if len(model.active_adapters) != 1:
        raise ValueError(
            f"Adapter-only vLLM sync serves a single adapter, but {len(model.active_adapters)} are active "
            f"({model.active_adapters}). Activate one adapter, or sync merged weights instead."
        )
    peft_config = model.peft_config[model.active_adapters[0]]
    if not isinstance(peft_config, LoraConfig):
        raise ValueError(
            f"Adapter-only vLLM sync supports LoRA adapters only, got `{type(peft_config).__name__}`. Sync merged "
            f"weights instead."
        )
    if peft_config.modules_to_save:
        raise ValueError(
            f"`modules_to_save={peft_config.modules_to_save}` cannot ride in a LoRA adapter: those modules are "
            f"fully trained, not a low-rank delta, so vLLM would keep serving the base checkpoint's copy. Sync "
            f"merged weights instead."
        )
    if peft_config.use_dora:
        raise ValueError(
            "DoRA adapters are not servable by vLLM: the magnitude vector has no slot in the adapter format. Sync "
            "merged weights instead."
        )
    if peft_config.bias != "none":
        raise ValueError(
            f"`bias='{peft_config.bias}'` trains bias terms, which are not part of a LoRA adapter and would be "
            f"dropped on the way to vLLM. Use `bias='none'`, or sync merged weights instead."
        )
    if peft_config.target_parameters:
        raise ValueError(
            f"`target_parameters={peft_config.target_parameters}` (fused-MoE parameter LoRA) needs the adapter to be "
            f"declared with `is_3d_lora_weight=True` against a server started with "
            f"`--enable-mixed-moe-lora-format`. vLLM does not inspect the checkpoint, so a wrong declaration loads "
            f"the weights into the wrong stacked buffers and silently produces garbage. Sync merged weights instead."
        )
    target_modules = peft_config.target_modules or []
    if isinstance(target_modules, str):  # a regex matching module names, rather than a collection of them
        target_modules = [target_modules]
    if any("lm_head" in module or "embed_tokens" in module for module in target_modules):
        raise ValueError(
            f"`target_modules={peft_config.target_modules}` targets the head or the embeddings. vLLM restricts LoRA "
            f"on those layers and drops the head delta from `prompt_logprobs` entirely "
            f"(https://github.com/vllm-project/vllm/issues/51594), so the trainer and the server would disagree on "
            f"every logprob. Sync merged weights instead."
        )
    return peft_config


def save_lora_adapter(model: "PeftModel", accelerator: Accelerator, adapter_name: str, dest_dir: str) -> None:
    """
    Save one PEFT adapter to `dest_dir`, gathering sharded parameters as needed.

    This is a collective: every process must call it, not only the main one, because materializing a sharded parameter
    all-gathers across ranks.

    `dest_dir` is named after the adapter and holds `adapter_config.json` at its root, the layout vLLM's filesystem
    LoRA resolver expects when it serves `<cache_dir>/<adapter_name>`. It is published with an atomic rename, so a
    server reading it over a shared filesystem never observes a half-flushed adapter.

    Args:
        model ([`~peft.PeftModel`]):
            PEFT model holding the adapter.
        accelerator ([`~accelerate.Accelerator`]):
            Accelerator whose device sharded parameters are gathered on.
        adapter_name (`str`):
            Name of the adapter to save.
        dest_dir (`str`):
            Directory the adapter is published to. Replaced if it already exists.
    """
    state_dict = {
        name: param for name, param in model.state_dict().items() if "lora_" in name and f".{adapter_name}." in name
    }
    materialized = {}
    for name, param in state_dict.items():
        # With `fsdp_offload_params`, the local shard lives on CPU; the all-gather needs it on device.
        if param.is_cpu:
            param = param.to(accelerator.device)
        if isinstance(param, DTensor):  # FSDP2
            param = param.full_tensor()
        materialized[name] = param.detach().cpu()

    if not accelerator.is_main_process:
        return
    # Staged under a dot-prefixed name: `dest_dir`'s parent is an adapter namespace, so a plain `<name>.tmp` sibling
    # would be a half-written adapter sitting in it, and a crash mid-save would leak one there permanently.
    tmp_dir = os.path.join(os.path.dirname(dest_dir), f".{os.path.basename(dest_dir)}.tmp")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)
    model.save_pretrained(tmp_dir, selected_adapters=[adapter_name], state_dict=materialized, safe_serialization=True)
    # PEFT nests every adapter but `"default"` in a subdirectory, while vLLM expects `adapter_config.json` at the root
    # of the path it is handed.
    saved_dir = tmp_dir if adapter_name == "default" else os.path.join(tmp_dir, adapter_name)
    shutil.rmtree(dest_dir, ignore_errors=True)
    os.rename(saved_dir, dest_dir)  # atomic within one filesystem
    shutil.rmtree(tmp_dir, ignore_errors=True)


def select_adapter_sync(server_info: dict, model: "PeftModel", args: AsyncGRPOConfig) -> bool:
    """
    Decide whether weight syncs push the adapter alone instead of merged base weights.

    A pure function of what the server reported and what is about to be trained, so it can be tested with the dict
    `/server_info` returns rather than with a server. Anything vLLM's adapter format cannot express falls back to the
    merged path with a warning. A server that is merely misconfigured — too low a rank, too few adapter slots — raises
    instead, rather than silently syncing 100x the bytes its launch line asked for.

    Args:
        server_info (`dict`):
            The server's resolved `vllm_config`, as returned by
            [`~trl.generation.async_vllm_client.AsyncVLLMClient.get_server_info`]. Only `lora_config` and
            `parallel_config.data_parallel_size` are read.
        model ([`~peft.PeftModel`]):
            The PEFT-wrapped model about to be trained.
        args ([`AsyncGRPOConfig`]):
            The training arguments; `max_staleness` sets how many adapter versions the server must hold at once.

    Returns:
        `bool`:
            Whether syncs should push only the adapter.
    """
    # `lora_config` is `None` unless the server was started with `--enable-lora`, making it the capability probe.
    server_lora_config = server_info["lora_config"]
    if server_lora_config is None:
        logger.warning(
            "Training a PEFT model against a vLLM server started without `--enable-lora`: every weight sync has "
            "to merge the adapter into the base model and push the full weights. Restart the server with "
            "`--enable-lora --max-lora-rank <r>` and `VLLM_ALLOW_RUNTIME_LORA_UPDATING=1` in its environment to "
            "push only the adapter instead (~1% of the bytes)."
        )
        return False

    # `/v1/load_lora_adapter` reaches only the data-parallel replica that answered it, leaving the others on
    # the base model. The merged path has no such problem, so hand data-parallel servers to it.
    if server_info["parallel_config"]["data_parallel_size"] > 1:
        logger.warning(
            "Falling back to merged-weight vLLM sync: the server runs with `--data-parallel-size > 1`, where a "
            "loaded adapter reaches only the replica that received the request."
        )
        return False

    try:
        peft_config = validate_lora_for_vllm_sync(model)
    except ValueError as error:
        logger.warning(f"Falling back to merged-weight vLLM sync. {error}")
        return False

    # `rank_pattern` can lift individual modules above the base `r`.
    adapter_rank = max([peft_config.r, *peft_config.rank_pattern.values()])
    if server_lora_config["max_lora_rank"] < adapter_rank:
        raise ValueError(
            f"The adapter has rank {adapter_rank}, but the vLLM server was started with `--max-lora-rank "
            f"{server_lora_config['max_lora_rank']}`. Restart it with `--max-lora-rank "
            f"{round_lora_rank(adapter_rank)}`."
        )
    # `max_staleness` bounds how many policies back a sample may still be consumed from, so `max_staleness + 1`
    # adapters can be named by in-flight requests at once. Each sync loads the new version before unloading the one
    # that just fell out of that window, so the server briefly holds one more — and vLLM does not reject the extra
    # load, it silently evicts the oldest still-servable adapter.
    min_loras = args.max_staleness + 2
    if server_lora_config["max_loras"] < min_loras:
        raise ValueError(
            f"The vLLM server was started with `--max-loras {server_lora_config['max_loras']}`, but "
            f"`max_staleness={args.max_staleness}` lets requests still name {min_loras - 1} distinct policy "
            f"versions, and each sync loads the next one before unloading the oldest. Restart the server with "
            f"`--max-loras {min_loras}`, or lower `max_staleness`."
        )
    return True


class AsyncGRPOTrainer(_BaseTrainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language
    Models](https://huggingface.co/papers/2402.03300). This trainer is the asynchronous version of GRPO, where
    generation is offloaded to an external vLLM server that runs asynchronously alongside training, decoupling rollout
    from the gradient update loop.

    Example:

    ```python
    >>> from trl import AsyncGRPOTrainer
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
        model (`str` or [`~transformers.PreTrainedModel`]):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or a
              path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
              with the architecture declared in its config: [`~transformers.AutoModelForImageTextToText`] for a
              vision-language checkpoint (whose vision tower is loaded but frozen, see [Training Vision Language
              Models](async_grpo_trainer#training-vision-language-models)), [`~transformers.AutoModelForCausalLM`]
              otherwise.
            - A [`~transformers.PreTrainedModel`] object. Its attention implementation is switched to the trainer's
              tree attention backend for packed training.

            The model name from the config is used to identify the model on the vLLM server used for generation.
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
            `functools.partial`, or a callable class instance — lambdas and closures will fail at startup. Tools are
            registered under their `__name__`, so they must be module-level functions. The child process also runs with
            `CUDA_VISIBLE_DEVICES=""`, so a GPU-backed reward model runs on CPU (slow), not the trainer's GPU.
        args ([`AsyncGRPOConfig`], *optional*):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`], *optional*):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset are
            ignored. The samples must be in [conversational](dataset_formats#conversational) format: each sample
            contains structured messages (e.g., role and content).

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
        peft_config ([`~peft.PeftConfig`], *optional*):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped. When the vLLM server was
            started with `--enable-lora`, a LoRA adapter is synced to it as an adapter rather than as merged weights;
            see the LoRA section of the AsyncGRPO documentation.
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
    [`~trainer.async_grpo_trainer.AsyncRolloutLoop`] and set it as the trainer's `loop_cls`. TRL builds the loop in the
    rollout process:

    ```python
    >>> class CurriculumTrainer(AsyncGRPOTrainer):
    ...     loop_cls = CurriculumRolloutLoop
    ```
    """

    loop_cls: type[AsyncRolloutLoop] = AsyncRolloutLoop
    _rollout_loop_kwargs: dict[str, Any] = {}
    loss_is_scaled_for_ga = True

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
        model: "str | PreTrainedModel",
        reward_funcs: RewardFunc | list[RewardFunc] | None = None,
        args: AsyncGRPOConfig | None = None,
        train_dataset: Dataset | IterableDataset | None = None,
        processing_class: PreTrainedTokenizerBase | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None] = (None, None),
        peft_config: "PeftConfig | None" = None,
        tools: list[Callable] | None = None,
        environment_factory: EnvironmentFactory | dict[str, EnvironmentFactory] | None = None,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else get_config_model_id(model.config)
            model_name = model_name.split("/")[-1]
            args = AsyncGRPOConfig(f"{model_name}-AsyncGRPO")

        # Training arguments
        self.epsilon_low = args.epsilon
        self.epsilon_high = args.epsilon_high if args.epsilon_high is not None else args.epsilon
        self.temperature = args.temperature

        register_tree_attention()

        # Model
        # PEFT initializes the adapter weights randomly, so set_seed must be done before creating the model to ensure
        # reproducibility.
        set_seed(args.seed)
        if isinstance(model, str):
            model_init_kwargs = dict(args.model_init_kwargs or {})  # copy to avoid mutating model_init_kwargs
            model_init_kwargs.setdefault("trust_remote_code", args.trust_remote_code)
            model_init_kwargs.setdefault("dtype", args.dtype)
            model_revision = model_init_kwargs.get("revision")
            model = create_model_from_path(
                model,
                device_map=None,
                attn_implementation=TREE_ATTENTION,
                **model_init_kwargs,
            )
        else:
            model_revision = None
            model.set_attn_implementation(TREE_ATTENTION)
            if args.model_init_kwargs is not None:
                logger.warning(
                    "You passed `model_init_kwargs` to the `AsyncGRPOConfig`, but your model is already instantiated. "
                    "The `model_init_kwargs` will be ignored."
                )

        if args.use_liger_kernel:
            raise NotImplementedError("`use_liger_kernel` is not supported yet.")

        text_config = model.config.get_text_config()
        layer_types = set(getattr(text_config, "layer_types", None) or []) - {"full_attention", "sliding_attention"}
        # Check tree attention compat
        if layer_types:
            if layer_types == {"linear_attention"} and Version(transformers.__version__) < Version("5.15.0"):
                raise ValueError(
                    f"Tree packing of {type(model).__name__}'s gated DeltaNet layers requires transformers>=5.15.0: "
                    "before it, these layers call the functions they stored at init rather than the module-level "
                    "ones tree packing patches, so they would scan across samples."
                )
            if layer_types != {"linear_attention"} or not patch_tree_gdn(model):
                raise ValueError(
                    f"{type(model).__name__} has {sorted(layer_types)} layers. Tree packing routes attention and "
                    "transformers' gated DeltaNet layers along each sample's own path; any other recurrent or "
                    "convolutional layer would scan across samples."
                )
        if any(name.endswith(".sinks") for name, _ in model.named_parameters()):
            raise ValueError(
                f"{type(model).__name__} uses attention sinks, which the FA3 kernel used by tree packing does not "
                "support."
            )

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

        add_fused_lm_head(
            model.get_base_model() if is_peft_model(model) else model,
            temperature=self.temperature,
            outputs=("log_probs", "entropy"),
        )

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

        # PEFT. Placed after `add_fused_lm_head`, which reads the bare `lm_head` and would otherwise have to
        # traverse `base_model.model` to find it.
        if peft_config is not None:
            if not is_peft_available():
                raise ImportError(
                    "You passed `peft_config` but the `peft` library is not installed. "
                    "Install it with `pip install trl[peft]`."
                )
            if not isinstance(peft_config, PeftConfig):
                raise TypeError(
                    f"`peft_config` must be a `peft.PeftConfig` instance (e.g. `peft.LoraConfig`), "
                    f"got {type(peft_config).__name__}."
                )
            if is_peft_model(model):
                raise ValueError(
                    "You passed a `PeftModel` instance together with a `peft_config` to the trainer. Please first merge "
                    "and unload the existing adapter, save the resulting base model, and then pass that base model along "
                    "with the new `peft_config` to the trainer."
                )
            # Unlike `GRPOTrainer`, no `autocast_adapter_dtype=False` branch (it works around a DeepSpeed ZeRO-3
            # dtype mismatch, and AsyncGRPO is FSDP2-only) and no "ref" adapter (there is no reference model).
            model = get_peft_model(model, peft_config)

        # Prompt-learning methods (PromptTuning, PrefixTuning, P-Tuning) add virtual tokens rather than weights, and
        # the trainer syncs weights to the vLLM server, so the server would generate without them.
        if is_peft_model(model) and any(isinstance(cfg, PromptLearningConfig) for cfg in model.peft_config.values()):
            raise ValueError(
                "Prompt-learning PEFT methods (PromptTuning, PrefixTuning, P-Tuning) are not supported. The trainer "
                "syncs weights to the vLLM server, and these methods add virtual tokens instead, so the server would "
                "generate without them. Use a weight-based adapter such as LoRA instead."
            )

        # `add_fused_lm_head` computes logits from `lm_head.weight` directly. On a PEFT-wrapped head that is the
        # base layer's weight, so the adapter delta is never applied: the trainer scores a policy that does not exist
        # while the server serves the real one, and the importance ratio is wrong on every token with nothing raised. Checked on
        # the module rather than on `target_modules`, so a regex that happens to match the head is caught too.
        # `SFTTrainer` refuses the same configuration.
        if is_peft_model(model):
            from peft.tuners.tuners_utils import BaseTunerLayer

            if isinstance(model.get_output_embeddings(), BaseTunerLayer):
                raise ValueError(
                    "`AsyncGRPOTrainer` does not support a PEFT adapter on `lm_head`: its chunked log-probability "
                    "computation reads the head's base weight and would silently ignore the adapter. Remove `lm_head` "
                    "from `target_modules`."
                )

        # When using gradient checkpointing with PEFT, we need to enable input gradients. transformers.Trainer handles
        # this from 5.0.0 (https://github.com/huggingface/transformers/pull/42558)
        if (
            is_peft_model(model)
            and args.gradient_checkpointing
            and Version(transformers.__version__) < Version("5.0.0")
        ):
            model.enable_input_require_grads()

        if tools or environment_factory:
            # jmespath is only needed by the legacy `response_schema` parser, which is all transformers < 5.13 ships.
            # The new-style `response_template` parser doesn't use it, so don't require it on newer versions.
            if not _SUPPORTS_RESPONSE_TEMPLATE and not is_jmespath_available():
                raise ImportError(
                    "Using tools with AsyncGRPOTrainer on transformers below 5.13.0 requires the jmespath library for "
                    "response parsing. Please install it with `pip install jmespath`, or upgrade transformers to "
                    "5.13.0 or higher, which doesn't need it."
                )
            if not supports_tool_calling(processing_class):
                raise ValueError(
                    "The provided chat template does not support tool calling. The template must be able to render a "
                    "full tool-calling conversation (user -> assistant with tool_calls -> tool)."
                )

        # Reward functions
        if reward_funcs is None:
            reward_funcs = []
        elif not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]

        # At least one reward source is required: either `reward_funcs`, an environment that owns the reward via a
        # `get_reward` method, or a `loop_cls` whose `generate_one` sets `Rollout.reward`. Environments are created in the
        # rollout worker, which checks that they define `get_reward`.
        if (
            not reward_funcs
            and environment_factory is None
            and self.loop_cls.generate_one is AsyncRolloutLoop.generate_one
        ):
            raise ValueError(
                "No reward source provided. Pass `reward_funcs`, an `environment_factory` whose environment defines a "
                "`get_reward` method, or a `loop_cls` whose `generate_one` sets `Rollout.reward`."
            )

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
            if args.max_steps <= 0:
                raise ValueError(
                    "When training without a `train_dataset` (the environment owns the data and returns the prompt "
                    "from `reset()`), `max_steps` must be set to a positive value to define the training length. Set "
                    "it via `AsyncGRPOConfig(max_steps=...)`."
                )
            num_placeholder_rows = args.per_device_train_batch_size * args.gradient_accumulation_steps
            train_dataset = Dataset.from_dict({"prompt": [[{"role": "user", "content": ""}]] * num_placeholder_rows})
        elif not isinstance(train_dataset, (Dataset, IterableDataset)):
            raise TypeError(
                f"`train_dataset` must be a `Dataset` or `IterableDataset`, got `{type(train_dataset).__name__}`."
            )

        # Vision dataset detection
        dataset_sample = next(iter(train_dataset))
        if "image" in dataset_sample or "images" in dataset_sample:
            raise ValueError(
                "The dataset appears to be vision-related (contains 'image' or 'images' keys), but `AsyncGRPOTrainer` "
                "trains on text only, vision-language models included. Please remove the images from your dataset."
            )

        # Initialize the Trainer
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Add tags to the model
        self.model.add_model_tags(self._tag_names)

        precision = self.accelerator.mixed_precision
        self.compute_dtype = {
            "bf16": "bfloat16",
            "fp16": "float16",
            "no": str(self.model.dtype).removeprefix("torch."),
        }.get(precision, precision)
        self._peak_flops_per_device = get_peak_flops_per_device(self.accelerator, self.compute_dtype)

        # Epoch handling: stop after num_train_epochs full passes over the PROMPT dataset, counted as distinct
        # prompt-groups trained (fork-independent).
        self._trained_groups: set[int] = set()
        self._dropped_groups: set[int] = set()
        # Tracks restart to match `num_train_epochs`
        self._groups_before_resume = 0
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

        # The base Trainer's data-skip replay loop doesn't apply to AsyncGRPO's live rollout queue;
        # force it off regardless of what the user passed.
        if not self.args.ignore_data_skip:
            logger.warning(
                "`ignore_data_skip` is forced to `True` for AsyncGRPO because the base Trainer's skip-and-replay "
                "loop does not apply to a live rollout queue."
            )
        self.args.ignore_data_skip = True

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
        # Adapter-only vLLM sync is derived from a PEFT model plus a server started with `--enable-lora`, rather
        # than configured. Rank 0 probes and broadcasts the answer below.
        self._lora_sync = False
        self._lora_name = "trl-policy"
        # Absolute: the path is resolved by the *server's* process, which has its own working directory and may not
        # be on this machine. vLLM reads a path it cannot resolve as a Hub repo id, failing deep inside the engine.
        self._lora_dir = os.path.abspath(os.path.join(self.args.output_dir, ".vllm_lora"))
        # Captured once, so a trainer that later activates another adapter does not ship that one as the policy.
        self._adapter_name = model.active_adapters[0] if is_peft_model(model) else None
        # Weight sync and the token-budget query target the vLLM server from the config; the client is the single
        # place that talks to it, independent of how rollouts are produced.
        self.vllm_client = (
            AsyncVLLMClient(self.args.vllm_server_base_url, self.args.vllm_server_timeout)
            if self.accelerator.is_main_process
            else None
        )
        # Rank 0 probes the server for the sync mode, which every rank must agree on: one arm runs a collective
        # adapter save, the other a collective parameter gather, and a split decision hangs both. A failed probe is
        # broadcast in its place and raised on every rank before rank 0 builds anything else, since raising on rank 0
        # alone would leave the others waiting in this collective.
        if is_peft_model(model):
            lora_sync = None
            if self.accelerator.is_main_process:
                try:
                    lora_sync = self._init_lora_sync(model)
                except Exception as error:
                    lora_sync = error
            lora_sync = broadcast_object_list([lora_sync], from_process=0)[0]
            if isinstance(lora_sync, Exception):
                raise lora_sync
            self._lora_sync = lora_sync
        # Create worker and queue on rank 0
        if self.accelerator.is_main_process:
            if self._lora_sync:
                # The adapter reaches the server as a directory path over HTTP, so there is no NCCL transfer group to
                # build and no manifest to collect.
                self.weight_transfer = None
            else:
                # Collect weight metadata once — names/dtypes/shapes are fixed for the lifetime of training.
                # DTensor.shape returns the global shape without triggering any all-gather.
                weight_names, weight_dtype_names, weight_shapes = [], [], []
                for name, param in _iter_vllm_named_params(model):
                    weight_names.append(name)
                    weight_dtype_names.append(str(param.dtype).split(".")[-1])
                    weight_shapes.append(list(param.shape))
                self.weight_transfer = WeightTransferClient(
                    vllm_client=self.vllm_client,
                    weight_update_info={
                        "names": weight_names,
                        "dtype_names": weight_dtype_names,
                        "shapes": weight_shapes,
                    },
                    compute_dtype="bfloat16" if self.args.bf16 else "float16" if self.args.fp16 else None,
                )

            self.rollout_worker = AsyncRolloutWorker(
                self.args,
                loop_cls=self.loop_cls,
                model_name=get_config_model_id(model.config),
                lora_name=self._lora_name if self._lora_sync else None,
                dataset=train_dataset,
                reward_funcs=reward_funcs,
                processing_class=processing_class,
                eos_token_ids=self.eos_token_ids,
                tools=tools,
                environment_factory=environment_factory,
                **self._rollout_loop_kwargs,
            )
            self.rollout_queue = self.rollout_worker.rollout_buffer
        else:
            self.rollout_queue = None
            self.rollout_worker = None
            self.weight_transfer = None

        # Add callbacks. Cold weight sync + worker start on train begin, then periodic weight syncs.
        self.add_callback(_OptimizerTimeCallback(self))
        self.add_callback(_TrainBeginCallback(self))
        self.add_callback(StepIntervalCallback(self._sync_weight, self.args.weight_sync_steps))
        self.add_callback(StepIntervalCallback(self._log_step_metrics, 1))
        if self._epoch_stop_groups is not None:
            self.add_callback(_EpochStopCallback(self, self._epoch_stop_groups))

    def _init_lora_sync(self, model: "PeftModel") -> bool:
        """Probe the server and decide the sync mode. Main process only; the decision itself is [`select_adapter_sync`]."""
        self.vllm_client.wait_for_server_ready()
        lora_sync = select_adapter_sync(self.vllm_client.get_server_info(), model, self.args)
        if lora_sync:
            logger.info(
                f"Adapter-only vLLM sync enabled: syncs publish the '{self._adapter_name}' adapter as "
                f"'{self._lora_name}-v{{N}}' from {self._lora_dir}, which the vLLM server must be able to read."
            )
        return lora_sync

    def get_train_dataloader(self) -> DataLoader:
        num_processes = self.accelerator.num_processes
        if self.accelerator.is_main_process:
            dataset = RolloutQueueDataset(
                rollout_queue=self.rollout_queue,
                model_version_fn=lambda: self.model_version,
                check_health_fn=self.rollout_worker.check_health,
                stale_after_s=self.args.rollout_worker_timeout,
                metrics=self._metrics["train"],
                dropped_groups=self._dropped_groups,
                token_allowance=self.rollout_worker.token_allowance,
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
            self._rollout_dataset.lookahead_tokens = (
                max(self.args.max_staleness, 1)
                * self.args.gradient_accumulation_steps
                * num_processes
                * self.args.token_budget
            )
            self.rollout_worker.token_allowance.value = self._rollout_dataset.lookahead_tokens
            dataset = TokenBudgetBatcher(dataset, num_processes, self.args.token_budget, self._metrics["train"])
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
                    token_budget=self.args.token_budget or 0,
                ),
                num_workers=0,
            )
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        mask_bool = inputs["attention_mask"].bool()
        selected = inputs["token_index"] >= 0
        input_ids = inputs["input_ids"][mask_bool].unsqueeze(0)
        position_ids = inputs["position_ids"][mask_bool].unsqueeze(0)
        shift_labels = inputs["shift_labels"][mask_bool].unsqueeze(0)
        token_index = inputs["token_index"][selected]
        old_log_probs = inputs["old_log_probs"][selected]
        advantages = inputs["advantages"][selected]
        segment_id = inputs["segment_id"][selected]
        counts = inputs["sampling_mask_counts"][mask_bool]
        width = int(counts.max())
        sampling_mask = None
        if width:
            candidates = inputs["sampling_mask"][inputs["sampling_mask"] >= 0]
            row = torch.repeat_interleave(torch.arange(len(counts), device=counts.device), counts)
            col = torch.arange(len(candidates), device=counts.device) - torch.repeat_interleave(
                counts.cumsum(0) - counts, counts
            )
            sampling_mask = candidates.new_full((len(counts), width), -1)
            sampling_mask[row, col] = candidates
            sampling_mask = sampling_mask.unsqueeze(0)
        sampler_ids = inputs["sampler_ids"][mask_bool].unsqueeze(0) if self.args.score_centering else None

        forward_start = time.time()
        # MoE models: request router logits so the forward returns the load-balancing loss
        router_kwargs = {"output_router_logits": True} if self.aux_loss_enabled else {}
        outputs = model(
            input_ids=input_ids,
            position_ids=position_ids,
            shift_labels=shift_labels,
            support_ids=sampling_mask,
            score_ids=sampler_ids,
            fused_lm_head=True,
            **router_kwargs,
            **unpad_tree(inputs),
        )
        log_probs = outputs.log_probs.flatten()[token_index]
        entropy = outputs.entropy.flatten()[token_index]
        self._last_forward_time_s = time.time() - forward_start

        log_ratio = log_probs - old_log_probs
        coef_1 = torch.exp(log_ratio)
        low, high = self.args.vllm_importance_sampling_clip_min, self.args.vllm_importance_sampling_clip_max
        is_masked = (coef_1 < low) | (coef_1 > high)
        if self.args.score_centering:
            # Score centering: https://huggingface.co/papers/2609.20807, Appendix A.2
            score_log_probs = outputs.score_log_probs[0][token_index]
            sampler_log_probs = inputs["sampler_log_probs"][mask_bool][token_index]
            with torch.no_grad():
                q = sampler_log_probs.exp()
                p = score_log_probs.exp() * (q > 0)
                head = torch.where((p >= low * q) & (p <= high * q), p, 0.0)
                tail_ratio = (1 - p.sum(-1)).clamp(min=1e-6) / (1 - q.sum(-1)).clamp(min=1e-6)
                tail = ((tail_ratio >= low) & (tail_ratio <= high)).float()
                residual = head - tail.unsqueeze(-1) * p
                weight = torch.where(is_masked, 0.0, coef_1)
            correction = (residual * score_log_probs).sum(-1)
            per_token_loss = -advantages * (weight * log_probs - correction)
        else:
            coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
            per_token_loss1 = coef_1 * advantages
            per_token_loss2 = coef_2 * advantages
            per_token_loss = -torch.min(per_token_loss1, per_token_loss2) * ~is_masked

        # DDP/FSDP averages gradients across ranks (world_size). To get correct per-token normalization we scale by
        # 1/tokens_per_rank = world_size / global_n_tokens, so after DDP averaging the effective scale is 1/global_n_tokens.
        loss = per_token_loss.sum()
        global_n_tokens = inputs["global_n_tokens"][0]
        world_size = self.accelerator.num_processes
        tokens_per_rank = (global_n_tokens / world_size).clamp(min=1.0)
        loss = loss / tokens_per_rank.to(torch.float32)
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
            local_sampler_mass_sum = q.sum() if self.args.score_centering else torch.zeros_like(local_count)

            # Batch all-reduce: [ratio_sum, logp_diff_sum, entropy_sum, low_clip_sum, high_clip_sum, region_clip_sum,
            # masked_sum, sampler_mass_sum, count]
            stats = torch.stack(
                [
                    local_ratio_sum,
                    local_logp_diff_sum,
                    local_entropy_sum,
                    local_low_clip_sum,
                    local_high_clip_sum,
                    local_region_clip_sum,
                    local_masked_sum,
                    local_sampler_mass_sum,
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
                global_sampler_mass_sum,
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
            if self.args.score_centering:
                self._metrics["train"]["sampling/sampler_mass"].append((global_sampler_mass_sum / global_count).item())

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
        self.state.num_input_tokens_seen += int(self._step_forward_tokens)
        metrics["num_tokens"] = [self.state.num_input_tokens_seen]

        if self.accelerator.is_main_process:
            metrics["perf/rollout_wait_s"].append(self._rollout_dataset.wait_s)
            self._rollout_dataset.wait_s = 0.0
            if self._rollout_dataset.step_tokens:
                self._rollout_dataset.lookahead_tokens = (
                    max(self.args.max_staleness, 1) * self._rollout_dataset.step_tokens
                )
                self._rollout_dataset.token_allowance.value = (
                    self._rollout_dataset.consumed_tokens + self._rollout_dataset.lookahead_tokens
                )
            self._rollout_dataset.step_tokens = 0
            metrics["batch/groups_per_step"].append(float(len(self._trained_groups) - self._last_groups_trained))
            self._last_groups_trained = len(self._trained_groups)

        # Throughput and MFU are reported on TWO bases, because for async RL one number cannot answer both questions.
        ## `fwd_bwd_` divides by `perf/fwd_bwd_s`: the compute alone.
        ## `wall_clock_` divides by `perf/step_s`: the whole step, rollout waits included  and says what fraction of the allocation actually became training.
        if self._step_forward_tokens > 0:
            mean_seq_len = self._step_seq_len_weighted / self._step_forward_tokens
            flops_per_token = compute_flops_per_token(self.model.config.get_text_config(), int(mean_seq_len))
            world_size = self.accelerator.num_processes
            metrics["perf/fwd_bwd_forwarded_tokens_per_s"].append((self._step_forward_tokens, fwd_bwd_s))
            if self._peak_flops_per_device is not None and flops_per_token is not None:
                metrics["perf/fwd_bwd_mfu"].append(
                    compute_mfu(
                        flops_per_token, self._step_forward_tokens / fwd_bwd_s, world_size, self._peak_flops_per_device
                    )
                )
            if step_s is not None:
                metrics["perf/wall_clock_forwarded_tokens_per_s"].append((self._step_forward_tokens, step_s))
                metrics["perf/wall_clock_trained_tokens_per_s"].append((self._step_trained_tokens, step_s))
                if self._peak_flops_per_device is not None and flops_per_token is not None:
                    metrics["perf/wall_clock_mfu"].append(
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
        if self.accelerator.is_main_process and self.rollout_worker:
            while True:
                try:
                    for key, value in self.rollout_worker.metrics_queue.get_nowait().items():
                        # The worker only generates training rollouts, so its metrics always go to the train metrics
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
        for name, param in _iter_vllm_named_params(self.accelerator.unwrap_model(self.model)):
            full = param.full_tensor() if isinstance(param, DTensor) else param.detach()
            if full.device != device:
                full = full.to(device)
            yield name, full

    def _sync_weight(self):
        """Publish the current policy to vLLM, then bump the version the rollout worker requests.

        Dispatches to the sync mode picked at init, logging `perf/weight_sync_*` for both here so they stay comparable.
        """
        t0 = time.time()
        if self._lora_sync:
            t_pause, t_barrier, t_transfer = self._sync_weight_lora(t0)
        else:
            t_pause, t_barrier, t_transfer = self._sync_weight_merged(t0)
        weight_sync_s = time.time() - t0
        # Total time and its three phases
        self._metrics["train"]["perf/weight_sync_s"].append(weight_sync_s)
        self._metrics["train"]["perf/weight_sync_pause_s"].append(t_pause - t0)
        self._metrics["train"]["perf/weight_sync_barrier_s"].append(t_barrier - t_pause)
        self._metrics["train"]["perf/weight_sync_transfer_s"].append(t_transfer - t_barrier)
        logger.info(f"Weight sync: done. Total {weight_sync_s:.1f}s")

    def _sync_weight_merged(self, t0: float) -> tuple[float, float, float]:
        """Fold the adapter, if any, into the base weights and stream every parameter to vLLM over NCCL.

        Args:
            t0 (`float`):
                When the sync started, for the phase log lines.

        Returns:
            `tuple[float, float, float]`:
                The pause, barrier and transfer phase marks, which [`_sync_weight`] turns into metrics.
        """
        # Under DDP `self.model` is the wrapper, which exposes none of the PEFT API used below.
        model = self.accelerator.unwrap_model(self.model)

        logger.info("Weight sync: pausing vLLM...")
        if self.accelerator.is_main_process:
            self.weight_transfer.pause()
            self.model_version += 1
            if self.rollout_worker:
                self.rollout_worker.update_model_version(self.model_version)
        t_pause = time.time()
        logger.info(f"Weight sync: pause took {t_pause - t0:.1f}s, waiting for all ranks...")

        self.accelerator.wait_for_everyone()
        t_barrier = time.time()

        logger.info(f"Weight sync: transferring weights... (barrier took {t_barrier - t_pause:.1f}s)")
        # vLLM only knows the base checkpoint's parameters, so the adapter is folded into them for the send. The
        # `finally` is not optional: leaving it merged would train merged weights from the next step on.
        if is_peft_model(model):
            with implicit_replication():
                model.merge_adapter()
        try:
            if self.accelerator.is_main_process:
                self.weight_transfer.send_weights(self._streaming_iter())
            else:
                # Non-rank-0 processes must still participate in full_tensor() collectives for FSDP2.
                for _ in self._streaming_iter():
                    pass
        finally:
            if is_peft_model(model):
                with implicit_replication():
                    model.unmerge_adapter()
        t_transfer = time.time()

        self.accelerator.wait_for_everyone()

        logger.info(f"Weight sync: resuming vLLM... (transfer took {t_transfer - t_barrier:.1f}s)")
        if self.accelerator.is_main_process:
            self.weight_transfer.resume()
        return t_pause, t_barrier, t_transfer

    def _sync_weight_lora(self, t0: float) -> tuple[float, float, float]:
        """Publish the trained adapter to vLLM under a fresh versioned name, moving no base weight.

        Each sync publishes `{lora_name}-v{N}` instead of overwriting the previous adapter in place, because
        `load_inplace=True` makes the server re-read the adapter from disk on every request
        (https://github.com/vllm-project/vllm/pull/41482) and vLLM keys its prefix cache on the adapter name alone
        (https://github.com/vllm-project/vllm/issues/42125), so an in-place swap would serve the previous policy's KV
        blocks. A new name is a fresh cache namespace, so no `reset_prefix_cache()` is needed.

        Args:
            t0 (`float`):
                When the sync started, for the phase log lines.

        Returns:
            `tuple[float, float, float]`:
                The pause, barrier and transfer phase marks, which [`_sync_weight`] turns into metrics.
        """
        version = self.model_version + 1
        adapter_dir = os.path.join(self._lora_dir, f"{self._lora_name}-v{version}")
        # Under DDP `self.model` is the wrapper, which does not expose `save_pretrained`.
        model = self.accelerator.unwrap_model(self.model)

        logger.info("Weight sync: pausing vLLM...")
        if self.accelerator.is_main_process:
            self.vllm_client.pause()
        t_pause = time.time()
        logger.info(f"Weight sync: pause took {t_pause - t0:.1f}s, waiting for all ranks...")

        self.accelerator.wait_for_everyone()
        t_barrier = time.time()

        logger.info(f"Weight sync: transferring adapter... (barrier took {t_barrier - t_pause:.1f}s)")
        # Every rank calls this, not just rank 0: materializing a sharded adapter parameter is a collective, even
        # though only the main process writes the files.
        save_lora_adapter(model, self.accelerator, self._adapter_name, adapter_dir)
        t_transfer = time.time()

        self.accelerator.wait_for_everyone()

        logger.info(f"Weight sync: resuming vLLM... (transfer took {t_transfer - t_barrier:.1f}s)")
        if self.accelerator.is_main_process:
            # The adapter has to exist on the server BEFORE the version moves below: the rollout worker derives the
            # adapter it requests from `model_version`, so bumping first would name one that is not loaded yet.
            self.vllm_client.load_lora_adapter(f"{self._lora_name}-v{version}", adapter_dir)
            self.vllm_client.resume()
            stale_version = version - (self.args.max_staleness + 1)
            if stale_version > 0:
                self.vllm_client.unload_lora_adapter(f"{self._lora_name}-v{stale_version}")
            # The files outlive the unload by one sync: vLLM resolves `lora_path` lazily inside the engine, where
            # a missing directory is not a 404 but a fatal `HFValidationError`.
            if stale_version > 1:
                stale_dir = f"{self._lora_name}-v{stale_version - 1}"
                shutil.rmtree(os.path.join(self._lora_dir, stale_dir), ignore_errors=True)
            self.model_version = version
            if self.rollout_worker:
                self.rollout_worker.update_model_version(self.model_version)
        return t_pause, t_barrier, t_transfer

    def _save(self, output_dir=None, state_dict=None):
        if not self.is_fsdp_enabled or self.accelerator.mixed_precision not in ("bf16", "fp16"):
            return super()._save(output_dir, state_dict)
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[self.compute_dtype]
        for name, tensor in state_dict.items():
            if tensor.is_floating_point():
                state_dict[name] = tensor.to(dtype)
        super()._save(output_dir, state_dict)
        model = self.accelerator.unwrap_model(self.model)
        if not is_peft_model(model):
            config = copy.deepcopy(model.config)
            config.dtype = self.compute_dtype
            config.save_pretrained(output_dir)

    def _save_checkpoint(self, model, trial):
        if self.accelerator.is_main_process:
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
            checkpoint_dir = os.path.join(self._get_output_dir(trial=trial), checkpoint_folder)
            os.makedirs(checkpoint_dir, exist_ok=True)
            consumed = self._trained_groups | self._dropped_groups
            first_unconsumed = next(g for g in itertools.count() if g not in consumed)
            prompt_index = self.rollout_worker._loop_kwargs["dataset_start_index"] + first_unconsumed
            # `model_version` rides along so adapter names keep counting across a resume: restarting at v1 would
            # republish a different adapter under a name a still-running server already holds.
            rollout_state = {"prompt_index": prompt_index, "model_version": self.model_version}
            with open(os.path.join(checkpoint_dir, "rollout_state.json"), "w", encoding="utf-8") as f:
                json.dump(rollout_state, f)
        super()._save_checkpoint(model, trial)

    def _inner_training_loop(self, *args, **kwargs):
        # When resuming, pass the saved prompt position to the worker before _TrainBeginCallback fires.
        # Skipped for IterableDataset since len() isn't available on streaming datasets.
        # Always reset first so a stale value from a prior train() call is never carried over.
        if self.accelerator.is_main_process:
            self.rollout_worker._loop_kwargs["dataset_start_index"] = 0
            self._groups_before_resume = 0
            resume_from_checkpoint = kwargs.get("resume_from_checkpoint")
            if resume_from_checkpoint is not None:
                rollout_state_file = os.path.join(resume_from_checkpoint, "rollout_state.json")
                # IterableDataset is skipped deliberately: streaming datasets have no len() and can't be repositioned.
                if not os.path.isfile(rollout_state_file):
                    logger.warning(
                        "rollout_state.json not found in the checkpoint; "
                        "the rollout worker will restart from prompt 0."
                    )
                elif not isinstance(self.train_dataset, Dataset):
                    logger.warning("Resuming with an IterableDataset; the rollout worker will restart from prompt 0.")
                else:
                    with open(rollout_state_file, encoding="utf-8") as f:
                        rollout_state = json.load(f)
                    self.rollout_worker._loop_kwargs["dataset_start_index"] = rollout_state["prompt_index"]
                    self._groups_before_resume = rollout_state["prompt_index"]
                    # Older checkpoints predate this field; resuming from one restarts numbering at v1, which is
                    # only safe against a freshly started server.
                    self.model_version = rollout_state.get("model_version", 0)
                    self.rollout_worker.update_model_version(self.model_version)
        try:
            return super()._inner_training_loop(*args, **kwargs)
        finally:
            if self.accelerator.is_main_process:
                if self.rollout_worker:
                    self.rollout_worker.stop()
                if self.weight_transfer:
                    self.weight_transfer.destroy()
