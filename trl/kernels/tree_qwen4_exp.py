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

"""
Qwen4-Exp over a tree-packed row: the n-gram embedding, the short convolution and the QSA indexer follow each sample's
own path instead of the flat row.

A packed row is a prefix trie laid out depth-first, given as segments with parents (see `~packing.TreeRow.pack`). Three
maps derived from it stand in for "the tokens before this one":

- `prev` (`n`): the packed position before a token on its own path, `-1` at a path start.
- `depth` (`n`): the token's position on its path, the row's `position_ids`.
- `visible` (`n`, `n`): whether a position sees another, `True` when the other is on its path and not after it.

They are built once per forward, for all its layers and for the gradient-checkpointing recompute.
"""

import math
from typing import Any

import torch
import torch.nn.functional as F


# Queries scored against the indexer's blocks at a time: the scores of a chunk are `(queries, blocks)`, the step's
# widest temporary
_QUERY_CHUNK_SIZE = 1024

_MAPS: tuple[torch.Tensor, dict[str, torch.Tensor]] | None = None


def tree_maps(tree: dict[str, Any]) -> dict[str, torch.Tensor]:
    """
    The `prev`, `depth` and `visible` maps of a packed row, built once per forward and shared by all its layers.

    Args:
        tree (`dict[str, Any]`):
            The forward's tree keyword arguments: `tree_cu_seqlens_q`, `tree_cu_seqlens_k`, `tree_kv_index` and
            `tree_segment_parents`.

    Returns:
        `dict[str, torch.Tensor]`: `prev` and `depth` (`n`) and `visible` (`n`, `n`).
    """
    global _MAPS
    key = tree["tree_cu_seqlens_q"]
    if _MAPS is None or _MAPS[0] is not key:
        # Holding the key keeps its identity unique, so a later row cannot be mistaken for this one
        _MAPS = (key, _build_maps(tree))
    return _MAPS[1]


def _build_maps(tree: dict[str, Any]) -> dict[str, torch.Tensor]:
    device = tree["tree_cu_seqlens_q"].device
    cu_seqlens_q, cu_seqlens_k = tree["tree_cu_seqlens_q"].long(), tree["tree_cu_seqlens_k"].long()
    parents, kv_index = tree["tree_segment_parents"].long(), tree["tree_kv_index"].long()
    n_tokens, n_segments = int(cu_seqlens_q[-1]), parents.numel()
    positions = torch.arange(n_tokens, device=device)
    segment = torch.repeat_interleave(
        torch.arange(n_segments, device=device),
        cu_seqlens_q[1:] - cu_seqlens_q[:-1],
        output_size=n_tokens,
    )
    starts = cu_seqlens_q[:-1][segment]
    # A segment's path holds its ancestors' paths, then the segment itself, so a token's depth is its position on that
    # path: its offset in the segment plus the ancestors the path prepends
    lengths, path_lengths = cu_seqlens_q[1:] - cu_seqlens_q[:-1], cu_seqlens_k[1:] - cu_seqlens_k[:-1]
    depth = positions - starts + (path_lengths - lengths)[segment]
    prev = torch.where(
        positions == starts,
        torch.where(parents >= 0, cu_seqlens_q[parents.clamp_min(0) + 1] - 1, -1)[segment],
        positions - 1,
    )

    visible = torch.zeros(n_tokens, n_tokens, dtype=torch.bool, device=device)
    bounds, path_bounds = cu_seqlens_q.tolist(), cu_seqlens_k.tolist()
    for s in range(n_segments):
        rows = torch.arange(bounds[s], bounds[s + 1], device=device)
        path = kv_index[path_bounds[s] : path_bounds[s + 1]]
        # A query sees its own path up to itself, which is the segment path's first `depth + 1` tokens
        columns = torch.arange(path.numel(), device=device)
        visible[rows.unsqueeze(1), path.unsqueeze(0)] = columns <= depth[rows].unsqueeze(1)
    return {"prev": prev, "depth": depth, "visible": visible}


def tree_ngram_shift_right(
    token_ids: torch.Tensor, shift: int, eos_token_id: int, tree: dict[str, Any]
) -> torch.Tensor:
    """
    The `shift`-th previous token of every position on its own path, `eos_token_id` where the walk leaves the path or
    crosses an end of sequence.

    Args:
        token_ids (`torch.Tensor`):
            `(batch, length)` token history: the caller's `eos_token_id` context, then the row's own tokens.
        shift (`int`):
            How many tokens back to read.
        eos_token_id (`int`):
            Token the walk stops at, and the value returned where it does.
        tree (`dict[str, Any]`):
            The forward's tree keyword arguments.

    Returns:
        `torch.Tensor`: `(batch, length)`, the context columns unchanged.
    """
    if shift == 0:
        return token_ids
    prev = tree_maps(tree)["prev"]
    context = token_ids.shape[1] - prev.numel()
    tokens = token_ids[:, context:]
    current = torch.arange(prev.numel(), device=token_ids.device).expand(tokens.shape[0], -1)
    on_path = torch.ones_like(current, dtype=torch.bool)
    for _ in range(shift):
        current = prev[current]
        on_path &= current >= 0
        current = current.clamp_min(0)
        on_path &= tokens.gather(1, current) != eos_token_id
    shifted = torch.where(on_path, tokens.gather(1, current), torch.full_like(current, eos_token_id))
    return torch.cat([token_ids[:, :context], shifted], dim=1)


def tree_ple_short_conv(hidden_states: torch.Tensor, conv1d: torch.nn.Conv1d, tree: dict[str, Any]) -> torch.Tensor:
    """
    The dilated depthwise short convolution over each path: tap `k` reads `(kernel_size - 1 - k) * dilation` steps
    back along the path, and zero before its start.

    Args:
        hidden_states (`torch.Tensor`):
            `(1, channels, state_len + length)`, the caller's left padding included, as the model passes it to the
            convolution.
        conv1d (`torch.nn.Conv1d`):
            The layer's depthwise convolution, which sets the kernel size, the dilation and the weights.
        tree (`dict[str, Any]`):
            The forward's tree keyword arguments.

    Returns:
        `torch.Tensor`: `(1, channels, length)`.
    """
    prev = tree_maps(tree)["prev"]
    kernel_size, dilation = conv1d.kernel_size[0], conv1d.dilation[0]
    tokens = hidden_states[..., -prev.numel() :].transpose(1, 2)[0]
    # `back[s]` is the position `s` steps back along the path, and `-1` from the step that leaves it
    back = [torch.arange(prev.numel(), device=prev.device)]
    for _ in range((kernel_size - 1) * dilation):
        back.append(torch.where(back[-1] >= 0, prev[back[-1].clamp_min(0)], -1))
    out = torch.zeros(tokens.shape, dtype=torch.float32, device=tokens.device)
    for tap in range(kernel_size):
        source = back[(kernel_size - 1 - tap) * dilation]
        gathered = tokens.float().index_select(0, source.clamp_min(0))
        out += torch.where(source.unsqueeze(1) >= 0, gathered, 0.0) * conv1d.weight[:, 0, tap].float()
    if conv1d.bias is not None:
        out = out + conv1d.bias.float()
    return F.silu(out.to(tokens.dtype)).transpose(0, 1).unsqueeze(0)


def tree_qsa_select_tokens(
    q: torch.Tensor,
    raw_keys: torch.Tensor,
    full_cos: torch.Tensor,
    full_sin: torch.Tensor,
    tree: dict[str, Any],
    k_layernorm: torch.nn.Module,
    block_topk: int,
    compress_ratio: int,
) -> torch.Tensor:
    """
    The QSA selection over a packed row: a block is `compress_ratio` consecutive depths of one path, keyed and pooled
    by its last token, so a query keeps the blocks whose last token is on its path, then the tail after them.

    Args:
        q (`torch.Tensor`):
            `(batch, queries, heads, head_dim)` indexer queries, after their layer norm and rotation.
        raw_keys (`torch.Tensor`):
            `(batch, keys, head_dim)` indexer keys, after the cache update; the forward has no cache here.
        full_cos (`torch.Tensor`), `full_sin` (`torch.Tensor`):
            `(batch, keys, head_dim)` rotations the indexer holds.
        tree (`dict[str, Any]`):
            The forward's tree keyword arguments.
        k_layernorm (`torch.nn.Module`):
            The indexer's key layer norm, applied to the pooled keys.
        block_topk (`int`):
            Number of blocks a query keeps.
        compress_ratio (`int`):
            Depths per block.

    Returns:
        `torch.Tensor`: `(1, 1, queries, keys)` of the tokens each query may attend to.
    """
    prev, depth, visible = (tree_maps(tree)[name] for name in ("prev", "depth", "visible"))
    n_tokens, device = depth.numel(), depth.device
    keys = ((depth % compress_ratio) == compress_ratio - 1).nonzero().flatten()
    members = keys.unsqueeze(1)
    for _ in range(compress_ratio - 1):
        members = torch.cat([prev[members[:, 0]].unsqueeze(1), members], dim=1)
    selected = torch.zeros(1, 1, n_tokens, n_tokens, dtype=torch.bool, device=device)
    n_blocks = keys.numel()
    if n_blocks == 0:
        return selected
    block_states = _pooled_block_keys(raw_keys, full_cos, full_sin, members, k_layernorm).float()
    block_seen = visible.index_select(1, keys)
    lowest = torch.finfo(torch.float32).min
    keep = min(block_topk, n_blocks)
    for start in range(0, n_tokens, _QUERY_CHUNK_SIZE):
        rows = torch.arange(start, min(start + _QUERY_CHUNK_SIZE, n_tokens), device=device)
        queries = q[0][rows].float()
        scores = torch.relu(queries @ block_states.transpose(-1, -2)).sum(-2) / math.sqrt(q.shape[-1])
        scored = scores.masked_fill(~block_seen[rows], lowest)
        top = scored.topk(keep, dim=-1).indices
        kept = torch.gather(scored, 1, top) != lowest
        # Blocks of different branches can hold the same token, so the union may write a token more than once
        union = torch.zeros(rows.numel(), n_tokens + 1, dtype=torch.bool, device=device)
        chunk_rows = torch.arange(rows.numel(), device=device).unsqueeze(1).unsqueeze(2)
        tokens = torch.where(kept.unsqueeze(-1), members[top], n_tokens)
        union[chunk_rows.expand(-1, keep, compress_ratio), tokens] = True
        tail = depth.unsqueeze(0) >= (((depth[rows] + 1) // compress_ratio) * compress_ratio).unsqueeze(1)
        selected[0, 0, rows] = (union[:, :n_tokens] | tail) & visible[rows]
    return selected


def _pooled_block_keys(
    raw_keys: torch.Tensor,
    full_cos: torch.Tensor,
    full_sin: torch.Tensor,
    members: torch.Tensor,
    k_layernorm: torch.nn.Module,
) -> torch.Tensor:
    """The block keys: the members' mean, the indexer's key layer norm, and the rotation at the first member's."""
    pooled = raw_keys[0].float().index_select(0, members.flatten())
    pooled = pooled.view(*members.shape, -1).mean(dim=1).to(raw_keys.dtype)
    pooled = k_layernorm(pooled).unsqueeze(1)
    starts = members[:, 0]
    return _rope(pooled, full_cos[0].index_select(0, starts), full_sin[0].index_select(0, starts)).squeeze(1)


def _rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """The half rotation of the first `cos.shape[-1]` dimensions, as the modeling module applies it to keys."""
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    rotary_dim = cos.shape[-1]
    rotary, rest = x[..., :rotary_dim], x[..., rotary_dim:]
    half = rotary.shape[-1] // 2
    rotated = torch.cat([-rotary[..., half:], rotary[..., :half]], dim=-1)
    return torch.cat([rotary * cos + rotated * sin, rest], dim=-1)
