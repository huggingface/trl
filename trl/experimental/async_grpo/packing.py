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

from functools import cache
from typing import Any

import torch
from transformers import AttentionInterface
from transformers.integrations.flash_attention import get_target_dtype
from transformers.masking_utils import AttentionMaskInterface


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
        input_ids = sample["input_ids"]
        children, shared = self.roots, 0
        for key in zip(input_ids, [*input_ids[1:], None], strict=True):
            if key not in children:
                break
            children = self.children[children[key]]
            shared += 1
        novel = len(input_ids) - shared
        return len(self.tokens) + novel <= budget and self.trained + sum(sample["completion_mask"]) <= budget

    def add(self, sample: dict[str, Any]) -> None:
        input_ids = sample["input_ids"]
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
        self.trained += sum(sample["completion_mask"])

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
                - `tree_segment_lengths` (`S`): tokens of each segment.
                - `tree_segment_parents` (`S`): segment each segment hangs off, `-1` for a root.
                - `token_index` (`M`): position whose prediction is each trained token. A shared position serves every
                  sample that shares it, so there are more loss terms than labelled positions.
                - `old_log_probs` (`M`), `advantages` (`M`): generator log-probability and advantage of each target.
                - `segment_id` (`M`): index of the sample each target came from.
        """
        input_ids, position_ids, offsets, parents = [], [], [0], []
        packed = [0] * len(self.tokens)
        stack = [(node, -1, 0) for node in reversed(self.roots.values())]
        while stack:
            node, parent, depth = stack.pop()
            segment = len(parents)
            parents.append(parent)
            while True:
                packed[node] = len(input_ids)
                input_ids.append(self.tokens[node])
                position_ids.append(depth)
                depth += 1
                if len(self.children[node]) != 1:
                    break
                node = next(iter(self.children[node].values()))
            offsets.append(len(input_ids))
            stack.extend((child, segment, depth) for child in reversed(self.children[node].values()))

        shift_labels = [-100] * len(input_ids)
        token_index, old_log_probs, advantages, segment_id = [], [], [], []
        for i, (sample, nodes) in enumerate(zip(self.samples, self.sample_nodes, strict=True)):
            for p in range(1, len(nodes)):
                if sample["completion_mask"][p]:
                    shift_labels[packed[nodes[p - 1]]] = sample["input_ids"][p]
                    token_index.append(packed[nodes[p - 1]])
                    old_log_probs.append(sample["old_log_probs"][p])
                    advantages.append(sample["advantage"])
                    segment_id.append(i)
        return {
            "input_ids": torch.tensor(input_ids),
            "position_ids": torch.tensor(position_ids),
            "shift_labels": torch.tensor(shift_labels),
            "tree_segment_lengths": torch.tensor(offsets[1:]) - torch.tensor(offsets[:-1]),
            "tree_segment_parents": torch.tensor(parents),
            "token_index": torch.tensor(token_index, dtype=torch.long),
            "old_log_probs": torch.tensor(old_log_probs, dtype=torch.float32),
            "advantages": torch.tensor(advantages, dtype=torch.float32),
            "segment_id": torch.tensor(segment_id, dtype=torch.long),
        }


TREE_ATTENTION = "tree_fa3"


def tree_attention_kwargs(inputs: dict[str, torch.Tensor], device: torch.device) -> dict[str, Any]:
    """
    Tree attention as plain FA3 varlen attention: each segment's queries attend to the K/V of its root-to-segment path.

    FA3 aligns the causal mask bottom-right when a sequence has fewer queries than keys, so a segment's queries, which
    are the last tokens of their path, see exactly their ancestors plus their own causal past. The K/V of shared
    prefixes is gathered once per descendant segment, but every other layer still runs once per unique token.

    Args:
        inputs (`dict[str, torch.Tensor]`):
            This rank's batch, holding the padded `tree_segment_lengths` and `tree_segment_parents` of its row.
        device (`torch.device`):
            Device the attention runs on.

    Returns:
        `dict[str, Any]`, the keyword arguments [`tree_attention`] reads:
            - `tree_kv_index`: packed position of every K/V row, the paths of all segments laid end to end.
            - `tree_cu_seqlens_q`, `tree_cu_seqlens_k`: segment boundaries in the row, path boundaries in the index.
            - `tree_max_seqlen_q`, `tree_max_seqlen_k`: longest segment, longest path.
    """
    lengths = inputs["tree_segment_lengths"][0]
    segments = lengths > 0
    offsets = [0, *lengths[segments].cumsum(0).tolist()]
    paths, kv_index, cu_seqlens_k = [], [], [0]
    for segment, parent in enumerate(inputs["tree_segment_parents"][0][segments].tolist()):
        paths.append((paths[parent] if parent != -1 else []) + list(range(offsets[segment], offsets[segment + 1])))
        kv_index.extend(paths[-1])
        cu_seqlens_k.append(len(kv_index))
    return {
        "tree_kv_index": torch.tensor(kv_index, dtype=torch.long, device=device),
        "tree_cu_seqlens_q": torch.tensor(offsets, dtype=torch.int32, device=device),
        "tree_cu_seqlens_k": torch.tensor(cu_seqlens_k, dtype=torch.int32, device=device),
        "tree_max_seqlen_q": int(lengths.max()),
        "tree_max_seqlen_k": max(len(path) for path in paths),
    }


@cache
def _flash_attn3():
    from kernels import get_kernel

    return get_kernel("kernels-community/flash-attn3", version=1)


def tree_attention(
    module,
    query,
    key,
    value,
    attention_mask,
    *,
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
    dtype = get_target_dtype(query, module) or query.dtype
    query, key, value = (x.transpose(1, 2)[0].to(dtype) for x in (query, key, value))
    out = _flash_attn3().flash_attn_varlen_func(
        query,
        key.index_select(0, tree_kv_index),
        value.index_select(0, tree_kv_index),
        tree_cu_seqlens_q,
        tree_cu_seqlens_k,
        tree_max_seqlen_q,
        tree_max_seqlen_k,
        softmax_scale=scaling,
        causal=True,
        window_size=(sliding_window - 1, 0) if sliding_window else (-1, -1),
        softcap=softcap or 0.0,
    )
    out = out[0] if isinstance(out, tuple) else out
    return out.unsqueeze(0), None


def register_tree_attention() -> None:
    AttentionInterface.register(TREE_ATTENTION, tree_attention)
    AttentionMaskInterface.register(TREE_ATTENTION, lambda **kwargs: None)
