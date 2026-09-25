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

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class TreeLayout:
    """
    Segment tree of a packed forest.

    Segment `s` holds the packed tokens `[offsets[s], offsets[s + 1])` and hangs off segment `parents[s]` (`-1` for a
    root). A segment ends at a branch or at a row end, which is what makes every original row exactly the concatenation
    of whole segments along one root-to-node path.

    Args:
        offsets (`tuple[int, ...]`):
            Token boundaries of the segments, one more entry than `parents`, starting at `0`.
        parents (`tuple[int, ...]`):
            Parent segment of each segment, `-1` for a root. Parents always precede their children.
    """

    offsets: tuple[int, ...]
    parents: tuple[int, ...]

    @property
    def num_tokens(self) -> int:
        return self.offsets[-1]

    @property
    def position_ids(self) -> torch.Tensor:
        """Original row position of each packed token, which is its depth in the forest."""
        depths = []
        for parent in self.parents:
            depths.append(0 if parent == -1 else depths[parent] + self.offsets[parent + 1] - self.offsets[parent])
        return torch.tensor(
            [d + i for s, d in enumerate(depths) for i in range(self.offsets[s + 1] - self.offsets[s])]
        )


class PrefixForest:
    """
    One prefix trie per `group_id`: rows of different groups never share a token.

    Grown one row at a time so the planner can price a candidate before committing it ([`~PrefixForest.probe`]), which
    is what lets the token budget bound *unique* tokens rather than raw ones. Restricting sharing to a single group is
    what keeps that price additive across groups, so the planner stays a greedy bin-packer.
    """

    def __init__(self):
        self.tokens: list[int] = []  # node -> its token
        self.children: list[dict[int, int]] = []  # node -> {token: child node}
        self.terminal: list[bool] = []  # node -> some row ended exactly here
        self.roots: dict[Any, dict[int, int]] = {}  # group_id -> {token: node}

    def probe(self, input_ids: list[int], group_id: Any) -> tuple[int, int]:
        """
        Price a row against the forest without inserting it.

        Returns:
            `tuple[int, int]`:
                - Number of tokens of `input_ids` not already in `group_id`'s trie, i.e. what the row would add to the
                  packed sequence.
                - Attention cost of those tokens, `Σ (depth + 1)`: a packed token attends to its ancestors plus itself,
                  so the sum is the number of score pairs the row would add.
        """
        children = self.roots.get(group_id, {})
        depth = 0
        while depth < len(input_ids):
            node = children.get(input_ids[depth])
            if node is None:
                break
            children = self.children[node]
            depth += 1
        return len(input_ids) - depth, sum(range(depth + 1, len(input_ids) + 1))

    def insert(self, input_ids: list[int], group_id: Any) -> None:
        children = self.roots.setdefault(group_id, {})
        for token in input_ids:
            node = children.get(token)
            if node is None:
                node = len(self.tokens)
                children[token] = node
                self.tokens.append(token)
                self.children.append({})
                self.terminal.append(False)
            children = self.children[node]
        self.terminal[node] = True

    def walk(self, input_ids: list[int], group_id: Any) -> list[int]:
        """Trie node of every token of a row that was already inserted."""
        nodes = []
        children = self.roots[group_id]
        for token in input_ids:
            node = children[token]
            nodes.append(node)
            children = self.children[node]
        return nodes

    def linearize(self) -> tuple[list[int], TreeLayout, list[int]]:
        """
        Lay the forest out in DFS pre-order, collapsing unary chains into contiguous segments.

        Returns:
            `tuple[list[int], TreeLayout, list[int]]`:
                - The packed tokens, one entry per node.
                - The segment tree over them.
                - Packed position of each trie node.
        """
        tokens, parents, offsets = [], [], [0]
        node_to_packed = [0] * len(self.tokens)
        roots = [node for group in self.roots.values() for node in group.values()]
        stack = [(node, -1) for node in reversed(roots)]
        while stack:
            node, parent = stack.pop()
            segment = len(parents)
            parents.append(parent)
            while True:
                node_to_packed[node] = len(tokens)
                tokens.append(self.tokens[node])
                if self.terminal[node] or len(self.children[node]) != 1:
                    break
                node = next(iter(self.children[node].values()))
            offsets.append(len(tokens))
            stack.extend((child, segment) for child in reversed(list(self.children[node].values())))
        return tokens, TreeLayout(tuple(offsets), tuple(parents)), node_to_packed
