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
from itertools import accumulate
from typing import Any, Protocol

import torch

from .tree import TREE_ATTENTION, PrefixForest, build_tree_block_mask, dfs_intervals


@dataclass
class TrainingRow:
    """
    One DP rank's forward, and the flat list of loss terms taken over it.

    The selection is explicit — a list of `(packed position, target token)` pairs rather than a boolean mask — because
    under tree packing one packed position predicts the first trained token of *every* row that shares its prefix, each
    with its own target and advantage. A mask cannot say that; a gather can, and it expresses ordinary next-token
    shifting just as well.

    Args:
        input_ids (`torch.Tensor`):
            Tokens the decoder forwards, shape `(N,)`.
        position_ids (`torch.Tensor`):
            RoPE position of each forwarded token, shape `(N,)`.
        pred_index (`torch.Tensor`):
            Position in `input_ids` whose hidden state predicts each trained token, shape `(M,)`.
        target_id (`torch.Tensor`):
            Token each entry of `pred_index` must predict, shape `(M,)`.
        old_log_probs (`torch.Tensor`):
            Generator log-probability of each target, shape `(M,)`.
        advantages (`torch.Tensor`):
            Advantage of each target, shape `(M,)`.
        segment_id (`torch.Tensor`):
            Index of the original rollout row each target came from, shape `(M,)`. Kept explicit because packing
            destroys any way of reading completion boundaries back off `position_ids`.
        tree_enter (`torch.Tensor`, *optional*):
            DFS entry clock of each forwarded token, shape `(N,)`. Only tree packing fills these; together with
            `tree_leave` they are the whole attention mask.
        tree_leave (`torch.Tensor`, *optional*):
            DFS exit clock of each forwarded token, shape `(N,)`.
    """

    input_ids: torch.Tensor
    position_ids: torch.Tensor
    pred_index: torch.Tensor
    target_id: torch.Tensor
    old_log_probs: torch.Tensor
    advantages: torch.Tensor
    segment_id: torch.Tensor
    tree_enter: torch.Tensor | None = None
    tree_leave: torch.Tensor | None = None


class PackingProtocol(Protocol):
    """How a micro-batch becomes one row per DP rank, and what attention that row needs.

    Chosen once in [`AsyncGRPOTrainer.__init__`]; the planner, the collator and `compute_loss` all go through it and
    never branch on the mode. Implemented by [`SequencePacking`] and [`TreePacking`].

    Attributes:
        attn_implementation (`str`):
            What to load the model with, since the row's structure is what attention has to read.
    """

    attn_implementation: str

    def atoms(self, samples: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        """Split `samples` into units the planner may place independently."""
        ...

    def cost(self, atom: list[dict[str, Any]]) -> tuple[int, int]:
        """Forwarded tokens and attention cost of an atom, both additive over the atoms of a row."""
        ...

    def open_row(self) -> "SequenceRow | TreeRow":
        """Start an empty row: an accumulator the planner prices candidates against and fills."""
        ...

    def forward_kwargs(self, inputs: dict[str, torch.Tensor], tokens: torch.Tensor) -> dict[str, Any]:
        """Anything beyond `input_ids` and `position_ids` the decoder needs, read off this rank's stripped batch."""
        ...

    def pack(self, samples: list[dict[str, Any]]) -> TrainingRow:
        """Lay `samples` out as one row, and select the loss terms taken over it."""
        ...


def _loss_terms(samples: list[dict[str, Any]], packed: list[list[int]]) -> dict[str, torch.Tensor]:
    """
    The flat loss terms of a row, given where each sample's tokens landed in it.

    This is the whole of the selection contract, and it is the same for every packing: a trained token is scored off
    its predecessor's hidden state, so all a strategy has to say is where that predecessor ended up. Sequence packing
    answers with a running offset, tree packing with the packed position of a trie node.

    Args:
        samples (`list[dict]`):
            The rollout rows of one DP rank's row, in placement order.
        packed (`list[list[int]]`):
            For each sample, the packed position of each of its tokens.

    Returns:
        `dict[str, torch.Tensor]`: the `pred_index`, `target_id`, `old_log_probs`, `advantages` and `segment_id` fields
        of a [`TrainingRow`], each of length `M`.
    """
    pred_index, target_id, old_log_probs, advantages, segment_id = [], [], [], [], []
    for i, (sample, positions) in enumerate(zip(samples, packed, strict=True)):
        for p in range(1, len(sample["input_ids"])):
            if sample["completion_mask"][p]:
                pred_index.append(positions[p - 1])
                target_id.append(sample["input_ids"][p])
                old_log_probs.append(sample["old_log_probs"][p])
                advantages.append(sample["advantage"])
                segment_id.append(i)
    return {
        "pred_index": torch.tensor(pred_index, dtype=torch.long),
        "target_id": torch.tensor(target_id, dtype=torch.long),
        "old_log_probs": torch.tensor(old_log_probs, dtype=torch.float32),
        "advantages": torch.tensor(advantages, dtype=torch.float32),
        "segment_id": torch.tensor(segment_id, dtype=torch.long),
    }


class SequenceRow:
    """
    A row of samples laid end to end, priced as the planner fills it.

    `samples` is what has been placed so far and `load` the row's attention cost, `Σ Lᵢ²` — the planner equalizes that
    across ranks so none straggles at the gradient all-reduce.
    """

    def __init__(self):
        self.samples: list[dict[str, Any]] = []
        self.tokens = 0
        self.load = 0

    def holds(self, sample: dict[str, Any]) -> bool:
        return False

    def fits(self, sample: dict[str, Any], budget: int) -> bool:
        return self.tokens + len(sample["input_ids"]) <= budget

    def add(self, sample: dict[str, Any]) -> None:
        n = len(sample["input_ids"])
        self.samples.append(sample)
        self.tokens += n
        self.load += n * n


class TreeRow:
    """
    A row of samples folded into a prefix forest.

    A sample costs only its *novel* tokens, so the price depends on what the row already holds — which is why the
    planner asks the row rather than the sample. Sharing is confined to a `group_id`, so a sample can only be cheap in
    the row that already holds its group, and [`~TreeRow.holds`] sends it there.
    """

    def __init__(self):
        self.samples: list[dict[str, Any]] = []
        self.tokens = 0
        self.trained = 0
        self.load = 0
        self.forest = PrefixForest()
        self.groups: set[Any] = set()

    def holds(self, sample: dict[str, Any]) -> bool:
        return sample["group_id"] in self.groups

    def fits(self, sample: dict[str, Any], budget: int) -> bool:
        novel, _cost = self.forest.probe(sample["input_ids"], sample["group_id"])
        return self.tokens + novel <= budget and self.trained + sum(sample["completion_mask"]) <= budget

    def add(self, sample: dict[str, Any]) -> None:
        novel, cost = self.forest.probe(sample["input_ids"], sample["group_id"])
        self.forest.insert(sample["input_ids"], sample["group_id"])
        self.samples.append(sample)
        self.groups.add(sample["group_id"])
        self.tokens += novel
        self.trained += sum(sample["completion_mask"])
        self.load += cost


class SequencePacking:
    """
    Samples concatenated into one padding-free row per rank, `position_ids` resetting at each sequence start.

    FlashAttention reads the resets as `cu_seq_lens` and attends block-diagonally, so a sample sees only itself.
    Nothing is shared between samples, which makes a single sample the scheduling atom and its cost additive.
    """

    attn_implementation = "kernels-community/flash-attn3"

    def atoms(self, samples: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        """Units the planner may move independently. Nothing is shared, so a sample stands alone."""
        return [[sample] for sample in samples]

    def cost(self, atom: list[dict[str, Any]]) -> tuple[int, int]:
        """Forwarded tokens and attention cost of an atom, both additive over the atoms of a row."""
        lengths = [len(sample["input_ids"]) for sample in atom]
        return sum(lengths), sum(n * n for n in lengths)

    def open_row(self) -> "SequenceRow":
        return SequenceRow()

    def forward_kwargs(self, inputs: dict[str, torch.Tensor], tokens: torch.Tensor) -> dict[str, Any]:
        return {}

    def pack(self, samples: list[dict[str, Any]]) -> TrainingRow:
        lengths = [len(sample["input_ids"]) for sample in samples]
        offsets = list(accumulate(lengths, initial=0))
        return TrainingRow(
            input_ids=torch.tensor([token for sample in samples for token in sample["input_ids"]], dtype=torch.long),
            position_ids=torch.cat([torch.arange(n) for n in lengths]),
            **_loss_terms(samples, [list(range(o, o + n)) for o, n in zip(offsets, lengths, strict=False)]),
        )


class TreePacking:
    """
    Rows packed into a prefix forest, so a token shared by several rows is forwarded once.

    Multi-turn rollouts overlap heavily — a shared prompt, and after a fork a shared conversation — so the decoder's
    per-token work drops by the packing ratio (total tokens ÷ unique tokens). Attention still sees exactly what each
    row saw, through a FlexAttention block mask built from the forest's DFS stamps.

    Sharing only ever happens inside a `group_id` (the packer keeps one trie per group), so a sample is only ever cheap
    in the row that already holds its group. That is what keeps the planner a greedy bin-packer rather than a
    submodular partitioning problem: it places each sample in its group's row, and groups are additive across rows.
    """

    attn_implementation = TREE_ATTENTION

    def atoms(self, samples: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        """One atom per `group_id`: its samples have to stay together, or they share nothing."""
        groups: dict[Any, list[dict[str, Any]]] = {}
        for sample in samples:
            groups.setdefault(sample["group_id"], []).append(sample)
        return list(groups.values())

    def cost(self, atom: list[dict[str, Any]]) -> tuple[int, int]:
        """Unique tokens and attention cost of an atom, both additive over the atoms of a row.

        The attention cost is `Σ (depth + 1)` over the packed tokens, i.e. the number of score pairs the row's forward
        evaluates. It replaces `Σ Lᵢ²` for balancing because packing changes memory and time by different factors:
        memory follows unique tokens, time follows surviving score pairs.
        """
        forest = PrefixForest()
        tokens = attention = 0
        for sample in atom:
            novel, novel_cost = forest.probe(sample["input_ids"], sample["group_id"])
            forest.insert(sample["input_ids"], sample["group_id"])
            tokens += novel
            attention += novel_cost
        return tokens, attention

    def open_row(self) -> "TreeRow":
        return TreeRow()

    def forward_kwargs(self, inputs: dict[str, torch.Tensor], tokens: torch.Tensor) -> dict[str, Any]:
        enter, leave = inputs["tree_enter"][tokens], inputs["tree_leave"][tokens]
        return {"tree_block_mask": build_tree_block_mask(enter, leave)}

    def pack(self, samples: list[dict[str, Any]]) -> TrainingRow:
        forest = PrefixForest()
        for sample in samples:
            forest.insert(sample["input_ids"], sample["group_id"])
        tokens, layout, node_to_packed = forest.linearize()
        enter, leave = dfs_intervals(layout)

        packed_samples = [
            [node_to_packed[node] for node in forest.walk(sample["input_ids"], sample["group_id"])]
            for sample in samples
        ]
        loss_terms = _loss_terms(
            samples,
            packed_samples,
        )
        return TrainingRow(
            input_ids=torch.tensor(tokens, dtype=torch.long),
            position_ids=layout.position_ids,
            tree_enter=torch.tensor(enter, dtype=torch.int32),
            tree_leave=torch.tensor(leave, dtype=torch.int32),
            **loss_terms,
        )
