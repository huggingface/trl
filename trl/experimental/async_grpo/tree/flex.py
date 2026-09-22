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

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from transformers import AttentionInterface
from transformers.masking_utils import AttentionMaskInterface

from .layout import TreeLayout


TREE_ATTENTION = "tree_flex"

_compiled_flex_attention = torch.compile(flex_attention, dynamic=True, fullgraph=True)


def dfs_intervals(layout: TreeLayout) -> tuple[list[int], list[int]]:
    """
    Stamp every packed token with its DFS entry and exit clock.

    A subtree occupies a contiguous interval of pre-order indices, so these two integers per token turn the `O(depth)`
    ancestor walk into the `O(1)` test `enter[k] <= enter[q] < leave[k]`, which is what a FlexAttention `mask_mod`
    needs. [`~PrefixForest.linearize`] already lays the forest out in pre-order, so the entry clock is the packed
    position itself, and all that is left is where each segment's subtree ends.

    Args:
        layout ([`TreeLayout`]):
            Segment tree of the packed forest.

    Returns:
        `tuple[list[int], list[int]]`: entry and exit clocks, one pair per packed token.
    """
    offsets = layout.offsets
    subtree_end = list(offsets[1:])
    for segment in reversed(range(len(layout.parents))):
        parent = layout.parents[segment]
        if parent != -1:
            subtree_end[parent] = max(subtree_end[parent], subtree_end[segment])
    enter = list(range(layout.num_tokens))
    leave = [end for s, end in enumerate(subtree_end) for _ in range(offsets[s + 1] - offsets[s])]
    return enter, leave


def build_tree_block_mask(enter: torch.Tensor, leave: torch.Tensor, block_size: int = 128):
    """
    Build the FlexAttention block mask under which every packed token sees exactly its own ancestry.

    Args:
        enter (`torch.Tensor`):
            DFS entry clocks, shape `(num_tokens,)`, on the device the attention runs on.
        leave (`torch.Tensor`):
            DFS exit clocks, same shape and device.
        block_size (`int`, *optional*, defaults to `128`):
            Granularity at which the mask is stored: a block is kept whole, evaluated per element, or skipped.

    Returns:
        `~torch.nn.attention.flex_attention.BlockMask`: block mask for the packed sequence.
    """
    n = enter.numel()
    padding = -n % block_size
    enter = F.pad(enter, (0, padding), value=n)
    leave = F.pad(leave, (0, padding), value=-1)

    def tree_mask(b, h, q, k):
        return (enter[k] <= enter[q]) & (enter[q] < leave[k])

    return create_block_mask(
        tree_mask, B=1, H=1, Q_LEN=n, KV_LEN=n, device=enter.device, BLOCK_SIZE=block_size, _compile=True
    )


def tree_flex_attention(module, query, key, value, attention_mask, *, tree_block_mask, scaling=None, **kwargs):
    out = _compiled_flex_attention(
        query,
        key,
        value,
        block_mask=tree_block_mask,
        scale=scaling,
        enable_gqa=query.shape[1] != key.shape[1],
        kernel_options={"BACKEND": "TRITON"},
    )
    return out.transpose(1, 2), None


def _tree_mask_formatter(**kwargs):
    return None


def register_tree_attention() -> None:
    AttentionInterface.register(TREE_ATTENTION, tree_flex_attention)
    AttentionMaskInterface.register(TREE_ATTENTION, _tree_mask_formatter)
