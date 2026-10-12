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
from functools import cache

import torch
from transformers import AttentionInterface
from transformers.integrations.flash_attention import get_target_dtype
from transformers.masking_utils import AttentionMaskInterface

from .layout import TreeLayout


TREE_ATTENTION = "tree_fa3"


@dataclass(frozen=True)
class TreeAttentionPlan:
    """
    Tree attention as plain FA3 varlen attention: each segment's queries attend to the K/V of its root-to-segment path.

    FA3 aligns the causal mask bottom-right when a sequence has fewer queries than keys, so a segment's queries, which
    are the last tokens of their path, see exactly their ancestors plus their own causal past. The K/V of shared
    prefixes is gathered once per descendant segment, but every other layer still runs once per unique token.

    Args:
        kv_index (`torch.Tensor`):
            Packed position of every K/V row, the paths of all segments laid end to end.
        cu_seqlens_q (`torch.Tensor`):
            Query boundaries of the segments in the packed row, shape `(S + 1,)`.
        cu_seqlens_k (`torch.Tensor`):
            Path boundaries in `kv_index`, shape `(S + 1,)`.
        max_seqlen_q (`int`):
            Longest segment.
        max_seqlen_k (`int`):
            Longest path.
    """

    kv_index: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    max_seqlen_q: int
    max_seqlen_k: int

    @classmethod
    def build(cls, layout: TreeLayout, device: torch.device) -> "TreeAttentionPlan":
        paths, kv_index, cu_seqlens_k = [], [], [0]
        for segment, parent in enumerate(layout.parents):
            path = (paths[parent] if parent != -1 else []) + list(
                range(layout.offsets[segment], layout.offsets[segment + 1])
            )
            paths.append(path)
            kv_index.extend(path)
            cu_seqlens_k.append(len(kv_index))
        offsets = layout.offsets
        return cls(
            kv_index=torch.tensor(kv_index, dtype=torch.long, device=device),
            cu_seqlens_q=torch.tensor(offsets, dtype=torch.int32, device=device),
            cu_seqlens_k=torch.tensor(cu_seqlens_k, dtype=torch.int32, device=device),
            max_seqlen_q=max(b - a for a, b in zip(offsets[:-1], offsets[1:], strict=True)),
            max_seqlen_k=max(len(path) for path in paths),
        )


@cache
def _flash_attn3():
    from kernels import get_kernel

    return get_kernel("kernels-community/flash-attn3", version=1)


def tree_attention(module, query, key, value, attention_mask, *, tree_attention_plan, scaling=None, **kwargs):
    plan = tree_attention_plan
    dtype = get_target_dtype(query, module) or query.dtype
    query, key, value = (x.transpose(1, 2)[0].to(dtype) for x in (query, key, value))
    out = _flash_attn3().flash_attn_varlen_func(
        query,
        key.index_select(0, plan.kv_index),
        value.index_select(0, plan.kv_index),
        plan.cu_seqlens_q,
        plan.cu_seqlens_k,
        plan.max_seqlen_q,
        plan.max_seqlen_k,
        softmax_scale=scaling,
        causal=True,
    )
    out = out[0] if isinstance(out, tuple) else out
    return out.unsqueeze(0), None


def _tree_mask_formatter(**kwargs):
    return None


def register_tree_attention() -> None:
    AttentionInterface.register(TREE_ATTENTION, tree_attention)
    AttentionMaskInterface.register(TREE_ATTENTION, _tree_mask_formatter)
