# Copyright 2026 The HuggingFace Team. All rights reserved.
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

from contextvars import ContextVar
from functools import cache, wraps

import torch
from torch.nn.attention.flex_attention import and_masks, create_block_mask
from transformers.integrations.flex_attention import compile_friendly_flex_attention, get_flex_attention_lse_kwargs
from transformers.utils import is_torch_greater_or_equal

from .tree_paths import tree_path_sources


_packed_forward = ContextVar("inkling2_packed_forward", default=False)
_compiled_create_block_mask = torch.compile(create_block_mask)


class Inkling2Tree:
    def __init__(self, cu_q, parents, depths, position_ids, conv_width):
        n_tokens = position_ids.numel()
        tokens = torch.arange(n_tokens, device=cu_q.device)
        segments = torch.arange(len(parents), device=cu_q.device)
        segment = segments.repeat_interleave(cu_q.diff(), output_size=n_tokens)

        # A segment's descendants follow it contiguously in depth-first order, up to the next segment no deeper than
        # it. A key is then seen by the queries from itself to the end of its segment's subtree.
        stop = (segments[None] > segments[:, None]) & (depths[None] <= depths[:, None])
        subtree_end = cu_q[torch.where(stop.any(1), stop.int().argmax(1), len(parents))]
        self.reach = subtree_end[segment].int()
        self.position = position_ids.view(-1).int()
        self.block_masks = {}

        # Prefix each segment with its convolution history: the tokens before its first one on its path, oldest
        # first. Past the root they read a zero entry, at `n_tokens`.
        history = conv_width - 1
        self.conv_keep = tokens + (segment + 1) * history
        sources = tree_path_sources(cu_q, parents, n_tokens, conv_width)
        before = sources[cu_q[:-1].long(), 1:].flip(1)
        self.conv_gather = tokens.new_empty(n_tokens + len(parents) * history)
        self.conv_gather[self.conv_keep] = tokens
        history_slots = (cu_q[:-1] + segments * history)[:, None] + torch.arange(history, device=cu_q.device)
        self.conv_gather[history_slots] = before.masked_fill(before < 0, n_tokens)

    def block_mask(self, sliding_window):
        """The row's tree mask, built once and shared by every layer with the same window."""
        if sliding_window not in self.block_masks:
            reach, position = self.reach, self.position

            def tree(b, h, q_idx, kv_idx):
                return (kv_idx <= q_idx) & (q_idx < reach[kv_idx])

            def window(b, h, q_idx, kv_idx):
                return position[q_idx] - position[kv_idx] < sliding_window

            mask_mod = tree if sliding_window is None else and_masks(tree, window)
            n_tokens = position.numel()
            self.block_masks[sliding_window] = _compiled_create_block_mask(
                mask_mod, None, None, n_tokens, n_tokens, device=position.device
            )
        return self.block_masks[sliding_window]

    def attention(self, module, query, key, value, scaling, relative_logits, sinks, sliding_window):
        """
        Inkling 2's attention over the tree in one FlexAttention call: the relative bias is a `score_mod` reading the
        `(batch, heads, tokens, extent)` profiles at each pair's distance along its path, and the sinks rescale the
        output by the mass they leave to the keys.
        """
        position, extent = self.position, relative_logits.size(-1)

        def relative_bias(score, b, h, q_idx, kv_idx):
            distance = position[q_idx] - position[kv_idx]
            bias = relative_logits[b, h, q_idx, distance.clamp(0, extent - 1)]
            return score + torch.where(distance < extent, bias, 0)

        # FlexAttention's kernels need a head dim of at least 16: zero columns change neither the scores nor the output
        head_dim = value.size(-1)
        pad = max(16 - head_dim, 0)
        query, key, value = (torch.nn.functional.pad(x, (0, pad)) for x in (query, key, value))
        output, aux = compile_friendly_flex_attention(
            query,
            key,
            value,
            training=module.training,
            score_mod=relative_bias,
            block_mask=self.block_mask(sliding_window),
            scale=scaling,
            enable_gqa=True,
            **get_flex_attention_lse_kwargs(True),
        )
        lse = aux.lse if is_torch_greater_or_equal("2.9.0") else aux
        output = output[..., :head_dim] * torch.sigmoid(lse - sinks.squeeze(-1)).unsqueeze(-1).to(output.dtype)
        return output.transpose(1, 2), None


def _patch_convolution(cls):
    forward = cls.forward

    @wraps(forward)
    def tree_conv(
        self,
        hidden_states,
        dynamic_input=None,
        cache=None,
        conv_mask=None,
        cache_offset=0,
        *,
        tree_inkling2=None,
        **kwargs,
    ):
        if tree_inkling2 is None:
            return forward(self, hidden_states, dynamic_input, cache, conv_mask, cache_offset, **kwargs)

        def with_history(features):
            if features is None:
                return None
            zero = torch.zeros_like(features[:, :1])
            return torch.cat((features, zero), dim=1).index_select(1, tree_inkling2.conv_gather)

        return forward(
            self,
            with_history(hidden_states),
            with_history(dynamic_input),
            cache,
            with_history(conv_mask),
            cache_offset,
            **kwargs,
        ).index_select(1, tree_inkling2.conv_keep)

    cls.forward = tree_conv


def patch_tree_inkling2(model) -> bool:
    """
    Make Inkling 2's attention and short convolutions follow the tree of a tree-packed forward: its `"hybrid"` and
    `"hybrid_sliding"` layers. Given the tree of [`~packing.TreeRow.pack`], the attention runs over the whole tree in
    one FlexAttention call and the convolutions read each token's own path; otherwise they run unchanged.

    Args:
        model ([`~transformers.PreTrainedModel`]):
            Model to patch.

    Returns:
        `bool`: whether the model is an Inkling 2 model whose modeling this can patch.
    """
    text_config = model.config.get_text_config()
    if text_config.model_type not in {"inkling2_text", "inkling2_loop_text"}:
        return False
    if text_config.attention_dropout:
        raise ValueError("Tree packing of Inkling 2 requires attention_dropout=0.")
    if not _patch_modeling(text_config.model_type):
        return False
    model.set_attn_implementation("eager")
    return True


@cache
def _patch_modeling(model_type):
    if model_type == "inkling2_loop_text":
        from transformers.models.inkling2_loop import modeling_inkling2_loop as modeling

        prefix = "Inkling2Loop"
    else:
        from transformers.models.inkling2 import modeling_inkling2 as modeling

        prefix = "Inkling2"
    # Before transformers 5.24, the relative distance was computed inside each attention layer
    if "get_relative_distance" not in vars(modeling):
        return False
    text_model = vars(modeling)[f"{prefix}TextModel"]
    relative_bias = vars(modeling)[f"{prefix}RelativePositionBias"]
    _patch_convolution(vars(modeling)[f"{prefix}ShortConvolution"])
    _patch_convolution(vars(modeling)[f"{prefix}DynamicShortConvolution"])

    model_forward = text_model.forward
    get_relative_distance = modeling.get_relative_distance
    bias_forward = relative_bias.forward
    eager_attention = modeling.eager_attention_forward

    @wraps(model_forward)
    def tree_forward(self, *args, **kwargs):
        packed = "tree_kv_index" in kwargs
        if packed:
            kwargs["tree_inkling2"] = Inkling2Tree(
                kwargs["tree_cu_seqlens_q"],
                kwargs["tree_segment_parents"],
                kwargs["tree_segment_depths"],
                kwargs["position_ids"],
                self.config.conv_kernel_size,
            )
            kwargs["attention_mask"] = dict(full_attention=None, sliding_attention=None, linear_attention=None)
            kwargs["use_cache"] = False
        token = _packed_forward.set(packed)
        try:
            return model_forward(self, *args, **kwargs)
        finally:
            _packed_forward.reset(token)

    def tree_relative_distance(*args, **kwargs):
        # The packed attention computes relative distances along each path.
        if not _packed_forward.get():
            return get_relative_distance(*args, **kwargs)

    @wraps(bias_forward)
    def tree_bias(self, hidden_states, relative_distance, position_ids, **kwargs):
        # Recent transformers pass `position_ids` among the forward's keyword arguments
        if relative_distance is not None:
            return bias_forward(self, hidden_states, relative_distance, position_ids=position_ids, **kwargs)
        relative_states = self.linear(hidden_states).view(*hidden_states.shape[:-1], self.num_heads, -1)
        return (self.norm(relative_states, position_ids) @ self.proj).transpose(1, 2)

    @wraps(eager_attention)
    def tree_attention(module, query, key, value, attention_mask, scaling, tree_inkling2=None, **kwargs):
        if tree_inkling2 is None:
            return eager_attention(module, query, key, value, attention_mask, scaling, **kwargs)
        return tree_inkling2.attention(
            module,
            query,
            key,
            value,
            scaling,
            kwargs["position_bias"],
            kwargs["sinks"],
            kwargs.get("sliding_window"),
        )

    text_model.forward = tree_forward
    modeling.get_relative_distance = tree_relative_distance
    relative_bias.forward = tree_bias
    modeling.eager_attention_forward = tree_attention
    return True
