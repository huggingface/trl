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
from itertools import pairwise

import torch
from torch.utils.checkpoint import checkpoint

from .tree_gated_delta_rule import tree_path_sources


_packed_forward = ContextVar("inkling2_packed_forward", default=False)


class Inkling2Tree:
    def __init__(self, n_tokens, cu_q, cu_k, kv_index, parents, conv_width):
        q_offsets, k_offsets = torch.stack((cu_q, cu_k)).tolist()
        self.segments = list(zip(pairwise(q_offsets), pairwise(k_offsets), strict=True))
        self.kv_index = kv_index

        # Prefix each segment with its convolution history: the tokens before its first one on its path, oldest
        # first. Past the root they read a zero entry, at `n_tokens`.
        history = conv_width - 1
        tokens = torch.arange(n_tokens, device=cu_q.device)
        segments = torch.arange(len(parents), device=cu_q.device)
        segment = segments.repeat_interleave(cu_q.diff(), output_size=n_tokens)
        self.conv_keep = tokens + (segment + 1) * history
        sources = tree_path_sources(cu_q, parents, n_tokens, conv_width)
        before = sources[cu_q[:-1].long(), 1:].flip(1)
        self.conv_gather = tokens.new_empty(n_tokens + len(parents) * history)
        self.conv_gather[self.conv_keep] = tokens
        history_slots = (cu_q[:-1] + segments * history)[:, None] + torch.arange(history, device=cu_q.device)
        self.conv_gather[history_slots] = before.masked_fill(before < 0, n_tokens)

    def attention(self, eager_attention, module, query, key, value, scaling, bias, sinks, sliding_window):
        def attend(query, key, value, bias, sinks, query_start, key_start):
            q_positions = torch.arange(query.size(2), device=query.device) + query_start
            k_positions = torch.arange(key.size(2), device=query.device) + key_start
            distance = q_positions[:, None] - k_positions
            extent = bias.size(-1)
            index = distance.clamp(0, extent - 1).expand(*bias.shape[:2], -1, -1)
            position_bias = bias.gather(-1, index)
            position_bias = position_bias.masked_fill((distance < 0) | (distance >= extent), 0)
            visible = distance >= 0
            if sliding_window is not None:
                visible = visible & (distance < sliding_window)
            mask = query.new_zeros(distance.shape).masked_fill(~visible, -torch.inf)
            output, _ = eager_attention(
                module, query, key, value, mask, scaling, position_bias=position_bias, sinks=sinks
            )
            return output

        # Bound the eager attention tile, and recompute large segments instead of retaining their scores.
        outputs = []
        chunk_size = 128
        bias = bias.transpose(1, 2)
        for (q_start, q_end), (k_start, k_end) in self.segments:
            path_index = self.kv_index[k_start:k_end]
            path_key = key.index_select(2, path_index)
            path_value = value.index_select(2, path_index)
            prefix = len(path_index) - (q_end - q_start)
            window = sliding_window if sliding_window is not None else len(path_index)
            recompute = len(path_index) * (q_end - q_start) > chunk_size**2

            for start in range(q_start, q_end, chunk_size):
                end = min(start + chunk_size, q_end)
                path_start = max(0, prefix + start - q_start - window + 1)
                path_end = prefix + end - q_start
                args = (
                    query[:, :, start:end],
                    path_key[:, :, path_start:path_end],
                    path_value[:, :, path_start:path_end],
                    bias[:, :, start:end],
                    sinks[:, :, start:end],
                    prefix + start - q_start,
                    path_start,
                )
                outputs.append(
                    checkpoint(attend, *args, use_reentrant=False, preserve_rng_state=False)
                    if recompute
                    else attend(*args)
                )

        return torch.cat(outputs, dim=1), None


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


@cache
def patch_tree_inkling2(model_type):
    if model_type == "inkling2_loop_text":
        from transformers.models.inkling2_loop import modeling_inkling2_loop as modeling

        text_model = modeling.Inkling2LoopTextModel
        relative_bias = modeling.Inkling2LoopRelativePositionBias
        _patch_convolution(modeling.Inkling2LoopShortConvolution)
        _patch_convolution(modeling.Inkling2LoopDynamicShortConvolution)
    else:
        from transformers.models.inkling2 import modeling_inkling2 as modeling

        text_model = modeling.Inkling2TextModel
        relative_bias = modeling.Inkling2RelativePositionBias
        _patch_convolution(modeling.Inkling2ShortConvolution)
        _patch_convolution(modeling.Inkling2DynamicShortConvolution)

    model_forward = text_model.forward
    get_relative_distance = modeling.get_relative_distance
    bias_forward = relative_bias.forward
    eager_attention = modeling.eager_attention_forward

    @wraps(model_forward)
    def tree_forward(self, *args, **kwargs):
        packed = "tree_kv_index" in kwargs
        if packed:
            kwargs["tree_inkling2"] = Inkling2Tree(
                kwargs["position_ids"].size(-1),
                kwargs["tree_cu_seqlens_q"],
                kwargs["tree_cu_seqlens_k"],
                kwargs["tree_kv_index"],
                kwargs["tree_segment_parents"],
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
    def tree_bias(self, hidden_states, relative_distance, position_ids):
        if relative_distance is not None:
            return bias_forward(self, hidden_states, relative_distance, position_ids)
        relative_states = self.linear(hidden_states).view(*hidden_states.shape[:-1], self.num_heads, -1)
        return self.norm(relative_states, position_ids) @ self.proj

    @wraps(eager_attention)
    def tree_attention(module, query, key, value, attention_mask, scaling, tree_inkling2=None, **kwargs):
        if tree_inkling2 is None:
            return eager_attention(module, query, key, value, attention_mask, scaling, **kwargs)
        return tree_inkling2.attention(
            eager_attention,
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
