# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

"""Opt-in Qwen3.5 adapter. The plan is an explicit forward argument, including during checkpoint recomputation."""

from types import MethodType

import torch.nn.functional as F
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5GatedDeltaNet

from ....kernels.tree_gated_delta_rule import tree_causal_conv1d, tree_chunk_gated_delta_rule


def qwen3_5_tree_gdn(module, hidden_states, plan):
    """Qwen3.5's training GDN path with tree convolution and tree chunk recurrence."""
    batch, length, _ = hidden_states.shape
    qkv = tree_causal_conv1d(
        module.in_proj_qkv(hidden_states),
        module.conv1d.weight,
        module.conv1d.bias,
        plan,
        # Transformers' own conv runs the activation in the convolution's dtype, not fp32: it calls
        # `F.conv1d` on `hidden_states.to(weight.dtype)` and applies `ACT2FN` to that result. Matching it
        # is what keeps a tree-packed forward numerically the same as a sequence-packed one.
        activation_in_fp32=False,
    )
    q, k, v = qkv.split([module.key_dim, module.key_dim, module.value_dim], dim=-1)
    q = q.reshape(batch, length, module.num_k_heads, module.head_k_dim)
    k = k.reshape(batch, length, module.num_k_heads, module.head_k_dim)
    v = v.reshape(batch, length, module.num_v_heads, module.head_v_dim)
    beta = module.in_proj_b(hidden_states).sigmoid()
    g = -module.A_log.float().exp() * F.softplus(module.in_proj_a(hidden_states).float() + module.dt_bias)
    out = tree_chunk_gated_delta_rule(q, k, v, g, beta, plan)
    z = module.in_proj_z(hidden_states).reshape(-1, module.head_v_dim)
    out = module.norm(out.reshape(-1, module.head_v_dim), z).reshape(batch, length, module.value_dim)
    return module.out_proj(out)


def _tree_gdn_forward(self, hidden_states, cache_params=None, attention_mask=None, tree_gdn_plan=None):
    if tree_gdn_plan is None:
        return Qwen3_5GatedDeltaNet.forward(self, hidden_states, cache_params, attention_mask)
    if cache_params is not None:
        raise ValueError("Tree GDN is a training path; use_cache must be False.")
    return qwen3_5_tree_gdn(self, hidden_states, tree_gdn_plan)


def _tree_decoder_forward(
    self,
    hidden_states,
    position_embeddings,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    tree_gdn_plan=None,
    **kwargs,
):
    if tree_gdn_plan is None or self.block_type == "full_attention":
        return Qwen3_5DecoderLayer.forward(
            self,
            hidden_states,
            position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            **kwargs,
        )
    if past_key_values is not None:
        raise ValueError("Tree GDN is a training path; use_cache must be False.")
    # Invoke the module, so FSDP's pre/post-forward hooks also run when the GDN block itself is sharded.
    hidden_states = hidden_states + self.linear_attn(
        self.input_layernorm(hidden_states),
        tree_gdn_plan=tree_gdn_plan,
    )
    return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


def enable_qwen3_5_tree_gdn(model):
    """Enable `tree_gdn_plan` on a Qwen3.5 model without changing weights or state-dict keys.

    Pass the same plan and existing `tree_block_mask` on every packed forward, with `use_cache=False`. Ordinary
    forwards (without a plan) retain Transformers' implementation. Explicit kwargs also survive checkpointing.
    """
    for module in model.modules():
        if isinstance(module, Qwen3_5DecoderLayer):
            module.forward = MethodType(_tree_decoder_forward, module)
        elif isinstance(module, Qwen3_5GatedDeltaNet):
            module.forward = MethodType(_tree_gdn_forward, module)
