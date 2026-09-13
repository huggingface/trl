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

from functools import partial

import torch
from torch import nn
from torch.utils.checkpoint import CheckpointPolicy, create_selective_checkpoint_contexts


# Exact aten overloads for the SDPA backends (flash, memory-efficient, cuDNN, and the CPU fallback). Matched by
# identity, as recommended in https://pytorch.org/blog/activation-checkpointing-techniques (not every name exists in
# every torch build, hence the `getattr` default). Flash-attn's custom kernels are a separate case: they register
# under hashed namespaces (e.g. `_flash_attn2_cuda_f12afc9::varlen_fwd`) and so can only be matched by substring.
_ATEN_ATTENTION_PACKET_NAMES = (
    "_scaled_dot_product_flash_attention",
    "_scaled_dot_product_efficient_attention",
    "_scaled_dot_product_cudnn_attention",
    "_scaled_dot_product_flash_attention_for_cpu",
)


def _aten_attention_ops() -> set:
    return {
        packet.default
        for name in _ATEN_ATTENTION_PACKET_NAMES
        if (packet := getattr(torch.ops.aten, name, None)) is not None
    }


def _build_policy_fn(aten_attention_ops: set):
    def policy_fn(ctx, op, *args, **kwargs):
        if op in aten_attention_ops or "flash_attn" in str(op):
            return CheckpointPolicy.MUST_SAVE
        return CheckpointPolicy.PREFER_RECOMPUTE

    return policy_fn


def enable_selective_activation_checkpointing(model: nn.Module) -> None:
    """
    Enable eager selective activation checkpointing (SAC) on the model.

    Plain (full) activation checkpointing recomputes every op in the checkpointed region during the backward pass,
    including attention. Attention *backward* is irreducible, but its recompute is not: at long context, saving the
    attention output during the forward pass instead of recomputing it recovers most of the checkpointing slowdown for
    one extra hidden-state-sized tensor per layer.

    This wraps the model's `gradient_checkpointing_enable` so that, whenever gradient checkpointing is turned on (by
    the trainer, PEFT preparation, etc.), a policy-based [SAC context
    function](https://pytorch.org/docs/main/checkpoint.html#torch.utils.checkpoint.create_selective_checkpoint_contexts)
    is injected. The policy saves the attention op and recomputes everything else. Matching happens at the dispatcher
    level, so custom kernels that are not registered as torch ops are simply recomputed as usual. It runs fully eager
    (no `torch.compile` required) and forces non-reentrant checkpointing, which SAC relies on.

    Wrapping the model instance's method (rather than passing a `context_fn` through the config) keeps the
    non-serializable callable out of [`~transformers.TrainingArguments`] and covers every enable call site. The
    wrapper is idempotent, so calling this twice on the same model (e.g. across repeated `Trainer` inits in tests) is
    a no-op the second time.

    Args:
        model (`nn.Module`):
            Model on which to enable selective activation checkpointing. Must support `gradient_checkpointing_enable`.
    """
    if getattr(model.gradient_checkpointing_enable, "_is_sac_wrapped", False):
        return

    policy_fn = _build_policy_fn(_aten_attention_ops())
    context_fn = partial(create_selective_checkpoint_contexts, policy_fn)
    original_gradient_checkpointing_enable = model.gradient_checkpointing_enable

    def gradient_checkpointing_enable(gradient_checkpointing_kwargs: dict | None = None, **kwargs):
        gradient_checkpointing_kwargs = dict(gradient_checkpointing_kwargs or {})
        # SAC intercepts saved tensors, which only happens under non-reentrant checkpointing.
        gradient_checkpointing_kwargs["use_reentrant"] = False
        gradient_checkpointing_kwargs["context_fn"] = context_fn
        return original_gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs, **kwargs
        )

    gradient_checkpointing_enable._is_sac_wrapped = True
    model.gradient_checkpointing_enable = gradient_checkpointing_enable
