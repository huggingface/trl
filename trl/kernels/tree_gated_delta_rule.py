# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Tree GDN: vendored fused FLA kernels and a separate FP32 reference, with no FLA runtime dependency.

The chunk equations follow Gated DeltaNet's WY formulation (https://huggingface.co/papers/2412.06464).
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .tree_gdn_scan import TreeScan


@dataclass(frozen=True)
class TreeGDNPlan:
    """Device topology built once per packed forest, outside model layers.

    `levels` groups linear segments by depth. `chunk_tokens` pads each segment independently to chunks of 64;
    `output_tokens` removes those pads. `conv_indices` contains causal ancestors, oldest first. Gather indices use
    `num_tokens` for a zero-padding token. State storage is per chunk and per segment, never per token.
    """

    levels: tuple[torch.Tensor, ...]
    parents: torch.Tensor
    chunk_offsets: torch.Tensor
    chunk_tokens: torch.Tensor
    output_tokens: torch.Tensor
    child_offsets: torch.Tensor
    children: torch.Tensor
    conv_indices: tuple[torch.Tensor, ...]
    cu_seqlens: torch.Tensor
    chunk_indices: torch.Tensor

    @classmethod
    def build(cls, offsets, parents, device, conv_kernel_size=4):
        """Build from CPU segment boundaries and parent IDs (the existing `TreeLayout` fields)."""
        if len(offsets) != len(parents) + 1 or not parents or offsets[0] != 0:
            raise ValueError("Expected a nonempty segment forest with offsets starting at zero.")
        depths, groups, token_parents = [], [], []
        for s, p in enumerate(parents):
            if p < -1 or p >= s or offsets[s + 1] <= offsets[s]:
                raise ValueError("Segments must be nonempty and parents must precede children.")
            depth = 0 if p == -1 else depths[p] + 1
            depths.append(depth)
            if depth == len(groups):
                groups.append([])
            groups[depth].append(s)
            token_parents.append(offsets[p + 1] - 1 if p != -1 else offsets[-1])
            token_parents.extend(range(offsets[s], offsets[s + 1] - 1))

        chunk_offsets, chunk_tokens, output_tokens = [0], [], []
        children = [[] for _ in parents]
        for s, parent in enumerate(parents):
            length = offsets[s + 1] - offsets[s]
            output_tokens.extend(range(len(chunk_tokens), len(chunk_tokens) + length))
            chunk_tokens.extend(range(offsets[s], offsets[s + 1]))
            chunk_tokens.extend([offsets[-1]] * (-length % 64))
            chunk_offsets.append(len(chunk_tokens) // 64)
            if parent != -1:
                children[parent].append(s)
        child_offsets = [0]
        for row in children:
            child_offsets.append(child_offsets[-1] + len(row))

        ancestors = list(range(offsets[-1]))
        token_parents.append(offsets[-1])
        conv_indices = []
        for _ in range(conv_kernel_size):
            conv_indices.append(torch.tensor(ancestors, device=device))
            ancestors = [token_parents[t] for t in ancestors]
        return cls(
            tuple(torch.tensor(segments, device=device) for segments in groups),
            torch.tensor(parents, device=device),
            torch.tensor(chunk_offsets, device=device),
            torch.tensor(chunk_tokens, device=device),
            torch.tensor(output_tokens, device=device),
            torch.tensor(child_offsets, device=device),
            torch.tensor([s for row in children for s in row], device=device, dtype=torch.long),
            tuple(reversed(conv_indices)),
            torch.tensor(offsets, device=device, dtype=torch.long),
            torch.tensor(
                [(s, c) for s in range(len(parents)) for c in range(chunk_offsets[s + 1] - chunk_offsets[s])],
                device=device,
                dtype=torch.long,
            ),
        )


@torch.compile(dynamic=True, fullgraph=True)
def _chunk_algebra(q, k, v, g, beta, chunk_tokens, scale, normalize):
    # Work in fp32 through the triangular solve and state updates, rounding only the final output.
    def chunks(x):
        x = F.pad(x, (0, 0, 0, 0, 0, 1)) if x.ndim == 4 else F.pad(x, (0, 0, 0, 1))
        x = x.index_select(1, chunk_tokens).float()
        return x.reshape(-1, 64, *x.shape[2:]).transpose(1, 2)

    q, k, v, g, beta = [chunks(x) for x in (q, k, v, g, beta)]
    if normalize:
        q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
        k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    repeat = v.shape[1] // q.shape[1]
    q, k = q.repeat_interleave(repeat, dim=1), k.repeat_interleave(repeat, dim=1)
    q = q * scale
    g = g.cumsum(-1)
    lower = torch.ones(64, 64, device=q.device, dtype=torch.bool).tril()
    # Mask before exp: upper-triangular differences can overflow for large negative decays.
    decay = (g[..., :, None] - g[..., None, :]).masked_fill(~lower, 0).exp().masked_fill(~lower, 0)
    system = ((k * beta[..., None]) @ k.transpose(-1, -2) * decay).tril(-1)
    system = system + torch.eye(64, device=q.device)
    rhs = torch.cat((k * (beta * g.exp())[..., None], v * beta[..., None]), dim=-1)
    wu = torch.linalg.solve_triangular(system, rhs, upper=False, unitriangular=True)
    w, u = wu.split((k.shape[-1], v.shape[-1]), dim=-1)
    kbar = k * (g[..., -1:] - g).exp()[..., None]
    qbar = q * g.exp()[..., None]
    attention = (q @ k.transpose(-1, -2)) * decay
    return qbar, attention, kbar, w, u, g[..., -1].exp()


@torch.compile(dynamic=True, fullgraph=True)
def _chunk_output(qbar, attention, h, vnew, output_tokens, dtype):
    out = qbar @ h + attention @ vnew
    out = out.transpose(1, 2).reshape(1, -1, out.shape[1], out.shape[-1])
    return out.index_select(1, output_tokens).to(dtype)


def reference_tree_gated_delta_rule(q, k, v, g, beta, plan, *, scale=None, use_qk_l2norm_in_kernel=True):
    """Run GDN on a packed forest, returning outputs in packed order.

    Inputs use `(1, tokens, heads, dim)` layout; gates have no final dimension. `g` contains log decay and `beta`
    activated update strengths. Parents precede children in forward, children precede parents in backward. The scan
    sums branch gradients explicitly. Q/K head dimensions up to 128 cover Qwen3.5; value heads may outnumber key heads.
    """
    if q.shape[0] != 1 or q.shape[1] != plan.output_tokens.numel():
        raise ValueError("Tree GDN expects one flattened forest matching the execution plan.")
    if q.shape[-1] > 128 or v.shape[2] % q.shape[2] != 0:
        raise ValueError("Expected key dimensions <= 128 and a whole number of value heads per key head.")
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    with torch.autocast(device_type="cuda", enabled=False):
        qbar, attention, kbar, w, u, decay = _chunk_algebra(
            q,
            k,
            v,
            g,
            beta,
            plan.chunk_tokens,
            scale,
            use_qk_l2norm_in_kernel,
        )
        h, vnew = TreeScan.apply(kbar, w, u, decay, plan)
        return _chunk_output(qbar, attention, h, vnew, plan.output_tokens, v.dtype)


def tree_chunk_gated_delta_rule(q, k, v, g, beta, plan, *, scale=None, use_qk_l2norm_in_kernel=True):
    """FLA's fused chunk kernels with tree-native state propagation and branch-gradient reduction.

    Inputs are `(1, tokens, heads, dim)`; gates are `(1, tokens, value_heads)`. The plan supplies segment boundaries
    and chunk indices directly: no token padding, duplicated prefixes, or per-depth tensor gathers are needed.
    """
    from ._tree_gdn_fla.tree import TreeGatedDeltaRule

    if q.shape[0] != 1 or q.shape[1] != plan.output_tokens.numel():
        raise ValueError("Tree GDN expects one flattened forest matching the execution plan.")
    if q.shape[-1] > 128 or v.shape[2] % q.shape[2] != 0:
        raise ValueError("Expected key dimensions <= 128 and a whole number of value heads per key head.")
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    return TreeGatedDeltaRule.apply(q, k, v, g, beta, plan, scale, use_qk_l2norm_in_kernel)


@torch.compile(dynamic=True, fullgraph=True, options={"emulate_precision_casts": True})
def _tree_causal_conv1d(x, weight, bias, conv_indices, activation_in_fp32):
    # Cast before gathering: backward then sums all ancestor contributions in fp32 before rounding once.
    padded = F.pad(x.float(), (0, 0, 0, 1))
    weight = weight.float()
    # Accumulate in fp32, as ordinary convolution does.
    out = torch.zeros_like(x, dtype=torch.float32)
    for tap, indices in enumerate(conv_indices):
        out = out + padded.index_select(1, indices) * weight[:, 0, tap]
    if bias is not None:
        out = out + bias.float()
    # causal-conv1d fuses SiLU before rounding; torch.nn.Conv1d rounds its output before SiLU.
    if not activation_in_fp32:
        out = out.to(x.dtype)
    return F.silu(out).to(x.dtype)


def tree_causal_conv1d(x, weight, bias, plan, *, activation_in_fp32=False):
    """Depthwise Conv1D + SiLU; `x` is `(1, tokens, channels)`, `weight` is `(channels, 1, width)`."""
    if weight.shape[-1] != len(plan.conv_indices):
        raise ValueError("Convolution width must match the execution plan.")
    return _tree_causal_conv1d(x, weight, bias, plan.conv_indices, activation_in_fp32)
