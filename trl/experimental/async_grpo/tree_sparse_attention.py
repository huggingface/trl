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

"""
DeepSeek Sparse Attention (DSA) over a tree-packed row.

DSA attends each query to the `index_topk` keys its indexer scores highest, so on a packed tree the structure folds
into the indices: the indexer only scores the keys on the query's own root-to-node path, and the kernel gathers
whatever it selected. Attention runs in MLA's latent space, where every head shares one 576-dim key (the normalized KV
latent and the rotary key) and one 512-dim value (the latent), so a gathered tile serves all heads of a query at once.
"""

import sys

import torch
import triton
import triton.language as tl


_PATCHED = set()


@triton.jit
def _fwd_kernel(
    q,
    k,
    v,
    idx,
    o,
    lse,
    scale,
    HQ: tl.constexpr,
    TOPK: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    BH: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    i_t, i_h, i_v = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    o_h = i_h * BH + tl.arange(0, BH)
    o_v = i_v * BV + tl.arange(0, BV)
    m_h, m_v = o_h < HQ, o_v < DV

    b_m = tl.full([BH], float("-inf"), dtype=tl.float32)
    b_acc = tl.zeros([BH], dtype=tl.float32)
    b_o = tl.zeros([BH, BV], dtype=tl.float32)
    for i in range(0, TOPK, BS):
        o_s = i + tl.arange(0, BS)
        b_i = tl.load(idx + i_t * TOPK + o_s, mask=o_s < TOPK, other=-1).to(tl.int64)
        m_s = b_i >= 0
        b_s = tl.zeros([BH, BS], dtype=tl.float32)
        for j in range(0, DK, BK):
            o_d = j + tl.arange(0, BK)
            m_d = o_d < DK
            b_q = tl.load(
                q + (i_t * HQ + o_h)[:, None] * DK + o_d[None, :], mask=m_h[:, None] & m_d[None, :], other=0.0
            )
            b_k = tl.load(k + b_i[:, None] * DK + o_d[None, :], mask=m_s[:, None] & m_d[None, :], other=0.0)
            b_s = tl.dot(b_q, tl.trans(b_k), b_s)
        b_s = tl.where(m_s[None, :], b_s * scale, float("-inf"))
        b_m_new = tl.maximum(b_m, tl.max(b_s, 1))
        b_r = tl.exp(b_m - b_m_new)
        b_p = tl.exp(b_s - b_m_new[:, None])
        b_acc = b_acc * b_r + tl.sum(b_p, 1)
        b_v = tl.load(v + b_i[:, None] * DV + o_v[None, :], mask=m_s[:, None] & m_v[None, :], other=0.0)
        b_o = b_o * b_r[:, None] + tl.dot(b_p.to(b_v.dtype), b_v)
        b_m = b_m_new
    b_o = b_o / b_acc[:, None]
    tl.store(
        o + (i_t * HQ + o_h)[:, None] * DV + o_v[None, :], b_o.to(o.dtype.element_ty), mask=m_h[:, None] & m_v[None, :]
    )
    if i_v == 0:
        tl.store(lse + i_t * HQ + o_h, b_m + tl.log(b_acc), mask=m_h)


@triton.jit
def _bwd_kernel(
    q,
    k,
    v,
    idx,
    do,
    lse,
    delta,
    dq,
    dk,
    dv,
    scale,
    HQ: tl.constexpr,
    TOPK: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    BH: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    i_t, i_h = tl.program_id(0), tl.program_id(1)
    o_h = i_h * BH + tl.arange(0, BH)
    m_h = o_h < HQ
    b_lse = tl.load(lse + i_t * HQ + o_h, mask=m_h, other=0.0)
    b_delta = tl.load(delta + i_t * HQ + o_h, mask=m_h, other=0.0)

    for i in range(0, TOPK, BS):
        o_s = i + tl.arange(0, BS)
        b_i = tl.load(idx + i_t * TOPK + o_s, mask=o_s < TOPK, other=-1).to(tl.int64)
        m_s = b_i >= 0
        b_s = tl.zeros([BH, BS], dtype=tl.float32)
        for j in range(0, DK, BK):
            o_d = j + tl.arange(0, BK)
            m_d = o_d < DK
            b_q = tl.load(
                q + (i_t * HQ + o_h)[:, None] * DK + o_d[None, :], mask=m_h[:, None] & m_d[None, :], other=0.0
            )
            b_k = tl.load(k + b_i[:, None] * DK + o_d[None, :], mask=m_s[:, None] & m_d[None, :], other=0.0)
            b_s = tl.dot(b_q, tl.trans(b_k), b_s)
        b_p = tl.where(m_s[None, :], tl.exp(b_s * scale - b_lse[:, None]), 0.0)
        b_dp = tl.zeros([BH, BS], dtype=tl.float32)
        for j in range(0, DV, BV):
            o_d = j + tl.arange(0, BV)
            m_d = o_d < DV
            b_do = tl.load(
                do + (i_t * HQ + o_h)[:, None] * DV + o_d[None, :], mask=m_h[:, None] & m_d[None, :], other=0.0
            )
            b_v = tl.load(v + b_i[:, None] * DV + o_d[None, :], mask=m_s[:, None] & m_d[None, :], other=0.0)
            b_dp = tl.dot(b_do, tl.trans(b_v), b_dp)
            b_dv = tl.dot(tl.trans(b_p).to(b_do.dtype), b_do)
            tl.atomic_add(dv + b_i[:, None] * DV + o_d[None, :], b_dv, mask=m_s[:, None] & m_d[None, :], sem="relaxed")
        b_ds = b_p * (b_dp - b_delta[:, None]) * scale
        for j in range(0, DK, BK):
            o_d = j + tl.arange(0, BK)
            m_d = o_d < DK
            b_q = tl.load(
                q + (i_t * HQ + o_h)[:, None] * DK + o_d[None, :], mask=m_h[:, None] & m_d[None, :], other=0.0
            )
            b_k = tl.load(k + b_i[:, None] * DK + o_d[None, :], mask=m_s[:, None] & m_d[None, :], other=0.0)
            b_dq = tl.dot(b_ds.to(b_k.dtype), b_k)
            b_dk = tl.dot(tl.trans(b_ds).to(b_q.dtype), b_q)
            tl.atomic_add(
                dq + (i_t * HQ + o_h)[:, None] * DK + o_d[None, :],
                b_dq,
                mask=m_h[:, None] & m_d[None, :],
                sem="relaxed",
            )
            tl.atomic_add(dk + b_i[:, None] * DK + o_d[None, :], b_dk, mask=m_s[:, None] & m_d[None, :], sem="relaxed")


class _SparseAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, idx, scale):
        n, hq, dk = q.shape
        dv = v.shape[1]
        topk = idx.shape[1]
        bh = max(16, min(triton.next_power_of_2(hq), 64))
        bv = min(triton.next_power_of_2(dv), 128)
        o = torch.empty(n, hq, dv, dtype=q.dtype, device=q.device)
        lse = torch.empty(n, hq, dtype=torch.float32, device=q.device)
        _fwd_kernel[(n, triton.cdiv(hq, bh), triton.cdiv(dv, bv))](
            q, k, v, idx, o, lse, scale, HQ=hq, TOPK=topk, DK=dk, DV=dv, BH=bh, BS=64, BK=64, BV=bv
        )
        ctx.save_for_backward(q, k, v, idx, o, lse)
        ctx.scale = scale
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, idx, o, lse = ctx.saved_tensors
        n, hq, dk = q.shape
        dv = v.shape[1]
        do = do.contiguous()
        delta = (o.float() * do.float()).sum(-1)
        dq = torch.zeros(n, hq, dk, dtype=torch.float32, device=q.device)
        dk_ = torch.zeros(n, dk, dtype=torch.float32, device=q.device)
        dv_ = torch.zeros(n, dv, dtype=torch.float32, device=q.device)
        _bwd_kernel[(n, triton.cdiv(hq, 16))](
            q,
            k,
            v,
            idx,
            do,
            lse,
            delta,
            dq,
            dk_,
            dv_,
            ctx.scale,
            HQ=hq,
            TOPK=idx.shape[1],
            DK=dk,
            DV=dv,
            BH=16,
            BS=64,
            BK=64,
            BV=64,
        )
        return dq.to(q.dtype), dk_.to(k.dtype), dv_.to(v.dtype), None, None


def sparse_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, idx: torch.Tensor, scale: float
) -> torch.Tensor:
    """
    Attention of every query over its own selected keys.

    Args:
        q (`torch.Tensor` of shape `(N, HQ, DK)`):
            Queries.
        k (`torch.Tensor` of shape `(N, DK)`), v (`torch.Tensor` of shape `(N, DV)`):
            One key and value per position, shared by all heads.
        idx (`torch.Tensor` of shape `(N, TOPK)`):
            Positions each query attends to, `-1` where it selected fewer than `TOPK`.
        scale (`float`):
            Softmax scale.

    Returns:
        `torch.Tensor` of shape `(N, HQ, DV)`.
    """
    return _SparseAttention.apply(q.contiguous(), k.contiguous(), v.contiguous(), idx.contiguous(), scale)


@torch.no_grad()
def tree_select(indexer, hidden_states, q_resid, position_embeddings, position_ids, tree) -> torch.Tensor:
    """
    The indexer's top-k selection on a tree-packed row: each position scores only the keys on its path at or before its
    depth, a chunk of queries at a time so the dense scores never span the whole row.

    Returns:
        `torch.Tensor` of shape `(N, TOPK)`: selected positions, `-1` past the path's length.
    """
    modeling = sys.modules[type(indexer).__module__]
    n = hidden_states.shape[1]
    cos, sin = position_embeddings
    q = indexer.wq_b(q_resid).view(1, n, indexer.n_heads, indexer.head_dim)
    q_rot, q_pass = torch.split(q, [indexer.qk_rope_head_dim, indexer.head_dim - indexer.qk_rope_head_dim], dim=-1)
    k = indexer.k_norm(indexer.wk(hidden_states)).unsqueeze(2)
    k_rot, k_pass = torch.split(k, [indexer.qk_rope_head_dim, indexer.head_dim - indexer.qk_rope_head_dim], dim=-1)
    q_rot, k_rot = modeling.apply_rotary_pos_emb(q_rot, k_rot, cos, sin, unsqueeze_dim=2)
    q = torch.cat([q_rot, q_pass], dim=-1)[0].float()  # [N, H, D]
    k = torch.cat([k_rot, k_pass], dim=-1)[0, :, 0].float()  # [N, D]
    weights = indexer.weights_proj(hidden_states.to(indexer.weights_proj.weight.dtype))[0].float()
    weights = weights * indexer.n_heads**-0.5  # [N, H]

    cu_q, cu_k, kv_index = tree["tree_cu_seqlens_q"], tree["tree_cu_seqlens_k"], tree["tree_kv_index"]
    n_segments = len(cu_q) - 1
    segment_of_key = torch.repeat_interleave(torch.arange(n_segments, device=k.device), (cu_k[1:] - cu_k[:-1]).long())
    on_path = torch.zeros(n_segments, n, dtype=torch.bool, device=k.device)
    on_path[segment_of_key, kv_index] = True
    segment_of_query = torch.repeat_interleave(
        torch.arange(n_segments, device=k.device), (cu_q[1:] - cu_q[:-1]).long()
    )
    depth = position_ids[0]

    topk = min(indexer.index_topk, n)
    selected = torch.empty(n, topk, dtype=torch.int32, device=k.device)
    chunk = max(1, 2**28 // (indexer.n_heads * n))  # keep the dense scores of a chunk around 1 GiB
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        scores = torch.relu(torch.matmul(q[start:end], k.T) * indexer.softmax_scale)  # [BQ, H, N]
        scores = torch.matmul(weights[start:end].unsqueeze(1), scores).squeeze(1)  # [BQ, N]
        allowed = on_path[segment_of_query[start:end]] & (depth[None, :] <= depth[start:end, None])
        scores = scores.masked_fill(~allowed, float("-inf"))
        values, indices = scores.topk(topk, dim=-1)
        selected[start:end] = indices.masked_fill(values == float("-inf"), -1).to(torch.int32)
    return selected


def tree_attention_forward(
    self, hidden_states, position_embeddings, attention_mask, past_key_values=None, position_ids=None, **kwargs
):
    """`DeepseekV32Attention.forward` over a tree-packed row, in MLA's latent space."""
    modeling = sys.modules[type(self).__module__]
    tree = {key: kwargs[key] for key in ("tree_kv_index", "tree_cu_seqlens_q", "tree_cu_seqlens_k")}
    n = hidden_states.shape[1]

    q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))
    q_states = self.q_b_proj(q_resid).view(1, n, self.num_heads, self.qk_head_dim).transpose(1, 2)
    q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
    compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
    kv_pass, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
    k_pass = self.kv_a_layernorm(kv_pass).view(1, 1, n, self.kv_lora_rank)
    k_rot = k_rot.view(1, 1, n, self.qk_rope_head_dim)
    cos, sin = position_embeddings
    q_rot, k_rot = modeling.apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)

    eye = torch.eye(self.kv_lora_rank, dtype=hidden_states.dtype, device=hidden_states.device)
    w = self.kv_b_proj(eye).T.reshape(self.num_heads, self.qk_nope_head_dim + self.v_head_dim, self.kv_lora_rank)
    w_k, w_v = torch.split(w, [self.qk_nope_head_dim, self.v_head_dim], dim=1)
    q_latent = torch.cat([torch.einsum("hnd,hdc->nhc", q_pass[0], w_k), q_rot[0].transpose(0, 1)], dim=-1)
    k_latent = torch.cat([k_pass[0, 0], k_rot[0, 0]], dim=-1)

    selected = tree_select(self.indexer, hidden_states, q_resid, position_embeddings, position_ids, tree)
    o_latent = sparse_attention(q_latent, k_latent, k_pass[0, 0], selected, self.scaling)
    attn_output = torch.einsum("nhc,hvc->nhv", o_latent, w_v).reshape(1, n, -1)
    return self.o_proj(attn_output), None


def patch_tree_dsa(model) -> bool:
    """
    Make the model's DeepSeek sparse attention layers follow the tree of a tree-packed forward.

    The attention module's forward is replaced: given the tree of [`~packing.TreeRow.pack`] it runs
    [`tree_attention_forward`], otherwise it runs unchanged.

    Returns:
        `bool`: whether the model's modeling module has such layers.
    """
    modeling = sys.modules[type(model).__module__]
    attention = vars(modeling).get("DeepseekV32Attention")
    if attention is None:
        return False
    if modeling in _PATCHED:
        return True
    original = attention.forward

    def forward(self, *args, **kwargs):
        if "tree_kv_index" not in kwargs:
            return original(self, *args, **kwargs)
        return tree_attention_forward(self, *args, **kwargs)

    attention.forward = forward
    _PATCHED.add(modeling)
    return True
