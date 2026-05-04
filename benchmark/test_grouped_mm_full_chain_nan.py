# ruff: noqa
"""Trace where NaN first appears in PR #45621's grouped_mm_experts_forward.

Walks the FULL forward chain on a single GPU with sentinel-bearing routing,
checking for NaN at every intermediate. Reveals whether NaN comes from:
  1) The first `_grouped_linear` leaving sentinel rows uninitialized,
  2) `_apply_gate` (SwiGLU) propagating NaN at sentinels,
  3) The second `_grouped_linear` leaking NaN into valid rows via block-tiled reads,
  4) The multiply backward (the leak we already proved in test_grouped_mm_pr45621_nan_diag.py),
  5) Something else.

Compares 4 modes:
  M1: clamp pattern (current production)
  M2: PR #45621 raw  (no clamp, after-mask only)
  M3: PR #45621 + before-mask (the proposed one-liner fix)
  M4: PR #45621 + before-mask AT EVERY STAGE (zero sentinels after every grouped_mm)
"""

# ruff: noqa: T201, S101, E741
import torch
import torch.nn.functional as F
from transformers.integrations.moe import _grouped_mm


torch.manual_seed(0)
DEVICE = "cuda"
DTYPE = torch.bfloat16

# Production-realistic shapes for one rank of Qwen3-30B-A3B with EP=8.
E = 16  # local experts on this rank
H = 2048  # hidden_size
I = 768  # moe_intermediate_size
T = 1024  # tokens
TOP_K = 8


def _grouped_linear(input_, weight, offsets):
    """Mirror of moe.py::_grouped_linear (no bias, default layout).

    Default layout means weight is (E, output_dim, input_dim), so we need to
    transpose to (E, input_dim, output_dim) before the grouped_mm call —
    matching how moe.py::_grouped_linear does it for `is_transposed=False`.
    """
    return _grouped_mm(input_, weight.transpose(-2, -1), offs=offsets)


def swiglu_gate(proj_out_2I):
    """Mirror of self._apply_gate for gated experts: split, silu(gate)*up."""
    gate, up = proj_out_2I.chunk(2, dim=-1)
    return F.silu(gate) * up


def show(name, x):
    """Print NaN/inf/finite stats for a tensor."""
    n_nan = torch.isnan(x).sum().item()
    n_inf = torch.isinf(x).sum().item()
    if n_nan == 0 and n_inf == 0:
        norm = x.float().norm().item()
        print(f"  {name:32s}  shape={str(tuple(x.shape)):16s} finite=ok norm={norm:.4g}")
    else:
        n = x.numel()
        print(f"  {name:32s}  shape={str(tuple(x.shape)):16s} NaN={n_nan}/{n} inf={n_inf}/{n}")


def run(label, *, mode):
    """One forward pass with sentinel-bearing routing.

    mode in {"clamp", "pr45621", "before-mask", "before-mask-everywhere"}.
    """
    print(f"\n=== {label} ===")
    torch.manual_seed(0)

    # Build expert weights
    gate_up_proj = torch.randn(E, 2 * I, H, device=DEVICE, dtype=DTYPE) * 0.02
    down_proj = torch.randn(E, H, I, device=DEVICE, dtype=DTYPE) * 0.02
    gate_up_proj.requires_grad_(True)
    down_proj.requires_grad_(True)

    hidden_states = torch.randn(T, H, device=DEVICE, dtype=DTYPE).requires_grad_(True)

    # Routing: most tokens have ALL slots sentinel (matches Qwen3MoE under EP)
    # 7/8 of (token, slot) pairs are sentinel — high enough to trigger pattern-sensitive bugs.
    routing_logits = torch.randn(T, TOP_K, device=DEVICE, dtype=torch.float32) * 5.0  # peaky
    sample_w = F.softmax(routing_logits, dim=-1).reshape(-1).to(DTYPE)
    expert_ids = torch.randint(0, E, (T * TOP_K,), device=DEVICE, dtype=torch.int64)
    n_sentinel_tokens = int(T * 0.94)
    sentinel_mask = torch.zeros(T, TOP_K, device=DEVICE, dtype=torch.bool)
    sent_idx = torch.randperm(T, device=DEVICE)[:n_sentinel_tokens]
    sentinel_mask[sent_idx, :] = True
    sentinel_mask = sentinel_mask.reshape(-1)
    expert_ids = torch.where(sentinel_mask, torch.tensor(E, device=DEVICE, dtype=torch.int64), expert_ids)
    sample_w = sample_w.clone()
    sample_w[sentinel_mask] = 0.0  # RouterParallel does this upstream
    sample_w = sample_w.requires_grad_(True)

    print(f"  routing: {sentinel_mask.sum().item()} sentinels of {T * TOP_K} slots")

    # === Mirror grouped_mm_experts_forward up to the multiply ===
    if mode == "clamp":
        invalid = expert_ids >= E
        eids = expert_ids.clamp(0, E - 1)
    else:
        eids = expert_ids

    eids_g, perm = torch.sort(eids)
    sw_g = sample_w[perm]
    h_g = hidden_states[perm // TOP_K]
    show("input h_g", h_g.detach())

    # Histogram + offsets — drops sentinels in PR45621 mode (because eids include E)
    histc = eids_g.int()
    tokens_per_expert = (
        torch.histc(histc.float(), bins=E, min=0, max=E - 1)
        if DEVICE == "cpu"
        else torch.histc(histc, bins=E, min=0, max=E - 1)
    )
    offsets = torch.cumsum(tokens_per_expert, dim=0, dtype=torch.int32)
    print(
        f"  offsets[-1]={offsets[-1].item()}  total slots={T * TOP_K}  sentinels-skipped={T * TOP_K - offsets[-1].item()}"
    )

    # === Up projection: grouped_mm ===
    proj1 = _grouped_linear(h_g, gate_up_proj, offsets)  # (TK, 2*I)
    show("after up grouped_mm", proj1.detach())
    show("  ... at sentinel rows  ", proj1.detach()[eids_g >= E])
    show("  ... at valid rows     ", proj1.detach()[eids_g < E])

    if mode == "before-mask-everywhere":
        proj1 = proj1.masked_fill((eids_g >= E).unsqueeze(-1), 0.0)
        show("  zeroed sentinel rows  ", proj1.detach())

    # === SwiGLU gate ===
    proj2 = swiglu_gate(proj1)  # (TK, I)
    show("after _apply_gate", proj2.detach())
    show("  ... at sentinel rows  ", proj2.detach()[eids_g >= E])

    if mode == "before-mask-everywhere":
        proj2 = proj2.masked_fill((eids_g >= E).unsqueeze(-1), 0.0)

    # === Down projection: grouped_mm ===
    proj3 = _grouped_linear(proj2, down_proj, offsets)  # (TK, H)
    show("after down grouped_mm", proj3.detach())
    show("  ... at sentinel rows  ", proj3.detach()[eids_g >= E])
    show("  ... at valid rows     ", proj3.detach()[eids_g < E])

    # === Apply weights, restore order ===
    if mode in ("before-mask", "before-mask-everywhere"):
        proj3 = proj3.masked_fill((eids_g >= E).unsqueeze(-1), 0.0)
    weighted_out = proj3 * sw_g.unsqueeze(-1)
    if mode in ("clamp", "pr45621"):
        # post-mul mask; in clamp mode this is the original `invalid_mask[perm]`
        if mode == "clamp":
            inv_g = invalid[perm]
        else:
            inv_g = eids_g >= E
        weighted_out = weighted_out.masked_fill(inv_g.unsqueeze(-1), 0.0)

    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(perm.size(0), device=DEVICE)
    out_pre_sum = weighted_out[inv_perm]
    show("weighted_out (post mask)", out_pre_sum.detach())

    # Reduce: sum top_k contributions
    final = out_pre_sum.view(T, TOP_K, H).sum(dim=1)
    show("final out", final.detach())

    # === Backward ===
    fake_grad = torch.randn_like(final)
    final.backward(fake_grad)
    show("d hidden_states", hidden_states.grad)
    show("d gate_up_proj", gate_up_proj.grad)
    show("d down_proj", down_proj.grad)
    show("d sample_w", sample_w.grad)


for mode_label, mode in [
    ("M1 clamp (production)", "clamp"),
    ("M2 PR #45621 raw (after-mask only)", "pr45621"),
    ("M3 PR #45621 + before-mask on final proj (one-liner)", "before-mask"),
    ("M4 PR #45621 + before-mask at EVERY grouped_mm stage", "before-mask-everywhere"),
]:
    try:
        run(mode_label, mode=mode)
    except Exception as e:
        print(f"  CRASH: {type(e).__name__}: {str(e)[:200]}")
