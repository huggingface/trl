# ruff: noqa
"""Vacuum test: does 0 × NaN in forward leak NaN into backward gradient?

Hypothesis: PR #45621's grouped_mm pattern leaves sentinel rows of proj_out
uninitialized (= possibly NaN). The wrapper does:
    weighted_out = proj_out * sample_weights      # forward: NaN * 0 = NaN at sentinels
    weighted_out.masked_fill_(sentinel_mask, 0)   # forward: zero sentinels
    final = weighted_out[inv_perm].view(...).sum(dim=1)
This makes forward correct.

But in backward:
    d_proj_out      = d_weighted_out * sample_weights
    d_sample_weights = d_weighted_out * proj_out      # ← THE LEAK
At sentinel positions: d_weighted_out is some real number, proj_out is NaN
                       → d_sample_weights[sentinel] = NaN.
NaN flows back into router gate, optimizer zeros it, loss collapses.

Three modes test the hypothesis:
    A: proj_out fully initialized (all valid; baseline)
    B: proj_out has NaN at sentinel rows, NO pre-zero before mul   (PR #45621 pattern)
    C: proj_out has NaN at sentinel rows, pre-zero BEFORE mul       (proposed fix)
"""
# ruff: noqa: T201, S101, E741

import torch


torch.manual_seed(0)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

T = 8  # tokens
TOP_K = 2
S = T * TOP_K  # 16 routing slots
H = 4  # tiny hidden_dim


def run_mode(label: str, *, sentinel_makes_nan: bool, prezero_before_mul: bool):
    """One forward+backward simulating the wrapper's mul-then-mask path."""
    # Pretend proj_out came out of grouped_mm. Half of rows are "sentinels" — those with
    # sample_weights=0 — and may be NaN-tainted (sentinel_makes_nan=True) or fully init.
    sentinel_mask = torch.zeros(S, dtype=torch.bool, device=DEVICE)
    sentinel_mask[S // 2 :] = True  # second half of rows are sentinels

    sample_weights = torch.randn(S, device=DEVICE, dtype=torch.float32)
    sample_weights[sentinel_mask] = 0.0
    sample_weights = sample_weights.to(DTYPE).clone().requires_grad_(True)

    proj_out = torch.randn(S, H, device=DEVICE, dtype=DTYPE)
    if sentinel_makes_nan:
        # Simulate uninitialized memory after grouped_mm — fill with NaN to be deterministic.
        proj_out[sentinel_mask] = float("nan")
    proj_out = proj_out.clone().requires_grad_(True)

    if prezero_before_mul:
        # The proposed fix: zero proj_out at sentinel rows BEFORE multiplying.
        # We use masked_fill (not in-place) to keep autograd happy on a leaf w/ requires_grad.
        proj_out_used = proj_out.masked_fill(sentinel_mask.unsqueeze(-1), 0.0)
    else:
        proj_out_used = proj_out

    weighted = proj_out_used * sample_weights.unsqueeze(-1)  # (S, H)

    # Always zero the output rows for sentinels (that's the PR's existing post-mul mask).
    weighted_zero = weighted.masked_fill(sentinel_mask.unsqueeze(-1), 0.0)

    # Reduce-by-token (the .view().sum(dim=1) at the end of the wrapper).
    out = weighted_zero.view(T, TOP_K, H).sum(dim=1)

    fake_grad = torch.randn_like(out)
    out.backward(fake_grad)

    sw_grad = sample_weights.grad
    pj_grad = proj_out.grad
    sw_nan = torch.isnan(sw_grad).sum().item()
    pj_nan = torch.isnan(pj_grad).sum().item()

    print(f"[{label}]")
    print(f"  forward out finite     : {torch.isfinite(out).all().item()}")
    print(f"  sample_weights.grad nan: {sw_nan} of {S}")
    print(f"  proj_out.grad     nan  : {pj_nan} of {S * H}")
    print(f"  sw.grad sentinel slice : {sw_grad[sentinel_mask].float().tolist()}")
    print()


run_mode("A: proj_out fully init (baseline)", sentinel_makes_nan=False, prezero_before_mul=False)
run_mode(
    "B: proj_out NaN at sentinels, NO pre-zero (PR #45621 pattern)", sentinel_makes_nan=True, prezero_before_mul=False
)
run_mode(
    "C: proj_out NaN at sentinels, pre-zero BEFORE mul (proposed fix)",
    sentinel_makes_nan=True,
    prezero_before_mul=True,
)
