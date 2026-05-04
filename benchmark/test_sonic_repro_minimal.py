# ruff: noqa
"""Minimal self-contained repro scaffold for the sonic-moe sentinel/backward bug.

Dependencies: torch, kernels, nvidia-cutlass-dsl. NO transformers, NO distributed.

Background
----------
The kernel documents `expert_ids >= E` as a supported sentinel value used by EP
to mark non-local routing slots:

    functional/triton_kernels/__init__.py:174-177
        "Drop EP sentinels and out-of-tile lanes (both have `expert_ids >= E`).
         `safe_experts` remaps masked-off lanes to expert 0 ..."

    functional/triton_kernels/__init__.py:237-239
        "Sentinel lanes (expert == E) and the output-indexed tail [sum_valid, TK)
         ... are left untouched here — the caller zero-inits these arrays so
         downstream reads are well-defined."

    functional/__init__.py:452-453
        "Zero-init: EP sentinel lanes (expert == E) ... are not written by the
         routing kernel; downstream reads see well-defined zeros."

Observation
-----------
In production (Qwen3-30B-A3B SFT, DS-Z2 + EP=8 + chunked-CE, 8 H100 nodes), the
backward of `moe_general_routing_inputs` produces NaN gradients in `w1` and `w2`
when `expert_ids` contains the sentinel value `E`. We confirmed this with a
4-mode A/B test of our wrapper:

    | clamp expert_ids to [0,E-1] | masked_fill scores at sentinel | result    |
    |-----------------------------|--------------------------------|-----------|
    | YES                         | YES                            | trains    |
    | YES                         | NO                             | trains    |
    | NO                          | YES                            | NaN @ ~step 10 |
    | NO                          | NO                             | NaN @ ~step 10 |

→ clamping `expert_ids` is necessary; zeroing `router_scores` is redundant
  (RouterParallel already zeros them upstream).

This synthetic single-rank scaffold does NOT reliably reproduce the production
NaN — even with poisoned memory and 7/8 sentinel ratio at production-realistic
shapes, all three modes produce finite gradients. The bug appears to require
real-model gradient flow (48 stacked MoE layers + DTensor sharding + multi-step
optimizer) to manifest. We share this script as the API-contract demonstrator
(showing the wrapper pattern that fixes it) and would value upstream eyes on
what additional condition trips the gemm-backward.

Run on a single H100:
    source .venv/bin/activate
    python test_sonic_repro_minimal.py
"""

# ruff: noqa: T201, S101, E741
import torch
from kernels import get_kernel


KERNEL = get_kernel("IlyasMoutawwakil/sonic-moe", revision="main")
ActivationType = KERNEL.enums.ActivationType
moe_general_routing_inputs = KERNEL.moe_general_routing_inputs

DEVICE = "cuda"
DTYPE = torch.bfloat16

# Tiny-model EP=2 shapes (matches the 1-node TRL-stack repro that produced 10,000x grad inflation).
E = 2  # num_experts // EP_size (= 4 // 2)
H = 8  # tiny model hidden_size
I = 768  # tiny model moe_intermediate_size
T = 512  # matches the production tiny-model capture
TOP_K = 2
# Fraction of tokens with ALL slots as sentinel (= no valid expert chosen by router).
# In the production capture, 479/512 = 0.936 of tokens had this pattern.
SENTINEL_RATIO = 0.99


def poison_memory_pool() -> None:
    """Pre-fill the CUDA caching allocator with NaN so subsequent `empty_like`
    allocations reuse NaN-tainted memory."""
    big = [torch.empty(T * TOP_K * H, device=DEVICE, dtype=DTYPE).fill_(float("nan")) for _ in range(20)]
    del big
    torch.cuda.empty_cache()
    big = [torch.full((T * TOP_K * H,), float("nan"), device=DEVICE, dtype=DTYPE) for _ in range(10)]
    del big
    torch.cuda.synchronize()


def run(*, with_oob: bool, clamp: bool) -> dict:
    """One forward + backward step of `moe_general_routing_inputs`.

    Mirrors the wrapper used in production
    (https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/sonicmoe.py).
    """
    torch.manual_seed(0)

    # Concat layout: gate_up_proj is (E, 2*I, H), down_proj is (E, H, I).
    g = (torch.randn(E, 2 * I, H, device=DEVICE, dtype=DTYPE) * 0.02).requires_grad_(True)
    d = (torch.randn(E, H, I, device=DEVICE, dtype=DTYPE) * 0.02).requires_grad_(True)
    # Permute to (2*I, H, E) and (I, H, E) per kernel convention.
    w1 = g.permute(1, 2, 0)
    w2 = d.permute(1, 2, 0)

    x = torch.randn(T, H, device=DEVICE, dtype=DTYPE).requires_grad_(True)

    # Routing inputs: token_indices ascending (kernel requirement), router_scores
    # softmaxed, expert_ids in [0, E) by default.
    token_idx = torch.arange(T, device=DEVICE).unsqueeze(1).expand(-1, TOP_K).reshape(-1).int()
    router_scores = (
        torch.softmax(torch.randn(T, TOP_K, device=DEVICE, dtype=torch.float32), dim=-1)
        .reshape(-1)
        .to(DTYPE)
        .clone()
        .requires_grad_(True)
    )
    expert_ids = torch.randint(0, E, (T * TOP_K,), device=DEVICE, dtype=torch.int32)

    if with_oob:
        # Per-token sentinel pattern matching production capture:
        # most tokens have ALL slots sentinel (no valid expert assigned to them).
        # This is the failure mode under EP — when the global top-k for a token
        # selects only experts on other ranks, ALL of this rank's local routing
        # slots for that token become sentinels.
        n_all_sentinel_tokens = int(T * SENTINEL_RATIO)  # ~479 of 512 with 0.94 ratio
        sentinel_mask = torch.zeros(T, TOP_K, device=DEVICE, dtype=torch.bool)
        all_sent_idx = torch.randperm(T, device=DEVICE)[:n_all_sentinel_tokens]
        sentinel_mask[all_sent_idx, :] = True  # all TOP_K slots sentinel for these tokens
        sentinel_mask = sentinel_mask.reshape(-1)
        expert_ids = torch.where(sentinel_mask, torch.tensor(E, device=DEVICE, dtype=torch.int32), expert_ids)
        with torch.no_grad():
            router_scores.masked_fill_(sentinel_mask, 0.0)

    if clamp:
        expert_ids = expert_ids.clamp(0, E - 1)

    out, _ = moe_general_routing_inputs(
        x,
        router_scores,
        token_idx,
        expert_ids,
        w1,
        None,
        w2,
        None,
        E=E,
        activation_type=ActivationType.SWIGLU,
        stream_id=torch.cuda.current_stream(DEVICE).cuda_stream,
        is_inference_mode_enabled=False,
        concat_layout=True,
    )

    fake_grad_out = torch.randn_like(out)
    out.backward(fake_grad_out)

    return {
        "out_finite": torch.isfinite(out).all().item(),
        "out_nan": int(out.isnan().sum().item()),
        "out_norm": out.float().norm().item(),
        "g_grad_nan": int(g.grad.isnan().sum().item()),
        "d_grad_nan": int(d.grad.isnan().sum().item()),
        "g_grad_norm": g.grad.float().norm().item(),
        "d_grad_norm": d.grad.float().norm().item(),
        "g_grad_max": g.grad.float().abs().max().item(),
        "d_grad_max": d.grad.float().abs().max().item(),
        "g_grad_finite": torch.isfinite(g.grad).all().item(),
        "d_grad_finite": torch.isfinite(d.grad).all().item(),
        "h_grad_norm": x.grad.float().norm().item(),
        "h_grad_max": x.grad.float().abs().max().item(),
        "rs_grad_norm": router_scores.grad.float().norm().item(),
        "rs_grad_max": router_scores.grad.float().abs().max().item(),
    }


def fmt(name: str, r: dict) -> str:
    ok = r["out_finite"] and r["g_grad_finite"] and r["d_grad_finite"]
    status = "OK   " if ok else "FAIL "
    return (
        f"  [{status}] {name:30s}\n"
        f"        out: norm={r['out_norm']:.4g}\n"
        f"        g.grad: norm={r['g_grad_norm']:.4g}  max={r['g_grad_max']:.4g}\n"
        f"        d.grad: norm={r['d_grad_norm']:.4g}  max={r['d_grad_max']:.4g}\n"
        f"        h.grad: norm={r['h_grad_norm']:.4g}  max={r['h_grad_max']:.4g}\n"
        f"        rs.grad: norm={r['rs_grad_norm']:.4g}  max={r['rs_grad_max']:.4g}"
    )


print(f"sonic-moe sentinel scaffold  |  E={E} H={H} I={I} T={T} TOP_K={TOP_K} sentinel_ratio={SENTINEL_RATIO}")
print()
poison_memory_pool()

print("A) all expert_ids in [0, E-1]              — control")
print(fmt("A: in-range, no clamp", run(with_oob=False, clamp=False)))
print()
print("B) ~7/8 expert_ids == E (EP sentinel)      — exercises documented sentinel API")
print(fmt("B: with oob, no clamp", run(with_oob=True, clamp=False)))
print()
print("C) ~7/8 expert_ids == E + clamp(0,E-1)     — production workaround")
print(fmt("C: with oob, clamp", run(with_oob=True, clamp=True)))
print()
print("All three modes are expected to pass at this scale — the kernel's routing")
print("metadata correctly drops sentinels per its own comments. The production NaN")
print("requires a real-model gradient flow (48 MoE layers + DTensor + multi-step")
print("optimizer) that this synthetic scaffold does not reach.")
