# sonicmoe: handle EP sentinels in `sonicmoe_experts_forward` (workaround)

> **Status note (2026-04-26):** [transformers PR #45621](https://github.com/huggingface/transformers/pull/45621) is the *upstream* fix for this issue. It updates the sonic-moe kernel itself to drop sentinel `expert_ids` in its metadata stage (no compute on sentinel rows) and re-points `_HUB_KERNEL_MAPPING["sonic-moe"]` from `kernels-community/sonic-moe` (v1, no sentinel handling) to `IlyasMoutawwakil/sonic-moe@main` (with sentinel handling). The PR is **open and not yet merged**, and the new kernel build is still in a personal repo.
>
> **Until that lands**, the cached kernel pulled by `lazy_load_kernel("sonic-moe")` (`kernels-community/sonic-moe` v1) does **not** handle sentinels — running sonicmoe under EP crashes with `cudaErrorIllegalAddress`. The change documented below is a **wrapper-level workaround** that makes the existing kernel safe to use under EP. Once PR #45621 lands and the new kernel is published to `kernels-community/sonic-moe`, this workaround should be **reverted** (the PR explicitly removes the wrapper-level clamp).

## TL;DR

`sonicmoe_experts_forward` in `src/transformers/integrations/sonicmoe.py` passes raw `expert_ids` straight to the CuteDSL kernel. With Expert Parallelism (`enable_expert_parallel=True`), `RouterParallel` produces sentinel ids (`>= num_experts`) for tokens that route to non-local experts. The current published kernel then does an out-of-bounds gather on `gate_up_proj[expert_ids]` → CUDA illegal memory access, sticky for the rest of the process.

The grouped_mm path used to handle this with `expert_ids.clamp(...) + masked_fill_(..., 0.0)` (now refactored further by PR #45621 to skip sentinel compute entirely). The sonicmoe wrapper has no equivalent on the current published kernel, and crashes the moment any rank sees a sentinel.

## Reproduction

- Model: `Qwen/Qwen3-30B-A3B`
- 2 nodes × 8× H100, FSDP2 with 2D mesh dp=2, tp/ep=8
- TRL `--enable_expert_parallel --expert_parallel_size 8 --experts_implementation sonicmoe`

Crash:

```
File "...sonic-moe/build/torch-cuda/quack/autotuner.py", line 84, in _gpu_warmup
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
torch.AcceleratorError: CUDA error: an illegal memory access was encountered
```

The illegal access happens during the kernel's first call (the autotuner warmup). The reported failure site is `torch.randn` because CUDA errors are sticky — the actual OOB happened slightly earlier inside `moe_general_routing_inputs`, but every subsequent CUDA op fails with the same code.

## Root cause

Inside `Qwen3MoeExperts` (and other MoE modules) when EP is enabled:

- `self.num_experts` becomes the **local** expert count after EP partition (e.g. 16 with EP=8 on 128 experts).
- `top_k_index` (after `RouterParallel._prepare_output_fn`) contains values up to `num_global_experts - 1 = 127`. Tokens routed to non-local experts carry sentinel ids `≥ num_local_experts`.
- `gate_up_proj` is locally shaped `[num_local_experts, ...]` after the EP partition.

The current sonicmoe wrapper:

```python
expert_ids = top_k_index.reshape(-1).int()
output, _ = moe_general_routing_inputs(
    hidden_states, router_scores, token_idx, expert_ids,
    w1, b1, w2, b2,
    E=self.num_experts,                # = 16
    activation_type=...,
    stream_id=torch.cuda.current_stream(device).cuda_stream,
    is_inference_mode_enabled=...,
    concat_layout=self.is_concatenated,
)
```

The kernel internally indexes `w1[expert_ids[i]]` → for sentinel ids `≥ 16` this reads past the local weight buffer.

## Workaround (until PR #45621 lands)

Clamp `expert_ids` in-bounds and zero out the corresponding `router_scores` so the kernel's weighted-mul drops sentinel rows to zero output. Both ops are pure local GPU kernels — no host sync, no collective. The kernel still does compute on sentinel rows; we just neutralize the output.

The proper fix is what PR #45621 does upstream: a kernel-side update so the metadata stage drops sentinels from the per-expert histogram and scatter indices, skipping their compute entirely (analogous to what PR #45621 also did for `grouped_mm_experts_forward`). With the new kernel, no wrapper changes are needed.

```diff
diff --git a/src/transformers/integrations/sonicmoe.py b/src/transformers/integrations/sonicmoe.py
--- a/src/transformers/integrations/sonicmoe.py
+++ b/src/transformers/integrations/sonicmoe.py
@@ -90,6 +90,13 @@ def sonicmoe_experts_forward(
     # Flatten — token_indices must be int32, sorted ascending (required by sonic-moe)
     token_idx = torch.arange(num_tokens, device=device).unsqueeze(1).expand(-1, num_top_k).reshape(-1).int()
     router_scores = top_k_weights.reshape(-1).to(hidden_states.dtype)
     expert_ids = top_k_index.reshape(-1).int()

+    # EP sentinel handling: out-of-range expert_ids (>= num_experts) come from RouterParallel
+    # marking tokens routed to non-local experts. Clamp them in-bounds so the kernel's
+    # `gate_up_proj[expert_ids]` gather stays valid; mask their router scores to zero so the
+    # weighted output for those rows is zero.
+    invalid_mask = expert_ids >= self.num_experts
+    expert_ids = expert_ids.clamp(0, self.num_experts - 1)
+    router_scores = router_scores.masked_fill(invalid_mask, 0.0)
+
     # Map activation function
```

## Important: do NOT guard with `if invalid_mask.any():`

The `.any()` returns a 1-element GPU tensor; Python's `if` on it implicitly calls `.item()` which forces a CUDA→CPU sync. Under FSDP2 + EP this caused a 600 s NCCL watchdog timeout on `_REDUCE_SCATTER_BASE` — the host-side stall on one rank desyncs the FSDP collective stream relative to the EP all-reduce stream, and the dp_shard reduce-scatter never lines up across ranks.

Empirically:

- With `if invalid_mask.any():` guard → NCCL deadlock during the first gradient sync.
- Unconditional `masked_fill_` → trains cleanly, window MFU 30.7 % at 16k 2n FSDP2 EP=8.

The "optimization" the guard was trying to make is also worthless: with EP > 1, every rank sees sentinels every step (~87.5 % of `expert_ids` are sentinels at EP=8); with EP = 1, the masked_fill is one cheap GPU kernel walking an all-False mask. Save a kernel, deadlock the cluster — bad trade.

**General rule:** never branch Python on a GPU tensor inside a forward/backward path of a distributed model.

## Validation

Tested on `Qwen/Qwen3-30B-A3B` at 16k context, 2 nodes × 8 H100, FSDP2 DP=2, EP=8, sdpa, 50 steps:

| Config                                | Cumulative MFU | Window MFU (steady-state) | TPS        |
| ------------------------------------- | -------------- | ------------------------- | ---------- |
| grouped_mm + sdpa, no EP              | 24.05 %        | 24.41 %                   | 65,500     |
| grouped_mm + sdpa + EP=8 + PR #45621  | 26.98 %        | 27.50 %                   | 74,080     |
| **sonicmoe + sdpa + EP=8 (this fix)** | **25.88 %**    | **30.67 %**               | **82,610** |

Window MFU is flat from step 5 onwards (30.0 – 30.8 % every logging step), no warmup curve. Loss 1.6, no NaN. The fix is functional and beats the grouped_mm + EP=8 path by +12 % at steady state.

## Relationship to PR #45621

PR #45621 ("Better Grouped GEMM + EP", open as of 2026-04-26) is the upstream fix and **supersedes the workaround above**:

- `src/transformers/integrations/moe.py` — refactors `grouped_mm_experts_forward` so sentinels are sorted to the tail and dropped by `histc(max=num_experts-1)`, then skipped by the grouped GEMM via offsets (no wasted compute, up to 5× kernel-only speedup at EP=8).
- `src/transformers/integrations/sonicmoe.py` — **deletes** the wrapper-level clamp ("leave `expert_ids` unclamped — the kernel's metadata stage drops sentinels"). Adds CUDA / SM90 / kernels-package availability checks.
- `src/transformers/integrations/hub_kernels.py` — temporarily redirects `_HUB_KERNEL_MAPPING["sonic-moe"]` from `kernels-community/sonic-moe` (v1) to `IlyasMoutawwakil/sonic-moe@main`. The new kernel build has the metadata-stage sentinel skip; it has not yet been republished to the official `kernels-community/sonic-moe` repo.

### When to revert this workaround

Once PR #45621 merges **and** the new kernel build is published to `kernels-community/sonic-moe`:

1. Remove the `invalid_mask + clamp + masked_fill` block from `sonicmoe_experts_forward` (apply the PR's deletion).
2. Restore the `_HUB_KERNEL_MAPPING["sonic-moe"]` entry to point at `kernels-community/sonic-moe` (the PR's redirect to the personal repo is itself temporary).
3. Re-run the EP benchmarks; expect a further bump over the wrapper-workaround numbers because the kernel will skip sentinel compute instead of computing-then-masking.

Until then, the wrapper workaround is the only way to get sonicmoe + EP working without a CUDA crash.

---

## Why "skip" doesn't beat "clamp" by as much as the kernel-only numbers suggest

Empirical observation from re-testing the patched `IlyasMoutawwakil/sonic-moe` kernel: `sonicmoe + clamp + mask` (this workaround, OLD kernel, **does full compute on sentinel rows**) at 16k EP=8 hit **30.67 % window MFU**. The patched kernel that *actually skips sentinels* hit 28.31 % at EP=2 and currently asserts in the backward at EP ≥ 4. So in the only apples-to-apples regime where both work, **the wasteful clamp path wins**.

The math suggests skip should win by a lot: at EP=8, only 1/8 of routes are local, so skip = 8× fewer MoE FLOPs than clamp. PR #45621's micro-bench shows up to 6.5× kernel-only speedup at EP=8 for that reason. End-to-end MFU shows ~10 % regression instead. The reason is that **the freed compute cycles aren't being used for anything**.

### The default scheduling — serialized expert-then-allreduce

```
GPU compute stream:  [attn][ MoE compute ][      idle     ][next attn]...
GPU comm stream:                          [ EP all-reduce ]
```

The MoE forward does `out = experts(x); out = all_reduce(out); return out`. Compute and the EP all-reduce are serialized on the same logical timeline. When you `skip` sentinels the kernel returns earlier, the all-reduce starts earlier, and the GPU's compute units sit idle until the all-reduce completes and the next attention block starts. **Skipped compute = idle GPU = no wall-time savings.**

Doing the wasted compute (clamp + mask) keeps the same GPU cycles busy with junk that gets multiplied by zero. Same wall time, less idle.

### What torchtitan / Megatron-LM do — overlapped expert + comm

```
GPU compute stream:  [attn][MoE compute][next layer's MoE compute][next attn]...
GPU comm stream:                       [ EP all-reduce of prev MoE  ]
```

Production MoE training frameworks issue the EP all-reduce on a **separate CUDA stream** and immediately launch the next layer's expert compute on the main stream. By the time the next layer needs the all-reduced result, NCCL has finished. The freed compute cycles from `skip` actually get filled by next-layer's compute → real wall-time win.

This requires:

- **Two CUDA streams**, with explicit `current_stream.wait_stream(comm_stream)` syncs at the points that read all-reduced values.
- A "dispatcher" abstraction that holds the async all-reduce handle and lets the next layer launch without blocking on it.
- Care interacting with FSDP2's prefetch hooks (FSDP2's all-gather/reduce-scatter are already on background streams; layering in EP all-reduce on top is non-trivial).
- Often a `torch.compile` pass that figures out the right cross-layer scheduling.

### Implication for our results

- **`sonicmoe + clamp + mask` winning over `grouped_mm + skip` is consistent**, not a paradox. In a serialized schedule, FLOPs are nearly free below the latency floor; per-FLOP kernel speed (sonicmoe is faster) and simpler control flow (no sort/histc/cumsum) matter more.
- **PR #45621's kernel-only speedup is real** but currently doesn't translate into end-to-end speedup on TRL.
- **To realize the skip's benefit, you'd need overlapped expert+comm scheduling**. That's a sizable refactor of the MoE forward path and FSDP2 hook integration in transformers — a level above the kernel work in PR #45621.

This is also why our report's earlier observation "EP is MFU-neutral for 30B" (~3 pp gap between EP=16 and no-EP) holds even with the patched kernel: the all-reduce overhead dominates whatever compute savings the kernel finds, until comms can overlap with compute.
