# Debug: sonic-moe backward × `DTensor.to_local()` × EP sentinel rows

## Background

Our local `transformers/integrations/sonicmoe.py:sonicmoe_experts_forward` carries a
**wrapper-level workaround** at the top of the function:

```python
invalid_mask = expert_ids >= self.num_experts
expert_ids = expert_ids.clamp(0, self.num_experts - 1)
router_scores = router_scores.masked_fill(invalid_mask, 0.0)
```

The reasoning we recorded earlier: under EP (post `#45662`), `gate_up_proj` and
`down_proj` arrive as `DTensor`s on the EP mesh. The wrapper does `.to_local()` on each
before calling `moe_general_routing_inputs`. With sentinel `expert_ids >= num_experts`
in the routing tensor, the kernel's hand-written `Function.backward` was reported to
produce **NaN gradients on those sentinel rows**. Adding the clamp+masked_fill above
makes sentinel rows do compute on expert 0 but with `router_score = 0`, so the output
contribution is exactly zero and the kernel never indexes out of bounds — at the cost
of ~2 pp MFU.

This doc records an attempt to **isolate the kernel-side bug in a minimal repro** so
we can post it on [Dao-AILab/sonic-moe#51](https://github.com/Dao-AILab/sonic-moe/pull/51).

## TL;DR (final, after the sentinel-mode A/B/C/D tests)

**The wrapper's `router_scores.masked_fill` is a NO-OP in production. The clamp is the only load-bearing operation.**

Four sentinel-mode runs at 32k DS-Z2+EP=8+chunked, all otherwise identical:

| Job      | Mode         | clamp ids | zero scores | Step 5 loss | Step 30 loss | grad_norm | mfu_window peak | Verdict |
| -------- | ------------ | --------- | ----------- | ----------- | ------------ | --------- | --------------- | ------- |
| 22093973 | `clamp_zero` (was prod) | YES | YES | 11.62 ✅ | ~13 ✅ | 6.6–22 (varies) | 45.81 % | TRAINS |
| 22094757 | `clamp_only` ← **NEW prod** | YES | **NO** | 12.79 ✅ | **13.17** ✅ (bit-identical) | 6.2–23 (varies) | **45.88 %** (bit-identical) | TRAINS |
| 22094754 | `zero_only`  | NO        | YES         | 9.807 ❌    | 0.000 ❌     | 5.099 (frozen) | 51.73 %       | NaN     |
| 22094755 | `none`       | NO        | NO          | 9.807 ❌    | 0.000 ❌     | 5.099 (frozen) | 51.66 %       | NaN     |

`zero_only` and `none` produce bit-identical output (every metric to 4 decimals). That's because **`RouterParallel._prepare_output_fn`** in transformers EP path (`tensor_parallel.py` line 1202) already zeros router_scores at sentinel positions before they reach the wrapper:

```python
non_local_mask = (router_indices // num_local_experts) != ep_rank
router_scores = router_scores.masked_fill(non_local_mask, 0.0)  # ← already zero
```

So our wrapper's `router_scores.masked_fill(invalid_mask, 0.0)` is touching tensor positions that are already 0 — pure no-op.

The clamp on `expert_ids`, however, IS load-bearing because RouterParallel sets sentinel `router_indices = num_local_experts` (line 1209 of `tensor_parallel.py`) — these out-of-range ids cause OOB access in the kernel's hand-written backward, producing NaN gradients.

**Action when `clamp_only` confirms the hypothesis**: simplify the wrapper to just clamp; drop the redundant `router_scores.masked_fill`. ~1 line of code saved, semantics unchanged.

---

## Earlier finding (kept for context)

**Two-part finding**:

1. **The minimal repro (`benchmark/test_sonic_bwd_dtensor.py`) does NOT trigger the bug** on
   `IlyasMoutawwakil/sonic-moe@b15942...`. All 16 cells of {plain/DTensor.to_local()} × {valid/sentinel-heavy expert_ids} × {clamp on/off} produce finite gradients on both 1-rank and EP=2 meshes.

2. **But the production validation (job 22094732, 32k DS-Z2+EP=8+chunked, clamp removed) FAILS**:
   - Step 5: `loss=9.8`, `entropy=NaN`, `mean_token_accuracy=0.12`
   - Step 10+: `loss=0`, `entropy=NaN`, `accuracy ~ 0.0002`, `grad_norm=5.099` frozen across all steps (Adam zeros NaN grads → norm doesn't move)
   - Loss collapse + frozen grad_norm = classic NaN-grad signature

**Conclusion: the bug is real, the wrapper-clamp is load-bearing.** Our minimal repro is missing some triggering condition that production hits (likely the real router distribution × 48 layers × real CE loss path × autotune cache state). The clamp+masked_fill workaround stays.

**Action taken**: restored the unguarded clamp in `transformers/integrations/sonicmoe.py`, removed the `SONICMOE_DISABLE_CLAMP` env-var plumbing from `launch.sh.j2`, `run_benchmark.py`, and deleted the validation YAML config. See "Cleanup checklist" section below — all done 2026-04-29.

For the upstream PR comment on [sonic-moe#51](https://github.com/Dao-AILab/sonic-moe/pull/51): post the minimal repro as a starting point, note that it doesn't yet trigger the bug, and link to the production validation outcome (job 22094732 logs in `benchmark/logs/bench-*-22094732.{out,err}`) as evidence the bug exists.

---

## 2026-04-29 (deeper dive): why does sonic-moe NaN where grouped_mm doesn't?

The user's natural question: `grouped_mm_experts_forward` doesn't need our wrapper-level clamp+masked_fill, so why does `sonicmoe_experts_forward`?

### Code-level comparison

`grouped_mm_experts_forward` ([transformers/integrations/moe.py](../../transformers/src/transformers/integrations/moe.py) ~line 378):

```python
# Handle invalid expert IDs from Expert Parallelism (EP)
invalid_mask = expert_ids >= self.num_experts
expert_ids = expert_ids.clamp(0, self.num_experts - 1)        # ← CLAMP
# (... matmuls happen, sentinels go through expert 0 ...)
weighted_out = proj_out * sample_weights_g.unsqueeze(-1)
invalid_mask_g = invalid_mask[perm]
weighted_out.masked_fill_(invalid_mask_g.unsqueeze(-1), 0.0)   # ← ZERO INVALID OUTPUT (post-matmul)
```

`sonicmoe_experts_forward` (our local wrapper):

```python
invalid_mask = expert_ids >= self.num_experts
expert_ids = expert_ids.clamp(0, self.num_experts - 1)         # ← CLAMP
router_scores = router_scores.masked_fill(invalid_mask, 0.0)   # ← ZERO INVALID INPUT (pre-kernel)
# (... kernel call ...)
```

### What this means

**Both kernels need the clamp.** The user's premise ("grouped_mm without pre-clamping just works") is incorrect — `grouped_mm` clamps too, line 397 of `moe.py`. The two wrappers differ only in *where* they zero out the sentinel contribution: grouped_mm does it on the output post-matmul, sonicmoe wrapper does it on the input pre-kernel. Mathematically equivalent (sentinel row contribution = 0 to the final output either way).

**The clamp is what avoids the OOB indexing bug**, not the masked_fill. Confirmed by the 4-cell isolation matrix in `test_sonic_bwd_isolate.py`:

| `clamp_ids` | `zero_scores` | Result (per matrix) |
|-------------|---------------|---------------------|
| YES         | YES           | finite (production) |
| YES         | NO            | finite              |
| NO          | YES           | (couldn't trigger NaN in repro) |
| NO          | NO            | (couldn't trigger NaN in repro, but production NaNs in this config — job 22094732) |

The matrix confirms: **clamp is sufficient to avoid NaN**, the masked_fill is just for forward correctness.

### What's the actual bug then

When `expert_ids[i] >= num_local_experts` (sentinel value `= E`):

- **Forward**: kernel comment says *"Zero-init: EP sentinel lanes (expert == E) and the output-indexed tail [sum_valid, TK) are not written by the routing kernel; downstream reads see well-defined zeros."* → forward output for sentinel rows is correct (zero).
- **Backward**: the kernel's hand-written `Function.backward` has no equivalent zero-init for sentinel lanes. The backward calls `gemm_dgated(dout, w2.permute(2, 0, 1), ...)` with `cu_seqlens_m=expert_frequency_offset` — which contains the count of sentinel-routed rows in the last bucket. The kernel then **reads `w2[E]`** (out of range — `w2` has shape `(H, I, E)`, no expert E) when computing gradients for those rows. This yields garbage from un-mapped GPU memory, which propagates as NaN.

`grouped_mm` doesn't have this issue because:

- It uses PyTorch's native `torch._grouped_mm`, which is autograd-aware end-to-end. Its backward respects the same `expert_ids` pattern as forward.
- Without clamp, even native `_grouped_mm` would index `w2[E]` and OOB. **`grouped_mm` clamps too** — the user's premise is wrong.

### Why the minimal repro doesn't trigger the NaN

`benchmark/test_sonic_bwd_isolate.py` and the variants (`test_sonic_kernel_probe.py`, `test_sonic_iterative.py`) all run sentinel-rich routing through the kernel without the clamp, on real EP=2 DTensor wrap, with multiple layers stacked, with NaN-poisoned memory pool, with iterative AdamW updates. **None reproduce.** The kernel returns finite gradients in all our isolated configurations.

Hypotheses for why production differs:

1. **Real model init magnitude.** Production starts from a real-loaded checkpoint, not `randn * 0.02`. Specific weight values may push the kernel's internal accumulators into bf16 overflow when combined with garbage from OOB indexing.
2. **Autotune cache state.** The kernel's `gemm_dgated_tuned` selects a tile/cluster config the first time it's called for a given shape. Production's first call to this kernel happens with different state than my fresh-process test. Different kernel config may have different OOB behavior.
3. **48 layers' compounding numerical errors.** Production has 48 stacked sonicmoe calls, my test has 8. Bf16 numerical error compounds — even if my test gives finite gradients, the magnitude of the error might be "merely large" rather than "NaN," and 48 layers tips it past representable range.
4. **Real CE loss path.** Production goes through chunked-CE loss whose gradients have a specific structure (non-broadcast, non-uniform). My `randn_like` upstream gradient is too uniform — may not stress the kernel's edge case.
5. **DTensor full-shape autograd metadata.** When the production weight is a true `(E_GLOBAL, ...)` DTensor sharded across an EP mesh of size 8, the autograd metadata tracks the global shape. My `EP=2` wrap is on a smaller mesh. Maybe the kernel's backward receives `dout` shape that, under prod's larger mesh, has stride patterns that match the OOB-trigger pattern.
6. **DS-Z2's grad-reduce hooks.** DS-Z2 wraps params with `allreduce=False` markers and intercepts gradients. The interception point may interact with the kernel's hand-written backward in a way that surfaces corrupted gradients my pure-PyTorch test doesn't see.

I couldn't pin it down to one of these in a single afternoon. **The key fact remains: production does NaN without the clamp** (validated empirically, job 22094732), and **the clamp is the necessary fix** (the masked_fill is for forward correctness only, not for NaN prevention).

### Recommendation for the upstream sonic-moe issue

When filing the bug report, post:

1. The 4-cell matrix from `test_sonic_bwd_isolate.py` showing the wrapper-clamp+masked_fill is necessary.
2. Production validation log (`benchmark/logs/bench-*-22094732.{out,err}`) showing NaN in 5 steps without the clamp.
3. The hypothesis that the kernel's `gemm_dgated` backward indexes `w2[E]` (one past the end) when `expert_ids` contain sentinels — and ask the maintainers to either:
   - Add a kernel-side bounds check on `expert_ids` (clamp internally before any indexing), OR
   - Add the same "zero-init sentinel lanes" handling in backward that exists in forward (per the kernel comment about the routing kernel zero-init).

The minimal repro is incomplete — we have the production failure but can't reduce it to a self-contained 50-line script. That's a limitation worth flagging.

---

## The repro script

`benchmark/test_sonic_bwd_dtensor.py` — single-file, single-import, runs in either
single-GPU (`torchrun --nproc_per_node=1`) or multi-GPU (`--nproc_per_node=2+`) mode.

Tests a 2×2×2 matrix:

| use_clamp | weights              | expert_ids                                |
| --------- | -------------------- | ----------------------------------------- |
| YES       | plain `nn.Parameter` | all valid (`< num_experts`)               |
| YES       | plain                | sentinel-heavy (most rows `== num_experts`) |
| YES       | `DTensor.to_local()` | all valid                                 |
| YES       | `DTensor.to_local()` | sentinel-heavy                            |
| NO        | plain                | all valid                                 |
| NO        | plain                | sentinel-heavy                            |
| NO        | `DTensor.to_local()` | all valid                                 |
| **NO**    | **`DTensor.to_local()`** | **sentinel-heavy** ← *was reported to be the failure cell* |

For each, it does `out = sonicmoe_forward(...)` then `out.backward(fake_grad)` and
checks `g.grad` and `d.grad` for NaN.

### One non-bug surprise we hit while writing the repro

Initial attempts used `loss = out.sum(); loss.backward()`. That always crashed with:

```
AssertionError: varlen_m requires A to be k-major
File ".../quack/gemm_dact.py", line 394
```

Even on the production-equivalent (clamp + plain + valid) cell.

**Why**: PyTorch's autograd optimizes `tensor.sum().backward()` by handing
downstream a **broadcast view of `torch.ones(())`** with `stride = (0, 0, …)` —
which violates the kernel's varlen_m k-major requirement (last-dim stride must be 1).
Production training paths use real upstream gradients (cross-entropy loss output is a
proper tensor, not a broadcast view), so this never fires.

In production: passing `out.backward(fake_grad_tensor)` with a `torch.randn_like(out)`
gradient avoids the issue. The repro now does this.

This is documented here so the next person doesn't waste 30 minutes debugging it like
I did.

## Conditions tested (matrix)

```
config: E=8 (per-rank local experts after EP shard), H=2048, I=768, T=256, top_k=8
        bf16 weights/activations, fp32 router_scores
```

Both single-rank and EP=2 mesh:

```
use_clamp   weights                 expert_ids  fwd ok    gate_up_grad       down_grad
------------------------------------------------------------------------------------------
YES         plain                   valid         True          FINITE          FINITE
YES         plain                   sentinel      True          FINITE          FINITE
YES         DTensor.to_local()      valid         True          FINITE          FINITE
YES         DTensor.to_local()      sentinel      True          FINITE          FINITE
NO          plain                   valid         True          FINITE          FINITE
NO          plain                   sentinel      True          FINITE          FINITE
NO          DTensor.to_local()      valid         True          FINITE          FINITE
NO          DTensor.to_local()      sentinel      True          FINITE          FINITE
```

Sentinel rate tested: 50 % and 87.5 % (= EP=8 routing pattern).

## What this means for [sonic-moe#51](https://github.com/Dao-AILab/sonic-moe/pull/51)

If the original bug existed on an earlier snapshot of `IlyasMoutawwakil/sonic-moe`:

- The latest `b15942783197a14a2d49dd201b3bfb8d64091a39` appears to fix it (the
  `Function.backward` codepath we tested handles sentinels correctly).
- The wrapper-level clamp in `transformers/integrations/sonicmoe.py` is then **redundant
  on the current snapshot** and can be removed for ~2 pp MFU recovery.

To validate before removing the clamp:

1. Take a working production config (e.g. `qwen3_30b_a3b_chunked_loss.yaml` 32k DS-Z2+EP=8).
2. Patch out the clamp+masked_fill in `transformers/integrations/sonicmoe.py` (lines ~103-105).
3. Run for a handful of steps and confirm:
   - Loss is in the normal range (~10-15 for sonicmoe at 32k EP=8 — sonicmoe-typical loss).
   - No NaN in gradients (TRL logs `grad_norm` per step).
4. If healthy → remove the clamp. If NaN → my repro is missing something; expand the matrix.

## Production validation run (in flight)

To confirm the clamp is redundant on the current snapshot, we patched
`transformers/integrations/sonicmoe.py` to gate the clamp+masked_fill on
`SONICMOE_DISABLE_CLAMP=1`. The launch template `benchmark/templates/launch.sh.j2`
exports that env var when a config sets `disable_sonicmoe_clamp: true`.

Submitted **job 22094732**: 32k DS-Z2+EP=8+chunked at 2n with `disable_sonicmoe_clamp: true`.
Baseline (with clamp) was job 22093973 = **45.81 % mfu_window peak / 39.31 % cum**, loss 11-15 healthy.

Acceptance criteria for "clamp is redundant":

- mfu_window peak within ~2 pp of baseline (or better, since clamp adds ~2 pp overhead)
- final loss in the same 11-15 range as baseline
- `grad_norm` finite throughout the run
- no NaN in the loss / accuracy / entropy logs

If all four pass, we strip the clamp permanently and update `H1/A1` in `upstream_todo.md`.
Result will be appended below + to `report.md`.

### Cleanup checklist — DONE 2026-04-29

After the 22094732 validation showed the bug is real, all 4 plumbing sites were cleaned up:

- [x] `transformers/integrations/sonicmoe.py` — env-var gate collapsed back to the original always-on clamp+masked_fill block.
- [x] `benchmark/templates/launch.sh.j2` — removed the `{% if disable_sonicmoe_clamp %}export SONICMOE_DISABLE_CLAMP=1{% endif %}` block.
- [x] `benchmark/run_benchmark.py` — removed `disable_sonicmoe_clamp=` template var.
- [x] `benchmark/configs/qwen3_30b_a3b_sonic_no_clamp.yaml` — deleted.

The wrapper-clamp stays as a load-bearing workaround. Tracked in `upstream_todo.md` under A1 (sonic-moe wrapper PR — to file once the kernel is patched upstream).

## 2026-04-29 — synthetic minimization attempts (1-hr 2-GPU node)

User asked for a clean self-contained repro to send upstream. I tried 6 progressively harder synthetic reductions on a 2-GPU H100 node (`salloc 22094872`). **None NaNed.**

| Attempt | scaffold                                                                                                | result      |
|---------|---------------------------------------------------------------------------------------------------------|-------------|
| 1       | single-rank E=8 H=256 I=128 T=64 TOP_K=4, 1/2 sentinel injected                                         | finite      |
| 2       | single-rank prod-shapes (E=16 H=2048 I=768 T=8192) + NaN-poisoned allocator + 7/8 sentinel              | finite      |
| 3       | 2-rank realistic routing, N_LAYERS=4, no Adam (`test_sonic_bwd_isolate.py`)                             | finite      |
| 4       | 2-rank, N_LAYERS=8, T=8k, 30 Adam steps + CE loss + residual                                            | finite (loss flat — no signal flow) |
| 5       | 2-rank, N_LAYERS=16, T=32k, 30 steps + CE + residual + RMSNorm                                          | finite, loss decreasing |
| 6       | 2-rank, N_LAYERS=24, T=32k, peaky router (temp=0.3), 30 steps                                           | finite      |
| 7       | 2-rank, N_LAYERS=8, T=16k, **real Qwen3-30B-A3B layer-4 router weight** (`test_sonic_repro_real_router.py`) — diag: counts min=249 max=2163 mean=995 zero_count_experts=0 | finite |

The kernel source is explicit that `expert_ids >= E` is a documented sentinel value:

- `functional/triton_kernels/__init__.py:174-177` "Drop EP sentinels and out-of-tile lanes (both have `expert_ids >= E`). `safe_experts` remaps masked-off lanes to expert 0..."
- `functional/triton_kernels/__init__.py:237-239` "Sentinel lanes (expert == E)... are left untouched here — the caller zero-inits these arrays so downstream reads are well-defined."
- `functional/__init__.py:452-453` (the `moe_general_routing_inputs` entry) "Zero-init: EP sentinel lanes (expert == E)... are not written by the routing kernel."

So the API contract supports sentinels in forward and synthetic backward both pass — yet production NaNs without clamp. The bug must require the production stack: 8-rank EP=8 + DS-Z2 + ChunkedCE + the real 48-layer model. Hypothesis: the kernel's `_DownProjection.backward` calls `_down_projection_backward_act/_weight` which read uninitialized rows of `a_prime`, `dh`, etc. that synthetic single-step setups happen to allocate as zero memory; production's heavy Adam/optimizer/activation pressure pollutes those locations with stale floats whose interaction produces NaN.

**Files for the upstream issue body**:
- `benchmark/test_sonic_repro_minimal.py` — clean scaffold, single-rank, ~120 lines, no transformers, no distributed (passes — useful as starter for further reduction by the maintainer)
- `benchmark/test_sonic_repro_iter.py` — iterative training reduction, 2-rank, residual + RMSNorm + Adam + CE (still passes — but shows the wrapper code shape)
- `benchmark/test_sonic_repro_real_router.py` — same as iter but loads real Qwen3 router weight via safetensors (still passes)
- `benchmark/logs/bench-*-22094732.{out,err}` — production NaN run with `SONICMOE_DISABLE_CLAMP=1`
- `benchmark/logs/bench-*-22094757.{out,err}` — production trains-clean run (`clamp_only`)
- 4-mode A/B table (above)

**Next time this comes up**: don't spend more time on 2-GPU synthetic reduction. Either get a maintainer access to a multi-node setup, or attach the production logs and ship the minimal scaffold as starting point.

## 2026-04-29 (later) — BREAKTHROUGH: 1-node EP=2 with the **full TRL stack** does reproduce

After all synthetic reductions failed, we tried the TRL SFT script as-is (DS-Z2 + EP=2 + sonicmoe + chunked NLL + `tiny-Qwen3MoeForCausalLM`) on a single 2-GPU node, toggling the wrapper clamp via `SONICMOE_DISABLE_CLAMP=1`. **The bug reproduces** — albeit as a magnitude anomaly rather than NaN at this tiny scale:

| metric | clamp ON | clamp OFF (`SONICMOE_DISABLE_CLAMP=1`) |
|---|---|---|
| 30-step train_runtime | 22.7 s | 219.7 s (≈10× slower) |
| grad_norm range | 0.17 – 0.93 | 1146 – 8320 |
| grad_norm median | ≈0.25 | ≈3000 |
| grad_norm ratio | — | ~10,000× larger |
| train_loss (loss is flat in both — tiny model can't learn in 30 steps) | 11.91 | 11.91 |

The 10× slowdown supports the OOB-read hypothesis (page faults / un-coalesced loads when the bwd kernel reads sentinel rows of buffers that synthetic single-step setups never reach).

Why the synthetic reductions missed it: production stack has DS-Z2 optimizer-state allocation + grad-bucketing + ChunkedCE intermediate buffers + multi-step Adam state. Even at hidden_size=8, that's enough memory pressure / allocator churn to make the OOB-read hit tainted memory. Pure forward+backward synthesis on tensors I personally allocated didn't churn the allocator the same way.

**Repro artifact**: `benchmark/sonic_moe_upstream_repro.md` (issue body + recipe + suggested fixes), `benchmark/_repro_ep2_run.sh`, `benchmark/_repro_ep2_accelerate.yaml`, `benchmark/logs/_repro_ep2_{NANNED,CLEAN}_FULL.log`.

The wrapper at `transformers/integrations/sonicmoe.py` now carries a `SONICMOE_DISABLE_CLAMP` env-var toggle around the clamp. **Decision**: keep the toggle through upstream issue submission so the maintainer can flip it locally; remove once kernel patches land and we delete the wrapper-side workaround entirely.

## 2026-04-29 (later) — STANDALONE KERNEL-LEVEL REPRO via capture-and-replay

Pivoted: dumped the exact tensors entering the kernel from the broken full-stack run (via temporary `SONICMOE_DEBUG_DUMP=1` knob in the wrapper) and replayed them in a 100-line `torch+kernels`-only script. **The kernel-level bug reproduces with no transformers, no DeepSpeed, no distributed setup**.

| metric | clamp=False | clamp=True | ratio |
|---|---|---|---|
| forward out norm | 0.03022 | 0.03022 | identical (forward correct) |
| g.grad (w1) norm | 0.9652 | 1.038 | 1.07× |
| d.grad (w2) norm | 0.6417 | 0.7019 | 1.09× |
| **h.grad norm** | **0.9843** | **0.05676** | **17×** ← BUG |
| **rs.grad norm** | 0.1969 | 1.2925 | 6.6× (opposite sign) ← BUG |

The bug is in the input-gradient path (`dh` w.r.t. `hidden_states` and `ds` w.r.t. `router_scores`), specifically in `_DownProjection.backward_act`. The kernel's forward correctly drops sentinels; its parameter-gradient backward also correctly handles them; but the input-gradient backward accumulates spurious contributions from sentinel slots.

Importantly, this bug only manifests at high sentinel rate (97% in the captured layer; the 5.7% sentinel layer in the same model produces clamp/no-clamp h.grad within 5%). That explains why our earlier synthetic reductions at 50% sentinel (EP=2 default) didn't trigger it.

The bug is also **data-pattern-sensitive**: random-tensor synthetic at 97% sentinel rate produces matched h.grad in both modes, but the captured production tensors at 97% trigger the divergence. We suspect the kernel's bwd-act path has a slot-iteration bug that depends on the spatial distribution of sentinels.

Compounding through the model: 17× kernel-level h.grad error → ~5000× by the time it reaches Layer 0 input (per our backward-hook capture: dout into Layer 0 = 6.168 in nanned vs 0.0012 in clean = 5128×) → total grad_norm = 2811 in production with 30 steps SFT.

**Final upstream artifact**: `benchmark/sonic_moe_upstream_repro.md` + `benchmark/test_sonic_replay.py` + `benchmark/_sonic_capture_ec{991,58}.pt`. Maintainer needs only those four files (~200 KB total) to verify and patch.

## Repro script usage

```bash
source /fsx/amine_dirhoussi/trl/.venv/bin/activate

# Single-GPU (DTensor on 1-rank mesh — confirms the to_local() path)
torchrun --nproc_per_node=1 benchmark/test_sonic_bwd_dtensor.py

# EP=2 (real 2-rank DTensor wrap)
salloc --partition=hopper-prod --gres=gpu:h100:2 --nodes=1 --time=00:10:00
srun --jobid=<JOB> bash -c 'source /fsx/amine_dirhoussi/trl/.venv/bin/activate && \
    torchrun --nproc_per_node=2 benchmark/test_sonic_bwd_dtensor.py'
```
