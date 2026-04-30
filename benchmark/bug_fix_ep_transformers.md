# Bug Fix: Expert Parallelism in Transformers

## Three bugs, zero test coverage

Expert Parallelism in transformers has three bugs that combine to produce silently wrong results. All are in the weight-loading, routing, and expert forward paths. None were caught because **there are zero tests for EP correctness** — every test uses `tp_plan="auto"` (regular TP) which bypasses `RouterParallel` and EP sentinel handling entirely.

---

## Bug 1: RouterParallel shape mismatch

**File**: `src/transformers/integrations/tensor_parallel.py`, `RouterParallel._prepare_output_fn` (line 1094)

### What it does

When EP is enabled, the router produces three tensors:

```
router_logits:  (seq, num_experts)   e.g. (3, 128)
router_scores:  (seq, top_k)         e.g. (3, 8)     ← one weight per selected expert
router_indices: (seq, top_k)         e.g. (3, 8)     ← global expert IDs
```

`RouterParallel._prepare_output_fn` is supposed to remap these from global to local expert space. The index remapping is correct — it maps global indices to local `[0, num_local_experts-1]` and sets non-local indices to a sentinel. But the **score handling** is wrong.

### The bug (lines 1136-1137)

```python
router_scores = torch.zeros_like(router_logits).scatter_(1, router_indices, router_scores)
router_scores = router_scores[:, ep_rank * num_local_experts : (ep_rank + 1) * num_local_experts]
```

This scatters `router_scores` from `(seq, top_k=8)` into a sparse `(seq, 128)` matrix indexed by expert ID, then slices to `(seq, num_local_experts)`. The output shape is now `(seq, num_local_experts)` — **different from the input `(seq, top_k)`**.

But `router_indices` stays `(seq, top_k=8)`.

### Why this breaks everything

All three expert forward implementations pair `top_k_weights` element-wise with `top_k_index`:

**`grouped_mm_experts_forward`** (moe.py line 382-384):
```python
sample_weights = top_k_weights.reshape(-1)  # seq * num_local_experts elements
expert_ids = top_k_index.reshape(-1)        # seq * top_k elements  ← DIFFERENT SIZE
```

With EP=2 and 128 experts: `sample_weights` has `seq * 64` elements, `expert_ids` has `seq * 8`. The subsequent `perm = argsort(expert_ids)` produces indices into the wrong tensor — routing weights from the wrong tokens get paired with the wrong experts.

**`batched_mm_experts_forward`** (moe.py line 122): identical issue.

**Eager forward** (one_hot path): `top_k_weights[token_idx, top_k_pos]` uses top_k position (0-7) to index into an expert-indexed tensor (0 to num_local_experts-1) — gets the score for the wrong expert.

### The fix

Replace scatter+slice with `masked_fill` that preserves the `(seq, top_k)` shape:

```python
non_local_mask = (router_indices // num_local_experts) != ep_rank
router_scores = router_scores.masked_fill(non_local_mask, 0.0)  # stays (seq, top_k)
```

Non-local expert scores become 0. Local expert scores stay at their original positions. The shape stays `(seq, top_k)`, matching `router_indices`.

### Evidence

Tested with `benchmark/test_ep_shapes.py` (EP with hooks) vs `benchmark/test_no_ep.py` (ground truth without EP), input `[[1, 2, 3]]`:

**Before fix:**

| Test  | scores shape | indices shape | Expert out max | Logits[0,:5]                      |
| ----- | ------------ | ------------- | -------------- | --------------------------------- |
| No EP | (3, 8)       | (3, 8)        | 1.46           | [14.56, 10.69, 15.44, 8.50, 17.88] |
| EP=1  | (3, 128)     | (3, 8)        | **0.00**       | [7.75, 7.72, 8.13, 9.19, 4.53]   |
| EP=2  | (3, 64)      | (3, 8)        | 0.53           | [6.13, 5.47, 6.59, 12.50, 6.38]  |
| EP=4  | (3, 32)      | (3, 8)        | 0.28           | [8.00, 10.38, 10.25, 9.75, 7.69] |

EP=1 expert output is literally zero — MoE layers contribute nothing. All EP sizes produce completely wrong logits.

**After fix:**

| Test  | scores shape | indices shape | Expert out max | Logits[0,:5]                          |
| ----- | ------------ | ------------- | -------------- | ------------------------------------- |
| No EP | (3, 8)       | (3, 8)        | 1.46           | [14.56, 10.69, 15.44, 8.50, 17.88]   |
| EP=1  | (3, 8)       | (3, 8)        | **1.46**       | **[14.56, 10.69, 15.44, 8.50, 17.88]** |
| EP=2  | (3, 8)       | (3, 8)        | **1.46**       | **[14.63, 10.56, 15.44, 8.31, 17.88]** |
| EP=4  | (3, 8)       | (3, 8)        | **1.46**       | **[14.69, 10.69, 15.38, 8.31, 17.88]** |

Shapes match. Expert output is correct. Logits are within bf16 precision of ground truth.

---

## Bug 2: Weight loading uses wrong plan for regex

**File**: `src/transformers/modeling_utils.py`, line 4259

### The setup

`convert_and_load_state_dict_in_model` in `core_model_loading.py` uses the plan in two steps:

1. **Line 1141**: Build a regex from `tp_plan.keys()` — determines WHICH parameters get sharded
2. **Line 1209**: Look up `model.tp_plan[matched_pattern]` — determines HOW they get sharded

The `tp_plan` parameter comes from line 4259 in `modeling_utils.py`.

### The bug

```python
# modeling_utils.py line 4259
tp_plan=model._tp_plan,      # raw TP plan attribute
```

But line 1209 uses:
```python
model.tp_plan[matched_tp_pattern]  # tp_plan PROPERTY
```

The `tp_plan` property (line 1332-1338):
```python
@property
def tp_plan(self):
    if ... distributed_config.enable_expert_parallel:
        return self._ep_plan   # EP plan
    return self._tp_plan       # TP plan
```

So when EP is enabled:
- **Regex** is built from `_tp_plan` keys (attention + experts entries)
- **Values** are looked up from `_ep_plan` (different plan)

### Why this matters for expert-only EP

With the expert-only ep_plan (no attention entries):

```python
_tp_plan keys: ["layers.*.self_attn.q_proj", ..., "layers.*.mlp.experts.gate_up_proj", ...]
_ep_plan keys: ["layers.*.mlp.gate", "layers.*.mlp.experts.gate_up_proj", ...]
```

When loading `q_proj.weight`:
1. Regex (from `_tp_plan`) matches `layers.*.self_attn.q_proj` ✓
2. Lookup `model.tp_plan["layers.*.self_attn.q_proj"]` → searches `_ep_plan` → **KeyError** (not in expert-only plan)

### The fix

```python
# modeling_utils.py line 4259
tp_plan=model.tp_plan,        # use the property, not the raw attribute
```

Now both the regex and the lookup use the same plan. When EP is enabled, both use `_ep_plan`. When EP is not enabled, `model.tp_plan == model._tp_plan` — zero change for regular TP.

For expert-only EP: the regex only matches expert params. Attention params don't match → loaded as full replicated tensors → correct (FSDP2 handles attention sharding).

---

## Bug 3: `grouped_mm_experts_forward` doesn't handle EP sentinels

**File**: `src/transformers/integrations/moe.py`, `grouped_mm_experts_forward` (line 369)

### The bug

When EP is enabled, `RouterParallel` sets non-local expert indices to a sentinel value (`num_local_experts`). The `batched_mm_experts_forward` handles this correctly (lines 125-128 + 165):

```python
invalid_mask = expert_ids >= self.num_experts
expert_ids = expert_ids.clamp(0, self.num_experts - 1)
...
weighted_out.masked_fill_(invalid_mask.unsqueeze(-1), 0.0)
```

But `grouped_mm_experts_forward` has no such handling. It uses `histc` with `max=self.num_experts - 1` (line 399) to compute per-expert token counts. Sentinel values (`num_local_experts`) fall outside the bin range, so those tokens are not assigned to any expert. The native `torch.nn.functional.grouped_mm` only writes output for tokens within the offset ranges — sentinel positions get **uninitialized GPU memory**.

Then at line 437:
```python
weighted_out = proj_out * sample_weights_g.unsqueeze(-1)
```

The sentinel routing weights are `0.0` (correctly zeroed by the RouterParallel fix). But `0.0 * NaN = NaN` per IEEE 754. The NaN propagates through all subsequent layers.

### Why this wasn't visible before

With the old (buggy) RouterParallel that scattered scores into `(seq, num_local_experts)` shape, sentinel weights were accidentally **non-zero** (garbage from wrong positions in the mismatched tensor). `non_zero * uninitialized` often produces finite garbage rather than NaN. With the correct RouterParallel fix (Bug 1), sentinel weights are properly `0.0`, making `0.0 * NaN = NaN` deterministic.

### The fix

Same pattern as `batched_mm`: clamp sentinel IDs before processing, zero out after (lines 384-388 + 441):

```python
# Before sorting
invalid_mask = expert_ids >= self.num_experts
expert_ids = expert_ids.clamp(0, self.num_experts - 1)

# After weighted multiply
invalid_mask_g = invalid_mask[perm]
weighted_out.masked_fill_(invalid_mask_g.unsqueeze(-1), 0.0)
```

Clamping makes sentinels hit a valid expert (so `grouped_mm` writes real values instead of leaving garbage). The `masked_fill_` then zeros out those invalid contributions. This matches the `batched_mm` approach exactly.

### Evidence

EP=8 expert-only, Qwen3-30B-A3B, `test_ep_impl_compare.py`:

| impl | Before fix | After fix | Ground truth |
|------|-----------|-----------|-------------|
| grouped_mm | NaN | [14.75, 10.63, 15.44, 8.38, 18.00] | [14.56, 10.69, 15.44, 8.50, 17.88] |
| batched_mm | [14.75, 10.63, 15.44, 8.38, 18.00] | (same) | (same) |

EP=16 (2 nodes) grouped_mm after fix: `[14.63, 10.63, 15.44, 8.50, 17.88]` — matches ground truth.

---

## Test setup

### Ground truth (no EP, no TP)

```bash
# benchmark/test_no_ep.py — loads model with device_map="cuda:0", no distributed
python benchmark/test_no_ep.py
```

Hooks on layer 0's gate and experts print shapes and values. Produces reference logits.

### EP test with shape debugging

```bash
# benchmark/test_ep_shapes.py — loads with DistributedConfig(enable_expert_parallel=True)
torchrun --nproc_per_node=N benchmark/test_ep_shapes.py
```

Same hooks. Compare scores/indices shapes and logits against ground truth.

### What to check

1. **Shapes**: `scores` and `indices` must have same last dimension (both `top_k`)
2. **Expert output max**: Must be nonzero and close to ground truth (1.46 for this model/input)
3. **Logits**: Must be within ~0.2 of ground truth (bf16 precision + different reduce order)

---

## Why the old EP runs in report.md appeared to work

### Phase 1: "EP enabled" but not active (report lines 232-262)

The first EP benchmark runs passed `--enable_expert_parallel` but the `device_mesh` was never passed to `from_pretrained` (accelerate creates its own mesh internally). Without a device_mesh, `distribute_model()` never runs → no EP hooks applied → no expert sharding. These runs were pure FSDP2 + fused experts. They were later renamed to `_noep` in wandb.

MFU matched the fused-only runs within noise (~3% at 16k), confirming EP was inactive.

### Phase 2: True EP with device_mesh fix (report lines 315+)

After adding `dist.init_device_mesh("cuda", (world_size,))` in sft_trainer.py, `distribute_model()` ran and expert weights were physically sharded (128→16 per GPU). Shape verification confirmed sharding:

```
gate_up_proj: [16, 1536, 2048]  (was [128, 1536, 2048])
```

These runs used the ep_plan WITH attention entries (colwise/rowwise on q/k/v/o_proj + grouped_gemm on experts + ep_router on gate). The RouterParallel shape mismatch bug was present — **routing weights were wrong for all runs**.

### Why wrong routing didn't cause NaN or obviously bad loss

1. **Residual connections**: Each decoder layer adds MoE output to the residual. With wrong routing weights, expert outputs are near-zero or incorrectly weighted, but the residual connection preserves the hidden state signal. The model effectively skips MoE layers and relies on attention + residual for training signal.

2. **Short training (20 steps)**: Benchmark runs only train 20 steps. With wrong MoE contributions, the loss still decreases because attention gradients are correct. Over thousands of steps, the wrong routing would likely cause training instability.

3. **MFU computation doesn't validate correctness**: MFU measures throughput (tokens/second × flops/token / peak_flops). Wrong routing weights don't slow down computation — the GPU still does the same matmuls. MFU was "correct" in measuring hardware utilization, but the computation was mathematically wrong.

4. **The bug produces non-NaN results**: The shape mismatch causes wrong weights, not infinite weights. `grouped_mm_experts_forward` reads from wrong positions in the weight tensor, getting values that are finite (mostly zeros for non-selected experts). The result is incorrect but numerically stable for short runs.

### What the MFU numbers actually measured

The EP runs in the report (2.7-4.8% MFU) measured the throughput of a model that was:
- Correctly sharding expert weights across GPUs (GroupedGemmParallel works correctly)
- Correctly loading and distributing attention weights
- Correctly running the forward/backward computation
- **Incorrectly routing tokens to experts** (wrong routing weights from RouterParallel)

The throughput numbers (TPS, TPS/GPU) are valid — the hardware was doing real work at real speed. The MFU is valid as a hardware utilization metric. But the training was not producing correct gradients for the MoE layers.

### Phase 3: Expert-only EP (current work)

Removing attention from ep_plan to allow EP > num_kv_heads failed because:
1. Bug 2 (line 4259) caused KeyError when TP-plan regex matched attention params not in the expert-only EP plan
2. Even after that, Bug 1 (RouterParallel scatter) caused NaN in training due to wrong routing weights accumulating across many layers

Both bugs are now fixed. Expert-only EP with the fixes produces logits matching ground truth for EP=1, 2, 4. EP=8+ (beyond num_kv_heads) is now possible since attention is not in the plan.

---

## Summary of changes

| File | Line | Change | Why |
|------|------|--------|-----|
| `integrations/tensor_parallel.py` | 1136-1137 | Replace scatter+slice with masked_fill | Preserve `(seq, top_k)` shape for router_scores |
| `modeling_utils.py` | 4259 | `model._tp_plan` → `model.tp_plan` | Use EP plan for both regex matching and value lookup |
| `integrations/moe.py` | 384-388, 441 | Clamp sentinel IDs + masked_fill after multiply | Handle EP sentinels in grouped_mm (same as batched_mm) |
| `models/qwen3_moe/configuration_qwen3_moe.py` | ep_plan | Remove attention entries | Allow EP to scale beyond num_kv_heads=4 |
