# Comments for huggingface/transformers#45621

## Comment 1 — already posted

https://github.com/huggingface/transformers/pull/45621#issuecomment-4351066146

(Suggested zeroing `proj_out` before the multiply to avoid `0 × NaN` leaking into `d_sample_weights`.)

---

## Comment 2 — follow-up after end-to-end testing

Update on my earlier suggestion. I implemented the pre-zero (`proj_out.masked_fill_(sentinel_mask, 0)` before the multiply) and re-ran end-to-end on `Qwen/Qwen3-30B-A3B` with EP=8 + FSDP2 + sdpa, 16k ctx, 2 nodes. **It's necessary but not sufficient** — training still NaN'd at step 2 (`loss=0`, `entropy=NaN`).

After more digging, there are actually **three separate sentinel-poisoning paths** in the PR's wrapper, not one:

### The three leaks

```python
selected_hidden_states_g = hidden_states[perm // num_top_k]   # (S, H)
proj_out = _grouped_linear(selected_hidden_states_g, gate_up_proj, offsets)   # ←── leak 2
proj_out = proj_out.masked_fill(sentinel_mask, 0.0)
proj_out = self._apply_gate(proj_out)
proj_out = _grouped_linear(proj_out, down_proj, offsets)                      # ←── leak 3
proj_out.masked_fill_(sentinel_mask, 0.0)
weighted_out = proj_out * sample_weights_g.unsqueeze(-1)                      # ←── leak 1
```

**Leak 1** is what my first comment caught: `0 × NaN = NaN` in `d_sample_weights`. Closed by zeroing `proj_out` before the multiply.

**Leaks 2 & 3** come from `_grouped_mm`'s backward leaving its `d_input` partially uninitialized.

Walking through what happens for the up-projection:

```python
# 1. Build the per-(token, slot) input tensor for the kernel by gathering rows from hidden_states:
selected_hidden_states_g = hidden_states[perm // num_top_k]   # shape (S, H)

# 2. Run the kernel. It only does work for rows [0, offsets[-1]) — sentinels are excluded.
proj_out = _grouped_mm(selected_hidden_states_g, gate_up_proj, offsets)
```

In backward, autograd needs the gradient w.r.t. each input. The kernel produces `d_selected_hidden_states_g` (= the gradient that should flow back through step 1). **The kernel writes only the valid rows of that gradient — sentinel rows are left as whatever uninitialized memory the buffer happened to be allocated with.** In a fresh process that's often zero; in production with a hot allocator it's frequently NaN.

Then autograd has to undo step 1, the gather. The backward of `hidden_states[perm // num_top_k]` is the inverse: a *scatter-add* into `d_hidden_states` — for every row `r` of `d_selected_hidden_states_g`, it adds that row to `d_hidden_states` at token index `perm[r] // num_top_k`. **Sentinel rows have NaN, so NaN gets added to `d_hidden_states` at every token a sentinel row was paired with.**

Under the ~94 % all-sentinel-token pattern EP=8 produces, almost every token has at least one sentinel slot, so almost every row of `d_hidden_states` ends up with NaN added to it. From there NaN spreads through the previous transformer block (residual + LayerNorm + attention + previous MoE), and Adam writes NaN into every parameter on the optimizer step.

Step 1 forward: clean (the wrapper's `masked_fill` after `_grouped_mm` zeros sentinel rows of the *output*). Step 1 backward: NaN gets scattered into `d_hidden_states` via the path above. Step 2 forward: weights are NaN → `loss=0, entropy=NaN`.

The same chain repeats for the down projection (leak 3): kernel leaves `d_(SwiGLU output)[sentinel]` uninitialized, and that NaN flows back through SwiGLU into the up-projection's output gradient.

### Wrapper-side fix that works (50 steps clean, mfu_window 27.5–28%)

Wrap each `_grouped_mm` call with `masked_fill(sentinel, 0.0)` on BOTH the input and the output:

```python
sentinel_mask_g = (expert_ids_g >= self.num_experts).unsqueeze(-1)

# Up projection
selected_hidden_states_g = selected_hidden_states_g.masked_fill(sentinel_mask_g, 0.0)
proj_out = _grouped_linear(selected_hidden_states_g, gate_up_proj, offsets)
proj_out = proj_out.masked_fill(sentinel_mask_g, 0.0)

proj_out = self._apply_gate(proj_out)   # SwiGLU(0) = 0, sentinel rows stay 0

# Down projection — same pattern
proj_out = proj_out.masked_fill(sentinel_mask_g, 0.0)
proj_out = _grouped_linear(proj_out, down_proj, offsets)
proj_out = proj_out.masked_fill(sentinel_mask_g, 0.0)

# Multiply (closes leak 1: post-`_grouped_mm` mask above means proj_out[sentinel] = 0,
# so backward's `d_sample_weights = (d_w * proj_out).sum(-1)` is 0×0 = 0, not 0×NaN)
weighted_out = proj_out * sample_weights_g.unsqueeze(-1)
```

Why this works without any explicit hook: `masked_fill`'s backward is `d_x = d_y.masked_fill(mask, 0)` — it **unconditionally overwrites the gradient with zero at masked positions**, regardless of what `d_y` was at those positions. So if the kernel produces NaN sentinel rows in `d_input`, the very next op in the autograd graph is `masked_fill`'s backward, which replaces them with zero before they can flow into the upstream gather and poison `d_hidden_states`. The post-`_grouped_mm` `masked_fill` also gives forward correctness (sentinel rows of the output are zero, so SwiGLU and the multiply can't see garbage).

Result on a 50-step run:

| step | loss   | entropy | mean_token_acc | mfu_window |
|------|--------|---------|----------------|------------|
| 5    | 12.05  | 1.19    | 0.66           | —          |
| 25   | 13.30  | 1.63    | 0.64           | 28.05 %    |
| 50   | 12.60  | 1.56    | 0.65           | 28.01 %    |

`mfu_window` averages **27.5–28.0%** — matches the throughput claim in this PR. No NaN, entropy/accuracy stable across 50 steps.

### Recommended fix is kernel-side, not wrapper-side

The hook approach works but is invasive (the wrapper now has to know about the kernel's uninitialized-tail behavior). The cleaner fix is in `torch._grouped_mm`'s backward: **zero-init `d_input`** — i.e., replace whatever `torch.empty_like(input)` is with `torch.zeros_like(input)`. With that, all three leaks vanish:

- Leak 1 disappears because `proj_out` (= `d_input` of the down projection in the autograd graph viewed from the multiply forward, NOT in backward — but conceptually the kernel's *forward* output also has the same uninitialized-tail issue, so zero-init applies there too if the upstream kernel allocates output similarly).
- Leak 2 + 3 disappear because `d_input` for both grouped_mm calls is now zero at sentinel rows, so the `index_add_` scatters zeros, not NaN.

Then this PR's wrapper as-written is safe — only the existing `masked_fill_` after the multiply is needed for forward correctness, no hooks required. That keeps the wrapper as clean as you proposed.

### Vacuum test for leak 1 (in case useful for a regression test)

```python
import torch
torch.manual_seed(0)
S, H = 16, 4
sentinel = torch.zeros(S, dtype=torch.bool); sentinel[S // 2:] = True

sw = torch.randn(S); sw[sentinel] = 0
sw = sw.clone().requires_grad_(True)

proj = torch.randn(S, H)
proj[sentinel] = float("nan")            # mimic uninitialized memory
proj = proj.clone().requires_grad_(True)

w = (proj * sw.unsqueeze(-1)).masked_fill(sentinel.unsqueeze(-1), 0.0)
out = w.view(S // 2, 2, H).sum(dim=1)
out.backward(torch.randn_like(out))

assert torch.isnan(sw.grad).sum() == 0   # FAILS without pre-zero on proj
```

### Suggested wrapper-side patch on top of this PR

Here's the exact diff against the current `grouped_mm_experts_forward` in this PR (the four `masked_fill` calls + dropping the now-redundant post-multiply `masked_fill_`). This is what we ran for the 50-step benchmark above. Happy to push it as a commit on this PR if you'd prefer.

```diff
--- a/src/transformers/integrations/moe.py
+++ b/src/transformers/integrations/moe.py
@@ -402,11 +402,15 @@ def grouped_mm_experts_forward(
     tokens_per_expert = torch.histc(histc_input, bins=self.num_experts, min=0, max=self.num_experts - 1)
     offsets = torch.cumsum(tokens_per_expert, dim=0, dtype=torch.int32)

+    sentinel_mask_g = (expert_ids_g >= self.num_experts).unsqueeze(-1)
     if self.has_bias:
         # Clamp now that the layout has been built — needed for the per-row bias gather below to stay
         # in-bounds. Bias added to sentinel positions falls in rows the kernel skips, so harmless.
         expert_ids_g.clamp_(0, self.num_experts - 1)

+    # `_grouped_mm` only writes rows [0, offsets[-1]) of its output (forward) and `d_input`
+    # (backward). Sentinel-row outputs and gradients are uninitialized — under production memory
+    # pressure they can be NaN. Wrap each `_grouped_mm` input AND output with `masked_fill(sentinel, 0)`.
+    # Backward of `masked_fill` unconditionally zeros at masked positions, absorbing whatever NaN the
+    # kernel writes into `d_input` so it doesn't reach `hidden_states.grad` via the upstream gather.
+
     # Select expert weights and biases
     ...
     # --- Up projection per expert (grouped) ---
+    selected_hidden_states_g = selected_hidden_states_g.masked_fill(sentinel_mask_g, 0.0)
     proj_out = _grouped_linear(
         selected_hidden_states_g, selected_weights, offsets, bias=selected_biases, is_transposed=self.is_transposed
     )  # (S, 2 * intermediate_dim) or (S, intermediate_dim)
+    proj_out = proj_out.masked_fill(sentinel_mask_g, 0.0)

     # Apply gating or activation (sentinel rows are 0 -> stay 0)
     if self.has_gate:
         proj_out = self._apply_gate(proj_out)
     else:
         proj_out = self.act_fn(proj_out)

     # --- Down projection per expert (grouped) ---
+    proj_out = proj_out.masked_fill(sentinel_mask_g, 0.0)
     proj_out = _grouped_linear(
         proj_out, selected_weights, offsets, bias=selected_biases, is_transposed=self.is_transposed
     )  # (S, hidden_dim)
+    proj_out = proj_out.masked_fill(sentinel_mask_g, 0.0)

     # Apply routing weights
     weighted_out = proj_out * sample_weights_g.unsqueeze(-1)  # (S, hidden_dim)

-    # EP sentinel handling: `proj_out` rows past `offsets[-1]` are left uninitialized by grouped_mm,
-    # so `proj_out[sentinel] * 0 = 0 * NaN = NaN` can leak from allocator pool reuse. Zero them here
-    # so the downstream reduction stays finite even when the routing weight was already zero.
-    weighted_out.masked_fill_((expert_ids_g >= self.num_experts).unsqueeze(-1), 0.0)
-
     # Restore original order
     inv_perm = torch.empty_like(perm)
     inv_perm[perm] = torch.arange(perm.size(0), device=device)
```

Net effect: 4 added `masked_fill(sentinel, 0)` (input + output of each `_grouped_linear`), 1 removed `weighted_out.masked_fill_` (now redundant — `proj_out[sentinel]` is already 0 by the time we multiply). Same trade-off you accepted in this PR for the post-multiply zero, just applied earlier in the chain so backward is also covered.

If you'd rather fix this kernel-side (`zeros_like` for `d_input` in `torch._grouped_mm`'s backward), I'm happy to drop these wrapper-side `masked_fill`s once the kernel patch lands.

Happy to share full launch logs (`benchmark/logs/bench-22095475.{out,err}`) and the diagnostic NaN-detect patch we used to locate the leaks if useful.

---

## GitHub-suggestion blocks (ready to paste in PR review)

For each block below: open the **Files changed** tab on PR #45621, navigate to `src/transformers/integrations/moe.py`, click-drag to select the line range, click the "Insert suggestion" icon (third toolbar icon, looks like ±▢), paste the body of the matching block.

### Suggestion 1 of 4 — define `sentinel_mask_g`

**File**: `src/transformers/integrations/moe.py`
**Select lines**: `403` – `404` (the `offsets = torch.cumsum(...)` line and the blank line after it)

````suggestion
    offsets = torch.cumsum(tokens_per_expert, dim=0, dtype=torch.int32)

    sentinel_mask_g = (expert_ids_g >= self.num_experts).unsqueeze(-1)
````

### Suggestion 2 of 4 — wrap up `_grouped_linear` with masked_fill before & after

**File**: `src/transformers/integrations/moe.py`
**Select lines**: `423` – `426` (`# --- Up projection per expert (grouped) ---` through `# (S, 2 * intermediate_dim) or ...`)

````suggestion
    # --- Up projection per expert (grouped) ---
    selected_hidden_states_g = selected_hidden_states_g.masked_fill(sentinel_mask_g, 0.0)
    proj_out = _grouped_linear(
        selected_hidden_states_g, selected_weights, offsets, bias=selected_biases, is_transposed=self.is_transposed
    )  # (S, 2 * intermediate_dim) or  (S, intermediate_dim) depending on whether we have gating
    proj_out = proj_out.masked_fill(sentinel_mask_g, 0.0)
````

### Suggestion 3 of 4 — wrap down `_grouped_linear` with masked_fill before & after

**File**: `src/transformers/integrations/moe.py`
**Select lines**: `440` – `443` (`# --- Down projection per expert (grouped) ---` through `# (S, hidden_dim)`)

````suggestion
    # --- Down projection per expert (grouped) ---
    proj_out = proj_out.masked_fill(sentinel_mask_g, 0.0)
    proj_out = _grouped_linear(
        proj_out, selected_weights, offsets, bias=selected_biases, is_transposed=self.is_transposed
    )  # (S, hidden_dim)
    proj_out = proj_out.masked_fill(sentinel_mask_g, 0.0)
````

### Suggestion 4 of 4 — drop the now-redundant post-multiply `masked_fill_`

**File**: `src/transformers/integrations/moe.py`
**Select lines**: `446` – `452` (`# Apply routing weights` through the blank line after `weighted_out.masked_fill_(...)`)

````suggestion
    # Apply routing weights (sentinel rows of proj_out are already 0 from the masked_fills above,
    # so 0 × 0 = 0 — no need for a post-multiply mask).
    weighted_out = proj_out * sample_weights_g.unsqueeze(-1)  # (S, hidden_dim)

````

After the four suggestions are accepted, the diff is +5/−4 lines and the function trains cleanly end-to-end at the same MFU.
