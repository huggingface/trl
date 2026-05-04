# Sonic-MoE backward bug — standalone repro (kernel-only, no transformers/DS)

## TL;DR

`moe_general_routing_inputs` produces incorrect input-gradients (`dh` w.r.t. `hidden_states`, and `ds` w.r.t. `router_scores`) when `expert_ids` contains the EP sentinel value `num_experts`, even though:

- the routing-metadata kernel explicitly documents `expert_ids >= E` as a supported sentinel and remaps them with `safe_experts` (`functional/triton_kernels/__init__.py:174-177`),
- the forward output is bit-identical to the no-sentinel (clamped) case,
- the parameter gradients (`dw1`, `dw2`) match the no-sentinel case to ~5%.

But `dh` and `ds` differ by **~17× and ~6.6×** respectively, and the difference compounds through subsequent backward layers in real training.

## Standalone reproduction

One file, ~120 lines. No transformers, no DeepSpeed, no distributed setup — just `torch` + `kernels`. Random tensors. Runs in ~10 s on one H100.

```bash
pip install kernels nvidia-cutlass-dsl
python test_sonic_repro_minimal.py
```

The trigger is a specific **per-token sentinel pattern**: at high EP factor, the global top-k for many tokens lands on experts owned by other ranks, so ALL of _this_ rank's routing slots for those tokens become sentinels. Uniformly-random sentinels (where each token still has at least one valid slot probabilistically) do _not_ trigger the bug — the script forces ~94% of tokens to have _all_ TOP_K slots set to sentinel, matching the production capture.

## Output

### Synthetic (no .pt file, just `python test_sonic_repro_minimal.py`):

```
E=2 H=8 I=768 T=512 TOP_K=2; 94% of tokens have ALL slots as sentinel:

mode                                                clamp=False    clamp=True   ratio
B/C: 94% all-sentinel tokens (sentinel_ratio=0.94)
  forward out norm                                  0.01595        0.01595      identical
  g.grad (w1) norm                                  1.259          1.259        identical
  d.grad (w2) norm                                  0.8848         0.8848       identical
  h.grad norm                                       0.04826        0.01953      ← 2.5× (BUG)
  rs.grad norm                                      0.09662        0.08504      ← 1.14×
```

### Capture replay (using `_sonic_capture_ec991.pt` from a real Qwen3MoE training step) — same bug, more pronounced because of specific tensor data:

```
                                                    clamp=False    clamp=True   ratio
forward out norm                                    0.03022        0.03022      identical
g.grad (w1) norm                                    0.9652         1.038        1.07×
d.grad (w2) norm                                    0.6417         0.7019       1.09×
h.grad norm                                         0.9843         0.05676      ← 17×    (BUG)
rs.grad norm                                        0.1969         1.2925       ← 6.6×   (BUG)
```

### Sentinel-pattern dependence (synthetic, sweep over `SENTINEL_RATIO`):

| ratio of all-sentinel tokens | h.grad clamp=off | h.grad clamp=on | trigger?            |
| ---------------------------- | ---------------- | --------------- | ------------------- |
| 0.50 (uniform sentinels)     | 0.054            | 0.053           | no                  |
| 0.80                         | 0.035            | 0.033           | no                  |
| **0.94**                     | **0.048**        | **0.020**       | **yes**             |
| 0.97                         | 0.013            | 0.013           | no (seed-dependent) |
| **0.99**                     | **0.022**        | **0.007**       | **yes**             |

The bug appears when _many tokens have zero valid slots_. Uniformly-distributed sentinels (each token still has at least one valid slot probabilistically) don't trigger it.

## Why this matters

Under transformers' EP integration, every MoE layer at EP=N has a sentinel rate ≈ (N-1)/N. With EP ≥ 2 the rate is ≥ 50 %; with EP=8 it is 87.5 %. The `dh` returned by the kernel propagates as the gradient into the previous layer's output, so the magnitude error compounds through every subsequent backward layer. In a real Qwen3-30B-A3B 8-node EP=8 SFT run we observed:

- with the wrapper-side `expert_ids.clamp(0, num_experts-1)`: training is healthy, grad_norm ≈ 0.25
- without the clamp: grad_norm ≈ 3000, NaN by step 10

Reduces to a 1-node EP=2 + `tiny-Qwen3MoeForCausalLM` test that shows ~10,000× grad_norm inflation in 30 SFT steps.

## Mechanism (hypothesis)

`_DownProjection.backward` (`functional/__init__.py:274-328`) calls `_down_projection_backward_act` with the metadata buffers (`expert_frequency_offset`, `x_gather_idx`, `s_scatter_idx`). The metadata kernel correctly drops sentinel slots in forward via the `expert_ids < E` mask (`functional/triton_kernels/__init__.py:177`). But somewhere in the bwd-act path, contributions to `dh` (and `ds`) are accumulated for slots that should have been dropped. The most likely culprit: when sentinel slots' `x_gather_idx` and `s_scatter_idx` were left at their zero-init defaults by the metadata kernel, the bwd-act kernel reads at index 0 (writing/accumulating into `dh[token_0]` or via `s_scatter_idx[0]`), producing the magnitude inflation we see.

### Parallel: same family as the `torch._grouped_mm` backward bug we just diagnosed in transformers PR #45621

We found and fixed an identical-shaped bug for the `grouped_mm_experts_forward` wrapper in transformers PR #45621 (see `benchmark/grouped_mm_pr45621_comment.md` and `benchmark/study_moe_autograd.md`). The mechanism there:

1. `torch._grouped_mm` writes only rows `[0, offsets[-1])` of `d_input` in backward; sentinel rows past `offsets[-1]` are left as whatever uninitialized memory `torch.empty_like(input)` returned. Under production allocator pressure this is often NaN.
2. The wrapper's gather backward is an `index_add_` that scatters those NaN rows into `d_hidden_states` at the corresponding tokens (because `dest[idx[r]] += src[r]` and IEEE 754 says `real + NaN = NaN`).
3. Result: `d_hidden_states` gets poisoned token-by-token under high sentinel rates (almost every token has at least one sentinel slot).

We strongly suspect the sonic-moe kernel has the same issue *internally* — the gather/grouped_mm/scatter chain that's split across the wrapper for `grouped_mm_experts_forward` is fused into `moe_general_routing_inputs` here. If any intermediate `(S, …)`-shaped buffer in the bwd-act path is `torch.empty_like(...)` and only valid rows are written, the same NaN-via-`+=`-into-`dh` story plays out, just inside the kernel.

The signature that we see in our capture-replay (large `dh` / `ds` magnitude difference, parameter gradients correct, forward correct, pattern-sensitive on sentinel density) **is exactly what we'd see if a `(S, H)` working tensor for `dh` accumulation is `empty_like` and the kernel scatter-adds only over valid rows.** Worth looking at any `torch.empty_like(...)` allocation in `_down_projection_backward_act` and `_up_projection_backward_act` — particularly buffers that get scatter-add'ed into `dh` or `ds`.

## Suggested fixes (any one suffices)

0. **Replace `torch.empty_like(...)` with `torch.zeros_like(...)`** for any backward intermediate whose tail past `offsets[-1]` (or equivalent) is not written by the kernel. This is the one-line analog of the proper fix we'd want for `torch._grouped_mm`. Likely catches the bug at the source — the buffer's sentinel rows are then 0 instead of stale memory, and `+=` on 0 is a no-op.
1. **Bounds-check `expert_ids` inside `_down_projection_backward_act` and `_up_projection_backward_act`**: skip slots where `expert_ids >= E`, matching the forward-routing semantics.
2. **Pass `expert_indices` into the bwd-act kernels and skip sentinel slots there** (currently only `expert_frequency_offset` is passed).
3. **Adopt the "histogram-tail-drop" pattern from transformers PR #45621**: sort sentinels to the tail and use `histc(max=E-1)` so the bwd kernel never sees them at all.

Note: option (0) is not a viable wrapper-side workaround for sonic-moe (unlike for `torch._grouped_mm` where transformers PR #45621 splits the gather and scatter into separate ops the wrapper can `masked_fill` between). Sonic-moe's gather/scatter is fused inside one `moe_general_routing_inputs` autograd Function, so there's no place between them for an external `masked_fill` to act as a NaN firewall. The wrapper-level workaround is the existing `expert_ids.clamp(0, num_experts - 1)`, which makes the kernel never enter its sentinel-skip path at all (cost: ~2 pp MFU because all rows go through the matmul).

## Files in this repro

- `test_sonic_repro_minimal.py` (~120 lines, depends only on `torch` + `kernels`) — synthetic, no `.pt` dependency, primary repro
- `test_sonic_replay.py` (~120 lines, same dependencies) — replays a captured tensor `.pt` for stronger evidence
- `_sonic_capture_ec991.pt` (95 KB) — captured tensors from a real MoE layer with 991/1024 sentinel slots
- `_sonic_capture_ec58.pt` (95 KB) — control capture with 58/1024 sentinels (no bug)

## Verification path for a maintainer

1. `python test_sonic_replay.py _sonic_capture_ec991.pt` — see the 17× / 6.6× divergence.
2. `python test_sonic_replay.py _sonic_capture_ec58.pt` — see ≤5 % divergence (kernel correct at low sentinel rate).
3. Apply any of the suggested fixes; re-run (1) and confirm divergence drops to ≤5 %.

## Production downstream context (for reference, not needed to verify the bug)

- Wrapper file: `transformers/integrations/sonicmoe.py` — the load-bearing workaround is `expert_ids = expert_ids.clamp(0, self.num_experts - 1)` (sentinels mapped to expert `E-1` with score=0). Costs ~2 pp MFU vs an ideal kernel-side fix because the kernel still processes sentinel rows even though they multiply by 0.

---

## GitHub comment (copy-paste, gist file: `repro_sonic_moe_backward_sentinel.py`)

````markdown
## Sonic-MoE: backward produces wrong `dh` / `ds` when `expert_ids` contains EP sentinels

`moe_general_routing_inputs` produces incorrect input-gradients (`∂L/∂hidden_states`, `∂L/∂router_scores`) when many tokens have all `top_k` routing slots set to the sentinel `num_experts`. Forward output and parameter gradients (`∂L/∂w1`, `∂L/∂w2`) are correct in the same call. The kernel documents `expert_ids >= E` as a supported sentinel (`functional/triton_kernels/__init__.py:174-177, 237-239`, `functional/__init__.py:452-453`), but the bwd-act path leaks gradient contributions from sentinel slots into `dh` / `ds`.

### Standalone repro (gist below, ~120 lines, `torch + kernels` only)

```bash
pip install kernels nvidia-cutlass-dsl
python repro_sonic_moe_backward_sentinel.py
```

Three modes on the same kernel call (`E=2 H=8 I=768 T=512 TOP_K=2`):

- **A**: control (no sentinels)
- **B**: ~94 % of tokens have ALL slots = sentinel, no clamp (broken EP config)
- **C**: same as B + `expert_ids = expert_ids.clamp(0, E-1)` (our current wrapper workaround)

### Output

```
                  B (clamp=off)   C (clamp=on)
forward out norm  0.01595         0.01595        ← identical (forward is correct)
g.grad norm       1.259           1.259          ← identical (param grads correct)
d.grad norm       0.8848          0.8848         ← identical
h.grad norm       0.04826         0.01953        ← 2.5× — BUG
rs.grad norm      0.09662         0.08504        ← 1.14× — BUG
```

Only the upstream-flowing gradients diverge. With a captured production tensor at the same shapes, `h.grad` diverges 17× and `rs.grad` 6.6× — full data and `.pt` capture available on request.

### Sentinel-pattern dependence

| % tokens with all slots sentinel | `h.grad` clamp=off | clamp=on  | triggers? |
| -------------------------------- | ------------------ | --------- | --------- |
| 50 (uniform random)              | 0.054              | 0.053     | no        |
| 80                               | 0.035              | 0.033     | no        |
| **94**                           | **0.048**          | **0.020** | **yes**   |
| **99**                           | **0.022**          | **0.007** | **yes**   |

Trigger condition: many tokens with **zero** valid slots. Uniformly-distributed sentinels (each token retains at least one valid slot) do not trigger it.

### Why it matters

Under transformers' EP integration (#45662), every MoE layer sees an EP sentinel rate ≈ (N-1)/N. At EP=8 (87.5 % sentinels per rank) we observe NaN gradients by step ~10 in real Qwen3-30B-A3B SFT without the wrapper-side clamp; with the clamp, training is healthy. The bug reduces to a 1-node EP=2 + `tiny-Qwen3MoeForCausalLM` run that shows ~10 000× total `grad_norm` inflation in 30 steps.

### Mechanism (hypothesis)

`_DownProjection.backward` calls `_down_projection_backward_act` with `expert_frequency_offset`, `x_gather_idx`, `s_scatter_idx`. Sentinel slots have `x_gather_idx[i] = s_scatter_idx[i] = 0` (zero-init from the metadata kernel). The bwd-act path likely iterates over all `[0, TK)` and writes into `dh[token_0]` / `ds[slot_0]` for sentinel slots, producing the magnitude inflation.

This shape exactly matches a bug we just diagnosed and fixed in transformers' `grouped_mm_experts_forward` (PR #45621): `torch._grouped_mm`'s backward writes only `[0, offsets[-1])` of `d_input`, leaving sentinel rows uninitialized (`torch.empty_like` returns stale memory, often NaN under allocator pressure). When that NaN flows through the upstream gather backward (`hidden_states[idx]`), the gather backward is `index_add_` (`dest[idx[r]] += src[r]`), and IEEE 754 says `real + NaN = NaN`, so a single sentinel row poisons the destination token. Under EP=8, ~87.5 % of slots are sentinels and almost every token has at least one sentinel slot, so almost every row of `d_hidden_states` ends up NaN.

In sonic-moe the gather and scatter are *fused* inside `moe_general_routing_inputs` (instead of split across the autograd graph as in `grouped_mm_experts_forward`), but the structure of "kernel writes valid rows only, scatter-add accumulates a tail of stale memory into `dh`" can play out identically. Worth checking whether any backward intermediate inside the kernel is `torch.empty_like(...)` (instead of `torch.zeros_like(...)`).

### Suggested fixes (any one suffices)

0. **Replace `torch.empty_like(...)` with `torch.zeros_like(...)`** for any backward intermediate whose tail past `offsets[-1]` is not written by the kernel — particularly buffers that subsequently get scatter-add'ed into `dh` or `ds`. This is the one-line analog of the cleanest fix for `torch._grouped_mm`.
1. Bounds-check `expert_ids` inside `_down_projection_backward_act` / `_up_projection_backward_act`; skip slots with `expert_ids >= E`.
2. Pass `expert_indices` into the bwd-act kernels and skip sentinel slots there (currently only `expert_frequency_offset` is passed).
3. Adopt the histogram-tail-drop pattern from transformers PR #45621 — sort sentinels to the tail, use `histc(max=E-1)` so the bwd kernel never sees them.

Wrapper-side note: the `masked_fill`-firewall trick we used to fix transformers PR #45621 doesn't apply here, because sonic-moe's gather/scatter is fused into one autograd Function — there's no intermediate tensor between them that the wrapper can `masked_fill` to absorb NaN. The current wrapper-side workaround is `expert_ids.clamp(0, num_experts-1)`, which keeps the kernel from entering its sentinel-skip path at all (costs ~2 pp MFU because every slot goes through the matmul).

Happy to verify any patch against our production Qwen3-30B-A3B SFT.
````
