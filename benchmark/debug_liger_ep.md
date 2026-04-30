# Debug: Liger kernel + Qwen3-MoE + EP

## Goal

H1 in `upstream_todo.md`: Liger crashes with `CUDA error: device-side assert triggered` on Qwen3-30B-A3B in our setup. We want:

1. A minimal reproduction that pinpoints the exact failure.
2. Understanding of where in Liger (file:line) the assert fires.
3. Workarounds we can use **without** patching Liger (alternative kernels for SwiGLU, fused_linear_cross_entropy, RMSNorm, RoPE).

## Initial mental model (turned out partially wrong)

The H1 entry in `upstream_todo.md` claimed: "Liger's `liger_kernel.transformers.swiglu` assumes standard 2D MLP weight shapes `[intermediate, hidden]`, but transformers' MoE fused expert architecture uses 3D tensors `[num_experts, intermediate, hidden]` with indexed access per expert. Liger globally patches the MLP module and doesn't dispatch to the expert-indexed forward path."

**This is wrong as stated.** Liger has a dedicated `LigerExperts` class in `liger_kernel/transformers/swiglu.py:40` that IS 3D-aware. For transformers v5+ (our version) `apply_liger_kernel_to_qwen3_moe` swaps `modeling_qwen3_moe.Qwen3MoeExperts = LigerExperts` and rebinds the `forward` method on already-instantiated experts modules via `_patch_swiglu_module(decoder_layer.mlp.experts, LigerExperts)`.

`LigerExperts.forward` is **structurally identical** to transformers' `Qwen3MoeExperts.forward`:

- `expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts)`
- `if expert_idx == self.num_experts: continue` (sentinel skip)
- Loop over `expert_hit`, indexing `self.gate_up_proj[expert_idx]` and `self.down_proj[expert_idx]`

The only difference is the activation: transformers does `act_fn(gate) * up` (eager SiLU + multiply), Liger does `LigerSiLUMulFunction.apply(gate, up)` (fused Triton kernel for `silu(gate) * up`).

So the failure can't be "Liger doesn't handle 3D weights." It must be something else — most likely interaction between EP semantics and Liger's `LigerSiLUMulFunction` Triton kernel.

## Hypothesis space (before reproducing)

1. **Sentinel handling under EP**: under transformers' `RouterParallel`, `top_k_index` for non-local tokens is set to a sentinel (`num_local_experts` or `num_experts`). The `if expert_idx == self.num_experts: continue` skip relies on `self.num_experts` being the right value. Under `_patch_swiglu_module`, `self` is the original `Qwen3MoeExperts` instance — where `self.num_experts = config.num_experts` (full 128, not sharded). If EP sets `gate_up_proj.shape[0] = num_local_experts = 16` while `self.num_experts = 128`, then `F.one_hot(top_k_index, num_classes=128)` is fine but `expert_idx` from `expert_hit.nonzero()` may exceed 16 for sentinel tokens, and `gate_up_proj[expert_idx]` indexes out of bounds → `device-side assert`.

2. **`LigerSiLUMulFunction` mishandles odd shapes** (small batches, very wide intermediate, etc.). Less likely given Liger is heavily tested on dense models.

3. **Liger's monkey-patch ordering with EP**: the EP `from_pretrained` device-mesh wrap may not play well with the post-init `_patch_swiglu_module` rebinding (DTensor params, mesh-aware accessors).

## Repro plan

1. Smallest possible model: a 2-layer Qwen3-MoE config (manually constructed).
2. Apply `apply_liger_kernel_to_qwen3_moe(model=model)` on a CPU/single-GPU model.
3. Run forward pass with synthetic input.
4. Reproduce on a single GPU first (no EP) — does Liger work at all on Qwen3-MoE without EP?
5. If single-GPU works, escalate to multi-rank with EP.

Repro script will live at `benchmark/test_liger_qwen3_moe.py`.

---

## Investigation log

### 2026-04-29 — single-GPU sanity (no EP)

Tested with a tiny 2-layer Qwen3-MoE model (8 experts, top_k=2, 256 hidden, vocab=1024) using `benchmark/test_liger_qwen3_moe.py`:

| Step | Result |
| --- | --- |
| 1. Build `Qwen3MoeForCausalLM` on CUDA | ✅ |
| 2. Forward without Liger | ✅ logits finite |
| 3. `apply_liger_kernel_to_qwen3_moe(model=model)` | ✅ patched (`_get_name() = LigerExperts`) |
| 4. Forward with Liger | ✅ logits finite, max diff vs eager = 8.79e-3 |
| 5. Backward (loss + backprop) | ✅ grad finite |

**Conclusion: the original H1 entry was wrong.** Liger's `LigerExperts` class IS 3D-aware and works fine on Qwen3-MoE without EP. The hypothesis "Liger crashes because of 3D weights" is bogus.

### 2026-04-29 — EP=2 reproduction

Same tiny model, 2 GPUs, `from_pretrained(distributed_config=DistributedConfig(enable_expert_parallel=True))`. Repro at `benchmark/test_liger_qwen3_moe_ep.py`. Failure reproduces:

```
File "/.venv/lib/python3.11/site-packages/liger_kernel/transformers/swiglu.py", line 69, in forward
  expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
torch.AcceleratorError: CUDA error: device-side assert triggered
```

**Diagnostics from the patched trace** (rank 0, layer 0):

```
top_k_index dtype=torch.int64, shape=(32, 2), min=0, max=4, unique=[0, 1, 2, 3, 4]
self.num_experts = 4
gate_up_proj.shape = (8, 256, 256)
```

So `top_k_index.max() = 4` and `num_classes = 4` — out of range for `F.one_hot`.

### 2026-04-29 — but eager forward succeeds with the same inputs!

Step 3 (eager EP forward, no Liger) saw the **same** `top_k_index` (max=4) and **same** `num_experts` (4), but completed without error. Why?

**The dispatcher is the difference.** Inspecting `model.config._experts_implementation` after `from_pretrained(enable_expert_parallel=True)`:

```
model.config._experts_implementation: grouped_mm
```

Transformers automatically sets this to `"grouped_mm"` when EP is enabled. The `@use_experts_implementation` decorator on `Qwen3MoeExperts` wraps `forward` in a dispatcher that calls `experts_interface.get_interface(self.config._experts_implementation, original_forward)` — which routes to `grouped_mm_experts_forward` (an EP-aware kernel that uses `torch._grouped_mm` and handles sentinels via masking, **no `F.one_hot` call**).

So step 3 never calls `F.one_hot`. The "eager" `Qwen3MoeExperts.forward` (with `F.one_hot`) is dead code under EP.

Liger's `_patch_swiglu_module(experts, LigerExperts)` does:

```python
_bind_method_to_module(module, "forward", liger_module.forward)
```

This replaces the dispatcher entirely with `LigerExperts.forward`, which calls `F.one_hot(top_k_index, num_classes=self.num_experts)` directly. With EP active:

- `self.num_experts` was overwritten by `GroupedGemmParallel.shard_tensor` to `num_local_experts` (= total // ep_size).
- `top_k_index` was processed by `RouterParallel._prepare_output_fn` which sets the sentinel value to `num_local_experts` (line 1209 of `tensor_parallel.py`).
- So `top_k_index.max() == num_classes` — exactly the OOB case `F.one_hot` errors on.

`LigerExperts.forward` does have the sentinel-skip: `if expert_idx == self.num_experts: continue` (line 75). But that runs INSIDE the `for expert_idx in expert_hit:` loop — **after** `F.one_hot` has already errored.

### Root cause (one-liner)

> Liger's `_patch_swiglu_module` replaces the EP-aware experts dispatcher (`grouped_mm_experts_forward`) with `LigerExperts.forward`, which uses `F.one_hot(top_k_index, num_classes=num_local_experts)` — but transformers' `RouterParallel` sets sentinel = `num_local_experts`, making the index OOB for `F.one_hot`.

### 2026-04-29 — Workaround validated (no Liger patch)

`apply_liger_kernel_to_qwen3_moe(model=model, swiglu=False, rope=True, rms_norm=True, fused_linear_cross_entropy=True)` — disable just the swiglu/experts patch, keep all the other Liger components.

| Step | Result |
| --- | --- |
| 5. Forward with Liger (swiglu=False) | ✅ logits finite, max diff vs eager = 7.81e-3 |

**Recommendation: pass `swiglu=False` to `apply_liger_kernel_to_qwen3_moe` whenever EP is enabled.** Keep transformers' `grouped_mm_experts_forward` (or `sonicmoe_experts_forward`) for the experts; Liger handles RMSNorm + RoPE + fused linear cross-entropy.

What we lose: Liger's `LigerSiLUMulFunction` (fused SiLU + multiply Triton kernel). What we keep: `LigerRMSNorm`, `liger_rotary_pos_emb`, and `LigerFusedLinearCrossEntropyLoss` — usually the bigger MFU wins anyway.

### Alternative kernels for SwiGLU under EP

Already integrated and EP-aware:

- **`grouped_mm_experts_forward`** (transformers): default when `enable_expert_parallel=True`. Uses `torch._grouped_mm` for the gate_up_proj batched matmul and a `clamp + masked_fill` to handle sentinels. Standard PyTorch op since 2.5+.
- **`sonicmoe_experts_forward`** (transformers, optional via `--experts_implementation sonicmoe`): CuteDSL-fused MoE kernel ("MegaBlocks-like"). Used in the chunked-CE long-context champions. Higher MFU than `grouped_mm` at large batch.

Liger's `LigerSiLUMulFunction` is a separate Triton kernel for `silu(gate) * up`. Under EP it would need to be invoked from inside `grouped_mm_experts_forward` or `sonicmoe_experts_forward`, replacing the existing `_apply_gate` (transformers' default `act_fn(gate) * up`). That's a transformers-side patch, not a Liger one.

### Updated H1 status

The original H1 in `upstream_todo.md` ("Liger fails on 3D weights") was incorrect. Updated H1 wording: "Liger's swiglu patch bypasses transformers' EP dispatcher and calls `F.one_hot(num_classes=num_local_experts)` with sentinel index = `num_local_experts` → OOB. Workaround: `apply_liger_kernel_to_qwen3_moe(swiglu=False)`."

### TODO

- [ ] Push a small Liger PR: under EP-active config (detected via `model.has_ep` or `model.config._experts_implementation`), the `swiglu=True` path should default to a no-op (or call the dispatcher). Open issue first with this repro.
- [ ] Benchmark Liger-without-swiglu vs eager+grouped_mm at 30B Qwen3-MoE to quantify the MFU gain from the other Liger components.
