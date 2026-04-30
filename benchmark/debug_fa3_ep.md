# Debug: FA3 + Expert Parallelism

## Status: RESOLVED (2026-04-27) — was a self-inflicted offline-mode race in `trl/scripts/sft.py`

## The original symptom

FA3 (`kernels-community/vllm-flash-attn3`) + EP appeared to crash during FSDP2 preparation:

```
[RANK 0] The model parameters are not sharded by DTensor, we skip the TP preparation.
KeyError: "Invalid mesh_dim_names ('dp_shard_cp',)" specified. Valid mesh_dim_names are ['tp']."
```

Earlier debugging (2026-04-17) blamed an interaction between FA3's kernel injection and the DTensor hook system in `tensor_parallel.py`. That analysis was wrong: the actual crash on 2026-04-27 was a hub-fetch failure earlier in `from_pretrained`:

```
ValueError: An error occurred while trying to load from 'kernels-community/vllm-flash-attn3':
Cannot reach .../api/models/.../tree/main/build: offline mode is enabled.
```

The earlier mesh-dim crash either was an older bug since fixed in transformers/accelerate, or was masking the true cause and got mis-attributed.

## What actually happens (2026-04-27 codepath)

`trl/scripts/sft.py` flips `HF_HUB_OFFLINE=1` for EP runs to avoid a 16-rank × 16-shard race against the hub during weight loading. Before flipping offline, it pre-warms the sonicmoe kernel so the cached entry survives the offline flip.

What the pre-warm missed: FA3 loads via **two separate code paths** during `from_pretrained`, both of which call `get_kernel(repo_id)` independently:

1. `transformers.integrations.hub_kernels.load_and_register_attn_kernel` — registers the attention function in `ALL_ATTENTION_FUNCTIONS` via `get_kernel`.
2. `transformers.modeling_flash_attention_utils.lazy_import_flash_attention` — sets module-level `_flash_fn`, `_flash_varlen_fn`, `_pad_fn`, `_unpad_fn` via `_lazy_imports`, which itself calls `get_kernel`.

Pre-warming sonicmoe (which uses `@functools.cache` on a single function) didn't pre-warm either FA3 path. After the offline flip, the model `__init__` hits `_check_and_adjust_attn_implementation` → `lazy_import_flash_attention` → hub call → `OfflineModeIsEnabled` → ValueError.

## The fix

Pre-warm both FA3 paths before the offline flip. In `trl/scripts/sft.py`, inside the `if training_args.enable_expert_parallel:` block:

```python
if model_args.attn_implementation and "/" in model_args.attn_implementation:
    from transformers.integrations.hub_kernels import load_and_register_attn_kernel
    from transformers.modeling_flash_attention_utils import lazy_import_flash_attention

    load_and_register_attn_kernel(model_args.attn_implementation)
    lazy_import_flash_attention(model_args.attn_implementation)
```

The heuristic `"/" in model_args.attn_implementation` distinguishes hub-kernel attention IDs (`kernels-community/vllm-flash-attn3`) from built-in names (`sdpa`, `eager`, `flash_attention_2`).

## Verification

Job 22092267 (2 nodes, 16k, FSDP2 + EP=8 + FA3 + sonicmoe, 5 steps): training launched cleanly, no offline-mode error, **window MFU 42.66 %**, TPS ~115k. See `benchmark/report.md` (2026-04-27 FA3+EP section) for the full per-step table.

## Caveat: numerical correctness is unrelated

While FA3 + EP now runs at high throughput, the loss/grad_norm/entropy go to NaN immediately. This is the pre-existing **RouterParallel sentinel bug** (`benchmark/CLAUDE.md`, "Critical Bug: RouterParallel Shape Mismatch"). It applies to all EP runs on this branch regardless of attention backend; FA3 doesn't introduce it and doesn't fix it. The MFU number is meaningful (real matmul throughput); the gradients are not.

## Was the earlier "DTensor count" theory wrong?

Yes. The 2026-04-17 analysis observed that sdpa+EP runs did NOT emit `"The model parameters are not sharded by DTensor"` while FA3+EP runs DID, and concluded FA3 was somehow stripping DTensors. Re-checking on 2026-04-27 with `benchmark/debug_fa3_ep_dtensor.py` (2 GPUs, world_size=2):

- Both sdpa+EP and FA3+EP report **0/531 DTensor params** after `from_pretrained`.
- Both have `model._tp_size = 2` and a device mesh.

The DTensor conversion happens during weight loading in `core_model_loading.py:1351` (`if device_mesh and tp_plan:`), but only for parameter names matching the EP plan (gate, experts) — not for attention or embedding weights. With 2 GPUs the EP-only plan has many keys but the loader still produces 0 DTensors at the end of `from_pretrained` for both attention backends. Whatever DTensor-creation difference the earlier analysis observed has since been refactored away or wasn't actually reproducible.

## Lessons

- **Self-inflicted hub-offline failures look identical to upstream incompatibilities.** When a "framework bug" only fires under EP, audit your own EP-specific setup code before diving into framework internals.
- **Pre-warming kernels by hand is fragile.** Each kernel-loading function is a separate cache. Hub-offline mode should be set with knowledge of what's been warmed; a missing pre-warm step looks like a network failure.
- **Both FA3 entry points must be warmed when going offline.** `load_and_register_attn_kernel` and `lazy_import_flash_attention` are called from independent transformers code paths.
