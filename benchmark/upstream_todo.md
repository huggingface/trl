# TODO: Local patches & not-working items (wrap-up)

Self-contained context for a fresh session. Goal: pick up debugging at night without re-reading the full report.

## Where things live

- **Branch**: `ds-ep-integration` (transformers fork) — cloned from `qwen3-moe-ep-v2`
- **Working dirs**: `/fsx/amine_dirhoussi/{transformers,accelerate,trl}` (all installed editable in `/fsx/amine_dirhoussi/trl/.venv`)
- **Full patch detail**: `benchmark/local_only_patches.md` (line numbers + rationale)
- **Latest results**: `benchmark/report.md` (chronological), `benchmark/sft_benchmark_notion.md` (consolidated tables)
- **Open transformers PRs**: `#45662` (EP+FSDP DTensor), `#45621` (sonicmoe Ilyas patch), `#45548` (DS-Z3 EP loading), `#45649` (FSDP cpu_ram_efficient_loading), `#45473` (RouterParallel routing), `#45436` (EP 2D mesh)

## Status legend

- 🟢 **Ready to PR** — needs writing up but no blocker
- 🟡 **Keep local** — proper fix is in another repo (PyTorch / accelerate / DS)
- ⛔ **Blocked / not yet attempted** — needs investigation
- 🔧 **Active workaround in place** — works for current benchmarks but not production-correct
- 🧪 **To test / explore** — frontier work, not a blocker
- 🗑️ **Debug-only** — remove before any PR

---

## A. Transformers — needs upstream PR

### A1. 🟢 `integrations/sonicmoe.py` — EP/DTensor support for the sonicmoe kernel

**File**: `transformers/src/transformers/integrations/sonicmoe.py` (3 hunks — see `local_only_patches.md` §1)

**What**: `.to_local()` on `gate_up_proj`/`down_proj` DTensors before `.permute()`, plus wrapper-level `clamp(0, num_experts-1)` for sentinel rows. (The `masked_fill` was removed 2026-04-29 — `RouterParallel` already zeros sentinel scores upstream.)

**Why blocked from PR-ing yet**: Depends on `#45621` (sonicmoe → IlyasMoutawwakil/sonic-moe@main) **and** `#45662` (EP+FSDP DTensor wrap) — both must land first. The wrapper clamp is the workaround for the kernel's hand-written backward producing NaN gradients on sentinels when going through `DTensor.to_local()`.

**Cost of the workaround**: ~2 pp MFU (kernel runs compute on sentinel rows, then multiplies by 0).

**Sonic-moe-side upstream issue (Dao-AILab/sonic-moe)**: standalone kernel-only repro found 2026-04-29 — see `benchmark/sonic_moe_upstream_repro.md` and `benchmark/test_sonic_repro_minimal.py`. The kernel produces wrong upstream-flowing input gradients (`dh`, `ds`) when many tokens have all `top_k` slots set to the EP sentinel. Forward output and parameter gradients (`dw1`, `dw2`) are correct. `[ ]` File the upstream issue with this repro.

**Next steps**:

1. File the Dao-AILab/sonic-moe issue with `benchmark/sonic_moe_upstream_repro.md` + the standalone scripts.
2. Wait for `#45621` + `#45662` to merge.
3. Open follow-up PR citing both, explaining the autograd-through-`to_local` backward gap.
4. Once kernel patches land, **remove the clamp** from the wrapper.
5. (Optional) revisit kernel-native sentinel-skip as a non-zero perf win.

---

## B. Transformers — keep local (proper fix is elsewhere)

### B1. 🟡 `trainer.py:_clip_grad_norm` skip when `has_ep`

**File**: `transformers/src/transformers/trainer.py` (in `Trainer._clip_grad_norm`)

**What**: Returns `tensor(0.0)` instead of calling `accelerator.clip_grad_norm_` when `model.has_ep` is set.

**Why local-only**: `clip_grad_norm_` calls `_foreach_norm` which stacks per-param norms; stacking DTensors on different meshes (EP mesh + FSDP DP mesh) errors with `RuntimeError: All operands in aten.stack.default must have the same mesh`. Proper fix is in PyTorch's `clip_grad_norm_` (or accelerate's wrapper), not the Trainer.

**Risk**: Returns 0 grad-norm for telemetry; **gradients are not actually clipped to `max_grad_norm`**. Fine for benchmarks. Unsafe for real training.

**Next steps**:

1. File a PyTorch issue describing the cross-mesh stack failure with a minimal repro.
2. Consider an accelerate-level wrapper that handles per-mesh grouping, then unblock the proper fix.
3. See also Patch C in section D — under DS-Z3 the skip should be gated off (DS handles MoE grad norms via its own per-group path).

---

## C. Accelerate — could upstream

### C1. 🟢 `accelerator.py:_prepare_tp` skip when `has_ep`

**File**: `.venv` site-packages — `/fsx/amine_dirhoussi/trl/.venv/lib/python3.11/site-packages/accelerate/accelerator.py` (~line 1607). Patched in place — 5 lines after the existing "no DTensor → skip" guard.

**What**: Returns early from `_prepare_tp` when `model.has_ep` is set.

**Why needed**: Post-`#45662`, EP params become DTensors → existing skip guard doesn't fire → `_prepare_tp` reaches `from transformers.integrations.tensor_parallel import ReplicateParallel` → **`ImportError`** (class added upstream after our fork point on `qwen3-moe-ep-v2`).

**Two upstream paths**:

1. Add a `ReplicateParallel` shim to the transformers fork (small but ugly — papers over version mismatch).
2. Push the `has_ep` skip into accelerate upstream (cleaner — single check).

**Next steps**:

1. Pick path 2. Open accelerate PR (single change, easy review).
2. Reference `#45662` so reviewer sees the EP-DTensor context.

**Revert**: `uv pip install --reinstall --no-deps accelerate`

### C2. 🟢 EP + `cpu_ram_efficient_loading=True` — guard + proper coexistence fix

**Files**: `accelerate/utils/fsdp_utils.py:fsdp2_prepare_model` (proper fix) and `transformers/src/transformers/modeling_utils.py` (`from_pretrained`, immediate guard).

**What's broken**: When `enable_expert_parallel=True` and `cpu_ram_efficient_loading=True` are both set, loading silently produces a broken model — no crash, just degenerate loss / wrong outputs (easy to miss). Two interacting steps in `fsdp2_prepare_model` cause it:

1. `model.to("meta")` destroys _everything_, including the EP per-rank-local expert slices that were just sharded by transformers' EP plan.
2. `fsdp2_load_full_state_dict` then broadcasts rank-0's full state dict to every rank, overwriting the per-rank EP partition with rank-0's slice.

Same root family as the DS-Z3 rank-ordering mismatch (D-blocker): "DP-replica sync assumes every rank holds the same data" is incompatible with EP's per-rank partitioning. Today there is no in-tree warning — users reaching for `cpu_ram_efficient_loading` (the canonical big-model loading recipe) won't know it conflicts with EP.

**The duplicated-load problem under EP** (measured 2026-04-29 on Qwen3-30B-A3B EP=4 by walking `model.named_parameters()` after `from_pretrained` — with AND without `FSDP_CPU_RAM_EFFICIENT_LOADING=True ACCELERATE_USE_FSDP=True`, both behave identically):

| param group                          | per-rank size | DTensor?                 | replicated across ranks?           |
| ------------------------------------ | ------------- | ------------------------ | ---------------------------------- |
| `embed_tokens` + `lm_head`           | 1.16 GiB      | no                       | yes                                |
| q/k/v/o projections (48 layers)      | 1.69 GiB      | no                       | yes                                |
| norms + router gates                 | 25 MiB        | no                       | yes                                |
| **dense subtotal (replicated)**      | **2.87 GiB**  | **plain `torch.Tensor`** | **yes — same bytes on every rank** |
| `experts.gate_up_proj` + `down_proj` | 13.50 GiB     | yes                      | no — sharded `(128,…) → (32,…)`    |

Each rank materializes its own copy of 2.87 GiB of dense params on its own GPU before FSDP2 ever runs (16.4 GiB total / rank, peak 16.57 GiB during load). Scales linearly with rank count.

**Important: today's `cpu_ram_efficient_loading=True` is a no-op at the loading stage under EP** (verified empirically — same 2.87 GiB replicated with both env vars set). The rank-0-only short-circuits in `modeling_utils.py:4654` and `:4690` only gate `_move_missing_keys_from_meta_to_cpu_or_disk` and `_initialize_missing_keys`. The actual disk loading in `core_model_loading.py:convert_and_load_state_dict_in_model` has **no rank-0 short-circuit** — every rank calls `spawn_materialize(tensor, device=device_map[""])` with `device_map[""]` set to the local cuda. So even if you "fixed" the broadcast clobber, you'd still get duplicated disk loads on every rank.

**Proper fix requires TWO patches**, not one:

**Patch 1 — transformers loading-side rank-0 gate**: in `convert_and_load_state_dict_in_model`, when `is_fsdp_enabled()` is true and current rank ≠ 0, only materialize params that ARE matched by the TP/EP plan (so each rank gets its EP slice). Skip `spawn_materialize` for plain (non-plan) params on non-rank-0 — leave them on meta to be filled by the FSDP2 broadcast.

**Patch 2 — accelerate `fsdp2_prepare_model` EP-aware broadcast**: capture EP params (`ignored_params`) before the meta move, skip them in the broadcast loop, restore each rank's own snapshot after `fully_shard`. Sketch below.

**Patch 1 detail (transformers, `core_model_loading.py`)**: today line 1375 unconditionally calls `spawn_materialize(tensor, device=device_map[""])` for any param not matched by the TP/EP plan. Add a rank-0 gate gated on `is_fsdp_enabled()`:

```python
if future_or_tensor is None:
    if is_fsdp_enabled() and not is_local_dist_rank_0():
        # rank > 0 under cpu_ram_efficient_loading: leave non-plan params on meta;
        # FSDP2's broadcast will fill them from rank 0.
        future_or_tensor = lambda: torch.empty_like(tensor[...].to("meta"))
    else:
        param_device = get_device(device_map, renamed_key, valid_torch_device=True)
        future_or_tensor = spawn_materialize(thread_pool, tensor, param_device, _dtype)
```

Sketch only — actual implementation needs to thread the `meta` placeholder through `set_param_for_module` so the param is actually attached to the module on rank > 0 (otherwise FSDP2's `fully_shard` won't see it). The point is: today every rank reads from disk; the gate makes only rank 0 read.

**Patch 2 detail (accelerate, `fsdp_utils.fsdp2_prepare_model`)**: capture EP params (`ignored_params`) **before** the meta move, skip them in the broadcast loop, restore each rank's own snapshot **after** `fully_shard`. Sketch:

```python
ignored_params = fsdp2_kwargs.get("ignored_params") or set()
original_sd = model.state_dict()
ignored_data = {n: p.data.clone()
                for n, p in model.named_parameters() if p in ignored_params}

if fsdp2_plugin.cpu_ram_efficient_loading:
    model = model.to(torch.device("meta"))

fully_shard(model, **fsdp2_kwargs)

if fsdp2_plugin.cpu_ram_efficient_loading:
    sd_for_broadcast = {k: v for k, v in original_sd.items() if k not in ignored_data}
    fsdp2_load_full_state_dict(accelerator, model, sd_for_broadcast, ...)

for name, data in ignored_data.items():
    parent, attr = get_module_from_name(model, name)
    getattr(parent, attr).data = data.to(accelerator.device)
```

Plus ~3 lines in `fsdp2_load_full_state_dict` to filter non-DTensor params on the rank>0 branch (it currently iterates `meta_sharded_sd.items()` blindly, so iteration counts must stay in sync with the filtered rank-0 dict).

**Why accelerate owns Patch 2**: this is fundamentally an FSDP-prepare concern — accelerate already exposes `ignored_params` to FSDP, this just makes the meta-move + broadcast respect that exclusion.

**Immediate-action `ValueError` safety net** (until Patches 1+2 land): raise in `from_pretrained` when both `distributed_config.enable_expert_parallel` and `cpu_ram_efficient_loading=True` are set. Message should explain the rank-0 broadcast would overwrite the EP partition (Patch 2 unblocks this) AND that today the flag is a no-op anyway under EP (Patch 1 unblocks this). Plus a one-line note in the FSDP / EP docs.

**Next steps** (ordered by dependency):

1. Open transformers PR for the `ValueError` guard — ships immediately, blocks silent failure once Patch 2 is in flight.
2. Open transformers PR for Patch 1 (rank-0 gate in `convert_and_load_state_dict_in_model`). Smoke-test: with `FSDP_CPU_RAM_EFFICIENT_LOADING=True ACCELERATE_USE_FSDP=True`, rank > 0 should show 0 GiB cuda PLAIN (currently 2.87 GiB) and the dense params should be on meta after `from_pretrained`.
3. Open accelerate PR for Patch 2 (`fsdp2_prepare_model` + `fsdp2_load_full_state_dict`).
4. End-to-end smoke-test: 30B-A3B EP=8 + FSDP2 + `cpu_ram_efficient_loading=True` should load and produce healthy loss for ~50 steps. Per-rank load-time peak should drop from 16.4 GiB to ~13.5 GiB (only the EP slice).

**(Stretch)** Make the transformers loader fully EP-aware so each rank loads its own slice from disk _without_ the rank-0 broadcast roundtrip — would require plumbing the EP mesh into the meta-init path. Larger scope, probably blocked on `#45662`. Track as a follow-up if anyone needs it.

---

## D-works. DeepSpeed-Z2 + EP — fixed and working

> **Status**: ✅ **WORKING** as of 2026-04-28. DS-Z2+EP=8 is the long-context champion path (45.81 % MFU at 32k, 57.23 % at 64k, 69.10 % at 128k — all on 2-4 nodes). Loss healthy with sonicmoe + clamp wrapper. This is the recipe used in every chunked-CE long-context champion in the report.

**What we changed to make it work** (across 3 repos — all local, all need upstreaming):

1. **`transformers/integrations/tensor_parallel.py:GroupedGemmParallel.post_shard_wrap`** — branch on backend. When DeepSpeed is active (detected via `_hf_deepspeed_config_weak_ref`), tag the EP-sharded param with DS's MoE convention instead of wrapping it as a DTensor:
    ```python
    param.allreduce = False
    param.group_name = f"ep_size_{ep_size}"
    return param  # plain tensor, no DTensor wrap
    ```
    Without this, EP params would be FSDP-style DTensors which DS-Z2's optimizer can't parse.
2. **`transformers/trainer.py:create_accelerator_and_postprocess`** — before `Accelerator()` instantiation, call `deepspeed.utils.groups._create_expert_and_data_parallel(ep_size)` so DS knows about the MoE groups when `deepspeed.initialize` runs `_broadcast_model` and `is_moe_param`.
3. **DS engine `has_moe_layers` detection patch** (`/fsx/amine_dirhoussi/trl/.venv/lib/python3.11/site-packages/deepspeed/runtime/engine.py`) — extend the loop that auto-detects native DS MoE layers to ALSO recognize external EP via `any(getattr(p, "allreduce", True) is False for p in module.parameters(recurse=False))`. Without this, `expert_data_parallel_group=None` is passed to the DS optimizer and Z2's `_configure_moe_settings` errors with `TypeError: 'NoneType' object is not subscriptable`.
4. **`transformers/trainer.py:create_optimizer`** — split MoE params into a separate optimizer group via `split_params_into_different_moe_groups_for_optimizer` so DS routes their grad reduce through `expert_data_parallel_group` (the small group) instead of the full DP group.
5. **`transformers/trainer.py:_clip_grad_norm`** — `has_ep` skip gated to FSDP-only path (DS handles MoE grad norms via its own per-group mechanism).
6. **TRL side** (in `sft_trainer.py` EP branch — see G2): pre-initialize `deepspeed.comm` if SP is configured, so EP's `distribute_model` can use DS process groups.
7. **Sonicmoe wrapper-level clamp** (`transformers/integrations/sonicmoe.py`, A1 in this todo): kernel sees DTensor.to_local() inputs and produces NaN gradients on sentinel rows; wrapper clamp+masked_fill is the workaround.

**Cost**: ~2 pp MFU vs theoretical (kernel does compute on sentinel rows, then masks them — see A1).

**Why Z2 works but Z3 doesn't**: short version — `deepspeed/runtime/zero/stage_1_and_2.py` has full `is_moe_param` plumbing (param-group split, per-group grad reduce via `expert_data_parallel_group`). `stage3.py` has **zero** MoE awareness. Z3 also makes a deeper assumption — "the same param name has the same logical data on every rank" — which transformers' EP violates: `model.layers.0.mlp.experts.gate_up_proj` on rank 0 holds experts 0–15 but on rank 8 holds 128–143; Z3's all-gather across the full DP group concatenates these into garbage. Z2 doesn't all-gather params (only grads/optim), so it sidesteps the issue.

**Full reasoning + iteration log + code references in [`benchmark/debug_sp_ep_sonic.md`](./debug_sp_ep_sonic.md)** (sections "2026-04-28: Root cause of why this is hard with ZeRO-3" line ~298 and "Practical path that works today: ZeRO-2 + EP" line ~310).

**Upstream split** (when ready):

1. Patches (1)+(2)+(4)+(5) → transformers PR depending on `#45662` landing first.
2. Patch (3) → DS PR (or push our fix into deepspeed `engine.py`'s MoE detection path).
3. TRL side (6) → falls under G2 in this todo.
4. (7) → A1 in this todo.

---

## D. DeepSpeed-Z3 + EP — 3-patch tagging approach (tried, broken loss)

> **Issue**: DS-Z3 + EP doesn't work because DS's MoE infrastructure assumes **explicit all-to-all token dispatch** (its `expert_parallel_group` is the all-to-all comm group, and `deepspeed.moe.layer.MoE` moves tokens across ranks based on routing), but transformers' EP is **replicate-mask + all-reduce** — every rank holds the full token batch, masks routing scores for non-local experts, all-reduces expert outputs at the end. The architectures aren't directly composable. We did try the 3-patch tagging approach (commit `cd52547f87`): tag EP params with `allreduce=False` + `group_name="ep_size_8"` so DS routes them through `expert_data_parallel_group`, create the DS expert groups in `Trainer.create_accelerator_and_postprocess`, gate `_clip_grad_norm` `has_ep` skip on FSDP-only. Smoke test at 16k Z3 ran 50 steps but loss=32-37 — DS's broadcast inside `expert_data_parallel_group` clobbered EP slices on DP-of-EP siblings (DS expects every rank in that group to hold the _same_ expert slice, which is true in DS native MoE; transformers' EP partitions by `self.rank` so siblings hold _different_ slices). Earlier "skip DTensors at broadcast/partition" patches also failed downstream with `'NoneType' has no attribute 'ds_numel'`. Full iteration log + DS group-construction notes in `debug_sp_ep_sonic.md` (Iteration 4 + commit table). The path forward is to align transformers' EP partitioning with DS's `expert_parallel_group` rank ordering before the broadcast — not yet attempted.

**Status**: Implemented in commit `cd52547f87` of the `ds-ep-integration` branch — ran 50 steps at 16k Z3 but loss=32-37 (broadcast clobbered EP slices on DP-of-EP siblings, see issue block above). All 3 patches reverted. Worth retrying because DS-Z3+SP+EP would compound the 19% (SP) win with EP's expert sharding for long-context MoE. See `local_only_patches.md` §5 and `debug_sp_ep_sonic.md` (Iteration 4 + commit table) for full reasoning and rank-ordering analysis.

**Key insight (still load-bearing)**: DS has first-class MoE/EP infrastructure in `deepspeed/moe/` — convention is `param.allreduce = False` + `param.group_name = "ep_size_N"`. The plumbing works; what failed is that DS's `expert_data_parallel_group` member ranks expect to hold the same expert slice (DS's MoE convention), but transformers' EP partitions by `self.rank` so DP-of-EP siblings hold different slices.

### D1. 🔧 Patch A: `tensor_parallel.py:GroupedGemmParallel.post_shard_wrap` — backend branch

**What** (already implemented in `cd52547f87`): when DS-Z3 active, return plain tensor with `param.allreduce = False` + `param.group_name = f"ep_size_{ep_size}"`. Otherwise (FSDP path), keep the DTensor wrap.

**Status**: code wrote and tested; the tagging itself is correct but doesn't help if the rank ordering issue (below) isn't also fixed.

### D2. 🔧 Patch B: `Trainer.create_accelerator_and_postprocess` — create DS expert groups

**What** (already implemented in `cd52547f87`): before `Accelerator()` instantiation, gated on `model.has_ep and is_deepspeed_zero3_enabled()`, call `deepspeed.utils.groups._create_expert_and_data_parallel(self.model.tp_size)`.

**Why ordering matters**: after `torch.distributed` init (so groups can be created), before `deepspeed.initialize` (which calls `_broadcast_model` and inspects `is_moe_param`).

**Status**: implemented; groups are correctly constructed but they assume the wrong rank-to-slice mapping (see D-blocker below).

### D3. 🔧 Patch C: gate the existing `_clip_grad_norm has_ep` skip

**What** (already implemented in `cd52547f87`): skip the skip when `is_deepspeed_zero3_enabled()` — DS handles MoE grad norms via its own per-group path.

### D-blocker. ⛔ Rank-ordering mismatch

The remaining work to make D1–D3 actually train. With 16 GPUs and EP=8:

- DS computes `expert_parallel_group = [[0..7], [8..15]]` and `expert_data_parallel_group = [[0,8], [1,9], ..., [7,15]]`. DS expects rank 0 and rank 8 to hold the **same** expert slice (they are DP replicas of EP rank 0).
- Transformers' `GroupedGemmParallel.shard_tensor` partitions experts by `self.rank` directly, so rank 0 holds experts `[0..15]` and rank 8 holds experts `[0..15]` only if `self.rank` resets per-EP-group. If `self.rank` is the global rank, rank 8 holds experts `[128..143]` — and DS's broadcast inside `[0,8]` will overwrite rank 8's slice with rank 0's, destroying half the model.

**Fix paths**:

1. Make transformers' EP partition use the same intra-EP-group rank that DS uses (`global_rank % ep_size`) — requires plumbing the EP mesh into `GroupedGemmParallel` so it knows the EP-local rank, not just the global rank.
2. Construct DS's `expert_parallel_group` to match transformers' partitioning order. Instead of calling DS's `_create_expert_and_data_parallel` helper, build the groups manually with `dist.new_group(ranks=...)` so the rank ordering matches what transformers EP produces.
3. Verify with a forward smoke test: each rank in `expert_data_parallel_group[g]` should produce identical experts for indices `[start, end)`. If not, the broadcast will corrupt weights.

**Estimated effort**: 1 day to verify which slicing convention each side actually uses (read both `GroupedGemmParallel.shard_tensor` and `deepspeed/utils/groups._create_expert_and_data_parallel`), then 1 day to wire the matching groups + smoke test.

---

## E. Compile-related blockers (long-context)

These come from the recent compile sweep — see `report.md` "Known errors / next steps" section.

### E1. ⛔ FSDP + EP + compile: Adam `_group_tensors_by_device_and_dtype` crash

> **Issue**: FSDP + EP + compile crashes at the Adam optimizer step because EP DTensors live on the EP mesh (size 8) and FSDP DP DTensors live on the FSDP DP mesh (size 16); under compile, `_group_tensors_by_device_and_dtype` strict-asserts that grouped tensors share device+dtype, and the two different DTensor meshes split foreach grouping incorrectly. We tried 3 variants — `FSDP+EP=8 + sonicmoe + compile`, `FSDP+EP=8 + grouped_mm + compile`, and the same with `--optim adamw_torch` (non-fused Adam) — all 3 crashed identically. DS-Z2+EP+compile works because DS uses plain `nn.Parameter` (no DTensor mesh).

**Repro**: any FSDP + EP=8 + sonicmoe + compile config (16k+, 2n+). Tested 3 variants:

- FA3 + sonicmoe + compile
- FA3 + grouped_mm + compile
- sonicmoe + compile + `--optim adamw_torch` (non-fused Adam)

All hit the same compiled foreach assert at the optimizer step.

**Root cause**: `nn.Parameter` inputs are a mix of:

- EP DTensors on the EP `device_mesh` (size 8)
- FSDP DP DTensors on the FSDP DP mesh (size 16)

Under compile, `_group_tensors_by_device_and_dtype` strict-asserts grouped tensors share device+dtype; with two different DTensor meshes in play, the foreach grouping splits incorrectly. Non-fused Adam doesn't help — `foreach` groups regardless of `--optim`.

**Why DS-Z2+EP+compile works**: DS uses plain `nn.Parameter` (with `allreduce=False` / `group_name` markers), no DTensor mesh.

**Fix paths** (ordered by likely effort):

1. Make EP wrap return DTensor on a 2D `(dp, ep)` mesh shared with FSDP (single mesh family) — cleanest, biggest scope.
2. Custom `_group_tensors_by_device_and_dtype` that treats EP and FSDP DTensors as compatible if device+dtype match — surgical patch in PyTorch.
3. Skip `foreach` entirely under EP — perf hit, easiest.

**Next steps**: Try (2) as a local patch first to confirm it unblocks the optimizer step; then decide between (1) and (2) for upstream.

### E2. ⛔ FSDP + EP + CP=4 + compile @ 64k: Triton cache file-not-found

> **Issue**: FSDP + EP + CP=4 + compile @ 64k crashes during inductor compilation with `InductorError: RuntimeError: CUDA driver error: file not found` on one or more ranks. We already set per-rank `TRITON_CACHE_DIR=/tmp/triton-rank-${RANK}-${HOSTNAME}` in `sft.py` before any torch import, which fixes startup contention; the crash now happens later in the inductor cache pipeline, suggesting `~/.cache/torch_inductor` (or similar) is still on FSx and has cross-rank contention. Likely needs `TORCHINDUCTOR_CACHE_DIR` overridden per-rank too.

**Repro**: 64k FSDP + EP=8 + CP=4 + sonicmoe + compile.

**Symptom**: `InductorError: RuntimeError: CUDA driver error: file not found` from one or more ranks during inductor compilation. Even with per-rank `TRITON_CACHE_DIR` set in `sft.py` (top of file, before any torch import).

**Hypothesis**: per-rank dir trick fixed startup contention; a later compile-graph cache step still hits a shared FSx-mounted path.

**Fix paths**:

1. Move `TRITON_CACHE_DIR` to node-local `/tmp` instead of FSx-mounted home.
2. Verify `os.uname().nodename` is unique per node (it is) and the dir exists before any torch import (it does).
3. Check if inductor caches into `~/.cache/torch_inductor` separately and needs a per-rank override too. Set `TORCHINDUCTOR_CACHE_DIR` per-rank.

**Next steps**: Try (3) first — `TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-rank-${RANK}-${HOSTNAME}` next to the existing TRITON env. Cheap to test.

---

## F. Architectural ceiling — EP expert buffer

### F1. ⛔ EP-replicated expert buffer @ 32k+ — kernel rewrite

> **Issue**: EP doesn't fit at 32k+ per-rank seq because transformers' EP replicates routing across all EP ranks (it's TP-style EP, not all-to-all), so every rank materializes a `seq × num_local_experts × moe_intermediate × 2 bytes` activation tensor — 18.55 GiB at 32k per-rank, 37.09 GiB at 64k. We tried DS-Z2+EP=8 at 32k @ 2n and 4n with and without compile, and FSDP+EP=8+CP=8 at 256k (per-rank seq=32k after CP); all 5 attempts OOM at the same single allocation. Workarounds today: halve per-rank seq via CP=2 (FSDP+EP+CP=2 at 32k = 15.6% MFU, training-correct) or drop EP entirely and use SP (DS-Z3+SP=2 at 32k = 21.98% MFU, current 32k champion). Real fix is a streaming kernel rewrite that doesn't materialize the full tensor.

**Symptom**: 18.55 GiB OOM at 32k per-rank seq (any EP=8 config). 37.09 GiB at 64k. Same allocation regardless of FSDP↔DS-Z2↔DS-Z3, compile on/off, 2n/4n.

**What it is**: `seq × num_local_experts × moe_intermediate × 2 bytes` per rank. Transformers' EP replicates routing across all EP ranks (it's a TP-style EP, not all-to-all), so every rank materializes this full tensor.

**Numbers (Qwen3-30B-A3B, EP=8)**:
| Per-rank seq | Buffer |
| ------------ | ------------ |
| 16k | 9.275 GiB |
| 32k | 18.55 GiB |
| 64k | 37.09 GiB |

**Why workarounds work today**:

- 16k EP=8 fits (9.3 GiB out of ~80 GB, room for params/optim/grads).
- 32k FSDP+EP+CP=2 fits (CP halves per-rank seq → 9.3 GiB buffer) — that's why 15.6 % MFU 32k baseline survives.
- 256k FSDP+EP+CP=8 OOMs (256k/8 = 32k per-rank → 18.55 GiB → no headroom).

**Fix paths** (ordered by scope):

1. **Combine EP with seq sharding** at long context (EP+CP=2 at 32k, EP+CP=4 at 64k...). Already proven correct — just slow (15.6 % at 32k vs 21.98 % SP-only).
2. **Drop EP degree** at long context (EP=4 → buffer halves; EP=2 → quarters). Loses the 16k throughput champion's edge.
3. **Stream the expert dispatch** — kernel-level rewrite to avoid materializing the full `(seq, num_local_experts, moe_intermediate)` tensor. Biggest payoff but biggest scope.

**Next steps**:

1. Profile (1) at 64k FSDP+EP=8+CP=4 to see if the buffer-halved path beats DS-Z3+SP=4 (currently 20.5 %).
2. Consider (3) as a long-term project — file an issue / RFC in transformers MoE integrations.

---

## G. TRL — known runtime requirements (no patch yet)

### G1. 🔧 SP runs need `--pad_to_multiple_of 8`

**Symptom**: DS-Z3 + SP + Ulysses crashes around step 25 with `ValueError: batch's seqlen=X isn't divisible by sp-size=N`.

**Workaround**: Pass `--pad_to_multiple_of 8` (or `sp_size`, whichever is larger) to `trl/scripts/sft.py`.

**Proper upstream fix**: `pad_to_multiple_of=sp_size` default in the SFT collator when SP is detected. Small TRL-side change.

**Next steps**: Open a TRL PR adding the auto-default in `SFTTrainer` when `accelerator.parallelism_config.sp_size > 1`.

### G2. 🟢 TRL local changes vs `main` — for upstream after benchmarks land

> **Issue**: Branch `benchmark-sft-moe` carries TRL-side changes that must be split into upstream PRs (or merged into one big "MoE benchmark support" PR) before this branch can be deleted. They live in 4 files: `trl/scripts/sft.py`, `trl/trainer/sft_config.py`, `trl/trainer/sft_trainer.py`, `trl/trainer/utils.py`. All on top of TRL `main` (which already has `patch_chunked_lm_head` in `utils.py` and the merged chunked-loss PR #5575 — see I1).

**Inventory of changes** (committed + uncommitted on this branch):

| File             | Change                                                                 | Why                                                                                                                                                                                                                                                                           |
| ---------------- | ---------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `sft_config.py`  | new field `enable_expert_parallel: bool`                               | flag to pass `DistributedConfig(enable_expert_parallel=True)` to `from_pretrained`                                                                                                                                                                                            |
| `sft_config.py`  | new field `expert_parallel_size: int \| None`                          | EP degree (must divide num_experts and num_kv_heads); defaults to world_size                                                                                                                                                                                                  |
| `sft_config.py`  | new field `experts_implementation: str \| None`                        | sets `model.config._experts_implementation` to `'sonicmoe'` or `'grouped_mm'`                                                                                                                                                                                                 |
| `sft_trainer.py` | EP branch in `__init__`                                                | when `enable_expert_parallel`, creates the device mesh (1D world or 2D `(dp, ep)`), wires `distributed_config`, pre-initializes `deepspeed.comm` if SP is configured (so EP's `distribute_model` can use DS process groups)                                                   |
| `sft_trainer.py` | sets `model.config._experts_implementation`                            | toggled by config flag                                                                                                                                                                                                                                                        |
| `sft_trainer.py` | MFU instrumentation                                                    | `_flops_per_token`, `_world_size`, `_last_log_time`, `_last_log_tokens` populated in `__init__`; `mfu` (cumulative) + `tps_window` + `mfu_window` (per-log-window) added in `log()`. CP/SP overcount of `num_input_tokens_seen` corrected by dividing by `cp_size × sp_size`. |
| `sft.py`         | top-level per-rank `TRITON_CACHE_DIR`                                  | avoids file-not-found races when 16 ranks share the FSx-mounted default cache. Set before any torch import.                                                                                                                                                                   |
| `sft.py`         | top-level legacy TF32 flags                                            | `torch.backends.cuda.matmul.allow_tf32 = True` and `torch.backends.cudnn.allow_tf32 = True` set at import time. Workaround for inductor reading legacy API while transformers' `enable_tf32()` only sets the new API.                                                         |
| `sft.py`         | EP branch — pass model as string to SFTTrainer                         | so SFTTrainer (not the script) handles `from_pretrained` with the EP device mesh                                                                                                                                                                                              |
| `sft.py`         | `HF_HUB_OFFLINE` pre-warm dance                                        | for EP runs only: pre-warm sonicmoe and FA3 kernels via their hub paths, then flip `HF_HUB_OFFLINE=1` (and patch `huggingface_hub.constants.HF_HUB_OFFLINE`). Workaround for the multi-node hub cache race (see H3).                                                          |
| `sft.py`         | gate `save_model` on `save_strategy != "no"`                           | benchmark runs use `--save_strategy no` and don't want the post-train save                                                                                                                                                                                                    |
| `utils.py`       | new `compute_flops_per_token(config, seq_len)`                         | counts attn + MLP + (for MoE) `num_experts_per_tok × MoE MLP` per token, ×3 for fwd+bwd                                                                                                                                                                                       |
| `utils.py`       | new `compute_mfu(flops_per_token, tps, world_size, peak_flops=835e12)` | percentage of peak GPU FLOPS                                                                                                                                                                                                                                                  |

**Note**: `fuse_moe_experts(model)` was a legacy helper for transformers <5.6 (when `nn.ModuleList(experts)` broke FSDP2's `reduce_scatter` due to per-rank shape mismatch). Modern transformers ships fused 3D `gate_up_proj` / `down_proj` natively, so this helper is **no longer needed** — drop it from the branch and don't upstream.

**Suggested upstream split** (in order of independence, each one ship-able alone):

1. **MFU instrumentation** (`compute_flops_per_token`, `compute_mfu`, the `log()` additions). Self-contained, useful for everyone, no MoE coupling. Should land first.
2. **Legacy TF32 flags + per-rank Triton cache dir** in `sft.py`. Tiny PR, fixes a real PyTorch 2.10+ inductor crash + multi-node training bug. No MoE coupling.
3. **`enable_expert_parallel` + `expert_parallel_size` + `experts_implementation` config fields and the EP branch in SFTTrainer**. The biggest, most coupled chunk. Needs the upstream PRs (#45662, #45621, #45473, #45436) to land first; until then, the EP path requires our local transformers fork. Ship after the transformers PRs.
4. **`HF_HUB_OFFLINE` pre-warm dance** + `--save_strategy no` gate. Workaround for H3; ship as a TRL infra PR or a doc note depending on whether HF Hub upstream fixes the cache lock-file issue first.
5. **`pad_to_multiple_of` auto-default for SP** (G1 above). Independent of the rest.

**Status today**: nothing upstreamed. All 4 files diverge from `main` and need to be re-based and split before any PR. `fuse_moe_experts` (helper + config flag + trainer call) should be removed from the branch entirely — it's a no-op on current transformers.

---

## H. Kernel / library incompatibilities (still broken)

### H1. 🔧 Liger kernel + Qwen3-MoE under EP — workaround: `swiglu=False`

> **Issue (corrected 2026-04-29 after minimal repro)**: the previous diagnosis ("Liger fails on 3D weights") was wrong — `LigerExperts` IS 3D-aware and works fine on Qwen3-MoE without EP (verified by `benchmark/test_liger_qwen3_moe.py`, max diff vs eager = 8.79e-3). The actual failure under EP: Liger's `_patch_swiglu_module(experts, LigerExperts)` rebinds `experts.forward = LigerExperts.forward`, which **bypasses transformers' `@use_experts_implementation` dispatcher** that normally routes to `grouped_mm_experts_forward` when EP is active. `LigerExperts.forward` then calls `F.one_hot(top_k_index, num_classes=self.num_experts)` directly, where `self.num_experts` was overwritten by `shard_tensor` to `num_local_experts` AND the sentinel value (set by `RouterParallel._prepare_output_fn` to `num_local_experts`) equals `num_classes` → out-of-range → `device-side assert triggered`. Full investigation log at [`benchmark/debug_liger_ep.md`](./debug_liger_ep.md), repros at `benchmark/test_liger_qwen3_moe.py` and `test_liger_qwen3_moe_ep.py`.

**Workaround (validated 2026-04-29, swiglu=False under EP=2 returns logits within 7.8e-3 of eager)**:

```python
apply_liger_kernel_to_qwen3_moe(
    model=model,
    swiglu=False,            # ← skip the broken swiglu patch
    rope=True,
    rms_norm=True,
    fused_linear_cross_entropy=True,
)
```

Keeps `LigerRMSNorm`, `liger_rotary_pos_emb`, `LigerFusedLinearCrossEntropyLoss` (the bigger MFU wins anyway) and lets transformers' `grouped_mm_experts_forward` / `sonicmoe_experts_forward` handle the experts under EP.

**Cost**: lose `LigerSiLUMulFunction` (fused SiLU+multiply Triton kernel). Under EP the experts already use `grouped_mm` or `sonicmoe` which are themselves heavily fused, so the missing kernel is a small fraction of MoE compute.

**Fix paths to upstream**:

1. **Liger-side** (cleanest): make `apply_liger_kernel_to_qwen3_moe` detect EP (via `model.has_ep` or `model.config._experts_implementation in {"grouped_mm", "sonicmoe", "batched_mm"}`) and **skip the swiglu patch automatically**. Or: rewrite `LigerExperts.forward` to dispatch to the underlying EP forward when present (via `experts_interface.get_interface(...)`).
2. **transformers-side**: register a Liger-aware `_apply_gate` so users can opt in via `model.config._experts_implementation = "liger_gate"` and reuse the `grouped_mm` matmul path with Liger's fused activation.

**Next steps**:

- [ ] File a Liger issue with the minimal repro (`test_liger_qwen3_moe_ep.py`) and propose path (1).
- [ ] Benchmark `swiglu=False` Liger vs eager+grouped_mm on Qwen3-30B-A3B to quantify the contribution of the non-swiglu Liger components.

### H2. ⛔ FA3 + CP incompatible — accelerate restricts CP to sdpa

> **Issue**: When `parallelism_config.cp_size > 1` and `attn_implementation` is FA3, accelerate raises `Context parallelism is supported only with SDPA attention`. It's a hard guard in accelerate — CP+FA3 cannot coexist on this stack. Concrete cost: long-context MoE has to choose between FA3 throughput (SP path) and ring-attention seq sharding (CP path), can't have both. Today's FSDP+EP+CP=2 32k baseline runs at 15.6% MFU on sdpa; if FA3 were available with CP, it would likely add ~5 pp.

**Fix paths**:

1. Upstream a FA3 + CP integration in accelerate (FA3 supports causal masking and seq sharding internally; the guard is conservative).
2. Use SP instead of CP at long context (current strategy — DS-Z3+SP+FA3 at 64k gives 20% MFU).

**Next steps**: file an accelerate feature request, or open a PR after verifying FA3 actually composes with the CP scheduler at the Python level.

### H3. 🔧 Multi-node HF Hub cache race condition

> **Issue**: On 4n+ runs, ranks on the second/third/fourth node intermittently fail to resolve model shards from the HF Hub cache (concurrent writes to `~/.cache/huggingface` corrupt the lock files). Workaround: `HF_HUB_OFFLINE=1` in the launch script — but that breaks FA3's `load_and_register_attn_kernel` two-phase kernel load (needs hub access). Current gating: `HF_HUB_OFFLINE=1` only when `enable_expert_parallel` is set, after sonicmoe + FA3 are pre-warmed in the SFT script. Works but fragile (any new kernel that needs hub fetch will silently fail).

**Fix paths**:

1. Pre-fetch all needed kernels (FA3, sonicmoe) before flipping offline mode — current state, still fragile.
2. Use a per-rank cache dir (`HF_HOME=/tmp/hf-rank-${RANK}`) to avoid lock contention. Costs disk + warm-up time per rank.
3. Upstream fix: HF Hub cache should use file-locking that works on FSx-mounted homes.

**Next steps**: option 2 is the cleanest workaround; investigate if `HF_HUB_DOWNLOAD_TIMEOUT` + retries already mitigates this in newer huggingface_hub.

---

## I. Frontier — experiments to test next

### I1. 🧪 Chunked loss in SFT — [trl#5575](https://github.com/huggingface/trl/pull/5575)

> **Hypothesis**: long-context MoE is partly bottlenecked by the final `(batch × seq, vocab=151936)` logit tensor materialization in cross-entropy — at 64k seq this is ~20 GB in bf16 per rank. Chunked loss processes the logit/CE computation along the seq dim in chunks, materializing only one chunk at a time. Same idea as Liger's fused CE, but works on MoE (Liger doesn't, see H1) and integrates at the SFT trainer level rather than as a kernel patch.

**Why interesting on this stack**:

- Bypasses the H1 (Liger+MoE) blocker — chunked loss doesn't touch the SwiGLU/expert path.
- Could reduce activation memory at 32k+ enough to fit configs that currently OOM (e.g., FSDP+EP=8 at 32k without CP).
- Stacks with SP/CP — chunking is over the materialized logit tensor, orthogonal to seq sharding.

**Test plan** (cheap, single config):

1. Pull the PR locally onto a `chunked-loss-test` branch off `benchmark-sft-moe`.
2. Re-run the current 32k FSDP+EP=8+CP=2 baseline (15.6% MFU, 73.6 GB peak) with chunked loss enabled. Compare MFU, peak mem, loss curve.
3. If it shaves enough memory, try the configs that OOM today: 32k DS-Z2+EP=8 (currently 18.55 GiB OOM) and 64k FSDP+EP=8+CP=2 (per-rank seq=32k, hits the same buffer).
4. If it works, add a column to `sft_benchmark_notion.md` for chunked-loss-on/off and re-sweep the long-context configs.

**Caveats**:

- Need to confirm the PR is compatible with packing (`--packing --packing_strategy wrapped`) — chunking over packed sequences may need awareness of sample boundaries.
- If chunked loss interacts with `num_input_tokens_seen` or grad accumulation, the MFU formula correction (cp_size × sp_size) may need a similar tweak.

**Next steps**: clone the PR, smoke-test on Qwen3-30B-A3B at 16k first to verify correctness (loss curve matches non-chunked), then push to 32k+ to see if it unblocks any OOM configs.

### I2. 🧪 **Single-node 30B**: longest context on 1× 8-GPU node

> **Why this matters**: if Qwen3-30B-A3B can train on a single H100 SXM5 node (8× 80 GB = 640 GB total), it dramatically lowers the barrier to MoE SFT — no multi-node Slurm, no cross-node EFA, no Z3 cross-node hangs. Today everything runs at 2 nodes minimum because of the long-context champions; we have not seriously stress-tested 1n.

**Goal**: find the maximum context length where one of these recipes trains correctly on 1n × 8 GPUs:

- DS-Z3 + SP=8 (intra-node only — entire SP group fits on one node, no cross-node Ulysses)
- DS-Z3 + SP=4 (per-rank seq = ctx/4)
- DS-Z2 + EP=8 (1n EP — one EP group = one node)
- FSDP-Z3 + EP=8 (no SP) at small ctx
- All combined with chunked-CE OR Liger (whichever wins from the current sweep)

**Memory budget per H100 (80 GB) on 1n × 8 GPUs**:

- DS-Z3 sharded: params 30B/8 = 3.75 GB · optim (Adam fp32 m+v) 30B × 8B / 8 = 30 GB · grads 30B × 2B / 8 = 7.5 GB → **~41 GB/rank fixed cost**, leaves 39 GB for activations + temp buffers
- DS-Z2 + EP=8 single node: per-rank ~7-15 GB params (experts sharded), full optim states (no DP to shard across) = ~30 GB optim → **~40-45 GB/rank fixed**, leaves ~35 GB
- FSDP-Z3: same as DS-Z3 essentially, ~41 GB fixed

Activation memory at per-rank seq for Qwen3-30B-A3B (with grad checkpointing): ~25-30 GB for 32k per-rank, much less at smaller per-rank seq. So the per-rank-seq=32k sweet spot from earlier translates to roughly **256k total context on 1n SP=8** as a starting point, possibly more with chunked.

**Test plan**:

1. Establish baseline: 16k DS-Z2+EP=8 + chunked @ 1n. Should fit easily.
2. Push: 32k DS-Z2+EP=8 + chunked @ 1n. If fits, try 64k.
3. SP path: 64k DS-Z3+SP=8 + chunked + compile @ 1n (per-rank seq=8k — should fly).
4. Long: 128k DS-Z3+SP=8 + chunked + compile @ 1n (per-rank=16k).
5. Stretch: 256k DS-Z3+SP=8 + chunked + compile @ 1n (per-rank=32k — may OOM, may not).
6. Whichever recipe wins the EP-vs-Liger sweep, re-test the OOM ceiling with that variant.

**Metrics**: report MFU window, peak GPU mem %, loss curve, TPS at each ctx. Compare to the 2n+ champions to quantify the "single-node penalty" (if any — e.g. losing the EP-replication win might not matter at small DP).

**Why this could be huge**: a recipe like "256k context Qwen3-30B-A3B SFT on a single H100 node" is dramatically more accessible than the current multi-node pattern. Concrete deliverable would be a `benchmark/configs/qwen3_30b_a3b_1n_max.yaml` with the verified-working recipe at the longest context that fits.

These are TRL-repo additions (managed by TRL git, not "uncommitted patches"):

- `benchmark/fetch_peak_gpu_mem.py` — keep. Required for trackio peak-mem queries scoped to a run's start time.
- `benchmark/CLAUDE.md` — keep. Documents the Peak GPU Mem mandatory rule, `report.md` first / `sft_benchmark_notion.md` second write order.

---

## TODO checklist (living document — update after each attempt / PR)

> Update rule: every time we try a fix, file an issue, or open a PR, check the box (or add a sub-bullet with the link/date/status). If a fix lands upstream and merges back into the stack, delete the item entirely.

### 🟢 Ready to PR

- [ ] **C1 — accelerate `_prepare_tp` `has_ep` skip** → open accelerate PR (single check, references `#45662` for context). Revert: `uv pip install --reinstall --no-deps accelerate`.
- [ ] **C2 — EP + `cpu_ram_efficient_loading=True` (3 PRs, in order)** → (a) transformers `ValueError` guard in `from_pretrained` when both flags set, plus EP doc note (immediate safety net); (b) transformers Patch 1: rank-0 gate in `convert_and_load_state_dict_in_model` so non-rank-0 leaves non-plan params on meta (today it's a no-op, every rank still loads — verified empirically 2026-04-29); (c) accelerate Patch 2: `fsdp2_prepare_model` (~10 lines capture/restore `ignored_params` around meta move) + `fsdp2_load_full_state_dict` filter (~3 lines, rank>0 branch). End-to-end test: peak/rank should drop from 16.4 GiB to ~13.5 GiB on 30B-A3B EP=4.
- [ ] **G1 — SP `--pad_to_multiple_of` auto-default** → TRL PR adding the auto-default in `SFTTrainer` when `accelerator.parallelism_config.sp_size > 1`.
- [x] **H1 (debug + sweep)** — Liger + Qwen3-MoE under EP: root-cause identified, workaround `--liger_kernel_config '{"swiglu":false}'` validated, full sweep run vs chunked-CE champions. Liger wins at every context 16k–1M (+0.3 to +25 pp peak MFU). New all-time MoE record: 128k @ 76.29 % (40.5 % causal-adjusted). Full investigation in `benchmark/debug_liger_ep.md`, results in `report.md` + `sft_benchmark_notion.md`. **Still TODO**: file a Liger PR/issue with the repro at `benchmark/test_liger_qwen3_moe_ep.py` proposing the dispatcher-aware fix (auto-detect EP via `model.config._experts_implementation`).
- [x] **TRL `_preparse_dict_args` helper** — added to `trl/scripts/sft.py` to JSON-decode `--liger_kernel_config` before HfArgumentParser sees it (HfArgumentParser falls back to `type=dict` and rejects JSON strings). 12 lines, no other changes. Should also become an upstream TRL PR if HfArgumentParser doesn't fix this.
- [ ] **H2 — FA3 + CP gated to sdpa in accelerate** → file an accelerate feature request, or open a PR after verifying FA3 actually composes with the CP scheduler at the Python level.

### 🟢 PR-after-deps-land

- [ ] **A1 — sonicmoe EP/DTensor support (`.to_local()` + clamp wrapper)** → blocked on `#45621` (sonicmoe Ilyas patch) + `#45662` (EP+FSDP DTensor wrap). Write up follow-up PR explaining the autograd-through-`to_local()` backward gap.
- [ ] **Deepseep EP-works — DS-Z2 + EP, 7 patches across 3 repos**, the recipe behind every long-context champion in the report. Split into:
    - [ ] transformers PR — patches 1/2/4/5 (`tensor_parallel.py`, `trainer.py:create_accelerator_and_postprocess`, `trainer.py:create_optimizer`, `trainer.py:_clip_grad_norm`)
    - [ ] DeepSpeed PR — patch 3 (extend `engine.py:_configure_distributed_model` to detect external EP via `param.allreduce is False`)
    - [ ] TRL PR — patch 6 (pre-init `deepspeed.comm` in SFTTrainer EP branch — falls under G2 below)

### 🟡 Keep local (proper fix is upstream elsewhere)

- [ ] **B1 — `_clip_grad_norm` skip when `has_ep`** → file a PyTorch issue describing the cross-mesh `_foreach_norm` stack failure with a minimal repro. Long-term fix is in PyTorch's `clip_grad_norm_` or accelerate's wrapper.

### ⛔ Investigation — smallest test surface first

- [ ] **E2 — Triton cache file-not-found at FSDP+EP+CP+compile @ 64k** → try `TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-rank-${RANK}-${HOSTNAME}` next to existing `TRITON_CACHE_DIR`. Cheapest test.
- [ ] **E1 — FSDP+EP+compile Adam `_group_tensors_by_device_and_dtype`** → try the surgical `_group_tensors_by_device_and_dtype` patch in PyTorch (treat EP and FSDP DTensors as compatible if device+dtype match).
- [ ] **H3 — multi-node HF Hub cache race** → try per-rank `HF_HOME=/tmp/hf-rank-${RANK}` to avoid lock contention. Investigate whether newer `huggingface_hub` already has `HF_HUB_DOWNLOAD_TIMEOUT` + retries.
- [ ] **FSDP + per-chunk lm_head NCCL hang** (chunked-CE AND Liger FLCE both hit it). Both `--loss_type chunked_nll` and `--use_liger_kernel true` (FLCE) compute the lm_head matmul in chunks of 256 valid tokens; under FSDP the lm_head weight is sharded across DP, so each chunk triggers an all-gather → ~519 collectives enqueued, only 8 complete → NCCL timeout. **Current workaround: use DS-Z2 instead of FSDP** for the EP+chunked or EP+Liger path (DS-Z2 doesn't shard params across DP, so no per-chunk gather). Proper fix paths: (a) exclude `lm_head` from FSDP wrap, (b) wrap the chunk loop in `model.lm_head.unshard_context()` so the weight stays gathered for all chunks in a step, (c) move the chunked patch before FSDP wrap. See `report.md` for the chunked-CE repro and the 16k FSDP+EP+Liger crash log (job 22094539).

### ⛔ Larger projects

- [ ] **D-blocker — DS-Z3 + EP rank-ordering fix** on top of the existing 3-patch tagging from commit `cd52547f87`. Either align transformers' EP partition to use `global_rank % ep_size` or build DS's `expert_parallel_group` manually with matching rank order. 1–2 days. Detailed plan in `debug_sp_ep_sonic.md` Iteration 4.
- [ ] **F1 — streaming expert dispatch (kernel rewrite)** → RFC in transformers MoE integrations to avoid materializing the full `(seq, num_local_experts, moe_intermediate)` activation tensor. Biggest payoff (unblocks 32k+ EP without CP), biggest scope.

### 🧪 Frontier — completed and validated

- [x] **I1 — Chunked loss in SFT** ([trl#5575](https://github.com/huggingface/trl/pull/5575), already merged into TRL `main`). Cherry-picked onto branch on 2026-04-28; unlocked DS-Z2+EP at 32k–128k (45–69 % MFU window) and SP+chunked at 256k–1M (32–60 % MFU window). See "chunked-CE results" in report.md and sft_benchmark_notion.md.
- [x] **Liger × MoE × EP unblocked** (2026-04-29). Disproved "3D weights" hypothesis with single-GPU repro. Found root cause: `_patch_swiglu_module` bypasses transformers' EP-aware dispatcher; Liger's `LigerExperts.forward` hits `F.one_hot` with `num_classes == sentinel`. Fix: `--liger_kernel_config '{"swiglu":false}'`. Liger wins at every context 16k–1M (+0.3 to +25 pp peak vs chunked). Repros: `benchmark/test_liger_qwen3_moe{,_ep}.py`. Investigation log: `benchmark/debug_liger_ep.md`.
- [x] **MFU causal-correction post-hoc adjustment** (2026-04-29). Helper `benchmark/adjust_mfu_causal.py` (CLI + CSV mode) computes `adjusted_mfu = reported_mfu × adj_factor` where `adj_factor` subtracts half the attention-score FLOPs (causal). Applied to all 3 headline tables in `sft_benchmark_notion.md`. Raw MFU column kept untouched per user request. Verified all 36 cells against an independent recompute from `compute_flops_per_token`.

### 🧪 Frontier — pending

- [ ] **MFU formula causal correction (in-source fix)** — currently `compute_flops_per_token` uses the non-causal convention; we apply the correction post-hoc via `benchmark/adjust_mfu_causal.py`. A proper TRL PR would update the formula at source (or expose a `causal=True` knob) so the raw MFU column matches the Llama 2/3 / DS-Ulysses convention. Tradeoff: would change all historical numbers — keep the helper as the safe path for now and flag this as a separate PR if anyone wants the convention switched.
- [ ] **Push 1n past 32k without offload** (deferred — open variants from I2): (a) bf16 optim states (Adafactor or 8-bit Adam), (b) different EP topology that also shards optim across the EP-data-parallel group, (c) activation-checkpointing variants. Each is independent; pursue if 1n is a frequent demand.

### 🗑️ Cleanup — to delete before PRs

- [x] **`SONICMOE_DISABLE_CLAMP` debug knob** — added 2026-04-29 for one validation run, removed same day after the result confirmed the wrapper-clamp is load-bearing (job 22094732: loss collapsed to 0 + entropy NaN + grad_norm frozen by step 10 without the clamp). All 4 plumbing sites restored to pre-debug state. Investigation log: `benchmark/debug_sonic_bwd_dtensor.md`.

- [x] **B2 — `TRANSFORMERS_SKIP_EP_DTENSOR_WRAP=1` env-var bypass** → removed 2026-04-28 (5-line bypass in `tensor_parallel.py` + `skip_ep_dtensor_wrap` plumbing in `run_benchmark.py` and `launch.sh.j2`).

## Revert everything (clean slate)

DONT DO THIS UNLESS TOLD SO BY THE USER:

```bash
# Transformers (working-tree only, no commits to undo)
cd /fsx/amine_dirhoussi/transformers
git checkout -- src/transformers/{modeling_utils.py,trainer.py,core_model_loading.py,integrations/tensor_parallel.py,integrations/moe.py,integrations/sonicmoe.py,integrations/hub_kernels.py}

# Accelerate (in-place .venv patch — single location)
cd /fsx/amine_dirhoussi/trl
uv pip install --reinstall --no-deps accelerate

# DeepSpeed already reverted in tree; reinstall as a safety net
uv pip install --reinstall --no-deps deepspeed
```
