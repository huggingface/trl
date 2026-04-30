# Debug: DS-Z3 + SP + EP + sonicmoe

Working notebook for unblocking long-context MoE via DeepSpeed Ulysses SP combined with Expert Parallelism. CP path is heavily MFU-penalized (≤8% at 64k); SP gets ~19% but EP+SP was historically blocked.

## TL;DR (jump to the verdict)

- **DS-Z2 + EP works** — new path validated this session. Loss healthy (12.43), 28.6% window MFU at 16k 2n. All patches committed on branch [`ds-ep-integration`](#branch-ds-ep-integration-summary) of `/fsx/amine_dirhoussi/transformers`.
- **DS-Z3 + EP doesn't** — `stage3.py` has zero `is_moe_param` plumbing; only `stage_1_and_2.py` does.
- **SP + EP doesn't compose** with transformers' replicate-mask EP (any ZeRO stage). Different architectures: Ulysses shards seq, transformers EP assumes full-batch on every rank, all-reduce of the two = garbage. DS native MoE composes with SP only because it does explicit all-to-all token dispatch (which transformers EP doesn't).
- **Long-context recipe stays**: DS-Z3 + SP + sonicmoe (no EP) at 19–20% MFU.

## Branch `ds-ep-integration` summary

Lives in `/fsx/amine_dirhoussi/transformers`, branched from `qwen3-moe-ep-v2`. Six commits:

| SHA | What it does |
| --- | --- |
| `69502dd55e` | **Baseline**: local benchmarking changes (PR #45662 + #45621 cherry-picks + sonicmoe EP wrapper-clamp). The starting point — same code that gives 40.4% on FSDP+EP. |
| `cd52547f87` | **3-patch tagging approach** (initial DS+EP try): tag EP params with `allreduce=False` + `group_name` in `GroupedGemmParallel.post_shard_wrap`; create `expert_parallel_group`/`expert_data_parallel_group` in Trainer; gate `_clip_grad_norm` skip on FSDP-only. Smoke test ran 50 steps at 16k Z3 but loss=32-37 (broadcast clobbered EP slices on DP-of-EP siblings). |
| `567b07ddcb` | **Z3-only attempt**: also tag with `ds_id` to make `is_zero_param` skip the EP params in `_convert_to_zero_parameters`. Hit `'Parameter' has no convert_to_zero_parameters` next. Reverted. |
| `cb84f38a09` | Reverted ds_id hack, kept tagging. |
| `f16bffc8c5` | **Broaden DS detection** (was Z3-only): `post_shard_wrap` now branches on any active DS config (Z1/Z2/Z3) via `_hf_deepspeed_config_weak_ref`, not just `is_deepspeed_zero3_enabled()`. |
| `0c146cd8f7` | **Trainer optimizer split for MoE**: in `Trainer.create_optimizer`, when `has_ep` + DS, run `deepspeed.moe.utils.split_params_into_different_moe_groups_for_optimizer` on the param groups. Otherwise ZeRO-2's `_configure_moe_settings` asserts. |

### What's in the branch (ready-to-use for DS-Z2 + EP)

Net diff from `qwen3-moe-ep-v2`:

- `src/transformers/integrations/tensor_parallel.py`: `GroupedGemmParallel.post_shard_wrap` branches on backend — under DS (any ZeRO stage), tags `param.allreduce = False` + `param.group_name = f"ep_size_{N}"`; under FSDP, wraps as DTensor.
- `src/transformers/trainer.py`: in `create_accelerator_and_postprocess`, calls `_create_expert_and_data_parallel(model.tp_size)` when `has_ep` + DS (so the named groups exist before engine init); in `create_optimizer`, splits param groups via `split_params_into_different_moe_groups_for_optimizer` when `has_ep` + DS; the `_clip_grad_norm` `has_ep` skip is gated to FSDP-only.
- Plus the baseline transformers patches (from PR #45662 / #45621 cherry-picks) for FSDP+EP.

### What's NOT in the branch (extra patches outside transformers)

These need to be re-applied to `.venv/.../deepspeed/` for DS-Z2+EP runs to start. Not committed because they patch the installed DeepSpeed package, not transformers.

```python
# .venv/.../deepspeed/runtime/engine.py:1430 — extend has_moe_layers detection
for _, module in self.module.named_modules():
    if isinstance(module, MoE):
        self.has_moe_layers = True
        self.num_experts.append(module.num_experts)
    elif any(getattr(p, "allreduce", True) is False for p in module.parameters(recurse=False)):
        # External MoE (transformers EP): tagged params trigger MoE-aware bookkeeping
        self.has_moe_layers = True
        self.num_experts.append(getattr(module, "num_experts", 0))
```

Without this DS patch, `has_moe_layers` stays False, `expert_data_parallel_group=None` is passed to the optimizer, and ZeRO-2's `_configure_moe_settings` errors with `TypeError: 'NoneType' object is not subscriptable`. To revert: `pip install --force-reinstall --no-deps deepspeed`.

### To use this branch

1. `cd /fsx/amine_dirhoussi/transformers && git checkout ds-ep-integration`
2. Apply the DeepSpeed engine patch above to `.venv/.../deepspeed/runtime/engine.py`
3. Submit a DS-Z2 + EP run via `benchmark/configs/qwen3_30b_a3b_dsz2_ep_smoke.yaml` (sets `zero_stage: 2`).

## Goal

Run `Qwen3-30B-A3B` SFT at 64k+ context with **DS-Z3 + SP + EP=8 + FA3 + sonicmoe**. Per-rank seq stays ≤16k via Ulysses sequence sharding while EP shards experts intra-node. Same 40% window MFU we got at 16k EP=8, scaled to long context.

**Status (resolved this session): the original goal is unreachable — SP and transformers-EP can't compose at all (verified in-session — see "Long-context fundamental: SP + transformers-EP doesn't compose" below). The salvage is DS-Z2+EP at 16k, working at ~28.6 % MFU.**

## Stack

All edits below are uncommitted local changes. They are needed simultaneously — removing any one will break the run differently.

### 1. Transformers fork — `/fsx/amine_dirhoussi/transformers` (branch `qwen3-moe-ep-v2`)

- **`modeling_utils.py`**:
  - `has_ep`, `ep_sharded_param_names` properties.
  - `_wrap_ep_params_as_dtensor` staticmethod called once at end of `from_pretrained`. Wraps each EP-sharded param as `DTensor.from_local(p.data, ep_mesh, [Shard(0)], run_check=False)`.
  - Env-var bypass: `TRANSFORMERS_SKIP_EP_DTENSOR_WRAP=1` (debug).
- **`integrations/moe.py`**: `to_local()` helper applied at three weight-access sites in `grouped_mm_experts_forward`. Wrapper-level clamp+masked_fill for sentinel handling (already there).
- **`integrations/sonicmoe.py`**:
  - `from torch.distributed.tensor import DTensor` import.
  - `gate_up_proj.to_local()` / `down_proj.to_local()` before `permute(*perm)`.
  - **Wrapper clamp+masked_fill on `expert_ids`** (mirrors grouped_mm). The kernel-native sentinel-skip caused NaN in backward when weights are DTensor-derived; the wrapper clamp does compute on sentinel rows but `router_scores` is zeroed for them, so they contribute 0.
- **`trainer.py`**:
  - `not has_ep` gate on `ParallelismConfig` auto-build.
  - `ignored_modules` auto-wire from `ep_sharded_param_names`.
  - `_clip_grad_norm` returns `tensor(0.0)` when `model.has_ep` (clip-norm op stacks per-param norms across mismatched meshes — FSDP DP mesh vs EP mesh — and crashes; we skip the clip for EP runs).

### 2. Accelerate — `/fsx/amine_dirhoussi/trl/.venv/lib/python3.11/site-packages/accelerate/accelerator.py`

In `Accelerator._prepare_tp` (right after the existing "no DTensor → skip" guard, around line 1601):

```python
# EP-only models: experts are DTensors on the EP mesh (not the TP mesh accelerate
# is about to set up). Skip TP preparation — non-EP params stay plain tensors and
# FSDP2 handles them on the FSDP mesh, the same path that worked before EP params
# became DTensors.
if getattr(model, "has_ep", False):
    return result
```

Without this skip, accelerate tries `from transformers.integrations.tensor_parallel import ReplicateParallel` which doesn't exist in our fork. The skip restores the natural flow that fired before our EP+FSDP fix made EP params DTensors.

### 3. DeepSpeed — `/fsx/amine_dirhoussi/trl/.venv/lib/python3.11/site-packages/deepspeed/runtime/engine.py`

In `DeepSpeedEngine._broadcast_model`, inside `is_replicated(p)`:

```python
def is_replicated(p):
    if hasattr(p, "ds_status") and p.ds_status is not ZeroParamStatus.AVAILABLE:
        return False
    elif hasattr(p, 'ds_optim_param'):
        return False
    # EP DTensors are sharded per rank (each rank holds a different slice).
    # `dist.broadcast` doesn't accept DTensor inputs and they shouldn't be broadcast anyway.
    from torch.distributed.tensor import DTensor
    if isinstance(p.data, DTensor):
        return False
    return True
```

Without this, `_broadcast_model` calls `dist.broadcast(p.data, ..., group=self.seq_data_parallel_group)` on every param at engine init. EP DTensors can't be broadcast — `RuntimeError: found no DeviceMesh from dtensor args for c10d::broadcast_`. Skipping DTensor params is correct: they're sharded, not replicated, so there's nothing to broadcast.

## Reverting all of this

```bash
# Transformers — uncommitted, just stash or checkout
cd /fsx/amine_dirhoussi/transformers && git checkout -- src/transformers/{modeling_utils.py,trainer.py,integrations/moe.py,integrations/sonicmoe.py}

# Accelerate + DeepSpeed — re-install
cd /fsx/amine_dirhoussi/trl
.venv/bin/pip install --force-reinstall accelerate deepspeed
```

## Failure modes encountered (chronological)

1. **EP + FSDP corruption** — `fsdp2_load_full_state_dict` broadcasts rank-0's expert slice to all ranks → ranks 1-7 lose their experts → loss=62, NaN. Fixed by transformers PR #45662 logic (DTensor wrap + ignored_modules).
2. **`ReplicateParallel` ImportError** — accelerate's `_prepare_tp` reaches an import line that didn't exist before EP params were DTensors. Fixed by accelerate-side `has_ep` skip above.
3. **Clip-grad-norm mesh mismatch** — `clip_grad_norm_` stacks per-param norms; FSDP DTensors and EP DTensors live on different meshes → "All operands in aten.stack.default must have the same mesh". Fixed by trainer-side skip when `has_ep`.
4. **Adam mixed-types** — `_fused_adamw_` rejects mixed Tensor/DTensor lists. Avoided by keeping the EP DTensor wrap.
5. **Sonicmoe NaN at EP < DP_size** — kernel-native sentinel-skip + DTensor weights → first-backward produces NaN. Fixed by wrapper-level clamp+masked_fill (kernel does compute on sentinels, router_scores is zero for them).
6. **DS-Z3 `_broadcast_model` on DTensor** — `dist.broadcast` on EP DTensor → `c10d::broadcast_` no-DeviceMesh. Fixed by DS-side skip above.

## Pending failure modes (not yet hit on this stack)

- **DS batch-size assertion at SP+EP** (job 22092281 historical): `train_batch_size != micro_batch_per_gpu * grad_acc * world_size`. DS counts only the EP-carved sub-mesh as DP world. Fix would set `train_batch_size = "auto"` or compute the correct DP size for the SP+EP combo. **May resurface** once the broadcast issue is unblocked.
- **Long-context EP=8 alone OOMs at 37 GiB** regardless of node count (jobs 22092511-14). Per-rank seq must stay ≤16k → CP or SP needed for >16k. SP path is what this debug is targeting.

## Iteration 2 — jobs 22092516, 22092517 still hit `c10d::broadcast_`

The first DS patch (`_broadcast_model.is_replicated`) was wrong location. The actual broadcast hits at engine init via:

```
parameter_offload.py:217 → _convert_to_zero_parameters
partition_parameters.py:1102 → _zero_init_param  (called per-param)
partition_parameters.py:1123 → dist.broadcast(param.data, 0, group)
```

DS-Z3's Stage3 path tries to broadcast every param from rank 0 then partition it across ranks for ZeRO-3 sharding. EP DTensors must be **skipped entirely** — they're already sharded per rank by EP and DS-Z3's partition would corrupt them (broadcast rank 0's slice to all, then re-shard).

### Patch 2 — `partition_parameters.py:_convert_to_zero_parameters`

```python
def _convert_to_zero_parameters(self, param_list):
    for param in param_list:
        if is_zero_param(param):
            continue

        # EP DTensors are sharded per rank by transformers' EP machinery (each rank holds
        # a different expert slice). DS-Z3 would broadcast rank 0's slice to all ranks
        # (destroying data) and try to re-partition it. Skip them — the EP path manages
        # these params end-to-end.
        from torch.distributed.tensor import DTensor
        if isinstance(param.data, DTensor):
            continue

        param.data = param.data.to(self.local_device)
        self._zero_init_param(param)
```

Resubmitted as jobs 22092518 (16k smoke) and 22092519 (64k SP=4 EP=8). If this passes the broadcast, next likely failure is the **DS batch-size assertion** documented above.

## Iteration 3 — DTensor params have no `ds_numel`

Past the broadcast (patch 2 worked), now hit:

```
AttributeError: 'DTensor' object has no attribute 'ds_numel'
File "deepspeed/runtime/zero/parameter_offload.py:269", in mark_persistent_parameters
    if param.ds_numel + total_persistent_parameters > model_threshold:
```

DS iterates *all* params expecting them to have `ds_numel` (set by `_convert_to_deepspeed_param` which we skipped for DTensors). Need to also skip DTensors in the iteration.

### Patch 3 — `parameter_offload.py:mark_persistent_parameters`

```python
def mark_persistent_parameters(self, param_threshold, model_threshold):
    from torch.distributed.tensor import DTensor

    persistent_params = []
    total_persistent_parameters = 0
    params_count = 0
    for name, param in self.module.named_parameters(recurse=True):
        # EP DTensors are managed outside DS-Z3 (skipped in _convert_to_zero_parameters).
        # They have no `ds_numel` and are not subject to ZeRO-3 persistence accounting.
        if isinstance(param.data, DTensor):
            continue
        if param.ds_numel + total_persistent_parameters > model_threshold:
            continue
        ...
```

Resubmitted as 22092520/22092521. Expecting more "DTensor has no ds_*" errors at every DS iteration over params; will patch each as it surfaces.

## Iteration 4 — `partition_numel` next; took the wrong abstraction

Got past `ds_numel` (patch 3) → next: `AttributeError: 'DTensor' object has no attribute 'partition_numel'` in `stage3.py:_configure_zero_optimizer`. Skipping each ds_* access is not tractable (30+ sites).

### Wrong conclusion drawn

I initially concluded "DS-Z3 has no opt-out for externally-managed parameters" and pivoted away. **That was wrong.** DeepSpeed has a complete MoE/EP infrastructure I missed — see `deepspeed/moe/`.

### How DeepSpeed actually models MoE/EP

DS-Z3 has a first-class concept of "expert parameters" that bypass the standard ZeRO-3 path. The convention is two attributes per expert param:

```python
param.allreduce = False             # marks it as MoE expert, not subject to standard allreduce
param.group_name = "ep_size_8"      # names the expert-parallel group
```

`deepspeed/moe/experts.py:Experts.__init__` sets these on every expert param. `deepspeed/moe/utils.py:is_moe_param(p)` checks for them. The runtime then branches at every relevant site:

| Site | Behavior for MoE params |
|---|---|
| `engine.py:_broadcast_model` | broadcast inside `expert_data_parallel_group[group_name]` (the DP-of-EP group), not the full DP group |
| ZeRO-3 partition | params are partitioned across `expert_data_parallel_group` (DP replicas of this EP rank's slice), not flattened across all GPUs |
| Optimizer | `configure_moe_param_groups` creates a separate param group keyed by `group_name`, so optimizer step uses the right comm group |
| `expert_parallel_group` | DS sets up its own all-to-all comm group (not used directly by ZeRO; this is for the MoE layer's token routing) |

`expert_parallel_group` and `expert_data_parallel_group` are computed by `deepspeed/utils/groups.py` based on the world size and EP size. With 16 GPUs and EP=8: `expert_parallel_group = [[0..7], [8..15]]`, `expert_data_parallel_group = [[0,8], [1,9], ..., [7,15]]`.

### What this means for our DS-Z3 + EP stack

To make DS-Z3 + EP work, we don't need to skip params (the patch-each-site approach I tried). We need to **tag them correctly** so DS routes them through the MoE path it already has. The minimal change is, after transformers' EP partition_tensor runs but before `deepspeed.initialize()`:

```python
for name, param in model.named_parameters():
    if _get_parameter_tp_plan(name, model.tp_plan, is_weight=True) == "grouped_gemm":
        param.allreduce = False
        param.group_name = f"ep_size_{model.tp_size}"
```

…and then make sure the matching `expert_parallel_group` / `expert_data_parallel_group` are created in `deepspeed/utils/groups`. Either:

- (a) Tell DS the EP size before `deepspeed.initialize()` (likely via a config/runtime hint) so it builds the groups itself, or
- (b) Construct the groups manually with the same names DS expects, before init.

### Where the DTensor wrap fits in

PR #45662 wraps EP params as `DTensor` because Adam's `_foreach_mul_` rejects mixed Tensor/DTensor lists when FSDP wraps the rest as DTensors. **Under DS-Z3, FSDP isn't active** — DS owns parameter management end-to-end. DS's optimizer already handles MoE params in their own param group (separate `_foreach_mul_` call), so the mixed-types issue doesn't arise there.

This means: under DS-Z3, **the DTensor wrap may not be needed at all** for EP params. The right fix is probably:

- Gate `_wrap_ep_params_as_dtensor` on the active backend: only wrap when FSDP is active. Under DS-Z3, leave as plain `nn.Parameter` and add the `allreduce=False` / `group_name` markers instead.
- This also avoids every `'DTensor' object has no attribute ds_*'` error we hit, because DS's MoE path expects plain tensors.

### Status

Not implemented; documented for future work. Path forward is non-trivial but not "fundamentally infeasible" as I initially wrote — DS already has the abstraction we need; we just have to tag the right params and provide matching process groups.

### 2026-04-28 update: implementation attempt — ran into deeper DS plumbing

Branch `ds-ep-integration` (in `/fsx/amine_dirhoussi/transformers`) implemented the 3 transformers-side patches:
- A: tag EP params with `allreduce=False` + `group_name` in `GroupedGemmParallel.post_shard_wrap` under DS-Z3
- B: create `expert_parallel_group`/`expert_data_parallel_group` via `_create_expert_and_data_parallel(ep_size)` in `Trainer.create_accelerator_and_postprocess`
- C: gate `_clip_grad_norm` skip on FSDP-only

Then iterated on a 16k 2n DS-Z3 + EP=8 + sdpa + sonicmoe smoke (jobs 22092855 → 22092859 → 22092874 → 22092877 → 22092878):

| Iter | Trigger | Symptom | What I tried |
| --- | --- | --- | --- |
| 1 | tagging only | runs 50 steps but loss=32–37 (not 12) | `_zero_init_param` broadcasts on full DP group → clobbers EP slices on ranks 8–15 |
| 2 | + `param.ds_id = id(param)` (so `is_zero_param` skips) | `'Parameter' has no attribute 'convert_to_zero_parameters'` in `parameter_offload._convert_to_zero_parameters` | reverted ds_id; tried DS-side patch |
| 3 | DS patch: `_zero_init_param` skips broadcast for `is_moe_param` | same loss=32–37 (broadcast was only half the issue; `param.partition()` also shards across full DP group) | extended patch to skip both broadcast AND partition for MoE |
| 4 | DS patch: skip both broadcast and partition for MoE | `'NoneType' has no attribute 'ds_numel'` (downstream code reads `param.ds_tensor.ds_numel`, which is None when partition is skipped) | tried partitioning across `expert_data_parallel_group` |
| 5 | DS patch: set `param.ds_process_group = ep_dp_group`, then broadcast + partition in that group | `ValueError: output tensor size must be equal to world_size times input tensor size` (forward all-gather still uses the full DP group; partition into smaller group → all-gather expects N× full-group, gets N× small-group). |

### Why this won't work without upstream DS changes

DS-Z3 plumbs the data-parallel group into the `Init` instance at module-creation time. Every ZeRO-3 collective then references the same group via `param.ds_process_group` and the engine's `dp_process_group`. DS's native MoE flow (the only working path for DS+EP) constructs `Init` once with the FULL DP group for non-MoE params and a separate `Init` (via `deepspeed.moe.layer.MoE`'s context manager) with the smaller DP-of-EP group for MoE params. Each set of params gets its own correctly-sized collectives end-to-end.

We're trying to retrofit the smaller-group routing onto a single `Init` that was constructed with the full DP group. Setting `param.ds_process_group` post-hoc on individual params doesn't propagate to: forward all-gather hooks, backward reduce-scatter, persistent-param accounting, optimizer step, etc. Patching each one of those (we counted 30+ access sites in `stage3.py`) is the same dead end as before, just with different attribute names.

The only correct fix is **upstream DS-side**: either (a) extend `Init` to accept a `data_parallel_group_per_param` callback that returns the right group based on `is_moe_param(p)`, or (b) construct two separate `Init` instances in `deepspeed.initialize()` — one for MoE params (DP-of-EP group) and one for non-MoE params (full DP group). DS already has half the machinery (`is_moe_param`, the named groups, the optimizer-side `configure_moe_param_groups`); plumbing the smaller group through ZeRO-3's runtime collectives is the missing piece.

### Reverted, for now

- DS patch in `_zero_init_param` reverted (`pip install --force-reinstall deepspeed` if needed).
- Branch `ds-ep-integration` retains the 3 transformers-side patches as a starting point for a future attempt that includes the upstream DS work.
- Net status: DS-Z3 + EP remains unsupported. SP runs continue to use DS-Z3 without EP. EP runs continue to use FSDP2.

### 2026-04-28: Root cause of why this is hard with ZeRO-3

After reading more code: **`stage3.py` has zero `is_moe_param` checks** — ZeRO-3 has no MoE awareness in this DS version. Only `stage_1_and_2.py` does (lines 730–739, 1119, 1242, 2225 — `is_moe_param_group`, `expert_dp_process_group[param.group_name]` for grad reduce, etc.).

The deeper issue is a fundamental ZeRO-3 assumption: **the same param name has the same logical data on every rank**. transformers EP violates this — `model.layers.0.mlp.experts.gate_up_proj` on rank 0 holds experts 0–15, on rank 1 holds 16–31. They're *different logical tensors that happen to share a name*. ZeRO-3's all-gather across the full DP group would concatenate rank 0's expert-0-15 slice with rank 1's expert-16-31 slice etc., producing garbage when "reconstituting the full param".

DS native MoE handles this by using `deepspeed.zero.Init(data_parallel_group=ep_dp_group)` to construct the MoE submodule — giving each EP rank's params their own ZeRO-3 universe with the smaller DP-of-EP group as their world. Each EP rank's local-experts tensor is partitioned across just the 2 DP-of-EP ranks (rank 0 & rank 8 for our config). All-gather within that group correctly reconstructs the per-rank slice. Cross-EP-rank gathering never happens — different EP ranks hold different logical tensors and never need to mix.

For our `from_pretrained`-based flow to support this on ZeRO-3, we'd need either:
- Post-hoc surgery: after `from_pretrained`, re-register MoE params with DS using `Init(data_parallel_group=ep_dp_group)`, recomputing all ds_* fields with the smaller group. Invasive.
- An upstream DS change: add `is_moe_param` plumbing to `stage3.py` (mirroring what `stage_1_and_2.py` has). Per-param `ds_process_group` is already the data structure DS needs to read at all collectives — `_allgather_params_coalesced` just needs to size the output tensor based on `dist.get_world_size(get_partition_dp_group(param))` instead of the engine-wide `self.num_partitions`.

### Practical path that works today: ZeRO-2 + EP

`stage_1_and_2.py` has full MoE plumbing already. With our transformers-side patches plus a few small DS-side ones, ZeRO-2 + EP works:

**Patches needed (committed on `ds-ep-integration` branch in /fsx/amine_dirhoussi/transformers):**

1. **`tensor_parallel.py:GroupedGemmParallel.post_shard_wrap`** — under any active DS config (Z1/Z2/Z3), tag EP params with `param.allreduce = False` + `param.group_name = f"ep_size_{N}"`. Skip the DTensor wrap (DTensor is FSDP-only).
2. **`trainer.py:create_accelerator_and_postprocess`** — when `has_ep` + DS, call `deepspeed.utils.groups._create_expert_and_data_parallel(model.tp_size)` so DS's named groups exist before engine init.
3. **`trainer.py:create_optimizer`** — when `has_ep` + DS, run `deepspeed.moe.utils.split_params_into_different_moe_groups_for_optimizer` on the optimizer param groups. Otherwise ZeRO-2's `_configure_moe_settings` asserts "model has moe layers, but no group is marked moe".
4. **DS engine `_configure_distributed_model` (`engine.py:1430-1433`)** — extend the `isinstance(module, MoE)` loop to also trigger `has_moe_layers = True` when any module has a param with `allreduce is False`. Without this, `expert_data_parallel_group=None` is passed to the optimizer and Z2 can't find the smaller group at grad reduce time.

### Result (job 22092899, 16k 2n DS-Z2 + EP=8 + sdpa + sonicmoe, 50 steps)

| Metric | Value | Reference (FSDP+EP+sdpa) |
| --- | --- | --- |
| Final loss | **12.43** | 12.59 ✅ |
| Loss range steps 5-50 | 11.93–13.55 | 11.27–15.74 ✅ |
| mean_token_accuracy | 0.62–0.66 | 0.62–0.68 ✅ |
| Window MFU | **28.6%** | 31.0% (FSDP, sdpa) |
| grad_norm | 5.9–7.3 | (skip, DTensor mesh) |

**Mechanically correct, end-to-end.** ~2 pp behind FSDP+sdpa due to DS-Z2 comm overhead at 16k.

### Long-context fundamental: SP + transformers-EP doesn't compose

Tested DS-Z2 + SP=4 + EP=8 (originally-targeted combo) at both 16k (job 22092907) and 64k (22092901):

| Run | Loss | mean_token_acc | Window MFU |
| --- | --- | --- | --- |
| 16k DS-Z2 + EP=8 (no SP) | 12.43 ✅ | 0.65 ✅ | 28.6% |
| 16k DS-Z2 + SP=4 + EP=8 | 7.7 ❌ | 0.06 ❌ | 4.3% |
| 64k DS-Z2 + SP=4 + EP=8 | 8.1 ❌ | 0.05 ❌ | 19–20% (then NCCL timeout) |

**Why SP + transformers-EP is fundamentally incompatible:**

Ulysses SP shards the sequence dim across SP ranks: each rank only sees `seq / SP` tokens before attention all-to-all. transformers' EP uses **replicate-then-mask**: every rank assumed to hold the FULL token batch, mask routing scores for non-local experts, all-reduce expert outputs across EP. The all-reduce assumes "different ranks computed contributions to the same set of tokens" — but with SP, different ranks computed contributions to *different subsets* of tokens. Summing them is meaningless → garbage hidden states.

DS native MoE works with SP because `deepspeed.moe.layer.MoE` does **explicit all-to-all token dispatch** — tokens move across ranks based on routing. Our transformers EP doesn't (it's the simpler "no all-to-all" design the user has).

**To combine SP + EP, transformers' EP would need a true all-to-all dispatch** (move tokens to the rank holding their target experts, compute, send back). That's a significant rewrite. Out of scope.

### Final practical map

| Goal | Recipe | Status |
| --- | --- | --- |
| Best 16k MoE MFU | FSDP2 + EP=8 + FA3 + sonicmoe | 40.4% ✅ |
| 16k DS-Z2 + EP=8 (this session) | DS-Z2 + EP=8 + sdpa + sonicmoe | 28.6% ✅ (new path, ~2pp behind FSDP) |
| 64k+ MoE | DS-Z3 + SP + FA3 + sonicmoe (no EP) | 19–20% ✅ |
| 64k+ MoE with EP | not viable on this stack — would need all-to-all-dispatch EP, not the replicate-mask design |

EP and SP can't compose with transformers' current EP design, regardless of ZeRO stage. For short context, EP wins (40%); for long context, SP wins (19%); they don't combine.

### Reverted DS patches (since the `is_moe_param` route is the right path, not skip-everywhere)

- `engine.py:_broadcast_model.is_replicated` skip on DTensor — ✅ reverted
- `partition_parameters.py:_convert_to_zero_parameters` skip on DTensor — ✅ reverted
- `parameter_offload.py:mark_persistent_parameters` skip on DTensor — ✅ reverted

Stock DeepSpeed in `.venv` for now. Re-enabling DS-Z3 + EP requires the `is_moe_param` tagging approach above, not the DTensor-skip approach.

### Decision (for now): pivot to FSDP2 + EP + CP for long context

The CP path is MFU-penalized (historical 7.97% at 64k vs SP's 18.98%) but it's achievable tonight. With the EP+FSDP fix and sonicmoe wrapper clamp, training is correct — only the steady-state MFU is the question.

## Test plan

Submit two jobs in order of risk:

1. `dsz3_sp_ep_test.yaml` run 0: 16k smoke. Tests broadcast skip + sonicmoe wrapper clamp under DS-Z3. If it gets past `_broadcast_model` and trains healthy → DS broadcast skip works.
2. `dsz3_sp_ep_test.yaml` run 1: 64k SP=4 EP=8. The actual long-context target. May hit the DS batch-size assertion (pending failure mode above) — if so, fix and retry.
