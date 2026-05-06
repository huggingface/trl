# TODO: Local patches & not-working items (wrap-up)

Self-contained context for a fresh session. Goal: pick up debugging at night without re-reading the full report.

## Where things live

- **Branch**: `ds-ep-integration` (transformers fork) — cloned from `qwen3-moe-ep-v2`
- **Working dirs**: `/fsx/amine_dirhoussi/{transformers,accelerate,trl}` (all installed editable in `/fsx/amine_dirhoussi/trl/.venv`)
- **Full patch detail**: `benchmark/local_only_patches.md` (line numbers + rationale)
- **Latest results**: `benchmark/report.md` (chronological), `benchmark/sft_benchmark_notion.md` (consolidated tables)
- **Open transformers PRs**: `#45662` (EP+FSDP DTensor), `#45621` (sonicmoe Ilyas patch), `#45548` (DS-Z3 EP loading), `#45649` (FSDP cpu_ram_efficient_loading), `#45473` (RouterParallel routing), `#45436` (EP 2D mesh)
- **Cluster ops**: `ip-26-0-172-142` has a hardware NVLink-P2P fault (between local GPUs 0 and 4). Took down 3 of our 2026-05-02 sweep submissions (C4-first, R5-first, C2-second). Workaround: `sbatch --exclude=ip-26-0-172-142`. Should be reported to cluster ops for replacement.

## Status legend

- 🟢 **Ready to PR** — needs writing up but no blocker
- 🟡 **Keep local** — proper fix is in another repo (PyTorch / accelerate / DS)
- ⛔ **Blocked / not yet attempted** — needs investigation
- 🔧 **Active workaround in place** — works for current benchmarks but not production-correct
- 🧪 **To test / explore** — frontier work, not a blocker
- 🗑️ **Debug-only** — remove before any PR

---

## A. Transformers — needs upstream PR

### A1. ✅ `integrations/sonicmoe.py` + `integrations/moe.py` — RESOLVED 2026-05-01 by PR #45621

**Status**: PR author (Ilyas) pushed kernel-side fixes on 2026-05-01 morning that close both the sonic-moe sentinel bug and the grouped_mm autograd leak. We pulled the new commits into local and verified end-to-end (jobs 22099323 / 22099324 / 22099326).

**What changed**:

- **`sonicmoe.py`**: dropped `expert_ids.clamp(0, num_experts-1)` entirely. Kernel's metadata stage now correctly drops `expert_ids >= num_experts` from the histogram and scatter indices internally. Wrapper just unwraps DTensors and passes through. Plus a `_sonicmoe_wrapper` shim with `@torch._dynamo.allow_in_graph` for compile compatibility.
- **`moe.py grouped_mm_experts_forward`**: sentinel handling reduced to **2 wrapper-side `masked_fill_`s** — pre-mask on `selected_hidden_states_g` (backward firewall: zeros sentinel rows of `d_input` after the kernel writes them as uninit, before the gather scatter-add poisons `d_hidden_states`) + post-mask on `weighted_out` (forward firewall: kills `0 × NaN = NaN` before the per-token reduction). Intermediate sentinel-row NaN between the two grouped_mm calls is never consumed because the next grouped_mm only reads rows `< offsets[-1]`.

**Verification (2026-05-01, with kernel fix, no clamp)**:

| Run                                                    | mean / peak Win MFU | vs clamp baseline peak | Notes                              |
| ------------------------------------------------------ | ------------------- | ---------------------- | ---------------------------------- |
| sonicmoe + EP=8 + sdpa @ 16k 2n                        | 33.5 / 35.1 %       | +2.7 pp (vs 32.36 %)   | first validation (job 22099323)    |
| **sonicmoe + EP=8 + FA3 @ 16k 2n**                     | **45.4 / 48.2 %**   | **+7.8 pp** (vs 40.4 %) | new champion at 16k (22099324)     |
| **sonicmoe + EP=8 + FA3 + Liger @ 32k 2n**             | **63.5 / 65.0 %**   | **+8.4 pp** (vs 56.62 %) | new champion at 32k (22099326), highest 30B-on-2-node ever |
| sonicmoe + EP=8 + FA3 + chunked_nll @ 32k 2n           | TBD (job 22099328)  | TBD (vs 45.86 %)       | retry after first attempt NCCL'd at startup |

Loss was healthy across all runs (8.0–13.4) — the kernel fix is not just faster but **more correct** (gradients are clean, not just bypassed via clamp+score=0).

**Next steps**:

1. **DONE**: Pulled 5 PR #45621 commits into local — `80a6fe5a33 fix`, `68b7b0fe2d compilable sonicmoe`, `ad8226ce7c dtensor support`, `a663f4d79c more dtensor`, `74c3f2e3bd simpler`, `d3cae33e30 remove comment`.
2. **DONE**: Removed `expert_ids.clamp` from local `sonicmoe.py`. Wrapper is now byte-equal to PR head modulo our `to_local` helper for moe.py (which the PR doesn't touch).
3. **DONE**: Updated `report.md`, `sft_benchmark_notion.md`, `grouped_mm_pr45621_comment.md`, `sonic_moe_upstream_repro.md` with the new measurements + diagnosis.
4. **DONE (closes upstream sonic-moe issue)**: The Dao-AILab/sonic-moe-side bug we were going to file is now fixed — kernel author pushed it as part of this PR.
5. **PENDING**: Wait for `#45621` and `#45662` to merge into transformers main. Once they do, our fork's sonicmoe.py and moe.py become byte-equal to upstream and we can drop those local patches from the diff.
6. **PENDING**: Re-run all the historical champions in `sft_benchmark_notion.md` once #45621 lands and we're on a vanilla branch — the +5–8 pp delta we measured locally should reproduce, refreshing every "Win MFU" number in the consolidated tables.

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

### C2. 🔴 `tp_plan`-loader duplicates non-plan params on every rank — wastes GPU memory + disk I/O at scale

**Scope**: this is bigger than originally framed. The fix description below targets the FSDP+`cpu_ram_efficient_loading=True` corruption (Layer 2), but the underlying Layer 1 (every rank reads the full dense replica from disk to its own GPU) **affects every EP run we do, every TP run any other team does, regardless of `cpu_ram_efficient_loading`**. It's the canonical TP-plan load behavior, baked into `initialize_tensor_parallelism` since the feature shipped, and nobody noticed because the unit was kicked down the road via the loading-walkthrough doc's "in the canonical FSDP recipe..." framing — but the canonical recipe is broken under EP, so people use the non-canonical recipe, where the problem just sits there silently.

**Files**: `transformers/src/transformers/integrations/tensor_parallel.py:initialize_tensor_parallelism` (root cause) → `transformers/src/transformers/core_model_loading.py:convert_and_load_state_dict_in_model` (where the duplicate loads happen) → `accelerate/utils/fsdp_utils.py:fsdp2_prepare_model` (where the broadcast clobbers EP).

**The Layer 1 problem (always-on, hits all `tp_plan` users)**: `initialize_tensor_parallelism` unconditionally sets `device_map = cuda:LOCAL_RANK` whenever any of `tp_plan`, `tp_size`, or `device_mesh` is passed to `from_pretrained`. `_materialize_copy` then issues `tensor[...].to("cuda:N")` on every rank, for every parameter — including the **non-plan params** (dense weights — embeddings, q/k/v/o, norms, router gates) that are *replicated*, not sharded. So every rank pulls its own full copy of the dense replica from disk straight to its own GPU. Cost on real workloads:

| Config | Dense per rank | Ranks/node | Duplicate dense GPU mem/node | Duplicate disk I/O/node |
| --- | --- | --- | --- | --- |
| Qwen3-30B-A3B EP=8, 1 node  | 2.87 GiB | 8 | 22.96 GiB | 480 GB |
| Qwen3-30B-A3B EP=8, 2 nodes | 2.87 GiB | 8 | 45.92 GiB | 960 GB |
| Qwen3-235B-A22B EP=8, 16 nodes | ~5-6 GiB | 8 | ~640 GiB | 7.5 TB |

The 235B numbers match the "25+ min stalls" reported in `report.md:4525` for the 235B no-EP path on FSx, which has the same shape (every rank reads its full copy from FSx).

This wastes both GPU memory (FSDP would later shard the dense replica anyway → only one rank's worth was ever needed) and disk bandwidth (FSx hammered N× redundantly). For large MoEs at scale, this is the biggest single avoidable cost in `from_pretrained` today.

**The Layer 2 problem (silent corruption, fires only with `cpu_ram_efficient_loading=True`)**: when `enable_expert_parallel=True` and `cpu_ram_efficient_loading=True` are both set, loading silently produces a broken model — no crash, just degenerate loss / wrong outputs. Two interacting steps in `fsdp2_prepare_model` cause it:

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

### C3. 🔧 EP-aware DataLoader sharding — TRL piggy-backs on accelerate's TP-replication path; clean upstream fix still TBD

**Status**: ✅ ROOT CAUSE FOUND 2026-05-02. ✅ Local fix in place (TRL only). ⛔ Proper upstream fix (first-class `ep_size` in accelerate) NOT YET IMPLEMENTED — design captured below.

**Problem in one line**: when EP > 1 with multi-node, accelerate's dataloader gives every world rank a unique micro-batch, but the 8 ranks of an EP group must see the **same** micro-batch (EP shards experts only, not tokens). Mismatched shapes inside the EP group cause silent NCCL hangs at the EP all-reduce.

**Files** (both local fixes now live in transformers — TRL is fully out of the EP-DataLoader business):
- Local fix (DS path): `transformers/src/transformers/trainer.py::create_accelerator_and_postprocess` — injects the EP mesh (`model._device_mesh`) into `accelerator.state.ds_device_mesh` for DS+EP runs. Sits next to the existing DS+EP `_create_expert_and_data_parallel` block. accelerate's existing TP-replication branch in `prepare_data_loader` does the rest natively.
- Local fix (FSDP/MULTI_GPU path): `transformers/src/transformers/trainer.py::_get_dataloader` — monkey-patches `accelerator.prepare_data_loader` for one call with EP-corrected `num_processes`/`process_index`. FSDP can't use the mesh-injection trick because `state.device_mesh` is also read by `fsdp2_prepare_model` (`fsdp_utils.py:651`) which would crash without a matching `parallelism_config`.
- Investigation: `benchmark/report.md` (Tests D + E + F + G section).
- Primer / study material: `benchmark/dataloader_ep_primer.md` (how distributed dataloaders work and where EP breaks the assumption); `benchmark/sftrainer_lifecycle.md` (full lifecycle walkthrough — DS vs FSDP boundaries, where the fix sits in each path).

**Key insight (2026-05-04)**: EP and TP are equivalent for dataloader sharding — both are data-replicate dims (ranks within the dim see the same micro-batch). Accelerate's `prepare_data_loader` already has the right code path for TP (inspects `torch_device_mesh.mesh_dim_names` for `"tp"` at `data_loader.py:1119-1155`, divides `process_index`/`num_processes` by `submesh_tp_size`). We can piggy-back on that path without changing accelerate — just expose a mesh with `"tp"` in its dim names. The proper upstream fix would still rename this to `"ep"` for clarity, but for sharding semantics it's the same divisor.

**Why the bug exists in the first place**: accelerate's `ParallelismConfig` (`accelerate/parallelism_config.py:34`) has fields for `dp_replicate_size`, `dp_shard_size`, `tp_size`, `cp_size`, `sp_size` — **no `ep_size`**. EP isn't a first-class concept anywhere in accelerate. Transformers builds an EP device mesh internally via `apply_tp_plan` (with `mesh_dim_names=("dp", "tp")`) but never registers it with accelerate. So accelerate's `state.device_mesh` is None for EP runs — its TP-replication branch never fires — and in our 16-GPU EP=8 setup the dataloader ships 16 unique data shards instead of 2-replicated-8x.

**Why it stays silent for many steps**: with `--packing --pad_to_multiple_of 1`, each micro-batch packs to a variable seq_len. Most micro-batches happen to pack to exactly `max_length=65536` (LongAlign-10k samples are very long), so most of the time all 8 ranks of an EP group **coincidentally agree** on shape and the EP all-reduce succeeds. The bug fires whenever a single rank's batch packs to a different length — random walk over batches → eventually hits → hang.

**Empirical signature** (Tests D/E/F before the fix):
- Hang state in NCCL flight recorder: `state=scheduled, never started` (CUDA stream blocked because participants disagree on tensor size).
- 7 of 8 EP-group ranks fire watchdog timeouts; the 8th doesn't (its all-reduce launched a different-sized one).
- Different seed → different sample order → bug fires at a different step (Test D step 16 with seed=42, Test E step 10 with seed=99) — confirms data-driven.
- Identical fingerprint across kernels (sonicmoe vs grouped_mm both hang at step 16, same SeqNum=2201) — confirms the bug is **upstream** of the experts kernel.

**Local fix (current, 2026-05-04)**: piggy-back on accelerate's existing TP-replication path by exposing a `"tp"` mesh of size `ep_size` to accelerate's prepare_data_loader. Two paths because the wiring point differs:

- **DEEPSPEED path (the common one for EP)**: in `transformers/trainer.py::create_accelerator_and_postprocess`, when `is_deepspeed_enabled and model.has_ep`, assign `accelerator.state.ds_device_mesh = model._device_mesh`. The mesh transformers itself built (1D `("tp",)` of size `ep_size`, set in `tensor_parallel.distribute_model`) is reused — no new NCCL splits. Accelerate's `_prepare_device_mesh` returns `ds_device_mesh` for DS runs (`accelerator.py:2604-2612`), so prepare_data_loader's DEEPSPEED branch (lines 1120-1130) sees `"tp"` in dim names and natively divides `process_index = global_rank // ep_size, num_processes = world_size // ep_size`. **No prepare_data_loader patching, no override.** Lives next to the existing `_create_expert_and_data_parallel` block (~line 835) so the EP-specific accelerator setup is co-located.
- **FSDP / MULTI_GPU path**: in `transformers/trainer.py::_get_dataloader`, when `model.has_ep and is_fsdp_enabled`, monkey-patch `accelerator.prepare_data_loader` for one call with explicit `num_processes = world_size // ep_size`, `process_index = global_rank // ep_size`, and `torch_device_mesh=None`. Try/finally restores the original after the call. The clean piggy-back is not viable for FSDP because `state.device_mesh` is also read by `fsdp2_prepare_model` at `fsdp_utils.py:651` — slipping a 2D `(dp_shard, tp)` mesh in there would also need a matching `parallelism_config` populated, which changes FSDP wrap semantics. The override is the smaller surgery.

**Validated by Test G (job 22102668, 2026-05-02)**: identical config to the deterministic step-16 hang in Test D, with the override active → completed 30/30 steps cleanly, train_loss=1.555, healthy gradients. **Re-validated post-piggy-back refactor by job 22108306 (2026-05-04)** — same config (DS-Z2 + EP=8 + ctx 64k + sonicmoe + chunked_nll, 2 nodes), `ds_device_mesh` reuses transformers' EP mesh in `__init__`, override disabled for DS. 30/30 steps, `train_loss=1.555` (exact match), step-1 loss `1.973` matches Test G exactly. Confirms accelerate's existing TP-replication path in `prepare_data_loader` handles EP correctly once the mesh is exposed.

**Implementation note (mesh reuse, not creation)**: an earlier attempt (job 22108189) called `dist.init_device_mesh("cuda", (ep_size,), mesh_dim_names=("tp",))` to build a fresh 1D mesh and got `AttributeError: 'NoneType' object has no attribute 'group_name'` from torch's `_init_one_process_group` — `split_group` returns None on the second sub-mesh attempt because transformers' `apply_tp_plan` already consumed the world PG split. Workaround: reuse the mesh transformers itself built (`model._device_mesh`, set in `tensor_parallel.py:distribute_model`). Same dim name "tp", same ranks, no new NCCL groups.

**Proper upstream fix (TWO PRs, deliberately not yet written)**:

1. **accelerate** — add `ep` as a first-class parallelism dim:
   - `parallelism_config.py:34`: add `ep_size: Optional[int] = None` field, plus entries in `__repr__`, `total_size`, `dp_dim_names`, `non_data_parallel_dim_names`.
   - `data_loader.py:1146-1155` (non-DeepSpeed path) and 1119-1130 (DeepSpeed path): add `submesh_ep_size = torch_device_mesh["ep"].size() if "ep" in torch_device_mesh.mesh_dim_names else 1` and divide `process_index` by it. EP is semantically identical to TP for dataloader sharding (both replicate data within the dim) — the change mirrors the existing TP handling exactly.

2. **transformers** — make EP visible to accelerate:
   - When `DistributedConfig(enable_expert_parallel=True)` is set on the model, push `ep_size` into `accelerate.state.parallelism_config.ep_size` at trainer init.
   - Long-term cleaner version: unify the two meshes — let accelerate own the multi-dim mesh (`(dp, ep, tp, cp)`) and have transformers' EP layers consume `device_mesh["ep"]` instead of building their own.

**Why the upstream fix is non-trivial enough to warrant design**: transformers and accelerate currently each build their own device_mesh independently. Today they don't talk to each other. A clean fix needs ONE source of truth — either accelerate owns it (and transformers reads `device_mesh["ep"]` from it), or transformers owns it (and exposes the dim names accelerate can read). Picking which side owns it requires alignment with both repos' maintainers.

**Order of operations once we're ready**:

1. Open accelerate PR for the `ep_size` ParallelismConfig + data_loader changes. Should be small, easy review (mirrors existing TP handling).
2. Open transformers PR (depends on 1) wiring `DistributedConfig.enable_expert_parallel` into `ParallelismConfig.ep_size` at trainer init.
3. Once both merge, delete the local pieces in `transformers/trainer.py`: (a) the `ds_device_mesh` injection block in `create_accelerator_and_postprocess` (~10 lines), and (b) the FSDP-fallback `prepare_data_loader` patch block in `_get_dataloader` (~40 lines). Both are replaced by accelerate's native handling once `ep_size` is a first-class field.

**Revert (local fix only)**:
```bash
# Both blocks live in transformers now. TRL's _get_dataloader is empty of EP logic.
git -C /fsx/amine_dirhoussi/transformers checkout -- src/transformers/trainer.py
```

**Multi-night investigation history that this resolves**: D-works.bug (below) and the entire "ROOT CAUSE narrowing" sub-tree. Tests A-F (NCCL state, Triton streams, autograd accumulation, FSDP backend, kernel swap, gc.collect) all confirmed the bug was NOT what they targeted. Test E with seed=99 proved data-dependence (rules out step-counter). Test F with grouped_mm proved kernel-independence. Per-rank EP-boundary debug instrumentation (added in this session — see `tensor_parallel.py::_maybe_log_ep`) plus per-step batch metadata (`sft_trainer.py::compute_loss BATCH_DEBUG_DIR` block) showed rank 4 sending `[1, 54223]` while ranks 0-3,5-7 sent `[1, 65536]`. Hang explained.

---

## D-works. DeepSpeed-Z2 + EP — fixed and working

> **Status**: ✅ **WORKING** as of 2026-04-28. DS-Z2+EP=8 is the long-context champion path (45.81 % MFU at 32k, 57.23 % at 64k, 69.10 % at 128k — all on 2-4 nodes). Loss healthy with sonicmoe + clamp wrapper. This is the recipe used in every chunked-CE long-context champion in the report.
>
> ⚠️ **Caveat (2026-05-01, kernel-fix era)**: see D-works.bug below — without the wrapper clamp (post-PR-#45621 stack), the EP=8 + DS-Z2 + sonicmoe + multi-node path now hangs in `ALLREDUCE` on the EP process group (PG ID 2) after ~5–15 training steps. Reproduces across 4 independent runs (P1 at 128k 4n step ~5, P2 ×3 at 64k 2n step ~15-16). The current stable champions still use the **clamp era** code; re-validating without the clamp on multi-node EP=8 is currently bug-blocked.

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
2. Patch (3) → ✅ **RESOLVED 2026-05-06** by a transformers-side monkey-patch on `DeepSpeedEngine._configure_distributed_model`, installed once from `Trainer.create_accelerator_and_postprocess` next to the existing `_create_expert_and_data_parallel(ep_size)` call. Static analysis of the three options originally sketched (a/b/c, kept below for reference) showed (a) is insufficient on its own: `_configure_moe_settings` runs DURING `deepspeed.initialize` and bakes `expert_dp_process_group` into the optimizer's `real_dp_process_group` / `partition_count` lists at init time — setting `engine.has_moe_layers = True` AFTER `accelerator.prepare` returns is too late. The actual fix runs at the same lifecycle point as the in-venv `engine.py:1430` shim (within `_configure_distributed_model`, before `_configure_optimizer`) but lives entirely outside DS:

    ```python
    import deepspeed.runtime.engine as _ds_engine
    if not getattr(_ds_engine.DeepSpeedEngine, "_external_moe_patched", False):
        _orig_cdm = _ds_engine.DeepSpeedEngine._configure_distributed_model
        def _wrapped_cdm(engine, model):
            _orig_cdm(engine, model)
            if not engine.has_moe_layers:
                for _, mod in engine.module.named_modules():
                    if any(getattr(p, "allreduce", True) is False for p in mod.parameters(recurse=False)):
                        engine.has_moe_layers = True
                        engine.num_experts.append(getattr(mod, "num_experts", 0))
        _ds_engine.DeepSpeedEngine._configure_distributed_model = _wrapped_cdm
        _ds_engine.DeepSpeedEngine._external_moe_patched = True
    ```

    Validated end-to-end at 16k 2n DS-Z2 + EP=8 + sonicmoe + Liger (slurm 22112383, COMPLETED 3:43, 5/5 steps healthy, mfu_window peak 34 %). Stock DS works as-is — `uv pip install --reinstall --no-deps deepspeed` is no longer required. Lives in `transformers/trainer.py` around line 877. Original three options preserved below for reference:

    a. **Post-`deepspeed.initialize` engine attribute patch** (originally preferred) — after `accelerator.prepare(model, optimizer, ...)` returns the wrapped engine, write `engine.has_moe_layers = True` and `engine.optimizer.expert_data_parallel_group = groups._get_expert_data_parallel_group(f"ep_size_{ep_size}")` directly. **Insufficient on its own** — the ZeRO optimizer's `_configure_moe_settings` already ran (or didn't) at init based on the original `has_moe_layers`; setting it after doesn't replay that bookkeeping.

    b. **Pre-`deepspeed.initialize` module marker** — set `module._has_moe_layers = True` (or whatever attribute DS's detection loop checks) on each EP-experts module before `accelerator.prepare`. Would require wrapping experts in a `deepspeed.moe.layer.MoE`-shaped shim because the existing detection is `isinstance(module, MoE)` — collapses into option (c).

    c. **Wrap external EP modules in a `deepspeed.moe.layer.MoE`-shaped shim** — wraps EP modules in a class that quacks like DS's `MoE` (exposes `deepspeed_moe`, the right `group_name`, etc.) so DS's detection finds them naturally. Most invasive; superseded by the monkey-patch above.

3. TRL side (6) → falls under G2 in this todo.
4. (7) → A1 in this todo.

---

### D-works.bug. ✅ RESOLVED 2026-05-02 — root cause was EP-unaware dataloader sharding (see C3 above)

> **Resolution**: the multi-night investigation in the sub-sections below (P/Q sweep failures, P2 ×3, FSDP2 same-fingerprint, "shape-dependent ALLREDUCE-PG-ID-2" — every clue) is now explained by a single root cause: accelerate's dataloader gives every world rank a unique micro-batch when EP > 1, but EP-group ranks must share data. Mismatched shapes inside an EP group → silent NCCL hang on the EP all-reduce. See **C3** above for the full diagnosis, the local TRL workaround in place, and the proper upstream-fix design. Test G (job 22102668) validated the workaround end-to-end. Re-running the affected sweep configs is now unblocked.
>
> The historical investigation log below is preserved for reference. Do NOT continue narrowing — the fix is upstream-shaped and tracked under C3.

### D-works.bug-history. (resolved by C3) — multi-night investigation 2026-05-01 → 2026-05-02

> **Resolved by C3** (EP-unaware DataLoader sharding). All hypotheses below were misleading symptoms of one root cause: shape mismatch within an EP group at the EP all-reduce. Detailed test-by-test investigation log is in `report.md` (Tests A through G). This block is preserved as a quick-reference for the failing fingerprints and for the vacuum scripts (still useful as control reproducers).

**Failing-fingerprint catalog** (use these to recognize the same bug if it appears in another stack):

| Job | Config | Hang signature |
| --- | --- | --- |
| P1 (22099522) | 128k 4n EP=8 | step 5, `WorkNCCL(SeqNum=621, OpType=ALLREDUCE, NumelIn=268435456)` PG ID 2 |
| P2 ×3 (22099330/359/426) | 64k 2n EP=8 | step 15-16, `SeqNum=2201, NumelIn=134217728` PG ID 2 |
| 22099828 | 64k 2n FSDP2 EP=8 (control) | step 15, `SeqNum=2166, NumelIn=134217728` PG ID 2 |
| Q6/Q10 (no compile) | 128k 4n/8n EP=32/64 | silent zombie post-init (same root cause, no loud timeout) |
| 22102169 (final, with debug instrumentation) | 64k 2n EP=8 grouped_mm | step 16, identical `SeqNum=2201` — confirmed kernel-independent; per-rank debug logs proved rank 4 sent shape `[1, 54223]` while ranks 0-3,5-7 sent `[1, 65536]` |

**Hypotheses ruled out** (all wrong; preserved so we don't re-investigate):
- ❌ NCCL itself — isolated repro of the failing all-reduce shape ran 10000 iters cleanly
- ❌ Wrapper clamp / no-clamp split (the 2026-05-01 21:00 "bisect breakthrough" was a false positive)
- ❌ DS-Z2 / FSDP2 backend — both hang identically
- ❌ sonicmoe / grouped_mm kernel — both hang identically
- ❌ Liger / chunked_nll loss — both hang identically
- ❌ FA3 / sdpa attention — both hang identically
- ❌ Gradient-checkpointing recompute (vacuum v5 cleared it)
- ❌ NCCL_ALGO=Ring, NCCL_PROTO=Simple, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `.contiguous()` on AR input
- ❌ `accelerator.gather_for_metrics` at the world-PG (initial trace-dump misread; gather was a downstream symptom)
- ❌ "Buffer leak / cumulative collective volume" theory (vacuum v1-v5 transferred 80+ GB without issue)
- ❌ "Per-rank seq ≥ 48k threshold" theory — turned out to be data-distribution-dependent, not a true threshold

**Vacuum scripts (still useful — checked in)**:
- `benchmark/test_ep_allreduce_repro.py` — pure NCCL all-reduce tight loop
- `benchmark/test_ep_layer_vacuum.py` — autograd Functions + sentinels
- `benchmark/test_ep_layer_vacuum_v2.py` — + 48 layers + AdamW
- `benchmark/test_ep_layer_vacuum_v3.py` — + DP reducer
- `benchmark/test_ep_layer_vacuum_v5_sonicmoe.py` — + REAL sonicmoe + grad_ckpt
- `benchmark/test_ep_layer_vacuum_v6.py` — + world-PG `_all_gather_base` per step
- All complete cleanly at the failing 64k 2n EP=8 shape — they do not reproduce the bug because they replicate inputs across EP-group ranks, exactly avoiding the data-mismatch trigger. Useful as control reproducers if any future hang shows the same fingerprint and you want to rule out NCCL/cluster.

**NCCL trace dumps from the hangs**: `/fsx/amine_dirhoussi/nccl-traces/nccl_trace_22099859_*` and `_22102169_*`. The latter is the one used to diagnose the shape mismatch (parsed in `report.md` — Test G section).

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

### E3. ✅ RESOLVED 2026-05-02 — was a manifestation of the EP DataLoader bug (see C3)

> **Resolution**: every "+compile + multi-node hang pre-first-step" symptom listed below was the same EP-DataLoader-shape-mismatch hang, just with compile making the warmup window long enough that the data divergence didn't happen until step 0's first EP all-reduce. Post-fix sweep 2026-05-02:
>
> - **Q1 (64k 8n EP=64 + compile)** completed cleanly, mfu_window peak 82.12 %.
> - **Q2 (64k 4n EP=32 + compile)** completed cleanly, mfu_window peak 73.23 %.
> - **A1 (128k 4n EP=32 + compile)** completed cleanly, mfu_window peak 81.79 %, healthy first-step loss=1.9.
> - **R5 (256k 8n DS-Z3+SP=2 + compile)** completed cleanly, mfu_window peak 64.59 %, train_loss 1.562 ✅.
> - **R6 (1M 8n DS-Z3+SP=8 + compile)** completed cleanly, mfu_window peak 63.89 %, train_loss 1.563 ✅.
>
> "TCPStore broken pipe spam from non-zero ranks → silent zombie" was exactly what 7-of-8 EP-group ranks waiting on a stuck NCCL collective looks like when the watchdog timeout is 600s instead of 30s — the LOUD failure (heartbeat) was hidden but the underlying hang was the same as the SeqNum=2201 EP all-reduce we eventually traced. The compile-warmup is a red herring; the hang fires at step 0's first EP all-reduce regardless of whether compile is enabled.
>
> Heartbeat-timeout bump (`TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=600` etc.) is **kept** in `templates/launch.sh.j2` since it doesn't hurt and made the hang easier to capture with NCCL trace dumps. Investigation history below preserved for reference.

(Original investigation — superseded by C3, kept for context):

Multi-node +compile jobs (P3/P4/P5/Q2/Q3) hung pre-first-step with TCPStore broken pipe symptoms. Hypothesis at the time was inductor-compile/distributed-init coordination. Mitigations tried (heartbeat timeout, `compile_threads = 1`, `dynamic=False`, etc.) all failed to fix the hang because the actual root cause was the data shape mismatch at step 0's first EP all-reduce. With C3 in place, every config above runs end-to-end.

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

**Update 2026-05-02 — partially obsoleted by Liger + DS-Z2 path post-EP-fix**:
With Liger + DS-Z2 + EP path post-EP-DataLoader-fix (C3), we now train **64k 4n EP=32 at 72 % mfu_window** (R3, healthy loss=1.867), **128k 4n EP=32 at 81.73 % mfu_window** (R1), and **128k 8n EP=64 at 95.39 % mfu_window** (R4) — all WITHOUT CP. The Liger + chunked path frees the lm_head logit memory, leaving room for the EP-replicated activation buffer at higher per-rank seq. So the FSDP+EP+CP path that was the 32k workaround is no longer the only way; DS-Z2+EP+Liger now scales 16k → 128k cleanly. The architectural ceiling at 256k+ still requires SP path (DS-Z3+SP=2/8) — see R5/R6 in `report.md`.

**Fix paths still relevant**:

1. **EP+CP path** (15.6 % at 32k): no longer needed for 30B; might still be useful for 235B where EP=64 + CP=2 at 64k may fit where DS-Z2+EP alone OOMs.
2. **Drop EP degree** at long context: not needed; DS-Z2+EP=8/32/64 + Liger handles 32k → 128k now.
3. **Stream the expert dispatch** — kernel-level rewrite. Still the right long-term ceiling-breaker for 256k+ EP. Biggest payoff but biggest scope. RFC still TODO.

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

1. **MFU helper functions only** (`compute_flops_per_token` + `compute_mfu` in `utils.py`). NOTHING else — no `SFTTrainer.log` integration, no `_flops_per_token` field, no cp/sp correction wiring, no `mfu_window`/`tps_window`. Standalone helpers any user can call from their own training loop. Pre-PR fixes from analysis 2026-05-03: causal-correct attention (halve `attn_score_flops`), `embed_flops=0` + `lm_head_flops=2*V*h` always (currently inverted: counts embed lookup as a matmul, drops lm_head matmul when tied — they cancel for tied models, over-count by ~4pp for untied). Add docstring note that `compute_mfu` assumes user has already corrected TPS for cp/sp/tp over-counting (since `num_input_tokens_seen` gathers across the whole world). Add tests for Qwen3-4B and Qwen3-30B-A3B FLOPs/token expected values. Branch off `main`, NOT off this branch.
2. **`SFTTrainer.log` MFU integration** (deferred — separate PR after #1 lands). Wires `compute_flops_per_token` at `__init__`, adds `mfu`/`mfu_window`/`tps_window` to `log()`, divides TPS by `cp_size × sp_size` to correct `num_input_tokens_seen` over-counting. Adds `_last_log_time` / `_last_log_tokens` state. Decision needed: how to handle TP>1 (currently no divide; would need to read `tp_size` from `parallelism_config` and divide). Window vs cumulative is a useful contribution — most TRL users don't realize cumulative is dominated by step-1 compile cost.
3. **EP-aware DataLoader override** (`_get_dataloader` in SFTTrainer, deferred). Real fix for the EP-vs-DP-data confusion uncovered in MFU analysis 2026-05-03: with flat-mesh EP=world_size, FSDP gives each rank a unique batch but EP all-reduces partial expert outputs across the same world — summing different-data partials. Override patches `accelerate.prepare_data_loader` for one call to use `num_processes=world//ep_size` and `process_index=rank//ep_size`, so EP groups receive the same micro-batch. Already documented under C3, but worth re-flagging here as it's coupled to the MFU "is the all-reduce semantically meaningful" question.
4. **Legacy TF32 flags + per-rank Triton cache dir** in `sft.py`. Tiny PR, fixes a real PyTorch 2.10+ inductor crash + multi-node training bug. No MoE coupling.
5. **`enable_expert_parallel` + `expert_parallel_size` + `experts_implementation` config fields and the EP branch in SFTTrainer**. The biggest, most coupled chunk. Needs the upstream PRs (#45662, #45621, #45473, #45436) to land first; until then, the EP path requires our local transformers fork. Ship after the transformers PRs.
6. **Generalized kernel pre-warm + `HF_HUB_OFFLINE` flip** + `--save_strategy no` gate. Pre-warm was originally EP-only; generalized 2026-05-02 to all multi-node runs after B7 (235B 16n DS-Z3+SP, no EP) sat in Hub fetch for 10+ min. Workaround for H3; ship as a TRL infra PR or a doc note depending on whether HF Hub upstream fixes the cache lock-file issue first.
7. **`pad_to_multiple_of` auto-default for SP** (G1 above). Independent of the rest.

**Status today**: nothing upstreamed. All 4 files diverge from `main` and need to be re-based and split before any PR. `fuse_moe_experts` (helper + config flag + trainer call) should be removed from the branch entirely — it's a no-op on current transformers.

**Why split #1 from #2**: the helper functions are pure Python with no TRL coupling — anyone (Megatron user, custom trainer, vLLM benchmark) can call them. Landing them first lets the convention discussion (causal vs non-causal, embed/lm_head accounting, TP=1 assumption) happen in public on a tiny PR rather than getting tangled with `SFTTrainer.log` API churn. The "50% MFU got x.com skeptics" framing wants public review of the formula itself, not of the trainer integration.

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

### H4. ⛔ **(NEW 2026-05-01)** sonicmoe Triton kernels ignore `TRITON_CACHE_DIR` → FSx race on `/fsx/<user>/.triton/...token_gather_sum_kernel.source`

> **Issue**: Multi-node EP=32 sonicmoe runs fail with `FileNotFoundError: [Errno 2] No such file or directory: '/fsx/amine_dirhoussi/.triton/<HASH>/token_gather_sum_kernel.source'`. The error path is `~/.triton/<HASH>/`, **not** the per-rank `/tmp/triton-rank-${RANK}-${HOSTNAME}` we set up in `trl/scripts/sft.py:69-73`. So `TRITON_CACHE_DIR` is being silently ignored by sonicmoe's kernel-compile-and-load path.

**Reproduced (2026-05-01)** in 2-of-2 multi-node EP=32 sonicmoe runs:
- Q4 (job 22099531): 128k 4n EP=32 SP=2 + FA3 + Liger + sonicmoe — `token_gather_sum_kernel.source` not found.
- Q8 (job 22099534): 32k 4n EP=32 + FA3 + Liger + sonicmoe — `token_gather_sum_kernel.ptx` not found.

After the FileNotFoundError, the rank that hit it died, then NCCL collectives on the rest of the ranks deadlocked on the dead-rank's expected calls — eventually `ProcessGroupNCCL's watchdog got stuck for 600 seconds without making progress in monitoring enqueued collectives` (heartbeat-monitor itself hangs because CUDA is in unrecoverable state).

**Hypothesis**: sonicmoe is loaded as a `kernels-community/...` HuggingFace kernel via the `kernels` library. Its Triton kernel compilation (`@triton.jit`) probably runs **before** `sft.py` sets `TRITON_CACHE_DIR`, OR uses a different cache mechanism that bypasses `TRITON_CACHE_DIR` (e.g., `triton.runtime.cache` with a hardcoded path).

**Earlier-today datapoint that does NOT hit this**: Q9 (job 22099370) was 16k 2n FSDP no-EP + FA3 + Liger + sonicmoe + compile — succeeded with 38.74% peak. Difference: 16 ranks vs 32 ranks (less FSx contention probability), or the kernel was already cache-warm from prior runs.

**Fix paths**:

1. Set `TRITON_CACHE_DIR` via the `srun` env in `templates/launch.sh.j2` (export *before* python starts), not from inside `sft.py`. This avoids any race where Triton initializes before `sft.py:69` runs.
2. Pre-warm sonicmoe kernels in a single-rank subprocess at job startup (writes to a node-local `/tmp` cache); subsequent ranks then read-only.
3. Inspect the `kernels` library to see if it has its own per-rank cache config.
4. Worst case: copy sonicmoe's compiled `.source`/`.ptx` files into the venv at install time and never rebuild.

**Next steps**: (1) is one line in `launch.sh.j2` and addresses the ordering hypothesis. Try first.

### H5. ⛔ **(NEW 2026-05-02, RE-CLASSIFIED 2026-05-03)** "loss=0" is actually NaN being masked downstream — real bug at large EP+long-ctx configs

> **Status**: Found during the 2026-05-02 post-EP-DataLoader-fix sweep. **Not blocking the EP DataLoader fix** — fix itself is validated (R3, R5, R6, C4, Test G all converge cleanly). This is a separate numerical-correctness bug surfaced once the EP-multi-node configs started training end-to-end.

**Pattern** (rows where the bug fires):

| Run | Shape | Loss behavior | mfu_window peak |
| --- | --- | --- | --- |
| Q1 | 64k 8n EP=64 + Liger + compile | loss=0 from step 2 onward | 82.12 % |
| Q2 | 64k 4n EP=32 + Liger + compile | loss diverges 7 → 16 | 73.23 % |
| Q4 | 128k 4n SP=2 EP=32 + Liger | loss diverges 6 → 14 (separate G1 issue) | 66.15 % |
| R1 | 128k 4n EP=32 + Liger | loss diverges 1.9 → 13.7, grad_norm 1e6+ | 81.73 % |
| R2 | 64k 2n EP=16 + Liger | loss=0 from step 2 onward | 80.32 % |
| R4 | 128k 8n EP=64 + Liger | loss=0 from step 2 onward | 95.39 % |
| C1 | 32k 4n EP=32 + Liger | loss=0 from step 2 onward | 69.77 % |
| C2 | 32k 8n EP=64 + Liger | loss=0 from step 2 onward | 66.76 % |
| A1 | 128k 4n EP=32 + Liger + compile | loss diverges 1.9 → 11.4 | 81.79 % |
| A2 | 128k 4n EP=32 + Liger (sdpa) | loss diverges 7.7 → 13.9 | 41.98 % |

**Pattern** (rows where convergence is healthy):

| Run | Shape | Loss | Notes |
| --- | --- | --- | --- |
| Test G | 64k 2n EP=8 + sonicmoe + chunked_nll (no Liger) | train_loss 1.555 ✅ | non-Liger path, 30/30 steps clean |
| **R3** | 64k 4n EP=32 + Liger + sonic | train_loss 1.867 ✅ | first Liger healthy multi-node EP=32 |
| **R5** | 256k 8n DS-Z3+SP=2 + Liger + sonic + compile | train_loss 1.562 ✅ | DS-Z3 SP path |
| **R6** | 1M 8n DS-Z3+SP=8 + Liger + sonic + compile | train_loss 1.563 ✅ | DS-Z3 SP path |
| C4 | 16k 8n EP=64 + Liger + sonic | train_loss 1.62 ✅ | comm-bound but healthy |

**No single trigger isolated yet**. Hypotheses to test:
1. **All-reduce of routing weights**: in `MoeTensorParalellExperts._prepare_input_fn` we run `all_reduce_backward(top_k_weights, device_mesh)`. With Liger's FLCE replacing the lm_head, the gradient flow through routing weights might miss this all-reduce, leading to inconsistent updates per-rank. Cheapest test: instrument the all_reduce to verify same input on all EP-group ranks.
2. **Liger FLCE vs sonicmoe interaction at high EP**: FLCE flattens chunked logits; if sonicmoe expects a specific gradient shape that Liger feeds back differently at EP > 8, that could corrupt routing-weight gradients.
3. **First-step sentinel masking edge case**: when most experts get 0 tokens at high EP (EP=32 means 4 experts/rank), sonicmoe's masked_fill may interact badly with Liger's chunked CE.
4. **`compile` interaction with chunked-CE in Liger** specifically — Q2 + A1 both diverge in a similar way; non-compile Liger configs at the same shape (R1 vs A1) also diverge, so compile may not be the trigger.

**Investigation steps (cheap-to-expensive)**:
1. Re-run R1 with chunked_nll instead of Liger (Test-G recipe at 128k 4n) — does chunked_nll converge where Liger diverges? If yes → Liger-specific.
2. Re-run R1 with `--use_liger_kernel false` and chunked_nll — same shape, no Liger. If converges → Liger numerical issue.
3. Add per-step gradient-norm-on-routing-weights logging (similar to the EP-debug instrumentation already in `sft_trainer.py:compute_loss`) to compare healthy R3 vs failing R1.
4. Profile the FLCE backward path under EP — does it see the EP all-reduce hook on top_k_weights?

**Why this matters**: the post-EP-DataLoader-fix headline numbers are **throughput-correct but training-incorrect** for the rows above. Real training requires healthy convergence, which only R3/R5/R6/C4 currently achieve in the Liger path. Once H5 is fixed, the Q1/R1/R4 numbers (82-95 % mfu_window peak) become real-training-correct, which would set new SFT MoE training records.

**Workaround for users today**: at multi-node + high-EP + Liger shapes, validate convergence carefully before any production run. If loss=0 or diverges, fall back to `chunked_nll` (slower but converges).

**Update 2026-05-03 (235B sweep)**: ran X1 = 235B 32k 16n EP=128 with `chunked_nll` instead of Liger. **Same loss-zero result** (mfu_window 71.81%, train_loss 0.18 artifact). So at 235B-scale-cross-node-EP=128, the trigger is NOT Liger-specific. Updated hypothesis: trigger is **cross-node EP-group + large per-rank seq → routing-weights backward all-reduce is incorrect**. The H5 issue is in `MoeTensorParalellExperts._prepare_input_fn`'s `all_reduce_backward(top_k_weights, device_mesh)` at scale, NOT in the loss kernel.

**Update 2026-05-03 03:15 (H5 isolation test)**: ran 30B 32k 16n FSDP2+**EP=8 intra-node** + Liger (job 22104173). **STILL loss=0** from step 10. Final: train_loss 0.29, mfu_window peak 69.32 %.

So **cross-node EP is NOT the H5 trigger** — even intra-node EP=8 at 16n + 32k goes loss=0. **Revised hypothesis**: trigger is `(32k+ context) AND (16n+ scale)` regardless of EP topology. Possible underlying causes:
- FSDP at 128-rank scale: gradient bucketing or unsharding has an issue at large per-rank activation
- Cross-node DP all-reduce timing/ordering at scale
- A bf16 precision issue magnified by 16n × 32k

**Healthy shapes** (for reference, all post-EP-fix):
- 30B 64k 4n EP=32 + Liger (R3): healthy ✅
- 30B 32k 8n EP=64 + Liger (no — C2 was loss=0 at 8n EP=64)... hmm wait C2 was 8n though. Let me re-check.

Re-examining: C2 was 30B 32k 8n EP=64, loss=0. So 8n + 32k + EP=64 already loss=0. Not just 16n+. So:
- 8n EP=64 + 32k → loss=0 (C2)
- 8n EP=64 + 16k → healthy ✅ (B6 — 235B; 30B equivalent unmeasured)
- 8n EP=32 + 64k → healthy ✅ (R3 — 4n actually)
- 16n EP=8 + 32k → loss=0 (the current H5 isolation test)
- 16n EP=128 + 32k → loss=0 (B3v2 235B)

Pattern: **at "≥32k context AND large total rank count", loss=0**. The "EP topology" doesn't matter.

Most likely cause: **per-rank gradient or weight all-reduce at FSDP-scale 64+** — when there are many DP/EP ranks with 32k+ tokens, some accumulated gradient becomes zero (or gets clobbered to zero). Could be:
- bf16 underflow in summed gradients
- A masking ordering issue at scale
- An FSDP2 reshard-after-forward + EP interaction at large rank count

Path forward (E-load-style debugging applied to H5):
1. Add per-step `top_k_weights.grad` norm logging to `MoeTensorParalellExperts._prepare_input_fn` (post-AR).
2. Run on (a) failing shape and (b) healthy shape; compare grad norms across ranks.
3. If grad norm = 0 on all ranks immediately, the issue is upstream of EP backward.
4. If grad norm is non-zero on some ranks but zero on others, it's an EP-AR ordering issue.

For the 30B and 235B sweep: every healthy run is at smaller scales. Large-scale runs are throughput-correct but loss-correctness-broken. Marking this as **the next high-priority investigation** for any production training of 30B/235B at 32k+ context on 8+ nodes.

### H5 — verification update 2026-05-03 morning: NOT REPRODUCIBLE the next day

Re-ran the same configs that had loss=0 yesterday with H5 instrumentation:

| Diagnostic | Original (yesterday) | Re-run today | Verdict |
|---|---|---|---|
| **D1** (= C1 30B 32k 4n EP=32 + Liger) | loss=0 throughout | **loss 0.7-2.3 healthy 15/15 steps** ✅ | bug GONE |
| **D2** (same shape + chunked_nll) | n/a (new probe) | loss 0.7-2.3 healthy ✅ | healthy |
| **D3** (= R3 30B 64k 4n EP=32 + Liger) | healthy ✅ | healthy ✅ | matches |
| **E1** (= D1 BUT NO H5 instrumentation) | loss=0 in C1 | **loss 1.87 healthy** ✅ | bug GONE without instrumentation |
| **E2** (sync only via H5_FORCE_SYNC=1) | n/a | healthy | sync isn't the differentiator either |

**Key finding**: even E1 (baseline, no instrumentation, no `.item()` sync) is healthy. The C1→D1→E1 sequence — same shape, same code modulo opt-in instrumentation — gave opposite outcomes between days. **The H5 loss-zero is not reproducible the next day.**

Most plausible explanations:
1. **Cluster state**: yesterday's runs landed on different nodes (possibly transient-faulty); today's land elsewhere. We've confirmed `ip-26-0-172-142` had hardware issues; other nodes may have had transient issues we never identified.
2. **HF datasets cache state**: yesterday's runs computed the cache mid-sweep (cold/partial state); today's benefit from a fully-warm cache.
3. **Other transient cluster fabric issue at scale** that's gone today.

**Action**: not pursuing H5 as a kernel/code bug. Documented as "transient — re-test failing configs the next day before assuming kernel-level bug". The H5 instrumentation (`H5_DEBUG_DIR`, `H5_FORCE_SYNC=1`) remains in `sft_trainer.py` as **opt-in diagnostics** for any future occurrence. Low-priority going forward.

**Remaining open**: re-test R4 (30B 128k 8n DS-Z2+EP=64+Liger) — yesterday's highest mfu_window measurement (95.39%) but loss=0. **W1 in flight** (job 22105404). If healthy → recovers the highest-MFU number ever measured WITH healthy convergence as the new headline.

### H5 — RE-CLASSIFIED 2026-05-03 ~08:48 — actually NaN, not loss-zero

W1 (22105404, R4 redo at 30B 128k 8n DS-Z2+EP=64+Liger) **reproduced loss=0 at step 10**. So H5 is NOT transient — it's shape-deterministic. The H5 instrumentation finally revealed the real bug:

| step | loss.item() PRE-backward | nan? |
|---|---|---|
| 0 | 2.161430 | False (healthy) |
| 1 | NaN | **True** |
| 2 | NaN | True |
| ... all subsequent | NaN | True |

**The trainer's reported `loss=0` is NaN-masked downstream (likely in `_maybe_log_save_evaluate` or the grad scaler).** The actual `compute_loss` returns NaN starting at step 1. Same for `grad_norm=2.236` — that's a constant value across all post-step-1 steps, suggesting the grad clipping or norm computation hits a sentinel when fed NaN gradients.

**Origin (working theory)**: step 0 forward is healthy (loss=2.16). Step 0 backward + optimizer.step produces NaN weights for some param (bf16 overflow in optim state, or NaN gradient propagating through clip). Step 1 forward with those NaN weights → NaN loss. Cascade.

**Reframes yesterday's sweep**: every "loss=0 throughout" entry — Q1, Q2, R1, R2, R4, C1, C2, A1, A2, B3v2, X1 — was **actually NaN**, not zero. Throughput numbers (MFU, TPS) are still real because forward+backward+optim still execute (just with NaN math).

**Why the smaller shapes today don't reproduce it** (D1 = 32k 4n EP=32 healthy, but C1 yesterday was loss=0):
- Hypothesis: at smaller scale + smaller context, gradient magnitudes are smaller → bf16 stays in range → no NaN
- C1 yesterday may have been a borderline case that flipped over the bf16 edge that one time
- W1 (128k 8n EP=64) is FAR from the bf16 edge — robust NaN every run

**Investigation plan**:
1. **Add NaN check on weights post-optimizer.step** (one-line change in trainer.py). Identify which params become NaN.
2. **Try tighter `max_grad_norm`** (default 1.0 → 0.5 or 0.1). If reduces NaN frequency, it's gradient-magnitude-driven.
3. **Try fp32 optimizer state** (Adafactor or bitsandbytes 8-bit Adam in fp32 master). Rules out Adam fp32 → bf16 cast issue.
4. **Disable expert all-reduces individually** to find which one introduces NaN. Could be `_AllReduceForward` (post-experts) vs `_AllReduceBackward` (top_k_weights bw).
5. **Compare per-rank weight stats post-step-0 optim** between healthy R3 (64k 4n EP=32) and failing W1 (128k 8n EP=64). The failing config produces NaN-creating gradients somehow.

**Why this matters**: the throughput numbers from yesterday's "loss=0" runs (Q1, R4, B3v2, etc.) are valid throughput-only datapoints, NOT training-correct results. Production training of 30B/235B at long context + high EP **needs the NaN bug fixed first**.

**Priority**: high. This is the single largest blocker to using these recipes for real training.

---

## J. Qwen3-235B-A22B — operating envelope (2026-05-03 morning)

First end-to-end-correct multi-node 235B SFT measurements, post-EP-DataLoader-fix (C3) + Liger + flat-EP. Prior cluster best was **2.9 % MFU at 32k 8n with cpu_offload+CP**. Now (with healthy convergence):

| Shape | Recipe | mfu_window peak | rolling final | Loss | Verdict |
|---|---|---|---|---|---|
| 16k 8n | FSDP2+EP=64+Liger | 63.58 % | 29.20 % | 1.326 ✅ | works |
| 16k 16n | FSDP2+EP=128+Liger | 54.36 % | 27.52 % | 1.325 ✅ | works |
| **32k 8n** | **FSDP2+EP=64+Liger** | **76.86 %** | 53.28 % | **1.482 ✅** | **+26× window vs prior** |
| **32k 8n + compile** | **FSDP2+EP=64+Liger+compile** | **75.33 %** | **70.14 %** | **1.482 ✅** | **+17 pp rolling vs no-compile** |
| 32k 16n | FSDP2+EP=128+Liger | 86.72 % peak | 55.37 % | 0.22 † | H5-yesterday loss-zero (retest pending W1-style) |

**Memory boundaries we charted overnight**:

- 235B 4n 32k EP=32 (Y1): **OOM at first forward** → need ≥8 nodes for 235B at 32k
- 235B 8n 64k EP=64 (no test) — predicted to OOM (B2v2 used 70 GB at 32k; 64k doubles activation)
- 235B 16n 64k EP=128 (B4v2): **OOM at first forward** → 64k needs CP+sdpa or ≥32 nodes
- 235B 16n 128k DS-Z3+SP=8 no-EP: **CPU RAM overflow** (128 ranks × 470 GB > 2 TB/node) → needs `cpu_ram_efficient_loading=True` (broken with EP per C2; should work for no-EP; not yet tested)

**EP topology lesson at 235B scale**: EP=8 OOMs at NCCL init at 8 nodes (16 experts/rank × 235B ≈ 37 GB just expert weights). **Use flat EP=world_size** for 235B so each rank holds 1-2 experts.

**Open follow-ups for 235B** (not blocking; documented for next session):

1. **`cpu_ram_efficient_loading=True` for 235B no-EP path** — DS-Z3+SP=8 at 128k+ on 16n. C2's EP+ramcache bug doesn't apply to no-EP. Not yet tested. Would unlock 128k+/256k+ at 235B.
2. **235B compile delta on 16k 8n** (B6 + compile = analog to Y2 measuring +17 pp on 32k). Uses fewer resources, useful headline.
3. **235B with `--gradient_checkpointing_kwargs '{"use_reentrant": true}'`** — reentrant grad-ckpt may shrink activation memory enough to fit 64k 16n EP=128 (B4v2 OOMed by 4 GB).
4. **235B at 16 nodes EP=128 + chunked_nll** (X1 reproducer) — yesterday loss=0; W1-style retest pending.

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

### I3. 🧪 **(NEW 2026-05-02 night)** Investigate why model loading is slow under DeepSpeed and FSDP

**Observation**: every multi-node run spends a long time (sometimes minutes) before the first training step appears. The "model loading" phase between `Started: <ts>` and the first `'loss': ...` log is dominated by something we don't fully understand. Anecdotal evidence: 235B 16-node jobs sit silent for 5-10 minutes before training starts; 30B 4-8n jobs ~1-3 minutes. The wall-clock cost adds up across hundreds of benchmark runs.

**Why this matters**: faster loads mean (a) more time-to-first-result for short-iteration debugging, (b) lower compute-burn for the cluster (16-node × 5 min = 1.3 GPU-hours per submission wasted in init), (c) a sharper tail latency budget for production training (auto-resume, checkpoint-load).

**Hypothesis dump (to test)**:

1. **Disk I/O bottleneck on `from_pretrained`** — every rank reads the same shards from FSx. With 128 ranks all hitting the same files concurrently, FSx may serialize or rate-limit. Fix idea: prefetch shards to local NVMe per-node before training starts.
2. **State-dict broadcast under FSDP `cpu_ram_efficient_loading`** — when set, rank-0 loads the full state dict and broadcasts to all other ranks. Broadcast over EFA at 470 GB takes time. With it OFF (current default for our EP runs), every rank reads independently from FSx — also slow but in a different way.
3. **Triton / Inductor warmup before first step** — sonicmoe + FA3 + Liger all have JIT-compiled kernels that warm up on first forward. Per-rank Triton cache helps but cold cache means N ranks compile in parallel.
4. **`accelerator.prepare`** — wraps model with FSDP/DS. Internally does an all-gather of the full state to verify shape; for 235B this is expensive.
5. **Transformers EP plan apply** — `apply_tp_plan` walks the entire module tree and registers hooks; for 94-layer 235B this may be O(layers²) due to nested module dicts.
6. **Dataloader prefetch** — first batch of packed sequences from the dataloader requires tokenization + packing of LongAlign-10k samples. With 235B's slow forward, this might overlap; with 30B's fast forward, this might be on the critical path.

**Targeted experiments** (cheap, single-config repros):

- **E-load-1**: instrument `from_pretrained` → `accelerator.prepare` → `model(first_batch)` boundary timestamps with `time.time()` in `trl/scripts/sft.py`. Run a small 16k 2n DS-Z2+EP=8 (30B) and a 16k 8n FSDP2+EP=64 (235B) baseline. Compare per-stage timings.
- **E-load-2**: profile a 235B 8n run with `py-spy` attached to rank 0 and rank 1 between job-start and first-loss. Where is rank-0 spending time vs rank-N?
- **E-load-3**: with `HF_HUB_OFFLINE=1` + pre-warmed FSx cache, time `from_pretrained` for 30B and 235B at varying world sizes (1n / 4n / 8n / 16n). Is loading time linear in world size? Constant? Worse?
- **E-load-4**: measure FSx read bandwidth from a single rank vs concurrent reads from N ranks against the same shard file. Does FSx serialize?
- **E-load-5**: try `dataloader_persistent_workers=True` and `dataloader_num_workers=4` to overlap data prep with model load.
- **E-load-6**: check if `core_model_loading.py:convert_and_load_state_dict_in_model` can short-circuit reads on rank > 0 when EP is on (related to C2 — today every rank reads from disk even with `cpu_ram_efficient_loading=True`).

**Smallest test surface first**: E-load-1 (instrumentation). One commit to `sft.py` adds 5-10 timestamps with `_log("stage_X", time.time())`. Captures the headline numbers in a single 16k 2n run. Then prioritize the slowest stage.

**Status**: 🧪 E-load-1 + E-load-2 instrumentation in tree (2026-05-02 → 2026-05-03). `[LOAD-T]` markers in `sft.py` and `sft_trainer.py`; analyzer at `benchmark/analyze_load_timing.py`. Default-on; opt out with `LOAD_TIMING=0`. Three findings shipped:

1. **`from_pretrained` is fast** even for 235B (~17-19s at 8n FSDP2+EP=8). Not the bottleneck — early hypothesis 1 (FSx I/O) ruled out.

2. **Dataset preprocessing is the dominant first-run cost**: `_prepare_dataset` (tokenize + pack of LongAlign-10k) takes ~330s on rank 0, gated by `main_process_first()` so all ranks wait. B6 hit this cold (350s); subsequent runs hit cache and drop to 11s. **30× speedup on subsequent runs from the cache hit.** Fix path: pre-tokenize offline + `skip_prepare_dataset=True` to avoid the cold-cache cost on every fresh sweep.

3. **Post-`from_pretrained` is the dominant non-cache bottleneck**: at 235B 16k 8n with cached dataset, the gap from `trainer_train_start` to `first_training_step_entry` is **~4:50 min**. This is `accelerator.prepare(model)` (FSDP2 wrap = ~1000 `fully_shard` calls × mesh barrier) PLUS first-batch fetch PLUS first-forward Triton/Inductor warmup. We don't yet have a marker between FSDP wrap and first forward to split these.

4. **Sub-linear scaling with model size**: 30B 16k 2n took ~90s end-to-end vs 235B 16k 8n took ~325s — only 3.6× longer for 7.8× more params. Suggests post-fp cost is part-constant (mesh barriers, dispatch) and part-linear (param transfer).

**Next experiment (E-load-3)**: instrument `accelerator.prepare(model)` itself with a wrap-time marker. Either by monkey-patching `Accelerator.prepare` or adding a marker right before `_inner_training_loop` calls it. Once we have the FSDP wrap timing isolated, we can compute the first-forward delta directly.

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

- [ ] **C3 — EP-aware DataLoader sharding (TWO upstream PRs, deliberately not yet written)** → root cause of the multi-night EP-multi-node ALLREDUCE-PG-ID-2 hang. Local override in TRL (`sft_trainer.py:1568::_get_dataloader`, ~30 lines, in place 2026-05-02 and validated by Test G + the entire 235B sweep — every previously-failing EP-multi-node config now trains). Proper fix needs: (a) accelerate PR adding `ep_size` to `ParallelismConfig` + `submesh_ep_size` reduction in `prepare_data_loader` (mirror of TP handling); (b) transformers PR wiring `DistributedConfig(enable_expert_parallel=True)` into `accelerate.state.parallelism_config.ep_size` at trainer init. Design discussion needed first: who owns the multi-dim device mesh — accelerate or transformers — currently both build their own independently. Once both PRs merge, delete the local override. **Highest-leverage upstream PR remaining**.
- [ ] **C1 — accelerate `_prepare_tp` `has_ep` skip** → open accelerate PR (single check, references `#45662` for context). Revert: `uv pip install --reinstall --no-deps accelerate`.
- [ ] **C2 — `tp_plan` loader duplicate-load bug (now reframed broader, 4 actions in order)**:
    - (a) **File upstream issue** at huggingface/transformers — draft in `benchmark/upstream_issue_tp_plan_duplicate_load.md`. Frames Layer 1 as the universal-impact problem, Layer 2 as the EP-specific corruption.
    - (b) transformers Patch 1: rank-0 gate in `convert_and_load_state_dict_in_model` so non-rank-0 leaves non-plan params on meta. Universal: helps all TP/EP users (FSDP, DS-Z2, DS-Z3), not just `cpu_ram_efficient_loading=True`.
    - (c) transformers `ValueError` guard in `from_pretrained` when `enable_expert_parallel=True` and `cpu_ram_efficient_loading=True` are both set, plus EP doc note (immediate safety net for Layer 2 until (d) lands).
    - (d) accelerate Patch 2: `fsdp2_prepare_model` (~10 lines capture/restore `ignored_params` around meta move) + `fsdp2_load_full_state_dict` filter (~3 lines, rank>0 branch).
    - End-to-end test: peak/rank should drop from 16.4 GiB to ~13.5 GiB on 30B-A3B EP=4 after (b). Layer 2 corruption goes away after (d).
- [ ] **G1 — SP `--pad_to_multiple_of` auto-default** → TRL PR adding the auto-default in `SFTTrainer` when `accelerator.parallelism_config.sp_size > 1`.
- [x] **H1 (debug + sweep)** — Liger + Qwen3-MoE under EP: root-cause identified, workaround `--liger_kernel_config '{"swiglu":false}'` validated, full sweep run vs chunked-CE champions. Liger wins at every context 16k–1M (+0.3 to +25 pp peak MFU). New all-time MoE record: 128k @ 76.29 % (40.5 % causal-adjusted). Full investigation in `benchmark/debug_liger_ep.md`, results in `report.md` + `sft_benchmark_notion.md`. **Still TODO**: file a Liger PR/issue with the repro at `benchmark/test_liger_qwen3_moe_ep.py` proposing the dispatcher-aware fix (auto-detect EP via `model.config._experts_implementation`).
- [x] **TRL `_preparse_dict_args` helper** — added to `trl/scripts/sft.py` to JSON-decode `--liger_kernel_config` before HfArgumentParser sees it (HfArgumentParser falls back to `type=dict` and rejects JSON strings). 12 lines, no other changes. Should also become an upstream TRL PR if HfArgumentParser doesn't fix this.
- [ ] **H2 — FA3 + CP gated to sdpa in accelerate** → file an accelerate feature request, or open a PR after verifying FA3 actually composes with the CP scheduler at the Python level.

### 🟢 PR-after-deps-land

- [ ] **A1 — sonicmoe EP/DTensor support (`.to_local()` + clamp wrapper)** → blocked on `#45621` (sonicmoe Ilyas patch) + `#45662` (EP+FSDP DTensor wrap). Write up follow-up PR explaining the autograd-through-`to_local()` backward gap.
- [ ] **Deepseep EP-works — DS-Z2 + EP, 7 patches across 3 repos**, the recipe behind every long-context champion in the report. Split into:
    - [ ] transformers PR — patches 1/2/4/5 (`tensor_parallel.py`, `trainer.py:create_accelerator_and_postprocess`, `trainer.py:create_optimizer`, `trainer.py:_clip_grad_norm`) **+ the `_configure_distributed_model` monkey-patch that replaces the deleted DS PR** (validated 2026-05-06, slurm 22112383 — see D-works "Upstream split" item 2)
    - [x] ~~DeepSpeed PR — patch 3~~ — landed as a transformers-side monkey-patch on `_configure_distributed_model` (2026-05-06). Stock DS works as-is.
    - [ ] TRL PR — patch 6 (pre-init `deepspeed.comm` in SFTTrainer EP branch — falls under G2 below)

### 🟡 Keep local (proper fix is upstream elsewhere)

- [ ] **B1 — `_clip_grad_norm` skip when `has_ep`** → file a PyTorch issue describing the cross-mesh `_foreach_norm` stack failure with a minimal repro. Long-term fix is in PyTorch's `clip_grad_norm_` or accelerate's wrapper.

### ⛔ Investigation — smallest test surface first

- [ ] **E2 — Triton cache file-not-found at FSDP+EP+CP+compile @ 64k** → try `TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-rank-${RANK}-${HOSTNAME}` next to existing `TRITON_CACHE_DIR`. Cheapest test.
- [ ] **E1 — FSDP+EP+compile Adam `_group_tensors_by_device_and_dtype`** → try the surgical `_group_tensors_by_device_and_dtype` patch in PyTorch (treat EP and FSDP DTensors as compatible if device+dtype match).
- [ ] **H3 — multi-node HF Hub cache race** → try per-rank `HF_HOME=/tmp/hf-rank-${RANK}` to avoid lock contention. Investigate whether newer `huggingface_hub` already has `HF_HUB_DOWNLOAD_TIMEOUT` + retries.
- [ ] **H5 — actually NaN-loss being masked to 0 by trainer downstream** (RE-CLASSIFIED 2026-05-03 morning, REVISED 2026-05-03 03:15, **STATIC ANALYSIS 2026-05-04**) → W1 (R4 redo at 30B 128k 8n DS-Z2+EP=64+Liger) reproduced "loss=0" pattern. H5 instrumentation revealed `loss.item()` PRE-backward is **NaN** from step 1. W2 (22107681) shows step-0 grad_norm=2.66M, step-1 grad_norm=2913 (finite!), step-2 weights NaN — but ONLY for `embed_tokens.weight` + `layers.0.self_attn.{q,k,v,o}_proj.weight` (the 5 FSDP/DS-sharded entry-side params). MoE experts + layer 1+ stay finite. Post-bwd NaN-grad logger never fires → gradients themselves are finite; NaN appears inside Adam's update for those 5 shards specifically. **Static-analysis pass 2026-05-04 ruled out**: EP `_AllReduceBackward`/`_AllReduceForward` math (correct AR_sum semantics), DS-Z3 `reduce_scatter` averaging (`stage3.py:1442-1446` divides correctly), Liger FLCE chain-rule scaling (autograd handles `loss/num_items` correctly), `num_items_in_batch` computation, C3 dataloader (verified by bit-identical loss across rank 0/1/8/16/32/48/63 in W2 logs), `_maybe_log_ep` instrumentation (no compute side-effects). Bug is in `(grad → optim_state → param)` step, not in forward/backward math. Top hypotheses: (a) DS-Z3+EP=world entry-side weight loading puts a few elements out of bf16 normal range, only manifesting after step-1 optim, (b) bf16 v_t overflow on outlier element from step-0 huge grad propagates Inf → NaN at step 1, (c) DS-Z3 `unscale_and_clip_grads` corner case at world=64. Smallest tests to distinguish: add per-param grad-norm log (vs just NaN check), run with `--max_grad_norm 0.1`, run with `--optim adafactor` (rules out v_t overflow), run with FSDP2 instead of DS-Z3 at same shape (rules out DS-specific). **HIGH PRIORITY**. Static analysis is exhausted; needs cluster runs. Full analysis log appended to `report.md` 2026-05-04 section.
- [ ] **FSDP + per-chunk lm_head NCCL hang** (chunked-CE AND Liger FLCE both hit it). Both `--loss_type chunked_nll` and `--use_liger_kernel true` (FLCE) compute the lm_head matmul in chunks of 256 valid tokens; under FSDP the lm_head weight is sharded across DP, so each chunk triggers an all-gather → ~519 collectives enqueued, only 8 complete → NCCL timeout. **Current workaround: use DS-Z2 instead of FSDP** for the EP+chunked or EP+Liger path (DS-Z2 doesn't shard params across DP, so no per-chunk gather). Proper fix paths: (a) exclude `lm_head` from FSDP wrap, (b) wrap the chunk loop in `model.lm_head.unshard_context()` so the weight stays gathered for all chunks in a step, (c) move the chunked patch before FSDP wrap. See `report.md` for the chunked-CE repro and the 16k FSDP+EP+Liger crash log (job 22094539).

### ⛔ Larger projects

- [ ] **D-blocker — DS-Z3 + EP rank-ordering fix** on top of the existing 3-patch tagging from commit `cd52547f87`. Either align transformers' EP partition to use `global_rank % ep_size` or build DS's `expert_parallel_group` manually with matching rank order. 1–2 days. Detailed plan in `debug_sp_ep_sonic.md` Iteration 4.
- [ ] **F1 — streaming expert dispatch (kernel rewrite)** → RFC in transformers MoE integrations to avoid materializing the full `(seq, num_local_experts, moe_intermediate)` activation tensor. Biggest payoff (unblocks 32k+ EP without CP), biggest scope.

### 🧪 Frontier — completed and validated

- [x] **I1 — Chunked loss in SFT** ([trl#5575](https://github.com/huggingface/trl/pull/5575), already merged into TRL `main`). Cherry-picked onto branch on 2026-04-28; unlocked DS-Z2+EP at 32k–128k (45–69 % MFU window) and SP+chunked at 256k–1M (32–60 % MFU window). See "chunked-CE results" in report.md and sft_benchmark_notion.md.
- [x] **Liger × MoE × EP unblocked** (2026-04-29). Disproved "3D weights" hypothesis with single-GPU repro. Found root cause: `_patch_swiglu_module` bypasses transformers' EP-aware dispatcher; Liger's `LigerExperts.forward` hits `F.one_hot` with `num_classes == sentinel`. Fix: `--liger_kernel_config '{"swiglu":false}'`. Liger wins at every context 16k–1M (+0.3 to +25 pp peak vs chunked). Repros: `benchmark/test_liger_qwen3_moe{,_ep}.py`. Investigation log: `benchmark/debug_liger_ep.md`.
- [x] **MFU causal-correction post-hoc adjustment** (2026-04-29). Helper `benchmark/adjust_mfu_causal.py` (CLI + CSV mode) computes `adjusted_mfu = reported_mfu × adj_factor` where `adj_factor` subtracts half the attention-score FLOPs (causal). Applied to all 3 headline tables in `sft_benchmark_notion.md`. Raw MFU column kept untouched per user request. Verified all 36 cells against an independent recompute from `compute_flops_per_token`.

### 🧪 Frontier — pending

- [ ] **I3 — Slow model loading under DeepSpeed / FSDP** (NEW 2026-05-02 night) → instrumentation pass first (E-load-1: 5-10 `time.time()` markers in `sft.py` between `from_pretrained` → `accelerator.prepare` → first batch), then prioritize the slowest stage. See section I3.
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

# DeepSpeed: stock is now sufficient (engine.py monkey-patch lives in transformers/trainer.py).
# Only run the reinstall if a previous session wrote to engine.py in-place.
uv pip install --reinstall --no-deps deepspeed
```
