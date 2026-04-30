# DeepSpeed SP + EP Debug Guide

## Goal

Make DeepSpeed Ulysses Sequence Parallelism (SP) work together with transformers Expert Parallelism (EP) for MoE models when launched through `accelerate launch` with DeepSpeed ZeRO-3.

## What works

### SP+EP via torchrun (no accelerate, no ZeRO-3 env)

The script `benchmark/test_sp_ep_full.py` proves SP+EP work together when loaded outside accelerate:

```python
# Step 1: init both dist backends
dist.init_process_group("nccl")
ds_comm.init_distributed("nccl")

# Step 2: load model with EP (no DeepSpeed env → no zero.Init, no meta device)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-30B-A3B", dtype=torch.bfloat16,
    distributed_config=DistributedConfig(enable_expert_parallel=True),
    device_mesh=ep_mesh,
    attn_implementation="flash_attention_2",
)

# Step 3: register SP attention
mpu = UlyssesSPAttentionHF.register_with_transformers(
    model_name_or_path=model, sequence_parallel_size=2, ...
)

# Step 4: wrap with DeepSpeed ZeRO-3
model_engine, optimizer, _, _ = deepspeed.initialize(model=model, config=ds_config, mpu=mpu)

# Step 5: forward + backward → WORKS
```

Tested on 1 node (4 GPUs) and 2 nodes (16 GPUs). Forward pass produces correct logits, backward completes.

Run command:

```bash
srun --partition=hopper-prod --nodes=1 --gres=gpu:h100:4 --ntasks-per-node=4 --exclusive --time=00:10:00 --qos=normal \
  bash -c 'source /fsx/amine_dirhoussi/trl/.venv/bin/activate && SP_SIZE=2 torchrun --nproc_per_node=4 --master_port=29508 benchmark/test_sp_ep_full.py'
```

### SP without EP via accelerate launch

The 4B dense model and 30B MoE (without EP) work with SP through accelerate:

- 4B SP=2/4 at 16k/32k: all pass
- 30B SP=2 at 16k/32k: pass (MFU=5.8%/13.1%)
- 30B SP=4 at 64k: pass (MFU=11.6%)

These use DeepSpeed ZeRO-3 + SP without `--enable_expert_parallel`.

## What was breaking

### SP+EP via accelerate launch → hangs during model loading

When `accelerate launch` is used with DeepSpeed config + `--enable_expert_parallel`, the job hangs indefinitely. Output shows `NCCL version 2.27.5+cuda12.9` and then nothing for >10 minutes, 0% GPU utilization.

## The fundamental conflict

When `accelerate launch` configures DeepSpeed ZeRO-3, it sets environment variables that make `is_deepspeed_zero3_enabled()` return `True` inside `from_pretrained`. This single boolean gates dozens of code paths throughout model loading. EP needs a different path at every one of those gates:

|                        | EP (transformers)                                                                             | ZeRO-3 (DeepSpeed)                                                     |
| ---------------------- | --------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| Model creation         | Meta device (fast, 0 memory)                                                                  | `deepspeed.zero.Init()` (lazy partitioned params)                      |
| Weight loading         | Standard path: `convert_and_load_state_dict_in_model` + `shard_and_distribute_module`         | Zero3 path: `_load_state_dict_into_zero3_model` + `GatheredParameters` |
| Expert sharding        | `distribute_model()` registers hooks, `shard_and_distribute_module` shards during weight load | N/A — ZeRO-3 slices every param equally across all ranks               |
| Post-load finalization | `_move_missing_keys_from_meta_to_device` moves meta buffers to CPU                            | Skipped entirely (early return)                                        |
| Weight initialization  | Standard `initialize_weights()`                                                               | `GatheredParameters` + `initialize_weights()`                          |
| When DS wraps          | After everything: `deepspeed.initialize(model=already_loaded_model)`                          | During creation: `zero.Init()` inside `cls(config)`                    |

The torchrun test works because `is_deepspeed_zero3_enabled()` is `False` — every gate takes the standard path. The accelerate test fails because every gate takes the ZeRO-3 path.

## Root cause (confirmed via diagnostic tests v3-v7)

We ran 7 incremental diagnostic scripts (`test_ep_ds_loading_debug[1-7].py`) on 4xH100 to isolate the exact failure points. Each test simulated the accelerate environment by setting `HfDeepSpeedConfig(ds_config)` to make `is_deepspeed_zero3_enabled()` return `True`, then stepped through `from_pretrained` piece by piece.

### Bug 1: Model creation hangs (v3 confirmed)

The previous fix attempt added a `pass` branch in `get_init_context` for EP+DS:

```python
if _has_ep and is_deepspeed_zero3_enabled():
    pass  # no zero.Init, no meta device
```

This created the model with **real bf16 tensors on CPU**. For Qwen3-30B-A3B:

- 48 layers x 128 experts x 3 projections = 18,432 expert weight tensors
- ~30B total parameters x 2 bytes (bf16) = ~60GB per process
- 4 processes x 60GB = 240GB CPU RAM allocation
- Plus `init_weights()` runs kaiming_normal on all 30B params

v3 showed the last output was `"Inside context managers, calling cls(config)..."` — the model constructor never returned. Not an NCCL deadlock, just the CPU choking on 4 simultaneous 60GB random-init allocations.

The torchrun test doesn't have this problem because `is_deepspeed_zero3_enabled()` is `False`, so `get_init_context` takes the `else` branch: meta device. Meta tensors allocate zero memory and skip `init_weights`. Model creation takes 0.2s instead of infinity.

### Bug 2: Weight loading uses wrong path (v4-v5 confirmed)

Even if Bug 1 were fixed, `_load_pretrained_model` has this gate:

```python
# modeling_utils.py ~line 4240
if is_deepspeed_zero3_enabled() and not is_quantized:
    # zero3 path: _load_state_dict_into_zero3_model
    # uses deepspeed.zero.GatheredParameters
else:
    # standard path: convert_and_load_state_dict_in_model
    # uses shard_and_distribute_module for EP/TP sharding
```

With the DS env set, this always takes the zero3 path. That path calls `GatheredParameters(params_to_gather, modifier_rank=0)` which expects parameters created by `zero.Init()` (with `ds_id`, `ds_shape`, `ds_status` attributes). But our EP+DS model was created on meta device — the params have none of those attributes. `GatheredParameters` on non-zero3 params hangs waiting for an all-gather that no rank initiates.

The standard path is what EP needs: it calls `convert_and_load_state_dict_in_model` which:

1. Reads all 16 safetensor shard files on every rank (each rank sees all weights)
2. Applies `WeightConverter` operations (merges per-expert weights into fused tensors: `experts.*.gate_proj.weight` + `experts.*.up_proj.weight` -> `experts.gate_up_proj`)
3. For each weight matching the EP plan, calls `shard_and_distribute_module` -> `GroupedGemmParallel.shard_tensor()` which slices `[128, 1536, 2048]` -> `[32, 1536, 2048]` per GPU
4. Non-expert weights (attention, norms, embeddings) are loaded without sharding (replicated)

v4 showed: meta device creation (0.2s) + `distribute_model` (instant, hooks only) + standard loading path = completes in 16s. But with `tp_plan={}` (no EP sharding), expert weights stayed on meta (missing keys).

v5 confirmed the key mismatch: the checkpoint has 18,432 per-expert keys (`model.layers.0.mlp.experts.0.gate_proj.weight`) but the model expects 96 fused keys (`model.layers.0.mlp.experts.gate_up_proj`). The `WeightConverter` from `conversion_mapping.py` handles this merge — but only when passed as `weight_mapping` in the `load_config`.

v6 added `weight_mapping` -> 0 missing keys, expert weights correctly sharded `[128, 1536, 2048]` -> `[32, 1536, 2048]`.

### Bug 3: Post-load finalization skips meta buffer cleanup

Two more methods had `is_deepspeed_zero3_enabled()` early-returns:

```python
# _move_missing_keys_from_meta_to_device
if is_deepspeed_zero3_enabled() and not is_quantized:
    return  # skips moving inv_freq buffers from meta to CPU

# _initialize_missing_keys
if is_deepspeed_zero3_enabled() and not is_quantized:
    # uses GatheredParameters — hangs on non-zero3 params
```

With EP+DS, meta buffers (like `rotary_emb.inv_freq`) need the standard move path, not the zero3 early return.

### Bug 4: `model._tp_plan` vs `model.tp_plan` (pre-existing)

The standard loading path used `model._tp_plan` (raw TP plan dict) instead of `model.tp_plan` (property). When EP is enabled, the `tp_plan` property returns the EP plan:

```python
@property
def tp_plan(self):
    if hasattr(self.config, "distributed_config") and self.config.distributed_config.enable_expert_parallel:
        return self._ep_plan   # EP plan: experts-only sharding
    return self._tp_plan       # TP plan: attention + experts sharding
```

Using `_tp_plan` directly meant the TP plan regex matched attention layers that should NOT be sharded under EP (EP only shards experts, FSDP handles the rest). This caused shape mismatches.

## The fix

All changes in `transformers/src/transformers/modeling_utils.py`. The principle: when EP+DS, make `from_pretrained` follow the same code path as when `is_deepspeed_zero3_enabled()` is `False` (the torchrun path), then let `deepspeed.initialize()` wrap the result afterwards.

### Change 1: `get_init_context` — meta device instead of real tensors

```python
@classmethod
def get_init_context(
    cls, dtype, is_quantized, _is_ds_init_called, allow_all_kernels,
    distributed_config=None,  # NEW parameter
):
    init_contexts = [local_torch_dtype(dtype, cls.__name__), init.no_tie_weights(), apply_patches()]
    if allow_all_kernels:
        init_contexts.append(allow_all_hub_kernels())

    _has_ep = distributed_config is not None and getattr(distributed_config, "enable_expert_parallel", False)

    if _has_ep and is_deepspeed_zero3_enabled():
        # EP + DeepSpeed: use meta device (same as the normal non-DS path).
        # zero.Init is skipped because EP needs to shard experts via distribute_model()
        # hooks, which are incompatible with ZeRO-3 lazy parameters.
        # The standard weight loading path (not zero3) handles EP sharding via
        # shard_and_distribute_module. deepspeed.initialize() wraps the result later.
        init_contexts.extend([torch.device("meta"), init.meta_device_safe_creation_ops()])
    elif is_deepspeed_zero3_enabled():
        # ... existing zero.Init path (unchanged)
    else:
        # ... existing meta device path (unchanged)
```

The caller in `from_pretrained` passes `distributed_config`:

```python
model_init_context = cls.get_init_context(
    dtype, is_quantized, _is_ds_init_called, allow_all_kernels, distributed_config
)
```

**Why meta device, not `pass`?** Meta tensors allocate zero memory and skip `init_weights()`. Creating a 30B model on meta takes 0.2s. Creating it with real bf16 tensors takes 60+GB RAM per process and runs kaiming_normal on 30B params. The checkpoint weights overwrite everything anyway — real initialization is wasted work.

**Why not `zero.Init`?** `zero.Init` makes all parameters lazy (partitioned across ranks). When `distribute_model()` registers EP hooks and later `shard_and_distribute_module` tries to slice expert weights, it triggers `GatheredParameters` all-gathers at different parameters on different ranks — deadlock.

### Change 2: `from_pretrained` — clear device_map for EP+DS

```python
_has_ep = distributed_config is not None and getattr(distributed_config, "enable_expert_parallel", False)
if _has_ep and is_deepspeed_zero3_enabled():
    device_map = None
device_map = check_and_set_device_map(device_map)
```

`initialize_tensor_parallelism` sets `device_map = torch.device("cuda:LOCAL_RANK")`. This triggers accelerate's dispatch-based loading which splits shard files per rank. EP needs all ranks to load all files (each rank reads all 16 shards, then EP picks the experts belonging to that rank). Setting `device_map=None` makes the standard loader place tensors on CPU, which is fine — `deepspeed.initialize()` or `model.to(device)` moves them to GPU later.

### Change 3: `_load_pretrained_model` — skip zero3 loading for EP+DS

```python
_has_ep_with_ds = (
    getattr(getattr(model, "config", None), "distributed_config", None) is not None
    and model.config.distributed_config.enable_expert_parallel
    and is_deepspeed_zero3_enabled()
)

if is_deepspeed_zero3_enabled() and not is_quantized and not _has_ep_with_ds:
    # zero3 path: _load_state_dict_into_zero3_model (unchanged)
    ...
else:
    # standard path: convert_and_load_state_dict_in_model
    ...
    loading_info, disk_offload_index = convert_and_load_state_dict_in_model(
        model=model,
        state_dict=merged_state_dict,
        load_config=load_config,
        tp_plan=model.tp_plan,  # was model._tp_plan — use property for EP plan
        disk_offload_index=disk_offload_index,
    )
```

This is the critical change. The `else` branch runs the standard loading pipeline:

1. Opens all 16 safetensor files via `safe_open(..., device="cpu")`
2. For each checkpoint key, `rename_source_key` matches it against `WeightConverter` patterns:
    - `mlp.experts.*.gate_proj.weight` + `mlp.experts.*.up_proj.weight` -> `mlp.experts.gate_up_proj` (via `MergeModulelist` + `Concatenate`)
    - `mlp.experts.*.down_proj.weight` -> `mlp.experts.down_proj` (via `MergeModulelist`)
3. For keys matching `tp_plan` (which is `model.tp_plan` = EP plan when EP enabled):
    - `layers.*.mlp.experts.gate_up_proj` matches `grouped_gemm` -> `GroupedGemmParallel.shard_tensor()` slices dim 0 by EP rank
    - `layers.*.mlp.gate` matches `ep_router` -> `RouterParallel.shard_tensor()` handles router weights
    - `layers.*.mlp.experts` matches `moe_tp_experts` -> `MoeTensorParallelExperts` handles expert container
4. Non-matching keys (attention, norms, embeddings) are loaded as-is (replicated on all ranks)

The `model.tp_plan` property (not `model._tp_plan`) is essential: when `distributed_config.enable_expert_parallel` is set, it returns `_ep_plan` which only shards MoE expert layers. The raw `_tp_plan` would also shard attention layers (colwise/rowwise), causing shape mismatches with replicated checkpoint weights.

### Change 4: `_move_missing_keys_from_meta_to_device` — don't skip for EP+DS

```python
_has_ep = (
    getattr(getattr(self, "config", None), "distributed_config", None) is not None
    and self.config.distributed_config.enable_expert_parallel
)
# Exception: EP + DeepSpeed uses meta device (not zero.Init), so it needs the standard move path.
if is_deepspeed_zero3_enabled() and not is_quantized and not _has_ep:
    return
```

Without this, non-persistent buffers like `rotary_emb.inv_freq` (48 layers = 48 buffers) stay on meta device. The standard path materializes them as empty tensors on CPU, which `_initialize_missing_keys` then fills with correct values (via `initialize_weights()`).

### Change 5: `_initialize_missing_keys` — don't use GatheredParameters for EP+DS

```python
_has_ep = (
    getattr(getattr(self, "config", None), "distributed_config", None) is not None
    and self.config.distributed_config.enable_expert_parallel
)
if is_deepspeed_zero3_enabled() and not is_quantized and not _has_ep:
    # zero3 path: GatheredParameters + initialize_weights (unchanged)
    ...
else:
    self.initialize_weights()  # standard path
```

The zero3 path wraps `initialize_weights()` in `GatheredParameters` because zero3 params are partitioned and need gathering before init. EP+DS params are real (loaded from checkpoint) or empty (from meta), neither partitioned. `GatheredParameters` on these would hang.

## How the full flow works after the fix

```
accelerate launch with DeepSpeed ZeRO-3 + enable_expert_parallel
    |
    v
from_pretrained("Qwen/Qwen3-30B-A3B", distributed_config=EP, device_mesh=mesh)
    |
    |-- get_init_context: _has_ep=True, is_ds_zero3=True
    |   -> meta device (NOT zero.Init, NOT real tensors)
    |
    |-- cls(config): Qwen3MoeForCausalLM created on meta (0.2s, 0 memory)
    |   48 layers x 128 experts, all tensors on meta device
    |
    |-- distribute_model(model, tp_plan, distributed_config, device_mesh):
    |   Registers EP hooks on all modules (no weight movement):
    |   - GroupedGemmParallel._prepare_input_fn / _prepare_output_fn on expert layers
    |   - RouterParallel._prepare_input_fn / _prepare_output_fn on gate layers
    |   - MoeTensorParallelExperts._prepare_input_fn / _prepare_output_fn on expert containers
    |   Sets model.config.distributed_config = distributed_config
    |
    |-- _load_pretrained_model: _has_ep_with_ds=True
    |   -> STANDARD loading path (NOT zero3 path)
    |   |
    |   |-- Opens 16 safetensor files, reads 18,867 keys
    |   |-- WeightConverter merges per-expert keys:
    |   |   128 x gate_proj + 128 x up_proj -> 1 x gate_up_proj [128, 1536, 2048]
    |   |   128 x down_proj -> 1 x down_proj [128, 2048, 768]
    |   |   (per layer, 48 layers)
    |   |
    |   |-- EP plan sharding via shard_and_distribute_module:
    |   |   gate_up_proj [128, 1536, 2048] -> [32, 1536, 2048] (rank's 32 experts)
    |   |   down_proj [128, 2048, 768] -> [32, 2048, 768]
    |   |   gate (router) weights: replicated (ep_router plan)
    |   |
    |   |-- Non-expert weights loaded without sharding:
    |       q/k/v/o_proj, layer norms, embeddings, lm_head -> replicated on all ranks
    |
    |-- _finalize_model_loading:
    |   |-- _move_missing_keys_from_meta_to_device: _has_ep=True
    |   |   -> runs standard path (NOT early return)
    |   |   -> moves inv_freq buffers from meta to CPU
    |   |
    |   |-- _initialize_missing_keys: _has_ep=True
    |       -> runs standard initialize_weights() (NOT GatheredParameters)
    |       -> fills inv_freq with correct rotary embedding values
    |
    v
model returned: all params on CPU, experts sharded, ready for DS wrap
    |
    v
SFTTrainer.__init__ -> accelerator.prepare(model) -> deepspeed.initialize(model=model)
    |
    |-- DeepSpeed ZeRO-3 wraps the already-EP-sharded model:
    |   - Expert params (different per rank) are partitioned by ZeRO-3
    |   - Non-expert params (same on all ranks) are partitioned by ZeRO-3
    |   - All params get ds_id, ds_shape, ds_status attributes
    |
    v
model_engine: EP-sharded experts + ZeRO-3 memory optimization
forward/backward/step all work
```

## Verification results (test_ep_ds_loading_debug7.py)

4xH100, `is_deepspeed_zero3_enabled()=True`:

```
[   0.0s] 1. init dist, ws=4
[   3.5s]    is_ds_zero3=True
[   3.5s] 2. from_pretrained with EP+DS fix
[  33.8s]    from_pretrained OK!
[  33.8s] 3. Verifying model
[  33.8s]    experts type: Qwen3MoeExperts
[  33.8s]    Expert 'model.layers.0.mlp.experts.gate_up_proj': shape=[32, 1536, 2048], device=cpu
[  33.8s]    Meta params: 0, meta buffers: 0
[  39.6s] 4. Forward OK! logits=[1, 8, 151936], nan=False, max=16.375, min=-6.875
[  47.3s] 5. DeepSpeed init OK!
[  48.1s]    Forward: loss=12.0223
[  49.2s]    Backward + step OK!
```

## What was tried and failed (before the fix)

### Attempt 1: Skip `zero.Init` when EP active

Changed `get_init_context()` to add `_has_ep` check:

```python
_has_ep = distributed_config is not None and getattr(distributed_config, "enable_expert_parallel", False)
if is_deepspeed_zero3_enabled() and not _has_ep:
    # zero.Init block
```

**Result**: Skipped `zero.Init` but fell through to meta device path. Meta device itself is fine (v4 proved this), but the **weight loading** still took the zero3 path -> `GatheredParameters` on meta tensors -> hang. The loading path was the real blocker, not model creation.

### Attempt 2: Skip both `zero.Init` AND meta device (`pass` branch)

```python
if _has_ep and is_deepspeed_zero3_enabled():
    pass  # no context managers
elif is_deepspeed_zero3_enabled():
    # zero.Init
else:
    # meta device
```

**Result**: Created 30B model with real bf16 tensors on CPU. 4 processes x 60GB = 240GB CPU allocation + kaiming_normal on 30B params. Appeared as hang (0% GPU util, no output for 10+ minutes). Even if it completed, the zero3 loading path would still deadlock.

### Attempt 3: Skip TP-aware weight loading for EP+DS

```python
tp_plan={} if _ep_skip_tp_load else model.tp_plan
```

**Result**: This was in the standard loading branch, but EP+DS never reached it — the zero3 loading branch at the `if` above captured it first. Dead code for the EP+DS case.

### Attempt 4-5: Fix device_map

Various attempts to convert `device_map` from `torch.device` to dict or None.

**Result**: device_map fixes alone couldn't help because the zero3 loading path runs regardless of device_map.

### Attempt 6: Set `_is_ds_init_called = True`

**Result**: Skipped `zero.Init` in `get_init_context` but entered meta device path (the `else` branch). Meta device is correct for creation, but the zero3 loading path still ran.

### Attempt 7: Temporarily unset DeepSpeed config

**Result**: Without DS config, `is_deepspeed_zero3_enabled()` returns `False` everywhere, including in the model class `__init__` where it may be needed. Also, `model.tp_plan` defaulted to the TP plan (not EP plan), causing attention weight shape mismatches.

### Attempt 8: Load model then apply EP after

**Result**: Model loaded via zero.Init (lazy params), then `distribute_model()` tried to shard lazy params -> triggered all-gather at different params on different ranks -> deadlock.

### Why previous attempts failed: the loading path was the key

All attempts focused on model creation (which context managers to use). The actual blocker was the `if is_deepspeed_zero3_enabled()` gate in `_load_pretrained_model` that routes to `_load_state_dict_into_zero3_model`. No matter how the model was created, if DS env was set, the loading always went through the zero3 path. The fix had to change **both** model creation **and** loading **and** finalization.

## Current state of changes

### transformers fork (`/fsx/amine_dirhoussi/transformers`)

```bash
cd /fsx/amine_dirhoussi/transformers && git diff --stat HEAD
```

Files changed in `modeling_utils.py`:

1. `get_init_context` — accepts `distributed_config`, EP+DS uses meta device
2. `from_pretrained` — passes `distributed_config` to `get_init_context`, clears `device_map` for EP+DS
3. `_load_pretrained_model` — skips zero3 loading when EP+DS, uses `model.tp_plan` (was `model._tp_plan`)
4. `_move_missing_keys_from_meta_to_device` — doesn't early-return for EP+DS
5. `_initialize_missing_keys` — uses standard init for EP+DS

Other files changed:

- `integrations/tensor_parallel.py` — RouterParallel shape fix, `initialize_tensor_parallelism` validation
- `integrations/moe.py` — `grouped_mm_experts_forward` sentinel handling
- `models/qwen3_moe/configuration_qwen3_moe.py` — expert-only `base_model_ep_plan`

### TRL (`/fsx/amine_dirhoussi/trl`)

1. `trl/scripts/sft.py` — EP model loading as string (for SFTTrainer to handle)
2. `trl/trainer/sft_trainer.py` — `ds_comm.init_distributed` for SP, device_map handling for EP
3. `trl/trainer/utils.py` — `create_model_from_path` distributed_config check
4. `benchmark/templates/launch.sh.j2` — `HF_HOME` env var

## Test scripts

| Script                                       | What it tests                         | Status                |
| -------------------------------------------- | ------------------------------------- | --------------------- |
| `benchmark/test_sp_ep_full.py`               | SP+EP+DS via torchrun (no accelerate) | PASS                  |
| `benchmark/test_ep_ds_loading_debug7.py`     | EP+DS via simulated accelerate env    | PASS                  |
| `benchmark/test_ep_ds_loading_debug[1-6].py` | Incremental diagnostic scripts        | Used during debugging |
| `benchmark/test_sp_moe_debug.py`             | SP+EP step-by-step debug              | PASS                  |
| `benchmark/test_sp_bare.py`                  | Bare SP init (ds_comm check)          | PASS                  |
| `benchmark/test_ep_shapes.py`                | EP correctness (shape matching)       | PASS                  |
| `benchmark/test_no_ep.py`                    | Ground truth logits (no EP)           | PASS                  |

## Critical Bug: RouterParallel Shape Mismatch

### Status: CONFIRMED, NOT YET FIXED (as of 2026-04-15)

### The Bug

`RouterParallel._prepare_output_fn` in `tensor_parallel.py:1094-1145` scatters `router_scores` from shape `(seq, top_k)` into `(seq, num_local_experts)` via scatter+slice:

```python
router_scores = torch.zeros_like(router_logits).scatter_(1, router_indices, router_scores)
router_scores = router_scores[:, ep_rank * num_local_experts : (ep_rank + 1) * num_local_experts]
```

This changes the tensor shape and semantics. But ALL downstream expert forwards expect `top_k_weights` to have the SAME last dimension as `top_k_index`:

- `grouped_mm_experts_forward` (line 383): `sample_weights = top_k_weights.reshape(-1)` produces different size than `expert_ids = top_k_index.reshape(-1)`
- `batched_mm_experts_forward` (line 122): same issue
- Eager forward: `top_k_weights[token_idx, top_k_pos]` uses top_k position to index into expert-indexed tensor

### Evidence

Tested with `benchmark/test_ep_shapes.py` and `benchmark/test_no_ep.py`:

| Test  | scores shape | indices shape | Expert out max | Logits match ground truth? |
| ----- | ------------ | ------------- | -------------- | -------------------------- |
| No EP | (3, 8)       | (3, 8)        | 1.46           | YES (ground truth)         |
| EP=1  | (3, 128)     | (3, 8)        | 0.00           | NO                         |
| EP=2  | (3, 64)      | (3, 8)        | 0.53           | NO                         |
| EP=4  | (3, 32)      | (3, 8)        | 0.28           | NO                         |

- EP=1 produces ZERO expert output — MoE layers contribute nothing
- All EP sizes produce different (wrong) logits
- EP doesn't NaN in short inference but outputs are incorrect; training will diverge

### Root Cause

The scatter operation changes `router_scores` from per-top-k format `(seq, top_k)` to per-expert format `(seq, num_local_experts)`. The expert forward functions consume `top_k_weights` paired 1:1 with `top_k_index`, assuming both have the same shape. The shape mismatch causes wrong routing weights to be applied.

### Fix (not yet applied)

Replace the scatter+slice in `RouterParallel._prepare_output_fn` with a simple masked_fill that preserves the `(seq, top_k)` shape:

```python
non_local_mask = (router_indices // num_local_experts) != ep_rank
router_scores = router_scores.masked_fill(non_local_mask, 0.0)  # stays (seq, top_k)
```

This zeros out non-local expert scores while keeping the original tensor shape. The existing index remapping (lines 1138-1144) is correct and should be kept.

## MFU Computation

Functions in `trl/trainer/utils.py`:

- `compute_flops_per_token(config, seq_len)` — counts matmul FLOPs per token (forward + backward = 3x forward)
    - Dense: attn projections + attention scores + MLP
    - MoE: same attention + `num_experts_per_tok * MoE MLP` per routed layer
    - Detects MoE via `num_local_experts` or `num_experts` on config
- `compute_mfu(flops_per_token, tokens_per_second, world_size, peak_flops=835e12)` — percentage of peak GPU utilization

Integrated in `SFTTrainer.log()`: when `train_tokens_per_second` is in logs, computes MFU and adds to logs. CP correction: divides TPS by `cp_size` to account for token overcounting.

## Model Configs (Qwen3 MoE)

| Field                 | Qwen3-4B (dense) | Qwen3-30B-A3B | Qwen3-235B-A22B |
| --------------------- | ---------------- | ------------- | --------------- |
| model_type            | qwen3            | qwen3_moe     | qwen3_moe       |
| hidden_size           | 2560             | 2048          | 4096            |
| num_hidden_layers     | 36               | 48            | 94              |
| num_attention_heads   | 32               | 32            | 64              |
| num_key_value_heads   | 8                | 4             | 4               |
| intermediate_size     | 9728             | 6144          | 12288           |
| moe_intermediate_size | N/A              | 768           | 1536            |
| num_local_experts     | N/A              | 128           | 128             |
| num_experts_per_tok   | N/A              | 8             | 8               |
| vocab_size            | 151936           | 151936        | 151936          |

## Known Issues and Fixes Applied

1. **FSDP2 + MoE collective shape mismatch**: Different experts active on different ranks -> different gradient tensor sizes -> reduce_scatter fails. Fixed with `fuse_moe_experts()` in `trl/trainer/utils.py`.

2. **CP token overcounting**: `num_input_tokens_seen` inflated by cp_size in transformers. Fixed by dividing TPS by cp_size in MFU computation.

3. **device_map="auto" bypassing EP**: `create_model_from_path` re-adds `device_map="auto"` even after SFT trainer removes it. Fixed by checking for `distributed_config` in kwargs.

4. **JSON quoting in srun bash -c**: gradient_checkpointing_kwargs JSON broke through shell quoting. Fixed by using separate `launch.sh.j2` template.

5. **Slurm config**: `--gres=gpu:h100:8` (not `gpu:8`), no `--cpus-per-task` (use `--exclusive` instead).

6. **torch.compile + FSDP2 + MoE**: Still broken (inductor TF32 error). Not yet resolved.

## Benchmark Results Summary

See `benchmark/consolidated_report.md` for full tables. Key findings:

- Qwen3-4B (dense): FSDP2 works well, CP=2 works, MFU ~30% at 16k context
- Qwen3-30B-A3B (MoE): FSDP2 with fused experts works, EP is broken (RouterParallel bug)
- Qwen3-235B-A22B (MoE): Only FSDP2 with fused experts tested, EP not yet functional
