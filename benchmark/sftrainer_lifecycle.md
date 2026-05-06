# SFTTrainer lifecycle: from `from_pretrained` to first training step

A walkthrough of what happens between `python sft.py` and the first `loss.backward()`, with explicit transformers / accelerate / DeepSpeed-or-FSDP boundaries. Two scenarios are covered separately because their wiring diverges:

1. **DeepSpeed + Expert Parallelism**
2. **FSDP + Expert Parallelism**

If you understand single-GPU `Trainer.train()` already, skim section 0 for orientation and dive into 1 or 2.

---

## 0. The mental model

Three repos collaborate to start training. Each owns one well-defined slice of the work:

| Repo                        | Owns                                                                                                                                                                                                                                                             |
| --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **transformers**            | The model, weight loading, EP-aware sharding via `tp_plan`, the EP device mesh stored as `model._device_mesh`. The `Trainer` class that orchestrates everything.                                                                                                 |
| **accelerate**              | The `Accelerator` object, the `state.device_mesh` (FSDP path) and `state.ds_device_mesh` (DS path), the `prepare()` method that wraps model+optimizer+dataloader in a backend-specific way, and the `prepare_data_loader` that strides the dataset across ranks. |
| **DeepSpeed** _or_ **FSDP** | The actual model wrapping: ZeRO partitioning + grad reduction (DS) or `fully_shard` + per-mesh sharding (FSDP). Invoked through accelerate.                                                                                                                      |

Two device meshes are created during init. They are not the same object:

- `model._device_mesh` (transformers): the EP mesh. Built in `tensor_parallel.initialize_tensor_parallelism`. Used for the EP `all_reduce` inside MoE forward. Has dim name `"tp"`.
- `accelerator.state.device_mesh` / `state.ds_device_mesh` (accelerate): the mesh accelerate uses for **everything else** — dataloader sharding, FSDP wrapping, TP setup. Built lazily from `ParallelismConfig.build_device_mesh()` based on env vars set by `accelerate launch`.

The fact that these are _separate meshes that don't talk to each other_ is the source of the EP+dataloader bug discussed in §1 below.

There is also an asymmetric weight-loading pattern that distinguishes the two backends:

- **DS path**: every rank loads its own weights from disk (mmap-backed safetensors slices) and keeps them. DS partitions inside `deepspeed.initialize` later, but the per-rank loaded values are read in place — no meta-move, no broadcast.
- **FSDP path**: every rank still loads from disk (same mmap path), but accelerate then _throws away_ the per-rank values. `fsdp2_prepare_model` moves the model to `meta`, calls `fully_shard`, and `fsdp2_load_full_state_dict` broadcasts rank-0's values to every other rank's DTensor shards. Non-rank-0 ranks' loaded weights are written and never read. See `benchmark/fsdp2_loading_walkthrough.md` for the full RSS + page-cache breakdown of this dance and the `_move_missing_keys_from_meta_to_device` Path-B detour. The EP + `cpu_ram_efficient_loading` interaction this creates is tracked as C2 in `upstream_todo.md` (rank-0 broadcast clobbers each rank's per-rank EP-shard).

---

## 1. DeepSpeed + Expert Parallelism

Configuration assumed: 16 GPUs (2 nodes × 8), Qwen3-30B-A3B (128 experts, top-k=8), `enable_expert_parallel=True, expert_parallel_size=8`, DeepSpeed ZeRO-2.

### 1.1 Entry: `sft.py`

`trl/scripts/sft.py:202` instantiates the trainer. The model is constructed _before_ `SFTTrainer` is created, by `create_model_from_path`, which is just a thin wrapper around `from_pretrained`. SFTTrainer's pre-`super().__init__()` block in `sft_trainer.py` builds an EP-aware device mesh and passes it through:

```python
# trl/trainer/sft_trainer.py (~line 950)
if args.enable_expert_parallel:
    model_init_kwargs["distributed_config"] = DistributedConfig(enable_expert_parallel=True)
    world_size = dist.get_world_size()                     # 16
    ep_size = args.expert_parallel_size or world_size      # 8
    if ep_size < world_size:
        dp_size = world_size // ep_size                    # 2
        model_init_kwargs["device_mesh"] = dist.init_device_mesh(
            "cuda", (dp_size, ep_size), mesh_dim_names=("dp", "tp")
        )
    # ...
model = create_model_from_path(model, **model_init_kwargs)
```

Two things to notice:

- `enable_expert_parallel` is passed on the model via `DistributedConfig` (a transformers concept). The `Trainer` later checks `model.has_ep` to branch behavior.
- Although the parallelism is _expert parallel_, the dim is named `"tp"`. transformers and accelerate both use `"tp"` as the data-replicate dim name; renaming would just create more wiring.

### 1.2 `from_pretrained` and EP sharding

Inside `transformers/modeling_utils.py:from_pretrained` (~line 4067), with `device_mesh` passed in:

```python
device_map, device_mesh, tp_size = initialize_tensor_parallelism(
    tp_plan, tp_size=tp_size, device_mesh=device_mesh, device_map=device_map
)
```

`initialize_tensor_parallelism` (in `transformers/integrations/tensor_parallel.py`) extracts the 1D `"tp"` sub-mesh from the 2D parent and returns it. By the time `from_pretrained` returns, the 2D `("dp", "tp")` mesh is reduced to a 1D `("tp",)` mesh of size 8 — the **EP mesh** that powers the experts all-reduce.

Then at line 4218:

```python
if _torch_distributed_available and device_mesh is not None:
    model = distribute_model(model, tp_plan, distributed_config, device_mesh, tp_size)
```

`distribute_model` (`tensor_parallel.py:1618`) does three things:

1. Stores the EP mesh on the model: `model._device_mesh = device_mesh`.
2. Sets `model._tp_size = tp_size` (= 8 here).
3. Walks `model.tp_plan` (the EP version of TP plan) and adds per-module hooks: `RouterParallel`, `MoeTensorParallelExperts`, `GroupedGemmParallel`, etc. These hooks rewrite forward inputs/outputs to make EP work — most importantly inserting the experts all-reduce after the local-expert computation.

When `GroupedGemmParallel.post_shard_wrap` runs over an expert weight, it detects DeepSpeed via `_hf_deepspeed_config_weak_ref` and does:

```python
# transformers/integrations/tensor_parallel.py:1163-1165
param.allreduce = False
param.group_name = f"ep_size_{self.device_mesh.size()}"
return param  # plain nn.Parameter, NOT a DTensor
```

These are markers DeepSpeed's `is_moe_param(p)` consumes later. Crucially: under DeepSpeed, the EP-sharded experts are _plain tensors_ (not DTensors), because DS's stage_1_and_2 partitioner can't deal with DTensors. Compare with the FSDP path where they would be wrapped as DTensor on the EP mesh.

After `from_pretrained` returns:

- `model._device_mesh`: 1D `("tp",) = (8,)` EP mesh
- `model.has_ep`: `True`
- Experts: per-rank-local slice (16 of 128 experts on each rank), tagged with `allreduce=False, group_name="ep_size_8"`
- Dense weights: full replica on every rank (DS will partition them later via ZeRO)

A subtle point about where these weights actually live on the box: `_load_pretrained_model` always starts the same way regardless of backend — `safe_open(file).get_slice(k)` returns a `PySafeSlice`, and `_materialize_copy` calls `tensor[...]` to produce a `torch.Tensor` whose storage is a view into the mmap'd file region. **What happens next depends on `device_map`** (set by `initialize_tensor_parallelism`):

```python
# transformers/integrations/accelerate.py::check_and_set_device_map
# initialize_tensor_parallelism returned device_map = torch.device("cuda", local_rank)
# converted to:
device_map = {"": torch.device("cuda", local_rank)}

# core_model_loading.py:1366-1377
if device_mesh and tp_plan:
    # EP-sharded experts: shard with TP-style op, COPY directly to GPU
    future_or_tensor = spawn_tp_materialize(thread_pool, tensor, ..., device_map[""], dtype)
else:
    # Dense weights: COPY to whatever device_map resolves to (= cuda:LOCAL_RANK here)
    param_device = get_device(device_map, key, valid_torch_device=True)
    future_or_tensor = spawn_materialize(thread_pool, tensor, param_device, dtype)
```

`_materialize_copy` (`core_model_loading.py:936`) then does `tensor = tensor[...]` (free, mmap-backed CPU view) followed by `tensor = tensor.to(device="cuda:N", dtype=bf16)` (real bytes copy from page cache → GPU HBM). So under **DS+EP, every param ends up on GPU directly** — the CPU mmap is just an intermediate the kernel collects after the copy.

Verified empirically via `benchmark/debug_ep_loading.py ds --ep-size 4` on the tiny MoE: each call to `_materialize_copy` prints the `device` arg per rank — every param-load lands on `cuda:LOCAL_RANK`, never on cpu:

```text
# rank 0..3 each printing the cuda device passed into _materialize_copy
cuda:0
cuda:1
cuda:2
cuda:3
Loading weights:   4%|███| 1/25 ...
cuda:0
cuda:1
cuda:2
cuda:3
...
```

**Under EP, `cpu_ram_efficient_loading` does NOT change the materialize path** — every rank still loads to `cuda:LOCAL_RANK`. This is the key insight that took a debug session to lock down. The flag changes things *downstream* of `_load_pretrained_model`, not inside it. Tracing what's actually invariant vs gated:

- `get_init_context` (`modeling_utils.py:3654-3696`): the `torch.device("meta")` init context is added for **all non-DS-Z3 paths**, including DS+EP and FSDP+EP, regardless of `cpu_ram_efficient_loading`. So `cls(config)` runs on meta in every EP case.
- `_materialize_copy` (`core_model_loading.py:936`): uses `device_map[""]`. With EP, `initialize_tensor_parallelism` set `device_map = cuda:LOCAL_RANK`. So `_materialize_copy` always copies bytes from the mmap to `cuda:LOCAL_RANK` on every rank, regardless of `cpu_ram_efficient_loading`. **Verified empirically: `debug_ep_loading.py {ds,fsdp} --ep-size 4` prints `cuda:0..cuda:3` per rank in both modes; setting `--fsdp-cpu-ram-efficient` produces the same prints.**

What `cpu_ram_efficient_loading=True` *does* change (all gated on `is_fsdp_enabled()`, which requires both `ACCELERATE_USE_FSDP=True` AND `FSDP_CPU_RAM_EFFICIENT_LOADING=True`):

1. **Path B in `_move_missing_keys_from_meta_to_device`** (`modeling_utils.py:4622`) fires on non-rank-0 ranks. After materialize already put params on `cuda:LOCAL_RANK`, this loop walks every named parameter and replaces it with `torch.zeros_like(p, device="cpu")`, dropping the cuda values. So non-rank-0 ranks end up with **CPU zero placeholders** for every param after `_finalize_model_loading`. (Rank 0 stays on cuda — only Path A runs there, and `missing_keys` is empty for a complete checkpoint.)
2. Rank-0-only short-circuits in `_initialize_missing_keys` (`modeling_utils.py:4690`) skip re-init on non-rank-0.
3. Later, `accelerate.fsdp2_prepare_model` does `model.to("meta")` + `fully_shard` + `fsdp2_load_full_state_dict` — rank 0 broadcasts its (still-cuda) params to every other rank's DTensor shards. The non-rank-0 cpu zero placeholders from step 1 are never read; they only existed to give `fully_shard` shape/dtype metadata.

So `cpu_ram_efficient_loading` is misnamed for the EP case: the "savings" don't come from the loading stage (every rank still materializes its full per-rank copy on GPU). They come from the broadcast stage — non-rank-0 ranks would ordinarily duplicate the dense replica through `accelerate.prepare`'s state_dict copy. With the flag, the cpu zero placeholders are tiny (RSS overhead), so the per-rank GPU peak during prepare drops from "load + duplicate-on-CPU" to just "load."

**Important caveat for EP**: the flag's broadcast step (item 3) is what clobbers the EP partition. See §2.4(d) for the C2 issue — Patch 1 (rank-0 disk-load gate) AND Patch 2 (EP-aware broadcast) both needed. **Today, debugging the flag's effects under EP requires inspecting non-rank-0 state**, since rank-0 sees no difference. The debug script's default rank-0-only print won't surface anything.

After Step 2 (DS+EP), per-rank state:
- Experts: per-rank-local 16/128 slice, on GPU, plain `nn.Parameter` tagged with `allreduce=False, group_name="ep_size_8"`.
- Dense weights: full replica, on GPU, plain `nn.Parameter`.
- CPU RSS ≈ 0.5 GB (just Python overhead). Per-rank GPU memory ≈ ~5 GB experts + ~5 GB dense = ~10 GB before ZeRO.

DS does NOT move the model back to meta or broadcast — the per-rank values are read in place when `deepspeed.initialize` later partitions them under ZeRO. The dense replica gets sharded across the 16 ranks by ZeRO-2/3; the EP experts stay where they are (DS sees `allreduce=False` and routes them through the expert process group).

### 1.3 `super().__init__()` and `create_accelerator_and_postprocess`

Now `SFTTrainer.__init__` calls `super().__init__()`, which dives into `transformers/trainer.py:Trainer.__init__`. Very early, line 410:

```python
self.create_accelerator_and_postprocess()
```

This is the central wiring method. Walk through it (`trainer.py:759`):

**(a) Build accelerate args.** `_build_accelerator_args` packages dataloader config, FSDP plugin, gradient accumulation plugin, etc. Reads from `self.args`.

**(b) Construct the Accelerator.** Line 817:

```python
self.accelerator = Accelerator(**args)
```

`Accelerator.__init__` (`accelerate/accelerator.py:300`) reads env vars set by `accelerate launch`:

- `ACCELERATE_USE_DEEPSPEED=true` → `state.deepspeed_plugin` is set.
- `ACCELERATE_USE_PARALLELISM_CONFIG=true` → builds a `ParallelismConfig` from `PARALLELISM_CONFIG_TP_SIZE` etc. For our run, the yaml emits NO `parallelism_config` block (`tp=cp=pp=1` in the bookkeeping config), so `ACCELERATE_USE_PARALLELISM_CONFIG` is unset → `parallelism_config` is `None` → `state.device_mesh` is `None`.

**(c) Detect backend.** Lines 827-828:

```python
self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None  # True
self.is_fsdp_enabled       = getattr(self.accelerator.state, "fsdp_plugin", None) is not None       # False
```

**(d) DS-specific EP setup.** Lines 835-841:

```python
if self.is_deepspeed_enabled and getattr(self.model, "has_ep", False):
    from deepspeed.utils import groups as _ds_groups
    ep_size = self.model.tp_size                # 8
    group_name = f"ep_size_{ep_size}"
    if group_name not in _ds_groups._get_expert_parallel_group_dict():
        _ds_groups._create_expert_and_data_parallel(ep_size)
```

This pre-creates the named DeepSpeed process groups (the ones DS uses for `expert_data_parallel_group` reductions and the `_broadcast_model` call inside `deepspeed.initialize`). DS expects them to exist _before_ it constructs the engine.

**(e) EP DataLoader injection (the new piece, see §1.6).** Right after the DS group creation:

```python
if (
    self.is_deepspeed_enabled
    and getattr(self.model, "has_ep", False)
    and getattr(self.accelerator.state, "ds_device_mesh", None) is None
):
    ep_mesh = getattr(self.model, "_device_mesh", None)
    if ep_mesh is not None and "tp" in (ep_mesh.mesh_dim_names or ()):
        self.accelerator.state.ds_device_mesh = ep_mesh
```

To unpack what's going on here, three things need clarifying.

**What `ds_device_mesh` actually is.** It's a single attribute on `AcceleratorState` that exists specifically for the DEEPSPEED branch of `prepare_data_loader`. Its only consumer in all of accelerate is `Accelerator._prepare_device_mesh` (`accelerator.py:2604`):

```python
def _prepare_device_mesh(self):
    if self.distributed_type == DistributedType.DEEPSPEED and hasattr(self.state, "ds_device_mesh"):
        return self.state.ds_device_mesh    # ← used by prepare_data_loader
    else:
        return self.torch_device_mesh        # = state.device_mesh (FSDP path)
```

So `ds_device_mesh` is **purely a "tell prepare_data_loader how to stride data" channel**. It is not used to shard params, not used in DS engine init, not used for any collective. It's a sub-mesh accelerate inspects when computing `process_index = global_rank // tp_size, num_processes = world_size // tp_size` for the dataloader.

The reason it's a separate attribute from `state.device_mesh` is historical: accelerate added DS-AutoTP support before `ParallelismConfig` existed, so DS-TP got its own slot. The two slots feed the same downstream consumer (`prepare_data_loader`) but are populated by different code paths.

**Why DS needs a "tp"-named mesh at all.** ZeRO partitions params across all data-parallel ranks. With pure DS (no TP, no EP), every rank is a DP rank, so the dataloader correctly strides 1-shard-per-rank. But under TP or EP, some ranks share data: the 8 ranks of a TP group all run the same forward on the same tokens; the 8 ranks of an EP group all run the same forward through the dense layers and only diverge inside the experts all-reduce. For both TP and EP, the dataloader must give the same micro-batch to all ranks within the dim. Accelerate's existing dataloader-sharding code already handles this — when it sees `"tp"` in `mesh_dim_names`, it divides `process_index` by `tp_size` so all ranks in a TP group get the same shard index. It's a generic data-replicate mechanism.

**Why we have to set it ourselves under EP.** Accelerate populates `ds_device_mesh` on its own only for **DS-AutoTP**: when the DS config explicitly contains `tensor_parallel: { autotp_size: N }`. Our config does not — we use transformers' EP plan, not DS-AutoTP. `_prepare_deepspeed` (`accelerator.py:2160`) reads `autotp_size`, sees 0, skips the mesh creation. So without intervention, `ds_device_mesh` is `None`, accelerate's `prepare_data_loader` falls back to "every rank gets a unique shard," and the 8 ranks of an EP group end up with 8 different micro-batches → mismatched shapes inside the experts all-reduce → silent NCCL hang. We populate the slot ourselves with the EP mesh (which already has `"tp"` in its dim names — set by `initialize_tensor_parallelism`) so accelerate's existing TP-replication path fires. **No new mesh is created** — we reuse the one transformers built. (Trying to call `init_device_mesh` here fails because the world process group has already been split by transformers' EP setup — see §1.6 for the failure mode.)

After `create_accelerator_and_postprocess` returns:

- `self.accelerator` is alive but no model/optimizer is wrapped yet
- `self.accelerator.state.ds_device_mesh` is the 1D `("tp",) = (8,)` EP mesh
- `self.is_deepspeed_enabled` is `True`

### 1.4 The rest of `Trainer.__init__`

Continues setting up state: device placement (skipped for DS), data collator, optimizer-related attributes, etc. Nothing EP-specific until `train()` is called. Eventually `super().__init__()` returns to `SFTTrainer`, which finishes its own setup (loss func, MFU tracking, model tags) and returns control to the caller.

Then `sft.py:213` calls `trainer.train()`.

### 1.5 `train()` → `_prepare_for_training` → `accelerator.prepare`

`Trainer.train()` (line 1467) eventually reaches `_prepare_for_training` (line 1620). Two things matter:

```python
if self.is_deepspeed_enabled:
    self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)
# ...
model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
```

`accelerator.prepare` (`accelerator.py:1413`) dispatches based on backend. For DEEPSPEED (line 1547):

```python
if self.distributed_type == DistributedType.DEEPSPEED:
    result = self._prepare_deepspeed(*args)
```

`_prepare_deepspeed` (line 2129):

- Looks up `tensor_parallel.autotp_size` from the DeepSpeed config — 0 for us, so the standalone `init_device_mesh` block at line 2160 does NOT fire. To clarify what that block normally does and why we sidestep it:

  ```python
  # accelerate/accelerator.py:2148-2160 — DeepSpeed AutoTP path
  tp_size = deepspeed_plugin.deepspeed_config.get("tensor_parallel", {}).get("autotp_size", 0)
  if tp_size > 1:
      ...
      self.state.ds_device_mesh = init_device_mesh(self.device.type, (tp_size,), mesh_dim_names=("tp",))
  ```

  This fires **only when the DS config itself requests TP via `autotp_size`** (a DeepSpeed feature). When it fires, DS engine internally rewrites the model's linear layers to do its own tensor-parallel forward inside `deepspeed.initialize`, and accelerate creates a matching `ds_device_mesh` so the dataloader replicates data across the TP group. **DS-AutoTP and transformers-TP are two separate TP implementations** — they don't compose, you pick one.

  Our setup uses transformers' tp_plan (with EP enabled), not DS-AutoTP. So the DS config has no `tensor_parallel` block, `autotp_size=0`, and accelerate's mesh-creation step is skipped. `ds_device_mesh` would be `None` if we hadn't already set it in step (e) above.

  **Subtlety: transformers-TP-without-EP under DS has the same gap.** If you load a model with a plain `tp_plan` (no `enable_expert_parallel`) and run it under DeepSpeed, you also get `device_mesh = (tp_size,)` from `initialize_tensor_parallelism`, no `autotp_size` in the DS config, and `ds_device_mesh = None` → same dataloader-sharding bug. The fix in step (e) is gated on `model.has_ep` today; it should arguably be gated on "any transformers `tp_plan` is active" so it benefits TP-with-DS workflows too. We haven't generalized it because TP-with-DS isn't a recipe anyone runs in this repo (FSDP is the standard partner for TP), but the asymmetry is worth noting.
- Calls `deepspeed.initialize(...)` which returns the DS engine. Inside DS, `is_moe_param(p)` finds our tagged params, splits them into a separate optimizer group, and routes their grad reduce through `expert_dp_process_group`.

When `_prepare_deepspeed` reaches the dataloader (passed in args), it invokes `prepare_data_loader` — which is where the EP-aware sharding finally happens (see §1.6).

After `prepare()` returns:

- `model` is now `DeepSpeedEngine(transformers_model)`
- `self.optimizer` is a `DeepSpeedZeroOptimizer`
- The dataloader yields different micro-batches per _EP group_, identical micro-batches _within_ an EP group

### 1.6 Where the EP DataLoader fix actually fires

`accelerator.prepare_data_loader` (the public method, `accelerator.py:2669`) is called from `_prepare_one` for each `DataLoader` arg. It internally calls `accelerate.data_loader.prepare_data_loader` (the function), passing `torch_device_mesh = self._prepare_device_mesh()`:

```python
# accelerator.py:2604-2612
def _prepare_device_mesh(self):
    if self.distributed_type == DistributedType.DEEPSPEED and hasattr(self.state, "ds_device_mesh"):
        return self.state.ds_device_mesh        # <-- our injected EP mesh
    else:
        return self.torch_device_mesh           # = state.device_mesh (None for our run)
```

The function then inspects the mesh (`data_loader.py:1119-1130`):

```python
if torch_device_mesh:
    if state.distributed_type == DistributedType.DEEPSPEED:
        submesh_tp_size = 1
        if "tp" in torch_device_mesh.mesh_dim_names:
            submesh_tp_size = torch_device_mesh["tp"].size()    # 8
        process_index = process_index // submesh_tp_size        # rank 0..7 → 0; rank 8..15 → 1
        num_processes  = num_processes // submesh_tp_size       # 16 → 2
```

That's it. `BatchSamplerShard` strides the dataset with `num_processes=2, process_index=k`, so the 8 ranks of EP group A all read the same indices, and the 8 ranks of EP group B read a different (but internally consistent) set.

**Why we don't `init_device_mesh` ourselves:** an earlier attempt called

```python
self.accelerator.state.ds_device_mesh = dist.init_device_mesh(
    "cuda", (ep_size,), mesh_dim_names=("tp",)
)
```

This crashed with `AttributeError: 'NoneType' object has no attribute 'group_name'`. PyTorch's `_init_one_process_group` (`device_mesh.py:444`) calls `split_group(parent_pg=default_group, …)` to create the sub-group, and it returns `None` because the world PG was already split when transformers built the EP mesh in §1.2. The fix is to _reuse_ the mesh transformers built rather than build a new one — same dim name, same ranks, no new NCCL splits.

### 1.7 First training step

`_inner_training_loop` (line 1549) iterates:

```python
tr_loss_step = self.training_step(model, inputs, num_items_in_batch)
```

The model forward enters `RouterParallel._prepare_output_fn` and `MoeTensorParallelExperts._prepare_input_fn` hooks (added by `distribute_model`). The local-expert computation runs on each rank's 16 experts; an `all_reduce` then sums the partial outputs across the 8 ranks of the EP group. This `all_reduce` only succeeds if the input shapes agree — which is true precisely because the dataloader gave all 8 ranks the same micro-batch.

### 1.8 Boundary recap (DS + EP)

| Concern                                              | Owned by               | Where                                                                                         |
| ---------------------------------------------------- | ---------------------- | --------------------------------------------------------------------------------------------- |
| Build EP mesh, shard experts                         | transformers           | `from_pretrained` → `initialize_tensor_parallelism` + `distribute_model`                      |
| Tag MoE params for DS                                | transformers           | `GroupedGemmParallel.post_shard_wrap` — `param.allreduce=False, param.group_name="ep_size_N"` |
| Pre-create DS expert process groups                  | transformers Trainer   | `create_accelerator_and_postprocess` (line 835)                                               |
| Inject EP mesh into accelerator state for dataloader | transformers Trainer   | `create_accelerator_and_postprocess` (line 843, the new block)                                |
| Build accelerator state                              | accelerate             | `Accelerator()` constructor                                                                   |
| Wrap model+optimizer in DS engine                    | accelerate → DeepSpeed | `_prepare_deepspeed` → `deepspeed.initialize`                                                 |
| Stride dataset across EP groups                      | accelerate             | `prepare_data_loader` (uses `state.ds_device_mesh`)                                           |
| Routes MoE grads through `expert_dp_process_group`   | DeepSpeed              | `stage_1_and_2.py`                                                                            |

---

## 2. FSDP + Expert Parallelism

Configuration assumed: 16 GPUs, Qwen3-30B-A3B, `enable_expert_parallel=True, expert_parallel_size=8`, FSDP2.

The early phases (entry, model loading, `from_pretrained`, EP sharding) are _almost_ identical to the DS case. The divergence starts at `GroupedGemmParallel.post_shard_wrap` and amplifies through the rest of init.

### 2.1 Where it diverges from DS

#### Inside `from_pretrained` — experts as DTensors, not plain tensors

Same `initialize_tensor_parallelism` and `distribute_model` as before, but in `post_shard_wrap` (`tensor_parallel.py:1158-1168`) the DS detection branch does NOT fire (no DS config active). Instead:

```python
dt = DTensor.from_local(param.data, self.device_mesh, [Shard(0)], run_check=False)
return nn.Parameter(dt, requires_grad=param.requires_grad)
```

Each rank's expert slice is wrapped as a `DTensor` placed on the EP mesh. This matters because once FSDP wraps the model, it will see a mix of plain tensors (dense weights) and DTensors (expert weights). FSDP cannot compose with DTensors that live on a different mesh than its own; the `ignored_modules` mechanism (set up in §2.3) is what lets these coexist.

After `from_pretrained` returns:

- `model._device_mesh`: 1D `("tp",) = (8,)` EP mesh — same as DS path
- Experts: per-rank-local DTensor on the EP mesh
- Dense weights: full plain tensors replicated on every rank

#### `accelerate launch` configuration

The yaml template (`benchmark/templates/accelerate/fsdp2.yaml.j2`) emits:

```yaml
distributed_type: FSDP
fsdp_config:
    fsdp_version: 2
    fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
    fsdp_cpu_ram_efficient_loading: ...
```

If any of `tp/cp/pp > 1`, it also emits a `parallelism_config:` block. For pure EP runs (the case here), `tp=cp=pp=1`, so the block is absent — same as the DS case. accelerate's `state.parallelism_config` is `None`, `state.device_mesh` is `None`.

### 2.2 `Accelerator()` construction

Line 827-828 of `Trainer.create_accelerator_and_postprocess`:

```python
self.is_deepspeed_enabled = False
self.is_fsdp_enabled       = True       # state.fsdp_plugin is set
```

The DS-specific block (line 835) is skipped (`is_deepspeed_enabled` is False).

The new EP-DataLoader injection block (line 843) is also skipped — it's gated on `is_deepspeed_enabled`. **This is the gap.** FSDP doesn't have an equivalent of `state.ds_device_mesh`; its dataloader sharding goes through `state.device_mesh`, which is tied to `parallelism_config`. To inject a mesh there cleanly we'd need accelerate to also have a `parallelism_config` populated, and FSDP wrap reads `parallelism_config.fsdp_dim_names` later at `fsdp_utils.py:651` to slice the mesh for sharding. Without that wiring, the dataloader sees `state.device_mesh = None` and shards naively across all 16 ranks. Same bug as before the DS fix — but for FSDP.

This is the reason `Trainer._get_dataloader` keeps a fallback `prepare_data_loader` patch active for FSDP+EP — see §2.6 below.

### 2.3 FSDP+EP setup in `create_accelerator_and_postprocess`

After the DS block (skipped) and the EP-mesh-injection block (skipped), Trainer hits the FSDP block (line 844-851):

```python
if self.is_fsdp_enabled:
    fsdp_plugin = self.accelerator.state.fsdp_plugin
    ep_param_names = get_ep_sharded_param_names(self.model)
    if ep_param_names:
        module_names = list({n.rsplit(".", 1)[0] for n in ep_param_names})
        fsdp_plugin.ignored_modules = [self.model.get_submodule(n) for n in module_names]
    # ...
```

This is the FSDP-EP coexistence trick: tell FSDP _not_ to shard the modules whose params are EP DTensors. Otherwise FSDP would try to wrap them in addition to their EP sharding, and the resulting double-DTensor stack would fail in NCCL collectives.

After `create_accelerator_and_postprocess` returns:

- `self.accelerator.state.fsdp_plugin.ignored_modules` lists the MoE expert modules
- `self.accelerator.state.device_mesh` is `None` (no parallelism_config)

### 2.4 Weight loading and the rank-0 broadcast

Before getting to `accelerator.prepare`, recall the FSDP-specific loading dance — this is the boundary where transformers hands off to accelerate, and it has consequences for EP.

**(a) Every rank loads from disk.** Same mmap-backed path as the DS case: `_load_pretrained_model` calls `convert_and_load_state_dict_in_model` which materializes each tensor via `safe_open(...).get_slice(k)[...]`. The "load" is RSS-cheap because the bytes live in the kernel page cache, not process heap.

**(b) `_move_missing_keys_from_meta_to_device` Path B fires on non-rank-0.** Despite the misleading name, this function has a separate code path for FSDP non-rank-0 ranks (`modeling_utils.py:4622`):

```python
if is_fsdp_enabled() and not is_local_dist_rank_0() and not is_quantized:
    for key, param in self.named_parameters():
        value = torch.zeros_like(param, device="cpu")  # placeholder, overwritten by broadcast
        _load_parameter_into_model(self, key, value)
    for key, buffer in self.named_buffers():
        value = torch.zeros_like(buffer, device="cpu")
        _load_parameter_into_model(self, key, value)
    return
```

It walks **every** parameter (not just `missing_keys`) and replaces the just-loaded mmap-backed tensor with a fresh CPU placeholder. The reasoning: rank 0 will broadcast the canonical values during FSDP wrap, so non-rank-0 ranks don't need their own copies — but they need _some_ tensor of the right shape/dtype for `fully_shard` to derive metadata from. (See `benchmark/fsdp2_loading_walkthrough.md` §3 for the long story; the short version is that this loop is the source of CPU OOM under PR #45050 and the reason the proposed fix removes the parameter loop entirely.)

**(c) `fsdp2_prepare_model` moves to meta and broadcasts from rank 0.** Inside `accelerate/utils/fsdp_utils.py:621`, with `cpu_ram_efficient_loading=True`:

```python
original_sd = model.state_dict()                # snapshot before meta move
model = model.to(torch.device("meta"))          # all params/buffers → meta
fully_shard(module, **fsdp2_kwargs)             # each param becomes a DTensor on meta
fsdp2_load_full_state_dict(accelerator, model, original_sd, ...)
```

`fsdp2_load_full_state_dict` (`fsdp_utils.py:467`) is asymmetric:

- **rank 0** reads each param from `original_sd`, casts to GPU, broadcasts, then `distribute_tensor` shards it onto the FSDP mesh.
- **non-rank-0** never reads `original_sd`. It allocates a `torch.empty(..., device="cuda")`, receives the broadcast, and `distribute_tensor`s it.

So under FSDP, the per-rank values that transformers loaded in Step (a) **are written but never read on non-rank-0**. The canonical values come from rank-0's broadcast.

**(d) Why this collides with EP — two independent bugs, not one.** This is C2 in `upstream_todo.md`. The collision has two layers, either of which alone would already corrupt the EP partition; together they produce silent broken outputs (pre-2026-04-30) or a 10-minute NCCL SCATTER hang on the first param-load collective (post-2026-04-30; observed in job `22095410`, `report.md:3661`). The workaround today is **never combine `enable_expert_parallel=True` with `cpu_ram_efficient_loading=True`**.

_Layer 1 — the flag is a no-op at the loading stage under EP._ `cpu_ram_efficient_loading=True` is supposed to make only rank-0 read from disk. But the rank-0-only short-circuits in `modeling_utils.py:4654` and `:4690` only gate post-loading helpers (`_move_missing_keys_from_meta_to_cpu_or_disk`, `_initialize_missing_keys`). The actual disk loading in `core_model_loading.py:convert_and_load_state_dict_in_model` has **no rank-0 short-circuit** — every rank calls `spawn_materialize(tensor, device=device_map[""])`. Under EP, `initialize_tensor_parallelism` set `device_map[""] = cuda:LOCAL_RANK`, so every rank materializes its own copy of the dense replica on its own GPU. Verified empirically on 30B-A3B EP=4 (2026-04-29): 2.87 GiB dense replicated × 16 ranks regardless of whether the flag is set.

_Layer 2 — the broadcast clobbers the EP partition._ Step (c) (`fsdp2_prepare_model`) does `model.to("meta")` then `fsdp2_load_full_state_dict`, which broadcasts rank-0's pre-meta snapshot to every rank. For dense weights this is correctness-preserving — every rank had the same dense replica anyway. For **EP experts**, each rank held a *distinct* slice (rank 0 = experts 0–15, rank 1 = 16–31, …); the broadcast overwrites each rank's local shard with rank-0's experts 0–15. After this, every rank holds the same experts. MoE routing is broken silently — forward looks plausible (the all-reduce sums identical contributions × N), gradients are wrong, training diverges.

Same family as DS-Z3 + EP (D-blocker in upstream_todo): both ZeRO-3 and FSDP+`cpu_ram_efficient_loading` assume **"DP-replica sync — every rank holds the same logical data, just sharded differently"**. Normal DP/FSDP satisfies it; EP doesn't.

_Why it requires two patches:_
- **Patch 1 (transformers)**: rank-0 gate in `convert_and_load_state_dict_in_model`. When `is_fsdp_enabled() and rank > 0`, skip `spawn_materialize` for non-plan params; leave them on meta. Plan params (EP-sharded experts) still load per-rank because they're rank-distinct.
- **Patch 2 (accelerate)**: EP-aware broadcast in `fsdp2_prepare_model`. Capture EP params as `ignored_params` *before* the meta move, skip them in `fsdp2_load_full_state_dict`, restore each rank's snapshot after `fully_shard`.
- Plus an immediate `ValueError` guard in `from_pretrained` when both flags are set, until Patches 1+2 land.

Without both patches: Patch 1 alone cuts duplicate disk reads but Layer 2's broadcast still clobbers the EP partition. Patch 2 alone preserves the EP partition but every rank still loads its own copy from disk (Layer 1's no-op).

The clean transformers/accelerate boundary under FSDP is: **transformers loads, accelerate shards-and-distributes**. EP breaks that boundary because the "load" already produced a per-rank partition that the "distribute" step assumes is uniform.

### 2.5 `train()` → `accelerator.prepare`

The `prepare` dispatch in `accelerator.py:1551` reaches:

```python
elif self.is_fsdp2:
    result = self._prepare_fsdp2(*args)
```

`_prepare_fsdp2` calls `fsdp2_prepare_model` (`accelerate/utils/fsdp_utils.py`). The relevant passage (line 644-651):

```python
mesh = getattr(accelerator, "torch_device_mesh", None)         # = state.device_mesh = None
fsdp2_kwargs = {
    "mesh": mesh[tuple(accelerator.parallelism_config.fsdp_dim_names)] if mesh is not None else None,
    "ignored_params": ...,                                       # from ignored_modules above
    # ...
}
# then: fully_shard(module, **fsdp2_kwargs)
```

With `mesh=None`, FSDP shards across the full default process group (all 16 ranks). Dense weights end up sharded 1/16 per rank; expert DTensors are skipped via `ignored_params`.

In parallel, the dataloader is processed by `_prepare_one` → `prepare_data_loader`. The mesh-inspection block sees `torch_device_mesh = None` and falls through:

```python
# data_loader.py:1119
if torch_device_mesh:
    # ...
    process_index = process_index // (submesh_tp_size * submesh_cp_size)
    num_processes  = submesh_fsdp_size * submesh_dp_size
```

The `if` fails. accelerate falls back to default `state.num_processes = 16, state.process_index = global_rank`. **The dataloader gives every rank a unique micro-batch.** This is the bug the FSDP-fallback override in `Trainer._get_dataloader` exists to patch.

### 2.6 The `_get_dataloader` override (FSDP fallback) — lives in `transformers/Trainer`

When `model.has_ep and is_fsdp_enabled`, `Trainer._get_dataloader` monkey-patches `accelerator.prepare_data_loader` for the one call that wraps the dataset, then restores it:

```python
# transformers/src/transformers/trainer.py::_get_dataloader (excerpt)
ep_patch_active = (
    getattr(self.model, "has_ep", False)
    and self.is_fsdp_enabled
    and getattr(self.model, "_tp_size", 1) > 1
)
if ep_patch_active:
    ep_size = self.model._tp_size
    eff_num = self.accelerator.num_processes // ep_size      # 2
    eff_idx = self.accelerator.process_index // ep_size      # 0 for ranks 0-7, 1 for ranks 8-15
    orig_prepare_dl = self.accelerator.prepare_data_loader

    def _patched(dataloader, device_placement=None, slice_fn_for_dispatch=None):
        return _prep_dl(
            dataloader, self.accelerator.device,
            num_processes=eff_num,
            process_index=eff_idx,
            # ... + every other field copied from the accelerator config
            torch_device_mesh=None,                          # critical: bypass mesh inspection
        )

    self.accelerator.prepare_data_loader = _patched

try:
    dataloader = self.accelerator.prepare(DataLoader(dataset, **dataloader_params))
finally:
    if ep_patch_active:
        self.accelerator.prepare_data_loader = orig_prepare_dl
```

Why this lives in transformers, not TRL: same reasoning as the DS injection. `model.has_ep`, `model._tp_size`, and the model-level Trainer hooks all belong to transformers. TRL's `SFTTrainer._get_dataloader` is now a thin override that only adds the `[LOAD-T]` instrumentation print; it delegates to `super()._get_dataloader`, which is where the FSDP fallback lives.

Three things make this safe:

1. The patch is in effect for exactly one super call. Subsequent accelerate ops are unaffected.
2. `torch_device_mesh=None` skips the mesh inspection at line 1119 — accelerate doesn't try to derive `num_processes` from a mesh, it uses our explicit values.
3. All other dataloader-prep concerns (rng seeding, dispatch, even_batches, stateful) are passed through unchanged.

The override does the exact same math the DS path does _natively_ via `state.ds_device_mesh`. It exists only because FSDP's dataloader path won't read a mesh unless `parallelism_config` is also wired up — bigger surgery than we want to do under EP today.

### 2.7 Boundary recap (FSDP + EP)

| Concern                                                  | Owned by                            | Where                                                                         |
| -------------------------------------------------------- | ----------------------------------- | ----------------------------------------------------------------------------- |
| Build EP mesh, shard experts as DTensors                 | transformers                        | `from_pretrained` → `distribute_model`                                        |
| Load weights from disk (mmap, all ranks)                 | transformers                        | `_load_pretrained_model` → `convert_and_load_state_dict_in_model`             |
| Replace per-rank weights with placeholders on non-rank-0 | transformers                        | `_finalize_model_loading` → `_move_missing_keys_from_meta_to_device` (Path B) |
| Move model to meta and broadcast from rank 0             | accelerate                          | `fsdp2_prepare_model` → `fsdp2_load_full_state_dict`                          |
| Tell FSDP to skip MoE modules                            | transformers Trainer                | `create_accelerator_and_postprocess` (line 844)                               |
| EP DataLoader sharding                                   | transformers Trainer                | `Trainer._get_dataloader` FSDP-fallback `prepare_data_loader` patch           |
| FSDP wrap, dense-weight sharding                         | accelerate → torch.distributed.fsdp | `_prepare_fsdp2` → `fsdp2_prepare_model` → `fully_shard`                      |

---

## 3. Cheat sheet — what triggers what

```
sft.py
└── SFTTrainer.__init__
    ├── (pre-super) build (dp, tp) mesh, set DistributedConfig    [TRL]
    ├── from_pretrained                                            [transformers]
    │   ├── cls(config) under torch.device("meta")                 every rank, RSS ~0.5 GB
    │   ├── initialize_tensor_parallelism                          extracts "tp" sub-mesh from (dp,tp)
    │   ├── distribute_model                                       sets model._device_mesh, adds hooks
    │   │   └── post_shard_wrap                                    DS: tag plain param (allreduce=False)
    │   │                                                          FSDP: wrap as DTensor on EP mesh
    │   ├── _load_pretrained_model                                 every rank loads from disk
    │   │   ├── safe_open(...).get_slice(k)                        lazy PySafeSlice, no bytes copied
    │   │   └── _materialize_copy → tensor[...].to(device,dtype)   tensor[...] is mmap-backed CPU view
    │   │                                                          (page cache, not RSS); .to(device) is
    │   │                                                          where the actual copy happens.
    │   │                                                          ALL EP variants (DS, FSDP±cpu_ram):
    │   │                                                                device = cuda:LOCAL_RANK
    │   │                                                                → bytes copied straight to GPU
    │   │                                                          (cpu_ram_efficient_loading does NOT
    │   │                                                           change the materialize device under
    │   │                                                           EP; effect is in Path B below.)
    │   └── _finalize_model_loading
    │       └── _move_missing_keys_from_meta_to_device
    │           ├── DS branch: return                              no-op
    │           ├── FSDP rank-0 (Path A): missing_keys = ∅         no-op for complete checkpoint
    │           └── FSDP non-rank-0 (Path B):                      drops the cuda values
    │               for p in named_parameters():                   `_materialize_copy` just put on GPU
    │                 _load_parameter_into_model(p, zeros_like)    and replaces with cpu zeros.
    │                                                              For EP this clobbers the per-rank
    │                                                              expert slice → MoE routing broken
    │                                                              after later rank-0 broadcast (C2).
    │                                                              ⚠ also commits ~60 GB × (n-1) RSS
    │                                                              of cpu zeros on dense MoE models
    │                                                              (fsdp2_loading_walkthrough.md).
    │                                                              (Only fires when is_fsdp_enabled() —
    │                                                               needs both ACCELERATE_USE_FSDP and
    │                                                               FSDP_CPU_RAM_EFFICIENT_LOADING set.)
    └── super().__init__() (Trainer)                               [transformers]
        └── create_accelerator_and_postprocess
            ├── Accelerator(...)                                   [accelerate]   builds state, no model wrap yet
            ├── if DS+EP: _create_expert_and_data_parallel         [DeepSpeed]    pre-create expert PGs
            ├── if DS+EP: state.ds_device_mesh = model._device_mesh   <-- the new EP DataLoader fix
            └── if FSDP+EP: fsdp_plugin.ignored_modules = MoE      [transformers] tells fully_shard
                                                                                  to skip EP DTensors

trainer.train()
└── _inner_training_loop
    └── _prepare_for_training                                      [transformers]
        └── accelerator.prepare(model, optimizer)                  [accelerate]
            ├── DEEPSPEED → _prepare_deepspeed → deepspeed.initialize
            │              → DS uses param.allreduce=False markers
            │              → no meta-move, no broadcast — per-rank loaded values stay
            │              → prepare_data_loader sees state.ds_device_mesh, divides by ep_size
            └── FSDP2     → _prepare_fsdp2 → fsdp2_prepare_model
                          → original_sd = model.state_dict()
                          → model.to("meta")  [wipes loaded values]
                          → fully_shard       [params become meta-DTensors]
                          → fsdp2_load_full_state_dict
                              ├ rank 0: read original_sd, broadcast to all
                              └ non-rank-0: empty(gpu) + recv broadcast
                                            (their loaded values are NEVER read)
                          → ignored_params skips MoE (avoids double-DTensor)
                          → prepare_data_loader sees state.device_mesh=None, naive sharding (BUG)
                          → Trainer._get_dataloader FSDP fallback patches around it
```

---

## 4. Glossary

- **`model.has_ep`**: transformers convention. Set on `PreTrainedModel` when `DistributedConfig.enable_expert_parallel=True`. Trainer branches on this to enable EP-specific setup.
- **`model._device_mesh`**: the EP mesh (1D `("tp",)`) transformers built and uses for the experts all-reduce. NOT the same as `accelerator.state.device_mesh`.
- **`accelerator.state.device_mesh`**: the mesh accelerate built from `ParallelismConfig`. Used by FSDP wrap and the non-DS dataloader path. Is `None` when no `parallelism_config` is set.
- **`accelerator.state.ds_device_mesh`**: a separate attribute used only by the DEEPSPEED branch in `_prepare_device_mesh`. We reuse this slot to hand accelerate the EP mesh.
- **`parallelism_config.fsdp_dim_names`**: `["dp_replicate", "dp_shard_cp"]`. Read by FSDP wrap to slice the mesh for sharding. The reason we can't drop a 1D mesh into `state.device_mesh` for FSDP without also wiring up a `ParallelismConfig`.
- **`process_index` vs `effective_process_index`**: `process_index` is the global rank (0..15). `effective_process_index = process_index // ep_size` is what the dataloader uses to assign data-shards (0 or 1 in our run). All ranks within the same EP group share the same `effective_process_index`, so they read the same micro-batch.
- **mmap-backed tensor**: tensor whose storage is a view into a `safe_open(...)` mmap'd region of a `.safetensors` file. Looks like a regular `torch.Tensor` but its bytes live in the kernel page cache, not in process RSS. This is why every rank can "load" a 60 GB model into `model.state_dict()` without committing 60 GB × `ranks_per_node` of CPU RAM.
- **rank-0 broadcast pattern (FSDP)**: under `cpu_ram_efficient_loading=True`, only rank 0's loaded weights matter. accelerate moves the model to meta, fully-shards into DTensors, then `fsdp2_load_full_state_dict` reads each param from rank-0's pre-meta snapshot and broadcasts it to every other rank's DTensor shard. Non-rank-0 ranks' loaded values are written by transformers but never read — they only exist to give `fully_shard` shape/dtype metadata to derive sharding from.
