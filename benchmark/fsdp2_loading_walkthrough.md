# FSDP2 + `cpu_ram_efficient_loading` walkthrough on Qwen3-30B-A3B

This document walks through what _actually_ happens between `transformers.from_pretrained` and `accelerate.prepare(model)` when loading **Qwen/Qwen3-30B-A3B** with FSDP2 and `fsdp_cpu_ram_efficient_loading=true` on a 2-node × 8 H100 setup. The goal is to clarify why `_move_missing_keys_from_meta_to_device` is even called on a model where _no keys are missing_, why the function loops over **all** parameters on non-rank-0 ranks (not just missing ones), and how the regression introduced by transformers PR #45050 (`empty_like` → `zeros_like`) leads to CPU OOM at 8 ranks/node.

All file references are to the editable installs:

- `/fsx/amine_dirhoussi/transformers/src/transformers/`
- `/fsx/amine_dirhoussi/trl/.venv/lib/python3.11/site-packages/accelerate/`

---

## Call tree (for orientation)

The relevant code lives in three layers: `transformers.modeling_utils` (entry point + meta init + post-load processing), `transformers.core_model_loading` (the weight loader), and `accelerate.utils.fsdp_utils` (FSDP2 sharding + broadcast). Below is the chain of calls from `from_pretrained` through `accelerator.prepare(model)`. Indentation reflects call depth; comments are short reminders of what each frame contributes. `mu.py` = `transformers/modeling_utils.py`, `cml.py` = `transformers/core_model_loading.py`, `fu.py` = `accelerate/utils/fsdp_utils.py`.

```text
PreTrainedModel.from_pretrained(...)                                 mu.py:3735
│   ── entry point. Reads env (ACCELERATE_USE_FSDP, FSDP_CPU_RAM_EFFICIENT_LOADING),
│      resolves checkpoint_files, picks dtype, builds load_config.
│
├── cls.get_init_context(...)                                        mu.py:3648
│   │   Returns the list of context managers wrapping cls(config).
│   │   Crucially appends torch.device("meta") for the non-DeepSpeed path.
│   └── (returns [..., torch.device("meta"), init.meta_device_safe_creation_ops()])
│
├── with ContextManagers(model_init_context):
│       model = cls(config, ...)                                     mu.py:4170-4172
│       └── all params/buffers constructed on meta. RSS ~0.5 GB on every rank.
│
├── cls._load_pretrained_model(model, state_dict,                    mu.py:4258
│   │                          checkpoint_files, load_config)
│   │   ── per-rank weight loader. NO non-rank-0 short-circuit anymore.
│   │
│   ├── for file in checkpoint_files:                                mu.py:4327
│   │      file_pointer = safetensors.safe_open(file, "pt", "cpu")
│   │      for k in file_pointer.keys():
│   │          merged_state_dict[k] = file_pointer.get_slice(k)
│   │           └── PySafeSlice — lazy view, NO bytes copied yet.
│   │
│   ├── convert_and_load_state_dict_in_model(model,                  cml.py:1151
│   │   │                                    merged_state_dict, ...)
│   │   │   ── walks every key, materializes, applies WeightConverters,
│   │   │      installs onto the model. Removes loaded keys from missing_keys.
│   │   │
│   │   ├── for original_key, tensor in state_dict:                  cml.py:1294
│   │   │       (rename, dispatch to thread pool)
│   │   │
│   │   │   ├── spawn_materialize(thread_pool, tensor,               cml.py:944
│   │   │   │   │                 device, dtype)
│   │   │   │   │   ── submits a job to the pool (or returns a Callable).
│   │   │   │   │      The job calls _materialize_copy.
│   │   │   │   └── thread_pool.submit(lambda: _materialize_copy(...))
│   │   │   │
│   │   │   │       └── _materialize_copy(tensor, device, dtype)     cml.py:936
│   │   │   │             ├── tensor[...] ← reads from PySafeSlice.
│   │   │   │             │    Returns a torch.Tensor whose storage is
│   │   │   │             │    a VIEW into the safetensors mmap. Empirically
│   │   │   │             │    this does NOT commit RSS — bytes live in
│   │   │   │             │    the kernel page cache.
│   │   │   │             └── tensor.to(device=cpu, dtype=bf16)
│   │   │   │                  No-op if already cpu+bf16.
│   │   │   │
│   │   │   ├── (TP path) spawn_tp_materialize(...)                  cml.py:964
│   │   │   │     same pattern, but applies TP sharding op.
│   │   │   │
│   │   │   └── mapping.add_tensor(target, source, pattern, future)
│   │   │
│   │   └── for first_param_name, mapping in param_name_to_load:     cml.py:1386
│   │       │
│   │       ├── mapping.convert(first_param_name, model=...,         cml.py:725 (WeightConverter.convert)
│   │       │   │               loading_info=...)
│   │       │   │   ── resolves the futures (calls .result() on each),
│   │       │   │      applies the WeightConverter ops (Chunk, MergeModulelist, …).
│   │       │   └── (returns realized_value: dict[target_name -> Tensor])
│   │       │
│   │       └── set_param_for_module(model, target_name, param,      cml.py:1044
│   │           │                    loading_info, ...)
│   │           ├── loading_info.missing_keys.discard(target_name)
│   │           │    ← key is REMOVED from missing_keys here.
│   │           │      For Qwen3-30B with a complete checkpoint this
│   │           │      empties missing_keys completely.
│   │           ├── param_value._is_hf_initialized = True
│   │           └── setattr(module_obj, param_name, param_value)
│   │                ← model.layers[i]…weight now points at the
│   │                  mmap-backed tensor produced by _materialize_copy.
│   │
│   └── for k in all_pointer: k.__exit__(None, None, None)           mu.py:4353-4354
│        Closes the Python `safe_open`. The underlying Rust mmap is
│        kept alive via Arc refcount as long as any installed tensor
│        still views it (which is "all of them" at this point).
│
├── cls._finalize_model_loading(model, load_config, loading_info)    mu.py:4359
│   │   ── post-loading cleanup. THIS is where Step 3 happens.
│   │
│   ├── model.mark_tied_weights_as_initialized(loading_info)
│   │
│   ├── model._move_missing_keys_from_meta_to_device(                mu.py:4603
│   │       missing_and_mismatched, device_map, device_mesh, hf_quantizer)
│   │   ├── DeepSpeed branch        → return                         mu.py:4618
│   │   ├── FSDP non-rank-0 branch  → Path B                         mu.py:4622-4626
│   │   │     for k, p in self.named_parameters():
│   │   │         value = torch.zeros_like(p, device="cpu")          ← THE BUG (#45050)
│   │   │         _load_parameter_into_model(self, k, value)
│   │   │              └── setattr(...) replaces mmap-backed param   mu.py:516
│   │   │                  with a freshly committed CPU allocation.
│   │   │     for k, b in self.named_buffers():
│   │   │         value = torch.zeros_like(b, device="cpu")
│   │   │         _load_parameter_into_model(self, k, value)
│   │   └── default branch          → Path A                         mu.py:4631
│   │         for key in missing_keys - self.all_tied_weights_keys.keys():
│   │             ... (empty for a complete checkpoint, no-op on rank 0)
│   │
│   ├── model._initialize_missing_keys(is_quantized)                 mu.py:4649
│   │     On FSDP non-rank-0: just marks every param/buffer
│   │     `_is_hf_initialized = True` so the regular initializer
│   │     skips them (rank 0 will broadcast the canonical values).
│   │
│   ├── model.tie_weights(...)                                       mu.py:4382
│   └── model._adjust_missing_and_unexpected_keys(loading_info)      mu.py:4385

# ─── from_pretrained returns; trainer constructs and calls accelerator.prepare ───

Accelerator.prepare(model, optimizer, ...)                           accelerate/accelerator.py
└── prepare_model(...) → fsdp2_prepare_model(accelerator, model)     fu.py:621
    │
    ├── original_sd = model.state_dict()                             fu.py:643
    │     On rank 0: dict of mmap-backed tensors with real values.
    │     On non-rank-0: dict of zeros (post-#45050) or empty (pre-#45050)
    │                    or meta (proposed fix) tensors. Never read on non-rank-0.
    │
    ├── if cpu_ram_efficient_loading:                                fu.py:669
    │       capture original_non_persistent_buffers (rope caches…)
    │       model = model.to(torch.device("meta"))                   fu.py:683
    │       model.tie_weights()
    │
    ├── fully_shard(module, **fsdp2_kwargs) for each wrapped module  fu.py:694, 697
    │     Replaces every nn.Parameter with a DTensor whose local shard is on meta.
    │
    ├── fsdp2_load_full_state_dict(accelerator, model,               fu.py:467
    │   │                          original_sd, cpu_offload=...)
    │   │
    │   ├── meta_sharded_sd = model.state_dict()                     fu.py:484
    │   │
    │   ├── if accelerator.is_main_process:                          fu.py:513
    │   │       for (name, full_param), sharded_param in zip(...):
    │   │           full_param.detach().to("cuda")     ← rank-0 reads mmap
    │   │                                                 once per param into GPU
    │   │           dist.broadcast(full_param, src=0)
    │   │           distribute_tensor(full_param, device_mesh, placements)
    │   │
    │   └── else:                                                    fu.py:535
    │           for name, sharded_param in meta_sharded_sd.items():
    │               full_tensor = torch.empty(size, device="cuda")  ← non-rank-0
    │               dist.broadcast(full_tensor, src=0)                allocates a
    │               distribute_tensor(...)                             GPU empty,
    │                                                                  receives
    │                                                                  broadcast.
    │                                                                  full_sd is
    │                                                                  NEVER READ.
    │
    └── re-register original_non_persistent_buffers                  fu.py:710-721
```

The two function names that hide all the action: `_materialize_copy` (the supposed "real read" — actually mmap-backed and free) and `_move_missing_keys_from_meta_to_device` (the supposed "fix-up of missing keys" — actually replaces every param on FSDP non-rank-0 ranks regardless of `missing_keys`). Everything else in the doc reduces to clarifying what those two functions do and why their behavior interacts to produce the OOM.

---

## 0. The cast

Three numbers to keep in mind:

| Quantity                    | Value                |
| --------------------------- | -------------------- |
| Qwen3-30B-A3B params (bf16) | ~30 B × 2 B = ~60 GB |
| Ranks per node              | 8                    |
| Node CPU RAM (p5.48xlarge)  | ~2 TB                |

Two environment variables are set the moment `Accelerator(...)` is constructed (well before `from_pretrained` runs):

```text
ACCELERATE_USE_FSDP=True
FSDP_CPU_RAM_EFFICIENT_LOADING=True
```

These are what `transformers.is_fsdp_enabled()` checks (`integrations/fsdp.py:44-53`):

```python
def is_fsdp_enabled():
    return (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and strtobool(os.environ.get("ACCELERATE_USE_FSDP", "False")) == 1
        and strtobool(os.environ.get("FSDP_CPU_RAM_EFFICIENT_LOADING", "False")) == 1
    )
```

So inside `from_pretrained`, transformers "knows" we are in FSDP + cpu_ram_efficient mode and gates a few branches on `is_fsdp_enabled() and not is_local_dist_rank_0()`. That gate is the source of the bug.

Linux memory primer (matters because the diff between `empty_like` and `zeros_like` is exactly this):

```text
torch.empty(...)  →  malloc → anonymous mmap → virtual address space reserved.
                    Pages are NOT committed until they are written.
                    RSS is unaffected. /proc/<pid>/status shows VmSize↑, VmRSS unchanged.

torch.zeros(...)  →  same allocation, then writes 0 to every byte.
                    Every page is faulted → committed → counted in RSS.
                    RSS grows by the full tensor size.
```

Scratch repro (single process, no torch.distributed):

```python
import os, torch, psutil, gc
proc = psutil.Process(os.getpid())
SHAPE = (4_000_000_000 // 2,)  # 4 GB worth of bf16 elements

print(f"start RSS:  {proc.memory_info().rss/1e9:5.2f} GB")
t = torch.empty(SHAPE, dtype=torch.bfloat16, device="cpu")
print(f"after empty:{proc.memory_info().rss/1e9:5.2f} GB  # virtual only")
del t; gc.collect()

t = torch.zeros(SHAPE, dtype=torch.bfloat16, device="cpu")
print(f"after zeros:{proc.memory_info().rss/1e9:5.2f} GB  # committed")

# start RSS:  0.54 GB
# after empty:0.54 GB
# after zeros:4.54 GB
```

---

## 1. Step 1 — `from_pretrained`: the model is built on **meta** on every rank

```python
# transformers/modeling_utils.py:4167-4172
model_init_context = cls.get_init_context(dtype, is_quantized, _is_ds_init_called, allow_all_kernels)
config = copy.deepcopy(config)
with ContextManagers(model_init_context):
    model = cls(config, *model_args, **model_kwargs)   # ← every rank constructs on meta
```

Where `get_init_context` (line 3671-3675) plugs in `torch.device("meta")` for every non-DeepSpeed path:

```python
else:
    # meta_device_safe_creation_ops patches torch.linspace to default to CPU
    # so that custom models calling .item() during __init__ don't crash on meta tensors.
    init_contexts.extend([torch.device("meta"), init.meta_device_safe_creation_ops()])
```

After this returns, the per-rank state is:

```text
LOCAL_RANK = 0..7 on each node:

  model.layers[i].mlp.experts[j].gate_proj.weight  → meta:(2048, 768)  bf16
  model.layers[i].mlp.experts[j].up_proj.weight    → meta:(2048, 768)  bf16
  model.layers[i].mlp.experts[j].down_proj.weight  → meta:(768, 2048)  bf16
  model.layers[i].self_attn.q_proj.weight          → meta:(2048, 2048) bf16
  ...                                                                        ~  600 params total
                                                                             ~  30B numel
                                                                             RSS ≈ 0.5 GB (Python+torch overhead)
```

No CPU memory committed for any tensor yet. Same on every rank.

---

## 2. Step 2 — `_load_pretrained_model`: every rank "reads" the .safetensors but RSS stays flat

There is no non-rank-0 short-circuit in the loading code anymore (it was lost in the #41580 refactor). Every rank — rank 0 included — walks every shard. **But that does not mean every rank commits 60 GB of CPU RSS.** safetensors uses mmap; `slice[...]` returns a tensor whose storage points into the mmap'd region, which lives in the kernel page cache, not in process RSS.

The relevant block is at `modeling_utils.py:4321-4356`:

```python
all_pointer = set()
if state_dict is not None:
    merged_state_dict = state_dict
elif checkpoint_files is not None and checkpoint_files[0].endswith(".safetensors") and state_dict is None:
    merged_state_dict = {}
    for file in checkpoint_files:
        if load_config.disable_mmap or _is_on_hf_mount(file):
            with open(file, "rb") as _fh:
                merged_state_dict.update(_safe_load_bytes(_fh.read()))
            continue
        file_pointer = safe_open(file, framework="pt", device="cpu")
        all_pointer.add(file_pointer)
        for k in file_pointer.keys():
            merged_state_dict[k] = file_pointer.get_slice(k)   # lazy slice, no memory yet
...
loading_info, disk_offload_index = convert_and_load_state_dict_in_model(
    model=model, state_dict=merged_state_dict, ...
)
# finally close all opened file pointers
for k in all_pointer:
    k.__exit__(None, None, None)
```

`safe_open(...).get_slice(k)` returns a `PySafeSlice` — a lazy view with no bytes copied yet. The "materialization" happens in `_materialize_copy` (`core_model_loading.py:936-941`):

```python
def _materialize_copy(tensor: torch.Tensor, device=None, dtype=None) -> torch.Tensor:
    # This slicing is what actually loads the tensor from the safetensors slice object
    tensor = tensor[...]                    # ← does NOT commit RSS; tensor backed by mmap
    if dtype is not None or device is not None:
        tensor = tensor.to(device=device, dtype=dtype)   # no-op if already cpu+bf16
    return tensor
```

The naming is misleading: `tensor[...]` does not copy bytes into a new owned allocation. It produces a `torch.Tensor` whose storage is the mmap'd file region. Empirically (single process, real safetensors file ~5 GB, no torch.distributed):

```text
start                         RSS =  0.542 GB
after safe_open (100 keys)    RSS =  0.543 GB
after get_slice for all keys  RSS =  0.543 GB
after slice[...] for ALL keys RSS =  0.551 GB     # 4.979 GB of "tensor" data
after del + gc                RSS =  0.552 GB
after close                   RSS =  0.547 GB
```

Materializing ~5 GB of tensor data via `slice[...]` grew RSS by ~9 MB. The bytes are page-cache (kernel memory, reclaimable, not counted in RSS) — not allocated heap. Same is true for the path that goes through `_safe_load_bytes` on the hf-mount fallback at line 4329: it reads bytes into Python memory, so that one DOES commit RSS — but on this cluster the safetensors path is the one that runs (the `_is_on_hf_mount` check returns false for `/fsx/...` paths).

`set_param_for_module` (`core_model_loading.py:1044-1085`) then plugs the mmap-backed tensor into the model in place of the meta one:

```python
def set_param_for_module(model, target_name, param_value, loading_info, distributed_operation, hf_quantizer):
    module_path, _, param_name = target_name.rpartition(".")
    module_obj = model.get_submodule(module_path) if module_path else model
    ...
    if not isinstance(param_value, torch.nn.Parameter):
        if param_name not in module_obj._buffers:
            param_value = torch.nn.Parameter(param_value, requires_grad=param_value.is_floating_point())
    loading_info.missing_keys.discard(target_name)         # ← key is removed from missing_keys
    ...
    param_value._is_hf_initialized = True
    setattr(module_obj, param_name, param_value)           # ← swap meta for mmap-backed tensor
```

Note about the file_pointer close at lines 4353-4354: closing the `safe_open` does decrement Rust-side refcounts, but the underlying `Mmap` is owned by an `Arc` that is also retained by every `PySafeSlice` whose tensor we just installed on the model. So the mmap stays alive as long as those tensors are alive.

So **on every rank**, after Step 2 runs to completion:

```text
LOCAL_RANK = 0..7:

  model.layers[i].mlp.experts[j].gate_proj.weight  → mmap-backed:(2048, 768)  bf16
  model.layers[i].mlp.experts[j].up_proj.weight    → mmap-backed:(2048, 768)  bf16
  ...                                                                        ~  600 params total
  loading_info.missing_keys = ∅ (or very small — only tied weights, etc.)
                                                                             RSS ≈ 0.5 GB
                                                                             page cache: ~60 GB
                                                                               (shared across all
                                                                                ranks on the node,
                                                                                kernel-managed,
                                                                                does not count
                                                                                against cgroup as RSS)
```

For Qwen3-30B with a complete checkpoint, `loading_info.missing_keys` is empty after Step 2 — every parameter in `model.state_dict()` was found in the .safetensors and got `loading_info.missing_keys.discard(target_name)`-ed. So Path A of `_move_missing_keys_from_meta_to_device` is a no-op; the *only* effective code path on non-rank-0 is Path B in Step 3.

---

## 3. Step 3 — `_finalize_model_loading` calls `_move_missing_keys_from_meta_to_device`

This is the function whose name lies. The call site is `modeling_utils.py:4371-4376`:

```python
# Move missing (and potentially mismatched) keys and non-persistent buffers back to their expected device from
# meta device (because they were not moved when loading the weights as they were not in the loaded state dict)
model._move_missing_keys_from_meta_to_device(
    loading_info.missing_and_mismatched(),
    load_config.device_map,
    load_config.device_mesh,
    load_config.hf_quantizer,
)
```

The function itself (`modeling_utils.py:4603-4647`) has **two completely independent code paths** that share a name:

```python
def _move_missing_keys_from_meta_to_device(self, missing_keys, device_map, device_mesh, hf_quantizer) -> None:
    is_quantized = hf_quantizer is not None

    # ── DeepSpeed: no meta init in the first place, nothing to move
    if is_deepspeed_zero3_enabled() and not is_quantized:
        return

    # ── Path B: FSDP non-rank-0 SHORT-CIRCUIT (← does not use missing_keys at all)
    if is_fsdp_enabled() and not is_local_dist_rank_0() and not is_quantized:
        for key, param in self.named_parameters():               # ← ALL params
            value = torch.zeros_like(param, device="cpu")        # was empty_like before #45050
            _load_parameter_into_model(self, key, value)
        for key, buffer in self.named_buffers():                 # ← ALL buffers
            value = torch.zeros_like(buffer, device="cpu")       # was empty_like before #45050
            _load_parameter_into_model(self, key, value)
        return

    # ── Path A: the regular "missing keys" path (rank-0, single-process, etc.)
    for key in missing_keys - self.all_tied_weights_keys.keys():  # ← uses missing_keys
        param = self.get_parameter_or_buffer(key)
        param_device = get_device(device_map, key, valid_torch_device=True)
        value = torch.empty_like(param, device=param_device)
        if device_mesh is not None:
            shard_and_distribute_module(self, value, param, key, None, False, device_mesh.get_local_rank(), device_mesh)
        else:
            _load_parameter_into_model(self, key, value)
    for key, buffer in self.named_non_persistent_buffers():
        buffer_device = get_device(device_map, key, valid_torch_device=True)
        value = torch.empty_like(buffer, device=buffer_device)
        _load_parameter_into_model(self, key, value)
```

This is the answer to the original confusion: **on FSDP non-rank-0 ranks, `missing_keys` is ignored entirely**. Path B walks every entry in `self.named_parameters()` (and `self.named_buffers()`) regardless of whether a key was missing or not. The function name describes Path A; Path B has piggybacked on it since #29587 (back in March 2024) and is really a cleanup-and-replace operation:

> "Drop the just-loaded per-rank weights, replace with cheap placeholders that have the right shape/dtype, because rank 0 is going to broadcast the canonical values shortly."

`_load_parameter_into_model` is just `setattr` (`modeling_utils.py:516-523`):

```python
def _load_parameter_into_model(model: "PreTrainedModel", param_name: str, tensor: torch.Tensor):
    parent, param_type = get_module_from_name(model, param_name)
    if param_type in parent._parameters and not isinstance(tensor, nn.Parameter):
        tensor = nn.Parameter(tensor, requires_grad=tensor.is_floating_point())
    setattr(parent, param_type, tensor)
```

So each iteration drops the reference to the mmap-backed tensor (which had RSS cost ~0) and replaces it with a fresh CPU allocation (which may or may not have RSS cost depending on `empty_like` vs `zeros_like`). The companion function `_initialize_missing_keys` (line 4658-4670) then marks every param/buffer `_is_hf_initialized=True` on non-rank-0 so the regular weight initializer doesn't re-init values that are about to be broadcast over.

### 3.1 Why `empty_like → zeros_like` blows up RSS

The baseline at the start of Step 3, on every rank, is the mmap-backed state from Step 2: ~0.5 GB RSS, with 60 GB of page-cache pages mapped in. The `for k, p in named_parameters()` loop's only job is to replace the mmap-backed `param` with a fresh CPU tensor of the same shape/dtype. Whether that fresh tensor commits RSS or not is the entire bug:

```text
Per-rank RSS, non-rank-0 (LOCAL_RANK ∈ {1..7})

Pre-#45050 (empty_like)
─────────────────────────
   Start of Step 3:   [ 0.5 GB RSS,  60 GB page-cache mmap-backed params ]
   for k, p in named_parameters():
       value = empty_like(p, device="cpu")          # virtual-only allocation, RSS unchanged
       setattr(...)                                  # mmap-backed param ref dropped
   Step 3 end:        [ 0.5 GB RSS,  ~60 GB virtual placeholders, mmap can be reclaimed by kernel ]
                       RSS  ≈  0.5 GB                                                          ✓

Post-#45050 (zeros_like)
─────────────────────────
   Start of Step 3:   [ 0.5 GB RSS,  60 GB page-cache mmap-backed params ]
   for k, p in named_parameters():
       value = zeros_like(p, device="cpu")          # writes 0 to every byte → page-faults
                                                    # → 60 GB of fresh anonymous pages committed
       setattr(...)                                  # mmap-backed param ref dropped
   Step 3 end:        [ ~60 GB RSS of committed zeros ]
                       RSS  ≈  60 GB                                                           ✗
```

Key insight about the "peak" question: there is no transient 120 GB peak from the old + new tensors coexisting. The "old" tensor is mmap-backed (page cache, not RSS), and only the "new" `zeros_like` allocation contributes to RSS. The growth is monotonic from ~0.5 GB to ~60 GB across the loop iterations, plateauing at ~60 GB.

Per-node committed CPU (RSS, summed across the 8 ranks on a node) at the end of Step 3:

```text
                    rank0    rank1   rank2   rank3   rank4   rank5   rank6   rank7    sum
empty_like:          ~0.5 GB  ~0.5 GB ~0.5 GB ~0.5 GB ~0.5 GB ~0.5 GB ~0.5 GB ~0.5 GB  ~4 GB
zeros_like:          ~0.5 GB  ~60 GB  ~60 GB  ~60 GB  ~60 GB  ~60 GB  ~60 GB  ~60 GB   ~420 GB
                                                                                       ─────
                                                                                       OOM on 2 TB nodes
                                                                                       once you add CUDA
                                                                                       contexts, NCCL
                                                                                       buffers, dataset,
                                                                                       dataloader workers
```

(rank 0 stays low because it goes down Path A, where `missing_keys` is empty for a complete checkpoint, so the loop is a no-op. Its mmap-backed params are alive but not contributing to RSS.)

Cgroup OOM message (slurm step):

```text
slurmstepd: error: Detected 1 oom-kill event(s) in StepId=...
exitcode: -9 (SIGKILL)
```

---

## 4. Step 4 — `accelerator.prepare(model)` → `fsdp2_prepare_model`

Right after `from_pretrained` returns, the trainer calls `accelerator.prepare(...)`. The relevant function is in `accelerate/utils/fsdp_utils.py:621-707`:

```python
def fsdp2_prepare_model(accelerator, model: torch.nn.Module) -> torch.nn.Module:
    ...
    fsdp2_plugin = accelerator.state.fsdp_plugin
    fsdp2_plugin.set_auto_wrap_policy(model)
    original_sd = model.state_dict()                 # ← snapshot BEFORE we move to meta
    ...
    if fsdp2_plugin.cpu_ram_efficient_loading and not model_has_params4bit:
        # Capture non-persistent buffers (rope caches, etc.) BEFORE the meta move,
        # because they are NOT in original_sd (persistent=False) and won't be broadcast.
        non_persistent_buffer_fqns = get_non_persistent_buffers(model, recurse=True, fqns=True)
        original_non_persistent_buffers = copy.deepcopy(
            {k: v for k, v in model.named_buffers() if k in non_persistent_buffer_fqns}
        )
        model = model.to(torch.device("meta"))       # ← move ALL params/buffers to meta
        if hasattr(model, "tie_weights"):
            model.tie_weights()

    # Wrap with fully_shard. After this each param is a DTensor whose local shard is on meta.
    auto_wrap_policy_func = fsdp2_prepare_auto_wrap_policy(fsdp2_plugin, model)
    if auto_wrap_policy_func is not None:
        for module in get_module_children_bottom_up(model)[:-1]:
            if auto_wrap_policy_func(module) and not isinstance(module, FSDPModule):
                fully_shard(module, **fsdp2_kwargs)
    if not isinstance(model, FSDPModule):
        fully_shard(model, **fsdp2_kwargs)

    if fsdp2_plugin.cpu_ram_efficient_loading:
        from torch.distributed.fsdp import CPUOffloadPolicy
        fsdp2_load_full_state_dict(
            accelerator, model, original_sd,
            cpu_offload=isinstance(fsdp2_plugin.cpu_offload, CPUOffloadPolicy),
        )

    if fsdp2_plugin.cpu_ram_efficient_loading and not model_has_params4bit:
        # Re-register the non-persistent buffers we captured before the meta move
        for fqn, buffer_tensor in original_non_persistent_buffers.items():
            buffer_tensor = buffer_tensor.to(accelerator.device)
            ...
            parent_module.register_buffer(local_buffer_name, buffer_tensor, persistent=False)
        if hasattr(model, "tie_weights"):
            model.tie_weights()
```

And `fsdp2_load_full_state_dict` (`fsdp_utils.py:467-554`):

```python
def fsdp2_load_full_state_dict(accelerator, model, full_sd, cpu_offload=False):
    ...
    meta_sharded_sd = model.state_dict()        # DTensors with meta local shards
    sharded_sd = {}

    if accelerator.is_main_process:
        # rank 0: read each param from full_sd (the snapshot from before .to(meta))
        # and broadcast it.
        for (param_name, full_param), sharded_param in zip(full_sd.items(), meta_sharded_sd.values()):
            device_mesh = sharded_param.device_mesh
            full_param = full_param.detach().to(device_mesh.device_type)
            if isinstance(full_param, DTensor):
                full_param = full_param.to_local()
            dist.broadcast(full_param, src=0, group=dist.group.WORLD)
            sharded_tensor = distribute_tensor(full_param, device_mesh, sharded_param.placements)
            ...
            sharded_sd[param_name] = sharded_tensor
    else:
        # non-rank-0: NEVER reads full_sd. Just allocates a GPU empty and broadcasts into it.
        for param_name, sharded_param in meta_sharded_sd.items():
            device_mesh = sharded_param.device_mesh
            full_tensor = torch.empty(sharded_param.size(),
                                      device=device_mesh.device_type, dtype=sharded_param.dtype)
            dist.broadcast(full_tensor, src=0, group=dist.group.WORLD)
            sharded_tensor = distribute_tensor(full_tensor, device_mesh, sharded_param.placements)
            ...
            sharded_sd[param_name] = sharded_tensor

    # we set `assign=True` because our params are on meta device
    model.load_state_dict(sharded_sd, assign=True)
    return model
```

Two crucial properties of this code:

1. **`model.to(torch.device("meta"))` blows away whatever values `_move_missing_keys_from_meta_to_device` wrote.** Whether they were `empty_like` garbage, `zeros_like` zeros, or actually-on-meta tensors does not matter — they're all replaced by meta tensors at line 683.

2. **The non-rank-0 branch of `fsdp2_load_full_state_dict` never reads `full_sd`.** It only reads `meta_sharded_sd` for sizes/dtypes/placements, allocates a GPU `torch.empty`, and broadcasts. So whatever non-rank-0 wrote into its CPU params during Step 3 is **never observed by anyone**.

Conclusion: the parameter half of Step 3's loop on non-rank-0 is wasted work _and_ the source of the OOM.

---

## 5. The fix

```diff
diff --git a/src/transformers/modeling_utils.py b/src/transformers/modeling_utils.py
--- a/src/transformers/modeling_utils.py
+++ b/src/transformers/modeling_utils.py
@@ -4621,11 +4621,15 @@ class PreTrainedModel(...):
-        # In this case we need to move everything back
+        # On non-rank-0 FSDP ranks with cpu_ram_efficient_loading, parameters get their values from
+        # rank-0 broadcast in `fsdp2_load_full_state_dict`. Materializing them here as `zeros_like` on
+        # CPU forces a physical memory commit (~model_size × (ranks_per_node - 1)) → OOM on 30B+ MoE
+        # at 8 ranks/node. Leave parameters on meta; accelerate moves the model to meta before sharding
+        # anyway. Buffers (rope caches, etc.) are per-rank and not broadcast, so they still need a real
+        # placeholder — kept as `zeros_like` for the safe-NaN behavior PR #45050 was protecting.
         if is_fsdp_enabled() and not is_local_dist_rank_0() and not is_quantized:
-            for key, param in self.named_parameters():
-                value = torch.zeros_like(param, device="cpu")
-                _load_parameter_into_model(self, key, value)
             for key, buffer in self.named_buffers():
                 value = torch.zeros_like(buffer, device="cpu")
                 _load_parameter_into_model(self, key, value)
             return
```

Why this is safe:

- **Params**: by Step 4 they will be on meta anyway (line 683 of `fsdp_utils.py`), and the values get broadcast from rank 0 by `fsdp2_load_full_state_dict`. Skipping the CPU placeholder doesn't change behavior — it just removes a wasted alloc/dealloc.
- **Buffers**: `_initialize_missing_keys` skips re-init on non-rank-0 (line 4664-4670), `fsdp2_load_full_state_dict` only iterates persistent params/buffers (those in `state_dict()`), and non-persistent buffers are captured by accelerate **before** the meta move (line 678-681) and re-registered after. So:
    - Persistent buffers: in `original_sd`, broadcast like params, placeholder value irrelevant.
    - Non-persistent buffers: captured before any of this happens, restored after.
    - But: between Step 3 and the start of Step 4, anything that reads `model.named_buffers()` would observe the placeholder values. `zeros_like` is the safe choice (the bug PR #45050 was originally fixing — buffers like attention masks read with NaN bytes from `empty_like` would corrupt computations).

### 5.1 Per-rank state under the proposed fix

```text
LOCAL_RANK ≥ 1:
  Phase                              Params         Buffers       RSS
  ─────────────────────────────────  ─────────────  ────────────  ────────
  1. After cls(config)               meta           meta          ~0.5 GB
  2. After convert_and_load_…        mmap-backed    mmap-backed   ~0.5 GB
                                     (cpu, real)    (cpu, real)   (60 GB in
                                                                   page cache)
  3. After _move_missing_keys_…      meta           cpu/zeros     ~0.5 GB
                                     (proposed fix:                 (no commit:
                                      params loop                    params on
                                      removed)                       meta, only
                                                                     buffers
                                                                     committed —
                                                                     buffers ≪ 60 GB)
  4. After fsdp2_prepare_model
     (model.to(meta))                meta           cpu/zeros     ~0.5 GB
     (fully_shard)                   DTensor:       cpu/zeros     ~0.5 GB
                                       meta shard
     (fsdp2_load_full_state_dict)    DTensor:       cpu/zeros     ~0.5 GB
                                       gpu shard                  (broadcast → GPU)
     (re-register non-persistent     DTensor:       real          ~0.5 GB
      buffers from snapshot)           gpu shard    (re-registered)
```

Per-node CPU RSS (summed across 8 ranks) at end of Phase 4:

```text
                    rank0    rank1   rank2   rank3   rank4   rank5   rank6   rank7    sum
empty_like (old):    ~0.5 GB  ~0.5    ~0.5    ~0.5    ~0.5    ~0.5    ~0.5    ~0.5    ~4 GB
zeros_like (#45050): ~0.5 GB  ~60 GB  ~60 GB  ~60 GB  ~60 GB  ~60 GB  ~60 GB  ~60 GB   ~420 GB
proposed fix:        ~0.5 GB  ~0.5    ~0.5    ~0.5    ~0.5    ~0.5    ~0.5    ~0.5    ~4 GB
```

---

## 6. Validation

Tested with the same FSDP2 control config that previously OOMed:

- `Qwen/Qwen3-30B-A3B`, 2 nodes × 8 H100 NVL, FSDP2 DP=16, sdpa, `fsdp_cpu_ram_efficient_loading=true`
- Without the fix: cgroup OOM during `from_pretrained` (exitcode -9, SIGKILL)
- With the fix: training completes; MFU 18.44% at step 5 (mid-warmup), loss 1.598; control runs at step 20 reach ~23% MFU which matches the historical baseline of 23.1%
- Loss values are sane (no `inf`/`nan`), confirming the buffer-only `zeros_like` placeholder still protects against the original NaN bug PR #45050 was fixing

Bisect (control config: `Qwen/Qwen3-30B-A3B`, FSDP2, 2 nodes × 8 H100, DP=16, sdpa, max_steps=5, `fsdp_cpu_ram_efficient_loading=true`):

| Commit                       | Date                     | Result   | MFU    |
| ---------------------------- | ------------------------ | -------- | ------ |
| c43f15c71a                   | 2026-04-10               | PASS     | 21.46% |
| `a001f34439` (pre-#45050)    | 2026-04-13               | **PASS** | 21.13% |
| **`ff49f7c4cb` (PR #45050)** | **2026-04-13**           | **FAIL** | OOM    |
| e40b0c0195                   | 2026-04-13 (post-#45050) | FAIL     | OOM    |
| 8426e7e63d                   | 2026-04-15               | FAIL     | OOM    |
| 7a0d582ad4                   | 2026-04-20               | FAIL     | OOM    |
| 9dff7ca5c9                   | 2026-04-21               | FAIL     | OOM    |
| cbe7a02878                   | 2026-04-22               | FAIL     | OOM    |

---

## 7. Cheat sheet

End-to-end, what each rank does (Qwen3-30B-A3B, 8 ranks/node):

```text
        rank 0                            ranks 1-7
  ┌──────────────────────────────┐  ┌──────────────────────────────┐
  │ cls(config) on meta          │  │ cls(config) on meta          │  Step 1
  │   RSS  ~0.5 GB               │  │   RSS  ~0.5 GB               │
  └──────────────┬───────────────┘  └──────────────┬───────────────┘
                 │                                  │
                 ▼                                  ▼
  ┌──────────────────────────────┐  ┌──────────────────────────────┐
  │ safetensors mmap →           │  │ safetensors mmap →           │  Step 2
  │ params point into mmap       │  │ params point into mmap       │  (no rank-0
  │   RSS  ~0.5 GB               │  │   RSS  ~0.5 GB               │   short-circuit)
  │   page cache: ~60 GB         │  │   page cache: ~60 GB         │
  │   (kernel-managed, not RSS;  │  │   (shared across all ranks   │
  │    NOT counted by cgroup)    │  │    on the node)              │
  └──────────────┬───────────────┘  └──────────────┬───────────────┘
                 │                                  │
                 ▼                                  ▼
  ┌──────────────────────────────┐  ┌──────────────────────────────┐
  │ Path A: missing_keys = ∅     │  │ Path B: replace EVERY param  │  Step 3
  │   loop body never executes   │  │  with a fresh CPU tensor     │
  │   RSS ~0.5 GB                │  │  pre #45050 (empty_like):    │
  │                              │  │    virtual only → RSS ~0.5 GB│
  │                              │  │  post #45050 (zeros_like):   │
  │                              │  │    page-faulted → RSS ~60 GB │
  │                              │  │    × 7 ranks = 420 GB / node │
  │                              │  │    ⚠ OOM (cgroup -9)         │
  │                              │  │  proposed fix: skip params,  │
  │                              │  │    only buffers → RSS ~0.5 GB│
  └──────────────┬───────────────┘  └──────────────┬───────────────┘
                 │                                  │
                 ▼                                  ▼
  ┌──────────────────────────────┐  ┌──────────────────────────────┐
  │ original_sd = .state_dict()  │  │ original_sd = .state_dict()  │  Step 4
  │   (mmap-backed real values)  │  │   (placeholder values —      │  accelerator
  │ model.to("meta")             │  │    never read, see below)    │  .prepare
  │ fully_shard                  │  │ model.to("meta")             │
  │ fsdp2_load_full_state_dict:  │  │ fully_shard                  │
  │   for each param:            │  │ fsdp2_load_full_state_dict:  │
  │     full_param.to("cuda")    │  │   for each param:            │
  │     broadcast(src=0)         │  │     empty(gpu) ←             │
  │     distribute_tensor        │  │     broadcast(src=0) ←       │
  │                              │  │     distribute_tensor        │
  │   mmap stays cold; bytes     │  │   (full_sd never read on     │
  │   read once per param into   │  │    non-rank-0)               │
  │   GPU; CPU RSS stays ~0.5 GB │  │                              │
  └──────────────────────────────┘  └──────────────────────────────┘
```

The mystery the function name creates — _"why is `_move_missing_keys` called when there are no missing keys"_ — is resolved by Path B: on FSDP non-rank-0 it is not a "missing keys" function at all. It walks `self.named_parameters()` unconditionally and replaces each mmap-backed param with a fresh CPU tensor. With `empty_like` (pre-#45050) this is free — the new tensor is virtual-only and the dropped mmap reference releases page-cache pages back to the kernel. With `zeros_like` (post-#45050), the same loop commits `model_size × (ranks_per_node - 1)` of fresh anonymous pages per node. Removing the params loop entirely (and keeping only the buffer one) is correct because accelerate's `fsdp2_prepare_model` will move the model to meta and broadcast from rank 0 anyway, so the placeholder values on non-rank-0 are written but never read.
