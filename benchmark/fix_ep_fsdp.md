# Fix: EP + FSDP2 corruption on Qwen3-30B-A3B

**Status:** working. End-to-end SFT run with EP=8 + FSDP2 starts at first-step loss within the expected range and trains cleanly. Previously: loss=62 → 0/NaN (catastrophically broken — 7/8 of experts silently destroyed).

The fix lives **entirely in transformers**. Accelerate runs unmodified. TRL runs unmodified. The Trainer's auto-wired logic handles all subclasses (HF Trainer, SFTTrainer, GRPOTrainer, …).

This document captures:

1. The bug and how it manifested.
2. The root cause (proven by a focused diagnostic test).
3. The two pieces of the fix and why each is needed (empirically verified).
4. Exact diffs.
5. Verification tests.
6. How to reproduce / extend.

The repo involved:

- `/fsx/amine_dirhoussi/transformers` (branch `qwen3-moe-ep-v2`)

---

## 1. Symptom and history

### Working April 15, broken April 26+

| Date   | Slurm    | Mesh                     | First-step loss | grad_norm |
| ------ | -------- | ------------------------ | --------------- | --------- |
| Apr 15 | 22078957 | dp=16, tp=1, ep=16       | **1.596**       | 1.24      |
| Apr 15 | 22079140 | dp=16, tp=1, ep=8        | **1.596**       | normal    |
| Apr 26 | 22091902 | dp=2, tp=8, ep=8         | **37.2**        | broken    |
| Apr 26 | 22091929 | dp=2, tp=8, ep=2         | 9.01            | NaN       |
| Apr 26 | 22091987 | dp=4, tp=4, ep=4         | 29.80           | Inf       |
| Apr 27 | 22092267 | dp=2, tp=8, ep=8 + FA3   | 62.0            | NaN       |
| Apr 27 | 22092280 | dp=2, tp=8, ep=8 + DS-Z3 | 62.11           | NaN       |

The April 15 runs used PR [#45473](https://github.com/huggingface/transformers/pull/45473). Subsequent benchmark runs (April 24+) used a YAML pattern with `tp>1` in the accelerate config (to satisfy a `ParallelismConfig.total_size != num_processes` validation triggered by the trainer auto-build of `ParallelismConfig` from `model.tp_size`). That pattern caused the corruption documented below.

### What "broken" looked like

```
{'loss': '62', 'grad_norm': 'nan', 'mfu': '3.085', ...}
{'loss': '0',  'grad_norm': 'nan', 'mfu': '5.755', ...}
{'loss': '0',  'grad_norm': 'nan', 'mfu': '8.092', ...}
```

First step loss = 62 (vs ~2 expected for fresh Qwen3-30B-A3B base, or ~8–10 on real packed text). MFU/TPS look healthy — GPUs are doing real matmul work — but on the wrong weights.

---

## 2. Root cause

`from_pretrained(distributed_config=DistributedConfig(enable_expert_parallel=True))` correctly EP-shards experts across ranks. Each rank holds `(16, 1536, 2048)` for its 16 experts (rank 0: experts 0–15, rank 1: 16–31, … rank 7: 112–127). Verified: per-rank values differ as expected.

`accelerator.prepare(model, optim)` triggers `accelerate.utils.fsdp_utils.fsdp2_prepare_model`. With `cpu_ram_efficient_loading=True`:

1. Snapshot `original_sd = model.state_dict()` — captures rank-local data.
2. `model.to(torch.device("meta"))` — drops parameter values.
3. `fully_shard(model)` — wraps each parameter as a DTensor on the FSDP DP mesh.
4. `fsdp2_load_full_state_dict(accelerator, model, original_sd, ...)` — broadcasts rank 0's snapshot to all ranks, then redistributes per the FSDP placements.

Step 4 is the bug. Rank 0 iterates `original_sd` and calls `dist.broadcast(full_param, src=0)` for each param. For an EP-sharded param, rank 0's `full_param` is `(16, 1536, 2048)` containing **only rank 0's experts 0–15**. Every rank receives that data. After the subsequent `distribute_tensor(...)`, each rank holds a slice of _rank 0's 16 experts_ — not its own EP slice.

Net effect: experts 16–127 silently destroyed. Model has ~16 experts effectively present (sharded across 8 ranks via FSDP), router still picks among 128 IDs, most of those experts missing.

### Definitive evidence (`benchmark/debug_ep_fsdp.py`)

```
[rank 0] BEFORE — local[expert=0,0,0]= 1.031e-02  local[expert=15,0,0]=-2.478e-02
[rank 1] BEFORE — local[expert=0,0,0]=-1.575e-02  local[expert=15,0,0]= 2.429e-02
... (each rank distinct, as expected)

AFTER FSDP — full_tensor() shape: (16, 1536, 2048)
full_after[expert=0, 0, 0]  =  1.031e-02   ← matches rank 0's BEFORE
full_after[expert=15, 0, 0] = -2.478e-02   ← matches rank 0's BEFORE
IndexError: index 16 is out of bounds for dimension 0 with size 16  ← only 16 experts left
```

Reconstructed full tensor after FSDP wrap is `(16, …)`, not `(128, …)`. Only rank 0's 16 experts survived.

---

## 3. The fix — two complementary pieces

The fix has two parts. **Both are necessary** — empirically verified by removing each in isolation:

| Piece                                        | What it does                                                                                                          | Removing it →                                                                                                 |
| -------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| **`ignored_modules`** in Trainer             | Tells FSDP "skip the experts modules" — they're already EP-sharded                                                    | FSDP either crashes (`Invariant encountered: value was None`) or corrupts (broadcast destroys 7/8 of experts) |
| **DTensor wrap** at end of `from_pretrained` | Type-marker so Adam's `_foreach_mul_` doesn't reject mixed `Tensor + DTensor` against the FSDP-wrapped DTensor params | Optimizer crashes: `RuntimeError: aten._foreach_mul_.Scalar: got mixed torch.Tensor and DTensor`              |

The two pieces are orthogonal: `ignored_modules` keeps FSDP from re-sharding the EP params; DTensor wrap keeps the optimizer's foreach ops happy. Neither alone is sufficient.

### Empirical verification of each piece

**With only `ignored_modules`, no DTensor wrap:**

```
RuntimeError: aten._foreach_mul_.Scalar: got mixed torch.Tensor and DTensor,
need to convert all torch.Tensor to DTensor before calling distributed operators!
```

(in `optim.step()`, called from `Adam.step()` → `_foreach_mul_.Scalar`)

**With only DTensor wrap, no `ignored_modules`:**

```
File "torch/utils/_typing_utils.py", line 13, in not_none
    raise TypeError("Invariant encountered: value was None when it should not be")
```

(`fully_shard()` doesn't auto-skip DTensors on a non-FSDP mesh)

**With both** → loss=1.6 → 0.45 over 5 steps on overfit test, loss=8.8 healthy on real packed data.

---

## 4. Exact diffs

All in transformers. Three files.

### 4.1 `transformers/modeling_utils.py`

**Add** the `ep_sharded_param_names` property on `PreTrainedModel`. Recomputed from `tp_plan` on every access — survives parameter replacement during weight loading.

```diff
     @property
     def has_ep(self) -> bool:
         """Whether expert parallelism is enabled for this model."""
         distributed_config = getattr(getattr(self, "config", None), "distributed_config", None)
         return distributed_config is not None and getattr(distributed_config, "enable_expert_parallel", False)

+    @property
+    def ep_sharded_param_names(self) -> list[str]:
+        from .integrations.tensor_parallel import _get_parameter_tp_plan
+
+        if not self.has_ep:
+            return []
+        plan = self.tp_plan
+        return [
+            name
+            for name, _ in self.named_parameters()
+            if _get_parameter_tp_plan(parameter_name=name, tp_plan=plan, is_weight=True) == "grouped_gemm"
+        ]
+
     @property
     def tp_plan(self) -> dict[str, str]:
```

**Add** the DTensor wrap loop right after `_finalize_model_loading` in `from_pretrained` (post-load, post-replacement, before `from_pretrained` returns):

```diff
         loading_info, disk_offload_index = cls._load_pretrained_model(model, state_dict, checkpoint_files, load_config)
         loading_info = cls._finalize_model_loading(model, load_config, loading_info)
         model.eval()  # Set model in evaluation mode to deactivate Dropout modules by default
         model.set_use_kernels(use_kernels, kernel_config)
+
+        # Wrap EP-sharded params as DTensors so the optimizer's foreach ops don't
+        # error with "mixed Tensor and DTensor" against the FSDP-wrapped DTensor params.
+        if model.has_ep:
+            from .integrations.tensor_parallel import _get_parameter_tp_plan
+            from torch.distributed.tensor import DTensor, Shard
+
+            plan = model.tp_plan
+            for name, p in list(model.named_parameters()):
+                if _get_parameter_tp_plan(parameter_name=name, tp_plan=plan, is_weight=True) != "grouped_gemm":
+                    continue
+                parent, attr = get_module_from_name(model, name)
+                dt = DTensor.from_local(p.data, device_mesh, [Shard(0)], run_check=False)
+                setattr(parent, attr, nn.Parameter(dt, requires_grad=p.requires_grad))
```

### 4.2 `transformers/integrations/moe.py`

The `grouped_mm` kernel rejects DTensor weights mixed with regular tensor inputs. Add a one-line helper that pulls the local shard before calling the kernel:

```diff
+from torch.distributed.tensor import DTensor
 ...
+    def _local(p):
+        return p.to_local() if isinstance(p, DTensor) else p
+
     # Select expert weights and biases
     if self.has_gate:
-        selected_weights = self.gate_up_proj
+        selected_weights = _local(self.gate_up_proj)
         ...
     else:
-        selected_weights = self.up_proj
+        selected_weights = _local(self.up_proj)
         ...
-    selected_weights = self.down_proj
+    selected_weights = _local(self.down_proj)
```

The same `_local()` pattern must be applied to `batched_mm_experts_forward` and `sonicmoe_experts_forward` for those kernels to work under EP. (Not done in the current commit — only `grouped_mm` was needed for the verification SFT run.)

### 4.3 `transformers/trainer.py`

Two changes in `create_accelerator_and_postprocess` and the upstream `_build_accelerator_args`:

**(a)** Skip the `ParallelismConfig(tp_size=...)` auto-build when EP is active. Otherwise accelerate's `_prepare_tp` runs, expecting a TP setup that doesn't exist (and triggers an unrelated `ReplicateParallel` import error in the current accelerate-vs-transformers state):

```diff
-        if getattr(self.model, "tp_size", None) is not None and self.model.tp_size > 1:
+        # EP uses `model.tp_size` for its own mesh, but `_prepare_tp` shouldn't run —
+        # EP-sharded params are already DTensors on the EP mesh, not on a TP mesh.
+        if (
+            getattr(self.model, "tp_size", None) is not None
+            and self.model.tp_size > 1
+            and not getattr(self.model, "has_ep", False)
+        ):
             if self.args.parallelism_config is None:
                 if is_accelerate_available("1.12.0"):
                     ...
                     args["parallelism_config"] = ParallelismConfig(tp_size=self.model.tp_size)
```

**(b)** Auto-wire `fsdp_plugin.ignored_modules` from `model.ep_sharded_param_names` after the accelerator is created:

```diff
         # post accelerator creation setup
         if self.is_fsdp_enabled:
             fsdp_plugin = self.accelerator.state.fsdp_plugin
+            # EP-sharded experts must not be re-sharded by FSDP — their params are
+            # already DTensors on the EP mesh.
+            ep_param_names = getattr(self.model, "ep_sharded_param_names", []) or []
+            if ep_param_names:
+                module_names = list({n.rsplit(".", 1)[0] for n in ep_param_names})
+                fsdp_plugin.ignored_modules = [self.model.get_submodule(n) for n in module_names]
             for param in ["limit_all_gathers", "activation_checkpointing"]:
                 setattr(fsdp_plugin, param, self.args.fsdp_config.get(param, getattr(fsdp_plugin, param)))
```

### 4.4 Required runtime config: `cpu_ram_efficient_loading=False` for EP runs

```yaml
fsdp_config:
    fsdp_cpu_ram_efficient_loading: false
```

`benchmark/run_benchmark.py` currently defaults this to `True`. For EP runs it must be `False`.

#### Why `cpu_ram_efficient_loading=True` breaks EP, even with the rest of the fix in place

The whole reason `cpu_ram_efficient_loading=True` exists: in normal (non-EP) FSDP, only rank 0 actually loads checkpoint files into CPU RAM; non-rank-0 ranks have empty placeholders. After `fully_shard()`, rank 0 broadcasts each parameter to all ranks. Total CPU RAM usage is ~1× model size across the node instead of 8×.

For EP, that assumption is invalid: **all ranks have real, per-rank-unique data** (each holds its own 1/8 of experts). Two destructive steps inside `fsdp2_prepare_model` clobber the EP shards:

```python
# accelerate/utils/fsdp_utils.py:fsdp2_prepare_model

original_sd = model.state_dict()  # captures rank's local data (per-rank-unique for EP)

if fsdp2_plugin.cpu_ram_efficient_loading:
    ...
    model = model.to(torch.device("meta"))   # ← (1) drops EP DTensor data
    ...

fully_shard(model, ignored_params=ep_params)  # honors ignored_params,
                                              # but EP params are now meta tensors

if fsdp2_plugin.cpu_ram_efficient_loading:
    fsdp2_load_full_state_dict(accelerator, model, original_sd, ...)   # ← (2) broadcasts
```

(1) `model.to("meta")` is unconditional — runs **before** `fully_shard` looks at `ignored_modules`. Even though FSDP knows to leave EP params alone, their data is already gone after the meta move.

(2) `fsdp2_load_full_state_dict`: rank 0 iterates `original_sd` and calls `dist.broadcast(full_param, src=0)` for every param. Non-rank-0 ranks iterate `meta_sharded_sd = model.state_dict()` and call `distribute_tensor(full_tensor, sharded_param.device_mesh, sharded_param.placements)`. After the meta move, EP params on non-rank-0 ranks are meta tensors with no `device_mesh` attribute — the iteration paths diverge between rank 0 and rank>0 → NCCL deadlock.

Empirical confirmation (April 27, single H100 node, full transformers fix in place, ONLY `cpu_ram_efficient_loading=True`):

```
=== EP=8 single-node correctness test, world_size=8 ===
ignored_modules: 48 experts modules
EP params that are DTensors after from_pretrained: 96/96
... (acc.prepare hangs >55 minutes — killed manually)
```

`from_pretrained` produces 96/96 EP DTensors ✓. Trainer auto-wires `ignored_modules` ✓. Then `acc.prepare(model, optim)` hangs indefinitely — the broadcast deadlock above.

#### Why we don't fix the meta path in this PR

A clean fix routes the EP-DTensor data around the destructive meta move and the broadcast loop:

```python
# Snapshot EP-DTensor data before the meta move
ep_data = {n: p.data.clone() for n, p in model.named_parameters() if p in ignored_params}

# (existing) move to meta, fully_shard with ignored_params
model = model.to(torch.device("meta"))
fully_shard(model, ignored_params=ignored_params)

# Skip ignored params during broadcast (filter `original_sd`)
sd_for_broadcast = {k: v for k, v in original_sd.items() if k not in ep_data}
fsdp2_load_full_state_dict(accelerator, model, sd_for_broadcast, ...)

# Restore EP shards (no broadcast — each rank's own data)
for n, data in ep_data.items():
    parent, attr = get_module_from_name(model, n)
    getattr(parent, attr).data = data.to(accelerator.device)
```

That's an Accelerate-side change (~10 lines). We deliberately keep this PR transformers-only and disable `cpu_ram_efficient_loading` for EP at the YAML level. Routing around the meta move is a clear follow-up for an Accelerate PR.

---

## 5. Verification tests

All run on a single H100 node via a persistent `salloc` allocation:

```bash
salloc --no-shell --partition=hopper-prod --nodes=1 --gres=gpu:h100:8 \
  --ntasks-per-node=1 --exclusive --time=02:00:00 --qos=normal
# returns: salloc: Granted job allocation <JOBID>
```

Subsequent commands use `srun --jobid=<JOBID> --overlap`.

### 5.1 Corruption/preservation test — `benchmark/debug_ep_fsdp.py`

Compares per-rank `gate_up_proj` data BEFORE and AFTER `accelerator.prepare`. Each rank should see its OWN data preserved.

```bash
srun --jobid=<JOBID> --overlap --gres=gpu:h100:8 --ntasks=1 \
  bash -c 'cd /fsx/amine_dirhoussi/trl && source .venv/bin/activate && \
           torchrun --nproc_per_node=8 benchmark/debug_ep_fsdp.py'
```

Expected:

```
✓ EP shards PRESERVED — all 8 ranks still hold their original unique 16-expert slice.
  (Total experts retained across the EP group: 128/128)
```

### 5.2 5-step overfit test — `benchmark/debug_ep_train.py`

Same input each step, expect loss to decrease (overfit on a single batch).

```bash
srun --jobid=<JOBID> --overlap --gres=gpu:h100:8 --ntasks=1 \
  bash -c 'cd /fsx/amine_dirhoussi/trl && source .venv/bin/activate && \
           torchrun --nproc_per_node=8 benchmark/debug_ep_train.py'
```

Expected (April 27, 8 GPUs, EP=8):

```
EP params that are DTensors after from_pretrained: 96/96
acc.prepare done.
step 0: loss=1.6072  grad_norm=3.96
step 1: loss=1.1033  grad_norm=2.59
step 2: loss=0.8117  grad_norm=2.08
step 3: loss=0.6043  grad_norm=1.73
step 4: loss=0.4525  grad_norm=1.60
```

### 5.3 Real SFTTrainer run — full pipeline

Uses `trl/scripts/sft.py` end-to-end with EP=8 and the unmodified accelerate launcher. Each step uses a fresh batch from the shuffled dataloader; loss won't decrease monotonically but must stay finite and within healthy range (~7–12 for a fresh Qwen3-30B-A3B on real packed `THUDM/LongAlign-10k`).

```bash
srun --jobid=<JOBID> --overlap --gres=gpu:h100:8 --ntasks=1 \
  bash -c 'cd /fsx/amine_dirhoussi/trl && source .venv/bin/activate && \
    accelerate launch --num_processes=8 --num_machines=1 \
      --use_fsdp --fsdp_version 2 --fsdp_auto_wrap_policy transformer_based_wrap \
      --fsdp_cpu_ram_efficient_loading false \
      trl/scripts/sft.py \
      --model_name_or_path Qwen/Qwen3-30B-A3B \
      --dataset_name THUDM/LongAlign-10k --max_length 4096 \
      --per_device_train_batch_size 1 --gradient_checkpointing true \
      --bf16 true --dtype bfloat16 --packing --packing_strategy wrapped \
      --max_steps 5 --logging_steps 1 \
      --enable_expert_parallel \
      --output_dir /tmp/sft_ep_out --save_strategy no --report_to none'
```

Expected (April 27, single 8-GPU node):

```
'loss': '8.875', 'grad_norm': '23'
'loss': '9.398', 'grad_norm': '24.25'
'loss': '7.840', 'grad_norm': '31'
'loss': '7.973', 'grad_norm': '25'
'loss': '8.801', 'grad_norm': '28'
```

Loss bounces 7.8–9.4 across 5 different batches, finite grad_norm. Healthy.

### 5.4 What was NOT verified yet

- **`batched_mm_experts_forward` and `sonicmoe_experts_forward`** with the DTensor wrap. Both currently work without EP; will fail under EP+FSDP until they call `.to_local()` on weights. The verification tests above run with the default `grouped_mm` kernel only.
- **Multi-node SFT runs**. `run_benchmark.py` should be updated to default `cpu_ram_efficient_loading: false` when EP is enabled.
- **Combined SP + EP** (DS-Z3 + Ulysses + EP). Has a separate DeepSpeed `train_batch_size != micro_batch * grad_acc * world_size` assertion that needs config wiring.

---

## 6. Upstream paths

Single PR to **transformers**:

- `model.ep_sharded_param_names` property in `modeling_utils.py`
- DTensor wrap loop at end of `from_pretrained`
- `_local()` helper applied to all three expert kernels (`grouped_mm`, `batched_mm`, `sonicmoe`)
- `not has_ep` gate on `ParallelismConfig` auto-build in `Trainer._build_accelerator_args`
- `ignored_modules` auto-wire in `Trainer.create_accelerator_and_postprocess`

**No accelerate PR needed.** **No TRL PR needed.**

---

## 7. How to reproduce this state from scratch

Starting from:

- `transformers @ qwen3-moe-ep-v2` (commit `3c549912f7` — merge of `fix-deepspeed-ep-init` into `qwen3-moe-ep-v2`)
- accelerate 1.13.0+ (no patches)
- `trl @ benchmark-sft-moe` (no patches)

Apply the diffs in section 4 (transformers only). Run the three verification tests in section 5.

For multi-node SFT benchmarks, additionally update `benchmark/run_benchmark.py` to default `cpu_ram_efficient_loading: false` when `extra_args` contains `--enable_expert_parallel`.

---

## 8. Open follow-ups

- Apply `_local()` to `batched_mm_experts_forward` and `sonicmoe_experts_forward`. Both kernel paths fail under EP+FSDP today until they call `.to_local()` on weights.
- File the transformers PR (the five pieces in section 4 + the `_local()` extension to the other two kernels).
- Re-run the full SFT benchmark suite with the fix to capture corrected MFU/TPS numbers (the previous "32% MFU" measurements at EP=8 were on broken models — gradients were garbage but matmul throughput was real, so the kernel-level numbers may still be representative; need a clean rerun to confirm).
- Investigate `cpu_ram_efficient_loading=True` compatibility. The current fix requires it disabled because `model.to("meta")` happens before FSDP sees `ignored_modules`. A clean fix would route the EP params around the meta move entirely.
- Investigate whether torch FSDP2 could natively skip DTensors on non-FSDP meshes (the `Invariant encountered: value was None` crash hit when omitting `ignored_modules`). Removes the need for the `ignored_modules` setup.

---

# Appendix: end-to-end walk-through (concepts → high-level → low-level)

This section walks through the entire flow from `accelerate launch` all the way down to where the bug surfaces. Each concept is introduced before it's used. The intent is for someone unfamiliar with the codebase to follow without prior knowledge of transformers/accelerate/FSDP internals.

## A. Concepts you need first

### A.1 Rank, world, distributed launchers

When you launch a multi-GPU job with `torchrun --nproc_per_node=8 ...` or `accelerate launch --num_processes=8 ...`, **eight separate Python processes** start, one per GPU. They communicate via NCCL collective operations (broadcast, all-reduce, all-gather, etc.). Each process has:

- a **rank**: integer 0..7 identifying which process it is.
- a **world size**: 8 in this example — total number of processes participating in the distributed group.

Every line of Python you'll read below executes **on every rank**. When the code says `if accelerator.is_main_process: ...` or `if rank == 0: ...` it's gating which rank runs that block. Otherwise everyone runs the same code, but with potentially different local data.

### A.2 The "meta" device

`torch.device("meta")` is a fake device. A tensor on meta has **shape and dtype but no actual storage**:

```python
x = torch.zeros(1024, 1024, device="meta")
print(x.shape)   # torch.Size([1024, 1024])  ← shape is real
print(x.dtype)   # torch.float32             ← dtype is real
print(x.sum())   # error: cannot sum a meta tensor — there's no data
```

It's used as a memory-light placeholder when you want to construct the *structure* of a model (modules, parameter shapes) without paying the RAM cost of actually allocating storage.

When you do `model = model.to("meta")`, every parameter in the model has its underlying storage thrown away. The shapes and the parameter tree survive; the data does not.

### A.3 DTensor and device meshes

A **device mesh** is just a labeled grid of ranks: `init_device_mesh("cuda", (8,), mesh_dim_names=("tp",))` declares that 8 ranks form a 1-D group named `tp`. A 2-D mesh `(2, 4)` with names `("dp", "tp")` declares 2 DP groups of 4 TP ranks each.

A **DTensor** is a tensor that knows how it's laid out across a mesh:

```python
from torch.distributed.tensor import DTensor, Shard, Replicate

# Each rank has a 16-element local tensor; the global tensor is 128 elements
# sharded along dim 0 across an 8-rank mesh:
local = torch.arange(16) + 16 * rank  # rank 0: [0..15], rank 1: [16..31], ...
dt = DTensor.from_local(local, mesh, [Shard(0)])

# dt.shape -> (128,)            ← global view
# dt.to_local() -> 16 elements  ← this rank's slice
```

Two placements matter for us:

- `Shard(dim)`: each rank holds a different slice; the global tensor reconstructs by concatenation along `dim`.
- `Replicate()`: every rank holds the same data.

### A.4 FSDP2 in one paragraph

FSDP2 (`torch.distributed.fsdp.fully_shard`) is a way to fit a big model in distributed memory by **sharding its parameters along dim 0 across the FSDP DP mesh**. Conceptually: each rank stores `1/world_size` of every parameter. During forward, FSDP all-gathers the missing pieces of a layer just before that layer runs (so you have the full weight transiently), runs forward, then frees it. Backward does the same and reduce-scatters gradients. The implementation is "wrap the model with `fully_shard()` and forget about it."

`fully_shard(model, ignored_params={p1, p2, ...})` tells FSDP **don't shard these specific parameters** — leave them as plain `nn.Parameter`. They stay fully replicated (or whatever they were before) across the FSDP mesh.

### A.5 Expert Parallelism (EP)

In a Mixture-of-Experts model, the expert weights (here `gate_up_proj`, `down_proj`) are big — 128 experts × hidden size × intermediate size. **EP shards the expert dim across ranks**: with EP=8 and 128 experts, each rank holds 16 experts. Routing decisions are made on every rank (the router is replicated), but only the rank that owns the chosen expert actually computes its output; an all-reduce then sums per-rank contributions into the final hidden state.

For us the key fact is: each rank's `gate_up_proj` is shape `(16, 1536, 2048)` and contains **a different 16-expert slice** than other ranks. Rank 0 has experts 0–15, rank 1 has 16–31, …, rank 7 has 112–127. **Per-rank-unique data, not replicated.**

This is the opposite of what FSDP assumes about its inputs. FSDP assumes "before I shard this parameter, every rank holds the same full tensor"; for EP-sharded params, every rank holds a *different* partial tensor. The bug arises from this mismatch.

### A.6 The optimizer's foreach ops

Adam's `step()` doesn't loop over parameters one by one — it batches them into a single `_foreach_mul_`, `_foreach_add_` etc. call. These ops require **all participating tensors to be the same type**. After `fully_shard()`, FSDP-managed params become `DTensor`s. If EP params stay as plain `nn.Parameter`s, mixing the two in one foreach call errors out:

```
RuntimeError: aten._foreach_mul_.Scalar: got mixed torch.Tensor and DTensor
```

That's why our fix wraps EP params as DTensors too — so the optimizer sees a uniform DTensor population.

---

## B. The high-level entrypoint

The user's command:

```bash
accelerate launch --num_processes=8 --num_machines=1 \
  --use_fsdp --fsdp_version 2 --fsdp_auto_wrap_policy transformer_based_wrap \
  --fsdp_cpu_ram_efficient_loading false \
  trl/scripts/sft.py \
  --model_name_or_path Qwen/Qwen3-30B-A3B \
  --enable_expert_parallel \
  ...
```

`accelerate launch` is a thin wrapper around `torchrun`. It spawns 8 Python processes, each running `trl/scripts/sft.py` with `RANK`, `WORLD_SIZE`, `LOCAL_RANK` env vars set. From here on, we follow what each process does (the same code on every rank, with rank-local data).

---

## C. `trl/scripts/sft.py` — the entrypoint script

This is the user-facing script. Relevant flow (paraphrased):

```python
# trl/scripts/sft.py
def main(script_args, training_args, model_args, dataset_args):
    # 1) Choose how to give the model to the trainer
    if training_args.enable_expert_parallel:
        # Pass the model NAME (a string), let SFTTrainer call from_pretrained
        # internally with the right EP config
        model = model_args.model_name_or_path
        training_args.model_init_kwargs = dict(
            attn_implementation=model_args.attn_implementation,
            dtype=model_args.dtype,
            ...
        )
    else:
        # No EP: load the model object directly here
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            attn_implementation=model_args.attn_implementation,
            dtype=model_args.dtype,
        )

    # 2) Load dataset
    dataset = ...

    # 3) Create the trainer (this is where most of the magic happens)
    trainer = SFTTrainer(
        model=model,                # str (EP) or PreTrainedModel (no EP)
        args=training_args,
        train_dataset=dataset[...],
        ...
    )

    # 4) Run training
    trainer.train()
```

Two things to note:

- For EP runs, the model is passed as a **string** (`"Qwen/Qwen3-30B-A3B"`). `SFTTrainer` will call `from_pretrained` itself, with the right `distributed_config`.
- The actual data flow (load, EP-shard, FSDP-wrap, train) all happens inside `SFTTrainer.__init__` and `trainer.train()`.

---

## D. `trl/trainer/sft_trainer.py` — `SFTTrainer.__init__`

`SFTTrainer` extends `transformers.Trainer`. It does some SFT-specific setup (data collator, packing, etc.) and then calls `super().__init__(...)`. The relevant snippet for EP:

```python
# trl/trainer/sft_trainer.py, simplified
class SFTTrainer(Trainer):
    def __init__(self, model, args, ...):
        # If model is a string and EP is enabled, build the kwargs and load it
        if isinstance(model, str):
            model_init_kwargs = args.model_init_kwargs or {}
            if args.distributed_state.distributed_type in ["MULTI_GPU", "DEEPSPEED"]:
                model_init_kwargs["device_map"] = None
            if args.enable_expert_parallel:
                model_init_kwargs["distributed_config"] = DistributedConfig(
                    enable_expert_parallel=True
                )
                # Set up the EP mesh: dp × ep, where ep_size is the user's flag
                ep_size = getattr(args, "expert_parallel_size", None) or world_size
                if ep_size < world_size:
                    dp_size = world_size // ep_size
                    model_init_kwargs["device_mesh"] = dist.init_device_mesh(
                        "cuda", (dp_size, ep_size), mesh_dim_names=("dp", "tp")
                    )
                else:
                    model_init_kwargs["device_mesh"] = dist.init_device_mesh(
                        "cuda", (world_size,)
                    )
            # Call from_pretrained — this is where loading + EP-sharding happens
            model = create_model_from_path(model, **model_init_kwargs)

        # Hand off to transformers.Trainer
        super().__init__(model=model, args=args, ...)
```

Key takeaway: by the time `super().__init__()` runs, the model is already loaded and EP-sharded. Each rank has its own 16-expert slice in CPU RAM (for EP=8 on Qwen3-30B-A3B).

---

## E. `transformers/modeling_utils.py` — `from_pretrained` (loading + EP wrap)

Inside `create_model_from_path`, transformers' `PreTrainedModel.from_pretrained` runs. The relevant pieces (heavily simplified):

```python
# transformers/modeling_utils.py, simplified
@classmethod
def from_pretrained(cls, model_name_or_path, *, distributed_config=None,
                    device_mesh=None, **kwargs):
    # 1) Build empty model on meta device
    config = AutoConfig.from_pretrained(model_name_or_path)
    config.distributed_config = distributed_config
    model = cls(config)  # parameters are on meta

    # 2) If EP is active, set up the TP/EP plan and hooks
    if distributed_config is not None:
        # initialize_tensor_parallelism() builds the device mesh if not given,
        # and registers per-module hooks for routing/all-reduce
        device_map, device_mesh, tp_size = initialize_tensor_parallelism(
            tp_plan="auto", device_mesh=device_mesh, ...
        )
        model = distribute_model(model, model.tp_plan, distributed_config,
                                 device_mesh, tp_size)

    # 3) Load weights from disk into the model
    #    For EP-marked params (gate_up_proj, down_proj), `convert_and_load_state_dict`
    #    only places this rank's slice — rank 0 gets experts 0..15, rank 1 gets 16..31, etc.
    cls._load_pretrained_model(model, ...)
    cls._finalize_model_loading(model, ...)

    # 4) ★ NEW IN OUR FIX: wrap each EP-sharded param as a DTensor on the EP mesh.
    #    This is purely a type marker — the actual data still lives on the rank locally.
    cls._wrap_ep_params_as_dtensor(model, device_mesh)

    return model
```

Step 3 is where each rank actually fills in its slice of the experts. By the end of step 3 (before our new step 4), every rank has:

- Most params (attention, embeddings, lm_head, layer norms, …): full tensor, **same data on every rank**, dtype bf16, on CPU RAM.
- Expert params (`gate_up_proj`, `down_proj` of every MoE layer): tensor of shape `(16, 1536, 2048)` (for `gate_up_proj`), **different data per rank** — rank 0 has experts 0..15, rank 1 has 16..31, etc.

Step 4 is the new piece we added. It walks just the EP-sharded params and replaces each one's `nn.Parameter` with a new `nn.Parameter` whose `.data` is a `DTensor`. The `DTensor` records that this rank's local tensor (the `(16, ...)` slice) corresponds to a `Shard(0)` placement on the EP device mesh. Concretely:

```python
@staticmethod
def _wrap_ep_params_as_dtensor(model, device_mesh):
    if not model.has_ep:
        return
    for name, p in list(model.named_parameters()):
        plan = _get_parameter_tp_plan(parameter_name=name, tp_plan=model.tp_plan, is_weight=True)
        if plan != "grouped_gemm":
            continue
        parent, attr = get_module_from_name(model, name)
        # The rank's existing local tensor becomes the local shard of a DTensor on the EP mesh
        dt = DTensor.from_local(p.data, device_mesh, [Shard(0)], run_check=False)
        setattr(parent, attr, nn.Parameter(dt, requires_grad=p.requires_grad))
```

After this step:

- The data on each rank is the same as before — no copies, no broadcasts. Just metadata bookkeeping.
- `gate_up_proj.data` is now a `DTensor`, not a plain tensor. `gate_up_proj.data.to_local()` returns the original `(16, 1536, 2048)` per-rank slice.
- The optimizer's foreach ops will be able to mix this with FSDP-wrapped DTensors uniformly (see B.6).

---

## F. `transformers/trainer.py` — `Trainer.__init__` → `create_accelerator_and_postprocess`

Back in `super().__init__(...)` (which is `transformers.Trainer.__init__`), the trainer creates the `Accelerator` object and configures the FSDP plugin. The relevant block:

```python
# transformers/trainer.py — simplified
class Trainer:
    def __init__(self, model, args, ...):
        self.model = model      # already EP-sharded + DTensor-wrapped from from_pretrained
        ...
        self.create_accelerator_and_postprocess()
        ...

    def create_accelerator_and_postprocess(self):
        # Build accelerator kwargs (parallelism_config, fsdp_plugin, etc.)
        args = self._build_accelerator_args(...)
        self.accelerator = Accelerator(**args)
        self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None

        # Post-creation setup
        if self.is_fsdp_enabled:
            fsdp_plugin = self.accelerator.state.fsdp_plugin

            # ★ NEW IN OUR FIX: tell FSDP not to shard the EP experts modules
            ep_param_names = getattr(self.model, "ep_sharded_param_names", []) or []
            if ep_param_names:
                module_names = list({n.rsplit(".", 1)[0] for n in ep_param_names})
                fsdp_plugin.ignored_modules = [
                    self.model.get_submodule(n) for n in module_names
                ]
            ...
```

`fsdp_plugin.ignored_modules` is an existing accelerate API. When FSDP later wraps the model, accelerate translates `ignored_modules` into `ignored_params` for `fully_shard()`, and FSDP leaves those parameters alone.

There's a second related change in `_build_accelerator_args`. By default, transformers builds a `ParallelismConfig(tp_size=model.tp_size)` whenever `model.tp_size > 1`. For EP-active models this would trick accelerate into running its TP-prep path, which is wrong (EP isn't TP). We gate this on `not has_ep`:

```python
# transformers/trainer.py — simplified
def _build_accelerator_args(self, ...):
    args = {...}
    if (getattr(self.model, "tp_size", None) is not None
            and self.model.tp_size > 1
            and not getattr(self.model, "has_ep", False)):  # ★ NEW gate
        if self.args.parallelism_config is None:
            args["parallelism_config"] = ParallelismConfig(tp_size=self.model.tp_size)
    return args
```

After `create_accelerator_and_postprocess` returns:

- `self.model` is still the EP-sharded model from `from_pretrained` (with DTensor-wrapped expert params).
- `self.accelerator` is configured. `self.accelerator.state.fsdp_plugin.ignored_modules` lists every `mlp.experts` module — FSDP will skip them.
- **Nothing has actually been wrapped yet.** `accelerator.prepare(model, optim)` is what runs `fully_shard()`.

---

## G. `accelerator.prepare(model, optim)` — actual FSDP wrap

The call site is inside `trainer.train() → trainer._inner_training_loop() → ... → accelerator.prepare(model, optim)`.

`Accelerator.prepare` dispatches to `fsdp2_prepare_model` (since the FSDP plugin is FSDP2). That function lives in **accelerate**, not transformers. This is where the bug surfaces. Two paths, depending on `cpu_ram_efficient_loading`.

### G.1 The "OK" path: `cpu_ram_efficient_loading=False`

```python
# accelerate/utils/fsdp_utils.py — fsdp2_prepare_model, simplified
def fsdp2_prepare_model(accelerator, model):
    fsdp2_plugin = accelerator.state.fsdp_plugin
    fsdp2_kwargs = {...}

    # Translate ignored_modules → ignored_params for fully_shard
    if fsdp2_plugin.ignored_modules is not None:
        fsdp2_kwargs["ignored_params"] = get_parameters_from_modules(
            fsdp2_plugin.ignored_modules, model, accelerator.device
        )

    # ↓↓↓ skipped when cpu_ram_efficient_loading=False ↓↓↓
    # if fsdp2_plugin.cpu_ram_efficient_loading:
    #     model = model.to(torch.device("meta"))   # would destroy EP data
    # ↑↑↑

    # Actually wrap the model
    fully_shard(model, **fsdp2_kwargs)

    # ↓↓↓ also skipped ↓↓↓
    # if fsdp2_plugin.cpu_ram_efficient_loading:
    #     fsdp2_load_full_state_dict(...)   # would broadcast rank 0's data to all
    # ↑↑↑

    return model
```

What happens to the EP params during `fully_shard(...)` here? They're in `ignored_params`, so `fully_shard` doesn't touch them. They stay as `nn.Parameter`s with `.data` being a `DTensor` on the EP mesh — exactly as our `_wrap_ep_params_as_dtensor` left them.

What happens to the rest? FSDP wraps each transformer block, replacing each non-EP parameter's `.data` with a `DTensor` on the FSDP DP mesh, with `Shard(0)` placement. **No data movement at this point** — `fully_shard` records the placement and trims later as needed during forward.

End state: every parameter on every rank has `isinstance(p.data, DTensor) == True` (either on the EP mesh for experts, or on the FSDP mesh for everything else). The optimizer can run its foreach ops uniformly. Training works. ✓

### G.2 The "broken" path: `cpu_ram_efficient_loading=True`

This flag exists for a real reason: in the **non-EP** case, only rank 0 has actually loaded the full weights into CPU RAM. Ranks 1..7 have empty placeholder tensors. (`from_pretrained` skips the weight load on non-rank-0 ranks when the FSDP plugin advertises this flag.) Total CPU RAM is ~1× model size across the node instead of 8×. Rank 0 then broadcasts the actual values during `fsdp2_prepare_model`.

For EP this assumption is wrong — every rank loaded **its own** slice. But accelerate doesn't know that, and runs the same flow:

```python
# accelerate/utils/fsdp_utils.py — fsdp2_prepare_model with cpu_ram_efficient_loading=True
def fsdp2_prepare_model(accelerator, model):
    ...
    if fsdp2_plugin.ignored_modules is not None:
        fsdp2_kwargs["ignored_params"] = get_parameters_from_modules(...)

    # 1) Snapshot the state dict BEFORE any destructive op
    original_sd = model.state_dict()

    # 2) Drop ALL parameter values: every param becomes a meta tensor
    if fsdp2_plugin.cpu_ram_efficient_loading:
        model = model.to(torch.device("meta"))
        if hasattr(model, "tie_weights"):
            model.tie_weights()

    # 3) Wrap the model. fully_shard honors ignored_params, but ALL params
    #    (including ignored ones) are now meta-backed because step 2 nuked them.
    fully_shard(model, **fsdp2_kwargs)

    # 4) Re-fill the meta tensors with real data via a broadcast loop
    if fsdp2_plugin.cpu_ram_efficient_loading:
        fsdp2_load_full_state_dict(accelerator, model, original_sd, ...)

    return model
```

For each step, here's what actually happens with our EP setup:

**Step 1 — snapshot:** `original_sd` is built per rank. For EP params, rank 0's snapshot has the (16, ...) slice for experts 0..15; rank 1's snapshot has experts 16..31; etc. **Same key, different data per rank.**

**Step 2 — meta move:** Every parameter on every rank loses its storage. EP DTensors lose their underlying tensor data too — even though we marked them as `ignored_modules`, that hint hasn't been used yet (it gets used in step 3). So the meta move is **unconditional**: the data is gone on every rank, including the EP slices.

**Step 3 — `fully_shard`:** `ignored_params` is honored. Non-EP params get DTensor-wrapped on the FSDP mesh (still meta-backed; will be filled in step 4). EP params stay as plain `nn.Parameter`s — but their `.data` is meta. They have no `device_mesh` attribute (because FSDP didn't touch them).

**Step 4 — broadcast loop:** This is where the deadlock happens. `fsdp2_load_full_state_dict` looks like this:

```python
# accelerate/utils/fsdp_utils.py — fsdp2_load_full_state_dict, simplified
def fsdp2_load_full_state_dict(accelerator, model, full_sd, ...):
    meta_sharded_sd = model.state_dict()   # all params, meta-backed

    if accelerator.is_main_process:   # rank 0
        for (param_name, full_param), sharded_param in zip(
            full_sd.items(),                  # rank 0's snapshot
            meta_sharded_sd.values()
        ):
            device_mesh = sharded_param.device_mesh   # ← KEY LINE
            full_param = full_param.detach().to(device_mesh.device_type)
            dist.broadcast(full_param, src=0, group=dist.group.WORLD)   # rank 0 sends
            sharded_tensor = distribute_tensor(full_param, device_mesh,
                                              sharded_param.placements)
            ...
    else:   # ranks 1..7
        for param_name, sharded_param in meta_sharded_sd.items():
            device_mesh = sharded_param.device_mesh   # ← KEY LINE
            full_tensor = torch.empty(sharded_param.size(),
                                     device=device_mesh.device_type, ...)
            dist.broadcast(full_tensor, src=0, group=dist.group.WORLD)   # receivers
            ...
```

Notice the line `device_mesh = sharded_param.device_mesh`. This attribute exists on `DTensor`s (the FSDP-managed params after step 3) but **not on plain `nn.Parameter`s** (the EP params, which FSDP skipped).

When ranks 1..7 hit an EP param in their loop, `sharded_param.device_mesh` raises `AttributeError`. That rank exits the loop early or crashes. Meanwhile rank 0 keeps issuing `dist.broadcast(...)` calls for params that ranks 1..7 are no longer expecting. NCCL is a strict synchronous group operation — if some ranks aren't on the matching collective, it blocks indefinitely. After 10 minutes (default NCCL timeout) it should error, but in our test we observed the process hung for >55 minutes without exit. Either way, training never starts.

This is the deadlock. The fix would be to skip EP params in BOTH branches of the loop, AND restore their per-rank-local data outside the broadcast (since broadcasting rank 0's slice would corrupt other ranks' experts even if the loop didn't deadlock).

---

## H. The clean Accelerate-side fix (deferred)

A ~10-line patch to `accelerate.utils.fsdp_utils.fsdp2_prepare_model` would let `cpu_ram_efficient_loading=True` coexist with EP. The idea: capture the EP data **before** step 2 (the meta move), keep it aside per rank, and restore it **after** step 4. The broadcast loop also needs to skip EP params on both branches.

```python
# accelerate/utils/fsdp_utils.py:fsdp2_prepare_model — proposed patch
def fsdp2_prepare_model(accelerator, model):
    ...
    if fsdp2_plugin.ignored_modules is not None:
        fsdp2_kwargs["ignored_params"] = get_parameters_from_modules(...)
    ignored_params = fsdp2_kwargs.get("ignored_params") or set()

    original_sd = model.state_dict()

    # ★ Capture per-rank-local data for ignored params
    ignored_data = {n: p.data.clone()
                    for n, p in model.named_parameters() if p in ignored_params}

    if fsdp2_plugin.cpu_ram_efficient_loading:
        model = model.to(torch.device("meta"))   # destroys everything (including EP)

    fully_shard(model, **fsdp2_kwargs)

    if fsdp2_plugin.cpu_ram_efficient_loading:
        # ★ Filter out ignored params from the broadcast loop's input on rank 0
        sd_for_broadcast = {k: v for k, v in original_sd.items() if k not in ignored_data}
        fsdp2_load_full_state_dict(accelerator, model, sd_for_broadcast, ...)

    # ★ Restore ignored params from each rank's own snapshot — no broadcast
    for name, data in ignored_data.items():
        parent, attr = get_module_from_name(model, name)
        getattr(parent, attr).data = data.to(accelerator.device)

    return model
```

This would also require `fsdp2_load_full_state_dict` to filter non-DTensor params on its rank>0 branch (currently it iterates `meta_sharded_sd.items()` blindly), so the iteration counts stay in sync with the filtered rank-0 dict. ~3 more lines.

We did NOT include this in the transformers PR — it belongs upstream in accelerate. For now: set `cpu_ram_efficient_loading=False` for EP runs and the rest of the fix Just Works.

---

## I. Putting it all together: the full call chain at a glance

```
accelerate launch sft.py --enable_expert_parallel
        │
        ▼
torchrun-equivalent → 8 Python processes start, each runs sft.py
        │
        ▼
trl/scripts/sft.py:main()
        ├── if enable_expert_parallel: model = "Qwen/Qwen3-30B-A3B" (string)
        ├── trainer = SFTTrainer(model=...)
        └── trainer.train()
        │
        ▼
SFTTrainer.__init__()
        ├── if isinstance(model, str): build distributed_config + EP device_mesh
        ├── model = create_model_from_path(name, distributed_config=..., device_mesh=...)
        │           │
        │           ▼
        │     transformers/modeling_utils.py: PreTrainedModel.from_pretrained()
        │           ├── build empty model on meta
        │           ├── initialize_tensor_parallelism() + distribute_model()
        │           ├── _load_pretrained_model()      ← rank-local EP slices loaded into CPU RAM
        │           ├── _finalize_model_loading()
        │           └── ★ _wrap_ep_params_as_dtensor()  ← marks experts as DTensors on EP mesh
        │
        └── super().__init__(model=loaded_model)
                │
                ▼
        transformers/trainer.py: Trainer.__init__()
                ├── _build_accelerator_args()
                │     └── ★ skip ParallelismConfig auto-build when has_ep=True
                ├── self.accelerator = Accelerator(...)
                └── if is_fsdp_enabled:
                      ├── fsdp_plugin = self.accelerator.state.fsdp_plugin
                      └── ★ fsdp_plugin.ignored_modules = [<all experts modules>]
        │
        ▼
trainer.train() → ... → accelerator.prepare(model, optim)
        │
        ▼
accelerate/utils/fsdp_utils.py: fsdp2_prepare_model()
        ├── if cpu_ram_efficient_loading=False (REQUIRED for EP):
        │     └── fully_shard(model, ignored_params=ep_params)   ← clean path
        └── if cpu_ram_efficient_loading=True:
              ├── original_sd = model.state_dict()
              ├── model.to("meta")           ← destroys EP data
              ├── fully_shard(...)
              └── fsdp2_load_full_state_dict ← deadlocks because EP params have no
                                              device_mesh on rank>0 branch
```

The three ★ lines are our changes (in transformers). Everything else exists already and is unchanged.
