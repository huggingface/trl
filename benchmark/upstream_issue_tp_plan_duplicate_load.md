# Draft upstream issue — `tp_plan` loader duplicates non-plan params across ranks

Filed against: **`huggingface/transformers`**. (Accelerate has a related but smaller follow-up; transformers is where the root cause lives.)

Suggested title: **TP/EP `from_pretrained` duplicates non-plan params on every rank → N× GPU memory + N× disk I/O at load time**

---

## Summary

When `from_pretrained` is called with any of `tp_plan`, `tp_size`, or `device_mesh` (the TP/EP-aware loading path), `initialize_tensor_parallelism` unconditionally sets `device_map = cuda:LOCAL_RANK` for every rank. The downstream loader (`_materialize_copy` via `spawn_materialize`) then copies each parameter from the safetensors mmap to `cuda:LOCAL_RANK` on every rank — including the **non-plan parameters that are replicated, not sharded** (embeddings, attention projections, norms, router gates).

Net effect on a 16-rank Qwen3-30B-A3B EP=8 run:
- ~46 GiB of duplicated dense-replica VRAM across the cluster (2.87 GiB × 16 ranks) before FSDP/DS ever runs.
- ~960 GB of duplicated FSx reads at load time (every rank reads the full ~60 GB checkpoint).

For Qwen3-235B-A22B at 128 ranks: ~640 GiB of duplicated VRAM, ~7.5 TB of duplicate disk I/O. The 25+ minute load stalls observed at scale (sft, 235B, 16 nodes, EP=8) trace directly to this.

`FSDP_CPU_RAM_EFFICIENT_LOADING=True` does not fix it: the rank-0-only short-circuits in `modeling_utils.py:4654` and `:4690` only gate `_move_missing_keys_from_meta_to_cpu_or_disk` and `_initialize_missing_keys`. The actual disk loading in `core_model_loading.py:convert_and_load_state_dict_in_model` has no rank-0 short-circuit. Verified empirically: under EP=4 on Qwen3-30B-A3B, identical 2.87 GiB-per-rank dense replica with and without `FSDP_CPU_RAM_EFFICIENT_LOADING=True ACCELERATE_USE_FSDP=True`.

## Repro

```bash
# Two prints, one per rank, will show every rank loaded the same dense param to its own GPU.
torchrun --standalone --nproc-per-node=4 - <<'PY'
import os, torch, torch.distributed as dist
from transformers import AutoModelForCausalLM
from transformers.distributed.configuration_utils import DistributedConfig

dist.init_process_group(backend="nccl")
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
mesh = dist.init_device_mesh("cuda", (4,), mesh_dim_names=("tp",))
model = AutoModelForCausalLM.from_pretrained(
    "trl-internal-testing/tiny-Qwen3MoeForCausalLM",
    dtype=torch.bfloat16,
    distributed_config=DistributedConfig(enable_expert_parallel=True),
    device_mesh=mesh,
)
rank = int(os.environ["RANK"])
dense = next(p for n, p in model.named_parameters() if "self_attn.q_proj" in n)
for r in range(4):
    if r == rank:
        print(f"[rank {rank}] dense q_proj device={dense.device} shape={tuple(dense.shape)}")
    dist.barrier()
PY
```

Output: every rank prints its own `cuda:N` for the same dense q_proj (replicated, not sharded). In a system that respected the TP plan, only rank 0 should hold the dense values pre-FSDP-shard / pre-DS-broadcast.

## Root cause

`transformers/integrations/tensor_parallel.py:initialize_tensor_parallelism`:

```python
if device_mesh is None:
    ...
    tp_device = torch.device(device_type, local_rank)
    device_map = tp_device                   # ← unconditional cuda:LOCAL_RANK
else:
    device_map = torch.device(f"{device_mesh.device_type}:{int(os.environ['LOCAL_RANK'])}")
return device_map, device_mesh, tp_size
```

`device_map[""] = cuda:LOCAL_RANK` then propagates everywhere through `core_model_loading.py:convert_and_load_state_dict_in_model:1377`:

```python
if future_or_tensor is None:
    param_device = get_device(device_map, renamed_key, valid_torch_device=True)
    future_or_tensor = spawn_materialize(thread_pool, tensor, param_device, _dtype)
```

`spawn_materialize → _materialize_copy → tensor[...].to(device=cuda:N)` — every rank, every non-plan param.

## Proposed fix

The TP/EP plan distinguishes two classes of params:

- **Plan params** (sharded across the TP/EP dim — e.g. EP-sharded experts, TP column-/row-parallel linears): each rank legitimately wants its own slice on its own GPU.
- **Non-plan params** (dense replica — embeddings, attention projections, norms): every rank ends up with the same bytes after load. Only rank 0 needs to actually pull them from disk; the rest can be filled later by FSDP's broadcast / DS's `_broadcast_model` / direct `dist.broadcast`.

Two patches:

**Patch 1 (transformers, this issue)** — rank-0 gate in `convert_and_load_state_dict_in_model`. When loading non-plan params on rank > 0 in a distributed setup, leave them on `meta`. Plan params (EP-sharded, TP-sharded) keep loading per-rank as today.

```python
# core_model_loading.py near line 1375
if future_or_tensor is None:  # non-plan param
    if dist.is_initialized() and dist.get_rank() != 0:
        # Leave on meta; FSDP broadcast / DS _broadcast_model / explicit broadcast fills later.
        future_or_tensor = (lambda t=tensor: torch.empty(t.shape, dtype=_dtype or t.dtype, device="meta"))
    else:
        param_device = get_device(device_map, renamed_key, valid_torch_device=True)
        future_or_tensor = spawn_materialize(thread_pool, tensor, param_device, _dtype)
```

The exact placeholder threading needs care so `set_param_for_module` still attaches a tensor that `fully_shard` and DS-Z2/Z3 can see.

**Patch 2 (accelerate, follow-up issue)** — make `fsdp2_prepare_model` EP-aware so the rank-0 broadcast doesn't clobber the EP-sharded experts. Sketch in trl-internal `benchmark/upstream_todo.md` C2.

## Why this is bigger than `cpu_ram_efficient_loading`

The flag's intended behavior is exactly this fix — only rank 0 loads, others fill via broadcast — but only for the FSDP+meta-init path, gated on `is_fsdp_enabled()` (which requires `ACCELERATE_USE_FSDP=True` AND `FSDP_CPU_RAM_EFFICIENT_LOADING=True`). Two issues:

1. The flag's gating short-circuits don't cover `convert_and_load_state_dict_in_model` itself — so even when set, the disk reads happen on every rank (Layer 1 of the bug).
2. Under EP, the flag's later broadcast clobbers the per-rank EP-distinct expert slices (Layer 2 — separate but related).

Most users running EP today set `cpu_ram_efficient_loading=False` to avoid Layer 2. That keeps Layer 1 on indefinitely. The proposed Patch 1 fixes Layer 1 universally — for any TP/EP user, regardless of `cpu_ram_efficient_loading`, regardless of FSDP vs DS — by making the rank-0 gate live where the load actually happens.

## Impact

| Workload | Today | After Patch 1 |
| --- | --- | --- |
| 30B-A3B EP=8, 2 nodes | 45.92 GiB duplicate dense / 960 GB FSx I/O | ~3 GiB load (rank 0 only) + broadcast |
| 235B-A22B EP=8, 16 nodes | ~640 GiB duplicate dense / 7.5 TB FSx I/O | ~5 GiB load + broadcast |

Per-rank load-time peak GPU memory drops from "full dense replica + EP slice" to "just EP slice." For 30B-A3B EP=4 specifically: 16.4 GiB → ~13.5 GiB (verified locally with a manual rank-0 patch).

## Cross-references

- transformers `_move_missing_keys_from_meta_to_device` Path B (`modeling_utils.py:4622`) already implements the "non-rank-0 leaves params blank, gets them via FSDP broadcast" half of this idea — but only AFTER the duplicate disk reads in `convert_and_load_state_dict_in_model` already happened. This issue is "move that gate up one level."
- `is_fsdp_enabled()` (`integrations/fsdp.py:44`): the gate function we'd consult on the loader side.
- `benchmark/fsdp2_loading_walkthrough.md` (trl-internal): full RSS / page-cache analysis of the canonical FSDP loading path, which describes how the load-time RSS *would* look if Patch 1 were in place.
