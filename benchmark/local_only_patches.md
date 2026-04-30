# Local-only patches (not in any open PR)

Everything below lives only in the local environment (working tree of `/fsx/amine_dirhoussi/transformers` or in installed `accelerate`/`deepspeed`). Pulling these into upstream PRs is either out of scope, premature, or actively undesirable. Do not commit/push without re-reading the rationale.

Things already covered by open PRs (and therefore **not** in this list):

- `core_model_loading.py`, `integrations/tensor_parallel.py`, `integrations/moe.py` (most of it), `modeling_utils.py:has_ep`, `trainer.py:not has_ep` gate + `ignored_modules` wire — all in **#45662** (post-`9c712a5` refactor).
- `integrations/hub_kernels.py` (sonic-moe → `IlyasMoutawwakil/sonic-moe@main`) — in **#45621**.
- DS-Z3 + EP loading via accelerate launch — already merged into `qwen3-moe-ep-v2` from **#45548**.
- FSDP2 cpu_ram_efficient_loading OOM — already applied locally from **#45649**.

---

## 1. `transformers/integrations/sonicmoe.py` — EP/DTensor support for the sonicmoe kernel

**What changed (3 hunks):**

```python
# top of file
from torch.distributed.tensor import DTensor
```

```python
# inside sonicmoe_experts_forward, right after expert_ids/router_scores flattening:
invalid_mask = expert_ids >= self.num_experts
expert_ids = expert_ids.clamp(0, self.num_experts - 1)
router_scores = router_scores.masked_fill(invalid_mask, 0.0)
```

```python
# replace direct .permute() with .to_local() then .permute():
gate_up_proj = self.gate_up_proj.to_local() if isinstance(self.gate_up_proj, DTensor) else self.gate_up_proj
down_proj = self.down_proj.to_local() if isinstance(self.down_proj, DTensor) else self.down_proj
w1 = gate_up_proj.permute(*perm)
w2 = down_proj.permute(*perm)
```

**Why:**

- After PR #45662 wraps EP-sharded experts as `DTensor`s on the EP mesh, the custom CuteDSL kernel (`moe_general_routing_inputs`) doesn't accept DTensor inputs. Forward needs the local tensor — `.to_local()`.
- After `.to_local()`, the kernel's hand-written `Function.backward` produces NaN gradients on EP sentinels (`expert_ids >= num_experts`). The sentinel-skip path that #45621 designed works for `torch._grouped_mm` (a native autograd-aware op) but not for the kernel's hand-written backward going through a DTensor wrapper.
- Wrapper-level clamp + zeroing `router_scores` for sentinels makes the kernel do compute on those rows but the contribution is multiplied by 0 — net zero. ~2 pp MFU cost vs the kernel-native skip but trains correctly.

**Status as PR target:** This is the natural follow-up PR after both #45621 and #45662 land. Should cite both and explain why the wrapper clamp is the workaround for the kernel's autograd-through-DTensor.to_local backward gap.

---

## 2. `transformers/integrations/tensor_parallel.py` — `TRANSFORMERS_SKIP_EP_DTENSOR_WRAP=1` env-var bypass

**What changed:** added 5 lines to `GroupedGemmParallel.post_shard_wrap`:

```python
import os
if os.environ.get("TRANSFORMERS_SKIP_EP_DTENSOR_WRAP", "") == "1":
    return param
```

**Why:** debug/bisect tool. Used once to test "is the DTensor wrap what's breaking sonicmoe?" — confirmed it is, but Adam's `_fused_adamw_` mixed-types check makes the wrap mandatory anyway.

**Status as PR target:** No. This is debug-only and shouldn't ship. Remove before any PR.

---

## 3. `transformers/trainer.py` — `_clip_grad_norm` skip when `has_ep`

**What changed:** in `Trainer._clip_grad_norm`:

```python
def _clip_grad_norm(self, model):
    if is_sagemaker_mp_enabled() and self.args.fp16:
        return self.optimizer.clip_master_grads(self.args.max_grad_norm)
    if getattr(self.model, "has_ep", False):
        return torch.tensor(0.0, device=self.args.device)
    return self.accelerator.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
```

**Why:**

- After PR #45662, EP-sharded gradients are DTensors on the EP mesh; non-EP gradients are DTensors on the FSDP DP mesh. `accelerator.clip_grad_norm_` calls `torch._foreach_norm` which stacks per-param norms. Stacking DTensors on different meshes errors with `RuntimeError: All operands in aten.stack.default must have the same mesh`.
- Concrete cases: every FSDP2 + EP=8 + CP=N config (32k/64k/128k CP+EP) and FSDP2 + EP < DP_size (`dp=2 tp=8 ep=8`).
- Skipping returns a 0 grad-norm for telemetry; it loses the actual gradient clipping (i.e., gradients aren't clipped to `max_grad_norm`). That's fine for benchmarking but unsafe for real training.

**Status as PR target:** No. The proper fix is upstream in PyTorch's `clip_grad_norm_` (or accelerate's wrapper) to handle DTensors on mismatched meshes — not in transformers' Trainer. Keep local for benchmark runs only.

---

## 4. `accelerate/accelerator.py` — `_prepare_tp` skip when `has_ep`

**File:** `/fsx/amine_dirhoussi/trl/.venv/lib/python3.11/site-packages/accelerate/accelerator.py`

**What changed:** 5 lines added inside `_prepare_tp`, right after the existing "no DTensor → skip" guard (around line 1601):

```python
# EP-only models: experts are DTensors on the EP mesh (not the TP mesh accelerate
# is about to set up). Skip TP preparation — non-EP params stay plain tensors and
# FSDP2 handles them on the FSDP mesh, the same path that worked before EP params
# became DTensors.
if getattr(model, "has_ep", False):
    return result
```

**Why:**

- Pre-fix: EP params were plain `nn.Parameter` (the bug PR #45662 fixes). The `if not any(isinstance(p, DTensor) for p in model.parameters()): skip` guard fired → `_prepare_tp` returned without doing anything → run proceeded. The skip was firing *naturally* because no DTensors existed.
- Post-fix: EP params are DTensors → the existing skip doesn't fire → `_prepare_tp` reaches `from transformers.integrations.tensor_parallel import ReplicateParallel` (line 1616 of accelerator.py) → **`ImportError`** because that class doesn't exist in our `qwen3-moe-ep-v2` transformers fork (it was added upstream after the fork point).
- The `has_ep` skip restores the pre-fix flow: `_prepare_tp` returns early, FSDP2 wraps non-EP params on the FSDP DP mesh, EP params stay on the EP mesh — same as what worked before.

**Status as PR target:** Either: (a) add a `ReplicateParallel` shim/alias to the transformers fork (small but ugly — papers over the version mismatch), or (b) push the same `has_ep` skip into accelerate upstream. (b) is cleaner. Out of scope for the EP+FSDP correctness PR (#45662 only touches transformers); should be a separate accelerate PR if/when it's worth pushing.

**Reverting:** `pip install --force-reinstall accelerate`

---

## 5. DeepSpeed — *all reverted, but the path forward is real*

Initially patched 3 sites in DS-Z3 to skip DTensor params:

- `engine.py:_broadcast_model.is_replicated` — skip if `isinstance(p.data, DTensor)`
- `runtime/zero/partition_parameters.py:_convert_to_zero_parameters` — skip DTensor params
- `runtime/zero/parameter_offload.py:mark_persistent_parameters` — skip DTensor params

**All reverted.** The patch-each-site approach was the wrong abstraction — DS-Z3 has 30+ access sites for `param.ds_*` / `partition_numel` and skipping each is not tractable.

**The right abstraction (not yet implemented, documented for future work):** DeepSpeed has a complete first-class MoE/EP infrastructure in `deepspeed/moe/` that we missed. The convention is `param.allreduce = False` + `param.group_name = "ep_size_N"`. With these markers set, DS routes the param through the MoE path:

- `_broadcast_model` broadcasts inside `expert_data_parallel_group[group_name]` (DP-of-EP), not the full DP group.
- ZeRO-3 partitions inside `expert_data_parallel_group`, not across all ranks.
- Optimizer puts MoE params in a separate group (`configure_moe_param_groups`).
- `expert_parallel_group` is the all-to-all comm group for MoE token routing.

So the right DS-Z3+EP fix is **tagging not skipping**: after transformers' EP `partition_tensor` runs, mark the EP-sharded params with `allreduce=False` + `group_name`, and make sure DS's `expert_parallel_group` / `expert_data_parallel_group` are constructed with matching names. Likely also gate `_wrap_ep_params_as_dtensor` on backend (skip the DTensor wrap under DS-Z3, since DS doesn't need it — its MoE path expects plain tensors and handles the mixed-foreach issue via the separate optimizer group).

**Status as PR target:** Worth doing eventually — DS-Z3+SP+EP would compound 19% (SP) with EP's expert sharding for long-context MoE. Out of scope for the current EP+FSDP correctness PRs. Full notes in `debug_sp_ep_sonic.md` (Iteration 4 section).

**Workaround we ship today:** SP runs use DeepSpeed without EP (no DTensor params anywhere). EP runs use FSDP2 only. Stock DeepSpeed in the venv.

### Minimal-changes sketch for DS-Z3 + EP (3 patches, all transformers-side)

After reading `deepspeed/moe/{experts,layer,utils}.py` and `deepspeed/utils/groups.py`, the integration is small. We're not skipping or shimming DS — we're tagging EP params with the same convention DS already uses for its native `MoE` layer:

**Patch A (transformers, ~5 lines):** in `tensor_parallel.py:GroupedGemmParallel.post_shard_wrap`, branch on backend:

```python
def post_shard_wrap(self, param: nn.Parameter) -> nn.Parameter:
    from ..integrations import is_deepspeed_zero3_enabled
    if is_deepspeed_zero3_enabled():
        # Route via DS's MoE path: tag the param so is_moe_param(p) is True.
        # DS will broadcast inside expert_data_parallel_group, partition only across DP-of-EP,
        # and put it in a separate optimizer group — same as deepspeed.moe.layer.Experts does.
        param.allreduce = False
        param.group_name = f"ep_size_{self.device_mesh.size()}"
        return param  # plain tensor; DS's MoE path expects this
    # FSDP path: wrap as DTensor so Adam's foreach doesn't reject mixed Tensor/DTensor lists.
    dt = DTensor.from_local(param.data, self.device_mesh, [Shard(0)], run_check=False)
    return nn.Parameter(dt, requires_grad=param.requires_grad)
```

**Patch B (transformers, ~5 lines):** create DS's `expert_parallel_group` / `expert_data_parallel_group` before `deepspeed.initialize()` runs. Right place is `Trainer.create_accelerator_and_postprocess`, before the `Accelerator()` instantiation, gated on `model.has_ep and is_deepspeed_zero3_enabled()`:

```python
if getattr(self.model, "has_ep", False) and is_deepspeed_zero3_enabled():
    from deepspeed.utils import groups as ds_groups
    if not dist.is_initialized():
        # Already initialized by accelerate, just be defensive
        ...
    ds_groups._create_expert_and_data_parallel(self.model.tp_size)
```

This must run after `torch.distributed` is initialized (so the new groups can be created) and before `deepspeed.initialize` (which calls `_broadcast_model` and inspects `is_moe_param`).

**Patch C (transformers, 1 line):** also gate the existing `_clip_grad_norm` `has_ep` skip on `not is_deepspeed_zero3_enabled()` — DS's optimizer handles MoE grad norms via its own per-group path; it doesn't hit the cross-mesh stack issue. Under DS, let the standard path run.

**Why this is minimal:**

- No DS internals patched.
- No new abstraction. We use DS's existing `is_moe_param` convention (set in `deepspeed/moe/experts.py` for native MoE layers; we just set the same attributes on our EP params).
- The DTensor wrap under FSDP stays — it's still needed there for Adam mixed-types.
- The fix in #45548 (load EP via the standard non-zero3 path) already wired the loading; this builds on it.

**Conceptual note — why DS and transformers EP don't conflict at runtime:**

Transformers' EP **does not do all-to-all token dispatch**. It does *replicate-tokens-then-mask*: every rank holds the full token batch, the router masks routing scores for non-local experts (so those tokens contribute 0 on this rank), each rank's kernel computes only its local-expert contributions (with sentinel-skip), and a single `all_reduce` on the EP mesh at the end (via `GatherParallel`) sums partial outputs across EP ranks. Comm pattern is `replicate → mask → kernel → 1 all-reduce on model.tp mesh`, not `all-to-all → compute → all-to-all`.

DS's two MoE groups have very different purposes when transformers owns the forward:
- `expert_parallel_group` (DS's all-to-all comm group) — **dead code** for us. Transformers doesn't do all-to-all and the final `all_reduce` uses transformers' own device mesh, not DS's group object. The group exists in `dist`'s registry as harmless overhead.
- `expert_data_parallel_group` (DP-of-EP) — **load-bearing**. ZeRO-3 partitions inside this small group instead of across the full world, which is the correct behavior for already-EP-sharded experts. This is the only group that actually changes ZeRO-3's behavior in our setup.

So we're not "stacking" two routing systems — we're using DS only for ZeRO-3 parameter management bookkeeping. Forward stays 100% transformers; DS never enters the forward.

**What still needs verification (didn't run yet):**

- Does the `expert_parallel_size` we pass to `_create_expert_and_data_parallel` map 1-1 to transformers' EP mesh size? In our config (e.g. 16 GPUs, `--expert_parallel_size 8`), DS would create `[[0..7], [8..15]]` for `expert_parallel_group` and `[[0,8], [1,9], ..., [7,15]]` for `expert_data_parallel_group`. Transformers' EP mesh is `(8,)` per node. Need to verify the rank arrangement matches (intra-node EP, inter-node DP).
- Does our EP partitioning produce the same physical layout DS expects? `GroupedGemmParallel.shard_tensor` slices `[start:end]` by `self.rank`. If the rank ordering differs from DS's expert_parallel_group ranks, the broadcast group would mix experts that aren't actually equivalent.
- The sonicmoe-side workaround (wrapper clamp+mask) still applies under DS-Z3 if the kernel's hand-written backward also NaNs there — needs a forward+backward smoke test once the tagging is in.

**Estimated effort:** 1–2 days to wire and validate, vs the multi-day DS-internals refactor I was assuming. Path forward is real and tractable.

---

## 6. TRL benchmark utilities (`benchmark/`)

These are tooling, not transformers/accelerate patches:

- `benchmark/fetch_peak_gpu_mem.py` — script that resolves `job_id → run_name` via Slurm log filename, scopes a trackio query to the run's start time, and reports max `gpu/<rank>/allocated_memory` across all GPUs. Required because trackio reuses run names across resubmits, so unscoped queries return max across history.
- `benchmark/templates/launch.sh.j2` + `benchmark/run_benchmark.py` — `skip_ep_dtensor_wrap` plumbing (sets `TRANSFORMERS_SKIP_EP_DTENSOR_WRAP=1` env var). Same justification as patch #2: bisect-only.
- `benchmark/CLAUDE.md` — added rule that every benchmark table must include Peak GPU Mem fetched via `fetch_peak_gpu_mem.py`.

These are TRL repo additions, not packages — already managed via the TRL working tree's git, not part of the "uncommitted patches" risk.

---

## Required SFT runtime flag for SP runs (no patch — just config)

For any DS-Z3 + SP + Ulysses run, pass `--pad_to_multiple_of 8` to `trl/scripts/sft.py`. Without it, packed batches with `seqlen % sp_size != 0` crash Ulysses around step 25 with `ValueError: batch's seqlen=X isn't divisible by sp-size=N`. This is a known limitation of the pack-then-shard collator pattern; the proper upstream fix is `pad_to_multiple_of=sp_size` in the SFT collator default for SP runs (TRL-side change).

---

## Revert everything (returns to a clean state)

```bash
# Transformers (working-tree only, no commits to undo)
cd /fsx/amine_dirhoussi/transformers
git checkout -- src/transformers/{modeling_utils.py,trainer.py,core_model_loading.py,integrations/tensor_parallel.py,integrations/moe.py,integrations/sonicmoe.py,integrations/hub_kernels.py}

# Accelerate (re-install)
cd /fsx/amine_dirhoussi/trl
.venv/bin/pip install --force-reinstall --no-deps accelerate

# DeepSpeed already reverted; reinstall as a safety net
.venv/bin/pip install --force-reinstall --no-deps deepspeed
```

After revert, the only EP+FSDP fix remaining is whatever's pulled from open PRs (none until they land).
