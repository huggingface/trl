Hey team, I've been working on a working recipie to scale trl.SFTTrainer for Qwen3 30B/235B Moe models.

3 weeks as of now cluster runs across DS-Z2/Z3 + FSDP2 + CP + SP + EP + sonicmoe/grouped_mm/Liger/chunked-CE, 2–16 nodes and I surfaced some EP issues and fixed along the way. Everything now is stable and scales well ( benchmark results)
Now I want to cleanup and upstream these fixes bu I think I've hit a small wall so asking for guidance before I open more PRs.

Some open PRs from the campaign ( thx @Arthur and @Ferdinand Mom for the review :pray: )

- #45662: wraps EP modules in DTensor. Without it FSDP silently corrupts EP weights via rank-0 broadcast.
- #45548 Fix EP+DS-Z3 loading via `accelerate launch`. Loading-side only; Z3+EP still trains broken weights (rank-ordering blocker, see #5 below).

But the rest of the fix have a deeper problem, transformers and accelerate each build their own `DeviceMesh` that dont don't talk to each other. transformers builds `model._device_mesh` inside `from_pretrained`. accelerate builds `state.device_mesh` from `ParallelismConfig`. accelerate has no awareness EP exists. Which could be fine because transformers implements EP using TP but
this means that every workaround below is like: bridging transformers' EP state into accelerate (or deepspeed) at runtime.

:adhesive_bandage::adhesive_bandage: ducktape fixes:

Here are the problems I encountered and the how I fixed them

1. **Multi-node EP training silently hangs at variable seq_len.** accelerate's dataloader gives every world rank a unique micro-batch, but ranks inside the same EP group must share the same one (the experts are sharded across them and they all-reduce expert outputs at the end of the MoE layer). With `--packing`, micro-batches pack to variable seq_len, so eventually one rank in the EP group ends up with `(B, 54223)` while the others have `(B, 65536)` and the EP all-reduce hangs on a shape mismatch. Took 2 weeks of NCCL flight-recorder debugging to root-cause — symptom looks like a random NCCL bug.

   What I did:
   - **DS path**: at trainer init, `accelerator.state.ds_device_mesh = model._device_mesh`. accelerate already has this slot for DS runs — its `_prepare_device_mesh()` returns it, and `prepare_data_loader`'s DEEPSPEED branch divides `process_index` / `num_processes` by `submesh_tp_size`. Since transformers' EP mesh is dim-named `"tp"` (size = ep_size), I just hand it that mesh and the division falls out for free. No new mesh built, no NCCL splits.
   - **FSDP path**: same trick doesn't work because accelerate uses ONE mesh slot (`state.device_mesh`) for both the dataloader AND `fsdp2_prepare_model`. If I drop the EP mesh in there, FSDP wrap reads it too and gets confused (it expects a `dp_shard_cp` dim that the EP mesh doesn't have). So I monkey-patch `accelerator.prepare_data_loader` for one call from `_get_dataloader`, with explicit `num_processes=world//ep_size, process_index=rank//ep_size, torch_device_mesh=None`. try/finally restores. ugly.

2. **FSDP can't even start training on an EP model.** Ever since #45662 made the EP-sharded experts proper DTensors, calling `accelerator.prepare(model)` blows up before the first step with `ImportError: cannot import name 'ReplicateParallel'`. The deeper issue: accelerate's `_prepare_tp` has a "skip if no DTensor in model" early-return, but EP DTensors trip it positive — so accelerate falls through into TP wrapping logic that was never designed to handle a model another lib already TP-wrapped, and it tries to import a transformers symbol that doesn't exist on our fork's branch point.

   What I did: in-place `.venv` patch in `accelerator.py:_prepare_tp`, early-return on `getattr(model, "_device_mesh", None) is not None`. `_device_mesh` is set by `apply_tp_plan`, so basically I'm saying "if transformers has already TP-wrapped this model, accelerate keep your hands off." Heuristic — pattern-matches an implementation detail of `apply_tp_plan`, not a real interface.

3. **Gradient clipping crashes every FSDP+EP step.** Once you have both EP-sharded experts AND non-EP params FSDP-wrapped, your model has parameters living on two different parallel meshes — EP DTensors on the EP mesh (size 8), FSDP DTensors on the FSDP DP mesh (size 16). PyTorch's `clip_grad_norm_` calls `_foreach_norm` to stack per-param grad norms, and you can't stack DTensors on different meshes — it errors with `RuntimeError: All operands in aten.stack.default must have the same mesh`. Crashes from step 1.

   What I did: `Trainer._clip_grad_norm` returns `tensor(0.0)` when `model.has_ep`, gated to FSDP-only (DS handles MoE grad norms via its own per-group path).

   This is the hack I'm most worried about. Gradients aren't actually being clipped to `max_grad_norm`. Telemetry shows `grad_norm=0` which is a lie. Fine for benchmarks where I just want MFU numbers, **unsafe for any real prod training**. Real fix is in PyTorch's `clip_grad_norm_` (or accelerate's wrapper) handling per-mesh grouping.

4. **`torch.compile` is blocked under FSDP+EP.** Same dual-mesh root cause as (3) but at the optimizer level. Adam's fused foreach calls `_group_tensors_by_device_and_dtype` to bucket parameters before applying the update, and it strict-asserts that each bucket's tensors share device + dtype + layout. With EP DTensors and FSDP DTensors in the same param-group, the foreach grouping doesn't know how to handle cross-mesh tensors and the assert fires inside the compiled kernel.

   What I did: nothing yet. Hard blocker on compile × EP. Tested with `--optim adamw_torch` (non-fused) — same crash because foreach groups regardless of fused/non-fused. So I lose the +5–10 pp MFU compile typically gives on top of the FSDP+EP recipe.

   DS-Z2+EP+compile works fine because DS uses plain `nn.Parameter` (no DTensor mesh). The bug is specific to how FSDP+EP creates the dual-mesh state.

5. **Making MoE+EP play with DeepSpeed at all.** DS already has a notion of MoE layers — its native `deepspeed.moe.layer.MoE` class, plus a per-param convention (`param.allreduce=False` + `param.group_name="ep_size_N"`) that `is_moe_param()` looks for. transformers' EP doesn't use either: stock `GroupedGemmParallel.post_shard_wrap` just wraps the EP-sharded local tensor as a DTensor on the EP mesh. So when DS walks the model looking for MoE layers, it finds none — `engine.has_moe_layers=False`, `expert_data_parallel_group=None`, and Z2's `_configure_moe_settings` crashes with `TypeError: 'NoneType' object is not subscriptable`. Even if you sidestep the detection, DS's `_broadcast_model` and `is_moe_param` look up *named* process groups like `"ep_size_8"` that nothing in the EP path ever creates, so DS init crashes there too.

   What I did to bridge the two — three pieces, in order:

   - **Tag the params at shard time** (commit `cd52547f87`, extended to all DS stages in `f16bffc8c5`). Added a DS branch to `GroupedGemmParallel.post_shard_wrap`: when DS is active (detected via `_hf_deepspeed_config_weak_ref`), don't wrap as DTensor — return a plain `nn.Parameter` with `param.allreduce = False` and `param.group_name = f"ep_size_{ep_size}"`. These are the exact markers DS's `is_moe_param` looks for downstream. (Initially Z3-only, then expanded to Z1/Z2/Z3 once Z2 became the working path.)
   - **Pre-create the named DS groups.** `Trainer.create_accelerator_and_postprocess` calls `deepspeed.utils.groups._create_expert_and_data_parallel(ep_size)` before `Accelerator()` instantiates, so the named groups exist by the time DS init looks them up. That helper has a leading underscore — it's a private DS API the trainer is now hard-coupled to.
   - **Patch DS's auto-detect.** Even with the per-param tagging from above, DS's auto-detect for `engine.has_moe_layers` only scans for the native `deepspeed.moe.layer.MoE` class — it doesn't inspect param attributes. Extended the detection loop in `.venv/deepspeed/runtime/engine.py:_configure_distributed_model` to also recognize external EP via `any(getattr(p, "allreduce", True) is False for p in module.parameters(recurse=False))`. Lives in `.venv` so `uv pip install --reinstall deepspeed` wipes it. I want to avoid a DS-side PR for review-cycle reasons; alternative is writing `engine.optimizer.expert_data_parallel_group` directly post-init from the trainer — also a private API touchpoint.

   Plus splitting MoE params into their own optimizer group via DS's `split_params_into_different_moe_groups_for_optimizer` (so the grad reduce goes through the small `expert_data_parallel_group` instead of the full DP group), and gating the `_clip_grad_norm` skip from (3) to FSDP-only (DS handles MoE grad norms via its own per-group mechanism).

   With those in place, **DS-Z2+EP works end-to-end** and is actually our long-context champion — every healthy 30B+ benchmark on >32k context uses this recipe (45% MFU at 32k, 57% at 64k, 69% at 128k on 2–4 nodes).

6. **Z3+EP doesn't work, even with the recipe from (5).** What I observed: job runs cleanly, no crash, first-step loss ~62 (Qwen3-30B initial loss should be ~10–14), backward produces NaN gradients, training never recovers. So **the first forward already produces broken logits** — the model state is wrong by the time training starts. Reproducible across FA3+grouped_mm, FA3+sonicmoe, sdpa+sonicmoe — only the *DS-Z3+EP* combination fails (FSDP2+EP with the same kernels works).

   What I think is happening (this is hypothesis from reading both libs' code, **not directly verified** by a forward smoke test on per-rank expert weights):

   - **Z3 fundamentally doesn't grok MoE.** `deepspeed/runtime/zero/stage_1_and_2.py` has full `is_moe_param` plumbing (param-group split, per-group grad reduce via `expert_data_parallel_group`); `stage3.py` has *zero* MoE awareness. Z3 also assumes *"the same param name has the same logical data on every rank"* — which EP violates by construction (`model.layers.0.mlp.experts.gate_up_proj` holds experts 0–15 on rank 0 and 128–143 on rank 8). Z3's all-gather across the full DP group would concatenate these into garbage. Z2 doesn't all-gather params (only grads/optim), so it sidesteps this — Z3 can't.
   - **Rank-ordering mismatch in the broadcast group.** transformers' `GroupedGemmParallel.shard_tensor` partitions by global rank; DS builds `expert_data_parallel_group=[[0,8],[1,9],...]` and *expects siblings to hold the same slice*. If that holds, DS's `_broadcast_model` inside `[0,8]` overwrites rank 8's slice with rank 0's. **But I have a contradicting datapoint**: at world=64 with singleton groups (`[[0],[1],...,[63]]`) the broadcast can't fire across siblings, and the failure mode still shows up. So either rank-ordering isn't the root cause, or it's only one of multiple things going wrong.

   What I did: tried the recipe from (5) with 3 extra patches in commit `cd52547f87`. Confirmed it doesn't fix the broken-logits problem. Haven't done the smoke test that would isolate "is rank 0's expert N bit-identical to rank 8's expert N post-broadcast" — that's what would give me a real answer instead of intuition.

All of these are private-API touchpoints that break on upgrades.

## Where the team is already heading

Before posting this I dug through @3outeille's recent work to make sure I'm not asking questions you've already answered. Two open PRs on a `fsdp-vs-ddp` branch:

- **#45028** — _TP refactor for FSDP + TP integration_. Expands `DistributedConfig` to own the full parallelism story: `DistributedConfig(tp_size=2, tp_plan="auto", fsdp_size=2, fsdp_plan="auto", enable_sequence_parallel=True)`. The model exposes `model.device_mesh["fsdp"]` / `model.device_mesh["tp"]` as the single source of truth. accelerate's `ParallelismConfig` is out of the picture.
- **#44974** — _Refactor `core_model_loading` to support FSDP shard-on-read loading_. Replaces broadcast-based FSDP loading with per-rank shard-on-read. Pairs with #45028.

The training example in #45028 even uses `torchtitan.distributed.utils.clip_grad_norm_` with a TODO to add it to `transformers.distributed` — so even grad-norm computation is being pulled in-tree.

Reading this right, the direction is: **transformers owns the multi-dim mesh via `DistributedConfig`, parallelism support lives in transformers, accelerate steps out for distributed-config-driven runs.** That inverts the assumption I'd been making — most of my bridges have been pushing transformers' EP state *into* accelerate, but the team is going the other way and bringing parallelism *out of* accelerate. So my questions all need to be reframed against this direction.

## What I'd like guidance on (won't open more PRs until I have direction)

**Q1. Should EP fold into `DistributedConfig` alongside `tp_size` / `fsdp_size`?** `DistributedConfig` already has `enable_expert_parallel` (boolean); promoting it to `ep_size: int` so EP joins the same mesh family `("dp", "fsdp", "tp", "ep")` is the natural fit if I'm reading the refactor's intent correctly. With that in place, `model.device_mesh["ep"]` becomes the single source of truth — hacks 1, 2, 3, 4 collapse to nothing (single mesh family → no cross-mesh `_foreach_norm` or `_group_tensors_by_device_and_dtype` failures, no `_prepare_tp` skip, no dataloader monkey-patch). Hacks 5 and 6 are DS-specific and need separate attention but get easier when EP rank ordering can be defined in a single place.

**Q2. Timing.** How close are #45028 and #44974 to landing? If they're 1–2 weeks out, I'd happily wait and rebase the EP campaign onto them rather than push my bridges upstream — most of the bridges become unnecessary the moment `DistributedConfig` owns the EP dim. If the refactor is farther out (months), I'd want to know how much of the bridge is durable as a temporary recipe so I'm not opening PRs that get thrown away the moment your refactor lands.

**Q3. Does EP get its own dim name (`"ep"`) in `model.device_mesh`, or share with `"tp"`?** EP and TP are identical for dataloader sharding (both replicate data within the dim) but diverge for FSDP wrap rules (TP-shard params can be FSDP-wrapped on the orthogonal dim, EP-shard params can't), for grad-norm grouping, and for future Megatron-style EP+TP composition. Sharing `"tp"` is back-compat with stock transformers' EP today (which already names its mesh dim `"tp"`); a sibling `"ep"` keeps the distinction clean. I lean sibling, but defer to your refactor's conventions.

**Q4. Is the DS / FSDP code-path fork going to get cleaner?** This is a separate concern from the mesh question (Q1) — Q1 is "where's the source of truth?", this one is "how is backend-specific behavior dispatched?". Right now, "what does DS+EP do" isn't readable from any single function — the per-backend behavior is scattered as `is_deepspeed_enabled` / `is_fsdp_enabled` / `is_deepspeed_zero3_enabled()` conditionals in 20+ places across `trainer.py`, `modeling_utils.py`, `tensor_parallel.py`, and inside accelerate too (`accelerator.py`, `fsdp_utils.py`, `data_loader.py`). Each of my bridges from (5) had to fork in yet another place — DS path here, FSDP path over there, with shared logic woven between. Adding a new backend (or removing one) means touching every fork.

If `DistributedConfig` is becoming the single config, is there room in the refactor to also consolidate the dispatch — one handler / strategy object per backend, called once per lifecycle phase, instead of N inline branches? I don't have a concrete proposal, but the cost of new bridges compounds because each one ends up in a different file with a different set of conditionals around it, and "what does X+EP do" becomes a many-file walk.

**Q5. DS-Z3+EP — what's actually broken?** (hack #6). I've been *assuming* it's rank-ordering between transformers' EP partition and DS's `expert_data_parallel_group` — but that's pattern-matching on code, not a confirmed diagnosis. The contradicting datapoint is that with singleton `expert_data_parallel_group` at world=64, no broadcast runs across siblings and the failure still shows up. So one of: (a) rank-ordering is one of multiple things wrong, (b) it's the wrong hypothesis entirely and the real cause is Z3's full-DP all-gather of EP-distinct-data params, (c) something else inside Z3's `_broadcast_model` / `_zero_init_param` path that I haven't traced.

Two questions, in order:

- **Has anyone actually run the smoke test that would distinguish these?** i.e. compare per-rank expert weights right after `from_pretrained` vs right after `deepspeed.initialize`'s `_broadcast_model`. If rank 0's experts and rank 8's are bit-identical post-broadcast, rank-ordering is confirmed; if they're still distinct but the model still produces broken logits, the cause is elsewhere.
- **Assuming rank-ordering is at least part of it**, two ways to align it:
  - *Fix in transformers*: partition experts by `global_rank % ep_size` so DS siblings naturally hold the same slice. Changes EP semantics for users running EP without DS — the per-rank expert layout no longer matches the "global rank N holds experts N×16..(N+1)×16" convention.
  - *Fix in trainer*: keep transformers' partition; build DS's `expert_parallel_group` manually with `dist.new_group` so DS's rank ordering matches transformers'. More invasive, but doesn't touch user-visible EP semantics.

Both feel weird, and I'd rather verify the diagnosis first before picking either.

**Q6. Push back.** If EP is *intentionally* staying outside the #45028 / #44974 refactor (e.g., kept separate because it's tied to specific MoE kernels and doesn't generalize cleanly), say so — I'll commit to bridges-as-architecture, stop trying to disentangle, and treat the six hacks as the long-term home rather than scaffolding for a Phase 5.

Smallest concrete PR I can open today is #45662 (already up); rest is gated on direction here. Happy to walk through any of this in a call.

Pinging @3outeille specifically — would love your read on whether the EP campaign should fold into your refactor.
