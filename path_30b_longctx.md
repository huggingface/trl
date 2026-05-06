# Path to 30B MoE long-context SFT training

> **Tracking issue / draft PR for SFTTrainer scaling to frontier-scale MoE.**
> Train Qwen3-30B-A3B (and 235B-A22B) end-to-end with TRL's `SFTTrainer` at
> long context (16k → 1M) on 8×H100 nodes, with MFU competitive with dense
> training. This page tracks the cross-repo work (transformers / accelerate /
> TRL / DeepSpeed / Liger) that this depends on, the known issues, and the
> performance numbers we land along the way.
>
> Modeled after vLLM's [transformers v5 tracking PR](https://github.com/vllm-project/vllm/pull/30566).
> Community contributions and reproduction reports on different hardware are
> very welcome — please drop a comment with your config + numbers.

## Why this is non-trivial

Qwen3-30B-A3B has 128 experts (8 active per token) and a 152k vocab; out of
the box it does not train cleanly on any combination of FSDP2 / DeepSpeed /
Expert Parallel / Sequence Parallel / Context Parallel. A single recipe
(`SFTTrainer` + `--enable_expert_parallel` + Liger + sonicmoe + FA3 + chunked
loss) gets there, but it required fixing real bugs in 4 repos. This issue
tracks all of them.

## Status snapshot — Qwen3-30B-A3B (2026-05-06)

Best end-to-end-correct configurations on 2× / 4× / 8× p5.48xlarge nodes
(H100 SXM5, 989.5 TFLOPS bf16 peak). All numbers measured with the
`benchmark-sft-moe` branch of TRL + the patched transformers fork referenced
below. **Window MFU peak** is the per-log-window throughput; **adj.** is the
causal-corrected value (matches the Llama 2/3 / DS-Ulysses convention — half
the attention FLOPs disappear under causal masking).

| Context | Nodes | Recipe | Window MFU | Adj. MFU | Loss | Notes |
|---|---|---|---|---|---|---|
| 16k | 2 | FSDP2 + EP=8 + FA3 + sonicmoe (post-PR #45621 kernel fix) | **48.2 %** | 32.3 % | 13.4 ✅ | new champion since clamp era |
| 32k | 2 | DS-Z2 + EP=8 + FA3 + sonicmoe + Liger | **65.0 %** | 39.2 % | 8.0 ✅ | highest 30B-on-2-node MFU we have measured |
| 64k | 4 | DS-Z2 + EP=32 + FA3 + sonicmoe + Liger (R3) | **72.0 %** | 40.1 % | 1.87 ✅ | first end-to-end-correct multi-node EP=32 |
| 128k | 4 | DS-Z2 + EP=32 + FA3 + sonicmoe + Liger (R1) | **81.7 %** | 43.4 % | (see H5) | throughput real, convergence: see Known issues / H5 |
| 256k | 8 | DS-Z3 + SP=2 + FA3 + sonicmoe + Liger + compile | **63.6 %** | 32.8 % | 1.56 ✅ | SP path beyond the EP-buffer ceiling |
| 512k | 8 | DS-Z3 + SP=4 + FA3 + sonicmoe + Liger + compile | **63.3 %** | 32.1 % | ✅ | |
| 1M | 8 | DS-Z3 + SP=8 + FA3 + sonicmoe + Liger + compile | **62.3 %** | 31.4 % | 1.56 ✅ | first 1M-context MoE SFT result on this stack |

For Qwen3-235B-A22B early numbers (32k 8n EP=64 + Liger + compile = 70.1 %
window MFU, healthy loss), see the J section of `benchmark/upstream_todo.md`
in the `benchmark-sft-moe` branch.

## Hardware / software baseline

- **Cluster**: AWS p5.48xlarge — 8× H100 SXM5 80 GB per node, 32× EFA, 3200 Gbps inter-node aggregate
- **TRL**: `benchmark-sft-moe` branch (this branch will land via the PRs in §TRL below)
- **transformers**: 5.6.0.dev0 + the `qwen3-moe-ep-v2` series of fixes (see §transformers below)
- **accelerate**: 1.13.0 + an in-place `_prepare_tp` `has_ep` skip (§accelerate)
- **DeepSpeed**: stock + a runtime engine attribute patch (no DeepSpeed PR planned — workaround stays on TRL/transformers side)
- **PyTorch**: 2.10.0+cu128
- **Liger Kernel**: fused CrossEntropy + RMSNorm + RoPE (Triton). SwiGLU patch must be **disabled** under EP (`--liger_kernel_config '{"swiglu":false}'`) — see §Known issues / H1
- **Flash Attention 3**: `kernels-community/vllm-flash-attn3`
- **MoE kernel**: `kernels-community/sonic-moe` (CuteDSL), selected via `--experts_implementation sonicmoe`
- **Dataset**: `THUDM/LongAlign-10k` packed with `--packing --packing_strategy wrapped`

## Linked PRs

### transformers

Core EP support + correctness fixes. The first three are the load-bearing
ones — without them, EP > 1 produces silently wrong expert outputs.

- ✅ **Merged** [#45436](https://github.com/huggingface/transformers/pull/45436) — Add EP support for Qwen3 MoE, fix `GroupedGemmParallel` for 2D meshes
- ✅ **Merged** [#45473](https://github.com/huggingface/transformers/pull/45473) — Fix EP routing: `RouterParallel` shape, `tp_plan` property, `grouped_mm` sentinels (3 bugs that combined to produce silently wrong logits at every EP > 1; first regression tests for EP)
- 🟡 **Open** [#45621](https://github.com/huggingface/transformers/pull/45621) — sonicmoe kernel-side sentinel fix (drops the wrapper-level `expert_ids.clamp` workaround; +5–8 pp peak MFU at 16k–32k EP=8)
- 🟡 **Open** [#45662](https://github.com/huggingface/transformers/pull/45662) — EP + FSDP DTensor wrap (lets EP-sharded params survive FSDP2's `ignored_params` boundary)
- 🟡 **Open** [#45548](https://github.com/huggingface/transformers/pull/45548) — DeepSpeed-Z3 + EP weight loading
- 🟡 **Open** [#45649](https://github.com/huggingface/transformers/pull/45649) — FSDP `cpu_ram_efficient_loading` fixes
- ⏳ **Planned** — `tp_plan` loader rank-0 gate (Layer 1 of the duplicate-load issue: every TP/EP rank reads the full dense replica from disk; ~480 GB redundant disk I/O / node on 30B EP=8). Draft at `benchmark/upstream_issue_tp_plan_duplicate_load.md`.
- ⏳ **Planned** — `ValueError` guard when `enable_expert_parallel=True` and `cpu_ram_efficient_loading=True` are combined (Layer 2 of the same: silent corruption today)
- ⏳ **Planned** — DS-Z2 + EP transformers PR (4 patches: `tensor_parallel.py:GroupedGemmParallel.post_shard_wrap` backend branch, `Trainer.create_accelerator_and_postprocess` MoE-group setup, `Trainer.create_optimizer` MoE param split, `Trainer._clip_grad_norm` cross-mesh skip) + a post-`deepspeed.initialize` engine attribute patch that replaces the originally-planned DeepSpeed-side change
- ⏳ **Planned** — wire `DistributedConfig(enable_expert_parallel=True)` into `accelerate.state.parallelism_config.ep_size` once accelerate exposes that field

### accelerate

- ⏳ **Planned** — `_prepare_tp` early-return when `model.has_ep` (post-#45662 EP params become DTensors and trigger an `ImportError` on `ReplicateParallel`; single-line check)
- ⏳ **Planned** — first-class `ep_size` field in `ParallelismConfig` + `submesh_ep_size` divisor in `prepare_data_loader` (mirrors existing TP handling exactly). Today TRL piggy-backs on the TP-replication path by exposing the EP mesh as `"tp"`. **Highest-leverage upstream PR remaining.**
- ⏳ **Planned** — `fsdp2_prepare_model` capture/restore around the meta-move so EP `ignored_params` survive the FSDP rank-0 broadcast (paired with the transformers `cpu_ram_efficient_loading` work above)

### TRL

The `benchmark-sft-moe` branch will be split into a series of PRs against
`main`. Order is independence-first — anything that can ship alone goes
first.

1. ⏳ MFU helpers — `compute_flops_per_token` + `compute_mfu` in `trl/trainer/utils.py`. Pure-Python, no `SFTTrainer` coupling. Includes causal-correct attention and the `embed_flops=0` / `lm_head_flops=2*V*h` accounting fix.
2. ⏳ `SFTTrainer.log()` MFU integration (window + cumulative; corrects `num_input_tokens_seen` overcount by `cp_size × sp_size`)
3. ⏳ `enable_expert_parallel` + `expert_parallel_size` + `experts_implementation` config fields and the EP branch in `SFTTrainer.__init__` (depends on the transformers PRs landing)
4. ⏳ Generalized kernel pre-warm + `HF_HUB_OFFLINE` flip (workaround for HF Hub cache races on multi-node — see Known issues / H3)
5. ⏳ SP `--pad_to_multiple_of` auto-default when `accelerator.parallelism_config.sp_size > 1` (G1 in upstream-todo)
6. ⏳ Per-rank `TRITON_CACHE_DIR` + legacy TF32 flags in `trl/scripts/sft.py`
7. ✅ **Merged** [#5575](https://github.com/huggingface/trl/pull/5575) — Chunked cross-entropy loss for SFT (up to −50 % VRAM, unlocked the 32k → 1M long-context configs)

### DeepSpeed

**No upstream PR planned.** The originally-planned 1-line `engine.py` MoE
detection patch is being replaced by a post-`deepspeed.initialize` engine
attribute patch that lives entirely in `transformers/trainer.py`. Avoids
taking on a DS-side dependency. See D-works in `benchmark/upstream_todo.md`
for the three options being evaluated.

### Liger

- ⏳ **Planned** — `apply_liger_kernel_to_qwen3_moe` should auto-skip the
  SwiGLU patch when EP is active (today it silently bypasses transformers'
  `@use_experts_implementation` dispatcher, calling `F.one_hot` with an
  out-of-range sentinel → `device-side assert`). Workaround:
  `--liger_kernel_config '{"swiglu":false}'`. Repro: `benchmark/test_liger_qwen3_moe_ep.py`. Investigation log: `benchmark/debug_liger_ep.md`.

## Known issues / open blockers

> Status legend (matches `benchmark/upstream_todo.md`):
> 🟢 ready to PR · 🟡 keep local (fix is elsewhere) · ⛔ blocked / not yet attempted · 🔧 active workaround in place · 🧪 frontier · 🗑️ debug-only

- **🔧 EP-aware DataLoader sharding (C3)** — Local fix in `transformers/trainer.py` that exposes the EP mesh as `"tp"` so accelerate's existing TP-replication branch in `prepare_data_loader` divides correctly. Without it, every world rank gets a unique micro-batch but the 8 ranks of an EP group must see the **same** batch (EP shards experts only) → silent NCCL hang on the EP all-reduce after a random number of steps. Caused multi-night investigations (`SeqNum=2201` ALLREDUCE-PG-ID-2 fingerprint). **Highest-leverage upstream cleanup remaining.**
- **🟡 `_clip_grad_norm` cross-mesh failure (B1)** — `clip_grad_norm_` calls `_foreach_norm` which stacks per-param norms; stacking DTensors on different meshes (EP mesh + FSDP DP mesh) errors with `RuntimeError: All operands in aten.stack.default must have the same mesh`. Local skip returns `tensor(0.0)` for telemetry — gradients are not actually clipped. Fine for benchmarks, unsafe for production. Proper fix is in PyTorch.
- **⛔ FSDP + EP + compile Adam crash (E1)** — `_group_tensors_by_device_and_dtype` strict-asserts grouped tensors share device+dtype; EP DTensors (EP mesh, size 8) and FSDP DP DTensors (size 16) trip the assert. DS-Z2+EP+compile works because DS uses plain `nn.Parameter` (`allreduce=False` / `group_name` markers, no DTensor mesh). 3 fix paths under evaluation.
- **⛔ Architectural EP buffer ceiling at 32k+ per-rank seq (F1)** — transformers' EP replicates routing across all EP ranks (TP-style EP, not all-to-all), so every rank materializes a `seq × num_local_experts × moe_intermediate × 2 B` activation tensor: 18.55 GiB at 32k per-rank, 37.09 GiB at 64k. Workaround: chunked-CE / Liger frees ~20 GB from the lm_head logits which leaves room up to 128k EP=32 (R1, 81.7 % MFU). Real fix is a streaming kernel rewrite — biggest single ceiling-breaker for 256k+ EP, RFC TODO.
- **⛔ FA3 + CP incompatible (H2)** — accelerate hard-guards CP to sdpa attention. Long-context MoE has to choose between FA3 throughput (SP path) and ring-attention seq sharding (CP path). Cost: ~5 pp MFU at 32k FSDP+EP+CP=2 (15.6 % today; FA3 would close most of that).
- **🔧 Multi-node HF Hub cache race (H3)** — concurrent writes to `~/.cache/huggingface` corrupt the lock files on 4n+. Workaround: `HF_HUB_OFFLINE=1` after pre-warming sonicmoe + FA3 kernels in `trl/scripts/sft.py`.
- **⛔ sonicmoe Triton kernel ignores `TRITON_CACHE_DIR` (H4)** — multi-node EP=32+ runs intermittently fail with `FileNotFoundError` on `~/.triton/<HASH>/token_gather_sum_kernel.{source,ptx}`. Hypothesis: the `kernels` library compiles before `sft.py:69` sets the env var. Try setting it from `templates/launch.sh.j2` before Python starts.
- **⛔ Loss-zero / NaN at large EP + long-ctx + Liger / chunked (H5)** — at "≥32k context AND large total rank count", `loss.item()` PRE-backward is NaN from step 1, masked downstream so the trainer reports `loss=0`. NaN appears inside the optimizer step for the 5 FSDP/DS-sharded entry-side params (`embed_tokens.weight` + `layers.0.self_attn.{q,k,v,o}_proj.weight`). Static analysis ruled out the EP all-reduce math, DS-Z3 reduce-scatter averaging, Liger FLCE chain-rule scaling, and the C3 dataloader fix. Forward + backward math is fine; bug is in `(grad → optim_state → param)`. **High priority — the 81 % / 95 % window MFU rows on this stack are throughput-real but training-incorrect today.** Healthy convergence at 64k 4n EP=32 (R3, 1.87 loss) and at SP-path long-context (R5/R6, 1.56 loss).

## How to reproduce

The 32k 2-node champion (DS-Z2 + EP=8 + FA3 + sonicmoe + Liger, 65 % window
MFU, healthy loss):

```bash
git checkout benchmark-sft-moe          # ← will collapse to main as the PRs above land

# transformers + accelerate need the patched fork until the PRs in §transformers and §accelerate land
# (full revert recipe in benchmark/upstream_todo.md "Revert everything")
cd /path/to/transformers
git remote add aminediro git@github.com:aminediro/transformers.git  # placeholder name
git fetch aminediro qwen3-moe-ep-v2 && git checkout qwen3-moe-ep-v2

cd /path/to/trl
python benchmark/run_benchmark.py \
    --config benchmark/configs/qwen3_30b_a3b.yaml \
    --submit \
    --run-index <row-from-table-above>
```

Each row in `benchmark/configs/qwen3_30b_a3b.yaml` corresponds to one cell in
the headline table. Logs land in `benchmark/logs/`; results collection +
peak-mem queries via `benchmark/collect_results.py` and
`benchmark/fetch_peak_gpu_mem.py`.

## Help wanted

- Reproduction reports on non-H100 hardware (H200, MI300X, B200) — especially the EP=8 16k FSDP2 baseline
- Cross-checks of the MFU formula in `trl/trainer/utils.py:compute_flops_per_token` against your own training runs (Llama 2/3 / DS-Ulysses / Megatron numbers welcome)
- A clean repro for the H5 NaN-at-large-EP issue (the smallest config we have today is 16k 8n EP=64 — anything smaller would massively shorten the debug loop)
- Anyone looking at the FSDP + EP + compile Adam path (E1) — three fix paths sketched in `benchmark/upstream_todo.md`, none yet attempted

## Changelog

- **2026-05-06** — Issue opened. Headline numbers as of `benchmark-sft-moe@ee2876cc`.
