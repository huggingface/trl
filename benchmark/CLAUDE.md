# Benchmark Context for SFT MoE Scaling

## Goal

Test SFT training scaling limits for MoE models (Qwen3-4B dense, Qwen3-30B-A3B MoE, Qwen3-235B-A22B MoE) across parallelism strategies (FSDP2, DeepSpeed, CP, TP, PP, EP) on H100 NVL nodes. Measure MFU, TPS, TPS/GPU per config.

## Environment

- Cluster: Slurm, partition `hopper-prod`, QoS `normal` or `high` (AWS p5.48xlarge nodes)
- GPUs: H100 80GB SXM5 (bf16 dense Tensor Core peak: 989.5 TFLOPS) — verified via `nvidia-smi --query-gpu=name,memory.total` returning "NVIDIA H100 80GB HBM3, 81559 MiB"
- Python env: `/fsx/amine_dirhoussi/trl/.venv/bin/activate` (always use this, never install libs)
- Transformers fork: `/fsx/amine_dirhoussi/transformers` (installed editable in .venv)
- TRL repo: `/fsx/amine_dirhoussi/trl` (installed editable in .venv)

## Key Files

### TRL core changes

- `trl/trainer/utils.py` — `compute_flops_per_token()`, `compute_mfu()`, `fuse_moe_experts()`
- `trl/trainer/sft_trainer.py` — EP integration, MFU logging, fuse_moe_experts call
- `trl/trainer/sft_config.py` — `enable_expert_parallel`, `expert_parallel_size`, `fuse_moe_experts` flags

### Benchmark infrastructure

- `benchmark/run_benchmark.py` — renders Jinja2 templates + submits Slurm jobs
- `benchmark/collect_results.py` — parses Slurm logs, outputs CSV
- `benchmark/templates/sft.sbatch.j2` — Slurm job template
- `benchmark/templates/launch.sh.j2` — accelerate launch script
- `benchmark/templates/accelerate/fsdp2.yaml.j2` — FSDP2 accelerate config
- `benchmark/templates/accelerate/deepspeed_zero3.yaml.j2` — DeepSpeed config
- `benchmark/configs/qwen3_4b.yaml` — 4B dense benchmark runs
- `benchmark/configs/qwen3_30b_a3b.yaml` — 30B MoE benchmark runs
- `benchmark/configs/qwen3_235b_a22b.yaml` — 235B MoE benchmark runs
- `benchmark/report.md` — append-only timeline of experiments
- `benchmark/consolidated_report.md` — clean per-model summary tables
- `benchmark/bench_communication.md` — NCCL bandwidth benchmarks

### Transformers fork changes

- `/fsx/amine_dirhoussi/transformers/src/transformers/models/qwen3_moe/configuration_qwen3_moe.py` — `base_model_ep_plan` definition
- `/fsx/amine_dirhoussi/transformers/src/transformers/integrations/tensor_parallel.py` — `RouterParallel`, `MoeTensorParallelExperts`, `GroupedGemmParallel`
- `/fsx/amine_dirhoussi/transformers/src/transformers/integrations/moe.py` — `grouped_mm_experts_forward`, `batched_mm_experts_forward`
- `/fsx/amine_dirhoussi/transformers/src/transformers/modeling_utils.py` — line 4259 tp_plan mismatch
- `/fsx/amine_dirhoussi/transformers/src/transformers/core_model_loading.py` — weight loading + sharding

### Test scripts

- `benchmark/test_ep_fixed.py` — basic EP forward pass test
- `benchmark/test_ep_shapes.py` — shape debugging with hooks
- `benchmark/test_no_ep.py` — ground truth without EP

## Commands

### Submit benchmark jobs

```bash
# Dry-run (review generated scripts)
python benchmark/run_benchmark.py --config benchmark/configs/qwen3_30b_a3b.yaml

# Submit jobs
python benchmark/run_benchmark.py --config benchmark/configs/qwen3_30b_a3b.yaml --submit

# Submit specific run indices only
python benchmark/run_benchmark.py --config benchmark/configs/qwen3_30b_a3b.yaml --submit --run-index 0 2 5
```

### Debug EP interactively

```bash
# Single-GPU ground truth (no EP)
srun --partition=hopper-prod --nodes=1 --gres=gpu:h100:1 --ntasks-per-node=1 --exclusive --time=00:10:00 --qos=normal \
  bash -c 'source /fsx/amine_dirhoussi/trl/.venv/bin/activate && python benchmark/test_no_ep.py'

# Multi-GPU EP test (adjust nproc_per_node and gres)
srun --partition=hopper-prod --nodes=1 --gres=gpu:h100:4 --ntasks-per-node=4 --exclusive --time=00:10:00 --qos=normal \
  bash -c 'source /fsx/amine_dirhoussi/trl/.venv/bin/activate && torchrun --nproc_per_node=4 benchmark/test_ep_shapes.py'

# Multi-node EP test
srun --partition=hopper-prod --nodes=2 --gres=gpu:h100:8 --ntasks-per-node=1 --exclusive --time=00:10:00 --qos=normal \
  bash -c 'source /fsx/amine_dirhoussi/trl/.venv/bin/activate && torchrun --nproc_per_node=8 --nnodes=2 --node_rank=$SLURM_PROCID --master_addr=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -1) --master_port=29500 benchmark/test_ep_fixed.py'
```

### Monitor jobs

```bash
# Check queue
squeue -u $USER

# Watch specific job
squeue -j <JOB_ID>

# Check job output live
tail -f benchmark/logs/bench-<run_id>-<job_id>.out

# Check job errors
tail -f benchmark/logs/bench-<run_id>-<job_id>.err

# Cancel job
scancel <JOB_ID>
```

### Collect results

```bash
# From logs only (MFU, TPS, status)
python benchmark/collect_results.py --logs-dir benchmark/logs -o benchmark/results.csv

# With wandb peak GPU memory (recommended — always include peak memory in reports)
python benchmark/collect_results.py --logs-dir benchmark/logs --wandb-project trl-sft-benchmark -o benchmark/results.csv
```

**Important**: Always collect with `--wandb-project` to include Peak GPU Memory from wandb system metrics. All benchmark tables in `report.md` must include the `Peak GPU Mem` column. Runs shorter than ~2 minutes may not have wandb system metrics — for those, note "—" in the Peak GPU Mem column.

### Hard rule: never enable CPU offload (`cpu_offload: true`)

User feedback 2026-04-29: CPU offload completely kills MFU and metrics. **Do not submit any benchmark with `cpu_offload: true`** — even when a config OOMs and offload would unblock it, the resulting MFU is so degraded the run is not informative. If a 1n long-context run OOMs, look for non-offload workarounds (lower SP, EP, gradient_checkpointing tweaks, smaller batch) or accept the OOM ceiling.

### Mandatory: write order is `report.md` first, then `sft_benchmark_notion.md`, then `upstream_todo.md`

After every experiment / run / fix:

1. **Append to `benchmark/report.md` first.** It is the append-only chronological log: timestamp / dated section header, what was tried, what happened, the failure mode if any, the numbers if it worked. Do not edit older sections; new findings go at the bottom.
2. **Then update `benchmark/sft_benchmark_notion.md`.** It is the consolidated executive summary — clean tables, current best configs. When a new run produces a number that beats or replaces a row, update the row in place. When a row's verdict changes (e.g. broken → working), update it in place. The notion file should always reflect the current best, not the history.
3. **Then update `benchmark/upstream_todo.md`.** It is the living TODO checklist of local patches and not-working items. When something is tried (success or failure), tick the relevant `- [ ]` box (or add a sub-bullet with the date / link / status). When a PR is opened, add the link under the item. When a fix lands upstream and merges back into the stack, delete the item. The file should always reflect what's still open, not what's done — completed items get deleted, not archived.

Skipping (1) loses context for future debugging. Skipping (2) lets the notion file go stale while report.md gets bigger. Skipping (3) means the next session won't know what's still open vs already tried. All three happen.

When you finish a sweep, the natural cadence is: write the section in `report.md` (full detail), then sweep `sft_benchmark_notion.md` for any rows that should change, then check `upstream_todo.md` to see if the result closes any boxes (or surfaces new ones), then mention in chat what's new.

**When opening a PR or filing an issue**: update the relevant box in `upstream_todo.md` with the link inline (e.g. `[ ] G1 — SP --pad_to_multiple_of auto-default → TRL #5XXX (open 2026-04-30)`). When the PR merges and the patch is in the venv, delete the item.

### Mandatory: Peak GPU Mem on every benchmark table

Every row in every benchmark table (in `report.md` AND `sft_benchmark_notion.md`) **must** include Peak GPU Mem. No "—" placeholders unless wandb/trackio truly has no system metrics (rare, only for crashed runs that died before any system metric was logged).

How to fetch (current stack uses **trackio**, not wandb — system metrics are in `/fsx/amine_dirhoussi/.cache/trackio/trl-sft-benchmark.db:system_metrics`, keyed by `run_name`):

```bash
# Job ID → run name (via Slurm log) → peak GB / pct across all GPUs in the run's time window
python benchmark/fetch_peak_gpu_mem.py --job-id 22092737 22092738 22092739 ...
```

The script (`benchmark/fetch_peak_gpu_mem.py`) resolves each `job_id` from the Slurm log filename (`bench-<run_name>-<job_id>.out`), reads the `Started:` line to scope the time window (trackio reuses run names across re-submissions, so unscoped queries return max across history), and queries trackio's `system_metrics` JSON blob for `gpu/<rank>/allocated_memory` keys.

When updating a results table, run this against every job_id and paste the output into the Peak GPU Mem column. If any cell would say "—" because the run has no system metrics, investigate first — it usually means the trackio.db was deleted/rotated, or the run was canceled before the first system metric snapshot (~30s after start).

### Rename wandb runs (if labels are wrong)

```python
import wandb
api = wandb.Api()
runs = api.runs("trl-sft-benchmark")
for run in runs:
    if "some_condition" in run.name:
        run.name = run.name.replace("old", "new")
        run.update()
```

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

### Second Bug: `model._tp_plan` vs `model.tp_plan` at line 4259

`modeling_utils.py:4259` passes `model._tp_plan` (raw TP plan) to weight loading, but the weight loading at `core_model_loading.py:1209` looks up values via `model.tp_plan` (property that returns EP plan when EP enabled). The regex is built from TP plan keys, but values come from EP plan. This means:

- Keys only in EP plan (like `layers.*.mlp.gate`) never match the regex → router weights are never sharded (just replicated)
- Keys in both plans get the EP plan value (e.g., `grouped_gemm` instead of `packed_colwise`)

This creates an inconsistency where the hooks use the EP plan but weight loading uses a mix of TP plan regex + EP plan values.

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

1. **FSDP2 + MoE collective shape mismatch**: Different experts active on different ranks → different gradient tensor sizes → reduce_scatter fails. Fixed with `fuse_moe_experts()` in `trl/trainer/utils.py`.

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
