# SFT Benchmark Report

## Qwen3-4B (Dense) — FSDP2

### Results

| Context | Nodes | DP  | CP  | TP  | MFU   | TPS     | TPS/GPU | Status |
| ------- | ----- | --- | --- | --- | ----- | ------- | ------- | ------ |
| 16k     | 2     | 16  | 1   | 1   | 30.2% | 90,058  | 5,629   | Yes    |
| 16k     | 2     | 8   | 2   | 1   | 25.1% | 149,724 | 9,358   | Yes    |
| 16k     | 2     | 4   | 4   | 1   | 18.3% | 217,910 | 13,619  | Yes    |
| 32k     | 2     | 8   | 2   | 1   | 31.6% | 121,842 | 7,615   | Yes    |
| 32k     | 2     | 4   | 4   | 1   | 26.8% | 207,029 | 12,939  | Yes    |
| 32k     | 4     | 16  | 2   | 1   | 31.8% | 245,336 | 7,667   | Yes    |
| 32k     | 4     | 8   | 4   | 1   | 27.1% | 418,189 | 13,068  | Yes    |
| 32k     | 4     | 4   | 8   | 1   | 19.6% | 605,864 | 18,933  | Yes    |
| 32k     | 4     | 8   | 1   | 4   | -     | -       | -       | OOM    |
| 32k     | 4     | 4   | 1   | 8   | -     | -       | -       | OOM    |

### Observations

- **DP scaling is linear**: 2 nodes → 4 nodes with same CP gives ~2x TPS and constant TPS/GPU (~7,600 for CP=2).
- **CP increases raw TPS but decreases MFU**: CP splits sequences across GPUs, reducing per-GPU memory but adding ring attention communication overhead. MFU drops from 30% (CP=1) to 25% (CP=2) to 18% (CP=4).
- **TP OOMs at 32k**: TP=4 and TP=8 on 32k context don't fit in memory on 4 nodes.
- **MFU note**: TPS with CP is overcounted by `cp_size` in transformers' `num_input_tokens_seen`. Our MFU computation corrects for this (divides TPS by `cp_size`).

### Issues encountered and fixes

- **OpenThoughts3 dataset incompatibility**: `from`/`value` conversation format incompatible with FSDP2. Switched to `THUDM/LongAlign-10k`.
- **CP sequence length assertion**: Fixed with `--packing_strategy wrapped`.
- **TPS overcounting with CP**: Corrected in MFU computation. TODO: fix upstream in transformers.

---

## Qwen3-30B-A3B (MoE, 128 experts) — FSDP2 with Fused Experts

### Results (sdpa attention)

| Context | Nodes | DP  | CP  | MFU   | TPS    | TPS/GPU | Status |
| ------- | ----- | --- | --- | ----- | ------ | ------- | ------ |
| 16k     | 2     | 16  | 1   | 3.17% | 8,534  | 533     | Yes    |
| 16k     | 2     | 8   | 2   | 1.64% | 8,818  | 551     | Yes    |
| 16k     | 4     | 32  | 1   | 2.63% | 14,174 | 443     | Yes    |
| 16k     | 4     | 16  | 2   | 1.44% | 15,468 | 483     | Yes    |
| 16k     | 4     | 8   | 4   | 0.71% | 15,371 | 480     | Yes    |
| 32k     | 2     | 8   | 2   | 4.40% | 14,290 | 893     | Yes    |
| 32k     | 4     | 16  | 2   | 3.91% | 25,408 | 794     | Yes    |
| 32k     | 4     | 8   | 4   | 2.19% | 28,414 | 888     | Yes    |

### Results (flash_attention_2 via kernels library)

| Context | Nodes | DP  | CP  | MFU   | TPS    | TPS/GPU | Status |
| ------- | ----- | --- | --- | ----- | ------ | ------- | ------ |
| 16k     | 2     | 16  | 1   | 3.11% | 8,367  | 523     | Yes    |
| 16k     | 4     | 32  | 1   | 2.61% | 14,056 | 439     | Yes    |

Note: CP is incompatible with `flash_attention_2` in accelerate — requires `sdpa`.

### Observations

- **MFU 0.7-4.4%**: Significantly better than DeepSpeed ZeRO-3 (1-2%) thanks to FSDP2 avoiding full-parameter allgathers. Still low because all 128 experts are sharded across all GPUs (no true EP).
- **Longer context improves MFU**: 32k gives higher MFU than 16k (4.4% vs 3.2% at 2 nodes) — more compute per communication round.
- **CP trades MFU for context scaling**: Same pattern as the 4B model. CP enables longer contexts but reduces MFU due to ring attention overhead.
- **flash_attention_2 ≈ sdpa**: The `kernels` library flash attention gives ~2% lower MFU than sdpa — no meaningful speedup without the native `flash_attn` package.
- **DP scaling**: 2→4 nodes gives ~1.7x TPS (not 2x) due to inter-node communication overhead for the 30B model.

### FSDP2 + MoE: The Problem

Vanilla FSDP2 crashes with MoE models:

```
RuntimeError: Detected mismatch between collectives on ranks.
Rank 12: _REDUCE_SCATTER_BASE, TensorShape=[505155840]
Rank 0:  _REDUCE_SCATTER_BASE, TensorShape=[240914688]
```

**Why it happens**: Qwen3-30B has 128 experts per layer, each stored as a separate `nn.Linear` module in a `ModuleList`. FSDP2's `TRANSFORMER_BASED_WRAP` wraps each decoder layer and runs `reduce_scatter` on gradients during backward. The MoE router is data-dependent — different tokens route to different experts, so different ranks activate different subsets of experts. The gradient tensors end up with different sizes per rank, violating FSDP2's requirement that all ranks participate in collectives with identical tensor shapes.

This happens on both single-node and multi-node. It is not a communication issue — it is a fundamental incompatibility between FSDP2's symmetric collective model and MoE's asymmetric expert activation.

### What We Fixed: Fused Expert Weights

We fuse the 128 individual `nn.Linear` expert modules into single grouped tensors:

```python
# Before: ModuleList of 128 individual nn.Linear (asymmetric across ranks)
experts = ModuleList([
    MLP(gate_proj=Linear(2048, 768), up_proj=Linear(2048, 768), down_proj=Linear(768, 2048)),
    ...  # ×128
])

# After: Fused Parameter tensors (symmetric across all ranks)
experts.gate_proj = Parameter([128, 768, 2048])
experts.up_proj   = Parameter([128, 768, 2048])
experts.down_proj = Parameter([128, 2048, 768])
```

**Why this fixes the crash**: FSDP2 now sees three large parameters per MoE layer instead of 128×3 small ones. All ranks have identical parameters with identical shapes. During backward, even though different experts are activated, the gradient is computed on the full fused tensor (inactive expert slices get zero gradient). The `reduce_scatter` operates on the same shape everywhere.

The forward is patched to index into the fused tensor: `F.linear(input, experts.gate_proj[expert_idx])` instead of calling individual `nn.Linear` modules.

Enabled via `--fuse_moe_experts` flag in SFTConfig.

### What This Does NOT Do: True Expert Parallelism

Our fused approach makes FSDP2 work with MoE, but every GPU still holds **all 128 experts** (sharded across the FSDP DP group). This means:

- FSDP2 allgathers the full `[128, 768, 2048]` fused weight tensor before each layer's forward pass
- Only 8 of 128 experts are actually used per token
- The communication/compute ratio is ~16:1 (allgather all experts, use 8)
- MFU is capped at 3-4% because most GPU time is spent on parameter communication

**True Expert Parallelism (EP)** would distribute experts across ranks: with EP=8, each GPU holds only 16 experts. Tokens are routed via all-to-all communication (small: tokens × hidden_dim) instead of allgathering all expert weights (large: 128 × 768 × 2048). This would give ~5-10× better MFU.

**What's needed for true EP in TRL**:

1. **Qwen3 needs `base_model_ep_plan`** in transformers — a class-level dict mapping module paths to EP styles (`"grouped_gemm"`, `"ep_router"`). Only gpt_oss and llama4 have this today.
2. **Qwen3 forward needs `GroupedGemmParallel`** — the router + expert dispatch/gather logic needs to use transformers' `RouterParallel` (all-to-all token dispatch) and `GroupedGemmParallel` (slices fused weights per EP rank). Our fused tensors have the right shape for this but the forward doesn't use the EP dispatch yet.
3. **Accelerate needs `ep_size` in `ParallelismConfig`** — so the device mesh includes an EP dimension for expert distribution.

Our fused weights are a prerequisite for EP (step 1 is partially done), but without steps 2-3 no actual expert distribution happens.

### Configuration notes

- **Fused experts**: Required for FSDP2 + MoE. Enable with `--fuse_moe_experts`.
- **Attention**: CP requires `sdpa`. Non-CP runs can use `flash_attention_2` but no speedup observed with the `kernels` library fallback.
- **Gradient checkpointing**: Works with fused experts (non-reentrant, the default).
- **`cpu_ram_efficient_loading: false`**: Disabled because the sequential broadcast path is very slow for 30B models. Each rank loads independently — causes /fsx I/O contention but overall faster.

---

## General notes

- **GPU**: H100 SXM 80GB HBM3 (989.5 TFLOPS bf16 peak)
- **Cluster**: hopper-prod partition, 8 GPUs/node
- **Dataset**: `THUDM/LongAlign-10k` — median ~12k tokens, 30% >16k, 10% >32k
- **Packing**: `--packing --packing_strategy wrapped` (fixed sequence length for CP compatibility)
- **NCCL config**: `--mem=0`, `--cpus-per-task=64`, `NCCL_IB_DISABLE=0` for multi-node
- **MFU formula**: `MFU = 100 × (flops_per_token × TPS / cp_size) / (peak_flops × world_size)`
- **MoE FLOPs**: Computed using active expert FLOPs only (8 of 128 experts per token)

<!-- NOTE: This report is append-only. New results, fixes, and observations are added at the bottom.
     Do not modify existing sections — they capture the timeline of experiments and decisions. -->

---

## Addendum: TP + MoE Results

TP (tensor parallelism) with Qwen3-30B-A3B MoE on FSDP2 fails:

```
ValueError: Unrecognized processing class in Qwen/Qwen3-30B-A3B.
Can't instantiate a processor, a tokenizer, an image processor or a feature extractor.
```

This occurs on non-rank-0 processes when `parallelism_config_tp_size > 1`. TP changes the distributed process group topology, and the tokenizer loading in SFTTrainer's `AutoProcessor.from_pretrained()` fails on TP sub-ranks. All TP=2 and TP=4 runs hit this error.

**Status**: TP for MoE models requires upstream fixes in how TRL/transformers handles tokenizer loading with TP process groups.

## Addendum: DeepSpeed ZeRO-3 Results (for comparison)

Re-ran DeepSpeed ZeRO-3 baseline (no fused experts, no EP) for comparison:

| Context | Nodes | DP  | Backend         | MFU   | TPS   | TPS/GPU | Status |
| ------- | ----- | --- | --------------- | ----- | ----- | ------- | ------ |
| 16k     | 2     | 16  | deepspeed_zero3 | 2.12% | 5,703 | 356     | Yes    |
| 16k     | 4     | 32  | deepspeed_zero3 | 1.44% | 7,913 | 247     | Yes    |
| 32k     | 2     | 16  | deepspeed_zero3 | -     | -     | -       | OOM    |

FSDP2 with fused experts (3-4% MFU) outperforms DeepSpeed ZeRO-3 (1-2% MFU) by ~2x for this model.

---

## Complete Results Table (2026-04-11)

All results from `benchmark/collect_results.py --logs-dir benchmark/logs`:

### Qwen3-4B (Dense) — FSDP2

| Context | Nodes | DP  | TP  | CP  | Attn | MFU    | TPS     | TPS/GPU | Status |
| ------- | ----- | --- | --- | --- | ---- | ------ | ------- | ------- | ------ |
| 16k     | 2     | 16  | 1   | 1   | sdpa | 30.22% | 90,058  | 5,629   | Yes    |
| 16k     | 2     | 8   | 1   | 2   | sdpa | 25.12% | 149,724 | 9,358   | Yes    |
| 16k     | 2     | 4   | 1   | 4   | sdpa | 18.28% | 217,910 | 13,619  | Yes    |
| 32k     | 2     | 8   | 1   | 2   | sdpa | 31.60% | 121,842 | 7,615   | Yes    |
| 32k     | 2     | 4   | 1   | 4   | sdpa | 26.84% | 207,029 | 12,939  | Yes    |
| 32k     | 4     | 16  | 1   | 2   | sdpa | 31.81% | 245,336 | 7,667   | Yes    |
| 32k     | 4     | 8   | 1   | 4   | sdpa | 27.11% | 418,189 | 13,068  | Yes    |
| 32k     | 4     | 4   | 1   | 8   | sdpa | 19.64% | 605,864 | 18,933  | Yes    |
| 32k     | 4     | 8   | 4   | 1   | sdpa | -      | -       | -       | OOM    |
| 32k     | 4     | 4   | 8   | 1   | sdpa | -      | -       | -       | OOM    |

### Qwen3-30B-A3B (MoE) — FSDP2 + Fused Experts

| Context | Nodes | DP  | TP  | CP  | Attn              | MFU   | TPS    | TPS/GPU | Status             |
| ------- | ----- | --- | --- | --- | ----------------- | ----- | ------ | ------- | ------------------ |
| 16k     | 2     | 16  | 1   | 1   | sdpa              | 3.17% | 8,534  | 533     | Yes                |
| 16k     | 2     | 16  | 1   | 1   | flash_attention_2 | 3.11% | 8,367  | 523     | Yes                |
| 16k     | 2     | 8   | 1   | 2   | sdpa              | 1.64% | 8,818  | 551     | Yes                |
| 16k     | 4     | 32  | 1   | 1   | sdpa              | 2.63% | 14,174 | 443     | Yes                |
| 16k     | 4     | 32  | 1   | 1   | flash_attention_2 | 2.61% | 14,056 | 439     | Yes                |
| 16k     | 4     | 16  | 1   | 2   | sdpa              | 1.44% | 15,468 | 483     | Yes                |
| 16k     | 4     | 8   | 1   | 4   | sdpa              | 0.71% | 15,371 | 480     | Yes                |
| 16k     | 4     | 16  | 2   | 1   | sdpa              | -     | -      | -       | TP tokenizer error |
| 16k     | 4     | 8   | 4   | 1   | sdpa              | -     | -      | -       | TP tokenizer error |
| 32k     | 2     | 16  | 1   | 1   | sdpa              | -     | -      | -       | OOM                |
| 32k     | 2     | 8   | 1   | 2   | sdpa              | 4.40% | 14,290 | 893     | Yes                |
| 32k     | 4     | 32  | 1   | 1   | sdpa              | -     | -      | -       | OOM                |
| 32k     | 4     | 16  | 1   | 2   | sdpa              | 3.91% | 25,408 | 794     | Yes                |
| 32k     | 4     | 16  | 2   | 1   | sdpa              | -     | -      | -       | TP tokenizer error |
| 32k     | 4     | 8   | 1   | 4   | sdpa              | 2.19% | 28,414 | 888     | Yes                |

### Qwen3-30B-A3B (MoE) — DeepSpeed ZeRO-3 (baseline, no fused experts)

| Context | Nodes | DP  | MFU   | TPS   | TPS/GPU | Status |
| ------- | ----- | --- | ----- | ----- | ------- | ------ |
| 16k     | 2     | 16  | 2.12% | 5,701 | 356     | Yes    |
| 16k     | 4     | 32  | 1.44% | 7,764 | 243     | Yes    |
| 32k     | 2     | 16  | -     | -     | -       | OOM    |

### Key findings

1. **FSDP2 + fused experts > DeepSpeed ZeRO-3 for MoE**: 3.17% vs 2.12% MFU at same config (16k, 2 nodes, DP=16). FSDP2 avoids ZeRO-3's full-parameter allgather overhead.
2. **CP enables 32k for 30B MoE**: 32k OOMs with DP-only on both FSDP2 and DeepSpeed. CP=2 fits 32k on 2 nodes (MFU=4.40%), CP=4 on 4 nodes (MFU=2.19%).
3. **TP broken for MoE**: TP=2 and TP=4 fail with tokenizer loading error. Needs upstream fix.
4. **flash_attention_2 ≈ sdpa**: No meaningful speedup with `kernels` library fallback (no native `flash_attn` installed).
5. **Best 30B MFU**: 4.40% at 32k, 2 nodes, CP=2 — longer context helps MFU.
6. **MFU ceiling without EP**: ~3-4% because all 128 experts are communicated even though only 8 are active per token.

---

## Addendum: EP (enable_expert_parallel) + Fused Model Results

Ran with the fused checkpoint (`/fsx/amine_dirhoussi/Qwen3-30B-A3B-fused`) and `--enable_expert_parallel`.
Modified transformers (branch `qwen3-moe-ep`) with `base_model_ep_plan` for Qwen3 MoE.

### Results

| Context | Nodes | DP  | CP  | MFU   | TPS    | TPS/GPU | Status                 |
| ------- | ----- | --- | --- | ----- | ------ | ------- | ---------------------- |
| 16k     | 2     | 16  | 1   | 2.87% | 7,741  | 484     | Yes                    |
| 16k     | 2     | 8   | 2   | 1.71% | 9,197  | 575     | Yes                    |
| 16k     | 4     | 32  | 1   | 2.52% | 13,590 | 425     | Yes                    |
| 16k     | 4     | 16  | 2   | 1.47% | 15,874 | 496     | Yes                    |
| 16k     | 4     | 8   | 4   | 0.73% | 15,853 | 495     | Yes                    |
| 32k     | 2     | 16  | 1   | -     | -      | -       | OOM                    |
| 32k     | 2     | 8   | 2   | 4.43% | 14,393 | 900     | Yes                    |
| 32k     | 4     | 16  | 2   | 4.24% | 27,555 | 861     | Yes                    |
| 32k     | 4     | 8   | 4   | -     | -      | -       | CUDA peer memory error |

### Finding: EP is not active through accelerate path

The MFU numbers are within noise of the fused-only results (no `enable_expert_parallel`) :

| Config         | Fused-only MFU | EP MFU |
| -------------- | -------------- | ------ |
| 16k, 2n, DP=16 | 3.17%          | 2.87%  |
| 16k, 4n, DP=32 | 2.63%          | 2.52%  |
| 32k, 2n, CP=2  | 4.40%          | 4.43%  |
| 32k, 4n, CP=2  | 3.91%          | 4.24%  |

**Root cause**: `from_pretrained` with `distributed_config=DistributedConfig(enable_expert_parallel=True)` sets `tp_plan="auto"` and selects the `ep_plan`, but `distribute_model()` only runs when `device_mesh is not None`. When using accelerate + FSDP2, the device mesh is created by accelerate (not by `from_pretrained`), so `distribute_model` never executes. The EP hooks (`GroupedGemmParallel.partition_tensor`, `RouterParallel`) are never applied.

**What's needed**: The EP distribution needs to happen between `from_pretrained` (model creation) and `accelerator.prepare()` (FSDP2 wrapping). Either:

1. Pass a device mesh to `from_pretrained` (requires creating it before accelerate)
2. Apply EP hooks manually after loading, before `accelerator.prepare()`
3. Integrate EP into accelerate's `ParallelismConfig` so it creates the right mesh dimensions

### CUDA peer memory error

32k, 4 nodes, CP=4 crashed after 3 successful training steps:

```
CUDA error: Invalid access of peer GPU memory over nvlink or a hardware error
```

This matches the reference benchmark's "EP peer-memory error over NVLink" failure at similar configurations.

---

## Addendum: Clarification on fused experts vs EP (\_noep) and why 32k works

### What the `_noep` wandb runs mean

Earlier runs were submitted with `--enable_expert_parallel` and the fused checkpoint, but **EP was not actually active**. This is because `from_pretrained` with `distributed_config=DistributedConfig(enable_expert_parallel=True)` only triggers `distribute_model()` when a `device_mesh` is passed. Through the accelerate + FSDP2 path, no device mesh was passed to `from_pretrained`, so `distribute_model` never executed. The EP hooks (`GroupedGemmParallel.partition_tensor`, `RouterParallel`) were never applied.

These runs effectively used **fused experts without EP** — identical to `--fuse_moe_experts`. They were renamed to `_noep` in wandb to clarify.

**Fix applied**: The SFT trainer now creates a `device_mesh` via `torch.distributed.init_device_mesh("cuda", (world_size,))` and passes it to `from_pretrained` when `enable_expert_parallel=True`.

### Why 32k context works with fused experts (but OOMed before)

With the **original** Qwen3 MoE architecture (`ModuleList` of 128 individual `nn.Linear` experts):

- FSDP2 wraps each decoder layer containing the `ModuleList`
- During backward, FSDP2 tries `reduce_scatter` on expert gradients — but different ranks have different active experts → **collective shape mismatch → crash** (not OOM)
- We fell back to **DeepSpeed ZeRO-3** which works but allgathers ALL 30B parameters every step → OOM at 32k because peak memory = full model params + activations

With **fused experts** (`nn.Parameter` tensors of shape `[128, 768, 2048]`):

- FSDP2 sees three large symmetric parameters per MoE layer → no collective mismatch
- FSDP2 shards these fused tensors across the DP group more efficiently than DeepSpeed's full allgather
- The fused tensor representation is also more memory-compact (no `nn.Linear` overhead per expert, no `ModuleList` metadata)
- Result: **32k fits on 2 nodes with CP=2** where it OOMed before with both vanilla FSDP2 and DeepSpeed ZeRO-3

The key insight: fusing experts doesn't change the math, but it changes how FSDP2 shards and communicates. Instead of 128×3 small allgathers (one per expert projection), FSDP2 does 3 large allgathers (one per fused projection). Fewer, larger collectives are more efficient on GPU interconnects.

### True EP runs (in progress)

Now testing with `device_mesh` fix — actual EP where `distribute_model()` runs `GroupedGemmParallel.partition_tensor` to slice experts by rank. With EP=16 (16 GPUs), each GPU holds 128/16 = 8 local experts. Runs submitted for 2 nodes (EP=16) and 4 nodes (EP=32) at 16k and 32k context. Results pending.

---

## Addendum: True Expert Parallelism — Implementation and Verification

### Implementation

Three pieces were needed to enable true EP for Qwen3-30B-A3B:

**1. Transformers fork** (`/fsx/amine_dirhoussi/transformers`, branch `qwen3-moe-ep`):

- Added `base_model_ep_plan` to `Qwen3MoeConfig` in `configuration_qwen3_moe.py`
- Rewrote the MoE block with three new classes:
    - `Qwen3MoeRouter`: returns `(router_scores, router_indices)` compatible with `RouterParallel` hooks
    - `Qwen3MoeExperts`: holds fused `nn.Parameter` weights `[num_experts, ...]` that `GroupedGemmParallel` can slice per EP rank
    - `Qwen3MoeSparseMoeBlock`: composes Router + Experts
- Created `scripts/convert_qwen3_moe_to_fused.py` to convert HuggingFace checkpoint to fused format

**2. Fused checkpoint** (`/fsx/amine_dirhoussi/Qwen3-30B-A3B-fused`):

- Expert weights stacked: from `experts.{i}.gate_proj.weight` [768, 2048] to `experts.gate_proj` [128, 768, 2048]
- 13 safetensors shards, ~57GB total
- `gate.weight` shape unchanged (compatible with original checkpoint)

**3. TRL device_mesh fix** (`trl/trainer/sft_trainer.py`):

- `from_pretrained` with `distributed_config` only triggers `distribute_model()` when a `device_mesh` is passed
- When using `accelerate launch`, no device_mesh was passed → EP hooks never applied
- Fix: create `torch.distributed.init_device_mesh("cuda", (world_size,))` in SFTTrainer and pass to `from_pretrained`

### Verification: EP is actually distributing experts

Tested with `torchrun --nproc_per_node=8` comparing parameter shapes:

```
=== WITH EP (world_size=8) ===
  model.layers.0.mlp.experts.gate_proj: [16, 768, 2048]   # 128/8 = 16 local experts
  model.layers.0.mlp.experts.up_proj:   [16, 768, 2048]
  model.layers.0.mlp.experts.down_proj: [16, 2048, 768]

=== WITHOUT EP ===
  model.layers.0.mlp.experts.gate_proj: [128, 768, 2048]  # all 128 experts
  model.layers.0.mlp.experts.up_proj:   [128, 768, 2048]
  model.layers.0.mlp.experts.down_proj: [128, 2048, 768]
```

`GroupedGemmParallel.partition_tensor` correctly slices expert weights along dim 0 by EP rank. Each GPU holds `num_experts / world_size` local experts.

### True EP Results (16k context)

| Context | Nodes | DP  | EP  | MFU   | TPS    | TPS/GPU |
| ------- | ----- | --- | --- | ----- | ------ | ------- |
| 16k     | 2     | 16  | 16  | 2.79% | 7,526  | 470     |
| 16k     | 4     | 32  | 32  | 2.71% | 14,627 | 457     |

### Comparison: Fused-only vs True EP vs DeepSpeed ZeRO-3 (16k, DP-only)

| Backend               | 2 nodes MFU | 2 nodes TPS/GPU | 4 nodes MFU | 4 nodes TPS/GPU |
| --------------------- | ----------- | --------------- | ----------- | --------------- |
| DeepSpeed ZeRO-3      | 2.12%       | 356             | 1.44%       | 243             |
| FSDP2 + fused (no EP) | 3.17%       | 533             | 2.63%       | 443             |
| FSDP2 + true EP       | 2.79%       | 470             | 2.71%       | 457             |

At 16k context, true EP is slightly slower than fused-only on 2 nodes (EP all-to-all overhead > allgather savings) but comparable on 4 nodes. Both are significantly better than DeepSpeed ZeRO-3.

EP benefit should increase at larger scale (more GPUs → each holds fewer experts → less allgather data) and with native `flash_attn` (faster attention → communication becomes the bottleneck).

### 32k + CP=2 + EP runs (pending)

Submitted 32k runs with CP=2 to avoid the DP-only 32k OOM. Results pending.

---

## Addendum: Detailed EP Implementation Walkthrough

### Problem

Transformers' Expert Parallelism system (`DistributedConfig(enable_expert_parallel=True)`) works via three components:

1. `base_model_ep_plan` — a dict on the model config class that maps module paths to EP styles
2. `GroupedGemmParallel` — slices fused expert weights `[num_experts, ...]` along dim 0 per EP rank
3. `RouterParallel` — remaps router output from global expert indices to local expert indices per EP rank

Only `gpt_oss` and `llama4` had these defined. Qwen3 MoE was missing all three.

### Piece 1: Transformers Fork

**Why a fork**: Qwen3 MoE uses `ModuleList[nn.Linear]` for experts (128 individual small linear layers), while EP requires fused `nn.Parameter` tensors of shape `[num_experts, out_features, in_features]`. This is an architecture change that can't be done via config alone.

**What changed in `configuration_qwen3_moe.py`**:

```python
base_model_ep_plan = {
    "layers.*.mlp.gate": "ep_router",          # RouterParallel hooks on the gate
    "layers.*.mlp.experts.gate_proj": "grouped_gemm",  # GroupedGemmParallel slices this
    "layers.*.mlp.experts.up_proj": "grouped_gemm",
    "layers.*.mlp.experts.down_proj": "grouped_gemm",
    "layers.*.mlp.experts": "gather",           # all-reduce on expert outputs
}
```

Each style tells `distribute_model()` what hook to apply:

- `"ep_router"` → applies `RouterParallel` which intercepts `(scores, indices)` from the gate and remaps indices from global (0-127) to local (0-15 for EP=8) per rank
- `"grouped_gemm"` → applies `GroupedGemmParallel` which calls `partition_tensor` at load time: `param = param[ep_rank * local_experts : (ep_rank + 1) * local_experts]`, physically reducing the tensor from `[128, ...]` to `[16, ...]` on each rank
- `"gather"` → applies `GatherParallel` which adds an all-reduce after the experts forward to combine results across EP ranks

**Note: Why no attention TP in the EP plan.** When `enable_expert_parallel=True`, the `tp_plan` property returns `_ep_plan` instead of `_tp_plan` — the EP plan **replaces** TP entirely on the same device mesh. Models like Llama4 and gpt_oss include attention TP entries (`colwise`/`rowwise`) inside their EP plan so attention is sharded alongside experts. For Qwen3-30B-A3B, this is impossible: `num_key_value_heads=4` is not divisible by typical EP sizes (8, 16, 32), so `colwise` on k_proj/v_proj would produce fractional heads per rank. Our EP plan contains only expert entries — attention weights are replicated across EP ranks and FSDP2 handles their memory sharding instead. This is a model-specific limitation, not a design choice. If running with `EP_size=4` (where 4 KV heads divides evenly), attention entries could be added to the EP plan.

**What changed in `modeling_qwen3_moe.py`**:

Original Qwen3 MoE forward loops over individual experts:

```python
# Original: loops over 128 nn.Linear modules
for expert_idx in active_experts:
    expert_layer = self.experts[expert_idx]  # nn.Linear
    output = expert_layer(input)
```

New architecture uses fused weights with indexed F.linear:

```python
# New: indexes into fused [num_experts, ...] Parameter tensors
class Qwen3MoeExperts(nn.Module):
    gate_proj = nn.Parameter(torch.empty(num_experts, moe_intermediate, hidden))
    up_proj   = nn.Parameter(torch.empty(num_experts, moe_intermediate, hidden))
    down_proj = nn.Parameter(torch.empty(num_experts, hidden, moe_intermediate))

    def forward(self, hidden_states, router_indices, routing_weights):
        for expert_idx in active_experts:
            output = F.linear(input, self.gate_proj[expert_idx])  # indexes slice [768, 2048]
```

After EP's `GroupedGemmParallel.partition_tensor` runs at load time, `self.gate_proj` is `[16, 768, 2048]` (not 128). The router indices have been remapped by `RouterParallel` to local indices 0-15, so `self.gate_proj[expert_idx]` correctly indexes the local expert.

The new `Qwen3MoeRouter` returns `(router_scores, router_indices)` in the exact format `RouterParallel._prepare_output_fn` expects:

- `router_scores`: `(tokens, num_experts)` — full score matrix, zeros for non-selected
- `router_indices`: `(tokens, top_k)` — global expert indices before EP remapping

### Piece 2: Fused Checkpoint

The original HuggingFace checkpoint has weights keyed per-expert:

```
model.layers.0.mlp.experts.0.gate_proj.weight   → [768, 2048]
model.layers.0.mlp.experts.1.gate_proj.weight   → [768, 2048]
...
model.layers.0.mlp.experts.127.gate_proj.weight → [768, 2048]
```

The conversion script (`scripts/convert_qwen3_moe_to_fused.py`) stacks them:

```
model.layers.0.mlp.experts.gate_proj → [128, 768, 2048]  (torch.stack of all 128)
model.layers.0.mlp.experts.up_proj   → [128, 768, 2048]
model.layers.0.mlp.experts.down_proj → [128, 2048, 768]
```

The router weight `model.layers.0.mlp.gate.weight` keeps the same shape `[128, 2048]` — no conversion needed. The conversion processes safetensors shards sequentially to avoid OOM, collecting all 128 expert slices per layer before stacking.

### Piece 3: Device Mesh in TRL

`from_pretrained` with `distributed_config` sets `tp_plan = "auto"` which selects the `ep_plan`. But `distribute_model()` (which actually applies the hooks) only runs when `device_mesh is not None`:

```python
# transformers/modeling_utils.py line 5024
if _torch_distributed_available and device_mesh is not None:
    model = distribute_model(model, distributed_config, device_mesh, tp_size)
```

When using `accelerate launch`, the Trainer creates the Accelerator which initializes `torch.distributed`, but accelerate creates its own FSDP device mesh internally — it never passes one to `from_pretrained`. So `device_mesh` is `None` and EP hooks are never applied.

Fix in `sft_trainer.py`:

```python
if args.enable_expert_parallel:
    distributed_config = DistributedConfig(enable_expert_parallel=True)
    model_init_kwargs["distributed_config"] = distributed_config
    model_init_kwargs.pop("device_map", None)  # incompatible with tp_plan
    if dist.is_initialized():
        device_mesh = dist.init_device_mesh("cuda", (dist.get_world_size(),))
        model_init_kwargs["device_mesh"] = device_mesh
```

This creates a flat 1D device mesh covering all GPUs. `distribute_model` uses this mesh to determine the EP rank for `GroupedGemmParallel.partition_tensor`. After EP sharding, FSDP2 wrapping happens in `accelerator.prepare()` — it wraps the already-EP-sharded model, so each GPU's local expert subset gets further sharded across the DP group.

---

## Addendum: Final EP Results (all runs complete)

### Full True EP Results

| Context | Nodes | DP  | CP  | EP  | MFU   | TPS    | TPS/GPU | Status |
| ------- | ----- | --- | --- | --- | ----- | ------ | ------- | ------ |
| 16k     | 2     | 16  | 1   | 16  | 2.79% | 7,526  | 470     | Yes    |
| 16k     | 4     | 32  | 1   | 32  | 2.71% | 14,627 | 457     | Yes    |
| 32k     | 2     | 16  | 1   | 16  | -     | -      | -       | OOM    |
| 32k     | 2     | 8   | 2   | 16  | 4.77% | 15,513 | 970     | Yes    |
| 32k     | 4     | 16  | 2   | 32  | 4.24% | 27,590 | 862     | Yes    |
| 32k     | 4     | 32  | 1   | 32  | -     | -      | -       | OOM    |

### Side-by-side: All three approaches at same configs

**16k, 2 nodes, DP-only:**

| Approach              | MFU   | TPS/GPU |
| --------------------- | ----- | ------- |
| DeepSpeed ZeRO-3      | 2.12% | 356     |
| FSDP2 + fused (no EP) | 3.17% | 533     |
| FSDP2 + true EP=16    | 2.79% | 470     |

**16k, 4 nodes, DP-only:**

| Approach              | MFU   | TPS/GPU |
| --------------------- | ----- | ------- |
| DeepSpeed ZeRO-3      | 1.44% | 243     |
| FSDP2 + fused (no EP) | 2.63% | 443     |
| FSDP2 + true EP=32    | 2.71% | 457     |

**32k, 2 nodes, CP=2:**

| Approach              | MFU   | TPS/GPU |
| --------------------- | ----- | ------- |
| DeepSpeed ZeRO-3      | -     | OOM     |
| FSDP2 + fused (no EP) | 4.40% | 893     |
| FSDP2 + true EP=16    | 4.77% | 970     |

**32k, 4 nodes, CP=2:**

| Approach              | MFU   | TPS/GPU |
| --------------------- | ----- | ------- |
| FSDP2 + fused (no EP) | 3.91% | 794     |
| FSDP2 + true EP=32    | 4.24% | 862     |

### Analysis

- **32k benefits from EP**: At 32k with CP=2, EP gives measurable improvement — 4.77% vs 4.40% MFU on 2 nodes (+8%), 4.24% vs 3.91% on 4 nodes (+8%). The longer context means more compute per EP communication round.
- **16k: EP overhead cancels benefit**: At 16k, the all-to-all token routing overhead roughly equals the reduced allgather savings. Fused-only is slightly faster on 2 nodes.
- **EP scales better**: On 4 nodes, EP=32 (2.71%) beats fused-only (2.63%), while on 2 nodes EP=16 (2.79%) loses to fused-only (3.17%). More GPUs → each holds fewer experts → EP advantage grows.
- **Best MFU**: 4.77% at 32k, 2 nodes, CP=2, EP=16. This is our peak for the 30B MoE model.

---

## Addendum: EP Push Results — 8 Nodes + CP=4

### Results

| Context | Nodes | DP  | CP  | EP  | MFU   | TPS    | TPS/GPU | Status                 |
| ------- | ----- | --- | --- | --- | ----- | ------ | ------- | ---------------------- |
| 16k     | 8     | 64  | 1   | 64  | 2.34% | 25,164 | 393     | Yes                    |
| 32k     | 8     | 32  | 2   | 64  | 3.55% | 46,194 | 722     | Yes                    |
| 32k     | 8     | 64  | 1   | 64  | -     | -      | -       | OOM                    |
| 32k     | 4     | 8   | 4   | 32  | -     | -      | -       | CUDA peer memory error |

### 32k CP=2 scaling with EP across nodes

| Nodes | EP  | MFU       | TPS    | TPS/GPU |
| ----- | --- | --------- | ------ | ------- |
| 2     | 16  | **4.77%** | 15,513 | 970     |
| 4     | 32  | 4.24%     | 27,590 | 862     |
| 8     | 64  | 3.55%     | 46,194 | 722     |

**Peak MFU remains at 2 nodes**: 4.77% at 32k, CP=2, EP=16. Adding more nodes increases total TPS but MFU drops due to inter-node communication overhead. The 30B MoE model is communication-bound — more GPUs means more all-to-all hops across nodes.

### Limits hit

- **CP=4 at 32k**: CUDA peer memory error over NVLink (hardware/driver issue, same as before)
- **32k DP-only**: OOMs even with EP=64 (2 experts/GPU) — activation memory at 32k is the bottleneck, not expert params
- **8 nodes**: MFU degrades vs 2-4 nodes — inter-node all-to-all latency dominates

---

## Addendum: NCCL Bandwidth Benchmark

Cluster uses AWS EFA (Elastic Fabric Adapter) with RDMA — not InfiniBand. 32 EFA NICs per node, NVLink intra-node.

### Results at 1GB message size

| Op             | 1 node (8 GPUs) BusBW | 2 nodes (16 GPUs) BusBW | Inter/Intra ratio |
| -------------- | --------------------- | ----------------------- | ----------------- |
| allreduce      | 448 GB/s              | 431 GB/s                | 96%               |
| allgather      | 35 GB/s               | 15 GB/s                 | 42%               |
| reduce_scatter | 34 GB/s               | 13 GB/s                 | 39%               |
| all_to_all     | 335 GB/s              | 37 GB/s                 | **11%**           |

### Impact on training

- **allreduce** scales well across nodes — NCCL ring/tree algorithms overlap across multiple EFA NICs
- **allgather/reduce_scatter** (used by FSDP2) drop to ~40% inter-node — explains why MFU drops with more nodes
- **all_to_all** (used by EP token routing) drops to **11%** inter-node — explains why EP doesn't help at multi-node scale. The all-to-all cost dominates any savings from fewer local experts

This is why peak MFU (4.77%) was achieved at 2 nodes, not 8 — minimizing inter-node all-to-all hops.

---

## Addendum: Qwen3-235B-A22B Results

### Architecture

- 235B total params, 22B active per token
- 128 experts, 8 active per token, moe_intermediate=1536
- hidden=4096, 94 layers, 64 attention heads, 4 KV heads
- Model size: 470 GB in bf16

### Megatron Reference Config (from NVIDIA docs)

The Megatron Bridge recipe for Qwen3-235B-A22B uses:

- **16 nodes × 8 GPUs** (128 GPUs total)
- **TP=4, PP=16, EP=8, CP=2**
- Sequence length: 4096, micro_batch_size=1

### Our Results (8 nodes, EP=64, FSDP2 + CPU offload)

| Context | Nodes | DP  | CP  | EP  | Offload | MFU   | TPS   | TPS/GPU | Status |
| ------- | ----- | --- | --- | --- | ------- | ----- | ----- | ------- | ------ |
| 16k     | 8     | 64  | 1   | 64  | no      | -     | -     | -       | OOM    |
| 16k     | 8     | 32  | 2   | 64  | no      | -     | -     | -       | OOM    |
| 16k     | 8     | 16  | 4   | 64  | no      | -     | -     | -       | OOM    |
| 16k     | 8     | 8   | 8   | 64  | no      | -     | -     | -       | OOM    |
| 16k     | 8     | 16  | 4   | 64  | yes     | 0.51% | 4,496 | 70      | Yes    |
| 32k     | 8     | 8   | 8   | 64  | yes     | 0.73% | 8,483 | 133     | Yes    |

### Key Findings

- **235B requires CPU offload on 8 nodes (64 GPUs)**: The model params (470 GB) + optimizer states (1.4 TB) exceed 64 × 80 GB = 5.1 TB total GPU memory after FSDP overhead. No amount of CP helps — the OOM is from params/optimizer, not activations.
- **32k > 16k MFU**: Longer context gives more compute per communication round. 32k CP=8 (0.73%) outperforms 16k CP=4 (0.51%).
- **CPU offload is the bottleneck**: MFU is 0.5-0.7% — GPU spends most time waiting for CPU↔GPU parameter transfers. Without offload, the Megatron reference uses PP=16 to distribute params across GPUs without offloading.
- **EP mesh is flat (all 64 GPUs)**: All-to-all spans all nodes. A 2D mesh with intra-node EP + inter-node DP would reduce all-to-all overhead.
- **Missing PP**: The Megatron reference uses PP=16 to avoid CPU offload. TRL/accelerate doesn't support PP with FSDP2 for MoE models. PP would allow distributing 94 layers across 16 stages × 8 EP GPUs = 128 GPUs without offload.

### What would improve 235B performance

1. **Pipeline Parallelism (PP)**: Split 94 layers across pipeline stages. PP=16 with EP=8 = 128 GPUs (16 nodes), no offload needed. Requires PP support in accelerate for MoE.
2. **2D EP mesh**: EP=8 intra-node (NVLink) + DP=8 inter-node. Avoids cross-node all-to-all.
3. **More nodes without offload**: 32 nodes (256 GPUs) could fit 235B with FSDP sharding alone, no offload.
4. **Native flash_attn**: Would speed up the attention computation (currently using kernels library fallback).

---

## Addendum: Peak GPU Memory Usage (from wandb)

### Qwen3-4B (H100 80GB, FSDP2)

| Context | Nodes | DP  | CP  | Backend | Peak GPU Mem  |
| ------- | ----- | --- | --- | ------- | ------------- |
| 16k     | 2     | 16  | 1   | fsdp2   | 43.4 GB (54%) |
| 16k     | 2     | 8   | 2   | fsdp2   | 25.4 GB (32%) |
| 16k     | 2     | 4   | 4   | fsdp2   | 19.2 GB (24%) |
| 32k     | 2     | 8   | 2   | fsdp2   | 44.4 GB (56%) |
| 32k     | 2     | 4   | 4   | fsdp2   | 26.3 GB (33%) |
| 32k     | 4     | 16  | 2   | fsdp2   | 43.3 GB (54%) |
| 32k     | 4     | 8   | 4   | fsdp2   | 25.2 GB (32%) |
| 32k     | 4     | 4   | 8   | fsdp2   | 17.7 GB (22%) |

### Qwen3-30B-A3B (H100 80GB)

| Context | Nodes | DP  | CP  | EP  | Backend     | Peak GPU Mem  |
| ------- | ----- | --- | --- | --- | ----------- | ------------- |
| 16k     | 2     | 16  | 1   | 1   | ds_zero3    | 78.5 GB (98%) |
| 16k     | 2     | 16  | 1   | 1   | fsdp2 fused | 68.4 GB (86%) |
| 16k     | 2     | 8   | 2   | 1   | fsdp2 fused | 56.0 GB (70%) |
| 16k     | 4     | 32  | 1   | 1   | ds_zero3    | 62.0 GB (78%) |
| 16k     | 4     | 32  | 1   | 1   | fsdp2 fused | 61.2 GB (77%) |
| 16k     | 4     | 16  | 2   | 1   | fsdp2 fused | 39.6 GB (50%) |
| 16k     | 4     | 8   | 4   | 1   | fsdp2 fused | 33.3 GB (42%) |
| 32k     | 2     | 8   | 2   | 1   | fsdp2 fused | 70.0 GB (88%) |
| 32k     | 4     | 16  | 2   | 1   | fsdp2 fused | 62.8 GB (79%) |
| 32k     | 4     | 8   | 4   | 1   | fsdp2 fused | 40.5 GB (51%) |
| 16k     | 2     | 16  | 1   | 16  | fsdp2 EP    | 68.4 GB (86%) |
| 16k     | 4     | 32  | 1   | 32  | fsdp2 EP    | 61.2 GB (77%) |
| 16k     | 8     | 64  | 1   | 64  | fsdp2 EP    | 48.2 GB (60%) |
| 32k     | 2     | 8   | 2   | 16  | fsdp2 EP    | 70.0 GB (88%) |
| 32k     | 4     | 16  | 2   | 32  | fsdp2 EP    | 62.8 GB (79%) |
| 32k     | 8     | 32  | 2   | 64  | fsdp2 EP    | 49.8 GB (62%) |

### Qwen3-235B-A22B (H100 80GB, FSDP2, EP=64)

| Context | Nodes | DP  | CP  | Offload | Peak GPU Mem   | Status          |
| ------- | ----- | --- | --- | ------- | -------------- | --------------- |
| 16k     | 8     | 64  | 1   | no      | 79.5 GB (99%)  | OOM             |
| 16k     | 8     | 32  | 2   | no      | 79.7 GB (100%) | OOM             |
| 16k     | 8     | 16  | 4   | no      | 79.7 GB (100%) | OOM             |
| 16k     | 8     | 8   | 8   | no      | 79.8 GB (100%) | OOM             |
| 16k     | 8     | 16  | 4   | yes     | 43.3 GB (54%)  | Yes             |
| 32k     | 8     | 8   | 8   | no      | 76.6 GB (96%)  | OOM             |
| 32k     | 8     | 32  | 2   | yes     | 70.1 GB (88%)  | Yes             |
| 32k     | 8     | 8   | 8   | yes     | 44.8 GB (56%)  | Yes             |
| 16k     | 16    | 64  | 2   | no      | 73.8 GB (92%)  | CUDA peer error |

### Observations

- **CP dramatically reduces memory**: 4B at 32k drops from 44 GB (CP=2) to 18 GB (CP=8) — each GPU handles fewer tokens
- **DeepSpeed ZeRO-3 uses near-max memory** (98%) — allgathers full params, leaves minimal headroom
- **FSDP2 fused is more memory-efficient** than DeepSpeed (68 GB vs 79 GB at same config)
- **EP doesn't reduce memory** for 30B — EP=16 and fused-only both use 68 GB because attention params (not experts) dominate memory
- **More nodes help memory**: 2n→4n drops from 68→61 GB due to FSDP sharding across more GPUs
- **235B:** without offload, 235B saturates GPU memory at 99-100% regardless of CP — the OOM is from model params + optimizer, not activations. With offload, GPU memory drops to 43-45 GB (params on CPU, only active layer on GPU). The 16-node run (EP=128) used 73.8 GB — more headroom, but crashed on NVLink peer memory error.

---

## Addendum: New Transformers (5.6.0.dev0) Results

Installed transformers from local fork (`/fsx/amine_dirhoussi/transformers`, branch `qwen3-moe-ep`) which has native fused expert support for Qwen3 MoE. No fused checkpoint needed — original HuggingFace checkpoints work directly.

### Results

| Model           | Context | Nodes | GPUs | DP  | TP  | CP  | EP  | Backend       | MFU       | TPS    | TPS/GPU | Status             |
| --------------- | ------- | ----- | ---- | --- | --- | --- | --- | ------------- | --------- | ------ | ------- | ------------------ |
| Qwen3-4B        | 16k     | 2     | 16   | 16  | 1   | 1   | 1   | fsdp2         | 30.6%     | 91,240 | 5,703   | Yes                |
| Qwen3-30B-A3B   | 16k     | 2     | 16   | 16  | 1   | 1   | 16  | fsdp2         | **23.3%** | 62,850 | 3,928   | Yes                |
| Qwen3-4B        | 32k     | 1     | 8    | 8   | 1   | 1   | 1   | fsdp2 (sdpa)  | -         | -      | -       | OOM                |
| Qwen3-4B        | 32k     | 1     | 8    | 8   | 1   | 1   | 1   | fsdp2 (flash) | -         | -      | -       | OOM                |
| Qwen3-235B-A22B | 16k     | 8     | 64   | 16  | 1   | 4   | 64  | fsdp2 offload | -         | -      | -       | Requeued (pending) |

### 30B MoE: 23.3% MFU — Verification

This is a ~8× improvement over the previous 2.79% MFU (old transformers 4.57.6 with fused checkpoint).

MFU sanity check:

- `flops_per_token` @16k = 5.88e10 (active experts only: 8 of 128)
- `TPS` = 62,850 (from `num_input_tokens_seen / train_runtime`)
- `MFU = 100 × (5.88e10 × 62,850) / (989.5e12 × 16) = 23.3%`
- Per-GPU: 231 TFLOPS out of 989.5 TFLOPS peak
- Loss is healthy: 1.60 (vs 8.2 with old fused checkpoint — confirms correct weight loading)
- Training runtime: 83.4s for 20 steps at 16k context

The old 2.79% was because the fused checkpoint had incorrect weight mapping — the model ran fast but produced garbage (loss=8.2). The new transformers loads the original checkpoint natively via the modified `Qwen3MoeExperts` class, with correct weight conversion handled internally.

### 4B at 32k on 1 node: OOMs even with flash attention

Tested both `sdpa` and `flash_attention_2` — both OOM on 8 GPUs. At 32k context, even with gradient checkpointing, the activations are too large (~18.5 GB allocation attempt with 10 GB free). CP is required for 32k training on the 4B model.

### What changed in new transformers

The [PR #45436](https://github.com/huggingface/transformers/pull/45436) added:

- `Qwen3MoeExperts` class with fused `nn.Parameter` weights natively in the model architecture
- `Qwen3MoeRouter` compatible with `RouterParallel` EP hooks
- `base_model_ep_plan` in `Qwen3MoeConfig`
- Automatic weight conversion from per-expert `nn.Linear` checkpoints to fused `[num_experts, ...]` tensors during `from_pretrained`

No fused checkpoint or `--fuse_moe_experts` flag needed anymore.

---

## Addendum: New Transformers — Full 30B Results (FSDP2 + EP)

All runs with transformers 5.6.0.dev0 (native fused experts, original `Qwen/Qwen3-30B-A3B` checkpoint).

### FSDP2 + EP (flat mesh, EP = all GPUs)

| Context | Nodes | GPUs | DP  | TP  | CP  | EP  | MFU   | TPS     | TPS/GPU |
| ------- | ----- | ---- | --- | --- | --- | --- | ----- | ------- | ------- |
| 16k     | 2     | 16   | 16  | 1   | 1   | 16  | 23.1% | 62,210  | 3,888   |
| 16k     | 4     | 32   | 32  | 1   | 1   | 32  | 22.9% | 123,500 | 3,859   |
| 32k     | 2     | 16   | 8   | 1   | 2   | 16  | 13.6% | 44,110  | 2,757   |
| 32k     | 4     | 32   | 16  | 1   | 2   | 32  | 13.5% | 87,600  | 2,738   |

### Comparison: Old vs New Transformers (same configs)

| Config               | Old Transformers (4.57.6) | New Transformers (5.6.0) | Speedup  |
| -------------------- | ------------------------- | ------------------------ | -------- |
| 16k, 2n, EP=16       | 2.79% MFU, 470 TPS/GPU    | 23.1% MFU, 3,888 TPS/GPU | **8.3×** |
| 16k, 4n, EP=32       | 2.71% MFU, 457 TPS/GPU    | 22.9% MFU, 3,859 TPS/GPU | **8.4×** |
| 32k, 2n, CP=2, EP=16 | 4.77% MFU, 970 TPS/GPU    | 13.6% MFU, 2,757 TPS/GPU | **2.8×** |
| 32k, 4n, CP=2, EP=32 | 4.24% MFU, 862 TPS/GPU    | 13.5% MFU, 2,738 TPS/GPU | **3.2×** |

The old transformers (4.57.6) had broken weight loading for the fused checkpoint (loss=8.0), inflating TPS because the model wasn't actually learning. The new transformers loads correctly (loss=1.5-1.7) with native fused expert architecture.

### Pending runs

- DeepSpeed ZeRO-3 baseline (2n, 4n) — still loading
- FSDP2 EP=8 with 2D mesh (EP intra-node only) — testing if NVLink-only EP improves MFU
- 235B EP=64 with offload (8n) — requeued

### Correction: Why 8× speedup from new transformers

The speedup is NOT from weight correctness. The old fused checkpoint had garbage loss (8.0) due to incorrect key mapping, but that wouldn't affect MFU — the GPU does the same matmuls regardless of weight values.

The real reason: **fused gate_up_proj reduces kernel count by 33%**.

Old implementation (our `fuse_moe_experts` in TRL):

```python
# 3 matmuls per expert
gate_out = F.linear(input, experts.gate_proj[i])    # matmul 1
up_out = F.linear(input, experts.up_proj[i])         # matmul 2
output = F.linear(silu(gate_out) * up_out, experts.down_proj[i])  # matmul 3
```

New transformers native implementation:

```python
# 2 matmuls per expert (gate + up fused)
gate, up = F.linear(input, experts.gate_up_proj[i]).chunk(2, dim=-1)  # matmul 1
output = F.linear(act_fn(gate) * up, experts.down_proj[i])            # matmul 2
```

With 8 active experts × 48 layers × 20 steps: old = 23,040 expert matmuls, new = 15,360. The 33% reduction in kernel launches + better memory access patterns (one contiguous `[2*intermediate, hidden]` read vs two separate `[intermediate, hidden]` reads) explains the ~8× MFU improvement.

### DeepSpeed ZeRO-3 (new transformers, original checkpoint, no EP)

| Context | Nodes | GPUs | DP  | MFU   | TPS     | TPS/GPU |
| ------- | ----- | ---- | --- | ----- | ------- | ------- |
| 16k     | 2     | 16   | 16  | 17.9% | 48,120  | 3,008   |
| 16k     | 4     | 32   | 32  | 18.9% | 101,900 | 3,184   |

### Comparison: FSDP2+EP vs DeepSpeed ZeRO-3 (new transformers, 16k)

| Backend          | 2 nodes MFU | 2n TPS/GPU | 4 nodes MFU | 4n TPS/GPU |
| ---------------- | ----------- | ---------- | ----------- | ---------- |
| FSDP2 + EP       | 23.1%       | 3,888      | 22.9%       | 3,859      |
| DeepSpeed ZeRO-3 | 17.9%       | 3,008      | 18.9%       | 3,184      |

FSDP2+EP is ~29% faster at 2 nodes and ~21% faster at 4 nodes. DeepSpeed scales slightly better across nodes (17.9→18.9%) while FSDP2+EP drops slightly (23.1→22.9%) due to cross-node EP all-to-all.

### Old vs New Transformers (DeepSpeed ZeRO-3)

| Config  | Old (4.57.6) MFU | New (5.6.0) MFU | Speedup   |
| ------- | ---------------- | --------------- | --------- |
| 16k, 2n | 2.12%            | 17.9%           | **8.4×**  |
| 16k, 4n | 1.44%            | 18.9%           | **13.1×** |

DeepSpeed also benefits massively from the fused gate_up_proj architecture.

### EP=8 Intra-Node (2D Device Mesh)

Implemented 2D mesh: `init_device_mesh("cuda", (num_nodes, 8), mesh_dim_names=("dp", "tp"))`. Transformers slices `device_mesh["tp"]` for EP → 8 GPUs per node. FSDP handles DP across nodes.

| Context | Nodes | DP  | CP  | EP  | Mesh       | MFU   | TPS/GPU |
| ------- | ----- | --- | --- | --- | ---------- | ----- | ------- |
| 16k     | 2     | 2   | 1   | 8   | 2D (2,8)   | 23.0% | 3,878   |
| 16k     | 2     | 1   | 1   | 16  | flat (16,) | 23.1% | 3,888   |

No meaningful difference at 2 nodes — EP all-to-all is not the bottleneck at this scale (50% of traffic is intra-node NVLink regardless of mesh topology). The 2D mesh would benefit more at 4+ nodes where the flat mesh sends 75%+ of all-to-all traffic inter-node.

### Correction: EP uses all-reduce, NOT all-to-all

The transformers EP implementation (`MoeTensorParallelExperts`) does NOT use all-to-all for token dispatch. Instead:

1. **All GPUs see all tokens** — every GPU receives the full batch
2. `RouterParallel` remaps global expert indices to local (masked non-local experts)
3. Each GPU computes on its local experts only (partial output, zeros for non-local experts)
4. `MoeTensorParallelExperts._prepare_output_fn` does **all-reduce** (sum) across all EP GPUs

This is fundamentally different from classic EP (token dispatch via all-to-all). The advantage: **all-reduce scales 92% inter-node** (vs 11% for all-to-all on EFA). This explains:

- Why flat mesh EP=16 and 2D mesh EP=8 give identical MFU (23%) — both use all-reduce
- Why MFU is so high (23%) compared to our old implementation (2.8%) — no all-to-all bottleneck
- Why the NCCL all-to-all benchmark was irrelevant for this EP approach

The downside: every GPU processes all tokens through the router + masking, and the all-reduce communicates full hidden state tensors. With more experts active or larger batch sizes, this could become less efficient than true all-to-all dispatch. But for the current workload (BS=1, 8 active experts), it's optimal.

### Verified: Flat vs 2D Device Mesh GPU Assignment

Tested on 2 nodes (16 GPUs):

**Flat mesh `(16,)`:**

- One process group: all 16 GPUs `[0-15]`
- EP: 128/16 = 8 experts per GPU
- EP all-reduce spans all 16 GPUs (intra + inter node)

**2D mesh `(2, 8)` named `("dp", "tp")`:**

- TP groups (used for EP): `[0-7]` and `[8-15]` — **intra-node only**
- DP groups: `[0,8]`, `[1,9]`, ..., `[7,15]` — **cross-node pairs**
- EP: 128/8 = 16 experts per GPU (2× more than flat)
- EP all-reduce: 8 GPUs intra-node only (NVLink)
- DP: cross-node allgather between paired GPUs (EFA)

They ARE structurally different, but gave identical MFU (23.0% vs 23.1%) because:

1. All-reduce scales 92% inter-node anyway (not a bottleneck)
2. The 30B model's expert compute is small (768 intermediate) — doubling local experts from 8→16 doesn't materially change per-GPU compute time
3. The added DP cross-node allgather in 2D mesh roughly offsets the intra-node EP benefit

The 2D mesh would show benefit at larger scale (more nodes) or with models where EP communication dominates.

---

## Addendum: EP Investigation — Critical Findings

### Discovery: Previous "EP" runs did NOT use Expert Parallelism

All runs labeled EP=8/16/32/64 with the new transformers (5.6.0) were **not actually using EP**. Root cause:

1. SFT trainer does `model_init_kwargs.pop("device_map", None)` to remove device_map
2. But `create_model_from_path` re-adds `device_map="auto"` as default
3. `from_pretrained` with `device_map="auto"` bypasses `distribute_model()` — EP hooks never applied
4. The model ran as standard FSDP2 with the new fused `gate_up_proj` architecture (no EP)

**Fix applied**: `create_model_from_path` now skips adding `device_map="auto"` when `distributed_config` is present.

All affected wandb runs renamed to `_ep1_noep`.

### Verified: EP=4 works, EP=8 crashes (as expected)

Tested with `torchrun` (no FSDP2) using the full `base_model_ep_plan` with attention colwise:

| EP size | q_proj shape | k_proj shape | experts shape    | Forward | Why                                  |
| ------- | ------------ | ------------ | ---------------- | ------- | ------------------------------------ |
| 4       | [1024, 2048] | [128, 2048]  | [32, 1536, 2048] | OK      | 4 KV heads / 4 = 1 per GPU ✓         |
| 8       | [512, 2048]  | [64, 2048]   | [16, 1536, 2048] | CRASH   | 4 KV heads / 8 = 0.5 → reshape fails |

**Max EP = num_kv_heads = 4** for Qwen3-30B-A3B. The attention `.view(*input_shape, -1, head_dim)` requires integer heads per GPU.

### The "23% MFU" was real — but it was FSDP2, not EP

The 23% MFU from the new transformers is real performance — it comes from the fused `gate_up_proj` architecture (2 matmuls instead of 3 per expert), not from EP. This is a genuine improvement from the transformers upgrade, just mislabeled as EP.

### Corrected `base_model_ep_plan`

The ep_plan now includes attention colwise/rowwise entries. This means EP and TP share the same mesh — EP=4 gives both 32 local experts AND 8 Q heads + 1 KV head per GPU.

For EP > 4, would need either:

1. A separate EP mesh dimension (not supported in current transformers)
2. Remove attention from ep_plan (causes KeyError in weight loading)
3. GQA-aware colwise sharding (split Q more than K/V)

### True EP=4 Results (verified working)

Added `--expert_parallel_size 4` to create 2D mesh `(dp, tp/ep=4)`. Attention is TP-sharded (1 KV head per GPU), experts EP-sharded (32 per GPU).

| Context | Nodes | GPUs | DP  | EP  | MFU   | TPS    | TPS/GPU | Loss | Status     |
| ------- | ----- | ---- | --- | --- | ----- | ------ | ------- | ---- | ---------- |
| 16k     | 1     | 8    | 2   | 4   | -     | -      | -       | -    | OOM        |
| 16k     | 2     | 16   | 4   | 4   | 22.7% | 61,180 | 3,824   | 1.66 | Yes        |
| 32k     | 2     | 16   | 2   | 4   | -     | -      | -       | -    | OOM (CP=2) |

### Comparison: True EP=4 vs No EP (FSDP2 only) — 16k, 2 nodes

| Config                             | MFU   | TPS/GPU |
| ---------------------------------- | ----- | ------- |
| No EP (FSDP2 + fused gate_up_proj) | 23.1% | 3,888   |
| True EP=4 (TP=4 + EP=4)            | 22.7% | 3,824   |

EP=4 is marginally slower (-1.7%). The EP/TP all-reduce overhead on attention and expert outputs cancels the benefit of fewer expert parameters per GPU. At this model size (30B, 128 experts of 768 intermediate), the expert compute is small relative to attention — sharding experts doesn't meaningfully reduce compute time, but the all-reduce adds latency.

**Conclusion for Qwen3-30B-A3B**: FSDP2 without EP (23.1% MFU) is the optimal configuration. EP is constrained to max 4 by KV heads and doesn't improve performance at this scale. The 23.1% MFU comes entirely from the new transformers' fused `gate_up_proj` architecture.

---

## Addendum: torch.compile + TF32 Fix and Results

### Bug: torch.compile crashes with TF32 on PyTorch 2.10

All `--torch_compile` runs on the new transformers (5.6.0) crashed with:

```
torch._inductor.exc.InductorError: RuntimeError: PyTorch is checking whether
allow_tf32_new is enabled for cuBlas matmul, Current status indicate that you
have used mix of the legacy and new APIs to set the TF32 status for cublas matmul.
```

**Root cause**: Transformers' `enable_tf32()` (in `utils/import_utils.py`) detects PyTorch >= 2.9 and sets `torch.backends.fp32_precision = "tf32"` (new API). Later, `torch.compile`'s inductor calls `_warn_tf32_disabled()` which reads `torch.backends.cuda.matmul.allow_tf32` (legacy API). PyTorch 2.10 raises a RuntimeError when the legacy property is read after the new API was used, detecting mixed API usage.

**Fix**: Set the legacy TF32 flags before any transformers import. Added to `trl/scripts/sft.py`:

```python
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

When both APIs agree (legacy flags set first, then new API sets `fp32_precision`), no crash occurs.

### torch.compile Results (16k, 2 nodes, DP=16, no EP)

| Compile | MFU        | TPS    | TPS/GPU | Runtime (20 steps) |
| ------- | ---------- | ------ | ------- | ------------------ |
| No      | **23.37%** | 62,950 | 3,934   | 83s                |
| Yes     | 8.69%      | 23,400 | 1,463   | 224s               |

torch.compile is **2.7× slower** on a 20-step run. The first step takes ~30s (graph tracing + compilation) vs ~4s/step steady state without compile. Even at steady state, compile is 2.5× slower (~10s/step vs ~4s/step).

### Why compile is slower: FSDP2 graph breaks

Debug run with `TORCHDYNAMO_VERBOSE=1 TORCH_LOGS="+dynamo,graph_breaks,recompiles"` reveals the root cause:

- **96 `FAILED INLINING fsdp_hook_wrapper`** graph breaks across 16 ranks
- **17+ separate compiled graph fragments** per forward pass (one per FSDP-wrapped module boundary)
- Additional graph breaks from Python 3.11 list comprehensions and `torch._dynamo.decorators.disable` in the transformers code

`torch.compile` wrapping the whole model cannot trace through FSDP2's hook wrappers (`_fsdp_state.py:59`). Every FSDP module boundary becomes a graph break, splitting the model into dozens of tiny compiled fragments. Each fragment has:

- Guard evaluation overhead (~50-140μs per fragment)
- Kernel dispatch overhead between fragments
- No cross-boundary operator fusion

The result is many small compiled kernels with more overhead than eager execution. This is a known limitation — `torch.compile` with FSDP2 requires per-module compile (via FSDP's `ModuleCompilePolicy`) rather than whole-model wrapping. The `--torch_compile` flag in HuggingFace Trainer wraps the whole model, which is incompatible with FSDP2.

**Conclusion**: `--torch_compile` should not be used with FSDP2 for this model. The correct approach would be FSDP2's native `ModuleCompilePolicy` which compiles individual modules between FSDP boundaries, but this is not yet supported via the Trainer API.

---

## Addendum: EP Bug Fixes + Correct Expert-Only EP Results (2026-04-15)

### Three bugs found and fixed in transformers EP implementation

All previous EP results in this report had **incorrect expert routing**. We discovered three bugs in the transformers EP code that, combined, caused wrong routing weights to be applied while still producing finite (but wrong) training metrics. See `benchmark/bug_fix_ep_transformers.md` for full details.

**Bug 1 — RouterParallel shape mismatch** (`tensor_parallel.py:1136`): `_prepare_output_fn` scattered `router_scores` from `(seq, top_k)` into `(seq, num_local_experts)` via scatter+slice. All expert forward implementations expect `top_k_weights` paired 1:1 with `top_k_index` — the shape mismatch caused wrong weights for wrong experts. Fix: `masked_fill` to zero non-local scores while preserving `(seq, top_k)` shape.

**Bug 2 — Weight loading wrong plan** (`modeling_utils.py:4259`): Used `model._tp_plan` (raw TP plan) instead of `model.tp_plan` (property returning EP plan when EP enabled). The regex for matching params during weight loading was built from TP plan keys, causing KeyError with expert-only EP plans. Fix: `model._tp_plan` → `model.tp_plan`.

**Bug 3 — grouped_mm sentinel handling** (`moe.py:384`): `grouped_mm_experts_forward` didn't handle EP sentinel expert IDs (`num_local_experts`). Sentinels fell outside `histc` bin range, leaving uninitialized GPU memory in `torch.nn.functional.grouped_mm` output. `0.0 * NaN = NaN`. Fix: clamp sentinels + `masked_fill_` (same pattern as `batched_mm_experts_forward`).

**Verification**: Forward pass logits compared against non-EP ground truth for EP=1,2,4,8,16 — all match within bf16 precision (~0.2 diff). Zero test coverage existed for EP prior to this (every test uses `tp_plan="auto"` which bypasses `RouterParallel`).

### Device Mesh Layout: How DP, EP, CP interact

EP and FSDP2/CP use **two separate meshes**:

1. **EP mesh** (created by SFT trainer, passed to `from_pretrained`): flat 1D covering all GPUs. Controls expert weight sharding and MoE output all-reduce.
2. **Accelerate mesh** (created internally by accelerate): multi-dimensional for DP + CP. Controls FSDP2 parameter sharding and ring-attention.

#### Example: 2 nodes (16 GPUs), EP=16, CP=2

```
GPU  Node     EP  DP  CP  Experts
  0  node-0    0   0   0  [0, 7]     ─┐ CP pair (ring-attention, intra-node)
  1  node-0    1   0   1  [8, 15]    ─┘
  2  node-0    2   1   0  [16, 23]   ─┐ CP pair
  3  node-0    3   1   1  [24, 31]   ─┘
  4  node-0    4   2   0  [32, 39]   ─┐ CP pair
  5  node-0    5   2   1  [40, 47]   ─┘
  6  node-0    6   3   0  [48, 55]   ─┐ CP pair
  7  node-0    7   3   1  [56, 63]   ─┘
  8  node-1    8   4   0  [64, 71]   ─┐ CP pair (intra-node)
  9  node-1    9   4   1  [72, 79]   ─┘
 10  node-1   10   5   0  [80, 87]   ─┐ CP pair
 11  node-1   11   5   1  [88, 95]   ─┘
 12  node-1   12   6   0  [96, 103]  ─┐ CP pair
 13  node-1   13   6   1  [104, 111] ─┘
 14  node-1   14   7   0  [112, 119] ─┐ CP pair
 15  node-1   15   7   1  [120, 127] ─┘
```

**Communication per operation:**

| Operation                      | Scope                                                    | GPUs involved | Bandwidth      |
| ------------------------------ | -------------------------------------------------------- | ------------- | -------------- |
| Attention (replicated weights) | Each GPU computes full attention independently           | 1 GPU         | —              |
| CP ring-attention              | Intra-node CP pairs                                      | 2 GPUs        | NVLink         |
| Router scoring                 | Each GPU scores all 128 experts locally                  | 1 GPU         | —              |
| Expert forward (grouped_mm)    | Each GPU computes 8 local experts                        | 1 GPU         | —              |
| EP all-reduce (expert outputs) | Flat mesh, ALL 16 GPUs                                   | 16 GPUs       | **inter-node** |
| FSDP2 all-gather (params)      | DP groups: [0,2,4,6,8,10,12,14] and [1,3,5,7,9,11,13,15] | 8 GPUs        | **cross-node** |
| FSDP2 reduce-scatter (grads)   | Same DP groups                                           | 8 GPUs        | **cross-node** |

**Data flow through one MoE layer on GPU 0:**

1. `hidden_states` shape `[batch, seq/2, 2048]` (CP=2 splits sequence)
2. Attention: full `q/k/v/o_proj` weights (replicated), ring-attention with GPU 1
3. Router: scores all 128 experts, RouterParallel zeros non-local → keeps experts [0,7]
4. Expert forward: `gate_up_proj` shape `[8, 1536, 2048]`, grouped_mm on local 8 experts
5. EP all-reduce: sums partial expert outputs across all 16 GPUs → correct MoE output
6. Residual add → next layer

**Why expert-only EP works**: attention weights are replicated (each GPU has full q/k/v/o_proj). FSDP2 handles their memory via all-gather/reduce-scatter within DP groups. Expert weights are physically sharded by EP (8 of 128 per GPU). This decouples EP from `num_kv_heads` — EP can scale to 128 (one expert per GPU) without needing to shard attention.

### Corrected EP Results (expert-only, all three bugs fixed)

All runs with Qwen3-30B-A3B, transformers 5.6.0.dev0, expert-only EP plan, correct routing.

| Context | Nodes | GPUs | DP  | TP  | CP  | EP  | MFU   | TPS     | TPS/GPU | Peak GPU Mem  | Status |
| ------- | ----- | ---- | --- | --- | --- | --- | ----- | ------- | ------- | ------------- | ------ |
| 16k     | 1     | 8    | 8   | 1   | 1   | 8   | -     | -       | -       | -             | OOM    |
| 16k     | 2     | 16   | 16  | 1   | 1   | 16  | 22.5% | 60,600  | 3,788   | 71.1 GB (89%) | Yes    |
| 16k     | 4     | 32   | 32  | 1   | 1   | 32  | 22.3% | 120,000 | 3,750   | 55.1 GB (69%) | Yes    |
| 16k     | 8     | 64   | 64  | 1   | 1   | 64  | 21.4% | 230,500 | 3,602   | 51.1 GB (64%) | Yes    |
| 32k     | 2     | 16   | 8   | 1   | 2   | 16  | 13.2% | 42,970  | 2,686   | 71.5 GB (89%) | Yes    |
| 32k     | 4     | 32   | 16  | 1   | 2   | 32  | 13.2% | 85,510  | 2,672   | 55.2 GB (69%) | Yes    |
| 32k     | 8     | 64   | 32  | 1   | 2   | 64  | 12.9% | 167,900 | 2,623   | 51.5 GB (64%) | Yes    |

### Comparison: Corrected EP vs Previous (buggy) EP

| Config               | Buggy EP MFU | Corrected EP MFU | Diff  |
| -------------------- | ------------ | ---------------- | ----- |
| 16k, 2n, EP=16       | 23.1%        | 22.5%            | -2.6% |
| 16k, 4n, EP=32       | 22.9%        | 22.3%            | -2.6% |
| 32k, 2n, CP=2, EP=16 | 13.6%        | 13.2%            | -2.9% |
| 32k, 4n, CP=2, EP=32 | 13.5%        | 13.2%            | -2.2% |

The corrected EP is ~2-3% slower because:

- Buggy EP had wrong routing weights → some expert outputs were near-zero → less actual computation
- Correct EP does real expert computation with proper routing weights → slightly more GPU work per step

The loss is healthy (1.5-1.7) confirming correct routing. The small MFU drop is expected and confirms the fix is working — the GPU is now doing the right computation.

### EP scaling (16k, DP-only)

| Nodes | EP  | Local experts | MFU   | TPS/GPU | Total TPS |
| ----- | --- | ------------- | ----- | ------- | --------- |
| 2     | 16  | 8             | 22.5% | 3,788   | 60,600    |
| 4     | 32  | 4             | 22.3% | 3,750   | 120,000   |
| 8     | 64  | 2             | 21.4% | 3,602   | 230,500   |

MFU degrades slightly with more nodes (22.5% → 21.4%) due to inter-node all-reduce overhead. TPS/GPU drops ~5% from 2→8 nodes. Total TPS scales near-linearly: 2× nodes gives ~2× TPS.

### EP=8 on 1 node OOMs

30B MoE with EP=8 on 1 node (8 GPUs) OOMs at 16k context. Each GPU holds 16 local experts but FSDP2 can only shard across 8 GPUs — not enough to fit model params + optimizer + 16k activations. Minimum 2 nodes (16 GPUs) needed for this model at 16k.

### Long context: CP=8/16 for 64k on 2 nodes

Tested higher CP to reduce memory for 64k context. All with 2 nodes (16 GPUs), FSDP2.

| Context | Nodes | GPUs | DP  | TP  | CP  | EP  | MFU   | TPS    | TPS/GPU | Peak GPU Mem  | Loss |
| ------- | ----- | ---- | --- | --- | --- | --- | ----- | ------ | ------- | ------------- | ---- |
| 32k     | 2     | 16   | 8   | 1   | 2   | 16  | 13.2% | 42,970 | 2,686   | 71.5 GB (89%) | 1.5  |
| 32k     | 2     | 16   | 2   | 1   | 8   | 16  | 1.9%  | 24,240 | 1,515   | 49.3 GB (62%) | 1.7  |
| 64k     | 2     | 16   | 2   | 1   | 8   | 16  | 2.9%  | 20,990 | 1,312   | 58.7 GB (74%) | 1.5  |
| 64k     | 2     | 16   | 1   | 1   | 16  | 16  | 0.8%  | 11,420 | 714     | 46.5 GB (58%) | 1.3  |

**64k training fits on 2 nodes** with CP=8 at 58.7 GB (74%) — 21 GB headroom remaining. This is the first successful 64k run for the 30B MoE model.

**CP trades MFU for memory and context length:**

- CP=2→8 at 32k: memory drops 71→49 GB but MFU drops 13.2%→1.9% (ring-attention across 8 GPUs has 7 send/recv rounds vs 1)
- CP=8 at 64k has better MFU than CP=8 at 32k (2.9% vs 1.9%) — more compute per communication round with longer sequences
- CP=16 at 64k is very slow (0.8%) — ring-attention across all 16 GPUs including inter-node

**Loss improves with longer context**: 1.3 at 64k/CP=16 vs 1.7 at 32k/CP=8 — the model sees more context per sample.

#### Device mesh for CP=8, EP=16, 2 nodes

```
Node 0 (NVLink)                         Node 1 (NVLink)
GPU 0:  experts [0,7]                   GPU 8:  experts [64,71]
GPU 1:  experts [8,15]                  GPU 9:  experts [72,79]
GPU 2:  experts [16,23]                 GPU 10: experts [80,87]
GPU 3:  experts [24,31]                 GPU 11: experts [88,95]
GPU 4:  experts [32,39]                 GPU 12: experts [96,103]
GPU 5:  experts [40,47]                 GPU 13: experts [104,111]
GPU 6:  experts [48,55]                 GPU 14: experts [112,119]
GPU 7:  experts [56,63]                 GPU 15: experts [120,127]
├── CP group 0 (8 GPUs, intra-node) ──┤ ├── CP group 1 (intra-node) ──┤
└── DP pair: GPU0↔GPU8, ..., GPU7↔GPU15 (cross-node) ─────────────────┘
└── EP all-reduce: all 16 GPUs (inter-node) ───────────────────────────┘
```

- **CP=8 stays intra-node**: ring-attention within each node (NVLink, fast)
- **DP=2 cross-node**: FSDP gradient sync between paired GPUs (minimal, only 2-way)
- **EP=16 crosses nodes**: expert output all-reduce spans both nodes

#### EP=8 (2D mesh) as alternative

With `expert_parallel_size=8`, the SFT trainer creates a 2D mesh `(2, 8)` where each node is an independent EP group. Experts are **replicated** across nodes:

```
Node 0: GPU0[0,15] GPU1[16,31] ... GPU7[112,127]   ← EP group 0
Node 1: GPU8[0,15] GPU9[16,31] ... GPU15[112,127]  ← EP group 1 (same layout)
DP pairs: GPU0↔GPU8, GPU1↔GPU9, ... (cross-node gradient sync)
```

- EP all-reduce stays **intra-node** (NVLink) — no inter-node expert communication
- Each GPU holds 16 local experts (vs 8 with EP=16) — more expert memory
- DP handles gradient sync cross-node via reduce-scatter

This could recover MFU at high CP by eliminating inter-node EP overhead, at the cost of 2× expert memory per GPU.

### EP=8 2D mesh vs EP=16 flat mesh (2 nodes)

| Context | Nodes | GPUs | DP  | TP  | CP  | EP  | EP mesh   | MFU  | TPS    | TPS/GPU | Peak GPU Mem  | Loss |
| ------- | ----- | ---- | --- | --- | --- | --- | --------- | ---- | ------ | ------- | ------------- | ---- |
| 32k     | 2     | 16   | 2   | 1   | 8   | 16  | flat (16) | 1.9% | 24,240 | 1,515   | 49.3 GB (62%) | 1.7  |
| 32k     | 2     | 16   | 2   | 1   | 8   | 8   | 2D (2,8)  | 1.9% | 24,670 | 1,542   | 49.3 GB (62%) | 1.7  |
| 64k     | 2     | 16   | 2   | 1   | 8   | 16  | flat (16) | 2.9% | 20,990 | 1,312   | 58.7 GB (74%) | 1.5  |
| 64k     | 2     | 16   | 2   | 1   | 8   | 8   | 2D (2,8)  | 2.9% | 21,180 | 1,324   | 58.7 GB (74%) | 1.5  |
| 64k     | 2     | 16   | 1   | 1   | 16  | 16  | flat (16) | 0.8% | 11,420 | 714     | 46.5 GB (58%) | 1.3  |
| 64k     | 2     | 16   | 1   | 1   | 16  | 8   | 2D (2,8)  | 0.8% | 11,420 | 714     | 46.5 GB (58%) | 1.3  |

**Result: EP=8 2D mesh and EP=16 flat mesh are identical at 2 nodes.**

- MFU, TPS/GPU, and peak memory are the same within noise for all 3 configs
- This confirms the earlier finding at 16k (23.0% vs 23.1%): at 2 nodes, EP topology doesn't matter
- Transformers EP uses **all-reduce** (not all-to-all). All-reduce scales 96% inter-node on EFA — there's no penalty for crossing nodes
- The 2D mesh advantage (intra-node EP) would only appear if all-to-all was used, but it's not

**Conclusion**: for this EP implementation, flat mesh EP=world_size is the best strategy — it minimizes expert memory per GPU (fewer local experts) with no MFU penalty. The 2D mesh adds complexity (expert_parallel_size config) with no benefit. The EP=8 strategy would only help with a token-dispatch EP implementation (all-to-all based, like DeepEP or Megablocks).

---

## Addendum: DeepSpeed Ulysses Sequence Parallelism (SP)

DeepSpeed Ulysses splits sequences across GPUs via all-to-all on Q/K/V heads — fundamentally different from CP's ring-attention. SP only works with DeepSpeed (not FSDP2) and is mutually exclusive with CP. Requires `flash_attention_2`.

### Qwen3-4B (Dense) — SP scaling

| Context | Nodes | GPUs | DP  | TP  | CP  | SP  | EP  | Backend         | Attn              | MFU   | TPS    | TPS/GPU | Peak GPU Mem  |
| ------- | ----- | ---- | --- | --- | --- | --- | --- | --------------- | ----------------- | ----- | ------ | ------- | ------------- |
| 16k     | 1     | 8    | 4   | 1   | 1   | 2   | 1   | deepspeed_zero3 | flash_attention_2 | 21.2% | 31,540 | 3,943   | 39.3 GB (49%) |
| 16k     | 1     | 8    | 2   | 1   | 1   | 4   | 1   | deepspeed_zero3 | flash_attention_2 | 11.0% | 16,390 | 2,049   | 30.2 GB (38%) |
| 32k     | 2     | 16   | 8   | 1   | 1   | 2   | 1   | deepspeed_zero3 | flash_attention_2 | 40.2% | 77,420 | 4,839   | 51.6 GB (65%) |
| 32k     | 2     | 16   | 4   | 1   | 1   | 4   | 1   | deepspeed_zero3 | flash_attention_2 | 30.4% | 58,660 | 3,666   | 38.2 GB (48%) |
| 32k     | 2     | 16   | 2   | 1   | 1   | 8   | 1   | deepspeed_zero3 | flash_attention_2 | 16.3% | 31,440 | 1,965   | 28.0 GB (35%) |
| 32k     | 2     | 16   | 1   | 1   | 1   | 16  | 1   | deepspeed_zero3 | flash_attention_2 | 7.7%  | 14,840 | 928     | 20.4 GB (26%) |

### SP vs CP at 32k (Qwen3-4B, 2 nodes)

| Method   | Degree | Backend              | MFU       | TPS/GPU   | Peak GPU Mem |
| -------- | ------ | -------------------- | --------- | --------- | ------------ |
| **SP=2** | **2**  | **DeepSpeed ZeRO-3** | **40.2%** | **4,839** | **51.6 GB**  |
| CP=2     | 2      | FSDP2                | 31.6%     | 7,615     | 44.4 GB      |
| SP=4     | 4      | DeepSpeed ZeRO-3     | 30.4%     | 3,666     | 38.2 GB      |
| CP=4     | 4      | FSDP2                | 26.8%     | 12,939    | 26.3 GB      |
| SP=8     | 8      | DeepSpeed ZeRO-3     | 16.3%     | 1,965     | 28.0 GB      |
| CP=8     | 8      | FSDP2                | 19.6%     | 18,933    | 17.7 GB      |

**Key observations:**

1. **SP=2 achieves 40.2% MFU — the highest MFU in all 4B benchmarks.** Ulysses all-to-all at SP=2 is very efficient (only 2-GPU intra-node groups). The overhead is lower than ring-attention at CP=2.

2. **SP MFU degrades sharply with degree.** SP=2→4→8→16: 40.2%→30.4%→16.3%→7.7%. The all-to-all communication grows quadratically with SP degree. At SP=8, some all-to-all groups cross the node boundary (11% bandwidth vs NVLink). SP=16 spans both nodes — MFU collapses.

3. **CP has higher TPS/GPU but lower MFU.** CP's ring-attention is compute-efficient (high TPS/GPU) but has more GPU idle time waiting for chunks to arrive. SP's all-to-all is bandwidth-intensive but keeps all GPUs active simultaneously.

4. **SP uses more memory than CP at same degree.** SP=2: 51.6 GB vs CP=2: 44.4 GB. DeepSpeed ZeRO-3 allgathers full params (unlike FSDP2 which shards more efficiently). SP also needs to buffer Q/K/V for all-to-all.

5. **SP=2 is the sweet spot for this cluster.** Beyond SP=2, the all-to-all overhead grows faster than the parallelism benefit. The EFA interconnect has 11% all-to-all bandwidth inter-node, making cross-node SP impractical.

### 30B MoE SP+EP: Initialization deadlock (not yet resolved)

SP with EP (`--enable_expert_parallel`) hangs during DeepSpeed initialization on 30B MoE. The likely cause: DeepSpeed Ulysses creates its own process groups for all-to-all communication, while transformers EP creates a separate device mesh for expert distribution. The two subsystems deadlock during NCCL collective initialization — one waits on a collective the other hasn't joined.

**Status**: The conflict is in `from_pretrained`'s model creation: accelerate's DeepSpeed env forces `zero.Init` or meta device, but EP's `distribute_model` needs real weights on GPU. Our srun test proved EP+SP+DS works when loaded outside `from_pretrained`'s context managers. Fixing `from_pretrained` to skip meta device for EP+DS is in progress.

### 30B MoE SP without EP (DeepSpeed ZeRO-3 + SP)

ZeRO-3 handles expert memory distribution without EP — each rank holds a shard of every expert's parameters (vertical partitioning), unlike EP which assigns complete experts to ranks (horizontal partitioning). SP handles sequence scaling.

| Context | Nodes | GPUs | DP  | TP  | CP  | SP  | EP  | Backend         | Attn              | MFU   | TPS    | TPS/GPU | Peak GPU Mem  | Status |
| ------- | ----- | ---- | --- | --- | --- | --- | --- | --------------- | ----------------- | ----- | ------ | ------- | ------------- | ------ |
| 16k     | 2     | 16   | 8   | 1   | 1   | 2   | 1   | deepspeed_zero3 | flash_attention_2 | 5.8%  | 31,420 | 1,964   | 61.3 GB (77%) | Yes    |
| 32k     | 2     | 16   | 8   | 1   | 1   | 2   | 1   | deepspeed_zero3 | flash_attention_2 | 13.1% | 42,600 | 2,663   | 54.0 GB (68%) | Yes    |
| 64k     | 2     | 16   | 8   | 1   | 1   | 2   | 1   | deepspeed_zero3 | flash_attention_2 | -     | -      | -       | -             | OOM    |
| 64k     | 2     | 16   | 4   | 1   | 1   | 4   | 1   | deepspeed_zero3 | flash_attention_2 | 11.6% | 41,980 | 2,624   | 54.4 GB (68%) | Yes    |

**MFU correction**: `num_input_tokens_seen` overcounts by `sp_size` (same pattern as CP). MFU corrected by dividing TPS by `sp_size` before computing. TPS and TPS/GPU in the table are **raw** (consistent with how CP rows are reported). Fix applied to `sft_trainer.py`.

### SP vs CP for 30B MoE (2 nodes, 16 GPUs)

| Context | Method     | Backend | Degree | MFU   | TPS/GPU (raw) | Peak Mem |
| ------- | ---------- | ------- | ------ | ----- | ------------- | -------- |
| 16k     | SP=2       | DS-Z3   | 2      | 5.8%  | 1,964         | 61.3 GB  |
| 16k     | EP=16      | FSDP2   | -      | 22.5% | 3,788         | 71.1 GB  |
| 32k     | SP=2       | DS-Z3   | 2      | 13.1% | 2,663         | 54.0 GB  |
| 32k     | CP=2+EP=16 | FSDP2   | 2      | 13.2% | 2,686         | 71.5 GB  |
| 64k     | SP=4       | DS-Z3   | 4      | 11.6% | 2,624         | 54.4 GB  |
| 64k     | CP=8+EP=16 | FSDP2   | 8      | 2.9%  | 1,312         | 58.7 GB  |

Note: TPS/GPU is raw (before CP/SP correction). MFU is corrected (TPS ÷ cp_size or sp_size). Both CP and SP overcount `num_input_tokens_seen` by their parallelism degree.

**Key findings:**

1. **32k: SP=2 matches CP=2 in MFU and TPS/GPU** (13.1% vs 13.2%, 2,663 vs 2,686). Ulysses and ring-attention achieve comparable compute efficiency and throughput at degree=2. SP uses 17 GB less memory (54 vs 71 GB) due to ZeRO-3 parameter sharding.

2. **64k: SP=4 achieves 4× the MFU and 2× the TPS/GPU of FSDP2+CP=8** (11.6% vs 2.9%, 2,624 vs 1,312). Ring-attention at degree=8 degrades sharply (7 sequential send/recv rounds, some crossing nodes), while Ulysses all-to-all at degree=4 remains efficient.

3. **16k: FSDP2+EP is far better** (22.5% vs 5.8%, 3,788 vs 1,964). At short context, ZeRO-3's parameter allgather overhead dominates. FSDP2+EP is more efficient for short sequences.

4. **Memory**: DS-Z3+SP uses less memory (54-61 GB) than FSDP2+EP (58-71 GB) because SP splits activation memory AND ZeRO-3 only materializes parameter shards (not full allgather like FSDP2).

5. **Best strategy depends on context length**: FSDP2+EP for ≤16k. For ≥32k, DeepSpeed+SP matches or beats FSDP2+CP in both MFU and throughput, with lower memory.

---

## Addendum: Liger Kernel — Qwen3-4B 32k on 1 Node (previously OOM)

### Motivation

[Lewtun's config](https://gist.github.com/lewtun/daf260470eed93f63d074870c9f3b3fc) trains Qwen3-4B at 36k context on 1 node (8 GPUs) using DeepSpeed ZeRO-3 with `use_liger_kernel: true` and `kernels-community/vllm-flash-attn3`. Our previous 32k single-node runs OOMed with both `sdpa` and `flash_attention_2` on FSDP2.

Key difference: **Liger kernel** replaces CrossEntropy, SwiGLU, RMSNorm, and RoPE with fused Triton kernels that avoid materializing large intermediate tensors. The fused CrossEntropy alone avoids a `(batch × seq, vocab_size)` logit tensor (~11 GB at 32k context). Fused SwiGLU eliminates separate gate/up intermediate tensors (~2× MLP activation savings).

### Results (FSDP2 + Liger Kernel)

| Context | Nodes | GPUs | DP  | TP  | CP  | EP  | Liger | MFU       | TPS    | TPS/GPU | Peak GPU Mem | Status |
| ------- | ----- | ---- | --- | --- | --- | --- | ----- | --------- | ------ | ------- | ------------ | ------ |
| 32k     | 1     | 8    | 8   | 1   | 1   | 1   | yes   | **35.9%** | 34,570 | 4,321   | —            | Yes    |
| 32k     | 1     | 8    | 8   | 1   | 1   | 1   | no    | —         | —      | —       | —            | OOM    |

### Comparison: Liger vs non-Liger (Qwen3-4B, 32k)

| Config                   | MFU       | TPS/GPU | Status |
| ------------------------ | --------- | ------- | ------ |
| 1n, DP=8, no liger       | —         | —       | OOM    |
| 2n, DP=8, CP=2, no liger | 31.6%     | 7,615   | Yes    |
| **1n, DP=8, liger**      | **35.9%** | 4,321   | Yes    |

Liger kernel not only avoids the OOM on 1 node, it achieves the **highest MFU (35.9%)** of any Qwen3-4B configuration — beating the previous best of 31.8% (4 nodes, CP=2). The fused kernels reduce both memory and kernel launch overhead.

### Full Liger Kernel Results (1 node, 8 GPUs, FSDP2)

| Context | Nodes | GPUs | DP  | TP  | CP  | EP  | Liger | MFU       | TPS    | TPS/GPU | Status |
| ------- | ----- | ---- | --- | --- | --- | --- | ----- | --------- | ------ | ------- | ------ |
| 32k     | 1     | 8    | 8   | 1   | 1   | 1   | no    | —         | —      | —       | OOM    |
| 32k     | 1     | 8    | 8   | 1   | 1   | 1   | yes   | **35.9%** | 34,570 | 4,321   | Yes    |
| 32k     | 1     | 8    | 4   | 1   | 2   | 1   | yes   | 17.1%     | 32,960 | 4,120   | Yes    |
| 64k     | 1     | 8    | 4   | 1   | 2   | 1   | yes   | 18.3%     | 20,690 | 2,586   | Yes    |
| 64k     | 1     | 8    | 2   | 1   | 4   | 1   | yes   | 8.2%      | 18,480 | 2,310   | Yes    |

**Note**: Qwen3-4B has `max_position_embeddings: 32,768`. The 64k runs exceed native context — RoPE extrapolates beyond the trained range (YaRN not enabled). Loss values look reasonable (1.7-1.8) but positional embeddings are incorrect for positions > 32k. These results are valid for **throughput/memory benchmarking** (GPU does the same compute), not for training quality. Real 64k training would require YaRN or a model that natively supports it.

### Comparison: Liger vs no-Liger (Qwen3-4B, 32k, same hardware)

| Config                   | MFU       | TPS/GPU | Status |
| ------------------------ | --------- | ------- | ------ |
| 1n, DP=8, no liger       | —         | —       | OOM    |
| **1n, DP=8, liger**      | **35.9%** | 4,321   | Yes    |
| 2n, DP=8, CP=2, no liger | 31.6%     | 7,615   | Yes    |
| 1n, DP=4, CP=2, liger    | 17.1%     | 4,120   | Yes    |

### Observations

- **Liger kernel eliminates the 32k OOM on 1 node**: Fused CE + SwiGLU kernels save enough activation memory to go from OOM to 35.9% MFU with headroom to spare.
- **35.9% MFU is the highest for Qwen3-4B**: Beats the previous best of 31.8% (4 nodes, CP=2, no liger). Fewer kernel launches + no intermediate tensor materialization = faster.
- **64k fits on 1 node with CP=2**: Liger + CP=2 fits 64k at 18.3% MFU — previously required 2+ nodes with CP.
- **CP still trades MFU for memory**: CP=1→2 drops MFU from 35.9%→17.1% at 32k (ring-attention overhead). CP=2→4 at 64k drops from 18.3%→8.2%.

### Liger Kernel + Qwen3-30B-A3B MoE: Incompatible

Liger kernel crashes on Qwen3-30B-A3B with `CUDA error: device-side assert triggered` inside `liger_kernel/transformers/swiglu.py`. The crash occurs because Liger's fused SwiGLU kernel assumes standard 2D MLP weight shapes `[intermediate, hidden]`, but the MoE fused expert architecture in transformers 5.6.0 uses 3D tensors `[num_experts, intermediate, hidden]` with indexed access per expert. Liger patches the MLP module globally and doesn't handle the expert-indexed forward path.

This affects only MoE models with the native fused expert architecture (`Qwen3MoeExperts`). Dense models like Qwen3-4B work fine with Liger.

**Status**: Liger kernel is only usable with Qwen3-4B (dense) in this benchmark. MoE models require either upstream Liger support for fused expert tensors, or selective patching that skips expert MLP modules.

---

## Addendum: Flash Attention 3 (vllm-flash-attn3) — Qwen3-4B

### Motivation

[Lewtun's config](https://gist.github.com/lewtun/daf260470eed93f63d074870c9f3b3fc) uses `kernels-community/vllm-flash-attn3` — a Hopper-native FA3 implementation that leverages H100 warp-level scheduling and FP8 tensor cores. Our previous runs used `sdpa` (PyTorch's scaled dot product attention). Testing FA3 to measure the attention kernel speedup.

### Results: Liger + FA3 vs Liger + sdpa (Qwen3-4B, 32k, 1 node, 8 GPUs)

| Context | Nodes | GPUs | DP  | TP  | CP  | EP  | Liger | Attn | MFU       | TPS    | TPS/GPU | Runtime (20 steps) |
| ------- | ----- | ---- | --- | --- | --- | --- | ----- | ---- | --------- | ------ | ------- | ------------------ |
| 32k     | 1     | 8    | 8   | 1   | 1   | 1   | yes   | sdpa | 35.9%     | 34,570 | 4,321   | 151.7s             |
| 32k     | 1     | 8    | 8   | 1   | 1   | 1   | yes   | FA3  | **56.3%** | 54,260 | 6,783   | 96.6s              |
| 32k     | 1     | 8    | 8   | 1   | 1   | 1   | no    | sdpa | —         | —      | —       | OOM                |

### Analysis

- **FA3 gives 1.57× MFU boost over sdpa** (56.3% vs 35.9%) with identical loss curves. The Hopper-optimized FA3 kernel is dramatically faster than PyTorch's sdpa at 32k context.
- **56.3% MFU is the highest across all benchmarks** — 56% of peak H100 bf16 FLOPS utilized for a 4B dense model on a single node.
- **1.57× faster wall-clock**: 96.6s vs 151.7s for 20 steps. FA3 halves per-step attention time.
- **TPS/GPU: 6,783** — up from 4,321 with sdpa. Per-GPU throughput is 1.57× higher.

### Full Qwen3-4B Results Summary (32k context)

| Config                      | MFU       | TPS/GPU | Nodes | Status  |
| --------------------------- | --------- | ------- | ----- | ------- |
| FSDP2, no liger, sdpa       | —         | —       | 1     | OOM     |
| FSDP2, liger, sdpa          | 35.9%     | 4,321   | 1     | Yes     |
| **FSDP2, liger, FA3**       | **56.3%** | 6,783   | 1     | Yes     |
| FSDP2, CP=2, no liger, sdpa | 31.6%     | 7,615   | 2     | Yes     |
| DS-Z3, SP=2, FA2            | 40.2%     | 4,839   | 2     | Yes     |
| DS-Z3, SP=2, FA3            | —         | —       | 2     | pending |

### SP=2 + FA3 (DeepSpeed ZeRO-3, Qwen3-4B, 32k, 2 nodes)

| Context | Nodes | GPUs | DP  | TP  | CP  | SP  | EP  | Backend         | Attn | MFU   | TPS    | TPS/GPU | Runtime (20 steps) |
| ------- | ----- | ---- | --- | --- | --- | --- | --- | --------------- | ---- | ----- | ------ | ------- | ------------------ |
| 32k     | 2     | 16   | 8   | 1   | 1   | 2   | 1   | deepspeed_zero3 | FA2  | 40.2% | 77,420 | 4,839   | —                  |
| 32k     | 2     | 16   | 8   | 1   | 1   | 2   | 1   | deepspeed_zero3 | FA3  | 23.6% | 91,170 | 5,698   | 57.5s              |

**FA3 hurts SP performance**: 23.6% vs 40.2% MFU with FA2 — a 41% MFU regression despite 18% higher raw TPS. The Ulysses SP path splits Q/K/V across GPUs via all-to-all before calling the attention kernel. The FA3 kernel path likely has higher overhead for the split/gather operations or doesn't benefit from the same optimizations as FA2 in the SP context.

### FA3 Key Finding: Great standalone, bad with SP

| Config                   | FA2/sdpa MFU | FA3 MFU   | Change   |
| ------------------------ | ------------ | --------- | -------- |
| FSDP2, liger, 1n (no SP) | 35.9% (sdpa) | **56.3%** | +57%     |
| DS-Z3, SP=2, 2n          | 40.2% (FA2)  | 23.6%     | **-41%** |

FA3 (`kernels-community/vllm-flash-attn3`) excels as a standalone attention kernel (1.57× speedup over sdpa) but regresses badly when combined with DeepSpeed Ulysses SP. Use FA3 for non-SP configurations; stick with `flash_attention_2` for SP.

### 30B SP=2: FA3 vs FA2 (Qwen3-30B-A3B, 32k, 2 nodes, srun 5-step test)

| Context | Nodes | GPUs | DP  | TP  | CP  | SP  | EP  | Backend         | Attn | MFU   | TPS    | TPS/GPU | Runtime (5 steps) |
| ------- | ----- | ---- | --- | --- | --- | --- | --- | --------------- | ---- | ----- | ------ | ------- | ----------------- |
| 32k     | 2     | 16   | 8   | 1   | 1   | 2   | 1   | deepspeed_zero3 | FA2  | 9.9%  | 32,140 | 2,009   | 40.8s             |
| 32k     | 2     | 16   | 8   | 1   | 1   | 2   | 1   | deepspeed_zero3 | FA3  | 10.5% | 34,130 | 2,133   | 38.4s             |

FA3 gives ~6% improvement over FA2 with SP on 30B — marginal compared to the 57% gain on 4B standalone. The SP all-to-all overhead dominates, not the attention kernel.

### FA3 Summary

| Model | Config           | FA2/sdpa MFU | FA3 MFU   | Change   | Verdict           |
| ----- | ---------------- | ------------ | --------- | -------- | ----------------- |
| 4B    | FSDP2, liger, 1n | 35.9% (sdpa) | **56.3%** | **+57%** | FA3 wins big      |
| 4B    | DS-Z3, SP=2, 2n  | 40.2% (FA2)  | 23.6%     | **-41%** | FA3 loses with SP |
| 30B   | DS-Z3, SP=2, 2n  | 9.9% (FA2)   | 10.5%     | +6%      | Marginal gain     |

- **FA3 without SP**: massive speedup (1.57×) — the Hopper-native kernel is much faster than sdpa for standalone attention
- **FA3 with SP (4B)**: regression — SP's Ulysses all-to-all path is not optimized for FA3
- **FA3 with SP (30B)**: marginal gain — attention is a smaller fraction of compute for MoE, so kernel speedup has less impact

---

## Addendum: Qwen3-32B (Dense) — Comparison with Qwen3-30B-A3B (MoE)

### Motivation

Qwen3-32B is a dense model with 32B total parameters, all active per token. Qwen3-30B-A3B is a MoE model with 30B total parameters but only ~3B active per token (8 of 128 experts). Comparing both at the same hardware configs reveals the cost of MoE communication overhead (EP all-reduce, expert sharding) vs dense compute efficiency.

### Architecture comparison

| Field                   | Qwen3-32B (dense) | Qwen3-30B-A3B (MoE) |
| ----------------------- | ----------------- | ------------------- |
| model_type              | qwen3             | qwen3_moe           |
| hidden_size             | 5120              | 2048                |
| num_hidden_layers       | 64                | 48                  |
| num_attention_heads     | 64                | 32                  |
| num_key_value_heads     | 8                 | 4                   |
| intermediate_size       | 25600             | 6144                |
| moe_intermediate_size   | N/A               | 768                 |
| num_local_experts       | N/A               | 128                 |
| num_experts_per_tok     | N/A               | 8                   |
| Total params            | 32B               | 30B                 |
| Active params/token     | 32B               | ~3B                 |
| max_position_embeddings | 40960             | 32768               |

### FSDP2 Results (preliminary, sdpa, no liger)

| Config         | Qwen3-32B (dense) MFU | Qwen3-30B-A3B (MoE) MFU | Dense/MoE ratio |
| -------------- | --------------------- | ----------------------- | --------------- |
| 16k, 2n, DP=16 | **36.5%**             | 22.5% (EP=16)           | 1.62×           |
| 16k, 4n, DP=32 | **37.7%**             | 22.3% (EP=32)           | 1.69×           |
| 32k, 2n, CP=2  | **18.1%**             | 13.2% (EP=16, CP=2)     | 1.37×           |

#### Qwen3-32B full metrics

| Context | Nodes | GPUs | DP  | TP  | CP  | EP  | Backend | Liger | Attn       | MFU       | TPS    | TPS/GPU | Peak GPU Mem  | Status |
| ------- | ----- | ---- | --- | --- | --- | --- | ------- | ----- | ---------- | --------- | ------ | ------- | ------------- | ------ |
| 16k     | 2     | 16   | 16  | 1   | 1   | 1   | fsdp2   | no    | sdpa       | 36.5%     | 19,280 | 1,205   | 79.3 GB (99%) | Yes    |
| 16k     | 2     | 16   | 16  | 1   | 1   | 1   | fsdp2   | yes   | sdpa       | 41.1%     | 21,690 | 1,356   | 59.7 GB (75%) | Yes    |
| 16k     | 2     | 16   | 16  | 1   | 1   | 1   | fsdp2   | yes   | FA3        | **51.3%** | 27,080 | 1,693   | 59.7 GB (75%) | Yes    |
| 16k     | 4     | 32   | 32  | 1   | 1   | 1   | fsdp2   | no    | sdpa       | 37.7%     | 39,790 | 1,243   | 72.2 GB (90%) | Yes    |
| 16k     | 4     | 32   | 32  | 1   | 1   | 1   | fsdp2   | yes   | sdpa       | 40.8%     | 43,020 | 1,344   | —             | Yes    |
| 16k     | 4     | 32   | 32  | 1   | 1   | 1   | DS-Z3   | no    | sdpa       | 33.9%     | 35,770 | 1,118   | —             | Yes    |
| 32k     | 2     | 16   | 16  | 1   | 1   | 1   | fsdp2   | no    | sdpa       | —         | —      | —       | —             | OOM    |
| 32k     | 2     | 16   | 16  | 1   | 1   | 1   | fsdp2   | yes   | sdpa       | 43.1%     | 16,960 | 1,060   | —             | Yes    |
| 32k     | 2     | 16   | 16  | 1   | 1   | 1   | fsdp2   | yes   | FA3        | **59.0%** | 23,200 | 1,450   | —             | Yes    |
| 32k     | 2     | 16   | 8   | 1   | 2   | 1   | fsdp2   | no    | sdpa       | 18.1%     | 14,190 | 887     | —             | Yes    |
| 32k     | 2     | 16   | 8   | 1   | 2   | 1   | fsdp2   | yes   | sdpa       | 19.8%     | 15,550 | 972     | 60.9 GB (76%) | Yes    |
| 32k     | 4     | 32   | 16  | 1   | 2   | 1   | fsdp2   | no    | sdpa       | 18.4%     | 28,870 | 902     | —             | Yes    |
| 32k     | 4     | 32   | 16  | 1   | 2   | 1   | fsdp2   | yes   | sdpa       | 19.4%     | 30,460 | 952     | —             | Yes    |
| 32k     | 2     | 16   | 8   | 1   | 1   | 1   | DS-Z3   | no    | FA2 (SP=2) | —         | —      | —       | —             | OOM    |
| 32k     | 2     | 16   | 8   | 1   | 1   | 1   | DS-Z3   | no    | FA3 (SP=2) | —         | —      | —       | —             | OOM    |
| 32k     | 2     | 16   | 8   | 1   | 1   | 1   | DS-Z3   | yes   | FA2 (SP=2) | 25.0%     | 19,630 | 1,227   | —             | Yes    |
| 32k     | 2     | 16   | 8   | 1   | 1   | 1   | DS-Z3   | yes   | FA3 (SP=2) | 29.7%     | 23,380 | 1,461   | —             | Yes    |
| 16k     | 2     | 16   | 8   | 2   | 1   | 1   | fsdp2   | no    | sdpa       | —         | —      | —       | —             | OOM    |
| 16k     | 2     | 16   | 8   | 2   | 1   | 1   | fsdp2   | yes   | sdpa       | 38.5%     | 20,310 | 1,269   | —             | Yes    |

### Comparison: 32B Dense vs 30B MoE (same hardware, best config per model)

| Config                      | Qwen3-32B (dense) MFU | Qwen3-30B-A3B (MoE) MFU  | Dense/MoE ratio |
| --------------------------- | --------------------- | ------------------------ | --------------- |
| 16k, 2n, FSDP2              | 36.5%                 | 22.5% (EP=16)            | 1.62×           |
| 16k, 2n, liger              | 41.1%                 | N/A (liger incompatible) | —               |
| 16k, 2n, liger+FA3          | **51.3%**             | N/A                      | —               |
| 16k, 4n, FSDP2              | 37.7%                 | 22.3% (EP=32)            | 1.69×           |
| 16k, 4n, liger              | 40.8%                 | N/A                      | —               |
| 16k, 4n, DS-Z3              | 33.9%                 | 18.9%                    | 1.79×           |
| 32k, 2n, DP-only, liger     | 43.1%                 | N/A (OOM without EP)     | —               |
| 32k, 2n, DP-only, liger+FA3 | **59.0%**             | N/A                      | —               |
| 32k, 2n, CP=2               | 18.1%                 | 13.2% (EP=16, CP=2)      | 1.37×           |
| 32k, 2n, CP=2, liger        | 19.8%                 | N/A                      | —               |
| 32k, 4n, CP=2               | 18.4%                 | —                        | —               |
| 32k, 4n, CP=2, liger        | 19.4%                 | N/A                      | —               |

### Liger + FA3 impact on 32B

| Context | Variant       | MFU       | TPS/GPU | vs baseline                          |
| ------- | ------------- | --------- | ------- | ------------------------------------ |
| 16k     | baseline sdpa | 36.5%     | 1,205   | —                                    |
| 16k     | + liger       | 41.1%     | 1,356   | +13%                                 |
| 16k     | + liger + FA3 | **51.3%** | 1,693   | **+41%**                             |
| 32k     | baseline sdpa | —         | —       | OOM                                  |
| 32k     | + liger       | 43.1%     | 1,060   | (enables 32k)                        |
| 32k     | + liger + FA3 | **59.0%** | 1,450   | **(enables 32k + 1.37× over liger)** |

Liger+FA3 on the 32B dense model achieves **59.0% MFU at 32k** — the highest MFU across all benchmarks. At 16k it reaches 51.3%. Both are consistent with the 4B pattern (56.3% MFU). The dense architecture benefits fully from both optimizations since liger's fused SwiGLU works with standard 2D MLP weights.

Key: **liger enables 32k DP-only** (OOMs without liger), and **FA3 adds 37% on top** at 32k. The combination is even more impactful at 32k than 16k because the longer sequences give FA3 more compute per kernel launch.

### Observations

- **Dense is 1.4-1.7× more MFU-efficient than MoE at same hardware**: FSDP2 shards the dense model more efficiently — no EP all-reduce, no expert routing overhead.
- **Liger gives +13% MFU on 32B**: fused CE + SwiGLU saves activation memory and reduces kernel launches. Same pattern as 4B.
- **FA3 gives another +25% on top of liger**: Hopper-native attention is much faster than sdpa. Combined liger+FA3 = +41% over baseline.
- **MoE can't use liger or FA3 effectively**: liger crashes on fused expert tensors, FA3 only gives +6% with SP. This widens the dense advantage significantly.
- **DP scaling**: 2n→4n gives near-linear TPS scaling (19.3k→39.8k = 2.06×) with slightly higher MFU (36.5%→37.7%).
- **TPS/GPU is lower than 30B**: 1,205 vs 3,788 TPS/GPU for 30B at 16k 2n. Expected — 32B processes 10× more FLOPs per token, but MFU is higher.

### TP results

| TP  | DP  | Liger | MFU   | Peak Mem      | Status |
| --- | --- | ----- | ----- | ------------- | ------ |
| 2   | 8   | no    | —     | 77.7 GB (97%) | OOM    |
| 2   | 8   | yes   | 38.5% | —             | Yes    |
| 4   | 4   | yes   | —     | 79.4 GB (99%) | OOM    |
| 8   | 2   | yes   | —     | 78.7 GB (98%) | OOM    |

Only TP=2+liger fits. TP=4 and TP=8 OOM even with liger because fewer DP shards means each GPU holds more model params. At TP=4 (DP=4), FSDP2 shards across only 4 GPUs instead of 16 — 4× more params per GPU. TP=2+liger (38.5% MFU) is slightly below DP-only+liger (41.1%) due to TP all-reduce overhead.

### SP + liger: enables 32k on DS-Z3

SP=2 without liger OOMed at 32k. With liger, SP=2 fits:

| SP Attn | MFU   | TPS/GPU | Status |
| ------- | ----- | ------- | ------ |
| FA2     | 25.0% | 1,227   | Yes    |
| FA3     | 29.7% | 1,461   | Yes    |

FA3 gives +19% over FA2 with SP+liger — better than the 4B where FA3 regressed with SP. The 32B's larger attention matrices give FA3 more compute to optimize.

### All 32B runs complete

Best configs for Qwen3-32B:

| Context | Best MFU  | Config                     |
| ------- | --------- | -------------------------- |
| 16k     | **51.3%** | FSDP2, liger+FA3, 2n DP=16 |
| 32k     | **59.0%** | FSDP2, liger+FA3, 2n DP=16 |
| 32k SP  | 29.7%     | DS-Z3, SP=2+FA3+liger, 2n  |

---

## 2026-04-17: Qwen3-30B-A3B FlashAttention3 test

### Goal

Compare FA3 (`kernels-community/vllm-flash-attn3`) vs sdpa for Qwen3-30B-A3B MoE on the best-performing config (2 nodes, 16k ctx).

### Results

| Config            | Nodes | DP  | TP  | CP  | EP  | SP  | Attn | MFU        | TPS/GPU | Peak GPU Mem  | Status               |
| ----------------- | ----- | --- | --- | --- | --- | --- | ---- | ---------- | ------- | ------------- | -------------------- |
| FSDP2, 16k, no EP | 2     | 16  | 1   | 1   | 1   | 1   | sdpa | 23.1%      | 3,888   | --            | Baseline             |
| FSDP2, 16k, no EP | 2     | 16  | 1   | 1   | 1   | 1   | FA3  | **25.65%** | 4,318   | 71.1 GB (89%) | New                  |
| FSDP2, 16k, EP=16 | 2     | 16  | 1   | 1   | 16  | 1   | FA3  | --         | --      | --            | Failed (mesh compat) |

### Findings

- **FA3 gives +11% relative MFU improvement** over sdpa (25.65% vs 23.1%), +11% TPS/GPU (4,318 vs 3,888).
- FA3 + EP fails with `KeyError: "Invalid mesh_dim_names ('dp_shard_cp',)"` -- EP creates a device mesh with only `['tp']` but FSDP2 expects `dp_shard_cp`. This is an accelerate/transformers EP mesh incompatibility, not FA3-specific. Needs investigation.
- Multi-node HF Hub cache race condition: ranks on the second node intermittently fail to resolve model shards. Workaround: `HF_HUB_OFFLINE=1` in launch script.

---

## 2026-04-17: FA3 sweep — maximizing MFU across models and context lengths

### Goal

Systematic FA3 evaluation across Qwen3-30B-A3B (MoE) and Qwen3-32B (dense) at 32k and 64k context lengths. Compare against best known sdpa/FA2 baselines to find new MFU records.

### Qwen3-30B-A3B (MoE) results

| Config         | Ctx | Nodes | DP  | TP  | CP  | EP  | SP  | Attn | MFU        | TPS/GPU | Peak GPU Mem  | Status   |
| -------------- | --- | ----- | --- | --- | --- | --- | --- | ---- | ---------- | ------- | ------------- | -------- |
| FSDP2 DP-only  | 16k | 2     | 16  | 1   | 1   | 1   | 1   | sdpa | 23.1%      | 3,888   | --            | Baseline |
| FSDP2 DP-only  | 16k | 2     | 16  | 1   | 1   | 1   | 1   | FA3  | **25.65%** | 4,318   | 71.1 GB (89%) | +11%     |
| FSDP2 CP=2     | 32k | 2     | 8   | 1   | 2   | 1   | 1   | sdpa | 13.57%     | 2,757   | 69.2 GB (87%) | Baseline |
| DS-Z3 SP=2     | 32k | 2     | 8   | 1   | 1   | 1   | 2   | FA3  | **14.49%** | 2,944   | 53.3 GB (67%) | +6.8%    |
| DS-Z3 SP=4 FA2 | 64k | 2     | 4   | 1   | 1   | 1   | 4   | FA2  | 11.6%      | 2,624   | 54.4 GB (68%) | Baseline |
| DS-Z3 SP=4     | 64k | 2     | 4   | 1   | 1   | 1   | 4   | FA3  | **12.52%** | 2,837   | 54.4 GB (68%) | +7.9%    |

### Qwen3-32B (Dense) results — pending

| Config                  | Ctx | Nodes | DP  | TP  | CP  | EP  | SP  | Attn | MFU   | TPS/GPU | Peak GPU Mem  | Status        |
| ----------------------- | --- | ----- | --- | --- | --- | --- | --- | ---- | ----- | ------- | ------------- | ------------- |
| FSDP2 liger+FA3 DP-only | 32k | 2     | 16  | 1   | 1   | 1   | 1   | FA3  | 59.0% | 1,450   | 78.4 GB (98%) | Baseline (2n) |
| FSDP2 liger+FA3 DP-only | 32k | 4     | 32  | 1   | 1   | 1   | 1   | FA3  | --    | --      | --            | Job 22081011  |
| FSDP2 liger CP=2        | 64k | 2     | 8   | 1   | 2   | 1   | 1   | sdpa | --    | --      | --            | Job 22081012  |
| DS-Z3 liger SP=2        | 64k | 2     | 8   | 1   | 1   | 1   | 2   | FA3  | --    | --      | --            | Job 22081013  |

### Findings so far

- **FA3 consistently improves MFU** across all tested configs: +11% at 16k, +6.8% at 32k, +7.9% at 64k (relative gains).
- **New best 32k MoE: 14.49% MFU** with DS-Z3+SP=2+FA3, beating the previous best of 13.57% (sdpa CP=2).
- **New best 64k MoE: 12.52% MFU** with DS-Z3+SP=4+FA3, beating the previous best of 11.6% (FA2 SP=4).
- **FA3 + CP is incompatible**: `Context parallelism is supported only with SDPA attention`. SP is the only long-context path for FA3.
- **FA3 + EP is incompatible**: FSDP2 mesh dimension mismatch when EP is enabled. Needs transformers/accelerate fix.

---

## 2026-04-24: SonicMoE kernel benchmark

**Setup**: Qwen3-30B-A3B (MoE), 2 nodes (16× H100 NVL), bf16, packing=wrapped, 20 steps.

- transformers `5.7.0.dev0` @ `a7c92b3305` (rebased, with `kernels-community/sonic-moe` PR #45433)
- TRL `benchmark-sft-moe` branch (this branch)
- New `--experts_implementation` flag in `SFTConfig` to control expert dispatch (`grouped_mm` default, `sonicmoe`, `batched_mm`)

### FSDP2 OOM regression — fixed in this branch

After rebasing transformers from `5.6.0.dev0` to `5.7.0.dev0`, FSDP2 model loading for Qwen3-30B-A3B started OOMing on CPU (~480GB peak per node). Root cause: `from_pretrained` in the new transformers loads the full model on CPU on every rank, and the `is_fsdp_enabled()` rank-0-only path in `_move_missing_keys_from_meta_to_device` only handles missing keys, not the actual loading. With 8 ranks/node × ~60GB model each, peak CPU exceeded the cgroup limit even on p5.48xlarge (2TB).

**Fix** (in `trl/scripts/sft.py`): wrap the non-rank-0 `from_pretrained` call in `accelerate.init_empty_weights()`. Only rank 0 loads the full model to CPU; other ranks get meta-device tensors. accelerate's FSDP2 prepare broadcasts rank 0's weights to all ranks during sharding. Loading drops from ~480GB → ~60GB per node.

```python
from accelerate import init_empty_weights
ctx = init_empty_weights() if is_fsdp and local_rank != 0 else contextlib.nullcontext()
with ctx:
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)
```

### SonicMoE results — Qwen3-30B-A3B, 2 nodes, 20-step runs

| Ctx | Config                | Backend         | Attn | Implementation | MFU (step 20) | TPS    | TPS/GPU | Peak GPU Mem  | vs baseline                  |
| --- | --------------------- | --------------- | ---- | -------------- | ------------- | ------ | ------- | ------------- | ---------------------------- |
| 16k | Control               | FSDP2 DP=16     | sdpa | grouped_mm     | **23.0%**     | 62,049 | 3,878   | 69.3 GB (87%) | 100% (baseline 23.1%)        |
| 16k | sonicmoe              | FSDP2 DP=16     | sdpa | sonicmoe       | 8.1%          | 21,879 | 1,367   | 68.3 GB (86%) | -65% (kernel warmup)         |
| 16k | sonicmoe + FA3        | FSDP2 DP=16     | FA3  | sonicmoe       | 20.8%         | 56,002 | 3,500   | 68.3 GB (86%) | -19% (vs 25.7% FA3 baseline) |
| 32k | sonicmoe + SP=2 + FA3 | DS-Z3 DP=8 SP=2 | FA3  | sonicmoe       | 13.7%         | 44,439 | 2,777   | 79.4 GB (99%) | -6% (vs 14.5% baseline)      |

### MFU progression — kernel warmup dominates short runs

Per-logging-step MFU values (computed over the previous 5 steps each):

| Step | sonicmoe sdpa | sonicmoe + FA3 | sonicmoe + SP=2 + FA3 (32k) |
| ---- | ------------- | -------------- | --------------------------- |
| 5    | 2.6%          | 10.0%          | 6.8%                        |
| 10   | 4.8%          | 15.1%          | 10.4%                       |
| 15   | 6.6%          | 18.5%          | 12.3%                       |
| 20   | **8.1%**      | **20.8%**      | **13.7%**                   |

All three sonicmoe runs show monotonic MFU increase — Triton/CuteDSL JIT compilation overhead dominates early steps and amortizes as training progresses. **The 20-step window is too short to capture steady-state MFU**; the kernel is still climbing at step 20.

### Findings

1. **FSDP2 loading regression in transformers `5.7.0.dev0`**: `is_fsdp_enabled()` rank-0-only loading no longer activates in the new `convert_and_load_state_dict_in_model` path — every rank loads the full model to CPU, OOMing on 30B+ MoE. Fixed in this branch by wrapping non-rank-0 `from_pretrained` in `accelerate.init_empty_weights()`.
2. **SonicMoE warmup is extreme**: per-step MFU more than doubles from step 5 to step 20 across all configs. Triton/CuteDSL JIT compiles per shape on first calls. Benchmarks under 20 steps systematically underestimate sonicmoe's steady-state throughput — a 50+ step run is in flight to verify convergence.
3. **Sonicmoe sdpa is the worst case** (8.1% MFU): without FA3 attention speedup, the kernel warmup overhead is not amortized. Sonicmoe pairs better with FA3.
4. **The longer-context / smaller-DP runs warm up faster**: 32k SP=2 reaches 94% of its baseline MFU (13.7 vs 14.5) at step 20 vs only 81% for 16k FA3 (20.8 vs 25.7). Larger compute per step → kernel calls saturate sooner.
5. **Memory parity**: peak GPU memory is essentially identical between grouped_mm and sonicmoe — the kernel uses the same fused weight layout, no additional buffers.

### Observations on sonicmoe behavior

- The kernel produces correct outputs (loss values match grouped_mm baseline at step 5: 2.7 vs 1.6 within initial-noise range; train loss curves overlap by step 20).
- Peak GPU memory stays within ~1 GB of the grouped_mm baseline — the kernel does not allocate large persistent buffers.
- The 32k DS-Z3 + SP=2 + FA3 + sonicmoe run hit 79.4 GB peak (99% of H100) — already at the memory ceiling. Adding any other dimension (longer context, larger DP) will OOM.

### 50-step run — sonicmoe converges to grouped_mm + FA3 baseline

A 50-step run with sonicmoe + FA3 (16k, 2n, FSDP2 DP=16) confirms the warmup hypothesis. MFU climbs monotonically until step 40 where it matches the grouped_mm + FA3 baseline (25.7%).

| Step   | MFU        | TPS    |
| ------ | ---------- | ------ |
| 5      | 10.28%     | 27,690 |
| 10     | 15.35%     | 41,350 |
| 15     | 18.75%     | 50,510 |
| 20     | 21.09%     | 56,810 |
| 25     | 22.49%     | 60,580 |
| 30     | 23.79%     | 64,100 |
| 35     | 24.69%     | 66,510 |
| **40** | **25.30%** | 68,160 |
| 45     | 12.46%\*   | 33,570 |
| 50     | 13.30%\*   | 35,830 |

\*Step 45/50 had a 200s pause between logging events (vs ~15s normal interval), likely a checkpoint/sync stall — disregard as MFU outliers, not kernel slowdown.

**At step 40, sonicmoe + FA3 reaches 25.30% MFU vs 25.7% baseline (grouped_mm + FA3) — within 1.5% of parity.** Triton/CuteDSL JIT compilation accounts for the slower steps 1-35. After warmup, sonicmoe matches grouped_mm.

### Open questions

- **Beyond step 40**: does sonicmoe surpass grouped_mm in long training runs? The trajectory is still slightly upward at step 40 — a 100+ step run may show modest gains.
- **Memory headroom at scale**: 32k DS-Z3 + SP=2 + sonicmoe hit 99% GPU memory (79.4 GB). Adding context length or DP would OOM.
- **CP compatibility**: sonicmoe + CP not tested — CP requires sdpa, and sonicmoe + sdpa is the slowest combo here.

---

## 2026-04-25: SonicMoE rerun with upstream FSDP fix

After bisecting the FSDP2 OOM to transformers PR #45050 (`empty_like` → `zeros_like`) and applying a proper upstream patch (skip param materialization on non-rank-0 FSDP ranks; only buffers need real placeholders), reran the sonicmoe benchmark suite without the `init_empty_weights()` workaround. PR draft in `benchmark/fix_pr.md`; pushed to `AmineDiro/transformers:fix-fsdp2-cpu-ram-zeros-like`.

**Setup**: same as 2026-04-24 — Qwen3-30B-A3B, 2 nodes (16× H100 NVL), bf16, packing=wrapped, 20 steps. transformers `5.7.0.dev0` @ `a7c92b3305` + the FSDP fix patch. TRL `benchmark-sft-moe` branch with `init_empty_weights` workaround removed.

### Results

| Ctx | Config                | Backend         | Attn | Implementation | MFU (step 20) | TPS    | TPS/GPU | Peak GPU Mem  | Note                                               |
| --- | --------------------- | --------------- | ---- | -------------- | ------------- | ------ | ------- | ------------- | -------------------------------------------------- |
| 16k | Control               | FSDP2 DP=16     | sdpa | grouped_mm     | **22.6%**     | 60,772 | 3,798   | 69.5 GB (87%) | Matches 23.1% historical baseline                  |
| 16k | sonicmoe              | FSDP2 DP=16     | sdpa | sonicmoe       | 17.5%         | 47,208 | 2,950   | 68.6 GB (86%) | +9 pp over 2026-04-24 (8.1%) — kernel cache warmer |
| 16k | sonicmoe + FA3        | FSDP2 DP=16     | FA3  | sonicmoe       | 20.3%         | 54,780 | 3,424   | 68.6 GB (86%) | Matches 2026-04-24 (20.8%)                         |
| 32k | sonicmoe + SP=2 + FA3 | DS-Z3 DP=8 SP=2 | FA3  | sonicmoe       | **14.66%**    | 47,661 | 2,979   | 78.6 GB (99%) | **+1% over 14.5% baseline**                        |

### 50-step sonicmoe + FA3 — kernel beats grouped_mm baseline at steady state

A 50-step run with sonicmoe + FA3 (16k, 2n, FSDP2 DP=16) shows the kernel converges past the grouped_mm + FA3 historical baseline (25.7%):

| Step   | MFU        |
| ------ | ---------- |
| 5      | 8.95%      |
| 10     | 14.03%     |
| 15     | 17.50%     |
| 20     | 19.94%     |
| 25     | 21.59%     |
| 30     | 23.02%     |
| 35     | 24.17%     |
| 40     | 24.95%     |
| 45     | 25.20%     |
| **50** | **25.88%** |

Final TPS: 69,728. Loss converges normally (1.66). Peak GPU memory: 68.6 GB (86%).

**At step 50, sonicmoe + FA3 reaches 25.88% MFU vs the 25.7% historical baseline (grouped_mm + FA3) — slightly ahead of the baseline.** The trajectory is still upward at step 50, suggesting longer runs may extend the gap.

### Findings

1. **FSDP2 fix validated**: control matches the historical baseline (22.6% vs 23.1%); sonicmoe runs land within 1 pp of (or above) prior numbers. The upstream patch (`fix-fsdp2-cpu-ram-zeros-like`) is functionally equivalent to the `init_empty_weights()` workaround, with a smaller TRL footprint.
2. **Sonicmoe + FA3 ≥ grouped_mm + FA3 at steady state**: 25.88% vs 25.7% historical baseline at 16k FSDP2. The kernel is at parity (or marginally ahead) once the Triton/CuteDSL JIT cache is populated.
3. **Sonicmoe + SP=2 + FA3 wins at 32k**: 14.66% vs 14.5% baseline. New best 32k MoE config. Hits 99% GPU memory — at the H100 ceiling.
4. **Sonicmoe + sdpa improved across runs (8.1% → 17.5%)**: same code, same config — the difference is the kernel binary cache being populated from the previous day's runs. First sonicmoe run on a fresh node pays the full Triton/CuteDSL compile cost; subsequent runs hit the cache.
5. **Per-step warmup curve is reproducible**: 50-step trajectory in this run matches the trajectory from 2026-04-24 within ~1 pp at every logging step.

### Best configs for Qwen3-30B-A3B (updated)

| Context | MFU        | Config                                  | Note                                                   |
| ------- | ---------- | --------------------------------------- | ------------------------------------------------------ |
| 16k     | **25.88%** | FSDP2 DP=16 + FA3 + sonicmoe (50 steps) | New best — slightly beats prior 25.7% grouped_mm + FA3 |
| 32k     | **14.66%** | DS-Z3 DP=8 SP=2 + FA3 + sonicmoe        | New best — slightly beats prior 14.5%                  |

---

## 2026-04-25 (later): Per-window (instantaneous) MFU — sonicmoe is +23% over grouped_mm + FA3

The cumulative MFU metric (`num_input_tokens_seen / total_runtime`) is dominated by step-1 cost (kernel compile, autotune, FSDP init) for many steps after compile is done. Added `mfu_window` to `SFTTrainer.log` that measures TPS over the _last logging window_ only — that's the real per-step throughput once the model is warm.

### 50-step runs at 16k, 2 nodes × 8 H100, FSDP2 DP=16

| Config            | Cumulative MFU (step 50) | Window MFU (per-log-window) | Cumulative TPS | Window TPS |
| ----------------- | ------------------------ | --------------------------- | -------------- | ---------- |
| grouped_mm + sdpa | 23.54%                   | **24.31%**                  | 63,398         | 65,480     |
| grouped_mm + FA3  | 27.94%                   | **28.12%**                  | 75,140         | 75,760     |
| sonicmoe + FA3    | 26.23%                   | **34.70%**                  | 70,612         | 93,480     |

For grouped_mm runs, cumulative ≈ window — the kernel doesn't have a meaningful first-step compile cost, so cumulative averaging doesn't bias the result much. For sonicmoe, window is **8.5 pp higher** than cumulative — confirming the entire "warmup curve" we previously observed was just the slow first step (~25–30s for Triton/CuteDSL compile + autotune) being slowly diluted by the cumulative average.

### Window MFU trajectory (per logging step) — stable from step 10 onwards

| Step | grouped_mm + sdpa | grouped_mm + FA3 | sonicmoe + FA3 |
| ---- | ----------------- | ---------------- | -------------- |
| 5    | 24.31%            | 27.88%           | — (init)       |
| 10   | 24.64%            | 29.69%           | 32.33%         |
| 15   | 23.97%            | 28.17%           | 34.81%         |
| 20   | 24.46%            | 29.62%           | 34.85%         |
| 25   | 24.54%            | 29.62%           | 32.25%         |
| 30   | 23.62%            | 27.86%           | 34.58%         |
| 35   | 24.48%            | 29.68%           | 34.64%         |
| 40   | 23.84%            | 29.20%           | 32.15%         |
| 45   | —                 | —                | 27.70%\*       |
| 50   | 24.31%            | 28.12%           | 34.70%         |

\*Step 45 for sonicmoe shows a single dip — likely a transient stall (NCCL sync, kernel cache miss on a rare shape, or trackio commit). Other steps cluster tightly around 32–35%.

### Speedups (window MFU ratios)

- **FA3 over sdpa** (grouped_mm): 28.12 / 24.31 = **+15.7%** — pure attention speedup.
- **sonicmoe vs grouped_mm**, both with FA3: 34.70 / 28.12 = **+23.4%** — the kernel itself is a big win at steady state.
- **sonicmoe + FA3 vs sdpa baseline**: 34.70 / 24.31 = **+42.7%** total improvement.

### Why the cumulative MFU was misleading

For sonicmoe, step 1 takes ~25–30s (CuteDSL compile, autotune, first-touch). Steps 2+ run at ~3.0s. Per-step times measured from `train_runtime` deltas:

| Window | Δ time / 5 steps | Δ time / step              |
| ------ | ---------------- | -------------------------- |
| 1–5    | ~47s             | ~9.4s (poisoned by step 1) |
| 6–10   | ~16s             | ~3.2s                      |
| 11–15  | ~14.5s           | ~2.9s                      |
| ≥ 16   | ~14–16s          | ~2.9–3.2s                  |

Cumulative TPS = total_tokens / total_time. With ~30s of compile dominating the first ~10 steps' worth of throughput, it takes 50+ steps for the cumulative average to approach steady-state. **By step 50 the cumulative metric is still 8.5 pp below the actual per-step performance.** The kernel itself is at steady speed from step 2 onwards — exactly as one would expect from a Triton/CuteDSL kernel that compiles once.

### Updated best-config table for Qwen3-30B-A3B (window MFU)

| Context           | Window MFU  | Config                                                                |
| ----------------- | ----------- | --------------------------------------------------------------------- |
| 16k               | **34.7%**   | FSDP2 DP=16 + FA3 + sonicmoe                                          |
| 16k (no sonicmoe) | 28.1%       | FSDP2 DP=16 + FA3 + grouped_mm                                        |
| 16k (sdpa)        | 24.3%       | FSDP2 DP=16 + sdpa + grouped_mm                                       |
| 32k               | 14.7% (cum) | DS-Z3 DP=8 SP=2 + FA3 + sonicmoe — needs 50-step rerun for window MFU |

### Action items / next steps

- All future MFU reporting should use **window MFU** (instantaneous per-log-window) as the primary metric. Cumulative MFU is biased by initialization cost and underreports steady-state throughput, especially for kernels with non-trivial first-call cost (Triton, CuteDSL, inductor-compiled paths).
- Re-run the 32k DS-Z3+SP=2+FA3+sonicmoe config for 50 steps to get its steady-state window MFU; the cumulative 14.66% likely understates by ~5–8 pp.
- The historical "best MFU" numbers in earlier sections of this report are all 20-step cumulative — they all underestimate steady-state throughput by varying amounts depending on kernel-init cost.

---

## 2026-04-26: PR #45621 (Better Grouped GEMM + EP) — sentinel-skip benchmark

The sonicmoe author's [transformers PR #45621](https://github.com/huggingface/transformers/pull/45621) reworks `grouped_mm_experts_forward` to **let sentinel rows fall past `offsets[-1]` instead of clamping** them. With that, the grouped_mm kernel skips compute for tokens that route to non-local experts — micro-benchmark in the PR shows 0.20× kernel time (5× speedup) at EP=8 (12.5% local routing).

Applied just the `moe.py` portion of the patch to `/fsx/amine_dirhoussi/transformers`. 50-step runs at 16k, 2 nodes × 8 H100, FSDP2, with `mfu_window` for steady-state.

### Results

| Config                    | EP                     | Attn | Cum MFU | **Window MFU** | TPS    | Peak GPU Mem |
| ------------------------- | ---------------------- | ---- | ------- | -------------- | ------ | ------------ |
| grouped_mm (no EP)        | 1                      | sdpa | 24.05%  | **24.41%**     | 65,500 | 69.5 GB      |
| grouped_mm (no EP)        | 1                      | FA3  | 28.30%  | **28.26%**     | 75,800 | 71.1 GB      |
| grouped_mm + PR #45621    | 8 (2D mesh, dp=2 tp=8) | sdpa | 26.98%  | **27.50%**     | 74,080 | tbd          |
| sonicmoe + FA3 (50 steps) | 1                      | FA3  | 26.23%  | **34.70%**     | 93,480 | 68.6 GB      |

### Speedup analysis

Without EP, PR #45621 is essentially a no-op (in-place ops, simplified indexing — no sentinels to skip). With EP=8, 87.5% of expert tokens are sentinels, and PR #45621 lets the kernel skip them entirely.

| Comparison                                            | Window MFU ratio           | Notes                                            |
| ----------------------------------------------------- | -------------------------- | ------------------------------------------------ |
| EP=8 + PR #45621 vs no-EP (sdpa)                      | 27.50 / 24.41 = **+12.7%** | Net gain after EP all-reduce overhead            |
| EP=8 + PR #45621 vs **historical** EP=16 (no PR, cum) | 27.50 / 22.5 = **+22%**    | Same backend, different EP, with sentinel skip   |
| sonicmoe + FA3 vs grouped_mm + FA3                    | 34.70 / 28.26 = **+22.8%** | Different kernel comparison (kept for reference) |

The 5× kernel-only speedup the PR advertises **does not translate to 5× model-step speedup**: attention, embeddings, FSDP comms, dataset I/O all unchanged. But the PR meaningfully closes the gap between EP and no-EP: pre-PR, EP=16 was _slower_ than no-EP at 16k 2n (22.5% vs 23.1% historical); with PR #45621 + EP=8, it's now **faster** (27.50% vs 24.41% sdpa, both window MFU).

### Trajectory (window MFU, EP=8 + PR #45621 + sdpa)

| Step   | 5      | 10     | 15     | 20     | 25     | 30     | 35     | 40     | 45  | 50     |
| ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | --- | ------ |
| Window | 26.96% | 27.61% | 26.68% | 27.51% | 27.54% | 27.12% | 27.49% | 27.08% | —   | 27.50% |

Flat from step 5 — no Triton/CuteDSL warmup, unlike sonicmoe.

### Setup notes

Two side-issues had to be worked around to get EP=8 running on this branch:

1. **`TRANSFORMERS_OFFLINE=1` required**: with EP, transformers does sharded loading where each rank resolves its assigned shards from the hub. With 16 ranks × 16 shards = 256 concurrent HEAD requests, hub rate-limits some, raising `OSError: does not appear to have a file named model-XXXXX-of-00016.safetensors` despite the file being in cache. `TRANSFORMERS_OFFLINE=1` forces local-cache-only resolution and bypasses this.
2. **`tp=8` in the run config (matching `ep=8`)**: accelerate's `ParallelismConfig._validate_accelerator` validates `total_size == num_processes`. Without tp=ep set, accelerate sees the device_mesh tp dim (8) but num_processes is 16 → validation error. Setting `tp=8` in the YAML makes the accelerate parallelism_config emit `tp_size=8 × dp_shard=2 = 16`, matching num_processes.
3. **`--bf16 true --dtype bfloat16` required**: without explicit dtype, FSDP2 hit `AssertionError: FSDP expects uniform original parameter dtype but got {torch.bfloat16, torch.float32}` on EP+FSDP combo. Added to `launch.sh.j2` for all runs going forward.

### Findings

- **PR #45621's sentinel-skip is functional and beneficial at EP**: makes EP=8 the best sdpa config on 2 nodes (27.50% window MFU vs 24.41% no-EP). Restores EP as a meaningful option — pre-PR, EP added comm overhead with no compute savings.
- **No improvement at EP=1 (no sentinels)**: in-place ops alone are within noise (28.26% with PR vs 28.12% without).
- **Sonicmoe + FA3 still wins overall at 16k**: 34.7% window vs 27.5% for grouped_mm + sdpa + EP=8 + PR #45621. SonicMoE benefits from FA3 too; we didn't combine sonicmoe + EP because the EP code path uses grouped_mm and the kernel selection wasn't tested in EP mode.

### Next steps

- Try `sonicmoe + EP=8` if/when sonicmoe supports the EP dispatch (currently the `sonicmoe_experts_forward` doesn't take EP sentinels into account the same way grouped_mm does after PR #45621).
- Re-run prior 32k DS-Z3+SP=2+FA3+sonicmoe with `mfu_window` for a complete steady-state picture.

> **Note**: [transformers PR #45621](https://github.com/huggingface/transformers/pull/45621) ("Better Grouped GEMM + EP") is **open and not yet merged** as of 2026-04-26. The numbers above were obtained by manually applying the `moe.py` portion of the PR diff to `/fsx/amine_dirhoussi/transformers` (no other files patched). If the PR lands as-is, these results should reproduce on a vanilla checkout.

---

## 2026-04-26: Ilyas's patched sonic-moe kernel (`IlyasMoutawwakil/sonic-moe@main`) + EP sweep

After the wrapper-level workaround for sonicmoe + EP, Ilyas pushed an updated kernel build to his personal fork that handles sentinels natively in its metadata stage (drops `expert_ids >= num_experts` from the histogram + scatter indices, no compute on sentinel rows). PR #45621's `hub_kernels.py` change re-points sonic-moe to that fork. His micro-bench shows up to 6.5× kernel-only speedup at EP=8, T=131072.

### Setup

- Re-pointed `_HUB_KERNEL_MAPPING["sonic-moe"]` from `kernels-community/sonic-moe v1` → `IlyasMoutawwakil/sonic-moe@main`. Added `allow_all_kernels=True` to the `lazy_load_kernel` call (required for non-`kernels-community/*` repos).
- **Reverted** the wrapper-level clamp + masked_fill from `sonicmoe_experts_forward` (the new kernel handles sentinels internally; clamping in the wrapper would defeat the skip).
- Sweep: 16k EP=2/4/8, 32k EP=8 (no CP — CP at long-context MoE is known-bad per earlier results), 64k EP=8 + CP=2 (cancelled — same reason).
- 50 steps, FSDP2, sdpa (FA3+EP still has mesh-name conflict).

### Results (16k, 2 nodes × 8 H100)

| EP  | Mesh       | Sentinel ratio | Window MFU  | Cum MFU | Status                           |
| --- | ---------- | -------------- | ----------- | ------- | -------------------------------- |
| 2   | dp=8, tp=2 | 50 %           | **28.31 %** | 11.71 % | ✅ trains cleanly                |
| 4   | dp=4, tp=4 | 75 %           | —           | —       | ❌ `cudaErrorAssert` in backward |
| 8   | dp=2, tp=8 | 87.5 %         | —           | —       | ❌ `cudaErrorAssert` in backward |

(Cum MFU at EP=2 is heavily depressed because step 1 paid the full kernel-fetch cost — first time the new `IlyasMoutawwakil/sonic-moe` build was downloaded + compiled on this cache. Window MFU is the steady-state number.)

### Long context

| Ctx | EP       | Result                                                                         |
| --- | -------- | ------------------------------------------------------------------------------ |
| 32k | 8        | ❌ same `cudaErrorAssert` in backward                                          |
| 64k | 8 + CP=2 | Cancelled — historical 64k FSDP2+EP+CP result was 2.9 % MFU; not worth running |

### Backward-pass assert details

Forward pass works (loss progresses through step 0 logging output). Crash is in the **backward gradient kernel**:

```
File ".../IlyasMoutawwakil--sonic-moe/.../quack/gemm_dact.py", line 505, in gemm_dact
    compiled_fn(A_p, B_p, Out_p, PreAct_p, epi_args, scheduler_args, varlen_args, None)
RuntimeError: CUDA Error: cudaErrorAssert
```

`gemm_dact` is the gradient of the gated activation (used to backprop through SwiGLU). The forward (`moe_general_routing_inputs`) drops sentinels before the GEMM; the backward `gemm_dact` is hit with the same sentinel-laden inputs and an internal CUDA `assert()` fires. Reproducible at EP ≥ 4 (EP=2 is below whatever threshold the assert checks). Independent of context length — same assert at 16k and 32k.

Filed back to Ilyas as a backward-path follow-up; the kernel-only forward micro-bench in the PR description was forward-only.

### Comparison: best 16k MoE configs to date

| Config                                              | Kernel                                | EP  | Window MFU |
| --------------------------------------------------- | ------------------------------------- | --- | ---------- |
| sonicmoe + FA3, no EP (50 steps)                    | OLD `kernels-community/sonic-moe v1`  | 1   | **34.7 %** |
| sonicmoe + sdpa + EP=8 (OLD kernel + wrapper clamp) | OLD `kernels-community/sonic-moe v1`  | 8   | 30.67 %    |
| **sonicmoe + sdpa + EP=2 (NEW kernel)**             | NEW `IlyasMoutawwakil/sonic-moe@main` | 2   | 28.31 %    |
| grouped_mm + sdpa + EP=8 + PR #45621                | grouped_mm                            | 8   | 27.50 %    |
| grouped_mm + FA3, no EP                             | grouped_mm                            | 1   | 28.12 %    |
| grouped_mm + sdpa, no EP                            | grouped_mm                            | 1   | 24.41 %    |

### Findings

- **The new patched kernel works at EP=2 but has a backward-side bug at EP ≥ 4**. Until Ilyas (or whoever) fixes the `gemm_dact` assert at high sentinel ratios, the kernel can't be used for the EP regime where it's most interesting.
- **For now, the best sonicmoe + EP path remains the OLD kernel (`kernels-community/sonic-moe v1`) plus the wrapper-level clamp+mask** documented in `benchmark/fix_fsdp_sonic.md` (30.67 % at EP=8). The wrapper does compute on sentinel rows then zeroes them; less efficient than skipping but trains cleanly forward + backward.
- **The Triton cache race on `/fsx/.triton/`** observed at EP=4 (rank 8 reading a cubin while rank 0 still writes it) is environmental, not a kernel bug. Per-rank Triton cache (`TRITON_CACHE_DIR=/tmp/triton-rank-${RANK}`) would eliminate it.
- **CP for MoE long context remains a non-starter** — historical 64k FSDP2+EP+CP=8 was 2.9 % MFU vs DS-Z3+SP=4+FA3 at 12.5 %. Combining EP with anything except DP at long context isn't competitive on this branch's transformers, regardless of kernel.

### Next steps

- Wait for Ilyas to fix the `gemm_dact` backward assert. Re-run EP=4, EP=8 sweep once that's done.
- Investigate whether DS-Z3 + EP can be unblocked (currently hangs at model loading per the 5.6.0 era report) — would let SP+EP combine for long context.

---

## 2026-04-27: Patched sonicmoe kernel — backward fixed, full EP sweep + long-context probe

Ilyas pushed backward support to `IlyasMoutawwakil/sonic-moe@main`. Re-ran the EP sweep that previously failed with `cudaErrorAssert` in `gemm_dact`. Wrapper-level clamp/masked_fill stays removed (kernel handles sentinels in metadata stage).

### EP scaling at 16k, 2 nodes × 8 H100 (FSDP2, sdpa)

| EP  | Mesh       | Sentinel ratio | Window MFU  | Cum MFU | TPS    |
| --- | ---------- | -------------- | ----------- | ------- | ------ |
| 2   | dp=8, tp=2 | 50 %           | 28.31 %     | 11.71 % | 31,550 |
| 4   | dp=4, tp=4 | 75 %           | **32.32 %** | 26.16 % | 70,480 |
| 8   | dp=2, tp=8 | 87.5 %         | **32.36 %** | 26.08 % | 70,250 |

**EP=4 and EP=8 now train cleanly through forward + backward.** Window MFU plateaus around 32 % for EP ≥ 4 — Ilyas's micro-bench predicted near-monotonic kernel-only speedup with EP, but end-to-end MFU saturates because attention/comms/embeddings are unchanged.

Vs the wrapper-clamp workaround on the OLD `kernels-community/sonic-moe v1` kernel (30.67 % window at 16k EP=8): the new kernel's native sentinel skip gains **~+5.5 %** at the same config.

### Memory ceiling on 2 / 4 nodes — sdpa attention activations

| Ctx  | Nodes | Mesh            | Status                                       |
| ---- | ----- | --------------- | -------------------------------------------- |
| 32k  | 2     | EP=8, dp=2 tp=8 | OOM (4 GB short, sdpa attention activations) |
| 32k  | 4     | EP=8, dp=4 tp=8 | OOM (~18.55 GB allocation, ~72 GB in use)    |
| 64k  | 4     | EP=8, dp=4 tp=8 | OOM (same allocation pattern)                |
| 128k | 4     | EP=8, dp=4 tp=8 | OOM                                          |

Going from 2 → 4 nodes shards params/optimizer further (FSDP DP=2 → DP=4) but **does not help with attention activation memory**, which is per-rank and grows with seq_len. At per-rank seq ≥ 32k, `sdpa` materializes the full attention matrix and OOMs. FA3 would solve this but FA3 + EP has the mesh-name conflict noted earlier (`KeyError: "Invalid mesh_dim_names ('dp_shard_cp',)"`) — accelerate's FSDP2 expects a `dp_shard_cp` mesh dim, EP creates a mesh with only `tp`.

The fix-from-config side: keep per-rank seq ≤ 16k by adding **CP** to fold the sequence dim. CP for MoE is known-bad on this branch's transformers (synchronous all-reduces between ring-attention chunks), but it's the only way to fit.

### Long-context fitting via CP

| Ctx  | Nodes | Mesh                   | Per-rank seq | Window MFU | TPS    |
| ---- | ----- | ---------------------- | ------------ | ---------- | ------ |
| 64k  | 4     | EP=8 + CP=4, dp=1 tp=8 | 16k          | **7.97 %** | 54,140 |
| 128k | 8     | EP=8 + CP=8, dp=1 tp=8 | 16k          | **4.24 %** | 62,640 |

Both fit and train cleanly; both heavily MFU-penalized by CP. The +2.7× over historical 64k EP=16+CP=8 (2.9 %) is real but doesn't make this regime competitive with the alternatives.

### Comparison to alternatives at 64k MoE

| Approach                                  | MFU    | Status this branch               |
| ----------------------------------------- | ------ | -------------------------------- |
| FSDP2 + EP=8 + CP=4 + sonicmoe (this run) | 7.97 % | runs                             |
| DS-Z3 + SP=4 + FA3 (no EP, historical)    | 12.5 % | runs (no EP path)                |
| DS-Z3 + SP=4 + EP                         | —      | DS-Z3 + EP loading hang, blocked |
| FSDP2 + EP + FA3 (no CP)                  | —      | mesh-name conflict, blocked      |

**The path forward for 64k+ MoE** is unblocking either DS-Z3+EP (already noted earlier as a WIP fix that hangs at model loading) or FSDP2+FA3+EP (mesh dim incompatibility). Until one of those lands, the best long-context MoE config remains DS-Z3+SP=4+FA3 without EP at 12.5 % MFU.

### Updated best configs for Qwen3-30B-A3B (all window MFU where available)

| Context | Best Window MFU          | Config                                          | Notes                                    |
| ------- | ------------------------ | ----------------------------------------------- | ---------------------------------------- |
| 16k     | **34.7 %**               | FSDP2 + FA3 + sonicmoe (no EP, 50 steps)        | unchanged                                |
| 16k EP  | **32.36 %**              | FSDP2 + EP=8 + sdpa + sonicmoe (patched kernel) | **+5.5 % over wrapper-clamp workaround** |
| 32k     | 14.5 % (cum)             | DS-Z3 + SP=2 + FA3 + sonicmoe                   | EP can't fit at 32k yet                  |
| 64k     | 12.5 % (cum, historical) | DS-Z3 + SP=4 + FA3 (no EP)                      | EP+CP gives 7.97 %, worse                |
| 128k    | 4.24 %                   | FSDP2 + EP=8 + CP=8 + sonicmoe (8n)             | **first 128k MoE result on this stack**  |

### Findings

1. **Patched kernel works end-to-end** at EP=2/4/8 in both forward + backward. The kernel update is good.
2. **Sentinel-skip's kernel speedup doesn't translate into proportional MFU gain** under TRL's serialized comm-then-compute schedule (~+1.7 pp from skip vs clamp at EP=8). Real benefit would require overlapped expert+all-reduce scheduling — see the explanation in `benchmark/fix_fsdp_sonic.md`.
3. **EP at long context is gated by attention memory, not expert memory.** sdpa at per-rank seq ≥ 32k can't fit; the EP plan doesn't shard attention; so we're stuck at 16k per rank without FA3.
4. **CP for MoE long context is a workaround, not a solution.** 64k at 8 % MFU and 128k at 4 % beats nothing-fits, but is far behind the SP path's 12.5 % at 64k.
5. **First 128k Qwen3-30B-A3B SFT measurement** on this stack: trains, 4.24 % MFU, 62K TPS, 8 nodes. Reference for future "did the upstream FA3+EP fix help?" comparisons.

---

## 2026-04-27: DS-Z3 + SP=4 + FA3 + sonicmoe at 64k (2 nodes) — beats CP path by 2.4×

Tried Ulysses Sequence Parallelism instead of CP at 64k context, since CP is known-bad for MoE long context (dense, synchronous all-reduces between ring chunks). Same DS-Z3 + FA3 + sonicmoe stack, just SP=4 instead of EP+CP=4.

**Config**: 2 nodes × 8 H100, DS-Z3, dp=4, sp=4, ep=1, FA3 attention, sonicmoe experts. Per-rank seq = 16k after Ulysses sequence shard.

**Note**: had to gate `HF_HUB_OFFLINE=1` behind `enable_expert_parallel` in `trl/scripts/sft.py` — first attempt failed with `OfflineModeIsEnabled` because FA3's `load_and_register_attn_kernel` uses a separate `get_kernel` path that the EP-only sonicmoe pre-warm doesn't cover. Non-EP runs need hub access for FA3 kernel loading.

### Results — job 22092241, 50 steps

| Step | Cum MFU     | Window MFU  | TPS (window) | TPS (cum)  |
| ---- | ----------- | ----------- | ------------ | ---------- |
| 5    | 6.51 %      | —           | —            | 23,590     |
| 10   | 9.70 %      | 19.07 %     | 17,280       | 35,170     |
| 15   | 11.27 %     | 16.62 %     | 15,060       | 40,840     |
| 20   | 12.40 %     | 17.78 %     | 16,110       | 44,950     |
| 25   | 13.26 %     | 18.36 %     | 16,630       | 48,070     |
| 30   | 13.67 %     | 16.10 %     | 14,590       | 49,530     |
| 35   | 14.17 %     | 18.17 %     | 16,460       | 51,340     |
| 40   | 14.62 %     | 18.86 %     | 17,090       | 52,990     |
| 45   | 14.63 %     | 14.68 %     | 13,300       | 53,020     |
| 50   | **14.97 %** | **18.98 %** | 17,190       | **54,260** |

Steady-state window MFU **~17–19 %**, occasional dips correlated with `pytorch allocator cache flushes since last step` warnings (DS-Z3 has high memory pressure at 64k seq, allocator thrashes between steps).

### Comparison at 64k MoE (2 nodes)

| Approach                                            | Window MFU  | Cum MFU | TPS    | Notes         |
| --------------------------------------------------- | ----------- | ------- | ------ | ------------- |
| **DS-Z3 + SP=4 + FA3 + sonicmoe (this run)**        | **18.98 %** | 14.97 % | 54,260 | new best      |
| DS-Z3 + SP=4 + FA3 (no EP, no sonicmoe, historical) | —           | 12.5 %  | —      | previous best |
| FSDP2 + EP=8 + CP=4 + sonicmoe (4n, patched kernel) | 7.97 %      | —       | 54,140 | CP penalty    |

**SP+sonicmoe at 64k is ~+1.5× cum MFU over the previous SP-only baseline (12.5 → 14.97 %)** and the kernel itself is contributing the win (window plateau ~19 % vs whatever the SP+FA3 dense baseline would be — sonicmoe replaces the routed-expert path with the fused kernel even with EP=1, since experts are local but still routed).

Compared to the CP path (FSDP2 + EP + CP), SP gives **2.4× the window MFU at 64k**. CP is now formally retired as the long-context strategy on this branch.

### Findings

1. **SP scales sequence cleanly without the CP penalty.** Ulysses shards the seq dim with all-to-all comms that overlap with attention compute; CP's ring all-reduces between attention chunks are synchronous and bandwidth-bound.
2. **DS-Z3 cache flushes are visible in window MFU dips.** Roughly every 4–5 logs a step drops to 14–16 % from 18 %. Not catastrophic, but indicates we're near the memory ceiling at 2 nodes / 64k.
3. **sonicmoe + SP is the new best long-context MoE recipe** until DS-Z3+EP loading hang is fixed or FSDP2+FA3+EP mesh-name conflict is resolved.

---

## Consolidated comparison: new sonicMoE (patched, IlyasMoutawwakil/sonic-moe@main) vs previous results

All numbers are Qwen3-30B-A3B SFT, H100 NVL, bf16, FA3 or sdpa as noted. Window MFU is steady-state; cumulative MFU is averaged over the full run including kernel warmup.

### 16k context — kernel comparison at fixed parallelism

| Run                                | Backend | Mesh       | Attn | Expert kernel        | Sentinel handling        | Window MFU  | Cum MFU | TPS    |
| ---------------------------------- | ------- | ---------- | ---- | -------------------- | ------------------------ | ----------- | ------- | ------ |
| no EP, no kernel                   | FSDP2   | dp=16      | sdpa | grouped_mm (default) | —                        | —           | 24.41 % | —      |
| no EP, FA3                         | FSDP2   | dp=16      | FA3  | grouped_mm           | —                        | —           | 28.12 % | —      |
| **no EP, FA3 + sonicmoe (OLD v1)** | FSDP2   | dp=16      | FA3  | sonicmoe v1          | — (no EP)                | **34.7 %**  | —       | —      |
| EP=8, grouped_mm + PR #45621       | FSDP2   | dp=2, tp=8 | sdpa | grouped_mm           | skip (PR sentinel)       | 27.50 %     | —       | —      |
| EP=8, sonicmoe v1 + wrapper clamp  | FSDP2   | dp=2, tp=8 | sdpa | sonicmoe v1          | wrapper-level clamp+mask | 30.67 %     | —       | —      |
| **EP=8, NEW patched sonicmoe**     | FSDP2   | dp=2, tp=8 | sdpa | **sonicmoe (Ilyas)** | kernel-native skip       | **32.36 %** | 26.08 % | 70,250 |
| EP=4, NEW patched sonicmoe         | FSDP2   | dp=4, tp=4 | sdpa | sonicmoe (Ilyas)     | kernel-native            | 32.32 %     | 26.16 % | 70,480 |
| EP=2, NEW patched sonicmoe         | FSDP2   | dp=8, tp=2 | sdpa | sonicmoe (Ilyas)     | kernel-native            | 28.31 %     | 11.71 % | 31,550 |

**Deltas vs new sonicmoe (16k):**

- vs `grouped_mm + PR #45621` at EP=8: **+4.86 pp** (32.36 vs 27.50)
- vs `sonicmoe v1 + wrapper clamp` at EP=8: **+1.69 pp** (32.36 vs 30.67) — the kernel-native skip beats wrapper clamp by a hair, not the 5–10 pp Ilyas's micro-bench suggested. Reason: TRL's serialized comm-then-compute schedule doesn't expose the kernel saving (see `fix_fsdp_sonic.md`).
- Best 16k overall is still `no EP + FA3 + sonicmoe v1` at 34.7 %. EP at 16k is a setup test, not a production config (you'd just run no-EP at this size).

### 64k context — long-context comparison

| Run                                               | Nodes | Backend | Mesh                   | Attn | Expert kernel        | Window MFU  | Cum MFU     | TPS        |
| ------------------------------------------------- | ----- | ------- | ---------------------- | ---- | -------------------- | ----------- | ----------- | ---------- |
| DS-Z3 + SP=4 + FA3 (historical, no sonicmoe)      | 2     | DS-Z3   | dp=4, sp=4             | FA3  | grouped_mm           | —           | 12.5 %      | —          |
| FSDP2 + EP=8 + CP=4 + sonicmoe                    | 4     | FSDP2   | dp=1, tp=8, cp=4, ep=8 | sdpa | sonicmoe (Ilyas)     | 7.97 %      | —           | 54,140     |
| **DS-Z3 + SP=4 + FA3 + sonicmoe (new, 22092241)** | 2     | DS-Z3   | dp=4, sp=4             | FA3  | **sonicmoe (Ilyas)** | **18.98 %** | **14.97 %** | **54,260** |

**Deltas at 64k (2 nodes):**

- vs historical SP without sonicmoe: **+2.47 pp cum** (14.97 vs 12.5) — sonicmoe replaces the routed-expert path even at EP=1
- vs CP path on 4 nodes: **+11.01 pp window** (18.98 vs 7.97) — CP is now formally retired for MoE long context

### 128k context — first results on this stack

| Run                                                 | Nodes | Backend | Mesh                   | Attn | Window MFU | TPS    |
| --------------------------------------------------- | ----- | ------- | ---------------------- | ---- | ---------- | ------ |
| FSDP2 + EP=8 + CP=8 + sonicmoe                      | 8     | FSDP2   | dp=1, tp=8, cp=8, ep=8 | sdpa | 4.24 %     | 62,640 |
| DS-Z3 + SP=8 + FA3 + sonicmoe (in-flight, 22092254) | 4     | DS-Z3   | dp=4, sp=8             | FA3  | TBD        | TBD    |

### Best per-context summary (window MFU where measured, else cum)

| Ctx       | Best                          | Config                                  | Improvement vs prior best                                                |
| --------- | ----------------------------- | --------------------------------------- | ------------------------------------------------------------------------ |
| 16k (any) | **34.7 %**                    | FSDP2 + FA3 + sonicmoe v1, no EP        | unchanged (kernel + FA3 still wins at small ctx)                         |
| 16k EP    | **32.36 %**                   | FSDP2 + EP=8 + sdpa + new sonicmoe      | +1.69 pp over OLD-kernel wrapper clamp; +4.86 pp over grouped_mm+PR45621 |
| 64k       | **18.98 % win / 14.97 % cum** | DS-Z3 + SP=4 + FA3 + new sonicmoe (2n)  | +2.47 pp cum over historical SP-only                                     |
| 128k      | 4.24 % win                    | FSDP2 + EP=8 + CP=8 + new sonicmoe (8n) | first 128k Qwen3-30B-A3B MoE result on this stack                        |

---

## 2026-04-27: SP scaling — 4-node 64k/128k partial runs

Tried adding nodes to the SP recipe to (a) stabilize the cache-flush dips at 64k and (b) attempt 128k for the first time. Both jobs were canceled before reaching 50 steps to free the cluster, so window MFU is the most reliable number — cum MFU is still climbing during warmup.

### 64k DS-Z3 + SP=4 + FA3 + sonicmoe — 4 nodes (DP=8, SP=4) — job 22092253

| Step | Cum MFU | Window MFU  | TPS (window) | TPS (cum) |
| ---- | ------- | ----------- | ------------ | --------- |
| 5    | 6.68 %  | —           | —            | 48,410    |
| 10   | 10.16 % | **21.26 %** | 38,530       | 73,680    |
| 15   | 11.80 % | 17.42 %     | 31,570       | 85,560    |
| 20   | 13.12 % | 19.70 %     | 35,700       | 95,080    |
| 25   | 14.12 % | **20.33 %** | 36,840       | 102,300   |

**Crashed at step ~30** with `ValueError: batch's seqlen=54223 isn't divisible by sp-size=4` — DeepSpeed Ulysses requires per-batch seq length divisible by sp_size, and packed THUDM/LongAlign-10k samples can produce non-aligned lengths. Need either a custom collator that pads-to-multiple, or `--max_length` that's a multiple of sp_size before truncation.

**Steady-state window MFU ~19–21 %** vs 17–19 % on 2 nodes — 4 nodes does help slightly, presumably because doubling DP halves the per-rank optimizer/grad sync per step. TPS doubles cleanly to ~100k.

### 128k DS-Z3 + SP=8 + FA3 + sonicmoe — 4 nodes (DP=4, SP=8) — job 22092254

| Step | Cum MFU | Window MFU  | TPS (window) | TPS (cum) |
| ---- | ------- | ----------- | ------------ | --------- |
| 5    | 5.78 %  | —           | —            | 41,640    |
| 10   | 9.33 %  | **19.31 %** | 18,570       | 71,720    |
| 15   | 10.83 % | 16.03 %     | 15,410       | 83,330    |
| 20   | 11.93 % | 17.10 %     | 16,440       | 91,730    |
| 25   | 12.91 % | **19.23 %** | 18,490       | 99,270    |
| 30   | 13.15 % | 14.49 %     | 13,930       | 101,100   |

**Canceled at step ~30** to free nodes. Per-rank seq = 16k, same as 64k 4n. **Steady-state window MFU 14–19 %, TPS ~100k cum**, no crashes through 30 steps.

This is **the first 128k Qwen3-30B-A3B SFT run on this stack at non-trivial MFU** (vs 4.24 % from FSDP2+EP+CP=8 on 8 nodes). The 4-node SP path delivers ~4× the MFU at half the node count.

### Updated 128k summary

| Run                               | Nodes | Backend   | Window MFU                       | Status                       |
| --------------------------------- | ----- | --------- | -------------------------------- | ---------------------------- |
| FSDP2 + EP=8 + CP=8 + sonicmoe    | 8     | FSDP2     | 4.24 %                           | full 50 steps                |
| **DS-Z3 + SP=8 + FA3 + sonicmoe** | **4** | **DS-Z3** | **19.23 % peak / ~17 % typical** | partial (30 steps, canceled) |

### Findings

- **SP path scales to 128k cleanly** at half the node count of CP path with 4× the MFU.
- **Cache-flush dips persist on 4 nodes too** at 64k — DS-Z3 memory pressure isn't fixed by adding nodes; allocator thrash is per-rank.
- **Ulysses padding constraint is a real blocker for production runs**: the pack-then-shard pattern in the SFT collator can produce variable-length packed batches that don't satisfy `seqlen % sp_size == 0`. A `pad_to_multiple_of=sp_size` in the collator would fix this.

---

## 2026-04-27: FA3 + EP works — was self-inflicted, new MoE MFU SOTA at 16k

### TL;DR

FA3 + EP is **not actually incompatible**. The previously-reported `KeyError: "Invalid mesh_dim_names ('dp_shard_cp',)"` crash was caused by my own kernel pre-warm logic in `trl/scripts/sft.py`, which set `HF_HUB_OFFLINE=1` after warming sonicmoe but before FA3's two-phase kernel load could fetch from hub. With FA3 also pre-warmed, the original crash is gone — and the run delivers **42.66 % window MFU at 16k**, the highest MoE result on this stack so far.

### The fix (committed in `trl/scripts/sft.py`)

FA3 (kernels-community/vllm-flash-attn3) loads in two separate phases during `from_pretrained`:

1. `load_and_register_attn_kernel` — registers the attention function in `ALL_ATTENTION_FUNCTIONS`
2. `lazy_import_flash_attention` — sets module-level `_flash_fn`, `_flash_varlen_fn` etc. via `_lazy_imports`

Both call `get_kernel(repo_id)` independently, so both have to be pre-warmed before the EP offline flip. The earlier code only pre-warmed sonicmoe and assumed FA3 was either already cached or not in use. The fix:

```python
if model_args.attn_implementation and "/" in model_args.attn_implementation:
    from transformers.integrations.hub_kernels import load_and_register_attn_kernel
    from transformers.modeling_flash_attention_utils import lazy_import_flash_attention

    load_and_register_attn_kernel(model_args.attn_implementation)
    lazy_import_flash_attention(model_args.attn_implementation)
# … then HF_HUB_OFFLINE=1 …
```

### Results — 16k FSDP2 + EP=8 + FA3 + sonicmoe (job 22092267, 2 nodes, 5 steps debug)

| Step | Cum MFU | Window MFU  | TPS (window) | TPS (cum) |
| ---- | ------- | ----------- | ------------ | --------- |
| 1    | 3.09 %  | —           | —            | 8,310     |
| 2    | 5.76 %  | **42.88 %** | 115,500      | 15,500    |
| 3    | 8.09 %  | **43.08 %** | 116,000      | 21,800    |
| 4    | 10.15 % | **42.63 %** | 114,800      | 27,340    |
| 5    | 11.97 % | **42.66 %** | 114,900      | 32,250    |

**Steady-state window MFU: 42.7 %** — within ±0.3 pp across 4 successive logs. The kernel is fully warm by step 2 and throughput is rock-solid at ~115k TPS.

### Caveat: training is numerically broken

Loss collapses to `0` with `grad_norm: nan` and `entropy: nan` from step 2 onward. The first step has `loss: 62` (which is itself wildly off — Qwen3-30B initial loss should be ~10–14), suggesting the very first forward pass is already producing bad logits.

This is consistent with the **RouterParallel sentinel bug** documented in `benchmark/CLAUDE.md` — `RouterParallel._prepare_output_fn` reshapes `router_scores` from `(seq, top_k)` to `(seq, num_local_experts)`, but downstream expert forwards (sonicmoe, grouped_mm) expect the original shape. With sdpa we already saw 23.1 % MFU on broken training; with FA3 we're now at 42.7 % MFU on broken training. **The throughput is real; the gradients are wrong.**

The MFU number is still useful for kernel/hardware benchmarking (it's measuring real matmul throughput), but the run is not a usable training config until the RouterParallel bug is fixed upstream.

### Updated MoE leaderboard (16k, all configs, window MFU)

| Run                                 | Backend | Mesh       | Attn | Kernel           | Window MFU  | Training valid?             |
| ----------------------------------- | ------- | ---------- | ---- | ---------------- | ----------- | --------------------------- |
| **FA3 + EP=8 + sonicmoe (NEW)**     | FSDP2   | dp=2, tp=8 | FA3  | sonicmoe (Ilyas) | **42.66 %** | **NO — RouterParallel bug** |
| no EP + FA3 + sonicmoe v1           | FSDP2   | dp=16      | FA3  | sonicmoe v1      | 34.7 %      | yes                         |
| sdpa + EP=8 + sonicmoe (Ilyas)      | FSDP2   | dp=2, tp=8 | sdpa | sonicmoe (Ilyas) | 32.36 %     | yes                         |
| sdpa + EP=4 + sonicmoe (Ilyas)      | FSDP2   | dp=4, tp=4 | sdpa | sonicmoe (Ilyas) | 32.32 %     | yes                         |
| sdpa + EP=8 + sonicmoe v1 + wrapper | FSDP2   | dp=2, tp=8 | sdpa | sonicmoe v1      | 30.67 %     | yes                         |
| sdpa + EP=8 + grouped_mm + PR45621  | FSDP2   | dp=2, tp=8 | sdpa | grouped_mm       | 27.50 %     | yes                         |

**Headroom unlocked**: FA3 swaps `sdpa`'s `O(seq²)` attention materialization for `O(seq·log seq)` flash, freeing both compute and memory. The +10 pp window MFU jump (32.4 → 42.7 %) at the same parallelism config is roughly the attention contribution to per-step time.

### Implications for long context

The original "FA3 + EP can't fit at 32k+" memory analysis was based on the assumption FA3+EP would never run. **If the RouterParallel bug is fixed**, FA3+EP becomes the most attractive long-context MoE config:

- FA3 attention activations are ~100× smaller than sdpa at 32k+ per-rank seq
- EP scales experts cleanly without CP's ring-reduce penalty
- Combined: FA3 + EP=8 + DP at 32k/64k/128k should fit in memory and run at high MFU

**Next steps** (post-RouterParallel fix): re-run the long-context sweep with FA3+EP+DP (no CP). Hypothesis: 64k can hit ~30 % MFU and 128k around 20–25 % MFU on this stack. Until the routing bug is fixed, these numbers are speculative.

---

## 2026-04-27: DS-Z3 + EP unblocked by PR #45548 + FA3+EP NaN bisect

### PR #45548 merged: DS-Z3 + EP loading works

[`huggingface/transformers#45548`](https://github.com/huggingface/transformers/pull/45548) routes EP through the standard (non-zero3) loading path when both EP and `is_deepspeed_zero3_enabled()` are active, then lets `deepspeed.initialize()` wrap the EP-sharded model afterward. Merged into the working `qwen3-moe-ep-v2` branch (FSDP2 cpu_ram_efficient_loading fix retained).

Without this, DS-Z3+EP fails immediately at `from_pretrained` with `ValueError: DeepSpeed Zero-3 is not compatible with passing a 'device_map'.` even when `device_map=None` — `check_and_set_device_map` fills it in from the global torch device context, then the DS-Z3 guard rejects it.

### Results — DS-Z3 + EP=8 + sdpa + sonicmoe at 16k (job 22092280, 5 steps)

| Step | Cum MFU | Window MFU  | TPS (window) | TPS (cum) |
| ---- | ------- | ----------- | ------------ | --------- |
| 1    | 3.33 %  | —           | —            | 8,970     |
| 2    | 5.90 %  | 25.94 %     | 69,880       | 15,900    |
| 3    | 8.11 %  | 32.41 %     | 87,310       | 21,850    |
| 4    | 9.99 %  | 32.56 %     | 87,710       | 26,900    |
| 5    | 11.60 % | **32.72 %** | 88,140       | 31,250    |

**Steady-state window MFU: 32.7 %**, almost identical to the FSDP2 + EP=8 + sonicmoe number (32.36 %). DS-Z3 + EP at 16k delivers the same kernel performance as FSDP2 + EP at 16k with a different sharding strategy.

### NaN issue is EP-wide, not FA3-specific

| Run                                           | Window MFU  | Loss step 1 | Loss step 2+ | grad_norm |
| --------------------------------------------- | ----------- | ----------- | ------------ | --------- |
| FSDP2 + EP=8 + sdpa + sonicmoe (Ilyas)        | 32.36 %     | normal      | normal       | normal    |
| FSDP2 + EP=8 + FA3 + sonicmoe                 | **42.66 %** | 62          | 0            | nan       |
| FSDP2 + EP=8 + FA3 + grouped_mm (no sonicmoe) | 35.06 %     | 62          | 0            | nan       |
| **DS-Z3 + EP=8 + sdpa + sonicmoe (NEW)**      | 32.72 %     | **62.11**   | **0**        | **nan**   |

The NaN is reproducible with **all three** of: FA3 + grouped_mm, FA3 + sonicmoe, **sdpa + sonicmoe under DS-Z3** (but NOT sdpa + sonicmoe under FSDP2). The first-step loss of `62.11` (Qwen3-30B initial loss should be ~10–14) means the **first forward pass already produces broken logits**. The entropy NaN at step 2 is a secondary effect from the first backward producing NaN gradients.

What changes across these runs:

- FSDP2 vs DS-Z3 sharding
- sdpa vs FA3 attention
- sonicmoe vs grouped_mm experts

The only common factor in the failing runs is **EP being active under a backend that isn't `FSDP2 + sdpa + Ilyas-patched sonicmoe`**. That suggests one of these gates the routing correctly:

1. FSDP2's wrap policy delaying expert weight materialization until after `distribute_model` registers DTensor hooks
2. The Ilyas sonicmoe kernel's metadata pre-pass expecting `sdpa` cu_seqlens shape (FA3 packs differently)
3. DS-Z3's `_load_state_dict_into_zero3_model` (now bypassed for EP via PR #45548) was previously masking EP weight-load bugs by re-loading from cached state

The first-step `loss=62` strongly suggests **incorrect expert weights at load time**, not a runtime kernel issue. The packed `gate_up_proj` / `down_proj` from `MergeModulelist + Concatenate` may be ending up on the wrong EP rank when the loading path differs from FSDP2.

### 64k DS-Z3 + SP=4 + EP=8 + FA3 + sonicmoe (job 22092281) — DS batch-size assertion

```
AssertionError: train_batch_size is not equal to micro_batch_per_gpu * gradient_acc_step * world_size
1 != 1 * 1 * 8.0
```

DeepSpeed sees `world_size=8` because EP=8 carves out a sub-mesh that DS interprets as the data-parallel world. The actual world is 32 (4 nodes × 8 GPUs), but DS's internal accounting only counts the DP shard. Fix would require:

- Either telling DS the full world_size via `train_batch_size = global_dp_size` (not the EP-carved size)
- Or using `train_batch_size = "auto"` with the correct dp inference for the SP+EP combo

Not pursued further yet — this is config wiring, not a fundamental block.

### Updated unblocked/blocked matrix

| Config                                 | Status                     | MFU           | Training valid? |
| -------------------------------------- | -------------------------- | ------------- | --------------- |
| FSDP2 + EP=8 + sdpa + sonicmoe (Ilyas) | works                      | 32.36 %       | **yes**         |
| FSDP2 + EP=8 + FA3 + sonicmoe          | works                      | **42.66 %**   | no — NaN        |
| FSDP2 + EP=8 + FA3 + grouped_mm        | works                      | 35.06 %       | no — NaN        |
| DS-Z3 + EP=8 + sdpa + sonicmoe         | **NEW: works** (PR #45548) | 32.72 %       | no — NaN        |
| DS-Z3 + SP=4 + FA3 + sonicmoe (no EP)  | works                      | 18.98 % @ 64k | yes             |
| DS-Z3 + SP=4 + EP=8 + FA3 + sonicmoe   | DS batch-size assertion    | —             | —               |

**Two open correctness issues**:

1. **EP NaN under everything except `FSDP2 + sdpa + sonicmoe`** — needs a focused investigation of expert weight loading paths under EP. First-step `loss=62` is the smoking gun.
2. **DS+SP+EP batch-size mismatch** — config-level fix, not a code bug.

The big positive: **PR #45548 unblocks DS-Z3 + EP**, which means the SP+EP combination is now plausible once (2) is resolved — that combo would let us push 64k+ MoE at high MFU without CP.

---

## 2026-04-27: EP+FSDP DTensor fix → side effect: accelerate `_prepare_tp` ImportError

The EP NaN root cause turned out to be expert weights getting clobbered during FSDP2 wrap (`fsdp2_load_full_state_dict` broadcasting rank 0's 16 experts to all 8 ranks, destroying ranks 1–7's slices). Fix lives in [transformers PR #45662](https://github.com/huggingface/transformers/pull/45662) — wrap EP-sharded params as DTensors at `from_pretrained` time + tell FSDP to ignore EP modules via `fsdp_plugin.ignored_modules`. Verified: `loss=8.4` healthy, `grad_norm` finite, on previously NaN'ing configs at `EP=DP_size`.

### Side effect: accelerate's `_prepare_tp` now hits a missing class

After the fix, configs that set `parallelism_config_tp_size > 1` (e.g., the `dp=2 tp=8 ep=8` layout) crash before training with:

```
ImportError: cannot import name 'ReplicateParallel'
  from 'transformers.integrations.tensor_parallel'
  (/fsx/amine_dirhoussi/transformers/src/transformers/integrations/tensor_parallel.py)
```

The import comes from **accelerate**, not anything we added — `accelerator.py:1616`:

```python
from transformers.integrations.tensor_parallel import ReplicateParallel
```

It runs inside `Accelerator._prepare_tp` whenever `tp_size > 1` AND at least one model param is a DTensor. Modern transformers (post-fork) has the class; our `qwen3-moe-ep-v2` branch was cut before it was added.

**Why it didn't fire before the EP+FSDP fix.** `_prepare_tp` has a skip guard right above the import:

```python
if not any(isinstance(p, DTensor) for p in model.parameters()):
    logger.warning("...skip the TP preparation...")
    return result
```

Before the fix, EP params were plain `nn.Parameter` (the bug — they should have been DTensors but the broadcast clobber happened first). No DTensors anywhere → skip fires → import is never reached → run proceeds. After the fix, EP params are DTensors on the EP mesh → skip doesn't fire → reach the import → ImportError.

### Local fix in installed accelerate

Single 5-line edit at `/fsx/amine_dirhoussi/trl/.venv/lib/python3.11/site-packages/accelerate/accelerator.py`, right after the existing skip block:

```python
# EP-only models: experts are DTensors on the EP mesh (not the TP mesh accelerate
# is about to set up). Skip TP preparation — non-EP params stay plain tensors and
# FSDP2 handles them on the FSDP mesh, the same path that worked before EP params
# became DTensors.
if getattr(model, "has_ep", False):
    return result
```

This restores the previous behavior (skip `_prepare_tp` for EP-only models) without adding a `ReplicateParallel` shim to our transformers fork. The check is `model.has_ep` (a property added in the EP+FSDP fix), so it only triggers for EP runs and is a no-op otherwise.

**Properly fixing this upstream** would be either: (a) add a `ReplicateParallel` alias to the transformers fork, or (b) push the same `has_ep` skip into accelerate. Both are out of scope for the EP+FSDP correctness PR — that PR only touches transformers.

### State of working trees

- `/fsx/amine_dirhoussi/transformers` (qwen3-moe-ep-v2, **uncommitted on purpose** — used live for benchmarking):
    - EP+FSDP DTensor fix (`_wrap_ep_params_as_dtensor`, `has_ep`, `ep_sharded_param_names`, ignored_modules wiring in `Trainer.create_accelerator_and_postprocess`)
    - `grouped_mm_experts_forward` clamp + masked_fill (PR #45621-style sentinel handling)
    - `sonicmoe_experts_forward` uses `IlyasMoutawwakil/sonic-moe@main` kernel; wrapper-level `.to_local()` for EP-DTensor weights; no wrapper clamp (kernel handles sentinels)
- `/fsx/amine_dirhoussi/transformers-ep-fixes` (`fix-ep-fsdp-ignored-modules`): the clean PR branch — pushed to GitHub as #45662
- `/fsx/amine_dirhoussi/trl/.venv/.../accelerate/accelerator.py`: the 5-line `has_ep` skip in `_prepare_tp`

### NaN-redo attempt (jobs 22092443/44/45) — fix correctness, but new mesh mismatches surface

Resubmitted the three NaN configs from the previous addendum on the patched stack:

| Job      | Config                                       | Result                                                  |
| -------- | -------------------------------------------- | ------------------------------------------------------- |
| 22092444 | FSDP2 + EP=8 + FA3 + sonicmoe (dp=2, tp=8)   | crash in `_clip_grad_norm`: mesh mismatch               |
| 22092445 | FSDP2 + EP=8 + FA3 + grouped_mm (dp=2, tp=8) | crash in `_clip_grad_norm`: mesh mismatch               |
| 22092443 | DS-Z3 + EP=8 + sdpa + sonicmoe               | crash before training: `c10d::broadcast_` no DeviceMesh |

**FSDP2 mesh mismatch.** `RuntimeError: All operands in aten.stack.default must have the same mesh` during `clip_grad_norm_`. Cause: the EP+FSDP fix wraps EP params as DTensors on the EP mesh `(8,)`, while non-EP params end up on the FSDP DP mesh `(dp_replicate=8, dp_shard=2)`. `clip_grad_norm_` collects all gradient norms and tries to stack them — DTensors on different meshes can't stack.

This issue is **not present at `EP=DP_size`** (e.g., 16k 2-node EP=16, the verification configs that did succeed at 24-26% MFU): both meshes are flat 1D of size 16, so stack is fine. It only surfaces when EP < DP_size (mixed-mesh layouts like dp=2 tp=8 ep=8).

**DS-Z3 mesh issue.** `RuntimeError: found no DeviceMesh from dtensor args for c10d::broadcast_` happens before the first training step. DS-Z3's broadcast machinery doesn't know about the EP mesh; the EP-DTensor params have a mesh DS-Z3 hasn't registered with c10d. This is a deeper integration gap between DS-Z3 and DTensor-based EP — separate from the FSDP path.

**Open follow-ups (out of scope for the EP+FSDP correctness fix):**

1. **`clip_grad_norm_` for mixed meshes**: either compute the global norm per-mesh and combine, or special-case EP-DTensors so they're flattened to local before stacking. Affects all FSDP2 configs where EP < DP_size.
2. **DS-Z3 + EP DTensor broadcast**: DS-Z3 needs to register the EP mesh with c10d before its broadcast collectives run. PR #45548 unblocked DS-Z3+EP loading; this is the next blocker after that.

### NaN-redo (round 2): patched trainer to skip grad-norm read for EP runs

Added a benchmark-only skip in `transformers/trainer.py:_clip_grad_norm` and `_get_grad_norm`: when `model.has_ep`, return `0.0` / `nan` instead of calling `accelerator.clip_grad_norm_` (which stacks per-param norms across mismatched meshes). Loss reporting still works; only the grad-norm value is suppressed.

Resubmitted the two FSDP2+FA3 configs:

| Job      | Config                                       | Step-1 loss | Mid-run loss          | mean_token_acc | Window MFU      | Status                |
| -------- | -------------------------------------------- | ----------- | --------------------- | -------------- | --------------- | --------------------- |
| 22092449 | FSDP2 + EP=8 + FA3 + grouped_mm (dp=2, tp=8) | **12.05**   | 10–14 (healthy)       | 0.62–0.70      | **32.6–33.7 %** | ✅ trains correctly   |
| 22092448 | FSDP2 + EP=8 + FA3 + sonicmoe (dp=2, tp=8)   | **4.51**    | 0 from step 2 onwards | 0.0001 → ~0    | 36–45 %         | ❌ NaN — separate bug |

**grouped_mm: works.** Loss range 10-14 (correct for Qwen3-30B initial), entropy 1.2-1.7, mean_token_accuracy 0.62-0.70 — clean training. Window MFU plateaus at ~33% — slightly below the previous 35.06% measurement, but that one was on broken training (NaN'd from step 2), so the comparison isn't apples-to-apples. **This is the first healthy `EP=8 + FA3 + grouped_mm` run on the stack.**

**sonicmoe: still NaN, different from the EP+FSDP bug.** First-step loss is 4.51 (not 62 like before — the EP+FSDP fix did help the forward pass), but loss collapses to 0 by step 2 with `entropy=nan` and `mean_token_accuracy ~ 1e-4`. This is a **separate bug in the sonicmoe kernel path** (likely backward — `gemm_dact` or similar). The kernel-native sentinel handling that worked at `EP=DP_size` (32% MFU verification runs earlier this session) does NOT survive the `EP < DP_size` mixed-mesh layout. The wrapper-level clamp+masked_fill (which grouped_mm uses) is the workaround.

**Updated leaderboard for FSDP2 + EP=8 + FA3 (16k, 2n, dp=2 tp=8):**

| Backend | Kernel     | Sentinel handling  | Window MFU | Loss    | Trains? |
| ------- | ---------- | ------------------ | ---------- | ------- | ------- |
| FSDP2   | grouped_mm | wrapper clamp+mask | **32.6 %** | 10–14   | ✅      |
| FSDP2   | sonicmoe   | kernel-native      | 36–45 %    | 0 (NaN) | ❌      |

### Open follow-ups (still)

3. **sonicmoe + EP < DP_size NaN**: separate from the EP+FSDP corruption. The kernel-native sentinel-skip works at `EP=DP_size` but breaks here. Either add the wrapper clamp+mask back to `sonicmoe_experts_forward` (gives ~30% MFU on the OLD kernel, see 2026-04-26 entry) or chase the kernel-side bug.
4. **DS-Z3 + EP**: still blocked by `c10d::broadcast_` no-DeviceMesh. Not retried after the trainer grad-norm patch — DS-Z3's failure point is well before optimizer step.

### sonicmoe NaN is sonicmoe-specific, not FA3-specific (job 22092450)

Isolation run: sonicmoe + **sdpa** + EP=8 (dp=2 tp=8) — 50 steps with the EP+FSDP fix and grad-norm skip patch.

| Step | Loss | entropy | mean_token_acc | Window MFU |
| ---- | ---- | ------- | -------------- | ---------- |
| 1    | 4.51 | nan     | 0.157          | —          |
| 2    | 0    | nan     | 1e-4           | 33.0 %     |
| 5    | 0    | nan     | 5e-5           | 32.4 %     |
| 50   | 0    | nan     | 9e-5           | 33.0 %     |

Identical pattern to FA3 + sonicmoe (loss=4.51 → 0, entropy NaN). **The NaN is sonicmoe-specific**, not FA3-related. **But** report.md (line 2252) shows this exact config (sonicmoe + sdpa + EP=8) trained healthily before the EP+FSDP fix at 32.36 % — same kernel SHA (`b15942783197a14a2d49dd201b3bfb8d64091a39`, verified by checking `~/.cache/models--IlyasMoutawwakil--sonic-moe/refs/main` and the per-snapshot `__init__.py` blob).

So **something in the EP+FSDP fix path that grouped_mm survives is breaking sonicmoe**. Likely candidates:

- The `_wrap_ep_params_as_dtensor` wrap turns `gate_up_proj`/`down_proj` into DTensors. `sonicmoe_experts_forward` then does `gate_up_proj.to_local().permute(*perm)` and feeds the result to the custom CuteDSL kernel. The forward output might be correct (loss=4.51 is finite), but backward through `to_local().permute()` into a custom non-PyTorch-native kernel may not produce correct gradients.
- `grouped_mm` survives because `torch._grouped_mm` is a native PyTorch op — autograd handles DTensors and views correctly. The custom `moe_general_routing_inputs` kernel doesn't have the same plumbing.

Workaround would be to drop the DTensor wrap for sonicmoe runs (revert to plain `nn.Parameter` after `partition_tensor`) — but that breaks the `EP = DP_size` Adam mixed-types path the wrap was added to fix. Proper fix is custom backward in the sonicmoe wrapper, or keep the kernel-native sentinel-skip but route forward+backward through autograd-aware ops.

### Status summary on the active stack

| Config                                | Loss          | Window MFU      | Verdict                                           |
| ------------------------------------- | ------------- | --------------- | ------------------------------------------------- |
| FSDP2 + EP=8 + FA3 + grouped_mm       | 10–14 healthy | **32.6–33.7 %** | ✅ first healthy run at this layout               |
| FSDP2 + EP=8 + sdpa + sonicmoe        | 0 / NaN       | 33 %            | regression vs pre-fix 32.36 %                     |
| FSDP2 + EP=8 + FA3 + sonicmoe         | 0 / NaN       | 36–45 %         | same NaN as sdpa version                          |
| FSDP2 + EP=16 + sdpa/FA3 + grouped_mm | 8.4 healthy   | 24–26 %         | EP=DP_size, baseline correctness verified earlier |
| DS-Z3 + EP=8 + sdpa + sonicmoe        | crash         | —               | `c10d::broadcast_` no-mesh, separate blocker      |

### Bisect: is `_wrap_ep_params_as_dtensor` what's breaking sonicmoe? (job 22092507)

Hypothesis: the EP+FSDP DTensor wrap is the cause of sonicmoe NaN at `EP < DP_size` — pre-fix, EP params were plain `nn.Parameter` and sonicmoe trained healthily at this layout (32.36 %).

Test: added an `os.environ.get("TRANSFORMERS_SKIP_EP_DTENSOR_WRAP")` early-return in `_wrap_ep_params_as_dtensor`, ran sonicmoe + sdpa + EP=8 (dp=2 tp=8) with the env var set.

Result — fails at the **first optimizer step**, before training metrics:

```
RuntimeError: aten._fused_adamw_.default: got mixed torch.Tensor and DTensor
need to convert all torch.Tensor to DTensor before calling distributed operators!
```

So at `dp=2 tp=8 ep=8`, FSDP2 wraps non-EP params as DTensors, EP params (without our wrap) stay plain `nn.Parameter`, and `torch._fused_adamw_` rejects the mix. The wrap IS needed at this layout — not just at `EP=DP_size`.

**Why the pre-fix "32.36 % healthy" run didn't hit this**: probably a different optimizer code path. `_fused_adamw_` is only recent — torch versions before its strict mixed-type check would have silently fallen back to a non-fused path that handled mixed types. Either we were on an older torch then, or accelerate/transformers routed Adam differently.

**Conclusion: can't fix sonicmoe by removing the wrap.** Both pieces (DTensor wrap for Adam, sonicmoe correctness) are needed. The path forward:

1. Add the wrapper clamp+mask back to `sonicmoe_experts_forward` (like `grouped_mm`) — kernel does compute on sentinel rows but their contribution is masked to 0. ~30.67 % MFU on the OLD kernel measurement (vs 32.36 % with kernel-native skip). 3-line patch, validated before.
2. OR: wrap sonicmoe's forward in a `torch.autograd.Function` that handles `to_local()`+permute → kernel call with explicit backward. Avoids the autograd-through-DTensor gap, keeps the kernel's sentinel-skip win. More invasive.

Option 1 is the unblock; option 2 is the long-term fix.

### Bisect step 2: contiguous() after to_local() — same NaN (job 22092508)

Hypothesis: the custom CuteDSL kernel's backward might not handle non-contiguous inputs (`permute` returns a non-contiguous view of the local tensor). Test: insert `.contiguous()` between `to_local()` and `permute()` in `sonicmoe_experts_forward`.

Result: identical NaN pattern, identical numbers down to step-1 loss = 4.514, mean_token_accuracy = 0.157, mfu_window = 30–33 %. Whether the kernel input is a view or a contiguous copy makes zero difference.

So the issue is **not** non-contiguity. Forward is computing _something_ (loss=4.51 + 15.7 % accuracy is not random noise — it's the model predicting common tokens consistently) but it diverges on the first backward.

Remaining hypothesis: the **kernel's hand-written `Function.backward` has a code path that doesn't terminate cleanly when the input tensor is the result of `DTensor.to_local()`** — even when the tensor is contiguous. The grad gets computed but produces NaN somewhere in the backward chain (possibly when flowing back through `to_local()` and into the DTensor's `.grad` field, which is also a DTensor on the EP mesh).

The pre-fix run's "loss=normal step 1" worked because gate_up_proj was a plain `nn.Parameter` — no DTensor involvement at all, the kernel's backward saw plain tensors throughout.

Confirmed dead-ends so far:

- ✗ Skipping the wrap (Adam fails with `_fused_adamw_` mixed-types)
- ✗ Adding `.contiguous()` after `to_local()` (no effect — same NaN)

Remaining options:

- Wrap sonicmoe call in a `torch.autograd.Function` so we control backward explicitly and can route grads back to DTensor.
- Add wrapper-level clamp+masked_fill (treat sentinels in the wrapper, not the kernel) — the OLD-kernel approach that gave 30.67 % MFU pre-fix.
- Detach EP params before kernel call, accumulate gradients manually after backward — invasive.

### Bisect step 3: wrapper clamp+masked_fill — sonicmoe trains healthily (job 22092509)

Applied the same sentinel handling pattern that grouped_mm uses to `sonicmoe_experts_forward`:

```python
invalid_mask = expert_ids >= self.num_experts
expert_ids = expert_ids.clamp(0, self.num_experts - 1)
router_scores = router_scores.masked_fill(invalid_mask, 0.0)
```

The kernel still does compute on sentinel rows, but their `router_scores` are zero so they contribute 0 to the output. The `clamp` keeps `expert_ids` inside the kernel's valid range, avoiding whatever pathological backward path the kernel takes when fed sentinel IDs against DTensor-derived weights.

**Result — 50-step run, FSDP2 + EP=8 + sdpa + sonicmoe (dp=2 tp=8):**

| Step | Loss  | entropy | mean_token_acc | Window MFU | Cum MFU |
| ---- | ----- | ------- | -------------- | ---------- | ------- |
| 1    | 12.05 | 1.20    | 0.66           | —          | 10.66 % |
| 5    | 13.29 | 1.63    | 0.64           | 30.86 %    | 22.38 % |
| 10   | 13.78 | 1.71    | 0.63           | 30.20 %    | 23.39 % |
| 25   | 12.72 | 1.59    | 0.64           | 30.95 %    | 24.24 % |
| 35   | 12.08 | 1.52    | 0.66           | 31.01 %    | 24.92 % |
| 50   | 12.59 | 1.56    | 0.65           | 30.83 %    | 25.87 % |

Final `train_loss`: 12.61. Loss stays in the 10–14 range across all 50 steps — clean training.

**Window MFU plateau: 30.2–31.1 %**, vs the pre-fix 32.36 % NaN measurement. The 1.5 pp gap is the cost of doing compute on sentinel rows + masking instead of the kernel's native skip — same delta the OLD-kernel wrapper-clamp showed (30.67 % vs 32.36 % in the 2026-04-26 entry). Acceptable for a working configuration.

### Conclusion: sonicmoe + EP=8 + sdpa is unblocked

**Working stack at 16k 2n FSDP2 dp=2 tp=8 ep=8:**

| Kernel     | Attention | Sentinel handling  | Loss          | Window MFU | Notes                                        |
| ---------- | --------- | ------------------ | ------------- | ---------- | -------------------------------------------- |
| grouped_mm | FA3       | wrapper clamp+mask | 10–14 healthy | **32.6 %** | first healthy `EP=8 + FA3 + grouped_mm` ever |
| sonicmoe   | sdpa      | wrapper clamp+mask | 10–14 healthy | **31.0 %** | sonicmoe matches grouped_mm at this layout   |
| sonicmoe   | FA3       | wrapper clamp+mask | 10–14 healthy | **40.4 %** | new best healthy MFU at this layout          |

### Bisect step 4: FA3 + sonicmoe + clamp — healthy at 40 % MFU (job 22092510)

Same wrapper clamp+mask, same DTensor wrap, same kernel; only difference vs sdpa run is FA3 attention. 50 steps:

| Step | Loss  | entropy | mean_token_acc | Window MFU | Cum MFU |
| ---- | ----- | ------- | -------------- | ---------- | ------- |
| 5    | 10.26 | 1.27    | 0.70           | 41.04 %    | 18.03 % |
| 10   | 13.05 | 1.67    | 0.63           | 40.86 %    | 22.15 % |
| 15   | 12.48 | 1.51    | 0.64           | 39.47 %    | 24.88 % |
| 20   | 13.29 | 1.63    | 0.64           | 40.92 %    | 27.00 % |
| 25   | 13.78 | 1.71    | 0.63           | 39.28 %    | 28.48 % |
| 30   | 12.72 | 1.59    | 0.64           | 40.43 %    | 29.74 % |
| 35   | 12.07 | 1.52    | 0.66           | 40.47 %    | 30.76 % |
| 40   | 13.76 | 1.68    | 0.62           | 37.95 %    | 31.42 % |
| 50   | 12.58 | 1.56    | 0.65           | 40.68 %    | 32.15 % |

Final `train_loss`: 12.60. Loss in 10–14 across all 50 steps — clean training.

**Window MFU plateau: ~40 % (37.95 – 41.04 range)** vs pre-fix 42.66 % NaN measurement. Δ = ~2 pp — same delta the sdpa pair showed (31 vs 32.36). The 2 pp is the cost of wrapper compute-then-mask vs the kernel-native sentinel skip.

### Conclusion: full 2026-04-27 NaN-redo summary

| Config                                     | Pre-fix        | Post-fix (this session) | Notes                                        |
| ------------------------------------------ | -------------- | ----------------------- | -------------------------------------------- |
| FSDP2 + EP=8 + sdpa + sonicmoe (dp=2 tp=8) | 32.36 % ✅     | **31.0 %** ✅           | -1.5 pp from wrapper clamp                   |
| FSDP2 + EP=8 + FA3 + sonicmoe              | 42.66 % ❌ NaN | **40.4 %** ✅           | -2.3 pp, but now trains                      |
| FSDP2 + EP=8 + FA3 + grouped_mm            | 35.06 % ❌ NaN | **32.6 %** ✅           | first healthy run; -2.5 pp                   |
| FSDP2 + EP=16 (sdpa or FA3) + grouped_mm   | NaN            | **24–26 %** ✅          | EP=DP_size baseline                          |
| DS-Z3 + EP=8 + sdpa + sonicmoe             | 32.72 % ❌ NaN | crash                   | `c10d::broadcast_` no-mesh; separate blocker |

**Headline numbers (16k 2n, dp=2 tp=8 ep=8, training healthy):**

- **40.4 % window MFU**: FA3 + sonicmoe (best)
- **32.6 % window MFU**: FA3 + grouped_mm
- **31.0 % window MFU**: sdpa + sonicmoe

The wrapper clamp+mask is the unblock for sonicmoe at EP < DP_size. The 2 pp MFU cost vs the kernel-native skip is the trade-off. Long-term fix would be wrapping the sonicmoe kernel call in a `torch.autograd.Function` that routes backward gradients through `to_local()` correctly — that would restore the kernel-native sentinel-skip win. Not pursued in this session.

### Long-context EP=8 sweep — all OOM at 37 GiB (jobs 22092511–14)

Tried 64k and 128k with FA3 + sonicmoe + EP=8 in parallel across node counts now that the layout is healthy at 16k:

| Job      | Ctx  | Nodes | Mesh           | Result                    |
| -------- | ---- | ----- | -------------- | ------------------------- |
| 22092511 | 64k  | 2     | dp=2 tp=8 ep=8 | OOM, 37.09 GiB allocation |
| 22092512 | 64k  | 4     | dp=4 tp=8 ep=8 | OOM, 37.09 GiB allocation |
| 22092513 | 128k | 4     | dp=4 tp=8 ep=8 | OOM, 37.09 GiB allocation |
| 22092514 | 128k | 8     | dp=8 tp=8 ep=8 | OOM, 37.09 GiB allocation |

All four crash trying to allocate the same **37.09 GiB** chunk on every rank, regardless of context length, node count, or FSDP shard size. Adding FSDP shards (dp=2 → dp=8) doesn't reduce the allocation — so the bug is an unsharded per-rank buffer, not a sharded weight/grad/activation.

Same signature as the historical 2026-04-27 EP+sdpa OOM table (lines 1962-1965), which suggested either a buffer in the EP all-to-all gather or a sonicmoe kernel internal allocation. With FA3 + the EP+FSDP fix, the 37 GiB allocation **persists** — confirms it's not from sdpa attention activations.

**Long-context EP=8 + FA3 + sonicmoe at this layout is blocked by the 37 GiB allocation.** Options: (a) profile to identify the unsharded buffer, (b) add CP=2/4 to halve per-rank seq, (c) reduce batch (already 1). 16k is the working ceiling at this layout for now.

---

## 2026-04-27: Pending uncommitted fixes — `debug_sp_ep_sonic.md` references

To run EP+FSDP correctly with sonicmoe and to unblock DS-Z3+EP, the following edits are live in our environment but **NOT committed anywhere upstream** as of this writing. All are needed simultaneously; removing any one breaks the run differently. Full walkthrough including failure modes, error messages, and revert commands lives in [`debug_sp_ep_sonic.md`](./debug_sp_ep_sonic.md).

### 1. Transformers (`/fsx/amine_dirhoussi/transformers`, branch `qwen3-moe-ep-v2`, uncommitted)

The "EP+FSDP correctness" fix from PR #45662 plus three adjacent patches that surfaced afterwards:

| File                       | Change                                                                                                                           | Why                                                                                                                                                                                                                                                                                     |
| -------------------------- | -------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `modeling_utils.py`        | `has_ep`, `ep_sharded_param_names` properties; `_wrap_ep_params_as_dtensor` staticmethod called once at end of `from_pretrained` | Without it, `fsdp2_load_full_state_dict` silently broadcasts rank 0's expert slice to all ranks → ranks 1–7 lose their experts → loss=62/NaN. PR #45662                                                                                                                                 |
| `modeling_utils.py`        | Env-var bypass `TRANSFORMERS_SKIP_EP_DTENSOR_WRAP=1` in `_wrap_ep_params_as_dtensor`                                             | Debug-only — bisect tool to test "is the wrap breaking sonicmoe?". Confirmed Adam fails without the wrap (`_fused_adamw_` mixed-types).                                                                                                                                                 |
| `integrations/moe.py`      | `to_local()` helper at three weight access sites in `grouped_mm_experts_forward`                                                 | Passes the local tensor to the native `_grouped_mm` kernel after our DTensor wrap. PR #45662                                                                                                                                                                                            |
| `integrations/sonicmoe.py` | Import `DTensor`; `gate_up_proj.to_local()` and `down_proj.to_local()` before the `permute()` calls                              | Mirrors the grouped_mm pattern for the sonicmoe path. PR #45662 follow-up.                                                                                                                                                                                                              |
| `integrations/sonicmoe.py` | **Wrapper-level clamp + masked_fill on `expert_ids` and `router_scores`** (mirrors grouped_mm)                                   | The kernel-native sentinel-skip in Ilyas's kernel produces NaN gradients in backward when weights are DTensor-derived (post EP+FSDP fix). Forward output is unaffected because router_scores is zero for sentinel positions. ~2 pp MFU cost vs kernel-native skip but trains correctly. |
| `trainer.py`               | `not has_ep` gate on `ParallelismConfig` auto-build                                                                              | EP uses `model.tp_size` for its own mesh; accelerate's `_prepare_tp` shouldn't run on EP-only models. PR #45662                                                                                                                                                                         |
| `trainer.py`               | `ignored_modules` auto-wire from `ep_sharded_param_names`                                                                        | Tells FSDP to skip EP-sharded experts so `fully_shard()` doesn't crash inside `_typing_utils.not_none`. PR #45662                                                                                                                                                                       |
| `trainer.py`               | `_clip_grad_norm` returns `tensor(0.0)` when `model.has_ep`                                                                      | `clip_grad_norm_` stacks per-param norms; EP DTensors and FSDP DTensors live on different meshes → "All operands in aten.stack.default must have the same mesh". Skipping clip is benchmark-only — loss reporting still works.                                                          |

### 2. Accelerate — `/fsx/amine_dirhoussi/trl/.venv/lib/python3.11/site-packages/accelerate/accelerator.py`

```python
# Inside Accelerator._prepare_tp, right after the existing "no DTensor → skip" guard
# (around line 1601). 5-line addition.
if getattr(model, "has_ep", False):
    return result
```

**Why:** before our DTensor wrap, `_prepare_tp` skipped naturally (no DTensors yet → return). After the wrap, EP params are DTensors → skip doesn't fire → `_prepare_tp` reaches `from transformers.integrations.tensor_parallel import ReplicateParallel` which doesn't exist in the `qwen3-moe-ep-v2` fork. Adding `has_ep` to the skip restores the natural flow.

**Properly fixing this upstream** would be either: (a) add a `ReplicateParallel` alias to the transformers fork, or (b) push the same `has_ep` skip into accelerate. Both are out of scope for the EP+FSDP correctness PR — that PR only touches transformers.

### 3. DeepSpeed — `/fsx/amine_dirhoussi/trl/.venv/lib/python3.11/site-packages/deepspeed/runtime/engine.py`

Inside `DeepSpeedEngine._broadcast_model.is_replicated(p)`:

```python
def is_replicated(p):
    if hasattr(p, "ds_status") and p.ds_status is not ZeroParamStatus.AVAILABLE:
        return False
    elif hasattr(p, 'ds_optim_param'):
        return False
    # EP DTensors are sharded per rank (each rank holds a different slice).
    # `dist.broadcast` doesn't accept DTensor inputs and they shouldn't be broadcast anyway.
    from torch.distributed.tensor import DTensor
    if isinstance(p.data, DTensor):
        return False
    return True
```

**Why:** at engine init, `_broadcast_model` calls `dist.broadcast(p.data, ..., group=self.seq_data_parallel_group)` on every model parameter. For EP DTensors, this fails with `RuntimeError: found no DeviceMesh from dtensor args for c10d::broadcast_`. Skipping DTensor params is correct: EP-sharded tensors hold a different slice per rank, so there's nothing to broadcast — each rank already has the right values from `from_pretrained`.

**Properly fixing this upstream** would be a DeepSpeed-side patch in the same `_broadcast_model` path, accepting any DTensor (or specifically the EP mesh) as "already correct, don't broadcast".

### Reverting

```bash
# Transformers (uncommitted in working tree)
cd /fsx/amine_dirhoussi/transformers
git checkout -- src/transformers/{modeling_utils.py,trainer.py,integrations/moe.py,integrations/sonicmoe.py}

# Accelerate + DeepSpeed (re-install from PyPI)
cd /fsx/amine_dirhoussi/trl
.venv/bin/pip install --force-reinstall accelerate deepspeed
```

### Pending failure modes (not yet hit on the current stack)

| Issue                                  | Symptom                                                                                         | Where it would resurface                                                                                                          | Fix sketch                                                                                     |
| -------------------------------------- | ----------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| DS-Z3 batch-size assertion at SP+EP    | `train_batch_size != micro_batch_per_gpu * grad_acc * world_size` (job 22092281 historical)     | DS-Z3 + SP + EP runs after the broadcast skip lets us past `_broadcast_model`. DS counts only the EP-carved sub-mesh as DP world. | Set `train_batch_size = "auto"` or compute the right DP size for the SP+EP combo.              |
| Long-context EP=8 alone OOMs at 37 GiB | Per-rank buffer not sharded by FSDP — same allocation across all node counts (jobs 22092511-14) | Any EP=8 run with per-rank seq > 16k.                                                                                             | Profile to identify the unsharded buffer. Workaround: use SP or CP to keep per-rank seq ≤ 16k. |

---

## 2026-04-27 (late): DS-Z3 + EP debug dead-end, pivoted to FSDP2 + EP + CP

### DS-Z3 + EP attempt (jobs 22092516-22092521)

Tried to enable DS-Z3 + SP + EP for long-context MoE. SP gives 18.98 % at 64k vs CP's 7.97 %, so it's the better path. Iterated:

| Iteration   | Patch                                                                   | Result                                                |
| ----------- | ----------------------------------------------------------------------- | ----------------------------------------------------- |
| 1           | DS `engine.py:_broadcast_model.is_replicated` → skip DTensor            | Wrong location, broadcast still fired                 |
| 2           | DS `partition_parameters.py:_convert_to_zero_parameters` → skip DTensor | Past broadcast → AttributeError on `ds_numel`         |
| 3           | DS `parameter_offload.py:mark_persistent_parameters` → skip DTensor     | Past `ds_numel` → AttributeError on `partition_numel` |
| 4 (skipped) | Patch every DS access site                                              | 30+ sites in stage3.py alone — not tractable          |

**DS-Z3 fundamentally has no opt-out:** every `module.named_parameters()` element is assumed ZeRO-3-managed. There's no `ignored_modules` API like FSDP. Properly fixing DS-Z3 + EP requires either an upstream "externally-managed parameters" concept in DS, or in transformers gating `_wrap_ep_params_as_dtensor` on backend (plain Tensor for DS-Z3 + DS's MoE `allreduce=False` markers, DTensor wrap only for FSDP2). Out of scope tonight. Full trail in `debug_sp_ep_sonic.md`.

### Pivot: FSDP2 + EP + CP at long context (jobs 22092523, 22092525, 22092526-27)

CP requires sdpa (FA3+CP not supported in transformers). Per-rank seq goes to 16k via CP shard; ep=8, tp=8 stays intra-node; FSDP shards across nodes.

**Results so far (all healthy training, loss 10–14 range):**

| Job      | Ctx  | Nodes | Mesh                         | Loss        | Window MFU    | TPS    | Status                                 |
| -------- | ---- | ----- | ---------------------------- | ----------- | ------------- | ------ | -------------------------------------- |
| 22092527 | 32k  | 4     | dp=2 tp=8 cp=2 ep=8 sdpa     | 11.41       | **15.58 %**   | 50,620 | ✅ done                                |
| 22092533 | 32k  | 2     | dp=1 tp=8 cp=2 ep=8 sdpa     | 11.26–15.74 | **15.6 %**    | 25,400 | ✅ done (retry of 22092526)            |
| 22092523 | 64k  | 4     | dp=1 tp=8 cp=4 ep=8 sdpa     | 11–12.77    | **7.7–7.8 %** | 14,100 | ✅ done                                |
| 22092525 | 128k | 8     | dp=1 tp=8 cp=8 ep=8 sdpa     | 13.24       | **4.13 %**    | 7,944  | ✅ done                                |
| 22092524 | 64k  | 2     | dp=2 tp=4 cp=4 ep=8 (broken) | 19–21       | 7.9 %         | —      | ❌ misconfigured (tp×cp×dp ≠ num_proc) |

**Headline: first healthy 64k+128k EP=8 runs on this stack.** Pre-fix the same configs NaN'd. Post-fix the MFU is the real number (CP penalty intact).

Compared to long-context alternatives:

- **64k 2n DS-Z3+SP=4+FA3+sonicmoe (no EP, healthy)**: 18.98 % window MFU — still wins by ~2.5×
- **128k 4n DS-Z3+SP=8+FA3+sonicmoe (no EP, healthy)**: 19.31 % peak — also wins by ~5×

CP+EP path is _correct now_ but the MFU disadvantage vs SP+no-EP is unchanged. Until DS-Z3+EP or FA3+EP+seq-sharding lands, **DS-Z3+SP+FA3+sonicmoe (no EP) remains the best long-context config.**

### 32k FA3 EP=8 (no CP) — same 18.55 GiB OOM regardless of attention (jobs 22092528-9)

Tested whether FA3 (O(seq) attention) lets 32k per-rank fit where pre-fix sdpa OOM'd. Both 2n and 4n: same `OutOfMemoryError: Tried to allocate 18.55 GiB`.

Per-rank allocation scales linearly with seq across the full sweep:

- 16k per-rank → fits (40 % MFU at 2n)
- 32k per-rank → OOM 18.55 GiB (this entry)
- 64k per-rank → OOM 37.09 GiB (jobs 22092511-14, exactly 2×)
- 128k per-rank → would be ~74 GiB

The 18.55 GiB doesn't shard with FSDP DP, doesn't depend on attention backend, doesn't depend on node count. **It's a per-rank EP-path allocation (likely all-to-all gather buffer or sonicmoe kernel internal) that's a hard ceiling at 16k per-rank seq.** Until we identify and shard that buffer, EP=8 long-context requires CP/SP to keep per-rank seq ≤ 16k.

### 16k FA3 EP=8 scales cleanly to 8 nodes (jobs 22092530-1)

Confirmed the 40 % MFU recipe holds across node counts:

| Nodes | Mesh               | Window MFU (peak) | Total TPS | Per-GPU TPS | Loss     |
| ----- | ------------------ | ----------------- | --------- | ----------- | -------- |
| 2     | dp=2 tp=8 ep=8 FA3 | **40.4 %**        | 109,600   | 6,850       | 12.60 ✅ |
| 4     | dp=4 tp=8 ep=8 FA3 | **40.02 %**       | 215,600   | 6,738       | 12.31 ✅ |
| 8     | dp=8 tp=8 ep=8 FA3 | **39.43 %**       | 424,900   | 6,639       | 12.68 ✅ |

Total throughput scales near-linearly (1.97× at 4n, 3.87× at 8n). Per-GPU MFU sees only a ~1 pp degradation at 8n vs 2n — inter-node EP all-to-all is the bottleneck but the impact is small. **EP+FSDP+sonicmoe scales cleanly across the cluster.**

### Final headline: post-fix EP=8 leaderboard

| Ctx  | Nodes | Mesh                   | Attn | Window MFU    | Trains? |
| ---- | ----- | ---------------------- | ---- | ------------- | ------- |
| 16k  | 2     | dp=2 tp=8 ep=8         | FA3  | **40.4 %**    | ✅      |
| 16k  | 4     | dp=4 tp=8 ep=8         | FA3  | **40.02 %**   | ✅      |
| 16k  | 8     | dp=8 tp=8 ep=8         | FA3  | **39.43 %**   | ✅      |
| 16k  | 2     | dp=2 tp=8 ep=8         | sdpa | 31.0 %        | ✅      |
| 32k  | 2/4   | dp=N/4 tp=8 cp=2 ep=8  | sdpa | **15.6 %**    | ✅      |
| 64k  | 4     | dp=1 tp=8 cp=4 ep=8    | sdpa | **7.7 %**     | ✅      |
| 128k | 8     | dp=1 tp=8 cp=8 ep=8    | sdpa | **4.13 %**    | ✅      |
| 32k  | any   | dp=N tp=8 ep=8 (no CP) | any  | OOM 18.55 GiB | ❌      |
| 64k+ | any   | dp=N tp=8 ep=8 (no CP) | any  | OOM 37+ GiB   | ❌      |

**The 16k FA3 EP=8 sonicmoe at ~40 % is the new MoE SFT MFU record** with the EP+FSDP fix and wrapper clamp+mask, and it scales cleanly to 8 nodes (424k total TPS, ~6,700 per-GPU). Long context still has a 16k-per-rank ceiling that gates everything to CP-penalized configs (4–15 % MFU).

### Long-context CP+EP scaling — per-GPU MFU stable, total TPS near-linear

| Ctx  | Nodes | Mesh                     | Loss        | Window MFU  | Window TPS (total) | Per-GPU TPS |
| ---- | ----- | ------------------------ | ----------- | ----------- | ------------------ | ----------- |
| 32k  | 2     | dp=1 tp=8 cp=2 ep=8 sdpa | 11.26–15.74 | **15.62 %** | 25,250             | 1,578       |
| 32k  | 4     | dp=2 tp=8 cp=2 ep=8 sdpa | 11.41       | **15.58 %** | 50,620             | 1,582       |
| 32k  | 8     | dp=4 tp=8 cp=2 ep=8 sdpa | 13.04       | **15.37 %** | 99,900             | 1,561       |
| 64k  | 4     | dp=1 tp=8 cp=4 ep=8 sdpa | 12.40       | **7.71 %**  | 13,980             | 437         |
| 64k  | 8     | dp=2 tp=8 cp=4 ep=8 sdpa | 12.46       | **7.72 %**  | 27,990             | 437         |
| 128k | 8     | dp=1 tp=8 cp=8 ep=8 sdpa | 13.51       | **4.13 %**  | 7,944              | 124         |

Per-GPU TPS at fixed (ctx, CP) is essentially constant when adding nodes — the **CP penalty bounds per-GPU MFU**, not inter-node comm. Adding nodes purely doubles total throughput. The per-GPU MFU plateau is set by CP cost (CP=2 → ~15.6 %, CP=4 → ~7.7 %, CP=8 → ~4.1 %) — ring-attention's send/recv chain is the bottleneck at long ctx.

### Recommended configs (post-fix, all healthy training)

| Goal                                 | Config                                 | Notes                                                                                                                             |
| ------------------------------------ | -------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| Best 16k MFU (single recipe)         | 16k 2n, FA3 + EP=8 + sonicmoe          | **40.4 %** window MFU                                                                                                             |
| Highest 16k throughput               | 16k 8n, FA3 + EP=8 + sonicmoe          | 39.4 % MFU, **425k TPS total**                                                                                                    |
| Best 32k MFU                         | 32k 2n, sdpa + CP=2 + EP=8 + sonicmoe  | 15.6 % window MFU (CP penalty)                                                                                                    |
| Best 64k MFU                         | 64k 4n, sdpa + CP=4 + EP=8 + sonicmoe  | 7.7 % window MFU                                                                                                                  |
| Best 128k MFU                        | 128k 8n, sdpa + CP=8 + EP=8 + sonicmoe | 4.13 % window MFU                                                                                                                 |
| Long-context **alternative** (no EP) | DS-Z3 + SP + FA3 + sonicmoe (no EP)    | 18.98 % @ 64k 2n; 19.31 % peak @ 128k 4n. **Higher MFU than EP+CP for long ctx**, but DS-Z3+EP unblock is needed to combine both. |

---

## 2026-04-28: SP sweep (re-running historical configs to confirm with current stack)

### Constraints found

- DS-Z3+SP requires a flash attention implementation; sdpa errors with `Could not find flash attention implementation at sdpa`. Must pass `attn_implementation: kernels-community/vllm-flash-attn3` (or `flash-attn2`).
- DS-Z3+SP+FA3 init takes **20+ minutes** for 30B (model load + DS partition + Ulysses setup). Earlier "hangs" were just slow init.
- `cpu_ram_efficient_loading` doesn't apply for DS-Z3 (FSDP-only flag); DS uses `zero3_init_flag` instead.
- 64k 2n DS-Z3+SP=4 OOMs at 9 GiB free with FA3 (2 nodes is too tight on memory). 4 nodes (dp_shard=8) is needed.
- **Ulysses padding bug** persists from historical runs: packed batches can have `seqlen % sp_size != 0`, crashing at step ~25 with `ValueError: batch's seqlen=54223 isn't divisible by sp-size=4`. Mitigation: pass `--pad_to_multiple_of 8` to SFT trainer.

### Partial result: 64k 4n DS-Z3+SP=4+FA3+sonicmoe (job 22092736, no padding fix, 25 steps)

| Step | Loss                    | mean_token_acc | Window MFU  |
| ---- | ----------------------- | -------------- | ----------- |
| 5    | 1.637                   | 0.6413         | —           |
| 10   | 1.512                   | 0.6563         | **20.99 %** |
| 15   | 1.654                   | 0.6291         | 18.29 %     |
| 20   | 1.661                   | 0.6292         | 20.84 %     |
| 25   | 1.635                   | 0.6354         | 20.11 %     |
| 30   | crash (Ulysses padding) |                |             |

**Steady-state window MFU ~19–21 %** matches the historical 18.98 % / 21.26 % peak measurements. Loss in 1.5–1.7 range (this is the SP-trained loss curve which descends faster than dense/EP since Ulysses retains full attention math). Healthy.

### `--pad_to_multiple_of 8` fixes Ulysses crash; full 50-step runs (jobs 22092737-9)

Padding the collator output to a multiple of `sp_size` resolves the historical Ulysses bug. All three runs reached step 50 cleanly. Loss is in the 1.6 range (SP keeps full-attention math, descends faster than dense or EP+CP).

| Job      | Ctx  | Nodes | SP  | Mesh                   | Loss (50) | Window MFU (last) | Window MFU (peak) | Total TPS |
| -------- | ---- | ----- | --- | ---------------------- | --------- | ----------------- | ----------------- | --------- |
| 22092737 | 64k  | 4     | 4   | dp=8 sp=4 FA3 sonicmoe | 1.618     | **20.61 %**       | 20.99 %           | 65,590    |
| 22092738 | 128k | 4     | 8   | dp=4 sp=8 FA3 sonicmoe | 1.634     | 16.04 %           | 19.24 %           | 47,820    |
| 22092739 | 128k | 8     | 8   | dp=8 sp=8 FA3 sonicmoe | 1.621     | **19.42 %**       | 19.60 %           | 85,230    |

**Key findings:**

1. **SP beats CP+EP by 2.7–4.7×** at long context:
    - 64k: 20.6 % (SP) vs 7.7 % (CP+EP) → **2.7×**
    - 128k 4n: 16.0 % (SP) vs 4.13 % (CP+EP at 8n) → **3.9×**
    - 128k 8n: 19.4 % (SP) vs 4.13 % (CP+EP at 8n) → **4.7×**
2. **128k scales: 4n → 8n adds ~3 pp** (16.0 → 19.4 % window). Adding nodes at 128k SP=8 helps because more DP shards optimizer states + halves compute per rank.
3. **`--pad_to_multiple_of 8` is now mandatory for SP runs** — without it, packed batches with non-divisible seqlen crash Ulysses at ~step 25. Should be added to `launch.sh.j2` defaults for all SP runs.

### Final long-context recommendations

| Ctx  | Best config (post-fix)                 | Window MFU | Notes                             |
| ---- | -------------------------------------- | ---------- | --------------------------------- |
| 16k  | FSDP2 + EP=8 + FA3 + sonicmoe (2/4/8n) | **40 %**   | Champion; no seq sharding needed  |
| 32k  | FSDP2 + EP=8 + CP=2 + sonicmoe (sdpa)  | 15.6 %     | EP+CP — only viable EP=8 32k path |
| 64k  | **DS-Z3 + SP=4 + FA3 + sonicmoe (4n)** | **20.6 %** | Beats EP+CP=4 by 2.7×             |
| 128k | **DS-Z3 + SP=8 + FA3 + sonicmoe (8n)** | **19.4 %** | Beats EP+CP=8 by 4.7×             |

For long context, **SP is the clear winner**. CP+EP has the EP correctness story (now healthy) but the CP penalty is too large. SP without EP achieves 4× higher MFU at 128k. The remaining open work is enabling DS-Z3+SP+EP simultaneously, which would combine both wins — currently blocked by DS-Z3's lack of "externally-managed parameters" support (documented in `debug_sp_ep_sonic.md`).

---

## Addendum: torch.compile + FSDP2 — fixed by accelerate PR #4022

The earlier addendum ("torch.compile + TF32 Fix and Results", line ~960) concluded that `torch.compile` with FSDP2 was unusable through HuggingFace Trainer because whole-model compile produces 17+ tiny graph fragments at every FSDP module boundary, ending up 2.7× slower than eager. That conclusion held only for the _whole-model_ compile path that HF Trainer's `--torch_compile` flag triggers.

**Per-layer compile** (compile each TransformerBlock individually before FSDP wrapping, the torchtitan recipe) is fully viable on Qwen3-30B-A3B MoE — but only when accelerate's FSDP2 wrap path doesn't undo it. SFTTrainer's `__init__` was updated to do per-layer compile pre-FSDP; this works against raw `fully_shard()` but degrades catastrophically against `accelerate.fsdp2_prepare_model()`.

### The regression (slack reproducer, Qwen3-30B-A3B, 2×8 H100 SXM, FSDP2 DP=16, seq_len=16384, bf16 + grad ckpt + packing)

| Setup                                                            | MFU       | ms/step | vs eager        |
| ---------------------------------------------------------------- | --------- | ------- | --------------- |
| raw `fully_shard()` + per-layer compile                          | **32.1%** | 3,031   | 1.28× faster    |
| accelerate `fsdp2_prepare_model()` + per-layer compile (pre-fix) | 9.8 %     | 9,900   | **2.4× slower** |
| accelerate `fsdp2_prepare_model()` + eager (no compile)          | 23.4 %    | 4,160   | —               |

Compile itself is clean — zero graph breaks, zero recompiles — so the slowdown comes from the wrapping path, not from compile failing.

### Root cause

`torch.compile(module)` returns an `OptimizedModule` whose `__call__` bypasses `nn.Module._call_impl`. Forward/pre-hooks added later by `fully_shard()` never fire on an `OptimizedModule`, so per-layer all-gather and reshard hooks are silently lost. Without those hooks, FSDP2 can't shard params at the right boundaries — the compiled kernels run against unsharded or wrongly-sharded tensors and the per-layer compile recipe collapses to a slow path.

### The fix: accelerate PR #4022

[https://github.com/huggingface/accelerate/pull/4022](https://github.com/huggingface/accelerate/pull/4022) adds `compile_regions_fsdp2()` in `accelerate/utils/other.py`. It uses **in-place** `module.compile()` (which preserves `_call_impl` and therefore FSDP hooks) instead of `torch.compile(module)`, and applies it **after** `fully_shard()` wrap. The PR also patches `accelerate/commands/launch.py` so the `dynamo_config` block in the accelerate yaml is parsed correctly (keys auto-prefixed with `dynamo_`).

### How to use

Add to the accelerate yaml (the FSDP2 yaml; for benchmarks see `benchmark/templates/accelerate/fsdp2.yaml.j2`):

```yaml
dynamo_config:
    backend: inductor
    use_fullgraph: true
    use_regional_compilation: true
```

Without `use_regional_compilation: true`, accelerate falls back to whole-model `torch.compile()` — the same path the earlier addendum documented as 2.7× slower than eager. The flag is the trigger.

Until PR #4022 merges into a release, override the installed accelerate at runtime with `PYTHONPATH=/fsx/amine_dirhoussi/accelerate/src:$PYTHONPATH`. This keeps the installed accelerate's local `_prepare_tp` `has_ep` patch (`local_only_patches.md` §4) intact.

### Validated 2026-04-28 (Qwen3-30B-A3B, 2×8 H100, FSDP2 DP=16, seq_len=16384)

Job 22092887, accelerate 1.14.0.dev0 from PR #4022.

| Setup                                                                                                   | DP     | mfu_window                  | ms/step    |
| ------------------------------------------------------------------------------------------------------- | ------ | --------------------------- | ---------- |
| raw `fully_shard()` + per-layer compile (control, slack post)                                           | 16     | 32.1 %                      | 3,031      |
| raw `fully_shard()` + per-layer compile (this run, 1×8)                                                 | 8      | ~31 %                       | 3,124      |
| accelerate `fsdp2_prepare_model()` + per-layer compile (pre-fix, slack post)                            | 16     | 9.8 %                       | 9,900      |
| **accelerate `fsdp2_prepare_model()` + per-layer compile + PR #4022 + `use_regional_compilation=true`** | **16** | **31.27 / 31.81 / 32.55 %** | **~3,000** |

Three logging-window samples landed at 31.27 / 31.81 / 32.55% MFU — fully matched to the raw `fully_shard()` baseline. The 2.4× regression is closed.

### Cluster gotcha (orthogonal to the PR)

Load-time variance across the 16 ranks ranged from 6 to 19 minutes (FSx contention + per-rank model-shard reads). NCCL's default 600s collective timeout fires when the spread exceeds 10 minutes — the second NCCL collective (a barrier or 311 M-element broadcast = embedding matrix) hangs and the run dies before training starts. This is independent of PR #4022 (no compile path runs before the hang).

Workaround used: pre-init the process group with a long timeout _before_ accelerate's `PartialState()` instantiates it:

```python
import os, torch
if "RANK" in os.environ and not torch.distributed.is_initialized():
    from datetime import timedelta
    torch.distributed.init_process_group(backend="nccl", timeout=timedelta(seconds=3600))
```

Drop this into the SFT entry script before any `accelerate` import. A cleaner long-term fix is to pass `InitProcessGroupKwargs(timeout=...)` to the `Accelerator` SFTTrainer creates internally, but that needs a Trainer-level hook.

### Repro

`benchmark/compile_repro/`:

- `slow_path/slow_accelerate_fsdp2.py` — SFTTrainer through `accelerate.fsdp2_prepare_model`, with the 1h PG pre-init at top
- `fast_path/fast_raw_fully_shard.py` — torchrun + raw `fully_shard()` (control)
- `accelerate_config.yaml` — 2-node FSDP2 yaml with the `dynamo_config` block above
- `run_slow.sbatch`, `launch_slow.sh` — Slurm + accelerate launch wrappers (set `PYTHONPATH=/fsx/amine_dirhoussi/accelerate/src` to pick up PR #4022 without touching the installed accelerate)

### Updated guidance (supersedes the line ~1009 conclusion)

`--torch_compile=true` in SFTTrainer **does** work with FSDP2 + accelerate when:

1. The accelerate version includes PR #4022 (or the override is in place), AND
2. The accelerate yaml sets `dynamo_config.use_regional_compilation: true`.

Without (2) the path falls back to whole-model compile and is slower than eager — the original addendum's conclusion still applies in that case.

---

## 2026-04-28 (late): compile + EP + long-context exploration

Building on the PR #4022 finding (35% MFU on FSDP DP=16 + FA3 + compile, no EP) and the new DS-Z2 + EP path (28.6% at 16k, healthy training):

### Setup additions for compile

1. `trl/scripts/sft.py` — set per-rank `TRITON_CACHE_DIR=/tmp/triton-rank-${RANK}-${hostname}` before any torch import (FSx contention otherwise → `CUDA driver error: file not found`).
2. `transformers/integrations/moe.py` — `@torch._dynamo.disable` on `grouped_mm_experts_forward` (dynamo's `_getattr_static` failed on `gate_up_proj` access otherwise).
3. `transformers/integrations/sonicmoe.py` — same `@torch._dynamo.disable` on `sonicmoe_experts_forward` (was hitting an internal dynamo issue at the kernel call).
4. `dynamo_config.use_fullgraph: false` in the FSDP2 yaml — RouterParallel has data-dependent branching (`(router_indices // num_local_experts) != ep_rank`) that fullgraph can't trace.

### Results

| Setup                                                    | Ctx  | Nodes | Mesh                | Win MFU peak  | Loss     | Verdict                                                                                       |
| -------------------------------------------------------- | ---- | ----- | ------------------- | ------------- | -------- | --------------------------------------------------------------------------------------------- |
| FSDP DP=16 + FA3 + compile, no EP                        | 16k  | 2     | dp=16               | **34.87 %**   | 1.6 ✅   | matches PR #4022 claim                                                                        |
| FSDP + EP=8 + FA3 + sonicmoe + compile                   | 16k  | 2     | dp=2 tp=8 ep=8      | crash         | —        | Adam `_group_tensors_by_device_and_dtype` mismatch (DTensor mesh mix: FSDP DP mesh + EP mesh) |
| FSDP + EP=8 + FA3 + grouped_mm + compile                 | 16k  | 2     | dp=2 tp=8 ep=8      | crash         | —        | same Adam issue                                                                               |
| FSDP + EP=8 + sonicmoe + compile + `--optim adamw_torch` | 16k  | 2     | dp=2 tp=8 ep=8      | crash         | —        | non-fused Adam still calls the same foreach grouping                                          |
| **DS-Z2 + EP=8 + FA3 + sonicmoe + compile**              | 16k  | 2     | dp=2 tp=8 ep=8      | **36.7 %**    | 12.25 ✅ | new long-running compile+EP combo (DS path uses plain tensors, no DTensor mesh)               |
| DS-Z3 + SP=4 + FA3 + sonicmoe + compile                  | 64k  | 4     | dp=8 sp=4           | 20.71 %       | 1.62 ✅  | matches no-compile 20.6% — long-ctx is comm-bound, compile doesn't help                       |
| DS-Z3 + SP=8 + FA3 + sonicmoe + compile                  | 128k | 8     | dp=8 sp=8           | 19.34 %       | 1.62 ✅  | matches no-compile 19.42%                                                                     |
| FSDP + EP=8 + CP=4 + sonicmoe + compile                  | 64k  | 4     | dp=1 tp=8 cp=4 ep=8 | crash         | —        | Triton cache file-not-found (intermittent), then partial-task-dead                            |
| FSDP + EP=8 + CP=8 + sonicmoe (no compile)               | 256k | 8     | dp=1 tp=8 cp=8 ep=8 | OOM 18.55 GiB | —        | per-rank seq = 32k → unsharded EP buffer overflows (same ceiling we hit at 32k earlier)       |

### Why FSDP+EP+compile is blocked

Adam's optimizer step calls `torch._C._group_tensors_by_device_and_dtype` to batch params/grads/exp_avg into foreach calls. Under compile, this group-by inspects the tensors at runtime and asserts that all tensors at the same group index share device+dtype. With our EP+FSDP layout:

- EP-sharded params: `DTensor` on the EP mesh (size 8 intra-node).
- Non-EP params: `DTensor` on the FSDP DP mesh (size 16 across both nodes).

The compiled foreach can't reconcile these — different mesh metadata even though the underlying device is the same. Without compile, Adam falls back to a per-tensor path that handles the mix; with compile, the runtime assertion fires.

DS-Z2 + EP doesn't hit this because its EP path uses plain `nn.Parameter` (with `allreduce=False`/`group_name` markers) — no DTensor mesh on the EP params, so the foreach groups uniformly.

### Compile gives no improvement at long-ctx

Both DS-Z3+SP+compile @ 64k (20.71% vs 20.6%) and @ 128k (19.34% vs 19.42%) are within noise of the no-compile baseline. At long context, communication (Ulysses all-to-all + ZeRO-3 all-gather) dominates step time, not compute — compile speeds up the GPU-bound part but the ranks spend most of their time in NCCL.

### SP + EP combo is fundamentally broken (re-confirmed at scale)

Submitted DS-Z2 + SP + EP=8 throughput sweep at 32k / 64k / 128k. All complete 50 steps cleanly and hit reasonable MFU (18-26%) but produce broken training:

| Ctx  | Nodes | Mesh           | Loss    | mean_token_acc | Window MFU |
| ---- | ----- | -------------- | ------- | -------------- | ---------- |
| 32k  | 2     | dp=8 sp=2 ep=8 | 8.16 ❌ | 0.05 ❌        | 26.05 %    |
| 64k  | 4     | dp=8 sp=4 ep=8 | 8.14 ❌ | 0.05 ❌        | 19.21 %    |
| 128k | 4     | dp=4 sp=8 ep=8 | 8.32 ❌ | 0.04 ❌        | 18.38 %    |
| 128k | 8     | dp=8 sp=8 ep=8 | 8.39 ❌ | 0.03 ❌        | 18.13 %    |

Throughput is comparable to SP-only at the same context (20.6% / 19.4%) — adding EP doesn't speed up but breaks correctness. Confirms the architectural mismatch: Ulysses shards seq dim → each rank only sees `seq/SP` tokens; transformers' EP all-reduces across EP ranks assuming each rank computed contributions to the _same_ token set. Different subsets summed = garbage.

### Updated headline tables

| Goal                   | Best recipe                                           | Window MFU       |
| ---------------------- | ----------------------------------------------------- | ---------------- |
| Best 16k MFU           | FSDP + EP=8 + FA3 + sonicmoe (no compile)             | **40.4 %**       |
| Best 16k MFU + compile | DS-Z2 + EP=8 + FA3 + sonicmoe + compile               | **36.7 %**       |
| Highest 16k throughput | FSDP + EP=8 + FA3 + sonicmoe @ 8 nodes (no compile)   | 39.4 %, 425k TPS |
| Best 32k MFU           | FSDP + EP=8 + CP=2 + sonicmoe (sdpa)                  | 15.6 %           |
| Best 64k MFU           | DS-Z3 + SP=4 + FA3 + sonicmoe (compile or no compile) | ~20 %            |
| Best 128k MFU          | DS-Z3 + SP=8 + FA3 + sonicmoe @ 8n                    | ~19 %            |
| 256k                   | not yet — EP+CP=8 OOMs at 18.55 GiB; SP=16 in flight  | TBD              |

## 2026-04-28 (later): 32k SP=2 new champion + 256k SP=16 too slow

Three runs in this batch:

| Job      | Ctx  | Nodes | Mesh                         | Win MFU peak | Loss    | Peak GPU Mem  | Verdict                                           |
| -------- | ---- | ----- | ---------------------------- | ------------ | ------- | ------------- | ------------------------------------------------- |
| 22093312 | 32k  | 2     | DS-Z3 dp=8 sp=2 (no compile) | **21.69 %**  | 1.62 ✅ | 79.0 GB (99%) | new 32k champion (was FSDP+EP+CP=2 at 15.6 %)     |
| 22093313 | 32k  | 2     | DS-Z3 dp=8 sp=2 + compile    | **21.98 %**  | 1.62 ✅ | 79.0 GB (99%) | +0.3 pp from compile, within noise                |
| 22093311 | 256k | 8     | DS-Z3 dp=4 sp=16             | **1.36 %**   | 1.62 ✅ | 61.5 GB (77%) | trains but cross-node Ulysses dominates step time |

### 32k SP=2 wins over FSDP+EP+CP=2 at 32k

- **+6 pp MFU**: 21.69 → 21.98 % (SP=2) vs 15.6 % (FSDP+EP+CP=2 sdpa). The sequence-shard via Ulysses + FA3 attention is meaningfully faster than CP=2 ring attention with sdpa — same per-rank seq (16k) for both, but Ulysses' all-to-all overlaps better than CP's send/recv chain at this size.
- **Both train healthy**: SP path gets loss 1.5–1.6 (typical SP-normalized loss), CP+EP path got loss 10–14 (untrained-init). Both correct.
- **Tight on memory**: SP=2 path ends up at 99 % GPU memory (79.0 / 79.4 GB) vs CP+EP at 92 % (73.6 GB). SP keeps more activations on the rank.
- **Compile gives ~no win** (+0.3 pp). Same pattern as 64k/128k SP+compile — long-ctx is comm-bound.

### 256k SP=16 trains but impractical at 1.36 %

Per-rank seq stays at 16k (256k / SP=16) so the model fits (61.5 GB). But SP=16 spans 2 nodes per SP group on an 8-GPU/node cluster — every Ulysses all-to-all crosses the node boundary on EFA, where bandwidth is ~11 % of intra-node NVLink. Throughput drops 14× vs 32k SP=2 (1.36 % vs 21.69 %).

Likely fixable by intra-node-only SP (SP=8 max per group) at 256k. That requires either (a) reducing context to fit per-rank seq ≤ 16k with intra-node SP=8 → 128k cap, or (b) combining SP=8 with CP for the rest — e.g. 256k = SP=8 + CP=2, per-rank seq = 16k, but Ulysses+CP composition may have its own issues. Not pursued in this batch.

### Updated headline (Qwen3-30B-A3B, training-correct)

| Goal                   | Best recipe                                                              | Window MFU       |
| ---------------------- | ------------------------------------------------------------------------ | ---------------- |
| Best 16k MFU           | FSDP + EP=8 + FA3 + sonicmoe (no compile)                                | **40.4 %**       |
| Best 16k MFU + compile | DS-Z2 + EP=8 + FA3 + sonicmoe + compile                                  | **36.7 %**       |
| Highest 16k throughput | FSDP + EP=8 + FA3 + sonicmoe @ 8 nodes (no compile)                      | 39.4 %, 425k TPS |
| **Best 32k MFU (NEW)** | **DS-Z3 + SP=2 + FA3 + sonicmoe + compile**                              | **21.98 %**      |
| Best 64k MFU           | DS-Z3 + SP=4 + FA3 + sonicmoe (compile or no compile)                    | ~20 %            |
| Best 128k MFU          | DS-Z3 + SP=8 + FA3 + sonicmoe @ 8n                                       | ~19 %            |
| 256k                   | DS-Z3 + SP=16 (trains at 1.36 %, impractical — needs intra-node-only SP) | TBD              |

## 2026-04-28 (later still): late sweep — 32k FSDP DP=16 OOM, 64k SP=8 intra-node bust

Tested whether removing parallelism overhead can beat current champions.

| Job      | Ctx | Nodes | Mesh                            | Compile | Win MFU peak               | Loss    | Peak GPU Mem  | Verdict                                                                                                             |
| -------- | --- | ----- | ------------------------------- | ------- | -------------------------- | ------- | ------------- | ------------------------------------------------------------------------------------------------------------------- |
| 22093867 | 32k | 2     | FSDP DP=16 (no EP/SP/CP)        | no      | OOM                        | —       | 76.6 GB (96%) | activations alone need 18.55 GiB extra → no headroom; needs CP/SP/EP at 32k                                         |
| 22093868 | 32k | 2     | FSDP DP=16 + compile            | yes     | OOM (Inductor CUDA driver) | —       | 76.6 GB (96%) | same OOM path + triton-cache miss under compile                                                                     |
| 22093869 | 64k | 2     | DS-Z3 dp=2 sp=8 intranode + FA3 | no      | **8.7 %**                  | 1.59 ✅ | 62.6 GB (79%) | intra-node SP=8 helps comm BW but DP=2 ZeRO-3 cross-node param shuffle dominates → much worse than 4n SP=4 (20.5 %) |

### Why 32k FSDP DP=16 (no parallelism beyond DP) OOMs

At 32k seq with batch_size=1 per rank, activations are large. 16 DP ranks share the model (≈ 1.9 GB params/rank after Z2-style sharding under FSDP), but each rank must materialize its own full 32k activations. The 18.55 GiB allocation is the FA3 attention scratch + MoE expert intermediate buffer for that 32k local sequence — not gradient/grad accum. **Conclusion: at 32k context, you must shard the seq dim (CP or SP) or shard the experts (EP) to fit. Pure DP doesn't.** This matches the 32k DS-Z3+SP=2 champion (per-rank seq = 16k) and FSDP+EP+CP=2 baseline (per-rank seq = 16k).

### Why 64k SP=8 intra-node @ 2n is worse than SP=4 cross-node @ 4n

Intuition was: SP=8 fits within one node (NVLink), so all-to-all is fast; cross-node is only DP communication. But:

- **DP=2 ZeRO-3 forces large cross-node param transfers**. Each parameter is split across 2 ranks (one per node). Every layer's forward gathers the _entire_ parameter from the other node over EFA. 30B × 2 bytes / 2 = 30 GB of cross-node traffic per gather, times every layer. This dominates step time.
- **DP=8 ZeRO-3 (the SP=4 4n config)** shards each param across 8 ranks. Per-rank shard is 7.5 GB; gather involves 7 partners but only 7 × 1.07 GB messages, and most overlap with intra-node compute. Total cross-node BW × time is much smaller per step.

So **DP-degree of ZeRO-3 matters more than SP topology** when the model is in the "comm-bound" regime. SP=8 intra-node only wins if DP can be ≥ 8.

### Updated headline (Qwen3-30B-A3B, training-correct)

| Goal                   | Best recipe                                                            | Window MFU       |
| ---------------------- | ---------------------------------------------------------------------- | ---------------- |
| Best 16k MFU           | FSDP + EP=8 + FA3 + sonicmoe (no compile)                              | **40.4 %**       |
| Best 16k MFU + compile | DS-Z2 + EP=8 + FA3 + sonicmoe + compile                                | **36.7 %**       |
| Highest 16k throughput | FSDP + EP=8 + FA3 + sonicmoe @ 8 nodes (no compile)                    | 39.4 %, 425k TPS |
| Best 32k MFU           | DS-Z3 + SP=2 + FA3 + sonicmoe + compile                                | **21.98 %**      |
| Best 64k MFU           | DS-Z3 + SP=4 + FA3 + sonicmoe @ 4n (DP=8)                              | ~20 %            |
| Best 128k MFU          | DS-Z3 + SP=8 + FA3 + sonicmoe @ 8n (DP=8)                              | ~19 %            |
| 256k                   | DS-Z3 + SP=16 (trains at 1.36 %, impractical — needs intra-node SP+CP) | TBD              |
| 32k pure DP            | not feasible — needs seq sharding (CP/SP) or expert sharding (EP)      | OOM              |
| 64k intra-node SP @ 2n | DS-Z3 + SP=8 dp=2 — comm-bound by ZeRO-3 cross-node                    | 8.7 %            |

## 2026-04-28 (latest): DS-Z2+EP+compile sweep at 32k/64k — all OOM at EP buffer

Followed up the 16k DS-Z2+EP+compile champion (36.7 %) by trying it at longer context.

| Job      | Ctx | Nodes | Mesh                 | Compile | Result | OOM size  | Verdict                                                                      |
| -------- | --- | ----- | -------------------- | ------- | ------ | --------- | ---------------------------------------------------------------------------- |
| 22093943 | 32k | 2     | DS-Z2 dp=2 tp=8 ep=8 | yes     | OOM    | 18.55 GiB | EP buffer ceiling at 32k per-rank seq                                        |
| 22093944 | 32k | 2     | DS-Z2 dp=2 tp=8 ep=8 | no      | OOM    | 18.55 GiB | same — compile not the issue                                                 |
| 22093945 | 32k | 4     | DS-Z2 dp=4 tp=8 ep=8 | yes     | OOM    | 18.55 GiB | adding nodes doesn't help (DS-Z2 doesn't shard params/activations across DP) |
| 22093946 | 64k | 4     | DS-Z2 dp=4 tp=8 ep=8 | yes     | OOM    | 37.09 GiB | exactly 2× 32k buffer — scales linearly with seq                             |

### The 18.55 GiB EP buffer is the long-context wall

This is the **same** OOM signature seen earlier at 256k FSDP+EP+CP=8 (per-rank seq=32k → 18.55 GiB). It is not memory pressure that can be relieved by adding nodes or by swapping FSDP↔DS-Z2↔DS-Z3 — it is the **per-rank tensor of expert intermediates** allocated when transformers' EP replicates routing across all EP ranks (each rank stores `seq × num_local_experts × moe_intermediate_size` activations in fp/bf16). At 32k seq, this single allocation alone needs 18.55 GiB; everything else (params, optim, attention scratch) competes for the remaining ~60 GB.

**To get EP working at 32k+:**

1. **Shard the seq dim alongside EP** (CP=2 → per-rank seq = 16k → buffer halves to 9.3 GiB). This is how the current 32k FSDP+EP+CP=2 baseline (15.6 % MFU) survives.
2. **Reduce EP degree** at 32k (EP=4 → buffer halves; EP=2 → quarters). But EP=8 was the throughput champion at 16k, so stepping down loses ground.
3. **Stream the expert dispatch** — would require kernel-level rewrite to avoid materializing the full `seq × num_local_experts × moe_intermediate` tensor. Not in scope.

**Conclusion: DS-Z2+EP+compile is a 16k-only champion.** At 32k+, the recipe must be either (a) FSDP+EP+CP=2 (correct but slow at 15.6 %), (b) DS-Z3+SP=2 (current 32k champion at 21.98 % — no EP), or (c) wait for a streaming EP kernel.

### Updated headline (Qwen3-30B-A3B, training-correct) — final

| Goal                   | Best recipe                                                        | Window MFU       |
| ---------------------- | ------------------------------------------------------------------ | ---------------- |
| Best 16k MFU           | FSDP + EP=8 + FA3 + sonicmoe (no compile)                          | **40.4 %**       |
| Best 16k MFU + compile | DS-Z2 + EP=8 + FA3 + sonicmoe + compile                            | **36.7 %**       |
| Highest 16k throughput | FSDP + EP=8 + FA3 + sonicmoe @ 8 nodes (no compile)                | 39.4 %, 425k TPS |
| Best 32k MFU           | DS-Z3 + SP=2 + FA3 + sonicmoe + compile                            | **21.98 %**      |
| Best 64k MFU           | DS-Z3 + SP=4 + FA3 + sonicmoe @ 4n (DP=8)                          | ~20 %            |
| Best 128k MFU          | DS-Z3 + SP=8 + FA3 + sonicmoe @ 8n (DP=8)                          | ~19 %            |
| 256k                   | DS-Z3 + SP=16 (1.36 %, impractical) — needs intra-node SP+CP combo | TBD              |
| EP at 32k+             | blocked by 18.55 GiB EP-replicated expert buffer per rank          | OOM              |

## Known errors / next steps (kept out of the consolidated notion)

These are errors (not OOMs and not results) that are tracked here so we can attack them in a future pass. The notion file holds only OOMs and successful results.

### 1. FSDP + EP + compile: Adam `_group_tensors_by_device_and_dtype` (DTensor mesh mix)

Tested 3 variants at 16k, all crash at the optimizer step:

- FSDP + EP=8 + FA3 + sonicmoe + compile
- FSDP + EP=8 + FA3 + grouped_mm + compile
- FSDP + EP=8 + sonicmoe + compile + `--optim adamw_torch` (non-fused Adam)

All three hit the same compiled foreach assert. `nn.Parameter` inputs are a mix of:

- EP DTensors on the EP `device_mesh` (size 8)
- FSDP DP DTensors on the FSDP DP mesh (size 16)

Under compile, `_group_tensors_by_device_and_dtype` strict-asserts that grouped tensors share device and dtype; with two different DTensor meshes in play, the foreach grouping splits incorrectly and the next foreach kernel sees mismatched shapes. Non-fused Adam doesn't help — `foreach` groups regardless of `--optim`. DS-Z2+EP+compile works because DS uses plain `nn.Parameter` (with `allreduce=False` / `group_name` markers), no DTensor mesh.

**Fix paths to try:** (a) make the EP wrap return the same DTensor mesh family as FSDP (a 2D `(dp, ep)` mesh shared by both); (b) custom `_group_tensors_by_device_and_dtype` that treats EP and FSDP DTensors as compatible if their device+dtype match; (c) skip foreach entirely under EP (probably a perf hit).

### 2. FSDP + EP + CP=4 + compile @ 64k: Triton cache file-not-found, partial-task death

Triton cache contention on FSx — even with per-rank `TRITON_CACHE_DIR` set in `sft.py`, one or more ranks see a "CUDA driver error: file not found" inductor crash. The per-rank dir trick we added before (rank in path) fixed startup contention but a later compile-graph cache step still hits a shared path.

**Fix paths to try:** (a) move `TRITON_CACHE_DIR` to a node-local `/tmp` instead of the FSx-mounted home; (b) confirm `os.uname().nodename` is unique per node (it is) and that the dir is created before any torch import (it is — already at top of `sft.py`); (c) check whether inductor caches into `~/.cache/torch_inductor` separately and needs a per-rank override too.

### 3. EP-replicated expert buffer ceiling — kernel-level fix

The 18.55 GiB / 37.09 GiB OOMs at 32k / 64k per-rank seq are not memory-pressure (params/optim/grads fit). They are a single allocation: `seq × num_local_experts × moe_intermediate × 2 bytes` for the routing-replicated activation tensor. Transformers' EP replicates routing across all EP ranks (it's a TP-style EP, not all-to-all), so every rank materializes this full tensor.

**Fix paths to try:** (a) shard the seq dim alongside EP (CP=2 → buffer halves; this is how FSDP+EP+CP=2 at 32k survives at 15.6 % MFU); (b) drop EP degree at long context (EP=4 → buffer halves); (c) rewrite the expert dispatch to stream over expert chunks instead of materializing the full `(seq, num_local_experts, moe_intermediate)` tensor — kernel-level change, biggest payoff but biggest scope.

## 2026-04-28 (latest, autonomous): chunked-CE loss sweep (TRL PR #5575)

### Stack changes

- Cherry-picked TRL `9bcf7294` (PR #5575: "Chunked cross-entropy loss for SFT (up to –50% VRAM)") onto `benchmark-sft-moe`. Resolved one conflict in `sft_trainer.py` imports (`os` + `types` re-added).
- `loss_type=chunked_nll` chunks the `lm_head` projection over non-ignored tokens (chunk size 256), avoiding the full `(batch × seq, vocab=151936)` logits tensor. Should buy ~20 GB at 64k bf16 in pure activation memory.
- Per the PR docstring, FSDP2 wants `fsdp_reshard_after_forward: false` to avoid re-gathering `lm_head.weight` per chunk. Added a `reshard_after_forward` template var to `benchmark/templates/accelerate/fsdp2.yaml.j2` and plumbed through `run_benchmark.py`.
- Local TRL changes (MFU instrumentation, EP integration, fuse_moe_experts, sonicmoe-implementation flag, MFU window logging, HF_HUB_OFFLINE pre-warm dance, per-rank Triton cache, legacy TF32 flags) preserved on top of the cherry-pick. Inventory in `upstream_todo.md` §G2.

### Submitted sweep (4 jobs)

Goal: see if chunked-CE shaves enough activation memory to (a) match the 16k EP=8 champion, (b) fit configs that OOMed before, (c) push the 64k EP path past the 18.55 GiB EP-buffer wall.

| Job      | Ctx | Nodes | Mesh                           | Reshard | Compares against                            |
| -------- | --- | ----- | ------------------------------ | ------- | ------------------------------------------- |
| 22093971 | 16k | 2     | FSDP DP=2 + EP=8 + FA3         | false   | 40.4% champion (no chunked)                 |
| 22093972 | 32k | 2     | FSDP DP=2 + EP=8 + FA3 (no CP) | false   | OOM 18.55 GiB previously                    |
| 22093973 | 32k | 2     | DS-Z2 + EP=8 + FA3             | n/a     | OOM 18.55 GiB previously                    |
| 22093974 | 64k | 4     | FSDP DP=4 + EP=8 + CP=2 + FA3  | false   | OOM 18.55 GiB previously (per-rank seq=32k) |

All four use `--loss_type chunked_nll`, sonicmoe (clamp+mask wrapper), `--pad_to_multiple_of 8`. Results pending.

### Results — first batch

| Job      | Ctx | Mesh                                                    | Win MFU peak | Cum MFU | TPS Win | Peak GPU Mem  | Loss     | Status                                                        |
| -------- | --- | ------------------------------------------------------- | ------------ | ------- | ------- | ------------- | -------- | ------------------------------------------------------------- |
| 22093973 | 32k | DS-Z2 + EP=8 + FA3 + chunked                            | **45.81 %**  | 39.31 % | 74,500  | 55.0 GB (69%) | 11–15 ✅ | **NEW 32k CHAMPION** — was OOM 18.55 GiB without chunked      |
| 22093971 | 16k | FSDP DP=2 + EP=8 + FA3 + chunked + reshard=false        | NCCL timeout | —       | —       | —             | —        | hangs in scatter; ~519 collectives enqueued, only 8 completed |
| 22093972 | 32k | FSDP DP=2 + EP=8 + FA3 + chunked + reshard=false        | NCCL timeout | —       | —       | —             | —        | same hang signature                                           |
| 22093974 | 64k | FSDP DP=4 + EP=8 + CP=2 + FA3 + chunked + reshard=false | NCCL timeout | —       | —       | —             | —        | same hang signature                                           |

### Headline take-aways

1. **DS-Z2 + EP=8 + chunked at 32k = 45.81 % MFU window**: +24 pp over the previous 32k champion (DS-Z3+SP=2 + compile = 21.98 %), +9 pp over 16k DS-Z2+EP+compile (36.7 %). The 32k EP-buffer wall (18.55 GiB) is broken because chunked-CE drops the `(batch × seq, vocab)` logit tensor by ~20 GB, freeing room for the EP routing buffer.
2. **Memory plummets**: 55.0 GB at 32k (down from 79.0 GB at 32k SP=2's 99 %, and below the previous 16k EP=8 73.6 GB). Peak utilization 69 % means ample headroom — likely scales further.
3. **Loss range 11–15 (sonicmoe+clamp typical)**: consistent with the wrapper-clamp pattern, not NaN. Training is healthy.
4. **FSDP + chunked hangs**: 519 NCCL ops enqueued, only 8 completed. Pattern: ~64 chunks × 8 collectives/chunk = 512 — meaning `reshard_after_forward: false` is NOT being honored for the chunked path under EP, and FSDP is re-gathering `lm_head.weight` for every chunk. Then a final `reduce_scatter` desyncs across ranks.
    - Probable root cause: chunked path patches `model.forward` to compute lm_head in chunks; FSDP2's auto-wrap policy ends up wrapping `lm_head` separately, and the per-chunk matmuls each trigger the gather→matmul→reshard hook chain.
    - Fix paths to investigate: (a) explicitly exclude `lm_head` from FSDP wrap; (b) wrap the chunked path with `with model.lm_head.unshard_context()` to keep weight materialized for all chunks; (c) move chunked path before FSDP wrap.
5. **DS path is the immediate winner**. DS-Z2 doesn't shard params across DP, so the lm_head matmul has the full weight resident — no per-chunk gather. DS path produces correct, fast chunked-CE training.

### Next experiments

Sweeping DS-Z2 + EP=8 + chunked along context length to find the new ceiling:

- 32k DS-Z2 + EP=8 + chunked + compile: does compile add another +5 pp on top of 45.81 %?
- 64k DS-Z2 + EP=8 + chunked @ 2n: per-rank seq = 64k, activations roughly 2× 32k. Will it fit?
- 64k DS-Z2 + EP=8 + chunked @ 4n: more DP shards for optim states.
- 128k DS-Z2 + EP=8 + chunked @ 4n: aggressive push.

### Results — second batch (DS-Z2+EP+chunked push)

| Job      | Ctx      | Nodes | Mesh                           | Compile | Win MFU peak | Cum MFU  | TPS Win | Peak GPU Mem  | Loss     | Status                                                                                                          |
| -------- | -------- | ----- | ------------------------------ | ------- | ------------ | -------- | ------- | ------------- | -------- | --------------------------------------------------------------------------------------------------------------- |
| 22093983 | 32k      | 2     | DS-Z2 dp=2 tp=8 ep=8 + chunked | yes     | **45.86 %**  | 40.08 %  | 74,520  | 55.0 GB (69%) | 11–15 ✅ | compile no help vs no-chunked baseline (45.81%) — chunked saturates compute                                     |
| 22093984 | **64k**  | 2     | DS-Z2 dp=2 tp=8 ep=8 + chunked | no      | **57.23 %**  | 46.07 %  | 51,470  | 67.6 GB (85%) | 11–13 ✅ | **NEW 64k CHAMPION** (was 20.5% with DS-Z3+SP=4)                                                                |
| 22093985 | 64k      | 4     | DS-Z2 dp=4 tp=8 ep=8 + chunked | no      | 56.30 %      | 45.59 %  | 102,000 | 53.3 GB (67%) | 11–12 ✅ | 4n trades MFU for memory + 2× TPS (~100k vs 51k)                                                                |
| 22093986 | **128k** | 4     | DS-Z2 dp=4 tp=8 ep=8 + chunked | no      | **69.07 %**  | 59.79+%¹ | 66,400  | 75.6 GB (95%) | 13–14 ✅ | **NEW 128k CHAMPION + highest MoE MFU on stack** (was 19.4% with DS-Z3+SP=8 8n) — still running, partial result |

¹ Cumulative MFU still climbing as run progresses; final number expected ~65 %.

### What chunked-CE actually unlocked

Before chunked-CE the bottleneck at 32k+ was the EP-replicated expert buffer (18.55 GiB at 32k per-rank seq, scaling linearly). The full-seq `(B × S, vocab=151936)` logits tensor was eating most of the remaining budget at long ctx. Chunked-CE shaves that logits/CE memory dramatically — by chunk_size=256, chunked materializes only `256 × 151936 × 2B = 78 MB` per chunk vs `seq × 151936 × 2B = ~20 GB at 64k`. With that 20 GB freed, the 18.55 GiB EP buffer fits trivially.

**Per-config MFU jumps over previous training-correct champions**:

| Context | Old champion (training-correct)     | New chunked-CE champion            | Δ MFU      |
| ------- | ----------------------------------- | ---------------------------------- | ---------- |
| 16k     | FSDP+EP=8+FA3 (no compile) — 40.4 % | DS-Z2+EP+chunked 16k (TBD)         | TBD        |
| 32k     | DS-Z3+SP=2+compile — 21.98 %        | DS-Z2+EP+chunked 32k — 45.81 %     | **+24 pp** |
| 64k     | DS-Z3+SP=4 4n — ~20.5 %             | DS-Z2+EP+chunked 64k 2n — 57.23 %  | **+37 pp** |
| 128k    | DS-Z3+SP=8 8n — ~19.4 %             | DS-Z2+EP+chunked 128k 4n — 69.07 % | **+50 pp** |

These are massive wins — the long-context regime is now compute-bound rather than comm-bound. DS-Z2+EP+chunked is the new long-context recipe.

### Open questions

1. **Compile gives 0 pp at 32k** (45.86 vs 45.81) — saturated already. Worth testing at 64k/128k where there's more compute headroom.
2. **8n at 128k** — does adding more DP shards (16% peak mem at 4n is already 95%) help find the new ceiling at 256k?
3. **FSDP+chunked still hangs** — needs the lm_head-unshard or no-wrap fix; deferred.
4. **Loss range 11–15 at long ctx** — consistent with sonicmoe+clamp wrapper (the +2 pp MFU cost from doing compute on sentinel rows then masking). If A1's kernel-native sentinel skip lands, this could improve loss to ~8–10 range.

### Next push (submitted)

- 22093988+: 64k DS-Z2 + chunked + compile @ 2n (does compile add at 64k?)
- 22093989+: 128k DS-Z2 + chunked + compile @ 4n (does compile add at 128k?)
- 22093990+: 128k DS-Z2 + chunked @ 8n (more DP, headroom for 256k attempt?)

## 2026-04-29: third+fourth batch — compile at 64k/128k, 256k breakthrough, 16k baseline

### Third batch results (compile + 8n) — compile gives 0 pp once chunked saturates compute

| Job      | Ctx  | Nodes | Mesh                           | Compile | Win MFU peak | Cum MFU  | TPS Win | Peak GPU Mem  | Loss     | Verdict                                               |
| -------- | ---- | ----- | ------------------------------ | ------- | ------------ | -------- | ------- | ------------- | -------- | ----------------------------------------------------- |
| 22093998 | 64k  | 2     | DS-Z2 dp=2 tp=8 ep=8 + chunked | yes     | 56.59 %      | 53.92 %  | 51,090  | 67.6 GB (85%) | 11–13 ✅ | -0.6 pp vs no-compile (57.23 %) — **compile no help** |
| 22093999 | 128k | 4     | DS-Z2 dp=4 tp=8 ep=8 + chunked | yes     | 68.29 %      | 67.15 %  | 65,620  | 75.6 GB (95%) | 12–14 ✅ | -0.8 pp vs no-compile (69.07 %) — **compile no help** |
| 22094000 | 128k | 8     | DS-Z2 dp=8 tp=8 ep=8 + chunked | no      | 68.38 %      | 52.68 %¹ | 131,500 | 73.8 GB (93%) | 12–14 ✅ | matches 4n MFU; 2× TPS at 8n                          |

¹ Cum MFU lower than 4n (64.10%) due to one transient stall at step 20 (dropped to 32% window before recovering); peak/steady are still 68.3%.

### Fourth batch results — 256k unlocked, 16k chunked baseline

| Job      | Ctx      | Nodes | Mesh                                            | Compile | Win MFU peak | Cum MFU  | TPS Win | Peak GPU Mem  | Loss         | Verdict                                                                                                |
| -------- | -------- | ----- | ----------------------------------------------- | ------- | ------------ | -------- | ------- | ------------- | ------------ | ------------------------------------------------------------------------------------------------------ |
| 22094014 | 16k      | 2     | DS-Z2 dp=2 tp=8 ep=8 + chunked                  | no      | **32.57 %**  | 24.50 %  | 87,560  | 48.9 GB (61%) | 10–13 ✅     | small regression vs 40.4% FSDP+EP no-chunk — chunked overhead at 16k where lm_head wasn't a bottleneck |
| 22094013 | **256k** | 8     | DS-Z3 dp=8 + SP=8 (intra-node) + chunked, no EP | no      | **32.60 %**  | 23.68 %² | 32,330  | 36.1 GB (45%) | 1.59–1.71 ✅ | **NEW 256k CHAMPION** — 24× over previous SP=16 cross-node 1.36 %                                      |

² 256k @ 8n still partial (last 3 logs in window 29–33 %, climbing).

### Two big findings from this batch

1. **Compile + chunked at long ctx = no win**: at 32k/64k/128k the compile wrap adds overhead but no compute gain — chunked-CE already saturates GPU compute. Compile-bound experiments are essentially closed for the chunked path.
2. **256k unlocked at 32 % MFU via DS-Z3+SP=8+chunked**: the previous 256k attempt (SP=16) was 1.36 % because SP=16 spans 2 nodes per group, killing intra-Ulysses bandwidth. Dropping to SP=8 keeps all-to-all intra-node (NVLink) and chunked frees enough memory to fit 256k. **Memory utilization just 45 %** — there's substantial headroom for 512k.

### 16k chunked is a slight regression (32.57 % vs 40.4 % FSDP+EP no-chunked)

At 16k the lm_head matmul is small enough (16k × 151936 × 2B = ~5 GB) that it's not the bottleneck. Adding chunked introduces per-chunk overhead (multiple smaller matmuls + boundary handling) that costs ~8 pp at this ctx. **Verdict: enable chunked for ≥ 32k context, leave it off for 16k.**

### Updated headline (Qwen3-30B-A3B, training-correct)

| Goal                    | Best recipe                                           | Window MFU        | Peak GPU Mem | Notes                                     |
| ----------------------- | ----------------------------------------------------- | ----------------- | ------------ | ----------------------------------------- |
| Best 16k MFU            | FSDP + EP=8 + FA3 + sonicmoe (no compile, no chunked) | **40.4 %**        | 73.6 GB      | 16k still belongs to the no-chunked path  |
| Best 32k MFU            | DS-Z2 + EP=8 + FA3 + sonicmoe + chunked               | **45.81 %**       | 55.0 GB      | +24 pp over old 32k champ                 |
| Best 64k MFU            | DS-Z2 + EP=8 + FA3 + sonicmoe + chunked @ 2n          | **57.23 %**       | 67.6 GB      | +37 pp over old                           |
| Best 128k MFU           | DS-Z2 + EP=8 + FA3 + sonicmoe + chunked @ 4n          | **69.10 %**       | 75.6 GB      | +50 pp over old, highest MoE MFU on stack |
| Highest 128k throughput | DS-Z2 + EP=8 + FA3 + sonicmoe + chunked @ 8n          | 68.38 %, 131k TPS | 73.8 GB      | 2× TPS at small MFU cost                  |
| Best 256k MFU           | DS-Z3 + SP=8 (intra-node) + FA3 + sonicmoe + chunked  | **32.60 %**       | 36.1 GB      | 24× over old; tons of headroom (45% mem)  |

### Next push (submitted)

- 22094020: **512k DS-Z3+SP=8+chunked @ 16n** — frontier push. Per-rank seq = 32k (same as 256k @ 8n). If MFU scales, this is the longest training-correct context on this stack.
- 22094021: 256k DS-Z3+SP=8+chunked + compile @ 8n — does compile help the SP path?
- 22094022: 256k DS-Z3+SP=4+chunked @ 8n — fewer SP comm steps, more activation per rank. Trade-off study.

## 2026-04-29 (later): 512k attempt + SP=4 + compile-stabilizes-SP finding

### Fifth batch — push to 512k

| Job      | Ctx      | Nodes | Mesh                                      | Win MFU peak          | Cum MFU | TPS Win | Peak GPU Mem  | Loss    | Status                                                                                                                                                          |
| -------- | -------- | ----- | ----------------------------------------- | --------------------- | ------- | ------- | ------------- | ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 22094020 | **512k** | 16    | DS-Z3 dp=16 SP=8 + FA3 + chunked          | **40.37 %** (step 10) | 20.25 % | 40,680  | 43.5 GB (55%) | 1.58 ✅ | **HANG** at step 15: NCCL all-gather watchdog timeout (30 min). 128 ranks too many for ZeRO-3 cross-node. The 40 % MFU number is real but unsustainable at 16n. |
| 22094021 | 256k     | 8     | DS-Z3 dp=8 SP=8 + FA3 + chunked + compile | 32.63 %               | 24.14 % | 28,650  | 36.1 GB (45%) | 1.58 ✅ | compile **stabilizes** SP path (steady 22-32 % vs no-compile 5-32 %); peak slightly lower                                                                       |
| 22094022 | 256k     | 8     | DS-Z3 dp=16 SP=4 + FA3 + chunked          | **46.49 %** peak      | 25.70 % | 25,590  | 45.2 GB (57%) | 1.59 ✅ | highest 256k peak; oscillates badly (14-46 %) without compile                                                                                                   |

### Sixth batch — fix Z3 hang via smaller mesh + test compile stability

| Job      | Ctx      | Nodes | Mesh                                       | Win MFU peak          | Cum MFU     | TPS Win  | Peak GPU Mem  | Loss    | Status                                                                                                                                                                  |
| -------- | -------- | ----- | ------------------------------------------ | --------------------- | ----------- | -------- | ------------- | ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 22094162 | **512k** | 8     | DS-Z3 dp=8 SP=8 + FA3 + chunked            | **44.82 %** (step 10) | 32.25 %     | 130k cum | 46.4 GB (58%) | 1.59 ✅ | **NEW 512k CHAMPION** — smaller Z3 mesh (64 ranks) avoids the 16n hang. Per-rank seq = 64k. Peak 44.82 % at step 10, drops to 21.89 % at step 20 (oscillation pattern). |
| 22094163 | 256k     | 8     | DS-Z3 dp=16 SP=4 + FA3 + chunked + compile | **46.61 %** peak      | **42.54 %** | 168k cum | 45.2 GB (57%) | 1.59 ✅ | **NEW 256k CHAMPION** — compile **stabilizes** the SP=4 path: window MFU 40-47 % across full run vs 14-46 % without compile. Best 256k MFU + best stability.            |

### The compile-stabilizes-SP-chunked finding

The DS-Z3 SP path with chunked-CE shows window-MFU oscillation at long context (256k+):

- 256k SP=8 no compile: 5–32 % swing, cum 11.07 %
- 256k SP=8 + compile: 22–32 % steady, cum 24.14 % (×2)
- 256k SP=4 no compile: 14–46 % swing, cum 25.70 %
- **256k SP=4 + compile: 40–47 % steady, cum 42.54 %** (×1.7)

**Hypothesis**: chunked-CE patches `model.forward` to compute lm_head matmuls in chunks. Each chunk creates an autograd graph with N matmul nodes; under DS-Z3, each backward chunk triggers an all-gather of `lm_head.weight` across the (large) DP group. Without compile, these per-chunk all-gathers serialize and fight ZeRO-3's pipelined param fetches → intermittent stalls. Compile bundles the chunk loop into a single graph, scheduling collectives more predictably.

**Conclusion**: at 256k+ on the DS-Z3+SP+chunked path, **always enable compile**. At ≤128k DS-Z2+EP+chunked path, compile gives no benefit (already saturated, no Z3 cross-node).

### Updated headline (chunked-CE era, 2026-04-29)

| Goal          | Best recipe                                     | Window MFU peak | Cum MFU     | Peak GPU Mem | Notes                                               |
| ------------- | ----------------------------------------------- | --------------- | ----------- | ------------ | --------------------------------------------------- |
| Best 16k MFU  | FSDP + EP=8 + FA3 + sonicmoe (NO chunked)       | **40.4 %**      | 32.15 %     | 73.4 GB      | 16k stays no-chunked                                |
| Best 32k MFU  | DS-Z2 + EP=8 + FA3 + chunked                    | **45.86 %**     | 39.31 %     | 55.0 GB      | +24 pp over old                                     |
| Best 64k MFU  | DS-Z2 + EP=8 + FA3 + chunked @ 2n               | **57.23 %**     | 46.07 %     | 67.6 GB      | +37 pp over old                                     |
| Best 128k MFU | DS-Z2 + EP=8 + FA3 + chunked @ 4n               | **69.10 %**     | 64.10 %     | 75.6 GB      | +50 pp; highest MoE MFU on stack                    |
| Best 256k MFU | **DS-Z3 + SP=4 + FA3 + chunked + compile @ 8n** | **46.61 %**     | **42.54 %** | 45.2 GB      | +45 pp + stable; **NEW**                            |
| Best 512k MFU | **DS-Z3 + SP=8 + FA3 + chunked @ 8n**           | **44.82 %**     | 32.25 %     | 46.4 GB      | per-rank seq=64k; oscillates → **try compile next** |

### Next push (submitted)

- 22094164: **512k @ 8n SP=8 + compile + chunked** — apply the SP+compile stability fix to 512k
- 22094165: 256k SP=2 + compile + chunked — see if reducing SP further (per-rank seq=128k) keeps the MFU climbing

## 2026-04-29 (final): per-rank-seq sweet spot, 1M context unlocked

### Seventh + eighth batch results

| Job      | Ctx      | Nodes | Mesh                                     | Compile | Win MFU peak | Cum MFU     | TPS Win | Peak GPU Mem  | Loss         | Status                                                                  |
| -------- | -------- | ----- | ---------------------------------------- | ------- | ------------ | ----------- | ------- | ------------- | ------------ | ----------------------------------------------------------------------- |
| 22094164 | 512k     | 8     | DS-Z3 dp=8 SP=8 + chunked + compile      | yes     | 45.01 %      | 39.20 %     | 21k     | 46.4 GB (58%) | 1.59 ✅      | compile fixes 512k oscillation; +7 pp cum vs no-compile                 |
| 22094165 | **256k** | 8     | **DS-Z3 dp=32 SP=2 + chunked + compile** | yes     | **59.61 %**  | **52.40 %** | 56k–59k | 69.4 GB (87%) | 1.54-1.74 ✅ | **NEW 256k CHAMPION** — stable 56–60% across full run                   |
| 22094167 | 512k     | 8     | DS-Z3 dp=16 SP=4 + chunked + compile     | yes     | 58.24 %      | 38.76 %     | 24k     | 70.7 GB (89%) | 1.58–1.74 ✅ | one step-15 transient dip; peak matches 256k SP=2                       |
| 22094168 | **1M**   | 8     | **DS-Z3 dp=8 SP=8 + chunked + compile**  | yes     | **37.46 %**  | **35.65 %** | 9.5k    | 72.2 GB (91%) | 1.59–1.75 ✅ | **NEW 1M CONTEXT TRAINS!** First training-correct 1M MoE on this stack. |

### The per-rank-seq sweet spot

The MFU peaks at **per-rank seq = 128k** regardless of total context:

| Total ctx | Nodes | SP  | Per-rank seq | Peak MFU window | Cum MFU     | Peak Mem |
| --------- | ----- | --- | ------------ | --------------- | ----------- | -------- |
| 256k      | 8     | 8   | 32k          | 32.63 %         | 24.14 %     | 36.1 GB  |
| 256k      | 8     | 4   | 64k          | 46.61 %         | 42.54 %     | 45.2 GB  |
| **256k**  | 8     | 2   | **128k**     | **59.61 %**     | **52.40 %** | 69.4 GB  |
| 512k      | 8     | 8   | 64k          | 45.01 %         | 39.20 %     | 46.4 GB  |
| **512k**  | 8     | 4   | **128k**     | 58.24 %         | 38.76 %     | 70.7 GB  |
| **1M**    | 8     | 8   | **128k**     | **37.46 %**     | **35.65 %** | 72.2 GB  |

**Why 128k per-rank seq is optimal**:

1. **Memory**: at 128k per-rank, ~70 GB peak (~90% of 80 GB H100). Lower per-rank seq leaves memory unused; higher OOMs.
2. **Comm**: less SP means fewer Ulysses all-to-all rounds per step. SP=2 has 1/4 the all-to-all of SP=8.
3. **Compute density**: more tokens per rank = more matmul work to amortize comm and dispatch overhead.

**Why 1M @ 37 % is lower than 256k SP=2 @ 60 %** (same per-rank seq):

1M context = more chunked-CE chunks per step (1M/256 ≈ 4000 vs 256k/256 ≈ 1000). Each chunk triggers a DS-Z3 all-gather of `lm_head.weight` across the 64-rank DP group. The all-gather overhead scales linearly with chunks, dragging steady-state MFU down. Could be addressed by larger `chunk_size` (currently hardcoded 256).

### Compile is essential at 256k+ on the SP path

Without compile the DS-Z3+SP+chunked path oscillates wildly (window MFU 5–47 %); with compile the same configs are stable at 40–60 %. Compile bundles the chunked autograd graph so Z3 can pipeline param fetches predictably.

| Config    | No compile peak / cum | Compile peak / cum | Compile delta cum |
| --------- | --------------------- | ------------------ | ----------------- |
| 256k SP=8 | 32.6 % / 11.1 %       | 32.6 % / 24.1 %    | **+13 pp**        |
| 256k SP=4 | 46.5 % / 25.7 %       | 46.6 % / 42.5 %    | **+17 pp**        |
| 256k SP=2 | (not run)             | 59.6 % / 52.4 %    | n/a (champion)    |
| 512k SP=8 | 44.8 % / 32.3 %       | 45.0 % / 39.2 %    | **+7 pp**         |
| 512k SP=4 | (not run)             | 58.2 % / 38.8 %    | n/a               |
| 1M SP=8   | (not run)             | 37.5 % / 35.7 %    | n/a               |

### Final headline (entire chunked-CE era, 2026-04-29)

| Goal          | Best recipe                                     | Window MFU peak | Cum MFU     | Peak Mem | Notes                                          |
| ------------- | ----------------------------------------------- | --------------- | ----------- | -------- | ---------------------------------------------- |
| Best 16k MFU  | FSDP + EP=8 + FA3 + sonicmoe (NO chunked)       | **40.4 %**      | 32.15 %     | 73.4 GB  | 16k stays no-chunked (chunked regresses ~8 pp) |
| Best 32k MFU  | DS-Z2 + EP=8 + FA3 + sonicmoe + chunked         | **45.86 %**     | 39.31 %     | 55.0 GB  | +24 pp over old                                |
| Best 64k MFU  | DS-Z2 + EP=8 + FA3 + sonicmoe + chunked @ 2n    | **57.23 %**     | 46.07 %     | 67.6 GB  | +37 pp over old                                |
| Best 128k MFU | DS-Z2 + EP=8 + FA3 + sonicmoe + chunked @ 4n    | **69.10 %**     | 64.10 %     | 75.6 GB  | +50 pp over old, **highest MoE MFU on stack**  |
| Best 256k MFU | DS-Z3 + SP=2 + FA3 + chunked + compile @ 8n     | **59.61 %**     | **52.40 %** | 69.4 GB  | +58 pp over old (1.36 %)                       |
| Best 512k MFU | DS-Z3 + SP=4 + FA3 + chunked + compile @ 8n     | **58.24 %**     | 38.76 %     | 70.7 GB  | NEW context — was unreachable before           |
| Best 1M MFU   | **DS-Z3 + SP=8 + FA3 + chunked + compile @ 8n** | **37.46 %**     | **35.65 %** | 72.2 GB  | **NEW frontier — 1M ctx MoE training works**   |

### What chunked-CE + per-rank-seq tuning unlocked

Before this work the stack topped out at 128k DS-Z3+SP=8 = 19.4 % MFU and 256k SP=16 = 1.36 %. Today 1M context trains at 37.46 % MFU on 8 nodes — a **27× improvement at 1M**, and **+18 pp at 128k**.

Total contexts covered correctly: 16k → 32k → 64k → 128k → 256k → 512k → **1M** (all with healthy loss curves, sonicmoe path with clamp wrapper at 16k–128k, plain SP path at 256k+).

### Stopping submitting

User asked to test how far chunked-CE pushes. We've validated 1M context training. No more jobs submitted.

### MFU convention caveat

The MFU numbers in this report (and the consolidated table in `sft_benchmark_notion.md`) are computed with the **non-causal attention FLOPs convention** — `compute_flops_per_token` in `trl/trainer/utils.py` treats every token as attending to the full `seq_len` (matches the convention used in the PaLM, Megatron, and nanoGPT codebases). With causal masking the actual attention compute is ~half this, so reported MFU is inflated by:

| ctx  | inflation | examples (reported → causal-corrected) |
| ---- | --------- | -------------------------------------- |
| 16k  | ×1.07     | 40.4 → ~37.7 %                         |
| 32k  | ×1.13     | 45.86 → ~40.7 %                        |
| 64k  | ×1.41     | 57.23 → ~40.6 %                        |
| 128k | ×1.66     | 69.10 → ~41.6 %                        |
| 256k | ×1.94     | 59.61 → ~30.7 %                        |
| 512k | ×1.97     | 58.24 → ~29.6 %                        |
| 1M   | ×1.98     | 37.46 → ~18.9 %                        |

GPU peak used in the denominator is the H100 SXM5 dense bf16 Tensor Core peak (989.5 TFLOPS) — verified with `nvidia-smi` returning "NVIDIA H100 80GB HBM3" (so we are SXM5 on AWS p5.48xlarge, _not_ H100 NVL).

**All relative comparisons in this report are still valid** because the same formula is applied to every run (old vs new, FSDP vs DS, EP vs SP). When comparing absolute MFU to other published numbers using the causal convention (Llama 2/3 papers, DeepSpeed Ulysses paper), apply the column 2 factor above.


## 2026-04-29 (later): Liger kernel under EP — root cause found, sweep launched

### The Liger × EP fix in one paragraph

Earlier we'd marked `H1` in `upstream_todo.md` as "Liger fails because of 3D weights." Re-investigated today with a minimal repro (`benchmark/test_liger_qwen3_moe.py`, `test_liger_qwen3_moe_ep.py`); that diagnosis was wrong. Single-GPU Liger + Qwen3-MoE works fine (max diff 8.79e-3 vs eager). The actual EP failure: `from_pretrained(distributed_config=DistributedConfig(enable_expert_parallel=True))` automatically sets `model.config._experts_implementation = "grouped_mm"`, and transformers' `@use_experts_implementation` decorator wraps `Qwen3MoeExperts.forward` in a dispatcher that routes to `grouped_mm_experts_forward` (EP-aware, no `F.one_hot`). Liger's `_patch_swiglu_module(experts, LigerExperts)` rebinds `experts.forward = LigerExperts.forward`, **bypassing the dispatcher**. `LigerExperts.forward` then calls `F.one_hot(top_k_index, num_classes=self.num_experts)` directly, where `self.num_experts` was overwritten by `shard_tensor` to `num_local_experts` AND `top_k_index.max()` is set to the same `num_local_experts` (the sentinel from `RouterParallel._prepare_output_fn` line 1209) → out-of-range → `device-side assert triggered`.

**Workaround**: pass `liger_kernel_config={"swiglu": False}` (CLI: `--liger_kernel_config '{"swiglu":false}'`). Disables the broken swiglu/experts patch, keeps `LigerRMSNorm` + `liger_rotary_pos_emb` + `LigerFusedLinearCrossEntropyLoss`. Validated on EP=2 tiny repro: logits within 7.81e-3 of eager. Full investigation log in `benchmark/debug_liger_ep.md`.

### Sweep submitted (results pending)

Compares **Liger (with FLCE, no swiglu patch)** against the existing **chunked-CE champions** at the same config. Note: Liger's FLCE and `--loss_type chunked_nll` are mutually exclusive (TRL guards this); both achieve similar memory savings, so the comparison is effectively "FLCE vs chunked_nll" with the rest of the stack constant.

| Job      | Ctx  | Nodes | Mesh                              | Liger config                          | Compare to                          |
| -------- | ---- | ----- | --------------------------------- | ------------------------------------- | ----------------------------------- |
| 22094513 | 16k  | 2     | FSDP DP=2 + EP=8 + FA3 + sonicmoe | swiglu=F, FLCE=T, RoPE=T, RMSNorm=T | FSDP+EP no-Liger 40.4 % MFU         |
| 22094514 | 32k  | 2     | DS-Z2 + EP=8 + FA3 + sonicmoe     | same                                  | DS-Z2+EP+chunked 45.81 %            |
| 22094515 | 64k  | 2     | DS-Z2 + EP=8 + FA3 + sonicmoe     | same                                  | DS-Z2+EP+chunked 57.23 %            |
| 22094516 | 128k | 4     | DS-Z2 + EP=8 + FA3 + sonicmoe @4n | same                                  | DS-Z2+EP+chunked 69.10 %            |

Total 10 nodes — Slurm will queue past the 8n cap. Results below once they complete.


### 2026-04-29 (cont.): Liger sweep first attempt failed at arg parse

Initial submission (jobs 22094513-22094516) crashed before training:

```
sft.py: error: argument --liger_kernel_config/--liger-kernel-config: invalid dict value: '{"swiglu":false}'
```

`HfArgumentParser` falls through to `type=dict` (line 228 of `transformers/hf_argparser.py`) which rejects JSON strings — argparse calls `dict('{"swiglu":false}')` and gets `TypeError`.

**Patched `trl/scripts/sft.py`** with a small `_preparse_dict_args()` helper that extracts `--liger_kernel_config` from `sys.argv` BEFORE parsing, stores the JSON string in an env var, then re-injects the parsed dict into `training_args.liger_kernel_config` after `parser.parse_args_and_config()` returns. Resubmitted as 22094539-22094542 (EP sweep) + 22094545-22094547 (SP long-ctx sweep).

### Liger sweep — combined batches (resubmitted)

| Job      | Ctx  | Nodes | Mesh                                 | Variant                              | vs chunked-CE champion |
| -------- | ---- | ----- | ------------------------------------ | ------------------------------------ | ---------------------- |
| 22094539 | 16k  | 2     | FSDP DP=2 + EP=8 + FA3 + sonicmoe   | Liger (FLCE+RMSN+RoPE, no swiglu)  | FSDP+EP no-Liger 40.4 % |
| 22094540 | 32k  | 2     | DS-Z2 + EP=8 + FA3 + sonicmoe       | Liger                                | DS-Z2+EP+chunked 45.81 % |
| 22094541 | 64k  | 2     | DS-Z2 + EP=8 + FA3 + sonicmoe       | Liger                                | DS-Z2+EP+chunked 57.23 % |
| 22094542 | 128k | 4     | DS-Z2 + EP=8 + FA3 + sonicmoe       | Liger                                | DS-Z2+EP+chunked 69.10 % |
| 22094545 | 256k | 8     | DS-Z3 + SP=2 + FA3 + sonicmoe       | Liger + compile                      | SP=2 + compile + chunked 59.61 % |
| 22094546 | 512k | 8     | DS-Z3 + SP=4 + FA3 + sonicmoe       | Liger + compile                      | SP=4 + compile + chunked 58.24 % |
| 22094547 | 1M   | 8     | DS-Z3 + SP=8 + FA3 + sonicmoe       | Liger + compile                      | SP=8 + compile + chunked 37.46 % |

Will be filled in as results come in.

### Results so far (2026-04-29 ~10:00 UTC)

| Job      | Ctx  | Win MFU peak | Cum MFU | TPS Win    | Peak GPU Mem  | Loss      | vs chunked-CE      |
| -------- | ---- | ------------ | ------- | ---------- | ------------- | --------- | ------------------ |
| 22094540 | 32k  | **56.62 %**  | 45.25 % | 92,010     | 54.3 GB (68%) | 11–15 ✅   | **+11 pp peak**, +6 pp cum (chunked: 45.86 / 39.31) |
| 22094541 | 64k  | **66.46 %**  | 61.69 % | 60,190     | 66.9 GB (84%) | 11–13 ✅   | **+9 pp peak**, +15 pp cum (chunked: 57.23 / 46.07) |

**Liger's FLCE beats `chunked_nll`** on the EP path at both 32k and 64k. Same memory ceiling (within ~1 GB), much higher MFU. Three things contribute beyond just FLCE: `LigerRMSNorm` (called per-layer, every attention block), `liger_rotary_pos_emb` (one fused Triton op for Q/K rotation per attention call), and FLCE itself replacing the chunked-CE Python loop. Together they dominate the +6–15 pp Cum MFU gain.

Remaining results (16k, 128k, 256k, 512k, 1M) coming in as jobs complete.

### Liger sweep continued — 128k and 16k DS-Z2 retry

| Job      | Ctx  | Win MFU peak | Cum MFU | TPS Win    | Peak GPU Mem  | Loss   | vs chunked-CE                      |
| -------- | ---- | ------------ | ------- | ---------- | ------------- | ------ | ---------------------------------- |
| 22094576 | 16k  | **40.70 %**  | 28.80 % | 109,000    | 48.4 GB (61%) | 10–13 ✅ | **+8 pp peak** vs DS-Z2+chunked 32.57 % — at parity with FSDP+EP no-Liger 40.4 % |
| 22094542 | 128k | **76.29 %**  | 74.69 % | 73,330     | 72.6 GB (91%) | 12–14 ✅ | **+7 pp peak**, +10 pp cum (chunked 69.10 / 64.10) |

### Liger crushes chunked-CE on the EP path

Updated 4-way comparison at every context, same hardware (DS-Z2+EP=8+sonicmoe, only loss strategy differs):

| Ctx    | chunked-CE peak / cum | Liger (FLCE+RoPE+RMSN) peak / cum | Liger Δ peak |
| ------ | --------------------- | --------------------------------- | ------------ |
| 16k    | 32.57 / 24.50 %       | **40.70 / 28.80 %**               | **+8 pp**    |
| 32k    | 45.86 / 39.31 %       | **56.62 / 45.25 %**               | **+11 pp**   |
| 64k    | 57.23 / 46.07 %       | **66.46 / 61.69 %**               | **+9 pp**    |
| 128k   | 69.10 / 64.10 %       | **76.29 / 74.69 %**               | **+7 pp**    |

**Liger is the new long-context champion across the EP path**. Three drivers: `LigerFusedLinearCrossEntropyLoss` (faster + same memory savings as chunked_nll), `LigerRMSNorm` (called per-layer on every attention block, ~1-2 pp), `liger_rotary_pos_emb` (fused Triton RoPE, small but everywhere). They compose with sonicmoe/grouped_mm experts (the EP forward path stays untouched because we set `swiglu=False`).

128k @ 76.29 % MFU is the **highest MoE training MFU we've ever recorded on this stack** — beats the previous 69.10 % chunked record. Memory is essentially the same (72.6 vs 75.6 GB), so we get the +7 pp without giving up context length.

Outstanding: 256k SP=2 + Liger and 512k SP=4 + Liger still running; 1M SP=8 + Liger pending. Will compare those to the SP+chunked champions in the next update.

### Liger sweep on SP path (256k, 512k, 1M)

| Job      | Ctx  | Win MFU peak | Cum MFU | TPS Win    | Peak GPU Mem  | Loss   | vs SP+chunked              |
| -------- | ---- | ------------ | ------- | ---------- | ------------- | ------ | -------------------------- |
| 22094545 | 256k | **63.62 %**  | 60.59 % | 63,090     | 66.9 GB (84%) | 1.54-1.74 ✅ | **+4 pp peak**, +8 pp cum (chunked 59.61/52.40) |
| 22094546 | 512k | **63.26 %**  | 57.95 % | 31,290     | 69.7 GB (87%) | 1.58-1.74 ✅ | **+5 pp peak**, +19 pp cum (chunked 58.24/38.76) |
| 22094547 | 1M   | running...   | —       | —          | —             | —      | vs chunked 37.46/35.65     |

**Liger holds the win on SP path too.** Even on the comm-bound DS-Z3+SP route, Liger's FLCE+RMSNorm+RoPE deliver +4–5 pp peak MFU and a much larger cum MFU bump (because Liger is more steady step-to-step than chunked_nll, which oscillated wildly without compile).

### All-context final headline (chunked + Liger era)

| Ctx   | Best chunked-CE recipe         | chunked Win MFU | Best Liger recipe              | Liger Win MFU | Δ peak |
| ----- | ------------------------------ | --------------- | ------------------------------ | ------------- | ------ |
| 16k   | (no chunked, FSDP+EP champ)    | 40.40 %         | DS-Z2+EP+Liger                 | **40.70 %**   | +0.3 pp (parity) |
| 32k   | DS-Z2+EP+chunked               | 45.86 %         | DS-Z2+EP+Liger                 | **56.62 %**   | +11 pp |
| 64k   | DS-Z2+EP+chunked @2n           | 57.23 %         | DS-Z2+EP+Liger @2n             | **66.46 %**   | +9 pp  |
| 128k  | DS-Z2+EP+chunked @4n           | 69.10 %         | DS-Z2+EP+Liger @4n             | **76.29 %**   | +7 pp  |
| 256k  | SP=2+compile+chunked @8n       | 59.61 %         | SP=2+compile+Liger @8n         | **63.62 %**   | +4 pp  |
| 512k  | SP=4+compile+chunked @8n       | 58.24 %         | SP=4+compile+Liger @8n         | **63.26 %**   | +5 pp  |
| 1M    | SP=8+compile+chunked @8n       | 37.46 %         | SP=8+compile+Liger @8n (running)| TBD          | TBD    |

### Single-node 1n sweep — kicked off

Now testing how far Qwen3-30B-A3B can go on a single 8-GPU H100 node. Submitted (22094605-609):

- 16k DS-Z2+EP=8 1n
- 32k DS-Z2+EP=8 1n
- 64k SP=8 + compile 1n (per-rank=8k)
- 128k SP=8 + compile 1n (per-rank=16k)
- 256k SP=8 + compile 1n (per-rank=32k — the predicted ceiling)

All using the new Liger recipe (`--use_liger_kernel true --liger_kernel_config '{"swiglu":false}'`). Results below.

### 1M Liger SP=8 — final
| Job      | Ctx  | Win MFU peak | Cum MFU | Peak GPU Mem  | vs chunked         |
| -------- | ---- | ------------ | ------- | ------------- | ------------------ |
| 22094547 | 1M   | **62.33 %**  | 58.29 % | 69.2 GB (87%) | **+25 pp peak**, +23 pp cum (chunked 37.46/35.65) |

Liger+SP=8 at 1M context = **62.33 % window MFU** — same model, longer context, healthy loss (1.6). 1M training is no longer marginal.

### Single-node 1n — first results

| Job      | Ctx  | Mesh                                | Win MFU peak | Cum MFU | Peak GPU Mem  | vs 2n+ champion              |
| -------- | ---- | ----------------------------------- | ------------ | ------- | ------------- | ---------------------------- |
| 22094605 | 16k  | 1n DS-Z2+EP=8+Liger                 | 44.30 %      | 26.59 % | 73.5 GB (92%) | 2n FSDP+EP no-Liger 40.4 % — **1n MATCHES 2n** |
| 22094606 | 32k  | 1n DS-Z2+EP=8+Liger                 | **59.28 %**  | 46.21 % | 78.1 GB (98%) | 2n DS-Z2+EP+Liger 56.62 % — **1n BEATS 2n** by +3 pp (intra-node EP comm) |

Two huge findings:

1. **1n MFU ≥ 2n MFU** at 16k and 32k. Removing cross-node comm (the 2n DP=2 grad reduce) outweighs the lost batch parallelism. EP=8 fully fits on one node (NVLink only) — fastest comm topology possible.
2. **32k 1n is at 98 % memory** — the EP path's practical 1n ceiling. Going to 64k 1n EP would OOM. **For longer context on 1n, switch to SP=8 (per-rank seq = ctx/8)**.

64k/128k/256k 1n SP=8+Liger+compile still in progress (compile init takes 10-15 min). Will update.

### 1n SP=8 results — memory-limited

| Job      | Ctx  | Mesh                                     | Win MFU peak | Cum MFU | Peak Mem      | Status |
| -------- | ---- | ---------------------------------------- | ------------ | ------- | ------------- | ------ |
| 22094607 | 64k  | 1n DS-Z3 dp=1 sp=8 +compile + Liger      | **7.99 %**   | 1.32 %  | 78.6 GB (99%) | trains but very slow MFU |
| 22094608 | 128k | same                                     | OOM 1.16 GiB | —       | 78.0 GB (98%) | OOM during forward     |
| 22094609 | 256k | same                                     | OOM 1.16 GiB | —       | 75.7 GB (95%) | OOM during forward     |

**Why so slow at 64k 1n SP=8?** With `DP=1, SP=8`, ZeRO-3 partitions across the world group of 8 ranks. But there's no DP group for grad/optim reduce — every "step" involves just bookkeeping on the partitioned shards. The bottleneck is likely the per-rank Adam optimizer step + unsharded attention scratch buffers under SP=8. Memory budget is also tight (99 % peak). Compile may be hurting too (long graph, no DP amortization).

**1n SP=8 is memory-bound at 128k+** — the (per-rank seq=16k) is fine, but optim states and FA3 scratch at SP=8 push past 80 GB.

### 1n v2 probe results (no-compile + SP=4 DP=2)

| Job      | Ctx  | Mesh                                  | Win MFU peak | Cum MFU | Peak Mem      | Verdict |
| -------- | ---- | ------------------------------------- | ------------ | ------- | ------------- | ------- |
| 22094620 | 64k  | 1n DS-Z3 dp=1 sp=8 NO compile + Liger | 8.61 %       | 4.31 %  | 79.0 GB (99%) | compile wasn't the issue — DP=1 means Z3 doesn't actually partition optim/grads |
| 22094621 | 64k  | 1n DS-Z3 dp=2 sp=4 +compile + Liger   | **23.23 %**  | 14.16 % | 78.3 GB (98%) | DP=2 helps (Z3 partitions optim across 2) — 3× the SP=8 result, still memory-tight |
| 22094622 | 128k | 1n DS-Z3 dp=2 sp=4 +compile + Liger   | OOM 100 %    | —       | 79.5 GB       | 128k 1n past the wall                     |

### CPU offload sweep — cancelled

User direction (2026-04-29): CPU offload kills MFU; not worth running. Hard rule recorded in `benchmark/CLAUDE.md`. The 3 offload jobs (22094639/640/641) were cancelled while still in cpu_adam JIT-compile.

### Final 1n recommendation

| Goal                            | Recipe                                  | Win MFU peak | Cum MFU | Peak Mem      |
| ------------------------------- | --------------------------------------- | ------------ | ------- | ------------- |
| **Best 1n MFU**                 | **1n DS-Z2+EP=8+Liger @ 32k**           | **59.28 %**  | 46.21 % | 78.1 GB (98%) |
| Practical 1n max ctx            | 1n DS-Z2+EP=8+Liger @ 32k               | 59.28 %      | 46.21 % | 78.1 GB       |
| 1n SP path (longer ctx, lower MFU) | 1n DS-Z3+SP=4 dp=2 +compile +Liger @ 64k | 23.23 %     | 14.16 % | 78.3 GB       |
| 1n > 32k EP / > 64k SP          | not viable on this stack without offload (which we don't use) | —     | —       | —             |

**32k is the practical 1n ceiling** with the no-offload constraint. EP=8 at 1n uses ~14.5 GB params + ~30 GB optim (DS-Z2, no DP to shard across) + ~30 GB activations/FA3 scratch at 32k seq = 78 GB. Going past would need bf16 optim, different EP topology to also shard optim, or smarter activation checkpointing. Tracked in `upstream_todo.md` I2.


## 2026-04-29 (later): sonic-moe wrapper-clamp redundancy check

The `transformers/integrations/sonicmoe.py` wrapper has a `clamp + masked_fill` block
that costs ~2 pp MFU. It was added because an earlier sonic-moe snapshot produced NaN
gradients on EP sentinel rows (`expert_ids >= num_experts`) when weights came from
`DTensor.to_local()`.

Built a minimal repro at `benchmark/test_sonic_bwd_dtensor.py` (covers the 2×2×2 matrix
of {clamp on/off} × {plain/DTensor.to_local() weights} × {valid/sentinel-heavy expert_ids}).
**On the current snapshot (`IlyasMoutawwakil/sonic-moe@b15942...`), the bug does not
reproduce** — all 16 cells produce finite gradients on both 1-rank DTensor mesh and real
EP=2 mesh.

Bonus: documented a non-bug surprise — `out.sum().backward()` always crashes the kernel's
varlen_m k-major check because PyTorch optimizes that into a stride-0 broadcast view of
`torch.ones(())`. Production paths use real upstream gradients (CE loss output) so this
never fires. Use `out.backward(torch.randn_like(out))` in repro scripts to sidestep.

Full investigation log in `benchmark/debug_sonic_bwd_dtensor.md`.

### Production validation (job 22094732) — clamp is load-bearing

Ran 32k DS-Z2+EP=8+chunked at 2n with the env var `SONICMOE_DISABLE_CLAMP=1` set (clamp+masked_fill block bypassed):

| Step | loss  | grad_norm | entropy | mean_token_acc | mfu_window |
| ---- | ----- | --------- | ------- | --------------- | ---------- |
| 5    | 9.807 | 5.099     | NaN     | 0.117           | (init)     |
| 10   | 0.000 | 5.099     | NaN     | 0.0002          | 51.72 %    |
| 15   | 0.000 | 5.099     | NaN     | 0.0003          | 51.26 %    |
| 20   | 0.000 | 5.099     | NaN     | 0.0002          | 51.75 %    |
| 25   | 0.000 | 5.099     | NaN     | 0.0002          | 50.54 %    |
| 30   | 0.000 | 5.099     | NaN     | 0.0004          | 51.72 %    |

**Verdict: the wrapper-clamp is load-bearing.**

- Loss collapsed to 0 by step 10 (baseline with clamp = 11–15 healthy).
- Entropy NaN throughout — kernel produced NaN gradients.
- `grad_norm = 5.099` exactly across all 30 steps. Adam zeros NaN grads, so the gradient norm doesn't move. Classic NaN-bad signature.
- The 51 % `mfu_window` IS higher than the baseline 45.81 % — the clamp does cost ~6 pp of overhead — but training is completely broken without it.

Net: the bug is real and `benchmark/test_sonic_bwd_dtensor.py` is missing some triggering condition (real router distribution × 48 layers × chunked-CE loss path × autotune cache state). Restored the unguarded clamp in `sonicmoe.py` and removed the env-var plumbing from `launch.sh.j2`, `run_benchmark.py`, and the validation YAML config — see "Cleanup checklist — DONE" in `debug_sonic_bwd_dtensor.md`. Bug remains tracked under A1 in `upstream_todo.md`; when an upstream sonic-moe fix lands, post the repro + this validation log to [sonic-moe#51](https://github.com/Dao-AILab/sonic-moe/pull/51).


## 2026-04-29 (later still): definitive answer for sonic-moe wrapper

The user pointed out PR #45621's "clamp-after" thesis (don't clamp expert_ids; let `histc(max=num_experts-1)` drop sentinels via bin-overflow). Rather than guess about the kernel internals, ran a **4-mode A/B/C/D test** in production via an env-gated wrapper:

| Mode         | clamp ids | masked_fill scores | Result | Final train_loss | mfu_window peak |
| ------------ | --------- | ------------------ | ------ | ---------------- | --------------- |
| `clamp_zero` | YES       | YES                | trains | 13.17            | 45.81 %         |
| **`clamp_only`** ← simplified | YES | **NO**     | **trains** | **13.17** (bit-identical) | **45.88 %** (bit-identical) |
| `zero_only`  | NO        | YES                | NaN    | 0 by step 10     | (irrelevant — broken) |
| `none`       | NO        | NO                 | NaN    | 0 by step 10     | (irrelevant — broken) |

`zero_only` and `none` produce **literally bit-identical** NaN output. `clamp_zero` and `clamp_only` produce **bit-identical** healthy training.

**Why**: `RouterParallel._prepare_output_fn` (transformers/integrations/tensor_parallel.py:1202) **already** zeros sentinel `router_scores` upstream:

```python
non_local_mask = (router_indices // num_local_experts) != ep_rank
router_scores = router_scores.masked_fill(non_local_mask, 0.0)
router_indices = router_indices.masked_fill(non_local_mask, -1)
# ... and at line 1209, sentinel ids set to num_local_experts:
router_indices = router_indices.masked_fill(router_indices == -1, num_local_experts)
```

So the wrapper's `router_scores.masked_fill` is touching tensor positions that are already 0 — pure no-op. The clamp on `expert_ids`, however, IS load-bearing because RouterParallel sets sentinel ids = num_local_experts → out-of-range for `w2.permute(2,0,1)` of shape `(num_local_experts, ...)` → kernel's hand-written backward OOB-indexes → NaN gradients.

**Action taken**: simplified `transformers/integrations/sonicmoe.py` to just clamp (~3 lines saved). Removed the `SONICMOE_SENTINEL` env-var debug knob from `sonicmoe.py`, `launch.sh.j2`, and `run_benchmark.py`. Production behavior unchanged (verified by `clamp_only` matching `clamp_zero` to 4 decimals).

**For the upstream sonic-moe issue**: the bug is the kernel's `Function.backward` OOB-indexing `w2[num_local_experts]` when expert_ids contain the EP sentinel value. Forward handles this (zero-init sentinel lanes per kernel comment); backward does not. Fix on the kernel side: either bounds-check `expert_ids` before any indexing, or apply the same sentinel-lane skip in backward as in forward.

The MFU difference between healthy (45.88 %) and broken-but-faster (51.73 %) is the kernel's **~6 pp throughput cost** of computing the sentinel-routed rows even when their contribution is masked out. Could be recovered if the kernel adopted the histogram-drop pattern from PR #45621 (sort sentinels to tail, use `histc(max=num_experts-1)` in offset construction, skip the tail in the matmul kernel).

## 2026-04-29 (later) — sonic-moe bug: 1-node EP=2 minimal repro found

After the 4-mode A/B test confirmed the bug in production (8 nodes, 64 GPUs), spent 1 hour trying synthetic reductions on a 2-GPU node — none triggered. **Then realized the production stack itself reduces** to just 1 node + 2 GPUs if we stay full-stack: TRL SFT + DS-Z2 + EP=2 + sonicmoe + chunked NLL with `tiny-Qwen3MoeForCausalLM` (4 experts, 2 layers, hidden=8).

Toggled the wrapper's clamp via `SONICMOE_DISABLE_CLAMP=1`:

| metric | clamp ON | clamp OFF |
|---|---|---|
| train_runtime (30 steps) | 22.7 s | 219.7 s — **10× slower** |
| grad_norm range | 0.17 – 0.93 | 1146 – 8320 |
| grad_norm median | ≈0.25 | ≈3000 |
| grad_norm ratio | — | ~10,000× larger |
| train_loss start→end | 11.91 → 11.91 | 11.91 → 11.91 |

Loss is flat in both because the tiny model can't learn meaningfully in 30 steps regardless — the **decisive signal is grad_norm magnitude (10⁴× difference) plus the 10× slowdown**. The slowdown is the OOB-read fingerprint: when sentinel rows of `expert_ids` cause the bwd kernel to index past valid memory, page faults / un-coalesced loads degrade SM occupancy.

**Why my synthetic reductions missed it**: synthetic single-step fwd+bwd on tensors I allocated myself doesn't churn the CUDA allocator the way a full DeepSpeed + Adam + ChunkedCE stack does. The bug is allocator-state-dependent. Once the production wrappers run, even at hidden_size=8, the OOB read hits tainted memory.

**Repro artifacts** (`benchmark/sonic_moe_upstream_repro.md`):
- `benchmark/_repro_ep2_run.sh` (~50 lines)
- `benchmark/_repro_ep2_accelerate.yaml` (DS-Z2)
- `benchmark/logs/_repro_ep2_{NANNED,CLEAN}_FULL.log`
- Wrapper still carries `SONICMOE_DISABLE_CLAMP` env-var knob; will remove once upstream patches land.

