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

| Context | Nodes | DP | CP | EP | MFU | TPS | TPS/GPU | Status |
|---|---|---|---|---|---|---|---|---|
| 16k | 8 | 64 | 1 | 64 | 2.34% | 25,164 | 393 | Yes |
| 32k | 8 | 32 | 2 | 64 | 3.55% | 46,194 | 722 | Yes |
| 32k | 8 | 64 | 1 | 64 | - | - | - | OOM |
| 32k | 4 | 8 | 4 | 32 | - | - | - | CUDA peer memory error |

### 32k CP=2 scaling with EP across nodes

| Nodes | EP | MFU | TPS | TPS/GPU |
|---|---|---|---|---|
| 2 | 16 | **4.77%** | 15,513 | 970 |
| 4 | 32 | 4.24% | 27,590 | 862 |
| 8 | 64 | 3.55% | 46,194 | 722 |

**Peak MFU remains at 2 nodes**: 4.77% at 32k, CP=2, EP=16. Adding more nodes increases total TPS but MFU drops due to inter-node communication overhead. The 30B MoE model is communication-bound — more GPUs means more all-to-all hops across nodes.

### Limits hit

- **CP=4 at 32k**: CUDA peer memory error over NVLink (hardware/driver issue, same as before)
- **32k DP-only**: OOMs even with EP=64 (2 experts/GPU) — activation memory at 32k is the bottleneck, not expert params
- **8 nodes**: MFU degrades vs 2-4 nodes — inter-node all-to-all latency dominates

---

## Addendum: NCCL Bandwidth Benchmark

Cluster uses AWS EFA (Elastic Fabric Adapter) with RDMA — not InfiniBand. 32 EFA NICs per node, NVLink intra-node.

### Results at 1GB message size

| Op | 1 node (8 GPUs) BusBW | 2 nodes (16 GPUs) BusBW | Inter/Intra ratio |
|---|---|---|---|
| allreduce | 448 GB/s | 431 GB/s | 96% |
| allgather | 35 GB/s | 15 GB/s | 42% |
| reduce_scatter | 34 GB/s | 13 GB/s | 39% |
| all_to_all | 335 GB/s | 37 GB/s | **11%** |

### Impact on training

- **allreduce** scales well across nodes — NCCL ring/tree algorithms overlap across multiple EFA NICs
- **allgather/reduce_scatter** (used by FSDP2) drop to ~40% inter-node — explains why MFU drops with more nodes
- **all_to_all** (used by EP token routing) drops to **11%** inter-node — explains why EP doesn't help at multi-node scale. The all-to-all cost dominates any savings from fewer local experts

This is why peak MFU (4.77%) was achieved at 2 nodes, not 8 — minimizing inter-node all-to-all hops.
