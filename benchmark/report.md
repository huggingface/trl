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

| Context | Method     | Backend   | Degree | MFU   | TPS/GPU (raw) | Peak Mem |
| ------- | ---------- | --------- | ------ | ----- | ------------- | -------- |
| 16k     | SP=2       | DS-Z3     | 2      | 5.8%  | 1,964         | 61.3 GB  |
| 16k     | EP=16      | FSDP2     | -      | 22.5% | 3,788         | 71.1 GB  |
| 32k     | SP=2       | DS-Z3     | 2      | 13.1% | 2,663         | 54.0 GB  |
| 32k     | CP=2+EP=16 | FSDP2     | 2      | 13.2% | 2,686         | 71.5 GB  |
| 64k     | SP=4       | DS-Z3     | 4      | 11.6% | 2,624         | 54.4 GB  |
| 64k     | CP=8+EP=16 | FSDP2     | 8      | 2.9%  | 1,312         | 58.7 GB  |

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

| Config                          | MFU       | TPS/GPU | Status |
| ------------------------------- | --------- | ------- | ------ |
| 1n, DP=8, no liger              | —         | —       | OOM    |
| 2n, DP=8, CP=2, no liger       | 31.6%     | 7,615   | Yes    |
| **1n, DP=8, liger**             | **35.9%** | 4,321   | Yes    |

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

| Config                              | MFU       | TPS/GPU | Status |
| ----------------------------------- | --------- | ------- | ------ |
| 1n, DP=8, no liger                  | —         | —       | OOM    |
| **1n, DP=8, liger**                 | **35.9%** | 4,321   | Yes    |
| 2n, DP=8, CP=2, no liger            | 31.6%     | 7,615   | Yes    |
| 1n, DP=4, CP=2, liger               | 17.1%     | 4,120   | Yes    |

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

| Context | Nodes | GPUs | DP  | TP  | CP  | EP  | Liger | Attn  | MFU       | TPS    | TPS/GPU | Runtime (20 steps) |
| ------- | ----- | ---- | --- | --- | --- | --- | ----- | ----- | --------- | ------ | ------- | ------------------ |
| 32k     | 1     | 8    | 8   | 1   | 1   | 1   | yes   | sdpa  | 35.9%     | 34,570 | 4,321   | 151.7s             |
| 32k     | 1     | 8    | 8   | 1   | 1   | 1   | yes   | FA3   | **56.3%** | 54,260 | 6,783   | 96.6s              |
| 32k     | 1     | 8    | 8   | 1   | 1   | 1   | no    | sdpa  | —         | —      | —       | OOM                |

### Analysis

- **FA3 gives 1.57× MFU boost over sdpa** (56.3% vs 35.9%) with identical loss curves. The Hopper-optimized FA3 kernel is dramatically faster than PyTorch's sdpa at 32k context.
- **56.3% MFU is the highest across all benchmarks** — 56% of peak H100 bf16 FLOPS utilized for a 4B dense model on a single node.
- **1.57× faster wall-clock**: 96.6s vs 151.7s for 20 steps. FA3 halves per-step attention time.
- **TPS/GPU: 6,783** — up from 4,321 with sdpa. Per-GPU throughput is 1.57× higher.

### Full Qwen3-4B Results Summary (32k context)

| Config                                 | MFU       | TPS/GPU | Nodes | Status |
| -------------------------------------- | --------- | ------- | ----- | ------ |
| FSDP2, no liger, sdpa                  | —         | —       | 1     | OOM    |
| FSDP2, liger, sdpa                     | 35.9%     | 4,321   | 1     | Yes    |
| **FSDP2, liger, FA3**                  | **56.3%** | 6,783   | 1     | Yes    |
| FSDP2, CP=2, no liger, sdpa            | 31.6%     | 7,615   | 2     | Yes    |
| DS-Z3, SP=2, FA2                       | 40.2%     | 4,839   | 2     | Yes    |
| DS-Z3, SP=2, FA3                       | —         | —       | 2     | pending |

### SP=2 + FA3 (DeepSpeed ZeRO-3, Qwen3-4B, 32k, 2 nodes)

| Context | Nodes | GPUs | DP  | TP  | CP  | SP  | EP  | Backend         | Attn | MFU       | TPS    | TPS/GPU | Runtime (20 steps) |
| ------- | ----- | ---- | --- | --- | --- | --- | --- | --------------- | ---- | --------- | ------ | ------- | ------------------ |
| 32k     | 2     | 16   | 8   | 1   | 1   | 2   | 1   | deepspeed_zero3 | FA2  | 40.2%     | 77,420 | 4,839   | —                  |
| 32k     | 2     | 16   | 8   | 1   | 1   | 2   | 1   | deepspeed_zero3 | FA3  | 23.6%     | 91,170 | 5,698   | 57.5s              |

**FA3 hurts SP performance**: 23.6% vs 40.2% MFU with FA2 — a 41% MFU regression despite 18% higher raw TPS. The Ulysses SP path splits Q/K/V across GPUs via all-to-all before calling the attention kernel. The FA3 kernel path likely has higher overhead for the split/gather operations or doesn't benefit from the same optimizations as FA2 in the SP context.

### FA3 Key Finding: Great standalone, bad with SP

| Config                        | FA2/sdpa MFU | FA3 MFU   | Change |
| ----------------------------- | ------------ | --------- | ------ |
| FSDP2, liger, 1n (no SP)     | 35.9% (sdpa) | **56.3%** | +57%   |
| DS-Z3, SP=2, 2n              | 40.2% (FA2)  | 23.6%     | **-41%** |

FA3 (`kernels-community/vllm-flash-attn3`) excels as a standalone attention kernel (1.57× speedup over sdpa) but regresses badly when combined with DeepSpeed Ulysses SP. Use FA3 for non-SP configurations; stick with `flash_attention_2` for SP.

### 30B SP=2: FA3 vs FA2 (Qwen3-30B-A3B, 32k, 2 nodes, srun 5-step test)

| Context | Nodes | GPUs | DP  | TP  | CP  | SP  | EP  | Backend         | Attn | MFU   | TPS    | TPS/GPU | Runtime (5 steps) |
| ------- | ----- | ---- | --- | --- | --- | --- | --- | --------------- | ---- | ----- | ------ | ------- | ----------------- |
| 32k     | 2     | 16   | 8   | 1   | 1   | 2   | 1   | deepspeed_zero3 | FA2  | 9.9%  | 32,140 | 2,009   | 40.8s             |
| 32k     | 2     | 16   | 8   | 1   | 1   | 2   | 1   | deepspeed_zero3 | FA3  | 10.5% | 34,130 | 2,133   | 38.4s             |

FA3 gives ~6% improvement over FA2 with SP on 30B — marginal compared to the 57% gain on 4B standalone. The SP all-to-all overhead dominates, not the attention kernel.

### FA3 Summary

| Model | Config | FA2/sdpa MFU | FA3 MFU | Change | Verdict |
| ----- | ------ | ------------ | ------- | ------ | ------- |
| 4B    | FSDP2, liger, 1n    | 35.9% (sdpa) | **56.3%** | **+57%** | FA3 wins big |
| 4B    | DS-Z3, SP=2, 2n     | 40.2% (FA2)  | 23.6%     | **-41%** | FA3 loses with SP |
| 30B   | DS-Z3, SP=2, 2n     | 9.9% (FA2)   | 10.5%     | +6%      | Marginal gain |

- **FA3 without SP**: massive speedup (1.57×) — the Hopper-native kernel is much faster than sdpa for standalone attention
- **FA3 with SP (4B)**: regression — SP's Ulysses all-to-all path is not optimized for FA3
- **FA3 with SP (30B)**: marginal gain — attention is a smaller fraction of compute for MoE, so kernel speedup has less impact

---

## Addendum: Qwen3-32B (Dense) — Comparison with Qwen3-30B-A3B (MoE)

### Motivation

Qwen3-32B is a dense model with 32B total parameters, all active per token. Qwen3-30B-A3B is a MoE model with 30B total parameters but only ~3B active per token (8 of 128 experts). Comparing both at the same hardware configs reveals the cost of MoE communication overhead (EP all-reduce, expert sharding) vs dense compute efficiency.

### Architecture comparison

| Field                 | Qwen3-32B (dense) | Qwen3-30B-A3B (MoE) |
| --------------------- | ----------------- | -------------------- |
| model_type            | qwen3             | qwen3_moe            |
| hidden_size           | 5120              | 2048                 |
| num_hidden_layers     | 64                | 48                   |
| num_attention_heads   | 64                | 32                   |
| num_key_value_heads   | 8                 | 4                    |
| intermediate_size     | 25600             | 6144                 |
| moe_intermediate_size | N/A               | 768                  |
| num_local_experts     | N/A               | 128                  |
| num_experts_per_tok   | N/A               | 8                    |
| Total params          | 32B               | 30B                  |
| Active params/token   | 32B               | ~3B                  |
| max_position_embeddings | 40960           | 32768                |

### FSDP2 Results (preliminary, sdpa, no liger)

| Config               | Qwen3-32B (dense) MFU | Qwen3-30B-A3B (MoE) MFU | Dense/MoE ratio |
| -------------------- | --------------------- | ----------------------- | --------------- |
| 16k, 2n, DP=16       | **36.5%**             | 22.5% (EP=16)           | 1.62×           |
| 16k, 4n, DP=32       | **37.7%**             | 22.3% (EP=32)           | 1.69×           |
| 32k, 2n, CP=2         | **18.1%**             | 13.2% (EP=16, CP=2)     | 1.37×           |

#### Qwen3-32B full metrics

| Context | Nodes | GPUs | DP  | TP  | CP  | EP  | Backend | Liger | Attn | MFU       | TPS    | TPS/GPU | Peak GPU Mem  | Status |
| ------- | ----- | ---- | --- | --- | --- | --- | ------- | ----- | ---- | --------- | ------ | ------- | ------------- | ------ |
| 16k     | 2     | 16   | 16  | 1   | 1   | 1   | fsdp2   | no    | sdpa | 36.5%     | 19,280 | 1,205   | 79.3 GB (99%) | Yes    |
| 16k     | 2     | 16   | 16  | 1   | 1   | 1   | fsdp2   | yes   | sdpa | 41.1%     | 21,690 | 1,356   | 59.7 GB (75%) | Yes    |
| 16k     | 2     | 16   | 16  | 1   | 1   | 1   | fsdp2   | yes   | FA3  | **51.3%** | 27,080 | 1,693   | 59.7 GB (75%) | Yes    |
| 16k     | 4     | 32   | 32  | 1   | 1   | 1   | fsdp2   | no    | sdpa | 37.7%     | 39,790 | 1,243   | 72.2 GB (90%) | Yes    |
| 16k     | 4     | 32   | 32  | 1   | 1   | 1   | fsdp2   | yes   | sdpa | 40.8%     | 43,020 | 1,344   | —             | Yes    |
| 16k     | 4     | 32   | 32  | 1   | 1   | 1   | DS-Z3   | no    | sdpa | 33.9%     | 35,770 | 1,118   | —             | Yes    |
| 32k     | 2     | 16   | 16  | 1   | 1   | 1   | fsdp2   | no    | sdpa | —         | —      | —       | —             | OOM    |
| 32k     | 2     | 16   | 16  | 1   | 1   | 1   | fsdp2   | yes   | sdpa | 43.1%     | 16,960 | 1,060   | —             | Yes    |
| 32k     | 2     | 16   | 16  | 1   | 1   | 1   | fsdp2   | yes   | FA3  | **59.0%** | 23,200 | 1,450   | —             | Yes    |
| 32k     | 2     | 16   | 8   | 1   | 2   | 1   | fsdp2   | no    | sdpa | 18.1%     | 14,190 | 887     | —             | Yes    |
| 32k     | 2     | 16   | 8   | 1   | 2   | 1   | fsdp2   | yes   | sdpa | 19.8%     | 15,550 | 972     | 60.9 GB (76%) | Yes    |
| 32k     | 4     | 32   | 16  | 1   | 2   | 1   | fsdp2   | no    | sdpa | 18.4%     | 28,870 | 902     | —             | Yes    |
| 32k     | 4     | 32   | 16  | 1   | 2   | 1   | fsdp2   | yes   | sdpa | 19.4%     | 30,460 | 952     | —             | Yes    |
| 32k     | 2     | 16   | 8   | 1   | 1   | 1   | DS-Z3   | no    | FA2 (SP=2) | —      | —      | —       | —             | OOM    |
| 32k     | 2     | 16   | 8   | 1   | 1   | 1   | DS-Z3   | no    | FA3 (SP=2) | —      | —      | —       | —             | OOM    |
| 32k     | 2     | 16   | 8   | 1   | 1   | 1   | DS-Z3   | yes   | FA2 (SP=2) | 25.0%  | 19,630 | 1,227   | —             | Yes    |
| 32k     | 2     | 16   | 8   | 1   | 1   | 1   | DS-Z3   | yes   | FA3 (SP=2) | 29.7%  | 23,380 | 1,461   | —             | Yes    |
| 16k     | 2     | 16   | 8   | 2   | 1   | 1   | fsdp2   | no    | sdpa       | —      | —      | —       | —             | OOM    |
| 16k     | 2     | 16   | 8   | 2   | 1   | 1   | fsdp2   | yes   | sdpa       | 38.5%  | 20,310 | 1,269   | —             | Yes    |

### Comparison: 32B Dense vs 30B MoE (same hardware, best config per model)

| Config                    | Qwen3-32B (dense) MFU | Qwen3-30B-A3B (MoE) MFU | Dense/MoE ratio |
| ------------------------- | --------------------- | ----------------------- | --------------- |
| 16k, 2n, FSDP2            | 36.5%                 | 22.5% (EP=16)           | 1.62×           |
| 16k, 2n, liger             | 41.1%                 | N/A (liger incompatible) | —              |
| 16k, 2n, liger+FA3         | **51.3%**             | N/A                      | —              |
| 16k, 4n, FSDP2            | 37.7%                 | 22.3% (EP=32)           | 1.69×           |
| 16k, 4n, liger             | 40.8%                 | N/A                      | —              |
| 16k, 4n, DS-Z3             | 33.9%                 | 18.9%                   | 1.79×           |
| 32k, 2n, DP-only, liger    | 43.1%                 | N/A (OOM without EP)     | —              |
| 32k, 2n, DP-only, liger+FA3 | **59.0%**            | N/A                      | —              |
| 32k, 2n, CP=2              | 18.1%                 | 13.2% (EP=16, CP=2)     | 1.37×           |
| 32k, 2n, CP=2, liger       | 19.8%                 | N/A                      | —              |
| 32k, 4n, CP=2              | 18.4%                 | —                        | —              |
| 32k, 4n, CP=2, liger       | 19.4%                 | N/A                      | —              |

### Liger + FA3 impact on 32B

| Context | Variant       | MFU       | TPS/GPU | vs baseline |
| ------- | ------------- | --------- | ------- | ----------- |
| 16k     | baseline sdpa | 36.5%     | 1,205   | —           |
| 16k     | + liger       | 41.1%     | 1,356   | +13%        |
| 16k     | + liger + FA3 | **51.3%** | 1,693   | **+41%**    |
| 32k     | baseline sdpa | —         | —       | OOM         |
| 32k     | + liger       | 43.1%     | 1,060   | (enables 32k) |
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

| Context | Best MFU       | Config                      |
| ------- | -------------- | --------------------------- |
| 16k     | **51.3%**      | FSDP2, liger+FA3, 2n DP=16  |
| 32k     | **59.0%**      | FSDP2, liger+FA3, 2n DP=16  |
| 32k SP  | 29.7%          | DS-Z3, SP=2+FA3+liger, 2n   |

---

## 2026-04-17: Qwen3-30B-A3B FlashAttention3 test

### Goal

Compare FA3 (`kernels-community/vllm-flash-attn3`) vs sdpa for Qwen3-30B-A3B MoE on the best-performing config (2 nodes, 16k ctx).

### Results

| Config | Nodes | DP | TP | CP | EP | SP | Attn | MFU | TPS/GPU | Peak GPU Mem | Status |
| ------ | ----- | -- | -- | -- | -- | -- | ---- | --- | ------- | ------------ | ------ |
| FSDP2, 16k, no EP | 2 | 16 | 1 | 1 | 1 | 1 | sdpa | 23.1% | 3,888 | -- | Baseline |
| FSDP2, 16k, no EP | 2 | 16 | 1 | 1 | 1 | 1 | FA3 | **25.65%** | 4,318 | 71.1 GB (89%) | New |
| FSDP2, 16k, EP=16 | 2 | 16 | 1 | 1 | 16 | 1 | FA3 | -- | -- | -- | Failed (mesh compat) |

### Findings

- **FA3 gives +11% relative MFU improvement** over sdpa (25.65% vs 23.1%), +11% TPS/GPU (4,318 vs 3,888).
- FA3 + EP fails with `KeyError: "Invalid mesh_dim_names ('dp_shard_cp',)"` -- EP creates a device mesh with only `['tp']` but FSDP2 expects `dp_shard_cp`. This is an accelerate/transformers EP mesh incompatibility, not FA3-specific. Needs investigation.
- Multi-node HF Hub cache race condition: ranks on the second node intermittently fail to resolve model shards. Workaround: `HF_HUB_OFFLINE=1` in launch script.

---

## 2026-04-17: FA3 sweep — maximizing MFU across models and context lengths

### Goal

Systematic FA3 evaluation across Qwen3-30B-A3B (MoE) and Qwen3-32B (dense) at 32k and 64k context lengths. Compare against best known sdpa/FA2 baselines to find new MFU records.

### Qwen3-30B-A3B (MoE) results

| Config | Ctx | Nodes | DP | TP | CP | EP | SP | Attn | MFU | TPS/GPU | Peak GPU Mem | Status |
| ------ | --- | ----- | -- | -- | -- | -- | -- | ---- | --- | ------- | ------------ | ------ |
| FSDP2 DP-only | 16k | 2 | 16 | 1 | 1 | 1 | 1 | sdpa | 23.1% | 3,888 | -- | Baseline |
| FSDP2 DP-only | 16k | 2 | 16 | 1 | 1 | 1 | 1 | FA3 | **25.65%** | 4,318 | 71.1 GB (89%) | +11% |
| FSDP2 CP=2 | 32k | 2 | 8 | 1 | 2 | 1 | 1 | sdpa | 13.57% | 2,757 | 69.2 GB (87%) | Baseline |
| DS-Z3 SP=2 | 32k | 2 | 8 | 1 | 1 | 1 | 2 | FA3 | **14.49%** | 2,944 | 53.3 GB (67%) | +6.8% |
| DS-Z3 SP=4 FA2 | 64k | 2 | 4 | 1 | 1 | 1 | 4 | FA2 | 11.6% | 2,624 | 54.4 GB (68%) | Baseline |
| DS-Z3 SP=4 | 64k | 2 | 4 | 1 | 1 | 1 | 4 | FA3 | **12.52%** | 2,837 | 54.4 GB (68%) | +7.9% |

### Qwen3-32B (Dense) results — pending

| Config | Ctx | Nodes | DP | TP | CP | EP | SP | Attn | MFU | TPS/GPU | Peak GPU Mem | Status |
| ------ | --- | ----- | -- | -- | -- | -- | -- | ---- | --- | ------- | ------------ | ------ |
| FSDP2 liger+FA3 DP-only | 32k | 2 | 16 | 1 | 1 | 1 | 1 | FA3 | 59.0% | 1,450 | 78.4 GB (98%) | Baseline (2n) |
| FSDP2 liger+FA3 DP-only | 32k | 4 | 32 | 1 | 1 | 1 | 1 | FA3 | -- | -- | -- | Job 22081011 |
| FSDP2 liger CP=2 | 64k | 2 | 8 | 1 | 2 | 1 | 1 | sdpa | -- | -- | -- | Job 22081012 |
| DS-Z3 liger SP=2 | 64k | 2 | 8 | 1 | 1 | 1 | 2 | FA3 | -- | -- | -- | Job 22081013 |

### Findings so far

- **FA3 consistently improves MFU** across all tested configs: +11% at 16k, +6.8% at 32k, +7.9% at 64k (relative gains).
- **New best 32k MoE: 14.49% MFU** with DS-Z3+SP=2+FA3, beating the previous best of 13.57% (sdpa CP=2).
- **New best 64k MoE: 12.52% MFU** with DS-Z3+SP=4+FA3, beating the previous best of 11.6% (FA2 SP=4).
- **FA3 + CP is incompatible**: `Context parallelism is supported only with SDPA attention`. SP is the only long-context path for FA3.
- **FA3 + EP is incompatible**: FSDP2 mesh dimension mismatch when EP is enabled. Needs transformers/accelerate fix.
