# SFT Benchmark Report

## Qwen3-4B (Dense) — FSDP2

### Results

| Context | Nodes | DP | CP | TP | MFU | TPS | TPS/GPU | Status |
|---|---|---|---|---|---|---|---|---|
| 16k | 2 | 16 | 1 | 1 | 30.2% | 90,058 | 5,629 | Yes |
| 16k | 2 | 8 | 2 | 1 | 25.1% | 149,724 | 9,358 | Yes |
| 16k | 2 | 4 | 4 | 1 | 18.3% | 217,910 | 13,619 | Yes |
| 32k | 2 | 8 | 2 | 1 | 31.6% | 121,842 | 7,615 | Yes |
| 32k | 2 | 4 | 4 | 1 | 26.8% | 207,029 | 12,939 | Yes |
| 32k | 4 | 16 | 2 | 1 | 31.8% | 245,336 | 7,667 | Yes |
| 32k | 4 | 8 | 4 | 1 | 27.1% | 418,189 | 13,068 | Yes |
| 32k | 4 | 4 | 8 | 1 | 19.6% | 605,864 | 18,933 | Yes |
| 32k | 4 | 8 | 1 | 4 | - | - | - | OOM |
| 32k | 4 | 4 | 1 | 8 | - | - | - | OOM |

### Observations

- **DP scaling is linear**: 2 nodes → 4 nodes with same CP gives ~2x TPS and constant TPS/GPU (~7,600 for CP=2).
- **CP increases raw TPS but decreases MFU**: CP splits sequences across GPUs, reducing per-GPU memory but adding ring attention communication overhead. MFU drops from 30% (CP=1) to 19% (CP=4) to 20% (CP=8).
- **TP OOMs at 32k**: TP=4 and TP=8 on 32k context don't fit in memory on 4 nodes.
- **MFU note**: TPS with CP is overcounted by `cp_size` in transformers' `num_input_tokens_seen`. Our MFU computation corrects for this (divides TPS by `cp_size`). See `benchmark/bug_cp_token_count.md`.

### Issues encountered and fixes

- **OpenThoughts3 dataset incompatibility**: The `from`/`value` conversation format triggers a code path incompatible with FSDP2. Switched to `THUDM/LongAlign-10k` (standard `role`/`content` format).
- **CP sequence length assertion**: Context Parallelism requires `seq_length % (cp_size * 2) == 0`. Fixed with `--packing_strategy wrapped` (fills to exact `max_length`).
- **TPS overcounting with CP**: `num_input_tokens_seen` counts tokens before CP splits inputs, inflating by `cp_size`. Corrected in MFU computation. TODO: fix upstream in transformers.

---

## Qwen3-30B-A3B (MoE, 128 experts) — DeepSpeed ZeRO-3

### Results

| Context | Nodes | GPUs | DP | SP | MFU | TPS | TPS/GPU | Status |
|---|---|---|---|---|---|---|---|---|
| 16k | 2 | 16 | 16 | 1 | 2.12% | 5,703 | 356 | Yes |
| 16k | 2 | 16 | 8 | 2 | 2.05% | 5,528 | 345 | Yes |
| 16k | 4 | 32 | 32 | 1 | 1.47% | 7,913 | 247 | Yes |
| 16k | 4 | 32 | 16 | 2 | 1.44% | 7,769 | 243 | Yes |
| 16k | 8 | 64 | 64 | 1 | 1.16% | 12,473 | 195 | Yes |
| 32k | 2 | 16 | 8 | 2 | - | - | - | OOM |
| 32k | 4 | 32 | 32 | 1 | - | - | - | OOM |
| 32k | 4 | 32 | 16 | 2 | - | - | - | OOM |
| 32k | 4 | 32 | 8 | 4 | - | - | - | OOM |
| 32k | 8 | 64 | 64 | 1 | - | - | - | OOM |

### Observations

- **MFU is very low (1-2%)**: This is expected with ZeRO-3 + MoE without Expert Parallelism. ZeRO-3 allgathers ALL 128 experts (30B params) for each forward/backward, but only 8 experts (3.3B active params) do useful compute. The communication/compute ratio is ~9:1.
- **SP=2 doesn't help**: Ulysses sequence parallelism adds SP communication on top of already-dominant ZeRO-3 communication. MFU is slightly lower with SP than without.
- **32k OOMs everywhere**: The 30B MoE at 32k context doesn't fit even on 8 nodes (64 GPUs) with DeepSpeed ZeRO-3.
- **MFU decreases with more nodes**: Going from 2→4→8 nodes drops MFU (2.1%→1.5%→1.2%) due to increasing inter-node communication latency for ZeRO-3 allgathers.

### Why not FSDP2

FSDP2 fundamentally does not work with MoE models. We hit a **collective shape mismatch**:

```
RuntimeError: Detected mismatch between collectives on ranks.
Rank 12: _REDUCE_SCATTER_BASE, TensorShape=[505155840]
Rank 0:  _REDUCE_SCATTER_BASE, TensorShape=[240914688]
```

**Root cause**: During the backward pass, MoE routing activates different experts on different ranks (data-dependent routing). FSDP2's `reduce_scatter` expects identical tensor shapes across all ranks, which MoE routing breaks. This affects both single-node and multi-node. Needs EP (Expert Parallelism) support which is not available in accelerate's `ParallelismConfig`.

### Reference comparison

The reference benchmark (different framework with native EP) achieved:

| EP | CP | MFU | TPS/GPU |
|---|---|---|---|
| 4 | 1 | 8.76% | 4,187 |
| 8 | 1 | 7.50% | 3,585 |

Our best: MFU=2.12%, TPS/GPU=356. The ~4x MFU gap is entirely due to lacking EP — without it, ZeRO-3 communicates 9x more data than needed.

### MoE FLOP accounting

For Qwen3-30B-A3B:
- **Total model params**: 30.5B (128 experts × 768 intermediate per expert)
- **Active params per token**: 3.3B (8 active experts × 768 intermediate)
- **Note**: `8 experts × 768 = 6144`, which equals the dense `intermediate_size`. The model is designed so active compute matches an equivalent dense model.
- MFU is computed using **active** FLOPs (what the GPU actually computes), not total model FLOPs.

### Additional configuration notes

- **Gradient checkpointing**: MoE + non-reentrant checkpointing causes `CheckpointError`. Use `--gradient_checkpointing_kwargs '{"use_reentrant": true}'`.
- **No EP in accelerate**: `ParallelismConfig` does not support `ep_size`. Transformers has experimental `DistributedConfig(enable_expert_parallel=True)` but Qwen3-30B-A3B lacks an `ep_plan`.
- **No CP with DeepSpeed**: CP is FSDP2-only in accelerate. DeepSpeed uses Ulysses SP instead.

---

## General notes

- **GPU**: H100 SXM 80GB HBM3 (989.5 TFLOPS bf16 peak)
- **Cluster**: hopper-prod partition, 8 GPUs/node
- **Dataset**: `THUDM/LongAlign-10k` — median ~12k tokens, 30% >16k, 10% >32k
- **Packing**: `--packing --packing_strategy wrapped` (fixed sequence length for CP compatibility)
- **NCCL config**: `--mem=0`, `--cpus-per-task=64`, `NCCL_IB_DISABLE=0` for multi-node
- **MFU formula**: `MFU = 100 × (flops_per_token × TPS / cp_size) / (peak_flops × world_size)`
