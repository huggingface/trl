# SFT Benchmark Report

## Qwen3-4B (Dense)

**Backend:** FSDP2
**Status:** Working

No special configuration needed. Standard FSDP2 with `TRANSFORMER_BASED_WRAP` works out of the box.

### Issues encountered and fixes

- **`cast_forward_inputs` crash with OpenThoughts3 dataset:** The `open-thoughts/OpenThoughts3-1.2M` dataset uses `from`/`value` conversation format which triggers a code path incompatible with FSDP2. Switching to `THUDM/LongAlign-10k` (standard `role`/`content` format) resolved this. Standard conversational and language modeling datasets work fine.

- **CP `seq_length % (cp_size * 2) == 0` assertion:** Context Parallelism requires fixed sequence lengths divisible by `cp_size * 2`. The default `bfd` packing strategy can produce variable-length packed sequences. Fixed by using `--packing_strategy wrapped` which always fills to exactly `max_length`.

- **TPS overcounting with CP:** `num_input_tokens_seen` in transformers' `_inner_training_loop` counts tokens **before** CP splits the inputs, then `gather().sum()` sums across all ranks including CP ranks. This inflates TPS by a factor of `cp_size`. We correct for this in the MFU computation. TODO: fix upstream in transformers.

---

## Qwen3-30B-A3B (MoE, 128 experts)

**Backend:** DeepSpeed ZeRO-3 (not FSDP2)
**Status:** Working with DeepSpeed

### Why not FSDP2

FSDP2 fundamentally does not work with MoE models. We hit a **collective shape mismatch** error:

```
RuntimeError: Detected mismatch between collectives on ranks.
Rank 12: _REDUCE_SCATTER_BASE, TensorShape=[505155840]
Rank 0:  _REDUCE_SCATTER_BASE, TensorShape=[240914688]
```

**Root cause:** FSDP2's `TRANSFORMER_BASED_WRAP` wraps each `Qwen3MoeDecoderLayer`, which contains a `Qwen3MoeSparseMoeBlock` with 128 experts. During the backward pass, MoE routing activates different experts on different ranks (data-dependent routing). FSDP2's `reduce_scatter` expects all ranks to have identical tensor shapes for the same collective, but different active experts produce different gradient tensor sizes.

We tried:
- Removing `TRANSFORMER_BASED_WRAP` → OOM (no sub-module sharding, full model on each GPU)
- `cpu_ram_efficient_loading: true` vs `false` → same mismatch
- Single node vs multi-node → same mismatch
- All combinations of 4 and 8 nodes → same mismatch

This is a fundamental FSDP2 + MoE incompatibility. FSDP2 requires symmetric collectives across all ranks, which MoE routing breaks. Needs EP (Expert Parallelism) support which is not yet available in accelerate's `ParallelismConfig`.

### Why DeepSpeed ZeRO-3 works

DeepSpeed ZeRO-3 handles MoE correctly because it shards parameters at the optimizer/gradient level, not at the module level. All ranks maintain the same parameter structure — ZeRO-3 partitions optimizer states and gradients uniformly regardless of which experts were activated. The allgather/reduce-scatter operations are on full parameter tensors, not data-dependent subsets.

### Additional configuration for MoE

- **Gradient checkpointing:** MoE models with non-reentrant checkpointing (`use_reentrant=False`, TRL's default) cause `CheckpointError: Recomputed values have different metadata`. The reentrant variant works but requires `--gradient_checkpointing_kwargs '{"use_reentrant": true}'`. This is only needed for MoE models — dense models work with non-reentrant (the default).

- **No EP available:** Accelerate's `ParallelismConfig` does not include `ep_size`. Transformers has experimental `DistributedConfig(enable_expert_parallel=True)` support but Qwen3-30B-A3B doesn't have an `ep_plan` defined. The reference benchmarks used FSDP2+EP (via a different framework) which we cannot replicate. Our runs use pure DP with DeepSpeed ZeRO-3, plus Ulysses SP for sequence parallelism.

- **No CP with DeepSpeed:** DeepSpeed uses Ulysses sequence parallelism (SP) instead of ring-attention context parallelism (CP). SP is configured via `parallelism_config_sp_size` and `parallelism_config_sp_backend: deepspeed`.

---

## General notes

- **Dataset:** `THUDM/LongAlign-10k` — median ~12k tokens, 30% samples >16k, 10% >32k. Good for long-context benchmarking.
- **Packing:** `--packing --packing_strategy wrapped` required for CP compatibility (fixed sequence lengths).
- **MFU computation:** Added `compute_flops_per_token()` and `compute_mfu()` to `trl/trainer/utils.py`. Accounts for dense vs MoE architectures (active expert FLOPs only). Peak FLOPS default: H100 SXM 989.5 TFLOPS bf16.
- **NCCL config:** Multi-node runs require `--mem=0`, `--cpus-per-task=64`, `--gpus-per-node=8`, `NCCL_IB_DISABLE=0` in sbatch for proper inter-node communication.
