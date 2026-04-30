# trl - SFTTrainer benchmark

[https://aminedirohf-qwen3-sft-benchmark.static.hf.space/index.html](https://aminedirohf-qwen3-sft-benchmark.static.hf.space/index.html)

## Setup

### Cluster

- **Instance**: AWS p5.48xlarge
- **GPUs**: 8× NVIDIA H100 SXM 80GB per node (NVLink intra-node, 32× EFA inter-node)
- **Peak compute**: 989.5 TFLOPS bf16 per GPU
- **Inter-node bandwidth**: 3200 Gbps EFA (400 GB/s aggregate)

### Software

- **Framework**: TRL SFTTrainer (branch `benchmark-sft-moe`)
- **Backends**: FSDP2 (via accelerate 1.13.0) and DeepSpeed ZeRO-3
- **Transformers**: 5.6.0.dev0 (local fork with EP bug fixes)
- **PyTorch**: 2.10.0+cu128
- **Liger Kernel**: fused CrossEntropy + SwiGLU + RMSNorm + RoPE (Triton)
- **Flash Attention 3**: `kernels-community/vllm-flash-attn3` (Hopper-native FA3)

### Training Config

- **Dataset**: `THUDM/LongAlign-10k` - median ~12k tokens, 30% >16k, 10% >32k
- **Packing**: `-packing --packing_strategy wrapped` (fills to exact `max_length`)
- **Batch size**: 1 per device
- **Optimizer**: AdamW, lr=2e-5
- **Precision**: bf16
- **Gradient checkpointing**: enabled for all runs
- **Max steps**: 20 (logging every 5)

### MFU Formula

```
MFU = 100 × (flops_per_token × TPS / cp_size) / (peak_flops_per_gpu × world_size)
```

- `flops_per_token` counts matmul FLOPs for forward + backward (3x forward).
- For MoE, only the **active** experts are counted: `num_experts_per_tok * 3 matmuls * 2 * hidden * moe_intermediate` per MoE layer (8 of 128 experts for Qwen3 MoE). Attention FLOPs include Q/K/V/O projections and attention scores (`2 * n_heads * head_dim * seq_len`).
- TPS divided by `cp_size` (or `sp_size`) to correct for overcounting in `num_input_tokens_seen`.
- Peak GPU memory is collected from trackio system metrics (`gpu/X/allocated_memory`).
- **Two MFU variants are reported**: _cumulative_ (`total_tokens / total_runtime` since training start, the historical metric) and _window_ (`Δtokens / Δtime` over the last logging window only — added to `SFTTrainer.log` as `mfu_window`/`tps_window`). For kernels with non-trivial first-call cost (Triton, CuteDSL, inductor-compiled), cumulative under-reports steady-state by many points for ~50+ steps. **Window is the more accurate steady-state metric.** Most numbers in this report are 20-step cumulative; new sonicmoe section reports both.
- **Convention: non-causal attention FLOPs** — the formula treats every token as attending to the full `seq_len` (matches PaLM / Megatron / nanoGPT). Causal masking actually halves attention compute, so reported MFU is inflated relative to the causal-corrected convention (Llama 2/3, DeepSpeed Ulysses). See "MFU adjustment for causal masking" below.
- **GPU peak**: 989.5 TFLOPS bf16 dense (H100 SXM5). Verified with `nvidia-smi` returning "NVIDIA H100 80GB HBM3" — AWS p5.48xlarge is SXM5, not NVL.

#### MFU adjustment for causal masking (post-hoc, leaves raw column untouched)

Multiply the reported `MFU` by the **adjustment factor** below to get the causal-corrected number. Subtract or augment as a separate "adjusted_MFU" column — do not modify the original.

**Formula** (general, any decoder-only model):

```
adj_factor(seq_len) = (full_flops - L * 3 * 2 * n_heads * head_dim * seq_len) / full_flops
adjusted_MFU = MFU * adj_factor
```

where `full_flops = compute_flops_per_token(cfg, seq_len)` (the uncausal value used in the original calc), `L = num_hidden_layers`, `n_heads = num_attention_heads`, `head_dim = head_dim or hidden_size/n_heads`. The numerator subtracts the half of the attention-score FLOPs (`Q·K^T` + `attn·V` per layer, multiplied by ×3 for fwd+bwd, summed across layers) — that half disappears under causal masking.

**Lookup for Qwen3-30B-A3B** (`h=2048, L=48, n_heads=32, head_dim=128`):

| ctx       | reported FLOPs/token | causal-corrected FLOPs/token | adj_factor (×) |
| --------- | -------------------- | ---------------------------- | -------------- |
| 16k       | 58.77 GFLOPs          | 39.44 GFLOPs                  | **0.6711**     |
| 32k       | 97.43 GFLOPs          | 58.77 GFLOPs                  | **0.6032**     |
| 64k       | 174.74 GFLOPs         | 97.43 GFLOPs                  | **0.5576**     |
| 128k      | 329.35 GFLOPs         | 174.74 GFLOPs                 | **0.5305**     |
| 256k      | 638.59 GFLOPs         | 329.35 GFLOPs                 | **0.5158**     |
| 512k      | 1257.07 GFLOPs        | 638.59 GFLOPs                 | **0.5080**     |
| 1M        | 2494.02 GFLOPs        | 1257.07 GFLOPs                | **0.5040**     |

**Apply** to any reported MFU value: `adjusted_MFU = MFU * adj_factor`. Examples (Liger champions):

| ctx  | reported MFU | adj_factor | **adjusted MFU** |
| ---- | ------------ | ---------- | ---------------- |
| 16k  | 40.70 %      | 0.6711     | **27.3 %**       |
| 32k  | 56.62 %      | 0.6032     | **34.2 %**       |
| 64k  | 66.46 %      | 0.5576     | **37.1 %**       |
| 128k | 76.29 %      | 0.5305     | **40.5 %**       |
| 256k | 63.62 %      | 0.5158     | **32.8 %**       |
| 512k | 63.26 %      | 0.5080     | **32.1 %**       |
| 1M   | 62.33 %      | 0.5040     | **31.4 %**       |

**Quick reference — dependence on per-rank seq, NOT on parallelism**: the adjustment depends only on seq_len (the seq each token attends to in attention) and the model config. So for SP=N, the adjustment uses `total_ctx` if attention is computed before Ulysses gather (transformers does this), OR `total_ctx/N` per rank. For our SP runs we use `total_ctx` (which is what `compute_flops_per_token` is called with via `args.max_length`), so the factors above apply directly.

**Same lookup for the other models we benchmark** (compute once if needed):

| Model | h | L | n_heads | head_dim | adj_factor formula reduces to |
| ----- | -- | --- | ------- | -------- | ------------------------------ |
| Qwen3-4B (dense) | 2560 | 36 | 32 | 80 | similar shape; attention smaller fraction → higher adj_factor |
| Qwen3-32B (dense) | 5120 | 64 | 64 | 80 | similar |
| Qwen3-235B-A22B (MoE) | 4096 | 94 | 64 | 128 | bigger; attention larger fraction → lower adj_factor at long ctx |

---

## Changes from Transformers 4.57.6 to 5.6.0

### The problem with 4.57.6: manual fused checkpoint + TRL workarounds

Qwen3-30B-A3B stores each of its 128 experts as separate `nn.Linear` modules (384 small weight matrices per layer). This layout was incompatible with FSDP2 (asymmetric gradients across ranks → collective shape mismatch) and with transformers' EP system (which expects a single `[num_experts, ...]` tensor to slice across ranks).

To make training work on 4.57.6, we had to:

- **Create a fused checkpoint** ([aminediroHF/Qwen3-30B-A3B-fused](https://huggingface.co/aminediroHF/Qwen3-30B-A3B-fused)): a conversion script stacked all 128 per-expert weights into fused tensors. Gate and up projections remained separate.
- **Add `fuse_moe_experts()` to TRL**: a runtime function that fuses `ModuleList[nn.Linear]` into grouped `nn.Parameter` tensors, patching the forward to use indexed `F.linear` calls..
- **Fork transformers** with `Qwen3MoeExperts` / `Qwen3MoeRouter` classes +`base_model_ep_plan` for EP.
- **Create a 1D device mesh** in SFT trainer and pass it to `from_pretrained`, since accelerate doesn't expose one.

### What transformers 5.6.0 eliminated

The new transformers handles all of this natively:

- **Automatic `WeightConverter`**: `from_pretrained` converts per-expert checkpoint keys to fused tensors at load time - no pre-converted checkpoint needed. The original `Qwen/Qwen3-30B-A3B` HuggingFace checkpoint works directly.
- **`Qwen3MoeExperts`** with fused `nn.Parameter` weights built into the model architecture.
- **Fused `gate_up_proj`**: gate + up projections combined into one `[num_experts, 2*intermediate, hidden]` tensor, **2 matmuls per expert instead of 3**. This is the single biggest performance change.
- **`Qwen3MoeRouter`** compatible with `RouterParallel` EP hooks.

The fused `gate_up_proj` delivers ~8× MFU improvement (23% vs 2.8% on 30B at 16k). This is a kernel-efficiency gain - fewer matmul launches + better memory access (one contiguous `[2*intermediate, hidden]` read vs two separate `[intermediate, hidden]` reads). The `--fuse_moe_experts` flag and fused checkpoint are no longer needed.

### PRs opened to make EP work

Two PRs were needed on top of transformers 5.6.0 to enable and fix EP for Qwen3:

- [**PR #45436**](https://github.com/huggingface/transformers/pull/45436) - _Add EP support for Qwen3 MoE + fix GroupedGemmParallel for 2D meshes_: added `base_model_ep_plan` to `Qwen3MoeConfig` and fixed two bugs in `GroupedGemmParallel` that prevented EP from working with 2D device meshes (expert sharding used global CUDA device index instead of mesh-local rank, and `num_experts` was divided repeatedly on each weight instead of once).
- [**PR #45473**](https://github.com/huggingface/transformers/pull/45473) - _Fix EP routing: RouterParallel shape, tp_plan property, grouped_mm sentinels_: fixed three bugs that combined to produce silently wrong expert routing (wrong shape for routing weights, wrong plan used during weight loading, uninitialized memory for non-local expert sentinels). Added 4 regression tests - zero EP test coverage existed before. All prior EP benchmarks had incorrect routing.

---

## Fixes applied

### EP routing fixes - [PR #45473](https://github.com/huggingface/transformers/pull/45473)

All EP results from the previous report produced **wrong expert outputs** without NaN or obvious failures. Three issues:

1. **Router output shape changed** during EP remapping: the routing weights went from shape `(seq, top_k)` to `(seq, num_local_experts)`, but the all expert forward implementations expected them paired 1:1 with the indices at `(seq, top_k)`. Tested with EP=1, expert output was zero.
2. **Weight loading used the wrong parallelism plan**: the code built a regex from the TP plan but looked up values from the EP plan, causing crashes with expert-only EP configurations.
3. **Non-local expert tokens hit uninitialized GPU memory sometimes**: the `grouped_mm` kernel didn't handle EP sentinel values, leaving garbage in the output. After fixing bug 1, sentinel weights were correctly zero, but we started seeing `0.0 * NaN = NaN` deterministic.

> All verified by comparing forward pass logits against non-EP ground truth for EP=1,2,4,8,16 - all match within bf16 precision.

### EP didn't work on 2D device meshes - [PR #45436](https://github.com/huggingface/transformers/pull/45436)

Two bugs in the expert sharding code prevented EP from working when combined with DP:

- on a 2D mesh: the shard computation used the global GPU index instead of the mesh-local rank (so GPU 2 at local rank 0 tried to load experts 128-191, out of range)
- the expert count attribute was divided once per weight tensor instead of once per module (128→64→32 after two weights)

### DeepSpeed + EP model loading hung indefinitely (WIP, not yet upstreamed)

When using `accelerate launch` with DeepSpeed ZeRO-3 and EP enabled, model loading hung with 0% GPU utilization for 10+ minutes.

The root cause: DeepSpeed's environment variables force `from_pretrained` down a ZeRO-3-specific code path at every decision point (model creation, weight loading, buffer initialization). EP needs the standard code path at every one of those points, it creates the model on meta device, loads weights through the standard `WeightConverter` pipeline, then lets `deepspeed.initialize()` wrap the result afterwards. With ZeRO-3 active, the model was either loaded through ZeRO-3's `GatheredParameters` which deadlocked on non-ZeRO-3 parameters I think.

The fix makes `from_pretrained` detect EP+DeepSpeed and follow the standard loading path instead of the ZeRO-3 path. Root-caused via incremental diagnostic scripts (hellish debugging, thanks that claude exists but still was hard to find) . This is still work in progress, I will be pushed as a separate PR once confirmed stable across more configurations.

### TRL-side fixes

- **EP was silently inactive** 😫! `create_model_from_path` re-added `device_map="auto"` even after SFT trainer removed it, I missed that code path entirely, it was bypassing the EP distribution hooks. All early "EP" wandb runs were actually plain FSDP2 :sad. Fixed by checking for `distributed_config` in model kwargs.
- **Expert-only EP plan**: removed attention entries from `base_model_ep_plan` so EP can scale beyond `num_kv_heads=4`. Attention weights stay replicated; FSDP2 handles their memory.
- **torch.compile TF32 crash on PyTorch 2.10**: transformers sets the new TF32 API, inductor reads the legacy API, PyTorch errors on mixed usage. Fixed by setting the legacy flags before any transformers import. Still need to debug this problem in the future.

### Note: EP uses all-reduce, not all-to-all

**The transformers EP implementation does not dispatch tokens between GPUs via all-to-all.** Reading through the code helped me find the non-active EP.

Every GPU sees all tokens, computes only on its local experts (zeros for non-local), and an **all-reduce** combines the partial results. This is important because all-reduce scales 96% inter-node on EFA while all-to-all scales only 11% on our infra. It explains why EP mesh topology (flat vs 2D) makes no difference, and why EP MFU stays stable across nodes.

Per [PR #45436 review feedback](https://github.com/huggingface/transformers/pull/45436#issuecomment-2810073098) from @Ferdinand Mom : the upstream plan is to create a dedicated `ep` mesh dimension separate from `tp`, so EP can scale independently of attention and TP. Currently `model.tp_plan` returns the EP plan when EP is enabled, replacing TP entirely.

---

## Qwen3-4B (Dense), FSDP2

### Baseline (sdpa, no liger)

| Context | Nodes | GPUs | DP  | TP  | CP  | EP  | Backend | Attn | MFU   | TPS     | TPS/GPU | Peak GPU Mem |
| ------- | ----- | ---- | --- | --- | --- | --- | ------- | ---- | ----- | ------- | ------- | ------------ |
| 16k     | 2     | 16   | 16  | 1   | 1   | 1   | fsdp2   | sdpa | 30.6% | 91,240  | 5,703   | 43.4 GB      |
| 16k     | 2     | 16   | 8   | 1   | 2   | 1   | fsdp2   | sdpa | 25.1% | 149,724 | 9,358   | 25.4 GB      |
| 16k     | 2     | 16   | 4   | 1   | 4   | 1   | fsdp2   | sdpa | 18.3% | 217,910 | 13,619  | 19.2 GB      |
| 32k     | 2     | 16   | 8   | 1   | 2   | 1   | fsdp2   | sdpa | 31.6% | 121,842 | 7,615   | 44.4 GB      |
| 32k     | 2     | 16   | 4   | 1   | 4   | 1   | fsdp2   | sdpa | 26.8% | 207,029 | 12,939  | 26.3 GB      |
| 32k     | 4     | 32   | 16  | 1   | 2   | 1   | fsdp2   | sdpa | 31.8% | 245,336 | 7,667   | 43.3 GB      |
| 32k     | 4     | 32   | 8   | 1   | 4   | 1   | fsdp2   | sdpa | 27.1% | 418,189 | 13,068  | 25.2 GB      |
| 32k     | 4     | 32   | 4   | 1   | 8   | 1   | fsdp2   | sdpa | 19.6% | 605,864 | 18,933  | 17.7 GB      |
| 32k     | 1     | 8    | 8   | 1   | 1   | 1   | fsdp2   | sdpa | —     | —       | —       | — OOM        |

### DeepSpeed Ulysses Sequence Parallelism (SP)

| Context | Nodes | GPUs | DP  | SP  | Backend         | Attn | MFU (corrected) | TPS    | TPS/GPU | Peak GPU Mem |
| ------- | ----- | ---- | --- | --- | --------------- | ---- | --------------- | ------ | ------- | ------------ |
| 16k     | 1     | 8    | 4   | 2   | deepspeed_zero3 | FA2  | 10.6%           | 31,540 | 3,943   | 39.3 GB      |
| 16k     | 1     | 8    | 2   | 4   | deepspeed_zero3 | FA2  | 2.8%            | 16,390 | 2,049   | 30.2 GB      |
| 32k     | 2     | 16   | 8   | 2   | deepspeed_zero3 | FA2  | 20.1%           | 77,420 | 4,839   | 51.6 GB      |
| 32k     | 2     | 16   | 4   | 4   | deepspeed_zero3 | FA2  | 7.6%            | 58,660 | 3,666   | 38.2 GB      |
| 32k     | 2     | 16   | 2   | 8   | deepspeed_zero3 | FA2  | 2.0%            | 31,440 | 1,965   | 28.0 GB      |
| 32k     | 2     | 16   | 1   | 16  | deepspeed_zero3 | FA2  | 0.5%            | 14,840 | 928     | 20.4 GB      |

### Liger Kernel (FSDP2)

| Context | Nodes | GPUs | DP  | CP  | Liger | Attn | MFU       | TPS    | TPS/GPU | Peak GPU Mem  |
| ------- | ----- | ---- | --- | --- | ----- | ---- | --------- | ------ | ------- | ------------- |
| 32k     | 1     | 8    | 8   | 1   | yes   | sdpa | 35.9%     | 34,570 | 4,321   | 24.9 GB (31%) |
| 32k     | 1     | 8    | 4   | 2   | yes   | sdpa | 17.1%     | 32,960 | 4,120   | 20.3 GB (25%) |
| 32k     | 1     | 8    | 8   | 1   | yes   | FA3  | **56.3%** | 54,260 | 6,783   | 24.9 GB (31%) |
| 64k     | 1     | 8    | 4   | 2   | yes   | sdpa | 18.3%     | 20,690 | 2,586   | 26.0 GB (33%) |
| 64k     | 1     | 8    | 2   | 4   | yes   | sdpa | 8.2%      | 18,480 | 2,310   | 21.2 GB (27%) |

### FA3 with SP (DeepSpeed)

| Context | Nodes | GPUs | DP  | SP  | Backend         | Attn | MFU (corrected) | TPS    | TPS/GPU | Peak GPU Mem  |
| ------- | ----- | ---- | --- | --- | --------------- | ---- | --------------- | ------ | ------- | ------------- |
| 32k     | 2     | 16   | 8   | 2   | deepspeed_zero3 | FA2  | 20.1%           | 77,420 | 4,839   | 51.6 GB (65%) |
| 32k     | 2     | 16   | 8   | 2   | deepspeed_zero3 | FA3  | 23.6%           | 91,170 | 5,698   | 51.6 GB (65%) |

FA3 is **17% better** than FA2 with SP=2 on 4B (23.6% vs 20.1%). The original report incorrectly claimed FA3 regressed because the FA2 baseline had uncorrected SP overcounting.

### Best configs for Qwen3-4B

| Context | Best MFU  | Config                      |
| ------- | --------- | --------------------------- |
| 16k     | 30.6%     | FSDP2, sdpa, 2n, DP=16      |
| 32k     | **56.3%** | FSDP2, liger+FA3, 1n, DP=8  |
| 32k SP  | 23.6%     | DS-Z3, SP=2+FA3, 2n         |
| 64k     | 18.3%     | FSDP2, liger+sdpa, 1n, CP=2 |

### Analysis

- **Liger kernel eliminates single-node 32k OOM**: fused CE + SwiGLU saves enough activation memory to go from OOM to 35.9% MFU on 1 node.
- **FA3 gives 1.57× MFU boost** over sdpa standalone (56.3% vs 35.9%). Hopper-native FA3 halves per-step attention time.
- **FA3 improves SP on 4B by 17%**: 23.6% vs 20.1% MFU (corrected). Higher raw TPS (91k vs 77k) confirms FA3 is faster.
- **CP beats SP at all degrees for 4B** (after SP correction): CP=2 at 31.6% vs SP=2 at 20.1%, CP=4 at 26.8% vs SP=4 at 7.6%. SP degrades sharply with degree because Ulysses all-to-all grows quadratically and crosses node boundaries at SP≥8 (11% EFA bandwidth).
- **CP trades MFU for memory**: CP=1→2→4→8 reduces memory (43→25→19→18 GB) but MFU drops (30→25→18→20%).

---

## Qwen3-30B-A3B (MoE, 128 experts, 8 active), Corrected EP Results

All results with transformers 5.6.0.dev0, expert-only EP plan, all three routing bugs fixed.

### FSDP2 + EP (flat mesh, EP = world_size)

| Context | Nodes | GPUs | DP  | TP  | CP  | EP  | MFU   | TPS     | TPS/GPU | Peak GPU Mem  |
| ------- | ----- | ---- | --- | --- | --- | --- | ----- | ------- | ------- | ------------- |
| 16k     | 1     | 8    | 8   | 1   | 1   | 8   | —     | —       | —       | — OOM         |
| 16k     | 2     | 16   | 16  | 1   | 1   | 16  | 22.5% | 60,600  | 3,788   | 71.1 GB (89%) |
| 16k     | 4     | 32   | 32  | 1   | 1   | 32  | 22.3% | 120,000 | 3,750   | 55.1 GB (69%) |
| 16k     | 8     | 64   | 64  | 1   | 1   | 64  | 21.4% | 230,500 | 3,602   | 51.1 GB (64%) |
| 32k     | 2     | 16   | 8   | 1   | 2   | 16  | 13.2% | 42,970  | 2,686   | 71.5 GB (89%) |
| 32k     | 4     | 32   | 16  | 1   | 2   | 32  | 13.2% | 85,510  | 2,672   | 55.2 GB (69%) |
| 32k     | 8     | 64   | 32  | 1   | 2   | 64  | 12.9% | 167,900 | 2,623   | 51.5 GB (64%) |

### FSDP2 without EP (fused gate_up_proj only)

| Context | Nodes | GPUs | DP  | TP  | CP  | EP  | MFU   | TPS     | TPS/GPU |
| ------- | ----- | ---- | --- | --- | --- | --- | ----- | ------- | ------- |
| 16k     | 2     | 16   | 16  | 1   | 1   | 1   | 23.1% | 62,210  | 3,888   |
| 16k     | 4     | 32   | 32  | 1   | 1   | 1   | 22.9% | 123,500 | 3,859   |
| 32k     | 2     | 16   | 8   | 1   | 2   | 1   | 13.6% | 44,110  | 2,757   |
| 32k     | 4     | 32   | 16  | 1   | 2   | 1   | 13.5% | 87,600  | 2,738   |

### DeepSpeed ZeRO-3 (no EP)

| Context | Nodes | GPUs | DP  | MFU   | TPS     | TPS/GPU |
| ------- | ----- | ---- | --- | ----- | ------- | ------- |
| 16k     | 2     | 16   | 16  | 17.9% | 48,120  | 3,008   |
| 16k     | 4     | 32   | 32  | 18.9% | 101,900 | 3,184   |

### DeepSpeed ZeRO-3 + SP (no EP)

| Context | Nodes | GPUs | DP  | SP  | Attn | MFU   | TPS    | TPS/GPU | Peak GPU Mem |
| ------- | ----- | ---- | --- | --- | ---- | ----- | ------ | ------- | ------------ |
| 16k     | 2     | 16   | 8   | 2   | FA2  | 5.8%  | 31,420 | 1,964   | 61.3 GB      |
| 32k     | 2     | 16   | 8   | 2   | FA2  | 13.1% | 42,600 | 2,663   | 54.0 GB      |
| 64k     | 2     | 16   | 4   | 4   | FA2  | 11.6% | 41,980 | 2,624   | 54.4 GB      |
| 64k     | 2     | 16   | 8   | 2   | FA2  | —     | —      | —       | — OOM        |

### FSDP2 + EP: long context (CP=8/16, 2 nodes)

| Context | Nodes | GPUs | DP  | CP  | EP  | MFU   | TPS    | TPS/GPU | Peak GPU Mem  |
| ------- | ----- | ---- | --- | --- | --- | ----- | ------ | ------- | ------------- |
| 32k     | 2     | 16   | 8   | 2   | 16  | 13.2% | 42,970 | 2,686   | 71.5 GB (89%) |
| 32k     | 2     | 16   | 2   | 8   | 16  | 1.9%  | 24,240 | 1,515   | 49.3 GB (62%) |
| 64k     | 2     | 16   | 2   | 8   | 16  | 2.9%  | 20,990 | 1,312   | 58.7 GB (74%) |
| 64k     | 2     | 16   | 1   | 16  | 16  | 0.8%  | 11,420 | 714     | 46.5 GB (58%) |

### EP=8 2D mesh vs EP=16 flat mesh (2 nodes)

| Context | EP  | EP mesh   | MFU  | TPS/GPU | Peak GPU Mem |
| ------- | --- | --------- | ---- | ------- | ------------ |
| 32k     | 16  | flat (16) | 1.9% | 1,515   | 49.3 GB      |
| 32k     | 8   | 2D (2,8)  | 1.9% | 1,542   | 49.3 GB      |
| 64k     | 16  | flat (16) | 2.9% | 1,312   | 58.7 GB      |
| 64k     | 8   | 2D (2,8)  | 2.9% | 1,324   | 58.7 GB      |

Flat and 2D meshes give identical performance, because EP uses all-reduce (96% inter-node bandwidth), not all-to-all. Flat mesh EP=world_size is the best strategy: minimizes expert memory per GPU with no MFU penalty.

### True EP=4 (TP=4 + EP=4, attention sharded)

| Context | Nodes | GPUs | DP  | EP  | MFU   | TPS/GPU | Loss | Status |
| ------- | ----- | ---- | --- | --- | ----- | ------- | ---- | ------ |
| 16k     | 2     | 16   | 4   | 4   | 22.7% | 3,824   | 1.66 | Yes    |
| 16k     | 1     | 8    | 2   | 4   | —     | —       | —    | OOM    |
| 32k     | 2     | 16   | 2   | 4   | —     | —       | —    | OOM    |

EP=4 with attention sharding is marginally slower than no-EP (22.7% vs 23.1%). Max EP = `num_kv_heads` = 4 when attention is in the EP plan. The [planned separate EP mesh dimension](https://github.com/huggingface/transformers/pull/45436#issuecomment-2810073098) would remove this constraint.

### FA3 with SP on 30B (5-step srun test)

| Context | Nodes | GPUs | DP  | SP  | Attn | MFU   | TPS/GPU | Runtime (5 steps) |
| ------- | ----- | ---- | --- | --- | ---- | ----- | ------- | ----------------- |
| 32k     | 2     | 16   | 8   | 2   | FA2  | 9.9%  | 2,009   | 40.8s             |
| 32k     | 2     | 16   | 8   | 2   | FA3  | 10.5% | 2,133   | 38.4s             |

FA3 gives only ~6% improvement with SP on 30B. Attention is a smaller fraction of MoE compute.

### FSDP2 + FA3 (no CP, no EP)

| Context | Nodes | GPUs | DP  | Attn | MFU       | TPS/GPU | Peak GPU Mem  |
| ------- | ----- | ---- | --- | ---- | --------- | ------- | ------------- |
| 16k     | 2     | 16   | 16  | sdpa | 23.1%     | 3,888   | ~68 GB        |
| 16k     | 2     | 16   | 16  | FA3  | **25.7%** | 4,318   | 71.1 GB (89%) |

FA3 gives +11% MFU over sdpa at 16k (25.7% vs 23.1%). Less dramatic than on dense models (+57%) because attention is only 80% of MoE layer compute, but still a meaningful improvement.

> Note: FA3 + EP failed with device mesh incompatibility (`KeyError: "Invalid mesh_dim_names ('dp_shard_cp',)"`). EP creates a mesh with only `['tp']` but FSDP2 expects `dp_shard_cp`. Needs transformers/accelerate fix.

---

### SonicMoE kernel (CuteDSL fused MoE) — `--experts_implementation sonicmoe`

The `kernels-community/sonic-moe` CuteDSL kernel from [transformers PR #45433](https://github.com/huggingface/transformers/pull/45433) is selectable via TRL's new `--experts_implementation` flag. Replaces the default `grouped_mm` expert dispatch with a fused MoE kernel.

#### Cumulative vs window MFU — kernel compile cost dominates short runs

Sonicmoe's first forward pass takes ~25–30s (CuteDSL JIT compile + autotune + first-touch). Steps 2+ run at steady state (~3s/step). The standard MFU metric uses cumulative TPS = `total_tokens / total_runtime`, which gets dominated by the slow first step for many steps. Added `mfu_window` to `SFTTrainer.log` that measures TPS over the _last logging window only_ — the real per-step throughput once the model is warm.

#### 50-step results at 16k, 2 nodes × 8 H100, FSDP2 DP=16

| Implementation | Attn | Cumulative MFU (step 50) | **Window MFU** (steady-state) | Cumulative TPS | Window TPS | Peak GPU Mem  |
| -------------- | ---- | ------------------------ | ----------------------------- | -------------- | ---------- | ------------- |
| grouped_mm     | sdpa | 23.5%                    | **24.3%**                     | 63,398         | 65,480     | 69.5 GB (87%) |
| grouped_mm     | FA3  | 27.9%                    | **28.1%**                     | 75,140         | 75,760     | 71.1 GB (89%) |
| sonicmoe       | FA3  | 26.2%                    | **34.7%**                     | 70,612         | 93,480     | 68.6 GB (86%) |

For grouped_mm runs, cumulative ≈ window — no meaningful first-step compile cost. For sonicmoe, **window is 8.5 pp higher than cumulative** even after 50 steps; the cumulative metric needs ~100+ steps to fully approach steady state.

#### Speedups (window MFU, steady-state)

| Comparison                             | Ratio      | Source                               |
| -------------------------------------- | ---------- | ------------------------------------ |
| FA3 vs sdpa (grouped_mm)               | **+15.7%** | 28.1 / 24.3 — pure attention speedup |
| sonicmoe vs grouped_mm (both with FA3) | **+23.4%** | 34.7 / 28.1 — kernel itself          |
| sonicmoe + FA3 vs sdpa baseline        | **+42.7%** | 34.7 / 24.3 — combined               |

#### Window MFU trajectory (per logging step) — flat from step 10

| Step | grouped_mm + sdpa | grouped_mm + FA3 | sonicmoe + FA3 |
| ---- | ----------------- | ---------------- | -------------- |
| 5    | 24.3%             | 27.9%            | — (compile)    |
| 10   | 24.6%             | 29.7%            | 32.3%          |
| 15   | 24.0%             | 28.2%            | 34.8%          |
| 20   | 24.5%             | 29.6%            | 34.9%          |
| 25   | 24.5%             | 29.6%            | 32.3%          |
| 30   | 23.6%             | 27.9%            | 34.6%          |
| 35   | 24.5%             | 29.7%            | 34.6%          |
| 40   | 23.8%             | 29.2%            | 32.2%          |
| 50   | 24.3%             | 28.1%            | 34.7%          |

Sonicmoe stabilizes at ~33–35% from step 10 onwards (single dip at step 45 was a transient stall). Both grouped_mm runs are flat across the entire trajectory — confirming the warmup curve previously seen with sonicmoe was a measurement artifact, not actual kernel speedup.

#### 32k DS-Z3 + SP=2 + FA3 + sonicmoe (20 steps)

| Implementation | Backend          | MFU (cum)  | TPS    | TPS/GPU | Peak GPU Mem  |
| -------------- | ---------------- | ---------- | ------ | ------- | ------------- |
| grouped_mm     | DS-Z3 SP=2 + FA3 | 14.5%      | 47,100 | 2,944   | 53.3 GB (67%) |
| sonicmoe       | DS-Z3 SP=2 + FA3 | **14.66%** | 47,661 | 2,979   | 78.6 GB (99%) |

Marginal improvement on the cumulative metric — but the per-window MFU was not collected on this 20-step run. Hits 99% GPU memory; no further headroom for context scaling.

#### FSDP2 OOM regression in transformers 5.7.0.dev0 (relevant context)

Bisecting transformers `5.6.0.dev0` (working) → `5.7.0.dev0` (rebased) identified [PR #45050](https://github.com/huggingface/transformers/pull/45050) as the regression. That PR swapped `torch.empty_like` → `torch.zeros_like` for non-rank-0 FSDP placeholder tensors to fix a NaN issue. The change is benign on small models but on Linux with anonymous mmap it forces a _physical memory commit_ of every byte (vs lazy allocation for `empty_like`), multiplying CPU peak by `(ranks_per_node - 1) × model_size`. For Qwen3-30B at 8 ranks/node: ~480 GB peak → cgroup OOM during `from_pretrained`.

Upstream fix pushed at [`AmineDiro/transformers:fix-fsdp2-cpu-ram-zeros-like`](https://github.com/AmineDiro/transformers/tree/fix-fsdp2-cpu-ram-zeros-like): drop the parameter materialization on non-rank-0 ranks (FSDP broadcast overwrites them anyway); only materialize buffers (per-rank, not broadcast). All sonicmoe results above were collected with this fix applied.

#### SonicMoE findings

- **+23% over grouped_mm + FA3 at steady state** (window MFU). Best 16k MoE config to date, replacing the previous best of 25.7% grouped_mm + FA3 (which was a 20-step cumulative number, ~28% in window).
- **Cumulative MFU is misleading for kernels with compile cost**: under-reports steady-state throughput by 8.5 pp at 50 steps. All future MFU comparisons in this report should use window MFU as the primary metric where compile-heavy kernels are involved.
- **Memory parity with grouped_mm**: ~68.6 GB peak (vs 71.1 GB for grouped_mm + FA3). Same fused weight layout, no additional kernel buffers.
- **Sonicmoe + FA3 + sdpa**: not tested. CP requires sdpa, so sonicmoe + CP would also need sdpa.
- **Per-machine cold-start cost**: first sonicmoe run on a fresh node pays the full Triton/CuteDSL compile (~25s); subsequent runs on the same node hit the kernel binary cache in `~/.cache/huggingface/kernels` and start much faster. This affects job-1 throughput on a new cluster but not subsequent jobs.

### DS-Z3 + SP + FA3 (full runs)

| Context | Nodes | GPUs | DP  | SP  | Attn | MFU       | TPS/GPU | Peak GPU Mem  |
| ------- | ----- | ---- | --- | --- | ---- | --------- | ------- | ------------- |
| 32k     | 2     | 16   | 8   | 2   | FA2  | 13.1%     | 2,663   | 54.0 GB       |
| 32k     | 2     | 16   | 8   | 2   | FA3  | **14.5%** | 2,944   | 53.3 GB (67%) |
| 64k     | 2     | 16   | 4   | 4   | FA2  | 11.6%     | 2,624   | 54.4 GB       |
| 64k     | 2     | 16   | 4   | 4   | FA3  | **12.5%** | 2,837   | 54.4 GB (68%) |

FA3 improves SP consistently: +10.7% at 32k SP=2, +7.9% at 64k SP=4 (relative gains over FA2).

### Comparison: all approaches at 16k, 2 nodes

Sorted by Win MFU (last). The previously NaN'ing EP runs have been re-verified after the EP+FSDP DTensor fix (transformers PR #45662) plus a wrapper-level clamp+masked_fill in `sonicmoe_experts_forward` (mirroring `grouped_mm`'s sentinel handling). The "Train valid?" column reflects the **post-fix** state; the "Pre-fix Win MFU" column shows the original NaN measurement for reference.

| Approach                                | Kernel              | Cum MFU | Win MFU (last) | Pre-fix Win MFU | TPS/GPU (win) | Peak GPU Mem  | Train valid? |
| --------------------------------------- | ------------------- | ------- | -------------- | --------------- | ------------- | ------------- | ------------ |
| FSDP2 + FA3 + EP=8                      | sonicmoe (Ilyas)    | 32.15%  | **40.4%**      | 42.66% (NaN)    | 6,808         | 57.8 GB (73%) | ✓            |
| FSDP2 + FA3 + EP=8                      | grouped_mm          | 32.40%  | **32.6%**      | 35.06% (NaN)    | 5,503         | 58.3 GB (73%) | ✓            |
| **FSDP2 + FA3 + sonicmoe v1 (no EP)**   | sonicmoe v1         | 26.2%   | **34.7%**      | —               | 5,843         | 68.6 GB (86%) | ✓            |
| FSDP2 + EP=4 + sdpa                     | sonicmoe (Ilyas)    | 26.16%  | 32.32%         | —               | 5,440         | 62.4 GB (78%) | ❌ NaN       |
| FSDP2 + EP=8 + sdpa                     | sonicmoe (Ilyas)    | 25.87%  | **31.0%**      | 32.36% (NaN)    | 5,202         | 57.8 GB (73%) | ✓            |
| DS-Z3 + EP=8 + sdpa                     | sonicmoe (Ilyas)    | —       | —              | 32.72% (NaN)    | —             | —             | ❌ crash (`c10d::broadcast_` no-mesh) |
| FSDP2 + EP=8 + sdpa (wrapper-clamp v1)  | sonicmoe v1         | —       | 30.67%         | —               | —             | —             | ❌ NaN       |
| FSDP2 + EP=2 + sdpa                     | sonicmoe (Ilyas)    | 11.71%  | 28.31%         | —               | 4,766         | 76.4 GB (96%) | ❌ NaN       |
| FSDP2 + EP=8 + sdpa (PR #45621)         | grouped_mm          | —       | 27.50%         | —               | —             | —             | ❌ NaN       |
| FSDP2 + FA3 (no EP)                     | grouped_mm          | 27.9%   | 28.1%          | —              | 4,735         | 71.1 GB (89%) | ✓            |
| FSDP2 + FA3 (no EP, 20-step)            | grouped_mm          | 25.7%   | —              | —              | 4,318         | 71.1 GB (89%) | ✓            |
| FSDP2 + sdpa (no EP)                    | grouped_mm          | 23.5%   | 24.3%          | —              | 4,093         | 69.5 GB (87%) | ✓            |
| FSDP2 + EP=16 (flat, older)             | grouped_mm          | 22.5%   | —              | —              | 3,788         | 71.1 GB (89%) | likely ❌ (predates correctness audit) |
| DeepSpeed ZeRO-3                        | grouped_mm          | 17.9%   | —              | —              | 3,008         | ~78 GB        | ✓            |
| DS-Z3 + SP=2                            | grouped_mm          | 5.8%    | —              | —              | 1,964         | 61.3 GB (77%) | ✓            |

Headline:

- **40.4% window MFU** (`FSDP2 + FA3 + EP=8 + sonicmoe (Ilyas)`) is the highest *trainable* config at 16k after the EP+FSDP fix and wrapper clamp+mask. Loss 10–14 across all 50 steps.
- The pre-fix 42.66% NaN measurement was on broken EP weights (rank 0's experts broadcast to all ranks). The 2.3 pp drop after fix is the cost of wrapper-level compute-then-mask vs the kernel-native sentinel skip.
- 34.7% (FA3 + sonicmoe v1, no EP) is the no-EP baseline. EP=8 + FA3 + sonicmoe now beats it by +5.7 pp on the same hardware.
- Patched `sonicmoe (Ilyas)` is required for any EP > 1 forward pass (sentinel-aware kernel); without it, upstream `sonicmoe v1` segfaults on non-local rows.

### SP vs EP+CP at long context

| Ctx  | SP path (best, no EP)   | EP+CP path (best)        | SP advantage |
| ---- | ----------------------- | ------------------------ | ------------ |
| 64k  | 20.6 % (4n SP=4)        | 7.7 % (4n EP=8 CP=4)     | **2.7×**     |
| 128k | 19.4 % (8n SP=8)        | 4.13 % (8n EP=8 CP=8)    | **4.7×**     |

EP+CP is correct after the fix but heavily MFU-penalized (CP=4 → ~7%, CP=8 → ~4%). SP without EP wins long-context. Combining DS-Z3+SP+EP would compound both wins; blocked by DS-Z3 lacking opt-out for externally-managed (EP-DTensor) params (see `debug_sp_ep_sonic.md`).

### Comparison: all approaches at 32k, 2 nodes

| Approach                          | Kernel           | MFU (cum)  | TPS/GPU | Peak GPU Mem  | Train valid? |
| --------------------------------- | ---------------- | ---------- | ------- | ------------- | ------------ |
| **DS-Z3 + SP=2 + FA3 + sonicmoe** | sonicmoe (Ilyas) | **14.66%** | 2,979   | 78.6 GB (99%) | ✓            |
| DS-Z3 + SP=2 + FA3                | grouped_mm       | 14.5%      | 2,944   | 53.3 GB (67%) | ✓            |
| FSDP2 (no EP) + CP=2              | grouped_mm       | 13.6%      | 2,757   | ~70 GB        | ✓            |
| FSDP2 + EP=16 + CP=2              | grouped_mm       | 13.2%      | 2,686   | 71.5 GB (89%) | likely ❌     |
| DS-Z3 + SP=2 + FA2                | grouped_mm       | 13.1%      | 2,663   | 54.0 GB (68%) | ✓            |

### Comparison: long context 64k (post-fix, all training-correct)

| Approach                                  | Nodes | Cum MFU | Win MFU (last) | Win MFU (peak) | Total TPS | Peak GPU Mem  |
| ----------------------------------------- | ----- | ------- | -------------- | -------------- | --------- | ------------- |
| **DS-Z3 + SP=4 + FA3 + sonicmoe**         | 4     | 9.05%   | **20.61%**     | **20.99%**     | 65,590    | 65.9 GB (83%) |
| FSDP2 + EP=8 + CP=4 + sonicmoe (sdpa)     | 4     | 7.29%   | 7.71%          | 7.81%          | 13,980    | —             |
| FSDP2 + EP=8 + CP=4 + sonicmoe (sdpa)     | 8     | 7.13%   | 7.72%          | 7.74%          | 27,990    | —             |

### Comparison: long context 128k (post-fix, all training-correct)

| Approach                                  | Nodes | Cum MFU | Win MFU (last) | Win MFU (peak) | Total TPS | Peak GPU Mem  |
| ----------------------------------------- | ----- | ------- | -------------- | -------------- | --------- | ------------- |
| **DS-Z3 + SP=8 + FA3 + sonicmoe**         | 8     | 5.54%   | **19.42%**     | **19.60%**     | 85,230    | —             |
| DS-Z3 + SP=8 + FA3 + sonicmoe             | 4     | 6.22%   | 16.04%         | 19.24%         | 47,820    | 67.4 GB (85%) |
| FSDP2 + EP=8 + CP=8 + sonicmoe (sdpa)     | 8     | 3.91%   | 4.13%          | 4.13%          | 7,944     | —             |

### Best configs for Qwen3-30B-A3B (post-fix, all training-correct)

| Context | Nodes | Best Config                                          | Win MFU (last/peak) | Cum MFU | Total TPS | Peak GPU Mem  | Steps |
| ------- | ----- | ---------------------------------------------------- | ------------------- | ------- | --------- | ------------- | ----- |
| 16k     | 2     | **FSDP2 + EP=8 + FA3 + sonicmoe**                    | **40.4 / 40.4 %**   | 32.15 % | 109,600   | 73.4 GB (92%) | 50    |
| 16k     | 4     | FSDP2 + EP=8 + FA3 + sonicmoe                        | 40.0 / 40.4 %       | 32.40 % | 215,600   | 71.7 GB (90%) | 50    |
| 16k     | 8     | FSDP2 + EP=8 + FA3 + sonicmoe                        | 39.4 / 40.0 %       | —       | **424,900** | 71.2 GB (89%) | 50    |
| 32k     | 2     | **DS-Z3 + SP=2 + FA3 + sonicmoe + compile**          | **21.98 / 21.98 %** | 17.78 % | 57,800    | 79.0 GB (99%) | 50    |
| 32k     | 2     | DS-Z3 + SP=2 + FA3 + sonicmoe (no compile)           | 21.69 / 21.69 %     | 17.82 % | 57,900    | 79.0 GB (99%) | 50    |
| 32k     | 2     | FSDP2 + EP=8 + CP=2 + sonicmoe (sdpa)                | 15.6 / 15.6 %       | 13.99 % | 25,250    | 73.6 GB (92%) | 50    |
| 32k     | 4     | FSDP2 + EP=8 + CP=2 + sonicmoe (sdpa)                | 15.6 / 15.6 %       | 13.89 % | 50,620    | 71.9 GB (90%) | 50    |
| 32k     | 8     | FSDP2 + EP=8 + CP=2 + sonicmoe (sdpa)                | 15.4 / 15.4 %       | 13.47 % | 99,900    | 71.4 GB (90%) | 50    |
| 64k     | 4     | **DS-Z3 + SP=4 + FA3 + sonicmoe**                    | **20.6 / 21.0 %**   | 9.05 %  | 65,590    | 65.9 GB (83%) | 50    |
| 64k     | 4     | FSDP2 + EP=8 + CP=4 + sonicmoe (sdpa)                | 7.7 / 7.8 %         | 7.29 %  | 13,980    | 72.0 GB (90%) | 50    |
| 64k     | 8     | FSDP2 + EP=8 + CP=4 + sonicmoe (sdpa)                | 7.7 / 7.7 %         | 7.13 %  | 27,990    | 71.4 GB (90%) | 50    |
| 128k    | 4     | DS-Z3 + SP=8 + FA3 + sonicmoe                        | 16.0 / 19.2 %       | 6.22 %  | 47,820    | 67.4 GB (85%) | 50    |
| 128k    | 8     | **DS-Z3 + SP=8 + FA3 + sonicmoe**                    | **19.4 / 19.6 %**   | 5.54 %  | 85,230    | 63.9 GB (80%) | 50    |
| 128k    | 8     | FSDP2 + EP=8 + CP=8 + sonicmoe (sdpa)                | 4.13 / 4.13 %       | 3.91 %  | 7,944     | 71.3 GB (89%) | 50    |
| 256k    | 8     | DS-Z3 + SP=16 + FA3 + sonicmoe                       | 1.36 / 1.43 %       | 4.52 %  | 71,650    | 61.5 GB (77%) | 50 (impractical — cross-node Ulysses dominates step time) |
| 32k     | 2     | FSDP DP=16 + FA3 + sonicmoe (no EP, no SP, no CP)    | OOM 18.55 GiB       | —       | —         | 76.6 GB (96%) | activations alone OOM at 32k pure DP |
| 64k     | 2     | DS-Z3 + SP=8 dp=2 intra-node + FA3 + sonicmoe        | 8.7 %               | 2.73 %  | 17,060    | 62.6 GB (79%) | DP=2 ZeRO-3 cross-node param shuffle dominates; SP=4 4n (DP=8) is much better |

**Required setup (all SP runs):** `--pad_to_multiple_of 8` to avoid Ulysses ValueError on non-divisible packed seqlens. SP requires `attn_implementation: kernels-community/vllm-flash-attn3` (or FA2); sdpa errors at attention dispatch.

**Required setup (all EP runs, post-fix):** transformers PR #45662 (EP+FSDP DTensor wrap) + sonicmoe wrapper-level clamp+mask + accelerate `_prepare_tp` `has_ep` skip. Full patch list in `debug_sp_ep_sonic.md`.

### compile + EP/SP results (2026-04-28)

Stack: accelerate PR #4022 clone via `PYTHONPATH=/fsx/amine_dirhoussi/accelerate/src` + `dynamo_config.use_fullgraph: false` + `use_regional_compilation: true` + per-rank `TRITON_CACHE_DIR` + `@torch._dynamo.disable` on expert kernels.

| Context | Nodes | Config                                                | Win MFU (last/peak) | Cum MFU | Total TPS | Peak GPU Mem  | Loss | Verdict |
| ------- | ----- | ----------------------------------------------------- | ------------------- | ------- | --------- | ------------- | ---- | ------- |
| 16k     | 2     | FSDP DP=16 + FA3 + compile (no EP, control)           | 34.87 / 34.87 %     | 26.98 % | 73,260    | 67.8 GB (85%) | 1.6 ✅ | matches PR #4022 +6 pp |
| 16k     | 2     | **DS-Z2 + EP=8 + FA3 + sonicmoe + compile**           | **36.7 %**          | 29.75 % | 80,130    | 73.3 GB (92%) | 12.25 ✅ | new compile+EP combo |
| 16k     | 2     | DS-Z2 + EP=8 + sdpa + sonicmoe (no compile, ref)      | 28.6 %              | 25.87 % | 67,250    | 73.6 GB (92%) | 12.43 ✅ | baseline for the +8 pp comparison |
| 64k     | 4     | DS-Z3 + SP=4 + FA3 + sonicmoe + compile               | 20.51 / 20.71 %     | 9.05 %  | 113,700   | 65.9 GB (83%) | 1.62 ✅ | comm-bound, ~same as no-compile (20.6 %) |
| 128k    | 8     | DS-Z3 + SP=8 + FA3 + sonicmoe + compile               | 19.34 / 19.34 %     | 12.02 % | 184,800   | 63.9 GB (80%) | 1.62 ✅ | comm-bound, ~same as no-compile (19.42 %) |
| 32k+    | 2/4   | DS-Z2 + EP=8 + FA3 + sonicmoe (compile or not, 2n/4n) | OOM 18.55 GiB @ 32k, 37.09 GiB @ 64k | — | — | — | — | EP-replicated expert buffer = `seq × num_local_experts × moe_intermediate × 2B` per rank; scales linearly with seq, not relieved by adding DP or toggling compile |
| 256k    | 8     | FSDP + EP=8 + CP=8 + sonicmoe (no compile)            | OOM 18.55 GiB       | —       | —         | —             | — | same EP buffer ceiling: per-rank seq=32k after CP=8 |

**Take-aways:**

1. **DS-Z2+EP+compile = +8 pp** over DS-Z2+EP no-compile (28.6 → 36.7 %). Closes most of the gap to the FSDP+EP no-compile champion (40.4 %) while adding compile. New runner-up at 16k.
2. **FSDP+EP+compile is blocked** at the optimizer step. EP DTensor mesh + FSDP DP DTensor mesh confuses the compiled foreach grouping (`_group_tensors_by_device_and_dtype` strict assert under compile). Without compile, Adam's foreach falls back to per-tensor handling at runtime. DS-Z2+EP avoids this because its EP path uses plain `nn.Parameter` (with `allreduce=False`/`group_name` markers), no DTensor mesh.
3. **Compile gives no improvement at long context** (64k/128k). Bottleneck is communication (Ulysses all-to-all + ZeRO-3 all-gather), not compute. GPU spends most step time in NCCL.
4. **256k still gated** by the unsharded per-rank EP buffer (18.55 GiB at 32k per-rank seq). Even with CP=8 sharding 256k → 32k/rank, OOM. Need either CP=16 (16+ nodes) or DS-Z3+SP=16 (in flight).
5. **DS-Z2+EP is a 16k-only recipe.** At 32k DS-Z2+EP=8 OOMs at the same 18.55 GiB EP-replicated activation buffer regardless of compile (on/off) or nodes (2n/4n). At 64k the buffer doubles to 37.09 GiB. The 18.55 GiB is `seq × num_local_experts × moe_intermediate × 2 bytes` — transformers' EP replicates routing across all EP ranks, so the buffer scales linearly with per-rank seq. To go beyond 16k with EP, must shard the seq dim too (CP=2 brings buffer to 9.3 GiB, which is how FSDP+EP+CP=2 survives 32k at 15.6 %).

> **Update (2026-04-29)**: take-away (5) is partially superseded by chunked-CE results below. Chunked CE shaves ~20 GB at 64k from the lm_head logit tensor, freeing the 18.55 GiB EP buffer. DS-Z2+EP+chunked now works at 32k–128k. See "chunked-CE results" section below.

### chunked-CE results (2026-04-29) — **NEW LONG-CONTEXT CHAMPIONS**

Stack: TRL PR #5575 cherry-picked onto branch (`loss_type=chunked_nll`, chunk size 256). On FSDP2: needs `fsdp_reshard_after_forward: false` (added to `fsdp2.yaml.j2` template). All sonicmoe runs use the wrapper-level clamp+masked_fill (Ilyas patched). Loss range 11–15 for sonicmoe+EP path (clamp wrapper typical), 1.5–1.7 for SP-only path.

| Context | Nodes | Config                                                | Win MFU (last/peak) | Cum MFU | Total TPS | Peak GPU Mem  | Loss     | Verdict |
| ------- | ----- | ----------------------------------------------------- | ------------------- | ------- | --------- | ------------- | -------- | ------- |
| 16k     | 2     | DS-Z2 + EP=8 + FA3 + chunked                          | 32.57 %             | 24.50 % | 87,560    | 48.9 GB (61%) | 10–13 ✅  | regression vs 40.4% no-chunked — chunked overhead at 16k where lm_head wasn't a bottleneck. **Use no-chunked at 16k**. |
| **32k** | **2** | **DS-Z2 + EP=8 + FA3 + chunked**                      | **45.81 / 45.86 %** | 39.31 % | 74,520    | 55.0 GB (69%) | 11–15 ✅  | **NEW 32k CHAMPION** (was 21.98% with DS-Z3+SP=2). +24 pp. EP buffer fits because chunked frees 20 GB. |
| 32k     | 2     | DS-Z2 + EP=8 + FA3 + chunked + compile                | 45.86 %             | 40.08 % | 74,520    | 55.0 GB (69%) | 11–15 ✅  | compile no help vs no-compile (-0.05 pp) — chunked saturates compute at 32k+ |
| **64k** | **2** | **DS-Z2 + EP=8 + FA3 + chunked**                      | **57.23 / 57.23 %** | 46.07 % | 51,470    | 67.6 GB (85%) | 11–13 ✅  | **NEW 64k CHAMPION** (was 20.5% with DS-Z3+SP=4 4n). +37 pp. |
| 64k     | 4     | DS-Z2 + EP=8 + FA3 + chunked                          | 56.30 %             | 45.59 % | 102,000   | 53.3 GB (67%) | 11–12 ✅  | -1 pp MFU vs 2n, +2× TPS (~100k vs 51k) — better throughput choice |
| 64k     | 2     | DS-Z2 + EP=8 + FA3 + chunked + compile                | 56.59 %             | 53.92 % | 51,090    | 67.6 GB (85%) | 11–13 ✅  | -0.6 pp vs no-compile — compile no help |
| **128k** | **4** | **DS-Z2 + EP=8 + FA3 + chunked**                     | **69.07 / 69.10 %** | 64.10 % | 66,400    | 75.6 GB (95%) | 12–14 ✅  | **NEW 128k CHAMPION + highest MoE MFU on stack ever** (was 19.4% with DS-Z3+SP=8 8n). +50 pp. |
| 128k    | 8     | DS-Z2 + EP=8 + FA3 + chunked                          | 68.38 %             | 52.68 %¹| 131,500   | 73.8 GB (93%) | 12–14 ✅  | matches 4n MFU; 2× TPS at 8n |
| 128k    | 4     | DS-Z2 + EP=8 + FA3 + chunked + compile                | 68.29 %             | 67.15 % | 65,620    | 75.6 GB (95%) | 12–14 ✅  | compile no help (compute saturated) |
| **256k** | **8** | **DS-Z3 + SP=8 (intra-node) + FA3 + chunked, no EP** | **32.60 / 46.49 %** | 11.07 % | 32,330    | 36.1 GB (45%) | 1.59–1.71 ✅ | **NEW 256k CHAMPION** — 24× over old SP=16 cross-node (1.36%). MFU oscillates 5–32%. |
| 256k    | 8     | DS-Z3 + SP=8 + FA3 + chunked + compile                | 28.89 %             | 24.14 % | 28,650    | 36.1 GB (45%) | 1.58 ✅   | compile **stabilizes** SP path (steady ~22-32% vs no-compile 5-32%); peak slightly lower |
| 256k    | 8     | DS-Z3 + SP=4 + FA3 + chunked                          | **46.49 %** peak    | 25.70 % | 25,590    | 45.2 GB (57%) | 1.56–1.69 ✅ | **highest 256k MFU** (peak); fewer SP comm steps, more activation. Still oscillates. |
| 512k    | 16    | DS-Z3 + SP=8 + FA3 + chunked                          | 40.37 % peak (step 10) | 20.25 % | 40,680    | 43.5 GB (55%) | 1.58 ✅ (2 logs)  | ran 2 logs (steps 5/10), then **Z3 cross-node all-gather hung at 30-min watchdog** — 128 ranks too many for ZeRO-3 across nodes. Memory was fine. |

¹ Cum MFU lower than 4n (64.10%) due to one transient stall at step 20 (window dropped to 32% before recovering); peak/steady are 68%.

**Take-aways (chunked-CE)**:

1. **chunked-CE is the long-context unlock.** At 64k chunked frees ~20 GB from the `(B×S, vocab=151936)` logits tensor, which is enough to fit the 18.55 GiB EP-replicated expert buffer. The EP path now scales 16k→128k.
2. **DS-Z2 + EP=8 + chunked is the new long-context recipe.** Compute-bound (≥ 60% MFU at 128k), no Ulysses comm cost, healthy training. 16k stays no-chunked (slight regression).
3. **256k+ falls back to SP path** (no EP, since per-rank seq grows the EP buffer past memory). DS-Z3+SP=8+chunked at 256k @ 8n hits 32% peak; SP=4 hits 46% peak (more activation, less SP comm).
4. **Compile + chunked**: no peak MFU benefit (compute already saturated), but **compile stabilizes the DS-Z3 SP oscillation** at 256k+ (window MFU steady ~25-30% vs no-compile 5-32%). Useful for predictable wall-time.
5. **Z3 doesn't scale to 128 ranks**: 512k @ 16n hangs at the 30-min all-gather watchdog. Smaller mesh (8n) might work at smaller per-rank seq; 16n requires `expert_data_parallel_group`-style sharding (DS native MoE) to scale. To be tested.
6. **MFU oscillation at 256k+ SP path**: window MFU swings 5–46% across steps. Suspect intermittent NCCL stalls under heavy ZeRO-3 cross-node load. Compile mitigates by serializing the path.

### Final headline (chunked-CE era, 2026-04-29)

| Context | Best recipe                                            | Window MFU peak | adj peak | Cum MFU | adj cum  | Peak Mem | Status |
| ------- | ------------------------------------------------------ | --------------- | -------- | ------- | -------- | -------- | ------ |
| 16k     | FSDP + EP=8 + FA3 + sonicmoe (NO chunked)              | **40.4 %**      | 27.11 %  | 32.15 % | 21.58 %  | 73.4 GB  | 16k stays no-chunked |
| 32k     | DS-Z2 + EP=8 + FA3 + sonicmoe + chunked                | **45.86 %**     | 27.66 %  | 39.31 % | 23.71 %  | 55.0 GB  | **+24 pp** |
| 64k     | DS-Z2 + EP=8 + FA3 + sonicmoe + chunked @ 2n           | **57.23 %**     | 31.91 %  | 46.07 % | 25.69 %  | 67.6 GB  | **+37 pp** |
| 128k    | DS-Z2 + EP=8 + FA3 + sonicmoe + chunked @ 4n           | **69.10 %**     | 36.66 %  | 64.10 % | 34.01 %  | 75.6 GB  | **+50 pp** |
| 256k    | DS-Z3 + SP=2 + FA3 + sonicmoe + chunked + compile @ 8n | **59.61 %**     | 30.75 %  | 52.40 % | 27.03 %  | 69.4 GB  | **+58 pp** vs old SP=16 (1.36 %) |
| 512k    | DS-Z3 + SP=4 + FA3 + sonicmoe + chunked + compile @ 8n | **58.24 %**     | 29.59 %  | 38.76 % | 19.69 %  | 70.7 GB  | **NEW** — was unreachable before |
| **1M**  | **DS-Z3 + SP=8 + FA3 + sonicmoe + chunked + compile @ 8n** | **37.46 %** | 18.88 %  | **35.65 %** | 17.97 % | 72.2 GB | **NEW frontier — 1M ctx MoE training works** |

### NEW headline (Liger era, 2026-04-29 — supersedes chunked-CE row by row)

Liger replaces `--loss_type chunked_nll` with `--use_liger_kernel true --liger_kernel_config '{"swiglu":false}'` (FLCE + RMSNorm + RoPE; swiglu disabled because it bypasses transformers' EP dispatcher — see `debug_liger_ep.md`).

| Context | Best recipe                                                | Win MFU peak | adj peak | Cum MFU | adj cum | TPS Win  | Peak Mem | vs chunked-CE  |
| ------- | ---------------------------------------------------------- | ------------ | -------- | ------- | ------- | -------- | -------- | -------------- |
| 16k     | DS-Z2 + EP=8 + FA3 + sonicmoe + Liger                      | **40.70 %**  | 27.31 %  | 28.80 % | 19.33 % | 109,000  | 48.4 GB  | parity (chunked-DS-Z2 was 32.57 %; old FSDP+EP champ 40.4 %) |
| 32k     | DS-Z2 + EP=8 + FA3 + sonicmoe + Liger                      | **56.62 %**  | 34.15 %  | 45.25 % | 27.29 % | 92,010   | 54.3 GB  | **+11 pp peak** |
| 64k     | DS-Z2 + EP=8 + FA3 + sonicmoe + Liger @ 2n                 | **66.46 %**  | 37.06 %  | 61.69 % | 34.40 % | 60,190   | 66.9 GB  | **+9 pp peak**  |
| **128k**| **DS-Z2 + EP=8 + FA3 + sonicmoe + Liger @ 4n**             | **76.29 %**  | 40.47 %  | **74.69 %** | 39.62 % | 73,330  | 72.6 GB  | **+7 pp peak**, highest MoE MFU on stack |
| 256k    | DS-Z3 + SP=2 + FA3 + sonicmoe + Liger + compile @ 8n       | **63.62 %**  | 32.82 %  | 60.59 % | 31.25 % | 63,090   | 66.9 GB  | **+4 pp peak**, +8 pp cum |
| 512k    | DS-Z3 + SP=4 + FA3 + sonicmoe + Liger + compile @ 8n       | **63.26 %**  | 32.14 %  | 57.95 % | 29.44 % | 31,290   | 69.7 GB  | **+5 pp peak**, +19 pp cum |
| **1M**  | **DS-Z3 + SP=8 + FA3 + sonicmoe + Liger + compile @ 8n**   | **62.33 %**  | 31.41 %  | **58.29 %** | 29.38 % | 15,830  | 69.2 GB  | **+25 pp peak**, +23 pp cum |

### Single-node 1n results (Liger recipe)

| Context | Mesh                                          | Win MFU peak | adj peak | Cum MFU | adj cum | TPS Win  | Peak Mem      | vs 2n+              |
| ------- | --------------------------------------------- | ------------ | -------- | ------- | ------- | -------- | ------------- | ------------------- |
| 16k     | 1n DS-Z2+EP=8+Liger                           | 44.30 %      | 29.73 %  | 26.59 % | 17.84 % | 59,520   | 73.5 GB (92%) | matches 2n FSDP+EP champion (40.4 %) |
| 32k     | 1n DS-Z2+EP=8+Liger                           | **59.28 %**  | 35.76 %  | 46.21 % | 27.87 % | 47,930   | 78.1 GB (98%) | **beats 2n** by +3 pp (intra-node EP comm) |
| 64k     | 1n DS-Z3 dp=1 sp=8 +compile +Liger            | 7.99 %       | 4.46 %   | 1.32 %  | 0.74 %  | —        | 78.6 GB (99%) | very slow — DP=1 disables Z3 sharding |
| 64k     | 1n DS-Z3 dp=2 sp=4 +compile +Liger            | 23.23 %      | 12.95 %  | 14.16 % | 7.90 %  | —        | 78.3 GB (98%) | DP=2 lifts MFU 3× over SP=8 DP=1 |
| 128k+   | 1n SP=8/SP=4 + Liger                          | OOM          | —        | —       | —       | —        | 100 %         | over the wall, no offload allowed |

### Per-rank-seq sweet spot rule

**At long ctx, MFU is maximized when per-rank seq = 128k**. Tune SP to make `total_ctx / SP = 128k` per rank.

| Total ctx | Nodes | SP  | Per-rank seq | MFU peak | MFU cum  | Mem    |
| --------- | ----- | --- | ------------ | -------- | -------- | ------ |
| 256k      | 8     | 8   | 32k          | 32.6 %   | 24.1 %   | 36 GB  |
| 256k      | 8     | 4   | 64k          | 46.6 %   | 42.5 %   | 45 GB  |
| **256k**  | **8** | **2** | **128k**   | **59.6 %** | **52.4 %** | 69 GB |
| 512k      | 8     | 8   | 64k          | 45.0 %   | 39.2 %   | 46 GB  |
| **512k**  | **8** | **4** | **128k**   | 58.2 %   | 38.8 %   | 71 GB  |
| **1M**    | **8** | **8** | **128k**   | 37.5 %   | 35.7 %   | 72 GB  |

### Compile is mandatory at 256k+

DS-Z3+SP+chunked oscillates wildly without compile (window MFU 5–47 %); compile delivers stable 40–60 %. Cum MFU gain from compile: +13 pp at 256k SP=8, +17 pp at SP=4, +7 pp at 512k SP=8.

### Required setup for chunked-CE on FSDP2

`fsdp_reshard_after_forward: false` in accelerate config (added as conditional template var in `benchmark/templates/accelerate/fsdp2.yaml.j2`). Without it, `lm_head.weight` is re-gathered per chunk during backward, and FSDP+EP+chunked hangs at NCCL scatter on 16k+ context.

### Open: FSDP+chunked hangs at NCCL on EP+CP

FSDP path with chunked-CE hangs after ~519 collectives enqueued, only 8 completed. Pattern: chunk_size×ranks coincides with the queue depth, suggesting `lm_head.weight` regathers per chunk despite `reshard_after_forward: false`. DS-Z2+EP+chunked sidesteps this (DS doesn't shard params across DP). DS-Z3+SP+chunked also sidesteps (no EP). FSDP+EP+CP+chunked at 64k+ would unlock the comm-bound regime but currently broken — see `upstream_todo.md` future work.

### SP+EP throughput sweep (broken loss — MFU reference only)

DS-Z2 + SP + EP=8 was tested for MFU comparison but **training is fundamentally broken** (Ulysses shards seq, transformers EP assumes full per-rank batch — the all-reduce across EP combines DIFFERENT token subsets → garbage). MFU is roughly the same as SP-only at the same context — adding EP doesn't help, just breaks correctness.

| Context | Nodes | Mesh                | Win MFU | Cum MFU | Total TPS | Peak GPU Mem  | Loss | mean_token_acc |
| ------- | ----- | ------------------- | ------- | ------- | --------- | ------------- | ---- | -------------- |
| 32k     | 2     | dp=8 sp=2 ep=8      | 26.05 % | 21.75 % | 70,680    | 73.8 GB (93%) | 8.16 ❌ | 0.04 ❌ |
| 64k     | 4     | dp=8 sp=4 ep=8      | 19.21 % | 14.81 % | 107,300   | 67.9 GB (85%) | 8.14 ❌ | 0.05 ❌ |
| 128k    | 4     | dp=4 sp=8 ep=8      | 18.38 % | 6.22 %  | 46,600    | 69.4 GB (87%) | 8.32 ❌ | 0.04 ❌ |
| 128k    | 8     | dp=8 sp=8 ep=8      | 18.13 % | 8.36 %  | 128,600   | 58.4 GB (73%) | 8.39 ❌ | 0.03 ❌ |

### Analysis

- **8× MFU improvement from new transformers**: 25.7% vs 2.8% (old). The fused `gate_up_proj` (2 matmuls instead of 3 per expert) is the primary driver: fewer kernel launches and better memory access patterns.
- **EP correctness gap is the dominant blocker.** Every EP run on this branch (sdpa/FA3, sonicmoe v1 / sonicmoe Ilyas / grouped_mm, FSDP2 / DS-Z3) produces broken initial logits with first-step loss proportional to EP degree (EP=2 → 9.0, EP=4 → 29.8, EP=8 → 58–62). Only the no-EP baseline (`dp=16 + FA3 + sonicmoe v1`) starts cleanly (loss=2.15, grad_norm=3.28). All "EP" MFU numbers below measure real matmul throughput but **do not certify training correctness**. Highest-priority debugging target.
- **Patched sonicMoE (Ilyas) is required for any EP ≥ 2 forward pass.** Upstream `kernels-community/sonic-moe v1` does not skip sentinels and OOB-gathers on non-local rows (CUDA illegal access). Ilyas's fork handles sentinels in metadata stage. Until upstreamed, the kernel revision must be set explicitly in `transformers/integrations/hub_kernels.py`.
- **FA3 unlocks +10 pp window MFU at 16k EP=8** (32.36% → 42.66%). New throughput peak; once the EP correctness bug is fixed, this becomes the new SOTA training number.
- **DS-Z3+EP works** as of [PR #45548](https://github.com/huggingface/transformers/pull/45548). Same window MFU as FSDP2+EP, ~7 GB lower peak (50.3 vs 57.8 GB). Same correctness issue.
- **DS-Z3+SP+FA3+sonicMoE is the new long-context recipe and is training-correct**: 18.98% window @ 64k (2n, 99% mem, 50 steps), 19.31% peak window @ 128k (4n, 85% mem, 30 steps). Both initial losses ~1.7, decreasing. **4.5× peak MFU over the CP path at half the nodes** for 128k — and the CP path is also NaN-broken at long context, so the comparison is between "fast and correct" vs "slow and incorrect".
- **CP is retired for MoE in two ways**: (1) lower MFU (4–8% vs 19% for SP), (2) broken correctness — CP+EP runs at 64k/128k all show `loss=0, grad=nan` from step 1, same EP weight-loading bug.
- **FA3 consistently improves MFU**: +11% at 16k (25.7% vs 23.1%), +10.7% at 32k SP (14.5% vs 13.1%), +7.9% at 64k SP (12.5% vs 11.6%). FA3 is the best attention kernel for every MoE config tested.
- **FA3 + CP is incompatible**: accelerate's context parallelism requires sdpa. SP is the only long-context path for FA3.
- **Liger incompatible with MoE**: Liger's fused SwiGLU assumes 2D weight shapes `[intermediate, hidden]`, crashes on MoE fused expert 3D tensors `[num_experts, intermediate, hidden]`.
- **Flat mesh EP is optimal**: EP=8 (2D) and EP=16 (flat) give identical *throughput* because EP uses all-reduce, not all-to-all. (Both still NaN.)

### SonicMoE kernel variants

Two kernel revisions are tracked separately:

| Tag                        | Repo / revision                                                                        | Sentinel handling                                       | Status           | Use case                                                                                 |
| -------------------------- | -------------------------------------------------------------------------------------- | ------------------------------------------------------- | ---------------- | ---------------------------------------------------------------------------------------- |
| `sonicmoe v1`              | [`kernels-community/sonic-moe`](https://huggingface.co/kernels-community/sonic-moe)    | none — caller must pre-clamp non-local expert ids       | upstream         | EP=1 only (no sentinels), or with wrapper-level clamp+mask                               |
| `sonicmoe (Ilyas patched)` | [`IlyasMoutawwakil/sonic-moe@main`](https://huggingface.co/IlyasMoutawwakil/sonic-moe) | **kernel-native skip** for sentinel ids ≥ `num_experts` | NOT yet upstream | All EP sizes (2/4/8); forward + backward; required for any EP run that doesn't pre-clamp |

The Ilyas fork removes the wrapper-level `clamp + masked_fill` workaround documented in `benchmark/fix_fsdp_sonic.md` — the kernel handles sentinels in its own metadata-prep stage. All "NEW" 16k/32k/64k EP results below use the patched kernel; the older `kernels-community/sonic-moe v1` numbers (e.g. 30.67% wrapper-clamp baseline) are kept for comparison.

To switch between them, only the `revision` field in `transformers/integrations/hub_kernels.py` (`_HUB_KERNEL_MAPPING["sonic-moe"]`) changes. No code change in TRL or the wrapper is needed.

### Patched sonicMoE EP scaling at 16k (FSDP2, sdpa, 2 nodes × 8 H100, 50 steps)

All runs: `--enable_expert_parallel --experts_implementation sonicmoe`, kernel = `IlyasMoutawwakil/sonic-moe@main`. Numbers are end-of-50-step values from `benchmark/logs/` cross-checked against trackio (`/fsx/amine_dirhoussi/.cache/trackio/trl-sft-benchmark.db`). "Win peak" = max `mfu_window` observed across the run; "Win last" = value at final step. TPS (cum) is total cumulative; TPS (win) is steady-state from `tps_window`.

| EP  | Mesh       | World | Sentinel | Cum MFU | Win last | Win peak | TPS cum | TPS win | Peak GPU Mem  | Loss step 1 | Training | Slurm    |
| --- | ---------- | ----- | -------- | ------- | -------- | -------- | ------- | ------- | ------------- | ----------- | -------- | -------- |
| 2   | dp=8, tp=2 | 16    | 50%      | 11.71%  | 28.31%   | 31.44%   | 31,550  | 76,260  | 76.4 GB (96%) | 9.01        | ❌ NaN/Inf grad | 22091929 |
| 4   | dp=4, tp=4 | 16    | 75%      | 26.16%  | 32.32%   | **33.54%** | 70,480 | 87,050 | 62.4 GB (78%) | 29.80       | ❌ Inf grad → NaN | 22091987 |
| 8   | dp=2, tp=8 | 16    | 87.5%    | 26.08%  | 32.36%   | **33.54%** | 70,250 | ~87,000 | 57.8 GB (73%) | 58.99      | ❌ Loss → 0, NaN | 22091931 |

vs OLD `sonicmoe v1 + wrapper clamp` at EP=8 (also broken training, see below): 30.67% window — kernel-native skip gains **+1.7 pp** end-to-end.

vs `grouped_mm + PR #45621` at EP=8 (broken training): 27.50% window — patched sonicMoE gains **+4.86 pp** at the same parallelism.

Window MFU plateau at 32–34% for EP ≥ 4. **All EP runs above produce broken gradients** (NaN/Inf at step 1). The throughput numbers measure real matmul flops; they do not certify training correctness — see "EP correctness gap" below.

### FA3 + EP (FSDP2) — new MoE MFU peak at 16k (5-step debug runs)

The previously-reported FA3+EP "incompatibility" was a self-inflicted offline-mode race in `trl/scripts/sft.py`: `HF_HUB_OFFLINE=1` was set after pre-warming sonicmoe but before FA3's two-phase kernel load (`load_and_register_attn_kernel` + `lazy_import_flash_attention`) could fetch from hub. Pre-warming both FA3 paths before the offline flip resolves it. Full root-cause analysis in `benchmark/debug_fa3_ep.md`.

These are 5-step debug runs (max_steps=5), so cum MFU is heavily dragged by warmup. Window MFU is steady-state. Loss step 1 is the diagnostic for correctness.

| Run                         | Mesh       | Attn | Expert kernel    | Cum MFU | Win last | Win peak | TPS cum | TPS win | Peak GPU Mem  | Loss step 1 | Training | Slurm    |
| --------------------------- | ---------- | ---- | ---------------- | ------- | -------- | -------- | ------- | ------- | ------------- | ----------- | -------- | -------- |
| FA3 + EP=8 + sonicmoe (Ilyas) | dp=2, tp=8 | FA3 | sonicmoe (Ilyas) | 11.97% | **42.66%** | 43.08% | 32,250 | 114,900 | 57.8 GB (73%) | 62.0        | ❌ NaN @ step 2 | 22092267 |
| FA3 + EP=8 + grouped_mm     | dp=2, tp=8 | FA3  | grouped_mm       | 30.09%  | 35.06%   | 35.23%   | 81,050  | 94,450  | 58.3 GB (73%) | 62.0        | ❌ NaN @ step 2 | 22092277 |

**+10 pp window MFU vs sdpa+EP=8+sonicmoe (32.36 → 42.66 %)** — the attention contribution unlocked by FA3 at the same parallelism. The `FA3 + grouped_mm` cum MFU is much higher than `FA3 + sonicmoe` not because the kernel is faster, but because the second run hit an already-warm `~/.cache/huggingface/kernels/` from the first; per-step latency was 2.7s vs 7.5s on the first step. Window MFU is the apples-to-apples number.

Caveat: loss collapses to 0 with `grad_norm: nan` from step 2 onward (first-step `loss=62` vs expected ~2 for a Qwen3-30B base model). Open EP correctness bug, not specific to FA3 — same NaN pattern in sdpa+EP runs.

### DS-Z3 + EP — unblocked by [`transformers#45548`](https://github.com/huggingface/transformers/pull/45548)

Previously DS-Z3+EP failed at `from_pretrained` with `ValueError: DeepSpeed Zero-3 is not compatible with passing a 'device_map'` — even with `device_map=None`, `check_and_set_device_map` filled it from the global torch device context, then the DS-Z3 guard rejected it. PR #45548 adds a `PreTrainedModel.has_ep` property and routes EP+DS through the standard (non-zero3) loading path.

| Run                                  | Mesh       | Attn | Expert kernel    | Cum MFU | Win last | Win peak | TPS cum | TPS win | Peak GPU Mem  | Loss step 1 | Training | Slurm    |
| ------------------------------------ | ---------- | ---- | ---------------- | ------- | -------- | -------- | ------- | ------- | ------------- | ----------- | -------- | -------- |
| DS-Z3 + EP=8 + sdpa + sonicmoe (Ilyas) | dp=2, tp=8 | sdpa | sonicmoe (Ilyas) | 11.60% | 32.72%   | 32.72%   | 31,250  | 88,140  | 50.3 GB (63%) | 62.11       | ❌ NaN @ step 2 | 22092280 |

Same window MFU as FSDP2+EP=8+sonicmoe (32.36 %), with **~7 GB less peak memory** (50.3 vs 57.8 GB). DS-Z3's optimizer-state sharding wins on the memory side; FSDP2's expert-prefetch wins on raw kernel overlap. NaN issue identical.

### EP correctness gap — every EP run on this branch produces wrong gradients

Verified by reading `train/loss` step 1 and `train/grad_norm` from log files / trackio for every EP run since the patched sonicMoE landed:

| Config                                  | First-step loss | First-step grad_norm | Final-step loss | Final-step grad_norm |
| --------------------------------------- | --------------- | -------------------- | --------------- | -------------------- |
| Expected (Qwen3-30B-A3B base, fresh)    | ~2.0            | ~3                   | should decrease | should decrease      |
| FSDP2 + EP=8 + sdpa + sonicmoe (Ilyas)  | 58.99           | 318                  | 0               | nan                  |
| FSDP2 + EP=4 + sdpa + sonicmoe (Ilyas)  | 29.80           | inf                  | 0               | nan                  |
| FSDP2 + EP=2 + sdpa + sonicmoe (Ilyas)  | 9.01            | nan                  | 0               | nan                  |
| FSDP2 + EP=8 + FA3 + sonicmoe (Ilyas)   | 62.0            | nan                  | 0               | nan                  |
| FSDP2 + EP=8 + FA3 + grouped_mm         | 62.0            | nan                  | 0               | nan                  |
| DS-Z3 + EP=8 + sdpa + sonicmoe (Ilyas)  | 62.11           | 1.41                 | 0               | nan                  |

The "no-EP" baseline (FSDP2 + dp=16 + FA3 + sonicmoe v1) starts at `loss=2.15, grad_norm=3.28` — the expected initial state. Every EP run starts with broken initial logits (loss 9–62), proportional to EP degree. Some lower-EP variants don't NaN at step 1 (EP=2 has finite-but-NaN-grad, EP=4 has inf), but all converge to `loss=0, grad=nan` within ~10 steps.

This points at **EP weight loading**, not the routing kernel: forward at step 0 (before any optimizer update) already produces wrong logits proportional to EP degree. Most likely the packed `gate_up_proj`/`down_proj` from `MergeModulelist + Concatenate` is ending up on the wrong EP rank when sharding distributes experts across the EP group. Throughput numbers are still meaningful for kernel/hardware benchmarking; gradients are not.

### Long context — patched sonicMoE (training-correct ✓ vs broken ❌)

These runs use `EP=1` (just SP for sequence parallelism) and train cleanly — initial loss ~1.7, decreasing. The CP variants have EP active and inherit the EP correctness bug above (loss=0 at step 1, NaN gradients).

#### 64k

| Config                                 | Nodes | Backend | Mesh                  | Attn | Expert kernel    | Cum MFU | Win last | Win peak | TPS cum | TPS win | Peak GPU Mem  | Loss step 1 | Steps        | Training | Slurm    |
| -------------------------------------- | ----- | ------- | --------------------- | ---- | ---------------- | ------- | -------- | -------- | ------- | ------- | ------------- | ----------- | ------------ | -------- | -------- |
| FSDP2 + EP=8 + CP=4 + sonicmoe         | 4     | FSDP2   | dp=1,tp=8,cp=4,ep=8   | sdpa | sonicmoe (Ilyas) | 7.47%   | —        | 8.01%    | —       | —       | 53.0 GB (67%) | 0.0 (NaN)   | 50           | ❌ NaN   | —        |
| **DS-Z3 + SP=4 + FA3 + sonicmoe**      | **2** | DS-Z3   | dp=4,sp=4             | FA3  | sonicmoe (Ilyas) | **14.97%** | 18.98% | 19.07%   | 54,260  | 17,200  | 79.1 GB (99%) | 1.72        | 50           | ✓        | 22092241 |
| DS-Z3 + SP=4 + FA3 + sonicmoe (4n)     | 4     | DS-Z3   | dp=8,sp=4             | FA3  | sonicmoe (Ilyas) | 14.12%  | 20.33%   | 21.26%   | 102,300 | 36,840  | 65.9 GB (83%) | 1.64        | 25 (canceled, Ulysses padding crash @ step 30) | ✓ | 22092253 |

The 4-node 64k run was canceled at step 30 due to the separate Ulysses padding bug (`batch's seqlen=54223 isn't divisible by sp-size=4`). Window MFU peaked at 21.26%, ~+2 pp over 2-node — gain mostly from doubled DP throughput. 64k MoE at 99% GPU mem on 2 nodes is the hardware ceiling; 4 nodes drops to 83% but does not unlock more headroom for context.

CP path is **broken at 64k+ on this branch** — `loss=0` step 1 in the FSDP2+EP+CP=4 run, same NaN signature as the 16k EP runs. Both the CP version (broken correctness) and SP version (correct) measure 8% / 19% peak window MFU respectively; the SP path also saturates GPU memory while the CP path has 33% headroom (CP is comm-bound, not memory-bound).

#### 128k — first viable Qwen3-30B-A3B SFT result on this stack

| Config                                 | Nodes | Backend | Mesh                  | Attn | Expert kernel    | Cum MFU | Win last | Win peak | TPS cum | TPS win | Peak GPU Mem  | Loss step 1 | Steps                 | Training | Slurm    |
| -------------------------------------- | ----- | ------- | --------------------- | ---- | ---------------- | ------- | -------- | -------- | ------- | ------- | ------------- | ----------- | --------------------- | -------- | -------- |
| FSDP2 + EP=8 + CP=8 + sonicmoe         | 8     | FSDP2   | dp=1,tp=8,cp=8,ep=8   | sdpa | sonicmoe (Ilyas) | 4.07%   | —        | 4.25%    | 62,640  | —       | 51.5 GB (65%) | 0.0 (NaN)   | 50                    | ❌ NaN   | —        |
| FSDP2 + EP=8 + CP=4 + sonicmoe         | 4     | FSDP2   | dp=4,tp=8,cp=4,ep=8   | sdpa | sonicmoe (Ilyas) | —       | —        | —        | —       | —       | 76.8 GB (96%) | OOM         | OOM                   | —        | —        |
| **DS-Z3 + SP=8 + FA3 + sonicmoe**      | **4** | DS-Z3   | dp=4,sp=8             | FA3  | sonicmoe (Ilyas) | **13.15%** | 14.49% | **19.31%** | 101,100 | 13,930 | 67.4 GB (85%) | 1.75        | 30 (canceled to free nodes) | ✓        | 22092254 |

DS-Z3+SP=8+FA3+sonicmoe at 128k delivers **~4.5× the peak window MFU at half the node count** vs the FSDP2+EP+CP=8 baseline (19.31% on 4 nodes vs 4.25% on 8 nodes), and **trains correctly while the CP variant doesn't** (loss=0/NaN). 67.4 GB peak vs 51.5 GB CP peak — the SP run is closer to capacity but is using its memory for actual attention activations rather than CP staging buffers.

Per-rank seq is the same in both 64k 4n and 128k 4n SP configs (16k after Ulysses shard) — the FA3 + sonicmoe combo holds steady-state window MFU around 18–21% across both, gated by attention compute and DS-Z3 cache flushes (memory pressure causes 1–2 dips per 5 logging steps).

### Remaining blockers

1. **EP NaN bug** (broad): every EP ≥ 2 config produces broken initial gradients; loss/grad pattern proportional to EP degree (EP=2 has first-step loss=9, EP=4 has loss=29, EP=8 has loss=58–62). Affects every backend/kernel combination tested. Highest priority for end-to-end correctness.
2. **DS-Z3 + SP + EP combo**: DeepSpeed batch-size assertion (`world_size=8` after EP-mesh carving, expects 32). Config-level fix only — not a code bug.
3. **Ulysses SP padding constraint**: `batch_seqlen % sp_size == 0` enforced per-batch; packed THUDM/LongAlign-10k samples can violate. Needs `pad_to_multiple_of=sp_size` in collator.
4. **SonicMoE upstreaming**: `IlyasMoutawwakil/sonic-moe@main` needs to land in `kernels-community/sonic-moe` so future runs don't need the manual revision pin.

---

## Qwen3-32B (Dense), Comparison Model

### Full results

| Context | Nodes | GPUs | DP  | TP  | CP  | SP  | Backend | Liger | Attn     | MFU       | TPS    | TPS/GPU | Peak GPU Mem  |
| ------- | ----- | ---- | --- | --- | --- | --- | ------- | ----- | -------- | --------- | ------ | ------- | ------------- |
| 16k     | 2     | 16   | 16  | 1   | 1   | —   | fsdp2   | no    | sdpa     | 36.5%     | 19,280 | 1,205   | 79.3 GB (99%) |
| 16k     | 2     | 16   | 16  | 1   | 1   | —   | fsdp2   | yes   | sdpa     | 41.1%     | 21,690 | 1,356   | 59.7 GB (75%) |
| 16k     | 2     | 16   | 16  | 1   | 1   | —   | fsdp2   | yes   | FA3      | **51.3%** | 27,080 | 1,693   | 59.7 GB (75%) |
| 16k     | 4     | 32   | 32  | 1   | 1   | —   | fsdp2   | no    | sdpa     | 37.7%     | 39,790 | 1,243   | 72.2 GB (90%) |
| 16k     | 4     | 32   | 32  | 1   | 1   | —   | fsdp2   | yes   | sdpa     | 40.8%     | 43,020 | 1,344   | 44.3 GB (55%) |
| 16k     | 4     | 32   | 32  | 1   | 1   | —   | DS-Z3   | no    | sdpa     | 33.9%     | 35,770 | 1,118   | 76.6 GB (96%) |
| 16k     | 2     | 16   | 8   | 2   | 1   | —   | fsdp2   | yes   | sdpa     | 38.5%     | 20,310 | 1,269   | 79.4 GB (99%) |
| 32k     | 2     | 16   | 16  | 1   | 1   | —   | fsdp2   | no    | sdpa     | —         | —      | —       | — OOM         |
| 32k     | 2     | 16   | 16  | 1   | 1   | —   | fsdp2   | yes   | sdpa     | 43.1%     | 16,960 | 1,060   | 78.4 GB (98%) |
| 32k     | 2     | 16   | 16  | 1   | 1   | —   | fsdp2   | yes   | FA3      | **59.0%** | 23,200 | 1,450   | 78.4 GB (98%) |
| 32k     | 2     | 16   | 8   | 1   | 2   | —   | fsdp2   | no    | sdpa     | 18.1%     | 14,190 | 887     | 79.4 GB (99%) |
| 32k     | 2     | 16   | 8   | 1   | 2   | —   | fsdp2   | yes   | sdpa     | 19.8%     | 15,550 | 972     | 60.9 GB (76%) |
| 32k     | 4     | 32   | 16  | 1   | 2   | —   | fsdp2   | no    | sdpa     | 18.4%     | 28,870 | 902     | 73.3 GB (92%) |
| 32k     | 4     | 32   | 16  | 1   | 2   | —   | fsdp2   | yes   | sdpa     | 19.4%     | 30,460 | 952     | 45.5 GB (57%) |
| 32k     | 2     | 16   | 8   | 1   | 1   | 2   | DS-Z3   | yes   | FA2 (SP) | 25.0%     | 19,630 | 1,227   | 69.4 GB (87%) |
| 32k     | 2     | 16   | 8   | 1   | 1   | 2   | DS-Z3   | yes   | FA3 (SP) | 29.7%     | 23,380 | 1,461   | 69.4 GB (87%) |
| 32k     | 2     | 16   | 8   | 1   | 1   | 2   | DS-Z3   | no    | FA2 (SP) | —         | —      | —       | — OOM         |
| 32k     | 2     | 16   | 8   | 1   | 1   | 2   | DS-Z3   | no    | FA3 (SP) | —         | —      | —       | — OOM         |
| 32k     | 4     | 32   | 32  | 1   | 1   | —   | fsdp2   | yes   | FA3      | **58.7%** | 46,126 | 1,441   | 63.0 GB (79%) |
| 64k     | 2     | 16   | 8   | 1   | 2   | —   | fsdp2   | yes   | sdpa     | 20.7%     | 10,772 | 673     | 79.2 GB (99%) |
| 16k     | 2     | 16   | 8   | 2   | 1   | —   | fsdp2   | no    | sdpa     | —         | —      | —       | — OOM         |
| 16k     | 2     | 16   | 4   | 4   | 1   | —   | fsdp2   | no    | sdpa     | —         | —      | —       | — OOM         |
| 16k     | 2     | 16   | 2   | 8   | 1   | —   | fsdp2   | no    | sdpa     | —         | —      | —       | — OOM         |
| 32k     | 2     | 16   | 8   | 2   | 1   | —   | fsdp2   | no    | sdpa     | —         | —      | —       | — OOM         |
| 16k     | 2     | 16   | 4   | 4   | 1   | —   | fsdp2   | yes   | sdpa     | —         | —      | —       | — OOM         |
| 16k     | 2     | 16   | 2   | 8   | 1   | —   | fsdp2   | yes   | sdpa     | —         | —      | —       | — OOM         |

### Impact of Liger + FA3

| Context | Variant       | MFU       | TPS/GPU | vs baseline                    |
| ------- | ------------- | --------- | ------- | ------------------------------ |
| 16k     | baseline sdpa | 36.5%     | 1,205   | —                              |
| 16k     | + liger       | 41.1%     | 1,356   | +13%                           |
| 16k     | + liger + FA3 | **51.3%** | 1,693   | **+41%**                       |
| 32k     | baseline sdpa | —         | —       | OOM                            |
| 32k     | + liger       | 43.1%     | 1,060   | enables 32k                    |
| 32k     | + liger + FA3 | **59.0%** | 1,450   | enables 32k + 1.37× over liger |

### Dense vs MoE comparison (same hardware, best config per model)

| Config         | Qwen3-32B (dense) | Qwen3-30B-A3B (MoE) | Dense/MoE |
| -------------- | ----------------- | ------------------- | --------- |
| 16k, 2n, FSDP2 | 36.5%             | 22.5% (EP=16)       | 1.62×     |
| 16k, 4n, FSDP2 | 37.7%             | 22.3% (EP=32)       | 1.69×     |
| 16k, 4n, DS-Z3 | 33.9%             | 18.9%               | 1.79×     |
| 32k, 2n, CP=2  | 18.1%             | 13.2% (EP=16)       | 1.37×     |

### Best configs for Qwen3-32B

| Context  | Best MFU  | Config                      |
| -------- | --------- | --------------------------- |
| 16k      | **51.3%** | FSDP2, liger+FA3, 2n, DP=16 |
| 32k      | **59.0%** | FSDP2, liger+FA3, 2n, DP=16 |
| 32k (4n) | **58.7%** | FSDP2, liger+FA3, 4n, DP=32 |
| 32k SP   | 29.7%     | DS-Z3, SP=2+FA3+liger, 2n   |
| 64k      | **20.7%** | FSDP2, liger+sdpa, 2n, CP=2 |

### Analysis

- **Dense is 1.4-1.7× more MFU-efficient than MoE** at same hardware. No EP all-reduce overhead, no expert routing, FSDP shards more efficiently.
- **Liger + FA3 = 59% MFU at 32k**, the highest across all benchmarks. Liger enables DP-only 32k (OOM without), FA3 adds 37% on top.
- **Liger + FA3 scales to 4 nodes**: 58.7% MFU at 4n (32k) vs 59.0% at 2n, near-perfect scaling. Total TPS doubles (46,126 vs 23,200). Peak memory drops from 78 GB to 63 GB with more FSDP sharding.
- **64k works on 32B with liger + CP=2**: 20.7% MFU at 79.2 GB (99%), a tight fit but it runs. Can't use FA3 with CP (sdpa only). DS-Z3 SP+FA3+liger 64k still running (job 22081013).
- **SP+FA3 works well on 32B**: +19% over FA2 (29.7% vs 25.0%). Larger attention matrices give FA3 more compute to optimize.
- **TP=2 with liger** gives 38.5% MFU, slightly below DP-only+liger (41.1%) due to TP all-reduce overhead. TP≥4 OOMs even with liger. Without liger, all TP configs OOM on 32B.

---

## Qwen3-235B-A22B (MoE, 128 experts, 8 active) — FSDP2 + EP

Model: 235B total params, 22B active. 470 GB in bf16. 94 layers.

### Results (8 nodes, EP=64)

| Context | Nodes | GPUs | DP  | CP  | EP  | Offload | MFU   | TPS   | TPS/GPU | Peak GPU Mem   | Status          |
| ------- | ----- | ---- | --- | --- | --- | ------- | ----- | ----- | ------- | -------------- | --------------- |
| 16k     | 8     | 64   | 64  | 1   | 64  | no      | —     | —     | —       | 79.5 GB (99%)  | OOM             |
| 16k     | 8     | 64   | 32  | 2   | 64  | no      | —     | —     | —       | 79.7 GB (100%) | OOM             |
| 16k     | 8     | 64   | 16  | 4   | 64  | no      | —     | —     | —       | 79.7 GB (100%) | OOM             |
| 16k     | 8     | 64   | 8   | 8   | 64  | no      | —     | —     | —       | 79.8 GB (100%) | OOM             |
| 16k     | 8     | 64   | 16  | 4   | 64  | yes     | 0.51% | 4,496 | 70      | 43.3 GB (54%)  | Yes             |
| 32k     | 8     | 64   | 32  | 2   | 64  | yes     | 2.9%  | 8,483 | 133     | 70.1 GB (88%)  | Yes             |
| 32k     | 8     | 64   | 8   | 8   | 64  | yes     | 0.67% | 7,783 | 122     | 44.8 GB (56%)  | Yes             |
| 16k     | 16    | 128  | 64  | 2   | 128 | no      | —     | —     | —       | 73.8 GB (92%)  | CUDA peer error |

### Analysis

- **CPU offload required on 8 nodes**: model params (470 GB) + optimizer states (1.4 TB) exceed 64 × 80 GB = 5.1 TB total GPU memory after FSDP overhead. No amount of CP helps — OOM is from params/optimizer, not activations.
- **CP=2 gives best MFU (2.9%)**: lower CP means less ring-attention overhead. CP=8 drops to 0.67%. CP=4 at 16k gives 0.51%. All dominated by CPU↔GPU parameter transfer time.
- **16 nodes (128 GPUs) without offload**: 73.8 GB peak (should fit), but crashed on CUDA peer memory error from EP=128 cross-node all-to-all.
- **Missing PP**: Megatron reference uses TP=4, PP=16, EP=8, CP=2 on 128 GPUs. PP distributes layers across stages without offloading. TRL/accelerate lacks PP for MoE.

### What would improve 235B performance

1. **Pipeline Parallelism (PP)**: PP=16 with EP=8 = 128 GPUs, no offload needed
2. **2D EP mesh**: EP=8 intra-node (NVLink) + DP=8 inter-node — avoids cross-node all-to-all
3. **More nodes without offload**: 32 nodes (256 GPUs) could fit with FSDP sharding alone
4. **Native flash_attn**: would speed up attention compute

---

## torch.compile Results

### Original (pre-fix, 2026-04 mid)

| Model | Compile | MFU       | TPS/GPU | Runtime (20 steps) |
| ----- | ------- | --------- | ------- | ------------------ |
| 30B   | No      | **23.4%** | 3,934   | 83s                |
| 30B   | Yes     | 8.7%      | 1,463   | 224s               |

The 2.7× slowdown was traced to `accelerate.fsdp2_prepare_model()` calling `torch.compile(module)` (returns `OptimizedModule` whose `__call__` bypasses `nn.Module._call_impl`, breaking FSDP hooks). Fixed by [accelerate PR #4022](https://github.com/huggingface/accelerate/pull/4022).

### Post-fix (2026-04-28, accelerate PR #4022 + per-rank Triton cache)

| Setup                                       | Win MFU peak | Loss   | Verdict                                        |
| ------------------------------------------- | ------------ | ------ | ---------------------------------------------- |
| FSDP DP=16 + FA3 + compile (no EP)          | **34.87 %**  | 1.6 ✅ | matches the +6 pp claim from the PR           |
| **DS-Z2 + EP=8 + FA3 + sonicmoe + compile** | **36.7 %**   | 12.25 ✅ | new champion at 16k                          |
| FSDP + EP=8 + FA3 + sonicmoe + compile      | crash        | —      | Adam `_group_tensors_by_device_and_dtype` mismatch — DTensor mesh mix |
| FSDP + EP=8 + CP=4 + sonicmoe + compile     | Triton cache error | — | known cluster issue                       |

**Key finding:** compile + EP works only via the DS-Z2 path (uses plain tensors, no DTensor mesh mix). FSDP+EP+compile is blocked by the multi-mesh DTensor foreach grouping in the compiled optimizer. DS-Z2+EP+compile pulls **+8 pp** over DS-Z2+EP no-compile (28.6 → 36.7%).

**Setup notes for compile runs:**
- Use accelerate clone at `/fsx/amine_dirhoussi/accelerate/src` via `PYTHONPATH` (PR #4022 not yet released).
- Add `dynamo_config: {backend: inductor, use_fullgraph: false, use_regional_compilation: true}` to FSDP yaml.
- `use_fullgraph: false` required because RouterParallel has data-dependent branching that dynamo can't fullgraph.
- Per-rank `TRITON_CACHE_DIR` set in `trl/scripts/sft.py` (FSx contention otherwise).
- `@torch._dynamo.disable` on `grouped_mm_experts_forward` and `sonicmoe_experts_forward` (kernels already optimized; dynamo's `_getattr_static` failed on `gate_up_proj` access otherwise).

---

## NCCL Bandwidth Reference

| Op             | 1 node (NVLink) | 2 nodes (EFA) | Ratio   | Used by                         |
| -------------- | --------------- | ------------- | ------- | ------------------------------- |
| allreduce      | 448 GB/s        | 431 GB/s      | **96%** | Gradient sync, EP expert output |
| allgather      | 35 GB/s         | 15 GB/s       | 42%     | FSDP2 param unsharding          |
| reduce_scatter | 34 GB/s         | 13 GB/s       | 39%     | FSDP2 gradient sharding         |
| all_to_all     | 335 GB/s        | 37 GB/s       | **11%** | SP Ulysses token dispatch       |

Key: EP uses all-reduce (96% inter-node), not all-to-all. SP Ulysses uses all-to-all (11% inter-node). This explains why SP degrades sharply cross-node while EP doesn't.

---

## Cross-Model Summary

### Best MFU by model and context length

| Model         | Context  | Best MFU                                | Config                                       | Peak GPU Mem  | Train valid?         |
| ------------- | -------- | --------------------------------------- | -------------------------------------------- | ------------- | -------------------- |
| Qwen3-4B      | 16k      | 30.6%                                   | FSDP2, sdpa, 2n                              | 43.4 GB       | ✓                    |
| Qwen3-4B      | 32k      | **56.3%**                               | FSDP2, liger+FA3, 1n                         | 24.9 GB       | ✓                    |
| Qwen3-4B      | 64k      | 18.3%                                   | FSDP2, liger+sdpa, 1n, CP=2                  | 26.0 GB       | ✓                    |
| Qwen3-30B-A3B | 16k      | **34.7%** (Win last) / 26.2% (cum)      | FSDP2, FA3 + sonicmoe v1, 2n (no EP, dp=16)  | 68.6 GB (86%) | ✓                    |
| Qwen3-30B-A3B | 16k peak | **42.66%** (Win last) / 43.08% (peak)   | FSDP2, EP=8 + FA3 + sonicmoe (Ilyas), 2n     | 57.8 GB (73%) | ❌ NaN (open EP bug) |
| Qwen3-30B-A3B | 32k      | **14.66%** (cum, window not collected)  | DS-Z3, SP=2 + FA3 + sonicmoe (Ilyas), 2n     | 78.6 GB (99%) | ✓                    |
| Qwen3-30B-A3B | 64k 2n   | **18.98%** (Win last) / 14.97% (cum)    | DS-Z3, SP=4 + FA3 + sonicmoe (Ilyas), 2n     | 79.1 GB (99%) | ✓                    |
| Qwen3-30B-A3B | 64k 4n   | **21.26%** (Win peak) / 20.33% (last)   | DS-Z3, SP=4 + FA3 + sonicmoe (Ilyas), 4n     | 65.9 GB (83%) | ✓ (canceled @ 30)    |
| Qwen3-30B-A3B | 128k 4n  | **19.31%** (Win peak) / 14.49% (last)   | DS-Z3, SP=8 + FA3 + sonicmoe (Ilyas), 4n     | 67.4 GB (85%) | ✓ (canceled @ 30)    |
| Qwen3-32B     | 16k      | **51.3%**                               | FSDP2, liger+FA3, 2n                         | 59.7 GB       | ✓                    |
| Qwen3-32B     | 32k      | **59.0%**                               | FSDP2, liger+FA3, 2n                         | 78.4 GB       | ✓                    |
| Qwen3-32B     | 64k      | 20.7%                                   | FSDP2, liger+CP=2, 2n                        | 79.2 GB       | ✓                    |
| Qwen3-235B    | 16k      | 0.51%                                   | FSDP2, EP=64, CP=4, offload, 8n              | 43.3 GB       | ✓                    |
| Qwen3-235B    | 32k      | 2.9%                                    | FSDP2, EP=64, CP=2, offload, 8n              | 70.1 GB       | ✓                    |

### Key takeaways

1. **Fused gate_up_proj in new transformers is a huge win**: 23% vs 2.8% MFU on Qwen3-30B-A3B. Reduces expert matmuls from 3 to 2 with better memory access patterns. Enabled by native `Qwen3MoeExperts` ([PR #45436](https://github.com/huggingface/transformers/pull/45436)) support for Qwen3.
2. **Liger + FA3 is the best config for dense models**: 56–59% MFU on H100. Liger eliminates intermediate tensor materialization; FA3's Hopper-native kernels halve attention time. Not compatible with MoE for now (Liger's fused SwiGLU assumes 2D weight shapes; MoE fused experts are 3D). The single `use_liger` global flag is also a UX issue — would prefer per-kernel opt-in.
3. **EP is the dominant correctness blocker for Qwen3-30B-A3B**: throughput-wise, patched sonicmoe (Ilyas) at EP=8 reaches 32–43% window MFU, but **every EP run on this branch (sdpa/FA3, sonicmoe / grouped_mm, FSDP2 / DS-Z3) NaNs in training** with first-step loss 9–62 vs expected ~2. Only the no-EP baseline (`dp=16 + FA3 + sonicmoe v1`) trains correctly at 34.7% window MFU. The bug is in EP weight loading, not the routing kernel or attention backend.
4. **FA3 improves everything (throughput-wise)**: +11% on MoE standalone (FSDP2), +7–11% on MoE with SP (DS-Z3), +10 pp at 16k EP=8 (32.36% → 42.66%), +37–57% on dense with liger. FA3 is the best attention kernel for every non-CP config. But FA3 is incompatible with CP (accelerate's CP requires sdpa).
5. **Best (training-correct) strategy depends on context length**:
    - **≤16k dense**: FSDP2 + FA3 (+ liger).
    - **≤16k MoE (30B-A3B)**: FSDP2 + dp=16 + FA3 + sonicmoe v1 (no EP) — **34.7% window MFU**, only currently-correct EP-free option. EP variants reach 32–43% throughput but all NaN.
    - **32k dense**: FSDP2 + liger + FA3 (59% MFU).
    - **32k MoE (30B-A3B)**: DS-Z3 + SP=2 + FA3 + sonicmoe (Ilyas) (14.66%) > grouped_mm equivalents.
    - **64k dense**: FSDP2 + liger + CP=2 (20.7%).
    - **64k MoE (30B-A3B)**: DS-Z3 + SP=4 + FA3 + sonicmoe (Ilyas) (18.98% window, 2n; 21.26% peak, 4n) — only training-correct long-context recipe. CP path NaNs at 64k+ on this branch.
    - **128k MoE (30B-A3B)**: DS-Z3 + SP=8 + FA3 + sonicmoe (Ilyas) at 19.31% peak window (4n, 85% mem). First viable Qwen3-30B-A3B SFT result at this length on this stack; 4.5× MFU at half the nodes vs FSDP2+EP+CP=8 (which also NaNs).
6. **Dense models are 1.4–1.7× more MFU-efficient than MoE** at same total parameter count and hardware. MoE's advantage is lower active compute per token, not higher hardware utilization.
7. **235B needs Pipeline Parallelism**: CPU offload caps MFU at ~3% (32k CP=2) or <1% (16k/high CP). PP would distribute layers across pipeline stages without offloading. Missing from TRL/accelerate.
