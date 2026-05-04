# Blog post plan — "Training MoEs with TRL: how we made the open stack actually fast"

> **Working title options**
>
> - "Open MoE training, on the open stack — and why we report _real_ MFU"
> - "From 2.8% to 76% MFU: how we made TRL the fastest place to SFT a Qwen3-MoE"
> - "We trained Qwen3-30B-A3B at 1M context on 8 nodes. With TRL. With receipts."
>
> Pick one based on whether the framing leans more "rebuttal" or "results showcase". My recommendation: the second one — let the numbers carry the rebuttal.

---

## 0. The hook (lede + snark, 1–2 paragraphs)

A few weeks ago, an [open-weights/open-training post](https://www.workshoplabs.ai/blog/open-weights-open-training) and a flurry of dunk threads landed on `transformers` not being a "real" training framework. The argument goes: HuggingFace ships great inference and a great hub, but if you want to _train_ a frontier-size MoE you have to leave for Megatron / NeMo / a hand-rolled stack.

The premise is wrong. Or rather — it _was_ wrong about a month ago, and we've spent the last few weeks fixing it. This post is the receipt. We took the open stack — `trl.SFTTrainer` + `transformers` + `accelerate` + `deepspeed` — and pushed Qwen3-30B-A3B SFT from **2.8% MFU at 16k to 76% peak MFU at 128k**, then to **1M context training that actually steps cleanly**. Every fix is upstreamable. Most are already upstream or in-flight.

> **What "we did the work" looks like by the numbers** (call this out in the lede as the rebuttal-by-receipt):
>
> - **463 training runs** launched on AWS p5 between early-April and 04-29 — including the 32+ OOMs and the dead-end pivots, not just the wins.
> - **~8,900 H100 GPU-runs in aggregate** (one GPU's participation in one run = one GPU-run). Distribution across the sweep: 27 single-node runs, **309 two-node** (16 GPUs), 57 four-node (32 GPUs), 28 eight-node (64 GPUs), and 1 sixteen-node run (128 GPUs).
> - **81 config sweeps** written under `benchmark/configs/`, each a distinct `(model, ctx, parallelism, kernel)` permutation we wanted to verify.
> - **3 model architectures** stress-tested: Qwen3-4B (dense, easy mode), Qwen3-30B-A3B (MoE, the actual target), Qwen3-32B (dense anchor at the same parameter scale as the MoE).
> - **8 context lengths from 16k → 1,048,576** — a **64× span** on the same hardware and same trainer.
> - **51 result tables / 284 measured rows** in the consolidated [benchmark write-up](./sft_benchmark_notion.md) — including failed configs, kernel comparisons at fixed parallelism, and per-rank-seq sweet-spot scans.
> - **5 transformers PRs authored to make EP usable** (#45436, #45473, #45548, #45649, #45662), plus 1 accelerate PR validated (#4022) and 1 TRL PR cherry-picked (#5575).
>
> When people say "you don't have a serious training stack," they mean "you didn't run the matrix." We ran the matrix — **8,900 H100 GPU-runs of it.**

> **The "real MFU" flex** (the second piece of the rebuttal): every number in this post is reported in _two_ columns: the standard non-causal convention (the one PaLM, Megatron, nanoGPT, and most public MoE numbers use), and a **causal-corrected** column that subtracts the half of attention FLOPs that don't actually run under causal masking. We do this because Llama-2/3 and DeepSpeed-Ulysses report this way and we want our numbers to compose with theirs. The headline **76% peak at 128k is 40.5% causal-corrected** — still the highest published MoE MFU on this stack, and reported honestly.

---

## 1. Setup: why MoE, why SFT, what we benchmarked

This is the one context section before the technical thread. Three sub-points, each compact:

### 1.1 Why MoE is the architecture worth fixing

Recent releases have made MoE the **default architecture** for cost-conscious frontier work: Qwen3-30B-A3B (3B active / 30B total — 10× compute savings per token vs dense), DeepSeek-V3 (37B active / 671B total), Llama-4-Scout / Maverick, Mixtral, Qwen3-235B-A22B. The pattern is consistent: **training compute scales with active params, model capacity scales with total params, and inference is bounded by active** — so MoE is a Pareto win on serving cost.

For post-training (SFT / RLHF / GRPO / online-RL), this is *especially* attractive: you're tuning over many trajectories, often with low gradient-update count per token. Halving compute per token via MoE is a big deal.

**The catch**: if your training framework can't run MoE at >5% MFU, you give back the entire savings advantage in wall-clock time. **MoE is only a Pareto win if the framework is good.** This is the gap the post is about.

### 1.2 Why MoE is hard for FSDP-style frameworks

Three independent reasons, each load-bearing for a later Step:

1. **Asymmetric activation across ranks.** Different tokens route to different experts → different ranks accumulate gradient on different expert subsets. FSDP2's symmetric `reduce_scatter` requires all ranks to participate with the same shape. The naive `nn.ModuleList[nn.Linear]` layout makes this worse — the gradient *shapes* differ across ranks. Fix: fused `[num_experts, ...]` weight tensors so the gradient buffer is always the same shape (inactive expert slices just hold zeros). transformers 5.6.0 ships this natively. (Step 1.)
2. **Expert distribution wants its own mesh dimension.** With 128 experts you want some ranks to *own* a slice (Expert Parallelism, EP). EP wants its own mesh, which has to compose with FSDP DP, CP, and SP. Composability between all four is the entire reason §2 has 13 steps. (Step 2 + Step 5.)
3. **Token routing is a comm pattern, not a forward.** Two camps: (a) **all-to-all dispatch** — ship tokens to wherever the expert lives (DeepSpeed-MoE, Megatron); small comm cost (tokens × hidden), bandwidth-hungry, sensitive to inter-node BW. (b) **Replicate-then-mask** — every rank sees all tokens, masks routing scores for non-local experts, all-reduces partial outputs. Bigger comm but uses all-reduce, which on EFA runs at **96% of intra-node bandwidth** while all-to-all only hits **11%**. transformers picked (b). On EFA, that's the right choice by ~9× on the inter-node leg.

> **Snarky aside the post should keep**: "transformers EP = replicate-mask" is sometimes characterized as "not real EP" by people who think EP must mean all-to-all. On EFA hardware, **replicate-mask is the faster choice**.

### 1.3 Why SFT is the right thing to optimize first

Three lines:

- **SFT is the dense-compute half of every modern post-training recipe.** GRPO, DPO, RLHF, online-RL all interleave SFT-style forward/backward passes through the policy. Optimize `SFTTrainer` and the rest comes "for free."
- **Long context is the open frontier.** Reasoning RL, long-CoT distillation, agentic traces — all 32k+ and increasingly 128k+. The public guidance for "how to train a 30B MoE at 128k on commodity H100s" is *roughly: don't*.
- So the focus: `SFTTrainer × Qwen3-30B-A3B × long-context`.

### 1.4 The model matrix and hardware

**Models** (one line each):

- **Qwen3-4B (dense)** — easy mode. The "what should it look like when everything works" reference. Single-node fits.
- **Qwen3-30B-A3B (MoE, 128 experts, 8 active)** — the target.
- **Qwen3-32B (dense)** — the **comparison anchor**. Same parameter scale as 30B-A3B, no MoE machinery. Lets us measure "how much MFU does going MoE actually cost on the same hardware?"

**Hardware**: AWS p5.48xlarge, H100 SXM5 80GB, 8 GPUs/node, NVLink intra-node, EFA inter-node. **1 → 16 nodes** tested (8 → 128 GPUs).

**Backends covered**: FSDP2 (via accelerate), DeepSpeed ZeRO-2, ZeRO-3, with optional Ulysses SP, CP, EP, torch.compile, Liger, sonicMoE, chunked-CE.

(In the published post: include the full setup table from [`sft_benchmark_notion.md`](./sft_benchmark_notion.md) here — it's already the right format.)

---

## 2. The decision tree — one continuous thread

This is the spine of the post. Walk the reader through **the actual decisions a researcher makes** when handed a 30B MoE and a cluster, in the order they're forced to make them. Each step's *output state* (working recipe + measured MFU) becomes the *starting point* for the next step.

> **Format**: every step has a **STATE** (where we are now), a **WANT** (what we want next), a **TRY** (the obvious first thing), what **BROKE**, what we **FIXED/PIVOTED** to, and the new **MEASURED state** that the next step inherits. Read top-to-bottom; it's a single thread.

### Step 0 — The starting state

We have **Qwen3-30B-A3B**: 30B total params (60 GB bf16 weights), 128 experts per layer, 8 active per token. We want to SFT it with TRL on H100 SXM5 80 GB nodes. Single GPU obviously can't hold the model — sharding is mandatory.

### Step 1 — Pick a distributed strategy: FSDP2 or DeepSpeed-ZeRO-3?

- **WANT**: shard the model across ranks so it fits.
- **TRY**: FSDP2 (TRL/accelerate's first-class path). Run baseline.
- **BROKE**: `RuntimeError: Detected mismatch between collectives on ranks. Rank 12: TensorShape=[505155840], Rank 0: TensorShape=[240914688]`.
  - **WHY**: Qwen3-30B-A3B stores experts as `nn.ModuleList[nn.Linear]` (384 small weight matrices per layer). MoE routing is data-dependent → different ranks activate different experts → gradient *shapes* differ across ranks → FSDP2's symmetric `reduce_scatter` fails. Architectural incompatibility, not a bug.
- **FIX (initial)**: TRL-side `fuse_moe_experts()` helper + a forked checkpoint. Pre-stack 128 experts into `[128, ...]` tensors so all ranks have the same gradient shape (inactive expert slices just hold zeros).
- **PARALLEL TRY**: DS-ZeRO-3 baseline. Works without a fork, but **1–2% MFU**.
- **FIX (proper)**: transformers 5.6.0 ships native `Qwen3MoeExperts` with fused `gate_up_proj` (one `[128, 2*intermediate, hidden]` tensor; 2 matmuls per expert instead of 3). Drop our fork.
- **STATE → next step**: FSDP2 + native fused experts at 16k = **23% MFU**, training-correct, baseline established. Each rank still holds **all 128 experts** sharded across DP — so per-rank memory is `30B / world_size`, not `(num_local_experts/128) × 30B`. We're using DP-sharding but no expert distribution.

### Step 2 — Reduce per-rank expert memory: turn on Expert Parallelism

- **STATE**: 23% MFU at 16k, but every rank still gathers all 128 expert weights at every layer. We want each rank to *own* only `num_experts/EP` experts → less expert memory per rank → headroom for longer context or bigger batch later.
- **WANT**: actual expert sharding (EP), not just DP sharding of the fused tensor.
- **TRY**: `--enable_expert_parallel`.
- **BROKE 1**: TRL silently bypassed EP entirely. `create_model_from_path` re-added `device_map="auto"` after SFTTrainer removed it → `from_pretrained` took the device-map path → EP distribution hooks never ran. **All early "EP" wandb runs were plain FSDP2 with replicated experts.** Embarrassing miss.
- **FIX**: TRL-side check for `distributed_config` in model kwargs.
- **TRY (again)**: `--enable_expert_parallel`, this time actually wired.
- **BROKE 2**: forward outputs all-zero at EP=1 (no NaN). Forward outputs *plausible-looking* but bf16-divergent at EP=2/4/8.
- **DEBUG**: forward-output match against non-EP ground truth (which is the only correctness test that catches this). Five independent transformers bugs found:
  1. `RouterParallel._prepare_output_fn` — wrong shape under EP remapping.
  2. Weight-loading plan lookup — TP regex over EP plan → `KeyError`.
  3. `grouped_mm_experts_forward` — uninit memory in non-local-expert rows.
  4. `GroupedGemmParallel.shard_tensor` — global rank vs mesh-local rank.
  5. `num_experts` divided once per weight, not once per module.
- **FIX**: **[transformers#45436](https://github.com/huggingface/transformers/pull/45436)** + **[#45473](https://github.com/huggingface/transformers/pull/45473)** (both merged). 4 regression tests added — zero EP coverage existed before.
- **STATE → next step**: EP=4 with TP=4 at 16k = 22.7% MFU, **training-correct**. EP doesn't beat no-EP on throughput yet (we haven't optimized attn or expert kernels), but the EP foundation is now verifiable. **Phase 1 of EP is one PR away from finished — but we won't hit that PR until Step 5.**

### Step 3 — Speed up attention: swap sdpa → FA3

- **STATE**: 23% MFU at 16k, EP correct. Profile says ~30% of step time is attention. sdpa is the default; FA3 should be faster on Hopper.
- **WANT**: replace sdpa with `kernels-community/vllm-flash-attn3`.
- **TRY**: `attn_implementation=kernels-community/vllm-flash-attn3`. Run on dense Qwen3-4B first as the reference (no MoE machinery to get in the way).
- **MEASURE on dense Qwen3-4B at 32k**: 35.9% (sdpa+Liger) → **56.3% (FA3+Liger)**. **+57%.** FA3 is real.
- **MEASURE on MoE at 16k FSDP DP=16**: 23.1% → **25.7%**. **+11%.** Smaller because attention is a smaller fraction of MoE compute, but still net win.
- **STATE → next step**: 16k FSDP+EP=8+FA3 = better attention but **expert dispatch is still grouped_mm** (the default). The next bottleneck is the expert kernel.

### Step 4 — Speed up expert dispatch: swap grouped_mm → sonicMoE

- **STATE**: attention is FA3-fast, EP is correct, but `grouped_mm` is still the default expert kernel.
- **WANT**: a faster fused MoE kernel.
- **TRY**: `--experts_implementation sonicmoe` — `kernels-community/sonic-moe`, a CuteDSL fused kernel.
- **MEASURE**: 16k FSDP DP=16 + FA3 + sonicMoE = **34.7%**, vs grouped_mm+FA3 = 28.1%. **+23%** at steady state.
- **TRY**: stack sonicMoE with EP > 1.
- **BROKE**: kernel CUDA-illegal-accesses on rows whose `top_k` ids are all EP sentinels.
- **FIX**: **[transformers#45621](https://github.com/huggingface/transformers/pull/45621)** (Ilyas Moutawwakil's patch — *not* ours; we just pin to it). Kernel-native sentinel handling in metadata stage. (See also Step 5 fix-4/fix-5: backward leaks remained until #45621's 2026-05-01 follow-up commits closed them in the kernel — at which point we dropped the wrapper-side clamp + simplified the grouped_mm wrapper from 4 `masked_fill`s back to 2.)
- **STATE → next step**: sonicMoE + EP forward works. We have FA3 + sonicMoE + EP — but they don't compose cleanly with FSDP2 yet, because **EP params are still plain `nn.Parameter` while FSDP DP params are DTensors**. Adam's `_fused_adamw_` rejects mixed lists. We need to close that gap.

### Step 5 — Compose EP with FSDP2: DTensor wrap → close Phase 1 of EP

- **STATE**: each piece individually works (FSDP2 ✓, EP correct ✓, FA3 ✓, sonicMoE ✓), but stacking EP+FSDP+Adam blows up at the optimizer step.
- **WANT**: EP+FSDP composable end-to-end.
- **TRY**: stack them. Adam errors on mixed Tensor/DTensor list.
- **FIX 1**: **[transformers#45662](https://github.com/huggingface/transformers/pull/45662)** — wrap EP-sharded experts as DTensor on the EP mesh. Now everything is DTensor; Adam is happy.
- **BROKE (side effect of #45662)**: now that EP params are DTensors, accelerate's `_prepare_tp` "no DTensor → skip" guard stops firing → ImportError on `ReplicateParallel`.
- **FIX 2**: 5-line `has_ep` skip in accelerate's `_prepare_tp`. Local; queued as a standalone accelerate PR.
- **BROKE (second side effect)**: `clip_grad_norm_` calls `_foreach_norm`, which stacks per-param norms across meshes; FSDP-DP-mesh and EP-mesh DTensors error with `RuntimeError: All operands in aten.stack.default must have the same mesh`.
- **FIX 3**: skip grad-clip when `has_ep`. Local, returns 0 for telemetry. Proper fix is in PyTorch's `clip_grad_norm_`; PyTorch issue queued.
- **BROKE (kernel backward)**: sonicMoE's hand-written backward produces NaN gradients on EP sentinels through `DTensor.to_local()`. Same bug shape repeats inside `torch._grouped_mm`'s backward path (uninit `d_input` rows past `offsets[-1]`) — once #45621 takes the histogram-tail-drop route in the wrapper, the autograd graph leaks NaN via the gather's `index_add_` backward into `d_hidden_states`.
- **FIX 4 (workaround, ~2pp MFU cost — *temporary*)**: wrapper-level `clamp(0, num_experts-1)` in `sonicmoe_experts_forward`, plus a 4-`masked_fill` firewall in `grouped_mm_experts_forward`. Diagnosed via vacuum tests + capture-and-replay of production tensors (see `benchmark/study_moe_autograd.md` and `benchmark/grouped_mm_pr45621_comment.md`). Reported to Ilyas via PR #45621 comments.
- **FIX 5 (proper kernel-side fix, landed 2026-05-01)**: PR #45621 author landed kernel-side fixes that close both the sonic-moe sentinel bug and the grouped_mm autograd leak. Wrapper-side clamp dropped entirely from `sonicmoe.py`; `grouped_mm_experts_forward` simplified to 2 `masked_fill_`s (pre + post) since intermediate sentinel-row NaN is never consumed by the next grouped_mm.
- **MEASURE (clamp era, kept here for narrative)**: 16k FSDP2 + EP=8 + FA3 + sonicMoE + clamp = **40.4 % window MFU**, training-correct.
- **MEASURE (kernel-fix era, 2026-05-01, no clamp)**: same config = **45.4 / 48.2 % mean / peak window MFU**. **+5–8 pp** recovered — much more than the ~2 pp we initially expected, because the kernel's cleaner sentinel-skip path compounds with FA3 attention. Loss = 13.4, healthy. **The new 16k EP champion. Phase 1 of EP is done.**
- **STATE → next step**: 40.4% at 16k is great. Now we want longer context (the actual research goal — long-CoT SFT, agentic traces, R1-style reasoning RL all live at 32k+). Per-rank seq doesn't fit on 80 GB GPU at 32k. We need to shard the sequence dim.

### Step 6 — Extend context: try CP first (it's the obvious accelerate-native path)

- **STATE**: 40.4% at 16k. Want 32k.
- **WANT**: shard the sequence dim. Two options exist: **CP** (Context Parallelism, ring-attention) or **SP** (Ulysses Sequence Parallelism, all-to-all).
- **WHY CP first**: it's the native accelerate path (`parallelism_config.cp_size`), no DeepSpeed required, integrates with FSDP2.
- **TRY**: FSDP2 + EP=8 + CP=2 at 32k.
- **BROKE 1**: accelerate hard-guards `cp_size > 1` to `attn_implementation=sdpa`. **CP + FA3 doesn't compose** (the guard is conservative; FA3 supports causal masking and seq sharding internally, but the integration isn't there). So CP forces us back to sdpa.
- **MEASURE 1 (sdpa-only)**: 32k FSDP+EP=8+CP=2 + sdpa = **15.6% window MFU**. Works, but losing FA3 hurts.
- **TRY**: push CP higher to fit longer context. 64k FSDP+EP=8+CP=4.
- **BROKE 2**: every EP=8 config OOMs at the **same single 18.55 GiB allocation**. The EP-replicated expert activation buffer (`seq × num_local_experts × moe_intermediate × 2 bytes`) — transformers' EP is replicate-mask, every rank materializes the full per-batch buffer.
- **STATE → next step**: CP path is correct but capped at 32k MoE, slow (15.6% MFU), and can't use FA3. Time to try SP. SP needs DeepSpeed.

### Step 7 — Pivot to SP: DeepSpeed-Z3 + Ulysses

- **STATE**: CP is bottlenecked. SP composes with FA3, runs cross-node well in theory.
- **WANT**: DS-Z3 + Ulysses SP at 32k+.
- **TRY**: load Qwen3-30B-A3B with DS-Z3 + EP enabled.
- **BROKE 1**: `from_pretrained` hangs at 0% GPU for 10+ minutes. DS-Z3's env vars route every weight through the ZeRO-3 path; EP's weight loader expects the standard path.
- **FIX 1**: **[transformers#45548](https://github.com/huggingface/transformers/pull/45548)** (merged) — `PreTrainedModel.has_ep` property; `from_pretrained` detects EP+DS and bypasses the Z3 loading path.
- **TRY**: DS-Z3 + SP at 32k.
- **BROKE 2**: SP runs crash around step 25 with `ValueError: batch's seqlen=X isn't divisible by sp-size=N`. Packed `LongAlign-10k` samples produce non-divisible lengths.
- **FIX 2**: `--pad_to_multiple_of 8` (or `sp_size`). Queued as a TRL auto-default PR.
- **MEASURE**: 32k DS-Z3+SP=2+FA3 = **21.98% window MFU**. **+8 pp over CP path.** ✓
- **TRY**: stack SP with EP — should compound seq sharding (SP) with expert sharding (EP) for 64k+.
- **BROKE 3 (architectural, not a bug)**: DS-Z3 + SP + EP produces broken loss (loss=8, token_acc=0.04). Ulysses shards seq across SP, transformers' EP assumes full per-rank batch — the EP all-reduce combines DIFFERENT token subsets → garbage.
- **PIVOT**: Drop SP+EP stacking. SP alone at long context. Or — find another way to free memory and stay on the EP path.
- **STATE → next step**: SP path = 22% at 32k, scales to longer context but caps at the SP architectural limit. Meanwhile the EP path is stuck at 32k by the 18.55 GiB buffer. **What's the *other* big tensor we can shrink?** lm_head logits at 64k = ~20 GB.

### Step 8 — Free the lm_head: chunked-CE → DS-Z2 pivot → long-context champions

- **STATE**: stuck. EP+CP capped at 32k by EP buffer; DS-Z3+SP capped at 22%; SP+EP architecturally broken.
- **WANT**: free enough activation memory to fit the EP buffer at 32k+ per-rank.
- **TRY**: TRL [PR #5575](https://github.com/huggingface/trl/pull/5575) — chunked cross-entropy. Process `(B*S, vocab=151936)` logits in chunks of 256 instead of materializing all at once. Frees ~20 GB at 64k.
- **TRY**: FSDP2 + EP=8 + chunked at 32k.
- **BROKE**: NCCL hangs after ~519 collectives, only 8 complete. The per-chunk lm_head matmul triggers an all-gather under FSDP (lm_head is sharded across DP). Chunk size × ranks saturates the NCCL queue.
- **PIVOT**: switch to **DS-Z2**. ZeRO-2 doesn't shard params across DP → no per-chunk all-gather → chunked-CE works.
- **PRE-REQ for DS-Z2+EP**: a **7-patch recipe across 3 repos** (transformers `tensor_parallel.py` backend branch + DS engine MoE detection + `trainer.py` create_optimizer + `_clip_grad_norm` gating + sonicMoE wrapper-clamp + TRL EP branch + `pre-init deepspeed.comm`). All local; queued as a 3-PR upstream split (transformers / DeepSpeed / TRL).
- **MEASURE (clamp era)**:
  - 32k DS-Z2+EP=8+FA3+sonicMoE+chunked = **45.86%**. **+24 pp over the old DS-Z3+SP=2 32k champion.**
  - 64k = **57.23%**. **+37 pp.**
  - 128k = **69.10%**. **+50 pp.**
- **MEASURE (kernel-fix era, 2026-05-01, no clamp)**: 32k DS-Z2+EP=8+FA3+sonicMoE+chunked = ~50.7% mean / ~50.9% peak (job 22099328) — **+5 pp** over the clamp baseline. Loss = 8.0, healthy. The 64k / 128k re-runs are queued; the same +5–8 pp delta is expected to push the 128k record from 69 % → ≥75 %.
- **WHY this is the largest jump in the post**: chunked-CE removes the lm_head logit tensor; DS-Z2 removes the per-chunk all-gather; together they let the EP path scale to 128k. Each fix individually is small; the stack is the unlock.
- **STATE → next step**: 128k = 69.10% via the EP+chunked path. Past 128k per-rank seq, the 18.55 GiB EP buffer can't fit even with chunked. **EP path is over.** For 256k+ we have to drop EP entirely.

### Step 9 — Try torch.compile (in parallel — it's worth a side-quest at every long-context step)

- **STATE**: every recipe so far uses eager. torch.compile should give us 5–10 pp on dense forward.
- **WANT**: `--torch_compile`.
- **TRY**: turn it on. 16k FSDP DP=16.
- **BROKE**: 2.7× **slowdown** (23.4% → 8.7%). Opposite of what we wanted.
- **DEBUG**: `accelerate.fsdp2_prepare_model()` was calling `torch.compile(module)` and assigning the returned `OptimizedModule` to `model`. `OptimizedModule.__call__` bypasses `nn.Module._call_impl` → **FSDP2 forward pre/post hooks never fire** → params never gathered.
- **FIX**: **[accelerate#4022](https://github.com/huggingface/accelerate/pull/4022)** — call `torch.compile` at the right level so hooks survive.
- **MEASURE**: FSDP DP=16 + FA3 + compile @ 16k = **34.87%**. **+6 pp.** Matches the PR's claim.
- **TRY**: compile + EP under FSDP.
- **BROKE**: Adam's `_group_tensors_by_device_and_dtype` strict-asserts under compile when EP-mesh and FSDP-DP-mesh DTensors mix. Tried non-fused Adam (`--optim adamw_torch`) — same crash; foreach groups regardless.
- **PIVOT**: compile + EP under DS-Z2. DS-Z2 uses plain `nn.Parameter` (no DTensor mesh) → no foreach mesh-mix → works.
- **MEASURE**: DS-Z2+EP=8+FA3+sonicMoE+compile @ 16k = **36.7%**. **+8 pp over DS-Z2+EP no-compile.**
- **OBSERVATION (saved for Step 10)**: at long context (64k+) compile gives ~0 pp on peak — the path is comm-bound, not compute-bound. But compile **stabilizes** the SP path at 256k+ (window MFU steady 40–60% vs no-compile 5–47% oscillation). Useful at the extreme contexts.

### Step 10 — Try Liger (transferred from dense)

- **STATE**: We've measured Liger on dense models (Qwen3-4B/32B): **+13% to +57%** depending on stack. It should transfer.
- **WANT**: same Liger speedup on MoE.
- **TRY**: `apply_liger_kernel_to_qwen3_moe(model)`.
- **BROKE**: device-side assert. Crash. Conventional wisdom (Liger issue tracker, our own original notes): "Liger's fused SwiGLU assumes 2D weights; MoE fused experts are 3D. Liger doesn't support MoE."
- **WE ALMOST STOPPED HERE.** Took the conventional wisdom at face value and shelved Liger for MoE.
- **DEBUG (going back)**: built a [single-GPU repro](./test_liger_qwen3_moe.py) without EP. Liger ran fine. Max diff vs eager = 8.79e-3. **The 3D-weights story was wrong.**
- **DEBUG 2 (with EP)**: built [`test_liger_qwen3_moe_ep.py`](./test_liger_qwen3_moe_ep.py). Crashed only with EP. Real root cause: Liger's `_patch_swiglu_module(experts, LigerExperts)` rebinds `experts.forward = LigerExperts.forward`, **bypassing transformers' `@use_experts_implementation` dispatcher** that routes to the EP-aware kernel under EP. Then `F.one_hot(top_k_index, num_classes=self.num_experts)` collides with the EP sentinel value (= `num_local_experts` = `num_classes`) → out-of-range → device assert.
- **FIX**: `--liger_kernel_config '{"swiglu":false}'`. Disables only the broken patch. Keeps `LigerRMSNorm` + `liger_rotary_pos_emb` + `LigerFusedLinearCrossEntropyLoss` (the bigger MFU wins anyway). Lets transformers' EP-aware dispatcher handle experts.
- **MEASURE (clamp era)**: Liger wins at every context 16k–1M, +0.3 to +25 pp peak vs chunked-CE.
  - 32k: 56.62% (vs 45.86% chunked).
  - 64k: 66.46% (vs 57.23%).
  - **128k: 76.29% peak / 74.69% cumulative** — **all-time MoE MFU record on this stack.** (Causal-corrected: 40.5%.)
- **MEASURE (kernel-fix era, 2026-05-01, no clamp)**: PR #45621's kernel-side fix landed; we re-ran the 32k Liger champion and hit **63.5 % mean / 65.0 % peak window MFU** (+8.4 pp peak over 56.62 %). Loss = 8.0 (vs 11–15 with the clamp wrapper) — gradients are not just clean but *more correct* than the clamp's "expert E−1 with score 0" workaround. **Highest 30B-on-2-node Win MFU we've measured.** A full re-sweep of 64k / 128k / 256k+ with the new kernel is queued as the next batch — expect ~5–10 pp peak gains across the board.
- **UPSTREAM**: file a Liger PR — auto-detect EP via `model.config._experts_implementation` and skip the swiglu patch automatically. Repro included.
- **LESSON to emphasize in the post**: **the conventional wisdom was wrong, and we repeated it.** Almost shelved Liger for MoE entirely. The single-GPU repro disproved it in 30 minutes. *When debugging composition issues, isolate the smallest crash before believing the dominant explanation.*

### Step 11 — The accessibility question: how far can we push on a single node × 8 GPUs?

- **STATE**: every champion so far is multi-node (2n at 16k, 4n at 128k, 8n at 1M). Real users without a multi-node cluster are locked out.
- **WANT**: the longest context Qwen3-30B-A3B SFT can hit on **one** H100 SXM5 node (8 × 80 GB = 640 GB total). No CPU offload, no Slurm multi-node, no inter-node EFA.
- **WHY this matters for the post**: this is the "can a small lab do it?" question. Multi-node Slurm clusters are gatekeepers; a recipe that runs on one node turns 30B-A3B SFT into something a single 8×H100 server can do.
- **TRY**: throw the full optimization stack (FA3 + sonicMoE + EP=8 + Liger + chunked, then add compile + SP at long context) at 1n × 8.
- **MEASURE**:

  | Context | Recipe | Win MFU peak | Causal-adj | Peak Mem | vs 2n+ |
  | ------- | ------ | ------------ | ---------- | -------- | ------ |
  | **16k** | DS-Z2 + EP=8 + FA3 + sonicMoE + Liger | **44.30%** | **29.73%** | 73.5 GB (92%) | matches the 2n FSDP+EP champion (40.4%) |
  | **32k** | DS-Z2 + EP=8 + FA3 + sonicMoE + Liger | **59.28%** | **35.76%** | 78.1 GB (98%) | **beats 2n by +3 pp** — intra-node EP comm is faster than cross-node |
  | 64k | DS-Z3 dp=2 sp=4 + compile + Liger | 23.23% | 12.95% | 78.3 GB (98%) | DP=2 lifts MFU 3× over SP=8 DP=1 (DP=1 disables Z3 sharding entirely) |
  | 64k | DS-Z3 dp=1 sp=8 + compile + Liger | 7.99% | 4.46% | 78.6 GB (99%) | not viable — DP=1 makes Z3 a no-op |
  | 128k+ | SP=8 / SP=4 + Liger | OOM | — | 100% | over the wall — no offload allowed |

- **HEADLINE FINDING**: **at 32k, a single 1n × 8 H100 node beats the 2n recipe by +3 pp** (59.28% vs 56.62% peak window MFU). EP=8 fits intra-node on NVLink, so the EP all-reduce runs at ~448 GB/s instead of EFA's 431 GB/s — and you save the cross-node hop entirely. **One node is *better* than two for 30B-A3B at 32k.**
- **WHERE 1n stops working**: 64k drops sharply because losing DP forces SP=8/DP=1 (which neutralizes Z3 sharding) — only DP=2 SP=4 keeps any throughput. 128k+ OOMs without CPU offload.
- **STATE → next step**: 1n × 8 GPUs is a viable recipe up to 32k context (ahead of 2n!) and a usable-but-slow recipe at 64k (DP=2 SP=4). For 128k+ we go back to multi-node.

### Step 12 — Beyond 128k: drop EP, tune per-rank-seq, make compile mandatory

- **STATE**: 128k = 76.29% via EP+chunked+Liger (2n+). At 256k per-rank seq the EP buffer doesn't fit even with chunked. Must drop EP.
- **WANT**: 256k, 512k, 1M context that actually steps.
- **TRY**: DS-Z3 + SP=8 + chunked @ 256k.
- **MEASURE 1**: 256k @ SP=8 = **32.6%** peak window. Per-rank seq = 32k.
- **OBSERVATION**: window MFU oscillates 5–47% across steps (NCCL stalls under heavy ZeRO-3 cross-node load). Hard to use in practice.
- **TRY**: torch.compile + chunked + SP. Hypothesis: compile serializes the path → less variance.
- **MEASURE 2**: 256k @ SP=8 + compile = **28.89%** peak but **steady 22–32%** window. Variance gone. **Compile becomes mandatory at 256k+.**
- **TRY**: scan SP × per-rank-seq across 256k / 512k / 1M.
- **MEASURE → discovered the per-rank-seq sweet-spot rule**: MFU peaks when `per_rank_seq = 128k`. Tune SP to hit it.

  | Total ctx | Nodes | SP | Per-rank seq | Peak MFU |
  | --------- | ----- | -- | ------------ | -------- |
  | 256k | 8 | 8 | 32k | 32.6% |
  | 256k | 8 | 4 | 64k | 46.6% |
  | **256k** | **8** | **2** | **128k** | **63.62%** |
  | **512k** | **8** | **4** | **128k** | **63.26%** |
  | **1M** | **8** | **8** | **128k** | **62.33%** |

- **STATE (final)**: 1M context training works at 62.33% peak window MFU (31.41% causal-corrected). Loss healthy. **New frontier reached.**

### Step 13 — Honest reporting (last decision: how to publish the numbers)

- **STATE**: every champion measured. Now: how do we report?
- **OBSERVATION**: the standard MFU formula counts attention as if every token attends to the full seq (the PaLM/Megatron/nanoGPT convention). Causal masking actually halves attention compute → reported MFU is inflated relative to the Llama-2/3 / DS-Ulysses convention.
- **FIX**: post-hoc helper [`benchmark/adjust_mfu_causal.py`](./adjust_mfu_causal.py). Both columns reported in every headline table.
- **MEASURE (one more time, with the honesty column)**:
  - 16k: 40.4% raw → **27.1% causal-adjusted**
  - 64k: 66.46% raw → **37.1% causal-adjusted**
  - 128k: **76.29% raw → 40.5% causal-adjusted** (all-time MoE record on this stack)
  - 1M: 62.33% raw → **31.4% causal-adjusted**

### One-paragraph summary of the whole thread

**Start with FSDP2 + native fused experts (23%). Turn on EP and audit the 5 forward-correctness bugs (still 23%, but verifiable). Add FA3 (+10pp). Add sonicMoE (+6pp on top of grouped_mm at the same parallelism). Close the EP+FSDP composition gap with #45662 (40.4% at 16k). Want longer context: try CP, slow because no FA3; try SP, works at 22% at 32k after #45548 + padding fix; SP+EP architecturally broken. Free the lm_head with chunked-CE; FSDP+chunked hangs on per-chunk gather; pivot to DS-Z2 (no DP-param sharding, no gather). DS-Z2+EP+chunked wins at 32k–128k (45–69%). Add compile via accelerate#4022 (+8pp at 16k EP, no peak help at long context but stabilizes the SP path at 256k+). Transfer Liger from dense; it crashes; conventional wisdom blames 3D weights but a single-GPU repro disproves that — real bug is dispatcher bypass; `swiglu=False` works; Liger beats chunked-CE at every context, sets a new 128k record (76.29% / 40.5% causal). Side-question: how much can we do on one node? Throw the full stack at 1n × 8 — 32k actually beats 2n at 59.28% (intra-node EP comm wins). Beyond 128k drop EP, tune per-rank-seq=128k via SP, compile mandatory; 1M context unlocked at 62%. Both raw and causal-corrected MFU columns in every table.**

That's the whole thread. Thirteen decisions, in order, each one's output state being the next one's starting state. **No "Chain A vs Chain B" parallel framing.** One continuous walk through what a researcher would actually do.

---

### Notes for the published version

- Each "Step N → Step N+1" transition should be a small visual diagram in the post — a single arrow with the **MEASURED state** on the left and the **next WANT** on the right.
- The "we almost stopped here" beat (Step 10 Liger) is worth highlighting visually — it shows the research process honestly.
- The thread should fit on one page if printed; deeper detail lives in **§3 (technical deep-dives)** and **Appendix B (chronological diary)**.

---

### 2.X — Sidebar: the EP-PR series in detail (deep-dive, referenced from Steps 2 + 5)

> **Why this sidebar exists.** Steps 2 and 5 of the main thread compress the EP work into "5 forward bugs → #45436 + #45473" and "EP+FSDP composition → #45662 + side-effects". That's correct but undersells what was actually a **multi-week, multi-PR effort across two distinct phases**. This sidebar is the full version: every EP-related PR by date, who authored it, and what it unblocked. Read this if you want the EP receipts; skip it if you're following the linear thread.

> **The framing for this section in the post.** When we started, **EP was broken at every layer of the stack** — TRL silently bypassed it, transformers had five forward-correctness bugs, FSDP couldn't compose with it, accelerate's TP prep tripped on it, DeepSpeed-Z3 couldn't load it, and the fastest MoE kernel (sonicMoE) had no sentinel awareness. We opened **a series of five transformers PRs over two weeks** ([author search](https://github.com/huggingface/transformers/pulls?q=+is%3Apr+author%3AAmineDiro+)) to make it correct, then layered Ilyas Moutawwakil's sonic-moe kernel work on top to make it fast. This chain is the spine of the post; everything else (long context, compile, Liger) is downstream of EP being usable.

This chain has **two distinct phases**. The post should mark them explicitly:

- **Phase 1 — make EP correct.** Ship-or-die. Until forward outputs match the non-EP baseline in bf16 at every EP degree, every throughput number is a lie.
- **Phase 2 — make EP fast.** Once correct, claw back the perf compromises (sentinel-mask cost, wrapper-level clamp, conservative grad-clip skip).

#### Phase 1 — make EP correct (the PR series, 04-15 → 04-27)

Five transformers PRs, one TRL fix, one accelerate patch. Ordered by what each unblocks:

1. **TRL-side EP bypass** (local, before any transformers PR). `create_model_from_path` re-added `device_map="auto"` *after* SFTTrainer removed it → `from_pretrained` took the device-map path → EP distribution hooks never ran. **All early "EP" wandb runs were plain FSDP2 sharding** with experts replicated everywhere. Embarrassing, but the kind of bug only surfaces under a correctness audit.
   - **FIX**: TRL-side check for `distributed_config` in model kwargs. Lives on the `benchmark-sft-moe` branch; will land in the upstream TRL G2 PR.
2. **[transformers#45436](https://github.com/huggingface/transformers/pull/45436) — Qwen3-MoE EP plan + 2D-mesh fixes** (merged).
   - Added `base_model_ep_plan` to `Qwen3MoeConfig` (only `gpt_oss` and `llama4` had one before).
   - Fixed `GroupedGemmParallel.shard_tensor` to use mesh-local rank instead of global CUDA index (so GPU 2 at local-rank-0 stops trying to load experts 128–191).
   - Fixed `num_experts` divided once per module instead of once per weight tensor (was 128 → 64 → 32 after two weights).
3. **[transformers#45473](https://github.com/huggingface/transformers/pull/45473) — EP routing correctness** (merged).
   - Fixed `RouterParallel._prepare_output_fn` shape (was passing `(seq, num_local_experts)` where the kernel expected `(seq, top_k)` — produced **all-zero expert outputs** at EP=1).
   - Fixed weight-loading plan lookup (TP-plan regex over EP-plan values → `KeyError` on EP-only configs).
   - Made `grouped_mm_experts_forward` sentinel-aware (was leaving uninit GPU memory in non-local-expert rows).
   - **Added 4 regression tests** verifying forward-output match against non-EP ground truth at EP=1,2,4,8,16. **Zero EP test coverage existed before.**
4. **[transformers#45548](https://github.com/huggingface/transformers/pull/45548) — DS-Z3 + EP loading** (merged).
   - DS-Z3's env vars forced `from_pretrained` down a ZeRO-3 path at every weight-load decision. EP needs the standard path — it creates the model on meta, loads via `WeightConverter`, then lets `deepspeed.initialize` wrap.
   - Added `PreTrainedModel.has_ep` property; `from_pretrained` now detects EP+DS and bypasses the Z3-specific loading path.
   - Without this, DS-Z3+EP hung at 0% GPU utilization for 10+ minutes in `from_pretrained`.
5. **[transformers#45649](https://github.com/huggingface/transformers/pull/45649) — FSDP2 cpu_ram_efficient_loading OOM** (in flight).
   - Bisected to PR #45050: changed `torch.empty_like` → `torch.zeros_like` for non-rank-0 FSDP placeholders. Benign on small models, but on Linux with anonymous mmap it forces a *physical memory commit* of every byte. For 8 ranks × 30B-A3B per node = ~480 GB CPU peak → cgroup OOM during `from_pretrained`.
   - Fix: drop the parameter materialization on non-rank-0 ranks (FSDP's broadcast overwrites them anyway); keep buffer materialization (per-rank, not broadcast).
6. **[transformers#45662](https://github.com/huggingface/transformers/pull/45662) — EP + FSDP DTensor wrap** (in flight).
   - Adam's `_fused_adamw_` rejects mixed Tensor + DTensor parameter lists. Pre-fix: EP params were plain `nn.Parameter`, FSDP DP params were DTensors → Adam blew up under FSDP2.
   - Fix: wrap EP-sharded experts as `DTensor` on the EP mesh. Now everything is DTensor; Adam is happy.
   - **Side effect**: accelerate's existing `_prepare_tp` "no DTensor → skip" guard stops firing post-#45662 → ImportError on `ReplicateParallel` (a class added upstream after our fork point).
   - **FIX (local, queued as standalone accelerate PR)**: 5-line `has_ep` skip in `_prepare_tp`. [§4 of local_only_patches.](./local_only_patches.md)

**End of Phase 1**: EP=8 trains **correctly** under FSDP2 (verified by forward-output match) and DS-Z2. DS-Z3+EP loads (after #45548) but still hits a rank-ordering issue at the broadcast step — pivoted to DS-Z2+EP for long context (Step 8); proper DS-Z3+EP fix documented in [§D-blocker of todo](./upstream_todo.md).

#### Phase 2 — make EP fast (Ilyas's sonicMoE + the clamp tax)

Phase 1 produced a correct but conservatively-slow EP path. The remaining ~2 pp MFU sits in two places:

1. **The sentinel-mask kernel cost.** The default `grouped_mm` kernel computes on every row, including EP sentinels (rows whose experts live on a different rank). Sentinel rows are masked to zero on the wrapper side — compute work is wasted.
2. **The wrapper-level clamp + masked_fill in `sonicmoe_experts_forward`.** Needed because the sonic-moe kernel's hand-written `Function.backward` produces NaN gradients on EP sentinels when going through `DTensor.to_local()`. The clamp lets the kernel run on sentinel rows; `masked_fill` zeros their contribution. Net-correct, ~2 pp MFU cost vs a kernel-native sentinel-skip backward.

**The work to claw this back, all in flight or pending:**

1. **[transformers#45621](https://github.com/huggingface/transformers/pull/45621) — sonicMoE Ilyas patch pin** (in flight, **not authored by us** — credit Ilyas Moutawwakil). Pins the kernel revision to [`IlyasMoutawwakil/sonic-moe@main`](https://huggingface.co/IlyasMoutawwakil/sonic-moe). Ilyas added kernel-native sentinel handling in the metadata stage. **Forward** is now sentinel-aware — pre-#45621, EP > 1 segfaults outright; post-#45621, EP > 1 forward works.
2. **The wrapper clamp is still required for backward.** Ilyas's patch fixes forward. Backward through `DTensor.to_local()` still NaNs on EP sentinels. We filed an upstream issue at Dao-AILab/sonic-moe with a [standalone kernel-only repro](./sonic_moe_upstream_repro.md): the kernel produces wrong upstream-flowing input grads (`dh`, `ds`) when many tokens have all `top_k` slots set to the EP sentinel. Forward output and parameter grads (`dw1`, `dw2`) are correct.
3. **Once Dao-AILab fixes the backward**, we remove the wrapper clamp. **+2 pp MFU recovery for free** at every EP context length. This is a queued follow-up PR, not yet possible.
4. **Once the kernel-native skip is everywhere**, we file a follow-up transformers PR explaining the autograd-through-`to_local` backward gap and citing both #45621 and #45662 as dependencies.

**Phase 2 is not done.** The post should be honest about this: the 76.29% headline includes the ~2 pp clamp tax. With the kernel-native backward landed, the same recipe would push toward ~78%. (Or, in causal-corrected terms, ~41.5% instead of 40.5%.) **We're not claiming Phase 2 is finished — we're claiming it's a known, scoped, in-progress recovery.**

#### Throughout: the local glue

Two patches that don't fit Phase 1 or Phase 2 cleanly but are part of the EP-correctness story:

- **`trainer.py:_clip_grad_norm` `has_ep` skip.** `clip_grad_norm_` calls `_foreach_norm` which stacks per-param norms across meshes; FSDP-DP-mesh and EP-mesh DTensors can't stack (`RuntimeError: All operands in aten.stack.default must have the same mesh`). Local skip returns `tensor(0.0)` for telemetry. Fine for benchmarking; **unsafe for production** (gradients aren't actually clipped). The proper fix is in PyTorch's `clip_grad_norm_` — we'll file a PyTorch issue. [§3 of local_only_patches.](./local_only_patches.md)
- **`trainer.py:create_optimizer` MoE param-group split** (under DS-Z2 only). Split MoE params into a separate optimizer group via `split_params_into_different_moe_groups_for_optimizer` so DS routes their grad reduce through `expert_data_parallel_group` (the small group) instead of the full DP group. Will land in the upstream DS-Z2+EP recipe PR (Step 8's long-context champion path).

#### MEASURE — what each phase actually unlocked

| Phase | What | Impact |
| ----- | ---- | ------ |
| Pre-Phase-1 | TRL silently bypassed EP | All early EP numbers were FSDP-with-replicated-experts. Throughput plausible; sharding fictitious. |
| Phase 1 (#45436 + #45473) | EP forward outputs correct at EP=1,2,4,8,16 | Throughput numbers become **meaningful**. EP=4+TP=4 at 16k = 22.7%. |
| Phase 1 (#45548) | DS-Z3 + EP loads | Unblocks DS-Z3 + EP exploration (later pivots to DS-Z2 due to Phase 1 limit). |
| Phase 1 (#45649) | FSDP2 cpu_ram_efficient OOM removed | Unblocks 30B-A3B loading on 8 ranks/node — required for every multi-node EP run. |
| Phase 1 (#45662) | EP+FSDP composable | **40.4% window MFU at 16k** — the EP champion at short context. |
| Phase 2 (#45621, Ilyas) | sonicMoE forward sentinel-aware | EP > 1 with sonicMoE no longer segfaults. Unlocks the +23% kernel speedup at EP. |
| Phase 2 (Dao-AILab issue, pending) | sonicMoE backward sentinel-aware | **+2 pp MFU recovery** when the wrapper clamp can be removed. Open. |

---

## 3. Technical deep-dives — the topics that need more than the linear thread

These four sub-sections elaborate on Steps 2 + 5 + 8 + 9 of the decision tree. Skip them if you've followed the thread; read them if you want code-level detail. Order matters less here than in §2 — each is self-contained.

### 3.1 The EP correctness audit — five forward-output bugs and the regression tests we left behind

(Elaborates Step 2.)

The five bugs, one line each:

1. **Wrong shape for routing weights** (#45473) — EP remapping changed routing weights from `(seq, top_k)` to `(seq, num_local_experts)`, but the kernel expected the original. EP=1 produced **all-zero expert outputs** with no NaN.
2. **Weight loading used wrong plan** (#45473) — TP-plan regex over EP-plan values → `KeyError` on EP-only configs.
3. **Sentinel rows hit uninit GPU memory** (#45473) — `grouped_mm` didn't handle EP sentinel ids → garbage output. After fixing #1, we got `0.0 * NaN = NaN` deterministically.
4. **Global rank vs mesh-local rank in `GroupedGemmParallel.shard_tensor`** (#45436) — on a 2D mesh, GPU 2 at local-rank-0 tried to load experts 128–191 (out of range).
5. **`num_experts` divided once per weight, not once per module** (#45436) — 128 → 64 → 32 after two weights.

PR #45473 added **4 regression tests** comparing forward outputs against the non-EP baseline at EP=1,2,4,8,16. **Zero EP test coverage existed before.** The verification matrix that should exist in any framework shipping EP: `forward-output-vs-non-EP-baseline at every supported EP degree, in bf16`. Now it does.

### 3.2 The 18.55 GiB wall + the per-rank-seq sweet-spot rule

(Elaborates Step 6 → 8 → 11 → 12.)

Up to 16k per-rank seq, EP=8 trains beautifully (40% MFU). At 32k+ per-rank seq, every EP=8 config OOMs at the **same single allocation**:

| Per-rank seq | EP buffer |
| ------------ | --------- |
| 16k | 9.275 GiB |
| **32k** | **18.55 GiB** |
| 64k | 37.09 GiB |

This is `seq × num_local_experts × moe_intermediate × 2 bytes`, materialized once per rank because EP is replicate-mask (every rank sees the full token batch). **Linear in per-rank seq.** Chunked-CE freed up enough room (by removing the lm_head logit tensor from the activation budget) to fit it through 128k. Past 128k per-rank, even chunked can't save the EP path.

That's where the empirical rule comes in. From a sweep of `(total_ctx, SP) × per-rank-seq`:

| Total ctx | Nodes | SP | Per-rank seq | Peak MFU |
| --------- | ----- | -- | ------------ | -------- |
| 256k | 8 | 8 | 32k | 32.6% |
| 256k | 8 | 4 | 64k | 46.6% |
| **256k** | **8** | **2** | **128k** | **59.6%** |
| **512k** | **8** | **4** | **128k** | **58.2%** |
| **1M** | **8** | **8** | **128k** | **37.5%** |

**MFU is maximized when `per_rank_seq = 128k`.** Tune SP to hit that. We didn't find this from theory; we found it from running ~250 configurations. It's the kind of operational fact that only a sweep produces.

### 3.3 torch.compile × FSDP2 × EP — three bugs, two fixes, one open issue

(Elaborates Step 9.)

- **Bug 1 — broken FSDP2 hooks**: `accelerate.fsdp2_prepare_model()` was calling `torch.compile(module)` and assigning the returned `OptimizedModule` to `model`. `OptimizedModule.__call__` bypasses `nn.Module._call_impl` → **FSDP2 forward pre/post hooks never fire** → params never gathered → 2.7× slowdown (23.4% → 8.7% MFU).
  - **Fix**: [accelerate#4022](https://github.com/huggingface/accelerate/pull/4022) — call `torch.compile` at the right level so hooks survive. **+6 pp** confirmed (FSDP DP=16 + FA3 + compile @ 16k = 34.87%).
- **Bug 2 — Adam mesh-mix under EP+FSDP+compile**: EP DTensors on EP mesh + FSDP DP DTensors on FSDP mesh → `_group_tensors_by_device_and_dtype` strict-asserts under compile. Tried non-fused Adam (`--optim adamw_torch`) — same crash, foreach groups regardless.
  - **Pivot**: DS-Z2+EP+compile. DS uses plain `nn.Parameter` (no DTensor mesh) → no foreach mesh-mix. **+8 pp** at 16k (28.6 → 36.7%).
  - **Open issue**: surgical PyTorch patch to `_group_tensors_by_device_and_dtype` to treat EP and FSDP DTensors as compatible if device+dtype match. [§E1 of todo.](./upstream_todo.md)
- **Setup gotchas worth listing in the published post**: `dynamo_config.use_fullgraph: false` + `use_regional_compilation: true` + per-rank `TRITON_CACHE_DIR=/tmp/triton-rank-${RANK}` + `@torch._dynamo.disable` on expert kernels. Without all four, you crash.

### 3.4 DS-Z2 + EP — the 7-patch recipe behind every long-context champion

(Elaborates Step 8.)

DS-Z3+EP is **fundamentally hard**: DS-Z3's "every rank holds the same logical data per param name" assumption is violated by transformers' EP partitioning (rank 0 holds experts 0-15, rank 8 holds experts 128-143 for the same param name; Z3's all-gather concatenates them into garbage). DS-Z2 sidesteps this — it doesn't all-gather params, only grads/optim.

The recipe took **7 patches across 3 repos** to land:

1. `tensor_parallel.py:GroupedGemmParallel.post_shard_wrap` — backend branch. Under DS, return plain tensor with `param.allreduce = False` + `param.group_name = f"ep_size_{ep_size}"` (DS's MoE convention). Under FSDP, return DTensor.
2. `trainer.py:create_accelerator_and_postprocess` — call `deepspeed.utils.groups._create_expert_and_data_parallel(ep_size)` before `Accelerator()`, so DS knows about MoE groups when `deepspeed.initialize` runs.
3. DS engine's `has_moe_layers` detection extended to recognize external EP via `param.allreduce == False`.
4. `trainer.py:create_optimizer` — split MoE params into separate optimizer group via `split_params_into_different_moe_groups_for_optimizer`.
5. `_clip_grad_norm has_ep` skip gated to FSDP-only (DS handles MoE grad norms via per-group path).
6. TRL EP branch — pre-init `deepspeed.comm` if SP is configured.
7. SonicMoE wrapper-level clamp (same as Step 5 patch 4 in the linear thread).

This unblocked **every long-context champion** in the post: the 32k/64k/128k DS-Z2+EP+chunked recipe. Queued upstream as a [transformers PR + DeepSpeed PR + TRL PR split](./upstream_todo.md#%EF%B8%8F-pr-after-deps-land), citing #45662 as the dependency.

> **Honest reporting (folded in here)**: FSDP2+EP+CP also works after the chunked-CE unlock, but is **~3× slower** than DS-Z2+EP+chunked at 64k. The slowness is FSDP2's per-chunk all-gather of `lm_head` under chunked-CE: ~519 NCCL collectives enqueued, only 8 complete → NCCL timeout. Workaround: route through DS-Z2. Proper FSDP-side fix paths listed in [§"FSDP+chunked NCCL hang" of upstream_todo](./upstream_todo.md). We don't recommend FSDP+EP+CP+chunked; we report it.

### 3.5 The accelerate patches we shipped, in one place

Three accelerate-side changes touch the headline numbers:

- **[accelerate#4022](https://github.com/huggingface/accelerate/pull/4022)** — torch.compile + FSDP2 hook preservation (in flight, validated by us). +6 pp at 16k FSDP.
- **`_prepare_tp` skip on `has_ep`** — local 5-line patch ([§4 of local_only_patches](./local_only_patches.md)), ready to upstream as a standalone PR. Required after #45662 turns EP params into DTensors.
- **`fsdp2_prepare_model` EP-aware broadcast + `cpu_ram_efficient_loading` proper fix** — ~10 lines, documented in [§C2 of upstream_todo](./upstream_todo.md). Comes with a `ValueError` safety-net guard in transformers `from_pretrained` so users don't silently get broken loading. PR queued.

---

## 4. The PR plan — what's merged, in flight, and queued

A single dense table that doubles as a roadmap. This is the section that proves "we're not just blogging — we're shipping."

### 4.1 Already merged

- [transformers#45436](https://github.com/huggingface/transformers/pull/45436) — Qwen3-MoE EP support + 2D-mesh fix.
- [transformers#45473](https://github.com/huggingface/transformers/pull/45473) — EP routing correctness + 4 regression tests.
- [transformers#45548](https://github.com/huggingface/transformers/pull/45548) — DS-Z3 + EP loading.
- [trl#5575](https://github.com/huggingface/trl/pull/5575) — chunked-CE in SFT.

### 4.2 In flight (review / pending merge)

- [transformers#45621](https://github.com/huggingface/transformers/pull/45621) — sonicMoE Ilyas patch (sentinel-aware kernel).
- [transformers#45649](https://github.com/huggingface/transformers/pull/45649) — FSDP2 cpu_ram_efficient_loading fix.
- [transformers#45662](https://github.com/huggingface/transformers/pull/45662) — EP + FSDP DTensor wrap.
- [accelerate#4022](https://github.com/huggingface/accelerate/pull/4022) — torch.compile + FSDP2 hook fix.

### 4.3 Queued, ready to file

(Pulled directly from [upstream_todo.md](./upstream_todo.md) §"🟢 Ready to PR".)

- **accelerate `_prepare_tp` `has_ep` skip** — single 5-line PR, references #45662.
- **EP + `cpu_ram_efficient_loading=True` (3 PRs)** — (a) transformers `ValueError` guard (immediate safety net), (b) transformers Patch 1 rank-0 gate in `convert_and_load_state_dict_in_model`, (c) accelerate Patch 2 `fsdp2_prepare_model` + `fsdp2_load_full_state_dict` EP-aware broadcast.
- **TRL `pad_to_multiple_of` auto-default for SP** — small TRL PR.
- **DS-Z2+EP recipe (3-repo split)** — transformers PR (4 patches), DeepSpeed PR (engine MoE detection), TRL PR (EP branch + MFU instrumentation + Triton cache dir).
- **Liger × MoE × EP fix** — Liger upstream PR with our [repro](./test_liger_qwen3_moe_ep.py); proposes `apply_liger_kernel_to_qwen3_moe` auto-detect EP and skip the swiglu patch.
- **TRL G2 split** — MFU instrumentation as its own PR (no MoE coupling, useful for everyone), legacy TF32 + per-rank Triton cache dir as its own PR (fixes a real PyTorch 2.10+ inductor crash + multi-node training bug).
- **Sonic-moe upstream issue at Dao-AILab** — kernel produces wrong upstream-flowing input grads on rows where all top-k slots are EP sentinels. Standalone repro at [`benchmark/test_sonic_repro_minimal.py`](./test_sonic_repro_minimal.py).

### 4.4 Investigation — smaller research follow-ups

- **PyTorch issue**: cross-mesh `_foreach_norm` stack failure under FSDP+EP. Minimal repro pending. Long-term fix unblocks `clip_grad_norm_` for mixed-mesh DTensors.
- **PyTorch issue**: FSDP+EP+compile Adam `_group_tensors_by_device_and_dtype` mesh-mix. Surgical patch sketched in [§E1](./upstream_todo.md#e1--fsdp--ep--compile-adam-_group_tensors_by_device_and_dtype-crash).
- **accelerate FA3 + CP integration** — current `cp_size > 1` requires sdpa; FA3 supports causal masking and seq sharding internally and likely composes. Worth a feature request.

### 4.5 Larger projects (research)

- **DS-Z3 + EP rank-ordering fix** — 1–2 days; build the `expert_data_parallel_group` with rank ordering matching transformers' `self.rank`-based EP partitioning. Would unlock DS-Z3+SP+EP for long-context MoE (compounding the 19% SP win with EP's expert sharding).
- **Streaming expert dispatch (kernel rewrite)** — RFC in transformers MoE integrations. Avoid materializing the `(seq, num_local_experts, moe_intermediate)` activation tensor. Biggest payoff: unblocks 32k+ EP without CP/chunked.

---

## 5. What we don't have yet — honest limitations

A short section. The post will be more credible if we **own the gaps** instead of papering them.

- **No PP** — Pipeline Parallelism is not in TRL/accelerate. For Qwen3-235B (470 GB bf16), CPU offload is the only option on 8 nodes and caps MFU at ~3% (32k CP=2). Megatron's reference uses TP=4, PP=16, EP=8 on 128 GPUs; PP is the gap. Open work.
- **EP correctness is a work in progress.** Five bugs in one month, four regression tests added. Five PRs upstreamed. We're confident the current state is correct for the configurations we benchmarked (verified by forward-output match against non-EP baseline at EP=1,2,4,8,16). We're not confident the matrix is exhaustive.
- **DS-Z3 + EP** — not working yet (rank-ordering issue, [debug log](./debug_sp_ep_sonic.md)). Workaround: DS-Z2 + EP for short-to-medium context, DS-Z3 + SP (no EP) for long context. The proper fix is real but unimplemented.
- **FSDP+EP+compile** — blocked at the Adam optimizer step. Workaround: use DS-Z2+EP+compile (works, 36.7% MFU at 16k).
- **MFU formula in TRL is currently non-causal.** We apply causal correction via a [post-hoc helper](./adjust_mfu_causal.py); a proper TRL PR to update `compute_flops_per_token` at source is queued. **Until then we report both columns.**
- **`235B` MFU is bad** (0.5–3%). Until PP lands, this is a hardware constraint. Be explicit.

---

## 6. The wrap-up — what we want readers to take away

Three clean takeaways. No fluff.

1. **Open MoE training works.** As of late April 2026, the open stack (TRL + transformers + accelerate + DeepSpeed) trains Qwen3-30B-A3B at **76% peak window MFU at 128k context** — and at 1M context with healthy loss. That's competitive with the best published numbers from closed-source training stacks at the same scale.
2. **All the fixes are upstreamable.** We've merged 4 PRs, have 4 in flight, and have ~10 more queued and documented. The whole picture is in [`upstream_todo.md`](./upstream_todo.md). If you want to reproduce: clone the repos at the commit we tag, follow the recipes table at the bottom of this post.
3. **Report MFU honestly.** We added the `mfu_window` metric (steady-state, not cumulative) and a causal-correction helper. Both columns are in every table we ship. **The 76% peak is 40.5% causal-corrected** — and 40.5% causal-corrected at 128k MoE is, to our knowledge, the highest published number on this stack with this hardware.

End with the call-to-action: try the recipes from Step 8 / Step 10 (the long-context champions), file issues on the repos for anything that breaks, and **stop telling people the open stack can't train MoE**. It can. We just did.

---

## Appendix A — Recipes table for reproduction

A copy-paste-friendly final table (not numbers — _commands_) so the post is actionable. One row per `(model, ctx, recipe)` triple, with the actual `accelerate launch` invocation.

```
# 16k Qwen3-30B-A3B — FSDP champion (40.4% window)
accelerate launch \
    --config-file benchmark/configs/qwen3_30b_a3b_ep8_fa3_redo.yaml \
    trl/scripts/sft.py \
    --model_name_or_path Qwen/Qwen3-30B-A3B \
    --enable_expert_parallel \
    --experts_implementation sonicmoe \
    --attn_implementation kernels-community/vllm-flash-attn3 \
    ...

# 128k Qwen3-30B-A3B — Liger × EP champion (76.29% peak window / 40.5% causal-adjusted)
... (similar)

# 1M Qwen3-30B-A3B — SP champion (62.33% peak / 31.4% causal-adjusted)
... (similar)
```

(Author note: pull these from `benchmark/configs/*.yaml` and `benchmark/templates/launch.sh.j2`. The configs are already in the repo; the post just needs to point at them.)

---

## Appendix B — Timeline diary (anchor reference for how the changes shipped)

> **Why this is an appendix, not a main section.** `benchmark/report.md` is append-only, so its sections are dated and ordered as we ran them. The diary captures the **research process**; the linear thread in §2 captures the **logic**. Both are valuable — they just shouldn't share the same slot in the main flow. A reader who wants to know "how did this actually unfold day by day, with the backtracks and dead ends?" finds the answer here. A reader who only wants the technical thread skips this.

> **Format below**: one entry per (rough) day of work. Each has a one-line header, the **WANT/TRY/BREAK/FIX** beat (when applicable), and the **measured delta** at the end.

### 4.1 Phase 1 — Baseline (pre-04, transformers 4.57.6): **2.8% MFU**

- Fresh sweeps on Qwen3-4B (dense) and Qwen3-30B-A3B (MoE), default stack.
- 4B works fine — 30% MFU at 16k. Dense baseline established as the sanity reference.
- 30B explodes: `RuntimeError: Detected mismatch between collectives on ranks` in FSDP2's `reduce_scatter`. The `nn.ModuleList[nn.Linear]` expert layout is fundamentally incompatible with FSDP2's symmetric collective model.
- **WORKAROUND**: TRL-side `fuse_moe_experts()` helper + a forked checkpoint ([`aminediroHF/Qwen3-30B-A3B-fused`](https://huggingface.co/aminediroHF/Qwen3-30B-A3B-fused)) that pre-stacks the 128 experts. Training works.
- **MEASURE**: 2.8% MFU at 16k. Re-ran DS-Z3 baseline for comparison: 1–2% MFU. We're 2× DS-Z3 but still in the basement.
- **MOOD**: this is the baseline the dunk threads were citing. It was a fair criticism.

### 4.2 2026-04-11 — Complete-results table day

- Ran the full sweep (all combinations of `(model, ctx, DP, TP, CP, attn)`) and shipped the first version of `report.md`'s "Complete Results Table" so we'd have an honest baseline before anything changed.
- TP+MoE: **fails on every config** with a tokenizer-loading error on non-rank-0 (`AutoProcessor.from_pretrained` on TP sub-ranks). Filed as a separate todo; not in the critical path.
- This is the day we picked **SFTTrainer × Qwen3-30B-A3B × long-context** as the focus and started planning EP work.

### 4.3 2026-04-13 (ish) — Transformers 5.6.0 lands → **8× jump** (2.8% → 23%)

- The transformers core team shipped native `Qwen3MoeExperts` with fused `gate_up_proj` + automatic `WeightConverter` from the original Hub checkpoint.
- Dropped `fuse_moe_experts` helper and the forked checkpoint. The original `Qwen/Qwen3-30B-A3B` works directly.
- **MEASURE**: 23% MFU at 16k — **8× over the 2.8% baseline**. One PR bump. Not our PR; we were standing on the core team's work.
- Important framing: this is the *one* big jump that wasn't ours. We acknowledge it.

### 4.4 2026-04-15 — The EP correctness audit: PRs #45436 + #45473 land **(EP-series PRs 1+2 of 5)**

- WANT: enable expert parallelism so each rank only stores `num_experts/EP` slices.
- TRY: `enable_expert_parallel=True`. Ran a forward pass at EP=1 and compared logits to the non-EP baseline.
- BREAK: **all-zero expert outputs** with no NaN. EP routing was silently broken.
- DEBUG: read through `RouterParallel` + `GroupedGemmParallel`. Found **five** independent bugs:
  1. Wrong shape for routing weights under EP remapping.
  2. Wrong plan looked up during weight loading (TP regex over EP plan).
  3. Sentinel rows hit uninit GPU memory in `grouped_mm`.
  4. Global rank vs mesh-local rank in `shard_tensor` (out-of-range expert ids on 2D mesh).
  5. `num_experts` divided once per weight instead of once per module.
- FIX: split into **[transformers#45436](https://github.com/huggingface/transformers/pull/45436)** (Qwen3 EP plan + 2D-mesh fixes) and **[transformers#45473](https://github.com/huggingface/transformers/pull/45473)** (routing correctness + 4 regression tests verifying forward-output match against non-EP at EP=1,2,4,8,16). Both **merged**.
- MEASURE: EP=4 with TP=4 on 16k = 22.7% MFU, training-correct. EP itself doesn't beat no-EP on throughput yet, but **now it's verifiable.** Zero EP test coverage existed before; we left 4 tests behind.
- MOOD: we discovered that **all prior "EP" wandb runs were wrong** (loss curves looked plausible, but logits diverged from the no-EP ground truth in bf16). This is the section we're least proud of and most explicit about in the post.
- **THREAD MARKER**: this is the start of a 5-PR EP-correctness series spanning two weeks. PRs 1+2 (#45436, #45473) land today. PRs 3+4+5 (#45548 DS-Z3 loading, #45649 FSDP cpu_ram OOM, #45662 EP+FSDP DTensor) land 04-24 → 04-27. See the **EP-series thread table** later in this appendix and the **§2 sidebar** for the consolidated view.

### 4.5 2026-04-17 — FA3 sweep day (Hopper-native attention everywhere)

- WANT: replace sdpa with FA3. Hopper has tensor cores tuned for FA3's GEMM shapes.
- TRY: `attn_implementation=kernels-community/vllm-flash-attn3`.
- MEASURE on dense:
  - Qwen3-4B at 32k: 35.9% (sdpa+Liger) → **56.3%** (FA3+Liger). +57%.
  - Qwen3-32B at 32k: 43.1% (sdpa+Liger) → **59.0%** (FA3+Liger). +37%.
- MEASURE on MoE: +11% at 16k FSDP DP=16 (23.1 → 25.7%), +6% at 16k EP=8.
- BREAK: FA3 + CP fails — accelerate hard-guards `cp_size > 1` to `attn_implementation=sdpa`. Filed as a todo (cf. §H2 of `upstream_todo.md`); FA3 is our long-context attention, but only via SP, not CP.

### 4.6 2026-04-24 — SonicMoE first attempt (and a hidden FSDP regression) **(EP-series PR 3 of 5)**

- WANT: a faster expert-dispatch kernel than `grouped_mm`.
- TRY: `kernels-community/sonic-moe` (CuteDSL fused MoE) via `--experts_implementation sonicmoe`.
- BREAK 1 (orthogonal, but turns out to be on the EP critical path): FSDP2 OOMs during `from_pretrained` at `Qwen3-30B`-scale on multi-node EP runs.
  - DEBUG: bisected transformers `5.6.0.dev0` → `5.7.0.dev0` to a single-line change in [PR #45050](https://github.com/huggingface/transformers/pull/45050) — `torch.empty_like` → `torch.zeros_like` on non-rank-0 FSDP placeholder tensors. Benign on small models, but on Linux with anonymous mmap it forces a *physical memory commit* of every byte. For 8 ranks × 30B weights, that's ~480 GB CPU peak per node → cgroup OOM.
  - FIX: pushed [`AmineDiro/transformers:fix-fsdp2-cpu-ram-zeros-like`](https://github.com/AmineDiro/transformers/tree/fix-fsdp2-cpu-ram-zeros-like) → became **[transformers#45649](https://github.com/huggingface/transformers/pull/45649)** (in flight) — drop the broadcast-bound parameter materialization on non-rank-0 ranks (FSDP overwrites them anyway). Buffers (per-rank) still get zeros.
  - **THREAD MARKER**: this is **EP-series PR 3 of 5**. It's framed as "an FSDP fix" but in practice every multi-node EP run depended on it — without #45649, you can't even *load* the model on 16+ ranks, let alone train it. Counts as part of the EP-correctness pipeline.
- BREAK 2 (the actual sonicMoE issue): kernel cold-start (~25–30s for CuteDSL JIT + autotune + first-touch) made 20-step runs lie about steady-state.
- MEASURE (early, 20-step): cumulative MFU was confusingly close to grouped_mm. We almost concluded sonicMoE wasn't a win — the cold start dominated the average.

### 4.7 2026-04-25 — Window MFU + sonicMoE redux: **+23%**

- FIX: added `mfu_window` (Δtokens / Δtime over the last logging window) and `tps_window` to `SFTTrainer.log`. Cumulative becomes the historical metric; window becomes the steady-state metric.
- RE-RAN the sonicMoE sweep at 50 steps, both metrics.
- MEASURE: 16k FSDP DP=16, sonicMoE+FA3 = **34.7% window MFU** vs grouped_mm+FA3 = 28.1% window. **+23.4%** at steady state. The win was real; the 20-step cumulative was misleading.
- LESSON we want to land in the post: **report window MFU for any kernel with non-trivial first-call cost** (Triton, CuteDSL, inductor-compiled). Cumulative under-reports steady-state by 5–10 pp for 50+ steps.

### 4.8 2026-04-26 — Phase-2 starts: Ilyas's sonic-moe + EP scaling sweep **(EP-series Phase-2 begins)**

- **THREAD SHIFT**: today is the day **make-EP-fast** work begins. Phase 1 (correctness) still has two more PRs to land (45548, 45662, both 04-27). Phase 2 starts here, in parallel.
- WANT: sonicMoE that survives EP > 1 (kernel CUDA-illegal-accesses on rows whose `top_k` ids are all EP sentinels).
- FIX: **[transformers#45621](https://github.com/huggingface/transformers/pull/45621)** pins the kernel revision to `IlyasMoutawwakil/sonic-moe@main` — **authored by Ilyas Moutawwakil, not by us**. Ilyas added kernel-native sentinel handling in the metadata stage. Forward path is now sentinel-aware.
- Ran patched-sonicMoE × EP=2/4/8 sweep at 16k. Kernel works. Window MFU plateaus at 32–34% for EP ≥ 4.
- BUT: every EP run produces broken initial gradients (loss step-1 ranges 9–62, scales with EP degree; expected ~2). NaN by step 10. The kernel is fine — there's an **EP weight-loading bug downstream**, plus a separate **kernel-backward NaN on EP sentinels** that the wrapper-clamp will eventually paper over (Phase 2 ongoing).
- MEASURE: throughput peaks at 33.54% window. Training-correctness still gated by Phase-1 PRs that haven't landed yet.

### 4.9 2026-04-27 — The big day: Phase-1 EP correctness lands in full **(EP-series PRs 4+5 of 5; Phase 2 first workaround)**

This is the longest entry in the diary. Phase-1 of the EP-PR series **finishes today** with the last two PRs (#45548 and #45662); Phase-2's first workaround (the wrapper clamp) also lands today. Seven things happened in parallel; the post should highlight this with a "what one busy day looks like in framework debugging" framing.

1. **EP-series PR 4: DS-Z3 + EP loading** — **[transformers#45548](https://github.com/huggingface/transformers/pull/45548)** (merged). Before: `from_pretrained` hung at `device_map` checks because DS-Z3's env vars routed every weight through the ZeRO-3 path; EP needs the standard path. Adds `PreTrainedModel.has_ep` and routes EP+DS through the standard loader. **Result**: DS-Z3 + EP now loads (though it'll later run into the rank-ordering issue and force the DS-Z2 pivot).
2. **FA3 + EP "incompatibility" was self-inflicted** — bisected to `HF_HUB_OFFLINE=1` being set after sonicMoE pre-warmed but before FA3's two-phase kernel load could fetch. Fixed in `trl/scripts/sft.py` by pre-warming both FA3 paths first. **MEASURE**: FA3+EP=8+sonicMoE = 42.66% window MFU peak (the highest pre-correctness-fix number on the stack).
3. **EP-series PR 5: EP+FSDP DTensor wrap** — **[transformers#45662](https://github.com/huggingface/transformers/pull/45662)** (in flight). EP-sharded experts wrapped as DTensors on the EP mesh so Adam's `_fused_adamw_` doesn't reject mixed Tensor + DTensor lists. **This is the PR that closes Phase 1.**
4. **Side effect of #45662** (the cascading consequence the post should highlight): now that EP params are DTensors, accelerate's existing `_prepare_tp` "no DTensor → skip" guard stops firing → ImportError on `ReplicateParallel`. Patched locally with a 5-line `has_ep` skip. Queued as a standalone accelerate PR. *Every Phase-1 PR has had a downstream consequence that needed a follow-up patch — this is normal in framework debugging and worth being explicit about.*
5. **Phase-2 starts on the trainer side: NaN bisect → wrapper-clamp workaround** — Bisected the EP NaN gradient through 4 hypotheses (DTensor-wrap? `to_local()` non-contiguous? sentinel handling?). Landed on: kernel's hand-written backward NaNs on EP sentinels going through `DTensor.to_local()`. Wrapper-level `clamp + masked_fill` on sentinels in `sonicmoe_experts_forward`. **~2 pp MFU cost — the "clamp tax" we'll eventually claw back when Dao-AILab fixes the kernel backward.** Filed the upstream issue with a [standalone repro](./sonic_moe_upstream_repro.md).
6. **Long-context EP=8 sweep** — every config OOMed at the **same single 18.55 GiB allocation** (the EP-replicated expert buffer). 32k, 64k, 8 nodes — all hit the same wall. (This is the wall chunked-CE will demolish on 04-29.)
7. **DS-Z3 + EP → dead end → pivot to FSDP+EP+CP** — DS-Z3's broadcast inside `expert_data_parallel_group` overwrites EP slices because rank ordering doesn't match. Documented the rank-ordering analysis (`debug_sp_ep_sonic.md` Iteration 4) and pivoted long-context to FSDP+EP+CP. **Phase-1 #45548 unblocked DS-Z3 *loading*; the rank-ordering issue blocks DS-Z3 *training*. Different problem, separate fix path, deferred.**

**MEASURE end-of-day**: 16k post-correctness-fix EP=8 + FA3 + sonicMoE = **40.4% window MFU**, training-correct. The EP champion that stuck. **Phase 1 of the EP-PR series is complete.** Phase 2 is in progress (clamp tax sitting at ~2 pp).

### 4.10 2026-04-28 — torch.compile fix + chunked-CE first results

1. **torch.compile + FSDP2 fix lands**: **[accelerate#4022](https://github.com/huggingface/accelerate/pull/4022)** (in flight) — `accelerate.fsdp2_prepare_model` was calling `torch.compile` and breaking FSDP2's forward hooks (2.7× *slowdown*). After the fix: FSDP DP=16 + FA3 + compile = **34.87% window MFU**. +6 pp over no-compile baseline.
2. **DS-Z2 + EP + compile = new compile-EP champion** at 16k — **36.7% window MFU**. +8 pp over DS-Z2+EP no-compile. (FSDP+EP+compile blocked: Adam's `_group_tensors_by_device_and_dtype` mesh-mix; documented.)
3. **32k SP=2 unseats CP** as the long-context champion — DS-Z3+SP=2+FA3+sonicMoE @ 32k = 21.98% window MFU. Beats FSDP+EP+CP=2 at 32k (15.6%) by +6 pp.
4. **256k SP=16 first attempt** — DS-Z3+SP=16+chunked at 256k @ 8n = 1.36% MFU. Cross-node Ulysses dominates. Lesson: SP doesn't scale arbitrarily; need per-rank-seq tuning.
5. **Chunked-CE first batch**: cherry-picked **[trl#5575](https://github.com/huggingface/trl/pull/5575)** (since merged). 32k DS-Z2+EP=8+FA3+chunked = **45.86% window MFU peak**. **NEW 32k CHAMPION (+24 pp).** The 18.55 GiB EP buffer that ate every config the day before now fits because chunked-CE freed ~20 GB from the lm_head logits.

### 4.11 2026-04-29 — The tear

**Multiple breakthroughs in one day. This is the diary entry that earns the post.**

1. **64k chunked-CE (DS-Z2+EP=8+FA3+chunked, 2n)** — **57.23% window MFU peak**. **NEW 64k CHAMPION (+37 pp).**
2. **128k chunked-CE (4n)** — **69.10% peak / 64.10% cumulative**. **NEW 128k CHAMPION (+50 pp).** Highest MoE MFU on stack (pre-Liger).
3. **Compile no help at long context** — at 64k/128k the path is communication-bound (not compute-bound), compile gives 0 pp.
4. **256k breakthrough on the SP path** — DS-Z3+SP=2+FA3+chunked @ 8n = 59.61% peak (was 1.36% on 04-28). **+58 pp.** Per-rank seq = 128k.
5. **Per-rank-seq sweet-spot rule discovered** — sweeping SP × ctx, MFU peaks at `total_ctx / SP = 128k`. This wasn't from theory; we found it from the sweep matrix.
6. **Compile stabilizes the SP path** — without compile, window MFU at 256k oscillates 5–47%; with compile, steady 40–60%. Compile becomes mandatory at 256k+.
7. **512k unlocked** — DS-Z3+SP=4+chunked+compile @ 8n = 58.24% peak. (Previous attempts at 512k @ 16n hung on Z3 cross-node all-gather watchdog with 128 ranks; smaller mesh + per-rank tuning works.)
8. **1M context unlocked** — DS-Z3+SP=8+chunked+compile @ 8n = **37.46% peak / 35.65% cumulative**. **New frontier.** First time this stack trains a 30B MoE at 1M ctx.
9. **Liger × MoE × EP root cause** — disproved the "Liger doesn't do 3D weights" wisdom in 30 minutes with single-GPU repro. Real bug: `_patch_swiglu_module` bypasses transformers' EP dispatcher, then `F.one_hot` collides with the EP sentinel value. Workaround: `--liger_kernel_config '{"swiglu":false}'`.
10. **Liger sweep dominates chunked-CE at every context** (16k–1M, +0.3 to +25 pp peak):
    - 32k: 56.62% (vs 45.86% chunked)
    - 64k: 66.46% (vs 57.23%)
    - **128k: 76.29% peak / 74.69% cumulative** — **all-time MoE MFU record on this stack.**
    - 256k: 63.62% / 512k: 63.26% / 1M: 62.33%
11. **Causal-corrected MFU helper** — `benchmark/adjust_mfu_causal.py`. Applies the convention switch post-hoc (Llama 2/3 / DS-Ulysses convention vs PaLM/Megatron). Headline 76.29% → **40.5% causal-adjusted at 128k**; 62.33% at 1M → **31.4% causal-adjusted**. Both columns in every headline table from this point on.
12. **Single-node 1n sweep** — DS-Z2+EP=8+Liger @ 1n = 59.28% MFU at 32k (single node beats 2n by +3 pp due to intra-node EP comm). Important for accessibility framing — you don't need a multi-node cluster for 30B MoE up to 32k.

### 4.12 2026-04-30+ — Cleanup, write-up, this post

- Removed the `SONICMOE_DISABLE_CLAMP` and `TRANSFORMERS_SKIP_EP_DTENSOR_WRAP` debug knobs (one-shot bisect tooling).
- Reverted DS engine.py local edits in favor of the proper `param.allreduce=False` + `group_name` tagging path documented in [§D-works of upstream_todo](./upstream_todo.md).
- Drafting upstream PRs for: accelerate `_prepare_tp` skip, `cpu_ram_efficient_loading` EP coexistence (3 PRs), Liger × MoE × EP fix, sonic-moe upstream issue, TRL G2 split (MFU instrumentation + Triton cache + EP branch).
- Writing this post.

### 4.13 What the diary tells you that the headline tables don't

- **Most of the gain came from one busy week** (2026-04-25 → 2026-04-29). Five working days produced the 16k EP champion (40.4%), the 32k/64k/128k chunked-CE champions (45/57/69%), the 256k/512k/1M SP+Liger champions (60+%), and the 76% all-time record.
- **Almost every fix unblocked the next fix.** EP correctness (#45473) → EP+FSDP DTensor (#45662) → sonicMoE+EP backward NaN → wrapper-clamp → DS-Z3+EP loading (#45548) → DS-Z3+EP rank-ordering dead end → DS-Z2+EP recipe → chunked-CE → long-context champions. Pull any node out and the chain breaks.
- **Pivots are part of the work, not failures.** DS-Z3+EP was pursued for a full day before pivoting to DS-Z2. The work isn't wasted — the rank-ordering analysis is the path forward for the eventual DS-Z3+EP unblock.
- **The order matters because dependencies matter.** A reader who skips the diary and goes straight to the recipes will think "just use DS-Z2+EP+chunked+Liger." That recipe **could not have been validated** before EP was correctness-audited (#45473), before sonicMoE got sentinel-handling (#45621), before FSDP2 stopped OOMing during loading (#45649), and so on. Five PR dependencies for one recipe.

### 4.14b The EP-series thread — the multi-week PR thread that runs through every diary entry

**A standalone callout/sidebar in the published post.** A reader who only reads this box should walk away understanding: **EP wasn't a single fix; it was a five-PR series spanning two weeks**, plus a second-phase kernel-side recovery still in progress.

| # | Date landed | PR | Phase | Author | What it unblocked |
| --- | --- | --- | --- | --- | --- |
| 1 | 2026-04-15 | [transformers#45436](https://github.com/huggingface/transformers/pull/45436) | 1 (correct) | us | Qwen3 EP plan; mesh-local rank in `shard_tensor`; single `num_experts` divide |
| 2 | 2026-04-15 | [transformers#45473](https://github.com/huggingface/transformers/pull/45473) | 1 (correct) | us | Routing-weight shape; weight-loading plan lookup; sentinel-aware grouped_mm; +4 regression tests |
| 3 | 2026-04-24 | [transformers#45649](https://github.com/huggingface/transformers/pull/45649) (in flight) | 1 (correct) | us | FSDP2 cpu_ram_efficient OOM removed (#45050 regression); required to load 30B-A3B on 16+ ranks |
| 4 | 2026-04-27 | [transformers#45548](https://github.com/huggingface/transformers/pull/45548) | 1 (correct) | us | DS-Z3 + EP loading via `PreTrainedModel.has_ep` |
| 5 | 2026-04-27 | [transformers#45662](https://github.com/huggingface/transformers/pull/45662) (in flight) | 1 (correct) | us | EP+FSDP composability via DTensor-wrap on EP mesh — closes Phase 1 |
| — | 2026-04-26 | [transformers#45621](https://github.com/huggingface/transformers/pull/45621) (in flight) | **2 (fast)** | **Ilyas Moutawwakil** | sonicMoE forward sentinel-aware (kernel revision pin) |
| — | 2026-04-27 | local — `sonicmoe.py` wrapper clamp ([§1 of local_only_patches](./local_only_patches.md)) | 2 (fast, workaround) | us | Backward NaN on EP sentinels through `DTensor.to_local()`; ~2 pp MFU clamp tax |
| — | pending | Dao-AILab/sonic-moe upstream issue ([repro](./sonic_moe_upstream_repro.md)) | 2 (fast, root fix) | us (issue), Dao-AILab (fix) | Kernel-native backward sentinel handling → **+2 pp recovery when removed** |
| — | pending | accelerate `_prepare_tp` `has_ep` skip ([§4 of local_only_patches](./local_only_patches.md)) | 1 (downstream consequence of #45662) | us | Stops `_prepare_tp` from ImportError-ing on EP DTensor params |

**The post's framing of this thread**:

- **EP wasn't broken in one place — it was broken at every layer of the stack.** TRL silently bypassed it; transformers had five forward-correctness bugs, a loading hang under DS-Z3, an OOM under FSDP cpu_ram, and an Adam mesh-mix under FSDP composition; accelerate's TP prep tripped on the resulting DTensor wrap; the fastest available kernel (sonic-moe) had no sentinel awareness. **Each layer needed its own fix.**
- **Phase 1 (correctness)** is **done**. Five PRs, all merged or in flight. Three in-flight PRs are awaiting review, not awaiting more code from us.
- **Phase 2 (performance)** is **in progress**. Ilyas's #45621 fixed forward; we filed the backward issue at Dao-AILab with a standalone repro; the wrapper clamp is the bridge until the kernel-native backward lands. The clamp tax is ~2 pp MFU. The 76.29% headline includes it; without it we'd be at ~78%.
- **The post should not pretend Phase 2 is done.** The honest framing: "EP went from broken at every layer to correct everywhere and within ~2 pp of optimal — and we have a clear path to closing the gap."

### 4.14 Reference table (the dry version, for skimmers)

For the reader who wants the table-of-PRs to scan in 30 seconds:

| Date | What unlocked | Where the fix lives | Status |
| ---- | ------------- | ------------------- | ------ |
| pre-04 | Fused experts as workaround | TRL `fuse_moe_experts` helper | Retired |
| 2026-04-13 (ish) | Native `Qwen3MoeExperts` | transformers core | Upstream (not us) |
| 2026-04-15 | EP correctness + 2D mesh | [transformers#45436](https://github.com/huggingface/transformers/pull/45436) [#45473](https://github.com/huggingface/transformers/pull/45473) | **Merged** |
| 2026-04-17 | FA3 unlock everywhere | `kernels-community/vllm-flash-attn3` wiring | Configured |
| 2026-04-24 | FSDP2 cpu_ram_efficient OOM | [transformers#45649](https://github.com/huggingface/transformers/pull/45649) | **In flight** |
| 2026-04-24 | sonicMoE first benchmark | `--experts_implementation sonicmoe` flag in TRL | Local (G2) |
| 2026-04-25 | Window MFU metric | `SFTTrainer.log` adds `mfu_window` | Local (G2) |
| 2026-04-26 | sonicMoE EP sentinel-skip | [transformers#45621](https://github.com/huggingface/transformers/pull/45621) | **In flight** |
| 2026-04-27 | DS-Z3 + EP loading | [transformers#45548](https://github.com/huggingface/transformers/pull/45548) | **Merged** |
| 2026-04-27 | EP+FSDP DTensor wrap | [transformers#45662](https://github.com/huggingface/transformers/pull/45662) | **In flight** |
| 2026-04-27 | sonicMoE wrapper-clamp + accelerate `_prepare_tp` skip | local; queued upstream | Local |
| 2026-04-27 | FA3 + EP self-inflicted offline-mode race | `trl/scripts/sft.py` pre-warm dance | Local (G2) |
| 2026-04-28 | torch.compile + FSDP2 broken hooks | [accelerate#4022](https://github.com/huggingface/accelerate/pull/4022) | **In flight** |
| 2026-04-28 | DS-Z2 + EP recipe (3 repos, 7 patches) | local — see [upstream_todo.md §D-works](./upstream_todo.md) | Pending PR split |
| 2026-04-28 | chunked-CE in SFT (cherry-picked) | [trl#5575](https://github.com/huggingface/trl/pull/5575) | **Merged** |
| 2026-04-29 | Liger × MoE × EP root cause + workaround | local + Liger PR upcoming | Workaround shipping |
| 2026-04-29 | Per-rank-seq sweet spot + 1M context | benchmark methodology | Documented |
| 2026-04-29 | Causal-corrected MFU | `benchmark/adjust_mfu_causal.py` helper | Local |

---

## Appendix C — Methodology notes for skeptics

- **Hardware**: AWS p5.48xlarge, H100 SXM5 80GB, 8 GPUs/node, NVLink intra-node, EFA inter-node.
- **MFU formula** (full):
    ```
    MFU = 100 × (flops_per_token × TPS / cp_size) / (peak_flops × world_size)
    flops_per_token = active_attn_flops + active_moe_flops × 3  (×3 for fwd+bwd)
    ```
- **Causal correction**:
    ```
    adj_factor = 1 - (L × 3 × 2 × n_heads × head_dim × seq_len) / full_flops
    adjusted_MFU = raw_MFU × adj_factor
    ```
    (Halve attention-score FLOPs; matches Llama-2/3 / DeepSpeed-Ulysses convention.)
- **Window MFU**: `Δtokens / Δtime` over the last logging window. Steady-state metric; we recommend reporting it for any kernel with non-trivial first-call cost (Triton, CuteDSL, inductor-compiled). Cumulative under-reports steady-state by 5–10 pp for the first 50 steps.
- **Data**: `THUDM/LongAlign-10k` dataset, packed via `--packing --packing_strategy wrapped`, batch size 1 per device, 20-step measurements (50 for kernels with cold-start), Adam fp32, bf16 forward, gradient checkpointing on for all runs.
- **Trackio dashboards**: every run is logged; peak GPU memory pulled from `gpu/<rank>/allocated_memory` via [`fetch_peak_gpu_mem.py`](./fetch_peak_gpu_mem.py). Full results browsable at [aminedirohf-qwen3-sft-benchmark.static.hf.space](https://aminedirohf-qwen3-sft-benchmark.static.hf.space/index.html).

---

## Author notes (cut these from the published version)

**Tone**: snarky but not bitter. The dunk-thread crowd is correct that the open stack _was_ slow for MoE. We don't want to deny that — we want to show what it took to fix. The "they were right, then we fixed it" framing reads better than "they were always wrong".

**Length target**: 4000–6000 words. The §2 decision-tree thread is the main draw; §3 (technical deep-dives) is the technical credibility; §4 (PR plan) is the receipts. §1 (Setup), §5 (Limitations), §6 (Wrap) are short transitions. Appendix B (Timeline diary) is for readers who want the chronological process.

**Audience**: ML engineers / researchers familiar with FSDP/DS but maybe not with EP internals. Don't assume they know `RouterParallel` or DTensor mesh details — explain on first use. Do assume they know what FSDP2 / Z3 / SP / CP mean.

**Asks for the user before publishing**:

1. Choose the title (3 options at the top).
2. Confirm whether to name people in the EP credits (Ferdinand Mom for the upstream EP plan review; Ilyas for the patched sonic-moe kernel; etc.).
3. Decide whether to include the `aminediroHF/Qwen3-30B-A3B-fused` checkpoint as a historical artifact link or remove (it's no longer needed post-5.6.0).
4. Confirm the cluster name / hosting credit (AWS p5.48xlarge — is there a partner / sponsor to credit?).
5. Sign-off on the rebuttal tone in §0 (lede). Possible to soften if the workshop-labs team is a partner / friend.

**Pre-publish checks**:

- [ ] Re-run the 16k EP=8 + FA3 + sonicMoE config one more time and confirm the 40.4% number is reproducible on a freshly-pulled commit.
- [ ] Re-run the 128k Liger champion (76.29%) and confirm.
- [ ] Re-run the 1M champion and confirm.
- [ ] Verify all PR links land (no 404s).
- [ ] Have someone on the transformers core team review §3.1–3.2 (EP correctness audit + EP-buffer wall) for technical accuracy on EP internals.
- [ ] Have someone on the accelerate/DS team review §3.4 (the DS-Z2+EP recipe write-up) for accuracy on the 7-patch flow.
