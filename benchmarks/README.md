# Tree GDN benchmarks

## Review this branch

This branch ports only the GDN work onto `tree-attention`: the kernel, Qwen3.5 adapter, packing integration,
tests, diagnostics, and saved benchmark results. It excludes the unrelated MiMo examples and sampling changes.

Start with [the single-file kernel](../trl/kernels/_tree_gdn_fla/tree.py),
[its review guide and source provenance](../trl/kernels/_tree_gdn_fla/README.md),
[the public plan and convolution](../trl/kernels/tree_gated_delta_rule.py), and
[the Qwen3.5 adapter](../trl/experimental/async_grpo/tree/gdn.py).

**This is experimental review code, not a training-correctness sign-off.** Ordinary-packing gate-gradient and
model/FSDP discrepancies remain unresolved; details and the failing results are retained below. The kernel has no
FLA runtime dependency; FLA is needed only for reference tests and benchmarks. The paper-index entry and upstream
MIT license are included.

The saved timings predate this branch port. They measure forward-only and combined forward + backward, not
backward-only or complete trainer steps. No new GPU speedup claim is made by the port.

## Single-file cleanup regression (2026-10-07)

The vendored implementation is now entirely in `trl/kernels/_tree_gdn_fla/tree.py`: eight implementation files became
one, with 768 Python lines removed (about 31%). Only unused branches/options were removed; the active matrix-operation
order, casts, state propagation, and tuning candidates were preserved. This is not a precision fix.

Compared against pre-cleanup revision `a1ae2ef4976df67234ca401daae0c4b947ea260f` on H100 / `hopper-dev`:

- **36/36 exact kernel regressions passed**: outputs and Q/K/V/gate/beta gradients are bitwise equal with matching
  pinned launch configurations. Chains, shared prefixes, nested forks, multiple roots, FP32/BF16, weak/strong decay,
  dimensions 32/64/96/128, grouped/equal heads, normalization on/off, and partial chunks are covered.
- **One-rank and two-rank FSDP2 + checkpointing passed both SGD steps**: losses, every parameter gradient, and
  updated parameters match the old tree implementation exactly. Two-rank layouts differ by rank and step.
- **Ordinary-FLA suite: 49 passed, 1 failed.** The existing FP32, dimension-128, strong-decay nested-tree gate-gradient
  discrepancy is unchanged: `0.0053399233147501945` relative L2 (0.533992%) versus the 0.5% tolerance. No tolerance was
  loosened. All 16K/32K/64K parity checks passed, as did the repaired no-FLA subprocess test.
- **18 long-sequence before/after benchmarks completed**: 16K/32K/64K, four rollouts, prompt/fork layouts, packing
  ratios 1.00/2.29/2.67/3.37/3.56. Outputs and all gradients were also bitwise equal with independent autotuning.
  Forward + backward speed ratios were **0.9988–1.0097×** (before/after): effectively unchanged performance, not a
  new packing-speedup claim. Median of ten CUDA-event measurements after three warmups, BF16, 16 key / 32 value
  heads, dimension 128, Triton 3.7.1. See [the cleanup chart](tree-gdn-cleanup-long.png) and raw
  `tree-gdn-cleanup-long.json`; an SVG is saved alongside the PNG.

Artifacts: `tree-gdn-cleanup-parity.json`, `tree-gdn-cleanup-kernels.xml`,
`tree-gdn-cleanup-fsdp-single-rank0.json`, and `tree-gdn-cleanup-fsdp-two-rank{0,1}.json`.
The FSDP comparison checks **before versus after cleanup**, not tree versus ordinary packing; it does not resolve
the ordinary-packing discrepancies documented below. Separate one-GPU Slurm allocations required
`NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_NVLS_ENABLE=0 NCCL_IB_DISABLE=1`; the first attempt without these settings
failed in NCCL's first all-gather with `invalid device ordinal`, before executing a GDN kernel.

The historical cleanup comparison requires the pre-cleanup Git object
`a1ae2ef4976df67234ca401daae0c4b947ea260f`. It is retained in the original worktree's local backup, but is not an
ancestor of this standalone branch and is not included in a fresh clone of it. These commands require that object;
the ordinary-baseline tests and benchmarks below do not.

Reproduce from the repository root, using a suitable GPU allocation after obtaining the pre-cleanup revision:

```bash
PYTHONPATH=. python benchmarks/compare_tree_gdn_cleanup.py \
  --before a1ae2ef4976df67234ca401daae0c4b947ea260f --output benchmarks/tree-gdn-cleanup-parity.json
PYTHONPATH=. torchrun --standalone --nproc-per-node=2 benchmarks/compare_tree_gdn_cleanup_fsdp.py \
  --before a1ae2ef4976df67234ca401daae0c4b947ea260f --output benchmarks/tree-gdn-cleanup-fsdp-two.json
PYTHONPATH=. python benchmarks/compare_tree_gdn_cleanup.py --benchmark \
  --before a1ae2ef4976df67234ca401daae0c4b947ea260f --output benchmarks/tree-gdn-cleanup-long.json
python benchmarks/plot_tree_gdn_cleanup.py benchmarks/tree-gdn-cleanup-long.json
```

## Which result is which?

- `tree-gdn-h100-long.json`: **old FP32 prototype**, not the vendored implementation. Retained for comparison.
- `tree-gdn-h100-vendored.json`: the **vendored + tree-modified FLA kernel**, ordinary independent-sequence FLA,
  and the compiled PyTorch reference. This measures the GDN core, not the complete Qwen layer.
- `tree-gdn-h100-vendored-forks.json`: core-only nested forks, with 2/4/8 rollouts and 75% shared prompt length.
- `tree-gdn-h100-qwen-layer.json`: the **complete Qwen3.5 GDN layer**: input projections, convolution, GDN core,
  gated normalization, output projection, and their gradients. It compares ordinary Qwen's default implementation,
  Qwen's built-in PyTorch GDN-core fallback, and the vendored tree layer. This is not a full-model training benchmark.
- `tree-gdn-h100-qwen-fastconv.json`: the **primary optimized Qwen comparison**, with ordinary Qwen using both
  FLA and causal-conv1d 1.7.0. Same layer dimensions and prompt/fork layouts; Transformers 5.8.1.

PNG and SVG charts are stored next to each JSON. The title says `CORE` or `LAYER`; do not confuse them.
The earlier `qwen-layer` run uses ordinary `torch.nn.Conv1d` because causal-conv1d was not installed then. The fallback switches
only the GDN core; other layer operations and weights are identical. The precise default core and convolution are
recorded per case. The original prototype used Triton 3.6; vendored results use 3.7.1 and `FLA_TILELANG=0` for both
FLA and the tree path. Warmup/autotuning is excluded from latency and reported separately.

## Measured forward + backward speedups

One H100 80GB on `hopper-dev`, BF16, four rollouts, median of five timed iterations after three warmups.
Sequence length below is **per rollout**, not total duplicated tokens.

### Primary baseline: normal Qwen with FLA + causal-conv1d

| Layout                      | Packing ratio | 16K   | 32K   | 64K   |
| --------------------------- | ------------- | ----- | ----- | ----- |
| No sharing (prompt control) | 1.00×         | 0.77× | 0.78× | 0.79× |
| Shared prompt               | 1.23×         | 0.95× | 0.97× | 0.99× |
| Shared prompt               | 2.29×         | 1.78× | 1.80× | 1.86× |
| Shared prompt               | 3.37×         | 2.58× | 2.61× | 2.73× |
| Nested fork                 | 1.60×         | 1.27× | 1.27× | 1.25× |
| Nested fork                 | 2.67×         | 2.06× | 2.12× | 2.10× |
| Nested fork                 | 3.56×         | 2.72× | 2.78× | 2.79× |

**The current full tree layer needs meaningful sharing to win.** Its backward path is slower without sharing,
even though the vendored core itself delivers packing gains. This is a complete GDN-layer comparison, not full-model
training. See the [optimized-Qwen chart](tree-gdn-h100-qwen-fastconv-forward_backward-speedup_vs_baseline.png).
This run uses `PYTORCH_ALLOC_CONF=expandable_segments:True` and clears unused allocations between methods.

### Core comparison and earlier fallback baseline

The table below contains the original core comparison and the **Torch Conv1d fallback layer baseline**.
Use `qwen-fastconv` for comparison against the optimized normal Qwen layer; low-sharing trees can be slower there.

| Comparison / layout                     | Packing ratio | 16K   | 32K   | 64K   |
| --------------------------------------- | ------------- | ----- | ----- | ----- |
| Vendored core vs ordinary FLA / prompt  | 1.23×         | 1.10× | 1.10× | 1.11× |
| Vendored core vs ordinary FLA / prompt  | 2.29×         | 1.99× | 1.99× | 2.03× |
| Vendored core vs ordinary FLA / prompt  | 3.37×         | 2.83× | 2.87× | 2.92× |
| Tree layer vs normal Qwen / prompt      | 1.23×         | 1.48× | 1.48× | 1.62× |
| Tree layer vs normal Qwen / prompt      | 2.29×         | 2.71× | 2.75× | 3.06× |
| Tree layer vs normal Qwen / prompt      | 3.37×         | 3.96× | 4.05× | 4.50× |
| Tree layer vs normal Qwen / nested fork | 1.60×         | 1.97× | 1.97× | 2.14× |
| Tree layer vs normal Qwen / nested fork | 2.67×         | 3.21× | 3.26× | 3.53× |
| Tree layer vs normal Qwen / nested fork | 3.56×         | 4.21× | 4.28× | 4.69× |

The layer baseline uses the optimized FLA GDN core but **Torch Conv1d, not causal-conv1d**. Its speedups include
convolution/implementation differences, not only packing. The no-sharing controls already show about 1.19–1.32×.
Do not interpret these as gains over a fully fused-convolution Qwen installation or as end-to-end model speedups.
The separate PyTorch-core fallback takes about 2.31 seconds for 16K forward + backward and OOMs at 32K/64K in this run.

Review [core speedups](tree-gdn-h100-vendored-forward_backward-speedup_vs_fla.png),
[full-layer speedups](tree-gdn-h100-qwen-layer-forward_backward-speedup_vs_baseline.png), and
[sample layouts](tree-gdn-sample-layouts.png). Raw JSON, CSV, latency and memory charts are beside them.

Nested-fork **core-only** fanout sweep, versus ordinary FLA:

| Rollouts | Packing ratio | 16K   | 32K   | 64K                   |
| -------- | ------------- | ----- | ----- | --------------------- |
| 2        | 1.60×         | 1.39× | 1.40× | 1.41×                 |
| 4        | 2.67×         | 2.32× | 2.32× | 2.36×                 |
| 8        | 4.00×         | 3.47× | 3.52× | FLA OOM; tree 45.6 ms |

Both packed and duplicated inputs remain live in this harness. Its OOM observations are specific to this memory
setup, not proof that the ordinary kernel cannot fit with a different allocation strategy.

## Samples and packing ratios

Samples are deterministic synthetic hidden states (layer benchmark) or Q/K/V/gates (core benchmark). A shared history
uses exactly the same tensor values in every duplicated rollout and in the tree. These are generated tree layouts,
not generated natural-language text or pretrained-model activations. Each JSON includes segment lengths and parents
for the newer layout sweep, making the exact sample reconstructible.

**Shared prompt:** one prompt of P tokens followed by B independent completions of C tokens.

```text
prompt P ─┬─ completion C
          ├─ completion C
          ├─ completion C
          └─ completion C
```

Normal work: `B × (P + C)` tokens. Tree work: `P + B × C` tokens.
Packing ratio = normal logical tokens / tree unique tokens. It describes token savings, not a promised speedup:
state scans, branching, convolution and launch overhead still cost time. Different convolution/GVA implementations
can also change latency independently of sharing; the 1.00× control measures that implementation difference.

For four 16K rollouts:

| Prompt | Completion each | Logical tokens | Unique tokens | Packing ratio              |
| ------ | --------------- | -------------- | ------------- | -------------------------- |
| 0      | 16K             | 64K            | 64K           | 1.00× (no-sharing control) |
| 4K     | 12K             | 64K            | 52K           | 1.23×                      |
| 12K    | 4K              | 64K            | 28K           | 2.29×                      |
| 15K    | 1K              | 64K            | 19K           | 3.37×                      |

**Nested fork:** after the prompt, two continuations of C/2 tokens each fork again into B/2 leaves of C/2 tokens.
Every rollout is still P+C tokens. Tree work is `P + 2 × C/2 + B × C/2`, so both prompt and continuation sharing help.
For four 16K rollouts, the three nonzero-prompt cases above become packing ratios 1.60×, 2.67×, and 3.56×.
At P=0 the generator deliberately emits independent roots as the no-sharing control for either chart row.

Backward uses the same logical loss: the sum of outputs over all duplicated rollouts, divided by logical token count.
Tree outputs receive their path multiplicity, so shared-prefix gradients are counted once per original rollout.
Inputs/topology are prepared outside timing; layer projections, gathers, kernels, autograd, and gradient clearing are
inside timing. Memory is **incremental peak allocated memory** above the live inputs/weights/gradients after warmup,
not total process memory. An OOM is recorded for that method, not converted into an infinite speedup.

## Reproduce on one Hopper GPU

```bash
FLA_TILELANG=0 python -m benchmarks.tree_gdn \
  --output benchmarks/tree-gdn-h100-vendored.json --repeats 5

FLA_TILELANG=0 python -m benchmarks.tree_gdn --component layer \
  --topologies prompt fork --shared-fractions 0 0.25 0.75 0.9375 --qwen-native \
  --output benchmarks/tree-gdn-h100-qwen-layer.json --repeats 5

FLA_TILELANG=0 python -m benchmarks.tree_gdn --topologies fork --branch-counts 2 4 8 \
  --shared-fractions 0.75 --output benchmarks/tree-gdn-h100-vendored-forks.json --repeats 5

# With causal-conv1d installed, this measures the optimized normal Qwen layer.
FLA_TILELANG=0 PYTORCH_ALLOC_CONF=expandable_segments:True python -m benchmarks.tree_gdn --component layer \
  --topologies prompt fork --shared-fractions 0 0.25 0.75 0.9375 \
  --output benchmarks/tree-gdn-h100-qwen-fastconv.json --repeats 5

python -m benchmarks.plot_tree_gdn benchmarks/tree-gdn-h100-vendored.json
python -m benchmarks.plot_tree_gdn benchmarks/tree-gdn-h100-qwen-layer.json
python -m benchmarks.plot_tree_gdn benchmarks/tree-gdn-h100-qwen-fastconv.json
```

Defaults: 16K/32K/64K per rollout; four rollouts; 16 key heads, 32 value heads, 128 channels; hidden size 4096 for layers;
BF16 on H100 80GB. Use `--branch-counts 2 4 8` to vary fanout. `--topologies prompt fork` separates the two scenarios.
FLA is only needed for benchmarks/reference tests; production tree kernels are vendored with their MIT license.

Performance results are not a substitute for correctness validation. The initial 16 direct kernel comparisons passed.
The expanded run passed 35/36 tests, including BF16, 16K/32K/64K forward/backward, and a subprocess that blocks FLA imports.
Separately, all 12 sample-layout checks and both FP32/BF16 convolution checks for each of Torch Conv1d and causal-conv1d passed.
One strong-decay nested-tree FP32 gate-gradient test measured 0.534% relative error versus its 0.5% tolerance.
Full-model gate/bias gradients also exceed the original strict relative tolerance in some cases. These failures have
not been hidden by widening tolerances. Two-rank FSDP2 with checkpointing passed its first optimizer step (maximum
parameter-gradient relative error 1.21%), but failed its second step on `dt_bias` (15.8% versus a 6% threshold).
The separate one-GPU Slurm allocations needed host transport (`NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_NVLS_ENABLE=0
NCCL_IB_DISABLE=1`); this was a correctness test, not a distributed performance benchmark. **Numerical validation is
unfinished; do not treat these timings as a production-readiness claim.**

The initial full-layer sweep had an allocator OOM in the 64K no-sharing tree case; the identical no-sharing control
later in the fork sweep succeeded. This is an observed allocation-history sensitivity, not evidence that the shape
can never fit. New runs clear unused allocations between methods; allocator settings are recorded in their metadata.

## Numerical rerun with causal-conv1d 1.7.0

The fresh model/FSDP rerun still fails with unchanged tolerances. The current checkout uses `block_type`, while
installed Transformers 5.8.1 exposes `layer_type`. The initial unmodified rerun records that AttributeError in
`tree-gdn-fastconv-model-rerun.xml`. `run_tree_gdn_compat.py` adds only a test-time decoder attribute alias and more
informative assertion messages; it does not modify production code, numerical operations, or tolerances.

Results in `tree-gdn-fastconv-model-compat-rerun.xml` and `tree-gdn-fastconv-bf16-detail.xml`:

- Standalone layer: FP32 gradient relative L2 error 4.1057%; BF16 11.0697%, versus a 4% threshold.
- Hybrid model: FP32 `dt_bias` error 30.6795–30.6796%, with/without checkpointing, versus 4%.
- Chunk-aligned FP32 hybrid diagnostic: first failing parameter `A_log`, error 4.1235%, versus 4%.
- BF16 hybrid model: first failure is the elementwise embedding-gradient check (`atol=0.15`, `rtol=0.08`).
  Its relative L2 error is 1.2974% and maximum absolute difference is 0.25. Subsequent parameter checks do not run.
- Two-rank FSDP2 + checkpointing: iteration zero passes with maximum gradient relative L2 error 4.3457%.
  Iteration one fails on `model.layers.0.linear_attn.dt_bias` at 36.3430% versus 6%, identically on both ranks.
  Logs: `tree-gdn-fastconv-fsdp-rank0.log` and `tree-gdn-fastconv-fsdp-rank1.log`.

These are first-failure measurements, not maxima over unchecked parameters. The FSDP test independently updates
tree/reference weights after iteration zero; iteration one's discrepancy does not isolate an FSDP-specific cause.
The fused-convolution change has **not** resolved model or distributed training parity.

### Root-cause controls after the rerun

The current adapter hard-codes `activation_in_fp32=False`. This matches Torch Conv1d followed by SiLU, but does
**not** match this environment's causal-conv1d implementation, which applies SiLU in FP32 before BF16 rounding.
`tree-gdn-stage-diagnosis.json` measures about 0.29–0.31% relative Q/K/V differences before the GDN core; gates and
beta match exactly. The optional `--match-fused-conv` switch in the diagnostic runner overrides only that precision
choice, in-process. It is not a production fix.

With this override, all three ordinary-baseline BF16 model tests pass unchanged tolerances: maximum gradient relative
L2 error 1.5261% for the standalone layer and 2.7970% for the hybrid model with/without checkpointing. See
`tree-gdn-matched-conv-diagnostic.xml`.

Two-step BF16 diagnostic, iteration one's first failing `dt_bias` gradient:

| Convolution | Reference weights before each step | Model execution                          | Error    |
| ----------- | ---------------------------------- | ---------------------------------------- | -------- |
| Mismatched  | Independent optimizer updates      | FSDP2                                    | 36.3430% |
| Mismatched  | Copied from tree model             | FSDP2                                    | 9.5658%  |
| Matched     | Independent optimizer updates      | FSDP2                                    | 9.6488%  |
| Matched     | Copied from tree model             | FSDP2                                    | 8.0436%  |
| Matched     | Copied from tree model             | Unsharded, explicitly averaged gradients | 8.0436%  |

The identical residual with/without sharding rules out FSDP as the cause **in this reproducer**. The remaining
BF16 tree-versus-normal gradient discrepancy is unresolved and still exceeds 6%. Independent BF16 optimizer
updates amplify it. Logs for each control are saved alongside the original run.

Separately, `diagnose_tree_gdn.py` replaces only the ordinary layer's GDN core with an independent FP64 token
recurrence oracle, while keeping weights, inputs, output dtype, and objective identical. The strong-decay FP32
case exposes severe gate-gradient sensitivity to default Triton TF32 dot products:

| Implementation | `dt_bias` error vs oracle, default TF32 | IEEE FP32 dot control |
| -------------- | --------------------------------------- | --------------------- |
| Ordinary FLA   | 380.35%                                 | 0.0661%               |
| Vendored tree  | 374.46%                                 | 0.1383%               |

`A_log` errors similarly fall from 537.00%/528.88% to 0.0944%/0.2040%. Forward outputs were already close; forward
agreement alone missed the gate-gradient problem. Default TF32 also gives spurious first-token gate gradients
(ordinary max 0.005145; tree max 0.007921), although decay of the zero initial state must have derivative zero.
The gate backward subtracts dot-product-derived terms; reduced precision corrupts that cancellation. Repeating
tree Q/K heads to match ordinary Qwen did not materially change these gate errors, ruling out grouped-head handling
as the main cause in this case. These controls do not prove the remaining BF16 failure has the same cause.

Artifacts: `tree-gdn-oracle-diagnosis.json`, `tree-gdn-oracle-ieee.json`, and `tree-gdn-first-gate-diagnosis.json`.
The IEEE control sets `TRITON_F32_DEFAULT=ieee`; no production kernel or performance claim was changed.

The fresh direct-kernel suite passed 48/50 checks, including long sequences. The numerical failure remains the
0.534% FP32 nested-tree gate gradient. The other failure is a SyntaxError in the current no-FLA subprocess test's
embedded Python string, not evidence of a runtime FLA import. See `tree-gdn-fastconv-kernel-rerun.xml`.
