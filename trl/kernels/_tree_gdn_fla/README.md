# Tree-native FLA kernels

Vendored from the `flash-linear-attention==0.5.2` distribution, upstream
[fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention).
The upstream MIT license is retained in [LICENSE](LICENSE), along with source copyright headers.
There is no `fla` runtime import. Only the training chunk path is included; no backend registry, context parallelism,
generation cache, or package-level dispatch is vendored.

## Review order

All implementation logic lives in [`tree.py`](tree.py), in this order:

1. Device/compiler checks, then the forward/backward autograd pipeline and depth-ordered launches.
2. Tree state scans. Forward reads the parent's FP32 final state; backward sums children's FP32 input-state
   gradients in CSR order before scanning the segment backward. No atomics, prefix duplication, or state gathers.
3. Fused triangular solve, WY representation, chunk outputs and local gradients.
4. Gate cumulative sums and Q/K normalization.

Only the tree path is retained: one flattened forest, variable-length segments, 64-token chunks, scalar gates,
key-first states, and key dimensions up to 128. Unused transposed-state, per-key/constant-gate, ungated, fixed-length,
optional-output, and large-key/normalization-dimension branches have been removed. Grouped value heads and partial
chunks remain supported. The cleanup preserves the active arithmetic order, casts, and autotuning configurations;
it is not a numerical-precision fix.

Chunk-local kernels see each linear tree segment as a variable-length sequence and run over the entire forest in
one launch. Only state kernels launch by tree depth. This preserves partial chunks at branch boundaries without
padding or pretending that unrelated branches are one sequence. The plan computes all chunk/segment indices once.

FLA's custom autotune cache is replaced with Triton's standard autotuner. Upstream tuning configurations and relevant
device checks are retained; unused launchers and FLA dispatch decorators are omitted. Formatting follows TRL.
The public path targets Qwen3.5 (64-token chunks, key dimensions up to 128), and retains FLA's numerical precision,
which is not bitwise-equivalent to the separate FP32 reference implementation. Known gate-gradient and model-level
parity discrepancies remain under investigation. Matching the pre-cleanup tree kernel is not evidence that those
ordinary-packing discrepancies are resolved.

On Hopper, Triton >= 3.7.1 is required. The upstream backward correctness guard is retained. Benchmark metadata records
the compiler version and `FLA_TILELANG` setting; use `FLA_TILELANG=0` to compare both paths with the same Triton backend.

## Cleanup regression checks

Measured results and reproduction commands are in [the cleanup report](../../../benchmarks/tree-gdn-cleanup.md).

`benchmarks/compare_tree_gdn_cleanup.py` loads an older revision into an isolated temporary package, leaving the
worktree unchanged. Its default mode pins matching launch configurations and requires bitwise equality for the output
and all five input gradients. Cases cover chains, shared prefixes, nested forks, forests, strong/weak decay, FP32/BF16,
grouped/equal head counts, partial chunks, and normalization on/off. `--benchmark` instead uses independent autotuning
and measures forward + backward at 16K/32K/64K across prompt/fork sharing ratios.

`benchmarks/compare_tree_gdn_cleanup_fsdp.py` compares the old and new tree implementations inside a Qwen model with
FSDP2 and activation checkpointing. Both optimizer steps must match losses, every parameter gradient, and updated
weights exactly. Run with `torchrun`; a one-rank run exercises FSDP hooks but does not validate multi-rank collectives.

## Source provenance

Paths below are relative to the installed upstream `fla/` package. SHA-256 hashes identify the unmodified source files.

| Source | SHA-256 |
| --- | --- |
| `ops/gated_delta_rule/chunk_fwd.py` | `1760a7dd26db2118bb1af7b9b7c115a27c97372e6a4d2e913d1542c9fedcda9e` |
| `ops/gated_delta_rule/wy_fast.py` | `bf24a66524ea56383fe6c9a97ae6b060ebe9490589c8499fba06d6172dc4510f` |
| `ops/common/chunk_delta_h.py` | `ff461bcc40e9cd0b2ca24af9ae9c3c6d99b2deae047c3737687c93b3fcd6ea57` |
| `ops/common/chunk_o.py` | `f174d431ce67139204a0916f638b5af158dd5917e2cb37c0ca03f5ce735466a0` |
| `ops/utils/cumsum.py` | `0405701c46cee331088bfee395b3bf37f8384829dace0aa3a55a509844bcf4cb` |
| `modules/l2norm.py` | `30f7feebdbfa87b8e90c143ec62a7fa0781bfd736c856c0ec69452f0c778be3f` |
