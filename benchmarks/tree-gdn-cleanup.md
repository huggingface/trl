# Tree GDN single-file cleanup

The vendored implementation now lives in `trl/kernels/_tree_gdn_fla/tree.py`: eight implementation files became one,
with 768 Python lines removed (about 31%). Unused branches and options were removed; active arithmetic order, casts,
state propagation, and autotuning candidates were preserved. This is not a numerical-precision fix.

## Validation

Run on NVIDIA H100 80GB in `hopper-dev`, with PyTorch 2.11.0+cu130 and Triton 3.7.1. The pre-cleanup baseline is
`a1ae2ef4976df67234ca401daae0c4b947ea260f`, retained locally by `backup/tree-gdn-before-cleanup-20261007`.

- **36 exact kernel comparisons passed.** Outputs and Q/K/V/gate/beta gradients are bitwise equal with matching
  pinned launch configurations. Coverage includes chains, shared prefixes, nested forks, multiple roots, FP32/BF16,
  weak/strong decay, dimensions 32/64/96/128, grouped/equal heads, normalization on/off, and partial chunks.
- **One-rank and two-rank FSDP2 with checkpointing passed both SGD steps.** Losses, every parameter gradient, and
  updated weights match the old tree implementation exactly. The two-rank test varies topology by rank and step.
- **Ordinary-FLA suite: 49 passed, 1 failed.** The pre-existing FP32, dimension-128, strong-decay nested-tree
  gate-gradient error is unchanged: `0.0053399233147501945` relative L2 (0.533992%) versus a 0.5% tolerance.
  Tolerances were not loosened. All 16K/32K/64K parity checks and the repaired no-FLA subprocess test passed.
- **18 long-sequence comparisons completed.** Outputs and all gradients were also bitwise equal with independent
  autotuning at 16K/32K/64K across prompt/fork layouts and packing ratios 1.00/2.29/2.67/3.37/3.56.
  Before/after forward + backward speed ratios were **0.9988–1.0097×**: effectively unchanged performance.
  Measurements use BF16, 16 key / 32 value heads, dimension 128, four rollouts, and the median of ten CUDA-event
  measurements after three warmups. This compares old versus new tree kernels, not tree versus ordinary packing.

The exact FSDP regression likewise compares tree packing **before versus after cleanup**. It does not resolve the
previously observed ordinary-packing model-gradient discrepancies or establish full training readiness.

Review the [performance chart](tree-gdn-cleanup-long.png) ([SVG](tree-gdn-cleanup-long.svg)),
[raw timings](tree-gdn-cleanup-long.json), [exact kernel checks](tree-gdn-cleanup-parity.json),
[ordinary-FLA test report](tree-gdn-cleanup-kernels.xml),
[one-rank FSDP results](tree-gdn-cleanup-fsdp-single-rank0.json), and two-rank FSDP results for
[rank 0](tree-gdn-cleanup-fsdp-two-rank0.json) / [rank 1](tree-gdn-cleanup-fsdp-two-rank1.json).

## Reproduction

From the repository root inside a suitable GPU allocation:

```bash
PYTHONPATH=. python benchmarks/compare_tree_gdn_cleanup.py \
  --before a1ae2ef4976df67234ca401daae0c4b947ea260f --output benchmarks/tree-gdn-cleanup-parity.json
PYTHONPATH=. torchrun --standalone --nproc-per-node=2 benchmarks/compare_tree_gdn_cleanup_fsdp.py \
  --before a1ae2ef4976df67234ca401daae0c4b947ea260f --output benchmarks/tree-gdn-cleanup-fsdp-two.json
FLA_TILELANG=0 PYTHONPATH=. python -m pytest tests/kernels/test_tree_gdn_fla.py -q \
  --junitxml=benchmarks/tree-gdn-cleanup-kernels.xml
PYTHONPATH=. python benchmarks/compare_tree_gdn_cleanup.py --benchmark \
  --before a1ae2ef4976df67234ca401daae0c4b947ea260f --output benchmarks/tree-gdn-cleanup-long.json
python benchmarks/plot_tree_gdn_cleanup.py benchmarks/tree-gdn-cleanup-long.json
```

The actual two-rank run used two separate one-GPU Slurm allocations. These required host transport:
`NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_NVLS_ENABLE=0 NCCL_IB_DISABLE=1`.
Without those settings, the first attempt failed in NCCL's first all-gather with `invalid device ordinal`, before
executing a GDN kernel. This is a distributed correctness check, not a distributed performance benchmark.
