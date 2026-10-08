"""Compare the same logical rollouts with ordinary FLA and tree GDN on one GPU.

Run from the repository root: python -m benchmarks.tree_gdn --output results.json
Times include gathers and autograd, but exclude topology construction and input generation.
"""

import argparse
import gc
import importlib.metadata
import json
import os
import statistics
import time
from itertools import accumulate
from pathlib import Path

import torch
import triton
from fla.ops.gated_delta_rule import chunk_gated_delta_rule

from trl.kernels.tree_gated_delta_rule import TreeGDNPlan, tree_chunk_gated_delta_rule

from .tree_gdn_torch import torch_tree_gdn


def measure(fn, repeats):
    start = time.perf_counter()
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    warmup_s = time.perf_counter() - start
    torch.cuda.reset_peak_memory_stats()
    allocated = torch.cuda.memory_allocated()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    return dict(
        ms=statistics.median(times),
        peak_extra_mib=(torch.cuda.max_memory_allocated() - allocated) / 2**20,
        warmup_and_compile_s=warmup_s,
    )


def sample_layout(prefix, suffix, branches, topology):
    """Identical logical samples represented either as duplicated paths or shared tree segments."""
    offsets, parents, paths = [0], [], []

    def segment(length, parent):
        index = len(parents)
        paths.append(([] if parent == -1 else paths[parent]) + list(range(offsets[-1], offsets[-1] + length)))
        offsets.append(offsets[-1] + length)
        parents.append(parent)
        return index

    if prefix == 0:
        for _ in range(branches):
            segment(suffix, -1)
    else:
        root = segment(prefix, -1)
        if topology == "prompt":
            for _ in range(branches):
                segment(suffix, root)
        elif topology == "fork":
            if branches % 2:
                raise ValueError("Fork samples require an even number of leaves.")
            for _ in range(2):
                fork = segment(suffix // 2, root)
                for _ in range(branches // 2):
                    segment(suffix - suffix // 2, fork)
        else:
            # A multi-turn trajectory: each turn's assistant tokens are a leaf, and the observation that follows
            # extends the shared history, so depth grows with the turn count instead of staying at three.
            node, step = root, max(suffix // (2 * branches), 64)
            for turn in range(branches):
                segment(step, node)
                if turn < branches - 1:
                    node = segment(step, node)
    return tuple(offsets), tuple(parents), [path for s, path in enumerate(paths) if s not in parents]


def run(prefix, suffix, branches, args, topology="prompt"):
    offsets, parents, rows = sample_layout(prefix, suffix, branches, topology)
    plan = TreeGDNPlan.build(offsets, parents, "cuda")
    index = torch.tensor([t for row in rows for t in row], device="cuda")
    cu_cpu = torch.tensor([0, *accumulate(len(row) for row in rows)], dtype=torch.long)
    cu = cu_cpu.cuda()
    n, h, hv, d = offsets[-1], args.key_heads, args.value_heads, args.head_dim
    torch.manual_seed(0)
    layer = None
    if args.component == "core":
        packed = [torch.randn(1, n, h, d, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
        packed += [torch.randn(1, n, hv, d, device="cuda", dtype=torch.bfloat16)]
        packed += [
            -torch.rand(1, n, hv, device="cuda") * 0.05,
            torch.rand(1, n, hv, device="cuda", dtype=torch.bfloat16),
        ]
    else:
        from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen_module
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet, torch_chunk_gated_delta_rule

        from trl.experimental.async_grpo.tree.gdn import qwen3_5_tree_gdn

        config = Qwen3_5TextConfig(
            hidden_size=args.hidden_size,
            linear_num_key_heads=h,
            linear_num_value_heads=hv,
            linear_key_head_dim=d,
            linear_value_head_dim=d,
            dtype=torch.bfloat16,
        )
        with torch.device("cuda"):
            layer = Qwen3_5GatedDeltaNet(config, 0).to(torch.bfloat16)
        # Since Transformers 5.9 the layer calls the module-level function, which dispatches to FLA, instead of
        # holding it as an attribute, so swapping the implementation means rebinding the module global.
        default_chunk = torch_chunk_gated_delta_rule
        packed = [torch.randn(1, n, args.hidden_size, device="cuda", dtype=torch.bfloat16)]
    duplicated = [x.index_select(1, index).detach().requires_grad_() for x in packed]
    packed = [x.requires_grad_() for x in packed]
    logical = index.numel()
    # Identical sum-of-output loss: repeated prefix outputs get their rollout multiplicity.
    multiplicity = torch.bincount(index, minlength=n).view(1, n, *([1] * (packed[0].ndim - 2))).float() / logical

    def baseline(backward, native=False):
        for x in [*duplicated, *packed]:
            x.grad = None
        if layer is not None:
            layer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(backward):
            if layer is None:
                out, _ = chunk_gated_delta_rule(
                    *duplicated, cu_seqlens=cu, cu_seqlens_cpu=cu_cpu, use_qk_l2norm_in_kernel=True
                )
            else:
                qwen_module.torch_chunk_gated_delta_rule = default_chunk
                out = layer(duplicated[0].view(branches, prefix + suffix, args.hidden_size))
            if backward:
                (out.float().sum() / logical).backward()

    def tree(backward, compiled=False):
        for x in [*duplicated, *packed]:
            x.grad = None
        if layer is not None:
            layer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(backward):
            if compiled:
                out = torch_tree_gdn(*packed, plan, prefix, branches)
            else:
                out = (
                    tree_chunk_gated_delta_rule(*packed, plan)
                    if layer is None
                    else qwen3_5_tree_gdn(layer, packed[0], plan)
                )
            if backward:
                (out.float() * multiplicity).sum().backward()

    result = dict(
        prefix=prefix,
        suffix=suffix,
        topology=topology,
        rollout_length=prefix + suffix,
        segment_lengths=[b - a for a, b in zip(offsets, offsets[1:], strict=False)],
        segment_parents=parents,
        branches=branches,
        logical_tokens=logical,
        unique_tokens=n,
        packing_ratio=logical / n,
        segments=len(parents),
        depth=len(plan.levels),
        partial_chunks=sum((b - a) % 64 != 0 for a, b in zip(offsets, offsets[1:], strict=False)),
    )
    baseline_key = "fla_sequence" if layer is None else "qwen_default"
    if layer is not None:
        result["qwen_default_core"] = f"{default_chunk.__module__}.{default_chunk.__name__}"
        result["qwen_convolution"] = "causal_conv1d" if layer.causal_conv1d_fn is not None else "torch.nn.Conv1d"
        result["transformers_version"] = importlib.metadata.version("transformers")
        if layer.causal_conv1d_fn is not None:
            result["causal_conv1d_version"] = importlib.metadata.version("causal-conv1d")
    for mode, backward in [("forward", False), ("forward_backward", True)]:
        methods = {
            baseline_key: lambda backward=backward: baseline(backward),
            "triton_tree": lambda backward=backward: tree(backward),
        }
        if args.component == "core" and topology == "prompt" and prefix > 0:
            methods["compiled_tree"] = lambda backward=backward: tree(backward, compiled=True)
        if layer is not None and args.qwen_native:
            methods["qwen_native"] = lambda backward=backward: baseline(backward, native=True)
        measurements = {}
        for name, fn in methods.items():
            # A previous method's allocator layout must not decide whether this method fits.
            for x in [*duplicated, *packed]:
                x.grad = None
            if layer is not None:
                layer.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            try:
                measurements[name] = measure(fn, args.repeats)
            except torch.OutOfMemoryError as error:
                measurements[name] = {"error": str(error)}
                gc.collect()
                torch.cuda.empty_cache()
        for measurement in measurements.values():
            if "error" in measurement:
                continue
            measurement["logical_tokens_s"] = logical * 1000 / measurement["ms"]
            if "ms" in measurements[baseline_key]:
                measurement["speedup_vs_baseline"] = measurements[baseline_key]["ms"] / measurement["ms"]
                if layer is None:
                    measurement["speedup_vs_fla"] = measurement["speedup_vs_baseline"]
            if "compiled_tree" in measurements and "ms" in measurements["compiled_tree"]:
                measurement["speedup_vs_compiled"] = measurements["compiled_tree"]["ms"] / measurement["ms"]
        result[mode] = measurements
    print(json.dumps(result), flush=True)  # noqa: T201
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--key-heads", type=int, default=16)
    parser.add_argument("--value-heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--component", choices=["core", "layer"], default="core")
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--lengths", type=int, nargs="+", default=[16384, 32768, 65536])
    parser.add_argument("--shared-fractions", type=float, nargs="+", default=[0.25, 0.75, 0.9375])
    parser.add_argument("--branches", type=int, default=4)
    parser.add_argument("--branch-counts", type=int, nargs="+")
    parser.add_argument("--topologies", choices=["prompt", "fork", "chain"], nargs="+", default=["prompt"])
    parser.add_argument("--qwen-native", action="store_true", help="Also time Qwen's built-in PyTorch chunk fallback.")
    args = parser.parse_args()
    import fla

    results = dict(
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        triton=triton.__version__,
        fla=fla.__version__,
        fla_tilelang=os.environ.get("FLA_TILELANG", "auto"),
        allocator=os.environ.get("PYTORCH_ALLOC_CONF", os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "default")),
        dtype="bfloat16",
        component=args.component,
        hidden_size=args.hidden_size if args.component == "layer" else None,
        key_heads=args.key_heads,
        value_heads=args.value_heads,
        head_dim=args.head_dim,
        repeats=args.repeats,
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        compiled_scan_chunks_per_region=8,
        tree_implementation="vendored FLA 0.5.2 with tree-native state kernels",
        compiled_implementation="FP32 chunk algebra and eight-chunk compiled PyTorch scan",
        input_generation="Seeded synthetic hidden states (layer) or Q/K/V/gates (core); shared histories reuse identical values.",
        cases=[],
    )
    for topology in args.topologies:
        for branches in args.branch_counts or [args.branches]:
            for length in args.lengths:
                for fraction in args.shared_fractions:
                    prefix = int(length * fraction)
                    try:
                        result = run(prefix, length - prefix, branches, args, topology)
                    except torch.OutOfMemoryError as error:
                        result = dict(
                            prefix=prefix,
                            suffix=length - prefix,
                            branches=branches,
                            topology=topology,
                            error=str(error),
                        )
                        print(json.dumps(result), flush=True)  # noqa: T201
                    results["cases"].append(result)
                    args.output.write_text(json.dumps(results, indent=2) + "\n")
                    gc.collect()
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
