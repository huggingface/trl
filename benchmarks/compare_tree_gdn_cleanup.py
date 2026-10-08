"""Compare the current tree kernel to a Git revision, without changing the worktree.

Use pinned matching launch configurations for exact forward/backward regression checks;
use --benchmark for independently autotuned long-sequence latency comparisons.
"""

import argparse
import importlib
import json
import os
import subprocess
import sys
import tempfile
from functools import partial
from itertools import accumulate
from pathlib import Path

import torch
import triton
from triton.runtime.autotuner import Autotuner, Heuristics

from trl.kernels._tree_gdn_fla import tree as current
from trl.kernels.tree_gated_delta_rule import TreeGDNPlan


def load_before(revision, directory):
    package = Path(directory) / "tree_gdn_before"
    package.mkdir()
    prefix = "trl/kernels/_tree_gdn_fla/"
    paths = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", revision, prefix], text=True).splitlines()
    for path in paths:
        if path.endswith(".py"):
            (package / Path(path).name).write_bytes(subprocess.check_output(["git", "show", f"{revision}:{path}"]))
    sys.path.insert(0, directory)
    return importlib.import_module("tree_gdn_before.tree")


def matching_configs(before):
    # Keep tensor shapes, arithmetic order AND tile sizes identical. Autotuning is
    # timed separately: a different winning tile can change floating-point rounding.
    for name, kernel in vars(current).items():
        if isinstance(kernel, Heuristics):
            kernel = kernel.fn
        if not isinstance(kernel, Autotuner):
            continue
        original = next(
            vars(module)[name]
            for key, module in sys.modules.items()
            if key.startswith("tree_gdn_before.") and name in vars(module)
        )
        if isinstance(original, Heuristics):
            original = original.fn
        kernel.configs = [original.configs[0]]
        original.configs = [original.configs[0]]


def evaluate(module, data, plan, weight, normalize):
    inputs = [tensor.detach().requires_grad_() for tensor in data]
    out = module.TreeGatedDeltaRule.apply(*inputs, plan, data[0].shape[-1] ** -0.5, normalize)
    grads = torch.autograd.grad((out * weight).sum(), inputs)
    return out.detach(), *(grad.detach() for grad in grads)


def elapsed(fn, repeats):
    for _ in range(3):
        fn()
    samples = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return sorted(samples)[len(samples) // 2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    torch.set_float32_matmul_precision("highest")
    layouts = {
        "chain": ((0, 257), (-1,)),
        "star": ((0, 65, 194, 267, 272), (-1, 0, 0, 0)),
        "nested": ((0, 65, 68, 197, 261, 262), (-1, 0, 1, 1, 0)),
        "forest": ((0, 63, 192, 194, 259, 262), (-1, 0, 0, -1, 3)),
    }
    cases = []
    if args.benchmark:
        for length in (16384, 32768, 65536):
            for topology in ("prompt", "fork"):
                for fraction in (0.0, 0.75, 0.9375):
                    prefix = int(length * fraction)
                    suffix = length - prefix
                    if prefix == 0:
                        offsets, parents = tuple(i * length for i in range(5)), (-1,) * 4
                    elif topology == "prompt":
                        offsets = (0, *accumulate([prefix, *([suffix] * 4)]))
                        parents = (-1, 0, 0, 0, 0)
                    else:
                        # Two intermediate forks, each followed by two completions.
                        offsets = (0, *accumulate([prefix, *([suffix // 2] * 6)]))
                        parents = (-1, 0, 1, 1, 0, 4, 4)
                    cases.append(
                        dict(
                            topology=topology,
                            length=length,
                            fraction=fraction,
                            offsets=offsets,
                            parents=parents,
                            dim=128,
                            dtype="bfloat16",
                            decay=0.2,
                            heads=16,
                            value_heads=32,
                            normalize=True,
                        )
                    )
    else:
        for topology, (offsets, parents) in layouts.items():
            for dim in (32, 128):
                for dtype in ("float32", "bfloat16"):
                    for decay in (0.2, 20.0):
                        cases.append(
                            dict(
                                topology=topology,
                                offsets=offsets,
                                parents=parents,
                                dim=dim,
                                dtype=dtype,
                                decay=decay,
                                heads=2,
                                value_heads=4,
                                normalize=True,
                            )
                        )
        # Exercise the remaining live branches, including non-power-of-two dimensions.
        for dim in (64, 96):
            for dtype in ("float32", "bfloat16"):
                cases.append(
                    dict(
                        topology="forest",
                        offsets=layouts["forest"][0],
                        parents=layouts["forest"][1],
                        dim=dim,
                        dtype=dtype,
                        decay=0.2,
                        heads=2,
                        value_heads=2,
                        normalize=False,
                    )
                )
    if args.smoke:
        cases = cases[:1]
    report = dict(
        before=args.before,
        device=torch.cuda.get_device_name(),
        torch=torch.__version__,
        triton=triton.__version__,
        triton_f32_default=os.environ.get("TRITON_F32_DEFAULT", "tf32"),
        pinned_configs=not args.benchmark,
        cases=[],
    )
    with tempfile.TemporaryDirectory(prefix="tree-gdn-before-") as directory:
        before = load_before(args.before, directory)
        if not args.benchmark:
            matching_configs(before)
        for case in cases:
            print(f"START {case}", flush=True)  # noqa: T201
            torch.manual_seed(41)
            n, heads, value_heads, dim = case["offsets"][-1], case["heads"], case["value_heads"], case["dim"]
            dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[case["dtype"]]
            data = [torch.randn(1, n, heads, dim, device="cuda", dtype=dtype) for _ in range(2)]
            data += [torch.randn(1, n, value_heads, dim, device="cuda", dtype=dtype)]
            data += [
                -torch.rand(1, n, value_heads, device="cuda") * case["decay"],
                torch.rand(1, n, value_heads, device="cuda", dtype=dtype),
            ]
            if not case["normalize"]:
                data[0] /= dim**0.5
                data[1] /= dim**0.5
            weight = torch.randn_like(data[2])
            plan = TreeGDNPlan.build(case["offsets"], case["parents"], "cuda")
            expected = evaluate(before, data, plan, weight, case["normalize"])
            actual = evaluate(current, data, plan, weight, case["normalize"])
            errors = {}
            for name, got, ref in zip(("out", "dq", "dk", "dv", "dg", "dbeta"), actual, expected, strict=True):
                difference = got.float() - ref.float()
                errors[name] = dict(
                    exact=torch.equal(got, ref),
                    max_abs=difference.abs().max().item(),
                    relative_l2=(difference.norm() / ref.float().norm().clamp_min(1e-8)).item(),
                )
            row = dict(**case, errors=errors)
            del expected, actual
            if args.benchmark:
                row["packing_ratio"] = 4 * case["length"] / n
                row["before_ms"] = elapsed(partial(evaluate, before, data, plan, weight, True), args.repeats)
                row["after_ms"] = elapsed(partial(evaluate, current, data, plan, weight, True), args.repeats)
                row["speedup"] = row["before_ms"] / row["after_ms"]
            report["cases"].append(row)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(row), flush=True)  # noqa: T201
            if not args.benchmark:
                assert all(error["exact"] for error in errors.values()), errors
            del data, weight, plan
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
