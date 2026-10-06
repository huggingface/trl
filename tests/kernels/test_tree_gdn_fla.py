"""Vendored tree kernels versus the actual optimized ordinary-sequence FLA implementation.

FLA is a test-only dependency. The production implementation must never import it.
"""

import subprocess
import sys

import pytest
import torch
from test_tree_gdn import paths

from trl.kernels.tree_gated_delta_rule import TreeGDNPlan, tree_causal_conv1d, tree_chunk_gated_delta_rule


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Tree GDN requires a CUDA GPU")


@pytest.mark.parametrize("topology", ["prompt", "fork"])
@pytest.mark.parametrize("prefix", [0, 12])
@pytest.mark.parametrize("branches", [2, 4, 8])
def test_benchmark_sample_layout(topology, prefix, branches):
    from benchmarks.tree_gdn import sample_layout

    suffix = 16 - prefix
    offsets, parents, rows = sample_layout(prefix, suffix, branches, topology)
    assert len(rows) == branches and all(len(row) == 16 for row in rows)
    assert rows == paths(offsets, parents)
    assert {t for row in rows for t in row} == set(range(offsets[-1]))
    if prefix == 0:
        expected_unique = branches * 16
    elif topology == "prompt":
        expected_unique = prefix + branches * suffix
    else:
        expected_unique = prefix + suffix + branches * (suffix // 2)
    assert offsets[-1] == expected_unique


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_convolution_precision(dtype):
    causal_conv = pytest.importorskip("causal_conv1d").causal_conv1d_fn
    offsets, parents = (0, 65, 71, 80), (-1, 0, 0)
    plan = TreeGDNPlan.build(offsets, parents, "cuda")
    torch.manual_seed(5)
    x = torch.randn(1, 80, 32, device="cuda", dtype=dtype, requires_grad=True)
    w = torch.randn(32, 1, 4, device="cuda", dtype=dtype, requires_grad=True)
    b = torch.randn(32, device="cuda", dtype=dtype, requires_grad=True)
    refs = [t.detach().clone().requires_grad_() for t in (x, w, b)]
    out = tree_causal_conv1d(x, w, b, plan, activation_in_fp32=True)
    loss, ref_loss = 0, 0
    for row in paths(offsets, parents):
        ref = causal_conv(refs[0][:, row].transpose(1, 2), refs[1][:, 0], refs[2], activation="silu").transpose(1, 2)
        torch.testing.assert_close(out[:, row], ref, atol=0.02, rtol=0.02)
        weight = torch.randn_like(ref)
        loss = loss + (out[:, row] * weight).sum()
        ref_loss = ref_loss + (ref * weight).sum()
    for a, expected in zip(torch.autograd.grad(loss, (x, w, b)), torch.autograd.grad(ref_loss, refs), strict=True):
        error = (a.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-8)
        assert error < (1e-5 if dtype == torch.float32 else 0.02), error.item()


def test_no_fla_runtime_dependency():
    # A fresh interpreter avoids already-imported FLA modules hiding an accidental dependency.
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc
import sys

class BlockFLA(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "fla" or fullname.startswith("fla."):
            raise ImportError("FLA is deliberately unavailable")

sys.meta_path.insert(0, BlockFLA())
import torch
from trl.kernels.tree_gated_delta_rule import TreeGDNPlan, tree_chunk_gated_delta_rule

plan = TreeGDNPlan.build((0, 65, 70, 73), (-1, 0, 0), "cuda")
q, k = [torch.randn(1, 73, 2, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(2)]
v = torch.randn(1, 73, 4, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
g = torch.full((1, 73, 4), -0.1, device="cuda", requires_grad=True)
beta = torch.full_like(g, 0.5, dtype=torch.bfloat16, requires_grad=True)
out = tree_chunk_gated_delta_rule(q, k, v, g, beta, plan)
out.float().sum().backward()
assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in (q, k, v, g, beta))
assert not any(name == "fla" or name.startswith("fla.") for name in sys.modules)
""",
        ],
        check=True,
    )


@pytest.mark.parametrize("head_dim", [32, 128])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("decay_scale", [0.2, 20.0])
@pytest.mark.parametrize("topology", ["chain", "star", "nested", "forest"])
def test_vendored_against_fla(head_dim, dtype, decay_scale, topology, record_property):
    fla = pytest.importorskip("fla.ops.gated_delta_rule")
    layouts = {
        "chain": ((0, 257), (-1,)),
        "star": ((0, 65, 194, 267, 272), (-1, 0, 0, 0)),
        "nested": ((0, 65, 68, 197, 261, 262), (-1, 0, 1, 1, 0)),
        "forest": ((0, 63, 192, 194, 259, 262), (-1, 0, 0, -1, 3)),
    }
    offsets, parents = layouts[topology]
    plan = TreeGDNPlan.build(offsets, parents, "cuda")
    torch.manual_seed(41)
    n = offsets[-1]
    data = [torch.randn(1, n, 2, head_dim, dtype=dtype, device="cuda") for _ in range(2)]
    data += [torch.randn(1, n, 4, head_dim, dtype=dtype, device="cuda")]
    data += [-torch.rand(1, n, 4, device="cuda") * decay_scale, torch.rand(1, n, 4, dtype=dtype, device="cuda")]
    actual = [x.requires_grad_() for x in data]
    expected = [x.detach().clone().requires_grad_() for x in data]
    out = tree_chunk_gated_delta_rule(*actual, plan)
    loss, ref_loss = 0, 0
    for row in paths(offsets, parents):
        ref, _ = fla.chunk_gated_delta_rule(*(x[:, row] for x in expected), use_qk_l2norm_in_kernel=True)
        torch.testing.assert_close(out[:, row], ref, atol=0.004, rtol=0.03)
        weight = torch.randn_like(ref)
        loss = loss + (out[:, row] * weight).sum()
        ref_loss = ref_loss + (ref * weight).sum()
    max_error = 0
    for name, a, b in zip(
        ("q", "k", "v", "g", "beta"),
        torch.autograd.grad(loss, actual),
        torch.autograd.grad(ref_loss, expected),
        strict=True,
    ):
        relative_rms = (a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-8)
        assert relative_rms < (0.005 if dtype == torch.float32 else 0.025), (name, relative_rms.item())
        max_error = max(max_error, relative_rms.item())
    record_property("max_gradient_relative_rms", max_error)


@pytest.mark.parametrize("length", [16384, 32768, 65536])
def test_long_sequence_fla_parity(length, record_property):
    fla = pytest.importorskip("fla.ops.gated_delta_rule")
    prefix, suffix, branches = length * 3 // 4, length // 4, 4
    offsets = (0, prefix, *(prefix + (i + 1) * suffix for i in range(branches)))
    parents = (-1, 0, 0, 0, 0)
    plan = TreeGDNPlan.build(offsets, parents, "cuda")
    index = torch.tensor([t for row in paths(offsets, parents) for t in row], device="cuda")
    torch.manual_seed(17)
    n = offsets[-1]
    data = [torch.randn(1, n, 2, 128, dtype=torch.bfloat16, device="cuda") for _ in range(2)]
    data += [torch.randn(1, n, 4, 128, dtype=torch.bfloat16, device="cuda")]
    data += [-torch.rand(1, n, 4, device="cuda") * 0.05, torch.rand(1, n, 4, dtype=torch.bfloat16, device="cuda")]
    actual = [x.requires_grad_() for x in data]
    expected = [x.detach().clone().requires_grad_() for x in data]
    out = tree_chunk_gated_delta_rule(*actual, plan).index_select(1, index)
    cu_cpu = torch.arange(branches + 1, dtype=torch.long) * length
    ref, _ = fla.chunk_gated_delta_rule(
        *(x.index_select(1, index) for x in expected),
        cu_seqlens=cu_cpu.cuda(),
        cu_seqlens_cpu=cu_cpu,
        use_qk_l2norm_in_kernel=True,
    )
    torch.testing.assert_close(out, ref, atol=0.004, rtol=0.03)
    weight = torch.randn_like(ref)
    max_error = 0
    for name, a, b in zip(
        ("q", "k", "v", "g", "beta"),
        torch.autograd.grad((out * weight).sum(), actual),
        torch.autograd.grad((ref * weight).sum(), expected),
        strict=True,
    ):
        relative_rms = (a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-8)
        assert relative_rms < 0.025, (name, relative_rms.item())
        max_error = max(max_error, relative_rms.item())
    record_property("max_gradient_relative_rms", max_error)
