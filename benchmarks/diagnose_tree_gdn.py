"""Diagnostic only: identical Qwen layer weights/loss, independent FP64 recurrence oracle."""

import argparse
import copy
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

from trl.experimental.async_grpo.tree import gdn
from trl.kernels.tree_gated_delta_rule import TreeGDNPlan, tree_chunk_gated_delta_rule


def oracle(q, k, v, g, beta, **kwargs):
    output_dtype = v.dtype
    q, k, v, g, beta = [x.double() for x in (q, k, v, g, beta)]
    q, k = [x * torch.rsqrt(x.square().sum(-1, keepdim=True) + 1e-6) for x in (q, k)]
    q = q * q.shape[-1] ** -0.5
    state = v.new_zeros(q.shape[0], v.shape[2], k.shape[-1], v.shape[-1])
    outputs = []
    for t in range(q.shape[1]):
        state = state * g[:, t].exp()[..., None, None]
        delta = (v[:, t] - (state * k[:, t, :, :, None]).sum(-2)) * beta[:, t, :, None]
        state = state + k[:, t, :, :, None] * delta[..., None, :]
        outputs.append((state * q[:, t, :, :, None]).sum(-2))
    return torch.stack(outputs, 1).to(output_dtype), None


def aligned(q, k, v, g, beta, **kwargs):
    kwargs.pop("initial_state", None)
    kwargs.pop("output_final_state", None)
    prefix, state = chunk_gated_delta_rule(
        q[:, :65], k[:, :65], v[:, :65], g[:, :65], beta[:, :65], output_final_state=True, **kwargs
    )
    suffix, _ = chunk_gated_delta_rule(
        q[:, 65:], k[:, 65:], v[:, 65:], g[:, 65:], beta[:, 65:], initial_state=state, **kwargs
    )
    return torch.cat((prefix, suffix), 1), None


def repeated_tree(q, k, v, g, beta, plan):
    factor = v.shape[2] // q.shape[2]
    return tree_chunk_gated_delta_rule(
        q.repeat_interleave(factor, 2), k.repeat_interleave(factor, 2), v, g, beta, plan
    )


def run(dtype, fused):
    torch.manual_seed(3)
    config = Qwen3_5TextConfig(
        hidden_size=128,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        dtype=dtype,
    )
    with torch.device("cuda"):
        original = Qwen3_5GatedDeltaNet(config, 0).to(dtype)
    if not fused:
        original.causal_conv1d_fn = None
    data = torch.randn(1, 80, 128, device="cuda", dtype=dtype)
    rows = [list(range(71)), list(range(65)) + list(range(71, 80))]
    weights = [torch.randn(1, len(row), 128, device="cuda", dtype=dtype) for row in rows]
    index = torch.tensor([row + [80] * (74 - len(row)) for row in rows], device="cuda")
    plan = TreeGDNPlan.build((0, 65, 71, 80), (-1, 0, 0), "cuda")
    gradients, outputs, stages, first_gate = {}, {}, {}, {}
    for method in ("oracle64", "normal", "aligned", "tree", "tree_repeated"):
        layer = copy.deepcopy(original)
        x = data.detach().clone().requires_grad_()
        stages[method] = {}

        def capture(core, is_tree, method=method):
            def wrapped(q, k, v, *args, **kwargs):
                gate = args[0] if args else kwargs["g"]
                beta = args[1] if args else kwargs["beta"]

                def record_first_gate(grad):
                    first_gate[method] = grad[:, 0].detach().float()

                gate.register_hook(record_first_gate)
                for name, value in zip(("q", "k", "v", "g", "beta"), (q, k, v, gate, beta), strict=True):
                    if name in ("q", "k"):
                        value = value.repeat_interleave(v.shape[2] // value.shape[2], 2)
                    parts = (
                        [value[:, row] for row in rows]
                        if is_tree
                        else [value[r : r + 1, : len(row)] for r, row in enumerate(rows)]
                    )
                    stages[method][name] = torch.cat([part.detach().float().flatten() for part in parts])
                return core(q, k, v, *args, **kwargs)

            return wrapped

        if method.startswith("tree"):
            core = repeated_tree if method == "tree_repeated" else tree_chunk_gated_delta_rule
            gdn.tree_chunk_gated_delta_rule = capture(core, True)
            out = gdn.qwen3_5_tree_gdn(layer, x, plan)
            selected = [out[:, row] for row in rows]
        else:
            core = {"oracle64": oracle, "normal": chunk_gated_delta_rule, "aligned": aligned}[method]
            layer.chunk_gated_delta_rule = capture(core, False)
            out = layer(F.pad(x, (0, 0, 0, 1))[0, index])
            selected = [out[r : r + 1, : len(row)] for r, row in enumerate(rows)]
        sum((out.float() * weight).sum() for out, weight in zip(selected, weights, strict=True)).backward()
        gradients[method] = {
            "input": x.grad.float().detach(),
            **{n: p.grad.float().detach() for n, p in layer.named_parameters()},
        }
        outputs[method] = torch.cat([out.detach().flatten().float() for out in selected])
    result = {
        "dtype": str(dtype),
        "fused_convolution": fused,
        "triton_f32_default": os.environ.get("TRITON_F32_DEFAULT", "tf32"),
        "methods": {},
    }
    for method in gradients:
        result["methods"][method] = {
            # The initial state is zero, so the first token's decay must have zero derivative.
            "first_gate_gradient_abs_max": first_gate[method].abs().max().item(),
            "core_input_relative_to_normal": {
                name: ((value - stages["normal"][name]).norm() / stages["normal"][name].norm().clamp_min(1e-8)).item()
                for name, value in stages[method].items()
            },
            "output_relative_to_oracle": (
                (outputs[method] - outputs["oracle64"]).norm() / outputs["oracle64"].norm()
            ).item(),
            "gradients": {
                name: {
                    "norm": grad.norm().item(),
                    "relative_to_oracle": (
                        (grad - gradients["oracle64"][name]).norm()
                        / gradients["oracle64"][name].norm().clamp_min(1e-8)
                    ).item(),
                    "relative_to_normal": (
                        (grad - gradients["normal"][name]).norm() / gradients["normal"][name].norm().clamp_min(1e-8)
                    ).item(),
                }
                for name, grad in gradients[method].items()
            },
        }
    print(json.dumps(result), flush=True)  # noqa: T201
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--float32-only", action="store_true")
    parser.add_argument("--fused-only", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("benchmarks/tree-gdn-oracle-diagnosis.json"))
    args = parser.parse_args()
    results = []
    for dtype in (torch.float32,) if args.float32_only else (torch.float32, torch.bfloat16):
        for fused in (True,) if args.fused_only else (False, True):
            results.append(run(dtype, fused))
            args.output.write_text(json.dumps(results, indent=2) + "\n")
