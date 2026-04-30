"""Byte-exact replay of the kernel call captured from the broken full-stack repro.

Now also mirrors production's `DTensor.from_local(...).to_local()` autograd hop on
the expert weights, since that's what the real wrapper does and it's the only
difference vs my earlier synthetic tests.

Loads `/fsx/amine_dirhoussi/trl/benchmark/_sonic_capture.pt` (saved by the
SONICMOE_DEBUG_DUMP=1 path in `transformers/integrations/sonicmoe.py`) and runs
forward + backward with the EXACT same tensors that the kernel saw inside the
broken full-stack run. Reports grad norms.

If standalone-replay produces the same broken grad norms as production:
    → bug IS in the kernel call itself (not the autograd graph composition).
If standalone-replay produces healthy grad norms:
    → bug is post-kernel-call (DTensor.to_local backward, all_reduce_backward,
      DeepSpeed grad bucketing, or some other autograd-graph effect).
"""

# ruff: noqa: T201, S101, E741
import os

import torch
import torch.distributed as dist
from kernels import get_kernel
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard


os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29500")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")
dist.init_process_group(backend="nccl", rank=0, world_size=1)
torch.cuda.set_device(0)
mesh = init_device_mesh("cuda", (1,), mesh_dim_names=("ep",))

KERNEL = get_kernel("IlyasMoutawwakil/sonic-moe", revision="main")
ActivationType = KERNEL.enums.ActivationType
moe_general_routing_inputs = KERNEL.moe_general_routing_inputs

DEVICE = "cuda"
ACT_MAP = {"silu": "swiglu", "gelu": "geglu", "relu": "reglu"}

import sys


_path = sys.argv[1] if len(sys.argv) > 1 else "/fsx/amine_dirhoussi/trl/benchmark/_sonic_capture.pt"
cap = torch.load(_path, weights_only=False)
print(f"Loaded capture from: {_path}")
print("Keys:", list(cap.keys()))


def run(*, clamp: bool):
    h = cap["hidden_states"].to(DEVICE).clone().requires_grad_(True)
    rs = cap["router_scores"].to(DEVICE).clone().requires_grad_(True)
    tk = cap["token_idx"].to(DEVICE)
    ei = cap["expert_ids"].to(DEVICE)
    w1_orig = cap["w1"].to(DEVICE)
    w2_orig = cap["w2"].to(DEVICE)

    # Reconstruct the differentiable parameters in their pre-permute orientation
    # so backward writes to them. w1 was permute(1,2,0) of (E, 2*I, H). Undo:
    E = cap["num_experts"]
    # Wrapper used perm=(1,2,0) since is_transposed=False. Inverse is perm=(2,0,1).
    # gate_up_proj shape: (E, 2*I, H), down_proj shape: (E, H, I).
    g_orig = w1_orig.permute(2, 0, 1).contiguous()  # (1536,8,2) -> (2,1536,8)
    d_orig = w2_orig.permute(2, 0, 1).contiguous()  # (8,768,2) -> (2,8,768)

    g = g_orig.detach().clone().requires_grad_(True)
    d = d_orig.detach().clone().requires_grad_(True)
    # Wrap as DTensor and immediately call to_local — mimics the production wrapper
    # path where gate_up_proj/down_proj are real EP-sharded DTensors.
    g_dt = DTensor.from_local(g, mesh, [Shard(0)], run_check=False)
    d_dt = DTensor.from_local(d, mesh, [Shard(0)], run_check=False)
    g_local = g_dt.to_local()
    d_local = d_dt.to_local()
    w1 = g_local.permute(1, 2, 0)
    w2 = d_local.permute(1, 2, 0)
    # Sanity check
    assert w1.shape == cap["w1"].shape, f"{w1.shape} vs {cap['w1'].shape}"
    assert w2.shape == cap["w2"].shape, f"{w2.shape} vs {cap['w2'].shape}"
    assert w1.stride() == cap["w1"].stride(), f"{w1.stride()} vs {cap['w1'].stride()}"
    assert w2.stride() == cap["w2"].stride(), f"{w2.stride()} vs {cap['w2'].stride()}"
    assert torch.equal(w1, cap["w1"].to(DEVICE)), "w1 values differ"
    assert torch.equal(w2, cap["w2"].to(DEVICE)), "w2 values differ"

    expert_ids = ei.clone()
    if clamp:
        expert_ids = expert_ids.clamp(0, E - 1)

    activation_type = getattr(ActivationType, ACT_MAP.get(cap["act_name"], "swiglu").upper(), ActivationType.SWIGLU)

    out, _ = moe_general_routing_inputs(
        h,
        rs,
        tk,
        expert_ids,
        w1,
        None,
        w2,
        None,
        E=E,
        activation_type=activation_type,
        stream_id=torch.cuda.current_stream(DEVICE).cuda_stream,
        is_inference_mode_enabled=False,
        concat_layout=cap["is_concatenated"],
    )

    # Synthetic loss to drive a backward — match what production does conceptually.
    fake_grad_out = torch.randn_like(out)
    out.backward(fake_grad_out)

    return {
        "out_finite": torch.isfinite(out).all().item(),
        "out_nan": int(out.isnan().sum().item()),
        "out_norm": out.float().norm().item(),
        "out_max": out.float().abs().max().item(),
        "g_grad_norm": g.grad.float().norm().item(),
        "d_grad_norm": d.grad.float().norm().item(),
        "h_grad_norm": h.grad.float().norm().item() if h.grad is not None else None,
        "h_grad_max": h.grad.float().abs().max().item() if h.grad is not None else None,
        "rs_grad_norm": rs.grad.float().norm().item() if rs.grad is not None else None,
        "rs_grad_max": rs.grad.float().abs().max().item() if rs.grad is not None else None,
        "g_grad_nan": int(g.grad.isnan().sum().item()),
        "d_grad_nan": int(d.grad.isnan().sum().item()),
        "g_grad_max": g.grad.float().abs().max().item(),
        "d_grad_max": d.grad.float().abs().max().item(),
    }


print("\nKernel input stats:")
print(f"  hidden_states: {tuple(cap['hidden_states'].shape)} {cap['hidden_states'].dtype}")
print(f"  router_scores: {tuple(cap['router_scores'].shape)} sum={cap['router_scores'].float().sum().item():.4g}")
print(
    f"  expert_ids: {tuple(cap['expert_ids'].shape)} sentinel_count={(cap['expert_ids'] == cap['num_experts']).sum().item()}"
)
print(f"  num_experts={cap['num_experts']} is_concatenated={cap['is_concatenated']} act={cap['act_name']}")

print("\n=== Replay 1: clamp=False (production-bug config) ===")
r = run(clamp=False)
print(f"  out: norm={r['out_norm']:.4g} max={r['out_max']:.4g} nan={r['out_nan']}")
print(f"  g.grad: norm={r['g_grad_norm']:.4g} max={r['g_grad_max']:.4g} nan={r['g_grad_nan']}")
print(f"  d.grad: norm={r['d_grad_norm']:.4g} max={r['d_grad_max']:.4g} nan={r['d_grad_nan']}")
print(f"  h.grad: norm={r['h_grad_norm']} max={r['h_grad_max']}")
print(f"  rs.grad: norm={r['rs_grad_norm']} max={r['rs_grad_max']}")

print("\n=== Replay 2: clamp=True (control) ===")
r = run(clamp=True)
print(f"  out: norm={r['out_norm']:.4g} max={r['out_max']:.4g} nan={r['out_nan']}")
print(f"  g.grad: norm={r['g_grad_norm']:.4g} max={r['g_grad_max']:.4g} nan={r['g_grad_nan']}")
print(f"  d.grad: norm={r['d_grad_norm']:.4g} max={r['d_grad_max']:.4g} nan={r['d_grad_nan']}")
print(f"  h.grad: norm={r['h_grad_norm']} max={r['h_grad_max']}")
print(f"  rs.grad: norm={r['rs_grad_norm']} max={r['rs_grad_max']}")
