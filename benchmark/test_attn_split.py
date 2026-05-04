# ruff: noqa
"""
Test how EP attention sharding interacts with FSDP2.

Test 1: torchrun only (no FSDP2) — does forward work?
Test 2: accelerate + FSDP2 — does forward work? What are the actual tensor shapes?
"""

import os

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM
from transformers.distributed.configuration_utils import DistributedConfig


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    if rank == 0:
        print(f"World size: {world_size}")
        print()

    # ======= Test 1: Load with EP, NO FSDP2 =======
    if rank == 0:
        print("=== Test 1: EP only (no FSDP2) ===")

    device_mesh = dist.init_device_mesh("cuda", (world_size,))
    model_ep = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-30B-A3B",
        dtype=torch.bfloat16,
        distributed_config=DistributedConfig(enable_expert_parallel=True),
        device_mesh=device_mesh,
    )

    if rank == 0:
        attn = model_ep.model.layers[0].self_attn
        print(f"  q_proj.weight: {list(attn.q_proj.weight.shape)} type={type(attn.q_proj.weight).__name__}")
        print(f"  k_proj.weight: {list(attn.k_proj.weight.shape)} type={type(attn.k_proj.weight).__name__}")
        print(f"  experts.gate_up_proj: {list(model_ep.model.layers[0].mlp.experts.gate_up_proj.shape)}")

    x = torch.tensor([[1, 2, 3]], device="cuda")
    try:
        with torch.no_grad():
            out = model_ep(x, use_cache=False)
        if rank == 0:
            print(f"  Forward: OK, logits={out.logits.shape}")
    except Exception as e:
        if rank == 0:
            print(f"  Forward: FAILED — {e}")

    del model_ep
    torch.cuda.empty_cache()
    dist.barrier()

    # ======= Test 2: Load with EP, then wrap with FSDP2 =======
    if rank == 0:
        print()
        print("=== Test 2: EP + FSDP2 (accelerator.prepare) ===")

    from accelerate import Accelerator

    accelerator = Accelerator()

    device_mesh2 = dist.init_device_mesh("cuda", (world_size,))
    model_ep2 = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-30B-A3B",
        dtype=torch.bfloat16,
        distributed_config=DistributedConfig(enable_expert_parallel=True),
        device_mesh=device_mesh2,
    )

    if rank == 0:
        attn2 = model_ep2.model.layers[0].self_attn
        print("  Before FSDP2:")
        print(f"    q_proj.weight: {list(attn2.q_proj.weight.shape)} type={type(attn2.q_proj.weight).__name__}")
        print(f"    k_proj.weight: {list(attn2.k_proj.weight.shape)} type={type(attn2.k_proj.weight).__name__}")

    optimizer = torch.optim.AdamW(model_ep2.parameters(), lr=1e-5)
    model_ep2, optimizer = accelerator.prepare(model_ep2, optimizer)

    if rank == 0:
        print("  After FSDP2:")
        print(f"    model type: {type(model_ep2).__name__}")
        # Navigate through FSDP wrapper to find actual params
        try:
            # FSDP wraps the model, need to access inner module
            inner = model_ep2
            if hasattr(inner, "module"):
                inner = inner.module
            if hasattr(inner, "model"):
                inner = inner.model
            attn3 = inner.layers[0].self_attn
            print(f"    q_proj.weight: {list(attn3.q_proj.weight.shape)} type={type(attn3.q_proj.weight).__name__}")
            print(f"    k_proj.weight: {list(attn3.k_proj.weight.shape)} type={type(attn3.k_proj.weight).__name__}")
        except Exception as e:
            print(f"    Could not access inner params: {e}")

    x2 = torch.tensor([[1, 2, 3]], device=accelerator.device)
    try:
        with torch.no_grad():
            out2 = model_ep2(x2, use_cache=False)
        if rank == 0:
            print(f"  Forward: OK, logits={out2.logits.shape}")
            print(f"  logits sample: {out2.logits[0, -1, :5]}")
    except Exception as e:
        if rank == 0:
            print(f"  Forward: FAILED — {e}")

    # ======= Test 3: Load WITHOUT EP, wrap with FSDP2 (baseline) =======
    if rank == 0:
        print()
        print("=== Test 3: No EP, just FSDP2 (baseline) ===")

    del model_ep2, optimizer
    torch.cuda.empty_cache()
    dist.barrier()

    accelerator2 = Accelerator()
    model_no_ep = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-30B-A3B",
        dtype=torch.bfloat16,
    )

    if rank == 0:
        attn4 = model_no_ep.model.layers[0].self_attn
        print("  Before FSDP2:")
        print(f"    q_proj.weight: {list(attn4.q_proj.weight.shape)}")
        print(f"    k_proj.weight: {list(attn4.k_proj.weight.shape)}")
        print(f"    experts.gate_up_proj: {list(model_no_ep.model.layers[0].mlp.experts.gate_up_proj.shape)}")

    optimizer2 = torch.optim.AdamW(model_no_ep.parameters(), lr=1e-5)
    model_no_ep, optimizer2 = accelerator2.prepare(model_no_ep, optimizer2)

    x3 = torch.tensor([[1, 2, 3]], device=accelerator2.device)
    try:
        with torch.no_grad():
            out3 = model_no_ep(x3, use_cache=False)
        if rank == 0:
            print(f"  Forward: OK, logits={out3.logits.shape}")
            print(f"  logits sample: {out3.logits[0, -1, :5]}")
    except Exception as e:
        if rank == 0:
            print(f"  Forward: FAILED — {e}")

    dist.barrier()
    if rank == 0:
        print()
        print("=== Done ===")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
