# ruff: noqa
"""Test script for Qwen3-30B-A3B with fused expert weights.

Test 1: Single-GPU forward pass (no EP) - validates fused weights load correctly
Test 2: Multi-GPU EP forward pass - validates expert parallelism works

Usage:
    # Single-GPU forward pass
    python test_ep_qwen3_fused.py --test forward

    # Multi-GPU EP (run via torchrun)
    torchrun --nproc_per_node=8 test_ep_qwen3_fused.py --test ep
"""

import argparse
import os

import torch
import torch.distributed as dist


def test_forward(model_path):
    """Single-GPU forward pass test."""
    from transformers import AutoModelForCausalLM

    print(f"Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16)
    model = model.cuda()
    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

    input_ids = torch.tensor([[1, 2, 3, 4, 5]], device="cuda")
    with torch.no_grad():
        out = model(input_ids, use_cache=False)

    logits = out.logits
    print(f"OK: logits.shape={logits.shape}, nan={logits.isnan().any()}, inf={logits.isinf().any()}")
    print(f"Logits sample: {logits[0, -1, :5]}")
    assert not logits.isnan().any(), "NaN in logits!"
    assert not logits.isinf().any(), "Inf in logits!"
    print("PASSED: Single-GPU forward pass")


def test_ep(model_path):
    """Multi-GPU expert parallelism test."""
    from transformers import AutoModelForCausalLM
    from transformers.distributed.configuration_utils import DistributedConfig

    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    torch.cuda.set_device(local_rank)

    if rank == 0:
        print(f"Loading model with EP on {world_size} GPUs...")

    distributed_config = DistributedConfig(enable_expert_parallel=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        distributed_config=distributed_config,
    ).cuda()

    if rank == 0:
        # Print expert shapes to verify sharding
        for name, param in model.named_parameters():
            if "layers.0.mlp" in name:
                print(f"  {name}: {list(param.shape)}")

    input_ids = torch.tensor([[1, 2, 3, 4, 5]], device="cuda")
    with torch.no_grad():
        out = model(input_ids, use_cache=False)

    logits = out.logits
    if rank == 0:
        print(f"OK: logits.shape={logits.shape}, nan={logits.isnan().any()}, inf={logits.isinf().any()}")
        print(f"Logits sample: {logits[0, -1, :5]}")
    assert not logits.isnan().any(), f"NaN in logits on rank {rank}!"
    assert not logits.isinf().any(), f"Inf in logits on rank {rank}!"

    dist.barrier()
    if rank == 0:
        print("PASSED: Multi-GPU EP forward pass")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", choices=["forward", "ep"], required=True)
    parser.add_argument("--model_path", default="/fsx/amine_dirhoussi/Qwen3-30B-A3B-fused")
    args = parser.parse_args()

    if args.test == "forward":
        test_forward(args.model_path)
    elif args.test == "ep":
        test_ep(args.model_path)
