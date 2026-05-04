# ruff: noqa
"""Debug FA3 + EP DTensor stripping.

Loads Qwen3-30B-A3B with enable_expert_parallel=True under both sdpa and FA3,
counts DTensor parameters, and prints which modules are/aren't DTensor.

Run with torchrun on 2 GPUs:
    torchrun --nproc_per_node=2 benchmark/debug_fa3_ep_dtensor.py [sdpa|fa3]
"""

import os
import sys

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from transformers import AutoModelForCausalLM
from transformers.distributed import DistributedConfig


def main():
    impl = sys.argv[1] if len(sys.argv) > 1 else "sdpa"
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if impl == "fa3":
        attn = "kernels-community/vllm-flash-attn3"
    else:
        attn = "sdpa"

    if rank == 0:
        print(f"\n=== Loading with attn_implementation={attn}, world_size={world_size} ===\n", flush=True)

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(rank)

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-30B-A3B",
        dtype=torch.bfloat16,
        attn_implementation=attn,
        distributed_config=DistributedConfig(enable_expert_parallel=True),
    )

    # Count DTensor params
    total = 0
    dtensor_count = 0
    moe_dtensor = 0
    moe_total = 0
    attn_dtensor = 0
    attn_total = 0
    other_dtensor = 0
    other_total = 0

    sample_dtensor_names = []
    sample_non_dtensor_names = []

    for name, p in model.named_parameters():
        total += 1
        is_dt = isinstance(p, DTensor)
        if is_dt:
            dtensor_count += 1
            if len(sample_dtensor_names) < 5:
                sample_dtensor_names.append(name)
        else:
            if len(sample_non_dtensor_names) < 5:
                sample_non_dtensor_names.append(name)

        if "mlp.experts" in name or "mlp.gate" in name:
            moe_total += 1
            if is_dt:
                moe_dtensor += 1
        elif "self_attn" in name:
            attn_total += 1
            if is_dt:
                attn_dtensor += 1
        else:
            other_total += 1
            if is_dt:
                other_dtensor += 1

    if rank == 0:
        print(f"\n[{impl}] DTensor counts:", flush=True)
        print(f"  total params:    {dtensor_count}/{total}", flush=True)
        print(f"  MoE params:      {moe_dtensor}/{moe_total} (experts + gate)", flush=True)
        print(f"  Attention:       {attn_dtensor}/{attn_total}", flush=True)
        print(f"  Other (emb/ln):  {other_dtensor}/{other_total}", flush=True)
        print("\n  sample DTensor params:", flush=True)
        for n in sample_dtensor_names:
            print(f"    {n}", flush=True)
        print("\n  sample non-DTensor params:", flush=True)
        for n in sample_non_dtensor_names:
            print(f"    {n}", flush=True)
        print(f"\n  model._tp_size: {getattr(model, '_tp_size', 'N/A')}", flush=True)
        print(f"  model._device_mesh: {getattr(model, '_device_mesh', None)}", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
