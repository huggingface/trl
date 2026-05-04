# ruff: noqa
"""Reproduce the FA3+EP accelerate.prepare crash on 2 GPUs.

Runs the full path: from_pretrained(distributed_config=EP) -> Accelerator -> prepare().
This is what SFTTrainer does internally. We replicate it minimally to bisect FA3 vs sdpa.

Run with torchrun on 2 GPUs:
    torchrun --nproc_per_node=2 benchmark/debug_fa3_ep_prepare.py [sdpa|fa3]
"""

import os
import sys
import traceback

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from transformers import AutoModelForCausalLM
from transformers.distributed import DistributedConfig


def count_dtensor(model):
    n_dt = 0
    n_total = 0
    for _, p in model.named_parameters():
        n_total += 1
        if isinstance(p, DTensor):
            n_dt += 1
    return n_dt, n_total


def main():
    impl = sys.argv[1] if len(sys.argv) > 1 else "sdpa"
    rank = int(os.environ["RANK"])

    attn = "kernels-community/vllm-flash-attn3" if impl == "fa3" else "sdpa"

    # accelerate creates the PG + sets up FSDP2 mesh. We must use Accelerator,
    # because Accelerator wraps the trainer's prepare flow.
    from accelerate import Accelerator
    from accelerate.utils import FullyShardedDataParallelPlugin

    fsdp_plugin = FullyShardedDataParallelPlugin(
        fsdp_version=2,
        auto_wrap_policy="transformer_based_wrap",
        cpu_ram_efficient_loading=True,
    )
    acc = Accelerator(fsdp_plugin=fsdp_plugin)

    if rank == 0:
        print(f"\n=== {impl.upper()} + EP on world_size={acc.num_processes} ===\n", flush=True)
        print(f"  parallelism_config (pre-load): {acc.parallelism_config}", flush=True)

    # Load model with EP enabled — this is the SFTTrainer path
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-30B-A3B",
        dtype=torch.bfloat16,
        attn_implementation=attn,
        distributed_config=DistributedConfig(enable_expert_parallel=True),
    )

    n_dt, n_total = count_dtensor(model)
    if rank == 0:
        print("\n  After from_pretrained:", flush=True)
        print(f"    DTensor params: {n_dt}/{n_total}", flush=True)
        print(f"    model._tp_size: {getattr(model, '_tp_size', None)}", flush=True)
        print(
            f"    model._device_mesh dim_names: {getattr(model._device_mesh, 'mesh_dim_names', None) if model._device_mesh is not None else None}",
            flush=True,
        )

    # SFTTrainer also auto-builds ParallelismConfig from model.tp_size — replicate that
    if getattr(model, "tp_size", None) is not None and model.tp_size > 1:
        from accelerate.parallelism_config import ParallelismConfig

        pc = ParallelismConfig(tp_size=model.tp_size)
        acc.parallelism_config = pc
        # rebuild device mesh
        acc.state.device_mesh = pc.build_device_mesh(acc.device.type)
        if rank == 0:
            print("\n  After ParallelismConfig override:", flush=True)
            print(f"    parallelism_config: {pc}", flush=True)
            print(f"    fsdp_dim_names: {pc.fsdp_dim_names}", flush=True)
            print(f"    device_mesh dim_names: {acc.state.device_mesh.mesh_dim_names}", flush=True)

    # Now try accelerate.prepare — this is where the original crash hits
    if rank == 0:
        print("\n  Calling accelerator.prepare(model)...", flush=True)
    try:
        model_prepared = acc.prepare(model)
        n_dt2, n_total2 = count_dtensor(model_prepared)
        if rank == 0:
            print("  ✓ prepare succeeded", flush=True)
            print(f"    DTensor params after prepare: {n_dt2}/{n_total2}", flush=True)
    except Exception as e:
        if rank == 0:
            print(f"  ✗ prepare CRASHED: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
