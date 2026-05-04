# ruff: noqa
"""Test: DeepSpeed comm.is_initialized vs torch.distributed.is_initialized."""

import os

import torch
import torch.distributed as torch_dist


rank = int(os.environ.get("RANK", 0))
torch_dist.init_process_group("nccl")
torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

from deepspeed import comm as ds_comm


if rank == 0:
    print(f"torch.distributed.is_initialized() = {torch_dist.is_initialized()}", flush=True)
    print(f"deepspeed.comm.is_initialized()     = {ds_comm.is_initialized()}", flush=True)

# Initialize DeepSpeed comm backend
if rank == 0:
    print("\nCalling ds_comm.init_distributed('nccl')...", flush=True)
ds_comm.init_distributed("nccl")

if rank == 0:
    print("After ds_comm.init_distributed:", flush=True)
    print(f"  torch.distributed.is_initialized() = {torch_dist.is_initialized()}", flush=True)
    print(f"  deepspeed.comm.is_initialized()     = {ds_comm.is_initialized()}", flush=True)

# Now try SP init
import deepspeed.runtime.sequence_parallel.parallel_state_sp as mpu


try:
    mpu.initialize_sequence_parallel(sequence_parallel_size=2)
    if rank == 0:
        print("\nSP init OK!", flush=True)
except Exception as e:
    if rank == 0:
        print(f"\nSP init FAILED: {e}", flush=True)

torch_dist.destroy_process_group()
