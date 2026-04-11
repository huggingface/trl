# ruff: noqa: T201, B023
"""
NCCL bandwidth benchmark.

Usage:
    # 1 node
    srun --nodes=1 --gpus-per-node=8 --partition=hopper-prod --exclusive --mem=0 --time=0:10:00 \
        bash -c "source .venv/bin/activate && torchrun --nproc_per_node=8 benchmark/nccl_bench.py"

    # 2 nodes
    srun --nodes=2 --gpus-per-node=8 --partition=hopper-prod --exclusive --mem=0 --time=0:10:00 \
        bash -c 'source .venv/bin/activate && torchrun --nproc_per_node=8 --nnodes=$SLURM_JOB_NUM_NODES \
        --rdzv_backend=c10d --rdzv_endpoint=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -1):29500 \
        benchmark/nccl_bench.py'
"""

import os
import time

import torch
import torch.distributed as dist


def bench_collective(name, fn, data_bytes, warmup=5, iters=20):
    """Benchmark a collective and return (time_ms, algbw_GBs, busbw_GBs)."""
    world_size = dist.get_world_size()

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier()

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    avg_s = elapsed / iters
    algbw = data_bytes / avg_s  # bytes/s

    # Bus bandwidth correction factors (NCCL convention)
    if name == "allreduce":
        busbw = algbw * 2 * (world_size - 1) / world_size
    elif name in ("allgather", "reduce_scatter", "all_to_all"):
        busbw = algbw * (world_size - 1) / world_size
    else:
        busbw = algbw

    return avg_s * 1000, algbw / 1e9, busbw / 1e9


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    if rank == 0:
        print(f"NCCL Bandwidth Benchmark: {world_size} GPUs")
        print(f"{'Op':>15} {'Size':>10} {'Time(ms)':>10} {'AlgBW(GB/s)':>12} {'BusBW(GB/s)':>12}")
        print("-" * 62)

    sizes_mb = [1, 8, 64, 256, 512, 1024]

    for size_mb in sizes_mb:
        numel = size_mb * 1024 * 1024 // 2  # bf16 = 2 bytes per element
        data_bytes = numel * 2  # total bytes = size_mb * 1MB

        # --- allreduce: each rank contributes `data_bytes`, result is same size ---
        tensor = torch.randn(numel, dtype=torch.bfloat16, device="cuda")

        def allreduce_fn():
            dist.all_reduce(tensor)

        t, algbw, busbw = bench_collective("allreduce", allreduce_fn, data_bytes)
        if rank == 0:
            print(f"{'allreduce':>15} {size_mb:>8}MB {t:>9.2f}ms {algbw:>11.1f} {busbw:>11.1f}")

        # --- allgather: each rank sends `data_bytes`, total output = data_bytes * world_size ---
        # algbw = total_output / time (not per-rank input), matching NCCL-tests convention
        ag_input = torch.randn(numel, dtype=torch.bfloat16, device="cuda")
        ag_output = torch.empty(numel * world_size, dtype=torch.bfloat16, device="cuda")

        def allgather_fn():
            dist.all_gather_into_tensor(ag_output, ag_input)

        t, algbw, busbw = bench_collective("allgather", allgather_fn, data_bytes * world_size)
        if rank == 0:
            print(f"{'allgather':>15} {size_mb:>8}MB {t:>9.2f}ms {algbw:>11.1f} {busbw:>11.1f}")

        # --- reduce_scatter: input = data_bytes * world_size, output = data_bytes per rank ---
        rs_input = torch.randn(numel * world_size, dtype=torch.bfloat16, device="cuda")
        rs_output = torch.empty(numel, dtype=torch.bfloat16, device="cuda")

        def reduce_scatter_fn():
            dist.reduce_scatter_tensor(rs_output, rs_input)

        t, algbw, busbw = bench_collective("reduce_scatter", reduce_scatter_fn, data_bytes * world_size)
        if rank == 0:
            print(f"{'reduce_scatter':>15} {size_mb:>8}MB {t:>9.2f}ms {algbw:>11.1f} {busbw:>11.1f}")

        # --- all_to_all: each rank sends data_bytes total (data_bytes/world_size to each peer) ---
        a2a_input = torch.randn(numel, dtype=torch.bfloat16, device="cuda")
        a2a_output = torch.empty_like(a2a_input)

        def all_to_all_fn():
            dist.all_to_all_single(a2a_output, a2a_input)

        t, algbw, busbw = bench_collective("all_to_all", all_to_all_fn, data_bytes)
        if rank == 0:
            print(f"{'all_to_all':>15} {size_mb:>8}MB {t:>9.2f}ms {algbw:>11.1f} {busbw:>11.1f}")

        if rank == 0:
            print()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
