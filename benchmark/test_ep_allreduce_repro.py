# ruff: noqa
"""
Isolated EP-multi-node-hang repro.

Sets up 2 EP groups (intra-node) of 8 ranks each, then runs the same
size all-reduce that the failing job's EP path issues:
  shape (49152, 2048) bf16 = 96M elements = 192MB
  on intra-node 8-rank group (NVLink)

Loops 10000 iterations to see if it hangs the same way the training does.
"""

import os
import time

import torch
import torch.distributed as dist


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    # Build EP groups: ranks 0-7 are EP group A (node 0), 8-15 are EP group B (node 1).
    EP_SIZE = 8
    ep_group_id = rank // EP_SIZE
    ep_group_ranks = list(range(ep_group_id * EP_SIZE, (ep_group_id + 1) * EP_SIZE))

    # Each rank creates BOTH groups so torch sees a consistent global view.
    groups = []
    for gid in range(world_size // EP_SIZE):
        ranks = list(range(gid * EP_SIZE, (gid + 1) * EP_SIZE))
        g = dist.new_group(ranks=ranks)
        groups.append(g)
    my_ep_group = groups[ep_group_id]

    # Also build a world-PG gather collective (mimics gather_for_metrics).
    world_group = dist.group.WORLD

    # Use the same shape as the failing job (48k context).
    SEQ_LEN = 49152
    HIDDEN = 2048

    if rank == 0:
        print(f"world_size={world_size} EP_SIZE={EP_SIZE}", flush=True)
        print(
            f"EP groups: {[list(range(i * EP_SIZE, (i + 1) * EP_SIZE)) for i in range(world_size // EP_SIZE)]}",
            flush=True,
        )
        print(f"all_reduce shape ({SEQ_LEN}, {HIDDEN}) bf16 = {SEQ_LEN * HIDDEN * 2 / 1e9:.2f} GB", flush=True)

    x = torch.randn(SEQ_LEN, HIDDEN, dtype=torch.bfloat16, device=f"cuda:{local_rank}")

    NUM_ITER = 10000
    t0 = time.time()
    for it in range(NUM_ITER):
        # 1) EP-group all-reduce (the failing collective)
        dist.all_reduce(x, op=dist.ReduceOp.SUM, group=my_ep_group)

        # Mimic the world-group scalar gather every ~150 EP collectives (like gather_for_metrics).
        if it % 150 == 0:
            scalar = torch.tensor([rank], device=f"cuda:{local_rank}", dtype=torch.float32)
            gathered = torch.empty(world_size, device=f"cuda:{local_rank}", dtype=torch.float32)
            dist.all_gather_into_tensor(gathered, scalar, group=world_group)

        if it % 100 == 0 and rank == 0:
            elapsed = time.time() - t0
            print(f"[{elapsed:.1f}s] iter {it}/{NUM_ITER}", flush=True)

    elapsed = time.time() - t0
    if rank == 0:
        print(f"[{elapsed:.1f}s] DONE — {NUM_ITER} iterations completed without hang", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
