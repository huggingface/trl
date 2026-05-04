# ruff: noqa
"""
Vacuum repro v3 for the EP-multi-node hang.

V3 adds:
- DP-group all-reduce (mimics DS-Z2/FSDP grad reduce-scatter) after each backward
- DP groups are inter-node pairs: [0, 8], [1, 9], ..., [7, 15]

Hypothesis: the hang is caused by inter-node DP-group collectives interleaving
with intra-node EP-group all-reduces, sharing NCCL channels in a way that stalls
when one side is heavily loaded.
"""

import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from transformers.integrations.tensor_parallel import _AllReduceBackward, _AllReduceForward


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    EP_SIZE = 8
    HIDDEN = 2048
    INTERMEDIATE = 768
    NUM_EXPERTS = 128
    NUM_TOP_K = 8
    NUM_LOCAL_EXPERTS = NUM_EXPERTS // EP_SIZE  # 16
    NUM_LAYERS = 48
    SEQ_LEN = 65536

    # EP groups (intra-node)
    ep_group_id = rank // EP_SIZE
    ep_rank_in_group = rank % EP_SIZE
    ep_groups = []
    for gid in range(world_size // EP_SIZE):
        ranks_in_group = list(range(gid * EP_SIZE, (gid + 1) * EP_SIZE))
        g = dist.new_group(ranks=ranks_in_group)
        ep_groups.append(g)
    my_ep_pg = ep_groups[ep_group_id]

    # DP groups (inter-node) — each rank pairs with the rank EP_SIZE ranks away
    dp_groups = []
    for ep_rank in range(EP_SIZE):
        ranks_in_group = [ep_rank + g * EP_SIZE for g in range(world_size // EP_SIZE)]
        g = dist.new_group(ranks=ranks_in_group)
        dp_groups.append(g)
    my_dp_pg = dp_groups[ep_rank_in_group]

    if rank == 0:
        print(f"world_size={world_size} EP_SIZE={EP_SIZE} HIDDEN={HIDDEN} SEQ_LEN={SEQ_LEN}", flush=True)
        print(
            f"EP groups (intra-node): {[list(range(g * EP_SIZE, (g + 1) * EP_SIZE)) for g in range(world_size // EP_SIZE)]}",
            flush=True,
        )
        print(
            f"DP groups (inter-node): {[[er + g * EP_SIZE for g in range(world_size // EP_SIZE)] for er in range(EP_SIZE)]}",
            flush=True,
        )

    class EpMeshShim:
        def __init__(self, group, size):
            self._group = group
            self._size = size

        def get_group(self):
            return self._group

        def size(self):
            return self._size

    ep_mesh = EpMeshShim(my_ep_pg, EP_SIZE)

    weights = []
    for _ in range(NUM_LAYERS):
        gu = nn.Parameter(
            torch.randn(NUM_LOCAL_EXPERTS, HIDDEN, 2 * INTERMEDIATE, dtype=torch.bfloat16, device=f"cuda:{local_rank}")
            * 0.01
        )
        dn = nn.Parameter(
            torch.randn(NUM_LOCAL_EXPERTS, INTERMEDIATE, HIDDEN, dtype=torch.bfloat16, device=f"cuda:{local_rank}")
            * 0.01
        )
        weights.append((gu, dn))

    def fake_moe_layer(hidden, layer_idx):
        gu, dn = weights[layer_idx]
        top_k_global = torch.randint(0, NUM_EXPERTS, (SEQ_LEN, NUM_TOP_K), device=f"cuda:{local_rank}")
        top_k_weights = torch.softmax(
            torch.randn(SEQ_LEN, NUM_TOP_K, dtype=torch.bfloat16, device=f"cuda:{local_rank}"), dim=-1
        )
        owner_rank = top_k_global // NUM_LOCAL_EXPERTS
        non_local_mask = owner_rank != ep_rank_in_group
        top_k_weights = top_k_weights.masked_fill(non_local_mask, 0.0)

        h = _AllReduceBackward.apply(hidden, ep_mesh)
        w = _AllReduceBackward.apply(top_k_weights, ep_mesh)

        proj = torch.matmul(h, gu[0])
        gate, up = proj.chunk(2, dim=-1)
        proj = torch.nn.functional.silu(gate) * up
        proj = torch.matmul(proj, dn[0])
        weighted = proj * w.sum(dim=-1, keepdim=True).to(proj.dtype) * 0.1
        out = _AllReduceForward.apply(weighted, ep_mesh)
        return out

    all_params = [p for w in weights for p in w]
    opt = torch.optim.AdamW(all_params, lr=1e-5)

    NUM_STEPS = 30
    if rank == 0:
        print(f"\nRunning {NUM_STEPS} steps × {NUM_LAYERS} layers + DP-group reduce-scatter on grads", flush=True)

    t0 = time.time()
    for step in range(NUM_STEPS):
        opt.zero_grad()
        hidden = torch.randn(SEQ_LEN, HIDDEN, dtype=torch.bfloat16, device=f"cuda:{local_rank}", requires_grad=True)

        for layer_idx in range(NUM_LAYERS):
            hidden = fake_moe_layer(hidden, layer_idx)

        loss = hidden.float().sum()
        loss.backward()

        # Mimic DS-Z2: reduce gradients across DP group (inter-node)
        for p in all_params:
            if p.grad is not None:
                # Bucket-style reduce-scatter approximated with all_reduce
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, group=my_dp_pg)
                p.grad.div_(world_size // EP_SIZE)

        opt.step()

        if rank == 0:
            mem = torch.cuda.memory_allocated(local_rank) / 1e9
            peak = torch.cuda.max_memory_allocated(local_rank) / 1e9
            elapsed = time.time() - t0
            print(
                f"  [{elapsed:6.1f}s] step {step + 1}/{NUM_STEPS}  loss={loss.item():.3e}  alloc={mem:.2f}GB peak={peak:.2f}GB",
                flush=True,
            )

    if rank == 0:
        print(f"\n*** {NUM_STEPS} steps completed (vacuum v3 — with DP grad reducer) ***", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
