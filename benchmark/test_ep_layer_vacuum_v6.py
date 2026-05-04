# ruff: noqa
"""
Vacuum v6 — v5 + per-step world-PG _all_gather_base.

Adds the missing piece: between MoE layers per step, run scalar all_gather
on the WORLD process group (16 ranks) — exactly mimicking what
`accelerator.gather_for_metrics` does in compute_loss.

This is the LAST untested NCCL pattern. The trace dump from the real failing
run showed the hang at this exact collective: `_all_gather_base [1]→[16]`
on PG 0 (world group).
"""

import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from transformers.integrations.sonicmoe import _load_sonic_kernel, sonicmoe_experts_forward
from transformers.integrations.tensor_parallel import _AllReduceBackward, _AllReduceForward


class FakeMoeModule(nn.Module):
    def __init__(self, num_local_experts, hidden, intermediate, hidden_act="silu"):
        super().__init__()
        self.num_experts = num_local_experts
        self.has_gate = True
        self.has_bias = False
        self.is_transposed = False
        self.is_concatenated = True
        self.gate_up_proj = nn.Parameter(
            torch.randn(num_local_experts, 2 * intermediate, hidden, dtype=torch.bfloat16, device="cuda") * 0.01
        )
        self.down_proj = nn.Parameter(
            torch.randn(num_local_experts, hidden, intermediate, dtype=torch.bfloat16, device="cuda") * 0.01
        )

        class _Cfg:
            pass

        self.config = _Cfg()
        self.config.hidden_act = hidden_act


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
    NUM_LOCAL_EXPERTS = NUM_EXPERTS // EP_SIZE
    NUM_LAYERS = 48
    SEQ_LEN = 65536

    ep_group_id = rank // EP_SIZE
    ep_rank_in_group = rank % EP_SIZE
    ep_groups = []
    for gid in range(world_size // EP_SIZE):
        ranks_in_group = list(range(gid * EP_SIZE, (gid + 1) * EP_SIZE))
        g = dist.new_group(ranks=ranks_in_group)
        ep_groups.append(g)
    my_ep_pg = ep_groups[ep_group_id]

    dp_groups = []
    for er in range(EP_SIZE):
        ranks_in_group = [er + g * EP_SIZE for g in range(world_size // EP_SIZE)]
        g = dist.new_group(ranks=ranks_in_group)
        dp_groups.append(g)
    my_dp_pg = dp_groups[ep_rank_in_group]

    if rank == 0:
        print(f"v6: world={world_size} EP={EP_SIZE} SEQ_LEN={SEQ_LEN} NUM_LAYERS={NUM_LAYERS}", flush=True)
        print("  + REAL sonicmoe + grad_ckpt + DP grad reducer + WORLD-PG scalar gather (5×/step)", flush=True)
        print("Pre-loading sonic-moe kernel...", flush=True)

    _load_sonic_kernel()

    if rank == 0:
        print("Sonic kernel loaded.", flush=True)

    class EpMeshShim:
        def __init__(self, group, size):
            self._group = group
            self._size = size

        def get_group(self):
            return self._group

        def size(self):
            return self._size

    ep_mesh = EpMeshShim(my_ep_pg, EP_SIZE)

    moe_modules = [FakeMoeModule(NUM_LOCAL_EXPERTS, HIDDEN, INTERMEDIATE).cuda(local_rank) for _ in range(NUM_LAYERS)]

    def fake_moe_layer(hidden, moe_mod):
        top_k_global = torch.randint(0, NUM_EXPERTS, (SEQ_LEN, NUM_TOP_K), device=f"cuda:{local_rank}")
        top_k_weights = torch.softmax(
            torch.randn(SEQ_LEN, NUM_TOP_K, dtype=torch.bfloat16, device=f"cuda:{local_rank}"), dim=-1
        )
        owner_rank = top_k_global // NUM_LOCAL_EXPERTS
        non_local_mask = owner_rank != ep_rank_in_group
        top_k_local = torch.fmod(top_k_global, NUM_LOCAL_EXPERTS)
        top_k_local = torch.where(non_local_mask, NUM_LOCAL_EXPERTS, top_k_local)
        top_k_weights_masked = top_k_weights.masked_fill(non_local_mask, 0.0)

        h = _AllReduceBackward.apply(hidden, ep_mesh)
        w = _AllReduceBackward.apply(top_k_weights_masked, ep_mesh)
        out = sonicmoe_experts_forward(moe_mod, h, top_k_local, w)
        out = _AllReduceForward.apply(out, ep_mesh)
        return out

    def world_pg_gather_for_metrics():
        """Mimic accelerator.gather_for_metrics — _all_gather_base [1]→[16]."""
        scalar = torch.tensor([float(rank)], dtype=torch.float32, device=f"cuda:{local_rank}")
        out = torch.empty(world_size, dtype=torch.float32, device=f"cuda:{local_rank}")
        dist.all_gather_into_tensor(out, scalar, group=dist.group.WORLD)

    all_params = [p for mod in moe_modules for p in mod.parameters()]
    opt = torch.optim.AdamW(all_params, lr=1e-5)

    NUM_STEPS = 30
    if rank == 0:
        print(
            f"Running {NUM_STEPS} steps × ({NUM_LAYERS} sonicmoe layers + 5 world-PG gathers) + DP reducer", flush=True
        )

    t0 = time.time()
    for step in range(NUM_STEPS):
        opt.zero_grad()
        hidden = torch.randn(SEQ_LEN, HIDDEN, dtype=torch.bfloat16, device=f"cuda:{local_rank}", requires_grad=True)

        for layer_idx in range(NUM_LAYERS):
            hidden = checkpoint(fake_moe_layer, hidden, moe_modules[layer_idx], use_reentrant=False)

        loss = hidden.float().sum()

        # Mimic compute_loss → 5 world-PG gather_for_metrics (chunked_nll path)
        for _ in range(5):
            world_pg_gather_for_metrics()

        loss.backward()

        # DS-Z2-style DP grad reducer
        for p in all_params:
            if p.grad is not None:
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
        print(f"\n*** {NUM_STEPS} steps completed (vacuum v6 — REAL sonicmoe + WORLD-PG gather) ***", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
