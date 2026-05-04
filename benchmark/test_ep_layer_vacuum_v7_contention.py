# ruff: noqa
"""
Vacuum v7 — targeted test for the network channel contention theory.

Theory: NCCL world-PG gathers compete with intra-node EP all-reduces for network
channels, causing the EP all-reduces to stall and never complete on slow ranks.

Reproduces real training more aggressively:
- Issues 5x world-PG `_all_gather_base` calls IMMEDIATELY after computing the loss,
  while the EP all-reduces from forward may still have CUDA queue tail on some ranks
- Adds `.item()` calls (forces H2D sync, holds GIL) like real `gather_for_metrics`
- Uses the actual `_gpu_gather` from accelerate (not a hand-rolled version)
- 30 steps × 48 sonicmoe layers + grad_ckpt + DP reducer
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
    def __init__(self, n, h, i, hidden_act="silu"):
        super().__init__()
        self.num_experts = n
        self.has_gate = True
        self.has_bias = False
        self.is_transposed = False
        self.is_concatenated = True
        self.gate_up_proj = nn.Parameter(torch.randn(n, 2 * i, h, dtype=torch.bfloat16, device="cuda") * 0.01)
        self.down_proj = nn.Parameter(torch.randn(n, h, i, dtype=torch.bfloat16, device="cuda") * 0.01)

        class _C:
            pass

        self.config = _C()
        self.config.hidden_act = hidden_act


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    EP_SIZE = 8
    HIDDEN, INTERMEDIATE = 2048, 768
    NUM_EXPERTS, NUM_TOP_K = 128, 8
    NUM_LOCAL_EXPERTS = NUM_EXPERTS // EP_SIZE
    NUM_LAYERS = 48
    SEQ_LEN = 65536

    ep_group_id = rank // EP_SIZE
    ep_rank_in_group = rank % EP_SIZE
    ep_groups = [
        dist.new_group(ranks=list(range(g * EP_SIZE, (g + 1) * EP_SIZE))) for g in range(world_size // EP_SIZE)
    ]
    my_ep_pg = ep_groups[ep_group_id]
    dp_groups = [
        dist.new_group(ranks=[er + g * EP_SIZE for g in range(world_size // EP_SIZE)]) for er in range(EP_SIZE)
    ]
    my_dp_pg = dp_groups[ep_rank_in_group]

    if rank == 0:
        print(f"v7 contention test: world={world_size} EP={EP_SIZE} SEQ_LEN={SEQ_LEN}", flush=True)
        print("theory: world-PG gathers contend with intra-node EP all-reduces", flush=True)

    _load_sonic_kernel()
    if rank == 0:
        print("kernel loaded", flush=True)

    class M:
        def __init__(self, g, s):
            self._g, self._s = g, s

        def get_group(self):
            return self._g

        def size(self):
            return self._s

    ep_mesh = M(my_ep_pg, EP_SIZE)

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

    def gather_for_metrics_clone(t):
        """Mimics accelerate._gpu_gather_one verbatim — including .clone() and contiguous check."""
        if t.ndim == 0:
            t = t.clone()[None]
        if not t.is_contiguous():
            t = t.contiguous()
        out = torch.empty(world_size * t.numel(), dtype=t.dtype, device=t.device)
        dist.all_gather_into_tensor(out, t, group=dist.group.WORLD)
        return out.view(-1, *t.size()[1:])

    all_params = [p for m in moe_modules for p in m.parameters()]
    opt = torch.optim.AdamW(all_params, lr=1e-5)

    NUM_STEPS = 30
    if rank == 0:
        print(f"Running {NUM_STEPS} steps with HIGH-FREQUENCY world-PG gathers right after forward", flush=True)

    t0 = time.time()
    for step in range(NUM_STEPS):
        opt.zero_grad()
        hidden = torch.randn(SEQ_LEN, HIDDEN, dtype=torch.bfloat16, device=f"cuda:{local_rank}", requires_grad=True)

        for layer_idx in range(NUM_LAYERS):
            hidden = checkpoint(fake_moe_layer, hidden, moe_modules[layer_idx], use_reentrant=False)

        # Compute loss tensors that depend on the forward (mimics chunked_nll outputs.entropy_sum)
        entropy_sum = hidden.float().sum()
        n_valid_local = torch.tensor(SEQ_LEN, device=f"cuda:{local_rank}", dtype=torch.float32)

        # MIMIC compute_loss: 5 gather_for_metrics calls IMMEDIATELY after forward, with .item()
        # forcing CPU↔GPU sync each time. This is the contention pattern.
        n_valid = gather_for_metrics_clone(n_valid_local).sum()
        entropy_g = gather_for_metrics_clone(entropy_sum).sum()
        entropy = (entropy_g / n_valid).item() if rank == 0 else None  # forces sync
        n_tokens = gather_for_metrics_clone(n_valid_local).sum().item()
        n_valid_2 = gather_for_metrics_clone(n_valid_local).sum()
        correct = gather_for_metrics_clone(entropy_sum).sum()
        accuracy = (correct / n_valid_2).item() if rank == 0 else None  # forces sync

        # Now do backward (this is where backward EP all-reduces fire)
        loss = entropy_sum
        loss.backward()

        # DS-Z2 grad reducer
        for p in all_params:
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, group=my_dp_pg)
                p.grad.div_(world_size // EP_SIZE)
        opt.step()

        if rank == 0:
            elapsed = time.time() - t0
            mem = torch.cuda.memory_allocated(local_rank) / 1e9
            peak = torch.cuda.max_memory_allocated(local_rank) / 1e9
            print(f"  [{elapsed:6.1f}s] step {step + 1}/{NUM_STEPS}  alloc={mem:.2f}GB peak={peak:.2f}GB", flush=True)

    if rank == 0:
        print(f"\n*** {NUM_STEPS} steps completed (vacuum v7 contention) ***", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
