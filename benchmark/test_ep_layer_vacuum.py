# ruff: noqa
"""
Vacuum repro for the EP-multi-node hang.

Builds ONE Qwen3MoE MoE block, applies the same EP TP plan that the failing
training runs use, then sweeps sequence length running fwd+bwd in a loop.

Tests two things:
1. Does the hang reproduce with JUST one MoE layer + EP TP (no model, no DS-Z2,
   no FSDP, no dataloader, no dataset)?
2. At which sequence length does it start hanging?

Usage (2 nodes, 8 GPUs each):
    srun --nodes=2 --ntasks-per-node=1 --gres=gpu:h100:8 ... bash -c '
      torchrun --nnodes=2 --nproc-per-node=8 --node-rank=$SLURM_PROCID \\
               --master-addr=$MASTER_ADDR --master-port=29500 \\
               benchmark/test_ep_layer_vacuum.py
    '
"""

import os
import time

import torch
import torch.distributed as dist
from transformers import AutoConfig
from transformers.integrations.tensor_parallel import (
    _AllReduceBackward,
    _AllReduceForward,
)
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeSparseMoeBlock,
)


def log0(msg):
    if int(os.environ.get("RANK", 0)) == 0:
        print(msg, flush=True)


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    EP_SIZE = 8
    HIDDEN = 2048

    # Build EP groups: ranks split into intra-node groups of EP_SIZE.
    # ranks 0-7 = group A (node 0), 8-15 = group B (node 1)
    ep_group_id = rank // EP_SIZE
    groups = []
    for gid in range(world_size // EP_SIZE):
        ranks_in_group = list(range(gid * EP_SIZE, (gid + 1) * EP_SIZE))
        g = dist.new_group(ranks=ranks_in_group)
        groups.append(g)
    my_ep_pg = groups[ep_group_id]

    # Mesh shim that exposes get_group() / size() for the autograd Functions.
    class EpMeshShim:
        def __init__(self, group, size):
            self._group = group
            self._size = size

        def get_group(self):
            return self._group

        def size(self):
            return self._size

    ep_mesh = EpMeshShim(my_ep_pg, EP_SIZE)
    log0(f"world_size={world_size} EP_SIZE={EP_SIZE} my_ep_group_id={ep_group_id}")

    # ---- Build a single Qwen3 MoE block ----
    config = AutoConfig.from_pretrained("Qwen/Qwen3-30B-A3B")
    NUM_EXPERTS = config.num_local_experts
    NUM_TOP_K = config.num_experts_per_tok
    NUM_LOCAL_EXPERTS = NUM_EXPERTS // EP_SIZE  # =16 for 128/8

    # We're going to mimic the EP-sharded experts manually rather than load the full model.
    # Build one block on this rank holding ONLY the local experts (consistent with EP).
    # We'll use the GroupedGemmParallel-style sharding: each rank sees num_local_experts experts.
    log0(f"num_experts={NUM_EXPERTS}, num_top_k={NUM_TOP_K}, num_local={NUM_LOCAL_EXPERTS}")

    # Build the block — Qwen3MoeSparseMoeBlock expects full num_experts.
    # We'll then manually shard the experts to local-only by slicing the param.
    # Simpler: build it with num_experts=NUM_LOCAL_EXPERTS so it computes only local experts,
    # AND we manually run the EP sentinel handling + all_reduce.
    sub_config = AutoConfig.from_pretrained("Qwen/Qwen3-30B-A3B")
    # Don't change num_experts; the router still produces global indices, sharding is via masking.
    block = Qwen3MoeSparseMoeBlock(sub_config).bfloat16().cuda(local_rank)
    block.train()  # enable grad
    block.num_experts = NUM_EXPERTS  # ensure we know the global count

    # EP sentinel: for ranks not owning a given expert, the global expert id is replaced with
    # `num_local_experts` (the sentinel). RouterParallel does this; we replicate it in dummy data.

    log0("\nSweeping sequence length — fwd+bwd in a loop, mimicking 1 MoE layer per iter:")
    log0("  iters per shape = 200 (~step 1 of training × 4 layers worth of fwd-only EP all-reduces)")

    # Track cumulative bytes transferred per all-reduce call.
    for SEQ_LEN in [16384, 32768, 49152, 65536, 98304]:
        if rank == 0:
            ar_bytes = SEQ_LEN * HIDDEN * 2  # bf16
            print(f"\n=== SEQ_LEN={SEQ_LEN}  (each EP all-reduce: {ar_bytes / 1e6:.1f} MB) ===", flush=True)
        torch.cuda.empty_cache()
        dist.barrier()
        t0 = time.time()
        max_mem = 0

        for it in range(200):
            # ---- Build dummy input tensors (mimics what the model produces at this point) ----
            # hidden_states: (seq, hidden)
            hidden = torch.randn(
                SEQ_LEN, HIDDEN, dtype=torch.bfloat16, device=f"cuda:{local_rank}", requires_grad=True
            )
            # top_k_index: (seq, top_k) — global expert ids in [0, num_experts).
            # SET LOTS TO SENTINEL (simulating heavy non-local routing).
            # Each rank "owns" experts [ep_group_id*NUM_LOCAL .. (ep_group_id+1)*NUM_LOCAL).
            # Random global ids; ones outside local become sentinels.
            top_k_global = torch.randint(0, NUM_EXPERTS, (SEQ_LEN, NUM_TOP_K), device=f"cuda:{local_rank}")
            # weights: (seq, top_k) bf16
            top_k_weights = torch.softmax(
                torch.randn(SEQ_LEN, NUM_TOP_K, dtype=torch.bfloat16, device=f"cuda:{local_rank}"),
                dim=-1,
            )
            top_k_weights.requires_grad_(True)

            # Apply EP sentinel handling (this is what RouterParallel._prepare_output_fn does):
            owner_rank = top_k_global // NUM_LOCAL_EXPERTS
            ep_local_rank = ep_group_id  # within EP group, this rank's local-rank == ep_group_id (intra-group rank)
            # Actually within EP group, local_rank = (rank % EP_SIZE). owner_rank should match that.
            ep_local_rank_in_group = rank % EP_SIZE  # 0..7 within EP group
            non_local_mask = owner_rank != ep_local_rank_in_group
            top_k_local_id = torch.fmod(top_k_global, NUM_LOCAL_EXPERTS)
            top_k_local_id = torch.where(non_local_mask, NUM_LOCAL_EXPERTS, top_k_local_id)  # SENTINEL!
            top_k_weights_masked = top_k_weights.masked_fill(non_local_mask, 0.0)

            # ---- Mimic per-MoE-layer EP collective pattern ----
            # 1) all_reduce_backward on hidden_states (no-op fwd, all-reduce in bwd)
            hidden_in = _AllReduceBackward.apply(hidden, ep_mesh)
            # 2) all_reduce_backward on top_k_weights
            tkw_in = _AllReduceBackward.apply(top_k_weights_masked, ep_mesh)

            # 3) Fake MoE compute (simple matmul) — the kernel itself has been ruled out as the
            #    cause (same hang at 64k regardless of sonicmoe vs grouped_mm), so we use a cheap
            #    proxy that has the same I/O contract.
            #    Output shape: (seq, hidden) — partial expert outputs per rank.
            output_partial = hidden_in * tkw_in.sum(dim=-1, keepdim=True).to(hidden_in.dtype) * 0.1

            # 4) all_reduce_forward on output (THE failing collective)
            output = _AllReduceForward.apply(output_partial, ep_mesh)

            # ---- Backward ----
            loss = output.float().sum()
            loss.backward()

            # ---- Cleanup ----
            del hidden, top_k_global, top_k_weights, top_k_weights_masked
            del hidden_in, tkw_in, output_partial, output, loss

            cur_mem = torch.cuda.memory_allocated(local_rank)
            if cur_mem > max_mem:
                max_mem = cur_mem

            if it % 25 == 0 and rank == 0:
                elapsed = time.time() - t0
                print(
                    f"  [{elapsed:5.1f}s] iter {it:3d}/200  alloc={cur_mem / 1e9:.2f}GB peak={max_mem / 1e9:.2f}GB",
                    flush=True,
                )

        elapsed = time.time() - t0
        log0(f"  DONE SEQ_LEN={SEQ_LEN}: 200 iters in {elapsed:.1f}s, peak alloc {max_mem / 1e9:.2f}GB")

    log0(
        "\n*** ALL SEQ LENGTHS COMPLETED — bug is NOT in (autograd-wrapped EP all-reduce + sentinel masks) at this iteration count ***"
    )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
