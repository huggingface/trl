# ruff: noqa
"""
Visualize how DP, EP, CP meshes overlap for a given configuration.
Shows which GPUs participate in which collective operations.

Usage:
    torchrun --nproc_per_node=8 benchmark/test_mesh_layout.py          # 1 node
    # or via srun for multi-node
"""

import os
import socket

import torch
import torch.distributed as dist
from transformers import AutoConfig


dist.init_process_group("nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()
local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)
hostname = socket.gethostname().split(".")[0]

# Configuration (matching our benchmark: EP=world_size, CP from accelerate)
NUM_EXPERTS = 128
TOP_K = 8
CP_SIZE = int(os.environ.get("CP_SIZE", "1"))
EP_SIZE = world_size  # expert-only EP: flat mesh, all GPUs

config = AutoConfig.from_pretrained("Qwen/Qwen3-30B-A3B")

# ============================================================
# Mesh 1: EP mesh (created by SFT trainer, used by distribute_model)
# ============================================================
ep_mesh = dist.init_device_mesh("cuda", (world_size,))
ep_rank = ep_mesh.get_local_rank()
num_local_experts = NUM_EXPERTS // EP_SIZE
my_expert_range = f"[{ep_rank * num_local_experts}, {(ep_rank + 1) * num_local_experts - 1}]"

# ============================================================
# Mesh 2: Accelerate mesh (created by accelerate for FSDP2 + CP)
# ============================================================
if CP_SIZE > 1:
    dp_size = world_size // CP_SIZE
    accel_mesh = dist.init_device_mesh("cuda", (dp_size, CP_SIZE), mesh_dim_names=("dp", "cp"))
    dp_rank = accel_mesh["dp"].get_local_rank()
    cp_rank = accel_mesh["cp"].get_local_rank()
    dp_group_ranks = accel_mesh["dp"].mesh.tolist()  # which GPUs share my DP group
    cp_group_ranks = accel_mesh["cp"].mesh.tolist()
else:
    dp_size = world_size
    dp_rank = rank
    cp_rank = 0
    dp_group_ranks = list(range(world_size))
    cp_group_ranks = [rank]

# Gather all info to rank 0
info = {
    "rank": rank,
    "local_rank": local_rank,
    "host": hostname,
    "ep_rank": ep_rank,
    "dp_rank": dp_rank,
    "cp_rank": cp_rank,
    "experts": my_expert_range,
    "num_local_experts": num_local_experts,
}

all_info = [None] * world_size
dist.all_gather_object(all_info, info)

if rank == 0:
    node_map = {}
    for i in all_info:
        node_map.setdefault(i["host"], []).append(i)

    print(f"{'=' * 80}")
    print(f"MESH LAYOUT: {world_size} GPUs, EP={EP_SIZE}, CP={CP_SIZE}, DP={dp_size}")
    print(f"Model: Qwen3-30B-A3B ({NUM_EXPERTS} experts, top_k={TOP_K})")
    print(f"Local experts per GPU: {num_local_experts}")
    print(f"{'=' * 80}")
    print()

    # Per-GPU table
    print(f"{'GPU':>4} {'Node':<20} {'EP rank':>8} {'DP rank':>8} {'CP rank':>8} {'Experts':>12}")
    print("-" * 70)
    for i in sorted(all_info, key=lambda x: x["rank"]):
        print(
            f"{i['rank']:>4} {i['host']:<20} {i['ep_rank']:>8} {i['dp_rank']:>8} {i['cp_rank']:>8} {i['experts']:>12}"
        )
    print()

    # Operation breakdown
    print("OPERATIONS AND WHICH GPUS PARTICIPATE:")
    print("-" * 70)
    print()

    # Attention (replicated, no sharding)
    print("1. ATTENTION (replicated weights, no EP/TP sharding)")
    print("   - Each GPU has FULL attention weights (q/k/v/o_proj)")
    print("   - FSDP2 shards attention params across DP group for memory")
    print("   - Forward: FSDP2 all-gathers attention params within DP group")
    if CP_SIZE > 1:
        print(f"   - CP={CP_SIZE}: sequence split across CP pairs, ring-attention")
        # Show CP pairs
        cp_pairs = {}
        for i in all_info:
            key = i["dp_rank"]
            cp_pairs.setdefault(key, []).append(i["rank"])
        print("   - CP groups (ring-attention within each):")
        for dp_r in sorted(cp_pairs.keys()):
            gpus = sorted(cp_pairs[dp_r])
            nodes = [all_info[g]["host"] for g in gpus]
            cross_node = len(set(nodes)) > 1
            print(f"     DP={dp_r}: GPUs {gpus} {'(CROSS-NODE)' if cross_node else '(intra-node)'}")
    else:
        print("   - No CP: each GPU processes full sequence independently")
    print()

    # MoE routing
    print("2. MoE ROUTING (RouterParallel)")
    print(f"   - Router weight replicated on all {world_size} GPUs")
    print(f"   - Each GPU computes FULL router scores for all {NUM_EXPERTS} experts")
    print("   - RouterParallel remaps to local indices + zeros non-local scores")
    print(f"   - Each GPU only computes {num_local_experts} local expert outputs")
    print()

    # Expert computation
    print("3. EXPERT COMPUTATION (grouped_mm on local experts)")
    print(f"   - Expert weights sharded: each GPU has {num_local_experts} of {NUM_EXPERTS} experts")
    for node, gpus in sorted(node_map.items()):
        ranges = [g["experts"] for g in sorted(gpus, key=lambda x: x["rank"])]
        gpu_ids = [g["rank"] for g in sorted(gpus, key=lambda x: x["rank"])]
        print(f"   - {node}: GPUs {gpu_ids}")
        for g, r in zip(gpu_ids, ranges, strict=False):
            print(f"       GPU {g}: experts {r}")
    print()

    # All-reduce for expert outputs
    print("4. EXPERT OUTPUT ALL-REDUCE (MoeTensorParallelExperts)")
    print(f"   - Sums partial expert outputs across ALL {world_size} GPUs")
    print(f"   - EP mesh: flat ({world_size},) — every GPU participates")
    cross_node = len(node_map) > 1
    if cross_node:
        print(f"   - WARNING: all-reduce spans {len(node_map)} nodes (inter-node bandwidth)")
    else:
        print("   - Single node: NVLink bandwidth")
    print()

    # FSDP2
    print("5. FSDP2 (parameter sharding for memory)")
    print(f"   - DP shard size: {dp_size}")
    print(f"   - All-gathers {num_local_experts} local experts before forward (not all {NUM_EXPERTS})")
    print(f"   - Memory per GPU: ~{NUM_EXPERTS // EP_SIZE} expert params + full attention params")
    if CP_SIZE > 1:
        dp_groups = {}
        for i in all_info:
            dp_groups.setdefault(i["cp_rank"], []).append(i["rank"])
        print("   - DP groups (FSDP all-gather/reduce-scatter within each):")
        for cp_r in sorted(dp_groups.keys()):
            gpus = sorted(dp_groups[cp_r])
            nodes = set(all_info[g]["host"] for g in gpus)
            cross = len(nodes) > 1
            print(f"     CP={cp_r}: GPUs {gpus} {'(CROSS-NODE)' if cross else '(intra-node)'}")
    print()

    # Communication summary
    print("COMMUNICATION SUMMARY:")
    print("-" * 70)
    print(f"  EP all-reduce:     all {world_size} GPUs {'(INTER-NODE)' if cross_node else '(intra-node)'}")
    if CP_SIZE > 1:
        print(f"  CP ring-attention: {CP_SIZE} GPUs per group, {world_size // CP_SIZE} groups")
    print(f"  FSDP all-gather:   {dp_size} GPUs per group {'(INTER-NODE)' if cross_node else ''}")
    print(f"  FSDP reduce-scatter: {dp_size} GPUs per group")
    print()

    # Data flow for one token through one layer
    print("DATA FLOW (one MoE layer, one token on GPU 0):")
    print("-" * 70)
    g0 = all_info[0]
    print(f"  1. hidden_states enters layer (shape: [1, {config.hidden_size}])")
    print(f"  2. Attention: GPU {g0['rank']} computes full attention (weights replicated)")
    if CP_SIZE > 1:
        print("     - ring-attention with CP partner GPU(s)")
    print(f"  3. Router: computes scores for all {NUM_EXPERTS} experts")
    print(f"     - RouterParallel: keeps scores for experts {g0['experts']}, zeros rest")
    print(f"  4. Expert forward: computes {num_local_experts} local experts via grouped_mm")
    print(
        f"     - gate_up_proj shape: [{num_local_experts}, {2 * config.moe_intermediate_size}, {config.hidden_size}]"
    )
    print(f"  5. All-reduce: sums expert outputs across {world_size} GPUs → full MoE output")
    print("  6. Residual add → output hidden_states")

dist.destroy_process_group()
