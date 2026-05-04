# ruff: noqa
import os

import torch
import torch.distributed as dist


dist.init_process_group("nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()
local_rank = int(os.environ.get("LOCAL_RANK", 0))
hostname = os.uname().nodename.split(".")[0]

torch.cuda.set_device(local_rank)

# Flat mesh
flat_mesh = dist.init_device_mesh("cuda", (world_size,))
flat_group = flat_mesh.get_group()
flat_ranks = dist.get_process_group_ranks(flat_group)

if rank == 0:
    print(f"=== Flat mesh ({world_size},) ===")
    print(f"  All ranks in one group: {flat_ranks}")
    print(f"  Size: {flat_mesh.size()}")
    print(f"  EP would shard 128 experts across {flat_mesh.size()} GPUs = {128 // flat_mesh.size()} per GPU")
    print()

dist.barrier()

# 2D mesh
mesh_2d = dist.init_device_mesh("cuda", (world_size // 8, 8), mesh_dim_names=("dp", "tp"))
tp_mesh = mesh_2d["tp"]
dp_mesh = mesh_2d["dp"]

tp_ranks = dist.get_process_group_ranks(tp_mesh.get_group())
dp_ranks = dist.get_process_group_ranks(dp_mesh.get_group())

print(
    f"Rank {rank:2d} (node={hostname[-3:]}, gpu={local_rank}): "
    f"tp_group={tp_ranks}, dp_group={dp_ranks}, "
    f"tp_local_rank={tp_mesh.get_local_rank()}, dp_local_rank={dp_mesh.get_local_rank()}"
)

dist.barrier()
if rank == 0:
    print()
    print(f"=== 2D mesh ({world_size // 8}, 8) dp × tp ===")
    print("  TP groups: 8 GPUs each (intra-node)")
    print(f"  DP groups: {world_size // 8} GPUs each (one per node)")
    print(f"  EP uses tp_mesh → {tp_mesh.size()} GPUs = {128 // tp_mesh.size()} experts per GPU")
    print(f"  FSDP uses dp_mesh → shards across {dp_mesh.size()} nodes")

dist.destroy_process_group()
