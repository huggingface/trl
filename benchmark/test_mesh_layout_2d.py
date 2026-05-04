# ruff: noqa
"""Visualize 2D EP mesh: EP intra-node, DP inter-node."""

import os
import socket

import torch
import torch.distributed as dist


dist.init_process_group("nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()
local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)
hostname = socket.gethostname().split(".")[0]

EP_SIZE = int(os.environ.get("EP_SIZE", "8"))
NUM_EXPERTS = 128

dp_size = world_size // EP_SIZE
mesh = dist.init_device_mesh("cuda", (dp_size, EP_SIZE), mesh_dim_names=("dp", "tp"))
tp_mesh = mesh["tp"]
dp_mesh = mesh["dp"]

ep_rank = tp_mesh.get_local_rank()
dp_rank = dp_mesh.get_local_rank()
num_local = NUM_EXPERTS // EP_SIZE
my_experts = f"[{ep_rank * num_local}, {(ep_rank + 1) * num_local - 1}]"

info = {
    "rank": rank,
    "local_rank": local_rank,
    "host": hostname,
    "ep_rank": ep_rank,
    "dp_rank": dp_rank,
    "experts": my_experts,
}
all_info = [None] * world_size
dist.all_gather_object(all_info, info)

if rank == 0:
    print(f"{'=' * 70}")
    print(f"2D MESH: {world_size} GPUs, EP={EP_SIZE}, DP={dp_size}")
    print(f"Local experts per GPU: {num_local}")
    print(f"{'=' * 70}")
    print(f"{'GPU':>4} {'Node':<18} {'EP rank':>8} {'DP rank':>8} {'Experts':>14}")
    print("-" * 60)
    for i in sorted(all_info, key=lambda x: x["rank"]):
        print(f"{i['rank']:>4} {i['host']:<18} {i['ep_rank']:>8} {i['dp_rank']:>8} {i['experts']:>14}")

    # Show EP groups (which GPUs do all-reduce together for expert outputs)
    print("\nEP groups (expert all-reduce within each):")
    ep_groups = {}
    for i in all_info:
        ep_groups.setdefault(i["dp_rank"], []).append(i["rank"])
    for dp_r in sorted(ep_groups.keys()):
        gpus = sorted(ep_groups[dp_r])
        nodes = set(all_info[g]["host"] for g in gpus)
        cross = len(nodes) > 1
        print(f"  DP={dp_r}: GPUs {gpus} {'(CROSS-NODE)' if cross else '(intra-node)'}")

    # Show DP groups (which GPUs do FSDP all-gather/gradient sync)
    print("\nDP groups (FSDP gradient sync between replicas):")
    dp_groups = {}
    for i in all_info:
        dp_groups.setdefault(i["ep_rank"], []).append(i["rank"])
    for ep_r in sorted(dp_groups.keys()):
        gpus = sorted(dp_groups[ep_r])
        nodes = set(all_info[g]["host"] for g in gpus)
        cross = len(nodes) > 1
        exp = all_info[gpus[0]]["experts"]
        print(f"  EP={ep_r} (experts {exp}): GPUs {gpus} {'(CROSS-NODE)' if cross else '(intra-node)'}")

    print(
        f"\nKey insight: EP all-reduce stays {'INTRA-NODE (NVLink)' if not any(len(set(all_info[g]['host'] for g in sorted(ep_groups[dp_r]))) > 1 for dp_r in ep_groups) else 'CROSS-NODE'}"
    )
    print(
        f"DP gradient sync goes {'CROSS-NODE' if dp_size > 1 and len(set(i['host'] for i in all_info)) > 1 else 'intra-node'}"
    )

dist.destroy_process_group()
