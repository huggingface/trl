# ruff: noqa: T201
"""Compare grouped_mm vs batched_mm with EP to isolate the NaN source."""

import os
import sys

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM
from transformers.distributed.configuration_utils import DistributedConfig


impl = sys.argv[1] if len(sys.argv) > 1 else "grouped_mm"

dist.init_process_group("nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()
torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

mesh = dist.init_device_mesh("cuda", (world_size,))
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-30B-A3B",
    dtype=torch.bfloat16,
    distributed_config=DistributedConfig(enable_expert_parallel=True),
    device_mesh=mesh,
    experts_implementation=impl,
)

if rank == 0:
    print(f"EP={world_size} impl={impl} num_local_experts={model.model.layers[0].mlp.experts.num_experts}")

x = torch.tensor([[1, 2, 3]], device="cuda")
with torch.no_grad():
    out = model(x, use_cache=False)

if rank == 0:
    has_nan = out.logits.isnan().any().item()
    print(f"nan={has_nan} logits={out.logits[0, -1, :5].tolist()}")

dist.destroy_process_group()
