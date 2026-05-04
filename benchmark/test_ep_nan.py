# ruff: noqa
import os

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM
from transformers.distributed.configuration_utils import DistributedConfig


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
)

x = torch.tensor([[1, 2, 3]], device="cuda")
with torch.no_grad():
    out = model(x, use_cache=False)

if rank == 0:
    print(f"EP={world_size}")
    print(f"logits: {out.logits[0, -1, :5]}")
    print(f"nan: {out.logits.isnan().any()}")
    print(f"inf: {out.logits.isinf().any()}")

    # Check intermediate: run just the first layer
    emb = model.model.embed_tokens(x)
    print(f"embed output nan: {emb.isnan().any()}")

dist.destroy_process_group()
