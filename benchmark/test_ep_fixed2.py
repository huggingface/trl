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

# EP=2 — should work cleanly (128/2=64 experts, 4 KV heads / no split)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-30B-A3B",
    dtype=torch.bfloat16,
    distributed_config=DistributedConfig(enable_expert_parallel=True),
    device_mesh=mesh,
)

if rank == 0:
    attn = model.model.layers[0].self_attn
    experts = model.model.layers[0].mlp.experts
    print(f"EP={world_size}")
    print(f"q_proj: {list(attn.q_proj.weight.shape)}")
    print(f"k_proj: {list(attn.k_proj.weight.shape)}")
    print(f"experts.gate_up_proj: {list(experts.gate_up_proj.shape)}")

x = torch.tensor([[1, 2, 3]], device="cuda")
with torch.no_grad():
    out = model(x, use_cache=False)
if rank == 0:
    print(f"Forward: logits={out.logits.shape}")
    print(f"logits: {out.logits[0, -1, :5]}")
    print(f"has_nan: {out.logits.isnan().any()}")

dist.destroy_process_group()
