# ruff: noqa
"""Find which layer produces NaN with expert-only EP."""

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

# Hook every decoder layer to track where NaN starts
for i, layer in enumerate(model.model.layers):

    def make_hook(layer_idx):
        def hook(mod, input, output):
            # output is (hidden_states, ...) tuple
            hs = output[0] if isinstance(output, tuple) else output
            has_nan = hs.isnan().any().item()
            mx = hs.abs().max().item()
            if rank == 0 and (has_nan or layer_idx < 3 or layer_idx % 10 == 0):
                print(f"  Layer {layer_idx:2d}: nan={has_nan} max={mx:.4f}")
            if rank == 0 and has_nan and layer_idx > 0:
                # Check which sublayer introduced NaN
                pass

        return hook

    layer.register_forward_hook(make_hook(i))

    # Also hook the MoE block specifically
    if hasattr(layer.mlp, "experts"):

        def make_moe_hook(layer_idx):
            def hook(mod, input, output):
                has_nan = output.isnan().any().item()
                mx = output.abs().max().item()
                if rank == 0 and (has_nan or layer_idx < 3):
                    print(f"    MoE {layer_idx:2d}: nan={has_nan} max={mx:.4f}")

            return hook

        layer.mlp.experts.register_forward_hook(make_moe_hook(i))

if rank == 0:
    print(f"EP={world_size}, expert-only plan")
    print(f"ep_plan keys: {list(model.tp_plan.keys())}")

x = torch.tensor([[1, 2, 3]], device="cuda")
with torch.no_grad():
    out = model(x, use_cache=False)

if rank == 0:
    print(f"\nlogits nan={out.logits.isnan().any()}")

dist.destroy_process_group()
