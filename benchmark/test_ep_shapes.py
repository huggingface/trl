# ruff: noqa
"""Test to check the shapes flowing through RouterParallel and expert forward."""

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

layer0_gate = model.model.layers[0].mlp.gate
layer0_experts = model.model.layers[0].mlp.experts


# Hook on the gate to see router output (after RouterParallel hooks)
def gate_hook(mod, input, output):
    if rank == 0:
        router_logits, router_scores, router_indices = output
        print(
            f"[Gate output] logits:{router_logits.shape} scores:{router_scores.shape} indices:{router_indices.shape}"
        )
        print(f"  indices[0]: {router_indices[0].tolist()}")
        print(f"  scores[0,:12]: {[round(x, 4) for x in router_scores[0, :12].float().tolist()]}")


layer0_gate.register_forward_hook(gate_hook)


# Hook on experts to see input shapes (after MoeTensorParallelExperts._prepare_input_fn)
def experts_pre_hook(mod, input):
    if rank == 0:
        hidden_states, top_k_index, top_k_weights = input
        print(f"[Experts input] hs:{hidden_states.shape} idx:{top_k_index.shape} wts:{top_k_weights.shape}")
        print(f"  idx[0]: {top_k_index[0].tolist()}")
        print(f"  wts[0,:12]: {[round(x, 4) for x in top_k_weights[0, :12].float().tolist()]}")
        print(f"  num_experts={mod.num_experts}")


layer0_experts.register_forward_pre_hook(experts_pre_hook)


def experts_post_hook(mod, input, output):
    if rank == 0:
        print(f"[Experts output] shape:{output.shape} nan:{output.isnan().any()} max:{output.abs().max():.4f}")


layer0_experts.register_forward_hook(experts_post_hook)

if rank == 0:
    print(f"EP={world_size}")
    print(f"gate_up_proj: {list(layer0_experts.gate_up_proj.shape)}")
    print(f"ep_plan keys: {list(model.tp_plan.keys())}")
    print()

x = torch.tensor([[1, 2, 3]], device="cuda")
try:
    with torch.no_grad():
        out = model(x, use_cache=False)
    if rank == 0:
        print(f"\nlogits nan={out.logits.isnan().any()}")
        print(f"logits: {out.logits[0, -1, :5]}")
except Exception as e:
    if rank == 0:
        import traceback

        print(f"FAILED: {e}")
        traceback.print_exc()

dist.destroy_process_group()
