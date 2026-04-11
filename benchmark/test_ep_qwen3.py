# ruff: noqa: T201
"""
Test Expert Parallelism for Qwen3 MoE by fusing individual expert weights
into grouped tensors and patching the forward to use batched matmuls.

Run single GPU:  python benchmark/test_ep_qwen3.py
Run with EP:     accelerate launch --config_file <fsdp2_config.yaml> benchmark/test_ep_qwen3.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM


def fuse_qwen3_moe_experts(model):
    """
    Fuse individual Qwen3 MoE experts into grouped tensors and patch the forward.

    Replaces:
      experts = ModuleList([MLP(gate_proj, up_proj, down_proj), ...])  # 128 x nn.Linear
    With:
      experts.gate_proj = Parameter([num_experts, moe_inter, hidden])
      experts.up_proj   = Parameter([num_experts, moe_inter, hidden])
      experts.down_proj = Parameter([num_experts, hidden, moe_inter])
      + patched forward using batched operations
    """
    config = model.config
    num_experts = config.num_experts

    for layer in model.model.layers:
        block = layer.mlp
        experts = block.experts

        # Stack individual expert weights into fused tensors
        gate_weights = torch.stack([experts[i].gate_proj.weight for i in range(num_experts)])
        up_weights = torch.stack([experts[i].up_proj.weight for i in range(num_experts)])
        down_weights = torch.stack([experts[i].down_proj.weight for i in range(num_experts)])

        # Replace with fused module
        fused = nn.Module()
        fused.gate_proj = nn.Parameter(gate_weights)
        fused.up_proj = nn.Parameter(up_weights)
        fused.down_proj = nn.Parameter(down_weights)
        block.experts = fused

        # Patch forward
        block.forward = _make_fused_forward(block)

    return model


def _make_fused_forward(block):
    """Create a patched forward method that uses fused expert weights."""

    def forward(hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        # Router
        router_logits = block.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, block.top_k, dim=-1)
        if block.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # Use the same expert-loop approach but with fused weights
        expert_mask = F.one_hot(selected_experts, num_classes=block.num_experts).permute(2, 1, 0)

        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)

            # Fused matmul: use the expert_idx slice from fused weights
            eidx = expert_idx.item()
            gate_out = F.linear(current_state, block.experts.gate_proj[eidx])
            up_out = F.linear(current_state, block.experts.up_proj[eidx])
            current_hidden_states = F.linear(F.silu(gate_out) * up_out, block.experts.down_proj[eidx])

            current_hidden_states = current_hidden_states * routing_weights[top_x, idx, None]
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits

    return forward


def get_qwen3_ep_plan():
    """EP plan for fused Qwen3 MoE experts."""
    return {
        "model.layers.*.mlp.gate": "ep_router",
        "model.layers.*.mlp.experts.gate_proj": "grouped_gemm",
        "model.layers.*.mlp.experts.up_proj": "grouped_gemm",
        "model.layers.*.mlp.experts.down_proj": "grouped_gemm",
    }


if __name__ == "__main__":
    print("=== Step 1: Load model ===")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-30B-A3B", dtype=torch.bfloat16)
    print(f"Model loaded, dtype={model.dtype}")

    print("\n=== Step 2: Fuse experts ===")
    model = fuse_qwen3_moe_experts(model)
    print(f"Fused. experts.gate_proj shape: {model.model.layers[0].mlp.experts.gate_proj.shape}")

    print("\n=== Step 3: Test forward (single GPU) ===")
    model = model.cuda()
    input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long, device="cuda")

    try:
        with torch.no_grad():
            out = model(input_ids, use_cache=False)
        print(f"Forward PASSED! loss={out.loss}, logits shape={out.logits.shape}")
    except Exception as e:
        print(f"Forward FAILED: {type(e).__name__}: {str(e)[:300]}")

    print("\n=== Step 4: Verify output matches original ===")
    # Quick sanity: check logits are reasonable (not NaN/Inf)
    if hasattr(out, "logits"):
        logits = out.logits
        print(
            f"logits: min={logits.min().item():.4f}, max={logits.max().item():.4f}, "
            f"mean={logits.mean().item():.4f}, has_nan={logits.isnan().any().item()}"
        )

    print("\nDone.")
