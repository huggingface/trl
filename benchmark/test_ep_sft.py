# ruff: noqa: T201
"""
Test SFT training with fused Qwen3 MoE experts + FSDP2.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM

from trl import SFTConfig, SFTTrainer


def fuse_qwen3_moe_experts(model):
    """Fuse individual experts into grouped tensors and patch forward."""
    num_experts = model.config.num_experts

    for layer in model.model.layers:
        block = layer.mlp
        experts = block.experts

        gate_weights = torch.stack([experts[i].gate_proj.weight for i in range(num_experts)])
        up_weights = torch.stack([experts[i].up_proj.weight for i in range(num_experts)])
        down_weights = torch.stack([experts[i].down_proj.weight for i in range(num_experts)])

        fused = nn.Module()
        fused.gate_proj = nn.Parameter(gate_weights)
        fused.up_proj = nn.Parameter(up_weights)
        fused.down_proj = nn.Parameter(down_weights)
        block.experts = fused
        block.forward = _make_fused_forward(block)

    return model


def _make_fused_forward(block):
    def forward(hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        router_logits = block.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, block.top_k, dim=-1)
        if block.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        expert_mask = F.one_hot(selected_experts, num_classes=block.num_experts).permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)

            eidx = expert_idx.item()
            gate_out = F.linear(current_state, block.experts.gate_proj[eidx])
            up_out = F.linear(current_state, block.experts.up_proj[eidx])
            current_hidden_states = F.linear(F.silu(gate_out) * up_out, block.experts.down_proj[eidx])
            current_hidden_states = current_hidden_states * routing_weights[top_x, idx, None]
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits

    return forward


print("Loading model...")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-30B-A3B", dtype=torch.bfloat16)
print("Fusing experts...")
model = fuse_qwen3_moe_experts(model)

dataset = load_dataset("THUDM/LongAlign-10k", split="train")

training_args = SFTConfig(
    output_dir="/tmp/bench_30b_ep",
    max_steps=5,
    per_device_train_batch_size=1,
    logging_steps=1,
    save_strategy="no",
    report_to="none",
    max_length=4096,
    gradient_checkpointing=True,
    packing=True,
    packing_strategy="wrapped",
    include_num_input_tokens_seen="true",
)

print("Creating trainer...")
trainer = SFTTrainer(model=model, args=training_args, train_dataset=dataset)

print("Training...")
trainer.train()
print("SUCCESS!")
