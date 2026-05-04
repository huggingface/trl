# ruff: noqa
"""Ground truth test - no EP, no TP, just regular model loading."""

import torch
from transformers import AutoModelForCausalLM


model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-30B-A3B",
    dtype=torch.bfloat16,
    device_map="cuda:0",
)

experts = model.model.layers[0].mlp.experts
gate = model.model.layers[0].mlp.gate


def gate_hook(mod, input, output):
    router_logits, router_scores, router_indices = output
    print(f"[Gate] logits:{router_logits.shape} scores:{router_scores.shape} indices:{router_indices.shape}")
    print(f"  indices[0]: {router_indices[0].tolist()}")


gate.register_forward_hook(gate_hook)


def experts_pre_hook(mod, input):
    hidden_states, top_k_index, top_k_weights = input
    print(f"[Experts in] hs:{hidden_states.shape} idx:{top_k_index.shape} wts:{top_k_weights.shape}")
    print(f"  idx[0]: {top_k_index[0].tolist()}")
    print(f"  wts[0,:8]: {[round(x, 4) for x in top_k_weights[0, :8].float().tolist()]}")


experts.register_forward_pre_hook(experts_pre_hook)


def experts_post_hook(mod, input, output):
    print(f"[Experts out] shape:{output.shape} nan:{output.isnan().any()} max:{output.abs().max():.6f}")


experts.register_forward_hook(experts_post_hook)

x = torch.tensor([[1, 2, 3]], device="cuda:0")
with torch.no_grad():
    out = model(x, use_cache=False)
print(f"\nlogits: {out.logits[0, -1, :5]}")
