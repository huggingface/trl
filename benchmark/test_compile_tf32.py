# ruff: noqa
"""Minimal test: torch.compile + MoE + TF32 fix.

Verifies that setting legacy TF32 flags before transformers' enable_tf32()
prevents the inductor crash when torch.compile reads the legacy property.
"""

import torch


# FIX: set legacy TF32 flags before any transformers import
# This prevents the crash when inductor later reads torch.backends.cuda.matmul.allow_tf32
# after transformers' enable_tf32() sets torch.backends.fp32_precision (new API).
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from transformers import AutoModelForCausalLM


print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}")
print(f"fp32_precision = {torch.backends.fp32_precision}")
print(f"legacy allow_tf32 = {torch.backends.cuda.matmul.allow_tf32}")

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-30B-A3B",
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
)
model.to("cuda")

compiled = torch.compile(model)

# Run a forward pass to trigger compilation
input_ids = torch.randint(0, 1000, (1, 128), device="cuda")
print("Running compiled forward pass...")
with torch.no_grad():
    out = compiled(input_ids)

print(f"Output shape: {out.logits.shape}")
print("torch.compile + TF32 fix works!")
