"""FA4 as TE calls it (causal=True, window_size=(-1, 0)) vs torch SDPA, head_dim 256, forward and backward."""

import sys


try:
    import sitecustomize
except ImportError:
    sitecustomize = None
import torch
import torch.nn.functional as F
from flash_attn.cute import interface as fa4


print(
    "sitecustomize:",
    getattr(sitecustomize, "__file__", None),
    "patched:",
    getattr(fa4, "_miles_bench_fa4_window_fix", False),
)
torch.manual_seed(0)
q, k, v = (torch.randn(1, 1024, 8, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(3))
ref = F.scaled_dot_product_attention(
    q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
).transpose(1, 2)
ref.float().pow(2).sum().backward()
gref = [t.grad.clone() for t in (q, k, v)]
for t in (q, k, v):
    t.grad = None
out = fa4.flash_attn_func(q, k, v, causal=True, window_size=(-1, 0))
out = out[0] if isinstance(out, tuple) else out
print("fwd max abs err vs sdpa:", (out - ref).abs().max().item())
try:
    out.float().pow(2).sum().backward()
    print(
        "bwd max abs err dq/dk/dv:",
        [round((t.grad - g).abs().max().item(), 4) for t, g in zip((q, k, v), gref, strict=False)],
    )
except Exception as e:
    print("bwd failed:", type(e).__name__, e)
    sys.exit(1)
