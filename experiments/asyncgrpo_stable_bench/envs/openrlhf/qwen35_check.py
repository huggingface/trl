"""Qwen3.5 checks on OpenRLHF's own trainer model class (openrlhf.models.Actor), one GPU, no DeepSpeed.

1. Which kernels transformers bound for the gated-DeltaNet layers (fla chunk rule, causal_conv1d) in this venv.
2. Packing: log-probs of a sample packed after another one vs the same sample alone, through Actor.forward with
   packing_samples=True (OpenRLHF's path: unpad to one row, per-sample position_ids, attention_mask=None).
3. Speed: fwd+bwd time per token on a packed ~32k-token row (the W1 max_tokens_per_gpu), gradient checkpointing on,
   and the top CUDA kernels from torch.profiler so the fla kernels are visible by name.

    python envs/openrlhf/qwen35_check.py <model dir>
"""

import importlib
import sys
import time

import torch
from openrlhf.models import Actor


model_dir = sys.argv[1]

# 1. kernel binding
import transformers.models.qwen3_5.modeling_qwen3_5 as m  # noqa: E402


# use_kernel_func_from_hub_with_fallback wraps the resolved implementation with functools.wraps(torch_function), so the
# wrapper's __module__ is always transformers': read the bound `implementation` from the closure instead.
for name in (
    "torch_chunk_gated_delta_rule",
    "torch_recurrent_gated_delta_rule",
    "causal_conv1d_fn",
    "causal_conv1d_update",
):
    fn = getattr(m, name, None)
    impl = [
        c.cell_contents
        for c in (getattr(fn, "__closure__", None) or ())
        if callable(c.cell_contents) and c.cell_contents is not fn
    ]
    print(
        f"modeling_qwen3_5.{name} -> bound to {[f'{i.__module__}.{i.__name__}' for i in impl if hasattr(i, '__module__')]}"
    )
from fla.ops.gated_delta_rule import chunk_gated_delta_rule  # noqa: E402


print("fla import OK:", chunk_gated_delta_rule.__module__)
print("causal_conv1d import OK:", importlib.import_module("causal_conv1d").causal_conv1d_fn.__module__)

actor = Actor(model_dir, attn_implementation="flash_attention_2", param_dtype="bf16", packing_samples=True)
actor.model.cuda()
print("model class:", type(actor.model).__name__, "is_vlm:", actor.is_vlm)
for mod in actor.model.modules():
    if type(mod).__name__.endswith("GatedDeltaNet"):
        print(
            "GatedDeltaNet attrs:",
            {
                k: getattr(getattr(mod, k), "__module__", None)
                for k in ("chunk_gated_delta_rule", "causal_conv1d_fn")
                if hasattr(mod, k)
            },
        )
        break

# 2. packing boundary check
tok_ids = lambda n, seed: torch.randint(1000, 100000, (n,), generator=torch.Generator().manual_seed(seed)).tolist()  # noqa: E731
a, b = tok_ids(3000, 0), tok_ids(500, 1)
L = len(a) + len(b)


def batch(seqs):
    T = max(len(s) for s in seqs)
    ids = torch.zeros(len(seqs), T, dtype=torch.long)
    mask = torch.zeros(len(seqs), T, dtype=torch.long)
    for i, s in enumerate(seqs):  # left padding, like OpenRLHF's rollout batches
        ids[i, T - len(s) :] = torch.tensor(s)
        mask[i, T - len(s) :] = 1
    return ids.cuda(), mask.cuda()


actor.eval()
with torch.no_grad():
    ids, mask = batch([a, b])
    packed = actor(ids, attention_mask=mask, return_logprobs=True)[1, -len(b) + 1 :]  # b's next-token logprobs
    ids, mask = batch([b])
    alone = actor(ids, attention_mask=mask, return_logprobs=True)[0, -len(b) + 1 :]
d = (packed - alone).abs()
print(
    f"packing: b after a vs b alone: max |dlogp| = {d.max().item():.4f}, mean = {d.mean().item():.4f}, "
    f"first 8 tokens = {[round(x, 3) for x in d[:8].tolist()]}"
)

# 2b. the same check without packing (padded batch + attention_mask, what OpenRLHF does when packing is off)
unpacked = Actor(model_dir, attn_implementation="flash_attention_2", param_dtype="bf16", packing_samples=False)
unpacked.model.cuda().eval()
with torch.no_grad():
    ids, mask = batch([a, b])
    padded = unpacked(ids, attention_mask=mask, return_logprobs=True)[1, -len(b) + 1 :]
    ids, mask = batch([b])
    alone_u = unpacked(ids, attention_mask=mask, return_logprobs=True)[0, -len(b) + 1 :]
d = (padded - alone_u).abs()
print(
    f"no packing: b left-padded next to a vs b alone: max |dlogp| = {d.max().item():.4f}, mean = {d.mean().item():.4f}"
)
d = (alone_u - alone).abs()
print(f"b alone, unpacked vs packed path: max |dlogp| = {d.max().item():.4f}")
del unpacked
torch.cuda.empty_cache()

# 3. speed: one packed row of ~32k tokens (64 samples of 512), fwd + bwd, gradient checkpointing
actor.train()
actor.gradient_checkpointing_enable()
seqs = [tok_ids(512, 100 + i) for i in range(64)]
ids, mask = batch(seqs)
amask = mask[:, 1:].clone()


def step():
    lp = actor(ids, action_mask=amask, attention_mask=mask)
    (-lp.sum() / amask.sum()).backward()


for _ in range(2):
    step()
torch.cuda.synchronize()
t0 = time.time()
for _ in range(3):
    step()
torch.cuda.synchronize()
dt = (time.time() - t0) / 3
ntok = int(mask.sum())
print(
    f"speed: {ntok} tokens fwd+bwd in {dt:.3f}s = {dt / ntok * 1e6:.1f} us/tok, peak mem {torch.cuda.max_memory_allocated() / 1e9:.1f} GB"
)
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
) as prof:
    step()
    torch.cuda.synchronize()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15, max_name_column_width=70))
