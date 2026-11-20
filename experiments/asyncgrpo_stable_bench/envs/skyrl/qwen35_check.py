"""Qwen3.5 checks for the SkyRL FSDP trainer path, on 1 GPU, through SkyRL's own HFModelWrapper.

1. Which kernels transformers bound for the gated-DeltaNet layers (fla / causal-conv1d vs the torch fallback).
2. Whether SkyRL's sample packing (`trainer.remove_microbatch_padding=true`) leaks state across sample
   boundaries in the linear-attention layers: logprobs of a packed 2-sample micro-batch vs each sample alone.
3. Fwd+bwd time of one 8k-token sequence; `--no-fla` reruns it with fla and causal_conv1d hidden, for the A/B.

Usage: python qwen35_check.py [--no-fla]
"""

import statistics
import sys
import time


if "--no-fla" in sys.argv:
    # Hide the fast kernels before transformers imports them: this forces the pure-torch GDN path.
    sys.modules["fla"] = None
    sys.modules["causal_conv1d"] = None

import torch
import transformers.models.qwen3_5.modeling_qwen3_5 as qmod
from skyrl.backends.skyrl_train.workers.model_wrapper import HFModelWrapper
from transformers import AutoTokenizer


MODEL = "Qwen/Qwen3.5-2B"
torch.manual_seed(0)


def bound_impl(fn):
    """Return the module of the implementation transformers resolved behind a kernel-fallback wrapper."""
    seen, stack = set(), [fn]
    while stack:
        f = stack.pop()
        if id(f) in seen:
            continue
        seen.add(id(f))
        for cell in getattr(f, "__closure__", None) or ():
            try:
                v = cell.cell_contents
            except ValueError:
                continue
            if callable(v):
                mod = getattr(v, "__module__", "") or ""
                if mod.startswith(("fla", "causal_conv1d")):
                    return f"{mod}.{getattr(v, '__name__', '?')}"
                stack.append(v)
    return "torch fallback (transformers)"


print("chunk_gated_delta_rule ->", bound_impl(qmod.torch_chunk_gated_delta_rule))
print("causal_conv1d_fn       ->", bound_impl(qmod.causal_conv1d_fn))

tok = AutoTokenizer.from_pretrained(MODEL)
text = open(qmod.__file__).read()
ids = tok(text, return_tensors="pt").input_ids[0]


def logprobs(model, seqs):
    """Right-aligned prompt-style batch like SkyRL builds: left padding, then score the last n tokens."""
    L = max(len(s) for s in seqs)
    n = min(len(s) for s in seqs) - 1
    x = torch.full((len(seqs), L), tok.pad_token_id, dtype=torch.long)
    m = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, L - len(s) :] = s
        m[i, L - len(s) :] = 1
    with torch.no_grad():
        return model(x.cuda(), n, attention_mask=m.cuda()).float().cpu()


if "--no-fla" not in sys.argv:
    a, b = ids[:700], ids[2000:3200]
    for packed in (True, False):
        model = HFModelWrapper(
            MODEL, use_flash_attention_2=True, bf16=True, language_model_only=True, remove_microbatch_padding=packed
        ).cuda()
        model.eval()
        both = logprobs(model, [a, b])
        alone_a = logprobs(model, [a])
        alone_b = logprobs(model, [b])
        n = both.shape[1]
        da = (both[0] - alone_a[0, -n:]).abs().max().item()
        db = (both[1] - alone_b[0, -n:]).abs().max().item()
        print(
            f"remove_microbatch_padding={packed}: max|dlogp| sample0={da:.4f} sample1={db:.4f} (mean |lp| {alone_b.abs().mean():.3f})"
        )
        del model
        torch.cuda.empty_cache()

model = HFModelWrapper(MODEL, use_flash_attention_2=True, bf16=True, language_model_only=True).cuda()
model.gradient_checkpointing_enable()
model.train()
seq = ids[:8192].unsqueeze(0).cuda()
mask = torch.ones_like(seq)
times = []
for i in range(6):
    torch.cuda.synchronize()
    t = time.perf_counter()
    lp = model(seq, 8000, attention_mask=mask)
    lp.mean().backward()
    torch.cuda.synchronize()
    times.append(time.perf_counter() - t)
    model.zero_grad(set_to_none=True)
print(
    f"fwd+bwd 1x8192 tokens, grad ckpt: median {statistics.median(times[1:]):.3f}s (first {times[0]:.2f}s)",
    "[no-fla]" if "--no-fla" in sys.argv else "[fla]",
)
print("peak mem GB", torch.cuda.max_memory_allocated() / 1e9)
