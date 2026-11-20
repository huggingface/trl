"""Qwen3.5 check for verl (1 GPU): kernel binding, packed-vs-unpacked equivalence through verl's patch, fla vs torch timing.

python envs/verl/qwen35_check.py Qwen/Qwen3.5-2B
"""

import sys
import time

import torch
import transformers.models.qwen3_5.modeling_qwen3_5 as mq
from transformers import AutoModelForImageTextToText
from verl.models.transformers.monkey_patch import apply_monkey_patch


path = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3.5-2B"
torch.manual_seed(0)

import causal_conv1d  # noqa: F401
from fla.ops.gated_delta_rule import chunk_gated_delta_rule  # noqa: F401  (must import)


model = AutoModelForImageTextToText.from_pretrained(
    path, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
)
model = model.cuda()
print("class", type(model).__name__, "model_type", model.config.model_type)
lin = [m for m in model.modules() if isinstance(m, mq.Qwen3_5GatedDeltaNet)]
print("linear-attn layers", len(lin))
la = lin[0]
print("chunk_gated_delta_rule ->", la.chunk_gated_delta_rule.__module__, la.chunk_gated_delta_rule.__name__)
print("causal_conv1d_fn ->", None if la.causal_conv1d_fn is None else la.causal_conv1d_fn.__module__)
print("is_fast_path_available", mq.is_fast_path_available)

apply_monkey_patch(model, ulysses_sp_size=1, use_remove_padding=True, use_fused_kernels=False)
model.eval()
vocab = model.config.text_config.vocab_size
lens = [300, 777, 1200]
seqs = [torch.randint(0, vocab, (n,), device="cuda") for n in lens]


@torch.no_grad()
def single(s):
    return (
        model(input_ids=s[None], position_ids=torch.arange(len(s), device="cuda")[None], use_cache=False)
        .logits[0]
        .float()
    )


@torch.no_grad()
def packed(with_cu):
    ids = torch.cat(seqs)[None]
    pos = torch.cat([torch.arange(n, device="cuda") for n in lens])[None]
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), device="cuda", dtype=torch.long)
    kw = {"cu_seqlens": cu, "cu_seqlens_cpu": cu.cpu()} if with_cu else {}
    return model(input_ids=ids, position_ids=pos, use_cache=False, **kw).logits[0].float()


ref = torch.cat([single(s) for s in seqs])
for with_cu in (True, False):
    out = packed(with_cu)
    diffs = [(out[a:b] - ref[a:b]).abs().max().item() for a, b in zip([0, 300, 1077], [300, 1077, 2277], strict=False)]
    print(f"packed cu_seqlens={with_cu}: max|logit diff| per sample vs unpacked = {[round(d, 3) for d in diffs]}")


def fwd_bwd_ms(tokens=16384, n=8, iters=3):
    model.train()
    each = tokens // n
    ids = torch.randint(0, vocab, (1, tokens), device="cuda")
    pos = torch.arange(each, device="cuda").repeat(n)[None]
    cu = torch.arange(0, tokens + 1, each, device="cuda", dtype=torch.long)
    times = []
    for _ in range(iters + 1):
        torch.cuda.synchronize()
        t = time.time()
        out = model(input_ids=ids, position_ids=pos, cu_seqlens=cu, cu_seqlens_cpu=cu.cpu(), use_cache=False).logits
        out.float().mean().backward()
        torch.cuda.synchronize()
        times.append(time.time() - t)
        model.zero_grad(set_to_none=True)
    return 1000 * sum(times[1:]) / iters


print(f"fwd+bwd 16k packed tokens, fla + causal_conv1d: {fwd_bwd_ms():.0f} ms")
for m in lin:
    m.chunk_gated_delta_rule = mq.torch_chunk_gated_delta_rule
    m.causal_conv1d_fn = None
print(
    f"fwd+bwd 16k packed tokens, torch fallback (what you get without fla/causal-conv1d): {fwd_bwd_ms(iters=1):.0f} ms"
)
