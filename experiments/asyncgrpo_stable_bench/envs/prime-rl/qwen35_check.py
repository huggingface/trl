"""Qwen3.5-2B in prime-rl's trainer model (1 GPU): fla kernels bound, packing respects sample boundaries, speed, parity vs HF.

Loads the VLM checkpoint the way prime_rl.trainer.model does (AutoConfig + custom VLM class + from_pretrained), then:
1. counts calls into fla's chunk_gated_delta_rule / causal_conv1d during one forward,
2. packs two real samples and compares their logprobs with each sample run alone (and with a wrong single-sample
   seq_lens as a negative control),
3. times forward+backward on a 16k-token packed micro-batch,
4. compares next-token logprobs with transformers' Qwen3_5ForConditionalGeneration on the same text.
"""

import glob
import os
import subprocess
import sys
import time

import prime_rl.trainer.models.layers.gated_delta_net as gdn
import torch
from huggingface_hub import snapshot_download
from prime_rl.configs.trainer import ModelConfig
from prime_rl.trainer.model import resolve_auto_attn
from prime_rl.trainer.models import AutoModelForCausalLMPrimeRL, get_custom_vlm_cls
from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head
from safetensors.torch import load_file
from transformers import AutoConfig, AutoTokenizer


NAME = "Qwen/Qwen3.5-2B"
calls = {"chunk_gated_delta_rule": 0, "causal_conv1d": 0, "cu_seqlens_none": 0}
_orig_rule, _orig_conv = gdn.chunk_gated_delta_rule, gdn.causal_conv1d


def rule(*a, **k):
    calls["chunk_gated_delta_rule"] += 1
    calls["cu_seqlens_none"] += k.get("cu_seqlens") is None
    return _orig_rule(*a, **k)


def conv(*a, **k):
    calls["causal_conv1d"] += 1
    calls["cu_seqlens_none"] += k.get("cu_seqlens") is None
    return _orig_conv(*a, **k)


gdn.chunk_gated_delta_rule, gdn.causal_conv1d = rule, conv

runtime = ModelConfig()
resolve_auto_attn(runtime)
config = AutoConfig.from_pretrained(NAME, attn_implementation=runtime.attn)
config.use_cache = False
cls = get_custom_vlm_cls(config) or AutoModelForCausalLMPrimeRL
print("config:", type(config).__name__, config.model_type, "| class:", cls.__name__, "| attn:", runtime.attn)
t = time.time()
# Same sequence as prime_rl.trainer.model.setup_model / load_dcp_from_hf, minus FSDP: build on meta, materialize on
# GPU, init buffers, load the HF safetensors (dense Qwen3.5's HF->prime conversion only drops `mtp.*`).
with torch.device("meta"):
    model = cls.from_config(config, attn_implementation=runtime.attn, dtype=torch.bfloat16)
inject_prime_lm_head(model, chunk_size=None)
model.to_empty(device="cuda")
# to_empty() materializes each module's parameters separately, which unties lm_head from embed_tokens; the trainer
# FSDP-shards before materializing and asserts the tie. Re-tie here (the checkpoint ships no lm_head weight).
model.lm_head.weight = model.get_input_embeddings().weight
model.init_buffers_post_meta()
sd = {}
for f in glob.glob(os.path.join(snapshot_download(NAME), "*.safetensors")):
    sd.update({k: v for k, v in load_file(f, device="cuda").items() if not k.startswith("mtp.")})
missing, unexpected = model.load_state_dict(sd, strict=False)
assert model.lm_head.weight is model.get_input_embeddings().weight
print(
    "missing keys:",
    missing[:10],
    "| unexpected:",
    unexpected[:10],
    "| lm_head tied:",
    model.lm_head.weight is model.get_input_embeddings().weight,
)
del sd
print(
    f"loaded in {time.time() - t:.1f}s, is_vlm={model.is_vlm}, layer types:",
    {lt: config.text_config.layer_types.count(lt) for lt in set(config.text_config.layer_types)},
)

tok = AutoTokenizer.from_pretrained(NAME)
texts = [
    "Let x be a real number such that x^2 - 5x + 6 = 0. Find the sum of all possible values of x. " * 6,
    "The quick brown fox jumps over the lazy dog, and then it reasons step by step about prime numbers. " * 8,
]
ids = [torch.tensor(tok(s)["input_ids"], device="cuda") for s in texts]


def logprobs(input_ids, seq_lens):
    pos = torch.cat([torch.arange(n, device="cuda") for n in seq_lens]).unsqueeze(0)
    out = model(input_ids.unsqueeze(0), position_ids=pos, seq_lens=torch.tensor(seq_lens, device="cuda"))
    lp = torch.log_softmax(out["logits"].float(), -1)[0, :-1]
    return lp.gather(-1, input_ids[1:, None])[:, 0]


with torch.no_grad():
    packed_ids = torch.cat(ids)
    lens = [len(x) for x in ids]
    packed = logprobs(packed_ids, lens)
    print("kernel calls in one packed forward:", calls)
    alone = [logprobs(x, [len(x)]) for x in ids]
    n0 = lens[0]
    # packed[i] predicts token i+1; sample-2 predictions start at index n0 (token n0+1 from token n0).
    d1 = (packed[: n0 - 1] - alone[0]).abs().max().item()
    d2 = (packed[n0 : n0 + lens[1] - 1] - alone[1]).abs().max().item()
    wrong = logprobs(packed_ids.clone(), [sum(lens)])  # boundary dropped, position ids still restart
    d2_wrong = (wrong[n0 : n0 + lens[1] - 1] - alone[1]).abs().max().item()
    print(f"mean logprob alone: sample1 {alone[0].mean().item():.3f}, sample2 {alone[1].mean().item():.3f}")
    print(
        f"packing: sample1 max|dlogp|={d1:.4f}  sample2 max|dlogp|={d2:.4f}  "
        f"sample2 with boundary dropped max|dlogp|={d2_wrong:.4f}"
    )

# Speed: forward+backward on a 16384-token packed micro-batch of 8 x 2048 random tokens (bf16 weights, no FSDP).
model.train()
x = torch.randint(0, 150000, (16384,), device="cuda")
seq = [2048] * 8
pos = torch.cat([torch.arange(n, device="cuda") for n in seq]).unsqueeze(0)
for i in range(4):
    torch.cuda.synchronize()
    t = time.time()
    out = model(x.unsqueeze(0), position_ids=pos, seq_lens=torch.tensor(seq, device="cuda"))
    out["logits"].float().logsumexp(-1).mean().backward()
    torch.cuda.synchronize()
    print(
        f"fwd+bwd 16384 tokens, iter {i}: {time.time() - t:.3f}s, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB"
    )
model.zero_grad(set_to_none=True)

# Parity vs transformers on both texts, in a fresh process: importing prime_rl registers its own Qwen3.5 config
# classes with AutoConfig, which breaks transformers' own model class in this process.
del out, model
torch.cuda.empty_cache()
REF = "/tmp/qwen35_hf_ref.pt"
torch.save([x.cpu() for x in ids], REF)
subprocess.run(
    [
        sys.executable,
        "-c",
        f"""
import torch
from transformers import AutoModelForImageTextToText
ids = torch.load("{REF}")
hf = AutoModelForImageTextToText.from_pretrained("{NAME}", dtype=torch.bfloat16).cuda().eval()
with torch.no_grad():
    out = [torch.log_softmax(hf(input_ids=x[None].cuda()).logits.float(), -1)[0, :-1].gather(-1, x[1:, None].cuda())[:, 0].cpu() for x in ids]
torch.save(out, "{REF}")
""",
    ],
    check=True,
)
for i, hf_lp in enumerate(torch.load(REF)):
    d = (hf_lp.cuda() - alone[i]).abs()
    print(f"prime-rl vs transformers logprobs, sample{i + 1}: max {d.max().item():.4f} mean {d.mean().item():.5f}")
