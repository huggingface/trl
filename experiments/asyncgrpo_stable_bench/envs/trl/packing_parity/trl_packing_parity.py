"""
Packed-vs-alone log-prob parity for TRL's AsyncGRPOTrainer on Qwen/Qwen3.5-2B (1 GPU), through the trainer's own path.

The model is built the way `AsyncGRPOTrainer.__init__` builds it (`create_model_from_path(attn_implementation=
TREE_ATTENTION)`, `patch_tree_gdn`, `add_fused_lm_head(temperature=1.0, outputs=("log_probs", "entropy"))`), in bf16
(the trainer's compute dtype under FSDP2 bf16 mixed precision). Each row goes through TRL's real packing:
`TreeRow.add` -> `DataCollatorForRollout` (which calls `TreeRow.pack`) -> the same tensor slicing and model call as
`AsyncGRPOTrainer.compute_loss` (`unpad_tree`, `shift_labels`, `fused_lm_head=True`), then
`outputs.log_probs.flatten()[token_index]` split by `segment_id`. On B200 the tree attention runs FA4
(kernels-community/flash-attn4) and the gated DeltaNet layers run TRL's tree Triton kernels.

References for each sample:
  - alone/TRL: the sample in a TreeRow of its own (one chain), same path.
  - alone/HF: a second, independent model (sdpa attention, fla `chunk_gated_delta_rule`, `causal_conv1d`, no tree
    kwargs, plain lm_head + log_softmax in fp32).

Tests:
  A. 2 real samples from different prompts packed in one row.
  B. 6 real samples from different prompts packed in one row (~realistic row length).
  C. prefix sharing: 8 real completions of one prompt (a GRPO group) folded into one trie, plus 1 synthetic fork that
     shares a prompt AND the first 200 completion tokens of completion 0 (a depth-2 tree, as a multi-turn fork makes).
  Negative controls on rows A and C:
  N1. naive concatenation: same packed tokens and restarting position ids, but the tree replaced by one segment, so
      attention and the recurrent/conv state both cross every boundary (what packing without boundaries does).
  N2. recurrent-only leak (row A): tree attention kept, but the gated DeltaNet/conv tree says sample 2 is sample 1's
      child, so only the linear-attention state leaks.
  G. gradients on row C: d(sum of completion log-probs)/d(params), packed tree vs the sum over samples alone, and N1.
"""

import json
import os
import time
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoTokenizer

from trl.experimental.async_grpo.async_grpo_trainer import DataCollatorForRollout
from trl.experimental.async_grpo.packing import (
    TREE_ATTENTION,
    TreeRow,
    _flash_attn,
    patch_tree_gdn,
    register_tree_attention,
    unpad_tree,
)
from trl.trainer.utils import add_fused_lm_head, create_model_from_path


NAME = "Qwen/Qwen3.5-2B"
EVAL = "/fsx/amine_dirhoussi/stable-bench-runs/trl/trl-w1-S-shared-364895/eval-split/batch-20261110T040301/eval-samples-200.jsonl"
PROMPT = "{problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
MAX_COMPLETION = int(os.environ.get("MAX_COMPLETION_TOKENS", "6000"))
dev = "cuda"
torch.manual_seed(0)
results = {}

tok = AutoTokenizer.from_pretrained(NAME)
eos = tok.convert_tokens_to_ids("<|im_end|>")

# ---- real samples: eval completions of TRL's own W1-S checkpoint-200 ----------------------------------------------
by_problem = defaultdict(list)
for line in open(EVAL):
    r = json.loads(line)
    by_problem[r["problem"]].append(r["text"])


def encode(problem, text, max_completion=MAX_COMPLETION):
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": PROMPT.format(problem=problem)}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    c = tok(text, add_special_tokens=False)["input_ids"][: max_completion - 1] + [eos]
    return p, c


def make_sample(p, c, group_id):
    ids = np.asarray(p + c, dtype=np.int64)
    mask = np.zeros(len(ids), dtype=np.int64)
    mask[len(p) :] = 1
    return {
        "input_ids": ids,
        "completion_mask": mask,
        "old_log_probs": np.zeros(len(ids), dtype=np.float32),
        "sampling_mask": [[] for _ in ids],
        "sampler_ids": np.zeros((len(ids), 0), dtype=np.int64),
        "sampler_log_probs": np.zeros((len(ids), 0), dtype=np.float32),
        "advantage": 1.0,
        "group_id": group_id,
        "metrics": {},
    }


problems = [p for p, texts in by_problem.items() if len(texts) >= 8]
problems.sort(key=lambda p: -np.mean([len(t) for t in by_problem[p]]))
print(f"{len(by_problem)} problems in eval file, {len(problems)} with >= 8 completions", flush=True)

# ---- the trainer's model ---------------------------------------------------------------------------------------------
# patch_tree_gdn imports these two names when it patches, so count them before it runs
calls = defaultdict(int)
import trl.experimental.async_grpo.tree_gated_delta_rule as tg  # noqa: E402


for fn in ("tree_chunk_gated_delta_rule", "tree_causal_conv1d"):
    orig = getattr(tg, fn)

    def counted(*a, _orig=orig, _fn=fn, **k):
        calls[_fn] += 1
        return _orig(*a, **k)

    setattr(tg, fn, counted)

import trl.experimental.async_grpo.packing as pk  # noqa: E402


_orig_attn = pk.tree_attention


def _counted_attn(*a, **k):
    calls["tree_attention(FA varlen)"] += 1
    return _orig_attn(*a, **k)


pk.tree_attention = _counted_attn  # register_tree_attention reads the module global
register_tree_attention()
t0 = time.time()
model = create_model_from_path(NAME, device_map=None, attn_implementation=TREE_ATTENTION, dtype=torch.bfloat16)
assert patch_tree_gdn(model)
add_fused_lm_head(model, temperature=1.0, outputs=("log_probs", "entropy"))
model.to(dev).eval()
print(
    f"TRL model {type(model).__name__} loaded in {time.time() - t0:.0f}s; FA module: {_flash_attn().__name__}; "
    f"device {torch.cuda.get_device_name()}",
    flush=True,
)
collator = DataCollatorForRollout(tok.pad_token_id, num_processes=1, metrics=defaultdict(list), token_budget=0)


def trl_inputs(samples):
    """TreeRow -> collator -> the tensors compute_loss builds, on device."""
    row = TreeRow()
    for s in samples:
        row.add(s)
    batch = collator.torch_call([[row]])
    inputs = {k: v.to(dev) for k, v in batch.items()}
    mask_bool = inputs["attention_mask"].bool()
    selected = inputs["token_index"] >= 0
    kw = dict(
        input_ids=inputs["input_ids"][mask_bool].unsqueeze(0),
        position_ids=inputs["position_ids"][mask_bool].unsqueeze(0),
        shift_labels=inputs["shift_labels"][mask_bool].unsqueeze(0),
        support_ids=None,
        score_ids=None,
        fused_lm_head=True,
    )
    tree = unpad_tree(inputs)
    return kw, tree, inputs["token_index"][selected], inputs["segment_id"][selected], len(row.tokens)


def trl_logps(samples, tree_override=None, grad=False):
    kw, tree, token_index, segment_id, n_unique = trl_inputs(samples)
    if tree_override is not None:
        tree = tree_override(tree, kw)
    with torch.set_grad_enabled(grad):
        out = model(**kw, **tree)
        lp = out.log_probs.flatten()[token_index].float()
    per = [lp[segment_id == i] for i in range(len(samples))]
    return per, n_unique, kw["input_ids"].shape[1]


def one_segment(tree, kw):
    """Naive concatenation: one segment over the whole row, so nothing stops at a sample boundary."""
    n = kw["input_ids"].shape[1]
    return {
        "tree_kv_index": torch.arange(n, device=dev),
        "tree_cu_seqlens_q": torch.tensor([0, n], dtype=torch.int32, device=dev),
        "tree_cu_seqlens_k": torch.tensor([0, n], dtype=torch.int32, device=dev),
        "tree_max_seqlen_q": n,
        "tree_max_seqlen_k": n,
        "tree_segment_parents": torch.tensor([-1], dtype=torch.int32, device=dev),
        "tree_segment_depths": torch.tensor([0], dtype=torch.int32, device=dev),
        "tree_level_sizes": [1],
        "tree_n_chunks": -(-n // 64),
    }


def chain_recurrence(tree, kw):
    """
    Recurrent-only leak: the gated DeltaNet/conv tree is told sample 2's own segment is a child of sample 1's last
    segment, so sample 2 starts from sample 1's final state and conv history. Attention keeps the real tree.
    (The two prompts share the chat-template header, so row A is a root segment with two children.)
    """
    parents = tree["tree_segment_parents"].tolist()
    leaves = [i for i in range(len(parents)) if i not in parents]
    assert len(leaves) == 2, parents
    parents[leaves[1]] = leaves[0]
    depths = []
    for p in parents:
        depths.append(0 if p < 0 else depths[p] + 1)
    t = dict(tree)
    t["tree_segment_parents"] = torch.tensor(parents, dtype=torch.int32, device=dev)
    t["tree_segment_depths"] = torch.tensor(depths, dtype=torch.int32, device=dev)
    t["tree_level_sizes"] = [depths.count(d) for d in range(max(depths) + 1)]
    print("N2 tree: parents", tree["tree_segment_parents"].tolist(), "->", parents, flush=True)
    return t


hf = None


def hf_logps(sample):
    ids = torch.from_numpy(sample["input_ids"]).to(dev)
    with torch.no_grad():
        logits = hf(input_ids=ids[None]).logits[0, :-1].float()
        lp = torch.log_softmax(logits, -1).gather(-1, ids[1:, None])[:, 0]
    trained = torch.from_numpy(np.flatnonzero(sample["completion_mask"][1:])).to(dev)
    return lp[trained]


def compare(name, samples, packed, alone_trl, alone_hf=None, extra=None):
    rows = []
    for i, s in enumerate(samples):
        d = (packed[i] - alone_trl[i]).abs()
        row = {
            "sample": i,
            "tokens": int(len(s["input_ids"])),
            "completion_tokens": int(s["completion_mask"].sum()),
            "max_dlogp_vs_alone_trl": d.max().item(),
            "mean_dlogp_vs_alone_trl": d.mean().item(),
            "mean_logp": alone_trl[i].mean().item(),
        }
        if alone_hf is not None:
            dh = (packed[i] - alone_hf[i]).abs()
            row["max_dlogp_vs_alone_hf"] = dh.max().item()
            row["mean_dlogp_vs_alone_hf"] = dh.mean().item()
            da = (alone_trl[i] - alone_hf[i]).abs()
            row["max_dlogp_alone_trl_vs_hf"] = da.max().item()
            row["mean_dlogp_alone_trl_vs_hf"] = da.mean().item()
        rows.append(row)
    summary = {k: max(r[k] for r in rows) for k in rows[0] if k.startswith("max_")}
    summary.update({k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k.startswith("mean_dlogp")})
    results[name] = {"summary": summary, "per_sample": rows, **(extra or {})}
    print(f"\n== {name}: {json.dumps(summary)}", flush=True)
    for r in rows:
        print("   ", {k: round(v, 5) if isinstance(v, float) else v for k, v in r.items()}, flush=True)


# ---- rows ----------------------------------------------------------------------------------------------------------
# A: two samples, two different prompts (one completion each)
A = [
    make_sample(*encode(problems[0], by_problem[problems[0]][0]), 0),
    make_sample(*encode(problems[1], by_problem[problems[1]][0]), 1),
]
# B: six samples, six prompts
B = [make_sample(*encode(problems[i], by_problem[problems[i]][1]), i) for i in range(2, 8)]
# C: a GRPO group (8 completions of one prompt) + a depth-2 fork sharing the prompt and 200 completion tokens
g = problems[8]
C = [make_sample(*encode(g, by_problem[g][j], max_completion=3000), 100) for j in range(8)]
p0, c0 = encode(g, by_problem[g][0], max_completion=3000)
p1, c1 = encode(g, by_problem[g][1], max_completion=3000)
C.append(make_sample(p0, c0[:200] + c1[:2500], 100))


# D: row A's two samples without the chat template, so the two share no first token and each is a root segment of
# its own, chunked from its own first token exactly as when it runs alone. Isolates leakage from chunk-alignment noise.
def encode_raw(problem, text):
    p = tok(PROMPT.format(problem=problem), add_special_tokens=False)["input_ids"]
    c = tok(text, add_special_tokens=False)["input_ids"][: MAX_COMPLETION - 1] + [eos]
    return p, c


D = [
    make_sample(*encode_raw(problems[0], by_problem[problems[0]][0]), 0),
    make_sample(*encode_raw(problems[1], by_problem[problems[1]][0]), 1),
]
assert D[0]["input_ids"][0] != D[1]["input_ids"][0]

for name, S in (("A", A), ("B", B), ("C", C), ("D", D)):
    print(name, "lengths", [len(s["input_ids"]) for s in S], flush=True)

alone = {}
for name, S in (("A", A), ("B", B), ("C", C), ("D", D)):
    alone[name] = [trl_logps([s])[0][0] for s in S]

calls.clear()
pA, nA, fA = trl_logps(A)
print("tree kernel calls in packed row A forward:", dict(calls), flush=True)
pB, nB, fB = trl_logps(B)
pC, nC, fC = trl_logps(C)
nA1, _, _ = trl_logps(A, one_segment)
nA2, _, _ = trl_logps(A, chain_recurrence)
nC1, _, _ = trl_logps(C, one_segment)
pD, nD, fD = trl_logps(D)
nD1, _, _ = trl_logps(D, one_segment)

# independent reference model
hf = create_model_from_path(NAME, device_map=None, attn_implementation="sdpa", dtype=torch.bfloat16).to(dev).eval()
alone_hf = {name: [hf_logps(s) for s in S] for name, S in (("A", A), ("B", B), ("C", C))}
del hf
torch.cuda.empty_cache()

compare("A_two_prompts_packed", A, pA, alone["A"], alone_hf["A"], {"row_tokens": fA})
compare("B_six_prompts_packed", B, pB, alone["B"], alone_hf["B"], {"row_tokens": fB})
compare(
    "C_prefix_tree_group",
    C,
    pC,
    alone["C"],
    alone_hf["C"],
    {"row_tokens_unique": fC, "raw_tokens": sum(len(s["input_ids"]) for s in C)},
)
compare("D_two_raw_prompts_no_shared_token", D, pD, alone["D"], None, {"row_tokens": fD})
compare("N1_D_naive_concat", D, nD1, alone["D"])
compare("N1_A_naive_concat", A, nA1, alone["A"])
compare("N2_A_recurrent_state_leak_only", A, nA2, alone["A"])
compare("N1_C_naive_concat_tree_as_chain", C, nC1, alone["C"])

# ---- gradients on the prefix tree ---------------------------------------------------------------------------------
model.gradient_checkpointing_enable()
model.train()  # Qwen3.5 has no dropout; train() only turns checkpointing on
params = [p for p in model.parameters() if p.requires_grad]


def grads(fn):
    model.zero_grad(set_to_none=True)
    fn()
    return torch.cat([p.grad.float().flatten() for p in params if p.grad is not None])


def packed_loss(override=None):
    per, _, _ = trl_logps(C, override, grad=True)
    torch.stack([x.sum() for x in per]).sum().backward()


def alone_loss():
    for s in C:
        trl_logps([s], grad=True)[0][0].sum().backward()


g_alone = grads(alone_loss)
g_packed = grads(packed_loss)
g_naive = grads(lambda: packed_loss(one_segment))
g_alone2 = grads(alone_loss)  # run-to-run noise of the reference itself


def rel(a, b):
    return ((a - b).norm() / b.norm()).item()


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


results["G_prefix_tree_gradients"] = {
    "rel_l2_packed_vs_alone": rel(g_packed, g_alone),
    "cos_packed_vs_alone": cos(g_packed, g_alone),
    "rel_l2_naive_concat_vs_alone": rel(g_naive, g_alone),
    "cos_naive_concat_vs_alone": cos(g_naive, g_alone),
    "rel_l2_alone_rerun_vs_alone": rel(g_alone2, g_alone),
    "grad_norm_alone": g_alone.norm().item(),
}
print("\n== G:", json.dumps(results["G_prefix_tree_gradients"]), flush=True)

out = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), f"results-{os.environ.get('SLURM_JOB_ID', 'local')}.json"
)
json.dump(results, open(out, "w"), indent=1)
print("wrote", out)
