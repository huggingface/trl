"""Score centering's correction with k = 1, 2, 4, 8 sampler logprobs vs k = 20, as in AsyncGRPOTrainer.compute_loss.

The correction's gradient w.r.t. the logits at a position is sum_j r_j (e_j - p), so it is exact over the vocabulary.
Usage: score.py <policy ckpt> <gen.pt>
"""

import sys

import torch
from transformers import AutoModelForImageTextToText


policy, path = sys.argv[1], sys.argv[2]
rows = torch.load(path)
model = AutoModelForImageTextToText.from_pretrained(policy, dtype=torch.bfloat16).cuda().eval()
LOW, HIGH, KS = 0.5, 5.0, (1, 2, 4, 8, 20)
acc = {k: {"nz": 0, "d2": 0.0, "g2": 0.0} for k in KS}
acc_main2 = acc_masked = n_pos = 0
mass = dict.fromkeys(KS, 0.0)
for r in rows:
    ids = torch.tensor(r["prompt_ids"] + r["ids"], device="cuda")
    P = len(r["prompt_ids"])
    with torch.no_grad():
        logits = model(input_ids=ids[None]).logits[0, P - 1 : -1].float()
    top_ids = torch.tensor([[t for t, _ in x] + [0] * (20 - len(x)) for x in r["top"]], device="cuda")
    top_lp = torch.tensor([[l for _, l in x] + [-float("inf")] * (20 - len(x)) for x in r["top"]], device="cuda")
    x = ids[P:]
    old = torch.tensor(r["sampled_lp"], device="cuda")
    for s in range(0, len(x), 512):
        logp = torch.log_softmax(logits[s : s + 512], -1)
        p = logp.exp()
        w = (logp.gather(1, x[s : s + 512, None])[:, 0] - old[s : s + 512]).exp()
        masked = (w < LOW) | (w > HIGH)
        w = torch.where(masked, 0.0, w)
        main = -p * w[:, None]
        main.scatter_add_(1, x[s : s + 512, None], w[:, None])  # w (e_x - p)
        g = {}
        for k in KS:
            q = top_lp[s : s + 512, :k].exp()
            S = top_ids[s : s + 512, :k]
            pk = p.gather(1, S) * (q > 0)
            head = torch.where((pk >= LOW * q) & (pk <= HIGH * q), pk, 0.0)
            tail_ratio = (1 - pk.sum(-1)).clamp(min=1e-6) / (1 - q.sum(-1)).clamp(min=1e-6)
            tail = ((tail_ratio >= LOW) & (tail_ratio <= HIGH)).float()
            res = head - tail[:, None] * pk
            gk = -p * res.sum(-1, keepdim=True)
            gk.scatter_add_(1, S, res)
            g[k] = gk
            acc[k]["nz"] += int((res.abs().sum(-1) > 1e-12).sum())
            mass[k] += float(q.sum())
        ref = main - g[20]  # per-token gradient of (weight * log p - correction) w.r.t. logits, k = 20
        acc_main2 += float((ref**2).sum())
        for k in KS:
            acc[k]["d2"] += float(((g[k] - g[20]) ** 2).sum())
            acc[k]["g2"] += float((g[k] ** 2).sum())
        acc_masked += int(masked.sum())
        n_pos += len(w)
print(
    f"{path} policy={policy}: {len(rows)} completions, {n_pos} tokens, sampled tokens masked {acc_masked / n_pos:.2e}"
)
print(
    "k | sampler mass in top-k | positions with nonzero correction | |correction| / |loss grad| | |corr_k - corr_20| / |loss grad|"
)
for k in KS:
    a = acc[k]
    print(
        f"{k} | {mass[k] / n_pos:.4f} | {a['nz'] / n_pos:.2e} | {(a['g2'] / acc_main2) ** 0.5:.2e} | {(a['d2'] / acc_main2) ** 0.5:.2e}"
    )
