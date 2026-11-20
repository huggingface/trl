"""Per-arm summary for options.sbatch: pause stats, gen tok/s, API-server CPU/RSS, errors, TCP counters.

Usage: python options.py <logs/JOBID-apiN> [...]   (also accepts the older logs/<jobid> dirs)
"""

import json
import re
import statistics
import sys
from collections import Counter


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


for d in sys.argv[1:]:
    rows = [json.loads(l) for l in open(f"{d}/load.jsonl")]
    cpu = [json.loads(l) for l in open(f"{d}/cpu.jsonl")]
    vlog = open(f"{d}/vllm.log", errors="replace").read().splitlines()
    print(f"\n=== {d}")
    for arm in dict.fromkeys(r["arm"] for r in rows):
        r = [x for x in rows if x["arm"] == arm]
        ms = next(x for x in r if x["kind"] == "measure_start")["t"]
        me = next(x for x in r if x["kind"] == "measure_end")
        end = next((x for x in r if x["kind"] == "arm_end"), {})
        p = [x["pause_s"] for x in r if x["kind"] == "pause"]
        replies = [x for x in r if x["kind"] == "reply" and ms <= x["end"] <= me["t"]]
        # API servers: processes named VLLM::APIServer_N (or "vllm serve"/"python" frontends); engines are EngineCore_DPn
        api_cpu, api_rss, eng_cpu = Counter(), Counter(), Counter()
        n = 0
        for c in cpu:
            if not ms <= c["t"] <= me["t"]:
                continue
            n += 1
            for k, (cp, rss) in c["cpu_rss"].items():
                if "APIServer" in k:
                    api_cpu[k] += cp
                    api_rss[k] = max(api_rss[k], rss)
                elif "EngineCore" in k:
                    eng_cpu[k] += cp
        api = sorted(api_cpu)
        cpus = [api_cpu[k] / max(n, 1) for k in api]
        errs = [x for x in r if x["kind"] == "err"]
        err_kind = Counter(
            f"{x.get('exc', x.get('status'))}{'' if x.get('head') else ' (before headers)'}" for x in errs
        )
        # an error within 2 s of a /pause or /resume?
        pauses = [(x["start"], x["t"]) for x in r if x["kind"] == "pause"]
        near = sum(any(a - 2 <= x["t"] <= b + 2 for a, b in pauses) for x in errs)
        err_age = [x["t"] - x["start"] for x in errs]
        print(f"\n-- {arm}: {len(p)} pauses")
        if p:
            print(
                f"pause mean {statistics.mean(p):.2f} median {statistics.median(p):.2f} p90 {pct(p, 0.9):.2f} "
                f"max {max(p):.2f}  >1s: {sum(x > 1 for x in p)}  >10s: {sum(x > 10 for x in p)}"
            )
            print("pauses:", " ".join(f"{x:.1f}" for x in p))
        print(
            f"gen_tok_s {me['gen_tok_s'] or 0:.0f}  replies in window {len(replies)}  "
            f"reply tok/s {sum(x['tokens'] for x in replies) / (me['t'] - ms):.0f}  "
            f"bytes/reply {statistics.mean(x['bytes'] for x in replies) if replies else 0:.0f}"
        )
        if cpus:
            print(
                f"API servers ({len(api)}): CPU mean {statistics.mean(cpus):.2f} max {max(cpus):.2f} cores/server, "
                f"total {sum(cpus):.1f}; peak RSS mean {statistics.mean(api_rss[k] for k in api):.2f} "
                f"max {max(api_rss[k] for k in api):.2f} GiB"
            )
        if eng_cpu:
            print(f"engine cores: CPU mean {statistics.mean(v / max(n, 1) for v in eng_cpu.values()):.2f}")
        print(f"errors (whole arm, incl. fill): {me['errors']}; logged {len(errs)}: {dict(err_kind)}")
        if errs:
            print(
                f"  error request age s: median {statistics.median(err_age):.1f} max {max(err_age):.1f}; "
                f"within 2 s of a pause/resume: {near}"
            )
        if end.get("tcp"):
            keys = (
                "ListenOverflows",
                "ListenDrops",
                "OutRsts",
                "EstabResets",
                "AttemptFails",
                "TCPAbortOnData",
                "TCPAbortOnClose",
                "TCPAbortOnTimeout",
                "TCPBacklogDrop",
                "TCPReqQFullDrop",
                "SyncookiesSent",
            )
            print("tcp:", {k.split(".")[1]: v for k, v in end["tcp"].items() if k.split(".")[1] in keys})
    srv = [
        l
        for l in vlog
        if re.search(r"error|exception|traceback|reset|timeout|h11|uvicorn", l, re.I)
        and "HF_HUB_ENABLE_HF_TRANSFER" not in l
        and "warnings.warn" not in l
    ]
    print(f"\nvllm.log error-like lines: {len(srv)}")
    for l in Counter(re.sub(r"\d+", "N", l)[:160] for l in srv).most_common(8):
        print(" ", l)
    names = Counter(re.sub(r":\d+$", "", k) for c in cpu[:3] for k in c["cpu_rss"])
    print("process names seen:", dict(names))

if len(sys.argv) == 1 or True:
    m = [
        json.loads(l)
        for l in open("/fsx/amine_dirhoussi/stable-bench-runs/trl/trl-w1-S-recommended-364705/metrics.jsonl")
    ]
    for k in ("sampling/sampler_mass", "sampling/masked_mean", "entropy", "batch/masked_token_frac"):
        v = [x[k] for x in m if k in x]
        if v:
            print(
                f"364705 {k}: n {len(v)} mean {statistics.mean(v):.5f} min {min(v):.5f} max {max(v):.5f} "
                f"first10 {statistics.mean(v[:10]):.5f} last10 {statistics.mean(v[-10:]):.5f}"
            )
