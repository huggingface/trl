"""Load vLLM like the AsyncGRPO rollout worker and time /pause?mode=keep the way TRL does.

Usage: python load.py <url> <arm> <out.jsonl> [fill_s] [measure_s]
arms:
  comp8   /v1/completions, logprobs=8 (score centering, what run 364705 does)
  compK   /v1/completions, logprobs=K (comp4, comp2, comp1: fewer sampler logprobs)
  comp0   /v1/completions, logprobs=0 (shared arm, run 364684)
  gen8    /inference/v1/generate, logprobs=8, detokenize=False, flat_logprobs=True (proposed fix)
"""

import asyncio
import json
import random
import sys
import threading
import time

import aiohttp
import requests


URL, ARM, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
FILL_S = float(sys.argv[4]) if len(sys.argv) > 4 else 120
MEASURE_S = float(sys.argv[5]) if len(sys.argv) > 5 else 170
CONCURRENCY = 688
PAUSE_EVERY_S = 20
MODEL = "Qwen/Qwen3.5-2B"
rng = random.Random(0)
out = open(OUT, "a", buffering=1)


def log(**kw):
    out.write(json.dumps({"arm": ARM, "t": time.time(), **kw}) + "\n")


def length():
    # run 364705's completions: ~35% hit the 16384 cap, the rest spread out below it (mean ~10k)
    return 16384 if rng.random() < 0.35 else rng.randint(500, 16000)


def payload(n):
    prompt = [rng.randint(1000, 100000) for _ in range(100)]
    sp = {
        "max_tokens": n,
        "min_tokens": n,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "repetition_penalty": 1.0,
        "n": 1,
        "seed": rng.getrandbits(63),
    }
    if ARM.startswith("comp"):
        return "/v1/completions", {
            "model": MODEL,
            "prompt": prompt,
            "return_token_ids": True,
            "return_tokens_as_token_ids": True,
            "logprobs": int(ARM[4:]),
            **sp,
        }
    return "/inference/v1/generate", {
        "model": MODEL,
        "token_ids": prompt,
        "sampling_params": {**sp, "logprobs": int(ARM[3:]), "detokenize": False, "flat_logprobs": True},
    }


stats = {"done": 0, "tokens": 0, "bytes": 0, "errors": 0, "checked": False}
stop = asyncio.Event()


async def one(session):
    n = length()
    path, body = payload(n)
    t0 = time.time()
    t_head = None
    try:
        async with session.post(URL + path, json=body) as resp:
            t_head = time.time()
            raw = await resp.read()
            if resp.status != 200:
                stats["errors"] += 1
                if stats["errors"] < 5:
                    log(kind="error", status=resp.status, body=raw[:500].decode(errors="replace"))
                log(kind="err", start=t0, head=t_head, status=resp.status)
                return
    except Exception as e:
        stats["errors"] += 1
        if stats["errors"] < 5:
            log(kind="error", exc=repr(e))
        # every error, compactly: did it fail before the response headers (request side) or while reading the body?
        log(kind="err", start=t0, head=t_head, exc=type(e).__name__, msg=str(e)[:80])
        return
    stats["done"] += 1
    stats["tokens"] += n
    stats["bytes"] += len(raw)
    log(kind="reply", start=t0, end=time.time(), tokens=n, bytes=len(raw))
    if not stats["checked"]:
        stats["checked"] = True
        c = json.loads(raw)["choices"][0]
        if ARM.startswith("comp"):
            lp = c["logprobs"]
            log(
                kind="check",
                n_tokens=len(c["token_ids"]),
                n_logprobs=len(lp["token_logprobs"]),
                top0=lp["top_logprobs"][0] if lp.get("top_logprobs") else None,
            )
        else:
            content = c["logprobs"]["content"]
            log(kind="check", n_tokens=len(c["token_ids"]), n_logprobs=len(content), first=content[0])


async def worker(session):
    while not stop.is_set():
        await one(session)


def metric(name):
    try:
        text = requests.get(URL + "/metrics", timeout=120).text
    except Exception:
        return None
    return sum(float(l.split()[-1]) for l in text.splitlines() if l.startswith(name + "{") or l.startswith(name + " "))


def tcp_counters():
    """Kernel TCP counters of this node (client and server share it): listen overflows, resets sent, ..."""
    res = {}
    for f in ("/proc/net/netstat", "/proc/net/snmp"):
        lines = open(f).read().splitlines()
        for head, vals in zip(lines[::2], lines[1::2], strict=False):
            prefix = head.split(":")[0]
            for k, v in zip(head.split()[1:], vals.split()[1:], strict=False):
                if prefix in ("Tcp", "TcpExt"):
                    res[f"{prefix}.{k}"] = int(v)
    return res


def syncer(t_start):
    """TRL's VLLMClient.pause/resume: a fresh `requests` connection per call."""
    time.sleep(FILL_S)
    gen0, t_gen0 = metric("vllm:generation_tokens_total"), time.time()
    log(kind="measure_start", done=stats["done"])
    while time.time() - t_start < FILL_S + MEASURE_S:
        t0 = time.time()
        r = requests.post(URL + "/pause", params={"mode": "keep", "clear_cache": "false"}, timeout=600)
        t1 = time.time()
        time.sleep(0.25)  # stands in for the 0.2 s NCCL transfer
        t2 = time.time()
        requests.post(URL + "/resume", timeout=600)
        t3 = time.time()
        log(kind="pause", start=t0, pause_s=t1 - t0, resume_s=t3 - t2, status=r.status_code)
        time.sleep(max(0.0, PAUSE_EVERY_S - (time.time() - t0)))
    gen1, t_gen1 = metric("vllm:generation_tokens_total"), time.time()
    log(
        kind="measure_end",
        done=stats["done"],
        gen_tok_s=(gen1 - gen0) / (t_gen1 - t_gen0) if gen0 and gen1 else None,
        errors=stats["errors"],
    )


async def main():
    t_start = time.time()
    log(kind="arm_start", concurrency=CONCURRENCY)
    tcp0 = tcp_counters()
    connector = aiohttp.TCPConnector(limit=CONCURRENCY + 1)  # as the rollout worker: keep-alive pool
    async with aiohttp.ClientSession(connector=connector, timeout=aiohttp.ClientTimeout(total=3600)) as session:
        tasks = [asyncio.create_task(worker(session)) for _ in range(CONCURRENCY)]
        th = threading.Thread(target=syncer, args=(t_start,), daemon=True)
        th.start()
        while th.is_alive():
            await asyncio.sleep(1)
        stop.set()
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    # drop whatever is still running on the server before the next arm
    requests.post(URL + "/inference/v1/abort_requests", json={}, timeout=600)
    for _ in range(120):
        running = metric("vllm:num_requests_running")
        if not running:
            break
        time.sleep(1)
    tcp1 = tcp_counters()
    log(
        kind="arm_end",
        running_after_abort=running,
        tcp={k: tcp1[k] - tcp0[k] for k in tcp1 if k in tcp0 and tcp1[k] != tcp0[k] and not k.startswith("Tcp.Curr")},
    )


asyncio.run(main())
