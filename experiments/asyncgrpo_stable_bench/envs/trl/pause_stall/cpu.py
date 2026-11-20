"""Every 2 s, log the CPU share and RSS of each vLLM API server / engine core process (descendants of a pid)."""

import json
import os
import sys
import time


root, out = int(sys.argv[1]), open(sys.argv[2], "a", buffering=1)
tick = os.sysconf("SC_CLK_TCK")


def children(pid):
    pids, todo = [], [pid]
    while todo:
        p = todo.pop()
        pids.append(p)
        try:
            for tid in os.listdir(f"/proc/{p}/task"):
                with open(f"/proc/{p}/task/{tid}/children") as f:
                    todo += [int(c) for c in f.read().split()]
        except OSError:
            pass
    return pids


def sample():
    res = {}
    for p in children(root):
        try:
            name = open(f"/proc/{p}/cmdline").read().replace("\0", " ")[:50]
            stat = open(f"/proc/{p}/stat").read().rsplit(")", 1)[1].split()
            res[p] = (name, (int(stat[11]) + int(stat[12])) / tick, int(stat[21]) * 4096 / 2**30)
        except OSError:
            pass
    return res


prev, tprev = sample(), time.time()
while True:
    time.sleep(2)
    cur, t = sample(), time.time()
    row = {}
    for p, (name, cpu, rss) in cur.items():
        if p in prev:
            row[f"{name}:{p}"] = [round((cpu - prev[p][1]) / (t - tprev), 2), round(rss, 2)]
    out.write(json.dumps({"t": t, "cpu_rss": row}) + "\n")
    prev, tprev = cur, t
