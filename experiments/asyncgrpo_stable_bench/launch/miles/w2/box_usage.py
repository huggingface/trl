"""Local Apptainer boxes of a miles W2 job: every 5 s, as CSV on stdout. A box is a process whose command line holds
the job's box dir (Harbor's singularity staging lives under TMPDIR=<box dir>/tmp) and whose parent's doesn't; its
tree is summed. Columns: unix time, live boxes, box RSS (MB), box CPU seconds of live trees, host memory used (MB), 1-min load.

    python box_usage.py <box dir> [interval_s]
"""

import os
import sys
import time

TICK = os.sysconf("SC_CLK_TCK")
PAGE_MB = os.sysconf("SC_PAGE_SIZE") / 2**20
box_dir, every = sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
print("time,boxes,box_rss_mb,box_cpu_s,host_mem_used_mb,load1", flush=True)
while True:
    procs = {}
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            stat = open(f"/proc/{pid}/stat").read()
            f = stat[stat.rfind(")") + 2 :].split()
            cmd = open(f"/proc/{pid}/cmdline", "rb").read()
            rss = int(open(f"/proc/{pid}/statm").read().split()[1])
        except (OSError, IndexError):
            continue
        procs[int(pid)] = (int(f[1]), box_dir.encode() in cmd, rss * PAGE_MB, (int(f[11]) + int(f[12])) / TICK)
    children = {}
    for pid, (ppid, *_r) in procs.items():
        children.setdefault(ppid, []).append(pid)
    roots = [p for p, (pp, inbox, *_r) in procs.items() if inbox and not (pp in procs and procs[pp][1])]
    rss = cpu = 0.0
    for r in roots:
        stack = [r]
        while stack:
            p = stack.pop()
            rss += procs[p][2]
            cpu += procs[p][3]
            stack.extend(children.get(p, []))
    mem = {l.split(":")[0]: int(l.split()[1]) for l in open("/proc/meminfo")}
    used = (mem["MemTotal"] - mem["MemAvailable"]) / 1024
    print(f"{time.time():.0f},{len(roots)},{rss:.0f},{cpu:.1f},{used:.0f},{os.getloadavg()[0]:.2f}", flush=True)
    time.sleep(every)
