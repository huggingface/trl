"""Samples the CPU and memory of the local sandboxes (W2 with SANDBOX=local) every few seconds, as CSV on stdout.

Every non-Python child of the Harbor server is a box (Harbor's singularity env starts one `singularity exec` per trial;
terminus-2 runs in the server process itself). Columns: unix time, live boxes (the server's direct children), RSS summed over the
boxes' process trees (MB), CPU seconds used by the boxes so far (cumulative, including reaped processes), host memory used
(MB), 1-min load average.

    python box_usage.py <harbor server pid> [interval_s]
"""

import os
import sys
import time


TICK = os.sysconf("SC_CLK_TCK")
PAGE_MB = os.sysconf("SC_PAGE_SIZE") / 2**20


def snapshot():
    procs = {}
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            stat = open(f"/proc/{pid}/stat").read()
            fields = stat[stat.rfind(")") + 2 :].split()
            rss_pages = int(open(f"/proc/{pid}/statm").read().split()[1])
        except (OSError, IndexError):
            continue
        # fields[1] = ppid, [11..14] = utime stime cutime cstime
        procs[int(pid)] = (int(fields[1]), sum(int(x) for x in fields[11:15]), rss_pages)
    return procs


def cmdline(pid):
    try:
        return open(f"/proc/{pid}/cmdline", "rb").read()
    except OSError:
        return b""


def main():
    root = int(sys.argv[1])
    interval = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
    print("time,boxes,box_rss_mb,box_cpu_s,host_used_mb,load1", flush=True)
    while os.path.exists(f"/proc/{root}"):
        procs = snapshot()
        children = {}
        for pid, (ppid, _, _) in procs.items():
            children.setdefault(ppid, []).append(pid)
        # the server's own helper processes (Python) are not boxes
        boxes = [pid for pid in children.get(root, []) if b"python" not in cmdline(pid)]
        stack, cpu, rss = list(boxes), 0, 0
        while stack:
            pid = stack.pop()
            cpu += procs[pid][1]
            rss += procs[pid][2]
            stack.extend(children.get(pid, []))
        # boxes already reaped by the server count in its cutime + cstime
        server = open(f"/proc/{root}/stat").read()
        cpu += sum(int(x) for x in server[server.rfind(")") + 2 :].split()[13:15])
        mem = dict(line.split(":") for line in open("/proc/meminfo"))
        used_mb = (int(mem["MemTotal"].split()[0]) - int(mem["MemAvailable"].split()[0])) / 1024
        load1 = open("/proc/loadavg").read().split()[0]
        print(f"{time.time():.0f},{len(boxes)},{rss * PAGE_MB:.0f},{cpu / TICK:.1f},{used_mb:.0f},{load1}", flush=True)
        time.sleep(interval)


if __name__ == "__main__":
    main()
