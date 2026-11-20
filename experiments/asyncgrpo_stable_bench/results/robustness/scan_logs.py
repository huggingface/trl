"""Scan every benchmark run's logs for error/warning patterns.

Writes, next to this script (atomically, so reruns are idempotent):
  log_errors.csv        one row per (run_id, file, pattern_class)
  log_messages_top.csv  top 30 normalized messages per lib
  log_files.csv         every file scanned, its size and whether it was sampled

Stdlib only. Run through scan_logs.sbatch on hopper-cpu, never on the login node.
"""

import argparse
import csv
import os
import re
import sys
import time
from collections import Counter
from multiprocessing import Pool


LIBS = ["trl", "prime-rl", "verl", "skyrl", "miles", "openrlhf"]
HARNESS_DIRS = ["queue", "status", "tests"]  # our own tooling, reported as lib "harness"
LOG_SUFFIXES = (".log", ".out", ".err")
SKIP_DIRS = {
    "hf",
    "ckpt",
    "checkpoints",
    "megatron",
    "models",
    "datasets",
    "data",
    "wandb",
    "broadcasts",
    "imgsrc",
    "torch_compile_cache",
    "__pycache__",
    ".git",
    "weights",
}
SKIP_DIR_PREFIXES = ("checkpoint-", "global_step", "step_")
BIG_FILE = 2 * 1024**3  # files above this are sampled
SAMPLE_BYTES = 256 * 1024**2  # head and tail sample size for big files
TOP_N = 30
EXAMPLE_LEN = 200

CLASSES = [
    "traceback",
    "oom",
    "nccl_error",
    "timeout",
    "retry/disconnect",
    "cuda_error",
    "nan_inf",
    "warning_deprecation",
    "warning_other",
    "killed/signal",
    "error_other",
]

# Cheap byte-level prefilter: a line that doesn't match this can't match any class below.
PREFILTER = re.compile(
    rb"traceback|error|exception|warn|out of memory|oom|nccl|time ?out|timed out|deadline|retr(y|ies|ying)|"
    rb"disconnect|connection|broken ?pipe|nan|inf|killed|sig[a-z]{3,4}|signal|segmentation|core dumped|"
    rb"cancelled|terminated|deprecat|will be removed|cuda|xid|failed|fatal",
    re.IGNORECASE,
)

I = re.IGNORECASE
# Order matters: the first class that matches wins (error_other and warnings last).
RULES = [
    ("oom", re.compile(r"out of memory|OutOfMemoryError|\bOOM\b|oom[-_]kill|Cannot allocate memory", I)),
    (
        "cuda_error",
        re.compile(
            r"CUDA error|cudaError|CUBLAS_STATUS|CUDNN_STATUS|illegal memory access|"
            r"device-side assert|no kernel image is available|NVML_ERROR|\bXid\b"
        ),
    ),
    (
        "nccl_error",
        re.compile(
            r"NCCL (WARN|ERROR)|nccl\w*(Error|Failure)|ProcessGroupNCCL.*(error|timeout|abort)|"
            r"NCCL.*(timeout|timed out|failure|error)",
            I,
        ),
    ),
    (
        "timeout",
        re.compile(
            r"timed out|TimeoutError|\btime ?out (waiting|while|error|expired|exceeded|reached)|"
            r"Timeout (waiting|while|error|expired|exceeded|reached)|DEADLINE_EXCEEDED|deadline exceeded",
            I,
        ),
    ),
    (
        "retry/disconnect",
        re.compile(
            r"\bretry(ing)?\b|\bretries\b|ServerDisconnected|\bdisconnected\b|"
            r"Connection (reset|refused|closed|aborted|lost)|ConnectionResetError|"
            r"ConnectionRefusedError|Broken ?pipe|ECONNRESET|\breconnect",
            I,
        ),
    ),
    (
        "killed/signal",
        re.compile(
            r"\bKilled\b|SIGTERM|SIGKILL|SIGSEGV|SIGABRT|SIGBUS|SIGINT|Segmentation fault|"
            r"core dumped|(received|with|by|on) signal \d+|signal=SIG|\bCANCELLED\b|DUE TO (PREEMPTION|TIME LIMIT|NODE)|"
            r"\bTerminated\b"
        ),
    ),
]
NAN_INF = re.compile(r"(?<![\w.])(-?nan|NaN|-?inf|Infinity)(?![\w])")
NAN_CONTEXT = re.compile(r"loss|grad|norm|reward|logp|logprob|\bkl\b|_kl|ratio|adv|entropy", I)
ERROR_LEVEL = re.compile(r"\b(ERROR|CRITICAL|FATAL)\b")
ERROR_OTHER = re.compile(r"\bERROR\b|\bError\b|\w+Error\b|\w+Exception\b|\bFAILED\b|\bFailed\b|\bfatal\b", 0)
DEPRECATION = re.compile(r"DeprecationWarning|FutureWarning|deprecated|will be removed", I)
WARNING = re.compile(r"\bWARN(ING)?\b|\w*Warning\b|\bwarn(ing)?:", I)

# Long argument/config dumps mention nccl, timeout, inf, ... as setting names; never classify them.
CONFIG_DUMP = re.compile(r"server_args=|[Nn]amespace\(|\bargs=\{|\bconfig=\{")
# vLLM prefixes every traceback line with its log header; strip it before the traceback state machine.
VLLM_PREFIX = re.compile(r"^(DEBUG|INFO|WARNING|ERROR|CRITICAL) \d\d-\d\d \d\d:\d\d:\d\d \[[^\]]*\] ?")
TRACEBACK = re.compile(r"Traceback \(most recent call last\)")
TB_CONTINUE = re.compile(r"^(\s|File \"|During handling of the above|The above exception|\^|~)")

ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
LINE_PREFIX = re.compile(r"^\s*(\([^)]*pid=\d+[^)]*\)\s?|\[rank\d+\]:\s?|\[\d+/\d+\]\s?)+")
TS_PATTERNS = [
    (re.compile(r"(20\d\d)-(\d\d)-(\d\d)[ T](\d\d:\d\d:\d\d)"), lambda m: f"{m[1]}-{m[2]}-{m[3]}T{m[4]}"),
    (re.compile(r"\b(\d\d)-(\d\d) (\d\d:\d\d:\d\d)\b"), lambda m: f"2026-{m[1]}-{m[2]}T{m[3]}"),
    (re.compile(r"\b[IWEF](\d\d)(\d\d) (\d\d:\d\d:\d\d)"), lambda m: f"2026-{m[1]}-{m[2]}T{m[3]}"),
]
NORM_SUBS = [
    (re.compile(r"(20\d\d-\d\d-\d\d[ T]\d\d:\d\d:\d\d[.,\d]*Z?)|(\b\d\d-\d\d \d\d:\d\d:\d\d[.,\d]*)"), "<TS>"),
    (re.compile(r"\b[IWEF]\d{4} \d\d:\d\d:\d\d\.\d+"), "<TS>"),
    (re.compile(r"0x[0-9a-fA-F]+"), "<HEX>"),
    (re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"), "<UUID>"),
    (re.compile(r"\b[0-9a-fA-F]{12,}\b"), "<ID>"),
    (re.compile(r"\d+(\.\d+)?([eE][-+]?\d+)?"), "<N>"),
    (re.compile(r"\s+"), " "),
]
JOBID = re.compile(r"(?<!\d)(36\d{4})(?!\d)")


def clean(line: str) -> str:
    return LINE_PREFIX.sub("", ANSI.sub("", line)).rstrip()


def normalize(line: str) -> str:
    s = line
    for pat, rep in NORM_SUBS:
        s = pat.sub(rep, s)
    return s.strip()[:EXAMPLE_LEN]


def timestamp(line: str) -> str | None:
    for pat, fmt in TS_PATTERNS:
        m = pat.search(line)
        if m:
            return fmt(m)
    return None


def classify(line: str, allow_generic: bool = True) -> str | None:
    if len(line) > 300 and CONFIG_DUMP.search(line):
        return None
    for name, pat in RULES:
        if pat.search(line):
            return name
    if NAN_INF.search(line) and NAN_CONTEXT.search(line):
        return "nan_inf"
    if not allow_generic:
        return None
    if DEPRECATION.search(line):
        return "warning_deprecation"
    # An explicit log level decides first, so "WARNING ... Failed to load" stays a warning.
    if ERROR_LEVEL.search(line):
        return "error_other"
    if WARNING.search(line):
        return "warning_other"
    if ERROR_OTHER.search(line):
        return "error_other"
    return None


def iter_lines(path: str, size: int):
    """Yield raw byte lines; for big files only a head and a tail sample. \\r splits tqdm redraws."""
    spans = [(0, size)] if size <= BIG_FILE else [(0, SAMPLE_BYTES), (size - SAMPLE_BYTES, size)]
    with open(path, "rb") as f:
        for start, end in spans:
            f.seek(start)
            remaining = end - start
            buf = b""
            while remaining > 0:
                chunk = f.read(min(8 * 1024**2, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                buf += chunk
                parts = re.split(rb"[\r\n]", buf)
                buf = parts.pop()
                yield from parts
            if buf:
                yield buf


def scan_file(task):
    lib, run_id, rel, path, size = task
    stats = {}  # class -> [count, first_seen, last_seen, example]
    msgs = Counter()
    msg_example = {}
    in_tb = False
    n = 0

    def hit(cls, text, lineno):
        ts = timestamp(text) or f"line:{lineno}"
        st = stats.get(cls)
        if st is None:
            stats[cls] = [1, ts, ts, text.strip()[:EXAMPLE_LEN]]
        else:
            st[0] += 1
            st[2] = ts
        key = (cls, normalize(text))
        msgs[key] += 1
        msg_example.setdefault(key, text.strip()[:EXAMPLE_LEN])

    try:
        for raw in iter_lines(path, size):
            n += 1
            if not raw or not PREFILTER.search(raw):
                if in_tb and raw.strip():
                    line = clean(raw.decode("utf-8", "replace"))
                    if not TB_CONTINUE.match(VLLM_PREFIX.sub("", line)):
                        in_tb = False
                        hit("traceback", line, n)  # exception line ends the traceback
                        cls = classify(line, allow_generic=False)
                        if cls:
                            hit(cls, line, n)
                continue
            line = clean(raw[:4000].decode("utf-8", "replace"))
            if TRACEBACK.search(line):
                in_tb = True
                continue
            if in_tb:
                tb_line = VLLM_PREFIX.sub("", line)
                if TB_CONTINUE.match(tb_line) or not tb_line.strip():
                    continue
                in_tb = False
                hit("traceback", line, n)
                cls = classify(line, allow_generic=False)
                if cls:
                    hit(cls, line, n)
                continue
            cls = classify(line)
            if cls:
                hit(cls, line, n)
    except OSError as e:
        print(f"skip {path}: {e}", file=sys.stderr)
    top_msgs = msgs.most_common(500)
    return lib, run_id, rel, size, n, stats, [(k, c, msg_example[k]) for k, c in top_msgs]


def run_dir_map(lib_dir: str) -> dict:
    """Job id -> run dir name, from top-level dirs like trl-w1-S-shared-364684 or ...-364425-aux."""
    out = {}
    for name in os.listdir(lib_dir):
        m = re.search(r"(36\d{4})(-aux)?$", name)
        if m and os.path.isdir(os.path.join(lib_dir, name)):
            out.setdefault(m[1], name.removesuffix("-aux"))
    return out


def collect_tasks(root: str):
    tasks, seen = [], set()
    for lib in LIBS + HARNESS_DIRS:
        lib_dir = os.path.join(root, lib)
        if not os.path.isdir(lib_dir):
            continue
        lib_name = lib if lib in LIBS else "harness"
        jobmap = run_dir_map(lib_dir)
        for dirpath, dirnames, filenames in os.walk(lib_dir):
            dirnames[:] = [
                d
                for d in dirnames
                if d not in SKIP_DIRS
                and not d.startswith(SKIP_DIR_PREFIXES)
                and not os.path.islink(os.path.join(dirpath, d))
            ]
            for fn in filenames:
                if not fn.endswith(LOG_SUFFIXES):
                    continue
                path = os.path.join(dirpath, fn)
                if os.path.islink(path):
                    continue
                real = os.path.realpath(path)
                if real in seen:
                    continue
                seen.add(real)
                rel = os.path.relpath(path, root)
                top = os.path.relpath(path, lib_dir).split(os.sep)[0]
                if top != fn and re.search(r"36\d{4}(-aux)?$", top):
                    run_id = top.removesuffix("-aux")
                else:
                    m = JOBID.search(rel)
                    run_id = jobmap.get(m[1]) if m else None
                    run_id = run_id or f"{lib_name}:{os.path.splitext(fn)[0]}"
                tasks.append((lib_name, run_id, rel, path, os.path.getsize(path)))
    tasks.sort(key=lambda t: -t[4])
    return tasks


def write_csv(path: str, header: list, rows: list):
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/fsx/amine_dirhoussi/stable-bench-runs")
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--procs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    args = ap.parse_args()

    t0 = time.time()
    tasks = collect_tasks(args.root)
    total = sum(t[4] for t in tasks)
    print(f"{len(tasks)} log files, {total / 1e9:.2f} GB, {args.procs} procs", flush=True)

    err_rows, file_rows = [], []
    lib_msgs = {}  # lib -> Counter[(cls, msg)], plus example/run/file count
    with Pool(args.procs) as pool:
        for lib, run_id, rel, size, n, stats, top in pool.imap_unordered(scan_file, tasks):
            file_rows.append([lib, run_id, rel, size, size > BIG_FILE, n])
            for cls in CLASSES:
                if cls in stats:
                    c, first, last, ex = stats[cls]
                    err_rows.append([run_id, lib, rel, cls, c, first, last, ex])
            agg = lib_msgs.setdefault(lib, {"count": Counter(), "files": Counter(), "example": {}})
            for key, c, ex in top:
                agg["count"][key] += c
                agg["files"][key] += 1
                agg["example"].setdefault(key, (run_id, ex))

    err_rows.sort(key=lambda r: (r[1], r[0], r[2], CLASSES.index(r[3])))
    file_rows.sort(key=lambda r: (r[0], r[1], r[2]))
    msg_rows = []
    for lib in sorted(lib_msgs):
        agg = lib_msgs[lib]
        for rank, (key, c) in enumerate(agg["count"].most_common(TOP_N), 1):
            run_id, ex = agg["example"][key]
            msg_rows.append([lib, rank, key[0], key[1], c, agg["files"][key], run_id, ex])

    write_csv(
        os.path.join(args.out, "log_errors.csv"),
        ["run_id", "lib", "file", "pattern_class", "count", "first_seen", "last_seen", "example"],
        err_rows,
    )
    write_csv(
        os.path.join(args.out, "log_messages_top.csv"),
        ["lib", "rank", "pattern_class", "message", "count", "n_files", "example_run", "example"],
        msg_rows,
    )
    write_csv(
        os.path.join(args.out, "log_files.csv"),
        ["lib", "run_id", "file", "size_bytes", "sampled", "lines_scanned"],
        file_rows,
    )
    print(f"wrote {len(err_rows)} error rows, {len(msg_rows)} message rows in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
