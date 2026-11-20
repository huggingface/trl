"""Per-library robustness summary from failures.jsonl (+ log_errors.csv if present).

Writes summary.csv and summary.md next to this script. Validates every ledger row first
and exits non-zero on a bad row, so a malformed append is caught on the next run.
Stdlib only.
"""

import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime


HERE = os.path.dirname(os.path.abspath(__file__))
LIBS = ["trl", "prime-rl", "verl", "skyrl", "miles", "openrlhf", "nemo-rl", "all"]
PHASES = {"install", "setup", "smoke", "recipe", "w1", "eval", "harness", "w2"}
LIB_CATEGORIES = [
    "install_break",
    "missing_dep",
    "docs_or_example_mismatch",
    "unclear_setup",
    "config_trap",
    "crash",
    "hang",
    "silent_correctness",
    "perf_trap",
    "cluster_unfriendly",
]
CATEGORIES = set(LIB_CATEGORIES) | {"our_harness", "infra"}
SEVERITIES = ["blocker", "major", "minor"]
ATTRIBUTABLE = {"lib", "upstream_dep", "ours", "infra", "unknown"}
KINDS = ["failure", "failure_unattributed", "environment", "observation", "ours", "infra"]
FIELDS = [
    "id",
    "date",
    "lib",
    "phase",
    "category",
    "severity",
    "attributable_to",
    "upstream_dep",
    "symptom",
    "root_cause",
    "fix",
    "patched_lib_source",
    "upstream_reportable",
    "time_lost_min",
    "source",
    "kind",
    "retag_note",
]
LOG_CLASSES = [
    "traceback",
    "oom",
    "nccl_error",
    "timeout",
    "retry/disconnect",
    "cuda_error",
    "nan_inf",
    "killed/signal",
    "error_other",
    "warning_deprecation",
    "warning_other",
]
W1_RUN = re.compile(r"-w1-S-(shared|recommended)-\d+$")


def load_ledger(path):
    rows, errors, ids = [], [], set()
    with open(path) as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"line {n}: bad JSON ({e})")
                continue
            missing = [k for k in FIELDS if k not in r]
            if missing:
                errors.append(f"line {n} ({r.get('id')}): missing {missing}")
                continue
            checks = [
                (r["id"] not in ids, f"duplicate id {r['id']}"),
                (r["lib"] in LIBS, f"lib {r['lib']!r}"),
                (r["phase"] in PHASES, f"phase {r['phase']!r}"),
                (r["category"] in CATEGORIES, f"category {r['category']!r}"),
                (r["kind"] in KINDS, f"kind {r['kind']!r}"),
                ((r["severity"] == "info") == (r["kind"] == "observation"), "severity info iff kind observation"),
                (r["severity"] in SEVERITIES + ["info"], f"severity {r['severity']!r}"),
                ((r["kind"] == "ours") == (r["attributable_to"] == "ours"), "kind ours iff attributable_to ours"),
                (r["kind"] != "infra" or r["attributable_to"] == "infra", "kind infra needs attributable_to infra"),
                (
                    (r["kind"] == "failure_unattributed") == (r["attributable_to"] == "unknown"),
                    "kind failure_unattributed iff attributable_to unknown",
                ),
                (bool(r["retag_note"]), "retag_note empty"),
                (r["attributable_to"] in ATTRIBUTABLE, f"attributable_to {r['attributable_to']!r}"),
                (
                    (r["attributable_to"] == "upstream_dep") == bool(r["upstream_dep"]),
                    "upstream_dep must be set iff attributable_to is upstream_dep",
                ),
                (
                    (r["category"] == "our_harness") == (r["attributable_to"] == "ours"),
                    "category our_harness iff attributable_to ours",
                ),
                (
                    (r["category"] == "infra") == (r["attributable_to"] == "infra"),
                    "category infra iff attributable_to infra",
                ),
                (isinstance(r["patched_lib_source"], bool), "patched_lib_source must be bool"),
                (isinstance(r["upstream_reportable"], bool), "upstream_reportable must be bool"),
                (r["time_lost_min"] is None or isinstance(r["time_lost_min"], (int, float)), "time_lost_min"),
                (bool(re.match(r"^\d{4}-\d\d-\d\d(T\d\d:\d\dZ)?$", r["date"])), f"date {r['date']!r}"),
            ]
            bad = [msg for ok, msg in checks if not ok]
            if bad:
                errors.append(f"line {n} ({r['id']}): " + "; ".join(bad))
            ids.add(r["id"])
            rows.append(r)
    return rows, errors


def load_log_totals(path):
    """lib -> Counter(pattern_class -> count) over W1 runs, plus the set of W1 run ids seen."""
    totals, runs = defaultdict(Counter), defaultdict(set)
    if not os.path.exists(path):
        return totals, runs, False
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if W1_RUN.search(r["run_id"]):
                totals[r["lib"]][r["pattern_class"]] += int(r["count"])
                runs[r["lib"]].add(r["run_id"])
    return totals, runs, True


def tsum(rows):
    return sum(r["time_lost_min"] for r in rows if r["time_lost_min"] is not None)


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def main():
    rows, errors = load_ledger(os.path.join(HERE, "failures.jsonl"))
    if errors:
        print("failures.jsonl has invalid rows:\n  " + "\n  ".join(errors), file=sys.stderr)
        sys.exit(1)
    log_totals, w1_runs, have_logs = load_log_totals(os.path.join(HERE, "log_errors.csv"))

    by_lib = defaultdict(list)
    for r in rows:
        by_lib[r["lib"]].append(r)

    header = (
        ["lib", "failures", "blocker", "major", "minor"]
        + LIB_CATEGORIES
        + [
            "patches_needed",
            "upstream_reportable",
            "time_lost_min",
            "time_unknown_rows",
            "failure_unattributed_rows",
            "environment_rows",
            "observation_rows",
            "ours_rows",
            "ours_time_min",
            "infra_rows",
            "infra_time_min",
            "w1_runs_scanned",
        ]
        + [f"w1_log_{c}" for c in LOG_CLASSES]
    )
    table = []
    for lib in LIBS:
        lib_rows = by_lib.get(lib, [])
        kind = Counter(r["kind"] for r in lib_rows)
        head = [r for r in lib_rows if r["kind"] == "failure"]
        ours = [r for r in lib_rows if r["kind"] == "ours"]
        infra = [r for r in lib_rows if r["kind"] == "infra"]
        sev = Counter(r["severity"] for r in head)
        cat = Counter(r["category"] for r in head)
        lt = log_totals.get(lib, Counter())
        table.append(
            [lib, len(head)]
            + [sev[s] for s in SEVERITIES]
            + [cat[c] for c in LIB_CATEGORIES]
            + [
                sum(r["patched_lib_source"] for r in head),
                sum(r["upstream_reportable"] for r in head),
                tsum(head),
                sum(r["time_lost_min"] is None for r in head),
                kind["failure_unattributed"],
                kind["environment"],
                kind["observation"],
                len(ours),
                tsum(ours),
                len(infra),
                tsum(infra),
                len(w1_runs.get(lib, ())),
            ]
            + [lt[c] for c in LOG_CLASSES]
        )

    tmp = os.path.join(HERE, "summary.csv.tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(table)
    os.replace(tmp, os.path.join(HERE, "summary.csv"))

    col = {h: i for i, h in enumerate(header)}
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    md = [
        "# Robustness summary",
        "",
        f"Generated {now} from `failures.jsonl` ({len(rows)} rows) and `log_errors.csv`. "
        "The headline counts only `kind: failure` rows: the library itself broke, hung, crashed, gave wrong numbers, "
        "or its own docs, examples, defaults, lock or image didn't work as written. "
        "`failure_unattributed` (root cause unknown with our code in the failing path), `environment` "
        "(third-party stack problems any library on Qwen3.5 + B200 + torch 2.11-2.14 + cu130 would hit), "
        "`observation` (recovered warnings, cosmetic log lines, limitations not hit, API/design notes), "
        "`ours` (our harness and agents) and `infra` (spot preemption, node failures, cluster limits) are counted separately. "
        "Lib `all` holds problems that hit every library (cluster facts, shared tooling). "
        "Time lost is a rough estimate summed over `failure` rows that have one; `rows w/o time` counts rows without.",
        "",
        "Scrutiny bias: TRL is the subject of the benchmark, so its logs and code were read far more closely than the "
        "others'. Observation counts are therefore not comparable across libraries; only the `failure` counts are "
        "meant for comparison.",
        "",
        "## Headline per library",
        "",
    ]
    md.append(
        md_table(
            [
                "lib",
                "failures",
                "blocker",
                "major",
                "minor",
                "silent correctness",
                "source patches",
                "upstream-reportable",
                "time lost (min)",
                "rows w/o time",
                "failure unattributed",
                "environment",
                "observation",
                "ours",
                "infra",
            ],
            [
                [
                    r[col["lib"]],
                    r[col["failures"]],
                    r[col["blocker"]],
                    r[col["major"]],
                    r[col["minor"]],
                    r[col["silent_correctness"]],
                    r[col["patches_needed"]],
                    r[col["upstream_reportable"]],
                    r[col["time_lost_min"]],
                    r[col["time_unknown_rows"]],
                    r[col["failure_unattributed_rows"]],
                    r[col["environment_rows"]],
                    r[col["observation_rows"]],
                    r[col["ours_rows"]],
                    r[col["infra_rows"]],
                ]
                for r in table
            ],
        )
    )
    md += ["", "## Failures by category", ""]
    md.append(md_table(["lib"] + LIB_CATEGORIES, [[r[0]] + [r[col[c]] for c in LIB_CATEGORIES] for r in table]))
    md += ["", "## Excluded from the headline", ""]
    md.append(
        md_table(
            ["lib", "ours rows", "ours time (min)", "infra rows", "infra time (min)"],
            [
                [r[0], r[col["ours_rows"]], r[col["ours_time_min"]], r[col["infra_rows"]], r[col["infra_time_min"]]]
                for r in table
            ],
        )
    )
    md += ["", "## Log pattern totals for W1 runs", ""]
    if have_logs:
        md.append(
            "Raw line counts from `scan_logs.py` over every log file of the W1-S shared/recommended runs "
            "(runs still in progress are counted as of the scan). Warnings are dominated by a few repeated "
            "messages; see `log_messages_top.csv`."
        )
        md.append("")
        md.append(
            md_table(
                ["lib", "runs"] + LOG_CLASSES,
                [
                    [r[0], r[col["w1_runs_scanned"]]] + [r[col[f"w1_log_{c}"]] for c in LOG_CLASSES]
                    for r in table
                    if r[0] != "all"
                ],
            )
        )
    else:
        md.append("`log_errors.csv` not found; run `scan_logs.sbatch` first.")
    for title, pred in [
        ("Blockers", lambda r: r["severity"] == "blocker"),
        ("Silent correctness", lambda r: r["category"] == "silent_correctness"),
    ]:
        md += ["", f"## {title}", ""]
        sel = [r for r in rows if pred(r)]
        md.append(
            md_table(
                ["id", "lib", "kind", "category", "attributable", "patched", "symptom"],
                [
                    [
                        r["id"],
                        r["lib"],
                        r["kind"],
                        r["category"],
                        r["attributable_to"] + (f" ({r['upstream_dep']})" if r["upstream_dep"] else ""),
                        "yes" if r["patched_lib_source"] else "no",
                        r["symptom"].replace("|", "/"),
                    ]
                    for r in sel
                ],
            )
        )
    md += ["", "## Every row by kind", "", "Why each row got its `kind` (`retag_note` in the ledger).", ""]
    md.append(
        md_table(
            ["id", "kind", "severity", "note"],
            [
                [r["id"], r["kind"], r["severity"], r["retag_note"].replace("|", "/")]
                for k in KINDS
                for r in rows
                if r["kind"] == k
            ],
        )
    )
    md.append("")
    tmp = os.path.join(HERE, "summary.md.tmp")
    with open(tmp, "w") as f:
        f.write("\n".join(md))
    os.replace(tmp, os.path.join(HERE, "summary.md"))
    print(f"summary: {len(rows)} ledger rows, logs {'found' if have_logs else 'missing'}")


if __name__ == "__main__":
    main()
