"""Every benchmark figure. Run figures/build.sh, which builds the data first; never run this on the login node.

    python figures/charts.py [--diagnostics --runs REGEX]

Blog figures (default): figures/post/<NN_name>.{svg,png,csv} from figures/data/bench.json (figures/build_data.py), the
same charts as figures/interactive/index.html. PNGs are 2x.

Diagnostics (--diagnostics): the per-workload charts a-l in figures/out/<workload>/ from results/raw/<run_id>/
(collect/SCHEMA.md), with the plotted data as results/tables/<workload>/<chart>.csv. A field that is null for a run is
skipped for that run, and the chart says so in a note under the plot.
"""

import argparse
import glob
import json
import os
import re
import textwrap

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import matplotlib.patches  # noqa: E402
import matplotlib.ticker  # noqa: E402


BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(BENCH, "results", "raw")
TABLES = os.path.join(BENCH, "results", "tables")
OUT = os.path.join(BENCH, "figures", "out")
SKIP_STEPS = 5  # warmup steps left out of every steady-state number
TIMELINE_S = 300

# One color per library in every chart (categorical slots 1-6 of the dataviz reference palette, fixed order).
LIB_COLOR = {
    "trl": "#2a78d6",
    "prime-rl": "#eb6834",
    "verl": "#1baf7a",
    "skyrl": "#eda100",
    "miles": "#e87ba4",
    "openrlhf": "#008300",
}
TAG_STYLE = {"shared": "-", "recommended": "--"}
OTHER_STYLES = [":", "-.", (0, (5, 1, 1, 1))]
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#fcfcfb"
SEQ_BLUE = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
LIB_ORDER = ["trl", "prime-rl", "verl", "miles", "openrlhf", "skyrl"]
# Step-time components (validated as a set: violet / red / amber pass CVD and normal-vision separation); "other" is the
# unattributed rest, so it stays a recessive hatched gray. Rows name the library, so these hues never stand for one.
PARTS = ["train_compute", "wait_for_rollouts", "weight_sync", "other"]
PART_COLOR = {"train_compute": "#6250d6", "wait_for_rollouts": "#e34948", "weight_sync": "#eda100", "other": "#dddcd6"}
PART_LABEL = {"train_compute": "train compute", "wait_for_rollouts": "waiting for rollouts", "weight_sync": "weight sync",
              "other": "other / not logged"}  # fmt: skip
EMA_ALPHA = 0.1

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "font.size": 9,
        "svg.fonttype": "none",
    }
)


# ---------------------------------------------------------------- loading


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def read_gpu_csv(path):
    """gpu.csv -> DataFrame(node, t, gpu, util, mem_gb). Columns are found by name: their order differs per launcher."""
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, skipinitialspace=True, dtype=str)
    col = lambda prefix: next(c for c in df.columns if c.strip().startswith(prefix))  # noqa: E731
    num = lambda c: pd.to_numeric(df[col(c)].str.extract(r"([-\d.]+)")[0], errors="coerce")  # noqa: E731
    out = pd.DataFrame(
        {
            "node": df[col("node")].str.strip() if any(c.strip() == "node" for c in df.columns) else "node0",
            "t": pd.to_datetime(df[col("timestamp")].str.strip(), format="%Y/%m/%d %H:%M:%S.%f", errors="coerce"),
            "gpu": num("index"),
            "util": num("utilization.gpu"),
            "mem_gb": num("memory.used") * 2**20 / 1e9,  # MiB -> GB (1e9 bytes, the unit of peak_mem_gb)
        }
    )
    out = out.dropna(subset=["t", "gpu"])
    out["gpu"] = out["gpu"].astype(int)
    return out


def load_runs(pattern):
    runs = []
    for d in sorted(glob.glob(os.path.join(RAW, "*", ""))):
        run_id = os.path.basename(d.rstrip("/"))
        if pattern and not re.search(pattern, run_id):
            continue
        meta = json.load(open(os.path.join(d, "meta.json"))) if os.path.exists(os.path.join(d, "meta.json")) else {}
        lib = meta.get("lib") or run_id.split("-w")[0]
        rest = run_id[len(lib) + 1 :].split("-")  # <workload>-<size>-<tag...>-<job id>
        tag = "-".join(rest[2:-1]) if len(rest) >= 4 else "?"
        steps = pd.DataFrame(read_jsonl(os.path.join(d, "steps.jsonl")))
        if len(steps):
            steps = steps.drop(columns=["native"], errors="ignore").sort_values("step").reset_index(drop=True)
        runs.append(
            {
                "id": run_id,
                "lib": lib,
                "tag": tag,
                "label": f"{lib} {rest[1] if len(rest) > 1 else ''} {tag} #{rest[-1]}"
                + (" (patched lib)" if str(meta.get("validity", "")).startswith("patched") else ""),
                "meta": meta,
                "steps": steps,
                "eval": pd.DataFrame(read_jsonl(os.path.join(d, "eval.jsonl"))),
                "spans": os.path.join(d, "spans.jsonl"),
                "gpu": read_gpu_csv(os.path.join(d, "gpu.csv")),
            }
        )
    # Line style: shared solid, recommended dashed; other tags (smoke, ablation arms) take the next free style per lib.
    for lib in {r["lib"] for r in runs}:
        mine = [r for r in runs if r["lib"] == lib]
        free = OTHER_STYLES if any(r["tag"] in TAG_STYLE for r in mine) else ["-"] + OTHER_STYLES
        others = [r for r in mine if r["tag"] not in TAG_STYLE]
        for i, r in enumerate(others):
            r["ls"] = free[i % len(free)]
        for r in mine:
            r.setdefault("ls", TAG_STYLE.get(r["tag"]))
    return runs


def col(df, name):
    """The column as floats, or None when the run never logged it (missing or all null)."""
    if df is None or not len(df) or name not in df.columns:
        return None
    s = pd.to_numeric(df[name], errors="coerce")
    return None if s.isna().all() else s


# ---------------------------------------------------------------- helpers


# One output folder per workload, named after what was trained, and a header line on every chart of that workload.
_W1 = ("single-turn math ({size}) | {model}, full fine-tune, thinking off | train: DAPO-Math-17k, 32 prompts x 8 samples per step, "
       "max completion 16k | eval: AIME24 / AIME25 avg@16, MATH-500 | 1 node, 8 x B200: 4 inference + 4 training GPUs")
WORKLOADS = {
    ("w1", "S"): ("w1-S_math_qwen3.5-2B_dapo-math-17k", "W1-S: " + _W1.format(size="S", model="Qwen3.5-2B")),
    ("w1", "M"): ("w1-M_math_qwen3.5-9B_dapo-math-17k", "W1-M: " + _W1.format(size="M", model="Qwen3.5-9B")),
    ("w2", "M"): ("w2-M_agentic_qwen3.5-9B_termgrade",
                  "W2-M: multi-turn agentic terminal tasks | Qwen3.5-9B, full fine-tune, thinking off | TermGrade (28 train / 14 eval tasks) "
                  "in sandboxes (HF Sandboxes, local containers or the lib's own runtime: see run), terminus-2 agent, <= 30 calls and 600 s per episode | 8 tasks x 8 episodes per step | "
                  "1 node, 8 x B200: 4 inference + 4 training GPUs"),
    ("repro", "pause"): ("vllm-pause-repro_qwen3.5-2B",
                         "Standalone vLLM 0.33 repro, no trainer | Qwen3.5-2B | 688 requests in flight, /pause + /resume every 20 s, 600 s per arm (job 365055)"),
}  # fmt: skip
CTX = {"dir": "", "banner": None}  # set by main() per workload


def save(fig, name, table=None):
    out, tables = os.path.join(OUT, CTX["dir"]), os.path.join(TABLES, CTX["dir"])
    os.makedirs(out, exist_ok=True)
    os.makedirs(tables, exist_ok=True)
    if CTX["banner"]:  # just above the topmost drawn element; bbox_inches="tight" grows the canvas to fit it
        fig.canvas.draw()
        rd = fig.canvas.get_renderer()
        boxes = [ax.get_tightbbox(rd) for ax in fig.axes if ax.get_visible()] + [t.get_window_extent(rd) for t in fig.texts]
        boxes += [lg.get_window_extent(rd) for lg in fig.legends] + ([fig._suptitle.get_window_extent(rd)] if fig._suptitle else [])
        top = max(b.y1 for b in boxes) / fig.bbox.height
        width = max(60, int(fig.get_figwidth() * 15))  # characters per line at 8.5 pt
        fig.text(0.01, top + 0.012, textwrap.fill(CTX["banner"], width), ha="left", va="bottom", fontsize=8.5, color=INK,
                 bbox={"boxstyle": "round,pad=0.4", "fc": GRID, "ec": "none"})  # fmt: skip
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=160 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)
    if table is not None:
        table.to_csv(os.path.join(tables, f"{name}.csv"), index=False)
    print(f"wrote {CTX['dir']}/{name}: {0 if table is None else len(table)} rows")


def notes(fig, lines):
    """Notes go under the lowest drawn element (tick labels, x label, legend) so they never overlap it."""
    if lines:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bottom = min(ax.get_tightbbox(renderer).y0 for ax in fig.axes if ax.get_visible()) / fig.bbox.height
        fig.text(0.01, bottom - 0.02, "\n".join(lines), ha="left", va="top", fontsize=7.5, color=INK2)


def legend_runs(ax, runs, **kw):
    handles = [Line2D([], [], color=LIB_COLOR.get(r["lib"], MUTED), ls=r["ls"], lw=2, label=r["label"]) for r in runs]
    if handles:
        ax.legend(handles=handles, **kw)


def hours_since_first_step(r, t):
    t0 = r["steps"]["time"].iloc[0]
    return (np.asarray(t, dtype=float) - t0) / 3600


def steady(r):
    s = r["steps"]
    return s[s["step"] > SKIP_STEPS] if len(s) > SKIP_STEPS else s.iloc[0:0]


def no_data(ax, msg="no run has this field"):
    ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes, color=INK2)


def run_order(runs):
    """Library order of LIB_ORDER, shared before recommended, then the rest by tag."""
    rank = {"shared": 0, "recommended": 1}
    return sorted(
        runs,
        key=lambda r: (LIB_ORDER.index(r["lib"]) if r["lib"] in LIB_ORDER else 99, rank.get(r["tag"], 2), r["id"]),
    )


def ema(y):
    return y.ewm(alpha=EMA_ALPHA).mean()


def mean_parts(r, min_step):
    """Mean seconds per optimizer step of each PARTS component over steps >= min_step, or None without step_time.
    Returns (parts dict, n steps, notes)."""
    st = r["steps"]
    s = st[st["step"] >= min_step].copy()
    msgs = []
    # A row can span several optimizer steps (verl recommended logs once per 4); report seconds per optimizer step.
    span = st["step"].diff().fillna(st["step"].iloc[0]).loc[s.index]
    for c in ["step_time"] + [f"{p}_time" for p in PARTS[:-1]]:
        if c in s:
            s[c] = pd.to_numeric(s[c], errors="coerce") / span
    if (span > 1).any():
        msgs.append(f"{r['label']}: one row per {int(span.max())} optimizer steps, times divided by {int(span.max())}")
    total = col(s, "step_time")
    if total is None:
        return None, 0, msgs + [f"{r['label']}: no step_time from step {min_step}, no bar"]
    s = s[total.notna()]
    vals = {}
    for p in PARTS[:-1]:
        v = col(s, f"{p}_time")
        vals[p] = 0.0 if v is None else float(v.mean())
        if v is None:
            msgs.append(f"{r['label']}: {p}_time not logged (counted in other)")
    known = sum(vals.values())
    vals["other"] = float(total.mean()) - known
    if vals["other"] < 0:
        if vals["other"] < -0.5:
            msgs.append(f"{r['label']}: components sum to {known:.1f}s > mean step_time {total.mean():.1f}s (overlap)")
        vals["other"] = 0.0
    vals_logged = col(s, "weight_sync_time") is not None
    return vals | {"_sync_logged": vals_logged}, len(s), msgs


def lib_labels(ax, rows):
    """Run labels on the y axis in ink, with the library color as a square beside each."""
    ax.set_yticks(range(len(rows)), [r["label"] for r in rows])
    for i, r in enumerate(rows):
        ax.plot([-0.015], [i], marker="s", ms=7, color=LIB_COLOR.get(r["lib"], MUTED), transform=ax.get_yaxis_transform(),
                clip_on=False)  # fmt: skip
    ax.tick_params(axis="y", pad=14, length=0)


def stacked_parts(ax, rows, scale_key=None, fmt="{:.1f} s"):
    """Horizontal stacked bars of PARTS per row; with scale_key every component is divided by row[scale_key]."""
    for i, r in enumerate(rows):
        k = 1.0 / r[scale_key] if scale_key else 1.0
        left = 0.0
        for p in PARTS:
            v = r["parts"][p] * k
            if v > 0:
                ax.barh(i, v, left=left, height=0.62, color=PART_COLOR[p], edgecolor=SURFACE, linewidth=1.5,
                        hatch="////" if p == "other" else None)  # fmt: skip
                left += v
        ax.text(left, i, "  " + fmt.format(left), va="center", fontsize=7.5, color=INK)
    ax.grid(axis="y", visible=False)


def breakdown_figure(rows, title, tok_title, tok_big, tok_small, per_title, per_key, name, msgs):
    """Three aligned panels, one row per run: tokens per step (big vs small bar), seconds per step by component, and
    seconds per million `per_key` tokens by component. rows: dicts with label, lib, parts, big, small (tokens/step)."""
    for r in rows:
        r["sync_logged"] = r["parts"].pop("_sync_logged", True)
    n = len(rows)
    has_fwd = any("fwd" in r for r in rows)
    fig, axes = plt.subplots(1, 4, figsize=(17.5, 0.48 * max(n, 2) + 2.3), sharey=True,
                             gridspec_kw={"width_ratios": [1, 1.25, 1.25, 0.6]})  # fmt: skip
    a, b, c, d = axes
    for i, r in enumerate(rows):
        col_ = LIB_COLOR.get(r["lib"], MUTED)
        a.barh(i, r["big"] / 1e6, height=0.62, color=col_, alpha=0.3, lw=0)
        if r.get("fwd") is not None:
            # What the GPUs actually ran: after prefix dedup, plus padding. Outline, so it shows past the trained bar.
            a.barh(i, r["fwd"] / 1e6, height=0.78, fill=False, edgecolor=INK, lw=1.0, ls=(0, (2, 1.5)))
        if r.get("small") is not None:
            a.barh(i, r["small"] / 1e6, height=0.62, color=col_, lw=0)
            lbl = f"  {r['small'] / 1e6:.2f}M / {r['big'] / 1e6:.2f}M"
        else:
            lbl = f"  {r['big'] / 1e6:.2f}M"
        if r.get("fwd") is not None:
            lbl += f" / {r['fwd'] / 1e6:.2f}M"
        a.text(max(r["big"], r.get("fwd") or 0) / 1e6, i, lbl, va="center", fontsize=7.5, color=INK)
    a.grid(axis="y", visible=False)
    a.set_xlim(0, max(max(r["big"], r.get("fwd") or 0) for r in rows) / 1e6 * 1.6 if rows else 1)
    a.set(xlabel="million tokens per optimizer step", title=tok_title)
    lib_labels(a, rows)
    stacked_parts(b, rows)
    b.set(xlabel="seconds per optimizer step (mean)", title="Step time")
    b.set_xlim(0, max(sum(r["parts"].values()) for r in rows) * 1.18 if rows else 1)
    stacked_parts(c, rows, scale_key=per_key, fmt="{:.1f} s")
    c.set(xlabel=f"seconds per million {per_title} tokens", title=f"Step time per million {per_title} tokens")
    c.set_xlim(0, max(sum(r["parts"].values()) / r[per_key] for r in rows) * 1.18 if rows else 1)
    # Weight sync is 0.2-2 s of a 26-70 s step, too thin to see in the stacks: its own panel and scale.
    for i, r in enumerate(rows):
        v = r["parts"]["weight_sync"]
        if r.get("sync_logged", True):
            d.barh(i, v, height=0.62, color=PART_COLOR["weight_sync"], lw=0)
            d.text(v, i, f"  {v:.2f} s", va="center", fontsize=7.5, color=INK)
        else:
            d.text(0, i, "  not logged", va="center", fontsize=7.5, color=MUTED)
    d.grid(axis="y", visible=False)
    d.set_xlim(0, max([r["parts"]["weight_sync"] for r in rows] + [0.1]) * 1.6)
    d.set(xlabel="seconds per optimizer step (mean)", title="Weight sync alone")
    a.invert_yaxis()
    fig.suptitle(title, fontweight="bold", color=INK, y=1.0)
    tok_h = [plt.Rectangle((0, 0), 1, 1, color=MUTED), plt.Rectangle((0, 0), 1, 1, color=MUTED, alpha=0.3)]
    tok_l = [tok_small, tok_big]
    if has_fwd:
        tok_h.append(plt.Rectangle((0, 0), 1, 1, fill=False, ec=INK, ls=(0, (2, 1.5))))
        tok_l.append("forwarded by the GPUs (after prefix dedup, incl. padding)")
    fig.legend(handles=tok_h, labels=tok_l, loc="upper center", bbox_to_anchor=(0.16, 0.965), ncol=1, fontsize=7.5)
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, fc=PART_COLOR[p], hatch="////" if p == "other" else None, ec=SURFACE)
                        for p in PARTS], labels=[PART_LABEL[p] for p in PARTS], loc="upper center",
               bbox_to_anchor=(0.6, 0.965), ncol=4)  # fmt: skip
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    notes(fig, msgs)
    save(fig, name, pd.DataFrame([{k: v for k, v in r.items() if k != "parts"} | r["parts"] for r in rows]))


def read_fair():
    """results/tables/throughput_fair.csv (collect/throughput_fair.py over raw/): one trained-token definition for every
    library, and the useful (nonzero-advantage) share. Indexed by run_id."""
    p = os.path.join(TABLES, "throughput_fair.csv")
    return pd.read_csv(p).set_index("run_id") if os.path.exists(p) else pd.DataFrame()


def gpu_roles(g, n_engine):
    """Split a run's GPUs into (train, engine) device lists. gpu.csv has no roles; the inference engine preallocates its
    KV cache up to its memory setting, so its GPUs are the n_engine with the highest median memory after the first 10%
    of the run (checked by hand on every W1 run: vLLM/SGLang 143-180 GB flat vs trainers 26-70 GB, or verl's trainer
    swinging between 50 and 191 GB around an engine flat at 167-175)."""
    g = g.assign(dev=g["node"] + ":" + g["gpu"].astype(str))
    late = g[g["t"] >= g["t"].min() + (g["t"].max() - g["t"].min()) * 0.1]
    med = late.groupby("dev")["mem_gb"].median().sort_values(ascending=False)
    engine = list(med.index[:n_engine]) if n_engine else []
    return sorted(d for d in med.index if d not in engine), engine, g


# ---------------------------------------------------------------- charts


def reward_x(r, x):
    """x values of a run's steps: hours since the first step, optimizer step, or cumulative generated tokens (billions)."""
    s = r["steps"]
    if x == "hours":
        return hours_since_first_step(r, s["time"])
    if x == "step":
        return s["step"].to_numpy(dtype=float)
    g = col(s, "gen_tokens")
    return None if g is None else g.fillna(0).cumsum().to_numpy() / 1e9


def reward_curves(runs, x, name, xlabel, title):
    """(a), (b), (b2): reward_mean, one panel per library. Raw thin, EMA from step SKIP_STEPS + 1 thick; every other
    library's EMA in light gray behind for context."""
    libs = [lib for lib in LIB_ORDER if any(r["lib"] == lib for r in runs)]
    ncol = max(1, min(3, len(libs)))
    nrow = max(1, -(-len(libs) // ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.1 * nrow + 0.9), sharex=True, sharey=True, squeeze=False)
    rows, missing, curves = [], [], {}
    for r in runs:
        y, xs = col(r["steps"], "reward_mean"), reward_x(r, x)
        if y is None or xs is None:
            missing.append(f"no {'reward_mean' if y is None else 'gen_tokens'}: {r['label']}")
            continue
        late = (r["steps"]["step"] > SKIP_STEPS).to_numpy()
        e = pd.Series(np.nan, index=y.index)
        e[late] = ema(y[late])
        curves[r["id"]] = (r, xs, y, e)
        rows += [
            {"run_id": r["id"], "lib": r["lib"], x: a, "reward_mean": b, "reward_ema": v}
            for a, b, v in zip(xs, y, e, strict=False)
        ]
    for ax, lib in zip(axes.flat, libs, strict=False):
        for rid, (r, xs, y, e) in curves.items():  # context first, so the library's own lines sit on top
            if r["lib"] != lib and r["tag"] == "shared":
                ax.plot(xs, e, color=MUTED, lw=1, alpha=0.45)
        names = []
        for rid, (r, xs, y, e) in curves.items():
            if r["lib"] != lib:
                continue
            c = LIB_COLOR.get(lib, MUTED)
            ax.plot(xs, y, color=c, lw=0.7, alpha=0.3, ls=r["ls"])
            ax.plot(xs, e, color=c, lw=2.2, ls=r["ls"])
            names.append(f"{'solid' if r['ls'] == '-' else 'dashed' if r['ls'] == '--' else 'dotted'} {r['tag']} "
                         f"#{r['id'].rsplit('-', 1)[1]}")  # fmt: skip
        if x == "step":
            ax.axvspan(0, SKIP_STEPS + 0.5, color=GRID, alpha=0.6, lw=0)
        ax.set_title(lib, loc="left", fontsize=10)
        ax.text(0.99, 0.03, "\n".join(names), transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color=INK2)
        ax.set_ylim(0, 1)
    for ax in axes.flat[len(libs) :]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel(xlabel)
    for ax in axes[:, 0]:
        ax.set_ylabel("train reward_mean")
    fig.suptitle(title, fontweight="bold", color=INK)
    fig.tight_layout()
    notes(fig, [(f"shaded: steps 1-{SKIP_STEPS}. " if x == "step" else "")
                + f"thick: EMA (alpha {EMA_ALPHA}) from step {SKIP_STEPS + 1}; thin: raw per step; gray: other libraries' shared arm (EMA)",
                ] + (["early spike: an async lib's first steps train on the first rollouts to finish, which are short and mostly solved "
                "(TRL step 1: reward 0.86 at 4.8k mean tokens, ~11k later)"] if CTX["dir"].startswith("w1-S") else []) + missing)  # fmt: skip
    save(fig, name, pd.DataFrame(rows))


def eval_curves(runs):
    """(c): eval metrics vs optimizer step (top row) and vs wall-clock hours (bottom row)."""
    metrics = [
        ("aime24_avg16", "AIME24 avg@16"),
        ("aime25_avg16", "AIME25 avg@16"),
        ("math500_avg1", "MATH-500 avg@1"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey="col")
    rows, missing, plotted = [], [], []
    for r in runs:
        e = r["eval"]
        if not len(e):
            missing.append(f"no eval.jsonl: {r['label']}")
            continue
        e = e.sort_values("step")
        plotted.append(r)
        c = LIB_COLOR.get(r["lib"], MUTED)
        # eval.jsonl's time is when the eval ran; a checkpoint's wall-clock is the time of its optimizer step in steps.jsonl
        st = r["steps"]
        hrs = hours_since_first_step(r, np.interp(e["step"], st["step"], st["time"])) if len(st) else None
        for j, (k, _) in enumerate(metrics):
            y = col(e, k)
            if y is None:
                missing.append(f"no {k}: {r['label']}")
                continue
            axes[0, j].plot(e["step"], y, color=c, ls=r["ls"], lw=2, marker="o", ms=5)
            if hrs is not None:
                axes[1, j].plot(hrs, y, color=c, ls=r["ls"], lw=2, marker="o", ms=5)
        for i, (_, row) in enumerate(e.iterrows()):
            hours = None if hrs is None else hrs[i]
            rows.append(
                {
                    "run_id": r["id"],
                    "lib": r["lib"],
                    "step": row["step"],
                    "hours": hours,
                    **{k: row.get(k) for k, _ in metrics},
                }
            )
    base_model = "Qwen3.5-9B" if any("-w1-M-" in r["id"] for r in runs) else "Qwen3.5-2B"
    base = pd.DataFrame(read_jsonl(os.path.join(RAW, f"base-{base_model}", "eval.jsonl")))
    for j, (k, t) in enumerate(metrics):
        if len(base):
            for i in range(2):
                axes[i, j].axhline(base[k].iloc[0], color=MUTED, lw=1)
            axes[0, j].text(1.0, base[k].iloc[0], f"base {base[k].iloc[0]:.3f} ", va="bottom", ha="right", fontsize=7,
                            color=INK2, transform=axes[0, j].get_yaxis_transform())  # fmt: skip
        axes[0, j].set(title=t, xlabel="optimizer step")
        axes[1, j].set(xlabel="hours since first optimizer step")
        if not plotted:
            no_data(axes[0, j])
    axes[0, 0].set_ylabel("accuracy")
    axes[1, 0].set_ylabel("accuracy")
    fig.suptitle("Eval accuracy per checkpoint", fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    handles = [
        Line2D([], [], color=LIB_COLOR.get(r["lib"], MUTED), ls=r["ls"], lw=2, label=r["label"]) for r in plotted
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0))
    fig.text(0.01, -0.005, "\n".join(["bottom row: x = wall-clock of the checkpoint's optimizer step (steps.jsonl); grey line = base " + base_model]
             + missing), ha="left", va="top", fontsize=7.5, color=INK2)  # fmt: skip
    save(fig, "c_eval", pd.DataFrame(rows))


def forwarded_tokens(r, all_tokens, f, st):
    """Tokens per step the training GPUs actually forwarded (after prefix dedup, incl. padding), W1."""
    if r["lib"] == "trl":
        native = [x["native"] for x in read_jsonl(os.path.join(RAW, r["id"], "steps.jsonl")) if x["step"] > SKIP_STEPS]
        return float(np.mean([x["batch/forwarded_tokens_per_step"] for x in native]))
    if r["lib"] == "skyrl":
        return float(f["padded_forward_tok_s"]) * st
    return all_tokens


def step_breakdown(runs, fair):
    """(d): batch size, step time and step time per million useful tokens, W1 runs, steps > SKIP_STEPS. Tokens per
    step come from throughput_fair.csv (one trained-token definition across libraries; useful = nonzero advantage)."""
    rows, msgs = [], []
    for r in run_order([r for r in runs if "-w1-" in r["id"]]):
        if r["id"] not in fair.index:
            msgs.append(f"{r['label']}: not in throughput_fair.csv, no bar")
            continue
        parts, n, m = mean_parts(r, SKIP_STEPS + 1)
        msgs += m
        if parts is None:
            continue
        f = fair.loc[r["id"]]
        st = float(f["step_time_s"])
        row = {"run_id": r["id"], "label": r["label"], "lib": r["lib"], "steps": n, "parts": parts,
               "big": f["all_trained_tok_s"] * st, "small": f["useful_trained_tok_s"] * st,
               "useful_M": f["useful_trained_tok_s"] * st / 1e6, "useful_exact": bool(f["useful_is_exact"])}  # fmt: skip
        rows.append(row | {"fwd": forwarded_tokens(r, row["big"], f, st)})
    est = [r["label"] for r in rows if not r["useful_exact"]]
    msgs = ["tokens: prompt + completion of every trained sample (throughput_fair.csv); useful = samples with nonzero advantage. "
            "TRL and prime-rl drop zero-advantage groups before the trainer, so their useful = all",
            "forwarded: TRL logged (tree packing forwards each group's prompt once; inter-rank padding is stripped before the forward). "
            "verl, miles, prime-rl pack with no padding and OpenRLHF runs 1 unpadded sample per micro-batch, so forwarded = trained. "
            "SkyRL pads each sample to the step's max width",
            "useful share estimated from reward mean (not logged): " + ", ".join(est) if est else ""] + msgs  # fmt: skip
    breakdown_figure(rows, f"Step time vs batch size (steps > {SKIP_STEPS})", "Tokens per step",
                     "all trained", "useful (nonzero advantage)", "useful", "useful_M", "d_step_breakdown", [m for m in msgs if m])  # fmt: skip


def throughput(runs, fair):
    """(e): trained tokens/s/GPU from throughput_fair.csv, useful (nonzero advantage) solid, the rest light."""
    rows = []
    for r in run_order([r for r in runs if r["id"] in fair.index]):
        f, n = fair.loc[r["id"]], r["meta"].get("gpus_total") or 8
        rows.append({"run_id": r["id"], "label": r["label"], "lib": r["lib"], "gpus_total": n,
                     "all_tok_s_gpu": f["all_trained_tok_s"] / n, "useful_tok_s_gpu": f["useful_trained_tok_s"] / n,
                     "generated_tok_s_gpu": f["generated_tok_s"] / n, "useful_is_exact": bool(f["useful_is_exact"])})  # fmt: skip
    # Where the tokens go, per run: generated by the engine -> trained by the backward -> useful (nonzero advantage).
    # TRL and prime-rl drop zero-advantage groups before the trainer (generated > trained = useful); the others train
    # them (generated ~ trained > useful). Either way about half the generated tokens carry no gradient.
    stages = [("generated_tok_s_gpu", "generated (engine output, completions)", 0.18),
              ("all_tok_s_gpu", "trained (through the backward, prompt + completion)", 0.5),
              ("useful_tok_s_gpu", "useful (trained, nonzero advantage)", 1.0)]  # fmt: skip
    h = 0.26
    fig, ax = plt.subplots(figsize=(11, 0.95 * max(len(rows), 2) + 1.8))
    for i, r in enumerate(rows):
        c = LIB_COLOR.get(r["lib"], MUTED)
        for j, (k, _, a) in enumerate(stages):
            y = i + (j - 1) * h
            ax.barh(y, r[k], height=h * 0.9, color=c, alpha=a, lw=0)
            est = k == "useful_tok_s_gpu" and not r["useful_is_exact"]
            ax.text(r[k], y, f" {r[k]:,.0f}" + (" (est.)" if est else ""), va="center", fontsize=7, color=INK)
    lib_labels(ax, rows)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max([max(r[k] for k, _, _ in stages) for r in rows] + [1]) * 1.2)
    ax.set(
        xlabel="tokens / s / GPU (all 8 GPUs of the job)", title=f"Where the tokens go (steps > {SKIP_STEPS})"
    )
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=MUTED, alpha=a) for _, _, a in stages],
              labels=[lbl for _, lbl, _ in stages], loc="upper center", bbox_to_anchor=(0.5, -0.07 - 0.25 / len(rows)), ncol=3)  # fmt: skip
    notes(fig, ["TRL and prime-rl drop groups whose samples all got the same reward before the trainer, so their trained = useful "
                "and the waste shows as generated > trained; the other libraries train those groups (zero gradient), so trained > useful",
                "from results/tables/throughput_fair.csv, whole-job wall clock; (est.) = zero-advantage share fitted from reward mean"])  # fmt: skip
    save(fig, "e_throughput", pd.DataFrame(rows))


def peak_memory(runs):
    """(f): peak memory.used of the trainer GPUs and of the inference GPUs per run (nvidia-smi), plus the trainer's own
    torch peak (steps.jsonl peak_mem_gb) where the library logs it."""
    rows, msgs = [], []
    for r in run_order([r for r in runs if r["gpu"] is not None and len(r["gpu"])]):
        train, engine, g = gpu_roles(r["gpu"], r["meta"].get("gpus_inference") or 0)
        peak = g.groupby("dev")["mem_gb"].max()
        torch_peak = col(r["steps"], "peak_mem_gb")
        rows.append({"run_id": r["id"], "label": r["label"], "lib": r["lib"],
                     "train_gpus": " ".join(train), "engine_gpus": " ".join(engine),
                     "train_peak_gb": float(peak[train].max()) if train else None,
                     "engine_peak_gb": float(peak[engine].max()) if engine else None,
                     "torch_peak_gb": None if torch_peak is None else float(torch_peak.max())})  # fmt: skip
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.48 * max(len(rows), 2) + 1.8), sharey=True)
    for ax, k, title in [
        (axes[0], "train_peak_gb", "Trainer GPUs"),
        (axes[1], "engine_peak_gb", "Inference-engine GPUs"),
    ]:
        for i, r in enumerate(rows):
            if r[k] is None:
                continue
            c = LIB_COLOR.get(r["lib"], MUTED)
            ax.barh(i, r[k], height=0.62, color=c, lw=0)
            ax.text(r[k], i, f"  {r[k]:.0f} GB", va="center", fontsize=7.5, color=INK)
            if k == "train_peak_gb" and r["torch_peak_gb"] is not None:
                ax.plot(r["torch_peak_gb"], i, marker="D", ms=6, color=INK, mec=SURFACE, mew=1.5)
        ax.set_xlim(0, 230)
        ax.grid(axis="y", visible=False)
        ax.set(title=title, xlabel="peak memory.used per GPU (GB, nvidia-smi)")
    lib_labels(axes[0], rows)
    axes[0].invert_yaxis()
    axes[0].legend(handles=[Line2D([], [], marker="D", ls="", color=INK, ms=6)], labels=["torch peak (trainer's own log)"],
                   loc="lower right")  # fmt: skip
    fig.suptitle("Peak GPU memory, trainer vs inference engine", fontweight="bold", color=INK)
    fig.tight_layout()
    notes(fig, ["GPU roles are inferred: the engine's GPUs are the ones with the highest median memory (KV cache preallocated), so their peak is "
                "the engine's memory setting, not a need. nvidia-smi includes the allocator's cache."] + msgs)  # fmt: skip
    save(fig, "f_peak_memory", pd.DataFrame(rows))


def memory_over_time(runs):
    """(f2): memory.used per GPU over the run, one panel per run; trainer GPUs in the library color, engine GPUs gray."""
    have = run_order([r for r in runs if r["gpu"] is not None and len(r["gpu"])])
    ncol = 2
    nrow = max(1, -(-len(have) // ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(12, 1.9 * nrow + 0.8), sharey=True, squeeze=False)
    rows = []
    for ax, r in zip(axes.flat, have, strict=False):
        train, engine, g = gpu_roles(r["gpu"], r["meta"].get("gpus_inference") or 0)
        t0 = g["t"].min()
        g = g.assign(h=(g["t"] - t0).dt.total_seconds() / 3600)
        g["bin"] = (g["h"] * 60).round() / 60  # one point per minute: max over the minute
        m = g.groupby(["dev", "bin"])["mem_gb"].max().reset_index()
        for dev, d in m.groupby("dev"):
            is_train = dev in train
            ax.plot(d["bin"], d["mem_gb"], lw=1.1 if is_train else 0.9, color=LIB_COLOR.get(r["lib"], MUTED) if is_train else MUTED,
                    alpha=0.9 if is_train else 0.6)  # fmt: skip
        tp = m[m["dev"].isin(train)]["mem_gb"].max() if train else np.nan
        ax.set_title(f"{r['label']}   trainer peak {tp:.0f} GB", loc="left", fontsize=8.5)
        ax.set_ylim(0, 210)
        rows += [{"run_id": r["id"], "dev": a, "role": "train" if a in train else "engine", "hours": b, "mem_gb": c}
                 for a, b, c in m.values]  # fmt: skip
    for ax in axes.flat[len(have) :]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("GB")
    for ax in axes[-1]:
        ax.set_xlabel("hours since the first nvidia-smi sample")
    fig.suptitle("GPU memory over the run (nvidia-smi, 1-min max)", fontweight="bold", color=INK)
    fig.tight_layout()
    notes(fig, ["colored: trainer GPUs; gray: inference-engine GPUs (inferred: highest median memory)"])
    save(fig, "f2_memory_over_time", pd.DataFrame(rows))


def staleness_truncation(runs):
    """(g): staleness (mean, max) and truncated_fraction per step as EMAs, runs ordered by library, legend outside."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharex=True)
    rows, msgs, plotted = [], [], []
    keys = [("staleness_mean", "Mean staleness of trained samples", "policy versions"),
            ("staleness_max", "Max staleness of trained samples", "policy versions"),
            ("truncated_fraction", "Completions hitting max length", "fraction")]  # fmt: skip
    for r in run_order(runs):
        s = r["steps"]
        got = False
        for ax, (k, _, _) in zip(axes, keys, strict=False):
            y = col(s, k)
            if y is None:
                msgs.append(f"no {k}: {r['label']}")
                continue
            got = True
            c = LIB_COLOR.get(r["lib"], MUTED)
            ax.plot(s["step"], y, color=c, ls=r["ls"], lw=0.6, alpha=0.25)
            ax.plot(s["step"], ema(y), color=c, ls=r["ls"], lw=2)
            rows += [
                {"run_id": r["id"], "lib": r["lib"], "step": a, k: b, f"{k}_ema": e}
                for a, b, e in zip(s["step"], y, ema(y), strict=False)
            ]
        if got:
            plotted.append(r)
    for ax, (k, t, yl) in zip(axes, keys, strict=False):
        ax.set(title=t, ylabel=yl, xlabel="optimizer step")
    fig.suptitle(
        f"Off-policyness and truncation (thick: EMA alpha {EMA_ALPHA}; thin: raw)", fontweight="bold", color=INK
    )
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    handles = [
        Line2D([], [], color=LIB_COLOR.get(r["lib"], MUTED), ls=r["ls"], lw=2, label=r["label"]) for r in plotted
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.0))
    notes(fig, msgs)
    save(fig, "g_staleness_truncation", pd.DataFrame(rows))


def util_heatmap(runs):
    """(h): SM utilization per GPU over time, one panel per run."""
    have = [r for r in runs if r["gpu"] is not None and len(r["gpu"])]
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
    fig, axes = plt.subplots(max(len(have), 1), 1, figsize=(11, 1.0 + 1.9 * max(len(have), 1)), squeeze=False,
                             layout="constrained")  # fmt: skip
    rows, im = [], None
    for ax, r in zip(axes[:, 0], have, strict=False):
        g = r["gpu"].copy()
        g["dev"] = g["node"] + ":" + g["gpu"].astype(str)
        span = (g["t"].max() - g["t"].min()).total_seconds()
        bin_s = max(1, int(np.ceil(span / 600)))  # at most ~600 columns
        g["bin"] = ((g["t"] - g["t"].min()).dt.total_seconds() // bin_s).astype(int)
        m = g.pivot_table(index="dev", columns="bin", values="util", aggfunc="mean")
        im = ax.imshow(m.to_numpy(), aspect="auto", cmap=cmap, vmin=0, vmax=100, interpolation="nearest",
                       extent=[0, m.columns.max() * bin_s / 3600, len(m) - 0.5, -0.5])  # fmt: skip
        ax.set_yticks(range(len(m)), [d.split(":")[-1] for d in m.index], fontsize=7)
        ax.set_ylabel("GPU", fontsize=8)
        ax.grid(False)
        ax.set_title(f"{r['label']}  ({bin_s} s bins)", fontsize=9, loc="left")
        rows += [{"run_id": r["id"], "lib": r["lib"], "gpu": d, "hours": b * bin_s / 3600, "util_pct": v}
                 for d, row in m.iterrows() for b, v in row.items()]  # fmt: skip
    axes[-1, 0].set_xlabel("hours since the first nvidia-smi sample")
    if im is not None:
        fig.colorbar(im, ax=axes[:, 0].tolist(), label="utilization.gpu (%)", shrink=0.8, pad=0.01)
    else:
        no_data(axes[0, 0], "no gpu.csv")
    fig.suptitle("GPU utilization (nvidia-smi utilization.gpu)", fontweight="bold", color=INK)
    save(fig, "h_gpu_util", pd.DataFrame(rows))


def trl_timeline(runs, run_pattern=None, start=None):
    """(i): TRL spans over a ~5 min window: in-flight vLLM requests on top, trainer lanes below."""
    cands = [r for r in runs if r["lib"] == "trl" and os.path.exists(r["spans"]) and len(r["steps"])]
    if run_pattern:
        cands = [r for r in cands if re.search(run_pattern, r["id"])]
    if not cands:
        print("skip i_trl_timeline: no TRL run with spans.jsonl")
        return
    r = max(cands, key=lambda r: r["meta"].get("start_time") or 0)
    sp = pd.DataFrame(read_jsonl(r["spans"]))
    trainer = sp[sp["kind"] != "generate"]
    if start is None:  # first trainer span past the warmup steps, minus a little context
        late = trainer[trainer["step"] > SKIP_STEPS]
        start = (late if len(late) else trainer)["start"].min() - 15
    end = start + TIMELINE_S
    w = sp[(sp["end"] > start) & (sp["start"] < end)]
    fig, (top, bot) = plt.subplots(2, 1, figsize=(12, 5.5), sharex=True, gridspec_kw={"height_ratios": [1.3, 1]})
    gen = w[w["kind"] == "generate"]
    edges = sorted(set(np.clip(np.r_[gen["start"], gen["end"], start, end], start, end)))
    inflight = [((gen["start"] <= t) & (gen["end"] > t)).sum() for t in edges]
    top.fill_between(np.array(edges) - start, inflight, step="post", color=LIB_COLOR["trl"], alpha=0.25, lw=0)
    top.step(np.array(edges) - start, inflight, where="post", color=LIB_COLOR["trl"], lw=1.2)
    top.set(ylabel="in-flight requests", title=f"generate (vLLM): {r['label']}")
    lanes = ["train_step", "weight_sync", "wait"]
    lane_color = {"train_step": PART_COLOR["train_compute"], "weight_sync": PART_COLOR["weight_sync"], "wait": PART_COLOR["wait_for_rollouts"]}
    rows = []
    for i, k in enumerate(lanes):
        s = w[w["kind"] == k]
        # bars shorter than 0.4% of the window (weight syncs take ~0.5 s) are widened so they stay visible
        bars = [
            (max(a, start) - start, max(min(b, end) - max(a, start), 0.004 * TIMELINE_S))
            for a, b in zip(s["start"], s["end"], strict=False)
        ]
        bot.broken_barh(bars, (i - 0.35, 0.7), color=lane_color[k], edgecolor=SURFACE, linewidth=1)
        for (a, d), st in zip(bars, s["step"], strict=False):
            if d > 0.08 * TIMELINE_S and k == "train_step":
                bot.text(a + d / 2, i, f"step {st}", ha="center", va="center", fontsize=7, color=SURFACE)
    bot.set_yticks(range(len(lanes)), lanes)
    bot.invert_yaxis()
    bot.grid(axis="y", visible=False)
    bot.set(xlabel=f"seconds from {pd.to_datetime(start, unit='s'):%Y-%m-%d %H:%M:%S} UTC",
            title="trainer (rank 0)", xlim=(0, TIMELINE_S))  # fmt: skip
    for _, s in w.iterrows():
        rows.append({k: s.get(k) for k in ["kind", "rank", "worker", "step", "start", "end", "tokens"]})
    fig.tight_layout()
    notes(
        fig,
        [
            f"{len(gen)} generate requests in the window; spans carry no GPU id, so generate is drawn as concurrency",
            f"trainer bars shorter than {0.004 * TIMELINE_S:.1f} s are drawn at that width",
        ],
    )
    save(fig, "i_trl_timeline", pd.DataFrame(rows))


def latest_run(prefix):
    """results/raw/<prefix>-<job id> with the highest job id, or None."""
    ds = sorted(glob.glob(os.path.join(RAW, f"{prefix}-[0-9]*")), key=lambda d: int(d.rsplit("-", 1)[1]))
    return ds[-1] if ds else None


# Libraries with no W2 row, and why (envs/<lib>/NOTES.md).
W2_STATUS = [
    "SkyRL: not runnable as shipped on this workload (zeroes the whole group when an agent times out, ~50% of episodes; "
    "its fully-async loss rollout_is collapsed on W1); run 365144 used harness changes and was stopped",
    "TRL shared #364987 runs boxes on HF Sandboxes, TRL local #366992 on local apptainer containers, prime-rl #368590 on its "
    "own docker runtime (rootless podman)",
    "miles, verl: W2 runs in progress",
]


def w2_tokens(runs):
    """(k): W2 batch size and step time per library: trained tokens (SHARED_W2: every model call as its own sample) vs
    forwarded (what the trainer ran after its own prefix sharing), step time by component, and seconds per million
    trained tokens. Every non-smoke W2 run. Step 1 (startup) is left out."""
    rows, msgs = [], []
    for r in run_order([r for r in runs if "smoke" not in r["tag"]]):
        lib = r["lib"]
        s = r["steps"]
        s = s[s["step"] >= 2]
        tr, fw = col(s, "trained_tokens"), col(s, "forwarded_tokens")
        parts, n, m = mean_parts(r, 2)
        msgs += m
        if parts is None or tr is None or fw is None:
            continue
        msgs.append(f"{r['label']}: steps 2-{int(s['step'].max())} ({n} steps)")
        rows.append({"run_id": r["id"], "label": r["label"], "lib": lib, "steps": n, "parts": parts, "big": float(tr.mean()),
                     "small": float(fw.mean()), "trained_M": float(tr.mean()) / 1e6, "prefix_share_ratio": float((tr / fw).mean())})  # fmt: skip
    if not rows:
        print("skip k_w2_tokens: no W2 run")
        return
    breakdown_figure(rows, "Step time vs batch size (step 1 left out)", "Tokens per step",
                     "trained (every model call as its own sample)", "forwarded (after prefix sharing)", "trained", "trained_M",
                     "k_w2_tokens", ["trained / forwarded = how much the library's prefix sharing saves"] + msgs + W2_STATUS)  # fmt: skip


def pause_options():
    """(l): trl-014 options on the standalone vLLM repro (job 365055, 8 API servers, 688 in flight, /pause every 20 s,
    600 s per arm). The numbers are copied from envs/trl/PAUSE_STALL.md "Options measured" (summaries of
    envs/trl/pause_stall/logs/365055-api8), since the repro has no results/raw dir."""
    t = pd.DataFrame(
        [("top-8 (default)", 12.0, 38.2, 65.3), ("top-4", 6.7, 19.3, 79.9), ("top-2", 0.05, 0.21, 91.1),
         ("top-1", 0.04, 0.06, 91.9), ("no logprobs", 0.05, 0.07, 95.9)],
        columns=["num_sampler_logprobs", "pause_mean_s", "pause_max_s", "gen_ktok_s"],
    )  # fmt: skip
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    y = np.arange(len(t))
    a.barh(y, t["pause_mean_s"], height=0.6, color=LIB_COLOR["trl"])
    for i, r in t.iterrows():
        a.text(
            r["pause_mean_s"],
            i,
            f"  mean {r['pause_mean_s']:g} s, max {r['pause_max_s']:g} s",
            va="center",
            fontsize=7.5,
            color=INK,
        )
        b.barh(i, r["gen_ktok_s"], height=0.6, color=LIB_COLOR["trl"])
        b.text(r["gen_ktok_s"], i, f"  {r['gen_ktok_s']:g}k", va="center", fontsize=7.5, color=INK)
    a.set_yticks(y, t["num_sampler_logprobs"])
    a.invert_yaxis()
    a.set_xlim(0, 22)
    b.set_xlim(0, 115)
    for ax in (a, b):
        ax.grid(axis="y", visible=False)
    a.set(xlabel="weight-sync /pause latency, mean (s)", title="Pause latency")
    b.set(xlabel="generated tokens / s (thousands)", title="Generation throughput")
    fig.suptitle(
        "Sampler logprobs per token vs vLLM pause latency (8 API servers)", fontweight="bold", color=INK
    )
    fig.tight_layout()
    notes(fig, ["standalone vLLM repro, no trainer: 688 requests in flight, /pause?mode=keep + /resume every 20 s, 600 s per arm (job 365055)",
                "from envs/trl/PAUSE_STALL.md; top-8 with 16 API servers: 0.06 s mean pause, 89.5k tok/s"])  # fmt: skip
    save(fig, "l_pause_options", t)


def field_coverage(runs):
    """Fraction of non-null values per steps.jsonl field and run, to spot schema gaps."""
    fields = ["step_time", "train_compute_time", "wait_for_rollouts_time", "weight_sync_time", "num_rollouts",
              "gen_tokens", "trained_tokens", "padding_fraction", "reward_mean", "reward_std", "frac_zero_adv_groups",
              "completion_len_mean", "completion_len_max", "truncated_fraction", "staleness_mean", "staleness_max",
              "logprob_mismatch_kl", "grad_norm", "lr", "peak_mem_gb"]  # fmt: skip
    rows = []
    for r in runs:
        s = r["steps"]
        rows.append({"run_id": r["id"], "steps": len(s),
                     **{f: (float(pd.to_numeric(s[f], errors="coerce").notna().mean()) if f in s else 0.0) if len(s) else None
                        for f in fields}})  # fmt: skip
    t = pd.DataFrame(rows)
    os.makedirs(TABLES, exist_ok=True)
    t.to_csv(os.path.join(TABLES, "field_coverage.csv"), index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 50):
        print(t.set_index("run_id").T.round(2))


# ---------------------------------------------------------------- blog figures (figures/post/, from figures/data/bench.json)
#
# The figures the post uses, in post order. Same data, titles and colors as figures/interactive/index.html. Each one is
# written as SVG and as PNG at 2x (7.2 in wide at 100 px per inch -> 1440 px), with its plotted numbers as CSV next to it.

POST = os.path.join(BENCH, "figures", "post")
DATA = os.path.join(BENCH, "figures", "data", "bench.json")
PX = 100  # CSS px per inch of the figure; PNG at 2x
W = 7.2
NAMES = {"trl": "TRL", "prime-rl": "prime-rl", "verl": "verl", "openrlhf": "OpenRLHF", "skyrl": "SkyRL", "miles": "miles"}
SEV_COLOR = {"blocker": "#3d3c38", "major": "#8a8984", "minor": "#cfcec8"}  # neutral ramp: severity is not a library
BENCHES = [("aime24", "AIME24 (avg@16)"), ("aime25", "AIME25 (avg@16)"), ("math500", "MATH-500 (avg@1)")]


def tint(hex_color, f):
    """Mix a color toward the chart surface: f=1 keeps it, f=0 is the surface."""
    a = np.array([int(hex_color[i : i + 2], 16) for i in (1, 3, 5)], dtype=float)
    b = np.array([int(SURFACE[i : i + 2], 16) for i in (1, 3, 5)], dtype=float)
    return "#" + "".join(f"{int(round(v)):02x}" for v in b + (a - b) * f)


def shade(hex_color, f):
    """Mix a color toward ink."""
    a = np.array([int(hex_color[i : i + 2], 16) for i in (1, 3, 5)], dtype=float)
    return "#" + "".join(f"{int(round(v)):02x}" for v in a * (1 - f) + 20 * f)


def post_figure(h, ncols=1, nrows=1, title="", subtitle="", foot=(), **kw):
    """A figure with a left-aligned finding as title, a context line under it and footnotes below the plot."""
    fig, axes = plt.subplots(nrows, ncols, figsize=(W, h), layout="constrained", squeeze=False, **kw)
    sub = textwrap.fill(subtitle, int(W * 15.5)) if subtitle else ""
    foot = [textwrap.fill(f, int(W * 17.5)) for f in foot if f]
    head_in = 0.34 + 0.17 * (sub.count("\n") + 1 if sub else 0) + 0.08
    foot_in = 0.06 + 0.135 * sum(f.count("\n") + 1 for f in foot) + (0.06 if foot else 0)
    fig.get_layout_engine().set(rect=(0, foot_in / h, 1, 1 - (head_in + foot_in) / h), h_pad=0.08, w_pad=0.06)
    fig.text(0.008, 1 - 0.06 / h, title, ha="left", va="top", fontsize=12, fontweight="bold", color=INK)
    if sub:
        fig.text(0.008, 1 - 0.33 / h, sub, ha="left", va="top", fontsize=8.6, color=INK2, linespacing=1.35)
    if foot:
        fig.text(0.008, 0.04 / h, "\n".join(foot), ha="left", va="bottom", fontsize=7.2, color=MUTED, linespacing=1.4)
    for ax in axes.flat:
        ax.tick_params(length=0, pad=4)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color(MUTED)
        ax.grid(axis="x", visible=False)
        ax.set_axisbelow(True)
    return fig, axes


def post_save(fig, name, rows=None):
    os.makedirs(POST, exist_ok=True)
    fig.savefig(os.path.join(POST, f"{name}.svg"), bbox_inches="tight", pad_inches=0.12)
    fig.savefig(os.path.join(POST, f"{name}.png"), dpi=2 * PX, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    if rows is not None:
        pd.DataFrame(rows).to_csv(os.path.join(POST, f"{name}.csv"), index=False)
    print(f"wrote figures/post/{name}")


def end_labels(ax, items, x, min_gap=0.055, fontsize=7.6, dx=5):
    """Direct labels at the end of lines: items = [(y, text, color)], spread to min_gap of the axis height around their
    own y (both directions), so close lines keep labels near their ends."""
    lo, hi = ax.get_ylim()
    gap = min_gap * (hi - lo)
    items = sorted(items, key=lambda t: t[0])
    ys = [y for y, _, _ in items]
    for _ in range(50):
        moved = False
        for i in range(1, len(ys)):
            d = ys[i] - ys[i - 1]
            if d < gap - 1e-9:
                push = (gap - d) / 2
                ys[i - 1] -= push
                ys[i] += push
                moved = True
        if not moved:
            break
    for (_, text, _), y in zip(items, ys):
        ax.annotate(text, (x, y), xytext=(dx, 0), textcoords="offset points", va="center", ha="left", fontsize=fontsize,
                    color=INK, annotation_clip=False)


def in_progress_note(runs, evals=True):
    out = [f"{NAMES[r['lib']]}: {r['status_note']}" for r in runs
           if r["status"] != "done" and r.get("status_note") and (evals or r["status"] != "evals pending")]
    return ("In progress: " + "; ".join(out) + ".") if out else ""


def fig_w1_eval_curves(D):
    rows = []
    foot = [in_progress_note(D["w1"][size]["runs"]) for size in ("S", "M")]
    foot.append("miles: not run as shipped (its only correct attention path on B200 is ~7x slower). Not shown: runs with "
                "patched libraries or custom losses (" + ", ".join(e["run_id"].rsplit("-", 1)[-1] + f" {NAMES.get(e['lib'], e['lib'])}"
                for e in D["excluded"] if "recommended" not in e["reason"] and "-w1-" in e["run_id"]) + ") and recommended arms, except SkyRL's "
                "default dppo loss on the shared config (365171, dashed), shown because its rollout_is collapse comes from our mapping of the clip band.")  # fmt: skip
    fig, axes = post_figure(5.6, ncols=3, nrows=2,
                            title="On the same config, the libraries that trained cleanly land within eval noise",
                            subtitle="Held-out score per saved checkpoint, W1 math on built-in options. Dashed: base model. "
                                     "AIME has 30 problems, so checkpoints of one run move by about ±0.05.", foot=foot)  # fmt: skip
    for i, (size, model) in enumerate([("S", "Qwen3.5-2B"), ("M", "Qwen3.5-9B")]):
        w = D["w1"][size]
        for j, (k, bname) in enumerate(BENCHES):
            ax = axes[i, j]
            ax.set_title(bname if i == 0 else "")
            if j == 0:
                ax.set_ylabel(model, fontsize=9, fontweight="bold", color=INK)
            ax.set_xlim(0, 200)
            ax.set_ylim(0, 1.0 if k == "math500" or size == "M" else 0.6)
            ends = []
            for r in w["runs"]:
                ev = [e for e in r["evals"] if e.get(k) is not None]
                if not ev:
                    continue
                xs, ys = [0] + [e["step"] for e in ev], [w["base"][k]] + [e[k] for e in ev]
                c = LIB_COLOR[r["lib"]]
                ax.plot(xs, ys, color=c, lw=2 if r["lib"] == "trl" else 1.6, marker="o", ms=4, mec=SURFACE, mew=1,
                        zorder=3 if r["lib"] == "trl" else 2)  # fmt: skip
                ends.append((ys[-1], NAMES[r["lib"]] + (" rollout_is, our mapping (collapsed)" if r.get("collapsed") else ""), c))
                rows += [{"size": size, "lib": r["lib"], "run_id": r["run_id"], "bench": k, "step": x, "score": y} for x, y in zip(xs, ys)]
            for r in w.get("extra", []):
                ev = [e for e in r["evals"] if e.get(k) is not None]
                xs, ys = [0] + [e["step"] for e in ev], [w["base"][k]] + [e[k] for e in ev]
                c = LIB_COLOR[r["lib"]]
                ax.plot(xs, ys, color=c, lw=1.6, ls="--", marker="o", ms=4, mec=SURFACE, mew=1, zorder=2)
                ends.append((ys[-1], r["label"], c))
                rows += [{"size": size, "lib": r["lib"], "run_id": r["run_id"], "bench": k, "step": x, "score": y} for x, y in zip(xs, ys)]
            ax.axhline(w["base"][k], color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
            ax.set_xticks([0, 50, 100, 150, 200])
            if j == 2:
                end_labels(ax, ends, 200, min_gap=0.075)
            if i == 1:
                ax.set_xlabel("optimizer step")
        axes[i, 0].text(3, w["base"]["aime24"], "base", fontsize=7, color=INK2, va="bottom")
    post_save(fig, "01_w1_eval_curves", rows)


def fig_w1_step200(D):
    rows = []
    fig, axes = post_figure(4.4, ncols=3, nrows=2, title="At step 200, the three clean runs sit within ±0.05 of each other",
                            subtitle="Score at optimizer step 200, W1 math on built-in options. Dashed: base model. "
                                     "Bars would exaggerate gaps this small, so each library is a dot.",
                            foot=[in_progress_note(D["w1"]["S"]["runs"]), in_progress_note(D["w1"]["M"]["runs"])])  # fmt: skip
    for i, (size, model) in enumerate([("S", "Qwen3.5-2B"), ("M", "Qwen3.5-9B")]):
        w = D["w1"][size]
        runs = w["runs"]
        for j, (k, bname) in enumerate(BENCHES):
            ax = axes[i, j]
            ax.set_title(bname if i == 0 else "")
            if j == 0:
                ax.set_ylabel(model, fontsize=9, fontweight="bold", color=INK)
                ax.text(w["base"][k], -0.62, " base", fontsize=6.8, color=INK2, va="bottom")
            for y, r in enumerate(runs):
                e = next((e for e in r["evals"] if e["step"] == 200), None)
                c = LIB_COLOR[r["lib"]]
                if e is None:
                    ax.text(0.01, y, "in progress", transform=ax.get_yaxis_transform(), va="center", fontsize=7, color=MUTED, style="italic")
                    continue
                ax.plot([w["base"][k], e[k]], [y, y], color=tint(c, 0.35), lw=1.5, zorder=1)
                ax.plot(e[k], y, "o", ms=7, color=c if not r.get("collapsed") else SURFACE, mec=c, mew=1.6, zorder=3)
                right = e[k] >= w["base"][k]
                ax.annotate(f"{e[k]:.3f}", (e[k], y), xytext=(7 if right else -7, 0), textcoords="offset points",
                            ha="left" if right else "right", va="center", fontsize=6.8, color=INK2)
                rows.append({"size": size, "lib": r["lib"], "run_id": r["run_id"], "bench": k, "step200": e[k], "base": w["base"][k]})
            ax.axvline(w["base"][k], color=INK2, lw=1, ls=(0, (4, 3)))
            ax.set_xlim(0, 0.6 if size == "S" and k != "math500" else 1.05)
            ax.set_ylim(len(runs) - 0.4, -0.7)
            ax.set_yticks(range(len(runs)), [NAMES[r["lib"]] + (" (collapsed)" if r.get("collapsed") else "") if j == 0 else "" for r in runs])
            ax.grid(axis="y", visible=False)
            ax.grid(axis="x", visible=True)
            if i == 1:
                ax.set_xlabel("score")
    post_save(fig, "02_w1_step200", rows)


def hbar_rows(ax, labels):
    ax.set_yticks(range(len(labels)), labels)
    ax.set_ylim(len(labels) - 0.4, -0.6)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)


def fig_w1_wallclock(D):
    rows = []
    fig, axes = post_figure(3.7, ncols=2, title="prime-rl reaches step 200 in half TRL's time, with half the data per step",
                            subtitle="Hours from the first to the last optimizer step (200 steps), W1 math, one 8xB200 node. "
                                     "TRL's step holds ~2.7M trained tokens, prime-rl's ~1.4M.",
                            foot=["Training time only: job start-up (model load, vLLM start) is not included."])  # fmt: skip
    for j, (size, model) in enumerate([("S", "Qwen3.5-2B"), ("M", "Qwen3.5-9B")]):
        ax = axes[0, j]
        runs = D["w1"][size]["runs"]
        ax.set_title(model)
        for y, r in enumerate(runs):
            c = LIB_COLOR[r["lib"]]
            h = r.get("hours")
            if h is None:
                ax.text(0.1, y, "in progress", va="center", fontsize=7, color=MUTED, style="italic")
                continue
            done = r["last_step"] >= 200
            ax.barh(y, h, height=0.62, color=c if done else SURFACE, edgecolor=c, hatch=None if done else "////", lw=1.2)
            ax.text(h, y, f"  {h:.2f} h" + ("" if done else f"  (step {r['last_step']} of 200, running)"), va="center", fontsize=7.4, color=INK)
            rows.append({"size": size, "lib": r["lib"], "run_id": r["run_id"], "hours": h, "last_step": r["last_step"]})
        hbar_rows(ax, [NAMES[r["lib"]] + (" (collapsed)" if r.get("collapsed") else "") for r in runs])
        ax.set_xlim(0, max([r.get("hours") or 0 for r in runs] + [1]) * 1.45)
        ax.set_xlabel("hours to step 200")
    post_save(fig, "03_w1_wallclock", rows)


def fig_w1_throughput(D):
    t = [r for r in D["throughput_S"]]
    labels = [NAMES[r["lib"]] + (" (collapsed)" if r.get("collapsed") else "") for r in t]
    fig, axes = post_figure(3.6, ncols=2, title="Per useful trained token, TRL and prime-rl are level",
                            subtitle="Qwen3.5-2B, W1 math, steady state (steps after 5), whole 8-GPU job. Useful = tokens of "
                                     "groups with any reward spread (nonzero advantage); the rest still costs a backward in "
                                     "the libraries that train it.",
                            foot=["est.: the library doesn't log its zero-advantage share; estimated from the step reward "
                                  "(fit over 2,145 steps of the libraries that log it). Definitions: results/tables/throughput_fair.md."]
                            + ([in_progress_note([{"lib": r["lib"], "status": "x", "status_note": "throughput pending"} for r in t if r["status"] != "done"])]
                               if any(r["status"] != "done" for r in t) else []),
                            gridspec_kw={"width_ratios": [1.5, 1]})  # fmt: skip
    a, b = axes[0]
    a.set_title("Trained tokens / s")
    b.set_title("Generated tokens / s")
    for y, r in enumerate(t):
        if r["status"] != "done":
            a.text(1e3, y, "in progress", va="center", fontsize=7, color=MUTED, style="italic")
            continue
        c = LIB_COLOR[r["lib"]]
        a.barh(y, r["useful"], height=0.62, color=c)
        zero = r["all"] - r["useful"]
        if zero > 1:
            a.barh(y, zero, left=r["useful"], height=0.62, color=tint(c, 0.28), edgecolor=SURFACE, lw=1)
        a.text(r["all"], y, f"  {r['useful'] / 1e3:.1f}k useful" + ("" if r["useful_exact"] else " (est.)"), va="center", fontsize=7.2, color=INK)
        b.barh(y, r["generated"], height=0.62, color=c)
        b.text(r["generated"], y, f"  {r['generated'] / 1e3:.1f}k", va="center", fontsize=7.2, color=INK)
    hbar_rows(a, labels)
    hbar_rows(b, ["" for _ in t])
    a.set_xlim(0, 95e3)
    b.set_xlim(0, 130e3)
    for ax in (a, b):
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1e3:.0f}k"))
    tr = next((r for r in t if r["lib"] == "skyrl" and r.get("status") == "done"), None)
    a.annotate("light: trained, zero advantage", (0.99, 0.02), xycoords="axes fraction", ha="right", va="bottom", fontsize=7, color=INK2)
    post_save(fig, "04_w1_throughput", t)
    return tr


def stacked_steps(ax, rows, parts, colors, labels, value_fmt="{:.0f} s"):
    """Horizontal stacked bars, one row per run, segment labels inside when they fit."""
    xmax = max([sum(r["parts"][p] for p in parts) for r in rows if r.get("parts")] + [1])
    for y, r in enumerate(rows):
        if not r.get("parts"):
            ax.text(xmax * 0.01, y, r.get("status_note") or "in progress", va="center", fontsize=7, color=MUTED, style="italic")
            continue
        left = 0
        for p in parts:
            v = r["parts"][p]
            fc, hatch = colors(r["lib"], p)
            ax.barh(y, v, left=left, height=0.62, color=fc, hatch=hatch, edgecolor=SURFACE, lw=1)
            txt = f"{labels[p]} {value_fmt.format(v)}"
            if v / (xmax * 1.12) > 0.0115 * len(txt):
                txt_c = "white" if p in ("train", "vllm") else INK
                ax.text(left + v / 2, y, txt, ha="center", va="center", fontsize=6.6, color=txt_c)
            left += v
        ax.text(left, y, "  " + value_fmt.format(left), va="center", fontsize=7.6, color=INK, fontweight="bold")
    ax.set_xlim(0, xmax * 1.12)
    return xmax


def part_key(ax, items):
    """Key for stacked segments whose hue is the library's: neutral proxies for solid / light / dark / hatched."""
    proxy = {"solid": ("#5f5e5a", None), "light": ("#cfcec8", None), "dark": ("#262522", None), "hatched": ("#e3e2dc", "////"),
             "grey": ("#bdbcb5", None), "pale": ("#d6d5cf", None)}  # fmt: skip
    hs = [matplotlib.patches.Patch(fc=proxy[k][0], hatch=proxy[k][1], ec=SURFACE, label=t) for k, t in items]
    ax.legend(handles=hs, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(hs), fontsize=7, handlelength=1.2,
              handleheight=0.9, columnspacing=1.2, borderaxespad=0.2)


def part_colors(lib, p):
    c = LIB_COLOR[lib]
    return {"train": (c, None), "wait": (tint(c, 0.38), None), "sync": (shade(c, 0.45), None), "other": ("#e3e2dc", "////")}[p]


def fig_w1_step_breakdown(D):
    runs = [r for r in D["w1"]["S"]["runs"]]
    labels = {"train": "train", "wait": "wait", "sync": "sync", "other": "other"}
    fig, axes = post_figure(3.6, title="TRL's step is twice prime-rl's, and 15 s of it is rank-0 packing",
                            subtitle="Mean seconds per optimizer step, Qwen3.5-2B, W1 math, steps after 5. Train = forward + "
                                     "backward + optimizer, wait = waiting for rollouts, sync = weight sync, "
                                     "other = the rest of the step (for TRL, mostly rank 0 building packed micro-batches).",
                            foot=["OpenRLHF logs no wait or sync time: everything but train is in other. "
                                  "Weight-sync spans differ: TRL's includes vLLM's pause, prime-rl's is the trainer-side broadcast only.",
                                  in_progress_note(runs, evals=False)])  # fmt: skip
    ax = axes[0, 0]
    stacked_steps(ax, runs, ["train", "wait", "sync", "other"], part_colors, labels)
    part_key(ax, [("solid", "train"), ("light", "waiting for rollouts"), ("dark", "weight sync"), ("hatched", "other")])
    hbar_rows(ax, [NAMES[r["lib"]] + (" (collapsed)" if r.get("collapsed") else "") for r in runs])
    ax.set_xlabel("seconds per optimizer step")
    post_save(fig, "05_w1_step_breakdown", [{"lib": r["lib"], "run_id": r.get("run_id"), **(r.get("parts") or {})} for r in runs])


def fig_w1_reward(D):
    rows = []
    fig, axes = post_figure(3.5, ncols=2, title="Train reward climbs in every run except SkyRL's",
                            subtitle="Mean train reward per optimizer step, W1 math. Thin: per step; thick: moving average "
                                     "(EMA, α = 0.1). Libraries put different numbers of samples in a step, so compare shapes, not levels.",
                            foot=[in_progress_note(D["w1"]["M"]["runs"], evals=False)])  # fmt: skip
    for j, (size, model) in enumerate([("S", "Qwen3.5-2B"), ("M", "Qwen3.5-9B")]):
        ax = axes[0, j]
        ax.set_title(model)
        ends = []
        for r in D["w1"][size]["runs"]:
            st = [s for s in r["steps"] if s.get("reward") is not None]
            if not st:
                continue
            c = LIB_COLOR[r["lib"]]
            x = np.array([s["step"] for s in st])
            y = pd.Series([s["reward"] for s in st])
            e = y.ewm(alpha=EMA_ALPHA).mean()
            ax.plot(x, y, color=tint(c, 0.35), lw=0.7, zorder=1)
            ax.plot(x, e, color=c, lw=2, zorder=3 if r["lib"] == "trl" else 2)
            ends.append((float(e.iloc[-1]), NAMES[r["lib"]] + ("" if r["last_step"] >= 200 else f" (step {r['last_step']})"), c))
            rows += [{"size": size, "lib": r["lib"], "step": int(a), "reward": float(b), "ema": float(c_)} for a, b, c_ in zip(x, y, e)]
        ax.set_xlim(0, 200)
        ax.set_ylim(0, 1)
        ax.set_xlabel("optimizer step")
        end_labels(ax, ends, 200, min_gap=0.07)
    post_save(fig, "06_w1_train_reward", rows)


def fig_w2_step_breakdown(D):
    fam = lambda r: next((f["family"] for f in D["w2"]["failed"] if f["lib"] == r["lib"]), None) if r.get("role") == "failed" else r["family"]  # noqa: E731
    runs = sorted(D["w2"]["step_breakdown"] + w2_failed_rows(D), key=lambda r: fam(r) != "9B")
    labels = {"train": "train", "wait": "waiting for rollouts", "sync": "sync", "other": "other"}
    fig, axes = post_figure(2.2 + 0.32 * len(runs), title=w2_steps_title(runs),
                            subtitle="Mean seconds per optimizer step, W2 terminal tasks, steps after 1. Qwen3.5-9B unless the row "
                                     "says 35B-A3B. TRL before the Harbor fix: one Harbor server event loop for all 64 agents.",
                            foot=w2_steps_foot(D))  # fmt: skip
    ax = axes[0, 0]
    stacked_steps(ax, runs, ["train", "wait", "sync", "other"], part_colors, labels)
    part_key(ax, [("solid", "train"), ("light", "waiting for rollouts"), ("dark", "weight sync"), ("hatched", "other")])
    hbar_rows(ax, [r["label"] for r in runs])
    ax.set_xlabel("seconds per optimizer step")
    post_save(fig, "07_w2_step_breakdown", [{"run_id": r["run_id"], "label": r["label"], **(r.get("parts") or {})} for r in runs])


def w2_failed_rows(D, fam=None):
    """Failed or not-runnable W2 libraries as labelled rows (never a silent gap)."""
    return [{"lib": f["lib"], "label": NAMES.get(f["lib"], f["lib"]) + ("" if f["family"] == "9B" else f", {f['family']}"),
             "parts": None, "status": "failed", "status_note": f"✕ {f['note']}", "run_id": f.get("run_id"), "role": "failed"}
            for f in D["w2"].get("failed", []) if fam is None or f["family"] == fam]  # fmt: skip


def w2_steps_title(runs):
    trl = next((r for r in runs if r["lib"] == "trl" and r["role"] == "main" and r.get("parts")), None)
    old = next((r for r in runs if r["role"] == "before_fix" and r.get("parts")), None)
    if trl and old:
        return f"Sharding the Harbor server cut TRL's agentic step from {old['parts']['step_time']:.0f} s to {trl['parts']['step_time']:.0f} s"
    return "Before the Harbor fix, TRL's agentic steps were 3x prime-rl's, most of it waiting for rollouts"


def w2_steps_foot(D):
    runs = D["w2"]["step_breakdown"]
    out = ["TRL's run shards the Harbor server and passes the 30-call cap to terminus-2 through a launcher wrapper (no OpenEnv "
           "file edited); that's closer to a patch than a built-in option."]
    out += [f"Not shown: TRL's first W2 run on HF Sandboxes ({d['run_id'].rsplit('-', 1)[-1]}, {d['step_time']:.0f} s per step, same "
            "single Harbor server)." for d in D["w2"].get("dropped", []) if d.get("step_time")]
    out += [f"In progress: {r['label']}: {r['status_note']}." for r in runs if r["status"] != "done"]
    out += [f"{NAMES.get(f['lib'], f['lib'])} ({f['family']}): {f['detail']}" for f in D["w2"].get("failed", []) if f.get("detail")]
    return out


def fig_w2_waterfall(D):
    wf = D["w2"]["waterfall"]
    causes = wf["causes"]
    fig, axes = post_figure(3.5, title="Before the Harbor fix, four causes accounted for TRL's extra 400 s per step; the trainer wasn't one",
                            subtitle="Counterfactual chain from prime-rl's step to TRL's before the Harbor fix (local containers, run "
                                     f"{wf['before_run'].rsplit('-', 1)[-1]}), using step ≈ episodes "
                                     "per step × episode time / episodes in flight. From envs/trl/W2_STEP_TIME.md.",
                            foot=["Causes 1, 2 and 4 are fixed in the launcher (shard the Harbor server, pass the call cap): that's TRL's run in the step-time chart; "
                                  "cause 3 is the config gap (TRL's step is sized in tokens and refills), not speed."])  # fmt: skip
    ax = axes[0, 0]
    labels = ["prime-rl step"] + [f"{i}. {c['label']}" for i, c in enumerate(causes, 1)] + ["TRL before the Harbor fix, sum of the chain"]
    left = wf["start"]
    ax.barh(0, left, height=0.6, color=LIB_COLOR["prime-rl"])
    ax.text(left, 0, f"  {left} s", va="center", fontsize=7.6, color=INK)
    for i, c in enumerate(causes, 1):
        ax.barh(i, c["s"], left=left, height=0.6, color=tint(LIB_COLOR["trl"], 0.45), edgecolor=LIB_COLOR["trl"], lw=0.8)
        ax.plot([left, left], [i - 1 - 0.3, i + 0.3], color=MUTED, lw=0.7, ls=":")
        ax.text(left + c["s"], i, f"  +{c['s']} s", va="center", fontsize=7.6, color=INK)
        left += c["s"]
    n = len(causes) + 1
    ax.barh(n, left, height=0.6, color=LIB_COLOR["trl"])
    ax.text(left, n, f"  {left} s (measured {wf['measured']} s)", va="center", fontsize=7.6, color=INK, fontweight="bold")
    if wf.get("measured"):
        ax.plot([wf["measured"]] * 2, [n - 0.38, n + 0.38], color=INK, lw=1.6)
    hbar_rows(ax, labels)
    ax.set_xlim(0, left * 1.3)
    ax.set_xlabel("seconds per optimizer step")
    post_save(fig, "08_w2_gap_waterfall", [{"start": wf["start"], **c} for c in causes])


EP_PARTS = ["vllm", "agent", "box", "verifier", "outside"]
EP_LABEL = {"vllm": "in vLLM", "agent": "agent loop, not vLLM", "box": "box start + setup", "verifier": "verifier", "outside": "outside phases"}


def ep_colors(lib, p):
    c = LIB_COLOR[lib]
    return {"vllm": (c, None), "agent": (tint(c, 0.38), None), "box": ("#bdbcb5", None), "verifier": ("#d6d5cf", None),
            "outside": ("#e8e7e1", "////")}[p]  # fmt: skip


def fig_w2_episode(D):
    eps = D["w2"]["episode"]
    fig, axes = post_figure(3.1, title="Before the Harbor fix, TRL's episodes took 3x longer, and vLLM was only 71 s of the 548",
                            subtitle="Mean wall time of one training episode (steady, scored), split by phase. Both libraries "
                                     "run ~23-24 model calls and ~15k sampled tokens per episode with ~62 episodes in flight.",
                            foot=["TRL's vLLM time is from vLLM's own counters (3.1 s per call); prime-rl's is its interception span "
                                  "(4.5 s per call). 47% of TRL's episodes end at the 600 s agent limit, none of prime-rl's."])  # fmt: skip
    ax = axes[0, 0]
    rows = [e | {"status": "done"} for e in eps]
    stacked_steps(ax, rows, EP_PARTS, ep_colors, EP_LABEL)
    part_key(ax, [("solid", "in vLLM"), ("light", "agent loop, not vLLM"), ("grey", "box start + setup"), ("pale", "verifier"),
                  ("hatched", "outside Harbor's phases")])
    hbar_rows(ax, [e["label"] for e in eps])
    ax.set_xlabel("seconds per episode")
    post_save(fig, "09_w2_episode_time", [{"lib": e["lib"], "wall": e["wall"], **e["parts"]} for e in eps])


def other_split_note(H):
    o = H.get("other_splits", [])
    if not o:
        return ""
    base = [p for p in o if not p["run_id"] and p.get("mean") is not None]
    return ("Not in this chart: evals on other task splits (" + ", ".join(sorted({p["split"] for p in o})) + "), in progress"
            + "".join(f"; {p['split']} base {p['family']}: {p['mean']:.3f}" for p in base) + ".")


def fig_w2_heldout(D):
    H = D["w2"]["heldout"]
    fams = ["9B"] + (["35B-A3B"] if any(x["family"] == "35B-A3B" for x in H["base"] + H["series"]) else [])
    rows, pend = [], []
    for s in H["series"]:
        if s["pending"]:
            pend.append(f"{s['label']}: step {', '.join(map(str, s['pending']))}")
    pend += [f"base {b['family']}" for b in H["base"] if b["status"] != "done"]
    fig, axes = post_figure(3.9, ncols=len(fams),
                            title="Held-out pass rate moves by about one standard error; nothing separates the runs yet",
                            subtitle="14 held-out terminal tasks × 4 episodes (56 per point), one eval script (TRL's probe on HF "
                                     "Sandboxes) for every checkpoint. Bars: ±1 binomial standard error (~0.065); dashed line and "
                                     "grey band: base model ±1 SE.",
                            foot=["TRL: the sharded-Harbor run at max_staleness 8 (prime-rl's setting)."
                                  + (" Dashed \"TRL, staleness 4\": the earlier staleness-4 run, shown only until the staleness-8 "
                                     "evals land." if any(x.get("reference") for x in H["series"]) else "")
                                  + " Episodes of one task are correlated, so the real uncertainty is larger: ±1 SE over per-task "
                                  "means is ~0.08-0.10."]
                            + H.get("notes", []) + [other_split_note(H)] + [
                                  ("Evals in progress: " + "; ".join(pend) + ".") if pend else ""])  # fmt: skip
    for ax, fam in zip(axes[0], fams):
        base = next((b for b in H["base"] if b["family"] == fam and b.get("mean") is not None), None)
        series = [s for s in H["series"] if s["family"] == fam]
        ax.set_title(f"Qwen3.5-{fam}")
        if base:
            ax.axhspan(base["mean"] - base["se"], base["mean"] + base["se"], color="#efeee9", zorder=0)
            ax.axhline(base["mean"], color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
            ax.text(63, base["mean"] - base["se"] - 0.012, f"base model {base['mean']:.3f} ± {base['se']:.3f}", fontsize=7.2, color=INK2, va="top", ha="right")
            rows.append({"family": fam, "run_id": "base", "step": 0, "mean": base["mean"], "se": base["se"], "se_task": base.get("se_task"), "n": base["n"]})
        if not base and not any(p["status"] == "done" for q in series for p in q["points"]):
            ax.text(0.5, 0.5, "in progress", transform=ax.transAxes, ha="center", color=MUTED, style="italic")
        for k, s in enumerate(series):
            c = LIB_COLOR[s["lib"]]
            ls = "--" if s.get("reference") else "-"
            pts = [p for p in s["points"] if p["status"] == "done"]
            off = (k - (len(series) - 1) / 2) * 1.4
            xs = ([0] if base else []) + [p["step"] + off for p in pts]
            ys = ([base["mean"]] if base else []) + [p["mean"] for p in pts]
            ax.plot(xs, ys, color=c, lw=1.4, ls=ls, zorder=2, marker="o", ms=5, mec=SURFACE, mew=1,
                    label=s["label"] + ("" if pts else " (evals in progress)"))
            eb = ax.errorbar([p["step"] + off for p in pts], [p["mean"] for p in pts], yerr=[p["se"] for p in pts], fmt="o", ms=5,
                             color=c, mec=SURFACE, mew=1, elinewidth=1.3, capsize=0, zorder=3)  # fmt: skip
            for lc in eb[2]:
                lc.set_linestyle("-")
            rows += [{"family": fam, "run_id": s["run_id"], "step": p["step"], "mean": p["mean"], "se": p["se"], "se_task": p.get("se_task"), "n": p["n"]} for p in pts]
        for f in [f for f in D["w2"].get("failed", []) if f["family"] == fam]:
            ax.plot([], [], ls="none", marker="x", ms=6, mew=1.8, color=LIB_COLOR.get(f["lib"], MUTED),
                    label=f"{NAMES.get(f['lib'], f['lib'])}: {f['note']}")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="upper left", fontsize=6.8, handlelength=2.2, ncol=1, labelspacing=0.3)
        ax.set_xlim(-3, 64)
        ax.set_xticks([0, 20, 40, 60], ["base", "20", "40", "60"])
        ax.set_ylim(0, 1.12)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xlabel("training step of the checkpoint")
    axes[0, 0].set_ylabel("held-out pass rate", color=INK2)
    post_save(fig, "10_w2_heldout", rows)


def fig_w2_reward(D):
    R = [r for r in D["w2"]["reward"] if r["steps"]]
    fams = list(dict.fromkeys(r["family"] for r in R)) or ["9B"]
    fig, axes = post_figure(3.5, ncols=len(fams), title="Train reward per step on the agentic tasks, one run per library",
                            subtitle="Mean train reward per optimizer step on the 28 training tasks. Thin: per step; thick: moving "
                                     "average (EMA, α = 0.2). Step sizes differ and TRL's agents lose turns to the 600 s limit, so "
                                     "the held-out eval is the comparison that counts.",
                            foot=[f"{r['label']}: {r['status_note']}." for r in D["w2"]["reward"] if r["status"] != "done"])  # fmt: skip
    rows = []
    for ax, fam in zip(axes[0], fams):
        ax.set_title(f"Qwen3.5-{fam}")
        for r in [r for r in R if r["family"] == fam]:
            c = LIB_COLOR[r["lib"]]
            ls = "-"
            x = np.array([s["step"] for s in r["steps"]])
            y = pd.Series([s["reward"] for s in r["steps"]])
            e = y.ewm(alpha=0.2).mean()
            ax.plot(x, y, color=tint(c, 0.35), lw=0.7, ls=ls)
            ax.plot(x, e, color=c, lw=2, ls=ls, label=r["label"])
            rows += [{"run_id": r["run_id"], "step": int(a), "reward": float(b), "ema": float(q)} for a, b, q in zip(x, y, e)]
        ax.set_xlim(0, 60)
        ax.set_ylim(0, 1)
        ax.set_xlabel("optimizer step")
        ax.legend(loc="lower right", fontsize=7.2, handlelength=2.6)
    post_save(fig, "07a_w2_train_reward", rows)


def fig_w2_prefix(D):
    P = [p for p in D["w2"]["prefix"]]
    fig, axes = post_figure(3.0, title="prime-rl shares prefixes 16x, TRL's tree packing 7.6x",
                            subtitle="Tokens per optimizer step, W2, steps after 1. Light: every model call as its own sample "
                                     "(what a trainer would forward without prefix sharing). Solid: what the trainer actually forwarded.",
                            foot=["terminus-2 re-renders the chat every turn, so each call forks TRL's prefix tree; prime-rl "
                                  "appends sampled token ids and keeps one branch per episode."]
                            + [f"{p['label']}: {p['status_note']}." for p in P if p["status"] != "done"])  # fmt: skip
    ax = axes[0, 0]
    for y, p in enumerate(P):
        c = LIB_COLOR[p["lib"]]
        if not p.get("trained"):
            ax.text(0.3e6, y, p["status_note"] or "in progress", va="center", fontsize=7, color=MUTED, style="italic")
            continue
        ax.barh(y, p["trained"], height=0.62, color=tint(c, 0.25))
        ax.barh(y, p["forwarded"], height=0.62, color=c)
        ax.text(p["forwarded"], y, f"  {p['forwarded'] / 1e6:.2f}M", va="center", fontsize=7.2, color=INK)
        ax.text(p["trained"], y, f"  {p['trained'] / 1e6:.1f}M per-call  ·  {p['ratio']:.1f}x shared", va="center", fontsize=7.4, color=INK)
    hbar_rows(ax, [p["label"] for p in P])
    ax.set_xlim(0, max([p.get("trained") or 0 for p in P]) * 1.45)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1e6:.0f}M"))
    ax.set_xlabel("tokens per optimizer step")
    post_save(fig, "11_w2_prefix_sharing", [{k: v for k, v in p.items() if k != "per_step"} for p in P])


def fig_moe(D):
    M = D["moe"]
    st = [s for s in M["steps"] if s.get("step_time") is not None]
    x = np.array([s["step"] for s in st])
    c = LIB_COLOR["trl"]
    fig, axes = post_figure(4.8, ncols=2, nrows=2,
                            title="Qwen3.5-35B-A3B trains on one node: flat weight sync and memory over 200 steps",
                            subtitle=f"TRL, LoRA r=32, EP=4 × FSDP=4 on 4 trainer GPUs, vLLM on the other 4, W1 math with 16k "
                                     f"completions, staleness 4. {M['hours']:.1f} h for 200 steps (run {M['run_id'].rsplit('-', 1)[-1]}).",
                            foot=["Weight sync sends the merged LoRA plus every frozen tensor. Memory: max over the 4 trainer ranks of torch's peak. "
                                  "Needed TRL fixes that have since landed on main (#8236, #8294, #8295). The run saved no checkpoints, so there's no AIME eval.",
                                  "W2 runs on the 35B: " + (", ".join(M["w2_runs_seen"]) if M.get("w2_runs_seen") else "in progress") + "."])  # fmt: skip
    a, b, cc, d = axes.flat
    wait = np.array([s["wait"] or 0 for s in st])
    tot = np.array([s["step_time"] for s in st])
    train = np.array([(s["fwd_bwd"] or 0) + (s["optimizer"] or 0) for s in st])
    a.fill_between(x, 0, train, color=tint(c, 0.3), lw=0)
    a.plot(x, tot, color=tint(c, 0.55), lw=0.7)
    a.plot(x, pd.Series(tot).ewm(alpha=EMA_ALPHA).mean(), color=c, lw=2)
    a.text(0.03, 0.08, "train: fwd + bwd + optimizer", transform=a.transAxes, color=INK, fontsize=7)
    a.text(0.03, 0.92, "whole step (rest: mostly waiting for rollouts)", transform=a.transAxes, color=INK, fontsize=7, va="top")
    a.set_title("Step time (s)")
    a.set_ylim(0, max(tot) * 1.12)
    sync = np.array([s["sync"] for s in st])
    b.plot(x, sync, color=c, lw=1.6)
    b.set_ylim(0, max(sync) * 1.6)
    b.set_title("Weight sync (s)")
    alloc = np.array([s["peak_alloc_gb"] or np.nan for s in st])
    res = np.array([s["peak_reserved_gb"] or np.nan for s in st])
    cc.plot(x, res, color=tint(c, 0.5), lw=1.4)
    cc.plot(x, alloc, color=c, lw=1.8)
    cc.axhline(180, color=INK2, lw=1, ls=(0, (4, 3)))
    cc.text(2, 172, "B200: 180 GB", fontsize=7, color=INK2, va="top")
    cc.text(x[-1], np.nanmax(alloc), " peak allocated", fontsize=7, color=INK, va="bottom", ha="right")
    cc.text(x[-1], np.nanmax(res), " reserved", fontsize=7, color=INK, va="bottom", ha="right")
    cc.set_ylim(0, 190)
    cc.set_title("Peak memory per trainer GPU (GB)")
    z = np.array([s["zero_std"] for s in st])
    d.plot(x, z, color=tint(c, 0.45), lw=0.7)
    d.plot(x, pd.Series(z).ewm(alpha=EMA_ALPHA).mean(), color=c, lw=2)
    d.set_ylim(0, 1)
    d.set_title("Groups with all-equal rewards (no gradient)")
    for ax in axes.flat:
        ax.set_xlim(0, 200)
    for ax in (cc, d):
        ax.set_xlabel("optimizer step")
    post_save(fig, "12_moe_35b", st)


T2SQL_NAMES = {"trl-fixed": "TRL (with #8212 fix)", "trl": "TRL stock (stale prefix cache)", "prime-rl": "prime-rl", "skyrl": "SkyRL"}


def t2sql_style(lib):
    """Color, line style and alpha: the stock TRL run is a dashed, lighter TRL-blue series behind the fixed one."""
    if lib == "trl":
        return LIB_COLOR["trl"], (0, (4, 2)), 0.55
    return LIB_COLOR["trl" if lib == "trl-fixed" else lib], "-", 1.0


T2SQL_SPLITS = [("validation", "validation (398 tasks)"), ("arcwise", "arcwise (498 tasks)")]


def t2sql_runs_note(T):
    gp = {r["lib"]: r["gpus"] for r in T["runs"]}
    out = [f"{T2SQL_NAMES[r['lib']]}: training, step {r['last_step']} of {T['planned_steps']}" for r in T["runs"] if r["last_step"] < T["planned_steps"] and r["lib"] not in ("prime-rl", "skyrl")]
    return [f"GPUs per run: TRL {gp.get('trl-fixed')} (both runs), prime-rl {gp.get('prime-rl')}, SkyRL {gp.get('skyrl')} (SkyRL's recipe assumes 8; it ran on the 4 "
            "that were free). Hours count from job start, so they include startup (model load, vLLM, first rollouts).",
            "SkyRL was stopped by choice at step 42; step 40 is its last eval. "
            + (("In progress: " + "; ".join(out) + ". Later checkpoints are added as their evals land.") if out else "")]  # fmt: skip


def fig_t2sql_heldout(D):
    T = D["t2sql"]
    b = T["base"]
    fig, axes = post_figure(5.4, ncols=2, nrows=2,
                            title="Held-out SQL accuracy: with the #8212 fix TRL lands within about one SE of prime-rl and SkyRL, in fewer hours",
                            subtitle="Greedy pass rate of each saved checkpoint, Qwen3.5-35B-A3B with LoRA r32 on text-to-SQL, one eval "
                                     "script for every model. Left: wall-clock hours since the job started; right: GPU-hours. Bars: "
                                     "±1 binomial SE (~0.017). Dashed line and grey band: base model ±1 SE.",
                            foot=t2sql_runs_note(T) + ["Each library runs its own recipe (loss, lr, samples per question, staleness): "
                                                       "results/tables/t2sql_35b.md."])  # fmt: skip
    rows = []
    for i, (k, kname) in enumerate(T2SQL_SPLITS):
        for j, (xk, xname) in enumerate([("hours", "hours since job start"), ("gpu_hours", "GPU-hours since job start")]):
            ax = axes[i, j]
            ax.set_title(f"{kname} · vs {'hours' if j == 0 else 'GPU-hours'}")
            xmax = max([e[xk] or 0 for r in T["runs"] for e in r["evals"]] + [1]) * 1.06
            if b:
                ax.axhspan(b[k] - b[k + "_se"], b[k] + b[k + "_se"], color="#efeee9", zorder=0)
                ax.axhline(b[k], color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
                ax.text(xmax * 0.99, b[k] - b[k + "_se"] - 0.004, f"base {b[k]:.3f}", fontsize=7, color=INK2, va="top", ha="right")
            for r in T["runs"]:
                ev = [e for e in r["evals"] if e.get(xk) is not None and e.get(k) is not None]
                if not ev:
                    continue
                c, ls, al = t2sql_style(r["lib"])
                xs, ys = [0] + [e[xk] for e in ev], [b[k] if b else None] + [e[k] for e in ev]
                ax.plot(xs, ys, color=c, lw=1.6, ls=ls, alpha=al, zorder=2, label=T2SQL_NAMES[r["lib"]])  # GPUs are in the foot
                eb = ax.errorbar([e[xk] for e in ev], [e[k] for e in ev], yerr=[e[k + "_se"] for e in ev], fmt="o", ms=5, color=c,
                                 alpha=al, mec=SURFACE, mew=1, elinewidth=1.3, capsize=0, zorder=3)  # fmt: skip
                for lc in eb[2]:
                    lc.set_linestyle("-")
                for e in ev:
                    ax.annotate(f"{int(e['step'])}", (e[xk], e[k]), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=6.4, color=MUTED)
                if i == 0 and j == 0:
                    rows += [{"lib": r["lib"], "step": int(e["step"]), "hours": e["hours"], "gpu_hours": e["gpu_hours"],
                              **{s: e[s] for s, _ in T2SQL_SPLITS}, **{s + "_se": e[s + "_se"] for s, _ in T2SQL_SPLITS}} for e in ev]  # fmt: skip
            ax.set_xlim(0, xmax)
            ax.set_ylim(0.68, 0.92)
            if i == 1:
                ax.set_xlabel(xname)
            if j == 0:
                ax.set_ylabel("greedy pass rate", color=INK2)
    axes[0, 0].legend(loc="lower center", ncol=2, fontsize=7, handlelength=2.2, columnspacing=1.0, borderaxespad=0.2)
    post_save(fig, "15_t2sql_heldout", rows)


def fig_t2sql_train(D):
    T = D["t2sql"]
    fig, axes = post_figure(3.6, ncols=2,
                            title="Step time about doubles in both TRL and prime-rl as episodes get longer",
                            subtitle="Per optimizer step, text-to-SQL on Qwen3.5-35B-A3B. Thin: per step; thick: moving average "
                                     "(EMA, α = 0.2). Step 1 (startup) is left out of the step time. Steps hold different numbers of "
                                     "episodes and SkyRL samples 5 per question at T = 0.6 (the others 16 at T = 1.0), so compare "
                                     "reward shapes, not levels.",
                            foot=t2sql_runs_note(T)[:1] + ["prime-rl runs without refill: 960 rollouts per step, zero-advantage groups "
                                                           "dropped and not replaced, as TRL does. Its default refill run (#373642) "
                                                           "needed ~3,500 rollouts per 912-sample step by step 30 and was stopped at "
                                                           "step 40. SkyRL is synchronous on 4 GPUs."]
                            + t2sql_runs_note(T)[1:])  # fmt: skip
    a, s = axes[0]
    rows = []
    for r in T["runs"]:
        st = r["steps"]
        if not st:
            continue
        c, ls, al = t2sql_style(r["lib"])
        x = np.array([q["step"] for q in st])
        y = pd.Series([q["reward"] for q in st], dtype=float)
        a.plot(x, y, color=tint(c, 0.35), lw=0.7, alpha=al)
        a.plot(x, y.ewm(alpha=0.2).mean(), color=c, lw=2, ls=ls, alpha=al, label=T2SQL_NAMES[r["lib"]])  # GPUs are in the foot
        tt = [q for q in st if q["step"] > 1 and q["step_s"] is not None]
        xt = np.array([q["step"] for q in tt])
        yt = pd.Series([q["step_s"] for q in tt], dtype=float)
        s.plot(xt, yt, color=tint(c, 0.35), lw=0.7, alpha=al)
        s.plot(xt, yt.ewm(alpha=0.2).mean(), color=c, lw=2, ls=ls, alpha=al)
        rows += [{"lib": r["lib"], "step": q["step"], "hours": q["h"], "reward": q["reward"], "step_s": q["step_s"], "wait_s": q["wait_s"]} for q in st]
    a.set_ylim(0.3, 1)
    a.set_title("Train reward (each library's own batch)")
    a.legend(loc="lower right", ncol=2, fontsize=7, handlelength=2.0, columnspacing=1.0)
    s.set_ylim(0, None)
    s.set_title("Seconds per optimizer step")
    for ax in (a, s):
        ax.set_xlim(0, T["planned_steps"])
        ax.set_xlabel("optimizer step")
    post_save(fig, "16_t2sql_train", rows)


CATS = ["install_break", "missing_dep", "docs_or_example_mismatch", "unclear_setup", "config_trap", "crash", "hang",
        "silent_correctness", "perf_trap", "cluster_unfriendly"]  # fmt: skip


def fig_robustness(D):
    R = D["robustness"]["libs"]
    top = max(R, key=lambda r: r["failures"])
    trl = next(r for r in R if r["lib"] == "trl")
    fig, axes = post_figure(3.9, ncols=2, title=f"TRL logged {trl['failures']} library failures, none silent; "
                            f"{NAMES[top['lib']]} logged {top['failures']}, {top['silent']} of them silent",
                            subtitle=f"Failures per library in the robustness ledger (results/robustness/failures.jsonl, "
                                     f"{D['robustness']['n_rows']} rows): the library broke, hung, crashed, gave wrong numbers, or its "
                                     "docs, defaults, lock or image didn't work as written. Our own mistakes and cluster problems aren't counted.",
                            foot=["TRL is the subject of this benchmark, so its code and logs were read far more closely than anyone "
                                  "else's; a problem elsewhere is more likely to have gone unnoticed."],
                            gridspec_kw={"width_ratios": [1, 1.25]})  # fmt: skip
    a, b = axes[0]
    for y, r in enumerate(R):
        left = 0
        for s in ("blocker", "major", "minor"):
            if r[s]:
                a.barh(y, r[s], left=left, height=0.62, color=SEV_COLOR[s], edgecolor=SURFACE, lw=1)
                if r[s] >= 1:
                    a.text(left + r[s] / 2, y, str(r[s]), ha="center", va="center", fontsize=6.8, color="white" if s != "minor" else INK)
                left += r[s]
        if r["unattributed"]:
            a.barh(y, r["unattributed"], left=left, height=0.62, color=SURFACE, edgecolor=MUTED, hatch="////", lw=0.8)
            left += r["unattributed"]
        a.text(left, y, f"  {r['failures']}", va="center", fontsize=7.6, color=INK, fontweight="bold")
    hbar_rows(a, [NAMES[r["lib"]] for r in R])
    a.set_xlim(0, max(r["failures"] + r["unattributed"] for r in R) * 1.2)
    a.set_title("By severity")
    hs = [matplotlib.patches.Patch(fc=SEV_COLOR[s], ec=SURFACE, label=s) for s in ("blocker", "major", "minor")]
    hs.append(matplotlib.patches.Patch(fc=SURFACE, ec=MUTED, hatch="////", label="cause not attributed"))
    a.legend(handles=hs, loc="lower right", fontsize=6.8, handlelength=1.1, handleheight=0.9, borderaxespad=0.2)
    a.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    a.set_xlabel("failures")
    cats = [c for c in CATS if any(r["categories"].get(c) for r in R)]
    grid = np.array([[r["categories"].get(c, 0) for c in cats] for r in R], dtype=float)
    ramp = LinearSegmentedColormap.from_list("ink", [SURFACE, "#3d3c38"])
    b.imshow(grid, cmap=ramp, vmin=0, vmax=max(grid.max(), 1), aspect="auto")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i, j]:
                b.text(j, i, int(grid[i, j]), ha="center", va="center", fontsize=7, color="white" if grid[i, j] >= grid.max() / 2 else INK)
    b.set_xticks(range(len(cats)), [c.replace("_", " ").replace("docs or example mismatch", "docs mismatch") for c in cats], rotation=40, ha="right", fontsize=7)
    b.set_yticks(range(len(R)), [""] * len(R))
    b.grid(False)
    b.spines["bottom"].set_visible(False)
    b.set_title("By kind")
    post_save(fig, "13_robustness", [{k: v for k, v in r.items() if k not in ("rows", "categories")} | r["categories"] for r in R])


def fig_pause(D):
    t = D["pause"]
    fig, axes = post_figure(2.9, ncols=2, title="Top-8 sampler log-probs stall vLLM's pause for 12 s and cost a third of throughput",
                            subtitle="Standalone vLLM 0.33 repro, no trainer: Qwen3.5-2B, 8 API servers, 688 requests in flight, "
                                     "/pause + /resume every 20 s, 600 s per arm (job 365055).",
                            foot=["From envs/trl/PAUSE_STALL.md. With 16 API servers, top-8 pauses 0.06 s on average at 89.5k tok/s."])  # fmt: skip
    a, b = axes[0]
    c = LIB_COLOR["trl"]
    for y, r in enumerate(t):
        a.barh(y, r["pause_mean_s"], height=0.62, color=c)
        a.text(r["pause_mean_s"], y, f"  {r['pause_mean_s']:g} s (max {r['pause_max_s']:g})", va="center", fontsize=7.2, color=INK)
        b.barh(y, r["gen_ktok_s"], height=0.62, color=c)
        b.text(r["gen_ktok_s"], y, f"  {r['gen_ktok_s']:g}k", va="center", fontsize=7.2, color=INK)
    hbar_rows(a, [r["option"] for r in t])
    hbar_rows(b, ["" for _ in t])
    a.set_xlim(0, 22)
    b.set_xlim(0, 120)
    a.set_title("Mean /pause latency (s)")
    b.set_title("Generated tokens / s (thousands)")
    post_save(fig, "14_vllm_pause", t)


def post_figures():
    D = json.load(open(DATA))
    with plt.rc_context({"axes.titlesize": 9, "axes.titlelocation": "left", "axes.titlepad": 8}):
        _post_figures(D)


def _post_figures(D):
    fig_w1_eval_curves(D)
    fig_w1_step200(D)
    fig_w1_wallclock(D)
    fig_w1_throughput(D)
    fig_w1_step_breakdown(D)
    fig_w1_reward(D)
    fig_w2_reward(D)
    fig_w2_step_breakdown(D)
    fig_w2_waterfall(D)
    fig_w2_episode(D)
    fig_w2_heldout(D)
    fig_w2_prefix(D)
    if D.get("moe"):
        fig_moe(D)
    if D.get("t2sql"):
        fig_t2sql_heldout(D)
        fig_t2sql_train(D)
    fig_robustness(D)
    fig_pause(D)


def diagnostics(args):
    """The per-workload diagnostic charts (a-l) in figures/out/, from results/raw (filtered by --runs)."""
    runs = [r for r in load_runs(args.runs) if not r["meta"].get("validity")]  # patched / custom-code runs never plotted
    print("runs:", ", ".join(r["id"] for r in runs))
    with_steps = [r for r in runs if len(r["steps"])]
    fair = read_fair()
    for (wl, size), (folder, banner) in WORKLOADS.items():
        grp = [r for r in with_steps if f"-{wl}-{size}-" in r["id"] and "smoke" not in r["tag"]]
        if not grp:
            continue
        CTX.update(dir=folder, banner=banner)
        reward_curves(grp, "hours", "a_reward_vs_hours", "hours since first optimizer step", "Train reward vs wall-clock")
        reward_curves(grp, "step", "b_reward_vs_step", "optimizer step", "Train reward vs optimizer step")
        reward_curves(grp, "gen_tokens", "b2_reward_vs_gen_tokens", "generated tokens (billions, cumulative)",
                      "Train reward vs tokens generated")  # fmt: skip
        if wl == "w1":
            eval_curves(grp)
        if (wl, size) == ("w1", "S"):  # throughput_fair.csv covers W1-S only
            step_breakdown(grp, fair)
            throughput(grp, fair)
        if wl == "w2":
            w2_tokens(grp)
        peak_memory(grp)
        memory_over_time(grp)
        staleness_truncation(grp)
        util_heatmap(grp)
        trl_timeline(grp, args.timeline_run, args.timeline_start)
    CTX.update(dir=WORKLOADS[("repro", "pause")][0], banner=WORKLOADS[("repro", "pause")][1])
    pause_options()
    CTX.update(dir="", banner=None)
    field_coverage(runs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--diagnostics", action="store_true", help="also write the per-workload diagnostic charts to figures/out/")
    p.add_argument("--runs", default=None, help="diagnostics: regex on run_id (default: every dir in results/raw)")
    p.add_argument("--timeline-run", default=None, help="diagnostics: regex picking the TRL run of the timeline (default: latest)")
    p.add_argument("--timeline-start", type=float, default=None, help="diagnostics: unix time where the timeline window starts")
    args = p.parse_args()
    post_figures()
    if args.diagnostics:
        diagnostics(args)


if __name__ == "__main__":
    main()
