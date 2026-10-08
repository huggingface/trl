"""Render review charts next to a tree-GDN benchmark JSON file."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def plot_sample_layouts(directory):
    """Illustrate identical 16K logical rollouts with two different sharing patterns."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, fork in zip(axes, (False, True), strict=True):
        nodes = [(0.03, 0.43, "Prompt\n12K", "#cfe4ff")]
        edges = []
        if fork:
            for y in (0.23, 0.68):
                nodes.append((0.37, y, "Shared\n2K", "#d9efdc"))
                edges.append((0, len(nodes) - 1))
                parent = len(nodes) - 1
                for leaf_y in (y - 0.11, y + 0.11):
                    nodes.append((0.73, leaf_y, "Leaf\n2K", "#ffe4bd"))
                    edges.append((parent, len(nodes) - 1))
        else:
            for y in (0.10, 0.32, 0.54, 0.76):
                nodes.append((0.73, y, "Completion\n4K", "#ffe4bd"))
                edges.append((0, len(nodes) - 1))
        for parent, child in edges:
            px, py = nodes[parent][:2]
            cx, cy = nodes[child][:2]
            ax.annotate("", (cx, cy + 0.07), (px + 0.22, py + 0.07), arrowprops=dict(arrowstyle="->", color="#666666"))
        for x, y, label, color in nodes:
            ax.add_patch(
                FancyBboxPatch((x, y), 0.22, 0.14, boxstyle="round,pad=0.01", facecolor=color, edgecolor="#777777")
            )
            ax.text(x + 0.11, y + 0.07, label, ha="center", va="center", fontsize=10)
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.axis("off")
        ax.set_title(
            "Nested forks: 12K + 2K + 2K per rollout\n64K logical → 24K unique · 2.67× packing"
            if fork
            else "Shared prompt: 12K + 4K per rollout\n64K logical → 28K unique · 2.29× packing"
        )
    fig.suptitle("Two examples with four 16K rollouts: shared nodes are evaluated once", fontsize=13)
    fig.tight_layout()
    for extension in ("png", "svg"):
        fig.savefig(directory / f"tree-gdn-sample-layouts.{extension}", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    plot_sample_layouts(args.results.parent)
    data = json.loads(args.results.read_text())
    cases = [case for case in data["cases"] if "error" not in case]
    rows = []
    for case in cases:
        for mode in ("forward", "forward_backward"):
            for method, measurement in case[mode].items():
                rows.append(
                    dict(
                        component=data["component"],
                        topology=case.get("topology", "prompt"),
                        rollout_tokens=case["prefix"] + case["suffix"],
                        rollouts=case["branches"],
                        prompt_tokens=case["prefix"],
                        continuation_tokens=case["suffix"],
                        logical_tokens=case["logical_tokens"],
                        unique_tokens=case["unique_tokens"],
                        packing_ratio=case["packing_ratio"],
                        mode=mode,
                        method=method,
                        ms=measurement.get("ms", ""),
                        speedup_vs_normal=measurement.get(
                            "speedup_vs_baseline", measurement.get("speedup_vs_fla", "")
                        ),
                        peak_extra_mib=measurement.get("peak_extra_mib", ""),
                        status="OOM" if "error" in measurement else "ok",
                    )
                )
    if rows:
        with args.results.with_suffix(".csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
    lengths = sorted({c["prefix"] + c["suffix"] for c in cases})
    groups = sorted({(c.get("topology", "prompt"), c["branches"]) for c in cases})
    styles = {
        "fla_sequence": ("Ordinary FLA", "#777777"),
        "triton_tree": ("Tree · Triton scan", "#1675d1"),
        "compiled_tree": ("Tree · compiled PyTorch scan", "#e78821"),
        "qwen_default": ("Normal Qwen GDN (default FLA core)", "#555555"),
        "qwen_native": ("Normal Qwen (PyTorch core fallback)", "#b25baf"),
    }
    if "tree_implementation" in data:
        styles["triton_tree"] = ("Tree · vendored FLA/Triton", "#1675d1")
        styles["compiled_tree"] = ("Tree · compiled PyTorch (FP32)", "#e78821")
    for mode, metric, title, ylabel in [
        ("forward", "ms", "Forward latency", "Milliseconds (lower is better)"),
        ("forward_backward", "ms", "Forward + backward latency", "Milliseconds (lower is better)"),
        ("forward_backward", "speedup_vs_fla", "Training speedup over ordinary FLA", "Speedup (higher is better)"),
        (
            "forward_backward",
            "speedup_vs_baseline",
            "Training speedup over the normal implementation",
            "Speedup (higher is better)",
        ),
        (
            "forward_backward",
            "speedup_vs_compiled",
            "Training speedup over compiled PyTorch tree",
            "Speedup (higher is better)",
        ),
        ("forward_backward", "peak_extra_mib", "Incremental peak allocated memory", "MiB (lower is better)"),
    ]:
        if not any(metric in measurement for c in cases for measurement in c[mode].values()):
            continue
        fig, axes = plt.subplots(len(groups), len(lengths), figsize=(5 * len(lengths), 4 * len(groups)), squeeze=False)
        for row, (topology, branches) in zip(axes, groups, strict=True):
            for ax, length in zip(row, lengths, strict=True):
                selected = sorted(
                    [
                        c
                        for c in cases
                        if c["prefix"] + c["suffix"] == length
                        and c.get("topology", "prompt") == topology
                        and c["branches"] == branches
                    ],
                    key=lambda c: c["packing_ratio"],
                )
                for key, (label, color) in styles.items():
                    points = [
                        (c["packing_ratio"], c[mode][key][metric])
                        for c in selected
                        if key in c[mode] and metric in c[mode][key]
                    ]
                    if points:
                        ax.plot(*zip(*points, strict=True), marker="o", label=label, color=color, linewidth=2)
                    oom = sum("error" in c[mode].get(key, {}) for c in selected)
                    if oom:
                        ax.plot([], [], color=color, linestyle=":", label=f"{label}: {oom} OOM")
                scenario = "Shared prompt" if topology == "prompt" else "Nested forks"
                ax.set_title(f"{scenario} · {branches} rollouts × {length // 1024}K")
                ax.set_xlabel("Packing ratio (logical / unique tokens)")
                ax.set_ylabel(ylabel)
                ax.set_xlim(0.9, branches + 0.1)
                ax.grid(alpha=0.25)
                if metric.startswith("speedup"):
                    ax.axhline(1, color="#555555", linestyle="--", linewidth=1)
                if metric == "ms":
                    ax.set_yscale("log")
                ax.legend(fontsize=7)
        axes[0, 0].legend(fontsize=8)
        subtitle = f"{data['gpu']} · {data['dtype']} · {data['key_heads']}/{data['value_heads']} K/V heads × {data['head_dim']}"
        if data["component"] == "layer":
            conv = ", ".join(sorted({c["qwen_convolution"] for c in cases}))
            subtitle += (
                f"\nSynthetic hidden states · hidden size {data['hidden_size']} · normal Qwen convolution: {conv}"
            )
        fig.suptitle(f"{data['component'].upper()} · {title}\n{subtitle}")
        fig.tight_layout()
        for extension in ("png", "svg"):
            fig.savefig(args.results.with_name(f"{args.results.stem}-{mode}-{metric}.{extension}"), dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    main()
