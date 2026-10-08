"""Plot before/after cleanup timings saved by compare_tree_gdn_cleanup.py."""

import argparse
import json
from pathlib import Path

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    rows = json.loads(args.results.read_text())["cases"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True, layout="constrained")
    for ax, topology in zip(axes, ("prompt", "fork"), strict=True):
        for fraction in (0.0, 0.75, 0.9375):
            cases = sorted(
                (r for r in rows if r["topology"] == topology and r["fraction"] == fraction), key=lambda r: r["length"]
            )
            ax.plot(
                [r["length"] / 1024 for r in cases],
                [r["speedup"] for r in cases],
                marker="o",
                label=f"{cases[0]['packing_ratio']:.2f}× packing",
            )
        ax.axhline(1, color="black", linestyle="--", linewidth=1)
        ax.set(title=topology.title(), xlabel="Sequence length per rollout (Ki tokens)", xticks=[16, 32, 64])
        ax.grid(alpha=0.2)
        ax.legend()
    axes[0].set_ylabel("Before / after cleanup latency (higher is better)")
    speeds = [row["speedup"] for row in rows]
    axes[0].set_ylim(min(0.95, min(speeds) * 0.99), max(1.05, max(speeds) * 1.01))
    fig.suptitle(
        "Tree GDN core: forward + backward, H100, BF16, 4 rollouts\nIndependent autotuning; 1.0× means unchanged performance"
    )
    for extension in ("png", "svg"):
        fig.savefig(args.results.with_suffix(f".{extension}"), dpi=160)


if __name__ == "__main__":
    main()
