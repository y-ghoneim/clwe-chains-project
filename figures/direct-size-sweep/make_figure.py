"""The chain's gain over the direct map as the direct dictionary is cut (controlled scarcity).

See README.md. Reads experiments/controlled-scarcity/results/curve.csv (committed, 126a143): per
size, the mean over the directions present at that size (a pair is present when its full direct
dictionary is larger than the size). Writes direct_size_sweep.pdf and .png next to this file: the
gains of the chain, pooled direct and combination, with the pairs present under each size. Sizes
are evenly spaced (each doubles the last); "full" (each pair's whole dictionary) is set apart at
the right. Deterministic.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CURVE = REPO / "experiments" / "controlled-scarcity" / "results" / "curve.csv"
# dataviz reference palette: categorical slots 1-3 in fixed order; text and grid tokens (light)
METHODS = [("chain", "Chain", "#2a78d6", "o"), ("joined_mb", "Pooled direct", "#eb6834", "^"),
           ("combo", "Combination", "#1baf7a", "s")]
INK2, MUTED, GRID = "#52514e", "#898781", "#e1e0d9"
FULL_GAP = 1.6      # x distance from 16,000 to "full" (the sizes themselves are 1 apart)


def load() -> pd.DataFrame:
    c = pd.read_csv(CURVE, dtype={"budget": str})
    sizes = [b for b in c["budget"].unique() if b != "full"]
    order = sorted(sizes, key=int) + ["full"]
    xpos = {b: i for i, b in enumerate(order[:-1])}
    xpos["full"] = len(order) - 2 + FULL_GAP
    c["x"] = c["budget"].map(xpos)
    c["pairs"] = c["n_dirs"] // 2
    c["order"] = c["budget"].map(order.index)
    return c.sort_values(["order", "method"]).reset_index(drop=True)


def main() -> None:
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "font.family": "DejaVu Sans"})
    c = load()
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    ax.axhline(0, color=INK2, lw=0.8, zorder=1)
    for m, name, col, mk in METHODS:
        d = c[c["method"] == m]
        num, full = d[d["budget"] != "full"], d[d["budget"] == "full"]
        ax.plot(num["x"], num["mean_delta"], color=col, lw=2, marker=mk, ms=5, mec="white",
                mew=0.8, zorder=3, label=name)
        ax.plot(full["x"], full["mean_delta"], color=col, ls="none", marker=mk, ms=5,
                mec="white", mew=0.8, zorder=3)

    sizes = c[c["method"] == "chain"]
    xs = sizes["x"].to_numpy()
    ax.axvline((xs[-2] + xs[-1]) / 2, color=GRID, lw=1, ls=(0, (2, 2)), zorder=0)
    labels = []
    for _, r in sizes.iterrows():
        size = "full" if r["budget"] == "full" else f"{int(r['budget']):,}"
        labels.append(f"{size}\n({r['pairs']})")
    ax.set_xticks(xs, labels)
    ax.set_xlim(xs[0] - 0.4, xs[-1] + 0.4)
    ax.set_xlabel("direct dictionary size in word pairs (language pairs present)")
    ax.set_ylabel("gain over the direct map (MRR)")
    ax.grid(True, axis="y", color=GRID, lw=0.6, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, loc="upper right", labelcolor=INK2)
    fig.tight_layout(pad=0.3)
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"direct_size_sweep.{ext}", dpi=300,
                    metadata={"CreationDate": None} if ext == "pdf" else None)
    plt.close(fig)


if __name__ == "__main__":
    main()
